use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use parking_lot::Mutex;
use vulkano::buffer::{BufferAccess, BufferContents, BufferUsage, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync;
use vulkano::sync::{FenceSignalFuture, GpuFuture, NowFuture};
use burn_vk_types::TensorShape;
use crate::{DeviceOrCpuBuffer, TensorOpsShader, VK_INSTANCE, VulkanBackendTy};

//device reference is redundant
#[derive(Debug)]
pub struct VulkanDevice {
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) shader_module: TensorOpsShader,
    pub(crate) pipeline_cache: Arc<PipelineCache>,
    pub(crate) memory_allocator: StandardMemoryAllocator,
    pub(crate) descriptor_set_allocator: StandardDescriptorSetAllocator,
    pub(crate) command_buffer_allocator: StandardCommandBufferAllocator
}

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_storage_buffer_storage_class: true,
    ..DeviceExtensions::empty()
};

pub static SUPPORTED_DEVICES: LazyLock<Vec<(Arc<PhysicalDevice>, u32)>> = LazyLock::new(|| {
    VK_INSTANCE
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&DEVICE_EXTENSIONS))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
            // that supports compute operations.
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(&QueueFlags {
                    compute: true,
                    ..Default::default()
                }))
                .map(|i| (p, i as u32))
        })
        .collect()
});

impl VulkanDevice {
    pub fn from_index(i: usize) -> Option<(Arc<Device>, Arc<Queue>)> {
        SUPPORTED_DEVICES.get(i).map(|(physical_device, queue_family_index)| {
            let (device, mut queue) = Device::new(
                physical_device.clone(),
                DeviceCreateInfo {
                    enabled_features: Features {
                        vulkan_memory_model: true,
                        ..Features::default()
                    },
                    enabled_extensions: DEVICE_EXTENSIONS,
                    queue_create_infos: vec![QueueCreateInfo {
                        queue_family_index: *queue_family_index,
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            ).unwrap();
            (device, queue.next().unwrap())
        })
    }

    pub(crate) fn new_builder(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap()
    }


    //TODO need investigation
    pub(crate) fn new_buffer_array_data<T>(&self, data: impl Iterator<Item=T>) -> DeviceOrCpuBuffer<[T]> where [T]: BufferContents {
        let mut command_buffer_builder = self.new_builder();
        let buffer = DeviceLocalBuffer::from_data(
            &self.device.memory_allocator,
            data,
            BufferUsage {
                storage_buffer: true,
                ..Default::default()
            }, &mut command_buffer_builder).unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        DeviceOrCpuBuffer::Device(buffer)
    }

    //TODO need investigation
    pub(crate) fn new_buffer_array<T>(&self, len: usize) -> DeviceOrCpuBuffer<[T]> where [T]: BufferContents {
        DeviceOrCpuBuffer::Device(DeviceLocalBuffer::array(
            &self.memory_allocator,
            len as _,
            BufferUsage {
                storage_buffer: true,
                ..Default::default()
            },
            []
        ).unwrap())
    }

    pub(crate) fn clone_buffer<T: BufferContents>(&self, buffer: DeviceOrCpuBuffer<T>) -> DeviceOrCpuBuffer<T> {
        let mut command_buffer_builder = self.new_builder();
        let buffer = DeviceLocalBuffer::from_buffer(
            &self.device.memory_allocator,
            buffer.as_buffer_access().clone(),
            BufferUsage {
                storage_buffer: true,
                ..Default::default()
            }, &mut command_buffer_builder).unwrap();
        let command_buffer = command_buffer_builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
        DeviceOrCpuBuffer::Device(buffer)
    }

    pub(crate) fn run<T: BufferContents>(
        &self,
        pipeline: Arc<ComputePipeline>,
        args: impl IntoIterator<Item=Arc<dyn BufferAccess>>,
        dispatch: [u32; 3], constants: Option<T>
    ) -> FenceSignalFuture<CommandBufferExecFuture<NowFuture>> {
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            descriptors(args)
        );
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap();
        let layout = pipeline.layout().clone();
        builder
            .bind_pipeline_compute(pipeline)
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                layout.clone(),
                0,
                set
            );
        if let Some(constants) = constants {
            builder.push_constants(layout, 0, constants)
        }
        builder.dispatch(dispatch).unwrap();

        let command_buffer = builder.build().unwrap();
        sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
    }

    pub(crate) fn run_single<T: VulkanBackendTy>(
        &self,
        pipeline: Arc<ComputePipeline>,
        buffer: &DeviceOrCpuBuffer<[T]>
    ) -> DeviceOrCpuBuffer<[T]> where [T]: BufferContents {
        let output = self.new_buffer_array(buffer.len());
        let dispatch = vulkano::DeviceSize::div_ceil(buffer.len(), 64);
        self.run(
            pipeline,
            [buffer, &output].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            None
        ).wait(None).unwrap();
        output
    }

    pub(crate) fn run_single_assign<T: VulkanBackendTy>(
        &self,
        pipeline: Arc<ComputePipeline>,
        buffer: &DeviceOrCpuBuffer<[T]>
    ) where [T]: BufferContents  {
        let dispatch = vulkano::DeviceSize::div_ceil(buffer.len(), 64);
        self.run(
            pipeline,
            [buffer].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            None
        ).wait(None).unwrap();
    }

    pub(crate) fn run_multiple<T: VulkanBackendTy>(
        &self,
        pipeline: Arc<ComputePipeline>,
        lhs: &DeviceOrCpuBuffer<[T]>,
        rhs: &DeviceOrCpuBuffer<[T]>,
    ) -> DeviceOrCpuBuffer<[T]> where [T]: BufferContents {
        let output = self.new_buffer_array(lhs.as_typed_access().len());
        let dispatch = vulkano::DeviceSize::div_ceil(lhs.as_typed_access().len(), 64);
        self.run(
            pipeline,
            [lhs, rhs, &output].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            None
        ).wait(None).unwrap();
        output
    }

    pub(crate) fn run_multiple_assign<T: VulkanBackendTy> (
        &self,
        pipeline: Arc<ComputePipeline>,
        lhs: &DeviceOrCpuBuffer<[T]>,
        rhs: &DeviceOrCpuBuffer<[T]>,
    ) where [T]: BufferContents  {
        let dispatch = vulkano::DeviceSize::div_ceil(lhs.as_typed_access().len(), 64);
        self.run(
            pipeline,
            [lhs, rhs].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            None
        ).wait(None).unwrap();
    }

    pub(crate) fn run_multiple_b<T: VulkanBackendTy, U: Into<u32>, const N: usize>(
        &self,
        pipeline: Arc<ComputePipeline>,
        lhs: &DeviceOrCpuBuffer<[T]>,
        lhs_shape: &[U; N],
        rhs: &DeviceOrCpuBuffer<[T]>,
        rhs_shape: &[U; N]
    ) -> DeviceOrCpuBuffer<[T]> where [T]: BufferContents {
        let output = self.new_buffer_array(lhs.as_typed_access().len());
        let dispatch = vulkano::DeviceSize::div_ceil(lhs.as_typed_access().len(), 64);
        self.run(
            pipeline,
            [lhs, rhs, &output].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            Some([lhs_shape, rhs_shape].map(TensorShape::from_slice))
        ).wait(None).unwrap();
        output
    }

    pub(crate) fn run_multiple_assign_b<T: VulkanBackendTy, U: Into<u32>, const N: usize>(
        &self,
        pipeline: Arc<ComputePipeline>,
        lhs: &DeviceOrCpuBuffer<[T]>,
        lhs_shape: &[U; N],
        rhs: &DeviceOrCpuBuffer<[T]>,
        rhs_shape: &[U; N]
    ) where [T]: BufferContents {
        let dispatch = vulkano::DeviceSize::div_ceil(lhs.as_typed_access().len(), 64);
        self.run(
            pipeline,
            [lhs, rhs].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            Some([lhs_shape, rhs_shape].map(TensorShape::from_slice))
        ).wait(None).unwrap();
    }

}

static DEFAULT_DEVICE_OVERRIDE: LazyLock<Mutex<Option<VulkanDevice>>> = LazyLock::new(Mutex::default);

static DEFAULT_DEVICE: LazyLock<Arc<VulkanDevice>> = LazyLock::new(|| {
    let (physical_device, queue_family_index) = SUPPORTED_DEVICES.iter()
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();
    let (device, mut queue) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_features: Features {
                vulkan_memory_model: true,
                ..Features::default()
            },
            enabled_extensions: DEVICE_EXTENSIONS,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index: *queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    ).unwrap();
    let queue = queue.next().unwrap();

    let cache = PipelineCache::empty(device.clone()).unwrap();

    Arc::new(VulkanDevice {
        device: device.clone(),
        queue: queue.clone(),
        shader_module: TensorOpsShader::load(device.clone(), Some(cache.clone())),
        pipeline_cache: cache,
        memory_allocator: StandardMemoryAllocator::new_default(device.clone()),
        descriptor_set_allocator: StandardDescriptorSetAllocator::new(device.clone()),
        command_buffer_allocator: StandardCommandBufferAllocator::new(device, Default::default())
    })
});
#[derive(Clone, Debug)]
pub struct VulkanDeviceHandle(Arc<VulkanDevice>);

impl Default for VulkanDeviceHandle {
    fn default() -> Self {
        let device = if let Some(over) = std::mem::replace(&mut *DEFAULT_DEVICE_OVERRIDE.lock(), None) {
            Arc::new(over)
        } else {
            DEFAULT_DEVICE.clone()
        };
        Self(device)
    }
}

pub(crate) fn descriptors<T: IntoIterator<Item = Arc<dyn BufferAccess>>>(iter: T) -> impl Iterator<Item=WriteDescriptorSet> {
    (0u32..).zip(iter).map(|(x, y)| WriteDescriptorSet::buffer(x, y))
}

impl Deref for VulkanDeviceHandle {
    type Target = Arc<VulkanDevice>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
