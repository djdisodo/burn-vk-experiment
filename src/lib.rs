#![feature(once_cell)]
#![feature(int_roundings)]

use std::intrinsics::offset;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, LazyLock};
use burn::tensor::backend::Backend;
use bytemuck::Pod;
use parking_lot::Mutex;
use vulkano::buffer::{BufferAccess, BufferContents, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Features, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{sync, Version, VulkanLibrary};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::pipeline::cache::PipelineCache;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::{FenceSignalFuture, GpuFuture, NowFuture};
use vulkano_spirv_bind_macro::include_shader;

//absolute path for ide(relative path does compile)
include_shader!(TensorOpsShader, "C:\\Users/PC/burn-vk/shader\\shader.spv");

//device reference is redundant
#[derive(Debug)]
pub struct VulkanDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    shader_module: TensorOpsShader,
    pipeline_cache: Arc<PipelineCache>,
    memory_allocator: StandardMemoryAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    command_buffer_allocator: StandardCommandBufferAllocator
}

static VK_INSTANCE: LazyLock<Arc<Instance>> = LazyLock::new(|| {
    let library = VulkanLibrary::new().unwrap();
    Instance::new(
        library,
        InstanceCreateInfo {
            enumerate_portability: true,
            max_api_version: Some(Version::V1_1),
            ..Default::default()
        }
    ).unwrap()
});

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

    fn new_builder(&self) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        ).unwrap()
    }


    //TODO need investigation
    fn new_buffer<T: BufferContents>(&self, data: impl Iterator<Item=T>) -> Arc<dyn TypedBufferAccess<Content=T>> {
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
        buffer
    }

    //TODO need investigation
    fn new_buffer_raw<T: BufferContents>(&self) -> DeviceOrCpuBuffer<T> {
        let command_buffer_builder = self.new_builder();
        DeviceOrCpuBuffer::Device(DeviceLocalBuffer::new(
            &self.memory_allocator,
            BufferUsage {
                storage_buffer: true,
                ..Default::default()
            },
            []
        ).unwrap())
    }

    fn clone_buffer<T: BufferContents>(&self, buffer: DeviceOrCpuBuffer<T>) -> DeviceOrCpuBuffer<T> {
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

    fn run<T>(&self, pipeline: Arc<ComputePipeline>, args: impl IntoIterator<Item=Arc<dyn BufferAccess>>, dispatch: [u32; 3], constants: Option<T>) -> FenceSignalFuture<CommandBufferExecFuture<NowFuture>> {
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
        let future = sync::now(self.device.clone()).then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

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

pub fn descriptors<T: IntoIterator<Item = Arc<dyn BufferAccess>>>(iter: T) -> impl Iterator<Item=WriteDescriptorSet> {
    (0u32..).zip(iter).map(|(x, y)| WriteDescriptorSet::buffer(x, y))
}

impl Deref for VulkanDeviceHandle {
    type Target = Arc<VulkanDevice>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Default, Debug)]
pub struct Vulkan<T>(PhantomData<T>);

#[derive(Clone)]
pub enum DeviceOrCpuBuffer<T: BufferContents + ?Sized> {
    Device(Arc<DeviceLocalBuffer<T>>),
    Cpu(Arc<CpuAccessibleBuffer<T>>)
}

macro_rules! buffer_enum_foreach {
    ($self: expr, $ex: expr) => {
        match $self {
            DeviceOrCpuBuffer::Device(x) => ({$ex})(x),
            DeviceOrCpuBuffer::Cpu(x) => ({$ex})(x)
        }
    };
}

impl<T: BufferContents + ?Sized> DeviceOrCpuBuffer<T> where DeviceLocalBuffer<T>: BufferAccess, CpuAccessibleBuffer<T>: BufferAccess{
    pub fn as_buffer_access(&self) -> &Arc<dyn BufferAccess> {
        match self {
            DeviceOrCpuBuffer::Device(_) => {}
            DeviceOrCpuBuffer::Cpu(x) => {x.}
        }
        buffer_enum_foreach!(self, |x| x)
    }
}


pub struct VulkanTensor<T, const D: usize> where [T]: BufferContents {
    device: VulkanDeviceHandle,
    shape: DeviceOrCpuBuffer<[u32; D]>,
    buffer: DeviceOrCpuBuffer<[T]>
}

impl<T, const D: usize> Clone for VulkanTensor<T, D> where [T]: BufferContents {
    fn clone(&self) -> Self {
        let buffer = self.device.clone_buffer(self.buffer.clone());
        VulkanTensor {
            device: self.device.clone(),
            shape: self.shape.clone(),
            buffer
        }
    }
}

impl Backend for Vulkan<f32> {
    type Device = VulkanDeviceHandle;
    type Elem = f32;
    type FullPrecisionElem = Vulkan<f32>;
    type FullPrecisionBackend = Vulkan<f32>;
    type IntegerBackend = Vulkan<f32>;
    type TensorPrimitive<const D: usize> = VulkanTensor<f32, D>;
    type BoolTensorPrimitive<const D: usize> = ();

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "vulkan".into()
    }

    fn seed(seed: u64) {
        todo!()
    }
}

mod tensor;
mod module;