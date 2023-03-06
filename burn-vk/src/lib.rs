#![feature(once_cell)]
#![feature(int_roundings)]

use std::sync::{Arc, LazyLock};
use burn::tensor::backend::Backend;
use burn::tensor::ops::{Ones, Zeros};
use burn::tensor::Shape;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{Version, VulkanLibrary};
use vulkano::buffer::{BufferAccess, BufferContents, CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess};
use vulkano_spirv_bind_macro::include_shader;
use crate::device::VulkanDeviceHandle;
macro_rules! include_mod {
    ($id:ident) => {
        mod $id;
        pub use $id::*;
    };
}


//absolute path for ide(relative path does compile)
include_shader!(TensorOpsShader, "C:\\Users\\PC\\burn-vk\\burn-vk\\src\\../../burn_vk_shader.spv");

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

pub trait VulkanBackendTy {
    const NAME: &'static str;
}

macro_rules! def_backend_ty {
    ($ty:ty) => {
        impl VulkanBackendTy for $ty {
            const NAME: &'static str = stringify!($ty);
        }
    };
}

def_backend_ty!(f32);

pub mod device;

#[derive(Clone)]
pub enum DeviceOrCpuBuffer<T: BufferContents + ?Sized> {
    Device(Arc<DeviceLocalBuffer<T>>),
    Cpu(Arc<CpuAccessibleBuffer<T>>)
}

macro_rules! buffer_enum_foreach {
    ($self: expr, $ex: expr) => {
        match $self {
            DeviceOrCpuBuffer::Device(x) => $ex(x),
            DeviceOrCpuBuffer::Cpu(x) => $ex(x)
        }
    };
}

impl<T: BufferContents + ?Sized> DeviceOrCpuBuffer<T> where DeviceLocalBuffer<T>: BufferAccess, CpuAccessibleBuffer<T>: BufferAccess {
    pub fn as_buffer_access(&self) -> &Arc<dyn BufferAccess> {
        &buffer_enum_foreach!(self, |x| x)
    }
    pub fn as_typed_access(&self) -> &Arc<dyn TypedBufferAccess<Content=T>> {
        &buffer_enum_foreach!(self, |x| x)
    }
}



include_mod!(backend);

#[derive(Debug)]
pub struct VulkanTensor<T, const D: usize> where [T]: BufferContents {
    device: VulkanDeviceHandle,
    shape: Shape<D>,
    buffer: DeviceOrCpuBuffer<[T]>
}

impl<T, const D: usize> Ones for VulkanTensor<T, D>  where [T]: BufferContents {
    fn ones(&self) -> Self {
        todo!()
    }
}

impl<T, const D: usize> Zeros for VulkanTensor<T, D>  where [T]: BufferContents {
    fn zeros(&self) -> Self {
        todo!()
    }
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

///types
pub type TensorPrimitive<B: Backend, const D: usize> = B::TensorPrimitive<D>;
pub type BoolTensorPrimitive<B: Backend, const D: usize> = B::BoolTensorPrimitive<D>;