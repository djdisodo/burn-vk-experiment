use std::marker::PhantomData;
use burn::tensor::{Data, Shape};
use vulkano::buffer::BufferContents;
use crate::device::VulkanDeviceHandle;
use crate::VulkanTensor;

#[derive(Clone, Default, Debug)]
pub struct Vulkan32<T>(PhantomData<T>);

impl<T> Vulkan32<T> where [T]: BufferContents {
    pub fn reshape_into<const D1: usize, const D2: usize>(tensor: VulkanTensor<T, D1>, shape: Shape<D2>) -> VulkanTensor<T, D2> {
        let mut device = tensor.device.clone();
        VulkanTensor {
            device: tensor.device,
            shape,
            buffer: tensor.buffer
        }
    }

    fn __from_data<const D: usize>(data: Data<T, D>, device: VulkanDeviceHandle) -> VulkanTensor<T, D> {
        let buffer = device.new_buffer_array_data(data.value.into_iter());
        VulkanTensor {
            device,
            shape: data.shape,
            buffer
        }
    }

    fn __reshape<const D1: usize, const D2: usize>(tensor: &VulkanTensor<T, D1>, shape: Shape<D2>) -> VulkanTensor<T, D2> {
        let tensor = tensor.clone();
        Self::reshape_into(tensor, shape)
    }


}

include_mod!(fp32);
include_mod!(int64);