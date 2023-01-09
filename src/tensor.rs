use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use burn::tensor::{Data, Distribution, Shape};
use burn::tensor::backend::Backend;
use burn::tensor::ops::TensorOps;
use vulkano::buffer::{BufferAccess, BufferContents, TypedBufferAccess};
use crate::{Vulkan, VulkanDeviceHandle, VulkanTensor};

impl<T> Vulkan<T> where [T]: BufferContents {
    pub fn reshape_into<const D1: usize, const D2: usize>(tensor: VulkanTensor<T, D1>, shape: Shape<D2>) -> VulkanTensor<T, D2> {
        let mut device = tensor.device.clone();
        VulkanTensor {
            device: tensor.device,
            shape: device.new_buffer(shape.dims.into_iter().map(|x| x as u32)),
            buffer: tensor.buffer
        }
    }

    fn __from_data<const D: usize>(data: Data<T, D>, device: VulkanDeviceHandle) -> VulkanTensor<T, D> {
        let shape = device.new_buffer(data.shape.dims.into_iter().map(|x| x as u32));
        let buffer = device.new_buffer(data.value.into_iter());
        VulkanTensor {
            device,
            shape,
            buffer
        }
    }

    fn __reshape<const D1: usize, const D2: usize>(tensor: &VulkanTensor<T, D1>, shape: Shape<D2>) -> VulkanTensor<T, D2> {
        let tensor = tensor.clone();
        Self::reshape_into(tensor, shape)
    }


}

impl TensorOps<Self> for Vulkan<f32> {
    fn from_data<const D: usize>(data: Data<f32, D>, device: VulkanDeviceHandle) -> VulkanTensor<f32, D> {
        Self::__from_data(data, device)
    }

    fn from_data_bool<const D: usize>(data: Data<bool, D>, device: VulkanDeviceHandle) -> () {
        todo!()
    }

    fn random<const D: usize>(shape: Shape<D>, distribution: Distribution<f32>, device: VulkanDeviceHandle) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn shape<const D: usize>(tensor: &VulkanTensor<f32, D>) -> &Shape<D> {
        todo!()
    }

    fn to_data<const D: usize>(tensor: &VulkanTensor<f32, D>) -> Data<f32, D> {
        todo!()
    }

    fn into_data<const D: usize>(tensor: VulkanTensor<f32, D>) -> Data<f32, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(tensor: &<Self as Backend>::BoolTensorPrimitive<D>) -> &Shape<D> {
        todo!()
    }

    fn bool_to_data<const D: usize>(tensor: &<Self as Backend>::BoolTensorPrimitive<D>) -> Data<bool, D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(tensor: <Self as Backend>::BoolTensorPrimitive<D>) -> Data<bool, D> {
        todo!()
    }

    fn bool_to_device<const D: usize>(tensor: &<Self as Backend>::BoolTensorPrimitive<D>, device: VulkanDeviceHandle) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(tensor: &<Self as Backend>::BoolTensorPrimitive<D1>, shape: Shape<D2>) -> <Self as Backend>::BoolTensorPrimitive<D2> {
        todo!()
    }

    fn bool_index<const D1: usize, const D2: usize>(tensor: &<Self as Backend>::BoolTensorPrimitive<D1>, indexes: [Range<usize>; D2]) -> <Self as Backend>::BoolTensorPrimitive<D1> {
        todo!()
    }

    fn device<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanDeviceHandle {
        todo!()
    }

    fn to_device<const D: usize>(tensor: &VulkanTensor<f32, D>, device: VulkanDeviceHandle) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn empty<const D: usize>(shape: Shape<D>, device: VulkanDeviceHandle) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn add<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        let output = lhs.device.new_buffer_raw();
        let pipeline = lhs.device.shader_module.add_f32.clone();
        let dispatch = vulkano::DeviceSize::div_ceil(lhs.buffer.len(), 64 * size_of::<f32>());
        lhs.device.run(
            pipeline,
            [&lhs.buffer, &rhs.buffer, &output].map(|x| x.as_buffer_access().clone()),
            [dispatch as _, 0, 0],
            None
        ).wait(None).unwrap();
        VulkanTensor {
            device: lhs.device.clone(),
            buffer: output,
            shape: lhs.shape.clone()
        }
    }

    fn add_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn sub<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn sub_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn mul<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn div<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn div_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn matmul<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn neg<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn swap_dims<const D: usize>(tensor: &VulkanTensor<f32, D>, dim1: usize, dim2: usize) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(tensor: &VulkanTensor<f32, D1>, shape: Shape<D2>) -> VulkanTensor<f32, D2> {
        Self::__reshape(tensor, shape)
    }

    fn index<const D1: usize, const D2: usize>(tensor: &VulkanTensor<f32, D1>, indexes: [Range<usize>; D2]) -> VulkanTensor<f32, D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(tensor: &VulkanTensor<f32, D1>, indexes: [Range<usize>; D2], value: &VulkanTensor<f32, D1>) -> VulkanTensor<f32, D1> {
        todo!()
    }

    fn mask_fill<const D: usize>(tensor: &VulkanTensor<f32, D>, mask: &<Self as Backend>::BoolTensorPrimitive<D>, value: f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn equal<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn equal_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn greater_equal_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &VulkanTensor<f32, D>) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn lower_equal_scalar<const D: usize>(lhs: &VulkanTensor<f32, D>, rhs: &f32) -> <Self as Backend>::BoolTensorPrimitive<D> {
        todo!()
    }

    fn detach<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn mean<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, 1> {
        todo!()
    }

    fn sum<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, 1> {
        todo!()
    }

    fn mean_dim<const D: usize>(tensor: &VulkanTensor<f32, D>, dim: usize) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn sum_dim<const D: usize>(tensor: &VulkanTensor<f32, D>, dim: usize) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(tensor: &VulkanTensor<f32, D>) -> <<Self as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(tensor: &<<Self as Backend>::FullPrecisionBackend as Backend>::TensorPrimitive<D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn argmax<const D: usize>(tensor: &VulkanTensor<f32, D>, dim: usize) -> <<Self as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn argmin<const D: usize>(tensor: &VulkanTensor<f32, D>, dim: usize) -> <<Self as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn exp<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn log<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn powf<const D: usize>(tensor: &VulkanTensor<f32, D>, value: f32) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn sqrt<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn erf<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn cat<const D: usize>(tensors: &[VulkanTensor<f32, D>], dim: usize) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn relu<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<f32, D> {
        todo!()
    }
}

