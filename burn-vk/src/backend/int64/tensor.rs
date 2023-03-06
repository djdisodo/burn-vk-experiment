use std::ops::Range;
use burn::tensor::{Data, Distribution, Shape};
use burn::tensor::ops::TensorOps;
use crate::device::VulkanDeviceHandle;
use crate::{Vulkan32, VulkanTensor};

impl TensorOps<Vulkan32<i64>> for Vulkan32<i64> {
    fn from_data<const D: usize>(data: Data<i64, D>, device: &VulkanDeviceHandle) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn from_data_bool<const D: usize>(data: Data<bool, D>, device: &VulkanDeviceHandle) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn random<const D: usize>(shape: Shape<D>, distribution: Distribution<i64>, device: &VulkanDeviceHandle) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn shape<const D: usize>(tensor: &VulkanTensor<i64, D>) -> Shape<D> {
        todo!()
    }

    fn to_data<const D: usize>(tensor: &VulkanTensor<i64, D>) -> Data<i64, D> {
        todo!()
    }

    fn into_data<const D: usize>(tensor: VulkanTensor<i64, D>) -> Data<i64, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(tensor: &VulkanTensor<u8, D>) -> Shape<D> {
        todo!()
    }

    fn bool_to_data<const D: usize>(tensor: &VulkanTensor<u8, D>) -> Data<bool, D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(tensor: VulkanTensor<u8, D>) -> Data<bool, D> {
        todo!()
    }

    fn bool_to_device<const D: usize>(tensor: &VulkanTensor<u8, D>, device: &VulkanDeviceHandle) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(tensor: &VulkanTensor<u8, D1>, shape: Shape<D2>) -> VulkanTensor<u8, D2> {
        todo!()
    }

    fn bool_index<const D1: usize, const D2: usize>(tensor: &VulkanTensor<u8, D1>, indexes: [Range<usize>; D2]) -> VulkanTensor<u8, D1> {
        todo!()
    }

    fn device<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanDeviceHandle {
        todo!()
    }

    fn to_device<const D: usize>(tensor: &VulkanTensor<i64, D>, device: &VulkanDeviceHandle) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &VulkanDeviceHandle) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn add<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn add_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn sub<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn sub_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn mul<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn mul_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn div<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn div_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn matmul<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn neg<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn swap_dims<const D: usize>(tensor: &VulkanTensor<i64, D>, dim1: usize, dim2: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn reshape<const D1: usize, const D2: usize>(tensor: &VulkanTensor<i64, D1>, shape: Shape<D2>) -> VulkanTensor<i64, D2> {
        todo!()
    }

    fn index<const D1: usize, const D2: usize>(tensor: &VulkanTensor<i64, D1>, indexes: [Range<usize>; D2]) -> VulkanTensor<i64, D1> {
        todo!()
    }

    fn index_assign<const D1: usize, const D2: usize>(tensor: &VulkanTensor<i64, D1>, indexes: [Range<usize>; D2], value: &VulkanTensor<i64, D1>) -> VulkanTensor<i64, D1> {
        todo!()
    }

    fn mask_fill<const D: usize>(tensor: &VulkanTensor<i64, D>, mask: &VulkanTensor<u8, D>, value: i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn equal<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn equal_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn greater<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn greater_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn greater_equal<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn greater_equal_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn lower<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn lower_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn lower_equal<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &VulkanTensor<i64, D>) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn lower_equal_scalar<const D: usize>(lhs: &VulkanTensor<i64, D>, rhs: &i64) -> VulkanTensor<u8, D> {
        todo!()
    }

    fn detach<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn mean<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, 1> {
        todo!()
    }

    fn sum<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, 1> {
        todo!()
    }

    fn mean_dim<const D: usize>(tensor: &VulkanTensor<i64, D>, dim: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn sum_dim<const D: usize>(tensor: &VulkanTensor<i64, D>, dim: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<f32, D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(tensor: &VulkanTensor<f32, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn argmax<const D: usize>(tensor: &VulkanTensor<i64, D>, dim: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn argmin<const D: usize>(tensor: &VulkanTensor<i64, D>, dim: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn exp<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn log<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn powf<const D: usize>(tensor: &VulkanTensor<i64, D>, value: i64) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn sqrt<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn cos<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn sin<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn tanh<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn erf<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn cat<const D: usize>(tensors: &[VulkanTensor<i64, D>], dim: usize) -> VulkanTensor<i64, D> {
        todo!()
    }

    fn relu<const D: usize>(tensor: &VulkanTensor<i64, D>) -> VulkanTensor<i64, D> {
        todo!()
    }
}