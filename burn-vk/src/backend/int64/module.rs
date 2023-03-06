use burn::tensor::backend::Backend;
use burn::tensor::ops::{MaxPool2dBackward, MaxPool2dWithIndexes, ModuleOps};
use crate::{Vulkan32, VulkanTensor};

impl ModuleOps<Vulkan32<i64>> for Vulkan32<i64> {
    fn embedding(weights: &VulkanTensor<i64, 2>, indexes: &VulkanTensor<i64, 2>) -> VulkanTensor<i64, 3> {
        todo!()
    }

    fn embedding_backward(weights: &VulkanTensor<i64, 2>, output: &VulkanTensor<i64, 3>, indexes: &VulkanTensor<i64, 2>) -> VulkanTensor<i64, 2> {
        todo!()
    }

    fn conv2d(x: &VulkanTensor<i64, 4>, weight: &VulkanTensor<i64, 4>, bias: Option<&VulkanTensor<i64, 1>>, stride: [usize; 2], padding: [usize; 2]) -> VulkanTensor<i64, 4> {
        todo!()
    }

    fn max_pool2d(x: &VulkanTensor<i64, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> VulkanTensor<i64, 4> {
        todo!()
    }

    fn max_pool2d_with_indexes(x: &VulkanTensor<i64, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> MaxPool2dWithIndexes<Vulkan32<i64>> {
        todo!()
    }

    fn max_pool2d_with_indexes_backward(x: &VulkanTensor<i64, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], output_grad: &VulkanTensor<i64, 4>, indexes: &VulkanTensor<i64, 4>) -> MaxPool2dBackward<Vulkan32<i64>> {
        todo!()
    }
}