use burn::tensor::ops::{MaxPool2dBackward, MaxPool2dWithIndexes, ModuleOps};
use crate::{Vulkan32, VulkanTensor};

impl ModuleOps<Vulkan32<f32>> for Vulkan32<f32> {
    fn embedding(weights: &VulkanTensor<f32, 2>, indexes: &VulkanTensor<i64, 2>) -> VulkanTensor<f32, 3> {
        todo!()
    }

    fn embedding_backward(weights: &VulkanTensor<f32, 2>, output: &VulkanTensor<f32, 3>, indexes: &VulkanTensor<i64, 2>) -> VulkanTensor<f32, 2> {
        todo!()
    }

    fn conv2d(x: &VulkanTensor<f32, 4>, weight: &VulkanTensor<f32, 4>, bias: Option<&VulkanTensor<f32, 1>>, stride: [usize; 2], padding: [usize; 2]) -> VulkanTensor<f32, 4> {
        todo!()
    }

    fn max_pool2d(x: &VulkanTensor<f32, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> VulkanTensor<f32, 4> {
        todo!()
    }

    fn max_pool2d_with_indexes(x: &VulkanTensor<f32, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> MaxPool2dWithIndexes<Vulkan32<f32>> {
        todo!()
    }

    fn max_pool2d_with_indexes_backward(x: &VulkanTensor<f32, 4>, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], output_grad: &VulkanTensor<f32, 4>, indexes: &VulkanTensor<i64, 4>) -> MaxPool2dBackward<Vulkan32<f32>> {
        todo!()
    }
}