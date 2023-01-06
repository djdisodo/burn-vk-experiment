use burn::tensor::backend::Backend;
use burn::tensor::ops::ModuleOps;
use crate::{Vulkan, VulkanTensor};

impl ModuleOps<Vulkan<f32>> for Vulkan<f32> {
    fn embedding(weights: &VulkanTensor<f32, 2>, indexes: &<<Self as Backend>::IntegerBackend as Backend>::TensorPrimitive<2>) -> VulkanTensor<f32, 3> {
        todo!()
    }

    fn embedding_backward(weights: &VulkanTensor<f32, 2>, output: &VulkanTensor<f32, 3>, indexes: &<<Self as Backend>::IntegerBackend as Backend>::TensorPrimitive<2>) -> <Self as Backend>::TensorPrimitive<2> {
        todo!()
    }
}