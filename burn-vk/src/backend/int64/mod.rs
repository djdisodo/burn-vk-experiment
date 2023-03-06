use burn::tensor::backend::Backend;
use crate::device::VulkanDeviceHandle;
use crate::{Vulkan32, VulkanTensor};

impl Backend for Vulkan32<i64> {
    type Device = VulkanDeviceHandle;
    type Elem = i64;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = Vulkan32<f32>;
    type IntegerBackend = Vulkan32<i64>;
    type TensorPrimitive<const D: usize> = VulkanTensor<i64, D>;
    type BoolTensorPrimitive<const D: usize> = VulkanTensor<u8, D>;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "vulkan32".into()
    }

    fn seed(seed: u64) {
        todo!()
    }
}

include_mod!(tensor);
include_mod!(module);