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