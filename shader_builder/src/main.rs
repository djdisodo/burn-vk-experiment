use std::error::Error;
use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn Error>>{
    SpirvBuilder::new("../shader", "spirv-unknown-vulkan1.1")
        //.capability(Capability::Groups)
        .capability(Capability::Int8)
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}