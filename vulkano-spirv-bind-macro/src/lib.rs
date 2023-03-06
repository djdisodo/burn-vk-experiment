#![feature(proc_macro_span)]
#![feature(array_windows)]

use std::path::PathBuf;
use proc_macro2::{Ident, TokenStream};
use quote::{quote, TokenStreamExt};
use syn::parse::{Parse, ParseStream};
use syn::{LitStr, parse_macro_input};
use syn::token::Comma;
use vulkano::shader::spirv::{Instruction, Spirv};


struct IncludeShaderInput {
    ident: Ident,
    path: LitStr
}

impl Parse for IncludeShaderInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        let _: Comma = input.parse()?;
        let path: LitStr = input.parse()?;
        Ok(Self {
            ident,
            path,
        })
    }
}

#[proc_macro]
pub fn include_shader(tokens: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(tokens as IncludeShaderInput);
    let i_path = PathBuf::from(input.path.value());
    let path = if i_path.is_relative() {
        let mut source_path: PathBuf = proc_macro::Span::call_site().source_file().path().parent().unwrap().canonicalize().unwrap();
        source_path.push(i_path);
        source_path
    } else {
        i_path
    };
    let bytes = std::fs::read(path).unwrap();

    let spirv = Spirv::new(unsafe { std::slice::from_raw_parts(
                bytes.as_ptr() as *const _,
                bytes.len() / std::mem::size_of::<u32>(),
	)}).unwrap();
    let mut fields_ts = TokenStream::new();
    let mut fields_initialization = TokenStream::new();
    for x in spirv.iter_entry_point() {
        if let Instruction::EntryPoint {
            name,
            ..
        } = x {
            let ident = Ident::new(name, input.path.span().clone());
            fields_ts.append_all(quote! {
                pub #ident: std::sync::Arc<vulkano::pipeline::compute::ComputePipeline>,
            });

            fields_initialization.append_all(quote! {
                #ident: vulkano::pipeline::compute::ComputePipeline::new(
                    device.clone(),
                    shader_module.entry_point(stringify!(#ident)).unwrap(),
                    &(),
                    cache,
                    |_| ()
                ),
            })
        }
    }

    let input_ident = input.ident;
    let input_path = input.path;

    (quote! {
        pub struct #input_ident {
            #fields_ts
        }

        impl #input_ident {
            pub fn load(device: std::sync::Arc<vulkano::device::Device>, cache: std::option::Option<std::sync::Arc<vulkano::pipeline::cache::PipelineCache>>) -> Self {
                let shader_module = std::pin::Pin::new(unsafe {
                    vulkano::shader::ShaderModule::from_bytes(device, include_bytes!(#input_path)).unwrap()
                });
                Self {
                    #fields_initialization
                }
            }
        }
    }).into()
}