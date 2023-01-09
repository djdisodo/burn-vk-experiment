#![no_std]
#![feature(asm_experimental_arch, asm_const)]

use spirv_std::arch::atomic_store;
use spirv_std::glam::UVec3;
use spirv_std::spirv;
use core::arch::asm;
use spirv_std::num_traits::Zero;
use spirv_std::num_traits::Float;
use spirv_std::memory::Scope;

macro_rules! impl_relu {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [$ty],
        ) {
            if input.len() >= id.x as usize {
                return;
            }
            output[id.x as usize] = <$ty>::max(input[id.x as usize], <$ty>::zero());
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &mut [$ty],
        ) {
            if buffer.len() >= id.x as usize {
                return;
            }
            buffer[id.x as usize] = <$ty>::max(buffer[id.x as usize], <$ty>::zero());
        }
    };
}

impl_relu!(f32, relu_f32, relu_append_f32);

macro_rules! impl_add {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            //id.x is index, idx with . UwU
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            output[idx] = lhs[idx] + rhs[idx]
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            lhs[idx] += rhs[idx]
        }
    };
}

impl_add!(f32, add_f32, add_append_f32);




macro_rules! impl_add_scalar {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(push_constant)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx] + *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(push_constant)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            input[idx] += *scalar
        }
    };
}


impl_add_scalar!(f32, add_scalar_f32, add_scalar_append_f32);

macro_rules! impl_sub {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            output[idx] = lhs[idx] - rhs[idx]
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            lhs[idx] -= rhs[idx]
        }
    };
}


impl_sub!(f32, sub_f32, sub_append_f32);

macro_rules! impl_sub_scalar {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(push_constant)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx] - *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(push_constant)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            input[idx] += *scalar
        }
    };
}


impl_sub_scalar!(f32, sub_scalar_f32, sub_scalar_append_f32);

macro_rules! impl_mul {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            output[idx] = lhs[idx] * rhs[idx]
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            lhs[idx] *= rhs[idx]
        }
    };
}


impl_mul!(f32, mul_f32, mul_append_f32);

macro_rules! impl_mul_scalar {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(push_constant)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx] * *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(push_constant)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            input[idx] *= *scalar
        }
    };
}

impl_mul_scalar!(f32, mul_scalar_f32, mul_scalar_append_f32);

macro_rules! impl_div {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            output[idx] = lhs[idx] / rhs[idx]
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
        ) {
            let idx = id.x as usize;
            if lhs.len() >= idx {
                return;
            }
            lhs[idx] /= rhs[idx]
        }
    };
}

impl_div!(f32, div_f32, div_append_f32);

macro_rules! impl_div_scalar {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx] / *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            input[idx] /= *scalar
        }
    };
}

impl_div_scalar!(f32, div_scalar_f32, div_append_scalar_f32);

//maybe faster than cpu eq in case, not efficient tho
//thx eddyb
macro_rules! impl_eq {
    ($ty:ty, $id:ident) => {
        #[spirv(compute(threads(64)))]
        pub unsafe fn $id(
            #[spirv(local_invocation_id)] local_id: UVec3,
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] lhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] rhs: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut u8,
        ) {
            let idx = id.x as usize;
            // Always compute a value, so that `OpGroupAll` can work.
            let equal = if idx < lhs.len() { lhs[idx] == rhs[idx] } else { true };
            let all_equal = {
                let input = equal;
                let mut result = false;
                asm! {
                    "%bool = OpTypeBool",
                    "%u32 = OpTypeInt 32 0",
                    "%scope = OpConstant %u32 {scope}",
                    "%input = OpLoad %bool {input}",
                    //"%result = OpGroupAll %bool %scope %input", keep it like this till some fix
                    //"OpStore {result} %result",
                    scope = const Scope::Workgroup as u32,
                    input = in(reg) &input,
                    //result = in(reg) &mut result
                }
                result
            };
            // Here comes the hard part: you have to "elect a leader" to write back the result.
            if local_id.x == 0 {
                if !all_equal {
                    atomic_store::<_, {Scope::Subgroup as u32}, 0>(output, 0u8)
                }
            }
        }
    };
}
impl_eq!(f32, eq_f32);

#[repr(packed)]
struct SwapDimsArgs {
    a: u32,
    b: u32
}

/*
#[spirv(compute(threads(64)))]
        pub fn swap_dims(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_shape: &[u32],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &[f32],
            #[spirv(push_constant)] swap_dims_args: &SwapDimsArgs
        ) {
            let mut temp = [0u32; 16];
            if idx >= lhs.len() {
                return;
            }


        }
*/



macro_rules! impl_neg {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx] * (-1 as $ty);
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= buffer.len() {
                return;
            }
            buffer[idx] *= (-1 as $ty)
        }
    };
}

impl_neg!(f32, neg_f32, neg_append_f32);

macro_rules! impl_exp {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx].exp();
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= buffer.len() {
                return;
            }
            buffer[idx] = buffer[idx].exp();
        }
    };
}

impl_exp!(f32, exp_f32, exp_append_f32);

macro_rules! impl_ln {
    ($ty:ty, $id:ident, $id_append:ident) => {
        #[spirv(compute(threads(64)))]
        pub fn $id(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= input.len() {
                return;
            }
            output[idx] = input[idx].ln();
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] buffer: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if idx >= buffer.len() {
                return;
            }
            buffer[idx] = buffer[idx].ln();
        }
    };
}

impl_exp!(f32, ln_f32, ln_append_f32);
