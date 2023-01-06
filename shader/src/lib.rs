#![no_std]

use spirv_std::glam::UVec3;
use spirv_std::spirv;
use spirv_std::num_traits::Zero;

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
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
                return;
            }
            output[idx] = input[idx] + *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
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
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
                return;
            }
            output[idx] = input[idx] - *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
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
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [$ty],
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
                return;
            }
            output[idx] = input[idx] * *scalar;
        }

        #[spirv(compute(threads(64)))]
        pub fn $id_append(
            #[spirv(global_invocation_id)] id: UVec3,
            #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &mut [$ty],
            #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] scalar: &$ty,
        ) {
            let idx = id.x as usize;
            if input.len() >= idx {
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
            if input.len() >= idx {
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
            if input.len() >= idx {
                return;
            }
            input[idx] /= *scalar
        }
    };
}

impl_div_scalar!(f32, div_scalar_f32, div_append_scalar_f32);
