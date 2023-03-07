#![cfg_attr(target_arch = "spirv", no_std)]
///common types

//above 3 will not likely be used i guess, i leave it here so it can be changed soon
//the more it slows down so...
pub const TENSOR_D: usize = 3;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TensorShape<const N: usize = TENSOR_D> {
    pub data: [u32; N],
    pub len: u32
}

#[cfg(not(target_arch = "spirv"))]
unsafe impl<const N: usize> bytemuck::Zeroable for TensorShape<N> {}
#[cfg(not(target_arch = "spirv"))]
unsafe impl<const N: usize> bytemuck::Pod for TensorShape<N> {}

impl<const N: usize> TensorShape<N> {
    pub fn indices(&self, n: u32) -> [u32; N] {
        let mut ret = [0u32; N];
        let mut div = 1;
        // for (d, r) in (self.data.iter().zip(ret.iter_mut())).take(self.len as usize).rev() {
        for idx in (0..(self.len)).rev() {
            let idx = idx as usize;
            let d = &self.data[idx];
            let r = &mut ret[idx];
            *r = (n / div) % *d;
            div *= *d;
        }
        ret
    }

    pub fn flatten_indices(&self, indices: &[u32; N]) -> u32 {
        let mut ret = 0;
        let mut mul = 1;

        //for (i, d) in indices.iter().zip(self.data.iter()).rev() {
        for idx in (0..(self.len)).rev() {
            let idx = idx as usize;
            let i = &indices[idx];
            let d = &self.data[idx];
            ret += *i * mul;
            mul *= *d;
        }
        ret
    }

    /// unstable why?
    /// this function doesn't check if size of dimension to be broadcasted is 1
    pub fn broadcast_unstable(&self, indices: &mut [u32; N]) {
        //for (i, d) in indices.iter_mut().zip(self.data.iter()) {
        for idx in 0..self.len {
            let idx = idx as usize;
            let i = &mut indices[idx];
            let d = &self.data[idx];
            if *i >= *d {
                *i = 0;
            }
        }
    }

    pub fn from_slice(slice: &[u32]) -> Self {
        assert!(slice.len() <= N);
        let mut inner = [0; N];
        for (x, i) in slice.iter().zip(inner.iter_mut()) {
            *i = *x
        }
        Self {
            data: inner,
            len: slice.len() as u32
        }
    }
}