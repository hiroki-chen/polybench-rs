#![no_std]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]
#![feature(allocator_api)]

extern crate alloc;
extern crate sgx_alloc;
extern crate sgx_no_tstd;

pub mod datamining;
pub mod linear_algebra;
pub mod medley;
pub mod stencils;

pub mod config;
pub mod ndarray;
pub mod util;

mod cmath;
mod f32;
mod f64;