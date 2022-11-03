#![no_std]
#![feature(rustc_attrs)]
#![feature(core_intrinsics)]

extern crate alloc;

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