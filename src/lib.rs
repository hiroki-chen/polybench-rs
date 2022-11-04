#![cfg_attr(not(std), no_std)]

#![feature(rustc_attrs)]
#![feature(new_uninit)]
#![feature(core_intrinsics)]
#![feature(allocator_api)]

extern crate alloc;

pub mod datamining;
pub mod linear_algebra;
pub mod medley;
pub mod stencils;

pub mod config;
pub mod ndarray;
pub mod util;

#[cfg(not(std))]
mod cmath;
#[cfg(not(std))]
mod f32;
#[cfg(not(std))]
mod f64;
