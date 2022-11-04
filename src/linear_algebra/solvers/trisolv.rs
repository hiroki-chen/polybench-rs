#![allow(non_snake_case)]

use crate::config::linear_algebra::solvers::trisolv::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(
    n: usize,
    L: &mut MaybeUninit<Array2D<DataType, N, N>>,
    x: &mut MaybeUninit<Array1D<DataType, N>>,
    b: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let L = unsafe { L.assume_init_mut() };
    let x = unsafe { x.assume_init_mut() };
    let b = unsafe { b.assume_init_mut() };

    for i in 0..n {
        x[i] = -999.0;
        b[i] = i as DataType;
        for j in 0..=i {
            L[i][j] = (i + n - j + 1) as DataType * 2.0 / n as DataType;
        }
    }
}

unsafe fn kernel_trisolv<const N: usize>(
    n: usize,
    L: &Array2D<DataType, N, N>,
    x: &mut Array1D<DataType, N>,
    b: &Array1D<DataType, N>,
) {
    for i in 0..n {
        x[i] = b[i];
        for j in 0..i {
            x[i] -= L[i][j] * x[j];
        }
        x[i] = x[i] / L[i][i];
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut L = Array2D::<DataType, N, N>::maybe_uninit();
    let mut x = Array1D::<DataType, N>::maybe_uninit();
    let mut b = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut L, &mut x, &mut b);
        let L = L.assume_init();
        let b = b.assume_init();
        let mut x = x.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_trisolv(n, &L, &mut x, &b),
            timing_function,
        );
        util::consume(x);
        elapsed
    }
}

#[test]
fn check() {}
