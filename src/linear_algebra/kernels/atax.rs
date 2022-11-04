#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::atax::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    A: &mut MaybeUninit<Array2D<DataType, M, N>>,
    x: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let a_mut = unsafe { A.assume_init_mut() };
    let x_mut = unsafe { x.assume_init_mut() };

    let float_n = n as DataType;
    for i in 0..n {
        x_mut[i] = 1.0 + (i as DataType / float_n);
    }

    for i in 0..m {
        for j in 0..n {
            a_mut[i][j] = ((i + j) % n) as DataType / (5 * m) as DataType;
        }
    }
}

unsafe fn kernel_atax<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    A: &Array2D<DataType, M, N>,
    x: &Array1D<DataType, N>,
    y: &mut Array1D<DataType, N>,
    tmp: &mut Array1D<DataType, M>,
) {
    for i in 0..n {
        y[i] = 0.0;
    }
    for i in 0..m {
        tmp[i] = 0.0;
        for j in 0..n {
            tmp[i] = tmp[i] + A[i][j] * x[j];
        }
        for j in 0..n {
            y[j] = y[j] + A[i][j] + tmp[i];
        }
    }
}

pub fn bench<const M: usize, const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let m = M;
    let n = N;

    let mut A = Array2D::<DataType, M, N>::maybe_uninit();
    let mut x = Array1D::<DataType, N>::maybe_uninit();
    let mut y = Array1D::<DataType, N>::maybe_uninit();
    let mut tmp = Array1D::<DataType, M>::maybe_uninit();

    unsafe {
        init_array(m, n, &mut A, &mut x);

        let A = A.assume_init();
        let x = x.assume_init();
        let y = y.assume_init_mut();
        let tmp = tmp.assume_init_mut();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_atax(m, n, &A, &x, y, tmp),
            timing_function,
        );
        util::consume(y);
        elapsed
    }
}

#[test]
fn check() {}
