#![allow(non_snake_case)]

use crate::config::linear_algebra::blas::gesummv::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(
    n: usize,
    alpha: &mut DataType,
    beta: &mut DataType,
    A: &mut MaybeUninit<Array2D<DataType, N, N>>,
    B: &mut MaybeUninit<Array2D<DataType, N, N>>,
    x: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let B = unsafe { B.assume_init_mut() };
    let x = unsafe { x.assume_init_mut() };

    *alpha = 1.5;
    *beta = 1.2;
    for i in 0..n {
        x[i] = (i % n) as DataType / n as DataType;
        for j in 0..n {
            A[i][j] = ((i * j + 1) % n) as DataType / n as DataType;
            B[i][j] = ((i * j + 2) % n) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_gesummv<const N: usize>(
    n: usize,
    alpha: DataType,
    beta: DataType,
    A: &Array2D<DataType, N, N>,
    B: &Array2D<DataType, N, N>,
    tmp: &mut Array1D<DataType, N>,
    x: &Array1D<DataType, N>,
    y: &mut Array1D<DataType, N>,
) {
    for i in 0..n {
        tmp[i] = 0.0;
        y[i] = 0.0;
        for j in 0..n {
            tmp[i] = A[i][j] * x[j] + tmp[i];
            y[i] = B[i][j] * x[j] + y[i];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut A = Array2D::<DataType, N, N>::maybe_uninit();
    let mut B = Array2D::<DataType, N, N>::maybe_uninit();
    let tmp = Array1D::<DataType, N>::maybe_uninit();
    let mut x = Array1D::<DataType, N>::maybe_uninit();
    let y = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut alpha, &mut beta, &mut A, &mut B, &mut x);
        let A = A.assume_init();
        let B = B.assume_init();
        let x = x.assume_init();
        let mut tmp = tmp.assume_init();
        let mut y = y.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_gesummv(n, alpha, beta, &A, &B, &mut tmp, &x, &mut y),
            timing_function,
        );
        util::consume(y);
        elapsed
    }
}

#[test]
fn check() {}
