#![allow(non_snake_case)]

use crate::config::linear_algebra::blas::gemver::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(
    n: usize,
    alpha: &mut DataType,
    beta: &mut DataType,
    A: &mut MaybeUninit<Array2D<DataType, N, N>>,
    u1: &mut MaybeUninit<Array1D<DataType, N>>,
    v1: &mut MaybeUninit<Array1D<DataType, N>>,
    u2: &mut MaybeUninit<Array1D<DataType, N>>,
    v2: &mut MaybeUninit<Array1D<DataType, N>>,
    w: &mut MaybeUninit<Array1D<DataType, N>>,
    x: &mut MaybeUninit<Array1D<DataType, N>>,
    y: &mut MaybeUninit<Array1D<DataType, N>>,
    z: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let u1 = unsafe { u1.assume_init_mut() };
    let v1 = unsafe { v1.assume_init_mut() };
    let u2 = unsafe { u2.assume_init_mut() };
    let v2 = unsafe { v2.assume_init_mut() };
    let w = unsafe { w.assume_init_mut() };
    let x = unsafe { x.assume_init_mut() };
    let y = unsafe { y.assume_init_mut() };
    let z = unsafe { z.assume_init_mut() };

    *alpha = 1.5;
    *beta = 1.2;

    let float_n = n as DataType;

    for i in 0..n {
        u1[i] = i as DataType;
        u2[i] = ((i + 1) as DataType / float_n) / 2.0;
        v1[i] = ((i + 1) as DataType / float_n) / 4.0;
        v2[i] = ((i + 1) as DataType / float_n) / 6.0;
        y[i] = ((i + 1) as DataType / float_n) / 8.0;
        z[i] = ((i + 1) as DataType / float_n) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for j in 0..n {
            A[i][j] = (i * j % n) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_gemver<const N: usize>(
    n: usize,
    alpha: DataType,
    beta: DataType,
    A: &mut Array2D<DataType, N, N>,
    u1: &Array1D<DataType, N>,
    v1: &Array1D<DataType, N>,
    u2: &Array1D<DataType, N>,
    v2: &Array1D<DataType, N>,
    w: &mut Array1D<DataType, N>,
    x: &mut Array1D<DataType, N>,
    y: &Array1D<DataType, N>,
    z: &Array1D<DataType, N>,
) {
    for i in 0..n {
        for j in 0..n {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    for i in 0..n {
        for j in 0..n {
            x[i] = x[i] + beta * A[j][i] * y[j];
        }
    }

    for i in 0..n {
        x[i] = x[i] + z[i];
    }

    for i in 0..n {
        for j in 0..n {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut A = Array2D::<DataType, N, N>::maybe_uninit();
    let mut u1 = Array1D::<DataType, N>::maybe_uninit();
    let mut v1 = Array1D::<DataType, N>::maybe_uninit();
    let mut u2 = Array1D::<DataType, N>::maybe_uninit();
    let mut v2 = Array1D::<DataType, N>::maybe_uninit();
    let mut w = Array1D::<DataType, N>::maybe_uninit();
    let mut x = Array1D::<DataType, N>::maybe_uninit();
    let mut y = Array1D::<DataType, N>::maybe_uninit();
    let mut z = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(
            n, &mut alpha, &mut beta, &mut A, &mut u1, &mut v1, &mut u2, &mut v2, &mut w, &mut x,
            &mut y, &mut z,
        );

        let mut A = A.assume_init();
        let u1 = u1.assume_init();
        let v1 = v1.assume_init();
        let u2 = u2.assume_init();
        let v2 = v2.assume_init();
        let mut w = w.assume_init();
        let mut x = x.assume_init();
        let y = y.assume_init();
        let z = z.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || {
                kernel_gemver(
                    n, alpha, beta, &mut A, &u1, &v1, &u2, &v2, &mut w, &mut x, &y, &z,
                )
            },
            timing_function,
        );
        util::consume(w);
        elapsed
    }
}

#[test]
fn check() {}
