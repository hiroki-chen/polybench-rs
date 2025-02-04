#![allow(non_snake_case)]

use crate::config::linear_algebra::solvers::ludcmp::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(
    n: usize,
    A: &mut MaybeUninit<Array2D<DataType, N, N>>,
    b: &mut MaybeUninit<Array1D<DataType, N>>,
    x: &mut MaybeUninit<Array1D<DataType, N>>,
    y: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let b = unsafe { b.assume_init_mut() };
    let x = unsafe { x.assume_init_mut() };
    let y = unsafe { y.assume_init_mut() };

    let float_n = n as DataType;

    for i in 0..n {
        x[i] = 0.0;
        y[i] = 0.0;
        b[i] = (i + 1) as DataType / float_n / 2.0 + 4.0;
    }

    for i in 0..n {
        for j in 0..=i {
            A[i][j] = (-(j as isize) % n as isize) as DataType / n as DataType + 1.0;
        }
        for j in (i + 1)..n {
            A[i][j] = 0.0;
        }
        A[i][i] = 1.0;
    }

    A.make_positive_semi_definite();
}

unsafe fn kernel_ludcmp<const N: usize>(
    n: usize,
    A: &mut Array2D<DataType, N, N>,
    b: &Array1D<DataType, N>,
    x: &mut Array1D<DataType, N>,
    y: &mut Array1D<DataType, N>,
) {
    let mut w;
    for i in 0..n {
        for j in 0..i {
            w = A[i][j];
            for k in 0..j {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w / A[j][j];
        }
        for j in i..n {
            w = A[i][j];
            for k in 0..i {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w;
        }
    }

    for i in 0..n {
        w = b[i];
        for j in 0..i {
            w -= A[i][j] * y[j];
        }
        y[i] = w;
    }

    for i in (0..n).rev() {
        w = y[i];
        for j in (i + 1)..n {
            w -= A[i][j] * x[j];
        }
        x[i] = w / A[i][i];
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut A = Array2D::<DataType, N, N>::maybe_uninit();
    let mut b = Array1D::<DataType, N>::maybe_uninit();
    let mut x = Array1D::<DataType, N>::maybe_uninit();
    let mut y = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut A, &mut b, &mut x, &mut y);
        let b = b.assume_init();
        let mut A = A.assume_init();
        let mut x = x.assume_init();
        let mut y = y.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_ludcmp(n, &mut A, &b, &mut x, &mut y),
            timing_function,
        );
        util::consume(x);
        elapsed
    }
}

#[test]
fn check() {}
