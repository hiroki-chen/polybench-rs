#![allow(non_snake_case)]

use crate::config::linear_algebra::blas::syrk::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: &mut DataType,
    beta: &mut DataType,
    C: &mut MaybeUninit<Array2D<DataType, M, M>>,
    A: &mut MaybeUninit<Array2D<DataType, M, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let C = unsafe { C.assume_init_mut() };

    *alpha = 1.5;
    *beta = 1.2;
    for i in 0..m {
        for j in 0..n {
            A[i][j] = ((i * j + 1) % m) as DataType / m as DataType;
        }
    }

    for i in 0..m {
        for j in 0..m {
            C[i][j] = ((i * j + 2) % n) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_syrk<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: DataType,
    beta: DataType,
    C: &mut Array2D<DataType, M, M>,
    A: &Array2D<DataType, M, N>,
) {
    for i in 0..m {
        for j in 0..=i {
            C[i][j] *= beta;
        }
        for k in 0..n {
            for j in 0..=i {
                C[i][j] += alpha * A[i][k] * A[j][k];
            }
        }
    }
}

pub fn bench<const M: usize, const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let m = M;
    let n = N;

    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut C = Array2D::<DataType, M, M>::maybe_uninit();
    let mut A = Array2D::<DataType, M, N>::maybe_uninit();

    unsafe {
        init_array(m, n, &mut alpha, &mut beta, &mut C, &mut A);
        let A = A.assume_init();
        let mut C = C.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_syrk(m, n, alpha, beta, &mut C, &A),
            timing_function,
        );
        util::consume(C);
        elapsed
    }
}

#[test]
fn check() {}
