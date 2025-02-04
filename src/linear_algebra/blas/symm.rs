#![allow(non_snake_case)]

use crate::config::linear_algebra::blas::symm::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: &mut DataType,
    beta: &mut DataType,
    C: &mut MaybeUninit<Array2D<DataType, M, N>>,
    A: &mut MaybeUninit<Array2D<DataType, M, M>>,
    B: &mut MaybeUninit<Array2D<DataType, M, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let B = unsafe { B.assume_init_mut() };
    let C = unsafe { C.assume_init_mut() };

    *alpha = 1.5;
    *beta = 1.2;
    for i in 0..m {
        for j in 0..n {
            C[i][j] = ((i + j) % 100) as DataType / m as DataType;
            B[i][j] = ((n + i - j) % 100) as DataType / m as DataType;
        }
    }

    for i in 0..m {
        for j in 0..=i {
            A[i][j] = ((i + j) % 100) as DataType / m as DataType;
        }
        for j in (i + 1)..m {
            A[i][j] = -999 as DataType;
        }
    }
}

unsafe fn kernel_symm<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: DataType,
    beta: DataType,
    C: &mut Array2D<DataType, M, N>,
    A: &Array2D<DataType, M, M>,
    B: &Array2D<DataType, M, N>,
) {
    for i in 0..m {
        for j in 0..n {
            let mut temp2 = 0.0;
            for k in 0..i {
                C[k][j] += alpha * B[i][j] * A[i][k];
                temp2 += B[k][j] * A[i][k];
            }
            C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
        }
    }
}

pub fn bench<const M: usize, const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let m = M;
    let n = N;

    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut C = Array2D::<DataType, M, N>::maybe_uninit();
    let mut A = Array2D::<DataType, M, M>::maybe_uninit();
    let mut B = Array2D::<DataType, M, N>::maybe_uninit();

    unsafe {
        init_array(m, n, &mut alpha, &mut beta, &mut C, &mut A, &mut B);
        let A = A.assume_init();
        let B = B.assume_init();
        let mut C = C.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_symm(m, n, alpha, beta, &mut C, &A, &B),
            timing_function,
        );
        util::consume(C);
        elapsed
    }
}

#[test]
fn check() {}
