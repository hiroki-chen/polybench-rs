#![allow(non_snake_case)]

use crate::config::linear_algebra::blas::trmm::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: &mut DataType,
    A: &mut MaybeUninit<Array2D<DataType, M, M>>,
    B: &mut MaybeUninit<Array2D<DataType, M, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let B = unsafe { B.assume_init_mut() };

    *alpha = 1.5;
    for i in 0..m {
        for j in 0..i {
            A[i][j] = ((i + j) % m) as DataType / m as DataType;
        }
        A[i][i] = 1.0;
        for j in 0..n {
            B[i][j] = (((n + i) - j) % n) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_trmm<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    alpha: DataType,
    A: &Array2D<DataType, M, M>,
    B: &mut Array2D<DataType, M, N>,
) {
    for i in 0..m {
        for j in 0..n {
            for k in (i + 1)..m {
                B[i][j] += A[k][i] * B[k][j];
            }
            B[i][j] = alpha * B[i][j];
        }
    }
}

pub fn bench<const M: usize, const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let m = M;
    let n = N;

    let mut alpha = 0.0;
    let mut A = Array2D::<DataType, M, M>::maybe_uninit();
    let mut B = Array2D::<DataType, M, N>::maybe_uninit();

    unsafe {
        init_array(m, n, &mut alpha, &mut A, &mut B);
        let A = A.assume_init();
        let mut B = B.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_trmm(m, n, alpha, &A, &mut B),
            timing_function,
        );
        util::consume(B);
        elapsed
    }
}

#[test]
fn check() {}
