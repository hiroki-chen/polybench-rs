#![allow(non_snake_case)]

use crate::config::stencils::heat_3d::DataType;
use crate::ndarray::{Array3D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize, const TSTEPS: usize>(
    n: usize,
    A: &mut MaybeUninit<Array3D<DataType, N, N, N>>,
    B: &mut MaybeUninit<Array3D<DataType, N, N, N>>,
) {
    let A = unsafe { A.assume_init_mut() };
    let B = unsafe { B.assume_init_mut() };

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                B[i][j][k] = (i + j + (n - k)) as DataType * (10 as DataType) / n as DataType;
                A[i][j][k] = B[i][j][k];
            }
        }
    }
}

unsafe fn kernel_heat_3d<const N: usize, const TSTEPS: usize>(
    tsteps: usize,
    n: usize,
    A: &mut Array3D<DataType, N, N, N>,
    B: &mut Array3D<DataType, N, N, N>,
) {
    for _ in 1..tsteps {
        for i in 1..(n - 1) {
            for j in 1..(n - 1) {
                for k in 1..(n - 1) {
                    B[i][j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
                        + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
                        + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
                        + A[i][j][k];
                }
            }
        }
        for i in 1..(n - 1) {
            for j in 1..(n - 1) {
                for k in 1..(n - 1) {
                    A[i][j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
                        + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
                        + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
                        + B[i][j][k];
                }
            }
        }
    }
}

pub fn bench<const N: usize, const TSTEPS: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;
    let tsteps = TSTEPS;

    let mut A = Array3D::<DataType, N, N, N>::maybe_uninit();
    let mut B = Array3D::<DataType, N, N, N>::maybe_uninit();

    unsafe {
        init_array::<N, TSTEPS>(n, &mut A, &mut B);
        let mut A = A.assume_init();
        let mut B = B.assume_init();
        
        let elapsed = util::benchmark_with_timing_function(
            || kernel_heat_3d::<N, TSTEPS>(tsteps, n, &mut A, &mut B),
            timing_function,
        );
        util::consume(A);
        elapsed
    }
}

#[test]
fn check() {}
