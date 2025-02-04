#![allow(non_snake_case)]

use crate::config::linear_algebra::solvers::lu::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(n: usize, A: &mut MaybeUninit<Array2D<DataType, N, N>>) {
    let A = unsafe { A.assume_init_mut() };

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

unsafe fn kernel_lu<const N: usize>(n: usize, A: &mut Array2D<DataType, N, N>) {
    for i in 0..n {
        for j in 0..i {
            for k in 0..j {
                A[i][j] -= A[i][k] * A[k][j];
            }
            A[i][j] /= A[j][j];
        }
        for j in i..n {
            for k in 0..i {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut A = Array2D::<DataType, N, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut A);
        let mut A = A.assume_init();
        
        let elapsed =
            util::benchmark_with_timing_function(|| kernel_lu(n, &mut A), timing_function);
        util::consume(A);
        elapsed
    }
}

#[test]
fn check() {}
