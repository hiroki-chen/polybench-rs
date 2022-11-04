#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::mvt::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(
    n: usize,
    x1: &mut MaybeUninit<Array1D<DataType, N>>,
    x2: &mut MaybeUninit<Array1D<DataType, N>>,
    y_1: &mut MaybeUninit<Array1D<DataType, N>>,
    y_2: &mut MaybeUninit<Array1D<DataType, N>>,
    A: &mut MaybeUninit<Array2D<DataType, N, N>>,
) {
    let x1 = unsafe { x1.assume_init_mut() };
    let x2 = unsafe { x2.assume_init_mut() };
    let y_1 = unsafe { y_1.assume_init_mut() };
    let y_2 = unsafe { y_2.assume_init_mut() };
    let A = unsafe { A.assume_init_mut() };

    for i in 0..n {
        x1[i] = (i % n) as DataType / n as DataType;
        x2[i] = ((i + 1) % n) as DataType / n as DataType;
        y_1[i] = ((i + 3) % n) as DataType / n as DataType;
        y_2[i] = ((i + 4) % n) as DataType / n as DataType;
        for j in 0..n {
            A[i][j] = (i * j % n) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_mvt<const N: usize>(
    n: usize,
    x1: &mut Array1D<DataType, N>,
    x2: &mut Array1D<DataType, N>,
    y_1: &Array1D<DataType, N>,
    y_2: &Array1D<DataType, N>,
    A: &Array2D<DataType, N, N>,
) {
    for i in 0..n {
        for j in 0..n {
            x1[i] = x1[i] + A[i][j] * y_1[j];
        }
    }
    for i in 0..n {
        for j in 0..n {
            x2[i] = x2[i] + A[j][i] * y_2[j];
        }
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut A = Array2D::<DataType, N, N>::maybe_uninit();
    let mut x1 = Array1D::<DataType, N>::maybe_uninit();
    let mut x2 = Array1D::<DataType, N>::maybe_uninit();
    let mut y_1 = Array1D::<DataType, N>::maybe_uninit();
    let mut y_2 = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut x1, &mut x2, &mut y_1, &mut y_2, &mut A);

        let mut x1 = x1.assume_init();
        let mut x2 = x2.assume_init();
        let y_1 = y_1.assume_init();
        let y_2 = y_2.assume_init();
        let A = A.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_mvt(n, &mut x1, &mut x2, &y_1, &y_2, &A),
            timing_function,
        );
        util::consume(x1);
        util::consume(x2);
        elapsed
    }
}

#[test]
fn check() {}
