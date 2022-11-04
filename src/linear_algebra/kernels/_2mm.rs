#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::_2mm::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const NI: usize, const NJ: usize, const NK: usize, const NL: usize>(
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: &mut DataType,
    beta: &mut DataType,
    A: &mut MaybeUninit<Array2D<DataType, NI, NK>>,
    B: &mut MaybeUninit<Array2D<DataType, NK, NJ>>,
    C: &mut MaybeUninit<Array2D<DataType, NJ, NL>>,
    D: &mut MaybeUninit<Array2D<DataType, NI, NL>>,
) {
    *alpha = 1.5;
    *beta = 1.2;

    let a_mut = unsafe { A.assume_init_mut() };
    let b_mut = unsafe { B.assume_init_mut() };
    let c_mut = unsafe { C.assume_init_mut() };
    let d_mut = unsafe { D.assume_init_mut() };

    for i in 0..ni {
        for j in 0..nk {
            a_mut[i][j] = ((i * j + 1) % ni) as DataType / ni as DataType;
        }
    }
    for i in 0..nk {
        for j in 0..nj {
            b_mut[i][j] = (i * (j + 1) % nj) as DataType / nj as DataType;
        }
    }
    for i in 0..nj {
        for j in 0..nl {
            c_mut[i][j] = ((i * (j + 3) + 1) % nl) as DataType / nl as DataType;
        }
    }
    for i in 0..ni {
        for j in 0..nl {
            d_mut[i][j] = (i * (j + 2) % nk) as DataType / nk as DataType;
        }
    }
}

unsafe fn kernel_2mm<const NI: usize, const NJ: usize, const NK: usize, const NL: usize>(
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: DataType,
    beta: DataType,
    tmp: &mut Array2D<DataType, NI, NJ>,
    A: &Array2D<DataType, NI, NK>,
    B: &Array2D<DataType, NK, NJ>,
    C: &Array2D<DataType, NJ, NL>,
    D: &mut Array2D<DataType, NI, NL>,
) {
    for i in 0..ni {
        for j in 0..nj {
            tmp[i][j] = 0.0;
            for k in 0..nk {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
    for i in 0..ni {
        for j in 0..nl {
            D[i][j] *= beta;
            for k in 0..nj {
                D[i][j] += tmp[i][k] * C[k][j];
            }
        }
    }
}

pub fn bench<const NI: usize, const NJ: usize, const NK: usize, const NL: usize>(
    timing_function: &dyn Fn() -> u64,
) -> Duration {
    let ni = NI;
    let nj = NJ;
    let nk = NK;
    let nl = NL;

    let mut alpha = 0.0;
    let mut beta = 0.0;
    let mut A = Array2D::<DataType, NI, NK>::maybe_uninit();
    let mut B = Array2D::<DataType, NK, NJ>::maybe_uninit();
    let mut C = Array2D::<DataType, NJ, NL>::maybe_uninit();
    let mut D = Array2D::<DataType, NI, NL>::maybe_uninit();

    unsafe {
        init_array(
            ni, nj, nk, nl, &mut alpha, &mut beta, &mut A, &mut B, &mut C, &mut D,
        );

        let A = A.assume_init();
        let B = B.assume_init();
        let C = C.assume_init();
        let D = D.assume_init_mut();
        let mut tmp = Array2D::<DataType, NI, NJ>::maybe_uninit();
        let tmp = tmp.assume_init_mut();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, &A, &B, &C, D),
            timing_function,
        );
        util::consume(D);
        elapsed
    }
}

#[test]
fn check() {}
