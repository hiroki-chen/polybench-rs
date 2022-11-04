#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::doitgen::DataType;
use crate::ndarray::{Array1D, Array2D, Array3D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const NP: usize, const NQ: usize, const NR: usize>(
    nr: usize,
    nq: usize,
    np: usize,
    A: &mut MaybeUninit<Array3D<DataType, NR, NQ, NP>>,
    C4: &mut MaybeUninit<Array2D<DataType, NP, NP>>,
) {
    let a_mut = unsafe { A.assume_init_mut() };
    let c4_mut = unsafe { C4.assume_init_mut() };

    for i in 0..nr {
        for j in 0..nq {
            for k in 0..np {
                a_mut[i][j][k] = ((i * j + k) % np) as DataType / np as DataType;
            }
        }
    }
    for i in 0..np {
        for j in 0..np {
            c4_mut[i][j] = (i * j % np) as DataType / np as DataType;
        }
    }
}

unsafe fn kernel_doitgen<const NP: usize, const NQ: usize, const NR: usize>(
    nr: usize,
    nq: usize,
    np: usize,
    A: &mut Array3D<DataType, NR, NQ, NP>,
    C4: &Array2D<DataType, NP, NP>,
    sum: &mut Array1D<DataType, NP>,
) {
    for r in 0..nr {
        for q in 0..nq {
            for p in 0..np {
                sum[p] = 0.0;
                for s in 0..np {
                    sum[p] += A[r][q][s] * C4[s][p];
                }
            }
            for p in 0..np {
                A[r][q][p] = sum[p];
            }
        }
    }
}

pub fn bench<const NP: usize, const NQ: usize, const NR: usize>(
    timing_function: &dyn Fn() -> u64,
) -> Duration {
    let nr = NR;
    let nq = NQ;
    let np = NP;

    let mut A = Array3D::<DataType, NR, NQ, NP>::maybe_uninit();
    let sum = Array1D::<DataType, NP>::maybe_uninit();
    let mut C4 = Array2D::<DataType, NP, NP>::maybe_uninit();

    unsafe {
        init_array(nr, nq, np, &mut A, &mut C4);
        let mut A = A.assume_init();
        let C4 = C4.assume_init();
        let mut sum = sum.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_doitgen(nr, nq, np, &mut A, &C4, &mut sum),
            timing_function,
        );
        util::consume(A);
        elapsed
    }
}

#[test]
fn check() {}
