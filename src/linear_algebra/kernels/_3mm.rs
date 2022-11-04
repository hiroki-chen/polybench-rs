#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::_3mm::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<
    const NI: usize,
    const NJ: usize,
    const NK: usize,
    const NL: usize,
    const NM: usize,
>(
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    nm: usize,
    A: &mut MaybeUninit<Array2D<DataType, NI, NK>>,
    B: &mut MaybeUninit<Array2D<DataType, NK, NJ>>,
    C: &mut MaybeUninit<Array2D<DataType, NJ, NM>>,
    D: &mut MaybeUninit<Array2D<DataType, NM, NL>>,
) {
    let a_mut = unsafe { A.assume_init_mut() };
    let b_mut = unsafe { B.assume_init_mut() };
    let c_mut = unsafe { C.assume_init_mut() };
    let d_mut = unsafe { D.assume_init_mut() };

    for i in 0..ni {
        for j in 0..nk {
            a_mut[i][j] = ((i * j + 1) % ni) as DataType / (5 * ni) as DataType;
        }
    }
    for i in 0..nk {
        for j in 0..nj {
            b_mut[i][j] = ((i * (j + 1) + 2) % nj) as DataType / (5 * nj) as DataType;
        }
    }
    for i in 0..nj {
        for j in 0..nm {
            c_mut[i][j] = (i * (j + 3) % nl) as DataType / (5 * nl) as DataType;
        }
    }
    for i in 0..nm {
        for j in 0..nl {
            d_mut[i][j] = ((i * (j + 2) + 2) % nk) as DataType / (5 * nk) as DataType;
        }
    }
}

unsafe fn kernel_3mm<
    const NI: usize,
    const NJ: usize,
    const NK: usize,
    const NL: usize,
    const NM: usize,
>(
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    nm: usize,
    E: &mut Array2D<DataType, NI, NJ>,
    A: &Array2D<DataType, NI, NK>,
    B: &Array2D<DataType, NK, NJ>,
    F: &mut Array2D<DataType, NJ, NL>,
    C: &Array2D<DataType, NJ, NM>,
    D: &Array2D<DataType, NM, NL>,
    G: &mut Array2D<DataType, NI, NL>,
) {
    for i in 0..ni {
        for j in 0..nj {
            E[i][j] = 0.0;
            for k in 0..nk {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    for i in 0..nj {
        for j in 0..nl {
            F[i][j] = 0.0;
            for k in 0..nm {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }
    for i in 0..ni {
        for j in 0..nl {
            G[i][j] = 0.0;
            for k in 0..nj {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}

pub fn bench<
    const NI: usize,
    const NJ: usize,
    const NK: usize,
    const NL: usize,
    const NM: usize,
>(
    timing_function: &dyn Fn() -> u64,
) -> Duration {
    let ni = NI;
    let nj = NJ;
    let nk = NK;
    let nl = NL;
    let nm = NM;

    let E = Array2D::<DataType, NI, NJ>::maybe_uninit();
    let mut A = Array2D::<DataType, NI, NK>::maybe_uninit();
    let mut B = Array2D::<DataType, NK, NJ>::maybe_uninit();
    let F = Array2D::<DataType, NJ, NL>::maybe_uninit();
    let mut C = Array2D::<DataType, NJ, NM>::maybe_uninit();
    let mut D = Array2D::<DataType, NM, NL>::maybe_uninit();
    let G = Array2D::<DataType, NI, NL>::maybe_uninit();

    unsafe {
        init_array(ni, nj, nk, nl, nm, &mut A, &mut B, &mut C, &mut D);
        let A = A.assume_init();
        let B = B.assume_init();
        let C = C.assume_init();
        let D = D.assume_init();
        let mut E = E.assume_init();
        let mut F = F.assume_init();
        let mut G = G.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_3mm(ni, nj, nk, nl, nm, &mut E, &A, &B, &mut F, &C, &D, &mut G),
            timing_function,
        );
        util::consume(G);
        elapsed
    }
}

#[test]
fn check() {}
