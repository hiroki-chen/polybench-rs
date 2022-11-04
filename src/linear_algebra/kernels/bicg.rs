#![allow(non_snake_case)]

use crate::config::linear_algebra::kernels::bicg::DataType;
use crate::ndarray::{Array1D, Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    A: &mut MaybeUninit<Array2D<DataType, M, N>>,
    r: &mut MaybeUninit<Array1D<DataType, M>>,
    p: &mut MaybeUninit<Array1D<DataType, N>>,
) {
    let a_mut = unsafe { A.assume_init_mut() };
    let r_mut = unsafe { r.assume_init_mut() };
    let p_mut = unsafe { p.assume_init_mut() };

    for i in 0..n {
        p_mut[i] = (i % n) as DataType / n as DataType;
    }
    for i in 0..m {
        r_mut[i] = (i % m) as DataType / m as DataType;
        for j in 0..n {
            a_mut[i][j] = (i * (j + 1) % m) as DataType / m as DataType;
        }
    }
}

unsafe fn kernel_bicg<const M: usize, const N: usize>(
    m: usize,
    n: usize,
    A: &Array2D<DataType, M, N>,
    s: &mut Array1D<DataType, N>,
    q: &mut Array1D<DataType, M>,
    p: &Array1D<DataType, N>,
    r: &Array1D<DataType, M>,
) {
    for i in 0..n {
        s[i] = 0.0;
    }
    for i in 0..m {
        q[i] = 0.0;
        for j in 0..n {
            s[j] = s[j] + r[i] * A[i][j];
            q[i] = q[i] + A[i][j] * p[j];
        }
    }
}

pub fn bench<const M: usize, const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let m = M;
    let n = N;

    let mut A = Array2D::<DataType, M, N>::maybe_uninit();
    let s = Array1D::<DataType, N>::maybe_uninit();
    let q = Array1D::<DataType, M>::maybe_uninit();
    let mut p = Array1D::<DataType, N>::maybe_uninit();
    let mut r = Array1D::<DataType, M>::maybe_uninit();

    unsafe {
        init_array(m, n, &mut A, &mut r, &mut p);
        let A = A.assume_init_mut();
        let mut s = s.assume_init();
        let r = r.assume_init();
        let mut q = q.assume_init();
        let p = p.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_bicg(m, n, &A, &mut s, &mut q, &p, &r),
            timing_function,
        );
        util::consume(s);
        util::consume(q);
        elapsed
    }
}

#[test]
fn check() {}
