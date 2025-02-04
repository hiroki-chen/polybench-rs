#![allow(non_snake_case)]

use crate::config::stencils::adi::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize, const TSTEPS: usize>(
    n: usize,
    u: &mut MaybeUninit<Array2D<DataType, N, N>>,
) {
    let u = unsafe { u.assume_init_mut() };

    for i in 0..n {
        for j in 0..n {
            u[i][j] = (i + n - j) as DataType / n as DataType;
        }
    }
}

unsafe fn kernel_adi<const N: usize, const TSTEPS: usize>(
    tsteps: usize,
    n: usize,
    u: &mut Array2D<DataType, N, N>,
    v: &mut Array2D<DataType, N, N>,
    p: &mut Array2D<DataType, N, N>,
    q: &mut Array2D<DataType, N, N>,
) {
    let DX = 1.0 / n as DataType;
    let DY = 1.0 / n as DataType;
    let DT = 1.0 / tsteps as DataType;
    let B1 = 2.0;
    let B2 = 1.0;
    let mul1 = B1 * DT / (DX * DX);
    let mul2 = B2 * DT / (DY * DY);

    let a = -mul1 / 2.0;
    let b = 1.0 + mul1;
    let c = a;
    let d = -mul2 / 2.0;
    let e = 1.0 + mul2;
    let f = d;

    for _ in 1..tsteps {
        for i in 1..(n - 1) {
            v[0][i] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = v[0][i];
            for j in 1..(n - 1) {
                p[i][j] = -c / (a * p[i][j - 1] + b);
                q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i]
                    - f * u[j][i + 1]
                    - a * q[i][j - 1])
                    / (a * p[i][j - 1] + b);
            }

            v[n - 1][i] = 1.0;
            for j in (1..(n - 1)).rev() {
                v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
            }
        }

        for i in 1..(n - 1) {
            u[i][0] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = u[i][0];
            for j in 1..(n - 1) {
                p[i][j] = -f / (d * p[i][j - 1] + e);
                q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j]
                    - c * v[i + 1][j]
                    - d * q[i][j - 1])
                    / (d * p[i][j - 1] + e);
            }
            u[i][n - 1] = 1.0;
            for j in (1..(n - 1)).rev() {
                u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
            }
        }
    }
}

pub fn bench<const N: usize, const TSTEPS: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;
    let tsteps = TSTEPS;

    let mut u = Array2D::<DataType, N, N>::maybe_uninit();
    let v = Array2D::<DataType, N, N>::maybe_uninit();
    let p = Array2D::<DataType, N, N>::maybe_uninit();
    let q = Array2D::<DataType, N, N>::maybe_uninit();

    unsafe {
        init_array::<N, TSTEPS>(n, &mut u);
        let mut u = u.assume_init();
        let mut v = v.assume_init();
        let mut p = p.assume_init();
        let mut q = q.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_adi::<N, TSTEPS>(tsteps, n, &mut u, &mut v, &mut p, &mut q),
            timing_function,
        );
        util::consume(u);
        elapsed
    }
}

#[test]
fn check() {}
