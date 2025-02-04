use crate::config::linear_algebra::solvers::durbin::DataType;
use crate::ndarray::{Array1D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(n: usize, r: &mut MaybeUninit<Array1D<DataType, N>>) {
    let r = unsafe { r.assume_init_mut() };

    for i in 0..n {
        r[i] = (n + 1 - i) as DataType;
    }
}

unsafe fn kernel_durbin<const N: usize>(
    n: usize,
    r: &Array1D<DataType, N>,
    y: &mut Array1D<DataType, N>,
) {
    let mut z: [DataType; N] = MaybeUninit::uninit().assume_init();

    y[0] = -r[0];
    let mut beta = 1.0;
    let mut alpha = -r[0];
    for k in 1..n {
        beta = (1.0 - alpha * alpha) * beta;
        let mut sum = 0.0;
        for i in 0..k {
            sum += r[k - i - 1] * y[i];
        }
        alpha = -(r[k] + sum) / beta;

        for i in 0..k {
            z[i] = y[i] + alpha * y[k - i - 1];
        }
        for i in 0..k {
            y[i] = z[i];
        }
        y[k] = alpha;
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut r = Array1D::<DataType, N>::maybe_uninit();
    let y = Array1D::<DataType, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut r);
        let r = r.assume_init();
        let mut y = y.assume_init();

        let elapsed =
            util::benchmark_with_timing_function(|| kernel_durbin(n, &r, &mut y), timing_function);
        util::consume(y);
        elapsed
    }
}

#[test]
fn check() {}
