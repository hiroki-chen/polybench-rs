use crate::config::medley::floyd_warshall::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const N: usize>(n: usize, path: &mut MaybeUninit<Array2D<DataType, N, N>>) {
    let path = unsafe { path.assume_init_mut() };

    for i in 0..n {
        for j in 0..n {
            path[i][j] = (i * j % 7 + 1) as DataType;
            if (i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0 {
                path[i][j] = 999 as DataType;
            }
        }
    }
}

unsafe fn kernel_floyd_warshall<const N: usize>(n: usize, path: &mut Array2D<DataType, N, N>) {
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                path[i][j] = if path[i][j] < path[i][k] + path[k][j] {
                    path[i][j]
                } else {
                    path[i][k] + path[k][j]
                };
            }
        }
    }
}

pub fn bench<const N: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let n = N;

    let mut path = Array2D::<DataType, N, N>::maybe_uninit();

    unsafe {
        init_array(n, &mut path);
        let mut path = path.assume_init();
        
        let elapsed = util::benchmark_with_timing_function(
            || kernel_floyd_warshall(n, &mut path),
            timing_function,
        );
        util::consume(path);
        elapsed
    }
}

#[test]
fn check() {}
