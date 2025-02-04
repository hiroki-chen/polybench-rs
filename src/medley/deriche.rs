use crate::config::medley::deriche::DataType;
use crate::ndarray::{Array2D, ArrayAlloc};
use crate::util;
use core::mem::MaybeUninit;
use core::time::Duration;

unsafe fn init_array<const H: usize, const W: usize>(
    w: usize,
    h: usize,
    alpha: &mut DataType,
    img_in: &mut MaybeUninit<Array2D<DataType, W, H>>,
) {
    let img_in = unsafe { img_in.assume_init_mut() };

    *alpha = 0.25;
    for i in 0..w {
        for j in 0..h {
            img_in[i][j] = ((313 * i + 991 * j) % 65536) as DataType / 65535.0;
        }
    }
}

unsafe fn kernel_deriche<const H: usize, const W: usize>(
    w: usize,
    h: usize,
    alpha: DataType,
    img_in: &Array2D<DataType, W, H>,
    img_out: &mut Array2D<DataType, W, H>,
    y1: &mut Array2D<DataType, W, H>,
    y2: &mut Array2D<DataType, W, H>,
) {
    let mut xm1;
    let mut tm1;
    let mut ym1;
    let mut ym2;
    let mut xp1;
    let mut xp2;
    let mut tp1;
    let mut tp2;
    let mut yp1;
    let mut yp2;

    let k = (1.0 - (-alpha).exp()) * (1.0 - (-alpha).exp())
        / (1.0 + 2.0 * alpha * (-alpha).exp() - ((2.0) * alpha).exp());
    let a5 = k;
    let a1 = a5;
    let a6 = k * (-alpha).exp() * (alpha - 1.0);
    let a2 = a6;
    let a7 = k * (-alpha).exp() * (alpha + 1.0);
    let a3 = a7;
    let a8 = -k * (-2.0 * alpha).exp();
    let a4 = a8;
    let b1 = (2.0 as DataType).powf(-alpha);
    let b2 = -(-2.0 * alpha).exp();
    let c2 = 1.0;
    let c1 = c2;

    for i in 0..w {
        ym1 = 0.0;
        ym2 = 0.0;
        xm1 = 0.0;
        for j in 0..h {
            y1[i][j] = a1 * img_in[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = img_in[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for i in 0..w {
        yp1 = 0.0;
        yp2 = 0.0;
        xp1 = 0.0;
        xp2 = 0.0;
        for j in (0..h).rev() {
            y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = img_in[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for i in 0..w {
        for j in 0..h {
            img_out[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    }

    for j in 0..h {
        tm1 = 0.0;
        ym1 = 0.0;
        ym2 = 0.0;
        for i in 0..w {
            y1[i][j] = a5 * img_out[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = img_out[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for j in 0..h {
        tp1 = 0.0;
        tp2 = 0.0;
        yp1 = 0.0;
        yp2 = 0.0;
        for i in (0..w).rev() {
            y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = img_out[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for i in 0..w {
        for j in 0..h {
            img_out[i][j] = c2 * (y1[i][j] + y2[i][j]);
        }
    }
}

pub fn bench<const H: usize, const W: usize>(timing_function: &dyn Fn() -> u64) -> Duration {
    let w = W;
    let h = H;

    let mut alpha = 0.0;
    let mut img_in = Array2D::<DataType, W, H>::maybe_uninit();
    let img_out = Array2D::<DataType, W, H>::maybe_uninit();
    let y1 = Array2D::<DataType, W, H>::maybe_uninit();
    let y2 = Array2D::<DataType, W, H>::maybe_uninit();

    unsafe {
        init_array(w, h, &mut alpha, &mut img_in);
        let img_in = img_in.assume_init();
        let mut img_out = img_out.assume_init();
        let mut y1 = y1.assume_init();
        let mut y2 = y2.assume_init();

        let elapsed = util::benchmark_with_timing_function(
            || kernel_deriche(w, h, alpha, &img_in, &mut img_out, &mut y1, &mut y2),
            timing_function,
        );
        util::consume(img_out);
        elapsed
    }
}

#[test]
fn check() {}
