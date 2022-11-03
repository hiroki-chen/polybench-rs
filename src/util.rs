use alloc::vec::Vec;
use core::fmt;
use core::time::Duration;

pub fn consume<T: fmt::Display>(dummy: T) -> T {
    #[cfg(feature = "print-result")]
    println!("{}", &dummy);

    unsafe {
        // Taken from bencher crate:
        // https://docs.rs/bencher/0.1.5/src/bencher/lib.rs.html#590-596
        let ret = core::ptr::read_volatile(&dummy);
        core::mem::forget(dummy);
        ret
    }
}

fn flush_llc_cache() {
    const LLC_CACHE_SIZE: usize = 32 * 1024 * 1024; // 32 MiB
    const NUM_ELEMS: usize = (LLC_CACHE_SIZE - 1) / core::mem::size_of::<usize>() + 1;

    let mut buf: Vec<usize> = Vec::with_capacity(NUM_ELEMS);
    buf.resize(NUM_ELEMS, Default::default());
    let sum: usize = buf.iter().sum();
    consume(sum);
}

/// Avoid invoking this interface.
#[inline(always)]
pub fn benchmark<F: FnOnce()>(f: F) -> Duration {
    flush_llc_cache();

    f();
    Duration::from_secs(0)
}

/// The target platform does not necessarily support std, so the time function should be given
/// by the user manually as a closure.
#[inline(always)]
pub fn benchmark_with_timing_function<F1, F2>(task: F1, mut time_function: F2) -> Duration
where
    F1: FnOnce(),
    F2: FnMut() -> u64,
{
    flush_llc_cache();

    let begin = time_function();
    task();
    let end = time_function();
    Duration::from_secs(end - begin)
}
