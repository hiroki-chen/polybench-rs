use alloc::vec::Vec;
use core::time::Duration;

/// An identity function that *__hints__* to the compiler to be maximally pessimistic about what
/// `black_box` could do.
///
/// Unlike [`std::convert::identity`], a Rust compiler is encouraged to assume that `black_box` can
/// use `dummy` in any possible valid way that Rust code is allowed to without introducing undefined
/// behavior in the calling code. This property makes `black_box` useful for writing code in which
/// certain optimizations are not desired, such as benchmarks.
///
/// Note however, that `black_box` is only (and can only be) provided on a "best-effort" basis. The
/// extent to which it can block optimisations may vary depending upon the platform and code-gen
/// backend used. Programs cannot rely on `black_box` for *correctness* in any way.
///
/// [`std::convert::identity`]: crate::convert::identity
#[inline(never)]
pub fn consume<T>(dummy: T) -> T {
    core::hint::black_box(dummy)
}

/// Optimize the locality by flushing the Last level cache (LLC),
/// which refers to the highest-numbered cache that is accessed by the cores prior to fetching from memory.
/// Things get different when we are inside the enclave.
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
pub fn benchmark_with_timing_function<F>(task: F, time_function: &dyn Fn() -> u64) -> Duration
where
    F: FnOnce(),
{
    flush_llc_cache();

    let begin = time_function();
    task();
    let end = time_function();
    Duration::from_nanos(end - begin)
}
