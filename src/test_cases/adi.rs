use polybench_rs::stencils::adi::bench;

fn bench_and_print<const N: usize, const TSTEPS: usize>() {
    let foo = || 0;

    let dims = format!("{:?}", (N, TSTEPS));
    let elapsed = bench::<N, TSTEPS>(|| 0).as_secs_f64();
    println!("{:<14} | {:<30} | {:.7} s", "adi", dims, elapsed);
}

fn main() {
    bench_and_print::<250, 125>();
    bench_and_print::<500, 250>();
    bench_and_print::<1000, 500>();
}
