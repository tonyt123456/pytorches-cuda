//! Compares the CUDA plugin against the CPU plugin through the core Tensor API.
//! Skips (passes) when no CUDA device is available. The CUDA plugin is loaded as the built shared
//! library; the CPU plugin is linked in.

use pytorches_core::plugin::{load_plugin_file, register_static};
use pytorches_core::{Device, Tensor};
use pytorches_plugin_abi as abi;
use std::path::PathBuf;
use std::sync::{Once, OnceLock};
use std::time::Instant;

/// The plugin cdylib. `cargo test` does not build a package's cdylib, so build it on first use into
/// the same target directory (`target/<profile>/`, one level above the test binary's `deps/`).
fn cuda_dll() -> PathBuf {
    let name = format!(
        "{}pytorches_plugin_cuda.{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_EXTENSION
    );
    let profile_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let dll = profile_dir.join(&name);
    if !dll.exists() {
        let mut build =
            std::process::Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
        build
            .args(["build", "--lib", "--manifest-path"])
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        if profile_dir.file_name().is_some_and(|n| n == "release") {
            build.arg("--release");
        }
        assert!(
            build.status().expect("run cargo").success(),
            "building the plugin failed"
        );
    }
    dll
}

/// Loads both plugins once; returns (cpu, cuda) devices, or None when CUDA is unavailable.
fn devices() -> Option<(Device, Device)> {
    static INIT: Once = Once::new();
    static OK: OnceLock<bool> = OnceLock::new();
    INIT.call_once(|| {
        register_static(pytorches_plugin_cpu::pytorches_plugin_entry).expect("cpu plugin");
        let ok = match load_plugin_file(&cuda_dll()) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("SKIP: cuda plugin unavailable: {e}");
                false
            }
        };
        OK.set(ok).unwrap();
    });
    if !*OK.get().unwrap() {
        return None;
    }
    Some((
        Device::parse("cpu:0").unwrap(),
        Device::parse("cuda:0").ok()?,
    ))
}

macro_rules! need_cuda {
    () => {
        match devices() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        }
    };
}

/// Deterministic pseudo-random data in [-1, 1).
fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8388608.0) - 1.0
        })
        .collect()
}

fn assert_close(what: &str, got: &[f32], want: &[f32], rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = rtol * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol || (g.is_nan() && w.is_nan()),
            "{what}[{i}]: cuda {g} vs cpu {w}"
        );
    }
}

fn pair(shape: &[usize], seed: u64, cpu: &Device, gpu: &Device) -> (Tensor, Tensor) {
    let d = data(shape.iter().product(), seed);
    (
        Tensor::from_vec_on(d.clone(), shape.to_vec(), cpu),
        Tensor::from_vec_on(d, shape.to_vec(), gpu),
    )
}

#[test]
fn broadcasting_binary_ops() {
    let (cpu, gpu) = need_cuda!();
    let cases: [(&[usize], &[usize]); 7] = [
        (&[2, 3], &[3]),
        (&[4, 1], &[1, 5]),
        (&[2, 3, 4], &[3, 1]),
        (&[2, 1, 4], &[1, 3, 1]),
        (&[1], &[5, 7]),
        (&[], &[3, 3]),
        (&[3, 33, 17], &[3, 33, 17]),
    ];
    for (i, (sa, sb)) in cases.iter().enumerate() {
        let (ac, ag) = pair(sa, 10 + i as u64, &cpu, &gpu);
        let (bc, bg) = pair(sb, 50 + i as u64, &cpu, &gpu);
        // Keep divisors away from zero.
        let (bc, bg) = (
            bc.add(&Tensor::scalar_on(2.0, &cpu)),
            bg.add(&Tensor::scalar_on(2.0, &gpu)),
        );
        for (name, f) in [
            ("add", Tensor::add as fn(&Tensor, &Tensor) -> Tensor),
            ("sub", Tensor::sub),
            ("mul", Tensor::mul),
            ("div", Tensor::div),
        ] {
            let c = f(&ac, &bc);
            let g = f(&ag, &bg);
            assert_eq!(c.shape(), g.shape());
            assert_close(
                &format!("{name} {sa:?}{sb:?}"),
                &g.to_vec(),
                &c.to_vec(),
                1e-6,
            );
        }
    }
}

/// Operand layouts for the 2-D broadcast kernel (inner dim of at least 32 with stride 0 or 1): a bias row,
/// a column, leading dims that merge, and layouts that must fall back to the general kernel.
#[test]
fn broadcasting_two_d_paths() {
    let (cpu, gpu) = need_cuda!();
    let cases: [(&[usize], &[usize]); 7] = [
        (&[100, 64], &[64]),
        (&[100, 64], &[100, 1]),
        (&[64], &[100, 64]),
        (&[3, 50, 64], &[64]),
        (&[3, 50, 64], &[50, 1]), // inner stride 0 for one operand, a broadcast middle dim: general kernel
        (&[1, 100], &[64, 100]),
        (&[7, 1000], &[1, 1000]),
    ];
    for (i, (sa, sb)) in cases.iter().enumerate() {
        let (ac, ag) = pair(sa, 70 + i as u64, &cpu, &gpu);
        let (bc, bg) = pair(sb, 90 + i as u64, &cpu, &gpu);
        let (bc, bg) = (
            bc.add(&Tensor::scalar_on(2.0, &cpu)),
            bg.add(&Tensor::scalar_on(2.0, &gpu)),
        );
        for (name, f) in [
            ("add", Tensor::add as fn(&Tensor, &Tensor) -> Tensor),
            ("sub", Tensor::sub),
            ("mul", Tensor::mul),
            ("div", Tensor::div),
        ] {
            assert_close(
                &format!("{name} {sa:?}{sb:?}"),
                &f(&ag, &bg).to_vec(),
                &f(&ac, &bc).to_vec(),
                1e-6,
            );
        }
    }
    // A transposed operand (inner stride != 1) against a plain one.
    let (ac, ag) = pair(&[100, 64], 5, &cpu, &gpu);
    let (bc, bg) = pair(&[64, 100], 6, &cpu, &gpu);
    assert_close(
        "t + plain",
        &ag.t().add(&bg).to_vec(),
        &ac.t().add(&bc).to_vec(),
        1e-6,
    );
}

#[test]
fn transpose_strided_copy() {
    let (cpu, gpu) = need_cuda!();
    for shape in [[2usize, 3], [33, 65], [1, 7], [100, 1], [257, 129]] {
        let (c, g) = pair(&shape, 7, &cpu, &gpu);
        assert_eq!(g.t().to_vec(), c.t().to_vec(), "transpose {shape:?}");
        assert_eq!(g.t().t().to_vec(), c.to_vec());
    }
}

#[test]
fn unary_ops() {
    let (cpu, gpu) = need_cuda!();
    let (c, g) = pair(&[1000], 3, &cpu, &gpu);
    assert_close("neg", &g.neg().to_vec(), &c.neg().to_vec(), 0.0);
    assert_close("relu", &g.relu().to_vec(), &c.relu().to_vec(), 0.0);
    assert_close("tanh", &g.tanh().to_vec(), &c.tanh().to_vec(), 1e-5);
    assert_close("exp", &g.exp().to_vec(), &c.exp().to_vec(), 1e-5);
    let (pc, pg) = (
        c.mul(&c).add(&Tensor::scalar_on(0.01, &cpu)),
        g.mul(&g).add(&Tensor::scalar_on(0.01, &gpu)),
    );
    assert_close("log", &pg.log().to_vec(), &pc.log().to_vec(), 1e-5);
}

#[test]
fn matmul_sizes() {
    let (cpu, gpu) = need_cuda!();
    for (m, k, n) in [
        (33, 65, 17),
        (1, 1, 1),
        (5, 129, 130),
        (128, 8, 128),
        (512, 512, 512),
        (7, 300, 1),
    ] {
        let (ac, ag) = pair(&[m, k], 1, &cpu, &gpu);
        let (bc, bg) = pair(&[k, n], 2, &cpu, &gpu);
        assert_close(
            &format!("matmul {m}x{k}x{n}"),
            &ag.matmul(&bg).to_vec(),
            &ac.matmul(&bc).to_vec(),
            2e-4,
        );
    }
}

#[test]
fn matmul_gradients_odd_shapes() {
    // Backward uses MATMUL_T (transposed operands), so check non-square shapes where a wrong flag or
    // leading dimension cannot cancel out. Both inputs need gradients here.
    let (cpu, gpu) = need_cuda!();
    for (m, k, n) in [
        (33, 65, 17),
        (1, 1, 1),
        (5, 129, 130),
        (7, 300, 1),
        (1, 40, 64),
    ] {
        let run = |dev: &Device| {
            let mk = |shape: &[usize], seed| {
                Tensor::from_vec_on(data(shape.iter().product(), seed), shape.to_vec(), dev)
            };
            let (a, b, w) = (
                mk(&[m, k], 1).requires_grad_(true),
                mk(&[k, n], 2).requires_grad_(true),
                mk(&[m, n], 3),
            );
            a.matmul(&b).mul(&w).sum().backward();
            vec![a.grad().unwrap().to_vec(), b.grad().unwrap().to_vec()]
        };
        let (c, g) = (run(&cpu), run(&gpu));
        assert_close(&format!("grad_a {m}x{k}x{n}"), &g[0], &c[0], 2e-4);
        assert_close(&format!("grad_b {m}x{k}x{n}"), &g[1], &c[1], 2e-4);
    }
}

#[test]
fn matmul_skips_unneeded_gradient() {
    let (_, gpu) = need_cuda!();
    let x = Tensor::from_vec_on(data(6, 1), vec![2, 3], &gpu);
    let w = Tensor::from_vec_on(data(12, 2), vec![3, 4], &gpu).requires_grad_(true);
    x.matmul(&w).sum().backward();
    assert!(w.grad().is_some());
    assert!(x.grad().is_none());
}

#[test]
fn axpy_in_place() {
    let (cpu, gpu) = need_cuda!();
    for n in [1usize, 255, 4097, 1 << 20] {
        let (ac, ag) = pair(&[n], 1, &cpu, &gpu);
        let (bc, bg) = pair(&[n], 2, &cpu, &gpu);
        ac.axpy_(-0.25, &bc);
        ag.axpy_(-0.25, &bg);
        assert_close(&format!("axpy n={n}"), &ag.to_vec(), &ac.to_vec(), 1e-6);
    }
}

#[test]
fn vectorized_paths_match_cpu() {
    // The 128-bit binary and row-sum kernels only run for lengths that are a multiple of 4; check both
    // sides of that boundary, single-chunk and many-chunk reductions, and 2-D rows.
    let (cpu, gpu) = need_cuda!();
    for n in [
        4usize,
        7,
        8,
        1000,
        4096,
        4100,
        65537,
        1 << 20,
        (1 << 22) + 4,
        (1 << 22) + 3,
    ] {
        let (ac, ag) = pair(&[n], 1, &cpu, &gpu);
        let (bc, bg) = pair(&[n], 2, &cpu, &gpu);
        assert_close(
            &format!("add n={n}"),
            &ag.add(&bg).to_vec(),
            &ac.add(&bc).to_vec(),
            1e-6,
        );
        assert_close(
            &format!("div n={n}"),
            &ag.div(&bg.add(&bg.mul(&bg)).add(&Tensor::scalar_on(2.0, &gpu)))
                .to_vec(),
            &ac.div(&bc.add(&bc.mul(&bc)).add(&Tensor::scalar_on(2.0, &cpu)))
                .to_vec(),
            1e-5,
        );
        assert_close(
            &format!("sum n={n}"),
            &ag.sum().to_vec(),
            &ac.sum().to_vec(),
            1e-3,
        );
    }
    // d/d(bias[r]) of sum(w * (x + bias)) is the row sum of w, reduced over the last axis (inner == 1).
    for (rows, cols) in [
        (3usize, 8usize),
        (5, 6),
        (256, 4096),
        (300, 1000),
        (1, 1 << 16),
    ] {
        let run = |dev: &Device| {
            let mk = |shape: &[usize], seed| {
                Tensor::from_vec_on(data(shape.iter().product(), seed), shape.to_vec(), dev)
            };
            let (x, w) = (mk(&[rows, cols], 3), mk(&[rows, cols], 4));
            let bias = mk(&[rows, 1], 5).requires_grad_(true);
            x.add(&bias).mul(&w).sum().backward();
            bias.grad().unwrap().to_vec()
        };
        assert_close(
            &format!("rowsum {rows}x{cols}"),
            &run(&gpu),
            &run(&cpu),
            1e-3,
        );
    }
}

#[test]
fn sums_and_means() {
    let (cpu, gpu) = need_cuda!();
    // sum() reduces a flat tensor (inner == 1); big ones exercise the multi-block path.
    for n in [1usize, 7, 1000, 100_000, 3_000_001] {
        let (c, g) = pair(&[n], 4, &cpu, &gpu);
        assert_close(
            &format!("sum {n}"),
            &g.sum().to_vec(),
            &c.sum().to_vec(),
            1e-3 + 1e-6 * n as f32,
        );
        assert_close(
            &format!("mean {n}"),
            &g.mean().to_vec(),
            &c.mean().to_vec(),
            1e-4,
        );
    }
    // Axis sums happen through broadcast backward: inner > 1, small/large totals, 3-D.
    for (sa, sb) in [
        (vec![5000usize, 3], vec![3usize]),
        (vec![64, 40, 30], vec![40, 1]),
        (vec![300, 200], vec![1, 200]),
        (vec![2, 3, 4], vec![2, 1, 4]),
    ] {
        let (ac, ag) = pair(&sa, 5, &cpu, &gpu);
        let (bc, bg) = pair(&sb, 6, &cpu, &gpu);
        let (ac, bc) = (ac.requires_grad_(true), bc.requires_grad_(true));
        let (ag, bg) = (ag.requires_grad_(true), bg.requires_grad_(true));
        ac.add(&bc).sum().backward();
        ag.add(&bg).sum().backward();
        assert_close(
            "grad b",
            &bg.grad().unwrap().to_vec(),
            &bc.grad().unwrap().to_vec(),
            1e-5,
        );
        assert_close(
            "grad a",
            &ag.grad().unwrap().to_vec(),
            &ac.grad().unwrap().to_vec(),
            0.0,
        );
    }
}

#[test]
fn randn_fill_golden() {
    let (cpu, gpu) = need_cuda!();
    let v = Tensor::randn_on(&[4], 1234, &gpu).to_vec();
    for (got, want) in v
        .iter()
        .zip([0.6574781f32, -0.08203645, -2.2065563, 0.70945626])
    {
        assert!((got - want).abs() < 1e-4, "{v:?}");
    }
    let (g, c) = (
        Tensor::randn_on(&[100_000], 99, &gpu),
        Tensor::randn_on(&[100_000], 99, &cpu),
    );
    assert_close("randn vs cpu", &g.to_vec(), &c.to_vec(), 1e-4);
    assert_eq!(Tensor::full_on(&[3, 5], 2.5, &gpu).to_vec(), vec![2.5; 15]);
    assert_eq!(Tensor::zeros_on(&[1], &gpu).to_vec(), vec![0.0]);
    let t = Tensor::zeros_on(&[2, 2], &gpu);
    t.copy_(&Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
    assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn mlp_gradients_match() {
    let (cpu, gpu) = need_cuda!();
    let run = |dev: &Device| {
        let mk = |shape: &[usize], seed| {
            Tensor::from_vec_on(data(shape.iter().product(), seed), shape.to_vec(), dev)
        };
        let x = mk(&[32, 10], 1);
        let t = mk(&[32, 3], 2);
        let w1 = mk(&[10, 20], 3).requires_grad_(true);
        let b1 = mk(&[20], 4).requires_grad_(true);
        let w2 = mk(&[20, 3], 5).requires_grad_(true);
        let diff = x.matmul(&w1).add(&b1).relu().matmul(&w2).sub(&t);
        let loss = diff.mul(&diff).mean();
        loss.backward();
        vec![
            loss.to_vec(),
            w1.grad().unwrap().to_vec(),
            b1.grad().unwrap().to_vec(),
            w2.grad().unwrap().to_vec(),
        ]
    };
    let (c, g) = (run(&cpu), run(&gpu));
    for (i, (gv, cv)) in g.iter().zip(&c).enumerate() {
        assert_close(&format!("mlp tensor {i}"), gv, cv, 1e-4);
    }
}

#[test]
fn allocation_one_gib() {
    let (_, gpu) = need_cuda!();
    let n = (1usize << 30) / 4;
    {
        let t = Tensor::full_on(&[n], 3.0, &gpu);
        let u = t.add(&Tensor::scalar_on(1.0, &gpu)); // another 1 GiB
        gpu.synchronize();
        let v = u.to_vec();
        assert!(v[0] == 4.0 && v[n / 2] == 4.0 && v[n - 1] == 4.0);
        assert_eq!(u.sum().item(), 4.0 * n as f32);
    }
    let info = gpu.info();
    eprintln!("{info:?}");
    assert_eq!(info.kind, abi::KIND_CUDA);
    assert!(info.total_memory.unwrap() > 1 << 30 && info.free_memory.unwrap() > 0);
}

#[test]
fn out_of_memory_is_a_status_not_a_crash() {
    let _ = need_cuda!();
    unsafe {
        let lib = libloading::Library::new(cuda_dll()).unwrap();
        let entry: abi::EntryFn = *lib.get::<abi::EntryFn>(abi::ENTRY_SYMBOL).unwrap();
        let vt = &*entry();
        let mut p = std::ptr::null_mut();
        let st = (vt.alloc)(0, 64usize << 30, &mut p);
        assert_eq!(st, abi::STATUS_OUT_OF_MEMORY);
        let msg = std::ffi::CStr::from_ptr((vt.last_error)())
            .to_string_lossy()
            .into_owned();
        eprintln!("oom message: {msg}");
        assert!(msg.contains("OUT_OF_MEMORY"), "{msg}");
        // The device still works afterwards.
        let st = (vt.alloc)(0, 1 << 20, &mut p);
        assert_eq!(st, abi::STATUS_OK);
        (vt.free)(0, p);
    }
}

#[test]
fn benchmark() {
    let (_, gpu) = need_cuda!();
    let n = 4096;
    let a = Tensor::randn_on(&[n, n], 1, &gpu);
    let b = Tensor::randn_on(&[n, n], 2, &gpu);
    let _ = a.matmul(&b);
    gpu.synchronize();
    let iters = 5;
    let t0 = Instant::now();
    let mut c = a.matmul(&b);
    for _ in 1..iters {
        c = a.matmul(&b);
    }
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * (n as f64).powi(3);
    eprintln!(
        "BENCH matmul {n}^3: {:.2} ms, {:.0} GFLOPs (c[0]={})",
        dt * 1e3,
        flops / dt / 1e9,
        c.to_vec()[0]
    );

    let m = 64usize << 20; // 64M floats = 256 MiB per tensor
    let x = Tensor::randn_on(&[m], 3, &gpu);
    let y = Tensor::randn_on(&[m], 4, &gpu);
    let _ = x.add(&y);
    gpu.synchronize();
    let t0 = Instant::now();
    let mut z = x.add(&y);
    for _ in 1..10 {
        z = x.add(&y);
    }
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64() / 10.0;
    eprintln!(
        "BENCH add {m} f32: {:.2} ms, {:.0} GB/s (z[0]={})",
        dt * 1e3,
        12.0 * m as f64 / dt / 1e9,
        z.to_vec()[0]
    );

    let t0 = Instant::now();
    let s = x.sum();
    gpu.synchronize();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "BENCH sum {m} f32: {:.2} ms, {:.0} GB/s (s={})",
        dt * 1e3,
        4.0 * m as f64 / dt / 1e9,
        s.item()
    );
}

// ---- streams, events, pinned memory (docs/plugin-abi.md section 5.1) -----------------------------

#[test]
fn cuda_provides_the_streams_extension() {
    let (_cpu, gpu) = need_cuda!();
    assert!(gpu.supports_streams() && gpu.supports_pinned());
    let s = gpu.create_stream().unwrap();
    assert_ne!(s.handle(), 0);
    let e = s.record_new().unwrap();
    e.sync().unwrap();
    assert!(e.is_done().unwrap());
}

/// A copy on a side stream, ordered before compute on the default stream by an event, gives the right
/// answer; ops run on a side stream through the current-stream guard; results match the CPU.
#[test]
fn side_stream_copy_then_compute() {
    let (cpu, gpu) = need_cuda!();
    let (m, k, n) = (256usize, 512usize, 128usize);
    let a = data(m * k, 1);
    let b = data(k * n, 2);
    let mut pin_a = gpu.alloc_pinned(m * k * 4).unwrap();
    let mut pin_b = gpu.alloc_pinned(k * n * 4).unwrap();
    assert!(pin_a.is_pinned());
    for (dst, src) in [(&mut pin_a, &a), (&mut pin_b, &b)] {
        for (i, v) in src.iter().enumerate() {
            dst.as_mut_slice()[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let side = gpu.create_stream().unwrap();
    let ga = Tensor::full_on(&[m, k], 0.0, &gpu);
    let gb = Tensor::full_on(&[k, n], 0.0, &gpu);
    gpu.synchronize();
    ga.copy_from_pinned_async(&pin_a, 0, &side).unwrap();
    gb.copy_from_pinned_async(&pin_b, 0, &side).unwrap();
    let copied = side.record_new().unwrap();
    // default stream waits for the copies, then multiplies
    gpu.default_stream().wait(&copied).unwrap();
    let out = ga.matmul(&gb);
    let want = Tensor::from_vec_on(a.clone(), vec![m, k], &cpu).matmul(&Tensor::from_vec_on(b.clone(), vec![k, n], &cpu));
    assert_close("matmul after side-stream copy", &out.to_vec(), &want.to_vec(), 1e-3);

    // the same product computed on the side stream (cuBLAS follows the current stream)
    let on_side = {
        let _g = side.enter();
        ga.matmul(&gb)
    };
    side.synchronize().unwrap();
    assert_close("matmul on side stream", &on_side.to_vec(), &want.to_vec(), 1e-3);

    // and read back asynchronously
    let mut back = gpu.alloc_pinned(m * n * 4).unwrap();
    out.copy_to_pinned_async(&mut back, 0, &side).unwrap();
    side.synchronize().unwrap();
    let got: Vec<f32> = back.as_slice().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_close("async read-back", &got, &want.to_vec(), 1e-3);
}

/// Freed memory must not be recycled while work enqueued before the free is still running on another
/// stream: a long product on a side stream writes into a block that is freed straight away, and the next
/// allocation of the same size, filled on the default stream, must not be overwritten by it.
#[test]
fn free_during_side_stream_work_does_not_corrupt_later_allocations() {
    let (_cpu, gpu) = need_cuda!();
    let side = gpu.create_stream().unwrap();
    let n = 2048usize;
    let a = Tensor::from_vec_on(data(n * n, 3), vec![n, n], &gpu);
    let b = Tensor::from_vec_on(data(n * n, 4), vec![n, n], &gpu);
    gpu.synchronize();
    for round in 0..10 {
        {
            let _g = side.enter();
            let mut x = a.matmul(&b);
            for _ in 0..8 {
                x = x.matmul(&b); // keep the side stream busy for a while
            }
            drop(x); // freed with work in flight
        }
        let fill = (round + 1) as f32;
        let y = Tensor::full_on(&[n, n], fill, &gpu); // same size: would reuse the freed block
        gpu.synchronize();
        let v = y.to_vec();
        assert!(v.iter().all(|&e| e == fill), "round {round}: a freed block was reused while a stream still used it");
    }
}

/// The Flux pattern in miniature: weights live in pinned host memory and are streamed through two GPU
/// "shell" buffers while the previous layer computes. Serial (copy, then compute) against double-buffered
/// (copy on a side stream, ordered with events). Both must give the same answer; overlap must be faster.
#[test]
fn streamed_weights_overlap_copy_and_compute() {
    let (_cpu, gpu) = need_cuda!();
    let (layers, d, batch) = (16usize, 4096usize, 768usize);
    let layer_bytes = d * d * 4;
    let mut host = gpu.alloc_pinned(layers * layer_bytes).unwrap();
    assert!(host.is_pinned());
    for l in 0..layers {
        let w = data(d * d, 100 + l as u64);
        let dst = &mut host.as_mut_slice()[l * layer_bytes..(l + 1) * layer_bytes];
        for (i, v) in w.iter().enumerate() {
            dst[i * 4..i * 4 + 4].copy_from_slice(&(v * 0.02).to_le_bytes());
        }
    }
    let x0 = Tensor::from_vec_on(data(batch * d, 7), vec![batch, d], &gpu);
    let shells = [Tensor::full_on(&[d, d], 0.0, &gpu), Tensor::full_on(&[d, d], 0.0, &gpu)];
    let main = gpu.default_stream();
    let copy = gpu.create_stream().unwrap();
    gpu.synchronize();

    let serial = |x: &Tensor| {
        let mut x = x.clone();
        for l in 0..layers {
            let sh = &shells[l % 2];
            sh.copy_from_pinned_async(&host, l * layer_bytes, &copy).unwrap();
            copy.synchronize().unwrap();
            x = x.matmul(sh).tanh();
            main.synchronize().unwrap(); // the next copy overwrites a shell the host must know is idle
        }
        x
    };
    let overlapped = |x: &Tensor| {
        let mut x = x.clone();
        let mut copied = Vec::new();
        shells[0].copy_from_pinned_async(&host, 0, &copy).unwrap();
        copied.push(copy.record_new().unwrap());
        let mut used: Vec<Option<_>> = (0..layers).map(|_| None).collect();
        for l in 0..layers {
            if l + 1 < layers {
                // the shell layer l+1 loads into was last read by layer l-1
                if l >= 1 {
                    copy.wait(used[l - 1].as_ref().unwrap()).unwrap();
                }
                shells[(l + 1) % 2].copy_from_pinned_async(&host, (l + 1) * layer_bytes, &copy).unwrap();
                copied.push(copy.record_new().unwrap());
            }
            main.wait(&copied[l]).unwrap();
            x = x.matmul(&shells[l % 2]).tanh();
            used[l] = Some(main.record_new().unwrap());
        }
        x
    };

    let time = |f: &dyn Fn(&Tensor) -> Tensor| {
        let mut best = f64::MAX;
        let mut last = None;
        for _ in 0..3 {
            gpu.synchronize();
            let t = Instant::now();
            let y = f(&x0);
            gpu.synchronize();
            best = best.min(t.elapsed().as_secs_f64());
            last = Some(y);
        }
        (best, last.unwrap())
    };
    time(&serial); // warm up: cuBLAS handles, allocator
    let (t_serial, y_serial) = time(&serial);
    let (t_over, y_over) = time(&overlapped);
    let gib = (layers * layer_bytes) as f64 / (1u64 << 30) as f64;
    eprintln!(
        "streamed {gib:.1} GiB of weights through 2 shells: serial {:.1} ms, overlapped {:.1} ms ({:.2}x)",
        t_serial * 1e3,
        t_over * 1e3,
        t_serial / t_over
    );
    // CPU reference, to say which one is wrong if they differ
    let mut r = Tensor::from_vec_on(data(batch * d, 7), vec![batch, d], &_cpu);
    for l in 0..layers {
        let w: Vec<f32> = data(d * d, 100 + l as u64).iter().map(|v| v * 0.02).collect();
        r = r.matmul(&Tensor::from_vec_on(w, vec![d, d], &_cpu)).tanh();
    }
    let want = r.to_vec();
    let (a, b) = (y_serial.to_vec(), y_over.to_vec());
    let worst = |x: &[f32]| x.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
    eprintln!("max abs error vs cpu: serial {:.2e}, overlapped {:.2e}", worst(&a), worst(&b));
    assert_close("serial vs cpu", &a, &want, 1e-3);
    assert_close("overlapped vs cpu", &b, &want, 1e-3);
    assert_close("overlapped vs serial", &y_over.to_vec(), &y_serial.to_vec(), 1e-5);
    assert!(t_over < t_serial * 0.9, "overlap gave no speed-up: serial {t_serial:.4}s, overlapped {t_over:.4}s");
}
