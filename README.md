# pytorches-cuda

The NVIDIA CUDA backend for [PyTorches](https://github.com/tonyt123456/pytorches), shipped as its
own package so the main project does not have to bundle every vendor's plugin.

```
pip install pytorches-universe pytorches-cuda
```

There is nothing to import. PyTorches finds the plugin through the `pytorches.plugins` entry point,
checks its manifest, probes it in a child process (a broken driver cannot crash your program), and
registers `cuda:N` devices. `pytorches.doctor()` shows what was found.

It uses the CUDA **driver** API loaded at runtime (and cuBLAS if present), so no CUDA toolkit is needed
on the machine, only an NVIDIA driver. Kernels ship as PTX, which the driver compiles for whatever GPU
is installed.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | The plugin: implements the `pytorches-plugin-abi` vtable |
| `kernels/` | CUDA C source and the committed PTX (`scripts/build-kernels.ps1` regenerates it; nvcc is only needed for that) |
| `python/pytorches_cuda/` | The Python package that carries the library and `pytorches_plugin_cuda.json` (manifest) |
| `setup.py` | Runs `cargo build --release` and packages the library as a `py3-none-<platform>` wheel |
| `tests/conformance.rs` | Compares every op against the CPU plugin through the core; skips without a GPU |

## Develop

```
cp .cargo/config.toml.example .cargo/config.toml   # build against a local ../pytorches checkout
cargo test                                         # builds the cdylib, runs the conformance suite
python -m build --wheel                            # dist/pytorches_cuda-*.whl
```

The plugin depends on `pytorches-plugin-abi` (and, for tests, `pytorches-core` and
`pytorches-plugin-cpu`) from the main repository's `main` branch. ABI version 4 is required.

## Releasing

Tag `vX.Y.Z`; the workflow builds Windows and manylinux wheels and publishes to PyPI by trusted
publishing. Bump the version in `pyproject.toml` and `Cargo.toml` together.
