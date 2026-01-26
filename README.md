<h1 align="center">MCHEP</h1>

<p align="justify">
  <b>MCHEP</b> is a highly parallelizable Monte Carlo integration routine. Specifically, it
  supports multi-threads/cores parallelization, Single Instruction Multiple Data (SIMD)
  instructions, and GPU acceleration. Currently, it implements two adaptive multidimensional
  integrations, namely VEGAS and VEGAS+ (with adaptive stratified sampling) as presented in
  the paper <a href="https://arxiv.org/pdf/2009.05112">arXiv:2009.05112</a>.
</p>

Installation
------------

To install the C/C++ APIs, you first need to install `cargo` and `cargo-c`. Then, in order
to also properly install the C++ header, you need to define the environment vaiable:
```bash
export CARGO_C_MCHEP_INSTALL_PREFIX=${prefix}
```
where `${prefix}` is the path to where the library will be installed. Then run the following
command:

```bash
cargo cinstall --release --prefix=${prefix} --manifest-path mchep_capi/Cargo.toml
```

Finally, you need to set the environment variables:
```bash
export LD_LIBRARY_PATH=${prefix}/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=${prefix}/lib/pkgconfig:$PKG_CONFIG_PATH
```
and check that `mchep` is properly found in the `PKG_CONFIG_PATH`:
```bash
pkg-config mchep_capi --libs
```

Documentation
-------------

- [SIMD Integration Tutorial](docs/SIMD_TUTORIAL.md) - How to use SIMD acceleration with the C/C++ API
- [Benchmark Results](mchep_capi/bench/BENCHMARK_RESULTS.md) - Extensive performance comparison with CUBA library

## MCHEP Feature Availability

### Rust API

| Feature | Vegas | VegasPlus |
|---------|-------|-----------|
| Multi-threaded (Rayon) | ✅ | ✅ |
| SIMD | ✅ | ✅ |
| GPU | ✅ | ✅ |

### C/C++ API

| Feature | Vegas | VegasPlus |
|---------|-------|-----------|
| Multi-threaded (Rayon) | ✅ | ✅ |
| SIMD | ✅ | ✅ |
| GPU | ❌ | ❌ |

Performance
-----------

The following benchmark compares MCHEP against the [CUBA](https://feynarts.de/cuba/) library
across different integrand complexities, simulating typical High Energy Physics (HEP) workloads:

<p align="center">
  <img src="docs/benchmark_scaling.png" alt="MCHEP vs CUBA Scaling Benchmark" width="100%">
</p>

**Left plot**: Integration throughput (evaluations per millisecond) vs. computational cost
per evaluation. MCHEP with SIMD+AVX consistently outperforms both MCHEP scalar and CUBA
across all complexity levels.

**Right plot**: Speedup factor of MCHEP implementations compared to CUBA Vegas.
MCHEP SIMD+AVX achieves **4-6x speedup** over CUBA for typical HEP workloads.

| QCD Complexity | Typical Cost | MCHEP Scalar | MCHEP SIMD+AVX | CUBA | Speedup vs CUBA |
|----------------|--------------|--------------|----------------|------|-----------------|
| LO (tree-level) | ~10k FLOPs | 2,500 evals/ms | 7,500 evals/ms | 1,800 evals/ms | **4.2x** |
| NLO (1-loop) | ~1M FLOPs | 48 evals/ms | 153 evals/ms | 34 evals/ms | **4.5x** |
| NNLO (2-loop) | ~10M FLOPs | 5 evals/ms | 18 evals/ms | 5 evals/ms | **3.9x** |

Combined with multi-core parallelization (16 cores), MCHEP can reduce month-long
calculations to days. See [benchmark details](mchep_capi/bench/BENCHMARK_RESULTS.md)
for comprehensive comparisons.
