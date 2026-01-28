<h1 align="center">MCHEP</h1>

<p align="justify">
  <b>MCHEP</b> is a highly parallelizable Monte Carlo integration routine. Specifically, it
  supports multi-threads/cores parallelization, Single Instruction Multiple Data (SIMD)
  instructions, and GPU acceleration. Currently, it implements two adaptive multidimensional
  integrations, namely VEGAS and VEGAS+ (with adaptive stratified sampling) as presented in
  the paper <a href="https://arxiv.org/pdf/2009.05112">arXiv:2009.05112</a>.
</p>

<h2>Installation</h2>

<p align="justify">
  To install the C/C++ APIs, you first need to install <code>cargo</code> and <code>cargo-c</code>.
  Then, in order to also properly install the C++ header, you need to define the environment variable:
</p>

```bash
export CARGO_C_MCHEP_INSTALL_PREFIX=${prefix}
```

<p align="justify">
  where <code>${prefix}</code> is the path to where the library will be installed. Then run the
  following command:
</p>

```bash
cargo cinstall --release --prefix=${prefix} --manifest-path mchep_capi/Cargo.toml
```

<p align="justify">
  Finally, you need to set the environment variables:
</p>

```bash
export LD_LIBRARY_PATH=${prefix}/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=${prefix}/lib/pkgconfig:$PKG_CONFIG_PATH
```

<p align="justify">
  and check that <code>mchep</code> is properly found in the <code>PKG_CONFIG_PATH</code>:
</p>

```bash
pkg-config mchep_capi --libs
```

<h2>Performance</h2>

<p align="justify">
  The following benchmark compares MCHEP against the <a href="https://feynarts.de/cuba/">CUBA</a>
  library across different integrand complexities, simulating typical High Energy Physics (HEP)
  workloads:
</p>

<p align="center">
  <img src="docs/benchmark_scaling.png" alt="MCHEP vs CUBA Scaling Benchmark" width="100%">
</p>

<p align="justify">
  <b>Left plot</b>: Integration throughput (evaluations per millisecond) vs. computational cost
  per evaluation. MCHEP with SIMD+AVX consistently outperforms both MCHEP scalar and CUBA
  across all complexity levels.
</p>

<p align="justify">
  <b>Right plot</b>: Speedup factor of MCHEP implementations compared to CUBA Vegas.
  MCHEP SIMD+AVX achieves <b>4-6x speedup</b> over CUBA for typical HEP workloads.
</p>

<div align="center">
<table>
  <tr>
    <th>QCD Complexity</th>
    <th>Typical Cost (FLOPS)</th>
    <th>MCHEP Scalar (evals/ms)</th>
    <th>MCHEP SIMD+AVX (evals/ms)</th>
    <th>CUBA (evals/ms)</th>
    <th>Speedup vs CUBA</th>
  </tr>
  <tr>
    <td>LO</td>
    <td>~10k</td>
    <td>2,500</td>
    <td>7,500</td>
    <td>1,800</td>
    <td>4.2x</td>
  </tr>
  <tr>
    <td>NLO</td>
    <td>~1M</td>
    <td>48</td>
    <td>153</td>
    <td>34</td>
    <td>4.5x</td>
  </tr>
  <tr>
    <td>NNLO</td>
    <td>~10M</td>
    <td>5</td>
    <td>18</td>
    <td>5</td>
    <td>3.9x</td>
  </tr>
</table>
</div>

<p align="justify">
  Combined with multi-core parallelization (16 cores), MCHEP can reduce month-long calculations
  to days. See <a href="mchep_capi/bench/BENCHMARK_RESULTS.md">benchmark details</a> for
  comprehensive comparisons.
</p>

<h2>Feature Availability</h2>

<div align="center">
<table>
  <tr>
    <th>Feature</th>
    <th colspan="2">Rust API</th>
    <th colspan="2">C/C++ API</th>
    <th colspan="2">Python API</th>
  </tr>
  <tr>
    <th></th>
    <th>Vegas</th>
    <th>VegasPlus</th>
    <th>Vegas</th>
    <th>VegasPlus</th>
    <th>Vegas</th>
    <th>VegasPlus</th>
  </tr>
  <tr>
    <td>Multi-threaded (Rayon)</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td>SIMD</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td>GPU</td>
    <td align="center">✅</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
  <tr>
    <td>MPI</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
  </tr>
  <tr>
    <td>MPI+SIMD</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">✅</td>
    <td align="center">❌</td>
    <td align="center">❌</td>
  </tr>
</table>
</div>

<h2>Documentation</h2>

<ul>
  <li><a href="docs/SIMD_TUTORIAL.md">SIMD Integration Tutorial</a> - How to use SIMD acceleration with the C/C++ API</li>
  <li><a href="mchep_capi/bench/BENCHMARK_RESULTS.md">Benchmark Results</a> - Extensive performance comparison with CUBA library</li>
</ul>
