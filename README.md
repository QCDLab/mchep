<h1 align="center">MCHEP</h1>

<p align="justify">
  <b>NeoPDF</b> is a highly parallelizable Monte Carlo integration routine. Specifically, it
  supports multi-threads/cores parallelization, Single Instruction Multiple Data (SIMD)
  instructions, and GPU acceleration. Currently, it implements two adaptive multidimensional
  integrations, namely VEGAS and VEGAS+ (with adaptive stratified sampling) as presented in
  the paper <a href="https://arxiv.org/pdf/2009.05112">arXiv:2009.05112</a>.
</p>

Installation
------------

To install the C/C++ APIs, simply run the following command:

```bash
cargo cinstall --release --prefix=/path/to/installation --manifest-path mchep_capi/Cargo.toml
```
