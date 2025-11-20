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
