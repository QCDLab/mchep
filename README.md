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

## Usage Examples

Here are some examples of how to use `MCHEP` with its different features. The examples will integrate a 2D Gaussian function over the domain `[-1, 1] x [-1, 1]`.

### Multi-threading (default)

By default, `MCHEP` uses the Rayon library to parallelize the integration over all available CPU cores. This is the standard way to use the library and requires no special configuration.

**1. `Cargo.toml`:**

Add `mchep` to your dependencies. If you are creating an example in a sub-directory of the `mchep` project, you can use a path dependency:

```toml
[dependencies]
mchep = { path = "../mchep" }
```

*(Note: The path may vary depending on your project structure.)*

**2. `main.rs`:**

```rust
use mchep::integrand::Integrand;
use mchep::vegas::Vegas;

// Define the function to be integrated
struct GaussianIntegrand;

impl Integrand for GaussianIntegrand {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> f64 {
        (-(x[0].powi(2)) - x[1].powi(2)).exp()
    }
}

fn main() {
    let integrand = GaussianIntegrand;
    let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];

    // Create a new VEGAS integrator
    let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
    vegas.set_seed(1234);

    // Perform the integration
    let result = vegas.integrate(&integrand, None);

    println!("Result: {:?}", result);
    // Expected value is ~2.230985
}
```

**3. Run:**

```bash
cargo run --release
```

### SIMD Acceleration

`MCHEP` can leverage SIMD instructions to perform multiple calculations at once, which can significantly speed up the integration. This requires implementing the `SimdIntegrand` trait and using the `integrate_simd` method.

**1. `Cargo.toml`:**

Your `Cargo.toml` should include the `wide` crate, which `mchep` uses for SIMD operations.

```toml
[dependencies]
mchep = { path = "../mchep" }
wide = "0.7"
```

**2. `main.rs`:**

```rust
use mchep::integrand::SimdIntegrand;
use mchep::vegas::Vegas;
use wide::f64x4;

// Define the SIMD version of the integrand
struct GaussianSimdIntegrand;

impl SimdIntegrand for GaussianSimdIntegrand {
    fn dim(&self) -> usize {
        2
    }

    fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
        let x = points[0];
        let y = points[1];
        (-(x * x) - (y * y)).exp()
    }
}

fn main() {
    let integrand = GaussianSimdIntegrand;
    let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];

    let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
    vegas.set_seed(1234);

    let result = vegas.integrate_simd(&integrand, None);

    println!("Result (SIMD): {:?}", result);
}
```

**3. Run:**

To get the best performance, compile with optimizations for your specific CPU.

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release
```

### GPU Acceleration

`MCHEP` can offload the integration to the GPU using the `burn` machine learning framework with its `wgpu` backend. This is ideal for very large numbers of evaluations and complex integrands.

**Prerequisites:** You need to have a `wgpu`-compatible graphics driver installed. This typically means having up-to-date drivers for your GPU and potentially installing the Vulkan SDK.

**1. `Cargo.toml`:**

Enable the `gpu` feature for `mchep`.

```toml
[dependencies]
mchep = { path = "../mchep", features = ["gpu"] }
burn = "0.13" # Or a version compatible with mchep's burn dependency
```

**2. `main.rs`:**

```rust
use mchep::vegas::Vegas;
#[cfg(feature = "gpu")]
use {
    mchep::integrand::BurnIntegrand,
    burn::prelude::*
};

// Define the GPU-compatible integrand
#[cfg(feature = "gpu")]
struct GaussianBurnIntegrand;

#[cfg(feature = "gpu")]
impl<B: Backend> BurnIntegrand<B> for GaussianBurnIntegrand {
    fn dim(&self) -> usize {
        2
    }

    fn eval_burn(&self, points: Tensor<B, 2>) -> Tensor<B, 1> {
        let n_points = points.dims()[0];
        let x = points.clone().slice([0..n_points, 0..1]);
        let y = points.clone().slice([0..n_points, 1..2]);
        let x2 = x.clone() * x;
        let y2 = y.clone() * y;
        let neg_x2_y2 = (x2 + y2).mul_scalar(-1.0);
        neg_x2_y2.exp().squeeze(1)
    }
}


fn main() {
    #[cfg(feature = "gpu")]
    {
        let integrand = GaussianBurnIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
        vegas.set_seed(1234);

        let result = vegas.integrate_gpu(&integrand, None);
        println!("Result (GPU): {:?}", result);
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("GPU feature not enabled. Compile with --features mchep/gpu");
    }
}
```

**3. Run:**

```bash
cargo run --release --features mchep/gpu
```

### MPI Parallelization

For large-scale integrations, `MCHEP` can be run in parallel across multiple nodes using MPI. This requires an MPI implementation installed on your system. This example uses `VegasPlus`.

**Prerequisites:** An MPI implementation like Open MPI or MPICH must be installed.

**1. `Cargo.toml`:**

Enable the `mpi` feature for `mchep` and add the `mpi` crate.

```toml
[dependencies]
mchep = { path = "../mchep", features = ["mpi"] }
mpi = "0.6"
```

**2. `main.rs`:**

```rust
use mchep::integrand::Integrand;
use mchep::vegasplus::VegasPlus;
use mpi::traits::*;

struct GaussianIntegrand;

impl Integrand for GaussianIntegrand {
    fn dim(&self) -> usize {
        2
    }

    fn eval(&self, x: &[f64]) -> f64 {
        (-(x[0].powi(2)) - x[1].powi(2)).exp()
    }
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    let integrand = GaussianIntegrand;
    let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
    let mut vegas = VegasPlus::new(20, 200_000, 50, 0.5, 4, 0.75, boundaries);

    let result = vegas.integrate_mpi(&integrand, &world, None);

    if rank == 0 {
        println!("Result (MPI): {:?}", result);
    }
}
```

**3. Run:**

Compile the program as usual, then run it with `mpirun`. The following command runs the integration on 4 processes.

```bash
# First, build the executable with the mpi feature
cargo build --release --features mchep/mpi

# Then, run with mpirun. The executable path might differ.
mpirun -n 4 target/release/<your_executable_name>
```