# Tutorials

This section provides practical examples of how to use MCHEP for multi-dimensional integration. MCHEP supports both Rust and C++ APIs, and offers various acceleration features like multi-threading, SIMD, GPU, and MPI.

## Basic Integration (Vegas)

The VEGAS algorithm uses adaptive importance sampling to focus evaluations in regions where the integrand is largest.

=== ":simple-rust: Rust"

    ```rust
    use mchep::vegas::{Vegas, VegasResult};
    use mchep::integrand::Integrand;

    // Define your integrand
    struct Gaussian;

    impl Integrand for Gaussian {
        fn dim(&self) -> usize { 2 }
        fn eval(&self, x: &[f64]) -> f64 {
            let r2 = x[0]*x[0] + x[1]*x[1];
            (-r2).exp()
        }
    }

    fn main() {
        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 1.5, boundaries);
        
        let result = vegas.integrate(&Gaussian, None);
        println!("Result: {} +/- {}", result.value, result.error);
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <iostream>
    #include <cmath>

    int main() {
        std::vector<std::pair<double, double>> boundaries = {
            {0.0, 1.0}, {0.0, 1.0}
        };

        mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);

        auto integrand = [](const std::vector<double>& x) {
            double r2 = x[0]*x[0] + x[1]*x[1];
            return std::exp(-r2);
        };

        VegasResult result = vegas.integrate(integrand);
        std::cout << "Result: " << result.value << " +/- " << result.error << std::endl;
        return 0;
    }
    ```

## Adaptive Stratified Sampling (VegasPlus)

VEGAS+ adds adaptive stratified sampling, which is particularly effective for integrands with multiple or sharp peaks.

=== ":simple-rust: Rust"

    ```rust
    use mchep::vegasplus::VegasPlus;
    use mchep::integrand::Integrand;

    struct PeakedIntegrand;
    impl Integrand for PeakedIntegrand {
        fn dim(&self) -> usize { 2 }
        fn eval(&self, x: &[f64]) -> f64 {
            // A sharp peak at (0.5, 0.5)
            let d2 = (x[0]-0.5).powi(2) + (x[1]-0.5).powi(2);
            (-100.0 * d2).exp()
        }
    }

    fn main() {
        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        // Params: n_iter, n_eval, n_bins, alpha, n_strat, beta, boundaries
        let mut vp = VegasPlus::new(10, 100_000, 50, 1.5, 4, 0.75, boundaries);
        
        let result = vp.integrate(&PeakedIntegrand, None);
        println!("Result: {} +/- {}", result.value, result.error);
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <iostream>
    #include <cmath>

    int main() {
        std::vector<std::pair<double, double>> boundaries = {
            {0.0, 1.0}, {0.0, 1.0}
        };

        // Params: n_iter, n_eval, n_bins, alpha, n_strat, beta, boundaries
        mchep::VegasPlus vp(10, 100000, 50, 1.5, 4, 0.75, boundaries);

        auto integrand = [](const std::vector<double>& x) {
            double d2 = std::pow(x[0]-0.5, 2) + std::pow(x[1]-0.5, 2);
            return std::exp(-100.0 * d2);
        };

        VegasResult result = vp.integrate(integrand);
        std::cout << "Result: " << result.value << " +/- " << result.error << std::endl;
        return 0;
    }
    ```

## Integration with Target Accuracy

Instead of running for a fixed number of iterations, you can specify a target accuracy (in percent). The integrator will stop as soon as the estimated relative error falls below this threshold.

=== ":simple-rust: Rust"

    ```rust
    use mchep::vegas::Vegas;
    use mchep::integrand::Integrand;

    fn main() {
        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vegas = Vegas::new(100, 100_000, 50, 1.5, boundaries);
        
        // Stop when relative error is below 0.1%
        let target_accuracy = Some(0.1);
        let result = vegas.integrate(&MyIntegrand, target_accuracy);
        
        println!("Final Accuracy: {}%", (result.error / result.value) * 100.0);
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <iostream>

    int main() {
        std::vector<std::pair<double, double>> boundaries = {{0.0, 1.0}, {0.0, 1.0}};
        mchep::Vegas vegas(100, 100000, 50, 1.5, boundaries);

        // Stop when relative error is below 0.1%
        double target_accuracy = 0.1;
        VegasResult result = vegas.integrate(my_integrand, target_accuracy);

        std::cout << "Final Accuracy: " << (result.error / result.value) * 100.0 << "%" << std::endl;
        return 0;
    }
    ```

## SIMD Acceleration

SIMD (Single Instruction Multiple Data) allows evaluating multiple points simultaneously (typically 4 for `f64x4`). This provides significant speedups for compute-heavy integrands.

=== ":simple-rust: Rust"

    ```rust
    use mchep::vegas::Vegas;
    use mchep::integrand::SimdIntegrand;
    use wide::f64x4;

    struct SimdGaussian;

    impl SimdIntegrand for SimdGaussian {
        fn dim(&self) -> usize { 2 }
        fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
            let x = points[0];
            let y = points[1];
            let r2 = x*x + y*y;
            (-r2).exp() // wide::f64x4 implements .exp()
        }
    }

    fn main() {
        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 1.5, boundaries);
        
        let result = vegas.integrate_simd(&SimdGaussian, None);
        println!("SIMD Result: {} +/- {}", result.value, result.error);
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <iostream>
    #include <array>
    #include <cmath>

    int main() {
        std::vector<std::pair<double, double>> boundaries = {{0.0, 1.0}, {0.0, 1.0}};
        mchep::Vegas vegas(10, 100000, 50, 1.5, boundaries);

        // x vector size is dim * 4 (SoA layout)
        auto simd_integrand = [](const std::vector<double>& x) {
            std::array<double, 4> results;
            for (int i = 0; i < 4; ++i) {
                double r2 = x[0*4 + i]*x[0*4 + i] + x[1*4 + i]*x[1*4 + i];
                results[i] = std::exp(-r2);
            }
            return results;
        };

        VegasResult result = vegas.integrate_simd(simd_integrand);
        std::cout << "SIMD Result: " << result.value << std::endl;
        return 0;
    }
    ```

## GPU Acceleration

MCHEP supports GPU integration in Rust using the `Burn` deep learning framework. This allows you to leverage the massive parallelism of modern GPUs for evaluation-heavy integrands.

=== ":simple-rust: Rust"

    ```rust
    // Requires "gpu" feature enabled
    use mchep::vegas::Vegas;
    use mchep::integrand::BurnIntegrand;
    use burn::prelude::*;

    struct GpuGaussian;

    impl<B: Backend> BurnIntegrand<B> for GpuGaussian {
        fn dim(&self) -> usize { 2 }

        fn eval_burn(&self, points: Tensor<B, 2>) -> Tensor<B, 1> {
            // points shape: [n_points, dim]
            let x = points.clone().slice([0..points.dims()[0], 0..1]);
            let y = points.clone().slice([0..points.dims()[0], 1..2]);
            
            let x2 = x.clone() * x;
            let y2 = y.clone() * y;
            
            let neg_r2 = (x2 + y2).mul_scalar(-1.0);
            neg_r2.exp().squeeze(1)
        }
    }

    fn main() {
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
        
        // integrate_gpu automatically initializes the WGPU backend
        let result = vegas.integrate_gpu(&GpuGaussian, None);
        println!("GPU Result: {} +/- {}", result.value, result.error);
    }
    ```

=== ":simple-cplusplus: C++"

    > GPU integration is currently only supported via the Rust API.

## Distributed Integration (MPI)

For very large integration tasks, MCHEP can be distributed across multiple nodes using MPI.

=== ":simple-rust: Rust"

    ```rust
    // Requires "mpi" feature enabled
    use mchep::vegasplus::VegasPlus;
    use mchep::integrand::Integrand;
    use mpi::traits::*;

    fn main() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();

        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vp = VegasPlus::new(10, 100_000, 50, 1.5, 4, 0.75, boundaries);
        
        // integrate_mpi distributes evaluations across the MPI communicator
        let result = vp.integrate_mpi(&MyIntegrand, &world, None);

        if rank == 0 {
            println!("MPI Result: {} +/- {}", result.value, result.error);
        }
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <mpi.h>
    #include <iostream>

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::vector<std::pair<double, double>> boundaries = {{0.0, 1.0}, {0.0, 1.0}};
        mchep::VegasPlus vp(10, 100000, 50, 1.5, 4, 0.75, boundaries);

        auto integrand = [](const std::vector<double>& x) {
            return std::exp(-(x[0]*x[0] + x[1]*x[1]));
        };

        // Use integrate_mpi with an MPI communicator
        VegasResult result = vp.integrate_mpi(integrand, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "MPI Result: " << result.value << std::endl;
        }

        MPI_Finalize();
        return 0;
    }
    ```

## MPI + SIMD Acceleration

Combining MPI and SIMD provides the ultimate performance, distributing vector-optimized evaluations across multiple nodes.

=== ":simple-rust: Rust"

    ```rust
    // Requires both "mpi" and "simd" features
    use mchep::vegasplus::VegasPlus;
    use mchep::integrand::SimdIntegrand;
    use mpi::traits::*;

    fn main() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();

        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vp = VegasPlus::new(10, 100_000, 50, 1.5, 4, 0.75, boundaries);
        
        // integrate_mpi_simd uses both distributed and vector acceleration
        let result = vp.integrate_mpi_simd(&MySimdIntegrand, &world, None);

        if rank == 0 {
            println!("MPI+SIMD Result: {} +/- {}", result.value, result.error);
        }
    }
    ```

=== ":simple-cplusplus: C++"

    ```cpp
    #include <mchep.hpp>
    #include <mpi.h>
    #include <iostream>

    int main(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::vector<std::pair<double, double>> boundaries = {{0.0, 1.0}, {0.0, 1.0}};
        mchep::VegasPlus vp(10, 100000, 50, 1.5, 4, 0.75, boundaries);

        auto simd_integrand = [](const std::vector<double>& x) {
            std::array<double, 4> results;
            for (int i = 0; i < 4; ++i) {
                results[i] = std::exp(-(x[0*4+i]*x[0*4+i] + x[1*4+i]*x[1*4+i]));
            }
            return results;
        };

        // Use integrate_mpi_simd
        VegasResult result = vp.integrate_mpi_simd(simd_integrand, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "MPI+SIMD Result: " << result.value << std::endl;
        }

        MPI_Finalize();
        return 0;
    }
    ```
