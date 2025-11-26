use clap::Parser;
use mchep::benchmark::integrands::{
    GenzGaussian, GenzOscillatory, GenzProductPeak, EndpointSingularity,
};
#[cfg(feature = "gpu")]
use mchep::integrand::BurnIntegrand;
use mchep::integrand::{Integrand, SimdIntegrand};
use mchep::vegas::Vegas;
use mchep::vegasplus::VegasPlus;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug, Clone, Copy, PartialEq)]
enum Backend {
    All,
    Scalar,
    Simd,
    Gpu,
    Mpi,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "all" => Ok(Backend::All),
            "scalar" => Ok(Backend::Scalar),
            "simd" => Ok(Backend::Simd),
            "gpu" => Ok(Backend::Gpu),
            "mpi" => Ok(Backend::Mpi),
            _ => Err(format!("Unknown backend: {}", s)),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Dimension
    #[clap(short, long, value_parser, default_value_t = 2)]
    dim: usize,
    /// Number of evaluations per iteration
    #[clap(short, long, value_parser, default_value_t = 100_000)]
    n_eval: usize,
    /// Number of iterations
    #[clap(short, long, value_parser, default_value_t = 10)]
    n_iter: usize,
    /// Output file
    #[clap(short, long, value_parser, default_value = "benchmark_results.json")]
    output: String,
    /// Backend to run
    #[clap(short, long, value_parser, default_value = "all")]
    backend: Backend,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    integrand: String,
    dim: usize,
    n_eval: usize,
    n_iter: usize,
    backend: String,
    algorithm: String,
    time_s: f64,
    value: f64,
    error: f64,
    chi2_dof: f64,
    analytical_result: f64,
}

trait BenchmarkFn {
    fn name(&self) -> &str;
    fn analytical_result(&self) -> f64;
    fn integrand(&self) -> Box<dyn Integrand + Sync + Send>;
    fn simd_integrand(&self) -> Box<dyn SimdIntegrand + Sync + Send>;
    #[cfg(feature = "gpu")]
    fn burn_integrand(&self) -> Box<dyn BurnIntegrand<mchep::gpu::GpuBackend> + Sync + Send>;
}

#[derive(Clone)]
struct GenzOscillatoryFn(GenzOscillatory);
impl BenchmarkFn for GenzOscillatoryFn {
    fn name(&self) -> &str {
        "GenzOscillatory"
    }
    fn analytical_result(&self) -> f64 {
        self.0.analytical_result()
    }
    fn integrand(&self) -> Box<dyn Integrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    fn simd_integrand(&self) -> Box<dyn SimdIntegrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    #[cfg(feature = "gpu")]
    fn burn_integrand(&self) -> Box<dyn BurnIntegrand<mchep::gpu::GpuBackend> + Send + Sync> {
        Box::new(self.0.clone())
    }
}

#[derive(Clone)]
struct GenzProductPeakFn(GenzProductPeak);
impl BenchmarkFn for GenzProductPeakFn {
    fn name(&self) -> &str {
        "GenzProductPeak"
    }
    fn analytical_result(&self) -> f64 {
        self.0.analytical_result()
    }
    fn integrand(&self) -> Box<dyn Integrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    fn simd_integrand(&self) -> Box<dyn SimdIntegrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    #[cfg(feature = "gpu")]
    fn burn_integrand(&self) -> Box<dyn BurnIntegrand<mchep::gpu::GpuBackend> + Send + Sync> {
        Box::new(self.0.clone())
    }
}

#[derive(Clone)]
struct GenzGaussianFn(GenzGaussian);
impl BenchmarkFn for GenzGaussianFn {
    fn name(&self) -> &str {
        "GenzGaussian"
    }
    fn analytical_result(&self) -> f64 {
        self.0.analytical_result()
    }
    fn integrand(&self) -> Box<dyn Integrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    fn simd_integrand(&self) -> Box<dyn SimdIntegrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    #[cfg(feature = "gpu")]
    fn burn_integrand(&self) -> Box<dyn BurnIntegrand<mchep::gpu::GpuBackend> + Send + Sync> {
        Box::new(self.0.clone())
    }
}

#[derive(Clone)]
struct EndpointSingularityFn(EndpointSingularity);
impl BenchmarkFn for EndpointSingularityFn {
    fn name(&self) -> &str {
        "EndpointSingularity"
    }
    fn analytical_result(&self) -> f64 {
        self.0.analytical_result()
    }
    fn integrand(&self) -> Box<dyn Integrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    fn simd_integrand(&self) -> Box<dyn SimdIntegrand + Send + Sync> {
        Box::new(self.0.clone())
    }
    #[cfg(feature = "gpu")]
    fn burn_integrand(&self) -> Box<dyn BurnIntegrand<mchep::gpu::GpuBackend> + Send + Sync> {
        Box::new(self.0.clone())
    }
}

fn main() {
    let args = Args::parse();
    let mut results = Vec::new();

    let rank = {
        #[cfg(feature = "mpi")]
        {
            if args.backend == Backend::Mpi || args.backend == Backend::All {
                let universe = mpi::initialize().unwrap();
                universe.world().rank()
            } else {
                0
            }
        }
        #[cfg(not(feature = "mpi"))]
        {
            0
        }
    };

    run_benchmarks(
        args.dim,
        args.n_eval,
        args.n_iter,
        args.backend,
        &mut results,
    );

    if rank == 0 {
        let json = serde_json::to_string_pretty(&results).unwrap();
        let mut file = File::create(&args.output).unwrap();
        file.write_all(json.as_bytes()).unwrap();
        println!("Benchmark results written to {}", args.output);
    }
}

fn run_benchmarks(
    dim: usize,
    n_eval: usize,
    n_iter: usize,
    backend: Backend,
    results: &mut Vec<BenchmarkResult>,
) {
    let rank = {
        #[cfg(feature = "mpi")]
        {
            mpi::environment::world().rank()
        }
        #[cfg(not(feature = "mpi"))]
        {
            0
        }
    };

    if rank == 0 {
        println!(
            "Running benchmarks for dim={} n_eval={} n_iter={}",
            dim, n_eval, n_iter
        );
    }

    let boundaries: Vec<(f64, f64)> = (0..dim).map(|_| (0.0, 1.0)).collect();

    let benchmarks: Vec<Box<dyn BenchmarkFn>> = vec![
        Box::new(GenzOscillatoryFn(GenzOscillatory::new(dim).clone())),
        Box::new(GenzProductPeakFn(GenzProductPeak::new(dim).clone())),
        Box::new(GenzGaussianFn(GenzGaussian::new(dim).clone())),
        Box::new(EndpointSingularityFn(EndpointSingularity::new(dim).clone())),
    ];

    for benchmark in benchmarks {
        run_one_integrand(
            benchmark.as_ref(),
            dim,
            n_eval,
            n_iter,
            &boundaries,
            backend,
            results,
        );
    }
}

fn run_one_integrand(
    benchmark: &dyn BenchmarkFn,
    dim: usize,
    n_eval: usize,
    n_iter: usize,
    boundaries: &[(f64, f64)],
    backend: Backend,
    results: &mut Vec<BenchmarkResult>,
) {
    let name = benchmark.name();
    let analytical_result = benchmark.analytical_result();

    let rank = {
        #[cfg(feature = "mpi")]
        {
            mpi::environment::world().rank()
        }
        #[cfg(not(feature = "mpi"))]
        {
            0
        }
    };

    if backend == Backend::All || backend == Backend::Scalar {
        if rank == 0 {
            println!("Running {} - Vegas Scalar", name);
            let mut vegas = Vegas::new(n_iter, n_eval, 50, 0.5, boundaries);
            let start = Instant::now();
            let result = vegas.integrate(benchmark.integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "Scalar".to_string(),
                algorithm: "Vegas".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });

            println!("Running {} - Vegas+ Scalar", name);
            let mut vegas_plus = VegasPlus::new(n_iter, n_eval, 50, 0.5, 4, 0.75, boundaries);
            let start = Instant::now();
            let result = vegas_plus.integrate(benchmark.integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "Scalar".to_string(),
                algorithm: "Vegas+".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });
        }
    }

    if backend == Backend::All || backend == Backend::Simd {
        if rank == 0 {
            println!("Running {} - Vegas SIMD", name);
            let mut vegas = Vegas::new(n_iter, n_eval, 50, 0.5, boundaries);
            let start = Instant::now();
            let result = vegas.integrate_simd(benchmark.simd_integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "SIMD".to_string(),
                algorithm: "Vegas".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });

            println!("Running {} - Vegas+ SIMD", name);
            let mut vegas_plus = VegasPlus::new(n_iter, n_eval, 50, 0.5, 4, 0.75, boundaries);
            let start = Instant::now();
            let result = vegas_plus.integrate_simd(benchmark.simd_integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "SIMD".to_string(),
                algorithm: "Vegas+".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });
        }
    }

    #[cfg(feature = "gpu")]
    if backend == Backend::All || backend == Backend::Gpu {
        if rank == 0 {
            println!("Running {} - Vegas GPU", name);
            let mut vegas = Vegas::new(n_iter, n_eval, 50, 0.5, boundaries);
            let start = Instant::now();
            let result = vegas.integrate_gpu(benchmark.burn_integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "GPU".to_string(),
                algorithm: "Vegas".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });

            println!("Running {} - Vegas+ GPU", name);
            let mut vegas_plus = VegasPlus::new(n_iter, n_eval, 50, 0.5, 4, 0.75, boundaries);
            let start = Instant::now();
            let result = vegas_plus.integrate_gpu(benchmark.burn_integrand().as_ref(), None);
            let duration = start.elapsed();

            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "GPU".to_string(),
                algorithm: "Vegas+".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });
        }
    }

    #[cfg(feature = "mpi")]
    if backend == Backend::All || backend == Backend::Mpi {
        let world = mpi::environment::world();
        if rank == 0 {
            println!("Running {} - Vegas+ MPI", name);
        }
        let mut vegas_plus = VegasPlus::new(n_iter, n_eval, 50, 0.5, 4, 0.75, boundaries);
        let start = Instant::now();
        let result = vegas_plus.integrate_mpi(benchmark.integrand().as_ref(), &world, None);
        let duration = start.elapsed();

        if rank == 0 {
            results.push(BenchmarkResult {
                integrand: name.to_string(),
                dim,
                n_eval,
                n_iter,
                backend: "MPI".to_string(),
                algorithm: "Vegas+".to_string(),
                time_s: duration.as_secs_f64(),
                value: result.value,
                error: result.error,
                chi2_dof: result.chi2_dof,
                analytical_result,
            });
        }
    }
}
