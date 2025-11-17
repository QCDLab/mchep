//! The main VEGAS integrator.

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use std::convert::TryInto;
use wide::f64x4;

use crate::grid::Grid;
use crate::integrand::{Integrand, SimdIntegrand};

/// Stores the result of a VEGAS integration.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VegasResult {
    /// The estimated value of the integral.
    pub value: f64,
    /// The estimated statistical error (one standard deviation).
    pub error: f64,
    /// The chi-squared per degree of freedom of the partial results.
    pub chi2_dof: f64,
}

/// The VEGAS Monte Carlo integrator.
pub struct Vegas {
    /// The number of dimensions.
    dim: usize,
    /// The number of integration iterations.
    n_iter: usize,
    /// The number of integrand evaluations per iteration.
    n_eval: usize,
    /// The random number generator.
    rng: Pcg64,
    /// The adaptive grids for each dimension.
    grids: Vec<Grid>,
    /// The integration boundaries for each dimension, as (min, max) tuples.
    boundaries: Vec<(f64, f64)>,
}

impl Vegas {
    /// Creates a new VEGAS integrator.
    ///
    /// # Arguments
    ///
    /// * `n_iter`: The number of iterations to perform.
    /// * `n_eval`: The number of integrand evaluations per iteration.
    /// * `n_bins`: The number of bins for the adaptive grid in each dimension.
    /// * `alpha`: The grid damping factor. Should be between 0.0 and 1.0.
    /// * `boundaries`: A slice of `(min, max)` tuples defining the integration domain for each dimension.
    pub fn new(
        n_iter: usize,
        n_eval: usize,
        n_bins: usize,
        alpha: f64,
        boundaries: &[(f64, f64)],
    ) -> Self {
        let dim = boundaries.len();
        assert!(dim > 0, "Number of dimensions must be positive.");
        let grids = (0..dim).map(|_| Grid::new(n_bins, alpha)).collect();
        Vegas {
            dim,
            n_iter,
            n_eval,
            rng: Pcg64::from_entropy(),
            grids,
            boundaries: boundaries.to_vec(),
        }
    }

    /// Sets the seed for the random number generator.
    ///
    /// # Arguments
    ///
    /// * `seed`: The seed to use.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = Pcg64::seed_from_u64(seed);
    }

    /// Returns the number of dimensions of the integrator.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Integrates the given function using the VEGAS algorithm.
    pub fn integrate<F: Integrand + Sync>(&mut self, integrand: &F) -> VegasResult {
        assert_eq!(
            integrand.dim(),
            self.dim,
            "Integrand dimension does not match integrator dimension."
        );

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = self.run_iteration(integrand);

            if iter > 0 {
                iter_results.push(iter_val);
                iter_errors.push(iter_err);
            }

            for grid in &mut self.grids {
                grid.refine();
            }
        }

        // Combine results from all iterations (excluding warm-up)
        self.combine_results(&iter_results, &iter_errors)
    }

    /// Integrates the given function using the VEGAS algorithm with SIMD.
    pub fn integrate_simd<F: SimdIntegrand + Sync>(&mut self, integrand: &F) -> VegasResult {
        assert_eq!(
            integrand.dim(),
            self.dim,
            "Integrand dimension does not match integrator dimension."
        );

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = self.run_iteration_simd(integrand);

            if iter > 0 {
                iter_results.push(iter_val);
                iter_errors.push(iter_err);
            }

            for grid in &mut self.grids {
                grid.refine();
            }
        }

        self.combine_results(&iter_results, &iter_errors)
    }

    /// Runs a single iteration of the VEGAS algorithm in parallel.
    fn run_iteration<F: Integrand + Sync>(&mut self, integrand: &F) -> (f64, f64) {
        for grid in &mut self.grids {
            grid.reset_importance_data();
        }

        let n_bins = self.grids[0].n_bins();
        let n_eval = self.n_eval;
        let dim = self.dim;

        let mut random_ys: Vec<f64> = vec![0.0; n_eval * dim];
        self.rng.fill(&mut random_ys[..]);

        let (sum_f, sum_f2, d_updates, _, _) = random_ys
            .par_chunks_exact(dim)
            .fold(
                || {
                    (
                        0.0,
                        0.0,
                        vec![vec![0.0; n_bins]; dim],
                        vec![0.0; dim],
                        vec![0; dim],
                    )
                },
                |mut acc, y_vec| {
                    let (sum_f_thread, sum_f2_thread, d_updates_thread, point, bin_indices) =
                        &mut acc;

                    let mut jacobian = 1.0;
                    for d in 0..dim {
                        let y = y_vec[d];
                        let (bin_idx, x_unit, jac_vegas) = self.grids[d].map(y);
                        jacobian *= jac_vegas;
                        bin_indices[d] = bin_idx;

                        let (min, max) = self.boundaries[d];
                        point[d] = min + (max - min) * x_unit;
                        jacobian *= max - min;
                    }

                    let f_val = integrand.eval(point);
                    let weighted_f = f_val * jacobian;
                    let f2 = weighted_f * weighted_f;

                    *sum_f_thread += weighted_f;
                    *sum_f2_thread += f2;

                    let d_val = f2 / n_eval as f64;
                    for d in 0..dim {
                        d_updates_thread[d][bin_indices[d]] += d_val;
                    }

                    acc
                },
            )
            .reduce(
                || {
                    (
                        0.0,
                        0.0,
                        vec![vec![0.0; n_bins]; dim],
                        vec![0.0; dim],
                        vec![0; dim],
                    )
                },
                |mut a, b| {
                    a.0 += b.0;
                    a.1 += b.1;
                    for d in 0..dim {
                        for i in 0..n_bins {
                            a.2[d][i] += b.2[d][i];
                        }
                    }
                    a
                },
            );

        for d in 0..self.dim {
            self.grids[d].d.copy_from_slice(&d_updates[d]);
        }

        let avg_f = sum_f / self.n_eval as f64;
        let avg_f2 = sum_f2 / self.n_eval as f64;
        let variance = (avg_f2 - avg_f * avg_f) / (self.n_eval - 1).max(1) as f64;
        let error = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        (avg_f, error)
    }

    /// Runs a single iteration of the VEGAS algorithm using SIMD.
    fn run_iteration_simd<F: SimdIntegrand + Sync>(&mut self, integrand: &F) -> (f64, f64) {
        for grid in &mut self.grids {
            grid.reset_importance_data();
        }

        let n_bins = self.grids[0].n_bins();
        let n_packets = self.n_eval / 4;
        let n_eval_simd = n_packets * 4;

        let mut ys_soa: Vec<f64> = vec![0.0; self.dim * n_eval_simd];
        self.rng.fill(&mut ys_soa[..]);

        let (sum_f, sum_f2, d_updates) = (0..n_packets)
            .into_par_iter()
            .map(|p_idx| {
                let mut jacobian_v = f64x4::splat(1.0);
                let mut point_v = vec![f64x4::splat(0.0); self.dim];
                let mut bin_indices_arr = [[0; 4]; 32];

                for d in 0..self.dim {
                    let offset = d * n_eval_simd + p_idx * 4;
                    let y_packet = f64x4::new((&ys_soa[offset..offset + 4]).try_into().unwrap());

                    let (x_unit_v, jac_vegas_v, bins_arr) = self.grids[d].map_simd(y_packet);
                    bin_indices_arr[d] = bins_arr;

                    let (min, max) = self.boundaries[d];
                    let jac_boundary = max - min;

                    jacobian_v *= jac_vegas_v * f64x4::splat(jac_boundary);
                    point_v[d] = f64x4::splat(min) + x_unit_v * f64x4::splat(jac_boundary);
                }

                let f_vals_v = integrand.eval_simd(&point_v);
                let weighted_f_v = f_vals_v * jacobian_v;

                let f_sum = weighted_f_v.reduce_add();
                let f2_sum = (weighted_f_v * weighted_f_v).reduce_add();

                let mut d_updates_thread = vec![0.0; self.dim * n_bins];
                let d_val_arr =
                    (weighted_f_v * weighted_f_v / f64x4::splat(n_eval_simd as f64)).to_array();

                for i in 0..4 {
                    for d in 0..self.dim {
                        d_updates_thread[d * n_bins + bin_indices_arr[d][i]] += d_val_arr[i];
                    }
                }

                (f_sum, f2_sum, d_updates_thread)
            })
            .reduce(
                || (0.0, 0.0, vec![0.0; self.dim * n_bins]),
                |mut a, b| {
                    a.0 += b.0;
                    a.1 += b.1;
                    for i in 0..a.2.len() {
                        a.2[i] += b.2[i];
                    }
                    a
                },
            );

        for d in 0..self.dim {
            let start = d * n_bins;
            let end = (d + 1) * n_bins;
            self.grids[d].d.copy_from_slice(&d_updates[start..end]);
        }

        let avg_f = sum_f / n_eval_simd as f64;
        let avg_f2 = sum_f2 / n_eval_simd as f64;
        let variance = (avg_f2 - avg_f * avg_f) / (n_eval_simd - 1).max(1) as f64;
        let error = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        (avg_f, error)
    }

    /// Combines the results from multiple iterations into a final estimate.
    fn combine_results(&self, values: &[f64], errors: &[f64]) -> VegasResult {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (&val, &err) in values.iter().zip(errors.iter()) {
            if err > 0.0 {
                let weight = 1.0 / (err * err);
                weighted_sum += val * weight;
                total_weight += weight;
            }
        }

        if total_weight == 0.0 {
            return VegasResult {
                value: 0.0,
                error: 0.0,
                chi2_dof: 0.0,
            };
        }

        let final_value = weighted_sum / total_weight;
        let final_error = (1.0 / total_weight).sqrt();

        let mut chi2 = 0.0;
        for (&val, &err) in values.iter().zip(errors.iter()) {
            if err > 0.0 {
                chi2 += ((val - final_value) / err).powi(2);
            }
        }
        let dof = (values.len() - 1).max(1) as f64;
        let chi2_dof = chi2 / dof;

        VegasResult {
            value: final_value,
            error: final_error,
            chi2_dof,
        }
    }
}

#[cfg(feature = "gpu")]
impl Vegas {
    pub fn integrate_gpu<F: crate::integrand::GpuIntegrand>(
        &mut self,
        integrand: &F,
    ) -> VegasResult {
        assert_eq!(integrand.dim(), self.dim);

        let gpu_integrator = match crate::gpu::GpuIntegrator::new(integrand) {
            Ok(integrator) => integrator,
            Err(e) => {
                panic!("Failed to initialize GPU integrator: {}", e);
            }
        };

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = match gpu_integrator.run_iteration(
                &mut self.rng,
                &mut self.grids,
                &self.boundaries,
                self.n_eval,
                self.dim,
            ) {
                Ok(res) => res,
                Err(e) => {
                    panic!("GPU iteration failed: {}", e);
                }
            };

            if iter > 0 {
                iter_results.push(iter_val);
                iter_errors.push(iter_err);
            }

            // Refine grids on CPU
            for grid in &mut self.grids {
                grid.refine();
            }
        }

        self.combine_results(&iter_results, &iter_errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrand::{Integrand, SimdIntegrand};
    use wide::f64x4;

    // Integral of the form exp(-x^2 - y^2) in [-1, 1]^2.
    struct GaussianIntegrand;

    impl Integrand for GaussianIntegrand {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> f64 {
            (-(x[0].powi(2)) - x[1].powi(2)).exp()
        }
    }

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

    const ANALYTICAL_RESULT: f64 = 2.230985;

    #[test]
    fn test_integrate_gaussian() {
        let integrand = GaussianIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
        vegas.set_seed(1234);
        let result = vegas.integrate(&integrand);

        assert!(
            (result.value - ANALYTICAL_RESULT).abs() < 2.5 * result.error,
            "Analytical={} vs. MCHEP={}+/-{}",
            ANALYTICAL_RESULT,
            result.value,
            result.error
        );
        assert!(result.chi2_dof < 1.5, "chi2_dof: {}", result.chi2_dof);
    }

    #[test]
    fn test_integrate_gaussian_simd() {
        let integrand = GaussianSimdIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas = Vegas::new(10, 100_000, 50, 0.5, boundaries);
        vegas.set_seed(1234);
        let result = vegas.integrate_simd(&integrand);

        assert!(
            (result.value - ANALYTICAL_RESULT).abs() < 2.5 * result.error,
            "Analytical={} vs. MCHEP (SIMD)={}+/-{}",
            ANALYTICAL_RESULT,
            result.value,
            result.error
        );
        assert!(result.chi2_dof < 1.5, "chi2_dof: {}", result.chi2_dof);
    }
}
