//! The VEGAS+ integrator, which adds adaptive stratified sampling.

use crate::grid::Grid;
use crate::integrand::{Integrand, SimdIntegrand};
use crate::vegas::VegasResult;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use wide::f64x4;

/// Stores the state of a single hypercube for stratified sampling.
#[derive(Debug, Clone)]
pub(crate) struct Hypercube {
    /// The number of samples to be evaluated in this hypercube.
    pub(crate) n_samples: usize,
    /// The estimated variance of the integrand within this hypercube
    /// from the last iteration.
    pub(crate) variance: f64,
}

impl Default for Hypercube {
    fn default() -> Self {
        Hypercube {
            n_samples: 2,
            variance: 0.0,
        }
    }
}

/// The VEGAS+ Monte Carlo integrator.
pub struct VegasPlus {
    pub(crate) dim: usize,
    pub(crate) n_iter: usize,
    pub(crate) n_eval: usize,
    pub(crate) rng: Pcg64,
    pub(crate) grids: Vec<Grid>,
    pub(crate) boundaries: Vec<(f64, f64)>,
    pub(crate) beta: f64,
    pub(crate) n_strat: usize,
    pub(crate) hypercubes: Vec<Hypercube>,
}

// Struct to hold the results of processing a single hypercube
struct HypercubeResult {
    sum_f: f64,
    sum_f2: f64,
    variance: f64,
    d_updates: Vec<Vec<f64>>,
}

impl VegasPlus {
    pub fn new(
        n_iter: usize,
        n_eval: usize,
        n_bins: usize,
        alpha: f64,
        n_strat: usize,
        beta: f64,
        boundaries: &[(f64, f64)],
    ) -> Self {
        let dim = boundaries.len();
        let n_hypercubes = n_strat.pow(dim as u32);
        assert!(
            n_eval >= 2 * n_hypercubes,
            "n_eval is too small for the given number of dimensions and stratifications."
        );

        let grids = (0..dim).map(|_| Grid::new(n_bins, alpha)).collect();
        let hypercubes = vec![Hypercube::default(); n_hypercubes];

        let mut vegas_plus = VegasPlus {
            dim,
            n_iter,
            n_eval,
            rng: Pcg64::from_entropy(),
            grids,
            boundaries: boundaries.to_vec(),
            beta,
            n_strat,
            hypercubes,
        };
        vegas_plus.reallocate_samples();
        vegas_plus
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

    /// Integrates the given function using the VEGAS+ algorithm.
    pub fn integrate<F: Integrand + Sync>(
        &mut self,
        integrand: &F,
        target_accuracy: Option<f64>,
    ) -> VegasResult {
        assert_eq!(integrand.dim(), self.dim);

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = self.run_iteration(integrand);

            if iter > 0 {
                iter_results.push(iter_val);
                iter_errors.push(iter_err);

                if let Some(acc_req) = target_accuracy {
                    if !iter_results.is_empty() {
                        let current_result = self.combine_results(&iter_results, &iter_errors);
                        if current_result.value != 0.0 {
                            let current_acc =
                                (current_result.error / current_result.value.abs()) * 100.0;
                            if current_acc < acc_req {
                                return current_result;
                            }
                        }
                    }
                }
            }

            for grid in &mut self.grids {
                grid.refine();
            }
            self.reallocate_samples();
        }

        self.combine_results(&iter_results, &iter_errors)
    }

    /// Integrates the given function using the VEGAS+ algorithm with SIMD.
    pub fn integrate_simd<F: SimdIntegrand + Sync>(
        &mut self,
        integrand: &F,
        target_accuracy: Option<f64>,
    ) -> VegasResult {
        assert_eq!(integrand.dim(), self.dim);

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = self.run_iteration_simd(integrand);

            if iter > 0 {
                iter_results.push(iter_val);
                iter_errors.push(iter_err);

                if let Some(acc_req) = target_accuracy {
                    if !iter_results.is_empty() {
                        let current_result = self.combine_results(&iter_results, &iter_errors);
                        if current_result.value != 0.0 {
                            let current_acc =
                                (current_result.error / current_result.value.abs()) * 100.0;
                            if current_acc < acc_req {
                                return current_result;
                            }
                        }
                    }
                }
            }

            for grid in &mut self.grids {
                grid.refine();
            }
            self.reallocate_samples();
        }

        self.combine_results(&iter_results, &iter_errors)
    }

    pub(crate) fn run_iteration<F: Integrand + Sync>(&mut self, integrand: &F) -> (f64, f64) {
        for grid in &mut self.grids {
            grid.reset_importance_data();
        }

        let n_bins = self.grids[0].n_bins();

        let seeds: Vec<u64> = (0..self.hypercubes.len()).map(|_| self.rng.gen()).collect();

        let results: Vec<HypercubeResult> = seeds
            .into_par_iter()
            .enumerate()
            .map(|(h_idx, seed)| {
                let mut seed_array = [0u8; 32];
                seed_array[..8].copy_from_slice(&seed.to_le_bytes());
                let mut thread_rng = Pcg64::from_seed(seed_array);
                let n_samples_h = self.hypercubes[h_idx].n_samples;
                let mut point = vec![0.0; self.dim];
                let mut bin_indices = vec![0; self.dim];
                let mut d_updates_thread =
                    (0..self.dim).map(|_| vec![0.0; n_bins]).collect::<Vec<_>>();

                let mut sum_f_h = 0.0;
                let mut sum_f2_h = 0.0;

                if n_samples_h == 0 {
                    return HypercubeResult {
                        sum_f: 0.0,
                        sum_f2: 0.0,
                        variance: 0.0,
                        d_updates: d_updates_thread,
                    };
                }

                for _ in 0..n_samples_h {
                    let mut jacobian = 1.0;
                    let mut y_vec = vec![0.0; self.dim];

                    let h_coords = self.get_hypercube_coords(h_idx);
                    for d in 0..self.dim {
                        let y_rand = thread_rng.gen::<f64>();
                        let strat_width = 1.0 / self.n_strat as f64;
                        y_vec[d] = (h_coords[d] as f64 + y_rand) * strat_width;
                    }

                    for d in 0..self.dim {
                        let (bin_idx, x_unit, jac_vegas) = self.grids[d].map(y_vec[d]);
                        jacobian *= jac_vegas;
                        bin_indices[d] = bin_idx;

                        let (min, max) = self.boundaries[d];
                        let jac_boundary = max - min;
                        point[d] = min + x_unit * jac_boundary;
                        jacobian *= jac_boundary;
                    }

                    let f_val = integrand.eval(&point);
                    let weighted_f = f_val * jacobian;
                    sum_f_h += weighted_f;
                    sum_f2_h += weighted_f * weighted_f;

                    let d_val = (weighted_f * weighted_f) / self.n_eval as f64;
                    for d in 0..self.dim {
                        d_updates_thread[d][bin_indices[d]] += d_val;
                    }
                }

                let avg_f_h = sum_f_h / n_samples_h as f64;
                let avg_f2_h = sum_f2_h / n_samples_h as f64;
                let var_g_h = (avg_f2_h - avg_f_h.powi(2)).max(0.0)
                    * (n_samples_h as f64 / (n_samples_h - 1).max(1) as f64);

                HypercubeResult {
                    sum_f: sum_f_h,
                    sum_f2: sum_f2_h,
                    variance: var_g_h.sqrt(),
                    d_updates: d_updates_thread,
                }
            })
            .collect();

        let mut total_value = 0.0;
        let mut total_variance = 0.0;
        let hypercube_volume = (1.0 / self.n_strat as f64).powi(self.dim as i32);

        for (h_idx, result) in results.iter().enumerate() {
            let n_samples_h = self.hypercubes[h_idx].n_samples;

            if n_samples_h > 0 {
                let avg_g_h = result.sum_f / n_samples_h as f64;
                total_value += hypercube_volume * avg_g_h;
            }

            if n_samples_h > 1 {
                let avg_g_h = result.sum_f / n_samples_h as f64;
                let avg_g2_h = result.sum_f2 / n_samples_h as f64;
                let var_g_h = (avg_g2_h - avg_g_h.powi(2)).max(0.0)
                    * (n_samples_h as f64 / (n_samples_h - 1) as f64);
                total_variance += hypercube_volume.powi(2) * var_g_h / n_samples_h as f64;
            }

            self.hypercubes[h_idx].variance = result.variance;
            for d in 0..self.dim {
                for i in 0..n_bins {
                    self.grids[d].d[i] += result.d_updates[d][i];
                }
            }
        }

        let error = total_variance.sqrt();

        (total_value, error)
    }

    pub(crate) fn run_iteration_simd<F: SimdIntegrand + Sync>(&mut self, integrand: &F) -> (f64, f64) {
        for grid in &mut self.grids {
            grid.reset_importance_data();
        }

        let n_bins = self.grids[0].n_bins();

        let seeds: Vec<u64> = (0..self.hypercubes.len()).map(|_| self.rng.gen()).collect();

        let results: Vec<HypercubeResult> = seeds
            .into_par_iter()
            .enumerate()
            .map(|(h_idx, seed)| {
                let mut seed_array = [0u8; 32];
                seed_array[..8].copy_from_slice(&seed.to_le_bytes());
                let mut thread_rng = Pcg64::from_seed(seed_array);
                let n_samples_h = self.hypercubes[h_idx].n_samples;
                let n_packets = n_samples_h / 4;
                let n_samples_h_simd = n_packets * 4;

                let mut d_updates_thread =
                    (0..self.dim).map(|_| vec![0.0; n_bins]).collect::<Vec<_>>();

                let mut sum_f_h = 0.0;
                let mut sum_f2_h = 0.0;

                if n_samples_h_simd == 0 {
                    return HypercubeResult {
                        sum_f: 0.0,
                        sum_f2: 0.0,
                        variance: 0.0,
                        d_updates: d_updates_thread,
                    };
                }

                let h_coords = self.get_hypercube_coords(h_idx);
                let strat_width = 1.0 / self.n_strat as f64;

                // SIMD part
                for _ in 0..n_packets {
                    let mut jacobian_v = f64x4::splat(1.0);
                    let mut point_v = vec![f64x4::splat(0.0); self.dim];
                    let mut bin_indices_arr = vec![[0; 4]; self.dim];

                    let mut y_vec_v = vec![f64x4::splat(0.0); self.dim];

                    for d in 0..self.dim {
                        let y_rand_v = f64x4::new([
                            thread_rng.gen::<f64>(),
                            thread_rng.gen::<f64>(),
                            thread_rng.gen::<f64>(),
                            thread_rng.gen::<f64>(),
                        ]);
                        y_vec_v[d] = (f64x4::splat(h_coords[d] as f64) + y_rand_v)
                            * f64x4::splat(strat_width);
                    }

                    for d in 0..self.dim {
                        let (x_unit_v, jac_vegas_v, bins_arr) = self.grids[d].map_simd(y_vec_v[d]);
                        jacobian_v *= jac_vegas_v;
                        bin_indices_arr[d] = bins_arr;

                        let (min, max) = self.boundaries[d];
                        let jac_boundary = max - min;
                        point_v[d] = f64x4::splat(min) + x_unit_v * f64x4::splat(jac_boundary);
                        jacobian_v *= f64x4::splat(jac_boundary);
                    }

                    let f_vals_v = integrand.eval_simd(&point_v);
                    let weighted_f_v = f_vals_v * jacobian_v;

                    sum_f_h += weighted_f_v.reduce_add();
                    sum_f2_h += (weighted_f_v * weighted_f_v).reduce_add();

                    let d_val_arr =
                        (weighted_f_v * weighted_f_v / f64x4::splat(self.n_eval as f64)).to_array();

                    for i in 0..4 {
                        for d in 0..self.dim {
                            d_updates_thread[d][bin_indices_arr[d][i]] += d_val_arr[i];
                        }
                    }
                }

                let mut var_g_h = 0.0;
                if n_samples_h_simd > 1 {
                    let avg_f_h = sum_f_h / n_samples_h_simd as f64;
                    let avg_f2_h = sum_f2_h / n_samples_h_simd as f64;
                    var_g_h = (avg_f2_h - avg_f_h.powi(2)).max(0.0)
                        * (n_samples_h_simd as f64 / (n_samples_h_simd - 1) as f64);
                }

                HypercubeResult {
                    sum_f: sum_f_h,
                    sum_f2: sum_f2_h,
                    variance: var_g_h.sqrt(),
                    d_updates: d_updates_thread,
                }
            })
            .collect();

        let mut total_value = 0.0;
        let mut total_variance = 0.0;
        let hypercube_volume = (1.0 / self.n_strat as f64).powi(self.dim as i32);

        for (h_idx, result) in results.iter().enumerate() {
            let n_samples_h = self.hypercubes[h_idx].n_samples;
            let n_samples_h_simd = (n_samples_h / 4) * 4;

            if n_samples_h_simd > 0 {
                let avg_g_h = result.sum_f / n_samples_h_simd as f64;
                total_value += hypercube_volume * avg_g_h;
            }

            if n_samples_h_simd > 1 {
                let avg_g_h = result.sum_f / n_samples_h_simd as f64;
                let avg_g2_h = result.sum_f2 / n_samples_h_simd as f64;
                let var_g_h = (avg_g2_h - avg_g_h.powi(2)).max(0.0)
                    * (n_samples_h_simd as f64 / (n_samples_h_simd - 1) as f64);
                total_variance += hypercube_volume.powi(2) * var_g_h / n_samples_h_simd as f64;
            }

            self.hypercubes[h_idx].variance = result.variance;
            for d in 0..self.dim {
                for i in 0..n_bins {
                    self.grids[d].d[i] += result.d_updates[d][i];
                }
            }
        }

        let error = total_variance.sqrt();

        (total_value, error)
    }

    /// Converts a hypercube index to its coordinates in the stratified grid.
    fn get_hypercube_coords(&self, index: usize) -> Vec<usize> {
        let mut coords = vec![0; self.dim];
        let mut current_index = index;
        for d in (0..self.dim).rev() {
            coords[d] = current_index % self.n_strat;
            current_index /= self.n_strat;
        }
        coords
    }

    pub(crate) fn reallocate_samples(&mut self) {
        let total_variance_damped: f64 = self
            .hypercubes
            .iter()
            .map(|h| h.variance.powf(self.beta))
            .sum();

        if total_variance_damped <= 0.0 {
            let samples_per_cube = (self.n_eval / self.hypercubes.len()).max(2);
            for cube in &mut self.hypercubes {
                cube.n_samples = samples_per_cube;
            }
            return;
        }

        let mut total_allocated = 0;
        for cube in &mut self.hypercubes {
            let fraction = cube.variance.powf(self.beta) / total_variance_damped;
            let desired_samples = (self.n_eval as f64 * fraction) as usize;
            cube.n_samples = desired_samples.max(2);
            total_allocated += cube.n_samples;
        }

        let remainder = self.n_eval.saturating_sub(total_allocated);
        if remainder > 0 {
            let mut sorted_cubes: Vec<(usize, f64)> = self
                .hypercubes
                .iter()
                .map(|h| h.variance)
                .enumerate()
                .collect();
            sorted_cubes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let n_hypercubes = self.hypercubes.len();
            for i in 0..remainder {
                self.hypercubes[sorted_cubes[i % n_hypercubes].0].n_samples += 1;
            }
        }
    }

    pub(crate) fn combine_results(&self, values: &[f64], errors: &[f64]) -> VegasResult {
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
        let dof = (values.len() - 1).max(1) as f64;
        let chi2 = values
            .iter()
            .zip(errors.iter())
            .fold(0.0, |acc, (&val, &err)| {
                if err > 0.0 {
                    acc + ((val - final_value) / err).powi(2)
                } else {
                    acc
                }
            });

        VegasResult {
            value: final_value,
            error: final_error,
            chi2_dof: chi2 / dof,
        }
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

    const ANALYTICAL_RESULT: f64 = 2.230985;

    #[test]
    fn test_integrate_gaussian_plus() {
        let integrand = GaussianIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas_plus = VegasPlus::new(20, 200_000, 50, 0.5, 4, 0.75, boundaries);
        let result = vegas_plus.integrate(&integrand, None);
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
    fn test_accuracy_goal_plus() {
        let integrand = GaussianIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas_plus = VegasPlus::new(10, 200_000, 50, 0.5, 4, 0.75, boundaries);
        vegas_plus.set_seed(4321);
        let result = vegas_plus.integrate(&integrand, Some(0.1));

        let accuracy = (result.error / result.value.abs()) * 100.0;
        assert!(accuracy < 0.1);
        assert!(
            (result.value - ANALYTICAL_RESULT).abs() < 3. * result.error,
            "Analytical={} vs. MCHEP={}+/-{}",
            ANALYTICAL_RESULT,
            result.value,
            result.error
        );
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

    #[test]
    fn test_integrate_gaussian_plus_simd() {
        let integrand = GaussianSimdIntegrand;
        let boundaries = &[(-1.0, 1.0), (-1.0, 1.0)];
        let mut vegas_plus = VegasPlus::new(20, 200_000, 50, 0.5, 4, 0.75, boundaries);
        vegas_plus.set_seed(1234);
        let result = vegas_plus.integrate_simd(&integrand, None);
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
