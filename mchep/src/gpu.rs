//! GPU-specific implementation for integration using Burn.

use crate::grid::Grid;
use crate::integrand::BurnIntegrand;
use burn::prelude::*;
use burn::tensor::Distribution;
use bytemuck;
use std::error::Error;

// Use WGPU backend for cross-platform GPU support.
pub type GpuBackend = burn::backend::Wgpu;

/// The Burn-based GPU integrator.
pub struct BurnIntegrator {
    device: <GpuBackend as Backend>::Device,
}

impl BurnIntegrator {
    /// Creates a new Burn integrator.
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let device = <GpuBackend as Backend>::Device::default();
        Ok(BurnIntegrator { device })
    }

    /// Runs a single iteration of the VEGAS algorithm on the GPU.
    pub fn run_iteration<F: BurnIntegrand<GpuBackend> + Sync>(
        &self,
        integrand: &F,
        grids: &mut [Grid],
        boundaries: &[(f64, f64)],
        n_eval: usize,
        dim: usize,
    ) -> Result<(f64, f64), Box<dyn Error>> {
        let ys: Tensor<GpuBackend, 2> =
            Tensor::random([n_eval, dim], Distribution::Uniform(0.0, 1.0), &self.device);

        let n_bins = grids[0].n_bins();
        let mut jacobian: Tensor<GpuBackend, 1> = Tensor::ones([n_eval], &self.device);
        let mut point: Tensor<GpuBackend, 2> = Tensor::zeros([n_eval, dim], &self.device);
        let mut all_bin_indices: Tensor<GpuBackend, 2, Int> =
            Tensor::zeros([n_eval, dim], &self.device);

        let boundaries_f32: Vec<(f32, f32)> = boundaries
            .iter()
            .map(|&(min, max)| (min as f32, max as f32))
            .collect();

        for d in 0..dim {
            let y_d = ys.clone().slice([0..n_eval, d..d + 1]).reshape([n_eval]);

            let bins_f32: Vec<f32> = grids[d].bins().iter().map(|&x| x as f32).collect();
            let grid_bins_d = Tensor::<GpuBackend, 1>::from_floats(&*bins_f32, &self.device);

            let n_bins_d = n_bins as f32;
            let y_scaled = y_d * n_bins_d;
            let bin_indices_d = y_scaled.clone().floor().int();
            let y_frac = y_scaled - bin_indices_d.clone().float();

            let bin_indices_clamped = bin_indices_d.clone().clamp(0, (n_bins - 1) as i32);

            let x_low = grid_bins_d.clone().gather(0, bin_indices_clamped.clone());
            let x_high = grid_bins_d.gather(0, bin_indices_clamped.clone() + 1);

            let width = x_high - x_low.clone();
            let x_unit_d = x_low + y_frac * width.clone();
            let jac_vegas_d = width * n_bins_d;

            jacobian = jacobian.clone() * jac_vegas_d;

            let (min_b, max_b) = boundaries_f32[d];
            let jac_boundary = max_b - min_b;
            let point_d = x_unit_d.mul_scalar(jac_boundary).add_scalar(min_b);
            jacobian = jacobian.clone() * jac_boundary;

            point = point.slice_assign([0..n_eval, d..d + 1], point_d.reshape([n_eval, 1]));
            all_bin_indices = all_bin_indices.slice_assign(
                [0..n_eval, d..d + 1],
                bin_indices_clamped.reshape([n_eval, 1]),
            );
        }

        let f_val = integrand.eval_burn(point);
        let weighted_f = f_val * jacobian;

        let sum_f: f32 = weighted_f.clone().sum().into_scalar();
        let sum_f2: f32 = (weighted_f.clone() * weighted_f.clone())
            .sum()
            .into_scalar();

        let avg_f = sum_f as f64 / n_eval as f64;
        let avg_f2 = sum_f2 as f64 / n_eval as f64;
        let variance = (avg_f2 - avg_f * avg_f) / (n_eval - 1).max(1) as f64;
        let error = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        let f2 = weighted_f.clone() * weighted_f;
        let d_val = f2 / n_eval as f32;

        for d in 0..dim {
            let bin_indices_d = all_bin_indices
                .clone()
                .slice([0..n_eval, d..d + 1])
                .reshape([n_eval]);
            let one_hot_matrix: Tensor<GpuBackend, 2> = bin_indices_d.one_hot(n_bins).float();
            let d_updates_d = d_val.clone().reshape([1, n_eval]).matmul(one_hot_matrix);
            let d_updates_data = d_updates_d.to_data();
            let d_updates_host: &[f32] = bytemuck::try_cast_slice(&d_updates_data.bytes).unwrap();
            grids[d]
                .d
                .copy_from_slice(&d_updates_host.iter().map(|&x| x as f64).collect::<Vec<_>>());
        }

        Ok((avg_f, error))
    }
}
