//! GPU-specific implementation for integration.

use crate::grid::Grid;
use crate::integrand::GpuIntegrand;
use cust::prelude::*;
use rand::Rng;
use std::error::Error;

pub struct GpuIntegrator {
    _context: Context,
    stream: Stream,
    module: Module,
}

impl GpuIntegrator {
    pub fn new<F: GpuIntegrand>(integrand: &F) -> Result<Self, Box<dyn Error>> {
        cust::init(CudaFlags::empty())?;

        let device = Device::get_device(0)?;
        let _context = Context::new(device)?;

        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        let ptx_path = integrand.ptx_path();
        let module = Module::from_file(ptx_path)?;

        Ok(GpuIntegrator {
            _context,
            stream,
            module,
        })
    }

    pub fn run_iteration(
        &self,
        grids: &mut [Grid],
        boundaries: &[(f64, f64)],
        n_eval: usize,
        dim: usize,
    ) -> Result<(f64, f64), Box<dyn Error>> {
        // TODO: For the time-being, for the sake of simplicity, we generate random numbers
        // on CPU and copy them over. A more advanced implementation would do this on the GPU.
        let mut rng = rand::thread_rng();
        let random_ys: Vec<f64> = (0..(n_eval * dim)).map(|_| rng.gen()).collect();

        let n_bins = grids[0].n_bins();
        let grid_bins: Vec<f64> = grids.iter().flat_map(|g| g.bins.clone()).collect();
        let boundaries_flat: Vec<f64> = boundaries
            .iter()
            .flat_map(|(min, max)| vec![*min, *max])
            .collect();

        let mut ys_gpu = random_ys.as_slice().as_dbuf()?;
        let mut grid_bins_gpu = grid_bins.as_slice().as_dbuf()?;
        let mut boundaries_gpu = boundaries_flat.as_slice().as_dbuf()?;

        let mut results_gpu = unsafe { DeviceBuffer::<f64>::uninitialized(n_eval)? };
        let mut d_updates_gpu = unsafe { DeviceBuffer::<f64>::uninitialized(dim * n_bins)? };
        d_updates_gpu.copy_from(&vec![0.0; dim * n_bins])?;

        let func = self.module.get_function("integrand_ker")?;
        let block_size = 256;
        let grid_size = (n_eval as u32 + block_size - 1) / block_size;

        unsafe {
            launch!(
                func<<<grid_size, block_size, 0, self.stream>>>(
                    ys_gpu.as_device_ptr(),
                    results_gpu.as_device_ptr(),
                    grid_bins_gpu.as_device_ptr(),
                    boundaries_gpu.as_device_ptr(),
                    d_updates_gpu.as_device_ptr(),
                    n_eval as u32,
                    dim as u32,
                    n_bins as u32
                )
            )?;
        }

        self.stream.synchronize()?;

        let mut results_host = vec![0.0; n_eval];
        results_gpu.copy_to(&mut results_host)?;

        let mut d_updates_host = vec![0.0; dim * n_bins];
        d_updates_gpu.copy_to(&mut d_updates_host)?;

        let sum_f: f64 = results_host.iter().sum();
        let sum_f2: f64 = results_host.iter().map(|f| f * f).sum();

        let avg_f = sum_f / n_eval as f64;
        let avg_f2 = sum_f2 / n_eval as f64;
        let variance = (avg_f2 - avg_f * avg_f) / (n_eval - 1).max(1) as f64;
        let error = if variance > 0.0 { variance.sqrt() } else { 0.0 };

        for d in 0..dim {
            let start = d * n_bins;
            let end = (d + 1) * n_bins;
            grids[d].d.copy_from_slice(&d_updates_host[start..end]);
        }

        Ok((avg_f, error))
    }
}

/// Example CUDA kernel. This would be in a .cu file and compiled to PTX.
pub const EXAMPLE_KERNEL: &str = r#"
extern "C" __device__ void map(
    const double y,
    const double* grid_bins,
    const int n_bins,
    int* bin_idx,
    double* x_unit,
    double* jac_vegas
) {
    double y_scaled = y * n_bins;
    *bin_idx = min(n_bins - 1, (int)floor(y_scaled));
    double y_frac = y_scaled - (*bin_idx);

    double x_low = grid_bins[*bin_idx];
    double x_high = grid_bins[*bin_idx + 1];
    double width = x_high - x_low;

    *x_unit = x_low + y_frac * width;
    *jac_vegas = width * n_bins;
}

// This is the user-provided part of the kernel
extern "C" __device__ double user_integrand(const double* x, int dim);

extern "C" __global__ void integrand_ker(
    const double* ys,
    double* results,
    const double* all_grid_bins,
    const double* boundaries,
    double* d_updates,
    unsigned int n_eval,
    unsigned int dim,
    unsigned int n_bins
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_eval) return;

    double point[32]; // Max dimensions, should be passed or a template
    int bin_indices[32];

    double jacobian = 1.0;

    for (unsigned int d = 0; d < dim; ++d) {
        double y = ys[idx * dim + d];
        const double* grid_bins_d = &all_grid_bins[d * (n_bins + 1)];

        int bin_idx;
        double x_unit;
        double jac_vegas;
        map(y, grid_bins_d, n_bins, &bin_idx, &x_unit, &jac_vegas);

        jacobian *= jac_vegas;
        bin_indices[d] = bin_idx;

        double min_b = boundaries[d * 2];
        double max_b = boundaries[d * 2 + 1];
        double jac_boundary = max_b - min_b;
        point[d] = min_b + x_unit * jac_boundary;
        jacobian *= jac_boundary;
    }

    double f_val = user_integrand(point, dim);
    double weighted_f = f_val * jacobian;
    results[idx] = weighted_f;

    double f2 = weighted_f * weighted_f;
    double d_val = f2 / n_eval;

    for (unsigned int d = 0; d < dim; ++d) {
        atomicAdd(&d_updates[d * n_bins + bin_indices[d]], d_val);
    }
}
"#;
