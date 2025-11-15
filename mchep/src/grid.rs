//! The adaptive grid used by the VEGAS algorithm.

use wide::f64x4;

/// Represents the adaptive grid for a single dimension.
#[derive(Debug, Clone)]
pub struct Grid {
    /// The number of bins in the grid.
    n_bins: usize,
    /// The grid boundaries, of size `n_bins + 1`.
    bins: Vec<f64>,
    /// The accumulated importance sampling data for each bin, of size `n_bins`.
    /// This corresponds to `d_i` from Eq. (17) of https://arxiv.org/pdf/2009.05112.
    pub(crate) d: Vec<f64>,
    /// The damping factor for grid adaptation.
    alpha: f64,
}

impl Grid {
    /// Creates a new uniform grid for a given number of bins.
    pub fn new(n_bins: usize, alpha: f64) -> Self {
        let mut bins = vec![0.0; n_bins + 1];
        for i in 0..=n_bins {
            bins[i] = i as f64 / n_bins as f64;
        }

        Grid {
            n_bins,
            bins,
            d: vec![0.0; n_bins],
            alpha,
        }
    }

    /// Resets the accumulated importance data to zero.
    pub fn reset_importance_data(&mut self) {
        self.d.fill(0.0);
    }

    /// Given a random number `y` in [0, 1], finds the corresponding grid bin,
    /// the mapped value `x`, and the jacobian for this dimension.
    pub fn map(&self, y: f64) -> (usize, f64, f64) {
        let y_scaled = y * self.n_bins as f64;
        let bin_index = y_scaled.floor() as usize;
        let y_frac = y_scaled - bin_index as f64;

        let bin_index = bin_index.min(self.n_bins - 1);

        let x_low = self.bins[bin_index];
        let x_high = self.bins[bin_index + 1];
        let width = x_high - x_low;

        let x = x_low + y_frac * width;
        let jacobian = width * self.n_bins as f64;

        (bin_index, x, jacobian)
    }

    /// Maps a packet of `4*y` values to `x` values and jacobians using SIMD.
    pub fn map_simd(&self, y_packet: f64x4) -> (f64x4, f64x4, [usize; 4]) {
        let n_bins_v = f64x4::splat(self.n_bins as f64);
        let y_scaled = y_packet * n_bins_v;
        let bin_indices_v = y_scaled.floor();
        let y_frac = y_scaled - bin_indices_v;

        let mut bin_indices_arr: [i64; 4] = bin_indices_v.to_array().map(|x| x as i64);

        for i in 0..4 {
            bin_indices_arr[i] = bin_indices_arr[i].min((self.n_bins - 1) as i64);
        }

        let x_low_arr = [
            self.bins[bin_indices_arr[0] as usize],
            self.bins[bin_indices_arr[1] as usize],
            self.bins[bin_indices_arr[2] as usize],
            self.bins[bin_indices_arr[3] as usize],
        ];
        let x_high_arr = [
            self.bins[bin_indices_arr[0] as usize + 1],
            self.bins[bin_indices_arr[1] as usize + 1],
            self.bins[bin_indices_arr[2] as usize + 1],
            self.bins[bin_indices_arr[3] as usize + 1],
        ];

        let x_low = f64x4::from(x_low_arr);
        let x_high = f64x4::from(x_high_arr);

        let width = x_high - x_low;
        let x = x_low + y_frac * width;
        let jacobian = width * n_bins_v;

        let final_bin_indices = [
            bin_indices_arr[0] as usize,
            bin_indices_arr[1] as usize,
            bin_indices_arr[2] as usize,
            bin_indices_arr[3] as usize,
        ];

        (x, jacobian, final_bin_indices)
    }

    /// Refines the grid based on the accumulated importance data.
    pub fn refine(&mut self) {
        let mut smoothed_d = self.d.clone();
        if self.n_bins > 2 {
            smoothed_d[0] = (7.0 * self.d[0] + self.d[1]) / 8.0;
            let n = self.n_bins - 1;
            smoothed_d[n] = (self.d[n - 1] + 7.0 * self.d[n]) / 8.0;
            for i in 1..n {
                smoothed_d[i] = (self.d[i - 1] + 6.0 * self.d[i] + self.d[i + 1]) / 8.0;
            }
        }

        let total_d: f64 = smoothed_d.iter().sum();
        if total_d <= 0.0 {
            return;
        }
        let avg_d = total_d / self.n_bins as f64;

        let mut compressed_d = vec![0.0; self.n_bins];
        for i in 0..self.n_bins {
            if smoothed_d[i] > 0.0 {
                let r = smoothed_d[i] / avg_d;
                compressed_d[i] = ((r - 1.0) / r.ln()).powf(self.alpha);
            } else {
                compressed_d[i] = 0.0;
            }
            if compressed_d[i].is_nan() || compressed_d[i].is_infinite() {
                compressed_d[i] = 1.0;
            }
        }

        let total_compressed_d: f64 = compressed_d.iter().sum();
        let desired_d_per_bin = total_compressed_d / self.n_bins as f64;

        let mut new_bins = vec![0.0; self.n_bins + 1];
        new_bins[0] = self.bins[0];
        new_bins[self.n_bins] = self.bins[self.n_bins];

        let mut current_d_sum = 0.0;
        let mut old_bin_idx = 0;

        for i in 1..self.n_bins {
            while current_d_sum < i as f64 * desired_d_per_bin {
                current_d_sum += compressed_d[old_bin_idx];
                old_bin_idx += 1;
                if old_bin_idx >= self.n_bins {
                    break;
                }
            }
            old_bin_idx -= 1;
            current_d_sum -= compressed_d[old_bin_idx];

            let overshoot = i as f64 * desired_d_per_bin - current_d_sum;
            let fraction = overshoot / compressed_d[old_bin_idx];

            let old_bin_width = self.bins[old_bin_idx + 1] - self.bins[old_bin_idx];
            new_bins[i] = self.bins[old_bin_idx] + fraction * old_bin_width;
        }

        self.bins = new_bins;
    }

    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
}
