//! The `Integrand` trait, which defines the function to be integrated.

use wide::f64x4;

/// A trait representing a function to be integrated.
///
/// Users of the library must implement this trait for their function.
pub trait Integrand {
    /// Returns the number of dimensions of the integration space.
    fn dim(&self) -> usize;

    /// Evaluates the function at a given point `x`.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `f64` representing the point in the integration space.
    ///
    /// # Returns
    ///
    /// The value of the function `f(x)`.
    fn eval(&self, x: &[f64]) -> f64;
}

/// A trait representing a function to be integrated using SIMD.
pub trait SimdIntegrand {
    /// Returns the number of dimensions of the integration space.
    fn dim(&self) -> usize;

    /// Evaluates the function on a packet of 4 points.
    fn eval_simd(&self, points: &[f64x4]) -> f64x4;
}

/// A trait representing a function to be integrated on the GPU.
///
/// Users of the library must implement this trait for their function.
#[cfg(feature = "gpu")]
pub trait GpuIntegrand {
    /// Returns the number of dimensions of the integration space.
    fn dim(&self) -> usize;

    /// Returns the path to the compiled PTX file for the integrand function.
    ///
    /// The PTX file should contain a `__global__` function with a signature like:
    /// `extern "C" __global__ void integrand_ker(const double* points, double* results, int n_points, int dim)`
    fn ptx_path(&self) -> &str;
}
