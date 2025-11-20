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

/// A trait representing a function to be integrated on the GPU using Burn.
///
/// Users of the library must implement this trait for their function.
#[cfg(feature = "gpu")]
pub trait BurnIntegrand<B: burn::prelude::Backend> {
    /// Returns the number of dimensions of the integration space.
    fn dim(&self) -> usize;

    /// Evaluates the function on a batch of points.
    ///
    /// # Arguments
    ///
    /// * `points`: A 2D tensor of shape `[n_points, dim]` representing the points
    ///             in the integration space.
    ///
    /// # Returns
    ///
    /// A 1D tensor of shape `[n_points]` with the function values `f(x)`.
    fn eval_burn(&self, points: burn::prelude::Tensor<B, 2>) -> burn::prelude::Tensor<B, 1>;
}
