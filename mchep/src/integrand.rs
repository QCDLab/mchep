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

/// A trait for integrands that, alongside the scalar value driving the
/// VEGAS/VEGAS+ importance-sampling estimator, also emit a side-channel of
/// auxiliary per-point observations -- for example, differential-grid fills
/// for a downstream interpolation grid (as used by e.g. `PineAPPL` in HEP
/// cross-section calculations), where every sampled point needs to be
/// recorded individually rather than only contributing to a single running
/// total.
pub trait ObservableIntegrand {
    /// The type of a single auxiliary observation recorded per point.
    type Observation: Send;

    /// Returns the number of dimensions of the integration space.
    fn dim(&self) -> usize;

    /// Evaluates the function at `x`.
    ///
    /// `fill_weight` is this point's total Monte Carlo weight: the
    /// VEGAS/VEGAS+ jacobian already normalized by this point's share of
    /// its hypercube's sample count and by the number of iterations
    /// contributing to the final result. Summing `fill_weight` (or any
    /// quantity proportional to it, e.g. a per-channel piece of `f(x)`)
    /// over every point across every counted iteration reproduces the same
    /// total the integrator itself reports. Implementations that need to
    /// record a differential quantity at `x` should scale their own
    /// per-observation weights by this value rather than re-deriving it.
    ///
    /// Returns the scalar value `f(x)` (unweighted, matching
    /// [`Integrand::eval`]) together with any observations to record at
    /// this point.
    fn eval(&self, x: &[f64], fill_weight: f64) -> (f64, Vec<Self::Observation>);
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
