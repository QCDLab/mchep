//! The `Integrand` trait, which defines the function to be integrated.

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
