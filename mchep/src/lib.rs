//! `MCHeP` is a multi-dimensional Monte Carlo integration routine.
//!
//! Currently, `MCHeP` implements the VEGAS (adaptive importance sampling)
//! and VEGAS+ adaptive stratified sampling algorithms.

pub mod grid;
pub mod integrand;
pub mod vegas;
pub mod vegasplus;

#[cfg(feature = "mpi")]
pub mod mpi;

#[cfg(feature = "gpu")]
pub mod gpu;
