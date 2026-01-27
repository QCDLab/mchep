//! The C-language interface for `MCHEP`

use mchep::integrand::{Integrand, SimdIntegrand};
use mchep::vegas::{Vegas, VegasResult};
use mchep::vegasplus::VegasPlus;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::slice;
use wide::f64x4;

#[cfg(feature = "mpi")]
use mpi::ffi::{RSMPI_COMM_SELF, RSMPI_COMM_WORLD};
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::*;

/// A C-compatible struct for integration boundaries.
#[repr(C)]
pub struct CBoundary {
    /// The minimum of the boundary.
    pub min: f64,
    /// The maximum of the boundary.
    pub max: f64,
}

/// The C-style integrand function pointer.
/// The first argument is the point `x` (an array of f64).
/// The second argument is the dimension.
/// The third is a user-provided `user_data` pointer.
pub type CIntegrand = extern "C" fn(*const f64, c_int, *mut c_void) -> f64;

/// The C-style SIMD integrand function pointer.
/// The first argument is the point `x` (an array of f64 in SoA layout, size dim * 4).
/// The second argument is the dimension.
/// The third is a user-provided `user_data` pointer.
/// The fourth is the result array (array of 4 f64).
pub type CSimdIntegrand = extern "C" fn(*const f64, c_int, *mut c_void, *mut f64);

/// A wrapper that implements the Rust `Integrand` trait.
struct CIntegrandWrapper {
    dim: usize,
    func: CIntegrand,
    user_data: *mut c_void,
}

impl Integrand for CIntegrandWrapper {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        (self.func)(x.as_ptr(), self.dim as c_int, self.user_data)
    }
}

/// A wrapper that implements the Rust `SimdIntegrand` trait.
struct CSimdIntegrandWrapper {
    dim: usize,
    func: CSimdIntegrand,
    user_data: *mut c_void,
}

impl SimdIntegrand for CSimdIntegrandWrapper {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
        let points_ptr = points.as_ptr() as *const f64;
        let mut result_arr = [0.0f64; 4];
        (self.func)(
            points_ptr,
            self.dim as c_int,
            self.user_data,
            result_arr.as_mut_ptr(),
        );
        f64x4::from(result_arr)
    }
}

/// This is unsafe, but required to integrate with Rayon.
/// The user of the C API is responsible for ensuring that the provided
/// integrand function is thread-safe.
unsafe impl Sync for CIntegrandWrapper {}
unsafe impl Sync for CSimdIntegrandWrapper {}

/// The opaque pointer to the Vegas integrator.
pub type VegasC = c_void;

/// Creates a new VEGAS integrator.
///
/// # Safety
///
/// `boundaries` must be a valid pointer to an array of `CBoundary`
/// of size `dim`.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_new(
    n_iter: usize,
    n_eval: usize,
    n_bins: usize,
    alpha: f64,
    dim: usize,
    boundaries: *const CBoundary,
) -> *mut VegasC {
    let boundaries_slice = unsafe { slice::from_raw_parts(boundaries, dim) };
    let rust_boundaries: Vec<(f64, f64)> =
        boundaries_slice.iter().map(|b| (b.min, b.max)).collect();

    let vegas = Vegas::new(n_iter, n_eval, n_bins, alpha, &rust_boundaries);
    let b = Box::new(vegas);
    Box::into_raw(b) as *mut VegasC
}

/// Integrates the given function using the VEGAS algorithm.
///
/// # Safety
/// `vegas_ptr` must be a valid pointer returned by `mchep_vegas_new`.
/// `integrand_func` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_integrate(
    vegas_ptr: *mut VegasC,
    integrand_func: CIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
) -> VegasResult {
    let vegas = unsafe { &mut *(vegas_ptr as *mut Vegas) };

    let integrand = CIntegrandWrapper {
        dim: vegas.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    vegas.integrate(&integrand, accuracy_opt)
}

/// Integrates the given function using the VEGAS algorithm with SIMD.
///
/// # Safety
///
/// `vegas_ptr` must be a valid pointer returned by `mchep_vegas_new`.
/// `integrand_func` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_integrate_simd(
    vegas_ptr: *mut VegasC,
    integrand_func: CSimdIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
) -> VegasResult {
    let vegas = unsafe { &mut *(vegas_ptr as *mut Vegas) };

    let integrand = CSimdIntegrandWrapper {
        dim: vegas.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    vegas.integrate_simd(&integrand, accuracy_opt)
}

/// Sets the seed for the random number generator.
///
/// # Safety
///
/// `vegas_ptr` must be a valid pointer returned by `mchep_vegas_new`.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_set_seed(vegas_ptr: *mut VegasC, seed: u64) {
    let vegas = unsafe { &mut *(vegas_ptr as *mut Vegas) };
    vegas.set_seed(seed);
}

/// Frees the memory of the VEGAS integrator.
///
/// # Safety
///
/// `vegas_ptr` must be a valid pointer returned by `mchep_vegas_new`
/// and must not be used afterward.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_free(vegas_ptr: *mut VegasC) {
    if !vegas_ptr.is_null() {
        drop(unsafe { Box::from_raw(vegas_ptr as *mut Vegas) });
    }
}

/// The opaque pointer to the VegasPlus integrator.
pub type VegasPlusC = c_void;

/// Creates a new VEGAS+ integrator.
///
/// # Safety
///
/// `boundaries` must be a valid pointer to an array of `CBoundary` of size `dim`.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_new(
    n_iter: usize,
    n_eval: usize,
    n_bins: usize,
    alpha: f64,
    n_strat: usize,
    beta: f64,
    dim: usize,
    boundaries: *const CBoundary,
) -> *mut VegasPlusC {
    let boundaries_slice = unsafe { slice::from_raw_parts(boundaries, dim) };
    let rust_boundaries: Vec<(f64, f64)> =
        boundaries_slice.iter().map(|b| (b.min, b.max)).collect();

    let vegas_plus = VegasPlus::new(
        n_iter,
        n_eval,
        n_bins,
        alpha,
        n_strat,
        beta,
        &rust_boundaries,
    );
    let b = Box::new(vegas_plus);
    Box::into_raw(b) as *mut VegasPlusC
}

/// Integrates the given function using the VEGAS+ algorithm.
///
/// # Safety
///
/// `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`.
/// `integrand_func` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_integrate(
    vegas_plus_ptr: *mut VegasPlusC,
    integrand_func: CIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
) -> VegasResult {
    let vegas_plus = unsafe { &mut *(vegas_plus_ptr as *mut VegasPlus) };

    let integrand = CIntegrandWrapper {
        dim: vegas_plus.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    vegas_plus.integrate(&integrand, accuracy_opt)
}

/// Integrates the given function using the VEGAS+ algorithm with SIMD.
///
/// # Safety
///
/// `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`.
/// `integrand_func` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_integrate_simd(
    vegas_plus_ptr: *mut VegasPlusC,
    integrand_func: CSimdIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
) -> VegasResult {
    let vegas_plus = unsafe { &mut *(vegas_plus_ptr as *mut VegasPlus) };

    let integrand = CSimdIntegrandWrapper {
        dim: vegas_plus.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    vegas_plus.integrate_simd(&integrand, accuracy_opt)
}

/// Sets the seed for the random number generator for VEGAS+.
///
/// # Safety
///
/// `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_set_seed(vegas_plus_ptr: *mut VegasPlusC, seed: u64) {
    let vegas_plus = unsafe { &mut *(vegas_plus_ptr as *mut VegasPlus) };
    vegas_plus.set_seed(seed);
}

/// Frees the memory of the VEGAS+ integrator.
///
/// # Safety
///
/// `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`
/// and must not be used afterward.
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_free(vegas_plus_ptr: *mut VegasPlusC) {
    if !vegas_plus_ptr.is_null() {
        drop(unsafe { Box::from_raw(vegas_plus_ptr as *mut VegasPlus) });
    }
}

// ============================================================================
// MPI Integration Functions
// ============================================================================

#[cfg(feature = "mpi")]
/// The raw MPI_Comm type for C interoperability.
/// This matches the MPI_Comm type from the system MPI library.
pub type MpiComm = mpi::ffi::MPI_Comm;

#[cfg(feature = "mpi")]
/// Integrates the given function using the VEGAS+ algorithm with MPI.
///
/// This function distributes the integration across MPI processes.
/// It should be called by all processes in the communicator.
/// The final result is returned on the root process (rank 0).
///
/// # Safety
///
/// - `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`.
/// - `integrand_func` must be a valid, thread-safe function pointer.
/// - `comm` must be a valid MPI communicator (e.g., MPI_COMM_WORLD).
/// - MPI must be initialized before calling this function.
///
/// # Example (C++)
///
/// ```cpp
/// #include <mpi.h>
/// #include <mchep.hpp>
///
/// int main(int argc, char** argv) {
///     MPI_Init(&argc, &argv);
///
///     auto* vp = mchep_vegas_plus_new(...);
///     VegasResult result = mchep_vegas_plus_integrate_mpi(
///         vp, my_integrand, nullptr, -1.0, MPI_COMM_WORLD);
///
///     int rank;
///     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
///     if (rank == 0) {
///         printf("Result: %f +/- %f\n", result.value, result.error);
///     }
///
///     mchep_vegas_plus_free(vp);
///     MPI_Finalize();
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_integrate_mpi(
    vegas_plus_ptr: *mut VegasPlusC,
    integrand_func: CIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
    comm: MpiComm,
) -> VegasResult {
    let vegas_plus = unsafe { &mut *(vegas_plus_ptr as *mut VegasPlus) };

    let integrand = CIntegrandWrapper {
        dim: vegas_plus.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    // Create a SimpleCommunicator from the raw MPI_Comm handle
    // We need to handle MPI_COMM_WORLD and MPI_COMM_SELF specially
    unsafe {
        if comm == RSMPI_COMM_WORLD {
            let world = SimpleCommunicator::world();
            vegas_plus.integrate_mpi(&integrand, &world, accuracy_opt)
        } else if comm == RSMPI_COMM_SELF {
            let self_comm = SimpleCommunicator::self_comm();
            vegas_plus.integrate_mpi(&integrand, &self_comm, accuracy_opt)
        } else {
            let user_comm = SimpleCommunicator::from_raw(comm);
            vegas_plus.integrate_mpi(&integrand, &user_comm, accuracy_opt)
        }
    }
}

#[cfg(feature = "mpi")]
/// Integrates the given SIMD function using the VEGAS+ algorithm with MPI.
///
/// This function combines MPI distribution across processes with SIMD
/// vectorization within each process, providing multiplicative speedup.
///
/// It should be called by all processes in the communicator.
/// The final result is returned on the root process (rank 0).
///
/// # Safety
///
/// - `vegas_plus_ptr` must be a valid pointer returned by `mchep_vegas_plus_new`.
/// - `integrand_func` must be a valid, thread-safe SIMD function pointer.
/// - `comm` must be a valid MPI communicator (e.g., MPI_COMM_WORLD).
/// - MPI must be initialized before calling this function.
///
/// # Example (C++)
///
/// ```cpp
/// #include <mpi.h>
/// #include <mchep.hpp>
/// #include <immintrin.h>
///
/// // AVX integrand function
/// void my_simd_integrand(const double* x, int dim, void* data, double* result) {
///     // x contains dim*4 values in SoA layout
///     // result should contain 4 output values
///     __m256d x0 = _mm256_loadu_pd(&x[0]);
///     // ... compute ...
///     _mm256_storeu_pd(result, output);
/// }
///
/// int main(int argc, char** argv) {
///     MPI_Init(&argc, &argv);
///
///     auto* vp = mchep_vegas_plus_new(...);
///     VegasResult result = mchep_vegas_plus_integrate_mpi_simd(
///         vp, my_simd_integrand, nullptr, -1.0, MPI_COMM_WORLD);
///
///     int rank;
///     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
///     if (rank == 0) {
///         printf("Result: %f +/- %f\n", result.value, result.error);
///     }
///
///     mchep_vegas_plus_free(vp);
///     MPI_Finalize();
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn mchep_vegas_plus_integrate_mpi_simd(
    vegas_plus_ptr: *mut VegasPlusC,
    integrand_func: CSimdIntegrand,
    user_data: *mut c_void,
    target_accuracy: f64,
    comm: MpiComm,
) -> VegasResult {
    let vegas_plus = unsafe { &mut *(vegas_plus_ptr as *mut VegasPlus) };

    let integrand = CSimdIntegrandWrapper {
        dim: vegas_plus.dim(),
        func: integrand_func,
        user_data,
    };

    let accuracy_opt = if target_accuracy > 0.0 {
        Some(target_accuracy)
    } else {
        None
    };

    // Create a SimpleCommunicator from the raw MPI_Comm handle
    // We need to handle MPI_COMM_WORLD and MPI_COMM_SELF specially
    unsafe {
        if comm == RSMPI_COMM_WORLD {
            let world = SimpleCommunicator::world();
            vegas_plus.integrate_mpi_simd(&integrand, &world, accuracy_opt)
        } else if comm == RSMPI_COMM_SELF {
            let self_comm = SimpleCommunicator::self_comm();
            vegas_plus.integrate_mpi_simd(&integrand, &self_comm, accuracy_opt)
        } else {
            let user_comm = SimpleCommunicator::from_raw(comm);
            vegas_plus.integrate_mpi_simd(&integrand, &user_comm, accuracy_opt)
        }
    }
}
