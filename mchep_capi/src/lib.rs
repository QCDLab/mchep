//! The C-language interface for `MCHEP`

use std::ffi::c_void;
use std::os::raw::c_int;
use std::slice;
use wide::f64x4;

use mchep::integrand::{Integrand, SimdIntegrand};
use mchep::vegas::{Vegas, VegasResult};

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
) -> VegasResult {
    let vegas = unsafe { &mut *(vegas_ptr as *mut Vegas) };

    let integrand = CIntegrandWrapper {
        dim: vegas.dim(),
        func: integrand_func,
        user_data,
    };

    vegas.integrate(&integrand)
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
) -> VegasResult {
    let vegas = unsafe { &mut *(vegas_ptr as *mut Vegas) };

    let integrand = CSimdIntegrandWrapper {
        dim: vegas.dim(),
        func: integrand_func,
        user_data,
    };

    vegas.integrate_simd(&integrand)
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
