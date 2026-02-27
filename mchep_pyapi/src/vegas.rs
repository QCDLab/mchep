//! VEGAS interface.

use std::convert::TryFrom;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use mchep::integrand::{Integrand, SimdIntegrand};
use mchep::vegas::{Vegas, VegasResult};
use mchep::vegasplus::VegasPlus;
use wide::f64x4;

// A wrapper for Python callables to implement the Integrand trait
#[pyclass(name = "Integrand")]
struct PyIntegrand {
    callable: Py<PyAny>,
    dim: usize,
}

impl Integrand for PyIntegrand {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval(&self, x: &[f64]) -> f64 {
        Python::attach(|py| {
            let args = (PyList::new(py, x).unwrap(),);
            self.callable
                .call1(py, args)
                .and_then(|result| result.extract::<f64>(py))
                .unwrap_or_else(|err| {
                    eprintln!("Error evaluating integrand: {err}");
                    0.0
                })
        })
    }
}

// CRITICAL: Mark as Send + Sync for parallel execution with Rayon
unsafe impl Send for PyIntegrand {}
unsafe impl Sync for PyIntegrand {}

#[pymethods]
impl PyIntegrand {
    #[new]
    fn new(callable: Py<PyAny>, dim: usize) -> Self {
        PyIntegrand { callable, dim }
    }
}

// A wrapper for Python callables to implement the SimdIntegrand trait
#[pyclass(name = "SimdIntegrand")]
struct PySimdIntegrand {
    callable: Py<PyAny>,
    dim: usize,
}

impl SimdIntegrand for PySimdIntegrand {
    fn dim(&self) -> usize {
        self.dim
    }

    fn eval_simd(&self, points: &[f64x4]) -> f64x4 {
        Python::attach(|py| {
            let py_points = PyList::empty(py);
            for d in 0..self.dim {
                let point_dim = PyList::new(py, &points[d].to_array()).unwrap();
                py_points.append(point_dim).unwrap();
            }

            let args = (py_points,);
            self.callable
                .call1(py, args)
                .and_then(|result| result.extract::<Vec<f64>>(py))
                .and_then(|result_vec| {
                    <[f64; 4]>::try_from(result_vec)
                        .map(f64x4::from)
                        .map_err(|_| {
                            PyValueError::new_err("Integrand must return a list of 4 floats.")
                                .into()
                        })
                })
                .unwrap_or_else(|err| {
                    eprintln!("Error evaluating SIMD integrand: {err}");
                    f64x4::splat(0.0)
                })
        })
    }
}

// CRITICAL: Mark as Send + Sync for parallel execution with Rayon
unsafe impl Send for PySimdIntegrand {}
unsafe impl Sync for PySimdIntegrand {}

#[pymethods]
impl PySimdIntegrand {
    #[new]
    fn new(callable: Py<PyAny>, dim: usize) -> Self {
        PySimdIntegrand { callable, dim }
    }
}

#[pyclass(name = "VegasResult")]
#[derive(Debug, Clone, Copy)]
struct PyVegasResult {
    #[pyo3(get)]
    value: f64,
    #[pyo3(get)]
    error: f64,
    #[pyo3(get)]
    chi2_dof: f64,
}

impl From<VegasResult> for PyVegasResult {
    fn from(result: VegasResult) -> Self {
        PyVegasResult {
            value: result.value,
            error: result.error,
            chi2_dof: result.chi2_dof,
        }
    }
}

#[pymethods]
impl PyVegasResult {
    fn __repr__(&self) -> String {
        format!(
            "VegasResult(value={:.6e}, error={:.6e}, chi2_dof={:.4})",
            self.value, self.error, self.chi2_dof
        )
    }

    fn __str__(&self) -> String {
        format!(
            "Value: {:.6e} ± {:.6e}, χ²/dof: {:.4}",
            self.value, self.error, self.chi2_dof
        )
    }
}

#[pyclass(name = "Vegas")]
struct PyVegas {
    vegas: Vegas,
    dim: usize,
}

#[pymethods]
impl PyVegas {
    #[new]
    #[pyo3(signature = (n_iter, n_eval, n_bins, alpha, boundaries))]
    fn new(
        n_iter: usize,
        n_eval: usize,
        n_bins: usize,
        alpha: f64,
        boundaries: Vec<(f64, f64)>,
    ) -> PyResult<Self> {
        let dim = boundaries.len();

        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err("alpha must be between 0.0 and 1.0"));
        }

        Ok(PyVegas {
            vegas: Vegas::new(n_iter, n_eval, n_bins, alpha, &boundaries),
            dim,
        })
    }

    fn set_seed(&mut self, seed: u64) {
        self.vegas.set_seed(seed);
    }

    #[pyo3(signature = (integrand, target_accuracy = None))]
    fn integrate_integrand(
        &mut self,
        py: Python,
        integrand: &PyIntegrand,
        target_accuracy: Option<f64>,
    ) -> PyVegasResult {
        py.detach(|| self.vegas.integrate(integrand, target_accuracy).into())
    }

    #[pyo3(signature = (callable, target_accuracy = None))]
    fn integrate(
        &mut self,
        py: Python,
        callable: Py<PyAny>,
        target_accuracy: Option<f64>,
    ) -> PyResult<PyVegasResult> {
        if !callable.bind(py).is_callable() {
            return Err(PyValueError::new_err("integrand must be callable"));
        }

        let integrand = PyIntegrand {
            callable,
            dim: self.dim,
        };

        Ok(py.detach(|| self.vegas.integrate(&integrand, target_accuracy).into()))
    }

    #[pyo3(signature = (integrand, target_accuracy = None))]
    fn integrate_simd_integrand(
        &mut self,
        py: Python,
        integrand: &PySimdIntegrand,
        target_accuracy: Option<f64>,
    ) -> PyVegasResult {
        py.detach(|| self.vegas.integrate_simd(integrand, target_accuracy).into())
    }

    #[pyo3(signature = (callable, target_accuracy = None))]
    fn integrate_simd(
        &mut self,
        py: Python,
        callable: Py<PyAny>,
        target_accuracy: Option<f64>,
    ) -> PyResult<PyVegasResult> {
        if !callable.bind(py).is_callable() {
            return Err(PyValueError::new_err("integrand must be callable"));
        }

        let integrand = PySimdIntegrand {
            callable,
            dim: self.dim,
        };

        Ok(py.detach(|| {
            self.vegas
                .integrate_simd(&integrand, target_accuracy)
                .into()
        }))
    }
}

#[pyclass(name = "VegasPlus")]
struct PyVegasPlus {
    vegas_plus: VegasPlus,
    dim: usize,
}

#[pymethods]
impl PyVegasPlus {
    #[new]
    #[pyo3(signature = (n_iter, n_eval, n_bins, alpha, n_strat, beta, boundaries))]
    fn new(
        n_iter: usize,
        n_eval: usize,
        n_bins: usize,
        alpha: f64,
        n_strat: usize,
        beta: f64,
        boundaries: Vec<(f64, f64)>,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyValueError::new_err("alpha must be between 0.0 and 1.0"));
        }

        if !(0.0..=1.0).contains(&beta) {
            return Err(PyValueError::new_err("beta must be between 0.0 and 1.0"));
        }

        let dim = boundaries.len();
        Ok(PyVegasPlus {
            vegas_plus: VegasPlus::new(n_iter, n_eval, n_bins, alpha, n_strat, beta, &boundaries),
            dim,
        })
    }

    fn set_seed(&mut self, seed: u64) {
        self.vegas_plus.set_seed(seed);
    }

    #[pyo3(signature = (integrand, target_accuracy = None))]
    fn integrate_integrand(
        &mut self,
        py: Python,
        integrand: &PyIntegrand,
        target_accuracy: Option<f64>,
    ) -> PyVegasResult {
        py.detach(|| {
            self.vegas_plus
                .integrate(integrand, target_accuracy)
                .into()
        })
    }

    #[pyo3(signature = (callable, target_accuracy = None))]
    fn integrate(
        &mut self,
        py: Python,
        callable: Py<PyAny>,
        target_accuracy: Option<f64>,
    ) -> PyResult<PyVegasResult> {
        if !callable.bind(py).is_callable() {
            return Err(PyValueError::new_err("integrand must be callable"));
        }

        let integrand = PyIntegrand {
            callable,
            dim: self.dim,
        };

        Ok(py.detach(|| {
            self.vegas_plus
                .integrate(&integrand, target_accuracy)
                .into()
        }))
    }

    #[pyo3(signature = (integrand, target_accuracy = None))]
    fn integrate_simd_integrand(
        &mut self,
        py: Python,
        integrand: &PySimdIntegrand,
        target_accuracy: Option<f64>,
    ) -> PyVegasResult {
        py.detach(|| {
            self.vegas_plus
                .integrate_simd(integrand, target_accuracy)
                .into()
        })
    }

    #[pyo3(signature = (callable, target_accuracy = None))]
    fn integrate_simd(
        &mut self,
        py: Python,
        callable: Py<PyAny>,
        target_accuracy: Option<f64>,
    ) -> PyResult<PyVegasResult> {
        if !callable.bind(py).is_callable() {
            return Err(PyValueError::new_err("integrand must be callable"));
        }

        let integrand = PySimdIntegrand {
            callable,
            dim: self.dim,
        };

        Ok(py.detach(|| {
            self.vegas_plus
                .integrate_simd(&integrand, target_accuracy)
                .into()
        }))
    }

    #[cfg(feature = "mpi")]
    #[pyo3(signature = (integrand, target_accuracy = None))]
    fn integrate_mpi_integrand(
        &mut self,
        py: Python,
        integrand: &PyIntegrand,
        target_accuracy: Option<f64>,
    ) -> PyVegasResult {
        

        // NOTE: MPI initialization should typically happen once at program start
        // This may cause issues if called multiple times.
        py.detach(|| {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            self.vegas_plus
                .integrate_mpi(integrand, &world, target_accuracy)
                .into()
        })
    }

    #[cfg(feature = "mpi")]
    #[pyo3(signature = (callable, target_accuracy = None))]
    fn integrate_mpi(
        &mut self,
        py: Python,
        callable: Py<PyAny>,
        target_accuracy: Option<f64>,
    ) -> PyResult<PyVegasResult> {
        

        if !callable.bind(py).is_callable() {
            return Err(PyValueError::new_err("integrand must be callable"));
        }

        let integrand = PyIntegrand {
            callable,
            dim: self.dim,
        };

        Ok(py.detach(|| {
            let universe = mpi::initialize().unwrap();
            let world = universe.world();
            self.vegas_plus
                .integrate_mpi(&integrand, &world, target_accuracy)
                .into()
        }))
    }

    fn __repr__(&self) -> String {
        format!("VegasPlus(dim={})", self.dim)
    }
}

/// Registers the `VEGAS` submodules with the parent Python module.
///
/// This function is typically called during the initialization of the
/// `MCHEP` Python package to expose the `VEGAS` and `VEGAPLUS` classes.
///
/// Parameters
/// ----------
/// `parent_module` : pyo3.Bound[pyo3.types.PyModule]
///     The parent Python module to which the `VEGAS` submodule will be added.
///
/// Returns
/// -------
/// pyo3.PyResult<()>
///     `Ok(())` if the registration is successful, or an error if the submodule
///     cannot be created or added.
///
/// # Errors
///
/// Raises an error if the (sub)module is not found or cannot be registered.
pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "vegas")?;
    m.setattr(pyo3::intern!(m.py(), "__doc__"), "Interface to Vegas")?;
    pyo3::py_run!(
        parent_module.py(),
        m,
        "import sys; sys.modules['mchep.vegas'] = m"
    );
    m.add_class::<PyIntegrand>()?;
    m.add_class::<PySimdIntegrand>()?;
    m.add_class::<PyVegasResult>()?;
    m.add_class::<PyVegas>()?;
    m.add_class::<PyVegasPlus>()?;
    parent_module.add_submodule(&m)
}
