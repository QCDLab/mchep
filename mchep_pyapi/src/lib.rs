//! Generate `PyO3` interface for `mchep`

use pyo3::prelude::*;

/// Python bindings for the `vegas` and `vegasplus` crates.
pub mod vegas;

/// `PyO3` Python module that contains all exposed classes from Rust.
#[pymodule]
fn mchep(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("version", env!("CARGO_PKG_VERSION"))?;
    vegas::register(m)?;
    Ok(())
}
