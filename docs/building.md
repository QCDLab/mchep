# Building and Installation

This page guides you through the process of installing MCHEP and its dependencies for different environments.

## Rust API

=== "Default"

    Add `mchep` to your `Cargo.toml`:

    ```toml
    [dependencies]
    mchep = "0.1.0"
    ```

=== "Development"

    Clone the repository and use a path dependency or git dependency:

    ```toml
    [dependencies]
    mchep = { git = "https://github.com/tanjona/mchep.git" }
    # Or for local development
    # mchep = { path = "../path/to/mchep" }
    ```

### Features

MCHEP provides several optional features:

*   `mpi`: Enables distributed integration using MPI.
*   `gpu`: Enables GPU acceleration using the Burn framework.
*   `simd`: Enables SIMD-optimized routines (enabled by default).

Example with features:

```bash
cargo build --features "mpi gpu"
```

---

## C/C++ API

The C/C++ API can be installed using a pre-built binary script or built from source.

=== "Default (Script)"

    Use the provided installation script to download and install the latest pre-built binaries:

    ```bash
    # Download and install to /usr/local (or your preferred prefix)
    ./install-capi.sh --prefix /usr/local
    ```

    This script automatically detects your platform, downloads the correct tarball from GitHub, and configures the `pkg-config` files.

=== "Development (Source)"

    To build from source, you need `cargo` and `cargo-c`:

    ```bash
    cargo install cargo-c
    
    # Define the installation prefix
    export CARGO_C_MCHEP_INSTALL_PREFIX=/usr/local

    # Install the library and headers
    cargo cinstall --release --prefix=$CARGO_C_MCHEP_INSTALL_PREFIX --manifest-path mchep_capi/Cargo.toml
    ```

### Environment Configuration

After installation, make sure your environment variables are set:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
```

Verify the installation:
```bash
pkg-config mchep_capi --libs --cflags
```

---

## Python API

=== "Default (Pip)"

    Install the latest version from PyPI:

    ```bash
    pip install mchep
    ```

=== "Development (Maturin)"

    To build the Python extension from source, you need `maturin`:

    ```bash
    cd mchep_pyapi
    maturin develop
    ```
