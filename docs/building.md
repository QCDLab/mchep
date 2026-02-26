# Building and Installation

This page guides you through the process of installing MCHEP and its dependencies for
different environments.

## Rust API

=== "Default"

    Add `mchep` to your `Cargo.toml`:

    ```toml
    [dependencies]
    mchep = { version = "0.1.0", features = ["simd"] }
    ```

=== "Development"

    Clone the repository and build it by running the following command:

    ```bash
    cargo build --features "mpi gpu"
    ```

    You can also include it in your project by passing the path to the cloned repository:

    ```toml
    [dependencies]
    mchep = { path = "./path/to/mchep", features = ["simd"] }
    ```

MCHEP provides several optional features such as `mpi` to enable distributed integration
using MPI, `gpu` to enable GPU acceleration using the Burn framework, and `simd` to enable
SIMD-optimized routines (enabled by default).

---

## C/C++ API

The C/C++ API can be installed using a pre-built binary script or built from source.

=== "Default"

    Use the provided installation script to download and install the latest pre-built binaries:

    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/QCDLab/mchep/refs/heads/master/install-capi.sh | sh
    ```

    To pass the installation directory for where to put the files, change the arguments of the shell as follows:

    ```bash
    .. | sh -s -- --prefix /custom/installation/path
    ```

    By default, the script will download the latest stable release. If you would like a specific
    version, pass the version along with `--version`:

    ```bash
    .. | sh -s -- --version 0.1.0
    ```

    This script automatically detects your platform, downloads the correct tarball from GitHub, and configures the `pkg-config` files.

=== "Development"

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

=== "Default"

    Install the latest version from PyPI:

    ```bash
    pip install mchep
    ```

=== "Development"

    To build the Python extension from source, you need `maturin`:

    ```bash
    cd mchep_pyapi
    maturin develop
    ```
