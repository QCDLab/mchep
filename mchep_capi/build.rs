//! A build script to install the OOP C++ interface to `MCHEP`

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=include/mchep.hpp");

    if let Ok(prefix) = env::var("CARGO_C_MCHEP_INSTALL_PREFIX") {
        let prefix_path = PathBuf::from(prefix);
        let include_path = prefix_path.join("include").join("mchep_capi");

        fs::create_dir_all(&include_path).expect("Failed to create include directory.");

        let source_header = PathBuf::from("include/mchep.hpp");
        let dest_header = include_path.join("mchep.hpp");

        fs::copy(&source_header, &dest_header).expect("Failed to copy header file.");
    }
}
