// This test is an integration test and must be run with mpirun.
// Example: mpirun -n 4 cargo test --test mpi_integration_test -- --nocapture

// This annotation ensures this test is only compiled when the "mpi" feature is enabled.
#[cfg(feature = "mpi")]
mod mpi_tests {
    use mchep::integrand::Integrand;
    use mchep::vegasplus::VegasPlus;
    use mpi::traits::*;
    struct GaussianIntegrand;

    impl Integrand for GaussianIntegrand {
        fn dim(&self) -> usize {
            2
        }

        fn eval(&self, x: &[f64]) -> f64 {
            let u = x[0];
            let v = x[1];
            let x_mapped = 2.0 * u - 1.0;
            let y_mapped = 2.0 * v - 1.0;
            let jacobian = 4.0;

            jacobian * (-(x_mapped.powi(2)) - y_mapped.powi(2)).exp()
        }
    }

    const ANALYTICAL_RESULT: f64 = 2.230985;

    #[test]
    fn test_mpi_gaussian_integration() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();
        let rank = world.rank();

        let integrand = GaussianIntegrand;
        let boundaries = &[(0.0, 1.0), (0.0, 1.0)];
        let mut vegas_plus = VegasPlus::new(10, 20_000, 50, 0.5, 4, 0.75, boundaries);
        let result = vegas_plus.integrate_mpi(&integrand, &world, None);

        if rank == 0 {
            assert!((result.value - ANALYTICAL_RESULT).abs() < 5.0 * result.error);
            assert!(result.chi2_dof < 5.0);
        }
    }
}
