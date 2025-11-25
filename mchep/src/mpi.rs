//! MPI-specific implementation for distributed integration.
use crate::integrand::Integrand;
use crate::vegas::VegasResult;
use crate::vegasplus::VegasPlus;
use mpi::collective::SystemOperation;
use mpi::topology::Communicator;
use mpi::traits::*;

impl VegasPlus {
    /// Integrates the given function in a distributed manner using MPI.
    ///
    /// This method should be called by all processes in the MPI communicator.
    /// The final result is returned on the root process (rank 0), while other
    /// processes will return a default `VegasResult`.
    ///
    /// # Arguments
    ///
    /// * `integrand`: The function to integrate. Must be `Sync`.
    /// * `world`: The MPI communicator.
    pub fn integrate_mpi<F: Integrand + Sync, C: Communicator>(
        &mut self,
        integrand: &F,
        world: &C,
        target_accuracy: Option<f64>,
    ) -> VegasResult {
        assert_eq!(integrand.dim(), self.dim());

        let rank = world.rank();
        let size = world.size();

        let n_eval_per_rank = self.n_eval / size as usize;
        let n_eval_orig = self.n_eval;
        self.n_eval = n_eval_per_rank;

        let mut iter_results = Vec::new();
        let mut iter_errors = Vec::new();

        for iter in 0..self.n_iter {
            let (iter_val, iter_err) = self.run_iteration(integrand);

            let mut local_data = self.serialize_adaptation_data();
            let mut global_data = vec![0.0; local_data.len()];

            world.all_reduce_into(
                &local_data[..],
                &mut global_data[..],
                SystemOperation::sum(),
            );

            self.deserialize_and_update_state(&global_data);

            let mut stop_signal = 0i32;

            if rank == 0 {
                let mut global_iter_val = 0.0;
                world.process_at_rank(0).reduce_into_root(&iter_val, &mut global_iter_val, SystemOperation::sum());

                let mut global_iter_err_sq: f64 = 0.0;
                let iter_err_sq = iter_err.powi(2);
                world.process_at_rank(0).reduce_into_root(
                    &iter_err_sq,
                    &mut global_iter_err_sq,
                    SystemOperation::sum(),
                );

                if iter > 0 {
                    iter_results.push(global_iter_val);
                    iter_errors.push(global_iter_err_sq.sqrt());
                }

                if let Some(acc_req) = target_accuracy {
                    if !iter_results.is_empty() {
                        let current_result = self.combine_results(&iter_results, &iter_errors);
                        if current_result.value != 0.0 {
                            let current_acc =
                                (current_result.error / current_result.value.abs()) * 100.0;
                            if current_acc < acc_req {
                                stop_signal = 1;
                            }
                        }
                    }
                }
            } else {
                world.process_at_rank(0).reduce_into(&iter_val, SystemOperation::sum());
                let iter_err_sq = iter_err.powi(2);
                world.process_at_rank(0).reduce_into(&iter_err_sq, SystemOperation::sum());
            }

            world.process_at_rank(0).broadcast_into(&mut stop_signal);

            if stop_signal == 1 {
                break;
            }

            self.reallocate_samples();
            for grid in &mut self.grids {
                grid.refine();
            }
        }

        self.n_eval = n_eval_orig;

        if rank == 0 {
            self.combine_results(&iter_results, &iter_errors)
        } else {
            VegasResult {
                value: 0.0,
                error: 0.0,
                chi2_dof: 0.0,
            }
        }
    }

    /// Serializes the grid `d` vectors and hypercube variances into a flat Vec.
    fn serialize_adaptation_data(&self) -> Vec<f64> {
        let n_grids = self.grids.len();
        let n_bins = self.grids[0].n_bins();
        let n_hypercubes = self.hypercubes.len();
        let mut data = vec![0.0; n_grids * n_bins + n_hypercubes];

        for d in 0..n_grids {
            data[d * n_bins..(d + 1) * n_bins].copy_from_slice(&self.grids[d].d);
        }

        let offset = n_grids * n_bins;
        for i in 0..n_hypercubes {
            data[offset + i] = self.hypercubes[i].variance;
        }

        data
    }

    /// Updates the state from a flat Vec of globally reduced adaptation data.
    fn deserialize_and_update_state(&mut self, global_data: &[f64]) {
        let n_grids = self.grids.len();
        let n_bins = self.grids[0].n_bins();
        let n_hypercubes = self.hypercubes.len();

        for d in 0..n_grids {
            self.grids[d]
                .d
                .copy_from_slice(&global_data[d * n_bins..(d + 1) * n_bins]);
        }

        let offset = n_grids * n_bins;
        for i in 0..n_hypercubes {
            self.hypercubes[i].variance = global_data[offset + i];
        }
    }
}
