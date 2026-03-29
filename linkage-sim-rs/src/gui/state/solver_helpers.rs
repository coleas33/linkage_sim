//! Shared solve-position-then-update-state helper.
//!
//! Six call-sites across state/ previously duplicated the same
//! `solve_position → update solver_status → update q / last_good_q` pattern.
//! This module consolidates it into a single method on `AppState`.

use nalgebra::DVector;

use crate::core::mechanism::Mechanism;
use crate::solver::kinematics::solve_position;

use super::{AppState, SolverStatus};

impl AppState {
    /// Solve the position problem and update state accordingly.
    ///
    /// On convergence, sets `self.q` and `self.last_good_q` to the solved `q`.
    ///
    /// On failure (non-convergence **or** solver error):
    /// - If `fallback_q` is `Some(fb)`, sets both `self.q` and `self.last_good_q`
    ///   to `fb`.
    /// - If `fallback_q` is `None`, leaves `self.q` and `self.last_good_q`
    ///   unchanged (the caller retains the previous valid pose).
    ///
    /// Always updates `self.solver_status`.
    ///
    /// Returns `true` when the solver converged.
    pub(crate) fn solve_and_update(
        &mut self,
        mech: &Mechanism,
        guess: &DVector<f64>,
        t: f64,
        tol: f64,
        max_iter: usize,
        fallback_q: Option<DVector<f64>>,
    ) -> bool {
        match solve_position(mech, guess, t, tol, max_iter) {
            Ok(result) => {
                self.solver_status = SolverStatus {
                    converged: result.converged,
                    residual_norm: result.residual_norm,
                    iterations: result.iterations,
                };
                if result.converged {
                    self.q = result.q.clone();
                    self.last_good_q = result.q;
                    true
                } else if let Some(fb) = fallback_q {
                    self.q = fb.clone();
                    self.last_good_q = fb;
                    false
                } else {
                    false
                }
            }
            Err(_) => {
                self.solver_status = SolverStatus {
                    converged: false,
                    residual_norm: f64::NAN,
                    iterations: 0,
                };
                if let Some(fb) = fallback_q {
                    self.q = fb.clone();
                    self.last_good_q = fb;
                }
                false
            }
        }
    }
}
