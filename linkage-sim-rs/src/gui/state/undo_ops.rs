//! Undo/redo operations on AppState.

use nalgebra::DVector;

use crate::gui::undo::MechanismSnapshot;
use crate::io::{load_mechanism_unbuilt, mechanism_to_json};
use super::AppState;

impl AppState {
    /// Create a snapshot of the current mechanism document state.
    ///
    /// Returns `None` if no mechanism is loaded or serialization fails.
    pub fn take_snapshot(&self) -> Option<MechanismSnapshot> {
        let mech = self.mechanism.as_ref()?;
        let json = mechanism_to_json(mech).ok()?;
        let json_str = serde_json::to_string(&json).ok()?;
        Some(MechanismSnapshot {
            mechanism_json: json_str,
            driver_angle: self.driver_angle,
            driver_omega: self.driver_omega(),
            driver_theta_0: self.driver_theta_0(),
            driver_joint_id: self.driver_joint_id.clone(),
            q: self.q.iter().copied().collect(),
        })
    }

    /// Restore mechanism state from a snapshot.
    ///
    /// Deserializes the mechanism JSON, builds, solves at t=0, and updates all
    /// document-level state. View transform, selection, and animation state are
    /// left unchanged.
    pub fn restore_snapshot(&mut self, snapshot: &MechanismSnapshot) {
        let Ok(mut mech) = load_mechanism_unbuilt(&snapshot.mechanism_json) else {
            log::warn!("Undo/redo: failed to deserialize mechanism snapshot");
            return;
        };
        if mech.build().is_err() {
            log::warn!("Undo/redo: failed to build mechanism from snapshot");
            return;
        }

        // Solve at the snapshot's driver angle, using the stored q as the initial guess.
        // Falls back to make_q() (all zeros) if the snapshot has no q data.
        let t = if snapshot.driver_omega.abs() > f64::EPSILON {
            (snapshot.driver_angle - snapshot.driver_theta_0) / snapshot.driver_omega
        } else {
            0.0
        };
        let q0 = if snapshot.q.len() == mech.state().n_coords() {
            DVector::from_vec(snapshot.q.clone())
        } else {
            mech.state().make_q()
        };
        self.solve_and_update(&mech, &q0, t, 1e-10, 50, Some(q0.clone()));

        // Restore the blueprint from the snapshot JSON so it stays in sync.
        self.blueprint = serde_json::from_str(&snapshot.mechanism_json).ok();

        self.mechanism = Some(mech);
        self.driver_angle = snapshot.driver_angle;
        // Reconstruct driver_kind from the restored blueprint, then
        // overlay the snapshot's omega/theta_0 onto it. Linear takes
        // priority over revolute (matches rebuild() / load_from_string).
        self.driver_kind = if let Some(ref bp) = self.blueprint {
            if let Some(ld) = bp.linear_drivers.first() {
                super::DriverKind::Linear {
                    stroke: ld.length_0,
                    velocity: snapshot.driver_omega,
                    length_0: snapshot.driver_theta_0,
                }
            } else if !bp.drivers.is_empty() {
                super::DriverKind::Revolute {
                    angle: snapshot.driver_angle,
                    omega: snapshot.driver_omega,
                    theta_0: snapshot.driver_theta_0,
                }
            } else {
                super::DriverKind::None
            }
        } else {
            super::DriverKind::None
        };
        self.driver_joint_id = snapshot.driver_joint_id.clone();
        self.playing = false;
        self.compute_forces(self.driver_angle);
        self.update_grashof();
        self.compute_validation();
        self.mark_sweep_dirty();
    }

    /// Push the current state onto the undo stack before an undoable action.
    ///
    /// No-op if no mechanism is loaded (nothing to snapshot).
    pub fn push_undo(&mut self) {
        if let Some(snapshot) = self.take_snapshot() {
            self.undo_history.push(snapshot);
            self.dirty = true;
        }
    }

    /// Run `op` as an undoable mutation: push an undo snapshot, run `op`,
    /// then rebuild the mechanism. This is the canonical pattern for every
    /// blueprint mutation — using this helper ensures neither step is
    /// forgotten.
    ///
    /// The invariant documented in `docs/ai/04-memory.yaml` says:
    /// *Every state mutation that changes the blueprint MUST call push_undo()
    /// BEFORE the mutation and rebuild() AFTER.*
    /// This helper encodes that invariant so new mutations can't get it wrong.
    pub fn mutate_and_rebuild<F>(&mut self, op: F)
    where
        F: FnOnce(&mut Self),
    {
        self.push_undo();
        op(self);
        self.rebuild();
    }

    /// Undo the last action: restore the previous mechanism state.
    pub fn undo(&mut self) {
        let Some(current) = self.take_snapshot() else {
            return;
        };
        if let Some(previous) = self.undo_history.undo(current) {
            self.restore_snapshot(&previous);
        }
    }

    /// Redo the last undone action: restore the next mechanism state.
    pub fn redo(&mut self) {
        let Some(current) = self.take_snapshot() else {
            return;
        };
        if let Some(next) = self.undo_history.redo(current) {
            self.restore_snapshot(&next);
        }
    }

    /// Returns true if there is at least one state to undo.
    pub fn can_undo(&self) -> bool {
        self.undo_history.can_undo()
    }

    /// Returns true if there is at least one state to redo.
    pub fn can_redo(&self) -> bool {
        self.undo_history.can_redo()
    }
}
