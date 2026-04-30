//! Firmware export adapters for trajectory mode.
//!
//! Each adapter consumes a `SweepData` (post-`compute_trajectory`) plus the
//! active `ControlTarget` and `Trajectory` and emits a string in the
//! controller-specific format. JSON is the v1 universal target; G-code,
//! Aerotech AeroBasic, Beckhoff TwinCAT NC PTP, and Galil DMC are planned
//! follow-ups (~150 LoC each).
//!
//! See: docs/superpowers/specs/2026-04-29-trajectory-position-control-design.md §11

pub mod json;

use crate::gui::state::Trajectory;
use crate::gui::sweep::SweepData;
use crate::solver::inverse_kinematics::ControlTarget;

/// A firmware adapter converts a computed trajectory + target into a
/// controller-specific command stream.
///
/// Adapters are pure: they read the `SweepData` produced by
/// `compute_trajectory`, the active `ControlTarget`, the source
/// `Trajectory`, and the input-parameter unit label (driven by the
/// caller's `DriverKind` — `"rad"` for revolute, `"m"` for linear).
/// They never touch `AppState` directly.
pub trait FirmwareAdapter {
    /// Convert trajectory data to the adapter's output format.
    fn emit(
        &self,
        data: &SweepData,
        target: &ControlTarget,
        trajectory: &Trajectory,
        input_units: &str,
    ) -> Result<String, String>;

    /// File extension WITHOUT the leading dot (e.g. "json", "gcode", "nc").
    fn file_extension(&self) -> &'static str;

    /// Human-readable name shown in the export menu.
    fn display_name(&self) -> &'static str;
}

pub use json::JsonAdapter;
