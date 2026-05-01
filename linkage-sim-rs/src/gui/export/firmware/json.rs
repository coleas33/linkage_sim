//! JSON firmware adapter — universal target.
//!
//! Emits a single pretty-printed JSON document with metadata + per-sample
//! objects. Schema is versioned via `FIRMWARE_JSON_SCHEMA_VERSION`; bump on
//! breaking changes so consumers can validate compatibility.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::gui::state::Trajectory;
use crate::gui::sweep::SweepData;
use crate::solver::inverse_kinematics::{ControlTarget, InverseSolveStatus};

use super::FirmwareAdapter;

/// Schema version for the JSON firmware export. Bump on breaking changes.
pub const FIRMWARE_JSON_SCHEMA_VERSION: &str = "linkage-traj-firmware-v1";

#[derive(Debug, Serialize, Deserialize)]
pub struct FirmwareJsonEnvelope {
    /// Schema version string. Consumers should validate this matches.
    pub schema_version: String,
    /// Target observable kind ("Angle", "WorldX", "WorldY", "Projection", "Distance").
    pub target_kind: String,
    /// Target observable units ("rad" or "m").
    pub target_units: String,
    /// Input parameter units ("rad" for revolute drivers, "m" for linear drivers).
    pub input_units: String,
    /// Number of samples in the trajectory.
    pub n_samples: usize,
    /// Total duration in seconds.
    pub duration_seconds: f64,
    /// Trajectory samples in time order.
    pub samples: Vec<FirmwareJsonSample>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FirmwareJsonSample {
    /// Sample time in seconds.
    pub t: f64,
    /// Desired output observable value at this t.
    pub target: f64,
    /// Achieved output observable (may differ from target on failure samples).
    pub achieved: f64,
    /// Tracking residual (achieved - target).
    pub residual: f64,
    /// Back-solved actuator input parameter value.
    pub u: f64,
    /// Back-solved actuator input rate.
    pub u_dot: f64,
    /// Back-solved actuator input acceleration (FD-derived).
    pub u_ddot: f64,
    /// Required actuator force in Newtons. None if not computed or non-finite.
    pub f_actuator_n: Option<f64>,
    /// Solver status at this sample. "Converged" for successful samples;
    /// failure modes carry diagnostic detail.
    pub status: String,
    /// Per-body pose snapshot at this sample, keyed by body ID. Each entry
    /// is `[x_m, y_m, theta_rad]`. `None` when the source `SweepData` did
    /// not capture pose (older sweeps; non-trajectory mode).
    /// `BTreeMap` for stable JSON key order across exports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pose: Option<BTreeMap<String, [f64; 3]>>,
}

/// Concrete `FirmwareAdapter` that emits the JSON envelope above.
pub struct JsonAdapter;

impl FirmwareAdapter for JsonAdapter {
    fn emit(
        &self,
        data: &SweepData,
        target: &ControlTarget,
        trajectory: &Trajectory,
        input_units: &str,
    ) -> Result<String, String> {
        let n = data.target_values.as_ref().map(|v| v.len()).ok_or_else(|| {
            "No trajectory data — switch to Trajectory mode and click Compute first."
                .to_string()
        })?;
        if n < 2 {
            return Err(format!(
                "Insufficient trajectory data ({} samples; need >= 2)",
                n
            ));
        }

        // All trajectory-mode optional vectors must be populated to length n
        // by `compute_trajectory`. We mirror the CSV exporter's contract:
        // present-or-error rather than silently filling NaN.
        let target_values = data.target_values.as_ref().unwrap();
        let achieved_values = data
            .achieved_values
            .as_ref()
            .ok_or_else(|| "achieved_values missing".to_string())?;
        let tracking_residual = data
            .tracking_residual
            .as_ref()
            .ok_or_else(|| "tracking_residual missing".to_string())?;
        let u_values = data
            .u_values
            .as_ref()
            .ok_or_else(|| "u_values missing".to_string())?;
        let u_dot_values = data
            .u_dot_values
            .as_ref()
            .ok_or_else(|| "u_dot_values missing".to_string())?;
        let u_ddot_values = data
            .u_ddot_values
            .as_ref()
            .ok_or_else(|| "u_ddot_values missing".to_string())?;
        let statuses = data
            .inverse_solve_statuses
            .as_ref()
            .ok_or_else(|| "inverse_solve_statuses missing".to_string())?;

        let times = trajectory.sample_times(n);

        // Pose snapshots are emitted as `{ body_id: [x, y, θ] }` per sample
        // when both `pose_body_order` and `pose_snapshots` are present.
        // Length mismatch between body order and per-sample row falls back
        // to None for that sample (defensive — never reached from
        // compute_trajectory which keeps the two in lockstep).
        let pose_body_order = data.pose_body_order.as_deref();
        let pose_snapshots = data.pose_snapshots.as_deref();

        let samples: Vec<FirmwareJsonSample> = (0..n)
            .map(|i| {
                let pose: Option<BTreeMap<String, [f64; 3]>> =
                    match (pose_body_order, pose_snapshots.and_then(|p| p.get(i))) {
                        (Some(order), Some(row)) if order.len() == row.len() => Some(
                            order
                                .iter()
                                .zip(row.iter())
                                .map(|(b, p)| (b.clone(), *p))
                                .collect(),
                        ),
                        _ => None,
                    };
                FirmwareJsonSample {
                    t: times[i],
                    target: target_values[i],
                    achieved: achieved_values[i],
                    residual: tracking_residual[i],
                    u: u_values[i],
                    u_dot: u_dot_values[i],
                    u_ddot: u_ddot_values[i],
                    // NaN/inf serialize as the JSON `null` sentinel via Option.
                    f_actuator_n: data
                        .actuator_forces
                        .as_ref()
                        .and_then(|v| v.get(i))
                        .copied()
                        .filter(|x| x.is_finite()),
                    status: format_status(&statuses[i]),
                    pose,
                }
            })
            .collect();

        let envelope = FirmwareJsonEnvelope {
            schema_version: FIRMWARE_JSON_SCHEMA_VERSION.to_string(),
            target_kind: target_kind_str(target).to_string(),
            target_units: target.unit_label().to_string(),
            input_units: input_units.to_string(),
            n_samples: n,
            duration_seconds: trajectory.duration(),
            samples,
        };

        serde_json::to_string_pretty(&envelope).map_err(|e| e.to_string())
    }

    fn file_extension(&self) -> &'static str {
        "json"
    }

    fn display_name(&self) -> &'static str {
        "Firmware JSON"
    }
}

fn target_kind_str(t: &ControlTarget) -> &'static str {
    match t {
        ControlTarget::Angle { .. } => "Angle",
        ControlTarget::WorldX { .. } => "WorldX",
        ControlTarget::WorldY { .. } => "WorldY",
        ControlTarget::Projection { .. } => "Projection",
        ControlTarget::Distance { .. } => "Distance",
    }
}

fn format_status(s: &InverseSolveStatus) -> String {
    use InverseSolveStatus::*;
    match s {
        Converged => "Converged".to_string(),
        Reachability {
            target,
            achieved_clamp,
            ..
        } => format!(
            "Reachability: target={:.6} clamped={:.6}",
            target, achieved_clamp
        ),
        Singularity { dg_du } => format!("Singularity: dg_du={:.3e}", dg_du),
        BranchJump { delta_q_norm } => format!("BranchJump: norm={:.6}", delta_q_norm),
        NonConvergent {
            iterations,
            residual,
        } => format!(
            "NonConvergent: iter={} residual={:.3e}",
            iterations, residual
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::{MotionProfile, TrajectoryProfile};
    use crate::gui::sweep::{empty_trajectory_sweep_data, SweepMode};
    use crate::solver::inverse_kinematics::Severity;

    /// Build a 3-sample trajectory `SweepData` fixture with all required
    /// vectors populated — mirrors the post-`compute_trajectory` shape.
    fn three_sample_trajectory_data() -> (SweepData, ControlTarget, Trajectory) {
        let target = ControlTarget::angle("crank");
        let traj = Trajectory::Profile(TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        });
        let mode = SweepMode::Trajectory {
            target: target.clone(),
            trajectory: traj.clone(),
            severity: Severity::Analysis,
            n_samples: 3,
        };
        let mut data = empty_trajectory_sweep_data(mode);
        data.target_values = Some(vec![0.0, 0.5, 1.0]);
        data.achieved_values = Some(vec![0.0, 0.5, 1.0]);
        data.tracking_residual = Some(vec![0.0, 0.0, 0.0]);
        data.u_values = Some(vec![0.0, 0.5, 1.0]);
        data.u_dot_values = Some(vec![1.0, 1.0, 1.0]);
        data.u_ddot_values = Some(vec![0.0, 0.0, 0.0]);
        data.actuator_forces = Some(vec![10.0, 20.0, 30.0]);
        data.inverse_solve_statuses = Some(vec![
            InverseSolveStatus::Converged,
            InverseSolveStatus::Converged,
            InverseSolveStatus::Converged,
        ]);
        (data, target, traj)
    }

    #[test]
    fn json_adapter_round_trip() {
        let (data, target, traj) = three_sample_trajectory_data();
        let adapter = JsonAdapter;
        let json_str = adapter.emit(&data, &target, &traj, "rad").unwrap();

        // Round-trip parse so the schema is verified end-to-end.
        let envelope: FirmwareJsonEnvelope = serde_json::from_str(&json_str).unwrap();
        assert_eq!(envelope.schema_version, FIRMWARE_JSON_SCHEMA_VERSION);
        assert_eq!(envelope.target_kind, "Angle");
        assert_eq!(envelope.target_units, "rad");
        assert_eq!(envelope.input_units, "rad");
        assert_eq!(envelope.n_samples, 3);
        assert!((envelope.duration_seconds - 1.0).abs() < 1e-9);
        assert_eq!(envelope.samples.len(), 3);
        assert!((envelope.samples[0].t - 0.0).abs() < 1e-9);
        assert!((envelope.samples[2].t - 1.0).abs() < 1e-9);
        assert!((envelope.samples[1].u - 0.5).abs() < 1e-9);
        assert_eq!(envelope.samples[0].f_actuator_n, Some(10.0));
        assert_eq!(envelope.samples[0].status, "Converged");

        // File extension and display name are stable.
        assert_eq!(adapter.file_extension(), "json");
        assert_eq!(adapter.display_name(), "Firmware JSON");
    }

    #[test]
    fn json_adapter_filters_nan_actuator_force() {
        let target = ControlTarget::angle("crank");
        let traj = Trajectory::Profile(TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        });
        let mode = SweepMode::Trajectory {
            target: target.clone(),
            trajectory: traj.clone(),
            severity: Severity::Analysis,
            n_samples: 2,
        };
        let mut data = empty_trajectory_sweep_data(mode);
        data.target_values = Some(vec![0.0, 1.0]);
        data.achieved_values = Some(vec![0.0, 1.0]);
        data.tracking_residual = Some(vec![0.0, 0.0]);
        data.u_values = Some(vec![0.0, 1.0]);
        data.u_dot_values = Some(vec![1.0, 1.0]);
        data.u_ddot_values = Some(vec![0.0, 0.0]);
        data.actuator_forces = Some(vec![f64::NAN, 5.0]);
        data.inverse_solve_statuses = Some(vec![InverseSolveStatus::Converged; 2]);

        let adapter = JsonAdapter;
        let json_str = adapter.emit(&data, &target, &traj, "rad").unwrap();
        let envelope: FirmwareJsonEnvelope = serde_json::from_str(&json_str).unwrap();
        assert_eq!(
            envelope.samples[0].f_actuator_n, None,
            "NaN should serialize as null"
        );
        assert_eq!(envelope.samples[1].f_actuator_n, Some(5.0));
        // Confirm the JSON text actually carries `null` (not `NaN`, which
        // would be invalid JSON).
        assert!(
            json_str.contains("\"f_actuator_n\": null"),
            "expected null sentinel for NaN actuator force"
        );
    }

    #[test]
    fn json_adapter_emits_per_sample_pose_when_present() {
        // When `pose_body_order` + `pose_snapshots` are populated (which
        // `compute_trajectory` always does), each FirmwareJsonSample
        // carries a `pose` map keyed by body ID.
        let (mut data, target, traj) = three_sample_trajectory_data();
        data.pose_body_order = Some(vec!["crank".to_string(), "coupler".to_string()]);
        data.pose_snapshots = Some(vec![
            vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.1]],
            vec![[0.0, 0.0, 0.5], [0.5, 0.1, 0.6]],
            vec![[0.0, 0.0, 1.0], [0.5, 0.2, 1.1]],
        ]);

        let adapter = JsonAdapter;
        let json_str = adapter.emit(&data, &target, &traj, "rad").unwrap();
        let envelope: FirmwareJsonEnvelope = serde_json::from_str(&json_str).unwrap();

        assert_eq!(envelope.samples.len(), 3);
        for sample in &envelope.samples {
            let pose = sample
                .pose
                .as_ref()
                .expect("pose should be present when SweepData carries it");
            assert!(pose.contains_key("crank"));
            assert!(pose.contains_key("coupler"));
        }
        // Sample 1 carries (x, y, θ) = (0.0, 0.0, 0.5) for crank.
        let crank_s1 = envelope.samples[1].pose.as_ref().unwrap()["crank"];
        assert!((crank_s1[2] - 0.5).abs() < 1e-9);
        // Coupler θ at sample 2 = 1.1.
        let coupler_s2 = envelope.samples[2].pose.as_ref().unwrap()["coupler"];
        assert!((coupler_s2[2] - 1.1).abs() < 1e-9);
    }

    #[test]
    fn json_adapter_omits_pose_when_absent() {
        // Backwards-compat: a SweepData without pose data must not include
        // a pose field in the emitted samples (skip_serializing_if).
        let (data, target, traj) = three_sample_trajectory_data();
        assert!(data.pose_body_order.is_none(), "fixture has no pose data");

        let adapter = JsonAdapter;
        let json_str = adapter.emit(&data, &target, &traj, "rad").unwrap();
        assert!(
            !json_str.contains("\"pose\""),
            "pose field should be omitted from JSON when SweepData has no pose: {}",
            json_str
        );
    }

    #[test]
    fn json_adapter_rejects_empty_data() {
        // Switching to trajectory mode but never running compute leaves
        // every optional vector at None — emit must surface an error rather
        // than silently producing a header-only file (the JSON analogue of
        // the CSV exporter's "header-only" path doesn't make sense here).
        let mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: MotionProfile::ConstantSpeed,
                start_value: 0.0,
                end_value: 1.0,
                duration: 1.0,
            }),
            severity: Severity::Analysis,
            n_samples: 5,
        };
        let data = empty_trajectory_sweep_data(mode);
        let target = ControlTarget::angle("crank");
        let traj = Trajectory::Profile(TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 1.0,
        });
        let adapter = JsonAdapter;
        let result = adapter.emit(&data, &target, &traj, "rad");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("No trajectory data"),
            "error should mention missing trajectory data"
        );
    }
}
