//! CSV export: sweep data and coupler trace data.

use std::io::Write;

use crate::gui::sweep::{SweepData, SweepMode};

/// (Native only) Write sweep data CSV to a file path. Calls
/// `write_sweep_csv` underneath; both layouts (angle/stroke + trajectory)
/// are produced by the cross-platform writer so the web download path
/// (`generate_sweep_csv_string`) emits identical bytes.
#[cfg(feature = "native")]
pub fn export_sweep_csv(path: &std::path::Path, sweep: &SweepData) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    write_sweep_csv(&mut file, sweep).map_err(|e| e.to_string())
}

/// (Cross-platform) Render sweep data CSV as a String.
///
/// Wraps `write_sweep_csv` over a `Vec<u8>` and decodes as UTF-8. The
/// writer only emits ASCII (numeric data + simple identifiers) so the
/// decode never fails in practice — surfaces an error defensively.
pub fn generate_sweep_csv_string(sweep: &SweepData) -> Result<String, String> {
    let mut buf: Vec<u8> = Vec::new();
    write_sweep_csv(&mut buf, sweep).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Write sweep data CSV to a generic `Write`. The cross-platform core
/// for both `export_sweep_csv` (native file I/O) and
/// `generate_sweep_csv_string` (web blob download).
///
/// In trajectory mode, emits the trajectory column layout (see
/// `write_trajectory_csv`). Otherwise emits the angle/stroke sweep layout:
/// driver angle, then sorted body angles, then transmission angle
/// (if available), then driver torque (if available). All angles in degrees,
/// torque in N*m.
fn write_sweep_csv(out: &mut impl Write, sweep: &SweepData) -> std::io::Result<()> {
    if matches!(sweep.sweep_mode, SweepMode::Trajectory { .. }) {
        return write_trajectory_csv(out, sweep);
    }

    // Collect and sort body IDs for deterministic column order.
    let mut body_ids: Vec<&String> = sweep.body_angles.keys().collect();
    body_ids.sort();

    // Header
    let mut headers = vec!["angle_deg".to_string()];
    for id in &body_ids {
        headers.push(format!("{}_theta_deg", id));
    }
    if sweep.transmission_angles.is_some() {
        headers.push("transmission_angle_deg".to_string());
    }
    if sweep.driver_torques.is_some() {
        headers.push("driver_torque_Nm".to_string());
    }
    let has_ma = !sweep.mechanical_advantage.is_empty()
        && sweep.mechanical_advantage.iter().any(|v| v.is_finite());
    if has_ma {
        headers.push("mechanical_advantage".to_string());
    }
    // Energy columns (always present — may be empty vecs, in which case we skip).
    let has_energy = !sweep.kinetic_energy.is_empty();
    if has_energy {
        headers.push("kinetic_energy_J".to_string());
        headers.push("potential_energy_J".to_string());
        headers.push("total_energy_J".to_string());
    }
    // Inverse dynamics torque column.
    let has_inv_dyn = !sweep.inverse_dynamics_torques.is_empty();
    if has_inv_dyn {
        headers.push("inverse_dynamics_torque_Nm".to_string());
    }
    // Actuator columns (present only when a LinearActuator force element exists).
    let has_actuator = sweep.actuator_forces.is_some();
    if has_actuator {
        headers.push("actuator_force_N".to_string());
    }
    let has_actuator_id = sweep.actuator_forces_id.is_some();
    if has_actuator_id {
        headers.push("actuator_force_id_N".to_string());
    }
    let has_actuator_len = sweep.actuator_lengths.is_some();
    if has_actuator_len {
        headers.push("actuator_length_m".to_string());
    }
    let has_actuator_speed = sweep.actuator_speeds.is_some();
    if has_actuator_speed {
        headers.push("actuator_speed_m_s".to_string());
    }
    let has_actuator_power = sweep.actuator_power.is_some();
    if has_actuator_power {
        headers.push("actuator_power_W".to_string());
    }
    let has_actuator_power_id = sweep.actuator_power_id.is_some();
    if has_actuator_power_id {
        headers.push("actuator_power_id_W".to_string());
    }
    // Output force column (present only when a ForceZone force element exists).
    let has_output_force = sweep.output_forces.is_some();
    if has_output_force {
        headers.push("output_force_N".to_string());
    }
    // Joint reaction magnitude columns (sorted by joint ID).
    let mut reaction_ids: Vec<&String> = sweep.joint_reaction_magnitudes.keys().collect();
    reaction_ids.sort();
    for jid in &reaction_ids {
        headers.push(format!("{}_reaction_N", jid));
    }
    // Coupler velocity magnitude columns (sorted by trace name).
    let mut vel_names: Vec<&String> = sweep.coupler_velocities.keys().collect();
    vel_names.sort();
    for name in &vel_names {
        headers.push(format!("{}_velocity_m_s", name));
    }
    // Coupler acceleration magnitude columns (sorted by trace name).
    let mut acc_names: Vec<&String> = sweep.coupler_accelerations.keys().collect();
    acc_names.sort();
    for name in &acc_names {
        headers.push(format!("{}_acceleration_m_s2", name));
    }
    // Toggle angles as a final informational column (sparse: only non-empty at toggle steps).
    let has_toggles = !sweep.toggle_angles.is_empty();
    if has_toggles {
        headers.push("toggle_angle_deg".to_string());
    }
    writeln!(out, "{}", headers.join(","))?;

    // Data rows
    for (i, angle) in sweep.angles_deg.iter().enumerate() {
        let mut row = vec![format!("{:.4}", angle)];
        for id in &body_ids {
            let val = sweep.body_angles[*id].get(i).copied().unwrap_or(f64::NAN);
            row.push(format!("{:.6}", val));
        }
        if let Some(ref ta) = sweep.transmission_angles {
            row.push(format!("{:.4}", ta.get(i).copied().unwrap_or(f64::NAN)));
        }
        if let Some(ref dt) = sweep.driver_torques {
            row.push(format!("{:.6}", dt.get(i).copied().unwrap_or(f64::NAN)));
        }
        if has_ma {
            let ma = sweep.mechanical_advantage.get(i).copied().unwrap_or(f64::NAN);
            row.push(format!("{:.6}", ma));
        }
        if has_energy {
            row.push(format!("{:.6}", sweep.kinetic_energy.get(i).copied().unwrap_or(f64::NAN)));
            row.push(format!("{:.6}", sweep.potential_energy.get(i).copied().unwrap_or(f64::NAN)));
            row.push(format!("{:.6}", sweep.total_energy.get(i).copied().unwrap_or(f64::NAN)));
        }
        if has_inv_dyn {
            row.push(format!("{:.6}", sweep.inverse_dynamics_torques.get(i).copied().unwrap_or(f64::NAN)));
        }
        if has_actuator {
            if let Some(ref forces) = sweep.actuator_forces {
                row.push(format!("{:.6}", forces.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_actuator_id {
            if let Some(ref forces) = sweep.actuator_forces_id {
                row.push(format!("{:.6}", forces.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_actuator_len {
            if let Some(ref lengths) = sweep.actuator_lengths {
                row.push(format!("{:.6}", lengths.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_actuator_speed {
            if let Some(ref speeds) = sweep.actuator_speeds {
                row.push(format!("{:.6}", speeds.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_actuator_power {
            if let Some(ref power) = sweep.actuator_power {
                row.push(format!("{:.6}", power.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_actuator_power_id {
            if let Some(ref power) = sweep.actuator_power_id {
                row.push(format!("{:.6}", power.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        if has_output_force {
            if let Some(ref forces) = sweep.output_forces {
                row.push(format!("{:.6}", forces.get(i).copied().unwrap_or(f64::NAN)));
            }
        }
        for jid in &reaction_ids {
            let val = sweep.joint_reaction_magnitudes[*jid]
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);
            row.push(format!("{:.6}", val));
        }
        for name in &vel_names {
            let val = sweep.coupler_velocities[*name]
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);
            row.push(format!("{:.6}", val));
        }
        for name in &acc_names {
            let val = sweep.coupler_accelerations[*name]
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);
            row.push(format!("{:.6}", val));
        }
        if has_toggles {
            // Toggle angles are sparse: emit the value only if this angle is a toggle.
            let angle_val = *angle;
            let is_toggle = sweep.toggle_angles.iter().any(|ta| (ta - angle_val).abs() < 1e-6);
            if is_toggle {
                row.push(format!("{:.4}", angle_val));
            } else {
                row.push(String::new());
            }
        }
        writeln!(out, "{}", row.join(","))?;
    }

    Ok(())
}

/// Write the trajectory-mode CSV layout (v1 superset).
///
/// Columns:
/// `t_seconds, target_value, achieved_value, residual,
///  u, u_dot, u_ddot, F_actuator_N, driver_torque_Nm, status`
///
/// Time axis is uniform across `[0, profile.duration]` to match the plot
/// panel's rendering. `F_actuator_N` is NaN when no LinearActuator is present
/// (trajectory compute does not populate `actuator_forces`).
fn write_trajectory_csv(out: &mut impl Write, sweep: &SweepData) -> std::io::Result<()> {
    // Trajectory data lives in the optional vectors. If the user hasn't
    // computed yet, emit just the header so the file is well-formed.
    let n = sweep.target_values.as_ref().map(|v| v.len()).unwrap_or(0);

    // Pose snapshot columns are appended after the fixed trajectory
    // columns when `pose_body_order` + `pose_snapshots` are populated.
    // This keeps the file consumable by tools that only know the v1
    // layout (they ignore extra columns) while letting downstream
    // replay tools recover the full q(t).
    let pose_body_order = sweep.pose_body_order.as_deref().unwrap_or(&[]);

    let mut header = String::from(
        "t_seconds,target_value,achieved_value,residual,u,u_dot,u_ddot,F_actuator_N,driver_torque_Nm,status",
    );
    for body_id in pose_body_order {
        header.push_str(&format!(
            ",q_x_{body}_m,q_y_{body}_m,q_theta_{body}_rad",
            body = body_id
        ));
    }
    writeln!(out, "{}", header)?;
    if n == 0 {
        return Ok(());
    }

    // Reconstruct the time axis from the active SweepMode::Trajectory's
    // trajectory duration (matches plot_panel/trajectory.rs).
    let duration = match &sweep.sweep_mode {
        SweepMode::Trajectory { trajectory, .. } => trajectory.duration(),
        _ => 1.0,
    };
    let denom = (n - 1).max(1) as f64;
    let times: Vec<f64> = (0..n).map(|i| (i as f64) * duration / denom).collect();

    // Required vectors (compute_trajectory always populates these to length n).
    let target = sweep.target_values.as_ref().unwrap();
    let achieved = sweep.achieved_values.as_ref().unwrap();
    let residual = sweep.tracking_residual.as_ref().unwrap();
    let u = sweep.u_values.as_ref().unwrap();
    let u_dot = sweep.u_dot_values.as_ref().unwrap();
    let u_ddot = sweep.u_ddot_values.as_ref().unwrap();
    let statuses = sweep.inverse_solve_statuses.as_ref();
    let pose_snapshots = sweep.pose_snapshots.as_ref();

    for i in 0..n {
        let t = times[i];
        // F_actuator_N: only present when a LinearActuator force element exists;
        // trajectory compute currently doesn't populate this, so emit NaN.
        let f_act = sweep
            .actuator_forces
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .unwrap_or(f64::NAN);
        // driver_torque_Nm comes from the per-sample statics solve in
        // compute_trajectory (Optional<Vec> populated to length n).
        let torque = sweep
            .driver_torques
            .as_ref()
            .and_then(|v| v.get(i))
            .copied()
            .unwrap_or(f64::NAN);
        let status_str = statuses
            .and_then(|v| v.get(i))
            .map(format_status)
            .unwrap_or_else(|| "Unknown".to_string());
        write!(
            out,
            "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            t,
            target[i],
            achieved[i],
            residual[i],
            u[i],
            u_dot[i],
            u_ddot[i],
            f_act,
            torque,
            status_str,
        )?;
        // Per-body pose columns (3 per body). Emit NaN if pose_snapshots
        // is missing or shorter than expected — defensive against
        // partial fixtures, never reachable from compute_trajectory.
        if !pose_body_order.is_empty() {
            let row = pose_snapshots.and_then(|p| p.get(i));
            for j in 0..pose_body_order.len() {
                let pose = row.and_then(|r| r.get(j));
                let (x, y, th) = pose
                    .map(|p| (p[0], p[1], p[2]))
                    .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                write!(out, ",{:.6},{:.6},{:.6}", x, y, th)?;
            }
        }
        writeln!(out)?;
    }

    Ok(())
}

/// Format an `InverseSolveStatus` as a single CSV field (no commas, no
/// quoting required). Failure variants include their numeric payload so
/// downstream tools can filter / sort by them.
fn format_status(s: &crate::solver::inverse_kinematics::InverseSolveStatus) -> String {
    use crate::solver::inverse_kinematics::InverseSolveStatus::*;
    match s {
        Converged => "Converged".to_string(),
        Reachability {
            target,
            achieved_clamp,
            ..
        } => format!("Reachability:{:.4}->{:.4}", target, achieved_clamp),
        Singularity { dg_du } => format!("Singularity:{:.2e}", dg_du),
        BranchJump { delta_q_norm } => format!("BranchJump:{:.4}", delta_q_norm),
        NonConvergent {
            iterations,
            residual,
            // `;` separator (not `,`) so the status field stays a single CSV column.
        } => format!("NonConvergent:iter={};res={:.2e}", iterations, residual),
    }
}

/// (Native only) Write coupler trace CSV to a file path.
#[cfg(feature = "native")]
pub fn export_coupler_csv(path: &std::path::Path, sweep: &SweepData) -> Result<(), String> {
    if sweep.coupler_traces.is_empty() {
        return Err("No coupler traces available".to_string());
    }
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    write_coupler_csv(&mut file, sweep).map_err(|e| e.to_string())
}

/// (Cross-platform) Render coupler trace CSV as a String.
///
/// Returns `Err` when the sweep has no coupler traces — matches the
/// `export_coupler_csv` precondition.
pub fn generate_coupler_csv_string(sweep: &SweepData) -> Result<String, String> {
    if sweep.coupler_traces.is_empty() {
        return Err("No coupler traces available".to_string());
    }
    let mut buf: Vec<u8> = Vec::new();
    write_coupler_csv(&mut buf, sweep).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Write coupler trace CSV to a generic `Write`. Cross-platform core for
/// `export_coupler_csv` and `generate_coupler_csv_string`.
///
/// Columns: driver angle, then sorted coupler trace x/y pairs. Coordinates
/// are in meters (SI). Caller is responsible for the empty-traces
/// precondition check (the wrappers do it before calling).
fn write_coupler_csv(out: &mut impl Write, sweep: &SweepData) -> std::io::Result<()> {
    let mut trace_names: Vec<&String> = sweep.coupler_traces.keys().collect();
    trace_names.sort();

    let mut headers = vec!["angle_deg".to_string()];
    for name in &trace_names {
        headers.push(format!("{}_x_m", name));
        headers.push(format!("{}_y_m", name));
    }
    writeln!(out, "{}", headers.join(","))?;

    for (i, angle) in sweep.angles_deg.iter().enumerate() {
        let mut row = vec![format!("{:.4}", angle)];
        for name in &trace_names {
            if let Some(points) = sweep.coupler_traces.get(*name) {
                if let Some(pt) = points.get(i) {
                    row.push(format!("{:.6}", pt[0]));
                    row.push(format!("{:.6}", pt[1]));
                } else {
                    row.push("NaN".to_string());
                    row.push("NaN".to_string());
                }
            }
        }
        writeln!(out, "{}", row.join(","))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Create minimal sweep data for testing.
    fn test_sweep_data() -> SweepData {
        let angles_deg: Vec<f64> = (0..5).map(|i| i as f64 * 90.0).collect();

        let mut body_angles = HashMap::new();
        body_angles.insert(
            "crank".to_string(),
            vec![0.0, 90.0, 180.0, 270.0, 360.0],
        );
        body_angles.insert(
            "rocker".to_string(),
            vec![30.0, 45.0, 60.0, 45.0, 30.0],
        );

        let mut coupler_traces = HashMap::new();
        coupler_traces.insert(
            "coupler.tip".to_string(),
            vec![
                [0.1, 0.2],
                [0.15, 0.25],
                [0.2, 0.2],
                [0.15, 0.15],
                [0.1, 0.2],
            ],
        );

        let mut joint_reaction_magnitudes = HashMap::new();
        joint_reaction_magnitudes.insert(
            "J1".to_string(),
            vec![5.0, 7.5, 6.0, 8.0, 5.0],
        );
        joint_reaction_magnitudes.insert(
            "J2".to_string(),
            vec![3.0, 4.5, 3.5, 5.0, 3.0],
        );

        let mut coupler_velocities = HashMap::new();
        coupler_velocities.insert(
            "coupler.tip".to_string(),
            vec![0.5, 0.8, 0.6, 0.7, 0.5],
        );

        let mut coupler_accelerations = HashMap::new();
        coupler_accelerations.insert(
            "coupler.tip".to_string(),
            vec![1.0, 1.5, 1.2, 1.3, 1.0],
        );

        SweepData {
            angles_deg,
            body_angles,
            coupler_traces,
            transmission_angles: Some(vec![80.0, 75.0, 90.0, 105.0, 80.0]),
            driver_torques: Some(vec![1.0, 1.5, 0.5, -0.5, 1.0]),
            kinetic_energy: vec![0.1, 0.2, 0.3, 0.2, 0.1],
            potential_energy: vec![0.5, 0.4, 0.3, 0.4, 0.5],
            total_energy: vec![0.6, 0.6, 0.6, 0.6, 0.6],
            inverse_dynamics_torques: vec![1.2, 1.8, 0.6, -0.4, 1.2],
            mechanical_advantage: vec![0.5, 0.45, 0.6, 0.55, 0.5],
            joint_reaction_magnitudes,
            coupler_velocities,
            coupler_accelerations,
            actuator_forces: None,
            actuator_forces_id: None,
            actuator_lengths: None,
            actuator_speeds: None,
            actuator_power: None,
            actuator_power_id: None,
            output_forces: None,
            profile_torques: None,
            profile_omega: None,
            profile_alpha: None,
            target_values: None,
            achieved_values: None,
            tracking_residual: None,
            u_values: None,
            u_dot_values: None,
            u_ddot_values: None,
            inverse_solve_statuses: None,
            pose_body_order: None,
            pose_snapshots: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: crate::gui::sweep::SweepMode::Angle,
        }
    }

    #[test]
    fn export_sweep_csv_writes_valid_file() {
        let sweep = test_sweep_data();
        let path = std::env::temp_dir().join("test_sweep_export.csv");

        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();

        // Header + 5 data rows
        assert_eq!(lines.len(), 6, "header + 5 data rows");

        // Header must contain expected columns
        let header = lines[0];
        assert!(header.starts_with("angle_deg"), "header should start with angle_deg");
        assert!(header.contains("crank_theta_deg"), "header should have crank column");
        assert!(header.contains("rocker_theta_deg"), "header should have rocker column");
        assert!(
            header.contains("transmission_angle_deg"),
            "header should have transmission angle"
        );
        assert!(
            header.contains("driver_torque_Nm"),
            "header should have driver torque"
        );
        assert!(
            header.contains("mechanical_advantage"),
            "header should have mechanical advantage"
        );
        assert!(
            header.contains("kinetic_energy_J"),
            "header should have kinetic energy"
        );
        assert!(
            header.contains("potential_energy_J"),
            "header should have potential energy"
        );
        assert!(
            header.contains("total_energy_J"),
            "header should have total energy"
        );
        assert!(
            header.contains("inverse_dynamics_torque_Nm"),
            "header should have inverse dynamics torque"
        );
        assert!(
            header.contains("coupler.tip_velocity_m_s"),
            "header should have coupler velocity"
        );
        assert!(
            header.contains("coupler.tip_acceleration_m_s2"),
            "header should have coupler acceleration"
        );

        // First data row should start with 0.0000
        assert!(
            lines[1].starts_with("0.0000"),
            "first data row should start with angle 0"
        );

        // Column count should be consistent
        let header_cols = header.split(',').count();
        for (i, line) in lines.iter().enumerate().skip(1) {
            let cols = line.split(',').count();
            assert_eq!(
                cols, header_cols,
                "row {} has {} columns, expected {}",
                i, cols, header_cols
            );
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_sweep_csv_without_optional_columns() {
        let mut sweep = test_sweep_data();
        sweep.transmission_angles = None;
        sweep.driver_torques = None;
        sweep.mechanical_advantage.clear();
        sweep.joint_reaction_magnitudes.clear();
        sweep.kinetic_energy.clear();
        sweep.potential_energy.clear();
        sweep.total_energy.clear();
        sweep.inverse_dynamics_torques.clear();
        sweep.coupler_velocities.clear();
        sweep.coupler_accelerations.clear();
        sweep.toggle_angles.clear();

        let path = std::env::temp_dir().join("test_sweep_no_optional.csv");

        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let header = contents.lines().next().unwrap();

        assert!(
            !header.contains("transmission_angle"),
            "header should not have transmission angle"
        );
        assert!(
            !header.contains("driver_torque"),
            "header should not have driver torque"
        );
        assert!(
            !header.contains("mechanical_advantage"),
            "header should not have mechanical advantage"
        );
        assert!(
            !header.contains("reaction_N"),
            "header should not have joint reaction columns"
        );
        assert!(
            !header.contains("kinetic_energy"),
            "header should not have energy columns"
        );
        assert!(
            !header.contains("inverse_dynamics"),
            "header should not have inverse dynamics column"
        );
        assert!(
            !header.contains("velocity"),
            "header should not have velocity columns"
        );
        assert!(
            !header.contains("acceleration"),
            "header should not have acceleration columns"
        );
        assert!(
            !header.contains("toggle"),
            "header should not have toggle column"
        );

        // angle_deg + crank + rocker = 3 columns
        assert_eq!(header.split(',').count(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_coupler_csv_writes_valid_file() {
        let sweep = test_sweep_data();
        let path = std::env::temp_dir().join("test_coupler_export.csv");

        export_coupler_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();

        // Header + 5 data rows
        assert_eq!(lines.len(), 6, "header + 5 data rows");

        let header = lines[0];
        assert!(header.starts_with("angle_deg"));
        assert!(header.contains("coupler.tip_x_m"));
        assert!(header.contains("coupler.tip_y_m"));

        // angle_deg + 1 trace * 2 (x, y) = 3 columns
        assert_eq!(header.split(',').count(), 3);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_coupler_csv_empty_traces_returns_error() {
        let mut sweep = test_sweep_data();
        sweep.coupler_traces.clear();

        let path = std::env::temp_dir().join("test_coupler_empty.csv");

        let result = export_coupler_csv(&path, &sweep);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No coupler traces"));

        // File should not have been created (we return before creating it)
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_sweep_csv_body_columns_are_sorted() {
        let sweep = test_sweep_data();
        let path = std::env::temp_dir().join("test_sweep_sorted.csv");

        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let header = contents.lines().next().unwrap();
        let cols: Vec<&str> = header.split(',').collect();

        // Body columns should be alphabetically sorted: crank before rocker
        let crank_idx = cols.iter().position(|c| *c == "crank_theta_deg").unwrap();
        let rocker_idx = cols.iter().position(|c| *c == "rocker_theta_deg").unwrap();
        assert!(
            crank_idx < rocker_idx,
            "crank column should come before rocker"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// Build a minimal trajectory-mode SweepData fixture with `n` samples.
    /// All Optional<Vec> fields populated; statuses include each variant
    /// at least once so `format_status` is exercised end-to-end.
    fn test_trajectory_sweep_data(n: usize) -> SweepData {
        use crate::gui::state::MotionProfile;
        use crate::gui::state::TrajectoryProfile;
        use crate::solver::inverse_kinematics::{
            ControlTarget, InverseSolveStatus, Severity,
        };
        assert!(n >= 5, "fixture covers all 5 status variants");

        let profile = TrajectoryProfile {
            shape: MotionProfile::ConstantSpeed,
            start_value: 0.0,
            end_value: 1.0,
            duration: 2.0,
        };
        let mode = SweepMode::Trajectory {
            target: ControlTarget::Angle {
                body_id: "crank".to_string(),
            },
            trajectory: crate::gui::state::Trajectory::Profile(profile),
            severity: Severity::Analysis,
            n_samples: n,
        };

        let target_values: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let achieved_values: Vec<f64> =
            target_values.iter().map(|t| t + 0.001).collect();
        let tracking_residual: Vec<f64> = achieved_values
            .iter()
            .zip(target_values.iter())
            .map(|(a, t)| a - t)
            .collect();
        let u_values: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let u_dot_values: Vec<f64> = vec![0.05; n];
        let u_ddot_values: Vec<f64> = vec![0.0; n];

        // One of each status variant in the first five slots; the rest Converged.
        let mut statuses = vec![
            InverseSolveStatus::Converged,
            InverseSolveStatus::Reachability {
                target: 1.5,
                achieved_clamp: 1.0,
                workspace_min: Some(-1.0),
                workspace_max: Some(1.0),
            },
            InverseSolveStatus::Singularity { dg_du: 1.0e-9 },
            InverseSolveStatus::BranchJump {
                delta_q_norm: 0.5,
            },
            InverseSolveStatus::NonConvergent {
                iterations: 50,
                residual: 1.0e-3,
            },
        ];
        statuses.resize(n, InverseSolveStatus::Converged);

        SweepData {
            angles_deg: (0..n).map(|i| i as f64).collect(),
            body_angles: HashMap::new(),
            coupler_traces: HashMap::new(),
            transmission_angles: None,
            // Statics-derived torque, populated in trajectory mode.
            driver_torques: Some((0..n).map(|i| 0.5 + 0.1 * i as f64).collect()),
            kinetic_energy: Vec::new(),
            potential_energy: Vec::new(),
            total_energy: Vec::new(),
            inverse_dynamics_torques: Vec::new(),
            mechanical_advantage: Vec::new(),
            joint_reaction_magnitudes: HashMap::new(),
            coupler_velocities: HashMap::new(),
            coupler_accelerations: HashMap::new(),
            actuator_forces: None,
            actuator_forces_id: None,
            actuator_lengths: None,
            actuator_speeds: None,
            actuator_power: None,
            actuator_power_id: None,
            output_forces: None,
            profile_torques: None,
            profile_omega: None,
            profile_alpha: None,
            target_values: Some(target_values),
            achieved_values: Some(achieved_values),
            tracking_residual: Some(tracking_residual),
            u_values: Some(u_values),
            u_dot_values: Some(u_dot_values),
            u_ddot_values: Some(u_ddot_values),
            inverse_solve_statuses: Some(statuses),
            pose_body_order: None,
            pose_snapshots: None,
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: mode,
        }
    }

    #[test]
    fn export_sweep_csv_trajectory_writes_full_layout() {
        let sweep = test_trajectory_sweep_data(5);
        let path = std::env::temp_dir().join("test_trajectory_export.csv");

        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();

        // Header + 5 rows.
        assert_eq!(lines.len(), 6, "header + 5 data rows");

        // Header is the trajectory v1 superset (no angle_deg / body columns).
        let header = lines[0];
        assert_eq!(
            header,
            "t_seconds,target_value,achieved_value,residual,u,u_dot,u_ddot,F_actuator_N,driver_torque_Nm,status"
        );

        // 10 columns on every data row.
        for (i, line) in lines.iter().enumerate().skip(1) {
            assert_eq!(
                line.split(',').count(),
                10,
                "row {} should have 10 columns: {}",
                i,
                line
            );
        }

        // Time axis: first row at t=0, last row at duration=2.0 (5 samples → step 0.5).
        let first_t = lines[1].split(',').next().unwrap().parse::<f64>().unwrap();
        assert!(first_t.abs() < 1e-9, "first row t≈0, got {}", first_t);
        let last_t = lines[5].split(',').next().unwrap().parse::<f64>().unwrap();
        assert!((last_t - 2.0).abs() < 1e-9, "last row t≈2.0, got {}", last_t);

        // F_actuator_N column is NaN when no actuator force element is present.
        let f_act_idx = 7;
        for line in lines.iter().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            assert_eq!(
                cols[f_act_idx], "NaN",
                "F_actuator_N should be NaN without an actuator: {}",
                line
            );
        }

        // Each status variant produces its own format token.
        assert!(lines[1].ends_with(",Converged"));
        assert!(lines[2].contains(",Reachability:"));
        assert!(lines[3].contains(",Singularity:"));
        assert!(lines[4].contains(",BranchJump:"));
        assert!(lines[5].contains(",NonConvergent:iter=50;res="));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn generate_sweep_csv_string_matches_file_export() {
        // Cross-platform companion must produce byte-identical output to
        // the file exporter (modulo newline normalization). This test
        // pins them together so future writer changes hit both paths.
        let sweep = test_sweep_data();
        let path = std::env::temp_dir().join("test_sweep_csv_string_parity.csv");
        export_sweep_csv(&path, &sweep).expect("file export should succeed");

        let from_file = std::fs::read_to_string(&path).expect("read file");
        let from_string = generate_sweep_csv_string(&sweep).expect("string export");
        // Normalize trailing newlines (writeln! adds OS-specific). Compare
        // line-by-line.
        let lines_file: Vec<&str> = from_file.lines().collect();
        let lines_str: Vec<&str> = from_string.lines().collect();
        assert_eq!(
            lines_file, lines_str,
            "file and String exports should produce identical content"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn generate_sweep_csv_string_trajectory_includes_pose() {
        // The trajectory layout shares write_trajectory_csv between the
        // file and String paths, so pose columns should also appear in
        // generate_sweep_csv_string output when SweepData carries them.
        let mut sweep = test_trajectory_sweep_data(5);
        sweep.pose_body_order = Some(vec!["crank".to_string()]);
        sweep.pose_snapshots = Some(
            (0..5).map(|i| vec![[0.1 * i as f64, 0.0, 0.0]]).collect(),
        );
        let csv = generate_sweep_csv_string(&sweep).expect("string export");
        assert!(
            csv.lines().next().unwrap().contains("q_theta_crank_rad"),
            "header should include pose column: {}",
            csv.lines().next().unwrap()
        );
        assert_eq!(csv.lines().count(), 6, "header + 5 data rows");
    }

    #[test]
    fn generate_coupler_csv_string_empty_traces_returns_error() {
        // String companion preserves the empty-traces precondition.
        let mut sweep = test_sweep_data();
        sweep.coupler_traces.clear();
        let result = generate_coupler_csv_string(&sweep);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No coupler traces"));
    }

    #[test]
    fn generate_coupler_csv_string_matches_file_export() {
        let sweep = test_sweep_data();
        let path = std::env::temp_dir().join("test_coupler_csv_string_parity.csv");
        export_coupler_csv(&path, &sweep).expect("file export should succeed");

        let from_file = std::fs::read_to_string(&path).expect("read file");
        let from_string = generate_coupler_csv_string(&sweep).expect("string export");
        let lines_file: Vec<&str> = from_file.lines().collect();
        let lines_str: Vec<&str> = from_string.lines().collect();
        assert_eq!(lines_file, lines_str);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_sweep_csv_trajectory_includes_pose_columns_when_present() {
        // When `pose_body_order` + `pose_snapshots` are populated (which
        // `compute_trajectory` always does), the trajectory CSV appends
        // 3 pose columns per body after the v1 fixed columns.
        let mut sweep = test_trajectory_sweep_data(5);
        sweep.pose_body_order = Some(vec!["crank".to_string(), "rocker".to_string()]);
        sweep.pose_snapshots = Some(
            (0..5)
                .map(|i| {
                    let f = i as f64;
                    vec![[0.1 * f, 0.2 * f, 0.3 * f], [0.4 * f, 0.5 * f, 0.6 * f]]
                })
                .collect(),
        );

        let path = std::env::temp_dir().join("test_trajectory_export_pose.csv");
        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();

        // Header now carries 6 extra pose columns (3 per body × 2 bodies).
        assert_eq!(
            lines[0],
            "t_seconds,target_value,achieved_value,residual,u,u_dot,u_ddot,F_actuator_N,driver_torque_Nm,status,\
             q_x_crank_m,q_y_crank_m,q_theta_crank_rad,q_x_rocker_m,q_y_rocker_m,q_theta_rocker_rad",
            "header should include per-body pose columns"
        );

        // 16 columns on every data row (10 trajectory + 6 pose).
        for (i, line) in lines.iter().enumerate().skip(1) {
            assert_eq!(
                line.split(',').count(),
                16,
                "row {} should have 16 columns: {}",
                i,
                line
            );
        }

        // Sample 2 (i=2): crank pose = (0.2, 0.4, 0.6), rocker = (0.8, 1.0, 1.2).
        let cols2: Vec<&str> = lines[3].split(',').collect();
        assert!((cols2[10].parse::<f64>().unwrap() - 0.2).abs() < 1e-6);
        assert!((cols2[11].parse::<f64>().unwrap() - 0.4).abs() < 1e-6);
        assert!((cols2[12].parse::<f64>().unwrap() - 0.6).abs() < 1e-6);
        assert!((cols2[13].parse::<f64>().unwrap() - 0.8).abs() < 1e-6);
        assert!((cols2[14].parse::<f64>().unwrap() - 1.0).abs() < 1e-6);
        assert!((cols2[15].parse::<f64>().unwrap() - 1.2).abs() < 1e-6);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_sweep_csv_trajectory_omits_pose_columns_when_absent() {
        // Backwards-compat path: a SweepData without pose_body_order should
        // emit the v1 layout exactly (no pose columns, 10 columns per row).
        let sweep = test_trajectory_sweep_data(5);
        assert!(sweep.pose_body_order.is_none(), "fixture has no pose data");

        let path = std::env::temp_dir().join("test_trajectory_export_no_pose.csv");
        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();

        // Header is the v1 string with no extra columns.
        assert_eq!(
            lines[0],
            "t_seconds,target_value,achieved_value,residual,u,u_dot,u_ddot,F_actuator_N,driver_torque_Nm,status"
        );
        for line in lines.iter().skip(1) {
            assert_eq!(line.split(',').count(), 10);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_sweep_csv_trajectory_empty_writes_header_only() {
        // Switching to trajectory mode but never running compute leaves all
        // optional vectors None. We still want a well-formed CSV (header only).
        let mut sweep = test_trajectory_sweep_data(5);
        sweep.target_values = None;
        sweep.achieved_values = None;
        sweep.tracking_residual = None;
        sweep.u_values = None;
        sweep.u_dot_values = None;
        sweep.u_ddot_values = None;
        sweep.inverse_solve_statuses = None;
        sweep.driver_torques = None;

        let path = std::env::temp_dir().join("test_trajectory_export_empty.csv");
        export_sweep_csv(&path, &sweep).expect("export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1, "header only when no trajectory data");
        assert!(lines[0].starts_with("t_seconds,"));

        let _ = std::fs::remove_file(&path);
    }
}
