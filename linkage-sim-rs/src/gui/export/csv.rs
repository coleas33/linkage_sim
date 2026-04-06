//! CSV export: sweep data and coupler trace data.

#[cfg(feature = "native")]
use std::io::Write;
#[cfg(feature = "native")]
use std::path::Path;

#[cfg(feature = "native")]
use crate::gui::sweep::SweepData;

/// Export sweep data to CSV file.
///
/// Columns: driver angle, then sorted body angles, then transmission angle
/// (if available), then driver torque (if available). All angles in degrees,
/// torque in N*m.
#[cfg(feature = "native")]
pub fn export_sweep_csv(path: &Path, sweep: &SweepData) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;

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
    writeln!(file, "{}", headers.join(",")).map_err(|e| e.to_string())?;

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
        writeln!(file, "{}", row.join(",")).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Export coupler trace data to CSV file.
///
/// Columns: driver angle, then sorted coupler trace x/y pairs. Coordinates
/// are in meters (SI).
#[cfg(feature = "native")]
pub fn export_coupler_csv(path: &Path, sweep: &SweepData) -> Result<(), String> {
    let mut trace_names: Vec<&String> = sweep.coupler_traces.keys().collect();
    trace_names.sort();

    if trace_names.is_empty() {
        return Err("No coupler traces available".to_string());
    }

    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;

    // Header: angle_deg, trace1_x, trace1_y, ...
    let mut headers = vec!["angle_deg".to_string()];
    for name in &trace_names {
        headers.push(format!("{}_x_m", name));
        headers.push(format!("{}_y_m", name));
    }
    writeln!(file, "{}", headers.join(",")).map_err(|e| e.to_string())?;

    // Data rows
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
        writeln!(file, "{}", row.join(",")).map_err(|e| e.to_string())?;
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
}
