//! Export functionality: CSV sweep data and coupler traces.

use std::io::Write;
use std::path::Path;

use super::state::SweepData;

/// Export sweep data to CSV file.
///
/// Columns: driver angle, then sorted body angles, then transmission angle
/// (if available), then driver torque (if available). All angles in degrees,
/// torque in N*m.
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
        writeln!(file, "{}", row.join(",")).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Export coupler trace data to CSV file.
///
/// Columns: driver angle, then sorted coupler trace x/y pairs. Coordinates
/// are in meters (SI).
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

/// Export the mechanism at its current pose as an SVG file.
pub fn export_mechanism_svg(
    path: &std::path::Path,
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<(), String> {
    use crate::core::constraint::Constraint;
    use crate::core::state::GROUND_ID;

    let state = mechanism.state();
    let bodies = mechanism.bodies();
    let joints = mechanism.joints();

    // Compute bounding box of all attachment points in world coords.
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (body_id, body) in bodies.iter() {
        for (_name, pt_local) in &body.attachment_points {
            let global = state.body_point_global(body_id, pt_local, q);
            min_x = min_x.min(global.x);
            min_y = min_y.min(global.y);
            max_x = max_x.max(global.x);
            max_y = max_y.max(global.y);
        }
    }

    // Guard against degenerate (empty or zero-size) bounding boxes.
    if min_x == f64::MAX || max_x == f64::MIN {
        return Err("No attachment points found in mechanism".to_string());
    }
    if (max_x - min_x).abs() < 1e-10 {
        max_x = min_x + 1.0;
    }
    if (max_y - min_y).abs() < 1e-10 {
        max_y = min_y + 1.0;
    }

    // Add margin (15% of bounding box on each side).
    let margin_x = (max_x - min_x) * 0.15;
    let margin_y = (max_y - min_y) * 0.15;
    min_x -= margin_x;
    min_y -= margin_y;
    max_x += margin_x;
    max_y += margin_y;

    let width = max_x - min_x;
    let height = max_y - min_y;

    // SVG coordinate system: Y grows down. Flip by negating Y in the transform.
    let svg_width = 800.0_f64;
    let svg_height = svg_width * (height / width);
    let scale = svg_width / width;

    let mut svg = String::new();
    svg.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{:.0}" height="{:.0}" viewBox="0 0 {:.0} {:.0}">
<style>
  .body {{ stroke: #4080c0; stroke-width: 2.5; fill: none; stroke-linecap: round; }}
  .joint-revolute {{ fill: none; stroke: #cccccc; stroke-width: 1.5; }}
  .joint-prismatic {{ fill: none; stroke: #cccccc; stroke-width: 1.5; }}
  .joint-fixed {{ fill: #999999; stroke: none; }}
  .ground-marker {{ fill: none; stroke: #888870; stroke-width: 1.2; }}
  .ground-hatch {{ stroke: #888870; stroke-width: 0.8; }}
  .label {{ font-family: sans-serif; font-size: 10px; fill: #666666; }}
</style>
<rect width="100%" height="100%" fill="#1e1e23"/>
"##,
        svg_width, svg_height, svg_width, svg_height
    ));

    // Helper: world coords to SVG coords (flips Y axis).
    let to_svg = |wx: f64, wy: f64| -> (f64, f64) {
        let sx = (wx - min_x) * scale;
        let sy = (max_y - wy) * scale;
        (sx, sy)
    };

    // Draw bodies.
    for (body_id, body) in bodies.iter() {
        if body_id == GROUND_ID {
            continue;
        }

        let mut point_names: Vec<&String> = body.attachment_points.keys().collect();
        point_names.sort();

        let screen_points: Vec<(f64, f64)> = point_names
            .iter()
            .map(|name| {
                let local = &body.attachment_points[*name];
                let global = state.body_point_global(body_id, local, q);
                to_svg(global.x, global.y)
            })
            .collect();

        if screen_points.len() >= 2 {
            for pair in screen_points.windows(2) {
                svg.push_str(&format!(
                    r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" class="body"/>
"#,
                    pair[0].0, pair[0].1, pair[1].0, pair[1].1
                ));
            }
        }

        // Body label at the first attachment point.
        if let Some(first) = screen_points.first() {
            svg.push_str(&format!(
                r#"<text x="{:.2}" y="{:.2}" class="label" text-anchor="middle">{}</text>
"#,
                first.0,
                first.1 - 8.0,
                body_id
            ));
        }
    }

    // Draw ground markers.
    if let Some(ground) = bodies.get(GROUND_ID) {
        for (_name, pt_local) in &ground.attachment_points {
            let global = state.body_point_global(GROUND_ID, pt_local, q);
            let (sx, sy) = to_svg(global.x, global.y);
            let s = 8.0_f64;
            // Downward-pointing triangle.
            svg.push_str(&format!(
                r#"<polygon points="{:.2},{:.2} {:.2},{:.2} {:.2},{:.2}" class="ground-marker"/>
"#,
                sx,
                sy,
                sx - s * 0.6,
                sy + s,
                sx + s * 0.6,
                sy + s
            ));
            // Hatch lines below the triangle.
            for i in 0..4 {
                let hx = sx - s * 0.5 + (i as f64) * s * 0.3;
                svg.push_str(&format!(
                    r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" class="ground-hatch"/>
"#,
                    hx,
                    sy + s,
                    hx - 3.0,
                    sy + s + 4.0
                ));
            }
        }
    }

    // Draw joints.
    for joint in joints {
        let global = state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
        let (sx, sy) = to_svg(global.x, global.y);
        let r = 4.0_f64;

        if joint.is_revolute() {
            svg.push_str(&format!(
                r#"<circle cx="{:.2}" cy="{:.2}" r="{:.1}" class="joint-revolute"/>
"#,
                sx, sy, r
            ));
        } else if joint.is_prismatic() {
            svg.push_str(&format!(
                r#"<rect x="{:.2}" y="{:.2}" width="{:.1}" height="{:.1}" class="joint-prismatic"/>
"#,
                sx - r,
                sy - r,
                r * 2.0,
                r * 2.0
            ));
        } else {
            svg.push_str(&format!(
                r#"<circle cx="{:.2}" cy="{:.2}" r="{:.1}" class="joint-fixed"/>
"#,
                sx,
                sy,
                r * 0.8
            ));
        }

        // Joint label.
        svg.push_str(&format!(
            r#"<text x="{:.2}" y="{:.2}" class="label" font-size="8">{}</text>
"#,
            sx + 6.0,
            sy - 4.0,
            joint.id()
        ));
    }

    svg.push_str("</svg>\n");

    std::fs::write(path, svg).map_err(|e| e.to_string())
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

    #[test]
    fn export_svg_produces_valid_file() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::CrankRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let path = std::env::temp_dir().join("test_mechanism.svg");
        export_mechanism_svg(&path, &mech, &q).expect("SVG export should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read file");
        assert!(contents.contains("<svg"), "should be valid SVG");
        assert!(contents.contains("class=\"body\""), "should contain body lines");
        assert!(
            contents.contains("class=\"joint-revolute\""),
            "should contain revolute joints"
        );
        assert!(
            contents.contains("</svg>"),
            "SVG should be properly closed"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn export_svg_empty_mechanism_returns_error() {
        use crate::core::mechanism::Mechanism;
        use nalgebra::DVector;

        let mut mech = Mechanism::new();
        // Add only ground so there are no attachment points.
        mech.build().expect("build should succeed");
        let q = DVector::zeros(0);

        let path = std::env::temp_dir().join("test_mechanism_empty.svg");
        let result = export_mechanism_svg(&path, &mech, &q);
        assert!(result.is_err(), "empty mechanism should return an error");

        let _ = std::fs::remove_file(&path);
    }
}
