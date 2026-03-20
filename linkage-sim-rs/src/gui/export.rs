//! Export functionality: CSV sweep data, coupler traces, SVG/PNG/GIF images.

#[cfg(feature = "native")]
use std::io::Write;
#[cfg(feature = "native")]
use std::path::Path;

#[cfg(feature = "native")]
use super::state::SweepData;
#[cfg(feature = "native")]
use crate::core::mechanism::Mechanism;

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

/// Generate the SVG string for a mechanism at its current pose.
///
/// This is the shared core used by both SVG file export and PNG rasterization.
#[cfg(feature = "native")]
pub fn generate_svg_string(
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<String, String> {
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

    Ok(svg)
}

/// Export the mechanism at its current pose as an SVG file.
#[cfg(feature = "native")]
pub fn export_mechanism_svg(
    path: &std::path::Path,
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<(), String> {
    let svg = generate_svg_string(mechanism, q)?;
    std::fs::write(path, svg).map_err(|e| e.to_string())
}

/// Rasterize an SVG string to RGBA pixel data at the given dimensions.
///
/// This is the shared rasterization core used by both PNG export and GIF
/// frame generation. Requires the `native` feature (depends on `resvg`).
#[cfg(feature = "native")]
fn rasterize_svg_to_rgba(
    svg_str: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let opt = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_str(svg_str, &opt)
        .map_err(|e| format!("Failed to parse SVG for rasterization: {}", e))?;

    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| format!("Failed to create {}x{} pixmap", width, height))?;

    pixmap.fill(resvg::tiny_skia::Color::WHITE);

    // Scale uniformly to fit, centering the shorter axis.
    let sx = width as f32 / tree.size().width();
    let sy = height as f32 / tree.size().height();
    let scale = sx.min(sy);
    let tx = (width as f32 - tree.size().width() * scale) / 2.0;
    let ty = (height as f32 - tree.size().height() * scale) / 2.0;

    let transform = resvg::tiny_skia::Transform::from_scale(scale, scale)
        .post_translate(tx, ty);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    Ok(pixmap.take())
}

/// Export the mechanism at its current pose as a PNG image.
///
/// Generates an SVG string, rasterizes it with resvg at the given dimensions,
/// and saves the result as a PNG file. Requires the `native` feature.
#[cfg(feature = "native")]
pub fn export_mechanism_png(
    path: &std::path::Path,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let svg_str = generate_svg_string(mechanism, q)?;
    let rgba = rasterize_svg_to_rgba(&svg_str, width, height)?;

    // Reconstruct a Pixmap from the raw RGBA data so we can use save_png.
    let pixmap = resvg::tiny_skia::Pixmap::from_vec(rgba, resvg::tiny_skia::IntSize::from_wh(width, height).unwrap())
        .ok_or_else(|| "Failed to reconstruct pixmap from RGBA data".to_string())?;

    pixmap.save_png(path)
        .map_err(|e| format!("Failed to save PNG: {}", e))
}

/// Export an animated GIF of the mechanism sweep.
///
/// Re-solves the mechanism position at each sampled sweep step, renders via
/// SVG + resvg, and encodes as a looping animated GIF. Requires the `native`
/// feature (depends on `resvg` and `gif`).
#[cfg(feature = "native")]
pub fn export_mechanism_gif(
    path: &std::path::Path,
    mech: &Mechanism,
    sweep: &SweepData,
    q_start: &nalgebra::DVector<f64>,
    omega: f64,
    theta_0: f64,
    width: u32,
    height: u32,
    frame_delay_cs: u16,
) -> Result<(), String> {
    use crate::solver::kinematics::solve_position;
    use gif::{Encoder, Frame, Repeat};

    let n_steps = sweep.angles_deg.len();
    if n_steps == 0 {
        return Err("No sweep data to export".to_string());
    }

    let file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create GIF file: {}", e))?;
    let mut encoder = Encoder::new(file, width as u16, height as u16, &[])
        .map_err(|e| format!("Failed to initialize GIF encoder: {}", e))?;
    encoder.set_repeat(Repeat::Infinite)
        .map_err(|e| format!("Failed to set GIF repeat: {}", e))?;

    // Target ~72 frames for a smooth animation; skip steps if sweep is denser.
    let step_skip = (n_steps / 72).max(1);
    let mut q_guess = q_start.clone();

    for i in (0..n_steps).step_by(step_skip) {
        let angle_rad = sweep.angles_deg[i].to_radians();
        let t = (angle_rad - theta_0) / omega;

        match solve_position(mech, &q_guess, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                if let Ok(svg_str) = generate_svg_string(mech, &result.q) {
                    if let Ok(rgba) = rasterize_svg_to_rgba(&svg_str, width, height) {
                        let mut rgba_buf = rgba;
                        let mut frame = Frame::from_rgba_speed(
                            width as u16,
                            height as u16,
                            &mut rgba_buf,
                            10,
                        );
                        frame.delay = frame_delay_cs;
                        encoder.write_frame(&frame)
                            .map_err(|e| format!("Failed to write GIF frame: {}", e))?;
                    }
                }
                q_guess = result.q;
            }
            _ => {
                // Skip frames that fail to converge; the animation will still
                // be useful with the frames that do converge.
            }
        }
    }

    Ok(())
}

/// Generate a DXF ASCII string for the mechanism at its current pose.
///
/// Exports: bodies as LINE entities, joints as CIRCLE entities (r=0.002m),
/// ground pivots as POINT entities. All coordinates in meters.
#[cfg(feature = "native")]
pub fn generate_dxf_string(
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<String, String> {
    use crate::core::constraint::Constraint;
    use crate::core::state::GROUND_ID;

    let state = mechanism.state();
    let bodies = mechanism.bodies();
    if bodies.len() <= 1 {
        return Err("No moving bodies to export".to_string());
    }

    let mut dxf = String::with_capacity(4096);

    // DXF header
    dxf.push_str("0\nSECTION\n2\nENTITIES\n");

    // Export bodies as LINE entities
    let mut body_ids: Vec<&String> = bodies.keys().collect();
    body_ids.sort();
    for body_id in &body_ids {
        let body = &bodies[*body_id];
        let mut pts: Vec<(&String, nalgebra::Vector2<f64>)> = body
            .attachment_points
            .iter()
            .map(|(name, local)| {
                let global = state.body_point_global(body_id, local, q);
                (name, global)
            })
            .collect();
        pts.sort_by_key(|(n, _)| n.as_str());

        // Lines between consecutive points
        for pair in pts.windows(2) {
            let (_, p1) = &pair[0];
            let (_, p2) = &pair[1];
            dxf.push_str(&format!(
                "0\nLINE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n11\n{:.6}\n21\n{:.6}\n31\n0.0\n",
                body_id, p1.x, p1.y, p2.x, p2.y
            ));
        }
        // Close polygon for 3+ points
        if pts.len() >= 3 {
            let (_, p_last) = pts.last().unwrap();
            let (_, p_first) = &pts[0];
            dxf.push_str(&format!(
                "0\nLINE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n11\n{:.6}\n21\n{:.6}\n31\n0.0\n",
                body_id, p_last.x, p_last.y, p_first.x, p_first.y
            ));
        }
    }

    // Export joints as CIRCLE entities (radius 0.002 m)
    let joint_radius = 0.002;
    for joint in mechanism.joints() {
        let pt = state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
        let layer = if joint.is_revolute() {
            "JOINTS_REVOLUTE"
        } else if joint.is_prismatic() {
            "JOINTS_PRISMATIC"
        } else {
            "JOINTS"
        };
        dxf.push_str(&format!(
            "0\nCIRCLE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n40\n{:.6}\n",
            layer, pt.x, pt.y, joint_radius
        ));
    }

    // Export ground attachment points as POINT entities
    if let Some(ground) = bodies.get(GROUND_ID) {
        for (_, local) in &ground.attachment_points {
            let global = state.body_point_global(GROUND_ID, local, q);
            dxf.push_str(&format!(
                "0\nPOINT\n8\nGROUND\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n",
                global.x, global.y
            ));
        }
    }

    // DXF footer
    dxf.push_str("0\nENDSEC\n0\nEOF\n");

    Ok(dxf)
}

/// Export mechanism geometry as a DXF file.
#[cfg(feature = "native")]
pub fn export_mechanism_dxf(
    path: &std::path::Path,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<(), String> {
    let dxf = generate_dxf_string(mechanism, q)?;
    std::fs::write(path, dxf).map_err(|e| e.to_string())
}

/// Generate an HTML report summarizing the current mechanism analysis.
///
/// Includes: mechanism diagram (embedded SVG), topology, dimensions, mass
/// properties, Grashof classification, torque/transmission/reaction envelopes,
/// force element summary, and energy data.
#[cfg(feature = "native")]
pub fn generate_html_report(
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
    sweep: &SweepData,
    grashof: Option<&crate::analysis::grashof::GrashofResult>,
    units: &super::state::DisplayUnits,
) -> Result<String, String> {
    use crate::analysis::envelopes::compute_envelope;
    use crate::core::state::GROUND_ID;

    let svg = generate_svg_string(mechanism, q).unwrap_or_else(|_| String::new());

    let mut html = String::with_capacity(16_000);
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n");
    html.push_str("<title>Linkage Mechanism Report</title>\n<style>\n");
    html.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #1a1a2e; }\n");
    html.push_str("h1 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }\n");
    html.push_str("h2 { color: #0f3460; margin-top: 28px; }\n");
    html.push_str("table { border-collapse: collapse; width: 100%; margin: 12px 0; }\n");
    html.push_str("th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }\n");
    html.push_str("th { background: #e8eaf6; font-weight: 600; }\n");
    html.push_str("tr:nth-child(even) { background: #f0f2f5; }\n");
    html.push_str(".diagram { text-align: center; margin: 16px 0; background: #1e1e23; border-radius: 8px; padding: 12px; }\n");
    html.push_str(".diagram svg { max-width: 100%; height: auto; }\n");
    html.push_str(".summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }\n");
    html.push_str(".card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }\n");
    html.push_str(".card h3 { margin: 0 0 8px 0; font-size: 14px; color: #555; }\n");
    html.push_str(".card .value { font-size: 22px; font-weight: 700; color: #0f3460; }\n");
    html.push_str(".footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 12px; color: #888; }\n");
    html.push_str("</style></head><body>\n");

    // ── Header ──────────────────────────────────────────────────────
    html.push_str("<h1>Linkage Mechanism Report</h1>\n");
    html.push_str(&format!("<p>Generated: {}</p>\n", chrono_now()));

    // ── Topology summary cards ──────────────────────────────────────
    let n_bodies = mechanism.bodies().len().saturating_sub(1);
    let n_joints = mechanism.joints().len();
    let dof = mechanism.state().n_coords() as isize - mechanism.n_constraints() as isize;
    let total_mass: f64 = mechanism.bodies().values()
        .filter(|b| b.id != GROUND_ID)
        .map(|b| b.mass)
        .sum();

    html.push_str("<div class='summary'>\n");
    html.push_str(&format!("<div class='card'><h3>Bodies</h3><div class='value'>{}</div></div>\n", n_bodies));
    html.push_str(&format!("<div class='card'><h3>Joints</h3><div class='value'>{}</div></div>\n", n_joints));
    html.push_str(&format!("<div class='card'><h3>DOF</h3><div class='value'>{}</div></div>\n", dof));
    html.push_str(&format!("<div class='card'><h3>Total Mass</h3><div class='value'>{:.3} kg</div></div>\n", total_mass));
    html.push_str("</div>\n");

    // ── Grashof classification ──────────────────────────────────────
    if let Some(gr) = grashof {
        let label = match gr.classification {
            crate::analysis::grashof::GrashofType::CrankRocker => "Crank-Rocker (Grashof)",
            crate::analysis::grashof::GrashofType::DoubleCrank => "Double-Crank (Grashof)",
            crate::analysis::grashof::GrashofType::DoubleRocker => "Double-Rocker (Grashof)",
            crate::analysis::grashof::GrashofType::ChangePoint => "Change-Point",
            crate::analysis::grashof::GrashofType::NonGrashof => "Non-Grashof",
        };
        html.push_str(&format!("<p><strong>Classification:</strong> {}</p>\n", label));
        html.push_str(&format!(
            "<p>Link lengths: {:.3}, {:.3}, {:.3}, {:.3}{}</p>\n",
            units.length(gr.link_lengths[0]),
            units.length(gr.link_lengths[1]),
            units.length(gr.link_lengths[2]),
            units.length(gr.link_lengths[3]),
            units.length_suffix()
        ));
    }

    // ── Mechanism diagram ───────────────────────────────────────────
    html.push_str("<h2>Mechanism Diagram</h2>\n");
    html.push_str("<div class='diagram'>\n");
    html.push_str(&svg);
    html.push_str("\n</div>\n");

    // ── Dimensions table ────────────────────────────────────────────
    html.push_str("<h2>Dimensions</h2>\n");
    html.push_str("<table><tr><th>Body</th><th>Segment</th><th>Length</th></tr>\n");
    let mut body_ids: Vec<&String> = mechanism.bodies().keys().collect();
    body_ids.sort();
    for body_id in &body_ids {
        if *body_id == GROUND_ID { continue; }
        let body = &mechanism.bodies()[*body_id];
        let mut pts: Vec<(&String, &nalgebra::Vector2<f64>)> = body.attachment_points.iter().collect();
        pts.sort_by_key(|(n, _)| n.as_str());
        for pair in pts.windows(2) {
            let (na, pa) = pair[0];
            let (nb, pb) = pair[1];
            let dist = (pb - pa).norm();
            html.push_str(&format!(
                "<tr><td>{}</td><td>{} \u{2192} {}</td><td>{:.3}{}</td></tr>\n",
                body_id, na, nb, units.length(dist), units.length_suffix()
            ));
        }
    }
    html.push_str("</table>\n");

    // ── Mass properties table ───────────────────────────────────────
    html.push_str("<h2>Mass Properties</h2>\n");
    html.push_str("<table><tr><th>Body</th><th>Mass (kg)</th><th>Izz (kg*m^2)</th><th>CG local</th></tr>\n");
    for body_id in &body_ids {
        if *body_id == GROUND_ID { continue; }
        let body = &mechanism.bodies()[*body_id];
        html.push_str(&format!(
            "<tr><td>{}</td><td>{:.4}</td><td>{:.6}</td><td>({:.4}, {:.4})</td></tr>\n",
            body_id, body.mass, body.izz_cg, body.cg_local.x, body.cg_local.y
        ));
    }
    html.push_str("</table>\n");

    // ── Torque envelope ─────────────────────────────────────────────
    if let Some(ref torques) = sweep.driver_torques {
        if let Some(env) = compute_envelope(torques) {
            html.push_str("<h2>Driver Torque Envelope</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!("<div class='card'><h3>Peak (abs)</h3><div class='value'>{:.3} N*m</div></div>\n",
                env.max_value.abs().max(env.min_value.abs())));
            html.push_str(&format!("<div class='card'><h3>RMS</h3><div class='value'>{:.3} N*m</div></div>\n", env.rms));
            html.push_str(&format!("<div class='card'><h3>Min</h3><div class='value'>{:.3} N*m</div></div>\n", env.min_value));
            html.push_str(&format!("<div class='card'><h3>Max</h3><div class='value'>{:.3} N*m</div></div>\n", env.max_value));
            html.push_str("</div>\n");
        }
    }

    // ── Transmission angle range ────────────────────────────────────
    if let Some(ref ta) = sweep.transmission_angles {
        if let Some(env) = compute_envelope(ta) {
            html.push_str("<h2>Transmission Angle</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!(
                "<div class='card'><h3>Min</h3><div class='value'>{:.1}\u{00b0}</div></div>\n",
                env.min_value
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Max</h3><div class='value'>{:.1}\u{00b0}</div></div>\n",
                env.max_value
            ));
            html.push_str("</div>\n");
            if env.min_value < 40.0 {
                html.push_str("<p style='color: #c62828;'><strong>Warning:</strong> Minimum transmission angle is below 40 degrees — poor force transmission in this region.</p>\n");
            }
        }
    }

    // ── Joint reaction peaks ────────────────────────────────────────
    if !sweep.joint_reaction_magnitudes.is_empty() {
        html.push_str("<h2>Joint Reaction Peaks</h2>\n");
        html.push_str("<table><tr><th>Joint</th><th>Peak Force (N)</th><th>Mean Force (N)</th></tr>\n");
        let mut jids: Vec<&String> = sweep.joint_reaction_magnitudes.keys().collect();
        jids.sort();
        for jid in jids {
            let vals = &sweep.joint_reaction_magnitudes[jid];
            if let Some(env) = compute_envelope(vals) {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{:.3}</td><td>{:.3}</td></tr>\n",
                    jid, env.max_value, env.mean
                ));
            }
        }
        html.push_str("</table>\n");
    }

    // ── Force element summary ───────────────────────────────────────
    let forces = mechanism.forces();
    if !forces.is_empty() {
        html.push_str("<h2>Force Elements</h2>\n");
        html.push_str("<table><tr><th>#</th><th>Type</th><th>Details</th></tr>\n");
        for (i, fe) in forces.iter().enumerate() {
            let (type_name, details) = force_element_summary(fe);
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                i + 1, type_name, details
            ));
        }
        html.push_str("</table>\n");
    }

    // ── Energy summary ──────────────────────────────────────────────
    if !sweep.kinetic_energy.is_empty() {
        if let (Some(ke_env), Some(pe_env)) = (
            compute_envelope(&sweep.kinetic_energy),
            compute_envelope(&sweep.potential_energy),
        ) {
            html.push_str("<h2>Energy Summary</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!("<div class='card'><h3>Peak KE</h3><div class='value'>{:.4} J</div></div>\n", ke_env.max_value));
            html.push_str(&format!("<div class='card'><h3>Peak PE</h3><div class='value'>{:.4} J</div></div>\n", pe_env.max_value));
            html.push_str("</div>\n");
        }
    }

    // ── Footer ──────────────────────────────────────────────────────
    html.push_str("<div class='footer'>\n");
    html.push_str("<p>Generated by Linkage Mechanism Simulator (Rust/egui)</p>\n");
    html.push_str("</div>\n");
    html.push_str("</body></html>\n");

    Ok(html)
}

/// Get current timestamp as a formatted string.
#[cfg(feature = "native")]
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple UTC timestamp without chrono dependency
    format!("Unix timestamp {}", now)
}

/// Summarize a force element as (type_name, detail_string).
#[cfg(feature = "native")]
fn force_element_summary(fe: &crate::forces::elements::ForceElement) -> (&'static str, String) {
    use crate::forces::elements::ForceElement;
    match fe {
        ForceElement::Gravity(e) => ("Gravity", format!("g = ({:.2}, {:.2}) m/s^2", e.g_vector[0], e.g_vector[1])),
        ForceElement::LinearSpring(e) => ("Linear Spring", format!("k = {:.1} N/m, L0 = {:.4} m", e.stiffness, e.free_length)),
        ForceElement::TorsionSpring(e) => ("Torsion Spring", format!("k = {:.2} N*m/rad, free = {:.2} rad", e.stiffness, e.free_angle)),
        ForceElement::LinearDamper(e) => ("Linear Damper", format!("c = {:.2} N*s/m", e.damping)),
        ForceElement::RotaryDamper(e) => ("Rotary Damper", format!("c = {:.2} N*m*s/rad", e.damping)),
        ForceElement::ExternalForce(e) => ("External Force", format!("F = ({:.2}, {:.2}) N on {}", e.force[0], e.force[1], e.body_id)),
        ForceElement::ExternalTorque(e) => ("External Torque", format!("T = {:.2} N*m on {}", e.torque, e.body_id)),
        ForceElement::GasSpring(e) => ("Gas Spring", format!("F0 = {:.1} N, stroke = {:.4} m", e.initial_force, e.stroke)),
        ForceElement::BearingFriction(e) => ("Bearing Friction", format!("drag = {:.3}, viscous = {:.3}", e.constant_drag, e.viscous_coeff)),
        ForceElement::JointLimit(e) => ("Joint Limit", format!("range = [{:.1}, {:.1}] rad, k = {:.0} N*m/rad", e.angle_min, e.angle_max, e.stiffness)),
        ForceElement::Motor(e) => ("Motor", format!("T_stall = {:.2} N*m, w_nl = {:.2} rad/s", e.stall_torque, e.no_load_speed)),
        ForceElement::LinearActuator(e) => ("Linear Actuator", format!("F = {:.1} N, v_max = {:.3} m/s", e.force, e.speed_limit)),
    }
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
            toggle_angles: Vec::new(),
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

    #[test]
    fn generate_svg_string_returns_valid_svg() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::CrankRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let svg = generate_svg_string(&mech, &q).expect("SVG generation should succeed");
        assert!(svg.contains("<svg"), "should contain SVG root element");
        assert!(svg.contains("</svg>"), "SVG should be properly closed");
        assert!(svg.contains("class=\"body\""), "should contain body lines");
    }

    #[test]
    fn generate_svg_string_empty_mechanism_returns_error() {
        use crate::core::mechanism::Mechanism;
        use nalgebra::DVector;

        let mut mech = Mechanism::new();
        mech.build().expect("build should succeed");
        let q = DVector::zeros(0);

        let result = generate_svg_string(&mech, &q);
        assert!(result.is_err(), "empty mechanism should return an error");
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_png_produces_valid_file() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::CrankRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let path = std::env::temp_dir().join("test_mechanism.png");
        export_mechanism_png(&path, &mech, &q, 1920, 1080)
            .expect("PNG export should succeed");

        // Verify the file exists and has reasonable size
        let metadata = std::fs::metadata(&path).expect("PNG file should exist");
        assert!(metadata.len() > 100, "PNG file should not be empty");

        // Verify PNG magic bytes
        let bytes = std::fs::read(&path).expect("should read PNG file");
        assert_eq!(
            &bytes[..4],
            &[0x89, 0x50, 0x4E, 0x47],
            "file should start with PNG magic bytes"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_png_empty_mechanism_returns_error() {
        use crate::core::mechanism::Mechanism;
        use nalgebra::DVector;

        let mut mech = Mechanism::new();
        mech.build().expect("build should succeed");
        let q = DVector::zeros(0);

        let path = std::env::temp_dir().join("test_mechanism_empty.png");
        let result = export_mechanism_png(&path, &mech, &q, 1920, 1080);
        assert!(result.is_err(), "empty mechanism should return an error");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_png_custom_dimensions() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        // Test with a smaller resolution
        let path = std::env::temp_dir().join("test_mechanism_small.png");
        export_mechanism_png(&path, &mech, &q, 640, 480)
            .expect("PNG export at 640x480 should succeed");

        let metadata = std::fs::metadata(&path).expect("PNG file should exist");
        assert!(metadata.len() > 100, "PNG file should not be empty");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_produces_valid_file() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;

        // Use AppState to get sweep data (which requires a full sample load).
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let mech = state.mechanism.as_ref().expect("mechanism should be loaded");
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some");

        let path = std::env::temp_dir().join("test_mechanism.gif");
        export_mechanism_gif(
            &path,
            mech,
            sweep,
            &state.q,
            state.driver_omega,
            state.driver_theta_0,
            400,
            300,
            5,
        )
        .expect("GIF export should succeed");

        // Verify the file exists and has reasonable size.
        let metadata = std::fs::metadata(&path).expect("GIF file should exist");
        assert!(metadata.len() > 100, "GIF file should not be empty");

        // Verify GIF magic bytes ("GIF89a" for animated GIFs).
        let bytes = std::fs::read(&path).expect("should read GIF file");
        assert_eq!(
            &bytes[..6],
            b"GIF89a",
            "file should start with GIF89a magic bytes"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_empty_sweep_returns_error() {
        use crate::gui::samples::{build_sample, SampleMechanism};

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let empty_sweep = SweepData {
            angles_deg: vec![],
            body_angles: HashMap::new(),
            coupler_traces: HashMap::new(),
            transmission_angles: None,
            driver_torques: None,
            kinetic_energy: vec![],
            potential_energy: vec![],
            total_energy: vec![],
            inverse_dynamics_torques: vec![],
            mechanical_advantage: vec![],
            joint_reaction_magnitudes: HashMap::new(),
            coupler_velocities: HashMap::new(),
            coupler_accelerations: HashMap::new(),
            toggle_angles: Vec::new(),
        };

        let path = std::env::temp_dir().join("test_mechanism_empty.gif");
        let result = export_mechanism_gif(
            &path,
            &mech,
            &empty_sweep,
            &q0,
            2.0 * std::f64::consts::PI,
            0.0,
            400,
            300,
            5,
        );
        assert!(result.is_err(), "empty sweep should return an error");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_crank_rocker_produces_valid_file() {
        use crate::gui::state::AppState;
        use crate::gui::samples::SampleMechanism;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let mech = state.mechanism.as_ref().expect("mechanism should be loaded");
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some");

        let path = std::env::temp_dir().join("test_crank_rocker.gif");
        export_mechanism_gif(
            &path,
            mech,
            sweep,
            &state.q,
            state.driver_omega,
            state.driver_theta_0,
            800,
            600,
            5,
        )
        .expect("CrankRocker GIF export should succeed");

        let metadata = std::fs::metadata(&path).expect("GIF file should exist");
        assert!(
            metadata.len() > 1000,
            "CrankRocker GIF should have multiple frames"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn rasterize_svg_to_rgba_produces_correct_size() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let svg = generate_svg_string(&mech, &q).expect("SVG generation should succeed");
        let rgba = rasterize_svg_to_rgba(&svg, 320, 240).expect("rasterization should succeed");

        // RGBA: 4 bytes per pixel
        assert_eq!(
            rgba.len(),
            320 * 240 * 4,
            "RGBA buffer should be width * height * 4 bytes"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generate_html_report_contains_expected_sections() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::gui::state::DisplayUnits;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q = if result.converged { result.q } else { q0 };

        let (sweep, _) = crate::gui::state::compute_sweep_data(
            &mech, &q, 2.0 * std::f64::consts::PI, 0.0, 9.81,
        );

        let units = DisplayUnits::default();

        let html = generate_html_report(&mech, &q, &sweep, None, &units)
            .expect("report generation should succeed");

        assert!(html.contains("Linkage Mechanism Report"), "should have title");
        assert!(html.contains("Mechanism Diagram"), "should have diagram section");
        assert!(html.contains("<svg"), "should have embedded SVG");
        assert!(html.contains("Dimensions"), "should have dimensions table");
        assert!(html.contains("Mass Properties"), "should have mass properties table");
        assert!(html.contains("kg"), "should have mass units");
    }

    #[cfg(feature = "native")]
    #[test]
    fn generate_dxf_contains_line_and_circle_entities() {
        use crate::gui::samples::{build_sample, SampleMechanism};

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q = if result.converged { result.q } else { q0 };

        let dxf = generate_dxf_string(&mech, &q).expect("DXF generation should succeed");

        assert!(dxf.contains("SECTION"), "should have SECTION header");
        assert!(dxf.contains("ENTITIES"), "should have ENTITIES section");
        assert!(dxf.contains("LINE"), "should have LINE entities for bodies");
        assert!(dxf.contains("CIRCLE"), "should have CIRCLE entities for joints");
        assert!(dxf.contains("POINT"), "should have POINT entities for ground");
        assert!(dxf.contains("EOF"), "should have EOF marker");
    }
}
