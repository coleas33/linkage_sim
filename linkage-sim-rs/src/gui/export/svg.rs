//! SVG generation and file export for mechanism diagrams.

use crate::core::mechanism::Mechanism;

/// Generate the SVG string for a mechanism at its current pose.
///
/// This is the shared core used by both SVG file export and PNG rasterization.
pub fn generate_svg_string(
    mechanism: &Mechanism,
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
  .body {{ stroke: #4696f0; stroke-width: 1.5; fill: #4696f0; fill-opacity: 0.25; stroke-linecap: round; stroke-linejoin: round; }}
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
            // Draw filled bars (matching canvas rendering) instead of hairlines.
            // half_width is in SVG coordinates; scale factor maps world -> SVG,
            // and the canvas uses 8 px half-width at ~150 px/m. We use a fixed
            // SVG-space half-width that looks proportional to the 800-wide SVG.
            let half_w = 6.0_f64;
            let draw_bar = |svg: &mut String, ax: f64, ay: f64, bx: f64, by: f64| {
                let dx = bx - ax;
                let dy = by - ay;
                let len = (dx * dx + dy * dy).sqrt().max(1e-10);
                let nx = -dy / len * half_w;
                let ny = dx / len * half_w;
                svg.push_str(&format!(
                    r#"<polygon points="{:.2},{:.2} {:.2},{:.2} {:.2},{:.2} {:.2},{:.2}" class="body"/>
"#,
                    ax + nx, ay + ny,
                    bx + nx, by + ny,
                    bx - nx, by - ny,
                    ax - nx, ay - ny,
                ));
            };
            for pair in screen_points.windows(2) {
                draw_bar(&mut svg, pair[0].0, pair[0].1, pair[1].0, pair[1].1);
            }
            if screen_points.len() >= 3 {
                let last = screen_points.last().unwrap();
                let first = &screen_points[0];
                draw_bar(&mut svg, last.0, last.1, first.0, first.1);
            }
        }

        // Body label at the centroid of all attachment points.
        if !screen_points.is_empty() {
            let centroid_x: f64 =
                screen_points.iter().map(|p| p.0).sum::<f64>() / screen_points.len() as f64;
            let centroid_y: f64 =
                screen_points.iter().map(|p| p.1).sum::<f64>() / screen_points.len() as f64;
            svg.push_str(&format!(
                r#"<text x="{:.2}" y="{:.2}" class="label" text-anchor="middle">{}</text>
"#,
                centroid_x,
                centroid_y - 14.0,
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
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<(), String> {
    let svg = generate_svg_string(mechanism, q)?;
    std::fs::write(path, svg).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
