//! Auto-generated labeled mechanism schematic (SVG).
//!
//! Produces a figure that matches the visual style of Figure 2 / Figure 3 in
//! `docs/superpowers/specs/2026-04-29-linkage-equations-reference.md` for the
//! user's currently-loaded mechanism, with constraint labels and a legend.
//!
//! Output structure (top-to-bottom of the SVG):
//!   1. Title strip (mechanism summary + DOF)
//!   2. Mechanism drawing area (left side, ~ x=20..460):
//!      - Ground hatching (only behind ground attachment points)
//!      - Bars (closed polygon between attachment points of each non-ground body)
//!      - Joints as open circles, clustered by world position so coincident
//!        joints render as one circle with multiple labels
//!      - Body labels (italic gray) at body centroid
//!      - Driver indicator (red) — curved arrow for revolute, line for linear
//!   3. Sidebar legend (right side, x=475..595):
//!      - Constraint rows grouped by joint type
//!      - Driver row (highlighted red)
//!      - Totals (n, m, DOF)
//!      - Multiplier note

use nalgebra::DVector;

use crate::core::constraint::JointConstraint;
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::gui::eq_rendering;

/// SVG canvas size — drawing area (top) + legend area (bottom).
///
/// Legend used to live in a sidebar at x=475..595 but constraint rows would
/// run off the bottom on mechanisms with many joints / drivers. The legend
/// is now stacked below the drawing area, which gives it the full canvas
/// width and unbounded vertical room.
const SVG_WIDTH: f64 = 600.0;
const SVG_HEIGHT: f64 = 500.0;

/// Drawing region where the mechanism is rendered (within the SVG canvas).
/// Leaves room above for the title strip and below for the legend.
const DRAW_X_MIN: f64 = 60.0;
const DRAW_X_MAX: f64 = 580.0;
const DRAW_Y_MIN: f64 = 60.0;
const DRAW_Y_MAX: f64 = 310.0;

/// Y position where the legend area starts (below the drawing region,
/// with a small gutter).
const LEGEND_Y_START: f64 = 340.0;
/// X position where the legend's text rows start.
const LEGEND_X_START: f64 = 30.0;

/// Tolerance for clustering joints that share a world position (in world units).
const POSITION_TOLERANCE: f64 = 1e-4;

/// Generate an SVG string showing the mechanism at pose `q` with all
/// constraint and body labels annotated. The output is a complete `<svg>`
/// document including viewBox, title, sidebar legend, and labeled geometry.
pub fn generate_schematic_svg(
    mech: &Mechanism,
    q: &DVector<f64>,
) -> Result<String, String> {
    // Bring the Constraint trait into scope so we can call id() / body_i_id()
    // / body_j_id() / n_equations() on concrete RevoluteDriver / LinearDriver
    // / JointConstraint values via the trait.
    use crate::core::constraint::Constraint as _;

    let state = mech.state();
    let bodies = mech.bodies();

    // ── 1. Compute world bounding box from all attachment points ──────────
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for (body_id, body) in bodies.iter() {
        for (_name, pt_local) in &body.attachment_points {
            let g = state.body_point_global(body_id, pt_local, q);
            min_x = min_x.min(g.x);
            min_y = min_y.min(g.y);
            max_x = max_x.max(g.x);
            max_y = max_y.max(g.y);
        }
    }
    // Include linear-driver attachment points so they don't fall outside the box.
    for drv in mech.linear_drivers() {
        let pa = drv.point_a();
        let pb = drv.point_b();
        let g_a = state.body_point_global(
            drv.body_i_id(),
            &nalgebra::Vector2::new(pa[0], pa[1]),
            q,
        );
        let g_b = state.body_point_global(
            drv.body_j_id(),
            &nalgebra::Vector2::new(pb[0], pb[1]),
            q,
        );
        min_x = min_x.min(g_a.x).min(g_b.x);
        min_y = min_y.min(g_a.y).min(g_b.y);
        max_x = max_x.max(g_a.x).max(g_b.x);
        max_y = max_y.max(g_a.y).max(g_b.y);
    }

    if min_x == f64::MAX || max_x == f64::MIN {
        return Err("No attachment points found in mechanism".to_string());
    }

    // Guard degenerate (zero-size) bounding boxes.
    if (max_x - min_x).abs() < 1e-12 {
        let cx = 0.5 * (min_x + max_x);
        min_x = cx - 0.5;
        max_x = cx + 0.5;
    }
    if (max_y - min_y).abs() < 1e-12 {
        let cy = 0.5 * (min_y + max_y);
        min_y = cy - 0.5;
        max_y = cy + 0.5;
    }

    // Pad the bounding box so labels don't get clipped against the drawing region.
    let pad_x = (max_x - min_x) * 0.20;
    let pad_y = (max_y - min_y) * 0.20;
    min_x -= pad_x;
    max_x += pad_x;
    min_y -= pad_y;
    max_y += pad_y;

    let world_w = max_x - min_x;
    let world_h = max_y - min_y;
    let region_w = DRAW_X_MAX - DRAW_X_MIN;
    let region_h = DRAW_Y_MAX - DRAW_Y_MIN;

    // Use a single isotropic scale so the geometry is not stretched.
    let scale = (region_w / world_w).min(region_h / world_h);

    // Centred mapping inside the drawing region.
    let region_cx = 0.5 * (DRAW_X_MIN + DRAW_X_MAX);
    let region_cy = 0.5 * (DRAW_Y_MIN + DRAW_Y_MAX);
    let world_cx = 0.5 * (min_x + max_x);
    let world_cy = 0.5 * (min_y + max_y);

    // Math y grows upward; SVG y grows downward → flip.
    let to_svg = |wx: f64, wy: f64| -> (f64, f64) {
        let sx = region_cx + (wx - world_cx) * scale;
        let sy = region_cy - (wy - world_cy) * scale;
        (sx, sy)
    };

    // ── 2. Compute body summaries we need for labels and the legend ────────
    let n_moving_bodies = bodies.len().saturating_sub(1);
    let n_constraints = mech.n_constraints();
    let n_coords = state.n_coords();
    let dof = n_coords as isize - n_constraints as isize;

    let mut svg = String::with_capacity(8_192);

    // ── 3. SVG header + styles ─────────────────────────────────────────────
    svg.push_str(&format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0}" height="{h:.0}" viewBox="0 0 {w:.0} {h:.0}" font-family="serif">
<defs>
  <marker id="ah_drv" viewBox="0 0 10 10" refX="9" refY="3" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M 0 0 L 9 3 L 0 6 z" fill="#a02020"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="white"/>
"##,
        w = SVG_WIDTH,
        h = SVG_HEIGHT
    ));

    // ── 4. Title strip ─────────────────────────────────────────────────────
    svg.push_str(&format!(
        r##"<text x="20" y="22" font-size="14" font-weight="bold">Mechanism schematic — {} moving bodies, {} constraints, DOF={}</text>
<text x="20" y="40" font-size="11" fill="#555">Pose at current q. See Equations panel for live numerical values.</text>
"##,
        n_moving_bodies, n_constraints, dof
    ));

    // ── 5. Ground hatching — draw under each ground attachment point ───────
    if let Some(ground) = bodies.get(GROUND_ID) {
        for (_name, pt_local) in &ground.attachment_points {
            let g = state.body_point_global(GROUND_ID, pt_local, q);
            let (sx, sy) = to_svg(g.x, g.y);
            // Short ground line below the pivot, with hatch tick-marks.
            let line_half_w = 18.0_f64;
            let y_below = sy + 8.0;
            svg.push_str(&format!(
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#333" stroke-width="1.6"/>
"##,
                sx - line_half_w, y_below, sx + line_half_w, y_below
            ));
            for i in 0..6 {
                let hx = sx - line_half_w + (i as f64) * (2.0 * line_half_w / 5.0);
                svg.push_str(&format!(
                    r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#333" stroke-width="1.0"/>
"##,
                    hx, y_below, hx - 4.0, y_below + 5.0
                ));
            }
        }
    }

    // ── 6. Bars (one closed polygon per non-ground body) ───────────────────
    for (body_id, body) in bodies.iter() {
        if body_id == GROUND_ID {
            continue;
        }
        // Sort attachment-point names for deterministic ordering.
        let mut names: Vec<&String> = body.attachment_points.keys().collect();
        names.sort();
        let pts: Vec<(f64, f64)> = names
            .iter()
            .map(|n| {
                let local = &body.attachment_points[*n];
                let g = state.body_point_global(body_id, local, q);
                to_svg(g.x, g.y)
            })
            .collect();

        if pts.len() < 2 {
            continue;
        }

        // Draw a thick line segment between every consecutive pair, then close
        // the loop for ternary+ plates. This matches the doc figures' bar style.
        for pair in pts.windows(2) {
            svg.push_str(&format!(
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#444" stroke-width="4" stroke-linecap="round"/>
"##,
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            ));
        }
        if pts.len() >= 3 {
            let first = pts[0];
            let last = *pts.last().unwrap();
            svg.push_str(&format!(
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#444" stroke-width="4" stroke-linecap="round"/>
"##,
                last.0, last.1, first.0, first.1
            ));
        }
    }

    // ── 7. Joint clusters: group by world position to handle coincident joints ─
    //
    // Each cluster has one circle and one combined label. Ground attachment
    // points without joints (rare; shouldn't normally happen) won't be drawn
    // as joints — only joint-pivots are circled.
    struct Cluster {
        sx: f64,
        sy: f64,
        labels: Vec<String>,
        kinds: Vec<JointKindForRender>,
        is_ground_pivot: bool,
    }
    #[derive(Copy, Clone, PartialEq, Eq)]
    enum JointKindForRender {
        Revolute,
        Prismatic,
        Fixed,
        CamFollower,
    }

    let mut clusters: Vec<Cluster> = Vec::new();
    let cluster_key = |x: f64, y: f64| -> (i64, i64) {
        let inv_tol = 1.0 / POSITION_TOLERANCE;
        ((x * inv_tol).round() as i64, (y * inv_tol).round() as i64)
    };
    // Map (x_key, y_key) → cluster index.
    let mut cluster_idx: std::collections::HashMap<(i64, i64), usize> =
        std::collections::HashMap::new();

    for joint in mech.joints() {
        let g = state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
        let (sx, sy) = to_svg(g.x, g.y);
        let key = cluster_key(g.x, g.y);
        let kind = match joint {
            JointConstraint::Revolute(_) => JointKindForRender::Revolute,
            JointConstraint::Prismatic(_) => JointKindForRender::Prismatic,
            JointConstraint::Fixed(_) => JointKindForRender::Fixed,
            JointConstraint::CamFollower(_) => JointKindForRender::CamFollower,
        };
        let is_ground = joint.body_i_id() == GROUND_ID || joint.body_j_id() == GROUND_ID;
        if let Some(&idx) = cluster_idx.get(&key) {
            clusters[idx].labels.push(joint.id().to_string());
            clusters[idx].kinds.push(kind);
            clusters[idx].is_ground_pivot |= is_ground;
        } else {
            cluster_idx.insert(key, clusters.len());
            clusters.push(Cluster {
                sx,
                sy,
                labels: vec![joint.id().to_string()],
                kinds: vec![kind],
                is_ground_pivot: is_ground,
            });
        }
    }

    // Render clusters.
    for c in &clusters {
        // Pick shape based on the dominant kind (first one in the cluster).
        let dominant = c.kinds[0];
        match dominant {
            JointKindForRender::Revolute | JointKindForRender::CamFollower => {
                svg.push_str(&format!(
                    r##"<circle cx="{:.2}" cy="{:.2}" r="7" fill="white" stroke="#222" stroke-width="2"/>
"##,
                    c.sx, c.sy
                ));
            }
            JointKindForRender::Prismatic => {
                svg.push_str(&format!(
                    r##"<rect x="{:.2}" y="{:.2}" width="14" height="10" fill="white" stroke="#222" stroke-width="2"/>
"##,
                    c.sx - 7.0,
                    c.sy - 5.0
                ));
            }
            JointKindForRender::Fixed => {
                svg.push_str(&format!(
                    r##"<rect x="{:.2}" y="{:.2}" width="12" height="12" fill="#222" stroke="#222" stroke-width="2"/>
"##,
                    c.sx - 6.0,
                    c.sy - 6.0
                ));
            }
        }

        // Combined label: "J1,J4" if multiple joints share this position.
        let combined = c.labels.join(",");
        // For ground pivots place the label below the circle to leave room for
        // the ground hatching above; otherwise place above-and-left.
        let (lx, ly) = if c.is_ground_pivot {
            (c.sx - 8.0, c.sy + 28.0)
        } else {
            (c.sx + 8.0, c.sy - 8.0)
        };
        svg.push_str(&format!(
            r#"<text x="{:.2}" y="{:.2}" font-size="11" font-weight="bold">{}</text>
"#,
            lx,
            ly,
            xml_escape(&combined)
        ));
    }

    // ── 8. Body labels (italic gray) at the centroid of each non-ground body ──
    for (body_id, body) in bodies.iter() {
        if body_id == GROUND_ID {
            continue;
        }
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut n = 0;
        for (_name, pt_local) in &body.attachment_points {
            let g = state.body_point_global(body_id, pt_local, q);
            let (sx, sy) = to_svg(g.x, g.y);
            sum_x += sx;
            sum_y += sy;
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let cx = sum_x / n as f64;
        let cy = sum_y / n as f64;
        // Offset slightly above the centroid so the label doesn't sit on top
        // of the bar line.
        svg.push_str(&format!(
            r##"<text x="{:.2}" y="{:.2}" font-size="12" font-style="italic" fill="#555" text-anchor="middle">{}</text>
"##,
            cx,
            cy - 10.0,
            xml_escape(body_id)
        ));
    }
    // Ground label (only if we have a ground body with attachment points).
    if let Some(ground) = bodies.get(GROUND_ID) {
        if let Some(first) = ground.attachment_points.values().next() {
            let g = state.body_point_global(GROUND_ID, first, q);
            let (_sx, sy) = to_svg(g.x, g.y);
            // Place near bottom of the drawing region, centred horizontally.
            svg.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-size="12" font-style="italic" fill="#555" text-anchor="middle">{}</text>
"##,
                0.5 * (DRAW_X_MIN + DRAW_X_MAX),
                sy + 30.0,
                "ground"
            ));
        }
    }

    // ── 9. Driver indicator (red) ──────────────────────────────────────────
    if let Some(drv) = mech.drivers().first() {
        // Find the joint that connects the driver's two bodies.
        let mut drv_joint_pos: Option<(f64, f64)> = None;
        for joint in mech.joints() {
            let bi = joint.body_i_id();
            let bj = joint.body_j_id();
            let pair = [bi, bj];
            if (pair.contains(&drv.body_i_id()) && pair.contains(&drv.body_j_id()))
                && joint.is_revolute()
            {
                let g = state.body_point_global(bi, &joint.point_i_local(), q);
                drv_joint_pos = Some(to_svg(g.x, g.y));
                break;
            }
        }
        if let Some((dx, dy)) = drv_joint_pos {
            // Curved arrow above the joint.
            let r = 18.0;
            svg.push_str(&format!(
                r##"<path d="M {:.2} {:.2} A {:.2} {:.2} 0 1 1 {:.2} {:.2}" stroke="#a02020" stroke-width="2" fill="none" marker-end="url(#ah_drv)"/>
"##,
                dx - r,
                dy,
                r,
                r,
                dx + r,
                dy
            ));
            let label = format!("{}: \u{03c9}", drv.id());
            svg.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-size="13" fill="#a02020" font-weight="bold">{}</text>
"##,
                dx + r,
                dy - r - 4.0,
                xml_escape(&label)
            ));
        }
    } else if let Some(drv) = mech.linear_drivers().first() {
        // Linear driver: draw colored line between P_a and P_b.
        let pa = drv.point_a();
        let pb = drv.point_b();
        let g_a = state.body_point_global(
            drv.body_i_id(),
            &nalgebra::Vector2::new(pa[0], pa[1]),
            q,
        );
        let g_b = state.body_point_global(
            drv.body_j_id(),
            &nalgebra::Vector2::new(pb[0], pb[1]),
            q,
        );
        let (ax, ay) = to_svg(g_a.x, g_a.y);
        let (bx, by) = to_svg(g_b.x, g_b.y);
        // Cylinder (thick) for first 60 % of the actuator, piston rod (thin)
        // for the remaining 40 %, matching Figure 3.
        let mid_x = ax + 0.6 * (bx - ax);
        let mid_y = ay + 0.6 * (by - ay);
        svg.push_str(&format!(
            r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#a02020" stroke-width="6" stroke-linecap="butt"/>
"##,
            ax, ay, mid_x, mid_y
        ));
        svg.push_str(&format!(
            r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#a02020" stroke-width="2.5"/>
"##,
            mid_x, mid_y, bx, by
        ));
        svg.push_str(&format!(
            r##"<circle cx="{:.2}" cy="{:.2}" r="4.5" fill="#a02020"/>
<circle cx="{:.2}" cy="{:.2}" r="4.5" fill="#a02020"/>
"##,
            ax, ay, bx, by
        ));
        let label = format!("{}: L(t)", drv.id());
        // Place label near the middle of the actuator line, slightly offset.
        let label_x = 0.5 * (ax + bx) + 8.0;
        let label_y = 0.5 * (ay + by);
        svg.push_str(&format!(
            r##"<text x="{:.2}" y="{:.2}" font-size="12" font-style="italic" fill="#a02020">{}</text>
"##,
            label_x,
            label_y,
            xml_escape(&label)
        ));
    }

    // ── 9b. LinearActuator force elements (orange, distinct from drivers)
    //
    // Force elements aren't constraints (don't appear in Φ or contribute to
    // m), but a LinearActuator is visually a hydraulic cylinder + piston
    // between two body points, same shape as a LinearDriver. Draw it in
    // orange so the reader doesn't confuse it with a driver constraint;
    // include the rated force in the label.
    use crate::forces::elements::ForceElement;
    for fe in mech.forces() {
        if let ForceElement::LinearActuator(act) = fe {
            let local_a = nalgebra::Vector2::new(act.point_a[0], act.point_a[1]);
            let local_b = nalgebra::Vector2::new(act.point_b[0], act.point_b[1]);
            let g_a = state.body_point_global(&act.body_a, &local_a, q);
            let g_b = state.body_point_global(&act.body_b, &local_b, q);
            let (ax, ay) = to_svg(g_a.x, g_a.y);
            let (bx, by) = to_svg(g_b.x, g_b.y);
            // Same cylinder/piston visual as the LinearDriver, in orange.
            let mid_x = ax + 0.6 * (bx - ax);
            let mid_y = ay + 0.6 * (by - ay);
            svg.push_str(&format!(
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#e08020" stroke-width="6" stroke-linecap="butt"/>
"##,
                ax, ay, mid_x, mid_y
            ));
            svg.push_str(&format!(
                r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#e08020" stroke-width="2.5"/>
"##,
                mid_x, mid_y, bx, by
            ));
            svg.push_str(&format!(
                r##"<circle cx="{:.2}" cy="{:.2}" r="4.5" fill="#e08020"/>
<circle cx="{:.2}" cy="{:.2}" r="4.5" fill="#e08020"/>
"##,
                ax, ay, bx, by
            ));
            let label = format!("F = {:.1} N", act.force);
            let label_x = 0.5 * (ax + bx) + 8.0;
            let label_y = 0.5 * (ay + by) + 14.0;
            svg.push_str(&format!(
                r##"<text x="{:.2}" y="{:.2}" font-size="12" font-style="italic" fill="#e08020">{}</text>
"##,
                label_x,
                label_y,
                xml_escape(&label)
            ));
        }
    }

    // ── 10. Constraint legend (below the drawing area) ───────────────────
    //
    // Legend used to live in a sidebar at x=475..595 but constraint rows
    // ran off the bottom on mechanisms with many joints / drivers. Now
    // it stacks below the drawing region with the full canvas width.
    let legend_x = LEGEND_X_START;
    let legend_header_y = LEGEND_Y_START;
    // A faint horizontal divider between the drawing area and the legend.
    // (Raw string uses `r##` because the SVG attribute contains `"#ccc"`,
    // which would prematurely close a single-# raw string.)
    svg.push_str(&format!(
        r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="#ccc" stroke-width="0.8"/>
"##,
        20.0,
        legend_header_y - 12.0,
        SVG_WIDTH - 20.0,
        legend_header_y - 12.0,
    ));
    svg.push_str(&format!(
        r#"<text x="{:.2}" y="{:.2}" font-size="13" font-weight="bold">Constraint rows</text>
"#,
        legend_x, legend_header_y,
    ));

    // Group joints by kind.
    let mut joint_groups: Vec<(&'static str, Vec<String>, usize)> = Vec::new();
    for &kind_label in &["revolute", "prismatic", "fixed", "cam-follower"] {
        let mut ids = Vec::new();
        let mut total_rows = 0;
        for joint in mech.joints() {
            let matches = match joint {
                JointConstraint::Revolute(_) => kind_label == "revolute",
                JointConstraint::Prismatic(_) => kind_label == "prismatic",
                JointConstraint::Fixed(_) => kind_label == "fixed",
                JointConstraint::CamFollower(_) => kind_label == "cam-follower",
            };
            if matches {
                ids.push(joint.id().to_string());
                total_rows += joint.n_equations();
            }
        }
        if !ids.is_empty() {
            joint_groups.push((kind_label, ids, total_rows));
        }
    }

    let mut y = legend_header_y + 22.0;
    for (label, ids, total_rows) in &joint_groups {
        let id_list = if ids.len() <= 4 {
            ids.join(",")
        } else {
            format!("{},...,{}", ids[0], ids[ids.len() - 1])
        };
        let line = format!(
            "{} ({}): {} rows",
            id_list, label, total_rows
        );
        svg.push_str(&format!(
            r#"<text x="{:.2}" y="{:.2}" font-size="11">{}</text>
"#,
            legend_x,
            y,
            xml_escape(&line)
        ));
        y += 18.0;
    }

    // Driver row(s) — render the symbolic form via shared eq_rendering helper
    // so the legend stays consistent with the Equations panel.
    let n_joints = mech.joints().len();
    for (i, drv) in mech.drivers().iter().enumerate() {
        let idx = n_joints + i;
        let symbolic = eq_rendering::symbolic_form(mech, idx);
        // Trim to a short driver-row form: use just "{id}: θ_j − θ_i − f(t) = 0"
        let line = format!(
            "{}: {}",
            drv.id(),
            short_driver_form(&symbolic, true)
        );
        svg.push_str(&format!(
            r##"<text x="{:.2}" y="{:.2}" font-size="11" fill="#a02020">{}</text>
"##,
            legend_x,
            y,
            xml_escape(&line)
        ));
        y += 18.0;
    }
    for (i, drv) in mech.linear_drivers().iter().enumerate() {
        let idx = n_joints + mech.drivers().len() + i;
        let symbolic = eq_rendering::symbolic_form(mech, idx);
        let line = format!(
            "{}: {}",
            drv.id(),
            short_driver_form(&symbolic, false)
        );
        svg.push_str(&format!(
            r##"<text x="{:.2}" y="{:.2}" font-size="11" fill="#a02020">{}</text>
"##,
            legend_x,
            y,
            xml_escape(&line)
        ));
        y += 18.0;
    }

    // Footer: totals.
    y += 6.0;
    svg.push_str(&format!(
        r#"<text x="{:.2}" y="{:.2}" font-size="11" font-style="italic">Total m = {} ; n = {} ; DOF = {}</text>
"#,
        legend_x, y, n_constraints, n_coords, dof
    ));
    y += 18.0;
    svg.push_str(&format!(
        r##"<text x="{:.2}" y="{:.2}" font-size="11" font-style="italic" fill="#666">λ: pin reactions [N], driver torque/force</text>
"##,
        legend_x, y
    ));

    svg.push_str("</svg>\n");
    Ok(svg)
}

/// Trim the multi-line full symbolic form down to just the right-hand portion
/// suitable for a one-line legend entry (e.g. `θ_j − θ_i − (θ₀+ωt) = 0`).
fn short_driver_form(symbolic: &str, is_revolute: bool) -> String {
    // `symbolic_form` returns strings like "Φ_rd: θ_j − θ_i − (...) = 0  (1 eq)".
    // Strip the prefix and trailing "(1 eq)" annotation.
    let s = symbolic
        .trim_start_matches("Φ_rd: ")
        .trim_start_matches("Φ_ld: ");
    // Drop trailing "  (1 eq)" if present.
    let trimmed = s.split("  (").next().unwrap_or(s).trim();
    if trimmed.is_empty() {
        if is_revolute {
            "\u{03b8}_j \u{2212} \u{03b8}_i \u{2212} f(t) = 0".to_string()
        } else {
            "\u{2016}P_b\u{2212}P_a\u{2016} \u{2212} L(t) = 0".to_string()
        }
    } else {
        trimmed.to_string()
    }
}

/// Minimal XML-escape for text node contents. Body and joint IDs in this
/// codebase are typically alphanumeric, but escape defensively in case a user
/// chooses an ID with `<`, `>`, `&`, etc.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use crate::core::mechanism::Mechanism;
    use crate::solver::kinematics::solve_position;
    use std::f64::consts::PI;

    fn build_fourbar_with_q() -> (Mechanism, DVector<f64>) {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0).unwrap();
        mech.build().unwrap();

        let mut q0 = mech.state().make_q();
        mech.state().set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        mech.state().set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        mech.state().set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let res = solve_position(&mech, &q0, 0.0, 1e-10, 50).unwrap();
        (mech, res.q)
    }

    #[test]
    fn schematic_svg_for_fourbar_contains_essentials() {
        let (mech, q) = build_fourbar_with_q();
        let svg = generate_schematic_svg(&mech, &q).unwrap();
        // Must contain SVG document framing.
        assert!(
            svg.contains("<svg"),
            "missing <svg element: first 100 chars = {:?}",
            &svg[..svg.len().min(100)]
        );
        assert!(svg.contains("</svg>"));
        // Must label every joint.
        for j in &["J1", "J2", "J3", "J4"] {
            assert!(svg.contains(j), "missing joint label {}", j);
        }
        // Must label every non-ground body.
        for b in &["crank", "coupler", "rocker"] {
            assert!(svg.contains(b), "missing body label {}", b);
        }
        // Must indicate driver.
        assert!(
            svg.contains("D1") || svg.contains("driver"),
            "missing driver indicator"
        );
        // Must show DOF in the title strip + footer.
        assert!(svg.contains("DOF"), "missing DOF in legend");
    }

    #[test]
    fn schematic_svg_well_formed() {
        let (mech, q) = build_fourbar_with_q();
        let svg = generate_schematic_svg(&mech, &q).unwrap();
        // Roughly well-formed: every <line opens with a corresponding close.
        // We use self-closed <line .../> tags, so the count of `<line ` must
        // equal the count of `/>` for those lines (matched against simple
        // self-close marker).
        let n_open = svg.matches("<line ").count();
        // Each <line has its own self-close /> on the same line.
        let n_self_close = svg.matches("/>").count();
        assert!(
            n_self_close >= n_open,
            "more <line tags ({}) than self-close markers ({})",
            n_open,
            n_self_close
        );
    }

    #[test]
    fn schematic_svg_for_sample_parallelogram_actuator() {
        // Smoke-test against a sample mechanism that has a linear actuator
        // force element. This exercises the full sample build path in addition
        // to the schematic generator.
        use crate::gui::samples::{build_sample, SampleMechanism};
        let (mech, q0) = build_sample(SampleMechanism::ParallelogramActuator);
        let res = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver ok");
        let svg = generate_schematic_svg(&mech, &res.q).expect("svg ok");
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        // Sample uses J1..J4 + D1.
        assert!(svg.contains("DOF"));
        assert!(svg.contains("D1"));
        // The ParallelogramActuator sample has a LinearActuator force element.
        // Verify the schematic now renders it: orange stroke (#e08020) for the
        // cylinder/piston, and an "F = ... N" label.
        assert!(
            svg.contains("#e08020"),
            "schematic should render LinearActuator force elements in orange"
        );
        assert!(
            svg.contains("F ="),
            "schematic should label LinearActuator with its rated force"
        );
    }

    #[test]
    fn schematic_svg_for_linear_driver_mechanism() {
        // Build a minimal 4-bar where the driver is replaced by a constant-
        // velocity LinearDriver between ground and the coupler. This covers
        // the linear-driver branch of the driver-indicator rendering.
        use crate::core::linear_driver::constant_velocity_linear_driver;

        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

        // LinearDriver between ground origin and the coupler's "B" point.
        // Initial length matches ‖(0,0) - (0.01,0)‖ = 0.01 m so the initial
        // pose satisfies the driver constraint.
        mech.add_linear_driver(constant_velocity_linear_driver(
            "D1",
            "ground",
            [0.0, 0.0],
            "coupler",
            [0.0, 0.0],
            0.0,
            0.01,
        ))
        .unwrap();
        mech.build().unwrap();

        let mut q0 = mech.state().make_q();
        mech.state().set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
        mech.state().set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
        mech.state().set_pose("rocker", &mut q0, 0.04, 0.005, 0.5);
        let res = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver ok");
        let svg = generate_schematic_svg(&mech, &res.q).expect("svg ok");

        // Linear driver indicator: should render two colored circles + label.
        assert!(svg.contains("D1"), "missing linear driver label");
        assert!(svg.contains("L(t)"), "missing linear driver L(t) text");
        // Sidebar should mention `‖P_b−P_a‖` (linear driver symbolic form).
        assert!(
            svg.contains("P_b") || svg.contains("\u{2016}"),
            "linear driver symbolic form missing from legend"
        );
    }

    #[test]
    fn schematic_svg_empty_mechanism_returns_error() {
        let mut mech = Mechanism::new();
        mech.build().unwrap();
        let q = DVector::zeros(0);
        let result = generate_schematic_svg(&mech, &q);
        assert!(result.is_err(), "empty mechanism should return an error");
    }
}
