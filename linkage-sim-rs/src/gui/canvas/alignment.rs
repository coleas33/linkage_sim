//! Alignment guide detection and snapping for canvas drag operations.
//!
//! When dragging a point, checks alignment against all other attachment points
//! in the mechanism and produces guide lines + snapped coordinates.

use crate::gui::state::{AlignmentAxis, AlignmentGuide, AppState};

/// Compute alignment guides and snap the dragged position.
///
/// Compares `(wx, wy)` against all attachment points in the blueprint
/// (excluding the point being dragged, identified by `exclude_name`).
/// Returns the snapped `(x, y)` and populates `guides_out`.
///
/// `snap_threshold` is in world units -- points within this distance
/// on a single axis are considered aligned.
pub fn compute_alignment_guides(
    state: &AppState,
    wx: f64,
    wy: f64,
    exclude_name: Option<&str>,
    snap_threshold: f64,
    guides_out: &mut Vec<AlignmentGuide>,
) -> (f64, f64) {
    let mut snapped_x = wx;
    let mut snapped_y = wy;

    // Track the closest alignment distance per axis so we snap to the
    // nearest aligned point, not just the first one found.
    let mut best_dx = snap_threshold;
    let mut best_dy = snap_threshold;

    if state.blueprint.is_none() {
        return (snapped_x, snapped_y);
    }
    let mech = match &state.mechanism {
        Some(m) => m,
        None => return (snapped_x, snapped_y),
    };

    let mech_state = mech.state();
    let q = &state.q;

    for (body_id, body) in mech.bodies().iter() {
        for (pt_name, local_pt) in &body.attachment_points {
            // Skip the point being dragged.
            if let Some(exclude) = exclude_name {
                if pt_name == exclude && body_id == "ground" {
                    continue;
                }
            }

            let global = mech_state.body_point_global(body_id, local_pt, q);
            let px = global.x;
            let py = global.y;

            let label = format!("{}/{}", body_id, pt_name);

            // Vertical alignment: same x-value.
            let dx = (wx - px).abs();
            if dx < best_dx {
                best_dx = dx;
                snapped_x = px;
                // Remove any previous vertical guide (we only keep the closest).
                guides_out.retain(|g| g.axis != AlignmentAxis::Vertical);
                guides_out.push(AlignmentGuide {
                    axis: AlignmentAxis::Vertical,
                    world_value: px,
                    label: label.clone(),
                });
            }

            // Horizontal alignment: same y-value.
            let dy = (wy - py).abs();
            if dy < best_dy {
                best_dy = dy;
                snapped_y = py;
                // Remove any previous horizontal guide (we only keep the closest).
                guides_out.retain(|g| g.axis != AlignmentAxis::Horizontal);
                guides_out.push(AlignmentGuide {
                    axis: AlignmentAxis::Horizontal,
                    world_value: py,
                    label: label.clone(),
                });
            }
        }
    }

    (snapped_x, snapped_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::AlignmentAxis;

    /// Helper: load a FourBar and return a ready-to-query AppState.
    fn fourbar_state() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state
    }

    #[test]
    fn no_guides_when_far_from_any_point() {
        let state = fourbar_state();
        let mut guides = Vec::new();
        let threshold = 0.001; // 1mm
        // Place cursor far from any attachment point.
        let (sx, sy) = compute_alignment_guides(&state, 999.0, 999.0, None, threshold, &mut guides);
        assert!(guides.is_empty(), "Expected no guides at 999,999");
        assert!((sx - 999.0).abs() < 1e-12);
        assert!((sy - 999.0).abs() < 1e-12);
    }

    #[test]
    fn horizontal_guide_when_y_aligned() {
        let state = fourbar_state();
        let mech = state.mechanism.as_ref().unwrap();
        let q = &state.q;

        // Find a ground attachment point to align with.
        let bodies = mech.bodies();
        let ground = &bodies["ground"];
        let (_pt_name, local_pt) = ground.attachment_points.iter().next().unwrap();
        let global = mech.state().body_point_global("ground", local_pt, q);

        // Place cursor at same y but different x.
        let mut guides = Vec::new();
        let threshold = 0.01;
        let (_, sy) = compute_alignment_guides(
            &state, global.x + 0.5, global.y + 0.001, None, threshold, &mut guides,
        );

        let h_guides: Vec<_> = guides.iter().filter(|g| g.axis == AlignmentAxis::Horizontal).collect();
        assert!(!h_guides.is_empty(), "Expected a horizontal guide when y is within threshold");
        assert!((sy - global.y).abs() < 1e-12, "y should snap to the aligned point");
    }

    #[test]
    fn vertical_guide_when_x_aligned() {
        let state = fourbar_state();
        let mech = state.mechanism.as_ref().unwrap();
        let q = &state.q;

        let bodies = mech.bodies();
        let ground = &bodies["ground"];
        let (_pt_name, local_pt) = ground.attachment_points.iter().next().unwrap();
        let global = mech.state().body_point_global("ground", local_pt, q);

        // Place cursor at same x but different y.
        let mut guides = Vec::new();
        let threshold = 0.01;
        let (sx, _) = compute_alignment_guides(
            &state, global.x + 0.001, global.y + 0.5, None, threshold, &mut guides,
        );

        let v_guides: Vec<_> = guides.iter().filter(|g| g.axis == AlignmentAxis::Vertical).collect();
        assert!(!v_guides.is_empty(), "Expected a vertical guide when x is within threshold");
        assert!((sx - global.x).abs() < 1e-12, "x should snap to the aligned point");
    }

    #[test]
    fn excluded_point_is_not_snapped_to() {
        let state = fourbar_state();
        let mech = state.mechanism.as_ref().unwrap();
        let q = &state.q;

        let bodies = mech.bodies();
        let ground = &bodies["ground"];
        let (pt_name, local_pt) = ground.attachment_points.iter().next().unwrap();
        let global = mech.state().body_point_global("ground", local_pt, q);

        // Place cursor exactly at the point and exclude it -- should not snap.
        let mut guides = Vec::new();
        let threshold = 0.01;
        let (_sx, _sy) = compute_alignment_guides(
            &state, global.x, global.y, Some(pt_name), threshold, &mut guides,
        );

        // The excluded point itself should not generate a guide. Other points
        // that happen to be within threshold still can, so we check that if
        // guides were generated they don't reference the excluded point.
        for g in &guides {
            assert!(
                !g.label.contains(&format!("ground/{}", pt_name)),
                "Excluded point should not appear in guides"
            );
        }
    }

    #[test]
    fn both_axes_snap_simultaneously() {
        let state = fourbar_state();
        let mech = state.mechanism.as_ref().unwrap();
        let q = &state.q;

        // Collect all ground attachment points.
        let bodies = mech.bodies();
        let ground = &bodies["ground"];
        let pts: Vec<_> = ground.attachment_points.iter().collect();

        if pts.len() >= 2 {
            let (_, lp0) = pts[0];
            let (_, lp1) = pts[1];
            let g0 = mech.state().body_point_global("ground", lp0, q);
            let g1 = mech.state().body_point_global("ground", lp1, q);

            // Place cursor near x of g0 and y of g1.
            let mut guides = Vec::new();
            let threshold = 0.01;
            let (_sx, _sy) = compute_alignment_guides(
                &state, g0.x + 0.001, g1.y + 0.001, None, threshold, &mut guides,
            );

            let has_v = guides.iter().any(|g| g.axis == AlignmentAxis::Vertical);
            let has_h = guides.iter().any(|g| g.axis == AlignmentAxis::Horizontal);
            assert!(has_v, "Expected a vertical guide");
            assert!(has_h, "Expected a horizontal guide");
        }
    }

    #[test]
    fn no_mechanism_returns_unsnapped() {
        let mut state = AppState::default();
        // Default state has a mechanism (empty), so explicitly clear it.
        state.mechanism = None;
        let mut guides = Vec::new();
        let (sx, sy) = compute_alignment_guides(&state, 1.0, 2.0, None, 0.01, &mut guides);
        assert!(guides.is_empty());
        assert!((sx - 1.0).abs() < 1e-12);
        assert!((sy - 2.0).abs() < 1e-12);
    }

    #[test]
    fn only_closest_point_produces_guide() {
        let state = fourbar_state();
        let mech = state.mechanism.as_ref().unwrap();
        let q = &state.q;

        // Collect all attachment points across all bodies.
        let bodies = mech.bodies();
        let all_pts: Vec<(String, f64, f64)> = bodies.iter()
            .flat_map(|(bid, b)| {
                b.attachment_points.iter().map(move |(pn, lp)| {
                    let g = mech.state().body_point_global(bid, lp, q);
                    (format!("{}/{}", bid, pn), g.x, g.y)
                })
            })
            .collect();

        if all_pts.len() >= 2 {
            let mut guides = Vec::new();
            // Use a large threshold so multiple points could potentially match,
            // then verify we only get one guide per axis.
            let threshold = 10.0;
            compute_alignment_guides(&state, 0.0, 0.0, None, threshold, &mut guides);

            let v_count = guides.iter().filter(|g| g.axis == AlignmentAxis::Vertical).count();
            let h_count = guides.iter().filter(|g| g.axis == AlignmentAxis::Horizontal).count();
            assert!(v_count <= 1, "Should have at most 1 vertical guide, got {}", v_count);
            assert!(h_count <= 1, "Should have at most 1 horizontal guide, got {}", h_count);
        }
    }
}
