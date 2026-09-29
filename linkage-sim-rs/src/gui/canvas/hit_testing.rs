//! Hit testing types and geometry helpers for canvas interaction.

use eframe::egui::Pos2;

use crate::core::state::GROUND_ID;
use crate::gui::state::AppState;
use crate::io::schema::is_blank_point_mass_id;

/// An attachment point hit target: screen + world position, body ID, point name.
#[derive(Clone)]
pub struct AttachmentHit {
    pub screen_pos: Pos2,
    /// World coordinates of this attachment point (for precise snapping).
    pub world_pos: [f64; 2],
    pub body_id: String,
    pub point_name: String,
}

/// Result of a body-segment hit test.
#[allow(dead_code)]
pub struct SegmentHit {
    pub body_id: String,
    pub world_pos: [f64; 2],
    pub screen_pos: Pos2,
    pub point_a_name: String,
    pub point_b_name: String,
}

/// Collected body segment for hit testing.
pub struct BodySegment {
    pub screen_a: Pos2,
    pub screen_b: Pos2,
    pub world_a: [f64; 2],
    pub world_b: [f64; 2],
    pub body_id: String,
    pub point_a_name: String,
    pub point_b_name: String,
}

/// Project a point onto a line segment. Returns projected point and distance,
/// or None if projection falls outside the segment.
pub fn project_onto_segment(point: Pos2, seg_a: Pos2, seg_b: Pos2) -> Option<(Pos2, f32)> {
    let ab = seg_b - seg_a;
    let ap = point - seg_a;
    let len_sq = ab.length_sq();
    if len_sq < 1e-10 {
        return None;
    }
    let t = ab.dot(ap) / len_sq;
    if t < 0.0 || t > 1.0 {
        return None;
    }
    let proj = seg_a + ab * t;
    let dist = point.distance(proj);
    Some((proj, dist))
}

/// Find the nearest body line segment to a screen point.
pub fn find_nearest_body_segment(
    point: Pos2,
    segments: &[BodySegment],
    max_distance: f32,
) -> Option<SegmentHit> {
    let mut best: Option<(f32, Pos2, [f64; 2], String, String, String)> = None;

    for seg in segments {
        if let Some((proj_screen, dist)) = project_onto_segment(point, seg.screen_a, seg.screen_b) {
            if dist <= max_distance {
                if best.as_ref().map_or(true, |(d, _, _, _, _, _)| dist < *d) {
                    let ab_screen = seg.screen_b - seg.screen_a;
                    let ap_screen = proj_screen - seg.screen_a;
                    let t = if ab_screen.length_sq() > 1e-10 {
                        ap_screen.length() / ab_screen.length()
                    } else {
                        0.0
                    };
                    let world_x = seg.world_a[0] + t as f64 * (seg.world_b[0] - seg.world_a[0]);
                    let world_y = seg.world_a[1] + t as f64 * (seg.world_b[1] - seg.world_a[1]);

                    best = Some((dist, proj_screen, [world_x, world_y], seg.body_id.clone(),
                                 seg.point_a_name.clone(), seg.point_b_name.clone()));
                }
            }
        }
    }

    best.map(|(_, screen_pos, world_pos, body_id, point_a_name, point_b_name)| SegmentHit {
        body_id, world_pos, screen_pos, point_a_name, point_b_name,
    })
}

/// Screen position of the weight (point mass) at body-local `local_pos` on
/// `body_id`, at the current pose and view. `None` when it is not finite (a
/// non-finite position, which the loader skips). Drawing and hit testing both
/// use this, so a weight is picked exactly where it is drawn.
pub fn point_mass_screen_pos(state: &AppState, body_id: &str, local_pos: [f64; 2]) -> Option<Pos2> {
    let [wx, wy] = state.body_local_to_world(body_id, local_pos);
    let [sx, sy] = state.view.world_to_screen(wx, wy);
    (sx.is_finite() && sy.is_finite()).then(|| Pos2::new(sx, sy))
}

/// The weight under `screen_pos`: `(body_id, weight_id)` of the weight whose
/// marker centre is nearest, within `radius_px` (inclusive).
///
/// Only weights the canvas draws and can address are candidates: weights on
/// ground are not drawn (the loader skips them) and blank ids are not
/// addressable. On a tie the first weight wins, with bodies sorted by id and
/// weights in list order, so the result never depends on HashMap order.
pub fn find_point_mass_at(state: &AppState, screen_pos: Pos2, radius_px: f32) -> Option<(String, String)> {
    let bp = state.blueprint.as_ref()?;
    let mut body_ids: Vec<&String> = bp.bodies.keys().filter(|id| id.as_str() != GROUND_ID).collect();
    body_ids.sort();

    let mut best: Option<(f32, &str, &str)> = None;
    for body_id in body_ids {
        for pm in &bp.bodies[body_id].point_masses {
            if is_blank_point_mass_id(&pm.id) {
                continue;
            }
            let Some(center) = point_mass_screen_pos(state, body_id, pm.local_pos) else { continue };
            let dist = screen_pos.distance(center);
            if dist <= radius_px && best.is_none_or(|(d, _, _)| dist < d) {
                best = Some((dist, body_id, &pm.id));
            }
        }
    }
    best.map(|(_, body_id, weight_id)| (body_id.to_string(), weight_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::vec2;

    use crate::gui::samples::SampleMechanism;
    use crate::io::PointMassJson;

    /// Four-bar with weight W1 (2 kg) on the coupler, in the default view
    /// (5000 px/m, so 1 mm is 5 px).
    fn fourbar_with_w1() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.add_point_mass("coupler", 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        state
    }

    fn weight_screen(state: &AppState, body: &str, id: &str) -> Pos2 {
        let pm = state.find_point_mass(body, id).expect("weight exists");
        point_mass_screen_pos(state, body, pm.local_pos).expect("finite position")
    }

    fn hit(body: &str, id: &str) -> Option<(String, String)> {
        Some((body.to_string(), id.to_string()))
    }

    #[test]
    fn point_mass_screen_pos_follows_the_body_pose_and_the_view() {
        let state = fourbar_with_w1();
        let [wx, wy] = state.body_local_to_world("coupler", [0.03, 0.02]);
        let [sx, sy] = state.view.world_to_screen(wx, wy);
        assert_eq!(point_mass_screen_pos(&state, "coupler", [0.03, 0.02]), Some(Pos2::new(sx, sy)));
    }

    #[test]
    fn point_mass_screen_pos_is_none_for_a_non_finite_position() {
        let state = fourbar_with_w1();
        assert_eq!(point_mass_screen_pos(&state, "coupler", [f64::NAN, 0.0]), None);
        assert_eq!(point_mass_screen_pos(&state, "coupler", [0.0, f64::INFINITY]), None);
    }

    #[test]
    fn find_point_mass_at_hits_the_weight_under_the_cursor() {
        let state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        assert_eq!(find_point_mass_at(&state, at, 8.0), hit("coupler", "W1"));
        assert_eq!(find_point_mass_at(&state, at + vec2(3.0, 4.0), 8.0), hit("coupler", "W1"), "5 px off");
        assert_eq!(find_point_mass_at(&state, at + vec2(-7.5, 0.0), 8.0), hit("coupler", "W1"), "7.5 px off");
    }

    #[test]
    fn find_point_mass_at_misses_empty_space() {
        let state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        assert_eq!(find_point_mass_at(&state, at + vec2(8.5, 0.0), 8.0), None, "just outside the radius");
        assert_eq!(find_point_mass_at(&state, at + vec2(0.0, 200.0), 8.0), None, "far away");
        assert_eq!(find_point_mass_at(&state, at, 0.0), hit("coupler", "W1"), "a zero radius still hits the centre");
    }

    #[test]
    fn find_point_mass_at_prefers_the_nearest_weight() {
        let mut state = fourbar_with_w1();
        // 2 mm along the coupler from W1: 10 px at 5000 px/m.
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.032, 0.02]).as_deref(), Some("W2"));
        let w1 = weight_screen(&state, "coupler", "W1");
        let w2 = weight_screen(&state, "coupler", "W2");
        assert!((w1.distance(w2) - 10.0).abs() < 0.01, "fixture spacing {}", w1.distance(w2));
        assert_eq!(find_point_mass_at(&state, w2, 12.0), hit("coupler", "W2"));
        assert_eq!(find_point_mass_at(&state, w1, 12.0), hit("coupler", "W1"));
        assert_eq!(find_point_mass_at(&state, w1 + (w2 - w1) * 0.7, 12.0), hit("coupler", "W2"));
    }

    #[test]
    fn find_point_mass_at_breaks_ties_by_sorted_body_then_list_order() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Added rocker first, so insertion order cannot explain the result.
        assert_eq!(state.add_point_mass("rocker", 1.0, [0.01, 0.005]).as_deref(), Some("W1"));
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.01, 0.005]).as_deref(), Some("W2"));
        assert_eq!(state.add_point_mass("coupler", 1.0, [0.01, 0.005]).as_deref(), Some("W3"));
        // Pose both links at the world origin so the three weights overlap exactly.
        let starts: Vec<usize> = ["coupler", "rocker"]
            .iter()
            .map(|b| state.mechanism.as_ref().unwrap().state().get_index(b).unwrap().q_start)
            .collect();
        for q0 in starts {
            for k in 0..3 {
                state.q[q0 + k] = 0.0;
            }
        }
        let at = weight_screen(&state, "rocker", "W1");
        assert_eq!(weight_screen(&state, "coupler", "W2"), at, "fixture: exact overlap");
        assert_eq!(find_point_mass_at(&state, at, 8.0), hit("coupler", "W2"));
    }

    #[test]
    fn find_point_mass_at_skips_ground_blank_ids_and_non_finite_positions() {
        let mut state = fourbar_with_w1();
        let local = state.find_point_mass("coupler", "W1").unwrap().local_pos;
        let at = weight_screen(&state, "coupler", "W1");
        let world = state.body_local_to_world("coupler", local);
        assert!(state.remove_point_mass_by_id("coupler", "W1"));
        // Weights the canvas does not draw or cannot address, all at W1's old spot.
        let weight = |id: &str, local_pos: [f64; 2]| PointMassJson { id: id.to_string(), label: None, mass: 1.0, local_pos };
        let bp = state.blueprint.as_mut().unwrap();
        bp.bodies.get_mut("ground").unwrap().point_masses.push(weight("G1", world));
        let coupler = bp.bodies.get_mut("coupler").unwrap();
        coupler.point_masses.push(weight("", local));
        coupler.point_masses.push(weight("  ", local));
        coupler.point_masses.push(weight("W9", [f64::NAN, local[1]]));
        assert_eq!(find_point_mass_at(&state, at, 50.0), None);
    }

    #[test]
    fn find_point_mass_at_without_a_blueprint_is_none() {
        let mut state = fourbar_with_w1();
        let at = weight_screen(&state, "coupler", "W1");
        state.blueprint = None;
        assert_eq!(find_point_mass_at(&state, at, 8.0), None);
    }
}
