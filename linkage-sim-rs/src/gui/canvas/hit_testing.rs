//! Hit testing types and geometry helpers for canvas interaction.

use eframe::egui::Pos2;

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
