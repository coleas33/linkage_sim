//! Polygon geometry utilities for linkage body shapes and force zones.
//!
//! Provides pure-math functions for:
//! - Polygon area via the shoelace formula
//! - Polygon centroid computation
//! - Sutherland-Hodgman polygon clipping against an axis-aligned bounding box
//! - Transforming a body-local rectangle to world-frame corners

use nalgebra::Vector2;

/// Compute the area of a simple (non-self-intersecting) polygon using the
/// shoelace formula. Returns the absolute area regardless of vertex winding.
///
/// Returns `0.0` for fewer than 3 vertices.
pub fn polygon_area(vertices: &[Vector2<f64>]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }

    let mut twice_area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        // Shoelace: sum of (x_i * y_{i+1} - x_{i+1} * y_i)
        twice_area += vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
    }
    twice_area.abs() / 2.0
}

/// Compute the centroid of a simple polygon.
///
/// Uses the standard formula that weights each edge's midpoint contribution by
/// the cross product of consecutive vertices. Returns the origin for degenerate
/// cases (fewer than 3 vertices or zero area).
pub fn polygon_centroid(vertices: &[Vector2<f64>]) -> Vector2<f64> {
    let n = vertices.len();
    if n < 3 {
        return Vector2::zeros();
    }

    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut signed_area_2 = 0.0;

    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i].x * vertices[j].y - vertices[j].x * vertices[i].y;
        cx += (vertices[i].x + vertices[j].x) * cross;
        cy += (vertices[i].y + vertices[j].y) * cross;
        signed_area_2 += cross;
    }

    if signed_area_2.abs() < 1e-15 {
        return Vector2::zeros();
    }

    let inv = 1.0 / (3.0 * signed_area_2);
    Vector2::new(cx * inv, cy * inv)
}

/// Clip a convex polygon against an axis-aligned bounding box using the
/// Sutherland-Hodgman algorithm.
///
/// Clips sequentially against the four AABB edges (left, right, bottom, top).
/// Returns the vertices of the clipped polygon, or an empty vec if there is
/// no overlap.
pub fn clip_polygon_to_aabb(
    polygon: &[Vector2<f64>],
    aabb_min: &Vector2<f64>,
    aabb_max: &Vector2<f64>,
) -> Vec<Vector2<f64>> {
    if polygon.is_empty() {
        return Vec::new();
    }

    let mut output: Vec<Vector2<f64>> = polygon.to_vec();

    // Each clip edge is defined by an "inside" test and an intersection helper.
    // We clip against four half-planes in sequence: left, right, bottom, top.

    // Clip against left edge (x >= aabb_min.x)
    output = clip_against_edge(&output, |p| p.x >= aabb_min.x, |a, b| {
        let t = (aabb_min.x - a.x) / (b.x - a.x);
        a + t * (b - a)
    });
    if output.is_empty() {
        return output;
    }

    // Clip against right edge (x <= aabb_max.x)
    output = clip_against_edge(&output, |p| p.x <= aabb_max.x, |a, b| {
        let t = (aabb_max.x - a.x) / (b.x - a.x);
        a + t * (b - a)
    });
    if output.is_empty() {
        return output;
    }

    // Clip against bottom edge (y >= aabb_min.y)
    output = clip_against_edge(&output, |p| p.y >= aabb_min.y, |a, b| {
        let t = (aabb_min.y - a.y) / (b.y - a.y);
        a + t * (b - a)
    });
    if output.is_empty() {
        return output;
    }

    // Clip against top edge (y <= aabb_max.y)
    clip_against_edge(&output, |p| p.y <= aabb_max.y, |a, b| {
        let t = (aabb_max.y - a.y) / (b.y - a.y);
        a + t * (b - a)
    })
}

/// Sutherland-Hodgman single-edge clip pass.
///
/// `is_inside` returns true if a point is on the inside of the clip edge.
/// `intersect` computes the intersection of edge (a -> b) with the clip boundary.
fn clip_against_edge(
    polygon: &[Vector2<f64>],
    is_inside: impl Fn(&Vector2<f64>) -> bool,
    intersect: impl Fn(&Vector2<f64>, &Vector2<f64>) -> Vector2<f64>,
) -> Vec<Vector2<f64>> {
    let n = polygon.len();
    if n == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(n + 1);
    for i in 0..n {
        let current = &polygon[i];
        let next = &polygon[(i + 1) % n];
        let cur_in = is_inside(current);
        let nxt_in = is_inside(next);

        match (cur_in, nxt_in) {
            (true, true) => {
                // Both inside: emit next
                result.push(*next);
            }
            (true, false) => {
                // Leaving: emit intersection
                result.push(intersect(current, next));
            }
            (false, true) => {
                // Entering: emit intersection, then next
                result.push(intersect(current, next));
                result.push(*next);
            }
            (false, false) => {
                // Both outside: emit nothing
            }
        }
    }
    result
}

/// Transform a body-local rectangle to world-frame corners.
///
/// The rectangle is centered on the body's local origin, offset by `offset` in
/// the local frame, with the given `width` and `height`. The body is at world
/// position `(body_x, body_y)` rotated by `body_theta` radians CCW.
///
/// Returns corners in CCW order: bottom-left, bottom-right, top-right, top-left
/// (relative to the unrotated local frame).
pub fn body_rect_to_world(
    body_x: f64,
    body_y: f64,
    body_theta: f64,
    width: f64,
    height: f64,
    offset: &Vector2<f64>,
) -> [Vector2<f64>; 4] {
    let hw = width / 2.0;
    let hh = height / 2.0;

    // Local-frame corners relative to body center, shifted by offset
    let local_corners = [
        Vector2::new(-hw + offset.x, -hh + offset.y),
        Vector2::new(hw + offset.x, -hh + offset.y),
        Vector2::new(hw + offset.x, hh + offset.y),
        Vector2::new(-hw + offset.x, hh + offset.y),
    ];

    let cos_t = body_theta.cos();
    let sin_t = body_theta.sin();

    local_corners.map(|lc| {
        Vector2::new(
            body_x + cos_t * lc.x - sin_t * lc.y,
            body_y + sin_t * lc.x + cos_t * lc.y,
        )
    })
}
