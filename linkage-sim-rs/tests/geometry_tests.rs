//! Tests for geometry utilities: polygon area, centroid, AABB clipping, and
//! body-rectangle-to-world transformation.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use std::f64::consts::FRAC_PI_2;

use linkage_sim_rs::geometry::{
    body_rect_to_world, clip_polygon_to_aabb, polygon_area, polygon_centroid,
};

// ---------------------------------------------------------------------------
// polygon_area
// ---------------------------------------------------------------------------

#[test]
fn polygon_area_unit_square() {
    let verts = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    assert_relative_eq!(polygon_area(&verts), 1.0, epsilon = 1e-12);
}

#[test]
fn polygon_area_triangle() {
    // Right triangle with legs of length 1 => area = 0.5
    let verts = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(0.0, 1.0),
    ];
    assert_relative_eq!(polygon_area(&verts), 0.5, epsilon = 1e-12);
}

#[test]
fn polygon_area_degenerate() {
    // Empty polygon
    assert_relative_eq!(polygon_area(&[]), 0.0, epsilon = 1e-12);

    // Single point
    let single = vec![Vector2::new(1.0, 2.0)];
    assert_relative_eq!(polygon_area(&single), 0.0, epsilon = 1e-12);

    // Collinear points (line segment) — zero area
    let line = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
    ];
    assert_relative_eq!(polygon_area(&line), 0.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// clip_polygon_to_aabb
// ---------------------------------------------------------------------------

#[test]
fn polygon_clip_full_overlap() {
    // A 0.6×0.6 square centered at origin fully inside a ±1 AABB.
    let body = vec![
        Vector2::new(-0.3, -0.3),
        Vector2::new(0.3, -0.3),
        Vector2::new(0.3, 0.3),
        Vector2::new(-0.3, 0.3),
    ];
    let aabb_min = Vector2::new(-1.0, -1.0);
    let aabb_max = Vector2::new(1.0, 1.0);
    let clipped = clip_polygon_to_aabb(&body, &aabb_min, &aabb_max);
    assert_relative_eq!(polygon_area(&clipped), 0.36, epsilon = 1e-10);
}

#[test]
fn polygon_clip_no_overlap() {
    // Body far outside the AABB.
    let body = vec![
        Vector2::new(5.0, 5.0),
        Vector2::new(6.0, 5.0),
        Vector2::new(6.0, 6.0),
        Vector2::new(5.0, 6.0),
    ];
    let aabb_min = Vector2::new(-1.0, -1.0);
    let aabb_max = Vector2::new(1.0, 1.0);
    let clipped = clip_polygon_to_aabb(&body, &aabb_min, &aabb_max);
    assert_relative_eq!(polygon_area(&clipped), 0.0, epsilon = 1e-10);
}

#[test]
fn polygon_clip_partial_overlap() {
    // Unit square from (0,0) to (1,1), clipped to AABB (0.5,0) to (2,2).
    // Overlap region is the rectangle (0.5,0) to (1,1) => area = 0.5.
    let body = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    let aabb_min = Vector2::new(0.5, 0.0);
    let aabb_max = Vector2::new(2.0, 2.0);
    let clipped = clip_polygon_to_aabb(&body, &aabb_min, &aabb_max);
    assert_relative_eq!(polygon_area(&clipped), 0.5, epsilon = 1e-10);
}

#[test]
fn polygon_clip_rotated_partial() {
    // A unit diamond (45° rotated square with vertices at distance 1 from origin)
    // clipped to the first quadrant. The diamond has vertices at (1,0), (0,1),
    // (-1,0), (0,-1). Total area = 2.0. First-quadrant portion = 0.5.
    let diamond = vec![
        Vector2::new(1.0, 0.0),
        Vector2::new(0.0, 1.0),
        Vector2::new(-1.0, 0.0),
        Vector2::new(0.0, -1.0),
    ];
    let aabb_min = Vector2::new(0.0, 0.0);
    let aabb_max = Vector2::new(2.0, 2.0);
    let clipped = clip_polygon_to_aabb(&diamond, &aabb_min, &aabb_max);
    assert_relative_eq!(polygon_area(&clipped), 0.5, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// polygon_centroid
// ---------------------------------------------------------------------------

#[test]
fn polygon_centroid_unit_square() {
    let verts = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    let c = polygon_centroid(&verts);
    assert_relative_eq!(c.x, 0.5, epsilon = 1e-12);
    assert_relative_eq!(c.y, 0.5, epsilon = 1e-12);
}

#[test]
fn polygon_centroid_of_clipped_region() {
    // Half-overlap from polygon_clip_partial_overlap: rectangle (0.5,0)–(1,1).
    // Centroid at (0.75, 0.5).
    let body = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.0, 1.0),
    ];
    let aabb_min = Vector2::new(0.5, 0.0);
    let aabb_max = Vector2::new(2.0, 2.0);
    let clipped = clip_polygon_to_aabb(&body, &aabb_min, &aabb_max);
    let c = polygon_centroid(&clipped);
    assert_relative_eq!(c.x, 0.75, epsilon = 1e-10);
    assert_relative_eq!(c.y, 0.5, epsilon = 1e-10);
}

// ---------------------------------------------------------------------------
// body_rect_to_world
// ---------------------------------------------------------------------------

#[test]
fn body_rect_to_world_no_rotation() {
    // 0.4×0.2 rectangle at body position (1, 2) with no rotation, no offset.
    let corners = body_rect_to_world(1.0, 2.0, 0.0, 0.4, 0.2, &Vector2::zeros());

    // Half-widths: 0.2, 0.1. Corners are body-center ± half-extents.
    let expected = [
        Vector2::new(1.0 - 0.2, 2.0 - 0.1),
        Vector2::new(1.0 + 0.2, 2.0 - 0.1),
        Vector2::new(1.0 + 0.2, 2.0 + 0.1),
        Vector2::new(1.0 - 0.2, 2.0 + 0.1),
    ];
    for (got, exp) in corners.iter().zip(expected.iter()) {
        assert_relative_eq!(got.x, exp.x, epsilon = 1e-12);
        assert_relative_eq!(got.y, exp.y, epsilon = 1e-12);
    }
}

#[test]
fn body_rect_to_world_90deg() {
    // 0.4×0.2 rectangle at origin rotated 90° CCW, no offset.
    let corners = body_rect_to_world(0.0, 0.0, FRAC_PI_2, 0.4, 0.2, &Vector2::zeros());

    // After 90° CCW rotation, local (+x, +y) → (-y, +x).
    // Local corners: (−0.2,−0.1), (0.2,−0.1), (0.2,0.1), (−0.2,0.1)
    // Rotated:       (0.1,−0.2),  (0.1,0.2),  (−0.1,0.2), (−0.1,−0.2)
    let expected = [
        Vector2::new(0.1, -0.2),
        Vector2::new(0.1, 0.2),
        Vector2::new(-0.1, 0.2),
        Vector2::new(-0.1, -0.2),
    ];
    for (got, exp) in corners.iter().zip(expected.iter()) {
        assert_relative_eq!(got.x, exp.x, epsilon = 1e-10);
        assert_relative_eq!(got.y, exp.y, epsilon = 1e-10);
    }
}
