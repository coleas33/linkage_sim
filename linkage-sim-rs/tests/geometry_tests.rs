//! Tests for geometry utilities: polygon area, centroid, AABB clipping,
//! body-rectangle-to-world transformation, and BodyGeometry validation.

use approx::assert_relative_eq;
use nalgebra::Vector2;
use std::f64::consts::FRAC_PI_2;

use linkage_sim_rs::core::body::{BodyGeometry, GeometryShape, CIRCLE_SEGMENTS};
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

#[test]
fn clip_polygon_to_aabb_empty_input() {
    let aabb_min = Vector2::new(-1.0, -1.0);
    let aabb_max = Vector2::new(1.0, 1.0);
    let clipped = clip_polygon_to_aabb(&[], &aabb_min, &aabb_max);
    assert!(clipped.is_empty());
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

#[test]
fn polygon_centroid_degenerate() {
    // Empty polygon returns origin.
    let c = polygon_centroid(&[]);
    assert_relative_eq!(c.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(c.y, 0.0, epsilon = 1e-12);

    // Collinear points (zero area) also return origin.
    let collinear = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(2.0, 0.0),
    ];
    let c = polygon_centroid(&collinear);
    assert_relative_eq!(c.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(c.y, 0.0, epsilon = 1e-12);
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

#[test]
fn body_rect_to_world_with_nonzero_offset() {
    // 0.4x0.2 rectangle at origin, no rotation, offset (0.5, 0.3).
    // The offset shifts the local-frame rectangle before world transform,
    // so corners become (offset.x +/- hw, offset.y +/- hh).
    let offset = Vector2::new(0.5, 0.3);
    let corners = body_rect_to_world(0.0, 0.0, 0.0, 0.4, 0.2, &offset);

    let expected = [
        Vector2::new(-0.2 + 0.5, -0.1 + 0.3), // (0.3, 0.2)
        Vector2::new(0.2 + 0.5, -0.1 + 0.3),  // (0.7, 0.2)
        Vector2::new(0.2 + 0.5, 0.1 + 0.3),   // (0.7, 0.4)
        Vector2::new(-0.2 + 0.5, 0.1 + 0.3),  // (0.3, 0.4)
    ];
    for (got, exp) in corners.iter().zip(expected.iter()) {
        assert_relative_eq!(got.x, exp.x, epsilon = 1e-12);
        assert_relative_eq!(got.y, exp.y, epsilon = 1e-12);
    }
}

// ---------------------------------------------------------------------------
// BodyGeometry
// ---------------------------------------------------------------------------

#[test]
fn body_geometry_valid() {
    let geo = BodyGeometry::new(0.06, 0.015, Vector2::new(0.0, 0.0));
    assert!(geo.is_ok());
    let geo = geo.unwrap();
    assert!((geo.width - 0.06).abs() < 1e-12);
    assert!((geo.height - 0.015).abs() < 1e-12);
    assert!((geo.area() - 0.0009).abs() < 1e-12);
}

#[test]
fn body_geometry_rejects_zero_dimension() {
    assert!(BodyGeometry::new(0.0, 0.015, Vector2::new(0.0, 0.0)).is_err());
    assert!(BodyGeometry::new(0.06, 0.0, Vector2::new(0.0, 0.0)).is_err());
    assert!(BodyGeometry::new(-0.01, 0.015, Vector2::new(0.0, 0.0)).is_err());
}

#[test]
fn body_geometry_with_offset() {
    let offset = Vector2::new(0.1, -0.05);
    let geo = BodyGeometry::new(0.2, 0.1, offset).unwrap();
    assert!((geo.offset.x - 0.1).abs() < 1e-12);
    assert!((geo.offset.y - (-0.05)).abs() < 1e-12);
    assert!((geo.area() - 0.02).abs() < 1e-12);
}

#[test]
fn body_geometry_rejects_negative_height() {
    assert!(BodyGeometry::new(0.06, -0.01, Vector2::new(0.0, 0.0)).is_err());
}

// ── Round shapes (schema 1.2.0, decisions R-1, R-3, R-5, R-6) ───────────

#[test]
fn a_circle_keeps_its_diameter_as_width_and_height() {
    let geo = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    assert_eq!(geo.shape, GeometryShape::Circle);
    assert_relative_eq!(geo.width, 0.13208);
    assert_relative_eq!(geo.height, 0.13208);
    assert_relative_eq!(
        geo.area(),
        std::f64::consts::PI * 0.13208 * 0.13208 / 4.0,
        epsilon = 1e-15
    );
}

#[test]
fn a_circle_needs_a_positive_finite_diameter() {
    for d in [0.0, -0.01, f64::NAN, f64::INFINITY] {
        assert!(BodyGeometry::circle(d, Vector2::zeros()).is_err(), "diameter {d}");
    }
}

#[test]
fn a_rectangle_outline_is_its_four_corners() {
    let offset = Vector2::new(0.05, 0.0);
    let geo = BodyGeometry::new(0.2, 0.1, offset).unwrap();
    assert_eq!(geo.shape, GeometryShape::Rectangle);
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    let corners = body_rect_to_world(1.0, 2.0, FRAC_PI_2, 0.2, 0.1, &offset);
    assert_eq!(outline.len(), 4);
    for (a, b) in outline.iter().zip(corners.iter()) {
        assert_relative_eq!(a.x, b.x, epsilon = 1e-12);
        assert_relative_eq!(a.y, b.y, epsilon = 1e-12);
    }
}

#[test]
fn a_circle_outline_is_a_polygon_on_the_circle_turned_with_the_body() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    let outline = geo.outline_world(1.0, 2.0, FRAC_PI_2);
    assert_eq!(outline.len(), CIRCLE_SEGMENTS);
    let centre = geo.centre_world(1.0, 2.0, FRAC_PI_2);
    assert_relative_eq!(centre.x, 1.0, epsilon = 1e-12);
    assert_relative_eq!(centre.y, 2.2, epsilon = 1e-12);
    for p in &outline {
        assert_relative_eq!((p - centre).norm(), 0.05, epsilon = 1e-12);
    }
    // Counter-clockwise, like the rectangle's corners: positive signed area.
    let n = outline.len();
    let signed: f64 = (0..n)
        .map(|i| {
            let (a, b) = (outline[i], outline[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        })
        .sum::<f64>()
        / 2.0;
    assert!(signed > 0.0, "signed area {signed}");
    assert_relative_eq!(polygon_area(&outline), signed, epsilon = 1e-15);
}

#[test]
fn a_circle_s_extreme_point_is_its_centre_plus_the_radius_along_the_direction() {
    let geo = BodyGeometry::circle(0.1, Vector2::new(0.2, 0.0)).unwrap();
    // However the body turns, the point does not turn with it.
    for theta in [0.0, 0.7, -2.0] {
        let centre = geo.centre_world(1.0, 2.0, theta);
        let low = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(0.0, -3.0));
        assert_relative_eq!(low.x, centre.x, epsilon = 1e-12);
        assert_relative_eq!(low.y, centre.y - 0.05, epsilon = 1e-12);
        let corner = geo.extreme_point_world(1.0, 2.0, theta, Vector2::new(1.0, 1.0));
        let r = 0.05 / 2f64.sqrt();
        assert_relative_eq!(corner.x, centre.x + r, epsilon = 1e-12);
        assert_relative_eq!(corner.y, centre.y + r, epsilon = 1e-12);
    }
}

#[test]
fn a_rectangle_s_extreme_point_is_its_farthest_corner_or_its_edge_midpoint() {
    let geo = BodyGeometry::new(0.2, 0.1, Vector2::zeros()).unwrap();
    // Level: the whole bottom edge meets a surface below; its midpoint.
    let level = geo.extreme_point_world(0.0, 0.0, 0.0, Vector2::new(0.0, -1.0));
    assert_relative_eq!(level.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(level.y, -0.05, epsilon = 1e-12);
    // Tilted: one corner is lowest.
    let theta = 0.3_f64;
    let tilted = geo.extreme_point_world(0.0, 0.0, theta, Vector2::new(0.0, -1.0));
    let corners = body_rect_to_world(0.0, 0.0, theta, 0.2, 0.1, &Vector2::zeros());
    let lowest = corners.iter().min_by(|a, b| a.y.total_cmp(&b.y)).unwrap();
    assert_relative_eq!(tilted.x, lowest.x, epsilon = 1e-12);
    assert_relative_eq!(tilted.y, lowest.y, epsilon = 1e-12);
}

#[test]
fn a_zero_direction_gives_the_shape_s_centre() {
    let shapes = [
        BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap(),
        BodyGeometry::circle(0.1, Vector2::new(0.05, 0.0)).unwrap(),
    ];
    for geo in shapes {
        for direction in [Vector2::zeros(), Vector2::new(f64::NAN, 1.0)] {
            let p = geo.extreme_point_world(1.0, 2.0, 0.4, direction);
            let c = geo.centre_world(1.0, 2.0, 0.4);
            assert_relative_eq!(p.x, c.x, epsilon = 1e-15);
            assert_relative_eq!(p.y, c.y, epsilon = 1e-15);
        }
    }
}

#[test]
fn a_rectangle_is_written_without_a_shape_key_and_old_files_load_as_rectangles() {
    let rect = BodyGeometry::new(0.2, 0.1, Vector2::new(0.05, 0.0)).unwrap();
    let json = serde_json::to_value(&rect).unwrap();
    assert!(json.get("shape").is_none(), "{json}");
    let old: BodyGeometry =
        serde_json::from_str(r#"{"width":0.2,"height":0.1,"offset":[0.05,0.0]}"#).unwrap();
    assert_eq!(old.shape, GeometryShape::Rectangle);
}

#[test]
fn a_circle_round_trips_and_keeps_width_and_height_for_older_builds() {
    let circle = BodyGeometry::circle(0.13208, Vector2::new(0.3, 0.05)).unwrap();
    let json = serde_json::to_value(&circle).unwrap();
    assert_eq!(json["shape"], "circle");
    // A build before 1.2.0 ignores "shape" and draws the circle's bounding square.
    assert_eq!(json["width"], json["height"]);
    let back: BodyGeometry = serde_json::from_value(json).unwrap();
    assert_eq!(back.shape, GeometryShape::Circle);
    assert_relative_eq!(back.width, 0.13208);
}

#[test]
fn files_with_shapes_are_schema_1_2_0() {
    assert_eq!(linkage_sim_rs::io::SCHEMA_VERSION, "1.2.0");
}
