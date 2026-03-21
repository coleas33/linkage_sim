//! Rigid body data structures.
//!
//! Bodies are first-class rigid objects with multiple named attachment points.
//! A binary bar is a body with two attachment points. A ternary plate is a body
//! with three. They are the same object type.
//!
//! All coordinates are in the body's local frame. All units are SI (meters, kg).

use std::collections::HashMap;

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::core::state::GROUND_ID;

/// A rigid body in the mechanism.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Body {
    /// Unique identifier.
    pub id: String,
    /// Named points in body-local coordinates (meters) where joints connect.
    pub attachment_points: HashMap<String, Vector2<f64>>,
    /// Body mass in kg. Zero for ground.
    pub mass: f64,
    /// Center of gravity in body-local coordinates (meters).
    pub cg_local: Vector2<f64>,
    /// Moment of inertia about z-axis through CG (kg·m²).
    pub izz_cg: f64,
    /// Named points for force element attachment (not structural joints).
    pub mount_points: HashMap<String, Vector2<f64>>,
    /// Named points tracked for output (path tracing) but not used for connections.
    pub coupler_points: HashMap<String, Vector2<f64>>,
}

impl Body {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            attachment_points: HashMap::new(),
            mass: 0.0,
            cg_local: Vector2::zeros(),
            izz_cg: 0.0,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
        }
    }

    /// Get an attachment point's local coordinates.
    pub fn get_attachment_point(&self, name: &str) -> Result<&Vector2<f64>, BodyError> {
        self.attachment_points
            .get(name)
            .ok_or_else(|| BodyError::AttachmentPointNotFound {
                point: name.to_string(),
                body: self.id.clone(),
                available: self.attachment_points.keys().cloned().collect(),
            })
    }

    /// Add a named attachment point in body-local coordinates (meters).
    pub fn add_attachment_point(
        &mut self,
        name: &str,
        x: f64,
        y: f64,
    ) -> Result<(), BodyError> {
        if self.attachment_points.contains_key(name) {
            return Err(BodyError::DuplicateAttachmentPoint {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        self.attachment_points
            .insert(name.to_string(), Vector2::new(x, y));
        Ok(())
    }

    /// Add a point mass at a local body position.
    ///
    /// Updates total mass, CG position, and moment of inertia using
    /// the parallel axis theorem.  Mirrors the Python
    /// `linkage_sim.core.point_mass.add_point_mass` function.
    pub fn add_point_mass(&mut self, mass: f64, local_pos: Vector2<f64>) {
        let old_mass = self.mass;
        let new_total = old_mass + mass;

        if new_total <= 0.0 {
            return;
        }

        // New CG is weighted average
        let new_cg = (self.cg_local * old_mass + local_pos * mass) / new_total;

        // Parallel axis: shift existing Izz from old CG to new CG,
        // add point mass contribution shifted from its position to new CG
        let d_old = (self.cg_local - new_cg).norm();
        let d_new = (local_pos - new_cg).norm();
        let new_izz = self.izz_cg + old_mass * d_old * d_old + mass * d_new * d_new;

        self.mass = new_total;
        self.cg_local = new_cg;
        self.izz_cg = new_izz;
    }

    pub fn add_mount_point(&mut self, name: &str, x: f64, y: f64) -> Result<(), BodyError> {
        if self.attachment_points.contains_key(name) {
            return Err(BodyError::MountPointNameCollision {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        if self.mount_points.contains_key(name) {
            return Err(BodyError::DuplicateMountPoint {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        self.mount_points.insert(name.to_string(), Vector2::new(x, y));
        Ok(())
    }

    pub fn resolve_force_point(&self, name: &str) -> Result<&Vector2<f64>, BodyError> {
        self.attachment_points
            .get(name)
            .or_else(|| self.mount_points.get(name))
            .ok_or_else(|| BodyError::ForcePointNotFound {
                point: name.to_string(),
                body: self.id.clone(),
                available_attachment: self.attachment_points.keys().cloned().collect(),
                available_mount: self.mount_points.keys().cloned().collect(),
            })
    }

    /// Add a named coupler point for output tracking.
    pub fn add_coupler_point(
        &mut self,
        name: &str,
        x: f64,
        y: f64,
    ) -> Result<(), BodyError> {
        if self.coupler_points.contains_key(name) {
            return Err(BodyError::DuplicateCouplerPoint {
                point: name.to_string(),
                body: self.id.clone(),
            });
        }
        self.coupler_points
            .insert(name.to_string(), Vector2::new(x, y));
        Ok(())
    }
}

/// Create the ground body with named fixed pivot locations.
///
/// Ground is fixed at the global origin with zero mass.
/// Attachment points are in global coordinates (since ground doesn't move).
pub fn make_ground(attachment_points: &[(&str, f64, f64)]) -> Body {
    let mut pts = HashMap::new();
    for &(name, x, y) in attachment_points {
        pts.insert(name.to_string(), Vector2::new(x, y));
    }
    Body {
        id: GROUND_ID.to_string(),
        attachment_points: pts,
        mass: 0.0,
        cg_local: Vector2::zeros(),
        izz_cg: 0.0,
        mount_points: HashMap::new(),
        coupler_points: HashMap::new(),
    }
}

/// Create a binary bar (two attachment points along x-axis).
///
/// The bar's local frame origin is at p1, with p2 at (length, 0).
/// CG is at the midpoint by default.
pub fn make_bar(
    body_id: &str,
    p1_name: &str,
    p2_name: &str,
    length: f64,
    mass: f64,
    izz_cg: f64,
) -> Body {
    let mut attachment_points = HashMap::new();
    attachment_points.insert(p1_name.to_string(), Vector2::new(0.0, 0.0));
    attachment_points.insert(p2_name.to_string(), Vector2::new(length, 0.0));
    Body {
        id: body_id.to_string(),
        attachment_points,
        mass,
        cg_local: Vector2::new(length / 2.0, 0.0),
        izz_cg,
        mount_points: HashMap::new(),
        coupler_points: HashMap::new(),
    }
}

#[derive(Debug, Error)]
pub enum BodyError {
    #[error("Attachment point '{point}' not found on body '{body}'. Available: {available:?}")]
    AttachmentPointNotFound {
        point: String,
        body: String,
        available: Vec<String>,
    },
    #[error("Attachment point '{point}' already exists on body '{body}'")]
    DuplicateAttachmentPoint { point: String, body: String },
    #[error("Coupler point '{point}' already exists on body '{body}'")]
    DuplicateCouplerPoint { point: String, body: String },
    #[error("Mount point '{point}' already exists on body '{body}'")]
    DuplicateMountPoint { point: String, body: String },
    #[error("Mount point name '{point}' collides with attachment point on body '{body}'")]
    MountPointNameCollision { point: String, body: String },
    #[error(
        "Force point '{point}' not found on body '{body}'. \
         Attachment points: {available_attachment:?}, Mount points: {available_mount:?}"
    )]
    ForcePointNotFound {
        point: String,
        body: String,
        available_attachment: Vec<String>,
        available_mount: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn make_bar_creates_correct_points() {
        let bar = make_bar("crank", "A", "B", 0.1, 1.0, 0.001);
        assert_eq!(bar.id, "crank");
        assert_abs_diff_eq!(bar.attachment_points["A"].x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(bar.attachment_points["B"].x, 0.1, epsilon = 1e-15);
        assert_abs_diff_eq!(bar.cg_local.x, 0.05, epsilon = 1e-15);
        assert_abs_diff_eq!(bar.mass, 1.0, epsilon = 1e-15);
    }

    #[test]
    fn make_ground_creates_correct_body() {
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        assert_eq!(ground.id, "ground");
        assert_abs_diff_eq!(ground.mass, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(ground.attachment_points["O2"].x, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(ground.attachment_points["O4"].x, 0.038, epsilon = 1e-15);
    }

    #[test]
    fn add_attachment_point_works() {
        let mut body = Body::new("test");
        body.add_attachment_point("A", 1.0, 2.0).unwrap();
        let pt = body.get_attachment_point("A").unwrap();
        assert_abs_diff_eq!(pt.x, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(pt.y, 2.0, epsilon = 1e-15);
    }

    #[test]
    fn duplicate_attachment_point_rejected() {
        let mut body = Body::new("test");
        body.add_attachment_point("A", 1.0, 2.0).unwrap();
        assert!(body.add_attachment_point("A", 3.0, 4.0).is_err());
    }

    #[test]
    fn missing_attachment_point_errors() {
        let body = Body::new("test");
        assert!(body.get_attachment_point("missing").is_err());
    }

    #[test]
    fn add_coupler_point_works() {
        let mut body = Body::new("test");
        body.add_coupler_point("C", 0.5, 0.3).unwrap();
        assert!(body.coupler_points.contains_key("C"));
    }

    #[test]
    fn add_mount_point_works() {
        let mut body = Body::new("test");
        body.add_mount_point("M1", 0.3, 0.1).unwrap();
        assert!(body.mount_points.contains_key("M1"));
        let pt = &body.mount_points["M1"];
        assert_abs_diff_eq!(pt.x, 0.3, epsilon = 1e-15);
        assert_abs_diff_eq!(pt.y, 0.1, epsilon = 1e-15);
    }

    #[test]
    fn duplicate_mount_point_rejected() {
        let mut body = Body::new("test");
        body.add_mount_point("M1", 0.3, 0.1).unwrap();
        assert!(body.add_mount_point("M1", 0.5, 0.2).is_err());
    }

    #[test]
    fn mount_point_name_collision_with_attachment_rejected() {
        let mut body = Body::new("test");
        body.add_attachment_point("A", 1.0, 2.0).unwrap();
        assert!(body.add_mount_point("A", 0.5, 0.2).is_err());
    }

    #[test]
    fn resolve_force_point_finds_attachment() {
        let mut body = Body::new("test");
        body.add_attachment_point("A", 1.0, 2.0).unwrap();
        let pt = body.resolve_force_point("A").unwrap();
        assert_abs_diff_eq!(pt.x, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(pt.y, 2.0, epsilon = 1e-15);
    }

    #[test]
    fn resolve_force_point_finds_mount() {
        let mut body = Body::new("test");
        body.add_mount_point("M1", 0.3, 0.1).unwrap();
        let pt = body.resolve_force_point("M1").unwrap();
        assert_abs_diff_eq!(pt.x, 0.3, epsilon = 1e-15);
        assert_abs_diff_eq!(pt.y, 0.1, epsilon = 1e-15);
    }

    #[test]
    fn resolve_force_point_errors_on_missing() {
        let body = Body::new("test");
        assert!(body.resolve_force_point("nope").is_err());
    }

    // ── Point mass tests ──────────────────────────────────────────────

    #[test]
    fn point_mass_at_cg_does_not_shift_cg() {
        let mut body = Body::new("test");
        body.mass = 2.0;
        body.cg_local = Vector2::new(0.5, 0.0);
        body.izz_cg = 0.01;

        body.add_point_mass(1.0, Vector2::new(0.5, 0.0));

        assert_abs_diff_eq!(body.mass, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(body.cg_local.x, 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(body.cg_local.y, 0.0, epsilon = 1e-15);
        // No shift so Izz only gains the point mass contribution (zero offset)
        assert_abs_diff_eq!(body.izz_cg, 0.01, epsilon = 1e-15);
    }

    #[test]
    fn point_mass_at_offset_shifts_cg() {
        let mut body = Body::new("test");
        body.mass = 1.0;
        body.cg_local = Vector2::new(0.0, 0.0);
        body.izz_cg = 0.0;

        // Add equal mass at (1, 0) -- CG should move to (0.5, 0)
        body.add_point_mass(1.0, Vector2::new(1.0, 0.0));

        assert_abs_diff_eq!(body.mass, 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(body.cg_local.x, 0.5, epsilon = 1e-15);
        assert_abs_diff_eq!(body.cg_local.y, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn point_mass_increases_inertia_with_parallel_axis() {
        let mut body = Body::new("test");
        body.mass = 2.0;
        body.cg_local = Vector2::new(0.0, 0.0);
        body.izz_cg = 0.1;

        // Add 1 kg at (0.3, 0) -- new CG at (0.1, 0)
        body.add_point_mass(1.0, Vector2::new(0.3, 0.0));

        let new_total = 3.0;
        let new_cg_x = (2.0 * 0.0 + 1.0 * 0.3) / new_total;
        let d_old = new_cg_x; // distance old CG moved: |0 - 0.1|
        let d_new = (0.3_f64 - new_cg_x).abs(); // distance point mass to new CG
        let expected_izz = 0.1 + 2.0 * d_old * d_old + 1.0 * d_new * d_new;

        assert_abs_diff_eq!(body.mass, new_total, epsilon = 1e-15);
        assert_abs_diff_eq!(body.cg_local.x, new_cg_x, epsilon = 1e-15);
        assert_abs_diff_eq!(body.izz_cg, expected_izz, epsilon = 1e-12);
    }

    #[test]
    fn multiple_point_masses_accumulate() {
        let mut body = Body::new("test");
        body.mass = 1.0;
        body.cg_local = Vector2::new(0.0, 0.0);
        body.izz_cg = 0.0;

        // Add three equal masses forming a symmetric pattern
        body.add_point_mass(1.0, Vector2::new(1.0, 0.0));
        body.add_point_mass(1.0, Vector2::new(0.0, 1.0));
        body.add_point_mass(1.0, Vector2::new(1.0, 1.0));

        assert_abs_diff_eq!(body.mass, 4.0, epsilon = 1e-15);
        // CG of four equal masses at (0,0), (1,0), (0,1), (1,1) is (0.5, 0.5)
        assert_abs_diff_eq!(body.cg_local.x, 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(body.cg_local.y, 0.5, epsilon = 1e-12);
        // Inertia should be positive and greater than zero
        assert!(body.izz_cg > 0.0);

        // Verify by computing expected Izz: four point masses at corners
        // around CG (0.5, 0.5), each at distance sqrt(0.5) = 0.5*sqrt(2)
        // Izz = 4 * 1.0 * 0.5 = 2.0
        assert_abs_diff_eq!(body.izz_cg, 2.0, epsilon = 1e-12);
    }
}
