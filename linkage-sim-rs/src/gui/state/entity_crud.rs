//! Entity creation and deletion operations on AppState.

use std::collections::HashMap;

use crate::core::state::GROUND_ID;
use crate::io::{BodyJson, JointJson};

use super::AppState;
use super::blueprint_ops::{joint_body_ids, joint_body_point_ids, joint_references_point, driver_body_ids, generate_unique_id};

impl AppState {
    // ── Raw blueprint helpers (no undo / no rebuild) ──────────────────

    /// Add a ground pivot (attachment point on the ground body).
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    /// Use this inside compound operations that batch a single undo + rebuild.
    pub(crate) fn add_ground_pivot_raw(&mut self, name: &str, x: f64, y: f64) {
        let Some(bp) = &mut self.blueprint else { return };
        let ground = bp.bodies.entry(GROUND_ID.to_string()).or_insert_with(|| BodyJson {
            attachment_points: HashMap::new(),
            mass: 0.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.0,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        });
        ground.attachment_points.insert(name.to_string(), [x, y]);
    }

    /// Add a revolute joint between two body attachment points.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    pub(crate) fn add_revolute_joint_raw(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        let joint_id = generate_unique_id("J", &bp.joints);
        bp.joints.insert(
            joint_id,
            JointJson::Revolute {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
                label: None,
            },
        );
    }

    /// Create a body from N world-coordinate points.
    ///
    /// The first point becomes local (0, 0); all other points are stored
    /// relative to the first. CG is set to the centroid of all local points.
    /// Default mass properties (mass=1, Izz=0.01) are assigned.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    pub(crate) fn add_body_with_points_raw(
        &mut self,
        body_id: &str,
        points: &[(String, [f64; 2])],
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if points.is_empty() {
            return;
        }

        // First point becomes the body-local origin.
        let origin = points[0].1;
        let mut attachment_points = HashMap::new();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for (name, world) in points {
            let local_x = world[0] - origin[0];
            let local_y = world[1] - origin[1];
            attachment_points.insert(name.clone(), [local_x, local_y]);
            sum_x += local_x;
            sum_y += local_y;
        }

        let n = points.len() as f64;
        let body = BodyJson {
            attachment_points,
            mass: 1.0,
            cg_local: [sum_x / n, sum_y / n],
            izz_cg: 0.01,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        };
        bp.bodies.insert(body_id.to_string(), body);
    }

    /// Add an attachment point to an existing body in body-local coordinates.
    ///
    /// Mutates the blueprint only. Does **not** push undo or rebuild.
    /// No-op if the body does not exist.
    pub(crate) fn add_attachment_point_local_raw(
        &mut self,
        body_id: &str,
        name: &str,
        local_x: f64,
        local_y: f64,
    ) {
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.attachment_points.insert(name.to_string(), [local_x, local_y]);
        }
    }

    // ── Update operations ────────────────────────────────────────────────

    /// Move a ground attachment point to new coordinates.
    ///
    /// Pushes undo, updates the point in the blueprint, and rebuilds.
    /// No-op if the blueprint or ground body is missing, or the point doesn't exist.
    pub fn update_ground_pivot_position(&mut self, name: &str, x: f64, y: f64) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let Some(ground) = bp.bodies.get_mut(GROUND_ID) else { return };
        if ground.attachment_points.contains_key(name) {
            ground.attachment_points.insert(name.to_string(), [x, y]);
        }
        self.rebuild();
    }

    /// Nudge all attachment points on a body by `(dx, dy)` in world coordinates.
    ///
    /// For the ground body, attachment points are already in world coordinates so
    /// the delta is applied directly. For moving bodies the delta is rotated into
    /// the body-local frame using the current pose before being applied.
    ///
    /// Pushes undo, mutates the blueprint, and rebuilds.
    /// No-op if the blueprint or body is missing.
    pub fn nudge_body(&mut self, body_id: &str, dx: f64, dy: f64) {
        self.push_undo();
        // For non-ground bodies we need to rotate the world-space delta into
        // the body-local frame so that the attachment point offsets shift
        // correctly.
        let (local_dx, local_dy) = self.world_delta_to_local(body_id, dx, dy);
        let Some(bp) = &mut self.blueprint else { return };
        let Some(body) = bp.bodies.get_mut(body_id) else { return };
        for point in body.attachment_points.values_mut() {
            point[0] += local_dx;
            point[1] += local_dy;
        }
        self.rebuild();
    }

    /// Nudge both attachment points of a joint by `(dx, dy)` in world coordinates.
    ///
    /// Each side of the joint is shifted independently using the appropriate
    /// body-local delta. Ground-side points use world coordinates directly.
    ///
    /// Pushes undo, mutates the blueprint, and rebuilds.
    /// No-op if the blueprint or joint is missing, or the joint has no point
    /// fields (e.g. `RevoluteDriver`).
    pub fn nudge_joint(&mut self, joint_id: &str, dx: f64, dy: f64) {
        self.push_undo();

        // Extract the four IDs we need before borrowing self.blueprint mutably.
        let ids = {
            let Some(bp) = &self.blueprint else { return };
            let Some(joint) = bp.joints.get(joint_id) else { return };
            let Some((bi, pi, bj, pj)) = joint_body_point_ids(joint) else { return };
            (bi.to_string(), pi.to_string(), bj.to_string(), pj.to_string())
        };
        let (body_i, point_i, body_j, point_j) = ids;

        // Convert world delta to each body's local frame.
        let (di_x, di_y) = self.world_delta_to_local(&body_i, dx, dy);
        let (dj_x, dj_y) = self.world_delta_to_local(&body_j, dx, dy);

        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(&body_i) {
            if let Some(pt) = body.attachment_points.get_mut(&point_i) {
                pt[0] += di_x;
                pt[1] += di_y;
            }
        }
        if let Some(body) = bp.bodies.get_mut(&body_j) {
            if let Some(pt) = body.attachment_points.get_mut(&point_j) {
                pt[0] += dj_x;
                pt[1] += dj_y;
            }
        }
        self.rebuild();
    }

    /// Convert a world-space delta `(dx, dy)` into body-local coordinates.
    ///
    /// For the ground body (which stores points in world coordinates) this
    /// returns the delta unchanged. For moving bodies the inverse rotation
    /// of the current pose angle is applied.
    fn world_delta_to_local(&self, body_id: &str, dx: f64, dy: f64) -> (f64, f64) {
        if body_id == GROUND_ID {
            return (dx, dy);
        }
        let theta = match &self.mechanism {
            Some(mech) => match mech.state().get_index(body_id) {
                Ok(idx) => self.q[idx.q_start + 2],
                Err(_) => return (dx, dy),
            },
            None => return (dx, dy),
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        // Inverse rotation: A^T * [dx, dy]
        (cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy)
    }

    // ── Create / delete operations ──────────────────────────────────────

    /// Add a named attachment point to a body at a world-coordinate position.
    ///
    /// Converts the world coordinates to body-local using the current pose, then
    /// delegates to `add_attachment_point_local_raw`. Pushes undo and rebuilds.
    /// No-op if the body does not exist or there is no blueprint.
    pub fn add_attachment_point_to_body(
        &mut self,
        body_id: &str,
        name: &str,
        world_x: f64,
        world_y: f64,
    ) {
        self.push_undo();
        let [lx, ly] = self.world_to_body_local(body_id, world_x, world_y);
        self.add_attachment_point_local_raw(body_id, name, lx, ly);
        self.rebuild();
    }

    /// Remove a named attachment point from a body.
    ///
    /// Cascades: any joint that references `(body_id, point_name)` is also
    /// removed. Pushes undo and rebuilds.
    /// No-op if the body or point does not exist, or there is no blueprint.
    pub fn remove_attachment_point(&mut self, body_id: &str, point_name: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        if let Some(body) = bp.bodies.get_mut(body_id) {
            body.attachment_points.remove(point_name);
        }
        bp.joints
            .retain(|_id, joint| !joint_references_point(joint, body_id, point_name));
        self.rebuild();
    }

    /// Add a new ground pivot (attachment point on the ground body).
    ///
    /// Pushes undo, adds the point, and rebuilds.
    pub fn add_ground_pivot(&mut self, name: &str, x: f64, y: f64) {
        self.push_undo();
        self.add_ground_pivot_raw(name, x, y);
        self.rebuild();
    }

    /// Add a new body from N world-coordinate points.
    ///
    /// Auto-generates a body ID via `next_body_id()`. The first point becomes
    /// local (0, 0); CG is at the centroid. Pushes undo, adds the body, and rebuilds.
    ///
    /// Returns the generated body ID.
    pub fn add_body_with_points(&mut self, points: &[(String, [f64; 2])]) -> String {
        self.push_undo();
        let body_id = self.next_body_id();
        self.add_body_with_points_raw(&body_id, points);
        self.rebuild();
        body_id
    }

    /// Remove a body and all joints/drivers that reference it.
    ///
    /// Pushes undo, removes the body and cascading references, and rebuilds.
    pub fn remove_body(&mut self, body_id: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };

        bp.bodies.remove(body_id);

        // Remove joints that reference this body.
        bp.joints.retain(|_id, joint| {
            let (bi, bj) = joint_body_ids(joint);
            bi != body_id && bj != body_id
        });

        // Remove drivers that reference this body.
        bp.drivers.retain(|_id, driver| {
            let (bi, bj) = driver_body_ids(driver);
            bi != body_id && bj != body_id
        });

        self.rebuild();
    }

    /// Add a revolute joint between two body attachment points.
    ///
    /// Generates a unique joint ID. Pushes undo, adds joint, and rebuilds.
    pub fn add_revolute_joint(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        self.push_undo();
        self.add_revolute_joint_raw(body_i, point_i, body_j, point_j);
        self.rebuild();
    }

    /// Add a prismatic joint between two bodies at the given attachment points.
    ///
    /// The slide axis defaults to the vector from point_i to point_j in body_i's
    /// local frame (normalized). delta_theta_0 defaults to 0.
    pub fn add_prismatic_joint(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };

        // Compute default axis from the direction between the two points.
        // Use the blueprint coordinates (local frame of body_i).
        let axis = if let (Some(bi), Some(bj)) = (bp.bodies.get(body_i), bp.bodies.get(body_j)) {
            if let (Some(pi), Some(pj)) = (
                bi.attachment_points.get(point_i),
                bj.attachment_points.get(point_j),
            ) {
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-12 {
                    [dx / len, dy / len]
                } else {
                    [1.0, 0.0]
                }
            } else {
                [1.0, 0.0]
            }
        } else {
            [1.0, 0.0]
        };

        let joint_id = generate_unique_id("J", &bp.joints);
        bp.joints.insert(
            joint_id,
            JointJson::Prismatic {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
                axis_local_i: axis,
                delta_theta_0: 0.0,
                label: None,
            },
        );
        self.rebuild();
    }

    /// Add a fixed joint between two bodies at the given attachment points.
    pub fn add_fixed_joint(
        &mut self,
        body_i: &str,
        point_i: &str,
        body_j: &str,
        point_j: &str,
    ) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        let joint_id = generate_unique_id("J", &bp.joints);
        bp.joints.insert(
            joint_id,
            JointJson::Fixed {
                body_i: body_i.to_string(),
                body_j: body_j.to_string(),
                point_i: point_i.to_string(),
                point_j: point_j.to_string(),
                delta_theta_0: 0.0,
                label: None,
            },
        );
        self.rebuild();
    }

    /// Remove a joint by ID.
    ///
    /// Pushes undo, removes the joint, and rebuilds.
    pub fn remove_joint(&mut self, joint_id: &str) {
        self.push_undo();
        let Some(bp) = &mut self.blueprint else { return };
        bp.joints.remove(joint_id);
        self.rebuild();
    }

    /// Generate a unique body ID (e.g. "body_1", "body_2", ...).
    pub fn next_body_id(&self) -> String {
        let Some(bp) = &self.blueprint else {
            return "body_1".to_string();
        };
        generate_unique_id("body_", &bp.bodies)
    }

    /// Generate a unique ground pivot name (e.g. "P1", "P2", ...).
    pub fn next_ground_pivot_name(&self) -> String {
        let Some(bp) = &self.blueprint else {
            return "P1".to_string();
        };
        if let Some(ground) = bp.bodies.get(GROUND_ID) {
            let mut i = 1;
            loop {
                let name = format!("P{}", i);
                if !ground.attachment_points.contains_key(&name) {
                    return name;
                }
                i += 1;
            }
        } else {
            "P1".to_string()
        }
    }

    /// Generate the next unused attachment point name for a body.
    ///
    /// Names follow the sequence A, B, ..., Z, AA, AB, ..., AZ, BA, ...
    /// (bijective base-26, always uppercase). Returns "A" when the body
    /// does not exist or the blueprint is absent.
    pub fn next_attachment_point_name(&self, body_id: &str) -> String {
        let Some(bp) = &self.blueprint else {
            return "A".to_string();
        };
        let existing: std::collections::HashSet<&String> = bp
            .bodies
            .get(body_id)
            .map(|b| b.attachment_points.keys().collect())
            .unwrap_or_default();

        let mut name = String::new();
        let mut n: usize = 0;
        loop {
            name.clear();
            let mut val = n;
            loop {
                name.push((b'A' + (val % 26) as u8) as char);
                val /= 26;
                if val == 0 {
                    break;
                }
                val -= 1;
            }
            let name_rev: String = name.chars().rev().collect();
            if !existing.contains(&name_rev) {
                return name_rev;
            }
            n += 1;
        }
    }

    /// Convert world coordinates to body-local coordinates using the body's
    /// current pose from `self.q`.
    ///
    /// Returns world coordinates unchanged for the ground body (which has no
    /// pose in `q`) or when the mechanism is not built.
    pub fn world_to_body_local(&self, body_id: &str, world_x: f64, world_y: f64) -> [f64; 2] {
        if body_id == GROUND_ID {
            return [world_x, world_y];
        }
        let q_start = match &self.mechanism {
            Some(mech) => match mech.state().get_index(body_id) {
                Ok(idx) => idx.q_start,
                Err(_) => return [world_x, world_y],
            },
            None => return [world_x, world_y],
        };
        let bx = self.q[q_start];
        let by = self.q[q_start + 1];
        let theta = self.q[q_start + 2];
        let dx = world_x - bx;
        let dy = world_y - by;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        [cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy]
    }
}
