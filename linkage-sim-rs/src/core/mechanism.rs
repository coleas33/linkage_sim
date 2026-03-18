//! Mechanism: the top-level assembly of bodies, joints, and state.
//!
//! A Mechanism owns the bodies, joints, and state vector mapping.
//! It provides the interface for building mechanisms and is the input
//! to all solvers.

use nalgebra::Vector2;
use std::collections::HashMap;
use thiserror::Error;

use crate::core::body::Body;
use crate::core::constraint::{
    make_fixed_joint, make_prismatic_joint, make_revolute_joint, Constraint, JointConstraint,
};
use crate::core::driver::{constant_speed_driver, make_revolute_driver, RevoluteDriver};
use crate::core::state::{State, GROUND_ID};

/// Row range in the lambda vector for a single constraint (joint or driver).
///
/// After `build()`, every joint and driver owns a contiguous block of rows
/// in the assembled constraint vector Phi and the multiplier vector lambda.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintRange {
    pub constraint_id: String,
    pub row_start: usize,
    pub n_equations: usize,
}

/// A planar mechanism: bodies connected by joint constraints.
pub struct Mechanism {
    bodies: HashMap<String, Body>,
    joints: Vec<JointConstraint>,
    drivers: Vec<RevoluteDriver>,
    state: State,
    built: bool,
    /// Deterministic body ordering used for the state vector q.
    /// Populated during `build()` — contains moving body IDs in alphabetical
    /// order (matching the Python reference implementation).
    body_order: Vec<String>,
    /// Row ranges in the lambda vector for each constraint, computed once in `build()`.
    constraint_ranges: Vec<ConstraintRange>,
}

impl Mechanism {
    pub fn new() -> Self {
        Self {
            bodies: HashMap::new(),
            joints: Vec::new(),
            drivers: Vec::new(),
            state: State::new(),
            built: false,
            body_order: Vec::new(),
            constraint_ranges: Vec::new(),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn bodies(&self) -> &HashMap<String, Body> {
        &self.bodies
    }

    pub fn joints(&self) -> &[JointConstraint] {
        &self.joints
    }

    pub fn n_drivers(&self) -> usize {
        self.drivers.len()
    }

    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Deterministic ordering of moving body IDs in the state vector q.
    /// Bodies are sorted alphabetically (matching the Python reference).
    /// Only valid after `build()`.
    pub fn body_order(&self) -> &[String] {
        &self.body_order
    }

    /// Row ranges in the lambda vector for each constraint (joints then drivers).
    /// Only valid after `build()`.
    pub fn constraint_ranges(&self) -> &[ConstraintRange] {
        &self.constraint_ranges
    }

    /// Total number of constraint equations across all joints and drivers.
    pub fn n_constraints(&self) -> usize {
        let joint_eqs: usize = self.joints.iter().map(|j| j.n_equations()).sum();
        let driver_eqs: usize = self.drivers.iter().map(|d| d.n_equations()).sum();
        joint_eqs + driver_eqs
    }

    /// Iterate over all constraints (joints + drivers) in order.
    pub fn all_constraints(&self) -> Vec<&dyn Constraint> {
        let mut all: Vec<&dyn Constraint> = Vec::new();
        for j in &self.joints {
            all.push(j);
        }
        for d in &self.drivers {
            all.push(d);
        }
        all
    }

    /// Add a body to the mechanism.
    pub fn add_body(&mut self, body: Body) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add body"));
        }
        if self.bodies.contains_key(&body.id) {
            return Err(MechanismError::DuplicateBody(body.id.clone()));
        }
        self.bodies.insert(body.id.clone(), body);
        Ok(())
    }

    /// Add a revolute joint between two bodies at named attachment points.
    pub fn add_revolute_joint(
        &mut self,
        joint_id: &str,
        body_i_id: &str,
        point_i_name: &str,
        body_j_id: &str,
        point_j_name: &str,
    ) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add joint"));
        }
        let pt_i = *self.get_attachment_point(body_i_id, point_i_name)?;
        let pt_j = *self.get_attachment_point(body_j_id, point_j_name)?;

        let joint = make_revolute_joint(joint_id, body_i_id, pt_i, body_j_id, pt_j);
        self.joints.push(JointConstraint::Revolute(joint));
        Ok(())
    }

    /// Add a fixed joint that locks all relative motion.
    pub fn add_fixed_joint(
        &mut self,
        joint_id: &str,
        body_i_id: &str,
        point_i_name: &str,
        body_j_id: &str,
        point_j_name: &str,
        delta_theta_0: f64,
    ) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add joint"));
        }
        let pt_i = *self.get_attachment_point(body_i_id, point_i_name)?;
        let pt_j = *self.get_attachment_point(body_j_id, point_j_name)?;

        let joint = make_fixed_joint(joint_id, body_i_id, pt_i, body_j_id, pt_j, delta_theta_0);
        self.joints.push(JointConstraint::Fixed(joint));
        Ok(())
    }

    /// Add a prismatic joint that allows sliding along one axis.
    pub fn add_prismatic_joint(
        &mut self,
        joint_id: &str,
        body_i_id: &str,
        point_i_name: &str,
        body_j_id: &str,
        point_j_name: &str,
        axis_local_i: Vector2<f64>,
        delta_theta_0: f64,
    ) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add joint"));
        }
        let pt_i = *self.get_attachment_point(body_i_id, point_i_name)?;
        let pt_j = *self.get_attachment_point(body_j_id, point_j_name)?;

        let joint = make_prismatic_joint(
            joint_id,
            body_i_id,
            pt_i,
            body_j_id,
            pt_j,
            axis_local_i,
            delta_theta_0,
        )
        .map_err(|e| MechanismError::InvalidJoint(e.to_string()))?;
        self.joints.push(JointConstraint::Prismatic(joint));
        Ok(())
    }

    /// Add a constant-speed revolute driver.
    pub fn add_constant_speed_driver(
        &mut self,
        driver_id: &str,
        body_i_id: &str,
        body_j_id: &str,
        omega: f64,
        theta_0: f64,
    ) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add driver"));
        }
        self.validate_body_exists(body_i_id)?;
        self.validate_body_exists(body_j_id)?;

        let driver = constant_speed_driver(driver_id, body_i_id, body_j_id, omega, theta_0);
        self.drivers.push(driver);
        Ok(())
    }

    /// Add a revolute driver with custom functions.
    pub fn add_revolute_driver(
        &mut self,
        driver_id: &str,
        body_i_id: &str,
        body_j_id: &str,
        f: impl Fn(f64) -> f64 + Send + Sync + 'static,
        f_dot: impl Fn(f64) -> f64 + Send + Sync + 'static,
        f_ddot: impl Fn(f64) -> f64 + Send + Sync + 'static,
    ) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("add driver"));
        }
        self.validate_body_exists(body_i_id)?;
        self.validate_body_exists(body_j_id)?;

        let driver = make_revolute_driver(driver_id, body_i_id, body_j_id, f, f_dot, f_ddot);
        self.drivers.push(driver);
        Ok(())
    }

    /// Finalize the mechanism: register moving bodies in the state vector.
    ///
    /// Bodies are registered in alphabetical order to ensure deterministic
    /// state vector layout. This matches the Python reference implementation
    /// (`sorted(self._bodies.keys())`). The resulting order is stored in
    /// `body_order` and can be queried via `body_order()`.
    pub fn build(&mut self) -> Result<(), MechanismError> {
        if self.built {
            return Err(MechanismError::AlreadyBuilt("build"));
        }

        let mut body_ids: Vec<_> = self.bodies.keys().cloned().collect();
        body_ids.sort();

        let mut order = Vec::new();
        for body_id in &body_ids {
            if body_id != GROUND_ID {
                self.state
                    .register_body(body_id)
                    .map_err(|e| MechanismError::StateError(e.to_string()))?;
                order.push(body_id.clone());
            }
        }
        self.body_order = order;

        // Compute constraint row ranges (joints first, then drivers).
        let mut ranges = Vec::new();
        let mut row = 0;
        for constraint in self.all_constraints() {
            let n_eq = constraint.n_equations();
            ranges.push(ConstraintRange {
                constraint_id: constraint.id().to_string(),
                row_start: row,
                n_equations: n_eq,
            });
            row += n_eq;
        }
        self.constraint_ranges = ranges;

        self.built = true;
        Ok(())
    }

    fn get_attachment_point(
        &self,
        body_id: &str,
        point_name: &str,
    ) -> Result<&Vector2<f64>, MechanismError> {
        let body = self
            .bodies
            .get(body_id)
            .ok_or_else(|| MechanismError::BodyNotFound(body_id.to_string()))?;
        body.get_attachment_point(point_name)
            .map_err(|e| MechanismError::InvalidJoint(e.to_string()))
    }

    fn validate_body_exists(&self, body_id: &str) -> Result<(), MechanismError> {
        if !self.bodies.contains_key(body_id) {
            return Err(MechanismError::BodyNotFound(body_id.to_string()));
        }
        Ok(())
    }
}

impl Default for Mechanism {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Error)]
pub enum MechanismError {
    #[error("Cannot {0} after build()")]
    AlreadyBuilt(&'static str),
    #[error("Body '{0}' already exists in the mechanism")]
    DuplicateBody(String),
    #[error("Body '{0}' not found in mechanism")]
    BodyNotFound(String),
    #[error("Invalid joint: {0}")]
    InvalidJoint(String),
    #[error("State error: {0}")]
    StateError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::body::{make_bar, make_ground};
    use nalgebra::Vector2;
    use std::f64::consts::PI;

    fn build_fourbar() -> Mechanism {
        // Classic 4-bar: ground(O2)--crank--coupler--rocker--ground(O4)
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();
        mech.add_body(rocker).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
            .unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
            .unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4")
            .unwrap();
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0)
            .unwrap();

        mech.build().unwrap();
        mech
    }

    #[test]
    fn fourbar_builds_correctly() {
        let mech = build_fourbar();
        assert!(mech.is_built());
        // 3 moving bodies × 3 coords each = 9
        assert_eq!(mech.state().n_coords(), 9);
        // 4 revolute joints × 2 + 1 driver = 9 constraints
        assert_eq!(mech.n_constraints(), 9);
    }

    #[test]
    fn cannot_add_after_build() {
        let mut mech = build_fourbar();
        let extra = make_bar("extra", "X", "Y", 0.1, 0.0, 0.0);
        assert!(mech.add_body(extra).is_err());
    }

    #[test]
    fn body_not_found_errors() {
        let mut mech = Mechanism::new();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        mech.add_body(ground).unwrap();
        assert!(mech
            .add_revolute_joint("J1", "ground", "O", "missing", "A")
            .is_err());
    }

    #[test]
    fn fourbar_constraint_count() {
        let mech = build_fourbar();
        assert_eq!(mech.n_constraints(), 9);
        assert_eq!(mech.all_constraints().len(), 5); // 4 joints + 1 driver
    }

    #[test]
    fn body_order_is_deterministic_alphabetical() {
        // Verify body ordering is alphabetical regardless of insertion order.
        let mech = build_fourbar();
        let order = mech.body_order();
        assert_eq!(order, &["coupler", "crank", "rocker"]);
    }

    #[test]
    fn body_order_matches_state_body_ids() {
        // body_order() and state().body_ids() must agree.
        let mech = build_fourbar();
        let order = mech.body_order();
        let state_ids = mech.state().body_ids();
        assert_eq!(order, state_ids.as_slice());
    }

    #[test]
    fn body_order_independent_of_insertion_order() {
        // Insert bodies in reverse-alphabetical order; ordering must still
        // be alphabetical after build().
        let ground = make_ground(&[("O2", 0.0, 0.0), ("O4", 0.038, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
        let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);
        let rocker = make_bar("rocker", "C", "D", 0.03, 0.0, 0.0);

        let mut mech = Mechanism::new();
        // Deliberately insert in reverse-alpha order
        mech.add_body(rocker).unwrap();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(coupler).unwrap();

        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
            .unwrap();
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
            .unwrap();
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
            .unwrap();
        mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4")
            .unwrap();

        mech.build().unwrap();

        assert_eq!(mech.body_order(), &["coupler", "crank", "rocker"]);
        assert_eq!(mech.body_order(), mech.state().body_ids().as_slice());
    }

    #[test]
    fn body_order_empty_before_build() {
        let mut mech = Mechanism::new();
        let ground = make_ground(&[("O", 0.0, 0.0)]);
        mech.add_body(ground).unwrap();
        assert!(mech.body_order().is_empty());
    }

    #[test]
    fn body_order_excludes_ground() {
        let mech = build_fourbar();
        assert!(
            !mech.body_order().contains(&"ground".to_string()),
            "body_order should not contain ground"
        );
    }

    #[test]
    fn constraint_ranges_match_manual_counting() {
        // Build a 4-bar and verify constraint_ranges matches walking
        // all_constraints() and manually accumulating row offsets.
        let mech = build_fourbar();

        let constraints = mech.all_constraints();
        let ranges = mech.constraint_ranges();

        assert_eq!(
            ranges.len(),
            constraints.len(),
            "constraint_ranges length must match all_constraints length"
        );

        let mut expected_row = 0;
        for (constraint, range) in constraints.iter().zip(ranges.iter()) {
            assert_eq!(range.constraint_id, constraint.id());
            assert_eq!(range.row_start, expected_row);
            assert_eq!(range.n_equations, constraint.n_equations());
            expected_row += constraint.n_equations();
        }

        // Total rows must equal n_constraints()
        assert_eq!(expected_row, mech.n_constraints());
    }

    #[test]
    fn constraint_ranges_for_slider_crank() {
        // Slider-crank: ground + crank + slider
        // Joints: revolute(ground-crank), revolute(crank-slider), prismatic(ground-slider)
        // Driver: constant speed on crank
        let ground = make_ground(&[("O", 0.0, 0.0), ("S", 0.0, 0.0)]);
        let crank = make_bar("crank", "A", "B", 0.1, 0.0, 0.0);
        let slider = make_bar("slider", "P", "Q", 0.2, 0.0, 0.0);

        let mut mech = Mechanism::new();
        mech.add_body(ground).unwrap();
        mech.add_body(crank).unwrap();
        mech.add_body(slider).unwrap();

        // Revolute: ground-crank (2 eq)
        mech.add_revolute_joint("R1", "ground", "O", "crank", "A")
            .unwrap();
        // Revolute: crank-slider (2 eq)
        mech.add_revolute_joint("R2", "crank", "B", "slider", "P")
            .unwrap();
        // Prismatic: ground-slider along x-axis (2 eq)
        mech.add_prismatic_joint(
            "P1",
            "ground",
            "S",
            "slider",
            "Q",
            Vector2::new(1.0, 0.0),
            0.0,
        )
        .unwrap();
        // Driver: constant speed (1 eq)
        mech.add_constant_speed_driver("D1", "ground", "crank", 2.0 * PI, 0.0)
            .unwrap();

        mech.build().unwrap();

        let ranges = mech.constraint_ranges();
        assert_eq!(ranges.len(), 4);

        // R1: revolute, 2 equations, starts at row 0
        assert_eq!(ranges[0].constraint_id, "R1");
        assert_eq!(ranges[0].row_start, 0);
        assert_eq!(ranges[0].n_equations, 2);

        // R2: revolute, 2 equations, starts at row 2
        assert_eq!(ranges[1].constraint_id, "R2");
        assert_eq!(ranges[1].row_start, 2);
        assert_eq!(ranges[1].n_equations, 2);

        // P1: prismatic, 2 equations, starts at row 4
        assert_eq!(ranges[2].constraint_id, "P1");
        assert_eq!(ranges[2].row_start, 4);
        assert_eq!(ranges[2].n_equations, 2);

        // D1: driver, 1 equation, starts at row 6
        assert_eq!(ranges[3].constraint_id, "D1");
        assert_eq!(ranges[3].row_start, 6);
        assert_eq!(ranges[3].n_equations, 1);
    }

    #[test]
    fn body_coord_range_matches_body_index() {
        // Verify body_coord_range agrees with get_index for every body.
        let mech = build_fourbar();
        let state = mech.state();

        for body_id in mech.body_order() {
            let idx = state.get_index(body_id).unwrap();
            let (start, end) = state.body_coord_range(body_id).unwrap();
            assert_eq!(start, idx.q_start);
            assert_eq!(end, idx.q_start + 3);
        }
    }
}
