//! 4-bar linkage detection, shared by the sweep (transmission angle, Grashof)
//! and the HTML report (Freudenstein and friends).

use nalgebra::{DVector, Vector2};

use crate::core::constraint::{Constraint, JointConstraint, RevoluteJoint};
use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;

/// Topology and link lengths of a planar 4-bar (crank, coupler, rocker, ground).
///
/// Lengths come from the revolute joint anchors, not from the bodies'
/// attachment-point lists, so a coupler point or an extra ground reference
/// point does not stop detection or change a length.
#[derive(Debug, Clone)]
pub(crate) struct FourbarGeometry {
    pub crank_id: String,
    pub coupler_id: String,
    pub rocker_id: String,
    /// Ground anchor of the crank pivot (J1), ground-local = world.
    pub o2: Vector2<f64>,
    /// Crank-side anchor of the crank-coupler joint (J2), crank-local.
    pub b_on_crank: Vector2<f64>,
    /// Rocker-side anchor of the coupler-rocker joint (J3), rocker-local.
    pub c_on_rocker: Vector2<f64>,
    /// Ground anchor of the rocker pivot (J4), ground-local = world.
    pub o4: Vector2<f64>,
    /// Crank length |O2 B|.
    pub a: f64,
    /// Coupler length |B C|.
    pub b: f64,
    /// Rocker length |C O4|.
    pub c: f64,
    /// Ground length |O2 O4|.
    pub d: f64,
}

impl FourbarGeometry {
    /// World positions of the four joints (O2, B, C, O4) at pose `q`.
    pub fn world_points(
        &self,
        mech: &Mechanism,
        q: &DVector<f64>,
    ) -> [Vector2<f64>; 4] {
        let st = mech.state();
        [
            self.o2,
            st.body_point_global(&self.crank_id, &self.b_on_crank, q),
            st.body_point_global(&self.rocker_id, &self.c_on_rocker, q),
            self.o4,
        ]
    }
}

/// Anchor of a revolute joint in the local frame of `body`.
fn anchor_on(joint: &RevoluteJoint, body: &str) -> Vector2<f64> {
    if joint.body_i_id() == body {
        *joint.point_i_local()
    } else {
        *joint.point_j_local()
    }
}

/// Detect a planar 4-bar: ground + 3 moving bodies, exactly 4 joints (all
/// revolute), exactly 1 driver (revolute, between ground and the crank) and
/// no linear driver. Returns `None` for any other topology (slider-cranks,
/// 6-bars, multi-driver mechanisms, non-revolute joints).
pub(crate) fn detect_fourbar(mech: &Mechanism) -> Option<FourbarGeometry> {
    // 4 bodies total = ground + 3 moving (crank, coupler, rocker).
    if mech.bodies().len() != 4 {
        return None;
    }
    let joints = mech.joints();
    if joints.len() != 4 || !joints.iter().all(|j| matches!(j, JointConstraint::Revolute(_))) {
        return None;
    }
    if mech.drivers().len() != 1 || !mech.linear_drivers().is_empty() {
        return None;
    }

    let drv = mech.drivers().first()?;
    // Crank is the non-ground side of the driver.
    let crank_id = if drv.body_i_id() == GROUND_ID {
        drv.body_j_id().to_string()
    } else if drv.body_j_id() == GROUND_ID {
        drv.body_i_id().to_string()
    } else {
        return None;
    };

    let find_revolute = |a: &str, b: &str| {
        joints.iter().find_map(|j| match j {
            JointConstraint::Revolute(r)
                if (r.body_i_id() == a && r.body_j_id() == b)
                    || (r.body_i_id() == b && r.body_j_id() == a) =>
            {
                Some(r)
            }
            _ => None,
        })
    };

    // Rocker = body connected to ground by a revolute joint, other than the crank.
    let rocker_id = joints.iter().find_map(|j| {
        let JointConstraint::Revolute(r) = j else { return None };
        let other = if r.body_i_id() == GROUND_ID {
            r.body_j_id()
        } else if r.body_j_id() == GROUND_ID {
            r.body_i_id()
        } else {
            return None;
        };
        (other != crank_id).then(|| other.to_string())
    })?;

    // Coupler is the remaining body.
    let coupler_id = mech
        .body_order()
        .iter()
        .find(|b| **b != crank_id && **b != rocker_id && b.as_str() != GROUND_ID)
        .cloned()?;

    let j1 = find_revolute(GROUND_ID, &crank_id)?;
    let j2 = find_revolute(&crank_id, &coupler_id)?;
    let j3 = find_revolute(&coupler_id, &rocker_id)?;
    let j4 = find_revolute(&rocker_id, GROUND_ID)?;

    let o2 = anchor_on(j1, GROUND_ID);
    let o4 = anchor_on(j4, GROUND_ID);
    let b_on_crank = anchor_on(j2, &crank_id);
    let c_on_rocker = anchor_on(j3, &rocker_id);

    Some(FourbarGeometry {
        a: (b_on_crank - anchor_on(j1, &crank_id)).norm(),
        b: (anchor_on(j2, &coupler_id) - anchor_on(j3, &coupler_id)).norm(),
        c: (c_on_rocker - anchor_on(j4, &rocker_id)).norm(),
        d: (o4 - o2).norm(),
        crank_id,
        coupler_id,
        rocker_id,
        o2,
        b_on_crank,
        c_on_rocker,
        o4,
    })
}

/// (crank, coupler, rocker, ground) link lengths of a detected 4-bar.
pub(crate) fn detect_fourbar_links(mech: &Mechanism) -> Option<(f64, f64, f64, f64)> {
    detect_fourbar(mech).map(|g| (g.a, g.b, g.c, g.d))
}
