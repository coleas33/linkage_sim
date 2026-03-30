//! Four-bar mechanism sample builders.

use nalgebra::{DVector, Vector2};
use std::f64::consts::PI;

use crate::core::body::{make_bar, make_ground, Body, BodyGeometry};
use crate::core::mechanism::Mechanism;
use crate::forces::elements::{ForceElement, ForceZoneElement, LinearActuatorElement};

use super::helpers::{
    attach_driver_to_grounded_revolute_with_theta0, fourbar_initial_q0,
    fourbar_rocker_angle_for_crank,
};

/// Grashof crank-rocker 4-bar linkage.
///
/// Link lengths (meters):
/// - ground: O2=(0,0) → O4=(0.038,0)
/// - crank:   0.01 m
/// - coupler: 0.04 m
/// - rocker:  0.03 m
///
/// Satisfies Grashof condition (shortest + longest < sum of other two):
///   0.01 + 0.04 < 0.038 + 0.03  →  0.05 < 0.068  ✓
pub(super) fn build_fourbar_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Link lengths and ground pivot locations.
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.038_f64, 0.0_f64);
    let l_crank = 0.01_f64;
    let l_coupler = 0.04_f64;
    let l_rocker = 0.03_f64;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

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

    // Determine initial crank angle. For J1-driven the crank angle is the
    // driver's theta_0. For J4-driven we compute a consistent crank angle
    // from the rocker angle via the 4-bar loop closure.
    let joint_id = driver_joint_id.unwrap_or("J1");

    let (theta_crank, theta_0) = match joint_id {
        "J1" => {
            // Default: drive crank at theta_crank = 0.
            (0.0_f64, 0.0_f64)
        }
        "J4" => {
            // Drive rocker. Compute a consistent crank angle from the
            // rocker's initial angle using the 4-bar loop closure.
            let theta_rocker = fourbar_rocker_angle_for_crank(
                o2, o4, l_crank, l_coupler, l_rocker, 0.0,
            );
            let theta_crank_for_j4 = 0.0_f64;
            (theta_crank_for_j4, theta_rocker)
        }
        _ => {
            // Let attach_driver_to_grounded_revolute_with_theta0 validate
            // and produce the proper error for unknown/non-grounded joints.
            (0.0, 0.0)
        }
    };

    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", theta_0)?;

    mech.build().map_err(|e| e.to_string())?;

    // Compute geometrically consistent initial poses from the crank angle.
    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, l_crank, l_coupler, l_rocker, theta_crank,
        "crank", "coupler", "rocker", false,
    );

    Ok((mech, q0))
}

/// Slider-crank linkage with a prismatic joint.
///
/// Link lengths (meters):
/// - crank:   0.01 m
/// - coupler: 0.04 m
/// - slider:  translates along X axis from ground/rail
pub(super) fn build_slider_crank_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let ground = make_ground(&[("O2", 0.0, 0.0), ("rail", 0.0, 0.0)]);
    let crank = make_bar("crank", "A", "B", 0.01, 0.0, 0.0);
    let coupler = make_bar("coupler", "B", "C", 0.04, 0.0, 0.0);

    let mut slider = Body::new("slider");
    slider.add_attachment_point("C", 0.0, 0.0).unwrap();

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(slider).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        .unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        .unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "slider", "C")
        .unwrap();

    mech.add_prismatic_joint(
        "P1",
        "ground",
        "rail",
        "slider",
        "C",
        nalgebra::Vector2::new(1.0, 0.0),
        0.0,
    )
    .unwrap();

    // Slider-crank only has J1 as grounded revolute; theta_0 = 0.0 (crank at theta=0).
    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let state = mech.state();
    let mut q0 = state.make_q();
    state.set_pose("crank", &mut q0, 0.005, 0.0, 0.0);
    state.set_pose("coupler", &mut q0, 0.025, 0.0, 0.0);
    state.set_pose("slider", &mut q0, 0.05, 0.0, 0.0);

    Ok((mech, q0))
}

/// Build a standard 4-bar linkage (d=ground, a=crank, b=coupler, c=rocker)
/// with ground pivots at O2=(0,0) and O4=(d,0).
///
/// A coupler point P is placed on the coupler at local x = coupler_point_x, y = 0
/// (i.e. at the midpoint if coupler_point_x = l_coupler/2).
///
/// `theta_crank_init` sets the initial crank angle. Use 0.0 for Grashof linkages
/// where the loop always closes at theta=0. For non-Grashof linkages the loop
/// may be geometrically invalid at theta=0 (e.g. rocker too long to reach the
/// crank tip), so pass a valid starting angle (e.g. PI/2).
///
/// The driver defaults to J1 (grounded revolute at O2, drives the crank).
fn build_standard_fourbar(
    crank_id: &str,
    coupler_id: &str,
    rocker_id: &str,
    l_ground: f64,
    l_crank: f64,
    l_coupler: f64,
    l_rocker: f64,
    coupler_point_x: f64,
    theta_crank_init: f64,
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (l_ground, 0.0_f64);

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar(crank_id, "A", "B", l_crank, 0.0, 0.0);
    let mut coupler = make_bar(coupler_id, "B", "C", l_coupler, 0.0, 0.0);
    coupler
        .add_coupler_point("P", coupler_point_x, 0.0)
        .unwrap();
    let rocker = make_bar(rocker_id, "C", "D", l_rocker, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", crank_id, "A")
        .unwrap();
    mech.add_revolute_joint("J2", crank_id, "B", coupler_id, "B")
        .unwrap();
    mech.add_revolute_joint("J3", coupler_id, "C", rocker_id, "C")
        .unwrap();
    mech.add_revolute_joint("J4", rocker_id, "D", "ground", "O4")
        .unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(
        &mut mech,
        joint_id,
        "D1",
        theta_crank_init,
    )?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(),
        o2,
        o4,
        l_crank,
        l_coupler,
        l_rocker,
        theta_crank_init,
        crank_id,
        coupler_id,
        rocker_id,
        false,
    );

    Ok((mech, q0))
}

/// Grashof crank-rocker 4-bar: d=4, a=2, b=4, c=3.
///
/// Grashof condition satisfied (shortest=2, longest=4): 2+4 < 4+3 → 6 < 7 ✓
/// Crank (a=2, shortest link grounded via O2) can rotate continuously.
/// Coupler point P at (2.0, 0) on coupler.
pub(super) fn build_crank_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 4.0, 3.0, 2.0,
        0.0,
        driver_joint_id,
    )
}

/// Non-Grashof double-rocker 4-bar: d=5, a=3, b=4, c=7.
///
/// No link can rotate fully; both input and output oscillate.
/// Coupler point P at (2.0, 0) on coupler.
/// Uses theta_crank = PI/2 as initial angle because at theta=0 the crank tip is
/// only 2 units from O4, which is less than |b-c| = 3 and the triangle cannot close.
pub(super) fn build_double_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        5.0, 3.0, 4.0, 7.0, 2.0,
        PI / 2.0,
        driver_joint_id,
    )
}

/// Grashof double-crank (drag-link) 4-bar: d=2, a=4, b=3.5, c=3.
///
/// Ground link is shortest: 2+3.5 < 4+3 → 5.5 < 7 ✓ (Grashof, ground shortest → double-crank).
/// Both crank and rocker rotate fully.
/// Coupler point P at (1.75, 0) on coupler.
pub(super) fn build_double_crank_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        2.0, 4.0, 3.5, 3.0, 1.75,
        0.0,
        driver_joint_id,
    )
}

/// Grashof parallelogram 4-bar: d=4, a=2, b=4, c=2.
///
/// Opposite links equal (a=c=2, b=d=4). Coupler translates without rotating.
/// Coupler point P at (2.0, 0) on coupler.
pub(super) fn build_parallelogram_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 4.0, 2.0, 2.0,
        0.0,
        driver_joint_id,
    )
}

/// Parallelogram 4-bar with a rectangular press plate on the coupler
/// passing through a vertical force zone.
///
/// Demonstrates: body geometry, force zones, crank angle limits.
/// Link lengths: ground=4, crank=2, coupler=4, rocker=2 (all x0.01m = cm scale)
pub(super) fn build_parallelogram_press(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.04_f64, 0.0_f64);
    let l_crank = 0.02_f64;
    let l_coupler = 0.04_f64;
    let l_rocker = 0.02_f64;

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);

    let mut coupler = make_bar("coupler", "B", "C", l_coupler, 0.0, 0.0);
    coupler.label = "Coupler (Press Plate)".to_string();
    coupler.geometry = Some(
        BodyGeometry::new(0.06, 0.015, Vector2::new(0.02, 0.0))
            .expect("valid geometry dimensions"),
    );
    // Mass for the press plate
    let plate_mass = 0.5_f64;
    let plate_w = 0.06_f64;
    let plate_h = 0.015_f64;
    coupler.mass = plate_mass;
    coupler.izz_cg = (1.0 / 12.0) * plate_mass * (plate_w * plate_w + plate_h * plate_h);
    coupler.add_coupler_point("P", 0.02, 0.0).unwrap();

    let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

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

    // Force zone: vertical downward force in the working region
    mech.add_force(ForceElement::ForceZone(ForceZoneElement {
        body_id: "coupler".to_string(),
        zone_min: [0.01, -0.005],
        zone_max: [0.03, 0.01],
        force: [0.0, -500.0],
        label: Some("Force Zone".to_string()),
    }));

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(),
        o2,
        o4,
        l_crank,
        l_coupler,
        l_rocker,
        0.0,
        "crank",
        "coupler",
        "rocker",
        false,
    );

    Ok((mech, q0))
}

/// Parallelogram 4-bar with a linear actuator driving the crank.
///
/// Same geometry as `Parallelogram` (d=4, a=2, b=4, c=2) but with:
/// - A mount point "M" at the crank midpoint (1.0, 0.0) in crank-local coords
/// - A new ground pivot "O_act" at (-1.0, -1.5) for the actuator base
/// - A linear actuator from ground "O_act" to crank mount "M"
///
/// The actuator triggers compound force expansion (cylinder + rod bodies).
pub(super) fn build_parallelogram_actuator(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (4.0_f64, 0.0_f64);

    let mut ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    // Actuator base pivot — offset below and behind the crank pivot.
    ground
        .add_attachment_point("O_act", -1.0, -1.5)
        .map_err(|e| e.to_string())?;

    let mut crank = make_bar("crank", "A", "B", 2.0, 0.0, 0.0);
    // Mount point at crank midpoint for the actuator.
    crank
        .add_mount_point("M", 1.0, 0.0)
        .map_err(|e| e.to_string())?;

    let mut coupler = make_bar("coupler", "B", "C", 4.0, 0.0, 0.0);
    coupler.add_coupler_point("P", 2.0, 0.0).unwrap();
    let rocker = make_bar("rocker", "C", "D", 2.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Linear actuator: ground "O_act" → crank mount "M".
    // Uses mount_point_name so compound expansion kicks in on serialization.
    mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
        body_a: "ground".to_string(),
        point_a: [-1.0, -1.5],
        point_a_name: Some("O_act".to_string()),
        body_b: "crank".to_string(),
        point_b: [1.0, 0.0],
        point_b_name: Some("M".to_string()),
        force: 50.0,
        speed_limit: 0.0,
        stroke_min: 0.0, stroke_max: 0.0,
        end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
    }));

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4, 2.0, 4.0, 2.0, 0.0,
        "crank", "coupler", "rocker", false,
    );

    Ok((mech, q0))
}

/// Chebyshev lambda straight-line mechanism (cognate of the ordinary Chebyshev).
///
/// Proportions: A₀A : AB : B₀B : BM : A₀B₀ = 1 : 2.5 : 2.5 : 2.5 : 2
/// With a=2: crank=2, coupler(AB)=5, rocker=5, extension(BM)=5, ground=4.
///
/// The 4-bar loop (ground=4, crank=2, AB=5, rocker=5) is identical to the
/// ordinary Chebyshev. The difference: the coupler extends 5 units past
/// the rocker joint to point M, which traces an approximate straight line.
///
/// Grashof: 2+5 < 5+4 → 7 < 9 ✓ (crank-rocker, full rotation).
pub(super) fn build_chebyshev_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (4.0_f64, 0.0_f64);

    let ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    let crank = make_bar("crank", "A", "B", 2.0, 0.0, 0.0);

    // Lambda coupler: bar rendered from B(0,0) to M(10,0).
    // C at (5,0) is the rocker attachment (intermediate on the bar).
    // M at (10,0) is the straight-line tracing endpoint.
    let mut coupler = make_bar("coupler", "B", "M", 10.0, 0.0, 0.0);
    coupler
        .add_attachment_point("C", 5.0, 0.0)
        .map_err(|e| e.to_string())?;
    coupler.add_coupler_point("M", 10.0, 0.0).unwrap();

    let rocker = make_bar("rocker", "C", "D", 5.0, 0.0, 0.0);

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    let joint_id = driver_joint_id.unwrap_or("J1");
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, joint_id, "D1", 0.0)?;

    mech.build().map_err(|e| e.to_string())?;

    // The 4-bar loop uses AB=5 (B to C distance) as the coupler length.
    let q0 = fourbar_initial_q0(
        mech.state(), o2, o4,
        2.0, 5.0, 5.0,
        0.0,
        "crank", "coupler", "rocker",
        false,
    );

    Ok((mech, q0))
}

/// Non-Grashof triple-rocker 4-bar: d=4, a=2, b=5, c=2.
///
/// No link satisfies Grashof (shortest+longest = 2+5 = 7, sum of others = 4+2 = 6; 7 > 6).
/// No link can make a full revolution; all three moving links oscillate.
/// Coupler point P at (2.5, 0) on coupler.
/// Uses theta_crank = PI/2 as initial angle because at theta=0 the crank tip is
/// only 2 units from O4, which is less than |b-c| = 3 and the triangle cannot close.
pub(super) fn build_triple_rocker_with_driver(
    driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    build_standard_fourbar(
        "crank", "coupler", "rocker",
        4.0, 2.0, 5.0, 2.0, 2.5,
        PI / 2.0,
        driver_joint_id,
    )
}

/// Compute the global position of coupler endpoint M for a given crank angle.
///
/// Solves the 4-bar loop closure analytically (above=true branch) and returns
/// (mx, my) in global coordinates.
pub(super) fn chebyshev_lambda_m_position(
    o2: (f64, f64),
    o4: (f64, f64),
    l_crank: f64,
    l_coupler_ab: f64,
    l_rocker: f64,
    l_total_coupler: f64,
    theta_crank: f64,
) -> (f64, f64) {
    let bx = o2.0 + l_crank * theta_crank.cos();
    let by = o2.1 + l_crank * theta_crank.sin();
    let dx = bx - o4.0;
    let dy = by - o4.1;
    let d = (dx * dx + dy * dy).sqrt();
    let alpha = dy.atan2(dx);
    let cos_beta = (d * d + l_rocker * l_rocker - l_coupler_ab * l_coupler_ab)
        / (2.0 * d * l_rocker);
    let cos_beta = cos_beta.clamp(-1.0, 1.0);
    let beta = cos_beta.acos();
    // above=true branch: alpha - beta + PI
    let theta_rocker = alpha - beta + PI;
    let cx = o4.0 - l_rocker * theta_rocker.cos();
    let cy = o4.1 - l_rocker * theta_rocker.sin();
    let theta_coupler = (cy - by).atan2(cx - bx);
    let mx = bx + l_total_coupler * theta_coupler.cos();
    let my = by + l_total_coupler * theta_coupler.sin();
    (mx, my)
}

/// Chebyshev lambda linkage with a linear actuator force element.
///
/// Custom proportions: ground=76mm, crank=44.4mm, coupler AB=91.9mm,
/// rocker=91.9mm, extension BM=91.9mm. Flipped to +y orientation.
///
/// The actuator base is placed below and to the left of the mechanism
/// (at ground level y=0) so the actuator pivots visibly as M traces its
/// approximate straight line. Stroke limits are computed by sweeping the
/// full 360-degree crank rotation to find the true min/max distance.
///
/// Driven by a standard revolute driver on J1 (like all other samples).
/// The LinearActuator is a force element only (not a driver constraint).
pub(super) fn build_chebyshev_lambda_actuator(
    _driver_joint_id: Option<&str>,
) -> Result<(Mechanism, DVector<f64>), String> {
    // Link lengths (meters).
    let o2 = (0.0_f64, 0.0_f64);
    let o4 = (0.076_f64, 0.0_f64);
    let l_crank = 0.0444_f64;
    let l_coupler_ab = 0.0919_f64; // B->C distance (4-bar loop coupler)
    let l_rocker = 0.0919_f64;
    let l_total_coupler = 0.1838_f64; // B->M (full lambda coupler)
    let theta_crank = 0.0_f64;

    // Lambda coupler: bar rendered from B(0,0) to M(l_total_coupler,0).
    // C at (l_coupler_ab,0) is the rocker attachment (intermediate point).
    // M at (l_total_coupler,0) is the straight-line tracing endpoint.
    let mut coupler = make_bar("coupler", "B", "M", l_total_coupler, 0.0, 0.0);
    coupler
        .add_attachment_point("C", l_coupler_ab, 0.0)
        .map_err(|e| e.to_string())?;
    coupler
        .add_mount_point("M_mount", l_total_coupler, 0.0)
        .map_err(|e| e.to_string())?;
    coupler.add_coupler_point("M", l_total_coupler, 0.0).unwrap();

    let crank = make_bar("crank", "A", "B", l_crank, 0.0, 0.0);
    let rocker = make_bar("rocker", "C", "D", l_rocker, 0.0, 0.0);

    // Compute average M y-position across the full crank rotation so the
    // actuator base is in line with the straight-line trace of M.
    let mut y_sum = 0.0_f64;
    for deg in 0..360 {
        let tc = (deg as f64).to_radians();
        let (_mx, my) = chebyshev_lambda_m_position(
            o2, o4, l_crank, l_coupler_ab, l_rocker, l_total_coupler, tc,
        );
        y_sum += my;
    }
    let avg_my = y_sum / 360.0;

    // Actuator base: to the left of the mechanism, at the average M height.
    // This keeps the actuator roughly in line with M's straight-line trace
    // rather than down at ground level (y=0).
    let act_base_x = -0.15_f64;
    let act_base_y = avg_my;

    let mut ground = make_ground(&[("O2", o2.0, o2.1), ("O4", o4.0, o4.1)]);
    ground
        .add_attachment_point("O_act", act_base_x, act_base_y)
        .map_err(|e| e.to_string())?;

    let mut mech = Mechanism::new();
    mech.add_body(ground).unwrap();
    mech.add_body(crank).unwrap();
    mech.add_body(coupler).unwrap();
    mech.add_body(rocker).unwrap();

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A").unwrap();
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B").unwrap();
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C").unwrap();
    mech.add_revolute_joint("J4", "rocker", "D", "ground", "O4").unwrap();

    // Standard revolute driver on J1 (crank), like all other samples.
    attach_driver_to_grounded_revolute_with_theta0(&mut mech, "J1", "D1", 0.0)?;

    // Compute stroke limits by sweeping the full crank rotation (1-degree steps).
    // The actuator length varies non-monotonically with crank angle, so we must
    // check all angles to find the true extremes.
    let mut stroke_min = f64::MAX;
    let mut stroke_max = f64::NEG_INFINITY;
    for deg in 0..360 {
        let tc = (deg as f64).to_radians();
        let (mx_i, my_i) = chebyshev_lambda_m_position(
            o2, o4, l_crank, l_coupler_ab, l_rocker, l_total_coupler, tc,
        );
        let dist = ((mx_i - act_base_x).powi(2) + (my_i - act_base_y).powi(2)).sqrt();
        if dist < stroke_min {
            stroke_min = dist;
        }
        if dist > stroke_max {
            stroke_max = dist;
        }
    }

    // Linear actuator force element between ground base and coupler M.
    mech.add_force(ForceElement::LinearActuator(LinearActuatorElement {
        body_a: "ground".to_string(),
        point_a: [act_base_x, act_base_y],
        point_a_name: Some("O_act".to_string()),
        body_b: "coupler".to_string(),
        point_b: [l_total_coupler, 0.0],
        point_b_name: Some("M_mount".to_string()),
        force: 2225.0,
        speed_limit: 0.0,
        stroke_min,
        stroke_max,
        end_stop_stiffness: 10000.0,
        end_stop_damping: 10.0,
        end_stop_restitution: 0.5,
    }));

    mech.build().map_err(|e| e.to_string())?;

    // Compute initial poses with above=true for +y orientation.
    // Pass the 4-bar loop coupler length (l_coupler_ab), not the total coupler.
    let q0 = fourbar_initial_q0(
        mech.state(),
        o2,
        o4,
        l_crank,
        l_coupler_ab,
        l_rocker,
        theta_crank,
        "crank",
        "coupler",
        "rocker",
        true,
    );

    Ok((mech, q0))
}
