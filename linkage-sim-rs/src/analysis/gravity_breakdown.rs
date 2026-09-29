//! Per-weight gravity power: which weights help and which hurt the actuator.
//!
//! A *weight source* is either a link's own mass (the blueprint's base
//! `mass` at its base `cg_local`) or a point mass the user attached to a
//! link. Gravity loads are linear in mass, and a built body's composite
//! mass and CG are the mass-weighted sum of its base mass and its point
//! masses (`core::body::Body::add_point_mass`), so the model's gravity load
//! is exactly the sum of the sources' gravity loads.
//!
//! At one pose with velocity `q_dot` (from the constant-rate velocity
//! solve) each source's gravity power is
//!
//! ```text
//! P_g,i = m_i * (g . v_i)     [W]
//! ```
//!
//! where `v_i` is the world velocity of the source's point
//! (`State::body_point_velocity`). `P_g,i > 0` means gravity does positive
//! work on that weight (it is coming down): the weight is **helping** the
//! actuator. The rules that turn powers into shares and labels live here
//! too, next to their named thresholds:
//!
//! - force share `-P_g,i / rate` ([`force_share`]; NaN near stroke
//!   reversal, [`EPS_REL_LDOT`]); the power share is `-P_g,i`,
//! - helping / hurting / neutral per weight ([`classify`], [`NEUTRAL_REL`]),
//! - braking where actuator power is negative ([`is_braking`],
//!   [`BRAKE_TOL_REL`]).
//!
//! The sweep (`gui::sweep`) applies them per sample. Spec:
//! docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md
//! (Track 2, section 2).

use nalgebra::{DVector, Vector2};
use serde::{Deserialize, Serialize};

use crate::core::mechanism::Mechanism;
use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::io::{point_mass_skip_reason, MechanismJson, PointMassJson};

/// Force shares are NaN where `|rate| < EPS_REL_LDOT * max|rate|` over the
/// sweep: near stroke reversal the actuator barely moves, so `-P_g / rate`
/// blows up although the force needed to hold the load does not.
pub const EPS_REL_LDOT: f64 = 0.01;

/// The actuator is braking (the load drives it) where its power is below
/// `-BRAKE_TOL_REL * max|P_act|` over the sweep; the tolerance keeps
/// round-off around zero power from flipping the label.
pub const BRAKE_TOL_REL: f64 = 1e-6;

/// A weight is neutral while `|P_g,i| <= NEUTRAL_REL * max|P_g,i|` over the
/// sweep for that weight (it is momentarily moving horizontally).
pub const NEUTRAL_REL: f64 = 0.01;

/// One weight the breakdown reports on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightSource {
    /// `"link:<body_id>"` for a link self-weight, else the point-mass id.
    pub id: String,
    /// Display name: the body label (or id) for a link self-weight, the
    /// point-mass label (or id) for a point mass.
    pub name: String,
    /// Body the weight rides on.
    pub body_id: String,
    /// Position in the body's local frame (m).
    pub local_pos: [f64; 2],
    /// Mass (kg), positive and finite.
    pub mass: f64,
    /// True for the link's own mass, false for a point mass.
    pub is_link_self_weight: bool,
}

/// Whether gravity on a weight helps or hurts the actuator at one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// Coming down: gravity does positive work (`P_g` above the band).
    Helping,
    /// Going up: the actuator lifts it (`P_g` below minus the band).
    Hurting,
    /// Moving (nearly) horizontally, or no finite power.
    Neutral,
}

/// `label` when it has visible text, else `fallback`: the display name of
/// a weight (label or id) or a link (label or body id).
pub fn display_name(label: Option<&String>, fallback: &str) -> String {
    match label {
        Some(text) if !text.trim().is_empty() => text.clone(),
        _ => fallback.to_string(),
    }
}

/// `name` with `id` in parentheses when they differ ("Robot (W1)"), else
/// just `id`: a display name that stays unique when two labels are equal.
pub fn name_with_id(name: &str, id: &str) -> String {
    if name == id {
        id.to_string()
    } else {
        format!("{name} ({id})")
    }
}

/// Display title of point mass `pm`: "W1", or "Robot torso (W1)" when it
/// has a label (the property panel's weight editors, the canvas readout).
pub fn point_mass_title(pm: &PointMassJson) -> String {
    name_with_id(&display_name(pm.label.as_ref(), &pm.id), &pm.id)
}

/// Every weight in blueprint `bp`, in a fixed order: first one link
/// self-weight per non-ground body whose base mass is positive and finite
/// (bodies sorted by id), then every point mass the loader applies (bodies
/// sorted by id, list order).
///
/// Point masses the loader skips (`io::point_mass_skip_reason`: on ground,
/// mass not positive and finite, non-finite position) are left out, so the
/// sources' gravity loads add up to the gravity load of the mechanism built
/// from `bp`; pass that mechanism to [`gravity_powers`]. A negative base
/// mass is invalid input and gets no source; whatever gravity load it adds
/// to the built mechanism shows up in the breakdown's non-gravity remainder.
pub fn weight_sources(bp: &MechanismJson) -> Vec<WeightSource> {
    let mut body_ids: Vec<&String> = bp.bodies.keys().filter(|id| id.as_str() != GROUND_ID).collect();
    body_ids.sort();

    let mut sources = Vec::new();
    for &body_id in &body_ids {
        let body = &bp.bodies[body_id];
        if body.mass.is_finite() && body.mass > 0.0 {
            sources.push(WeightSource {
                id: format!("link:{body_id}"),
                name: display_name(body.label.as_ref(), body_id),
                body_id: body_id.clone(),
                local_pos: body.cg_local,
                mass: body.mass,
                is_link_self_weight: true,
            });
        }
    }
    for &body_id in &body_ids {
        for pm in &bp.bodies[body_id].point_masses {
            if point_mass_skip_reason(body_id, pm).is_some() {
                continue;
            }
            sources.push(WeightSource {
                id: pm.id.clone(),
                name: display_name(pm.label.as_ref(), &pm.id),
                body_id: body_id.clone(),
                local_pos: pm.local_pos,
                mass: pm.mass,
                is_link_self_weight: false,
            });
        }
    }
    sources
}

/// Gravity power `m_i * (g . v_i)` (W) of each source at pose `q` with
/// velocity `q_dot`; positive = gravity does positive work (helping).
///
/// `mech` must be built from the blueprint the sources came from, and `g`
/// is the gravity vector (m/s^2), normally [`gravity_vector`] of `mech`. A
/// source on ground gets 0 (ground does not move); a source whose body is
/// not in `mech` (stale blueprint) gets NaN instead of a panic.
pub fn gravity_powers(
    mech: &Mechanism,
    sources: &[WeightSource],
    q: &DVector<f64>,
    q_dot: &DVector<f64>,
    g: [f64; 2],
) -> Vec<f64> {
    let state = mech.state();
    let g = Vector2::new(g[0], g[1]);
    sources
        .iter()
        .map(|s| {
            if !state.is_ground(&s.body_id) && state.get_index(&s.body_id).is_err() {
                return f64::NAN;
            }
            let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
            let v = state.body_point_velocity(&s.body_id, &local, q, q_dot);
            s.mass * g.dot(&v)
        })
        .collect()
}

/// The mechanism's gravity vector (m/s^2): the sum of its `Gravity` force
/// elements (the GUI keeps one, rotated by the mounting angle in
/// `AppState::sync_gravity`); zero when gravity is off.
pub fn gravity_vector(mech: &Mechanism) -> [f64; 2] {
    mech.forces().iter().fold([0.0, 0.0], |sum, force| match force {
        ForceElement::Gravity(g) => [sum[0] + g.g_vector[0], sum[1] + g.g_vector[1]],
        _ => sum,
    })
}

/// Largest finite `|value|` in `values`; 0 when none is finite.
pub fn max_abs_finite(values: &[f64]) -> f64 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .fold(0.0, |max, v| max.max(v.abs()))
}

/// Classify one weight at one sample from its gravity power `p_g` and the
/// largest `|P_g|` of the same weight over the sweep: helping above the
/// neutral band `NEUTRAL_REL * max_abs_p_g`, hurting below minus the band,
/// neutral inside it or when `p_g` is not finite.
pub fn classify(p_g: f64, max_abs_p_g: f64) -> Classification {
    if !p_g.is_finite() {
        return Classification::Neutral;
    }
    let band = NEUTRAL_REL * max_abs_p_g;
    if p_g > band {
        Classification::Helping
    } else if p_g < -band {
        Classification::Hurting
    } else {
        Classification::Neutral
    }
}

/// Force share `-p_g / rate` of one weight at one sample.
///
/// `rate` is the actuator extension rate dL/dt (m/s), or the driver rate
/// when the shares are driver-torque shares; `max_abs_rate` is its largest
/// finite `|value|` over the sweep. NaN near stroke reversal
/// (`|rate| < EPS_REL_LDOT * max_abs_rate`) and at a zero or non-finite
/// rate.
pub fn force_share(p_g: f64, rate: f64, max_abs_rate: f64) -> f64 {
    if !rate.is_finite() || rate == 0.0 || rate.abs() < EPS_REL_LDOT * max_abs_rate {
        return f64::NAN;
    }
    -p_g / rate
}

/// True when actuator power `p_act` is negative beyond the tolerance
/// `BRAKE_TOL_REL * max_abs_p_act` (the load drives the actuator); false
/// for NaN.
pub fn is_braking(p_act: f64, max_abs_p_act: f64) -> bool {
    p_act < -BRAKE_TOL_REL * max_abs_p_act
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::elements::{evaluate_gravity, ForceElement, GravityElement};
    use crate::io::{load_mechanism_unbuilt_from_json, MechanismJson, SCHEMA_VERSION};
    use crate::solver::kinematics::{solve_position, solve_velocity};
    use nalgebra::{DVector, Vector2};
    use serde_json::json;

    const G: f64 = 9.81;

    fn build(bp: &MechanismJson) -> Mechanism {
        let mut mech = load_mechanism_unbuilt_from_json(bp).expect("blueprint loads");
        mech.build().expect("mechanism builds");
        mech
    }

    /// Ground plus one bar pinned at the origin (body frame at the pivot, so
    /// its pose is (0, 0, theta)), driven at `omega` rad/s from theta = 0.
    /// Bar: 2 kg base mass at (0.5, 0); weights W1 = 3 kg at the tip (1, 0)
    /// and W2 "Robot" = 1.5 kg at (0.6, 0). Gravity (0, -9.81).
    fn single_bar_blueprint(omega: f64) -> MechanismJson {
        serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {"O": [0.0, 0.0]},
                           "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "bar": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                        "mass": 2.0, "cg_local": [0.5, 0.0], "izz_cg": 0.2,
                        "point_masses": [
                            {"id": "W1", "mass": 3.0, "local_pos": [1.0, 0.0]},
                            {"id": "W2", "label": "Robot", "mass": 1.5, "local_pos": [0.6, 0.0]}
                        ]}
            },
            "joints": {
                "J1": {"type": "revolute", "body_i": "ground", "point_i": "O",
                       "body_j": "bar", "point_j": "A"}
            },
            "drivers": {
                "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "bar",
                       "omega": omega, "theta_0": 0.0}
            },
            "forces": [{"type": "Gravity", "g_vector": [0.0, -G]}]
        }))
        .expect("single-bar blueprint parses")
    }

    /// Sources and gravity powers of the single bar at `theta`, driven at
    /// `omega` (pose solved at driver time theta / omega).
    fn single_bar_powers(theta: f64, omega: f64) -> (Vec<WeightSource>, Vec<f64>) {
        let bp = single_bar_blueprint(omega);
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("bar", &mut q, 0.0, 0.0, theta);
        let t = theta / omega;
        let solved = solve_position(&mech, &q, t, 1e-12, 50).expect("position solve");
        assert!(solved.converged, "bar pose at {theta} rad did not converge");
        let q_dot = solve_velocity(&mech, &solved.q, t).expect("velocity solve");
        let powers = gravity_powers(&mech, &sources, &solved.q, &q_dot, gravity_vector(&mech));
        (sources, powers)
    }

    /// Hand calculation for a bar pivoted at the origin: a point at
    /// body-local (x, 0) moves with v = omega * x * (-sin theta, cos theta),
    /// so with g = (0, -9.81) its gravity power is
    /// P = -m * 9.81 * omega * x * cos(theta). Covers the bar rising
    /// (cos theta > 0, P < 0), falling (cos theta < 0, P > 0) and moving
    /// horizontally (theta = 90, 270 deg, P = 0).
    #[test]
    fn gravity_power_single_bar_matches_hand_calculation() {
        let omega = 2.0;
        // (mass, x) of link:bar, W1, W2 in weight_sources order.
        let points = [(2.0, 0.5), (3.0, 1.0), (1.5, 0.6)];
        for deg in [0.0_f64, 30.0, 60.0, 90.0, 120.0, 180.0, 210.0, 270.0, 300.0, 330.0] {
            let theta = deg.to_radians();
            let (sources, powers) = single_bar_powers(theta, omega);
            assert_eq!(sources.len(), points.len());
            for (i, &(m, x)) in points.iter().enumerate() {
                let hand = -m * G * omega * x * theta.cos();
                assert!(
                    (powers[i] - hand).abs() <= 1e-9 * m * G * omega * x,
                    "{} at {deg} deg: P_g {} vs hand {hand}",
                    sources[i].id,
                    powers[i]
                );
            }
        }
    }

    /// Same pose, opposite direction: going up the weights cost power
    /// (hurting, P_g < 0); coming down gravity gives exactly that power back
    /// (helping, P_g > 0). The key physics statement of the spec.
    #[test]
    fn gravity_power_flips_sign_between_ascending_and_descending() {
        let theta = 30.0_f64.to_radians();
        let (sources, up) = single_bar_powers(theta, 2.0);
        let (_, down) = single_bar_powers(theta, -2.0);
        for i in 0..sources.len() {
            assert!(up[i] < 0.0, "{} rising: P_g {} should be < 0", sources[i].id, up[i]);
            assert!(down[i] > 0.0, "{} falling: P_g {} should be > 0", sources[i].id, down[i]);
            assert!((up[i] + down[i]).abs() <= 1e-12 * up[i].abs(), "{}: |P_g| differs", sources[i].id);
            assert_eq!(classify(up[i], up[i].abs()), Classification::Hurting);
            assert_eq!(classify(down[i], down[i].abs()), Classification::Helping);
        }
    }

    const FOURBAR_OMEGA: f64 = 1.5;
    const MOUNT_DEG: f64 = 30.0;

    /// Crank-rocker 4-bar (crank 1, coupler 3, rocker 2.5, ground 4 m) with
    /// base masses on every link, W1 "Robot" (5 kg, off the coupler line)
    /// and W2 (2 kg at the rocker tip), gravity rotated by a 30 deg mounting
    /// angle so both gravity components matter.
    fn fourbar_blueprint() -> MechanismJson {
        let mount = MOUNT_DEG.to_radians();
        let g = [-G * mount.sin(), -G * mount.cos()];
        serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {"O2": [0.0, 0.0], "O4": [4.0, 0.0]},
                           "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "crank": {"attachment_points": {"A": [0.0, 0.0], "B": [1.0, 0.0]},
                          "mass": 1.0, "cg_local": [0.5, 0.0], "izz_cg": 0.1},
                "coupler": {"attachment_points": {"B": [0.0, 0.0], "C": [3.0, 0.0]},
                            "mass": 3.0, "cg_local": [1.5, 0.0], "izz_cg": 2.25,
                            "point_masses": [
                                {"id": "W1", "label": "Robot", "mass": 5.0, "local_pos": [1.0, 0.4]}
                            ]},
                "rocker": {"attachment_points": {"D": [0.0, 0.0], "C": [2.5, 0.0]},
                           "mass": 2.0, "cg_local": [1.25, 0.0], "izz_cg": 1.0,
                           "point_masses": [
                               {"id": "W2", "mass": 2.0, "local_pos": [2.5, 0.0]}
                           ]}
            },
            "joints": {
                "J1": {"type": "revolute", "body_i": "ground", "point_i": "O2",
                       "body_j": "crank", "point_j": "A"},
                "J2": {"type": "revolute", "body_i": "crank", "point_i": "B",
                       "body_j": "coupler", "point_j": "B"},
                "J3": {"type": "revolute", "body_i": "coupler", "point_i": "C",
                       "body_j": "rocker", "point_j": "C"},
                "J4": {"type": "revolute", "body_i": "ground", "point_i": "O4",
                       "body_j": "rocker", "point_j": "D"}
            },
            "drivers": {
                "D1": {"type": "constant_speed", "body_i": "ground", "body_j": "crank",
                       "omega": FOURBAR_OMEGA, "theta_0": 0.0}
            },
            "forces": [{"type": "Gravity", "g_vector": g}]
        }))
        .expect("4-bar blueprint parses")
    }

    /// Newton solve of the 4-bar at crank angle `theta` from `guess`.
    fn solve_fourbar_near(mech: &Mechanism, guess: &DVector<f64>, theta: f64) -> DVector<f64> {
        let solved = solve_position(mech, guess, theta / FOURBAR_OMEGA, 1e-12, 50).expect("position solve");
        assert!(solved.converged, "4-bar pose at {theta} rad did not converge");
        solved.q
    }

    /// Solved elbow-up pose of the 4-bar at crank angle `theta`, seeded from
    /// the closed-form loop closure.
    fn fourbar_q(mech: &Mechanism, theta: f64) -> DVector<f64> {
        let (bx, by) = (theta.cos(), theta.sin());
        let (dx, dy) = (4.0 - bx, -by);
        let d = (dx * dx + dy * dy).sqrt();
        // Distance from B towards O4 to the foot of C, and C's height above B->O4.
        let a = (d * d + 3.0 * 3.0 - 2.5 * 2.5) / (2.0 * d);
        let h = (3.0 * 3.0 - a * a).sqrt();
        let (ux, uy) = (dx / d, dy / d);
        let (cx, cy) = (bx + a * ux - h * uy, by + a * uy + h * ux);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("crank", &mut q, 0.0, 0.0, theta);
        st.set_pose("coupler", &mut q, bx, by, (cy - by).atan2(cx - bx));
        st.set_pose("rocker", &mut q, 4.0, 0.0, cy.atan2(cx - 4.0));
        solve_fourbar_near(mech, &q, theta)
    }

    /// Potential energy `-m g . r` of source `s` at pose `q`.
    fn potential_energy(mech: &Mechanism, s: &WeightSource, q: &DVector<f64>, g: [f64; 2]) -> f64 {
        let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
        let r = mech.state().body_point_global(&s.body_id, &local, q);
        -s.mass * (g[0] * r.x + g[1] * r.y)
    }

    /// Independent reference: P_g is minus the rate of change of each
    /// weight's potential energy, finite-differenced over a small crank step
    /// (positions only, no velocity solve), for every link self-weight and
    /// point mass of the 4-bar under tilted gravity.
    #[test]
    fn gravity_power_matches_potential_energy_finite_difference() {
        let bp = fourbar_blueprint();
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        assert_eq!(sources.len(), 5, "3 link self-weights + 2 point masses");
        let g = gravity_vector(&mech);
        let h = 1e-3;
        let mut rows: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();
        for deg in (0..360).step_by(20) {
            let theta = (deg as f64).to_radians();
            let q = fourbar_q(&mech, theta);
            let q_dot = solve_velocity(&mech, &q, theta / FOURBAR_OMEGA).expect("velocity solve");
            let q_plus = solve_fourbar_near(&mech, &q, theta + h);
            let q_minus = solve_fourbar_near(&mech, &q, theta - h);
            let powers = gravity_powers(&mech, &sources, &q, &q_dot, g);
            let fd: Vec<f64> = sources
                .iter()
                .map(|s| {
                    let d_pe = potential_energy(&mech, s, &q_plus, g) - potential_energy(&mech, s, &q_minus, g);
                    -FOURBAR_OMEGA * d_pe / (2.0 * h)
                })
                .collect();
            rows.push((deg, powers, fd));
        }
        for (i, s) in sources.iter().enumerate() {
            let peak = rows.iter().fold(0.0_f64, |m, r| m.max(r.1[i].abs()));
            assert!(peak > 1.0, "{}: implausibly small peak gravity power {peak}", s.id);
            for (deg, powers, fd) in &rows {
                assert!(
                    (powers[i] - fd[i]).abs() <= 1e-5 * peak,
                    "{} at {deg} deg: P_g {} vs energy finite difference {}",
                    s.id,
                    powers[i],
                    fd[i]
                );
            }
        }
    }

    /// The link self-weights plus the point masses reproduce the built
    /// mechanism's total gravity power `Q_gravity . q_dot` (composite masses
    /// and CGs), and the point masses are a real part of that sum.
    #[test]
    fn link_self_weights_and_point_masses_sum_to_model_gravity_power() {
        let bp = fourbar_blueprint();
        let mech = build(&bp);
        let sources = weight_sources(&bp);
        let g = gravity_vector(&mech);
        let gravity = GravityElement { g_vector: g };
        let mut point_masses_matter = false;
        for deg in (0..360).step_by(15) {
            let theta = (deg as f64).to_radians();
            let q = fourbar_q(&mech, theta);
            let q_dot = solve_velocity(&mech, &q, theta / FOURBAR_OMEGA).expect("velocity solve");
            let powers = gravity_powers(&mech, &sources, &q, &q_dot, g);
            let model = evaluate_gravity(&gravity, mech.state(), mech.bodies(), &q).dot(&q_dot);
            let sum: f64 = powers.iter().sum();
            let scale: f64 = powers.iter().map(|p| p.abs()).sum::<f64>().max(1e-12);
            assert!(
                (sum - model).abs() <= 1e-9 * scale,
                "{deg} deg: sum of weight powers {sum} vs model gravity power {model}"
            );
            let links_only: f64 = sources
                .iter()
                .zip(&powers)
                .filter(|(s, _)| s.is_link_self_weight)
                .map(|(_, p)| p)
                .sum();
            point_masses_matter |= (links_only - model).abs() > 0.1 * scale;
        }
        assert!(point_masses_matter, "the point masses never changed the total: test is vacuous");
    }

    #[test]
    fn weight_sources_lists_link_self_weights_then_applied_point_masses() {
        let bp: MechanismJson = serde_json::from_value(json!({
            "schema_version": SCHEMA_VERSION,
            "bodies": {
                "ground": {"attachment_points": {}, "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0,
                           "point_masses": [{"id": "W9", "mass": 1.0, "local_pos": [0.0, 0.0]}]},
                "bar_d": {"attachment_points": {}, "mass": -1.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0},
                "bar_c": {"attachment_points": {}, "mass": 1.0, "cg_local": [0.1, 0.2], "izz_cg": 0.0,
                          "point_masses": [
                              {"id": "W4", "mass": 0.0, "local_pos": [0.0, 0.0]},
                              {"id": "W5", "label": "  ", "mass": 2.0, "local_pos": [1.0, 0.0]}
                          ]},
                "bar_a": {"attachment_points": {}, "mass": 0.0, "cg_local": [0.0, 0.0], "izz_cg": 0.0,
                          "point_masses": [{"id": "W3", "mass": 4.0, "local_pos": [0.5, 0.5]}]},
                "bar_b": {"attachment_points": {}, "mass": 2.5, "cg_local": [0.5, 0.0], "izz_cg": 0.0,
                          "label": "Arm",
                          "point_masses": [
                              {"id": "W2", "label": "Robot", "mass": 50.0, "local_pos": [1.0, 0.0]},
                              {"id": "W1", "mass": 3.0, "local_pos": [0.2, 0.0]}
                          ]}
            },
            "joints": {}
        }))
        .expect("blueprint parses");
        let sources = weight_sources(&bp);
        let summary: Vec<(&str, &str, &str, bool)> = sources
            .iter()
            .map(|s| (s.id.as_str(), s.name.as_str(), s.body_id.as_str(), s.is_link_self_weight))
            .collect();
        // Link self-weights first (bodies sorted by id; massless, negative-mass
        // and ground bodies have none), then applied point masses (bodies
        // sorted by id, list order; ground and zero-mass weights skipped).
        assert_eq!(
            summary,
            vec![
                ("link:bar_b", "Arm", "bar_b", true),
                ("link:bar_c", "bar_c", "bar_c", true),
                ("W3", "W3", "bar_a", false),
                ("W2", "Robot", "bar_b", false),
                ("W1", "W1", "bar_b", false),
                ("W5", "W5", "bar_c", false),
            ]
        );
        assert_eq!((sources[1].local_pos, sources[1].mass), ([0.1, 0.2], 1.0), "self-weight at base CG");
        assert_eq!((sources[3].local_pos, sources[3].mass), ([1.0, 0.0], 50.0), "point mass as stored");
    }

    #[test]
    fn gravity_power_is_zero_on_ground_and_nan_for_a_body_missing_from_the_mechanism() {
        let bp = single_bar_blueprint(2.0);
        let mech = build(&bp);
        let st = mech.state();
        let mut q = st.make_q();
        st.set_pose("bar", &mut q, 0.0, 0.0, 0.3);
        let q_dot = solve_velocity(&mech, &q, 0.15).expect("velocity solve");
        let source = |body: &str| WeightSource {
            id: "X".to_string(),
            name: "X".to_string(),
            body_id: body.to_string(),
            local_pos: [1.0, 0.0],
            mass: 1.0,
            is_link_self_weight: false,
        };
        let powers = gravity_powers(&mech, &[source("ground"), source("ghost"), source("bar")], &q, &q_dot, [0.0, -G]);
        assert_eq!(powers[0], 0.0, "ground does not move");
        assert!(powers[1].is_nan(), "a body the mechanism lacks gives NaN, not a panic");
        assert!(powers[2].is_finite() && powers[2] != 0.0);
    }

    #[test]
    fn gravity_vector_reads_the_mechanism_gravity_elements() {
        let mut mech = build(&fourbar_blueprint());
        let mount = MOUNT_DEG.to_radians();
        let g = gravity_vector(&mech);
        assert!((g[0] + G * mount.sin()).abs() < 1e-12 && (g[1] + G * mount.cos()).abs() < 1e-12, "{g:?}");

        mech.add_force(ForceElement::Gravity(GravityElement { g_vector: [1.0, 2.0] }));
        let summed = gravity_vector(&mech);
        assert!((summed[0] - (g[0] + 1.0)).abs() < 1e-12 && (summed[1] - (g[1] + 2.0)).abs() < 1e-12);

        let mut no_gravity = single_bar_blueprint(2.0);
        no_gravity.forces.clear();
        assert_eq!(gravity_vector(&build(&no_gravity)), [0.0, 0.0]);
    }

    #[test]
    fn classify_uses_a_one_percent_neutral_band() {
        use Classification::*;
        assert_eq!(classify(5.0, 100.0), Helping);
        assert_eq!(classify(-5.0, 100.0), Hurting);
        assert_eq!(classify(1.1, 100.0), Helping, "just above the 1 % band");
        assert_eq!(classify(-1.1, 100.0), Hurting, "just below minus the band");
        assert_eq!(classify(0.9, 100.0), Neutral, "inside the band");
        assert_eq!(classify(-0.9, 100.0), Neutral, "inside the band");
        assert_eq!(classify(0.0, 0.0), Neutral, "a weight that never moves vertically");
        assert_eq!(classify(f64::NAN, 100.0), Neutral);
        assert_eq!(classify(f64::INFINITY, 100.0), Neutral);
    }

    #[test]
    fn force_share_divides_by_rate_and_is_nan_near_reversal() {
        assert_eq!(force_share(10.0, 2.0, 4.0), -5.0);
        let above_band = force_share(-10.0, -0.05, 4.0);
        assert!((above_band - (-200.0)).abs() < 1e-9, "|rate| 0.05 >= 1 % of 4: {above_band}");
        assert!(force_share(10.0, 0.03, 4.0).is_nan(), "|rate| < 1 % of max: near reversal");
        assert!(force_share(10.0, -0.03, 4.0).is_nan());
        assert!(force_share(10.0, 0.0, 0.0).is_nan(), "no motion at all");
        assert!(force_share(10.0, f64::NAN, 4.0).is_nan());
        assert!(force_share(10.0, f64::INFINITY, 4.0).is_nan());
        assert!(force_share(f64::NAN, 2.0, 4.0).is_nan());
    }

    #[test]
    fn is_braking_needs_power_below_the_relative_tolerance() {
        assert!(is_braking(-1.0, 100.0));
        assert!(!is_braking(-1e-5, 100.0), "within 1e-6 * 100 of zero");
        assert!(!is_braking(0.0, 100.0));
        assert!(!is_braking(5.0, 100.0));
        assert!(!is_braking(f64::NAN, 100.0));
    }

    #[test]
    fn display_name_prefers_a_visible_label() {
        assert_eq!(display_name(Some(&"Robot".to_string()), "W1"), "Robot");
        assert_eq!(display_name(Some(&"  ".to_string()), "W1"), "W1");
        assert_eq!(display_name(None, "W1"), "W1");
    }

    #[test]
    fn name_with_id_adds_the_id_only_when_the_name_differs() {
        assert_eq!(name_with_id("W1", "W1"), "W1");
        assert_eq!(name_with_id("Robot torso", "W2"), "Robot torso (W2)");
        assert_eq!(name_with_id("Arm", "b2"), "Arm (b2)");
    }

    #[test]
    fn point_mass_title_is_the_id_or_the_label_with_the_id() {
        let mut pm = PointMassJson { id: "W1".to_string(), label: None, mass: 1.0, local_pos: [0.0, 0.0] };
        assert_eq!(point_mass_title(&pm), "W1");
        pm.label = Some("Robot torso".to_string());
        assert_eq!(point_mass_title(&pm), "Robot torso (W1)");
        pm.label = Some(" ".to_string());
        assert_eq!(point_mass_title(&pm), "W1", "a blank label shows the id");
    }

    #[test]
    fn max_abs_finite_skips_non_finite_values() {
        assert_eq!(max_abs_finite(&[1.0, -3.0, f64::NAN, 2.0, f64::INFINITY]), 3.0);
        assert_eq!(max_abs_finite(&[]), 0.0);
        assert_eq!(max_abs_finite(&[f64::NAN, f64::NEG_INFINITY]), 0.0);
    }
}
