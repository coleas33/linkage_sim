//! Payload weights (spec Track 2, section 2): independent reference for the
//! per-weight gravity power on the two actuator samples.
//!
//! Probe method from the spec: each weight's `P_g = m g . v` (velocity from
//! the solver's velocity solve) must equal minus the rate of change of its
//! potential energy `-m g . r`, finite-differenced over a small crank step
//! from positions alone. The link self-weights plus the point masses must
//! also add up to the built mechanism's total gravity power
//! `Q_gravity . q_dot` (composite masses; the massless compound actuator
//! bodies a rebuild adds must not break the sum).

use linkage_sim_rs::analysis::gravity_breakdown::{
    gravity_powers, gravity_vector, weight_sources, WeightSource,
};
use linkage_sim_rs::core::mechanism::Mechanism;
use linkage_sim_rs::forces::elements::{evaluate_gravity, GravityElement};
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;
use linkage_sim_rs::solver::kinematics::{solve_position, solve_velocity};
use nalgebra::{DVector, Vector2};

/// Crank step for the central difference (rad).
const H: f64 = 1e-3;

/// Load `sample` and add `weights` (body, kg, body-local position in m);
/// each add rebuilds the mechanism from the blueprint.
fn sample_with_weights(sample: SampleMechanism, weights: &[(&str, f64, [f64; 2])]) -> AppState {
    let mut state = AppState::default();
    state.load_sample(sample);
    for &(body, mass, pos) in weights {
        state.add_point_mass(body, mass, pos).expect("weight added");
    }
    state.sync_gravity();
    state
}

/// Newton solve at driver angle `theta` from `guess`.
fn solve_at(state: &AppState, mech: &Mechanism, guess: &DVector<f64>, theta: f64) -> DVector<f64> {
    let t = (theta - state.driver_theta_0()) / state.driver_omega();
    let solved = solve_position(mech, guess, t, 1e-12, 50).expect("position solve");
    assert!(solved.converged, "no pose at {theta} rad");
    solved.q
}

/// Potential energy `-m g . r` of source `s` at pose `q`.
fn potential_energy(mech: &Mechanism, s: &WeightSource, q: &DVector<f64>, g: [f64; 2]) -> f64 {
    let local = Vector2::new(s.local_pos[0], s.local_pos[1]);
    let r = mech.state().body_point_global(&s.body_id, &local, q);
    -s.mass * (g[0] * r.x + g[1] * r.y)
}

struct Row {
    deg: usize,
    powers: Vec<f64>,
    energy_fd: Vec<f64>,
    model_total: f64,
}

fn assert_matches_energy_reference(sample: SampleMechanism, weights: &[(&str, f64, [f64; 2])]) {
    let state = sample_with_weights(sample, weights);
    let mech = state.mechanism.as_ref().expect("mechanism built");
    let sources = weight_sources(state.blueprint.as_ref().expect("blueprint"));
    assert_eq!(
        sources.iter().filter(|s| !s.is_link_self_weight).count(),
        weights.len(),
        "{sample:?}: every added weight is a source"
    );
    assert!(sources.iter().any(|s| s.is_link_self_weight), "{sample:?}: links have mass");
    let g = gravity_vector(mech);
    assert!(g[0].abs() < 1e-12 && (g[1] + 9.81).abs() < 1e-12, "{sample:?}: gravity {g:?}");
    let gravity = GravityElement { g_vector: g };
    let omega = state.driver_omega();

    // Odd multiples of 5 deg: the parallelogram is a change point (all links
    // collinear, velocity not unique) at exactly 0 and 180 deg.
    let mut q = state.q.clone();
    let mut rows = Vec::new();
    for deg in (5..360).step_by(10) {
        let theta = (deg as f64).to_radians();
        q = solve_at(&state, mech, &q, theta);
        let t = (theta - state.driver_theta_0()) / omega;
        let q_dot = solve_velocity(mech, &q, t).expect("velocity solve");
        let q_plus = solve_at(&state, mech, &q, theta + H);
        let q_minus = solve_at(&state, mech, &q, theta - H);
        let energy_fd = sources
            .iter()
            .map(|s| {
                let d_pe = potential_energy(mech, s, &q_plus, g) - potential_energy(mech, s, &q_minus, g);
                -omega * d_pe / (2.0 * H)
            })
            .collect();
        rows.push(Row {
            deg,
            powers: gravity_powers(mech, &sources, &q, &q_dot, g),
            energy_fd,
            model_total: evaluate_gravity(&gravity, mech.state(), mech.bodies(), &q).dot(&q_dot),
        });
    }

    for (i, s) in sources.iter().enumerate() {
        let peak = rows.iter().fold(0.0_f64, |m, r| m.max(r.powers[i].abs()));
        assert!(peak > 0.0, "{sample:?} {}: weight never moves vertically", s.id);
        for r in &rows {
            assert!(
                (r.powers[i] - r.energy_fd[i]).abs() <= 1e-5 * peak,
                "{sample:?} {} at {} deg: P_g {} vs energy finite difference {}",
                s.id,
                r.deg,
                r.powers[i],
                r.energy_fd[i]
            );
        }
    }
    for r in &rows {
        let sum: f64 = r.powers.iter().sum();
        let scale: f64 = r.powers.iter().map(|p| p.abs()).sum::<f64>().max(1e-12);
        assert!(
            (sum - r.model_total).abs() <= 1e-9 * scale,
            "{sample:?} at {} deg: sum of weight powers {sum} vs model gravity power {}",
            r.deg,
            r.model_total
        );
    }
}

#[test]
fn parallelogram_actuator_gravity_power_matches_energy_reference() {
    assert_matches_energy_reference(
        SampleMechanism::ParallelogramActuator,
        &[("rocker", 50.0, [0.0, 0.0]), ("coupler", 20.0, [2.0, 0.5])],
    );
}

#[test]
fn chebyshev_lambda_actuator_gravity_power_matches_energy_reference() {
    assert_matches_energy_reference(
        SampleMechanism::ChebyshevLambdaActuator,
        &[("coupler", 50.0, [0.1838, 0.0]), ("rocker", 5.0, [0.0, 0.02])],
    );
}
