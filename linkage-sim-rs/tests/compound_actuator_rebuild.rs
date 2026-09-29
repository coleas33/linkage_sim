//! BL-022 regression: a no-op rebuild must not change actuator physics.
//!
//! `load_sample` keeps the sample's direct mechanism (actuator acting
//! pin-to-pin between the original bodies), but its blueprint keeps the
//! mount-point names, so the first `rebuild()` expands the actuator into a
//! compound cylinder + rod (`forces/compound.rs`). The expanded actuator
//! must see the same length (the pin-to-pin distance) and produce the same
//! sweep forces as the freshly loaded mechanism, in both stored-force and
//! sizing (`force = 0`) mode, and its end-stop penalty must stay inactive
//! while the pins are inside the stroke limits.

use linkage_sim_rs::forces::elements::{
    evaluate_linear_actuator, ForceElement, LinearActuatorElement,
};
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::{AppState, SweepData};
use nalgebra::{DVector, Vector2};

/// Relative tolerance from the BL-022 acceptance criterion.
const REL_TOL: f64 = 1e-6;
/// Absolute floor for the pose-level length check (m).
const ABS_FLOOR: f64 = 1e-9;
/// Series samples near a zero crossing are compared against this fraction
/// of the series' peak |value| instead of relatively against ~0: both
/// mechanisms solve positions to 1e-10, so e.g. a 0.003 N sample of a
/// ~1000 N sizing-force series legitimately differs by ~1e-8 N.
const PEAK_FLOOR: f64 = 1e-9;

fn first_actuator(state: &AppState) -> LinearActuatorElement {
    state
        .mechanism
        .as_ref()
        .expect("mechanism built")
        .forces()
        .iter()
        .find_map(|f| match f {
            ForceElement::LinearActuator(la) => Some(la.clone()),
            _ => None,
        })
        .expect("mechanism has a LinearActuator")
}

/// Set the actuator's stored force to zero (sizing mode) in the live
/// mechanism AND the blueprint, so the fresh and rebuilt mechanisms
/// describe the same model.
fn enter_sizing_mode(state: &mut AppState) {
    for f in state.mechanism.as_mut().unwrap().forces_mut() {
        if let ForceElement::LinearActuator(la) = f {
            la.force = 0.0;
        }
    }
    for f in &mut state.blueprint.as_mut().unwrap().forces {
        if let ForceElement::LinearActuator(la) = f {
            la.force = 0.0;
        }
    }
}

/// World distance between the ORIGINAL actuator pins (captured from the
/// freshly loaded, un-expanded mechanism) at the state's current pose.
fn pin_to_pin_length(state: &AppState, pins: &LinearActuatorElement) -> f64 {
    let mech = state.mechanism.as_ref().unwrap();
    let st = mech.state();
    let a = st.body_point_global(&pins.body_a, &Vector2::new(pins.point_a[0], pins.point_a[1]), &state.q);
    let b = st.body_point_global(&pins.body_b, &Vector2::new(pins.point_b[0], pins.point_b[1]), &state.q);
    (b - a).norm()
}

/// Length the (possibly remapped) actuator element itself sees at the
/// state's current pose.
fn element_length(state: &AppState, la: &LinearActuatorElement) -> f64 {
    let mech = state.mechanism.as_ref().unwrap();
    let st = mech.state();
    let a = st.body_point_global(&la.body_a, &Vector2::new(la.point_a[0], la.point_a[1]), &state.q);
    let b = st.body_point_global(&la.body_b, &Vector2::new(la.point_b[0], la.point_b[1]), &state.q);
    (b - a).norm()
}

fn assert_series_match(name: &str, fresh: &[f64], rebuilt: &[f64], angles: &[f64]) {
    assert_eq!(fresh.len(), rebuilt.len(), "{name}: series length differs");
    let peak = fresh.iter().filter(|v| v.is_finite()).fold(0.0_f64, |m, v| m.max(v.abs()));
    let floor = PEAK_FLOOR * peak;
    let mut n_finite = 0;
    for (i, (&f, &r)) in fresh.iter().zip(rebuilt).enumerate() {
        if f.is_nan() || r.is_nan() {
            assert!(
                f.is_nan() && r.is_nan(),
                "{name}[{i}] at {} deg: fresh {f} vs rebuilt {r} (only one is NaN)",
                angles[i]
            );
            continue;
        }
        let tol = REL_TOL * f.abs().max(r.abs()) + floor;
        assert!(
            (f - r).abs() <= tol,
            "{name}[{i}] at {} deg: fresh {f} vs rebuilt {r} (|diff| {} > tol {tol})",
            angles[i],
            (f - r).abs()
        );
        n_finite += 1;
    }
    assert!(n_finite > 0, "{name}: no finite samples to compare");
}

/// Load `sample`, sweep the fresh (direct) mechanism, do a no-op rebuild
/// (which expands the mount-point actuator into cylinder + rod), sweep
/// again, and assert the actuator series agree at every swept angle.
/// `range_deg` limits both sweeps identically (`None` = full 0..=360).
fn assert_rebuild_matches_fresh(sample: SampleMechanism, sizing: bool, range_deg: Option<(f64, f64)>) {
    let mut state = AppState::default();
    state.load_sample(sample);
    if sizing {
        enter_sizing_mode(&mut state);
    }
    if let Some((lo, hi)) = range_deg {
        state.sweep_range_enabled = true;
        state.sweep_angle_min_deg = lo;
        state.sweep_angle_max_deg = hi;
    }
    let pins = first_actuator(&state);
    assert!(
        !state.mechanism.as_ref().unwrap().bodies().contains_key("force_0_cyl"),
        "fresh sample must be the direct (un-expanded) mechanism"
    );
    state.compute_sweep();
    let fresh: SweepData = state.sweep_data.clone().expect("fresh sweep");

    state.rebuild();
    assert!(
        state.mechanism.as_ref().unwrap().bodies().contains_key("force_0_cyl"),
        "rebuild must expand the mount-point actuator into compound bodies"
    );
    assert!(state.solver_status.converged, "rebuilt mechanism must solve");

    // Current pose: the remapped element must see the pin-to-pin distance.
    let remapped = first_actuator(&state);
    let pin_len = pin_to_pin_length(&state, &pins);
    let elem_len = element_length(&state, &remapped);
    assert!(
        (pin_len - elem_len).abs() <= REL_TOL * pin_len + ABS_FLOOR,
        "rebuilt actuator length {elem_len} != pin-to-pin distance {pin_len}"
    );

    state.compute_sweep();
    let rebuilt: SweepData = state.sweep_data.clone().expect("rebuilt sweep");

    assert_eq!(fresh.angles_deg, rebuilt.angles_deg, "sweep angles differ");
    let angles = &fresh.angles_deg;
    assert_series_match(
        "actuator_lengths",
        fresh.actuator_lengths.as_ref().unwrap(),
        rebuilt.actuator_lengths.as_ref().unwrap(),
        angles,
    );
    assert_series_match(
        "actuator_forces",
        fresh.actuator_forces.as_ref().unwrap(),
        rebuilt.actuator_forces.as_ref().unwrap(),
        angles,
    );
}

#[test]
fn parallelogram_rebuild_matches_fresh_stored_force() {
    assert_rebuild_matches_fresh(SampleMechanism::ParallelogramActuator, false, None);
}

/// Sizing mode excludes the 0/360 deg change point: there the FRESH direct
/// mechanism's own sweep already aborts debug builds (pass-2 validation
/// `debug_assert`, BL-017), independent of the compound expansion. The
/// 180 deg change point is inside the range and does not abort.
#[test]
fn parallelogram_rebuild_matches_fresh_sizing_mode() {
    assert_rebuild_matches_fresh(SampleMechanism::ParallelogramActuator, true, Some((1.0, 359.0)));
}

#[test]
fn chebyshev_rebuild_matches_fresh_stored_force() {
    assert_rebuild_matches_fresh(SampleMechanism::ChebyshevLambdaActuator, false, None);
}

#[test]
fn chebyshev_rebuild_matches_fresh_sizing_mode() {
    assert_rebuild_matches_fresh(SampleMechanism::ChebyshevLambdaActuator, true, None);
}

/// With the pins inside the stroke limits (the sample's limits are the
/// pin-distance extremes over a full crank turn), the rebuilt actuator's
/// end-stop penalty must contribute nothing: with `force = 0` and zero
/// velocity its generalized force is identically zero at every pose.
#[test]
fn chebyshev_rebuild_end_stop_inactive_inside_stroke_limits() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    let pins = first_actuator(&state);
    assert!(pins.stroke_max > pins.stroke_min && pins.stroke_min > 0.0);

    state.rebuild();
    let mut remapped = first_actuator(&state);
    assert_eq!(remapped.body_a, "force_0_cyl", "actuator must be expanded");
    remapped.force = 0.0;

    let mut n_checked = 0;
    for deg in (0..360).step_by(5) {
        state.solve_at_angle((deg as f64).to_radians());
        assert!(state.solver_status.converged, "solve failed at {deg} deg");
        let pin_len = pin_to_pin_length(&state, &pins);
        // Only poses strictly inside the stroke window are in scope.
        if pin_len <= pins.stroke_min + 1e-9 || pin_len >= pins.stroke_max - 1e-9 {
            continue;
        }
        let mech = state.mechanism.as_ref().unwrap();
        let zero_qd = DVector::zeros(state.q.len());
        let q_act = evaluate_linear_actuator(&remapped, mech.state(), &state.q, &zero_qd);
        assert!(
            q_act.norm() < 1e-9,
            "phantom end-stop force at {deg} deg: |Q| = {} (pin length {pin_len}, stroke [{}, {}])",
            q_act.norm(),
            pins.stroke_min,
            pins.stroke_max
        );
        n_checked += 1;
    }
    assert!(n_checked > 10, "too few in-stroke poses checked ({n_checked})");
}
