//! BL-010 regression: the canvas actuator force label must show exactly the
//! `sweep.actuator_forces[i]` sample that the Actuator Force plot draws at
//! the current driver parameter. Promoted from the audit repro suite
//! (`tests/braindump_repro.rs`) with the assertions inverted.
//!
//! The label value is produced by `AppState::actuator_label_force`; the
//! canvas renderer only formats it (`canvas/rendering/force_render.rs`).

use linkage_sim_rs::forces::elements::{ForceElement, LinearActuatorElement};
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::{ActuatorLabelForce, AppState, SweepMode};

/// Clone the first `LinearActuator` element of the built mechanism (the one
/// whose force the sweep computes).
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
        .expect("sample has a LinearActuator")
}

/// Expected label at sample `i`: the sweep value when finite, else the
/// element's stored force.
fn expected_at(forces: &[f64], i: usize, la: &LinearActuatorElement) -> ActuatorLabelForce {
    if forces[i].is_finite() {
        ActuatorLabelForce::Computed(forces[i])
    } else {
        ActuatorLabelForce::Stored(la.force)
    }
}

/// Assert the label equals the plot sample at every sweep index, positioning
/// the driver via `set_driver` (angle in radians or stroke in metres).
fn assert_label_matches_every_sample(
    state: &mut AppState,
    la: &LinearActuatorElement,
    skip_wrapped_duplicate: bool,
    set_driver: fn(&mut AppState, f64),
) {
    let sweep = state.sweep_data.clone().expect("sweep computed");
    let forces = sweep.actuator_forces.as_ref().expect("actuator force series");
    assert_eq!(sweep.angles_deg.len(), forces.len());
    let mut n_computed = 0;
    for (i, &x) in sweep.angles_deg.iter().enumerate() {
        // A full 0..=360 sweep samples the same pose twice (0 and 360);
        // the label resolves that tie to the first sample, so only the
        // first is asserted bitwise.
        if skip_wrapped_duplicate && x >= 360.0 {
            continue;
        }
        set_driver(state, x);
        let got = state.actuator_label_force(la);
        let want = expected_at(forces, i, la);
        assert_eq!(
            got, want,
            "sample {} (driver x = {}): label {:?} != plot sample {:?}",
            i, x, got, want
        );
        if matches!(got, ActuatorLabelForce::Computed(_)) {
            n_computed += 1;
        }
    }
    assert!(n_computed > 0, "no finite actuator force sample in the sweep");
}

fn set_driver_angle_deg(state: &mut AppState, x_deg: f64) {
    state.driver_angle = x_deg.to_radians();
}

fn set_driver_stroke_m(state: &mut AppState, x_m: f64) {
    state.set_driver_stroke(x_m);
}

/// (a) Full 0..360 sweep, stored-force mode (la.force = 2225 N as shipped):
/// the label must read the sweep sample, never the stored value.
#[test]
fn label_matches_plot_sample_on_full_sweep() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    let la = first_actuator(&state);
    assert!(la.force.abs() > 1.0, "sample must ship in stored-force mode");
    state.compute_sweep();

    assert_label_matches_every_sample(&mut state, &la, true, set_driver_angle_deg);
}

/// Mechanism (1) of BL-010 in isolation: at the pose the mechanism loads at,
/// the plot's Statics series shows the computed force (about -2223 N) while
/// the pre-fix label printed the stored +2225 N.
#[test]
fn stored_force_mode_label_reads_sweep_not_stored_value() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    let la = first_actuator(&state);
    state.compute_sweep();
    let forces = state.sweep_data.as_ref().unwrap().actuator_forces.clone().unwrap();
    assert!(forces[0].is_finite());
    assert!(
        (forces[0] - la.force).abs() > 1.0,
        "fixture no longer distinguishes stored from computed: {} vs {}",
        forces[0],
        la.force
    );

    state.driver_angle = 0.0;
    assert_eq!(state.actuator_label_force(&la), ActuatorLabelForce::Computed(forces[0]));
}

/// (b) Seam-crossing range sweep 200..365 deg: `angles_deg` carries raw
/// values above 360 while the driver angle may be wrapped or not.
#[test]
fn label_matches_plot_sample_on_seam_crossing_range_sweep() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    let la = first_actuator(&state);
    state.sweep_range_enabled = true;
    state.sweep_angle_min_deg = 200.0;
    state.sweep_angle_max_deg = 365.0;
    state.compute_sweep();
    let sweep = state.sweep_data.clone().unwrap();
    assert!((sweep.angles_deg[0] - 200.0).abs() < 1e-9);
    assert!((sweep.angles_deg.last().unwrap() - 365.0).abs() < 1e-9);

    assert_label_matches_every_sample(&mut state, &la, false, set_driver_angle_deg);

    // Scrubbing to 365 deg via the plot-click path (driver_angle = clicked_x
    // - offset, unwrapped) and via a wrapped 5 deg driver angle must both
    // resolve to the 365 deg sample.
    let forces = sweep.actuator_forces.as_ref().unwrap();
    let last = forces.len() - 1;
    assert!(forces[last].is_finite());
    let offset = state.driver_display_offset;
    for driver_rad in [365.0_f64.to_radians() - offset, 5.0_f64.to_radians() - offset] {
        state.driver_angle = driver_rad;
        assert_eq!(
            state.actuator_label_force(&la),
            ActuatorLabelForce::Computed(forces[last]),
            "driver {:.3} rad should resolve to the 365 deg sample",
            driver_rad
        );
    }
}

/// (c) Stroke-mode sweep (linear driver): `angles_deg` carries METRES and the
/// driver parameter is the stroke, not an angle.
///
/// The forward sweep only fills `actuator_forces` when a `LinearActuator`
/// force element exists, and converting the sample's actuator into a linear
/// driver removes that element -- so it is re-added on top of the driver to
/// get a stroke-mode sweep that carries an actuator force series.
#[test]
fn label_matches_plot_sample_on_stroke_sweep() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ParallelogramActuator);
    let la = first_actuator(&state);
    let act_index = state
        .mechanism
        .as_ref()
        .unwrap()
        .forces()
        .iter()
        .position(|f| matches!(f, ForceElement::LinearActuator(_)))
        .unwrap();
    state.convert_actuator_to_linear_driver(act_index);
    state
        .blueprint
        .as_mut()
        .unwrap()
        .forces
        .push(ForceElement::LinearActuator(la.clone()));
    state.rebuild();
    state.compute_sweep();
    let sweep = state.sweep_data.clone().expect("stroke sweep computed");
    assert!(matches!(sweep.sweep_mode, SweepMode::Stroke));
    assert!(
        sweep.actuator_forces.is_some(),
        "LinearActuator element on a linear-driver mechanism populates actuator_forces"
    );

    assert_label_matches_every_sample(&mut state, &la, false, set_driver_stroke_m);
}
