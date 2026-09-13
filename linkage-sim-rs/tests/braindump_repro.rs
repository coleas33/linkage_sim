//! TEMPORARY audit repro tests for BL-017. Not for commit.
//! (BL-011 repro promoted to src/gui/sweep/mod.rs tests; BL-010 repros
//! promoted to tests/actuator_force_label.rs with assertions inverted.)
//! Run: cargo test --test braindump_repro -- --nocapture

use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;

/// Side discovery while reproducing BL-010: following the actuator-sizing
/// tutorial (tutorial.rs:204 "Step 2: Set Actuator Force to Zero") on the
/// ChebyshevLambdaActuator sample makes the very first sweep panic in any
/// debug build: the pass-2 driver lambda fails to collapse at some pose and
/// the data-dependent `debug_assert!` at solver/reactions.rs:666 fires.
#[test]
#[should_panic(expected = "Failed validation")]
fn bl010_side_sizing_mode_sweep_panics_debug_assert() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    if let Some(bp) = state.blueprint.as_mut() {
        for f in &mut bp.forces {
            if let linkage_sim_rs::forces::elements::ForceElement::LinearActuator(la) = f {
                la.force = 0.0; // sizing mode, per the tutorial
            }
        }
    }
    state.rebuild();
    state.compute_sweep(); // panics at reactions.rs:666 in debug builds
}
