//! TEMPORARY audit repro tests for BL-017. Not for commit.
//! (BL-011 repro promoted to src/gui/sweep/mod.rs tests; BL-010 repros
//! promoted to tests/actuator_force_label.rs with assertions inverted.)
//! Run: cargo test --test braindump_repro -- --nocapture

use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;

/// Side discovery while reproducing BL-010: following the actuator-sizing
/// tutorial (tutorial.rs:204 "Step 2: Set Actuator Force to Zero") on the
/// ChebyshevLambdaActuator sample made the very first sweep panic in any
/// debug build: the pass-2 driver lambda failed to collapse at some pose and
/// the data-dependent `debug_assert!` at solver/reactions.rs:666 fired.
/// Root cause was BL-022: the rebuild expands the mount-point actuator into
/// cylinder + rod, and the remapped force acted base -> rod slide, so its
/// length left the stroke window and the end-stop penalty (already in
/// pass-1 `q_forces`) was injected again in pass 2. With the force acting
/// pin to pin the sweep completes.
#[test]
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
    state.compute_sweep(); // panicked at reactions.rs:666 before BL-022
    let sweep = state.sweep_data.as_ref().expect("sweep computed");
    let forces = sweep.actuator_forces.as_ref().expect("actuator force series");
    assert!(forces.iter().any(|f| f.is_finite()), "no finite sizing force");
}
