//! Helpers shared by the tests of the GUI modules.

use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::gui::state::AppState;

/// Non-ground blueprint body (link) ids, sorted: a deterministic fixture
/// order, since `blueprint.bodies` is a `HashMap`.
pub(crate) fn sorted_link_ids(state: &AppState) -> Vec<String> {
    let mut ids: Vec<String> = state
        .blueprint
        .as_ref()
        .expect("blueprint")
        .bodies
        .keys()
        .filter(|k| k.as_str() != GROUND_ID)
        .cloned()
        .collect();
    ids.sort();
    ids
}

/// Set every LinearActuator's stored force in the blueprint and rebuild
/// (0 = sizing mode). Since BL-026 the sweep reports the required actuator
/// force in both modes.
pub(crate) fn set_actuator_stored_force(state: &mut AppState, force: f64) {
    for element in &mut state.blueprint.as_mut().expect("blueprint").forces {
        if let ForceElement::LinearActuator(la) = element {
            la.force = force;
        }
    }
    state.rebuild();
}
