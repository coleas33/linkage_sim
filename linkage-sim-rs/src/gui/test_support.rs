//! Helpers shared by the tests of the GUI modules.

use crate::core::state::GROUND_ID;
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
