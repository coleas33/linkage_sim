//! Canvas weights (point masses): placement hint.

use crate::gui::state::{format_mass_kg, AppState};
use crate::io::next_point_mass_id;

/// Hint while the + Mass tool waits for the drop point on `body_id`: the id
/// the new weight gets (`io::next_point_mass_id`, the default name) and its
/// mass (the toolbar field, `AppState::last_point_mass_kg`).
pub(super) fn place_mass_hint(state: &AppState, body_id: &str) -> String {
    // Without a blueprint no weight exists yet, so the next id is W1.
    let next_id = state
        .blueprint
        .as_ref()
        .map_or_else(|| "W1".to_string(), |bp| next_point_mass_id(&bp.bodies));
    format!(
        "Click to place weight {next_id} ({}) on '{body_id}' (Esc to cancel)",
        format_mass_kg(state.last_point_mass_kg)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;

    #[test]
    fn place_mass_hint_names_the_next_weight_and_the_field_mass() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.last_point_mass_kg = 3.5;
        assert_eq!(
            place_mass_hint(&state, "coupler"),
            "Click to place weight W1 (3.5 kg) on 'coupler' (Esc to cancel)"
        );

        state.add_point_mass("crank", 1.25, [0.0, 0.0]).expect("W1 added");
        assert_eq!(
            place_mass_hint(&state, "coupler"),
            "Click to place weight W2 (1.25 kg) on 'coupler' (Esc to cancel)",
            "W1 is taken, and adding it made 1.25 kg the last mass used"
        );
    }
}
