//! Weight (point mass) editing in the property panel (payload weights,
//! spec Track 2 section 3, "Property panel"):
//!
//! - [`draw_selected_weight`]: the editor of the weight selected on the
//!   canvas: name, mass, owning link and body-local position.
//! - [`draw_link_weights`]: the link editor's Weights section, shown for
//!   every moving link (even without weights) with an Add weight button.
//!
//! Both draw the same fields ([`draw_weight_fields`]). A field edits a copy
//! of the stored value and commits once, when the user finishes: a drag is
//! released, or the field loses focus after typing (Enter, Tab or a click
//! elsewhere); Esc drops the edit. Each commit is one `PendingPropertyEdit`,
//! which the `AppState` weight API applies as one undo step.

use std::ops::RangeInclusive;

use eframe::egui;

use crate::analysis::gravity_breakdown::{display_name, name_with_id, point_mass_title};
use crate::gui::canvas::WEIGHT_COLOR;
use crate::gui::state::{format_decimal, AppState, DisplayUnits};
use crate::io::{BodyJson, PointMassJson};

use super::pending_edits::PendingPropertyEdit;

/// Masses (kg) a weight mass field accepts when typed or dragged.
pub(crate) const WEIGHT_MASS_RANGE_KG: RangeInclusive<f64> = 0.001..=1000.0;

/// Id salt of the selected-weight editor's fields.
const SELECTED_SALT: &str = "selected_weight";
/// Id salt of the link editor's weight fields.
const LINK_EDITOR_SALT: &str = "link_editor";

/// A weight field's number widget over `value`: shows `format_decimal`
/// text and applies typed text only when the field loses focus without Esc
/// (`update_while_editing(false)`), so Esc drops what was typed.
fn field_drag_value(value: &mut f64) -> egui::DragValue<'_> {
    egui::DragValue::new(value)
        .update_while_editing(false)
        .custom_formatter(|v, _| format_decimal(v))
}

/// Mass settings of a weight field (kg). Out-of-range masses stored in a
/// file are left alone on idle frames (`clamp_existing_to_range(false)`:
/// egui would otherwise clamp them and report a change).
fn mass_field(field: egui::DragValue<'_>) -> egui::DragValue<'_> {
    field.speed(0.01).range(WEIGHT_MASS_RANGE_KG).clamp_existing_to_range(false).suffix(" kg")
}

/// The mass field (kg) of a weight, also the + Mass tool's toolbar field.
pub(crate) fn weight_mass_drag_value(mass: &mut f64) -> egui::DragValue<'_> {
    mass_field(field_drag_value(mass))
}

/// Show a number field over a copy of the stored `value` (`configure` sets
/// its speed, range and text) and return the number the user committed.
///
/// The copy lives in egui memory while the field is dragged or focused:
/// egui stops reporting `dragged()` on the release frame, so a copy made
/// afresh each frame would lose the drag. A commit is the release of a
/// drag or the loss of focus; it returns `None` when the committed number
/// is the stored value or the number the field was showing. Leaving a field
/// without typing makes egui read the shown (rounded) text back, which must
/// not become an edit, and Esc keeps the stored value. Idle frames never
/// commit, whatever the value.
fn committed_number(
    ui: &mut egui::Ui,
    key: &str,
    hover: &str,
    value: f64,
    configure: impl FnOnce(egui::DragValue<'_>) -> egui::DragValue<'_>,
) -> Option<f64> {
    let copy_id = ui.id().with(("committed_number", key));
    let mut edited = ui.data(|d| d.get_temp::<f64>(copy_id)).unwrap_or(value);
    let response = ui.add(configure(field_drag_value(&mut edited))).on_hover_text(hover);
    if response.dragged() || response.has_focus() {
        ui.data_mut(|d| d.insert_temp(copy_id, edited));
        return None;
    }
    ui.data_mut(|d| d.remove::<f64>(copy_id));
    let finished = response.drag_stopped() || response.lost_focus();
    let shown = format_decimal(value).parse().unwrap_or(value);
    (finished && edited != value && edited != shown).then_some(edited)
}

/// Id of the name field of weight `weight_id` on `body_id` in the editor
/// `salt` (the selected-weight editor and the link editor both show one).
fn name_field_id(salt: &str, body_id: &str, weight_id: &str) -> egui::Id {
    egui::Id::new(("weight_name", salt, body_id, weight_id))
}

/// The weight's name (label) field; blank shows the id. The text being
/// typed lives in egui memory while the field has focus. Returns the text
/// to commit when the field loses focus (Enter, Tab or a click elsewhere);
/// Esc drops it.
fn name_field(ui: &mut egui::Ui, salt: &str, body_id: &str, pm: &PointMassJson) -> Option<String> {
    let id = name_field_id(salt, body_id, &pm.id);
    let typing_id = id.with("typing");
    let mut text = ui
        .data(|d| d.get_temp::<String>(typing_id))
        .unwrap_or_else(|| pm.label.clone().unwrap_or_default());
    let response = ui
        .add(egui::TextEdit::singleline(&mut text).id(id).hint_text(pm.id.as_str()).desired_width(140.0))
        .on_hover_text("Weight name, shown on the canvas and in the Weight Breakdown plot. Leave blank to show the id.");
    if response.has_focus() {
        ui.data_mut(|d| d.insert_temp(typing_id, text));
        return None;
    }
    ui.data_mut(|d| d.remove::<String>(typing_id));
    (response.lost_focus() && !ui.input(|i| i.key_pressed(egui::Key::Escape))).then_some(text)
}

/// Name, mass and body-local position fields of weight `pm` on `body_id`,
/// and its Move to Link, Reposition and Delete buttons. `salt` keeps the
/// field ids of the two editors apart.
fn draw_weight_fields(
    ui: &mut egui::Ui,
    units: &DisplayUnits,
    salt: &str,
    body_id: &str,
    pm: &PointMassJson,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let ids = || (body_id.to_string(), pm.id.clone());
    egui::Grid::new(("weight_fields", salt, body_id, pm.id.as_str())).num_columns(2).show(ui, |ui| {
        ui.label("Name");
        if let Some(label) = name_field(ui, salt, body_id, pm) {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::SetPointMassLabel { body_id, weight_id, label: Some(label) });
        }
        ui.end_row();

        ui.label("Mass");
        if let Some(mass) = committed_number(ui, "mass", "Weight mass in kg", pm.mass, mass_field) {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::SetPointMassMass { body_id, weight_id, mass });
        }
        ui.end_row();

        ui.label("Position");
        ui.horizontal(|ui| {
            let [x, y] = pm.local_pos;
            let speed = units.length(0.001);
            let suffix = units.length_suffix();
            let new_x = committed_number(ui, "x", "Body-local X position", units.length(x), |f| {
                f.speed(speed).prefix("X ").suffix(suffix)
            });
            let new_y = committed_number(ui, "y", "Body-local Y position", units.length(y), |f| {
                f.speed(speed).prefix("Y ").suffix(suffix)
            });
            let local_pos = match (new_x, new_y) {
                (Some(nx), _) => Some([units.length_to_si(nx), y]),
                (None, Some(ny)) => Some([x, units.length_to_si(ny)]),
                (None, None) => None,
            };
            if let Some(local_pos) = local_pos {
                let (body_id, weight_id) = ids();
                *pending = Some(PendingPropertyEdit::SetPointMassPosition { body_id, weight_id, local_pos });
            }
        });
        ui.end_row();
    });
    ui.horizontal(|ui| {
        if ui.small_button("Move to Link").on_hover_text("Click a different link to move this weight there").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::ReassignPointMass { body_id, weight_id });
        }
        if ui.small_button("Reposition").on_hover_text("Click on the canvas to move this weight").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::RepositionPointMass { body_id, weight_id });
        }
        if ui.small_button("Delete").on_hover_text("Delete this weight").clicked() {
            let (body_id, weight_id) = ids();
            *pending = Some(PendingPropertyEdit::RemovePointMass { body_id, weight_id });
        }
    });
}

/// "coupler", or "Arm (b2)" for a labelled link.
fn link_name(state: &AppState, body_id: &str) -> String {
    let label = state.blueprint.as_ref().and_then(|bp| bp.bodies.get(body_id)).and_then(|b| b.label.as_ref());
    name_with_id(&display_name(label, body_id), body_id)
}

/// The editor of the weight selected on the canvas
/// (`SelectedEntity::Weight`): name, mass, owning link and body-local
/// position. Draws nothing for a stale selection (the weight was deleted,
/// undone or moved to another link since it was selected).
pub(super) fn draw_selected_weight(
    ui: &mut egui::Ui,
    state: &AppState,
    body_id: &str,
    weight_id: &str,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let Some(pm) = state.find_point_mass(body_id, weight_id) else { return };
    egui::CollapsingHeader::new(
        egui::RichText::new(format!("Weight {}", point_mass_title(pm))).color(state.nc(WEIGHT_COLOR)),
    )
    .id_salt("selected_weight")
    .default_open(true)
    .show(ui, |ui| {
        ui.label(format!("Link: {}", link_name(state, body_id)));
        draw_weight_fields(ui, &state.display_units, SELECTED_SALT, body_id, pm, pending);
    });
}

/// The link editor's Weights section for moving link `body_id`, shown even
/// when the link carries no weight: the fields of each weight, then an Add
/// weight button that adds a weight of the last mass used
/// (`AppState::last_point_mass_kg`) at the link's centre of mass.
pub(super) fn draw_link_weights(
    ui: &mut egui::Ui,
    state: &AppState,
    body_id: &str,
    body: &BodyJson,
    pending: &mut Option<PendingPropertyEdit>,
) {
    egui::CollapsingHeader::new(
        egui::RichText::new(format!("Weights ({})", body.point_masses.len())).color(state.nc(WEIGHT_COLOR)),
    )
    .id_salt(format!("point_masses_{body_id}"))
    .default_open(true)
    .show(ui, |ui| {
        for pm in &body.point_masses {
            ui.push_id(pm.id.as_str(), |ui| {
                ui.label(egui::RichText::new(point_mass_title(pm)).strong());
                draw_weight_fields(ui, &state.display_units, LINK_EDITOR_SALT, body_id, pm, pending);
            });
            ui.separator();
        }
        let hover = format!(
            "Add a {} kg weight at this link's centre of mass (the + Mass toolbar field sets the mass), then drag it on the canvas",
            format_decimal(state.last_point_mass_kg)
        );
        if ui.button("Add weight").on_hover_text(hover).clicked() {
            *pending = Some(PendingPropertyEdit::AddPointMass { body_id: body_id.to_string(), local_pos: body.cg_local });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::state::SelectedEntity;
    use crate::gui::test_support::{
        central_panel_frame, drawn_texts, drew_text, key_press, primary_button, text_rect, typed,
    };
    use super::super::draw_property_panel;

    // ── committed_number ────────────────────────────────────────────────

    /// A lone weight mass field over a stored value that only changes when
    /// the test says so, as the blueprint only changes on a commit.
    struct Field {
        ctx: egui::Context,
        stored: f64,
    }

    impl Field {
        fn new(stored: f64) -> Self {
            Self { ctx: egui::Context::default(), stored }
        }

        /// One frame; returns what the field committed in it.
        fn frame(&mut self, events: Vec<egui::Event>) -> Option<f64> {
            self.frame_output(events).0
        }

        fn frame_output(&mut self, events: Vec<egui::Event>) -> (Option<f64>, egui::FullOutput) {
            let stored = self.stored;
            let mut committed = None;
            let output = central_panel_frame(&self.ctx, events, |ui| {
                committed = committed_number(ui, "mass", "hover", stored, mass_field);
            });
            (committed, output)
        }

        /// Screen rect of the field, found by giving it keyboard focus with
        /// Tab (it is the only widget) and leaving again with Esc.
        fn rect(&mut self) -> egui::Rect {
            assert_eq!(self.frame(vec![key_press(egui::Key::Tab)]), None);
            let id = self.ctx.memory(|m| m.focused()).expect("Tab focuses the field");
            assert_eq!(self.frame(vec![key_press(egui::Key::Escape)]), None);
            self.ctx.read_response(id).expect("the field was drawn").rect
        }
    }

    #[test]
    fn a_typed_value_commits_once_when_enter_is_pressed() {
        let mut field = Field::new(2.0);
        assert_eq!(field.frame(Vec::new()), None, "idle");
        assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None, "focus");
        assert_eq!(field.frame(vec![typed("5.5")]), None, "typing is not a commit");
        assert_eq!(field.frame(vec![key_press(egui::Key::Enter)]), Some(5.5));
        assert_eq!(field.frame(Vec::new()), None, "one commit per edit");
    }

    #[test]
    fn a_typed_value_commits_when_focus_moves_away() {
        let mut field = Field::new(2.0);
        field.frame(vec![key_press(egui::Key::Tab)]);
        field.frame(vec![typed("7")]);
        assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), Some(7.0), "Tab leaves the only field");
    }

    #[test]
    fn typed_values_are_clamped_to_the_weight_mass_range() {
        for (text, want) in [("0", 0.001), ("-3", 0.001), ("5000", 1000.0)] {
            let mut field = Field::new(2.0);
            field.frame(vec![key_press(egui::Key::Tab)]);
            field.frame(vec![typed(text)]);
            assert_eq!(field.frame(vec![key_press(egui::Key::Enter)]), Some(want), "typed {text}");
        }
    }

    /// egui reads the shown (rounded) text back when a field loses focus;
    /// leaving a field without typing must not turn that rounding into an
    /// edit, whatever the stored precision.
    #[test]
    fn leaving_a_field_without_typing_commits_nothing() {
        for stored in [2.0, 30.0000004, 0.031234567891] {
            let mut field = Field::new(stored);
            assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None);
            assert_eq!(field.frame(vec![key_press(egui::Key::Tab)]), None, "{stored}: left without typing");
            assert_eq!(field.frame(Vec::new()), None);
        }
    }

    #[test]
    fn escape_drops_a_typed_value() {
        let mut field = Field::new(2.0);
        field.frame(vec![key_press(egui::Key::Tab)]);
        field.frame(vec![typed("9")]);
        assert_eq!(field.frame(vec![key_press(egui::Key::Escape)]), None);
        let (committed, output) = field.frame_output(Vec::new());
        assert_eq!(committed, None);
        assert!(drew_text(&output, "2 kg"), "the field shows the stored mass again");
    }

    /// egui stops reporting `dragged()` on the release frame, so a field
    /// over a per-frame copy must keep the dragged value itself.
    #[test]
    fn a_drag_commits_the_dragged_value_once_on_release() {
        let mut field = Field::new(2.0);
        let start = field.rect().center();
        let far = start + egui::vec2(40.0, 0.0);
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(start)]), None);
        assert_eq!(field.frame(vec![primary_button(start, true)]), None);
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(start + egui::vec2(20.0, 0.0))]), None, "mid-drag");
        assert_eq!(field.frame(vec![egui::Event::PointerMoved(far)]), None, "mid-drag");
        let committed = field.frame(vec![primary_button(far, false)]).expect("the release commits");
        // speed 0.01 kg per point over ~40 points (egui rounds the value).
        assert!((committed - 2.4).abs() < 0.1, "dragged to {committed}");
        assert_eq!(field.frame(Vec::new()), None);
    }

    #[test]
    fn idle_frames_commit_nothing_even_for_out_of_range_masses() {
        for stored in [1500.0, 0.0] {
            let mut field = Field::new(stored);
            for _ in 0..3 {
                assert_eq!(field.frame(Vec::new()), None, "{stored}");
            }
        }
    }

    // ── Property panel ──────────────────────────────────────────────────

    /// Four-bar with weight W1 (2 kg at (0.03, 0.02)) on the coupler,
    /// selected; the link editor shows the crank, so W1's fields appear
    /// once, in the selected-weight editor.
    fn selected_weight_state() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.add_point_mass("coupler", 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        state.selected = Some(SelectedEntity::Weight { body_id: "coupler".to_string(), weight_id: "W1".to_string() });
        state.link_editor_body = Some("crank".to_string());
        state
    }

    fn panel_frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> egui::FullOutput {
        central_panel_frame(ctx, events, |ui| draw_property_panel(ui, state))
    }

    fn click(ctx: &egui::Context, state: &mut AppState, at: egui::Pos2) {
        panel_frame(ctx, state, vec![egui::Event::PointerMoved(at)]);
        panel_frame(ctx, state, vec![primary_button(at, true)]);
        panel_frame(ctx, state, vec![primary_button(at, false)]);
    }

    /// Click into W1's name field in the selected-weight editor.
    fn focus_name_field(ctx: &egui::Context, state: &mut AppState) {
        panel_frame(ctx, state, Vec::new());
        let id = name_field_id(SELECTED_SALT, "coupler", "W1");
        let rect = ctx.read_response(id).expect("the name field is drawn").rect;
        click(ctx, state, rect.center());
        assert!(ctx.memory(|m| m.has_focus(id)), "the click focuses the name field");
    }

    #[test]
    fn the_selected_weight_shows_its_name_mass_link_and_position() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        for want in ["Weight W1", "Link: coupler", "2 kg", "X 30 mm", "Y 20 mm", "Move to Link", "Reposition"] {
            assert!(texts.iter().any(|t| t == want), "missing {want:?} in {texts:?}");
        }
    }

    #[test]
    fn typing_a_name_commits_it_once_when_enter_is_pressed() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![typed("Robot torso")]);
        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label, None, "typing is not a commit");
        assert_eq!(state.undo_history.undo_count(), depth);
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Enter)]);

        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label.as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1, "one name edit = one undo step");
        assert!(drew_text(&panel_frame(&ctx, &mut state, Vec::new()), "Weight Robot torso (W1)"));
    }

    #[test]
    fn escape_drops_a_typed_name() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![typed("Oops")]);
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Escape)]);
        panel_frame(&ctx, &mut state, Vec::new());

        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().label, None);
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn tab_from_the_name_to_the_mass_and_typing_commits_the_mass_once() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        focus_name_field(&ctx, &mut state);
        let depth = state.undo_history.undo_count();

        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Tab)]);
        panel_frame(&ctx, &mut state, vec![typed("5")]);
        assert_eq!(state.find_point_mass("coupler", "W1").unwrap().mass, 2.0, "typing is not a commit");
        panel_frame(&ctx, &mut state, vec![key_press(egui::Key::Enter)]);

        let pm = state.find_point_mass("coupler", "W1").unwrap();
        assert_eq!(pm.mass, 5.0);
        assert_eq!(pm.label, None, "leaving the untouched name field changed nothing");
        assert_eq!(state.undo_history.undo_count(), depth + 1, "one mass edit = one undo step");
    }

    #[test]
    fn a_stale_weight_selection_shows_no_weight_editor() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.selected = Some(SelectedEntity::Weight { body_id: "coupler".to_string(), weight_id: "W7".to_string() });
        let depth = state.undo_history.undo_count();

        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));

        assert!(!texts.iter().any(|t| t.starts_with("Weight ")), "{texts:?}");
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn the_link_editor_weights_section_shows_even_without_weights() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.selected = None;

        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        assert!(texts.iter().any(|t| t == "Weights (0)"), "crank has no weight: {texts:?}");
        assert!(texts.iter().any(|t| t == "Add weight"));

        state.link_editor_body = Some("coupler".to_string());
        let texts = drawn_texts(&panel_frame(&ctx, &mut state, Vec::new()));
        assert!(texts.iter().any(|t| t == "Weights (1)"), "{texts:?}");
        assert!(texts.iter().any(|t| t == "W1"));
        assert!(texts.iter().any(|t| t == "X 30 mm"));
    }

    #[test]
    fn add_weight_adds_the_last_mass_at_the_link_cg_and_selects_it() {
        let ctx = egui::Context::default();
        let mut state = selected_weight_state();
        state.last_point_mass_kg = 3.5;
        let cg = state.blueprint.as_ref().unwrap().bodies["crank"].cg_local;
        let depth = state.undo_history.undo_count();

        let output = panel_frame(&ctx, &mut state, Vec::new());
        let button = text_rect(&output, "Add weight").expect("the Add weight button is drawn");
        click(&ctx, &mut state, button.center());

        let pm = state.find_point_mass("crank", "W2").expect("the new weight gets the next id");
        assert_eq!(pm.mass, 3.5);
        assert_eq!(pm.local_pos, cg);
        assert_eq!(state.selected, Some(SelectedEntity::Weight { body_id: "crank".to_string(), weight_id: "W2".to_string() }));
        assert_eq!(state.undo_history.undo_count(), depth + 1);
    }
}
