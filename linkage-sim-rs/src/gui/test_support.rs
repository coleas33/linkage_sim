//! Helpers shared by the tests of the GUI modules (fixtures, input events,
//! and inspection of what a headless egui frame painted).

use eframe::egui;

use crate::core::state::GROUND_ID;
use crate::forces::elements::ForceElement;
use crate::gui::samples::SampleMechanism;
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

/// A primary-button press (`pressed`) or release at `pos`, with `modifiers`
/// held (e.g. Shift for a multi-selection click).
pub(crate) fn primary_button_with(pos: egui::Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers }
}

/// A primary-button press (`pressed`) or release at `pos`, no modifiers.
pub(crate) fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    primary_button_with(pos, pressed, egui::Modifiers::NONE)
}

/// A key press event with no modifiers.
pub(crate) fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: egui::Modifiers::NONE }
}

/// Text typed into the focused widget.
pub(crate) fn typed(text: &str) -> egui::Event {
    egui::Event::Text(text.to_string())
}

/// One headless frame of `draw` inside a central panel, with `events` as
/// the frame's input. Returns what egui painted.
pub(crate) fn central_panel_frame(
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput { events, ..Default::default() };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
}

/// Call `visit` on every shape egui painted in a frame, nested shapes
/// included, in paint order.
fn visit_shapes(output: &egui::FullOutput, mut visit: impl FnMut(&egui::Shape)) {
    fn walk(shape: &egui::Shape, visit: &mut impl FnMut(&egui::Shape)) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, visit)),
            other => visit(other),
        }
    }
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut visit);
    }
}

/// Every text egui drew in a frame (widgets, painter text, tooltips), in
/// paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    let mut texts = Vec::new();
    visit_shapes(output, |shape| {
        if let egui::Shape::Text(text) = shape {
            texts.push(text.galley.text().to_string());
        }
    });
    texts
}

/// Whether egui drew the text `needle` (exactly) in a frame.
pub(crate) fn drew_text(output: &egui::FullOutput, needle: &str) -> bool {
    drawn_texts(output).iter().any(|t| t == needle)
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    let mut found = None;
    visit_shapes(output, |shape| {
        match shape {
            egui::Shape::Text(text) if found.is_none() && text.galley.text() == needle => {
                found = Some(text.galley.rect.translate(text.pos.to_vec2()));
            }
            _ => {}
        }
    });
    found
}

/// The stroke colour of every line segment egui drew in a frame.
pub(crate) fn drawn_line_colors(output: &egui::FullOutput) -> Vec<egui::Color32> {
    let mut colors = Vec::new();
    visit_shapes(output, |shape| {
        if let egui::Shape::LineSegment { stroke, .. } = shape {
            colors.push(stroke.color);
        }
    });
    colors
}

/// The robot lift of the payload spec's hands-on checklist: Parallelogram +
/// Actuator with the sample's stored force (50 N; since BL-026 the sweep
/// reports the required force whatever it is), weight W1 (50 kg) at the
/// rocker tip and W2 (20 kg) on the coupler, swept 0..=360 deg.
pub(crate) fn swept_lift() -> AppState {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ParallelogramActuator);
    assert_eq!(state.add_point_mass("rocker", 50.0, [0.0, 0.0]).as_deref(), Some("W1"));
    assert_eq!(state.add_point_mass("coupler", 20.0, [2.0, 0.0]).as_deref(), Some("W2"));
    state.compute_sweep();
    state
}

/// Index of the sweep sample at `deg` (the driver angle, in degrees).
pub(crate) fn sample_at(state: &AppState, deg: f64) -> usize {
    let sweep = state.sweep_data.as_ref().expect("sweep computed");
    sweep
        .angles_deg
        .iter()
        .position(|&a| (a - deg).abs() < 1e-9)
        .unwrap_or_else(|| panic!("no sample at {deg} deg"))
}

/// Solve the mechanism at driver angle `deg`, so the canvas shows that pose
/// and the readouts read the sweep sample there.
pub(crate) fn pose_at(state: &mut AppState, deg: f64) {
    state.solve_at_angle(deg.to_radians());
    assert!(state.solver_status.converged, "the mechanism assembles at {deg} deg");
    assert_eq!(state.current_sweep_index(), Some(sample_at(state, deg)));
}
