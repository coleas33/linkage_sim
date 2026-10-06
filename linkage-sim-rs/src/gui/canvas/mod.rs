//! 2D mechanism canvas: rendering, pan/zoom, hit testing, drag, context menus.
//!
//! Split into submodules:
//! - `colors`: Color and sizing constants
//! - `hit_testing`: Hit target structs and geometry helpers
//! - `rendering`: Body/joint/force drawing, primitives, tooltips
//! - `interaction`: Drag, pan, zoom, tool modes, keyboard shortcuts
//! - `context_menu`: Right-click menus for joints, bodies, canvas

mod alignment;
mod colors;
mod context_menu;
mod hit_testing;
mod interaction;
mod rendering;

use eframe::egui::{self, FontId, Pos2};

use crate::gui::state::AppState;

pub use colors::{classification_color, to_grayscale, WEIGHT_COLOR};
use colors::*;
use hit_testing::{AttachmentHit, BodySegment};

/// Draw the 2D mechanism canvas with interaction.
pub fn draw_canvas(ui: &mut egui::Ui, state: &mut AppState) {
    let (response, painter) =
        ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
    let canvas_rect = response.rect;

    // Crosshair cursor while a trajectory canvas-pick is armed — gives a
    // clear "you're in a different mode" signal even if the trajectory panel
    // is collapsed.
    if state.pending_canvas_pick.is_some() && response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }

    // Sync mounting angle into view transform so canvas rendering rotates.
    state.view.mounting_angle = state.mounting_angle;

    // Adapt grid spacing to zoom level: finer grid as you zoom in (down to 0.1mm).
    state.grid.spacing_m = state.grid.zoom_spacing(canvas_rect.width(), state.view.scale);

    if state.pending_fit_to_view {
        state.fit_to_view(canvas_rect.width(), canvas_rect.height());
        state.pending_fit_to_view = false;
    }

    // Fill background.
    let bg = if state.nathan_mode { colors::to_grayscale(BG_COLOR) } else { BG_COLOR };
    painter.rect_filled(canvas_rect, 0.0, bg);

    // ── No mechanism message ────────────────────────────────────────────
    if state.mechanism.is_none() {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "No mechanism loaded",
            FontId::proportional(18.0),
            NO_MECH_TEXT_COLOR,
        );
        return;
    }

    // Snapshot solver state before immutable borrow.
    let show_debug = state.show_debug_overlay;
    let solver_converged = state.solver_status.converged;
    let solver_residual = state.solver_status.residual_norm;
    let solver_iterations = state.solver_status.iterations;
    let current_driver_joint = state.driver_joint_id.clone();

    // Collect hit-test data during the immutable rendering pass.
    let mut joint_hit_targets: Vec<(Pos2, String)> = Vec::new();
    let mut attachment_hit_targets: Vec<AttachmentHit> = Vec::new();
    let mut body_segments: Vec<BodySegment> = Vec::new();

    // ── Draw grid behind everything ──────────────────────────────────────
    rendering::draw_grid(&painter, canvas_rect, &state.view, &state.grid, state.nathan_mode);

    // ── Draw background image (behind mechanism, above grid) ────────────
    rendering::draw_background_image(&painter, canvas_rect, state);

    // ── Draw DXF overlay (behind mechanism, above background image) ─────
    crate::gui::dxf_import::draw_dxf_overlay(&painter, canvas_rect, state);

    // ── Render mechanism (immutable borrow scope) ────────────────────────
    let grounded_revolute_ids = rendering::render_mechanism(
        &painter,
        canvas_rect,
        state,
        &mut joint_hit_targets,
        &mut attachment_hit_targets,
        &mut body_segments,
    );

    // ── Post-immutable rendering (force arrows, overlays, hints) ─────────
    rendering::render_overlays(
        ui,
        &painter,
        canvas_rect,
        state,
        &joint_hit_targets,
        &attachment_hit_targets,
        &body_segments,
        solver_converged,
        solver_residual,
        solver_iterations,
        show_debug,
    );

    // ── Interaction: drag, pan, zoom, tools, selection ───────────────────
    let right_drag_ended = interaction::handle_interaction(
        ui,
        &painter,
        canvas_rect,
        &response,
        state,
        &joint_hit_targets,
        &attachment_hit_targets,
        &body_segments,
    );

    // ── Context menu ────────────────────────────────────────────────────
    context_menu::handle_context_menu(
        &response,
        state,
        &joint_hit_targets,
        &attachment_hit_targets,
        &body_segments,
        &grounded_revolute_ids,
        &current_driver_joint,
        right_drag_ended,
    );
}

#[cfg(test)]
mod tests {
    use crate::forces::elements::*;
    use super::rendering::fill_force_template;

    #[test]
    fn fill_template_linear_spring() {
        let template = ForceElement::LinearSpring(LinearSpringElement {
            body_a: String::new(), point_a: [0.0, 0.0], point_a_name: None,
            body_b: String::new(), point_b: [0.0, 0.0], point_b_name: None,
            stiffness: 500.0, free_length: 0.1,
        });
        let result = fill_force_template(
            &template,
            "ground", [1.0, 2.0], Some("pin_a".to_string()),
            "crank", [3.0, 4.0], None,
        );
        match result {
            ForceElement::LinearSpring(s) => {
                assert_eq!(s.body_a, "ground");
                assert_eq!(s.body_b, "crank");
                assert_eq!(s.point_a, [1.0, 2.0]);
                assert_eq!(s.point_b, [3.0, 4.0]);
                assert_eq!(s.point_a_name, Some("pin_a".to_string()));
                assert!(s.point_b_name.is_none());
                assert!((s.stiffness - 500.0).abs() < 1e-12);
                assert!((s.free_length - 0.1).abs() < 1e-12);
            }
            _ => panic!("expected LinearSpring"),
        }
    }

    #[test]
    fn fill_template_linear_damper() {
        let template = ForceElement::LinearDamper(LinearDamperElement {
            body_a: String::new(), point_a: [0.0, 0.0], point_a_name: None,
            body_b: String::new(), point_b: [0.0, 0.0], point_b_name: None,
            damping: 42.0,
        });
        let result = fill_force_template(
            &template, "a", [1.0, 0.0], None, "b", [0.0, 1.0], None,
        );
        match result {
            ForceElement::LinearDamper(d) => {
                assert_eq!(d.body_a, "a");
                assert_eq!(d.body_b, "b");
                assert!((d.damping - 42.0).abs() < 1e-12);
            }
            _ => panic!("expected LinearDamper"),
        }
    }

    #[test]
    fn fill_template_gas_spring() {
        let template = ForceElement::GasSpring(GasSpringElement {
            body_a: String::new(), point_a: [0.0, 0.0], point_a_name: None,
            body_b: String::new(), point_b: [0.0, 0.0], point_b_name: None,
            initial_force: 200.0, extended_length: 0.5, stroke: 0.2,
            damping: 1.0, polytropic_exp: 1.3,
        });
        let result = fill_force_template(
            &template, "frame", [0.1, 0.2], Some("mt".to_string()),
            "arm", [0.3, 0.4], Some("mt2".to_string()),
        );
        match result {
            ForceElement::GasSpring(g) => {
                assert_eq!(g.body_a, "frame");
                assert_eq!(g.body_b, "arm");
                assert_eq!(g.point_a_name, Some("mt".to_string()));
                assert_eq!(g.point_b_name, Some("mt2".to_string()));
                assert!((g.initial_force - 200.0).abs() < 1e-12);
                assert!((g.polytropic_exp - 1.3).abs() < 1e-12);
            }
            _ => panic!("expected GasSpring"),
        }
    }

    #[test]
    fn fill_template_linear_actuator() {
        let template = ForceElement::LinearActuator(LinearActuatorElement {
            body_a: String::new(), point_a: [0.0, 0.0], point_a_name: None,
            body_b: String::new(), point_b: [0.0, 0.0], point_b_name: None,
            force: 999.0, speed_limit: 0.5,
            stroke_min: 0.0, stroke_max: 0.0,
            end_stop_stiffness: 10000.0, end_stop_damping: 10.0, end_stop_restitution: 0.5,
        });
        let result = fill_force_template(
            &template, "g", [0.0, 0.0], None, "link", [1.0, 0.0], None,
        );
        match result {
            ForceElement::LinearActuator(a) => {
                assert_eq!(a.body_a, "g");
                assert_eq!(a.body_b, "link");
                assert!((a.force - 999.0).abs() < 1e-12);
                assert!((a.speed_limit - 0.5).abs() < 1e-12);
            }
            _ => panic!("expected LinearActuator"),
        }
    }
    /// Headless canvas clicks driving the weight (point-mass) handlers.
    mod weight_clicks {
        use eframe::egui::{self, Pos2};
        use nalgebra::Vector2;

        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::{AppState, EditorTool, SelectedEntity};
        use crate::gui::test_support::{key_press, primary_button, primary_button_with, sorted_link_ids};
        use super::super::draw_canvas;
        use super::super::hit_testing::tests::weight_screen;

        /// One canvas frame with `events`, `modifiers` held; returns egui's
        /// output (cursor icon etc.).
        fn frame_with(
            ctx: &egui::Context,
            state: &mut AppState,
            events: Vec<egui::Event>,
            modifiers: egui::Modifiers,
        ) -> egui::FullOutput {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                events,
                modifiers,
                ..Default::default()
            };
            ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw_canvas(ui, state));
            })
        }

        /// One canvas frame with `events` and no modifiers.
        pub(super) fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> egui::FullOutput {
            frame_with(ctx, state, events, egui::Modifiers::NONE)
        }

        fn click_with(ctx: &egui::Context, state: &mut AppState, pos: Pos2, modifiers: egui::Modifiers) {
            let _ = frame_with(ctx, state, vec![egui::Event::PointerMoved(pos)], modifiers);
            let _ = frame_with(ctx, state, vec![primary_button_with(pos, true, modifiers)], modifiers);
            let _ = frame_with(ctx, state, vec![primary_button_with(pos, false, modifiers)], modifiers);
        }

        fn click(ctx: &egui::Context, state: &mut AppState, pos: Pos2) {
            click_with(ctx, state, pos, egui::Modifiers::NONE);
        }

        /// Press at `from` and drag through the midpoint to `to` without
        /// releasing. egui reports the drag once the pointer has moved more
        /// than 6 px from the press.
        fn press_and_drag(ctx: &egui::Context, state: &mut AppState, from: Pos2, to: Pos2) {
            frame(ctx, state, vec![egui::Event::PointerMoved(from)]);
            frame(ctx, state, vec![primary_button(from, true)]);
            frame(ctx, state, vec![egui::Event::PointerMoved(from + (to - from) * 0.5)]);
            frame(ctx, state, vec![egui::Event::PointerMoved(to)]);
        }

        fn release(ctx: &egui::Context, state: &mut AppState, at: Pos2) {
            frame(ctx, state, vec![primary_button(at, false)]);
        }

        fn drag(ctx: &egui::Context, state: &mut AppState, from: Pos2, to: Pos2) {
            press_and_drag(ctx, state, from, to);
            release(ctx, state, to);
        }

        /// Four-bar with a 2 kg weight "W1" on the first sorted link, after two
        /// idle frames: the first applies the pending fit-to-view (so
        /// `state.view` is final), the second adapts the grid spacing to the
        /// fitted zoom (so `state.grid` snaps as a drag will). Returns the
        /// context, state, that link and a second link.
        pub(super) fn setup() -> (egui::Context, AppState, String, String) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::FourBar);
            let links = sorted_link_ids(&state);
            assert_eq!(state.add_point_mass(&links[0], 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state, links[0].clone(), links[1].clone())
        }

        fn world_of(state: &AppState, body: &str, local: [f64; 2]) -> [f64; 2] {
            let mech = state.mechanism.as_ref().unwrap();
            let p = mech.state().body_point_global(body, &Vector2::new(local[0], local[1]), &state.q);
            [p.x, p.y]
        }

        fn screen_of(state: &AppState, world: [f64; 2]) -> Pos2 {
            let [x, y] = state.view.world_to_screen(world[0], world[1]);
            Pos2::new(x, y)
        }

        /// The body-local point under screen position `pos` on `body`.
        fn local_under(state: &AppState, body: &str, pos: Pos2) -> [f64; 2] {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            state.world_to_body_local(body, wx, wy)
        }

        fn weight(body: &str, id: &str) -> SelectedEntity {
            SelectedEntity::Weight { body_id: body.to_string(), weight_id: id.to_string() }
        }

        /// The world point a drop at screen `pos` lands on: under the
        /// pointer, snapped to the grid when snapping is on.
        fn drop_world(state: &AppState, pos: Pos2) -> [f64; 2] {
            let [wx, wy] = state.view.screen_to_world(pos.x, pos.y);
            let (gx, gy) = state.grid.snap_point(wx, wy);
            [gx, gy]
        }

        /// Screen ends (sorted pin names) of a two-pin link's bar.
        fn link_ends(state: &AppState, body: &str) -> (Pos2, Pos2) {
            let mech = state.mechanism.as_ref().unwrap();
            let pins = &mech.bodies()[body].attachment_points;
            let mut names: Vec<&String> = pins.keys().collect();
            names.sort();
            assert_eq!(names.len(), 2, "fixture: a two-pin link");
            let end = |name: &String| screen_of(state, world_of(state, body, [pins[name].x, pins[name].y]));
            (end(names[0]), end(names[1]))
        }

        /// The screen point `t` of the way along a two-pin link's bar.
        fn along_link(state: &AppState, body: &str, t: f32) -> Pos2 {
            let (a, b) = link_ends(state, body);
            a + (b - a) * t
        }

        fn assert_close(what: &str, got: [f64; 2], want: [f64; 2]) {
            assert!(
                (got[0] - want[0]).abs() < 1e-9 && (got[1] - want[1]).abs() < 1e-9,
                "{what}: {got:?} vs {want:?}"
            );
        }

        #[test]
        fn reposition_click_moves_the_weight_by_id_as_one_undo_step() {
            let (ctx, mut state, body, _) = setup();
            let target = Pos2::new(120.0, 700.0); // empty canvas
            let expected = local_under(&state, &body, target);
            let depth = state.undo_history.undo_count();

            state.repositioning_point_mass = Some((body.clone(), "W1".to_string()));
            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W1").expect("W1 keeps its id and link");
            assert_close("reposition", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.repositioning_point_mass.is_none(), "the mode ends after one click");
        }

        #[test]
        fn move_to_link_click_keeps_world_position_id_and_mass() {
            let (ctx, mut state, body, other) = setup();
            let world = world_of(&state, &body, [0.03, 0.02]);
            let expected = state.world_to_body_local(&other, world[0], world[1]);
            // Click the middle of the other link's bar.
            let mech = state.mechanism.as_ref().unwrap();
            let pts: Vec<Vector2<f64>> = mech.bodies()[&other].attachment_points.values().copied().collect();
            assert_eq!(pts.len(), 2, "fixture: a two-pin link");
            let a = world_of(&state, &other, [pts[0].x, pts[0].y]);
            let b = world_of(&state, &other, [pts[1].x, pts[1].y]);
            let mid = screen_of(&state, [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
            let depth = state.undo_history.undo_count();

            state.reassigning_point_mass = Some((body.clone(), "W1".to_string()));
            click(&ctx, &mut state, mid);

            assert!(state.find_point_mass(&body, "W1").is_none(), "W1 left the old link");
            let pm = state.find_point_mass(&other, "W1").expect("W1 is on the clicked link");
            assert_close("reattach keeps the world position", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.reassigning_point_mass.is_none(), "the mode ends after one click");
        }

        #[test]
        fn place_mass_click_places_the_last_mass_on_the_snapped_grid_point() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            assert!(state.grid.snap_enabled, "fixture: snapping is on by default");
            let world = drop_world(&state, target);
            let [wx, wy] = state.view.screen_to_world(target.x, target.y);
            assert_ne!(world, [wx, wy], "fixture: the snap moves the placement");
            let expected = state.world_to_body_local(&body, world[0], world[1]);
            let depth = state.undo_history.undo_count();

            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W2").expect("the new weight gets the next id");
            assert_eq!(pm.mass, 3.5);
            assert_close("placement on the snapped grid point", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
            assert_eq!(state.selected, Some(weight(&body, "W2")), "the new weight is selected");
            assert_eq!(state.link_editor_body.as_deref(), Some(body.as_str()));
            assert_eq!(state.undo_history.undo_count(), depth + 1, "one placement = one undo step");
        }

        #[test]
        fn place_mass_with_snapping_off_lands_under_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            let expected = local_under(&state, &body, target);

            click(&ctx, &mut state, target);

            assert_close("unsnapped placement", state.find_point_mass(&body, "W2").unwrap().local_pos, expected);
        }

        /// The click that places a weight is not also a selection click: the
        /// snap can move the weight beyond the pick radius of the pointer,
        /// where selecting at the pointer would clear the selection.
        #[test]
        fn placing_a_weight_selects_it_even_when_the_snap_moves_it_off_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            let pick = super::super::colors::WEIGHT_HIT_RADIUS;
            let target = (0..40)
                .flat_map(|i| (0..20).map(move |j| Pos2::new(110.0 + i as f32 * 3.7, 640.0 + j as f32 * 3.1)))
                .find(|&p| screen_of(&state, drop_world(&state, p)).distance(p) > pick + 1.0)
                .expect("fixture: an empty-canvas point the snap moves off the pointer");
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());

            click(&ctx, &mut state, target);

            assert!(state.find_point_mass(&body, "W2").is_some());
            assert_eq!(state.selected, Some(weight(&body, "W2")));
        }

        #[test]
        fn the_place_mass_hint_names_the_next_weight_and_its_mass() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());

            let output = frame(&ctx, &mut state, Vec::new());

            let want = format!("Click to place weight W2 (3.5 kg) on '{body}' (Esc to cancel)");
            assert!(crate::gui::test_support::drew_text(&output, &want), "hint {want:?}");
        }

        #[test]
        fn clicking_a_weight_selects_it() {
            let (ctx, mut state, body, _) = setup();
            state.selected = Some(SelectedEntity::Body(body.clone()));
            let at = weight_screen(&state, &body, "W1");
            let depth = state.undo_history.undo_count();

            click(&ctx, &mut state, at);

            assert_eq!(state.selected, Some(weight(&body, "W1")));
            assert!(state.multi_selected.is_empty());
            assert_eq!(state.undo_history.undo_count(), depth, "selecting is not an edit");
        }

        #[test]
        fn clicking_empty_canvas_clears_a_weight_selection() {
            let (ctx, mut state, body, _) = setup();
            state.selected = Some(weight(&body, "W1"));

            click(&ctx, &mut state, Pos2::new(120.0, 700.0));

            assert_eq!(state.selected, None);
        }

        #[test]
        fn a_weight_on_a_pin_wins_the_click_over_the_joint() {
            let (ctx, mut state, body, _) = setup();
            // A weight exactly on one of the link's pins, where a joint is drawn too.
            let pin = {
                let mech = state.mechanism.as_ref().unwrap();
                let mut names: Vec<&String> = mech.bodies()[&body].attachment_points.keys().collect();
                names.sort();
                mech.bodies()[&body].attachment_points[names[0]]
            };
            let id = state.add_point_mass(&body, 1.0, [pin.x, pin.y]).expect("weight on the pin");
            let at = weight_screen(&state, &body, &id);

            click(&ctx, &mut state, at);

            assert_eq!(state.selected, Some(weight(&body, &id)));
        }

        #[test]
        fn shift_click_toggles_a_weight_in_the_multi_selection() {
            let (ctx, mut state, body, _) = setup();
            let at = weight_screen(&state, &body, "W1");

            click_with(&ctx, &mut state, at, egui::Modifiers::SHIFT);
            assert_eq!(state.multi_selected, vec![weight(&body, "W1")]);
            assert_eq!(state.selected, Some(weight(&body, "W1")));

            click_with(&ctx, &mut state, at, egui::Modifiers::SHIFT);
            assert!(state.multi_selected.is_empty());
        }

        #[test]
        fn hovering_a_weight_shows_a_grab_cursor_only_in_select_mode() {
            let (ctx, mut state, body, _) = setup();
            let at = weight_screen(&state, &body, "W1");
            let hover = |state: &mut AppState, pos: Pos2| {
                frame_with(&ctx, state, vec![egui::Event::PointerMoved(pos)], egui::Modifiers::NONE)
                    .platform_output
                    .cursor_icon
            };

            assert_eq!(hover(&mut state, at), egui::CursorIcon::Grab);
            assert_eq!(hover(&mut state, Pos2::new(120.0, 700.0)), egui::CursorIcon::Default);

            state.active_tool = EditorTool::PlaceMass;
            assert_eq!(hover(&mut state, at), egui::CursorIcon::Default, "no weight pick in other tools");
        }

        #[test]
        fn dragging_a_weight_moves_it_on_release_as_one_undo_step() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0); // empty canvas, far from every link
            assert!(state.grid.snap_enabled, "fixture: snapping is on by default");
            let world = drop_world(&state, to);
            let [wx, wy] = state.view.screen_to_world(to.x, to.y);
            assert_ne!(world, [wx, wy], "fixture: the snap moves the drop point");
            let expected = state.world_to_body_local(&body, world[0], world[1]);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&body, "W1").expect("W1 stays on its link");
            assert_close("drop on the snapped grid point", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1, "one drag = one undo step");
            assert_eq!(state.selected, Some(weight(&body, "W1")));
            assert!(state.weight_drag.is_none());

            state.undo();
            let pm = state.find_point_mass(&body, "W1").expect("undo keeps W1");
            assert_eq!(pm.local_pos, [0.03, 0.02], "one undo restores the start position");
        }

        #[test]
        fn a_drag_with_snapping_off_drops_exactly_under_the_pointer() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let expected = local_under(&state, &body, to);

            drag(&ctx, &mut state, from, to);

            assert_close("unsnapped drop", state.find_point_mass(&body, "W1").unwrap().local_pos, expected);
        }

        #[test]
        fn the_drag_preview_leaves_the_blueprint_alone_until_release() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let depth = state.undo_history.undo_count();

            press_and_drag(&ctx, &mut state, from, to);

            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02], "not moved yet");
            assert_eq!(state.undo_history.undo_count(), depth, "no undo entry mid-drag");
            let preview = state.weight_drag.clone().expect("a weight drag is in progress");
            assert_eq!(preview.body_id, body);
            assert_eq!(preview.weight_id, "W1");
            assert_eq!(preview.current_world, drop_world(&state, to));
            assert_eq!(state.selected, Some(weight(&body, "W1")), "pressing a weight selects it");
            let cursor = frame_with(&ctx, &mut state, vec![egui::Event::PointerMoved(to)], egui::Modifiers::NONE)
                .platform_output
                .cursor_icon;
            assert_eq!(cursor, egui::CursorIcon::Grabbing);

            release(&ctx, &mut state, to);

            assert_eq!(state.undo_history.undo_count(), depth + 1);
            assert!(state.weight_drag.is_none());
        }

        #[test]
        fn dropping_near_another_link_reattaches_the_weight_at_the_drop_point() {
            let (ctx, mut state, body, other) = setup();
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = along_link(&state, &other, 0.5);
            let expected = local_under(&state, &other, to);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, to);

            assert!(state.find_point_mass(&body, "W1").is_none(), "W1 left its link");
            let pm = state.find_point_mass(&other, "W1").expect("W1 is on the link it was dropped on");
            assert_close("reattached where it was dropped", pm.local_pos, expected);
            assert_eq!(pm.mass, 2.0);
            assert_eq!(state.undo_history.undo_count(), depth + 1, "a reattach is one undo step");
            assert_eq!(state.selected, Some(weight(&other, "W1")), "the selection follows the weight");
        }

        #[test]
        fn dropping_on_its_own_link_next_to_a_neighbour_keeps_the_weight_there() {
            let (ctx, mut state, body, _) = setup();
            state.grid.snap_enabled = false;
            // 90 % along its own link, beside the pin it shares with the rocker,
            // so the rocker's bar is inside the pick radius too.
            let to = along_link(&state, &body, 0.9);
            let (ra, rb) = link_ends(&state, "rocker");
            let near_rocker = super::super::hit_testing::project_onto_segment(to, ra, rb)
                .is_some_and(|(_, d)| d > 0.5 && d <= 60.0);
            assert!(near_rocker, "fixture: the drop point is within the pick radius of the rocker");
            let from = weight_screen(&state, &body, "W1");
            let expected = local_under(&state, &body, to);

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&body, "W1").expect("the nearest link wins: W1 stays");
            assert_close("moved along its own link", pm.local_pos, expected);
        }

        #[test]
        fn dragging_works_under_a_mounting_angle() {
            let (ctx, mut state, body, other) = setup();
            state.mounting_angle = 0.3;
            frame(&ctx, &mut state, Vec::new()); // the canvas copies it into the view
            state.grid.snap_enabled = false;
            let from = weight_screen(&state, &body, "W1");
            let to = along_link(&state, &other, 0.5);
            let expected = local_under(&state, &other, to);

            drag(&ctx, &mut state, from, to);

            let pm = state.find_point_mass(&other, "W1").expect("reattached in the rotated view");
            assert_close("rotated view drop", pm.local_pos, expected);
        }

        #[test]
        fn escape_cancels_a_weight_drag_without_an_edit() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            let to = Pos2::new(120.0, 700.0);
            let depth = state.undo_history.undo_count();

            press_and_drag(&ctx, &mut state, from, to);
            frame(&ctx, &mut state, vec![key_press(egui::Key::Escape)]);
            assert!(state.weight_drag.is_none(), "Esc drops the preview");
            release(&ctx, &mut state, to);

            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }

        #[test]
        fn releasing_a_weight_outside_the_canvas_cancels_the_drag() {
            let (ctx, mut state, body, _) = setup();
            let from = weight_screen(&state, &body, "W1");
            // Inside the window, but in the panel margin around the canvas.
            let outside = Pos2::new(3.0, 400.0);
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, outside);

            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }

        #[test]
        fn dragging_empty_canvas_still_pans_the_view() {
            let (ctx, mut state, body, _) = setup();
            let offset = state.view.offset;

            drag(&ctx, &mut state, Pos2::new(120.0, 700.0), Pos2::new(220.0, 650.0));

            assert_ne!(state.view.offset, offset, "the view panned");
            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
        }

        #[test]
        fn a_weight_does_not_drag_outside_select_mode() {
            let (ctx, mut state, body, _) = setup();
            state.active_tool = EditorTool::AddGroundPivot;
            let from = weight_screen(&state, &body, "W1");
            let depth = state.undo_history.undo_count();

            drag(&ctx, &mut state, from, Pos2::new(120.0, 700.0));

            assert!(state.weight_drag.is_none());
            assert_eq!(state.find_point_mass(&body, "W1").unwrap().local_pos, [0.03, 0.02]);
            assert_eq!(state.undo_history.undo_count(), depth);
        }
    }

    /// Headless canvas frames checking the weight readout: arrows coloured
    /// at the current pose, hover tooltip, selected-weight card.
    mod weight_readout {
        use eframe::egui::{self, Pos2};

        use crate::forces::elements::ForceElement;
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::{AppState, SelectedEntity};
        use crate::gui::test_support::{drawn_line_colors, drawn_texts, pose_at, swept_lift, visit_shapes};
        use super::super::colors::{
            FORCE_ZONE_OVERLAP_FILL, WEIGHT_COLOR, WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR,
            WEIGHT_NEUTRAL_COLOR, WEIGHT_RADIUS,
        };
        use super::super::hit_testing::tests::weight_screen;
        use super::weight_clicks::{frame, setup};

        /// The swept robot lift after two idle frames (fit to view, grid).
        fn lift() -> (egui::Context, AppState) {
            let mut state = swept_lift();
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state)
        }

        #[test]
        fn weight_arrows_take_the_classification_colour_of_the_current_pose() {
            let (ctx, mut state) = lift();
            for (deg, want, not) in [
                (45.0, WEIGHT_HURTING_COLOR, WEIGHT_HELPING_COLOR),
                (135.0, WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR),
            ] {
                pose_at(&mut state, deg);
                let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));
                assert!(colors.contains(&want), "{deg} deg: an arrow in {want:?}");
                assert!(!colors.contains(&not), "{deg} deg: no arrow in {not:?}");
                assert!(!colors.contains(&WEIGHT_COLOR), "{deg} deg: every weight is classified");
            }
            pose_at(&mut state, 90.0);
            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));
            assert!(colors.contains(&WEIGHT_NEUTRAL_COLOR), "90 deg: the weights move sideways");
        }

        #[test]
        fn a_weight_the_sweep_has_not_seen_gets_a_weight_coloured_arrow() {
            let (ctx, mut state) = lift();
            pose_at(&mut state, 45.0);
            // Moving W1 rebuilds; the sweep is only recomputed later (debounced).
            assert!(state.move_point_mass("rocker", "W1", "rocker", [0.05, 0.0]));

            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));

            assert!(colors.contains(&WEIGHT_COLOR), "W1's arrow waits for the new sweep");
            assert!(colors.contains(&WEIGHT_HURTING_COLOR), "W2 keeps its colour");
        }

        #[test]
        fn no_weight_arrows_without_gravity() {
            let (ctx, mut state) = lift();
            state.gravity_magnitude = 0.0;
            state.sync_gravity();
            state.compute_sweep();
            pose_at(&mut state, 45.0);

            let colors = drawn_line_colors(&frame(&ctx, &mut state, Vec::new()));

            for color in [WEIGHT_HELPING_COLOR, WEIGHT_HURTING_COLOR, WEIGHT_NEUTRAL_COLOR, WEIGHT_COLOR] {
                assert!(!colors.contains(&color), "no arrow in {color:?}");
            }
        }

        #[test]
        fn weights_have_no_permanent_label() {
            let (ctx, mut state, _, _) = setup();
            state.compute_sweep();
            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            assert!(!texts.iter().any(|t| t.contains("kg")), "{texts:?}");
        }

        /// Texts drawn while the pointer rests at `at`: a tooltip shows from
        /// its second frame (egui lays it out unseen first).
        fn texts_hovering(ctx: &egui::Context, state: &mut AppState, at: Pos2) -> Vec<String> {
            frame(ctx, state, vec![egui::Event::PointerMoved(at)]);
            drawn_texts(&frame(ctx, state, Vec::new()))
        }

        #[test]
        fn hovering_a_weight_shows_its_name_mass_and_share() {
            let (ctx, mut state, body, _) = setup();
            state.compute_sweep();
            let at = weight_screen(&state, &body, "W1");

            let texts = texts_hovering(&ctx, &mut state, at);

            assert!(texts.iter().any(|t| t == "W1"), "{texts:?}");
            assert!(texts.iter().any(|t| t == "Mass: 2 kg"), "{texts:?}");
            assert!(texts.iter().any(|t| t.starts_with("Torque share: ")), "four-bar: driver torque shares: {texts:?}");

            let texts = texts_hovering(&ctx, &mut state, Pos2::new(120.0, 700.0));
            assert!(!texts.iter().any(|t| t == "Mass: 2 kg"), "no tooltip away from the weight: {texts:?}");
        }

        #[test]
        fn the_selected_weight_shows_its_readout_next_to_it() {
            let (ctx, mut state, body, _) = setup();
            state.compute_sweep();
            state.selected = Some(SelectedEntity::Weight { body_id: body.clone(), weight_id: "W1".to_string() });

            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            let card = texts.iter().find(|t| t.starts_with("W1\nMass: 2 kg\n")).expect("the readout card");
            assert!(card.contains("Torque share: "), "{card:?}");

            // Hovering the selected weight adds no tooltip on top of its card.
            let at = weight_screen(&state, &body, "W1");
            let texts = texts_hovering(&ctx, &mut state, at);
            assert!(!texts.iter().any(|t| t == "Mass: 2 kg"), "{texts:?}");
            assert!(texts.iter().any(|t| t.starts_with("W1\nMass: 2 kg\n")), "the card stays: {texts:?}");
        }

        #[test]
        fn the_actuator_label_names_the_direction_and_who_does_the_work() {
            let (ctx, mut state) = lift();
            for (deg, word) in [(45.0, ", motoring"), (135.0, ", braking")] {
                pose_at(&mut state, deg);
                let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
                assert!(
                    texts.iter().any(|t| (t.contains(" push") || t.contains(" pull")) && t.ends_with(word)),
                    "{deg} deg: a label ending in {word:?} in {texts:?}"
                );
            }
        }

        /// A weight on a link inside a force zone draws after the zone's
        /// overlap highlight: the highlight is an additive fill, so a weight
        /// painted under it comes out the same yellow (BL-042: the user's
        /// 150 lb weight on the press tool could not be seen).
        #[test]
        fn a_weight_inside_a_force_zone_draws_over_the_zone_highlight() {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramPress);
            // A zone over the whole canvas: the coupler's geometry overlaps it at any pose.
            for force in &mut state.blueprint.as_mut().expect("blueprint").forces {
                if let ForceElement::ForceZone(zone) = force {
                    zone.zone_min = [-10.0, -10.0];
                    zone.zone_max = [10.0, 10.0];
                }
            }
            state.rebuild();
            assert_eq!(state.add_point_mass("coupler", 2.0, [0.02, 0.0]).as_deref(), Some("W1"));
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            let output = frame(&ctx, &mut state, Vec::new());

            let marker_fill = state.nc(WEIGHT_COLOR);
            let (mut highlight, mut marker, mut k) = (None, None, 0);
            visit_shapes(&output, |shape| {
                match shape {
                    egui::Shape::Path(path) if path.fill == FORCE_ZONE_OVERLAP_FILL => highlight = Some(k),
                    egui::Shape::Circle(c) if c.radius == WEIGHT_RADIUS && c.fill == marker_fill => marker = Some(k),
                    _ => {}
                }
                k += 1;
            });
            let highlight = highlight.expect("the coupler's overlap highlight is drawn");
            let marker = marker.expect("the weight marker is drawn");
            assert!(marker > highlight, "the weight (shape {marker}) is painted under the highlight (shape {highlight})");
        }
    }

    /// Headless canvas frames checking how a circle geometry and a force
    /// zone's contact point are drawn and dragged (decisions R-2, R-3, R-4).
    mod geometry_drawing {
        use eframe::egui::{self, Pos2};

        use crate::core::body::{BodyGeometry, GeometryShape};
        use crate::forces::elements::{ForceElement, ZoneAppMode};
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;
        use crate::gui::test_support::{drawn_texts, primary_button, visit_shapes};
        use super::weight_clicks::frame;

        /// The Parallelogram Press with a 20 mm wheel on its coupler where its
        /// rectangle was, the zone grown over the canvas, the zone's force at
        /// the wheel's contact point when `contact`; after two idle frames.
        fn press_with_wheel(contact: bool) -> (egui::Context, AppState) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::ParallelogramPress);
            let bp = state.blueprint.as_mut().expect("blueprint");
            let coupler = bp.bodies.get_mut("coupler").expect("coupler");
            let hub = coupler.geometry.as_ref().expect("the sample's geometry").offset;
            coupler.geometry = Some(BodyGeometry::circle(0.02, hub).unwrap());
            for force in &mut bp.forces {
                if let ForceElement::ForceZone(zone) = force {
                    zone.zone_min = [-10.0, -10.0];
                    zone.zone_max = [10.0, 10.0];
                    zone.at_contact_point = contact;
                }
            }
            state.rebuild();
            let ctx = egui::Context::default();
            frame(&ctx, &mut state, Vec::new());
            frame(&ctx, &mut state, Vec::new());
            (ctx, state)
        }

        fn zone(state: &AppState) -> crate::forces::elements::ForceZoneElement {
            state
                .mechanism
                .as_ref()
                .unwrap()
                .forces()
                .iter()
                .find_map(|f| if let ForceElement::ForceZone(z) = f { Some(z.clone()) } else { None })
                .expect("the press's zone")
        }

        #[test]
        fn a_circle_geometry_is_drawn_as_a_circle_of_its_radius() {
            let (ctx, mut state) = press_with_wheel(false);
            let output = frame(&ctx, &mut state, Vec::new());
            let geo = state.mechanism.as_ref().unwrap().bodies()["coupler"].geometry.clone().unwrap();
            assert_eq!(geo.shape, GeometryShape::Circle);
            let a = state.view.world_to_screen(0.0, 0.0);
            let b = state.view.world_to_screen(0.01, 0.0);
            let radius_px = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let pose = state.mechanism.as_ref().unwrap().state().get_pose("coupler", &state.q);
            let centre_world = geo.centre_world(pose.0, pose.1, pose.2);
            let centre = state.view.world_to_screen(centre_world.x, centre_world.y);
            let mut found = false;
            visit_shapes(&output, |shape| {
                if let egui::Shape::Circle(c) = shape {
                    if c.stroke.color == egui::Color32::from_rgb(255, 165, 0)
                        && (c.radius - radius_px).abs() < 0.5
                        && (c.center.x - centre[0]).abs() < 0.5
                        && (c.center.y - centre[1]).abs() < 0.5
                    {
                        found = true;
                    }
                }
            });
            assert!(found, "a circle of radius {radius_px} px centred at {centre:?} in the geometry colour");
        }

        #[test]
        fn the_contact_point_marker_reads_f_contact() {
            let (ctx, mut state) = press_with_wheel(true);
            let texts = drawn_texts(&frame(&ctx, &mut state, Vec::new()));
            assert!(texts.iter().any(|t| t == "F (contact)"), "{texts:?}");
            assert!(!texts.iter().any(|t| t == "F (locked)"), "{texts:?}");
        }

        #[test]
        fn dragging_the_contact_marker_locks_it_where_it_drops() {
            let (ctx, mut state) = press_with_wheel(true);
            let fz = zone(&state);
            let mech = state.mechanism.as_ref().unwrap();
            let world = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the contact point");
            let s = state.view.world_to_screen(world.x, world.y);
            let from = Pos2::new(s[0], s[1]);
            let to = from + egui::vec2(40.0, 30.0);
            // egui starts the drag once the pointer passes its click distance
            // (6 px) and reports the pointer's position then, which must still
            // be within the marker's hit radius (12 px): a first step of 8 px.
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from)]);
            frame(&ctx, &mut state, vec![primary_button(from, true)]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(from + egui::vec2(8.0, 0.0))]);
            frame(&ctx, &mut state, vec![egui::Event::PointerMoved(to)]);
            frame(&ctx, &mut state, vec![primary_button(to, false)]);
            let fz = zone(&state);
            assert_eq!(fz.app_mode(), ZoneAppMode::Locked, "the drop locks the point");
            let [wx, wy] = state.view.screen_to_world(to.x, to.y);
            let mech = state.mechanism.as_ref().unwrap();
            let dropped = crate::gui::canvas::rendering::force_zone_app_point_world(&fz, mech, mech.state(), &state.q)
                .expect("the locked point");
            assert!((dropped.x - wx).abs() < 1e-9 && (dropped.y - wy).abs() < 1e-9, "{dropped:?} vs ({wx}, {wy})");
        }
    }
}
