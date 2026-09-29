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

pub use colors::to_grayscale;
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
        use crate::gui::state::{AppState, EditorTool};
        use crate::gui::test_support::sorted_link_ids;
        use super::super::draw_canvas;

        fn frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| draw_canvas(ui, state));
            });
        }

        fn click(ctx: &egui::Context, state: &mut AppState, pos: Pos2) {
            let button = |pressed| egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: egui::Modifiers::NONE,
            };
            frame(ctx, state, vec![egui::Event::PointerMoved(pos)]);
            frame(ctx, state, vec![button(true)]);
            frame(ctx, state, vec![button(false)]);
        }

        /// Four-bar with a 2 kg weight "W1" on the first sorted link, after one
        /// idle frame (which applies the pending fit-to-view, so `state.view`
        /// is final). Returns the context, state, that link and a second link.
        fn setup() -> (egui::Context, AppState, String, String) {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::FourBar);
            let links = sorted_link_ids(&state);
            assert_eq!(state.add_point_mass(&links[0], 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
            let ctx = egui::Context::default();
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
        fn place_mass_click_uses_the_last_point_mass() {
            let (ctx, mut state, body, _) = setup();
            state.last_point_mass_kg = 3.5;
            state.active_tool = EditorTool::PlaceMass;
            state.place_mass_body = Some(body.clone());
            let target = Pos2::new(150.0, 650.0);
            let expected = local_under(&state, &body, target);

            click(&ctx, &mut state, target);

            let pm = state.find_point_mass(&body, "W2").expect("the new weight gets the next id");
            assert_eq!(pm.mass, 3.5);
            assert_close("placement", pm.local_pos, expected);
            assert_eq!(state.last_point_mass_kg, 3.5);
            assert_eq!(state.active_tool, EditorTool::Select);
        }
    }
}
