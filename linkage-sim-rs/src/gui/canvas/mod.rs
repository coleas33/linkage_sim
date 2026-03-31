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

use colors::*;
use hit_testing::{AttachmentHit, BodySegment};

/// Draw the 2D mechanism canvas with interaction.
pub fn draw_canvas(ui: &mut egui::Ui, state: &mut AppState) {
    let (response, painter) =
        ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
    let canvas_rect = response.rect;

    // Sync mounting angle into view transform so canvas rendering rotates.
    state.view.mounting_angle = state.mounting_angle;

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
}
