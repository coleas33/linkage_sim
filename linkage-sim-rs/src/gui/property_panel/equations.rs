//! Live "Equations" panel — symbolic Φ_J* + numerical residuals + λ multipliers.
//!
//! Renders an instantaneous snapshot of the constraint system at the current
//! pose `state.q` and time `t_mech`. Each constraint gets a `CollapsingHeader`
//! showing its symbolic form, residual norm, and (when statics solves) its
//! Lagrange multiplier with units. The driver row is highlighted with a
//! distinct color so the user can spot the input constraint at a glance.
//!
//! Performance: per draw, we run one `assemble_constraints` and one
//! `solve_statics` call. Both are cheap for typical 4-bar/6-bar mechanisms
//! (≪ 1 ms). The statics solve is recomputed every frame instead of cached
//! because the panel is purely diagnostic and the cost is negligible vs.
//! avoiding stale-data bugs from skipped invalidation paths.

use eframe::egui;

use crate::core::constraint::Constraint;
use crate::gui::eq_rendering::{
    kind_at, lambda_string, n_constraints, residual_norm_string, symbolic_form, EqKind,
};
use crate::gui::state::AppState;
use crate::solver::assembly::assemble_constraints;
use crate::solver::statics::solve_statics;

/// Draw the live equations panel into the given `ui`.
pub fn draw(state: &AppState, ui: &mut egui::Ui) {
    let Some(mech) = state.mechanism.as_ref() else {
        ui.label("No mechanism loaded.");
        return;
    };
    if !mech.is_built() {
        ui.label("Mechanism not built — cannot render equations.");
        return;
    }

    let n_coords = mech.state().n_coords();
    let n_eq = mech.n_constraints();
    let n_constr = n_constraints(mech);
    let dof = n_coords as isize - n_eq as isize;

    // Compute t_mech from the active driver state. Mirrors the convention
    // used by `solve_at_angle` / `solve_at_stroke`: t = (param - param_0) / rate
    // with a guard against zero rate.
    let t_mech = compute_t_mech(state);

    // Φ at current pose.
    let phi = assemble_constraints(mech, &state.q, t_mech);
    // λ — best-effort. If the statics solve fails we just omit multipliers.
    let lambdas = solve_statics(mech, &state.q, t_mech).ok().map(|s| s.lambdas);

    // ── Header ────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        ui.small(format!("m = {}", n_eq));
        ui.small("\u{2022}");
        ui.small(format!("n = {}", n_coords));
        ui.small("\u{2022}");
        let dof_color = if dof == 1 {
            egui::Color32::from_rgb(120, 200, 140)
        } else if dof < 1 {
            egui::Color32::from_rgb(220, 100, 100)
        } else {
            egui::Color32::from_rgb(220, 180, 60)
        };
        ui.colored_label(state.nc(dof_color), format!("DOF = {}", dof));
    });
    ui.add_space(4.0);

    // Quick legend so users learn the symbology.
    egui::CollapsingHeader::new(
        egui::RichText::new("Legend").small().weak(),
    )
    .id_salt("eq_legend")
    .default_open(false)
    .show(ui, |ui| {
        ui.small("Φ_rev: revolute joint (2 eqs)");
        ui.small("Φ_pri: prismatic joint (2 eqs)");
        ui.small("Φ_fix: fixed joint (3 eqs)");
        ui.small("Φ_cam: cam follower (1 eq)");
        ui.small("Φ_rd: revolute driver (1 eq, λ in N·m)");
        ui.small("Φ_ld: linear driver (1 eq, λ in N)");
    });

    ui.separator();

    // ── Per-constraint blocks ─────────────────────────────────────────
    for idx in 0..n_constr {
        let Some(c) = mech.all_constraints().nth(idx) else { continue };
        let Some(kind) = kind_at(mech, idx) else { continue };
        let id_label = c.id().to_string();
        let header_color = color_for_kind(state, kind);

        // Drivers get a distinct, bold header so users find them fast.
        let header_text = if kind.is_driver() {
            egui::RichText::new(format!("{}  [{}]  ▶ driver", id_label, kind.short_label()))
                .color(header_color)
                .strong()
        } else {
            egui::RichText::new(format!("{}  [{}]", id_label, kind.short_label()))
                .color(header_color)
        };

        egui::CollapsingHeader::new(header_text)
            .id_salt(format!("eq_panel_{}", id_label))
            .default_open(kind.is_driver())
            .show(ui, |ui| {
                // Body endpoints (helps the user link IDs to geometry).
                ui.small(format!(
                    "bodies: {} \u{2194} {}",
                    c.body_i_id(),
                    c.body_j_id()
                ));

                // Symbolic form — wrapped to the panel width.
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(symbolic_form(mech, idx))
                            .monospace()
                            .small(),
                    )
                    .wrap(),
                );

                // Numerical residual (Φ value).
                ui.small(residual_norm_string(mech, idx, &phi));

                // Lagrange multiplier.
                let lam_str = lambda_string(mech, idx, lambdas.as_ref());
                if !lam_str.is_empty() {
                    ui.small(lam_str);
                } else if lambdas.is_none() {
                    ui.small(
                        egui::RichText::new("λ unavailable (statics solve failed)").weak(),
                    );
                }
            });
    }
}

/// Color for a constraint header by kind. Goes through `state.nc()` so
/// Nathan Mode (grayscale) is honored automatically.
fn color_for_kind(state: &AppState, kind: EqKind) -> egui::Color32 {
    let raw = match kind {
        EqKind::Revolute => egui::Color32::from_rgb(80, 160, 255),
        EqKind::Prismatic => egui::Color32::from_rgb(180, 130, 255),
        EqKind::Fixed => egui::Color32::from_rgb(200, 200, 200),
        EqKind::CamFollower => egui::Color32::from_rgb(100, 220, 140),
        EqKind::RevoluteDriver | EqKind::LinearDriver => {
            egui::Color32::from_rgb(255, 190, 80)
        }
    };
    state.nc(raw)
}

/// Compute the mechanism time `t_mech` consistent with `solve_at_angle` /
/// `solve_at_stroke`: maps the active driver's parameter back to time using
/// the rate. Returns 0 if no driver or rate is near zero (defense in depth
/// against div-by-zero from edge-case driver configs).
fn compute_t_mech(state: &AppState) -> f64 {
    use crate::gui::state::DriverKind;
    match state.driver_kind {
        DriverKind::Revolute { angle: _, omega, theta_0 } => {
            if omega.abs() > f64::EPSILON {
                (state.driver_angle - theta_0) / omega
            } else {
                0.0
            }
        }
        DriverKind::Linear {
            stroke,
            velocity,
            length_0,
        } => {
            if velocity.abs() > f64::EPSILON {
                (stroke - length_0) / velocity
            } else {
                0.0
            }
        }
        DriverKind::None => 0.0,
    }
}
