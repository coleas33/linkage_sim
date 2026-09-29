//! Canvas weights (point masses): markers, gravity arrows, the hover and
//! selection readout, and the placement hint (payload weights, spec Track 2
//! section 3, "Canvas readout at the current pose").
//!
//! Each weight draws its marker and an arrow in the gravity direction whose
//! length grows with its mass, coloured by whether the weight helps (green),
//! hurts (red) or is neutral (gray) at the current pose:
//! `WeightBreakdown::classification` at `AppState::current_sweep_index`,
//! through `canvas::classification_color`, the palette the Weight Breakdown
//! plot uses. Before the sweep has a breakdown for the weight the arrow is
//! drawn in the weight colour. Hovering a weight shows its name, mass and
//! current force share in a tooltip; the selected weight shows the same
//! readout next to its marker. There are no permanent labels.

use eframe::egui::{self, Pos2, Stroke, Vec2};

use crate::analysis::gravity_breakdown::{gravity_vector, point_mass_title, Classification};
use crate::core::state::GROUND_ID;
use crate::gui::state::{format_mass_kg, AppState, SelectedEntity, ViewTransform};
use crate::gui::sweep::ShareBasis;
use crate::io::{next_point_mass_id, PointMassJson};

use super::super::colors::*;
use super::super::hit_testing::{find_point_mass_at, point_mass_screen_pos};
use super::super::interaction::weights_interactive;
use super::draw_pill_label;
use super::force_render::format_magnitude;
use super::primitives::draw_arrow;

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

/// Length (px) of the gravity arrow of a weight of `mass` when the heaviest
/// drawn weight is `max_mass`: `WEIGHT_ARROW_MAX_PX * mass / max_mass`, at
/// least `WEIGHT_ARROW_MIN_PX` so a light weight's arrow stays visible. The
/// minimum when either mass is not positive and finite.
pub(super) fn weight_arrow_length_px(mass: f64, max_mass: f64) -> f32 {
    if !(mass.is_finite() && mass > 0.0 && max_mass.is_finite() && max_mass > 0.0) {
        return WEIGHT_ARROW_MIN_PX;
    }
    let fraction = (mass / max_mass).min(1.0) as f32;
    (WEIGHT_ARROW_MAX_PX * fraction).max(WEIGHT_ARROW_MIN_PX)
}

/// Unit screen direction of the gravity vector `g` (world, m/s^2) in
/// `view`, which rotates world directions by the mounting angle and flips
/// y. `None` when gravity is off or not finite.
pub(super) fn gravity_screen_dir(view: &ViewTransform, g: [f64; 2]) -> Option<Vec2> {
    let norm = g[0].hypot(g[1]);
    if !(norm.is_finite() && norm > 0.0) {
        return None;
    }
    let [x0, y0] = view.world_to_screen(0.0, 0.0);
    let [x1, y1] = view.world_to_screen(g[0] / norm, g[1] / norm);
    let dir = Vec2::new(x1 - x0, y1 - y0);
    let length = dir.length();
    (length.is_finite() && length > 0.0).then(|| dir / length)
}

/// Where a weight stands at the current pose, read from the sweep's
/// weight breakdown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct WeightAtPose {
    pub(super) classification: Classification,
    /// Force share at the current sample; NaN near stroke reversal.
    pub(super) force_share: f64,
    pub(super) basis: ShareBasis,
    pub(super) is_stroke: bool,
}

/// The breakdown entry of weight `pm` on `body_id` at the current sweep
/// sample (`AppState::current_sweep_index`). `None` without a breakdown or
/// a current sample, and when the breakdown does not have the weight with
/// its current mass and position (the sweep is recomputed after an edit).
pub(super) fn weight_at_pose(state: &AppState, body_id: &str, pm: &PointMassJson) -> Option<WeightAtPose> {
    let sweep = state.sweep_data.as_ref()?;
    let breakdown = sweep.weight_breakdown.as_ref()?;
    let sample = state.current_sweep_index()?;
    let source = breakdown.sources.iter().position(|s| {
        !s.is_link_self_weight
            && s.id == pm.id
            && s.body_id == body_id
            && s.local_pos == pm.local_pos
            && s.mass == pm.mass
    })?;
    Some(WeightAtPose {
        classification: breakdown.classification(source, sample),
        force_share: *breakdown.force_share.get(source)?.get(sample)?,
        basis: breakdown.basis,
        is_stroke: sweep.sweep_mode.is_stroke(),
    })
}

fn classification_word(class: Classification) -> &'static str {
    match class {
        Classification::Helping => "helping",
        Classification::Hurting => "hurting",
        Classification::Neutral => "neutral",
    }
}

/// Readout name and unit of a force share: actuator force (N), driver
/// torque (N*m) with a revolute driver, driver force (N) with a linear one.
fn share_name_and_unit(basis: ShareBasis, is_stroke: bool) -> (&'static str, &'static str) {
    match basis {
        ShareBasis::ActuatorForce => ("Force share", "N"),
        ShareBasis::DriverTorque if is_stroke => ("Force share", "N"),
        ShareBasis::DriverTorque => ("Torque share", "N\u{00b7}m"),
    }
}

/// A signed share for the readout ("+874 N", "-1.2 kN"); "-" when it is
/// not finite (near stroke reversal).
fn format_share(value: f64, unit: &str) -> String {
    if !value.is_finite() {
        return "-".to_string();
    }
    let sign = if value < 0.0 { "-" } else { "+" };
    format!("{sign}{}", format_magnitude(value.abs(), unit))
}

/// Readout of weight `weight_id` on `body_id` at the current pose: its
/// title ("W1", "Robot (W1)"), its mass, and its current force share with
/// its classification ("Force share: +874 N (helping)"). The share reads
/// "-" near stroke reversal and while the sweep has no breakdown for the
/// weight. `None` when the weight does not exist.
pub(super) fn weight_readout_lines(state: &AppState, body_id: &str, weight_id: &str) -> Option<Vec<String>> {
    let pm = state.find_point_mass(body_id, weight_id)?;
    let share = match weight_at_pose(state, body_id, pm) {
        Some(at) => {
            let (name, unit) = share_name_and_unit(at.basis, at.is_stroke);
            format!("{name}: {} ({})", format_share(at.force_share, unit), classification_word(at.classification))
        }
        None => "Force share: -".to_string(),
    };
    Some(vec![point_mass_title(pm), format!("Mass: {}", format_mass_kg(pm.mass)), share])
}

/// Draw every weight on a moving link: its gravity arrow (see the module
/// docs), its marker (faded while it is dragged, ringed while selected)
/// and, on top, the readout of the selected weight. Weights whose screen
/// position is not finite are skipped.
pub(super) fn draw_weights(painter: &egui::Painter, state: &AppState) {
    let Some(bp) = &state.blueprint else { return };
    let gravity_dir = state.mechanism.as_ref().and_then(|m| gravity_screen_dir(&state.view, gravity_vector(m)));
    let max_mass = bp
        .bodies
        .iter()
        .filter(|(id, _)| id.as_str() != GROUND_ID)
        .flat_map(|(_, body)| body.point_masses.iter().map(|pm| pm.mass))
        .filter(|m| m.is_finite() && *m > 0.0)
        .fold(0.0, f64::max);
    let marker_color = state.nc(WEIGHT_COLOR);
    let selected_ring = Stroke::new(2.0, state.nc(BODY_SELECTED_COLOR));

    for (body_id, body) in &bp.bodies {
        if body_id == GROUND_ID {
            continue;
        }
        for pm in &body.point_masses {
            let Some(center) = point_mass_screen_pos(state, body_id, pm.local_pos) else { continue };
            // A weight being dragged fades where it is; the drag preview
            // (canvas interaction) draws it at the drop point.
            let dragged = state.weight_drag.as_ref().is_some_and(|d| d.body_id == *body_id && d.weight_id == pm.id);
            let fade = |c: egui::Color32| if dragged { c.linear_multiply(0.35) } else { c };
            if let Some(dir) = gravity_dir {
                let class_color = weight_at_pose(state, body_id, pm)
                    .map_or(WEIGHT_COLOR, |at| classification_color(at.classification));
                let tail = center + dir * WEIGHT_RADIUS;
                let tip = tail + dir * weight_arrow_length_px(pm.mass, max_mass);
                draw_arrow(painter, tail, tip, Stroke::new(WEIGHT_ARROW_WIDTH, fade(state.nc(class_color))));
            }
            painter.circle_filled(center, WEIGHT_RADIUS, fade(marker_color));
            let entity = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: pm.id.clone() };
            if state.selected.as_ref() == Some(&entity) || state.multi_selected.contains(&entity) {
                painter.circle_stroke(center, WEIGHT_RADIUS + 3.0, selected_ring);
            }
        }
    }

    if let Some(SelectedEntity::Weight { body_id, weight_id }) = &state.selected {
        let center = state
            .find_point_mass(body_id, weight_id)
            .and_then(|pm| point_mass_screen_pos(state, body_id, pm.local_pos));
        if let (Some(center), Some(lines)) = (center, weight_readout_lines(state, body_id, weight_id)) {
            let offset = WEIGHT_HIT_RADIUS + 4.0;
            draw_pill_label(painter, center + Vec2::new(offset, -offset), &lines.join("\n"), marker_color, egui::Align2::LEFT_BOTTOM);
        }
    }
}

/// Hover readout: a tooltip with the readout of the weight under
/// `hover_pos`, while weights answer the pointer (plain Select mode, no
/// weight drag). The selected weight already shows its readout on the
/// canvas and gets no tooltip. Returns true when a weight is under the
/// pointer, so the caller shows no joint or link tooltip beneath it.
pub(super) fn show_weight_tooltip(ui: &egui::Ui, state: &AppState, hover_pos: Pos2) -> bool {
    if !weights_interactive(state) || state.weight_drag.is_some() {
        return false;
    }
    let Some((body_id, weight_id)) = find_point_mass_at(state, hover_pos, WEIGHT_HIT_RADIUS) else {
        return false;
    };
    let hovered = SelectedEntity::Weight { body_id: body_id.clone(), weight_id: weight_id.clone() };
    if state.selected.as_ref() == Some(&hovered) {
        return true;
    }
    let Some(lines) = weight_readout_lines(state, &body_id, &weight_id) else { return false };
    egui::Tooltip::always_open(ui.ctx().clone(), ui.layer_id(), egui::Id::new("weight_tooltip"), egui::PopupAnchor::Pointer)
        .show(|ui: &mut egui::Ui| {
            let mut lines = lines.into_iter();
            if let Some(title) = lines.next() {
                ui.label(egui::RichText::new(title).strong());
            }
            for line in lines {
                ui.label(line);
            }
        });
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::sweep::WeightBreakdown;
    use crate::gui::test_support::{sample_at, swept_lift};
    use Classification::{Helping, Hurting, Neutral};

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

    #[test]
    fn weight_arrow_length_scales_linearly_up_to_the_heaviest_weight() {
        assert_eq!(weight_arrow_length_px(50.0, 50.0), WEIGHT_ARROW_MAX_PX);
        assert_eq!(weight_arrow_length_px(25.0, 50.0), WEIGHT_ARROW_MAX_PX / 2.0);
        assert_eq!(weight_arrow_length_px(1.0, 50.0), WEIGHT_ARROW_MIN_PX, "light weights keep a visible arrow");
        assert_eq!(weight_arrow_length_px(80.0, 50.0), WEIGHT_ARROW_MAX_PX, "never longer than the maximum");
        for (mass, max) in [(0.0, 50.0), (-1.0, 50.0), (f64::NAN, 50.0), (5.0, 0.0), (5.0, f64::INFINITY)] {
            assert_eq!(weight_arrow_length_px(mass, max), WEIGHT_ARROW_MIN_PX, "mass {mass}, max {max}");
        }
    }

    fn assert_down(dir: Option<Vec2>, what: &str) {
        let dir = dir.unwrap_or_else(|| panic!("{what}: gravity has a direction"));
        assert!(dir.x.abs() < 1e-5 && (dir.y - 1.0).abs() < 1e-5, "{what}: {dir:?} should point down the screen");
    }

    /// The GUI rotates gravity by the mounting angle (`sync_gravity`) and the
    /// view rotates the world by the same angle, so on screen the arrows
    /// always point straight down.
    #[test]
    fn gravity_points_down_the_screen_whatever_the_mounting_angle() {
        let mut view = ViewTransform::default();
        assert_down(gravity_screen_dir(&view, [0.0, -9.81]), "level");
        for theta in [0.3_f64, -1.2, std::f64::consts::PI] {
            view.mounting_angle = theta;
            let g = [-9.81 * theta.sin(), -9.81 * theta.cos()];
            assert_down(gravity_screen_dir(&view, g), &format!("mounting angle {theta}"));
        }
        view.mounting_angle = 0.3;
        let unrotated = gravity_screen_dir(&view, [0.0, -9.81]).unwrap();
        assert!(unrotated.x.abs() > 0.1, "a world vector turns with the view: {unrotated:?}");
    }

    #[test]
    fn gravity_screen_dir_is_none_when_gravity_is_off() {
        let view = ViewTransform::default();
        assert_eq!(gravity_screen_dir(&view, [0.0, 0.0]), None);
        assert_eq!(gravity_screen_dir(&view, [f64::NAN, -9.81]), None);
    }

    fn breakdown(state: &AppState) -> &WeightBreakdown {
        state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().unwrap()
    }

    fn source_index(b: &WeightBreakdown, id: &str) -> usize {
        b.sources.iter().position(|s| s.id == id).unwrap()
    }

    /// Put the driver at `deg` (the readouts only read the sweep sample).
    fn at_deg(state: &mut AppState, deg: f64) -> usize {
        state.driver_angle = deg.to_radians();
        sample_at(state, deg)
    }

    /// A sample where W1's force share is NaN (stroke reversal).
    fn reversal_sample(state: &AppState) -> usize {
        let b = breakdown(state);
        let angles = &state.sweep_data.as_ref().unwrap().angles_deg;
        (0..angles.len())
            .find(|&k| b.force_share[source_index(b, "W1")][k].is_nan() && b.gravity_power[0][k].is_finite())
            .expect("fixture: the lift reverses its stroke")
    }

    #[test]
    fn weight_at_pose_reads_the_breakdown_at_the_current_sample() {
        let mut state = swept_lift();
        let reversal_deg = state.sweep_data.as_ref().unwrap().angles_deg[reversal_sample(&state)];
        for (deg, want) in [(45.0, Some(Hurting)), (135.0, Some(Helping)), (90.0, Some(Neutral)), (reversal_deg, None)] {
            let k = at_deg(&mut state, deg);
            for (body, id) in [("rocker", "W1"), ("coupler", "W2")] {
                let pm = state.find_point_mass(body, id).unwrap().clone();
                let at = weight_at_pose(&state, body, &pm).unwrap_or_else(|| panic!("{id} at {deg} deg"));
                let b = breakdown(&state);
                let i = source_index(b, id);
                assert_eq!(at.classification, b.classification(i, k), "{id} at {deg} deg");
                let share = b.force_share[i][k];
                assert!(
                    at.force_share == share || (at.force_share.is_nan() && share.is_nan()),
                    "{id} at {deg} deg: {} vs {share}",
                    at.force_share
                );
                assert_eq!(at.basis, ShareBasis::ActuatorForce);
                assert!(!at.is_stroke);
                if let Some(want) = want {
                    assert_eq!(at.classification, want, "{id} at {deg} deg (lifting hurts, lowering helps, sideways is neutral)");
                }
            }
        }
    }

    #[test]
    fn weight_at_pose_is_none_without_a_matching_breakdown_entry() {
        let mut state = swept_lift();
        at_deg(&mut state, 45.0);
        let pm = state.find_point_mass("rocker", "W1").unwrap().clone();
        assert!(weight_at_pose(&state, "rocker", &pm).is_some());

        let moved = PointMassJson { local_pos: [0.1, 0.0], ..pm.clone() };
        assert_eq!(weight_at_pose(&state, "rocker", &moved), None, "the sweep predates the move");
        let heavier = PointMassJson { mass: 60.0, ..pm.clone() };
        assert_eq!(weight_at_pose(&state, "rocker", &heavier), None, "the sweep predates the mass edit");
        assert_eq!(weight_at_pose(&state, "coupler", &pm), None, "not on that link");

        state.sweep_data = None;
        assert_eq!(weight_at_pose(&state, "rocker", &pm), None, "no sweep");
    }

    #[test]
    fn readout_lines_show_the_title_mass_and_current_share() {
        let mut state = swept_lift();
        let k = at_deg(&mut state, 45.0);
        let share = breakdown(&state).force_share[source_index(breakdown(&state), "W1")][k];
        assert!(share.is_finite(), "fixture: a defined share at 45 deg");
        assert_eq!(
            weight_readout_lines(&state, "rocker", "W1").unwrap(),
            vec!["W1".to_string(), "Mass: 50 kg".to_string(), format!("Force share: {} (hurting)", format_share(share, "N"))]
        );

        // A name edit rebuilds without a new sweep: the share still shows.
        assert!(state.set_point_mass_label("rocker", "W1", Some("Robot torso".to_string())));
        let lines = weight_readout_lines(&state, "rocker", "W1").unwrap();
        assert_eq!(lines[0], "Robot torso (W1)");
        assert!(lines[2].ends_with("(hurting)"), "{lines:?}");

        let k = reversal_sample(&state);
        let deg = state.sweep_data.as_ref().unwrap().angles_deg[k];
        at_deg(&mut state, deg);
        let lines = weight_readout_lines(&state, "rocker", "W1").unwrap();
        assert!(lines[2].starts_with("Force share: - ("), "near stroke reversal: {lines:?}");

        state.sweep_data = None;
        assert_eq!(weight_readout_lines(&state, "rocker", "W1").unwrap()[2], "Force share: -");
        assert_eq!(weight_readout_lines(&state, "rocker", "W9"), None);
    }

    #[test]
    fn readout_shows_driver_torque_shares_without_an_actuator() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_point_mass("coupler", 2.0, [0.03, 0.02]).unwrap();
        state.compute_sweep();
        let k = at_deg(&mut state, 60.0);
        let b = breakdown(&state);
        let i = source_index(b, "W1");
        let want = format!(
            "Torque share: {} ({})",
            format_share(b.force_share[i][k], "N\u{00b7}m"),
            classification_word(b.classification(i, k))
        );
        assert_eq!(weight_readout_lines(&state, "coupler", "W1").unwrap()[2], want);
    }

    #[test]
    fn format_share_signs_the_value_and_shows_a_dash_when_undefined() {
        assert_eq!(format_share(874.2, "N"), "+874 N");
        assert_eq!(format_share(-1234.0, "N"), "-1.2 kN");
        assert_eq!(format_share(0.35, "N\u{00b7}m"), "+0.35 N\u{00b7}m");
        assert_eq!(format_share(0.0, "N"), "+0.00 N");
        assert_eq!(format_share(f64::NAN, "N"), "-");
        assert_eq!(format_share(f64::NEG_INFINITY, "N"), "-");
    }
}
