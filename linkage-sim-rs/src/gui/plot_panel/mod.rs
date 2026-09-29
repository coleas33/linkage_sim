//! Plot panel: sweep data visualization using egui_plot.
//!
//! Shows tabbed plots for coupler trace, body angles, and transmission angle.
//!
//! Clicking on any plot whose X-axis is driver angle scrubs the mechanism to
//! that angle.

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text as PlotText, VLine};

use super::state::{AngleUnit, AppState, DisplayUnits};
use super::sweep::{SweepData, SweepMode};

/// Selected plot tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlotTab {
    CouplerTrace,
    BodyAngles,
    TransmissionAngle,
    DriverTorque,
    InverseDynamics,
    Energy,
    MechanicalAdvantage,
    JointReactions,
    CouplerVelocity,
    CouplerAcceleration,
    ActuatorForce,
    ActuatorSpeed,
    ActuatorPower,
    WeightBreakdown,
    OutputForce,
}

/// Hover text of the Actuator Force tab. Sign convention of the
/// LinearActuator element (`forces::elements::evaluation`): positive =
/// extension.
const ACTUATOR_FORCE_TAB_TIP: &str = "Force in the linear actuator vs. driver angle. Red line = statics only (no inertia); cyan dashed = with inertia. Positive = extension (the actuator pushes its ends apart), negative = retraction (it pulls them together). Shaded bands mark where the load drives the actuator (braking). If a rated force is entered, a green safe-zone band is shown.";


// Plot submodules grouped by analysis domain
mod actuator;
mod coupler;
mod dynamics;
mod mechanics;
mod trajectory;
mod weights;

/// Draw the plot panel with tabbed plots.
///
/// When the user clicks on a plot whose X-axis is driver angle, the mechanism
/// is scrubbed to that angle.
pub fn draw_plot_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        ui.label("No mechanism loaded.");
        return;
    }

    if state.sweep_data.is_none() {
        ui.label("No sweep data available.");
        return;
    }

    // Trajectory mode bypasses the forward-sweep tabs entirely: the X-axis
    // is time, and the data lives in target_values/achieved_values/u_values
    // rather than angles_deg. Dispatch before the angles_deg guard since
    // trajectory sweeps populate angles_deg as empty. Render takes `&mut state`
    // so it can call `state.solve_for_trajectory_target` on click-to-scrub.
    if matches!(state.sweep_mode, SweepMode::Trajectory { .. }) {
        trajectory::render(state, ui);
        return;
    }

    // Forward-sweep tabs need an immutable borrow of sweep data alongside other
    // state reads; rebind here now that the trajectory dispatch is done.
    let Some(sweep) = &state.sweep_data else {
        return; // unreachable given the is_none() check above, but keeps borrow scoped
    };

    if sweep.angles_deg.is_empty() {
        ui.label("Sweep produced no data (solver failed at all angles).");
        return;
    }

    // Persistent tab selection via egui's memory.
    let tab_id = ui.id().with("plot_tab");
    let mut selected_tab = ui
        .memory(|mem| mem.data.get_temp::<PlotTab>(tab_id))
        .unwrap_or(PlotTab::CouplerTrace);

    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut selected_tab, PlotTab::CouplerTrace, "Coupler Trace")
            .on_hover_text("X-Y path traced by each coupler point over one full crank revolution. Use this to visualize the output motion path of the mechanism. Does not scrub on click.");
        ui.selectable_value(&mut selected_tab, PlotTab::BodyAngles, "Body Angles")
            .on_hover_text("Orientation of each moving body vs. driver angle. Shows how each link rotates as the crank turns. Click on the plot to scrub the mechanism to that angle.");

        // Only show transmission angle tab if data exists (4-bar only).
        let has_ta = sweep.transmission_angles.is_some();
        ui.add_enabled_ui(has_ta, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::TransmissionAngle,
                "Transmission Angle",
            ).on_hover_text("Angle between the coupler and output link at the driven joint. Ideal range is 40\u{b0}\u{2013}140\u{b0}; outside that range the mechanism transmits force poorly. Dashed lines mark the poor-transmission thresholds. Only available for 4-bar linkages.");
        });

        // Only show driver torque tab if data exists.
        let has_dt = sweep.driver_torques.is_some();
        let torque_tab_label = if sweep.sweep_mode.is_stroke() {
            "Actuator Force"
        } else {
            "Driver Torque"
        };
        let torque_tab_tip = if sweep.sweep_mode.is_stroke() {
            "Force required by the linear actuator at each stroke position, computed from static equilibrium (no inertia). Positive = extending, negative = retracting."
        } else {
            "Torque required at the driver joint to hold the mechanism in static equilibrium at each crank angle. Includes gravity and all applied forces but not inertia effects."
        };
        ui.add_enabled_ui(has_dt, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::DriverTorque,
                torque_tab_label,
            ).on_hover_text(torque_tab_tip);
        });

        // Only show inverse dynamics tab if data exists.
        let has_id = !sweep.inverse_dynamics_torques.is_empty();
        ui.add_enabled_ui(has_id, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::InverseDynamics,
                "Inv. Dynamics",
            ).on_hover_text("Driver torque including inertia effects (mass, acceleration, Coriolis). Shows the actual torque needed to drive the mechanism at the specified speed. Overlays the statics-only torque for comparison. If a motion profile is active, the profile torque is shown in green.");
        });

        // Only show energy tab if data exists.
        let has_energy = !sweep.kinetic_energy.is_empty();
        ui.add_enabled_ui(has_energy, |ui| {
            ui.selectable_value(&mut selected_tab, PlotTab::Energy, "Energy")
                .on_hover_text("Kinetic energy (from link velocities and inertias), gravitational potential energy, and their sum vs. driver angle. Useful for identifying energy storage opportunities (flywheels, counterbalances).");
        });

        // Only show mechanical advantage tab if data exists.
        let has_ma = !sweep.mechanical_advantage.is_empty()
            && sweep.mechanical_advantage.iter().any(|v| v.is_finite());
        ui.add_enabled_ui(has_ma, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::MechanicalAdvantage,
                "Mech. Advantage",
            ).on_hover_text("Ratio of output velocity to input velocity (velocity-based mechanical advantage). Values > 1 mean the output moves faster than the input; values < 1 mean force amplification. Dashed line at MA = 1 for reference. Singularities appear as spikes near toggle positions.");
        });

        // Only show joint reactions tab if data exists.
        let has_jr = !sweep.joint_reaction_magnitudes.is_empty();
        ui.add_enabled_ui(has_jr, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::JointReactions,
                "Joint Reactions",
            ).on_hover_text("Magnitude of the constraint reaction force at each joint vs. driver angle, from the statics solution. Use this to size bearings and pins \u{2014} the peak value determines the required load rating.");
        });

        // Only show coupler velocity tab if data exists.
        let has_cv = !sweep.coupler_velocities.is_empty();
        ui.add_enabled_ui(has_cv, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::CouplerVelocity,
                "Coupler Vel.",
            ).on_hover_text("Speed (magnitude of velocity vector) of each coupler point vs. driver angle. Depends on the driver speed setting. Useful for checking output velocity requirements and identifying dwell regions.");
        });

        // Only show coupler acceleration tab if data exists.
        let has_ca = !sweep.coupler_accelerations.is_empty();
        ui.add_enabled_ui(has_ca, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::CouplerAcceleration,
                "Coupler Accel.",
            ).on_hover_text("Acceleration magnitude of each coupler point vs. driver angle. High acceleration peaks indicate shock loads and inertia forces. Depends on the driver speed setting.");
        });

        // Only show actuator force tab when a LinearActuator is present.
        let has_af = sweep.actuator_forces.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_af, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorForce,
                "Actuator Force",
            ).on_hover_text(ACTUATOR_FORCE_TAB_TIP);
        });

        // Only show actuator speed tab when actuator speed data exists.
        let has_as = sweep.actuator_speeds.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_as, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorSpeed,
                "Actuator Speed",
            ).on_hover_text("Extension/retraction speed of the linear actuator in mm/s vs. driver angle. Use this to verify the actuator's speed rating is not exceeded at any point in the cycle.");
        });

        // Only show actuator power tab when actuator power data exists.
        let has_ap = sweep.actuator_power.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_ap, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::ActuatorPower,
                "Actuator Power",
            ).on_hover_text("Mechanical power (Force \u{d7} Speed) delivered by the linear actuator in Watts vs. driver angle. Red = statics only; cyan dashed = with inertia. Peak power determines the motor/pump sizing requirement. Negative power means the load drives the actuator (braking, shaded bands).");
        });

        // Only show the weight breakdown tab when the sweep has one (some
        // link mass or weight).
        let has_wb = sweep.weight_breakdown.is_some();
        ui.add_enabled_ui(has_wb, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::WeightBreakdown,
                "Weight Breakdown",
            ).on_hover_text("Each weight's share of the load the actuator (or, without an actuator, the driver) must carry vs. driver angle: one line per link self-weight and per placed weight, plus Other loads (springs, force zones, external loads, end stops) and the required Total. A line is green where that weight comes down and helps, red where the actuator lifts it, gray where it moves sideways. Force share = -m g\u{b7}v divided by the actuator speed dL/dt (by the driver rate without an actuator), left blank near stroke reversal; power share = -m g\u{b7}v, negative when the weight gives power back.");
        });

        // Only show output force tab when force zone data exists.
        let has_of = sweep.output_forces.as_ref().map_or(false, |v| !v.is_empty());
        ui.add_enabled_ui(has_of, |ui| {
            ui.selectable_value(
                &mut selected_tab,
                PlotTab::OutputForce,
                "Output Force",
            ).on_hover_text("Total force magnitude applied by all Force Zone elements at each driver angle. A Force Zone applies its full load whenever any part of the target body's geometry enters the zone (binary: full force or zero). Shows when and where in the cycle the mechanism encounters external loads.");
        });
    });

    ui.memory_mut(|mem| mem.data.insert_temp(tab_id, selected_tab));

    ui.horizontal(|ui| {
        ui.separator();
        ui.label(egui::RichText::new("Scroll to zoom, double-click to reset, click to scrub").small().weak())
            .on_hover_text("Scroll the mouse wheel to zoom in/out on the plot. Click and drag to pan. Double-click to reset the view. Single-click on any driver-angle plot to scrub the mechanism to that angle (the white vertical cursor follows your click).");

        // Explainer for the "(full)" suffix shown in legends when the
        // user has enabled a sweep range. The (full) curve is the
        // entire 0-360 sweep drawn faded/dashed behind the active
        // range, and the solid curve is the selected sub-range.
        if sweep.active_range.is_some() {
            ui.separator();
            ui.label(egui::RichText::new("ⓘ (full) = context").small().weak())
                .on_hover_text(
                    "When sweep range is enabled, each series is drawn twice:\n\
                     \u{2022} SOLID = the active sub-range you selected (e.g., J5)\n\
                     \u{2022} FADED DASHED = the full 0-360\u{b0} sweep for context (e.g., J5 (full))\n\n\
                     Both show the same data \u{2014} the dashed curve just shows what happens outside your active range. Disable 'Limit sweep range' in the sidebar to see only one curve per series."
                );
        }
    });

    // X-axis transform: in Angle mode, shift sweep angles by the
    // driver_display_offset so the plot X-axis matches the visible bar
    // direction (the Crank Angle slider). In Stroke mode, the X-axis
    // already tracks the actuator stroke (m, plotted as mm) and the
    // offset is zero by construction (R6).
    let offset_rad = state.driver_display_offset;
    let offset_deg = offset_rad.to_degrees();
    let display_sweep = sweep_in_display_frame(sweep, offset_deg);
    let sweep = &display_sweep;

    // Current-position cursor on plot X-axis. For linear drivers the
    // current X is the actuator stroke (m); plots multiply by 1000 to
    // display in mm. For revolute drivers it's the body-frame θ in
    // display units, with the offset added.
    let current_driver_display = if sweep.sweep_mode.is_stroke() {
        // Stroke value in metres; the plot consumers handle the
        // metres→mm conversion via their is_stroke branch.
        state.driver_stroke()
    } else {
        state.display_units.angle(state.driver_angle + offset_rad)
    };
    let nm = state.nathan_mode;

    // Each driver-angle-on-X-axis plot returns Some(x) when clicked, where x
    // is the X coordinate in display angle units. CouplerTrace (X vs Y) does
    // not participate in scrubbing.
    let clicked_display_angle: Option<f64> = match selected_tab {
        PlotTab::CouplerTrace => {
            coupler::draw_coupler_trace(ui, sweep, &state.display_units, nm);
            None
        }
        PlotTab::BodyAngles => {
            mechanics::draw_body_angles(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::TransmissionAngle => {
            mechanics::draw_transmission_angle(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::DriverTorque => {
            dynamics::draw_driver_torque(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::InverseDynamics => {
            dynamics::draw_inverse_dynamics(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::Energy => {
            dynamics::draw_energy(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::MechanicalAdvantage => {
            mechanics::draw_mechanical_advantage(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::JointReactions => {
            mechanics::draw_joint_reactions(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::CouplerVelocity => {
            coupler::draw_coupler_velocity(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::CouplerAcceleration => {
            coupler::draw_coupler_acceleration(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::ActuatorForce => {
            // Rated force input above the plot.
            ui.horizontal(|ui| {
                ui.label("Rated Force:").on_hover_text("Enter the actuator's maximum rated force from its datasheet. When set, a green band is drawn on the plot showing the safe operating region, and any portion of the cycle that exceeds the rating is highlighted in red.");
                ui.add(egui::DragValue::new(&mut state.actuator_rated_force)
                    .speed(10.0)
                    .range(0.0..=1e6)
                    .suffix(" N"))
                    .on_hover_text("Maximum continuous force rating of the actuator in Newtons. Set to 0 to hide the safety band.");
                if state.actuator_rated_force > 0.0 {
                    if ui.small_button("Clear").on_hover_text("Reset rated force to 0 and hide the safety band overlay").clicked() {
                        state.actuator_rated_force = 0.0;
                    }
                }
            });
            actuator::draw_actuator_force(ui, sweep, current_driver_display, &state.display_units, nm, state.actuator_rated_force)
        }
        PlotTab::ActuatorSpeed => {
            actuator::draw_actuator_speed(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::ActuatorPower => {
            actuator::draw_actuator_power(ui, sweep, current_driver_display, &state.display_units, nm)
        }
        PlotTab::WeightBreakdown => {
            ui.horizontal(|ui| {
                ui.label("Show:");
                ui.selectable_value(&mut state.weight_breakdown_show_power, false, "Force share")
                    .on_hover_text("Each weight's share of the required actuator force (N), or of the driver torque without an actuator. Speed-independent.");
                ui.selectable_value(&mut state.weight_breakdown_show_power, true, "Power share")
                    .on_hover_text("Each weight's share of the actuator power (W) at the driver's configured speed. Negative = the weight gives power back (helping).");
            });
            weights::draw_weight_breakdown(
                ui,
                sweep,
                current_driver_display,
                &state.display_units,
                nm,
                state.weight_breakdown_show_power,
            )
        }
        PlotTab::OutputForce => {
            coupler::draw_output_force(ui, sweep, current_driver_display, &state.display_units, nm)
        }
    };

    // Scrub the mechanism to the clicked X-coord. In Angle mode the
    // click is a display-frame angle; subtract the offset to recover
    // body-frame θ. In Stroke mode the click is in mm; convert to m
    // and call solve_at_stroke.
    if let Some(clicked_x) = clicked_display_angle {
        if display_sweep.sweep_mode.is_stroke() {
            state.solve_at_stroke(clicked_x * 1e-3);
        } else {
            let angle_rad = display_to_radians(clicked_x, &state.display_units)
                - offset_rad;
            state.solve_at_angle(angle_rad);
        }
    }
}

/// Return a copy of `SweepData` whose `angles_deg` and `toggle_angles`
/// are in display frame (body-frame θ + `offset_deg`). Only applied to
/// angle-mode sweeps — stroke-mode `angles_deg` carry metres and the
/// driver_display_offset (an angle quantity) is meaningless there. The
/// full clone is cheap enough for a per-frame panel redraw and keeps
/// plot-rendering code path simple.
fn sweep_in_display_frame(sweep: &SweepData, offset_deg: f64) -> SweepData {
    if sweep.sweep_mode.is_stroke() || offset_deg.abs() < 1e-12 {
        return sweep.clone();
    }
    let mut shifted = sweep.clone();
    for a in &mut shifted.angles_deg {
        *a += offset_deg;
    }
    for a in &mut shifted.toggle_angles {
        *a += offset_deg;
    }
    shifted
}

/// Convert a display-unit angle back to radians.
///
/// This is the inverse of `DisplayUnits::angle()`.
fn display_to_radians(display_angle: f64, units: &DisplayUnits) -> f64 {
    match units.angle {
        AngleUnit::Radians => display_angle,
        AngleUnit::Degrees => display_angle.to_radians(),
    }
}

/// Convert a sweep x value (`SweepData::angles_deg` entry, toggle angle or
/// range bound) to the plot's display x: metres to millimetres in stroke
/// mode, degrees to the configured display angle unit in angle mode.
///
/// The single place the plots map sweep x to screen x; clicks go back
/// through `display_to_radians` (angle) or `* 1e-3` (stroke).
fn sweep_x_to_display(x: f64, sweep: &SweepData, units: &DisplayUnits) -> f64 {
    if sweep.sweep_mode.is_stroke() {
        x * 1000.0
    } else {
        units.angle(x.to_radians())
    }
}

/// Return the x-axis label string for plots that sweep over the driver variable.
///
/// In angle mode this is `"Driver Angle (deg)"` or `"Driver Angle (rad)"`.
/// In stroke mode this is `"Actuator Stroke (mm)"`.
fn x_axis_label_for_sweep(sweep: &SweepData, units: &DisplayUnits) -> String {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Stroke (mm)".to_string()
    } else {
        let angle_label = match units.angle {
            AngleUnit::Degrees => "deg",
            AngleUnit::Radians => "rad",
        };
        format!("Driver Angle ({})", angle_label)
    }
}

/// Compute the display-space x-axis bounds (min, max) for an angle/stroke
/// sweep, or `None` if the sweep has no usable data.
///
/// The bounds go through `sweep_x_to_display`, the same conversion the
/// plots apply to their data points.
fn compute_default_x_bounds(sweep: &SweepData, units: &DisplayUnits) -> Option<(f64, f64)> {
    if sweep.angles_deg.is_empty() {
        return None;
    }
    let raw_min = sweep
        .angles_deg
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::INFINITY, f64::min);
    let raw_max = sweep
        .angles_deg
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !raw_min.is_finite() || !raw_max.is_finite() {
        return None;
    }
    if raw_max - raw_min < 1e-9 {
        return None;
    }
    Some((
        sweep_x_to_display(raw_min, sweep, units),
        sweep_x_to_display(raw_max, sweep, units),
    ))
}

/// Wrap a `Plot` with a fixed default x-axis bound matching the sweep's
/// data range, AND salt the plot id with the bounds key so egui_plot's
/// cached state gets invalidated when the data range changes.
///
/// Why both pieces:
/// - `default_x_bounds` disables auto-fit and pins the *reset* bounds —
///   stops the cursor `VLine` from expanding the visible x-range during
///   playback. But `default_x_bounds` does NOT override `egui::Memory`
///   that's already cached for the plot id (pan/zoom state, previously-
///   computed bounds from before the fix shipped, etc.) — egui keeps
///   re-using the cached state until the id changes.
/// - The id salt (rounded bounds tuple) makes the id change whenever the
///   data range changes. egui sees the new id as a fresh plot, has no
///   cached state for it, and uses our `default_x_bounds` directly.
///
/// User pan/zoom still works *within a single sweep recompute*; it
/// only resets when the underlying data range changes.
///
/// `plot_id_stem` should match the string passed to `Plot::new(...)`
/// (this helper effectively replaces that id with `(stem, bounds_key)`).
fn with_default_x_bounds<'a>(
    plot: Plot<'a>,
    plot_id_stem: &str,
    sweep: &SweepData,
    units: &DisplayUnits,
) -> Plot<'a> {
    let bounds = compute_default_x_bounds(sweep, units);
    let key: (i64, i64) = bounds
        .map(|(lo, hi)| {
            ((lo * 1000.0).round() as i64, (hi * 1000.0).round() as i64)
        })
        .unwrap_or((0, 0));
    let plot = plot.id(egui::Id::new((plot_id_stem, key)));
    if let Some((lo, hi)) = bounds {
        plot.default_x_bounds(lo, hi)
    } else {
        plot
    }
}

/// Return the y-axis label for the driver effort plot.
///
/// In angle mode (revolute driver) this is `"Driver Torque (N*m)"`.
/// In stroke mode (linear driver) this is `"Actuator Force (N)"`.
fn driver_effort_y_label(sweep: &SweepData) -> &'static str {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Force (N)"
    } else {
        "Driver Torque (N\u{00b7}m)"
    }
}

/// Return the series name for the driver effort in the plot legend.
///
/// In angle mode: `"Driver Torque"`. In stroke mode: `"Actuator Force"`.
fn driver_effort_series_name(sweep: &SweepData) -> &'static str {
    if sweep.sweep_mode.is_stroke() {
        "Actuator Force"
    } else {
        "Driver Torque"
    }
}

/// Detect a click on the plot and return the X coordinate in plot space.
///
/// Call this inside a `plot_ui` closure. Returns `Some(x)` when the user
/// clicks (not drags) on the plot area.
fn detect_plot_click(plot_ui: &egui_plot::PlotUi) -> Option<f64> {
    if plot_ui.response().clicked() {
        if let Some(coord) = plot_ui.pointer_coordinate() {
            return Some(coord.x);
        }
    }
    None
}

/// Plot coupler point traces: x vs y, converted to display length units.
///
/// When `sweep.active_range` is set, the full curve is drawn faded/dashed for
/// context and the active sub-range is overdrawn solid.

fn draw_toggle_markers(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
) {
    for (i, &toggle_val) in sweep.toggle_angles.iter().enumerate() {
        let toggle_display = sweep_x_to_display(toggle_val, sweep, units);
        plot_ui.vline(
            VLine::new(format!("toggle_{}", i), toggle_display)
                .color(egui::Color32::from_rgba_premultiplied(255, 60, 60, 100))
                .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                .width(2.5),
        );
    }
}

/// Draw vertical boundary markers at the sweep range limits.
///
/// When `active_range` is set, draws faint white dashed VLines at the
/// min and max angles (or stroke values) of the active range.
fn draw_range_boundary_markers(
    plot_ui: &mut egui_plot::PlotUi,
    sweep: &SweepData,
    units: &DisplayUnits,
) {
    if let Some((start, end)) = sweep.active_range {
        if let (Some(&min_val), Some(&max_val)) =
            (sweep.angles_deg.get(start), sweep.angles_deg.get(end))
        {
            let min_display = sweep_x_to_display(min_val, sweep, units);
            let max_display = sweep_x_to_display(max_val, sweep, units);
            let boundary_color =
                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 80);
            plot_ui.vline(
                VLine::new("range_min", min_display)
                    .color(boundary_color)
                    .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                    .width(1.0),
            );
            plot_ui.vline(
                VLine::new("range_max", max_display)
                    .color(boundary_color)
                    .style(egui_plot::LineStyle::Dashed { length: 3.0 })
                    .width(1.0),
            );
        }
    }
}

/// Create a faded version of a color for out-of-range plot data.
fn faded_color(color: egui::Color32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 60)
}

/// Helper for angle-based plots: draws a single data series with faded/solid
/// treatment when an active range is set.
///
/// - `name`: legend label for the series
/// - `color`: full-opacity color for the active range
/// - `width`: line width for the active (solid) portion
/// - `x_deg_and_y`: iterator of `(angle_deg, y_value)` pairs for the full sweep
/// - `convert_x`: function to convert angle in degrees to display x-value
/// - `sweep`: used to read `active_range` and `angles_deg`
///
/// When `active_range` is `None`, draws normally. When set, draws the full
/// curve faded/dashed and overdraws the active slice solid.
///
/// In angle mode, x values are in degrees and are converted to the display
/// angle unit. In stroke mode, x values are in meters and are converted to mm.
fn draw_angle_series_with_range(
    plot_ui: &mut egui_plot::PlotUi,
    name: &str,
    color: egui::Color32,
    width: f32,
    x_deg_and_y: &[(f64, f64)],
    sweep: &SweepData,
    units: &DisplayUnits,
    nathan_mode: bool,
) {
    use crate::gui::canvas::to_grayscale;
    let color = if nathan_mode { to_grayscale(color) } else { color };
    if x_deg_and_y.is_empty() {
        return;
    }

    let to_display = |x: f64| sweep_x_to_display(x, sweep, units);

    if let Some((start, end)) = sweep.active_range {
        // Full curve -- faded/dashed for context.
        let full: PlotPoints = x_deg_and_y
            .iter()
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(format!("{} (full)", name), full)
                .color(faded_color(color))
                .style(egui_plot::LineStyle::Dashed { length: 4.0 })
                .width(width * 0.5),
        );

        // Active range -- solid.
        // Use the degree boundaries to select the active subset.
        let min_deg = sweep.angles_deg.get(start).copied().unwrap_or(0.0);
        let max_deg = sweep.angles_deg.get(end).copied().unwrap_or(360.0);
        let active: PlotPoints = x_deg_and_y
            .iter()
            .filter(|&&(x_deg, _)| x_deg >= min_deg && x_deg <= max_deg)
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(name, active)
                .color(color)
                .width(width),
        );
    } else {
        // No range limit -- draw normally.
        let pts: PlotPoints = x_deg_and_y
            .iter()
            .map(|&(x_deg, y)| [to_display(x_deg), y])
            .collect();
        plot_ui.line(
            Line::new(name, pts)
                .color(color)
                .width(width),
        );
    }
}

/// Pair sweep x-values with a y-series, keeping every finite sample.
///
/// This is the only filtering the actuator force plot applies (BL-010): the
/// former Tukey-fence outlier drop silently removed finite samples the canvas
/// label still displayed, so plot and label disagreed at the same pose.
fn finite_series(xs: &[f64], ys: &[f64]) -> Vec<(f64, f64)> {
    xs.iter()
        .zip(ys.iter())
        .filter(|&(_, &y)| y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect()
}

/// A palette of distinguishable colors for plot series.
///
/// When `nathan_mode` is true, all colors are converted to grayscale.
fn series_colors(nathan_mode: bool) -> Vec<egui::Color32> {
    use crate::gui::canvas::to_grayscale;
    let raw = vec![
        egui::Color32::from_rgb(100, 200, 255), // light blue
        egui::Color32::from_rgb(255, 150, 80),  // orange
        egui::Color32::from_rgb(120, 220, 120), // green
        egui::Color32::from_rgb(255, 100, 100), // red
        egui::Color32::from_rgb(200, 150, 255), // purple
        egui::Color32::from_rgb(255, 220, 100), // yellow
        egui::Color32::from_rgb(150, 200, 200), // teal
        egui::Color32::from_rgb(255, 150, 200), // pink
    ];
    if nathan_mode {
        raw.into_iter().map(to_grayscale).collect()
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::state::{AngleUnit, DisplayUnits, LengthUnit};
    use crate::gui::sweep::{empty_trajectory_sweep_data, SweepMode};
    use crate::gui::test_support::central_panel_frame;

    fn deg_units() -> DisplayUnits {
        DisplayUnits {
            length: LengthUnit::Millimeters,
            angle: AngleUnit::Degrees,
        }
    }

    #[test]
    fn empty_sweep_returns_no_bounds() {
        // No data → helper signals "leave it on auto-fit" via None.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        sweep.angles_deg.clear();
        let bounds = compute_default_x_bounds(&sweep, &deg_units());
        assert!(bounds.is_none());
    }

    #[test]
    fn angle_mode_bounds_match_data_range_in_display_units() {
        // Sweep covering 66.5°–180° (body-frame degrees, per the convention
        // documented on `SweepData.angles_deg` for angle mode): helper
        // should report bounds in display-frame degrees that match the
        // data range exactly. This is the regression for the user's
        // "lines disappear before 66.5°" report — locking these bounds
        // hides the empty pre-66.5 region instead of letting auto-fit
        // expose it via the cursor.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        // angles_deg carries DEGREES in angle mode (despite the field
        // name conflating angle/stroke semantics); helper converts to
        // display unit via .to_radians() → units.angle(rad).
        sweep.angles_deg = vec![66.5, 120.0, 180.0];

        let (lo, hi) = compute_default_x_bounds(&sweep, &deg_units())
            .expect("nontrivial range yields bounds");
        assert!((lo - 66.5).abs() < 1e-6, "lo = {}", lo);
        assert!((hi - 180.0).abs() < 1e-6, "hi = {}", hi);
    }

    #[test]
    fn stroke_mode_converts_metres_to_millimetres() {
        // Stroke-mode angles_deg carries metres; bounds come back in mm.
        let mode = SweepMode::Stroke;
        let mut sweep = empty_trajectory_sweep_data(mode);
        sweep.angles_deg = vec![0.05, 0.10, 0.15]; // 50–150 mm

        let (lo, hi) = compute_default_x_bounds(&sweep, &deg_units())
            .expect("stroke range yields bounds");
        assert!((lo - 50.0).abs() < 1e-9);
        assert!((hi - 150.0).abs() < 1e-9);
    }

    #[test]
    fn degenerate_range_returns_no_bounds() {
        // All-equal x values would panic egui_plot's debug assert
        // (min < max). Helper returns None instead, falling back to auto-fit.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        sweep.angles_deg = vec![45.0; 5];
        assert!(compute_default_x_bounds(&sweep, &deg_units()).is_none());
    }

    #[test]
    fn nan_only_data_returns_no_bounds() {
        // A sweep where every angle is NaN (e.g. catastrophic solver
        // failure) shouldn't pin bounds at NaN. Returns None.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        sweep.angles_deg = vec![f64::NAN; 4];
        assert!(compute_default_x_bounds(&sweep, &deg_units()).is_none());
    }

    #[test]
    fn radians_mode_converts_degrees_to_radians_for_display() {
        // Same data, AngleUnit::Radians: bounds come back in radians.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        sweep.angles_deg = vec![0.0, 90.0, 180.0]; // body-frame degrees
        let units = DisplayUnits {
            length: LengthUnit::Millimeters,
            angle: AngleUnit::Radians,
        };
        let (lo, hi) = compute_default_x_bounds(&sweep, &units)
            .expect("nontrivial range yields bounds");
        assert!((lo - 0.0).abs() < 1e-9);
        assert!((hi - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn nan_padding_is_skipped() {
        // Mixed finite + NaN: bounds reflect the finite extent only.
        let mut sweep = empty_trajectory_sweep_data(SweepMode::Angle);
        sweep.angles_deg = vec![f64::NAN, 66.5, f64::NAN, 180.0, f64::NAN];
        let (lo, hi) = compute_default_x_bounds(&sweep, &deg_units())
            .expect("finite samples yield bounds");
        assert!((lo - 66.5).abs() < 1e-6);
        assert!((hi - 180.0).abs() < 1e-6);
    }

    /// The one sweep-x conversion every plot uses (data points, braking
    /// bands, toggle and range markers, default x bounds): metres to
    /// millimetres in stroke mode whatever the angle unit, degrees to the
    /// display angle unit in angle mode.
    #[test]
    fn sweep_x_to_display_converts_per_sweep_mode_and_angle_unit() {
        let angle = empty_trajectory_sweep_data(SweepMode::Angle);
        let stroke = empty_trajectory_sweep_data(SweepMode::Stroke);
        let rad_units = DisplayUnits { length: LengthUnit::Millimeters, angle: AngleUnit::Radians };
        assert!((sweep_x_to_display(90.0, &angle, &deg_units()) - 90.0).abs() < 1e-12);
        assert!((sweep_x_to_display(-45.0, &angle, &deg_units()) + 45.0).abs() < 1e-12);
        assert!((sweep_x_to_display(90.0, &angle, &rad_units) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(sweep_x_to_display(0.125, &stroke, &deg_units()), 125.0);
        assert_eq!(sweep_x_to_display(0.125, &stroke, &rad_units), 125.0);
        assert!(sweep_x_to_display(f64::NAN, &angle, &deg_units()).is_nan());
        assert!(sweep_x_to_display(f64::NAN, &stroke, &deg_units()).is_nan());
    }

    /// Flat curve with two near-singular spikes: the removed Tukey fence
    /// (Q1 - 10*IQR .. Q3 + 10*IQR, IQR floored at 1 N) would have dropped
    /// both spikes because the fence is only ~20 N wide on a flat curve.
    fn spiky_force_series() -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..24).map(|i| i as f64 * 15.0).collect();
        let mut ys = vec![-2222.8; 24];
        ys[5] = 5.0e4;
        ys[17] = -8.0e4;
        (xs, ys)
    }

    #[test]
    fn outlier_fence_removed_keeps_every_finite_sample() {
        // BL-010 mechanism (2): plot must draw exactly the samples the
        // canvas label reads — no silent outlier drop.
        let (xs, ys) = spiky_force_series();
        let series = finite_series(&xs, &ys);
        assert_eq!(series.len(), ys.len());
        for (i, &(x, y)) in series.iter().enumerate() {
            assert_eq!(x.to_bits(), xs[i].to_bits());
            assert_eq!(y.to_bits(), ys[i].to_bits());
        }
    }

    #[test]
    fn outlier_fence_removed_still_skips_non_finite_samples() {
        // NaN/inf (solver failures, dl_dt -> 0 guard) are the only samples
        // omitted; finite spikes stay.
        let (xs, mut ys) = spiky_force_series();
        ys[2] = f64::NAN;
        ys[9] = f64::INFINITY;
        ys[20] = f64::NEG_INFINITY;
        let series = finite_series(&xs, &ys);
        let finite_count = ys.iter().filter(|y| y.is_finite()).count();
        assert_eq!(series.len(), finite_count);
        assert_eq!(finite_count, ys.len() - 3);
        assert!(series.iter().any(|&(_, y)| y == 5.0e4));
        assert!(series.iter().any(|&(_, y)| y == -8.0e4));
        assert!(series.iter().all(|&(x, _)| x != 30.0 && x != 135.0 && x != 300.0));
    }

    #[test]
    fn outlier_fence_removed_short_series_unchanged() {
        // Fewer than 10 samples was the old filter's bypass; behaviour must
        // be identical either side of that threshold.
        let xs = [0.0, 90.0, 180.0];
        let ys = [1.0, 1.0e6, -1.0e6];
        assert_eq!(finite_series(&xs, &ys), vec![(0.0, 1.0), (90.0, 1.0e6), (180.0, -1.0e6)]);
        assert!(finite_series(&[], &[]).is_empty());
    }

    /// The code's convention is positive = extension (the actuator pushes
    /// its ends apart); the tab tip used to say "positive = tension".
    #[test]
    fn actuator_force_tab_tip_states_the_extension_sign_convention() {
        assert!(ACTUATOR_FORCE_TAB_TIP.contains("Positive = extension"), "{ACTUATOR_FORCE_TAB_TIP}");
        assert!(!ACTUATOR_FORCE_TAB_TIP.contains("Positive = tension"), "{ACTUATOR_FORCE_TAB_TIP}");
        assert!(!ACTUATOR_FORCE_TAB_TIP.contains("compression"), "{ACTUATOR_FORCE_TAB_TIP}");
    }

    /// Run one idle frame of the plot panel with `tab` selected (the panel
    /// keeps its tab in egui memory under `ui.id().with("plot_tab")`).
    fn plot_panel_frame(state: &mut AppState, tab: PlotTab) {
        let _ = central_panel_frame(&egui::Context::default(), Vec::new(), |ui| {
            let tab_id = ui.id().with("plot_tab");
            ui.memory_mut(|mem| mem.data.insert_temp(tab_id, tab));
            draw_plot_panel(ui, state);
        });
    }

    /// The Weight Breakdown tab (force and power view) and the actuator
    /// tabs with braking bands and a rated force render headlessly, and an
    /// idle frame changes neither the view toggle nor the pose.
    #[test]
    fn weight_breakdown_and_braking_band_tabs_render_idle_frames() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        state.add_point_mass("rocker", 50.0, [0.0, 0.0]).expect("weight added");
        state.compute_sweep();
        let breakdown = state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().expect("breakdown");
        assert!(breakdown.braking.iter().any(|&b| b), "the fixture has braking bands to draw");
        state.actuator_rated_force = 500.0;
        for show_power in [false, true] {
            state.weight_breakdown_show_power = show_power;
            let angle = state.driver_angle;
            for tab in [PlotTab::WeightBreakdown, PlotTab::ActuatorForce, PlotTab::ActuatorPower] {
                plot_panel_frame(&mut state, tab);
                assert_eq!(state.weight_breakdown_show_power, show_power, "{tab:?} idle frame");
                assert_eq!(state.driver_angle, angle, "{tab:?} idle frame");
            }
        }
    }

    /// Without an actuator the tab shows driver-torque shares; with no link
    /// mass and no weight the sweep has no breakdown, and the tab and the
    /// band-less actuator tab still render.
    #[test]
    fn weight_breakdown_tab_renders_driver_basis_and_missing_breakdown() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;
        use crate::gui::sweep::ShareBasis;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let breakdown = state.sweep_data.as_ref().unwrap().weight_breakdown.as_ref().expect("breakdown");
        assert_eq!(breakdown.basis, ShareBasis::DriverTorque);
        for show_power in [false, true] {
            state.weight_breakdown_show_power = show_power;
            plot_panel_frame(&mut state, PlotTab::WeightBreakdown);
        }

        for body in state.blueprint.as_mut().unwrap().bodies.values_mut() {
            body.mass = 0.0;
        }
        state.rebuild();
        state.compute_sweep();
        assert!(state.sweep_data.as_ref().unwrap().weight_breakdown.is_none());
        plot_panel_frame(&mut state, PlotTab::WeightBreakdown);
        plot_panel_frame(&mut state, PlotTab::ActuatorForce);
    }

    #[test]
    fn end_to_end_sweep_range_with_offset_pins_to_user_input_range() {
        // Reproduce the user-reported "plot shows 65°–120° when sweep range
        // is 22°–75°" symptom end-to-end. Wires `compute_sweep` →
        // `sweep_in_display_frame` → `compute_default_x_bounds` against a
        // 4-bar with a non-zero `driver_display_offset` and asserts the
        // resulting bounds equal the user's display-frame input range.
        //
        // Expected pipeline:
        //   1. blueprint_ops::compute_sweep subtracts offset → body-frame
        //      range passed to compute_sweep_data.
        //   2. compute_sweep_data writes body-frame degrees to angles_deg.
        //   3. sweep_in_display_frame adds offset back → display-frame.
        //   4. compute_default_x_bounds reads display-frame data → bounds
        //      should equal (22, 75).
        //
        // If this test fails, the bug is somewhere in steps 1–4.
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Rotate the crank's far attachment point by +43° in its local
        // frame so the body's A→B direction sits at +43° from +X. This
        // gives `driver_display_offset` ≈ 43° (matches the user's symptom).
        let crank_len = 2.0_f64;
        let new_bx = crank_len * 43f64.to_radians().cos();
        let new_by = crank_len * 43f64.to_radians().sin();
        if let Some(bp) = state.blueprint.as_mut() {
            if let Some(crank) = bp.bodies.get_mut("crank") {
                crank.attachment_points
                    .insert("B".to_string(), [new_bx, new_by]);
            }
        }
        state.rebuild();
        assert!(
            (state.driver_display_offset - 43f64.to_radians()).abs() < 1e-3,
            "test fixture should have ~43° offset, got {} rad",
            state.driver_display_offset,
        );

        // User configures display-frame sweep range 22°–75°.
        state.sweep_angle_min_deg = 22.0;
        state.sweep_angle_max_deg = 75.0;
        state.sweep_range_enabled = true;
        state.compute_sweep();

        let sweep = state
            .sweep_data
            .as_ref()
            .expect("compute_sweep should populate sweep_data");

        // Capture raw and shifted ranges for debug output on failure.
        let raw_min = sweep
            .angles_deg
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let raw_max = sweep
            .angles_deg
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        // Apply the same shift the plot panel does.
        let offset_deg = state.driver_display_offset.to_degrees();
        let display_sweep = sweep_in_display_frame(sweep, offset_deg);

        let shifted_min = display_sweep
            .angles_deg
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::INFINITY, f64::min);
        let shifted_max = display_sweep
            .angles_deg
            .iter()
            .copied()
            .filter(|x| x.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);

        let (lo, hi) = compute_default_x_bounds(&display_sweep, &deg_units())
            .expect("nontrivial sweep should yield bounds");

        assert!(
            (lo - 22.0).abs() < 1.0 && (hi - 75.0).abs() < 1.0,
            "expected display-frame bounds (22, 75), got ({}, {})\n  \
             raw angles_deg range = ({}, {})\n  \
             shifted angles_deg range = ({}, {})\n  \
             driver_display_offset = {} rad ({:.3}°)",
            lo, hi, raw_min, raw_max, shifted_min, shifted_max,
            state.driver_display_offset, offset_deg,
        );
    }
}
