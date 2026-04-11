//! Plot panel: sweep data visualization using egui_plot.
//!
//! Shows tabbed plots for coupler trace, body angles, and transmission angle.
//!
//! Clicking on any plot whose X-axis is driver angle scrubs the mechanism to
//! that angle.

use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text as PlotText, VLine};

use super::state::{AngleUnit, AppState, DisplayUnits};
use super::sweep::SweepData;

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
    OutputForce,
}


// Plot submodules grouped by analysis domain
mod actuator;
mod coupler;
mod dynamics;
mod mechanics;

/// Draw the plot panel with tabbed plots.
///
/// When the user clicks on a plot whose X-axis is driver angle, the mechanism
/// is scrubbed to that angle.
pub fn draw_plot_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        ui.label("No mechanism loaded.");
        return;
    }

    let Some(sweep) = &state.sweep_data else {
        ui.label("No sweep data available.");
        return;
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
            ).on_hover_text("Force in the linear actuator vs. driver angle. Red line = statics only (no inertia); cyan dashed = with inertia. Positive = tension (extending), negative = compression (retracting). If a rated force is entered, a green safe-zone band is shown.");
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
            ).on_hover_text("Mechanical power (Force \u{d7} Speed) delivered by the linear actuator in Watts vs. driver angle. Red = statics only; cyan dashed = with inertia. Peak power determines the motor/pump sizing requirement.");
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

    let current_driver_display = state.display_units.angle(state.driver_angle);
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
        PlotTab::OutputForce => {
            coupler::draw_output_force(ui, sweep, current_driver_display, &state.display_units, nm)
        }
    };

    // Scrub the mechanism to the clicked angle.
    if let Some(display_angle) = clicked_display_angle {
        let angle_rad = display_to_radians(display_angle, &state.display_units);
        state.solve_at_angle(angle_rad);
    }
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
    let is_stroke = sweep.sweep_mode.is_stroke();
    for (i, &toggle_val) in sweep.toggle_angles.iter().enumerate() {
        let toggle_display = if is_stroke { toggle_val * 1000.0 } else { units.angle(toggle_val.to_radians()) };
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
            let is_stroke = sweep.sweep_mode.is_stroke();
            let min_display = if is_stroke { min_val * 1000.0 } else { units.angle(min_val.to_radians()) };
            let max_display = if is_stroke { max_val * 1000.0 } else { units.angle(max_val.to_radians()) };
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

    let is_stroke = sweep.sweep_mode.is_stroke();
    let to_display = |x: f64| -> f64 {
        if is_stroke { x * 1000.0 } else { units.angle(x.to_radians()) }
    };

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

/// Filter extreme outliers from actuator force data using Tukey's fence method.
///
/// Near singularities the actuator force diverges (F = T*omega/dl_dt as dl_dt -> 0).
/// Even with a NaN threshold on dl_dt, values just above the threshold can be
/// orders of magnitude larger than the useful data, blowing out the Y-axis and
/// causing lag when zooming. This removes values beyond Q1 - 10*IQR .. Q3 + 10*IQR
/// (very conservative — only clips extreme spikes, preserves genuine peaks).
fn filter_actuator_outliers(pairs: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    if pairs.len() < 10 {
        return pairs;
    }
    let mut ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ys.len();
    let q1 = ys[n / 4];
    let q3 = ys[3 * n / 4];
    let iqr = (q3 - q1).max(1.0); // floor at 1 N to avoid zero IQR
    let lo = q1 - 10.0 * iqr;
    let hi = q3 + 10.0 * iqr;
    pairs.into_iter().filter(|(_, y)| *y >= lo && *y <= hi).collect()
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
