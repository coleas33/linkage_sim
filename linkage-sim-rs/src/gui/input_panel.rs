//! Angle slider, playback controls, load case selector, and solver status display.

use eframe::egui;
use super::state::{AppState, DriverKind, MotionProfile};
use crate::io::DriverJson;


/// Draw the input panel with animation controls and load case management.
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();

    // Revolute / no-driver: show the Crank Angle slider + sweep range.
    // Linear: show a parallel Actuator Stroke section instead.
    if state.driver_kind == DriverKind::Linear {
        draw_actuator_stroke_section(ui, state);
    } else {
    // ── Crank Angle ────────────────────────────────────────────────
    let accent = state.nc(egui::Color32::from_rgb(80, 160, 255));
    egui::CollapsingHeader::new(
        egui::RichText::new("Crank Angle").color(accent),
    )
        .id_salt("crank_section")
        .default_open(true)
        .show(ui, |ui| {
            {
                // ── Label: what is the crank angle? ────────────────
                // The slider value is the driver constraint's f(t) =
                // θⱼ − θᵢ (driven body vs. partner). For a ground-
                // mounted driver, partner = ground and zero crank
                // angle means the driver body's local +X is aligned
                // with world +X. Showing this explicitly so users
                // aren't left guessing which body the slider drives.
                if let Some(mech) = state.mechanism.as_ref() {
                    if let Some((partner, driver)) = mech.driver_body_pair() {
                        let reference_label = if partner == crate::core::state::GROUND_ID {
                            "world".to_string()
                        } else {
                            partner.to_string()
                        };
                        ui.small(
                            egui::RichText::new(format!(
                                "Driver: {} relative to {} (0\u{00B0} = {} +X)",
                                driver, partner, reference_label
                            ))
                            .color(egui::Color32::from_rgb(150, 150, 160)),
                        );
                    }
                }

                // ── Revolute driver: angle slider in degrees ────────
                // Slider operates in DISPLAY frame. `sweep_angle_min/max_deg`
                // are also stored in display frame; the sweep computation
                // and animation bounds subtract the driver_display_offset
                // when converting to the solver's body-frame θ. The
                // slider visible range is therefore always [0, 360] for
                // full rotation or the user's chosen display limits.
                let offset_rad = state.driver_display_offset;
                let offset_deg = offset_rad.to_degrees();
                let (slider_min, slider_max) = if state.sweep_range_enabled {
                    let a = state.sweep_angle_min_deg;
                    let b = state.sweep_angle_max_deg;
                    if a <= b { (a, b) } else { (b, a) }
                } else {
                    (0.0, 360.0)
                };

                // Current driver in display frame; normalise into the
                // slider range so the handle lands on the active range
                // rather than an aliased position.
                let body_deg = state.driver_angle.to_degrees();
                let mut display_angle_deg = body_deg + offset_deg;
                while display_angle_deg < slider_min { display_angle_deg += 360.0; }
                while display_angle_deg >= slider_min + 360.0 { display_angle_deg -= 360.0; }
                display_angle_deg = display_angle_deg.clamp(slider_min, slider_max);
                let prev_display_angle = display_angle_deg;
                let response = ui.add(
                    egui::Slider::new(&mut display_angle_deg, slider_min..=slider_max)
                        .suffix("\u{00B0}")
                        .step_by(0.5),
                ).on_hover_text("Drag to set the driver crank angle in degrees. 0° = visible bar along world +X (horizontal).");
                if response.dragged() {
                    if state.playing {
                        state.playing = false;
                        state.animation_direction = 1.0;
                    }
                    if let Some(sim) = &mut state.simulation {
                        sim.playing = false;
                    }
                }
                if (display_angle_deg - prev_display_angle).abs() > 1e-6 {
                    let body_angle_rad = display_angle_deg.to_radians() - offset_rad;
                    state.solve_at_angle(body_angle_rad);
                }
            }

            ui.horizontal(|ui| {
                let mode_label = if state.loop_mode { "Loop" } else { "Once" };
                if ui.button(mode_label)
                    .on_hover_text("Toggle between continuous loop and single-pass animation")
                    .clicked()
                {
                    state.loop_mode = !state.loop_mode;
                    state.animation_direction = 1.0;
                }
                if ui.button("Flip Branch")
                    .on_hover_text(
                        "Debug: reflect the mechanism across its ground line and re-solve \
                         to land on the alternate assembly branch. Useful when adjusting the \
                         sweep range causes the solver to jump configurations.",
                    )
                    .clicked()
                {
                    state.flip_assembly_branch();
                }
            });

            // ── Sweep Range ────────────────────────────────────────
            ui.separator();
            let prev_enabled = state.sweep_range_enabled;
            ui.checkbox(&mut state.sweep_range_enabled, "Limit Sweep Range")
                .on_hover_text("Restrict the crank sweep to a custom angular range instead of full 360\u{00B0}");
            if state.sweep_range_enabled != prev_enabled {
                state.mark_sweep_dirty();
            }
            if state.sweep_range_enabled {
                ui.horizontal(|ui| {
                    // sweep_angle_min_deg / max_deg are stored in DISPLAY
                    // frame (matches the Crank Angle slider). The sweep
                    // solver and animation bounds subtract the driver
                    // display offset when converting to body-frame θ.
                    ui.label("Min\u{00B0}:");
                    let min_resp = ui.add(egui::DragValue::new(&mut state.sweep_angle_min_deg)
                        .speed(0.5)
                        .range(0.0..=720.0)
                        .suffix("\u{00B0}"))
                        .on_hover_text("Start angle of the sweep range in display degrees (matches the Crank Angle slider). Values above 360\u{b0} are allowed so you can sweep across the 0/360 seam (e.g. 200\u{b0} to 365\u{b0}).");
                    ui.label("Max\u{00B0}:");
                    let max_resp = ui.add(egui::DragValue::new(&mut state.sweep_angle_max_deg)
                        .speed(0.5)
                        .range(0.0..=720.0)
                        .suffix("\u{00B0}"))
                        .on_hover_text("End angle of the sweep range in display degrees. Must be \u{2265} min. Set max above 360\u{b0} to sweep across the 0/360 seam.");

                    // Defer the sweep recompute until the user finishes
                    // editing (drag ended, field lost focus, or Enter
                    // pressed). This prevents transient inverted ranges
                    // during mid-type.
                    let min_done = min_resp.drag_stopped()
                        || min_resp.lost_focus();
                    let max_done = max_resp.drag_stopped()
                        || max_resp.lost_focus();
                    if min_done || max_done {
                        if state.sweep_angle_max_deg < state.sweep_angle_min_deg {
                            state.sweep_angle_max_deg = state.sweep_angle_min_deg;
                        }
                        state.mark_sweep_dirty();
                    }
                });
            }
        });
    } // end Crank Angle / Stroke dispatch

    // ── Gravity ──────────────────────────────────────────────────────
    let gravity_color = state.nc(egui::Color32::from_rgb(200, 160, 80));
    egui::CollapsingHeader::new(
        egui::RichText::new("Gravity").color(gravity_color),
    )
        .id_salt("gravity_section")
        .default_open(true)
        .show(ui, |ui| {
            let prev_g = state.gravity_magnitude;
            ui.add(
                egui::Slider::new(&mut state.gravity_magnitude, 0.0..=981.0)
                    .suffix(" m/s\u{00b2}")
                    .step_by(0.01)
                    .clamping(egui::SliderClamping::Always),
            ).on_hover_text("Gravitational acceleration magnitude in m/s\u{00b2} (9.81 = 1g)");
            ui.label(format!("({:.2} g)", state.gravity_magnitude / 9.81));
            if (state.gravity_magnitude - prev_g).abs() > 1e-9 {
                state.mark_sweep_dirty();
            }
        });

    // ── Mounting Angle ────────────────────────────────────────────────
    egui::CollapsingHeader::new(
        egui::RichText::new("Mounting Angle").color(gravity_color),
    )
        .id_salt("mounting_angle_section")
        .default_open(false)
        .show(ui, |ui| {
            let mut angle_deg = state.mounting_angle.to_degrees();
            let prev_deg = angle_deg;
            ui.add(
                egui::Slider::new(&mut angle_deg, -180.0..=180.0)
                    .suffix("\u{00B0}")
                    .step_by(0.5),
            ).on_hover_text("Rotate the mechanism mounting orientation in degrees");
            if (angle_deg - prev_deg).abs() > 1e-6 {
                state.mounting_angle = angle_deg.to_radians();
                state.mark_sweep_dirty();
                state.rebuild();
            }
        });

    // ── Driver ───────────────────────────────────────────────────────
    let driver_color = state.nc(egui::Color32::from_rgb(100, 220, 140));
    egui::CollapsingHeader::new(
        egui::RichText::new("Driver").color(driver_color),
    )
        .id_salt("driver_section")
        .default_open(true)
        .show(ui, |ui| {
            draw_load_case_selector(ui, state);
            if let Some(joint_id) = &state.driver_joint_id {
                let label_response = ui.label(format!("{} (right-click to change)", joint_id));
                if label_response.hovered() {
                    state.highlight_joint = Some(joint_id.clone());
                } else if state.highlight_joint.as_deref() == Some(joint_id.as_str()) {
                    state.highlight_joint = None;
                }
            } else {
                draw_no_driver_picker(ui, state);
            }
            draw_driver_type_selector(ui, state);
            draw_motion_profile_selector(ui, state);
        });

    // ── Simulation ───────────────────────────────────────────────────
    let sim_color = state.nc(egui::Color32::from_rgb(100, 180, 255));
    egui::CollapsingHeader::new(
        egui::RichText::new("\u{23F1} Simulation").color(sim_color),
    )
        .id_salt("simulation_section")
        .default_open(false)
        .show(ui, |ui| {
            draw_simulation_controls(ui, state);
        });
}

/// Actuator Stroke slider + stroke sweep range for linear drivers.
///
/// Mirror of the Crank Angle section, but the slider is in millimetres
/// and operates on `driver_stroke` (m) directly. No display offset is
/// applied (linear drivers don't have an angle-frame offset).
fn draw_actuator_stroke_section(ui: &mut egui::Ui, state: &mut AppState) {
    let accent = state.nc(egui::Color32::from_rgb(80, 160, 255));
    egui::CollapsingHeader::new(
        egui::RichText::new("Actuator Stroke").color(accent),
    )
        .id_salt("stroke_section")
        .default_open(true)
        .show(ui, |ui| {
            ui.small(
                egui::RichText::new(
                    "Linear driver: slider sets the actuator stroke in mm.",
                )
                .color(egui::Color32::from_rgb(150, 150, 160)),
            );

            // Slider bounds in mm. With sweep range enabled use the
            // user's [min, max]; otherwise fall back to the actuator's
            // stroke_min/stroke_max (cached on rebuild) or a window
            // around length_0 if no actuator hint was found.
            let (slider_min_mm, slider_max_mm) = if state.sweep_range_enabled {
                let a = state.sweep_angle_min_deg;
                let b = state.sweep_angle_max_deg;
                if a <= b { (a, b) } else { (b, a) }
            } else if state.sweep_stroke_max > state.sweep_stroke_min {
                (state.sweep_stroke_min * 1e3, state.sweep_stroke_max * 1e3)
            } else {
                let l0_mm = state.driver_theta_0 * 1e3;
                (l0_mm - 100.0, l0_mm + 100.0)
            };

            let mut stroke_mm = state.driver_stroke * 1e3;
            stroke_mm = stroke_mm.clamp(slider_min_mm, slider_max_mm);
            let prev_stroke_mm = stroke_mm;
            let resp = ui.add(
                egui::Slider::new(&mut stroke_mm, slider_min_mm..=slider_max_mm)
                    .suffix(" mm")
                    .step_by(0.1),
            ).on_hover_text("Drag to set actuator stroke in mm. Updates the mechanism kinematics by re-solving at the new stroke.");
            if resp.dragged() {
                if state.playing {
                    state.playing = false;
                    state.animation_direction = 1.0;
                }
                if let Some(sim) = &mut state.simulation {
                    sim.playing = false;
                }
            }
            if (stroke_mm - prev_stroke_mm).abs() > 1e-6 {
                state.solve_at_stroke(stroke_mm * 1e-3);
            }

            // Sweep range (mm). Stored in sweep_angle_min_deg/max_deg
            // by repurposing the field for stroke mode — no separate
            // field is required because only one of the two modes is
            // active at a time.
            ui.separator();
            let prev_enabled = state.sweep_range_enabled;
            ui.checkbox(&mut state.sweep_range_enabled, "Limit Sweep Range")
                .on_hover_text("Restrict the stroke sweep to a custom range instead of the full actuator stroke window.");
            if state.sweep_range_enabled != prev_enabled {
                state.mark_sweep_dirty();
            }
            if state.sweep_range_enabled {
                ui.horizontal(|ui| {
                    ui.label("Min:");
                    let min_resp = ui.add(egui::DragValue::new(&mut state.sweep_angle_min_deg)
                        .speed(0.5)
                        .range(0.0..=10000.0)
                        .suffix(" mm"))
                        .on_hover_text("Sweep range min stroke in mm.");
                    ui.label("Max:");
                    let max_resp = ui.add(egui::DragValue::new(&mut state.sweep_angle_max_deg)
                        .speed(0.5)
                        .range(0.0..=10000.0)
                        .suffix(" mm"))
                        .on_hover_text("Sweep range max stroke in mm. Must be \u{2265} min.");
                    let min_done = min_resp.drag_stopped() || min_resp.lost_focus();
                    let max_done = max_resp.drag_stopped() || max_resp.lost_focus();
                    if min_done || max_done {
                        if state.sweep_angle_max_deg < state.sweep_angle_min_deg {
                            state.sweep_angle_max_deg = state.sweep_angle_min_deg;
                        }
                        state.mark_sweep_dirty();
                    }
                });
            }
        });
}

/// Draw forward dynamics simulation controls: duration, run button,
/// timeline scrubber, and stop button.
fn draw_simulation_controls(ui: &mut egui::Ui, state: &mut AppState) {

    // Duration + Simulate button
    ui.horizontal(|ui| {
        ui.label("Duration:");
        ui.add(
            egui::DragValue::new(&mut state.simulation_duration)
                .speed(0.1)
                .range(1.0..=30.0)
                .suffix(" s"),
        ).on_hover_text("Forward dynamics simulation duration in seconds");

        let sim_active = state.simulation.is_some();
        if ui
            .add_enabled(!sim_active, egui::Button::new("Simulate"))
            .on_hover_text("Run forward dynamics from the current pose")
            .clicked()
        {
            let duration = state.simulation_duration;
            state.run_simulation(duration);
        }
    });

    // Timeline slider and stop button (only when simulation exists)
    if state.simulation.is_some() {
        // Extract values we need for the slider before mutably borrowing
        let t_end = state
            .simulation
            .as_ref()
            .map(|s| *s.times.last().unwrap_or(&1.0))
            .unwrap_or(1.0);
        let mut current_t = state
            .simulation
            .as_ref()
            .map(|s| *s.times.get(s.time_index).unwrap_or(&0.0))
            .unwrap_or(0.0);
        let prev_t = current_t;

        ui.horizontal(|ui| {
            ui.label("Time:");
            let response = ui.add(
                egui::Slider::new(&mut current_t, 0.0..=t_end)
                    .suffix(" s")
                    .step_by(t_end / 300.0),
            ).on_hover_text("Scrub through the simulation timeline");
            if response.dragged() {
                // Pause playback when scrubbing
                if let Some(sim) = &mut state.simulation {
                    sim.playing = false;
                }
            }
        });

        // If the slider moved, seek to the new time
        if (current_t - prev_t).abs() > 1e-9 {
            let new_q = {
                let Some(sim) = &mut state.simulation else {
                    unreachable!()
                };
                sim.elapsed = current_t;
                sim.time_index = sim
                    .times
                    .iter()
                    .position(|&t| t >= current_t)
                    .unwrap_or(sim.positions.len() - 1);
                let idx = sim.time_index;
                if idx < sim.positions.len() {
                    Some(sim.positions[idx].clone())
                } else {
                    None
                }
            };
            if let Some(q) = new_q {
                state.q = q;
            }
        }

        // Playback controls + stop
        ui.horizontal(|ui| {
            // Play/Pause for simulation playback
            let is_playing = state
                .simulation
                .as_ref()
                .map(|s| s.playing)
                .unwrap_or(false);
            let play_label = if is_playing { "Pause" } else { "Play" };
            if ui.button(play_label)
                .on_hover_text("Play/pause simulation timeline playback")
                .clicked()
            {
                if let Some(sim) = &mut state.simulation {
                    sim.playing = !sim.playing;
                    if sim.playing {
                        // Stop kinematic animation — only one can drive the canvas.
                        state.playing = false;
                        // If resuming at the end, restart from beginning
                        if sim.time_index >= sim.positions.len().saturating_sub(1) {
                            sim.time_index = 0;
                            sim.elapsed = 0.0;
                        }
                    }
                }
            }

            ui.label("Speed:");
            let mut speed = state
                .simulation
                .as_ref()
                .map(|s| s.speed)
                .unwrap_or(1.0);
            if ui
                .add(
                    egui::DragValue::new(&mut speed)
                        .speed(0.05)
                        .range(0.1..=5.0)
                        .suffix("x"),
                )
                .on_hover_text("Simulation playback speed multiplier")
                .changed()
            {
                if let Some(sim) = &mut state.simulation {
                    sim.speed = speed;
                }
            }

            if ui
                .button("Stop")
                .on_hover_text("Stop simulation and return to kinematic mode")
                .clicked()
            {
                state.simulation = None;
            }
        });
    }
}

/// Driver picker shown when the mechanism has no active driver.
///
/// Lists the grounded revolute joints and lets the user pick one to
/// become the driven input. If no grounded revolute joint exists, shows
/// an inline hint pointing at the + Ground tool (matching the canvas
/// context-menu copy) so the user always has a visible path forward.
fn draw_no_driver_picker(ui: &mut egui::Ui, state: &mut AppState) {
    let Some(mech) = state.mechanism.as_ref() else { return };
    let grounded = mech.grounded_revolute_joint_ids();

    if grounded.is_empty() {
        ui.small(
            egui::RichText::new(
                "No driver \u{2014} add a ground pivot with the + Ground tool, then pick one of its joints here.",
            )
            .italics()
            .color(egui::Color32::from_rgb(230, 180, 90)),
        );
        return;
    }

    ui.horizontal(|ui| {
        ui.label("No driver. Set:");
        egui::ComboBox::from_id_salt("driver_picker")
            .selected_text("(pick a joint)")
            .show_ui(ui, |ui| {
                for joint_id in &grounded {
                    if ui.selectable_label(false, joint_id).clicked() {
                        state.pending_driver_reassignment = Some(joint_id.clone());
                    }
                }
            })
            .response
            .on_hover_text(
                "Grounded revolute joints can be driven. Picking one sets it as the mechanism's input and rebuilds.",
            );
    });
}

/// Draw the load case selector: ComboBox for switching, +/- buttons, and
/// an editable name field for the active case.
fn draw_load_case_selector(ui: &mut egui::Ui, state: &mut AppState) {
    if state.load_cases.cases.is_empty() {
        return;
    }

    ui.separator();
    ui.strong("Load Case");

    // Collect values needed for the combo box to avoid borrow issues
    let active_index = state.load_cases.active_index;
    let active_name = state.load_cases.cases[active_index].name.clone();
    let case_count = state.load_cases.cases.len();

    // Combo box + add/remove buttons on the same row
    let mut new_active: Option<usize> = None;
    let mut add_case = false;
    let mut remove_case = false;

    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("load_case_selector")
            .selected_text(&active_name)
            .show_ui(ui, |ui| {
                for i in 0..case_count {
                    let case_name = state.load_cases.cases[i].name.clone();
                    if ui
                        .selectable_label(i == active_index, &case_name)
                        .clicked()
                    {
                        new_active = Some(i);
                    }
                }
            });

        if ui.button("+").on_hover_text("Add load case (copy current)").clicked() {
            add_case = true;
        }

        let remove_enabled = case_count > 1;
        if ui
            .add_enabled(remove_enabled, egui::Button::new("-"))
            .on_hover_text("Remove current load case")
            .clicked()
        {
            remove_case = true;
        }
    });

    // Editable name for the active case
    ui.horizontal(|ui| {
        ui.label("Name:");
        let name = &mut state.load_cases.cases[active_index].name;
        ui.text_edit_singleline(name);
    });

    // Apply deferred actions after UI rendering to avoid borrow conflicts
    if let Some(idx) = new_active {
        state.apply_load_case(idx);
    }
    if add_case {
        state.add_load_case();
    }
    if remove_case {
        state.remove_active_load_case();
    }
}

/// Draw a combo box to switch between "Constant Speed" and "Custom Expression"
/// driver modes, plus text fields for expression editing.
fn draw_driver_type_selector(ui: &mut egui::Ui, state: &mut AppState) {
    // Only show when there's a blueprint with at least one driver
    let Some(bp) = &state.blueprint else { return };
    if bp.drivers.is_empty() {
        return;
    }

    // Determine the current driver type from the blueprint
    let is_expression = bp
        .drivers
        .values()
        .next()
        .is_some_and(|d| matches!(d, DriverJson::Expression { .. }));

    ui.separator();
    ui.strong("Driver Function");

    // Combo box for driver type
    let current_label = if is_expression {
        "Custom Expression"
    } else {
        "Constant Speed"
    };
    let mut switch_to_expression = false;
    let mut switch_to_constant = false;

    egui::ComboBox::from_id_salt("driver_type_selector")
        .selected_text(current_label)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(!is_expression, "Constant Speed")
                .clicked()
                && is_expression
            {
                switch_to_constant = true;
            }
            if ui
                .selectable_label(is_expression, "Custom Expression")
                .clicked()
                && !is_expression
            {
                switch_to_expression = true;
            }
        });

    if switch_to_expression {
        // Initialize expression buffers with default linear driver
        state.expr_buf = "2*pi*t".to_string();
        state.expr_dot_buf = "2*pi".to_string();
        state.expr_ddot_buf = "0".to_string();
        state.expr_error = None;
        state.set_expression_driver("2*pi*t", "2*pi", "0");
        return;
    }

    if switch_to_constant {
        state.expr_error = None;
        state.set_constant_speed_driver(state.driver_omega, state.driver_theta_0);
        return;
    }

    // Speed editor (constant-speed mode only). Edits the driver's
    // angular velocity omega in RPM. Used for inverse dynamics
    // (inertial loads scale with omega^2) and for the playback timescale.
    // Static force calculations are correctly independent of speed.
    let mut apply_new_omega: Option<f64> = None;
    if !is_expression {
        let current_omega = state.driver_omega;
        let current_theta_0 = state.driver_theta_0;
        ui.horizontal(|ui| {
            ui.label("Speed:");
            let mut rpm = current_omega * 60.0 / (2.0 * std::f64::consts::PI);
            let rpm_resp = ui.add(
                egui::DragValue::new(&mut rpm)
                    .speed(1.0)
                    .range(-10000.0..=10000.0)
                    .suffix(" RPM"),
            ).on_hover_text(
                "Driver angular velocity in RPM. Statics are independent of this \
                 value (by design). Inverse dynamics inertial loads scale with \
                 omega squared. Negative = reverse rotation."
            );
            ui.label(format!("({:.2} rad/s)", current_omega));
            if rpm_resp.drag_stopped() || rpm_resp.lost_focus() {
                let new_omega = rpm * 2.0 * std::f64::consts::PI / 60.0;
                if (new_omega - current_omega).abs() > 1e-9 {
                    apply_new_omega = Some(new_omega);
                }
            }
            let _ = current_theta_0;
        });
    }
    if let Some(new_omega) = apply_new_omega {
        let theta_0 = state.driver_theta_0;
        state.set_constant_speed_driver(new_omega, theta_0);
        return;
    }

    // Show expression editor when in expression mode
    if is_expression {
        // Sync buffers from blueprint on first render (if empty)
        if state.expr_buf.is_empty() {
            if let Some(DriverJson::Expression {
                expr,
                expr_dot,
                expr_ddot,
                ..
            }) = bp.drivers.values().next()
            {
                state.expr_buf = expr.clone();
                state.expr_dot_buf = expr_dot.clone();
                state.expr_ddot_buf = expr_ddot.clone();
            }
        }

        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label("f(t) =");
            let response = ui.text_edit_singleline(&mut state.expr_buf);
            if response.lost_focus() || response.changed() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("f'(t) =");
            let response = ui.text_edit_singleline(&mut state.expr_dot_buf);
            if response.lost_focus() || response.changed() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("f''(t) =");
            let response = ui.text_edit_singleline(&mut state.expr_ddot_buf);
            if response.lost_focus() || response.changed() {
                changed = true;
            }
        });

        // Show error if present
        if let Some(err) = &state.expr_error {
            ui.colored_label(state.nc(egui::Color32::from_rgb(220, 80, 80)), err);
        }

        if changed {
            // Validate by attempting to parse (fast feedback)
            let expr = state.expr_buf.clone();
            let expr_dot = state.expr_dot_buf.clone();
            let expr_ddot = state.expr_ddot_buf.clone();

            match crate::core::driver::expression_driver(
                "validate", "a", "b", &expr, &expr_dot, &expr_ddot,
            ) {
                Ok(_) => {
                    state.expr_error = None;
                    state.set_expression_driver(&expr, &expr_dot, &expr_ddot);
                }
                Err(e) => {
                    state.expr_error = Some(e);
                }
            }
        }
    }
}

/// Draw a combo box for selecting the driver motion profile (Constant Speed vs
/// Trapezoidal) and, when Trapezoidal is selected, sliders for accel/decel
/// fractions.
fn draw_motion_profile_selector(ui: &mut egui::Ui, state: &mut AppState) {
    // Only relevant when a mechanism with a driver exists.
    let Some(bp) = &state.blueprint else { return };
    if bp.drivers.is_empty() {
        return;
    }

    ui.separator();
    ui.strong("Motion Profile");

    let selected_text = match state.motion_profile {
        MotionProfile::ConstantSpeed => "Constant Speed",
        MotionProfile::Trapezoidal { .. } => "Trapezoidal",
    };

    egui::ComboBox::from_id_salt("motion_profile")
        .selected_text(selected_text)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(
                    matches!(state.motion_profile, MotionProfile::ConstantSpeed),
                    "Constant Speed",
                )
                .clicked()
                && !matches!(state.motion_profile, MotionProfile::ConstantSpeed)
            {
                state.motion_profile = MotionProfile::ConstantSpeed;
                state.mark_sweep_dirty();
            }
            if ui
                .selectable_label(
                    matches!(state.motion_profile, MotionProfile::Trapezoidal { .. }),
                    "Trapezoidal",
                )
                .clicked()
                && !matches!(state.motion_profile, MotionProfile::Trapezoidal { .. })
            {
                state.motion_profile = MotionProfile::Trapezoidal {
                    accel_fraction: 0.25,
                    decel_fraction: 0.25,
                };
                state.mark_sweep_dirty();
            }
        });

    if let MotionProfile::Trapezoidal { accel_fraction, decel_fraction } = state.motion_profile {
        // Copy values out so sliders can work without holding a borrow on state.
        let mut af = accel_fraction;
        let mut df = decel_fraction;
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Accel:");
            if ui
                .add(
                    egui::Slider::new(&mut af, 0.05..=0.45)
                        .fixed_decimals(2)
                        .step_by(0.01),
                )
                .on_hover_text("Fraction of cycle spent accelerating")
                .changed()
            {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Decel:");
            if ui
                .add(
                    egui::Slider::new(&mut df, 0.05..=0.45)
                        .fixed_decimals(2)
                        .step_by(0.01),
                )
                .on_hover_text("Fraction of cycle spent decelerating")
                .changed()
            {
                changed = true;
            }
        });

        // Write back and mark dirty if changed.
        if changed {
            state.motion_profile = MotionProfile::Trapezoidal {
                accel_fraction: af,
                decel_fraction: df,
            };
            state.mark_sweep_dirty();
        }

        // Show computed peak omega for reference.
        let cruise_frac = 1.0 - af - df;
        if cruise_frac > 0.0 && state.driver_omega.abs() > 1e-15 {
            let total_angle = 2.0 * std::f64::consts::PI;
            let cycle_time = total_angle / state.driver_omega;
            let denom = 0.5 * af * cycle_time
                + cruise_frac * cycle_time
                + 0.5 * df * cycle_time;
            if denom.abs() > 1e-15 {
                let omega_peak = total_angle / denom;
                ui.label(
                    egui::RichText::new(format!(
                        "Peak: {:.1} rad/s ({:.0} RPM)",
                        omega_peak,
                        omega_peak * 60.0 / (2.0 * std::f64::consts::PI)
                    ))
                    .small()
                    .weak(),
                );
            }
        }
    }
}
