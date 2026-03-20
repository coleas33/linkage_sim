//! Angle slider, playback controls, load case selector, and solver status display.

use eframe::egui;
use super::state::AppState;
use crate::io::serialization::DriverJson;

/// Draw the input panel with animation controls and load case management.
pub fn draw_input_panel(ui: &mut egui::Ui, state: &mut AppState) {
    if !state.has_mechanism() {
        return;
    }

    ui.separator();
    ui.heading("Driver Input");

    // ── Load case selector ──────────────────────────────────────────
    draw_load_case_selector(ui, state);

    // ── Playback controls ────────────────────────────────────────────
    ui.horizontal(|ui| {
        let play_label = if state.playing { "Pause" } else { "Play" };
        if ui.button(play_label).clicked() {
            state.playing = !state.playing;
            if state.playing && !state.loop_mode {
                state.animation_direction = 1.0;
            }
        }

        ui.label("Speed:");
        ui.add(
            egui::Slider::new(&mut state.animation_speed_deg_per_sec, 10.0..=720.0)
                .suffix(" \u{00B0}/s")
                .logarithmic(true)
                .clamping(egui::SliderClamping::Always),
        );
    });

    ui.horizontal(|ui| {
        let mode_label = if state.loop_mode { "Loop" } else { "Once" };
        if ui.button(mode_label).clicked() {
            state.loop_mode = !state.loop_mode;
            state.animation_direction = 1.0;
        }
        if state.loop_mode {
            ui.label("(continuous, ping-pong at limits)");
        } else {
            ui.label("(sweep forward, stop at 360\u{00B0})");
        }
    });

    // ── Angle slider ─────────────────────────────────────────────────
    let mut angle_deg = state.driver_angle.to_degrees();
    let prev_angle = angle_deg;

    ui.horizontal(|ui| {
        ui.label("Crank angle:");
        let response = ui.add(
            egui::Slider::new(&mut angle_deg, 0.0..=360.0)
                .suffix("\u{00B0}")
                .step_by(0.5),
        );
        if response.dragged() && state.playing {
            state.playing = false;
            state.animation_direction = 1.0;
        }
    });

    if (angle_deg - prev_angle).abs() > 1e-6 {
        state.solve_at_angle(angle_deg.to_radians());
    }

    // ── Solver status ────────────────────────────────────────────────
    ui.separator();
    let status = &state.solver_status;
    ui.horizontal(|ui| {
        let color = if status.converged {
            egui::Color32::from_rgb(80, 200, 80)
        } else {
            egui::Color32::from_rgb(200, 60, 60)
        };
        ui.colored_label(color, "\u{25CF}");
        if status.converged {
            ui.label(format!(
                "Converged in {} iters (r = {:.2e})",
                status.iterations, status.residual_norm
            ));
        } else {
            ui.label(format!(
                "FAILED (r = {:.2e}) \u{2014} last good pose",
                status.residual_norm
            ));
        }
    });

    // ── Gravity controls ───────────────────────────────────────────────
    ui.separator();
    let prev_g = state.gravity_magnitude;
    ui.horizontal(|ui| {
        ui.label("Gravity:");
        ui.add(
            egui::Slider::new(&mut state.gravity_magnitude, 0.0..=981.0)
                .suffix(" m/s\u{00b2}")
                .step_by(0.01)
                .clamping(egui::SliderClamping::Always),
        );
    });
    ui.horizontal(|ui| {
        ui.add_space(55.0);
        ui.label(format!("({:.2} g)", state.gravity_magnitude / 9.81));
    });
    if (state.gravity_magnitude - prev_g).abs() > 1e-9 {
        state.mark_sweep_dirty();
    }

    // ── Driver info ──────────────────────────────────────────────────
    if let Some(joint_id) = &state.driver_joint_id {
        ui.label(format!("Driver: {} (right-click joint to change)", joint_id));
    }

    // ── Driver type selector + expression editor ────────────────────
    draw_driver_type_selector(ui, state);

    // ── Simulation ──────────────────────────────────────────────────
    draw_simulation_controls(ui, state);
}

/// Draw forward dynamics simulation controls: duration, run button,
/// timeline scrubber, and stop button.
fn draw_simulation_controls(ui: &mut egui::Ui, state: &mut AppState) {
    ui.separator();
    ui.strong("Simulation");

    // Duration + Simulate button
    ui.horizontal(|ui| {
        ui.label("Duration:");
        ui.add(
            egui::DragValue::new(&mut state.simulation_duration)
                .speed(0.1)
                .range(1.0..=30.0)
                .suffix(" s"),
        );

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
            );
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
            if ui.button(play_label).clicked() {
                if let Some(sim) = &mut state.simulation {
                    sim.playing = !sim.playing;
                    // If resuming at the end, restart from beginning
                    if sim.playing && sim.time_index >= sim.positions.len().saturating_sub(1) {
                        sim.time_index = 0;
                        sim.elapsed = 0.0;
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
            if ui.text_edit_singleline(&mut state.expr_buf).lost_focus() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("f'(t) =");
            if ui.text_edit_singleline(&mut state.expr_dot_buf).lost_focus() {
                changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("f''(t) =");
            if ui.text_edit_singleline(&mut state.expr_ddot_buf).lost_focus() {
                changed = true;
            }
        });

        // Show error if present
        if let Some(err) = &state.expr_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
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
