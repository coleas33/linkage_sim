//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod error_panel;
mod export;
mod input_panel;
mod parametric_panel;
mod plot_panel;
mod property_panel;
pub mod samples;
pub mod undo;

use eframe::egui;
pub use state::AppState;
use samples::SampleMechanism;
use state::{AngleUnit, EditorTool, LengthUnit, SelectedEntity};

/// Top-level application struct for eframe.
pub struct LinkageApp {
    state: AppState,
}

impl LinkageApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            state: AppState::default(),
        }
    }
}

impl eframe::App for LinkageApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Keyboard shortcuts ────────────────────────────────────────
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z) && !i.modifiers.shift) {
            self.state.undo();
        }
        if ctx.input(|i| {
            i.modifiers.command
                && (i.key_pressed(egui::Key::Y)
                    || (i.key_pressed(egui::Key::Z) && i.modifiers.shift))
        }) {
            self.state.redo();
        }
        // Ctrl+S — quick save to last path, or Save As if no path yet.
        #[cfg(feature = "native")]
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && !i.modifiers.shift) {
            if let Some(path) = self.state.last_save_path.clone() {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Quick save failed: {}", e);
                }
            } else if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .set_file_name("mechanism.json")
                .save_file()
            {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Save failed: {}", e);
                }
            }
        }
        // Ctrl+Shift+S — Save As (always shows file dialog).
        #[cfg(feature = "native")]
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::S) && i.modifiers.shift) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .set_file_name("mechanism.json")
                .save_file()
            {
                if let Err(e) = self.state.save_to_file(&path) {
                    log::error!("Save As failed: {}", e);
                }
            }
        }

        // Ctrl+N — New empty mechanism.
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::N)) {
            self.state.new_empty_mechanism();
        }

        // ── Debounced sweep recomputation ──────────────────────────────
        if self.state.sweep_dirty {
            let now = ctx.input(|i| i.time);
            // Stamp the dirty-since time on first detection.
            if self.state.sweep_dirty_since.is_none() {
                self.state.sweep_dirty_since = Some(now);
            }
            if let Some(since) = self.state.sweep_dirty_since {
                if (now - since) >= 0.2 {
                    self.state.compute_sweep();
                }
            }
            ctx.request_repaint();
        }

        // ── Animation / simulation stepping (before rendering) ────────
        let dt = ctx.input(|i| i.stable_dt) as f64;
        if self.state.step_simulation(dt) {
            ctx.request_repaint();
        }
        if self.state.step_animation(dt) {
            ctx.request_repaint();
        }

        // ── Process pending driver reassignment ──────────────────────
        if let Some(joint_id) = self.state.pending_driver_reassignment.take() {
            self.state.reassign_driver(&joint_id);
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New  Ctrl+N").clicked() {
                        self.state.new_empty_mechanism();
                        ui.close();
                    }
                    ui.menu_button("Load Sample", |ui| {
                        for sample in SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close();
                            }
                        }
                    });
                    // ── Native-only file dialogs ──────────────────────
                    #[cfg(feature = "native")]
                    {
                        ui.separator();
                        if ui.button("Open JSON...").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .pick_file()
                            {
                                if let Err(e) = self.state.load_from_file(&path) {
                                    log::error!("Failed to load mechanism: {}", e);
                                }
                            }
                            ui.close();
                        }
                        if !self.state.recent_files.is_empty() {
                            ui.menu_button("Recent Files", |ui| {
                                let mut load_path = None;
                                for path in &self.state.recent_files {
                                    let label = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().to_string())
                                        .unwrap_or_else(|| path.to_string_lossy().to_string());
                                    if ui
                                        .button(&label)
                                        .on_hover_text(path.to_string_lossy().to_string())
                                        .clicked()
                                    {
                                        load_path = Some(path.clone());
                                        ui.close();
                                    }
                                }
                                if let Some(path) = load_path {
                                    if let Err(e) = self.state.load_from_file(&path) {
                                        log::error!("Failed to load recent file: {}", e);
                                    }
                                }
                            });
                        }
                        if ui.button("Save  Ctrl+S").clicked() {
                            if let Some(path) = self.state.last_save_path.clone() {
                                if let Err(e) = self.state.save_to_file(&path) {
                                    log::error!("Save failed: {}", e);
                                }
                            } else if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .set_file_name("mechanism.json")
                                .save_file()
                            {
                                if let Err(e) = self.state.save_to_file(&path) {
                                    log::error!("Save failed: {}", e);
                                }
                            }
                            ui.close();
                        }
                        if ui.button("Save As...  Ctrl+Shift+S").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .set_file_name("mechanism.json")
                                .save_file()
                            {
                                if let Err(e) = self.state.save_to_file(&path) {
                                    log::error!("Failed to save mechanism: {}", e);
                                }
                            }
                            ui.close();
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                self.state.sweep_data.is_some(),
                                egui::Button::new("Export Sweep CSV..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("CSV", &["csv"])
                                .set_file_name("sweep_data.csv")
                                .save_file()
                            {
                                if let Some(ref sweep) = self.state.sweep_data {
                                    if let Err(e) = export::export_sweep_csv(&path, sweep) {
                                        log::error!("CSV export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                self.state.sweep_data.is_some(),
                                egui::Button::new("Export Coupler CSV..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("CSV", &["csv"])
                                .set_file_name("coupler_trace.csv")
                                .save_file()
                            {
                                if let Some(ref sweep) = self.state.sweep_data {
                                    if let Err(e) = export::export_coupler_csv(&path, sweep) {
                                        log::error!("Coupler CSV export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                self.state.mechanism.is_some(),
                                egui::Button::new("Export SVG..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("SVG", &["svg"])
                                .set_file_name("mechanism.svg")
                                .save_file()
                            {
                                if let Some(ref mech) = self.state.mechanism {
                                    if let Err(e) =
                                        export::export_mechanism_svg(&path, mech, &self.state.q)
                                    {
                                        log::error!("SVG export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                self.state.mechanism.is_some(),
                                egui::Button::new("Export PNG..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PNG", &["png"])
                                .set_file_name("mechanism.png")
                                .save_file()
                            {
                                if let Some(ref mech) = self.state.mechanism {
                                    if let Err(e) = export::export_mechanism_png(
                                        &path,
                                        mech,
                                        &self.state.q,
                                        1920,
                                        1080,
                                    ) {
                                        log::error!("PNG export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                self.state.sweep_data.is_some()
                                    && self.state.mechanism.is_some(),
                                egui::Button::new("Export GIF..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("GIF", &["gif"])
                                .set_file_name("mechanism.gif")
                                .save_file()
                            {
                                if let (Some(mech), Some(sweep)) =
                                    (&self.state.mechanism, &self.state.sweep_data)
                                {
                                    if let Err(e) = export::export_mechanism_gif(
                                        &path,
                                        mech,
                                        sweep,
                                        &self.state.q,
                                        self.state.driver_omega,
                                        self.state.driver_theta_0,
                                        800,
                                        600,
                                        5,
                                    ) {
                                        log::error!("GIF export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                    }
                    #[cfg(feature = "native")]
                    {
                        if ui
                            .add_enabled(
                                self.state.mechanism.is_some(),
                                egui::Button::new("Export DXF..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("DXF", &["dxf"])
                                .set_file_name("mechanism.dxf")
                                .save_file()
                            {
                                if let Some(ref mech) = self.state.mechanism {
                                    if let Err(e) =
                                        export::export_mechanism_dxf(&path, mech, &self.state.q)
                                    {
                                        log::error!("DXF export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                self.state.sweep_data.is_some()
                                    && self.state.mechanism.is_some(),
                                egui::Button::new("Generate Report (HTML)..."),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("HTML", &["html"])
                                .set_file_name("mechanism_report.html")
                                .save_file()
                            {
                                if let (Some(mech), Some(sweep)) =
                                    (&self.state.mechanism, &self.state.sweep_data)
                                {
                                    match export::generate_html_report(
                                        mech,
                                        &self.state.q,
                                        sweep,
                                        self.state.grashof_result.as_ref(),
                                        &self.state.display_units,
                                    ) {
                                        Ok(html) => {
                                            if let Err(e) = std::fs::write(&path, &html) {
                                                log::error!("Report write failed: {}", e);
                                            } else {
                                                // Open in browser
                                                let _ = open::that(&path);
                                            }
                                        }
                                        Err(e) => log::error!("Report generation failed: {}", e),
                                    }
                                }
                            }
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(self.state.can_undo(), egui::Button::new("Undo  Ctrl+Z"))
                        .clicked()
                    {
                        self.state.undo();
                        ui.close();
                    }
                    if ui
                        .add_enabled(self.state.can_redo(), egui::Button::new("Redo  Ctrl+Y"))
                        .clicked()
                    {
                        self.state.redo();
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts").clicked() {
                        self.state.show_shortcuts = true;
                        ui.close();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay");
                    ui.checkbox(&mut self.state.show_plots, "Plot Panel");
                    ui.checkbox(&mut self.state.show_parametric, "Parametric Study");
                    ui.checkbox(&mut self.state.show_forces, "Force Arrows");
                    ui.checkbox(&mut self.state.show_dimensions, "Link Dimensions");
                    let enabled = self.state.gravity_magnitude > 0.0;
                    let mut check = enabled;
                    if ui.checkbox(&mut check, "Gravity").changed() {
                        self.state.gravity_magnitude = if check { 9.81 } else { 0.0 };
                    }
                    ui.separator();
                    ui.label("Units:");
                    let mut use_mm = self.state.display_units.length == LengthUnit::Millimeters;
                    if ui.checkbox(&mut use_mm, "Millimeters").changed() {
                        self.state.display_units.length = if use_mm {
                            LengthUnit::Millimeters
                        } else {
                            LengthUnit::Meters
                        };
                    }
                    let mut use_deg = self.state.display_units.angle == AngleUnit::Degrees;
                    if ui.checkbox(&mut use_deg, "Degrees").changed() {
                        self.state.display_units.angle = if use_deg {
                            AngleUnit::Degrees
                        } else {
                            AngleUnit::Radians
                        };
                    }
                    ui.separator();
                    ui.label("Grid:");
                    ui.checkbox(&mut self.state.grid.show_grid, "Show Grid");
                    ui.checkbox(&mut self.state.grid.snap_enabled, "Snap to Grid");
                    ui.horizontal(|ui| {
                        ui.label("Spacing:");
                        let mut spacing_display =
                            self.state.display_units.length(self.state.grid.spacing_m);
                        if ui
                            .add(
                                egui::DragValue::new(&mut spacing_display)
                                    .speed(0.1)
                                    .range(0.001..=100.0)
                                    .suffix(self.state.display_units.length_suffix()),
                            )
                            .changed()
                        {
                            self.state.grid.spacing_m =
                                self.state.display_units.length_to_si(spacing_display);
                        }
                    });
                });
            });
        });

        // ── Delete / Backspace shortcut ───────────────────────────────────
        if ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
            match self.state.selected.take() {
                Some(SelectedEntity::Body(id)) => {
                    self.state.remove_body(&id);
                }
                Some(SelectedEntity::Joint(id)) => {
                    self.state.remove_joint(&id);
                }
                other => {
                    self.state.selected = other;
                }
            }
        }

        // --- Toolbar ---
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().button_padding = egui::vec2(6.0, 3.0);

                let tool = self.state.active_tool;

                if ui
                    .selectable_label(tool == EditorTool::Select, "Select")
                    .on_hover_text("Select and move entities")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::Select;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                if ui
                    .selectable_label(
                        tool == EditorTool::DrawLink || self.state.draw_link_start.is_some(),
                        "Draw Link",
                    )
                    .on_hover_text("Click and drag to draw a link")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::DrawLink;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                if ui
                    .selectable_label(
                        tool == EditorTool::AddBody || self.state.add_body_state.is_some(),
                        "+ Body",
                    )
                    .on_hover_text("Click to place attachment points, double-click or Enter to finish")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddBody;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                if ui
                    .selectable_label(tool == EditorTool::AddGroundPivot, "+ Ground")
                    .on_hover_text("Click canvas to place a ground pivot")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddGroundPivot;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }
            });
        });

        // --- Status bar ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(sample) = self.state.current_sample {
                    ui.label(format!("Mechanism: {}", sample.label()));
                    ui.separator();
                }

                if self.state.has_mechanism() {
                    let status = &self.state.solver_status;
                    let color = if status.converged {
                        egui::Color32::from_rgb(80, 200, 80)
                    } else {
                        egui::Color32::from_rgb(200, 60, 60)
                    };
                    ui.colored_label(color, "●");
                    ui.label(format!("‖Φ‖ = {:.2e}", status.residual_norm));
                    ui.separator();
                    ui.label(format!(
                        "\u{03b8} = {:.1}{}",
                        self.state.display_units.angle(self.state.driver_angle),
                        self.state.display_units.angle_suffix()
                    ));

                    if let Some(torque) = self.state.force_results.driver_torque {
                        ui.separator();
                        ui.label(format!("\u{03c4} = {:.3} N\u{00b7}m", torque));
                    }

                    if self.state.playing {
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(80, 200, 80), "PLAYING");
                    }

                    if let Some(sim) = &self.state.simulation {
                        ui.separator();
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 255),
                            "SIM",
                        );
                        ui.label(format!(
                            "t = {:.3} s",
                            sim.times.get(sim.time_index).unwrap_or(&0.0)
                        ));
                        if let Some(&drift) = sim.drift.get(sim.time_index) {
                            ui.label(format!("drift = {:.2e}", drift));
                        }
                    }

                    if let Some(mech) = &self.state.mechanism {
                        let dof = mech.state().n_coords() as isize - mech.n_constraints() as isize;
                        ui.separator();
                        ui.label(format!(
                            "Bodies: {} | Joints: {} | DOF: {}",
                            mech.bodies().len().saturating_sub(1),
                            mech.joints().len(),
                            dof,
                        ));
                    }

                    // Validation warnings from the computed ValidationWarnings struct.
                    let warn_color = egui::Color32::from_rgb(255, 180, 50);
                    let warnings = &self.state.validation_warnings;

                    if let Some(ref dof_msg) = warnings.dof_warning {
                        ui.separator();
                        ui.colored_label(warn_color, dof_msg);
                    }
                    if warnings.missing_driver {
                        ui.separator();
                        ui.colored_label(warn_color, "No driver");
                    }
                    if !warnings.disconnected_bodies.is_empty() {
                        ui.separator();
                        ui.colored_label(
                            warn_color,
                            format!(
                                "Disconnected: {}",
                                warnings.disconnected_bodies.join(", ")
                            ),
                        );
                    }

                    if !self.state.error_log.is_empty() {
                        ui.separator();
                        let label = format!("{} error(s)", self.state.error_log.len());
                        if ui
                            .colored_label(egui::Color32::from_rgb(220, 80, 80), &label)
                            .on_hover_text("Click to show error panel")
                            .clicked()
                        {
                            self.state.show_error_panel = !self.state.show_error_panel;
                        }
                    }
                } else {
                    ui.label("No mechanism loaded");
                }
            });
        });

        // --- Bottom panel: plots ---
        if self.state.show_plots {
            egui::TopBottomPanel::bottom("plot_panel")
                .resizable(true)
                .default_height(250.0)
                .show(ctx, |ui| {
                    plot_panel::draw_plot_panel(ui, &self.state);
                });
        }

        // --- Error panel (between plots and canvas) ---
        if self.state.show_error_panel && !self.state.error_log.is_empty() {
            egui::TopBottomPanel::bottom("error_panel")
                .resizable(true)
                .default_height(120.0)
                .show(ctx, |ui| {
                    error_panel::draw_error_panel(ui, &mut self.state);
                });
        }

        // --- Left panel: properties + input ---
        egui::SidePanel::left("left_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    property_panel::draw_property_panel(ui, &mut self.state);
                    ui.add_space(20.0);
                    input_panel::draw_input_panel(ui, &mut self.state);
                });
            });

        // --- Right panel: parametric study ---
        if self.state.show_parametric {
            egui::SidePanel::right("parametric_panel")
                .default_width(300.0)
                .resizable(true)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        parametric_panel::draw_parametric_panel(ui, &mut self.state);
                        ui.add_space(20.0);
                        ui.separator();
                        parametric_panel::draw_counterbalance_panel(ui, &mut self.state);
                    });
                });
        }

        // --- Central canvas ---
        egui::CentralPanel::default().show(ctx, |ui| {
            canvas::draw_canvas(ui, &mut self.state);
        });

        // ── Autosave recovery prompt ──────────────────────────────────
        if self.state.recovery_path.is_some() {
            let mut dismiss = false;
            let mut load = false;
            egui::Window::new("Recover Unsaved Work?")
                .collapsible(false)
                .resizable(false)
                .default_width(340.0)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("An autosave file was found from a previous session.");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Recover").clicked() {
                            load = true;
                        }
                        if ui.button("Discard").clicked() {
                            dismiss = true;
                        }
                    });
                });
            if load {
                if let Some(path) = self.state.recovery_path.take() {
                    if let Err(e) = self.state.load_from_file(&path) {
                        log::error!("Failed to recover autosave: {}", e);
                    }
                    // Clean up the autosave file after loading.
                    let _ = std::fs::remove_file(&path);
                }
            } else if dismiss {
                if let Some(path) = self.state.recovery_path.take() {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }

        // ── Keyboard shortcuts window ────────────────────────────────────
        if self.state.show_shortcuts {
            egui::Window::new("Keyboard Shortcuts")
                .collapsible(false)
                .resizable(false)
                .default_width(340.0)
                .open(&mut self.state.show_shortcuts)
                .show(ctx, |ui| {
                    egui::Grid::new("shortcuts_grid")
                        .num_columns(2)
                        .spacing([20.0, 6.0])
                        .show(ui, |ui| {
                            let shortcuts = [
                                ("Ctrl+N", "New empty mechanism"),
                                ("Ctrl+S", "Save (quick save to last path)"),
                                ("Ctrl+Shift+S", "Save As (choose new path)"),
                                ("Ctrl+Z", "Undo"),
                                ("Ctrl+Y / Ctrl+Shift+Z", "Redo"),
                                ("Delete / Backspace", "Delete selected entity"),
                                ("Escape", "Cancel current tool / operation"),
                                ("Enter / Double-click", "Finish multi-point body"),
                                ("Mouse wheel", "Zoom in/out"),
                                ("Right-click drag", "Pan canvas"),
                                ("Left-click", "Select / place point"),
                                ("Right-click joint", "Set Driver / Create Joint"),
                                ("Right-click body edge", "Add Pivot Here"),
                                ("Right-click canvas", "Add Ground Pivot / Body"),
                            ];
                            for (key, action) in shortcuts {
                                ui.strong(key);
                                ui.label(action);
                                ui.end_row();
                            }
                        });
                });
        }

        // ── Autosave tick ────────────────────────────────────────────────
        #[cfg(feature = "native")]
        self.state.tick_autosave(dt);
    }
}
