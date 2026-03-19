//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod export;
mod input_panel;
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
                    ui.menu_button("Load Sample", |ui| {
                        for sample in SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close();
                            }
                        }
                    });
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
                    if ui.button("Save JSON...").clicked() {
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
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay");
                    ui.checkbox(&mut self.state.show_plots, "Plot Panel");
                    ui.checkbox(&mut self.state.show_forces, "Force Arrows");
                    ui.checkbox(&mut self.state.enable_gravity, "Gravity");
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

        // --- Central canvas ---
        egui::CentralPanel::default().show(ctx, |ui| {
            canvas::draw_canvas(ui, &mut self.state);
        });
    }
}
