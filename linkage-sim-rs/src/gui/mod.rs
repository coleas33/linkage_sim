//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod input_panel;
mod plot_panel;
mod property_panel;
pub mod samples;
pub mod undo;

use eframe::egui;
pub use state::AppState;
use samples::SampleMechanism;
use state::{LengthUnit, AngleUnit};

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

        // ── Animation stepping (before rendering) ────────────────────
        let dt = ctx.input(|i| i.stable_dt) as f64;
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
                });
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

                    if self.state.playing {
                        ui.separator();
                        ui.colored_label(egui::Color32::from_rgb(80, 200, 80), "PLAYING");
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
