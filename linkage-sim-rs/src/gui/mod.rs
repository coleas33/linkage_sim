//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod input_panel;
mod property_panel;
pub mod samples;

use eframe::egui;
pub use state::AppState;
use samples::SampleMechanism;

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
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay");
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
                    ui.label(format!("θ = {:.1}°", self.state.driver_angle.to_degrees()));

                    if let Some(mech) = &self.state.mechanism {
                        ui.separator();
                        ui.label(format!(
                            "Bodies: {} | Joints: {} | DOF: {}",
                            mech.bodies().len() - 1,
                            mech.joints().len(),
                            mech.state().n_coords() as isize - mech.n_constraints() as isize
                        ));
                    }
                } else {
                    ui.label("No mechanism loaded");
                }
            });
        });

        // --- Left panel: properties + input ---
        egui::SidePanel::left("left_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    property_panel::draw_property_panel(ui, &self.state);
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
