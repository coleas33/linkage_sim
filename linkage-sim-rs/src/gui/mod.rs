//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod input_panel;
mod property_panel;
pub mod samples;

use eframe::egui;
pub use state::AppState;

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
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Linkage Simulator");
            ui.label("GUI shell — loading...");
        });
    }
}
