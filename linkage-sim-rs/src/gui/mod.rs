//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
mod error_panel;
mod export;
mod force_toolbar;
mod input_panel;
mod parametric_panel;
mod plot_panel;
mod property_panel;
pub mod samples;
pub mod sweep;
pub mod tutorial;
pub mod undo;

use std::collections::HashMap;

use eframe::egui;
pub use state::AppState;
pub use state::file_io::{decode_mechanism_from_url, encode_mechanism_for_url};
pub use sweep::{SweepData, SweepMode};
use samples::SampleMechanism;
use crate::core::state::GROUND_ID;
use state::{AngleUnit, EditorTool, LengthUnit, PlaceForceState, SelectedEntity};

/// Top-level application struct for eframe.
pub struct LinkageApp {
    state: AppState,
    /// Cached thumbnail textures for the sample mechanism gallery (native only).
    sample_thumbnails: HashMap<SampleMechanism, egui::TextureHandle>,
    /// Whether demo mode is active (auto-cycling through samples).
    demo_mode: bool,
    /// Accumulated time in the current demo sample (seconds).
    demo_timer: f64,
    /// Index into SampleMechanism::all() for the current demo sample.
    demo_sample_index: usize,
}

impl LinkageApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Professional dark theme inspired by CAD tools (SolidWorks, ANSYS)
        let mut visuals = egui::Visuals::dark();

        // Darker, more professional background tones
        visuals.panel_fill = egui::Color32::from_rgb(30, 32, 38);
        visuals.window_fill = egui::Color32::from_rgb(35, 37, 44);
        visuals.extreme_bg_color = egui::Color32::from_rgb(20, 22, 28);
        visuals.faint_bg_color = egui::Color32::from_rgb(38, 40, 48);

        // Accent color for selections and interactions
        visuals.selection.bg_fill = egui::Color32::from_rgb(40, 100, 200);
        visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

        // Widget styling — more rounded, cleaner
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(42, 44, 52);
        visuals.widgets.noninteractive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 62, 72));

        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 52, 62);
        visuals.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(70, 72, 82));

        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 65, 80);
        visuals.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 140, 220));

        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(40, 100, 200);
        visuals.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

        // Separator and window stroke
        visuals.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 58, 68));

        cc.egui_ctx.set_visuals(visuals);

        // Slightly larger default font for readability
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        cc.egui_ctx.set_style(style);

        let sample_thumbnails = generate_sample_thumbnails(&cc.egui_ctx);

        Self {
            state: AppState::default(),
            sample_thumbnails,
            demo_mode: false,
            demo_timer: 0.0,
            demo_sample_index: 0,
        }
    }

    /// Load a mechanism from a shared URL's JSON string.
    ///
    /// Called from the WASM entry point when a `?m=` parameter was decoded.
    pub fn load_shared_mechanism(&mut self, json_str: &str) {
        if let Err(e) = self.state.load_from_json_str(json_str) {
            log::error!("Failed to load shared mechanism: {}", e);
        } else {
            self.state.pending_fit_to_view = true;
            self.state.status_message = Some("Loaded mechanism from shared URL".to_string());
            self.state.status_message_time = 4.0;
        }
    }
}

impl eframe::App for LinkageApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ── Nathan Mode: override visuals to grayscale ──────────────
        if self.state.nathan_mode {
            apply_nathan_mode(ctx);
        }

        // ── Update window title to show filename and dirty state ──────
        let title = if let Some(ref path) = self.state.last_save_path {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if self.state.dirty {
                format!("Linkage Simulator \u{2014} {}*", name)
            } else {
                format!("Linkage Simulator \u{2014} {}", name)
            }
        } else if self.state.dirty {
            "Linkage Simulator \u{2014} unsaved*".to_string()
        } else {
            "Linkage Simulator".to_string()
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));

        // ── Tick status-message timer ─────────────────────────────────
        if self.state.status_message_time > 0.0 {
            self.state.status_message_time -= ctx.input(|i| i.unstable_dt) as f64;
            if self.state.status_message_time <= 0.0 {
                self.state.status_message = None;
                self.state.status_message_time = 0.0;
            }
        }

        // ── Drag-and-drop image import (works on native + WASM) ────────
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        for file in &dropped_files {
            if let Some(bytes) = &file.bytes {
                let name = file
                    .name
                    .as_str();
                let label = if name.is_empty() { "dropped_image" } else { name };
                match load_background_image_from_bytes(ctx, label, bytes) {
                    Ok(bg) => {
                        self.state.background_image = Some(bg);
                        self.state.status_message =
                            Some("Background image loaded (drag & drop)".to_string());
                        self.state.status_message_time = 3.0;
                    }
                    Err(e) => {
                        log::error!("Failed to load dropped image: {}", e);
                        self.state.status_message =
                            Some(format!("Image load failed: {}", e));
                        self.state.status_message_time = 4.0;
                    }
                }
                break; // Only process the first dropped file
            }
        }

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

        // ── Demo mode: auto-cycle through samples ────────────────────
        if self.demo_mode {
            self.demo_timer += dt;
            // Load first sample immediately on demo start, then every 5 seconds
            let should_advance = if self.demo_timer < 0.1 && self.state.mechanism.is_none() {
                true // first frame of demo: load immediately
            } else {
                self.demo_timer > 5.0
            };
            if should_advance {
                self.demo_timer = 0.2; // skip past the first-frame check
                let all = SampleMechanism::all();
                let sample = all[self.demo_sample_index % all.len()];
                self.demo_sample_index = (self.demo_sample_index + 1) % all.len();
                self.state.load_sample(sample);
                self.state.driver_angle = 0.0;
                self.state.solve_at_angle(0.0);
                // Snapshot the solved state so the animation solver has a
                // good initial guess.  Without this, last_good_q may be
                // stale from a previous (differently-dimensioned) mechanism
                // and the solver fails on the first frame, stopping playback.
                self.state.last_good_q = self.state.q.clone();
                self.state.q_at_zero = self.state.q.clone();
                self.state.playing = true;
                self.state.loop_mode = true;
                self.state.animation_direction = 1.0;
                self.state.animation_speed_deg_per_sec = 90.0;
                // Fit to view: set pending for 2 frames to handle layout changes
                self.state.pending_fit_to_view = true;
            }
            // Keep re-triggering fit for the first few frames after a sample loads
            // (canvas rect may change as panels settle)
            if self.demo_timer > 0.2 && self.demo_timer < 0.6 {
                self.state.pending_fit_to_view = true;
            }
            // Escape stops demo mode. Skip click detection for the first 0.5s
            // to avoid the "Watch Demo" button click from immediately stopping it.
            if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.demo_mode = false;
            }
            if self.demo_timer > 0.5 && ctx.input(|i| i.pointer.any_click()) {
                self.demo_mode = false;
            }
            ctx.request_repaint();
        }

        // ── Process pending driver reassignment ──────────────────────
        if let Some(joint_id) = self.state.pending_driver_reassignment.take() {
            self.state.reassign_driver(&joint_id);
        }

        // --- Menu bar ---
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                let file_resp = ui.menu_button("File", |ui| {
                    if ui.button("New  Ctrl+N")
                        .on_hover_text("Create a new empty mechanism (Ctrl+N)")
                        .clicked()
                    {
                        self.state.new_empty_mechanism();
                        ui.close();
                    }
                    let load_sample_resp = ui.menu_button("Load Sample", |ui| {
                        draw_sample_menu(ui, &mut self.state, &self.sample_thumbnails);
                    });
                    load_sample_resp.response.on_hover_text("Load a preset sample mechanism");
                    // ── Templates ─────────────────────────────────────
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.state.blueprint.is_some(),
                            egui::Button::new("Save as Template..."),
                        )
                        .on_hover_text("Save the current mechanism as a reusable template")
                        .clicked()
                    {
                        self.state.show_template_name_dialog = true;
                        self.state.template_name_buf = String::new();
                        ui.close();
                    }
                    if !self.state.saved_templates.is_empty() {
                        let load_tpl_resp = ui.menu_button("Load Template", |ui| {
                            let mut load_idx = None;
                            for (i, (name, _)) in self.state.saved_templates.iter().enumerate() {
                                if ui.button(name).clicked() {
                                    load_idx = Some(i);
                                    ui.close();
                                }
                            }
                            if let Some(idx) = load_idx {
                                self.state.load_template(idx);
                            }
                        });
                        load_tpl_resp.response.on_hover_text("Load a saved mechanism template");

                        let manage_tpl_resp = ui.menu_button("Manage Templates", |ui| {
                            let mut delete_idx = None;
                            for (i, (name, _)) in self.state.saved_templates.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(name);
                                    if ui.small_button("Delete").clicked() {
                                        delete_idx = Some(i);
                                    }
                                });
                            }
                            if let Some(idx) = delete_idx {
                                self.state.delete_template(idx);
                            }
                        });
                        manage_tpl_resp.response.on_hover_text("Delete saved templates");
                    }
                    // ── Share via URL ─────────────────────────────────
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.state.mechanism.is_some(),
                            egui::Button::new("Share via URL"),
                        )
                        .on_hover_text("Copy a shareable URL to the clipboard (loads in the web version)")
                        .clicked()
                    {
                        match self.state.generate_share_url() {
                            Ok(url) => {
                                ui.ctx().copy_text(url.clone());
                                self.state.status_message = Some(format!("Share URL copied ({} chars)", url.len()));
                                self.state.status_message_time = 4.0;
                            }
                            Err(e) => {
                                log::error!("Share URL generation failed: {}", e);
                                self.state.status_message = Some(format!("Share failed: {}", e));
                                self.state.status_message_time = 4.0;
                            }
                        }
                        ui.close();
                    }
                    // ── Native-only file dialogs ──────────────────────
                    #[cfg(feature = "native")]
                    {
                        ui.separator();
                        if ui.button("Open JSON...")
                            .on_hover_text("Load a mechanism from a JSON file")
                            .clicked()
                        {
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
                            let recent_resp = ui.menu_button("Recent Files", |ui| {
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
                            recent_resp.response.on_hover_text("Recently opened mechanism files");
                        }
                        if ui.button("Save  Ctrl+S")
                            .on_hover_text("Save mechanism to the current file (Ctrl+S)")
                            .clicked()
                        {
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
                        if ui.button("Save As...  Ctrl+Shift+S")
                            .on_hover_text("Save mechanism to a new JSON file (Ctrl+Shift+S)")
                            .clicked()
                        {
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
                            .on_hover_text("Export sweep data (angles, torques, reactions) to CSV")
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
                            .on_hover_text("Export coupler point trace coordinates to CSV")
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
                            .on_hover_text("Export mechanism as a scalable vector graphic")
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
                            .on_hover_text("Export mechanism as a PNG image (1920x1080)")
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
                            .on_hover_text("Export an animated GIF of the full crank cycle")
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
                            .on_hover_text("Export mechanism geometry as a DXF drawing file")
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
                            .on_hover_text("Generate an HTML report with plots and analysis summary")
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
                    if ui.button("Quit").on_hover_text("Close the application").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                file_resp.response.on_hover_text("File operations: new, open, save, export");
                let edit_resp = ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(self.state.can_undo(), egui::Button::new("\u{21A9} Undo  Ctrl+Z"))
                        .on_hover_text("Undo the last change (Ctrl+Z)")
                        .clicked()
                    {
                        self.state.undo();
                        ui.close();
                    }
                    if ui
                        .add_enabled(self.state.can_redo(), egui::Button::new("\u{21AA} Redo  Ctrl+Y"))
                        .on_hover_text("Redo the last undone change (Ctrl+Y)")
                        .clicked()
                    {
                        self.state.redo();
                        ui.close();
                    }
                });
                edit_resp.response.on_hover_text("Undo, redo, and editing operations");
                let help_resp = ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts")
                        .on_hover_text("Show all keyboard shortcuts")
                        .clicked()
                    {
                        self.state.show_shortcuts = true;
                        ui.close();
                    }
                    if ui.button("Tutorial: Build a 4-Bar")
                        .on_hover_text("Interactive step-by-step tutorial for building a 4-bar linkage")
                        .clicked()
                    {
                        self.state.new_empty_mechanism();
                        self.state.tutorial = tutorial::TutorialState::new_fourbar();
                        ui.close();
                    }
                });
                help_resp.response.on_hover_text("Keyboard shortcuts and help");
                let view_resp = ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay")
                        .on_hover_text("Show solver status, body IDs, and attachment point names on the canvas");
                    ui.checkbox(&mut self.state.show_plots, "Plot Panel")
                        .on_hover_text("Show/hide the sweep data plot panel at the bottom");
                    ui.add_enabled(false, egui::Checkbox::new(&mut self.state.show_parametric, "Parametric Study [WIP]"))
                        .on_hover_text("Parametric study panel (coming soon)");
                    ui.checkbox(&mut self.state.show_forces, "Force Arrows")
                        .on_hover_text("Show/hide joint reaction force arrows and force element visuals on the canvas");
                    ui.checkbox(&mut self.state.show_dimensions, "Link Dimensions")
                        .on_hover_text("Show/hide link length dimensions on the canvas");
                    ui.checkbox(&mut self.state.show_labels, "Show Labels")
                        .on_hover_text("Show/hide body and joint labels on the canvas");
                    let enabled = self.state.gravity_magnitude > 0.0;
                    let mut check = enabled;
                    if ui.checkbox(&mut check, "Gravity")
                        .on_hover_text("Enable/disable gravitational acceleration (9.81 m/s\u{00b2})")
                        .changed()
                    {
                        self.state.gravity_magnitude = if check { 9.81 } else { 0.0 };
                        self.state.mark_sweep_dirty();
                    }
                    ui.separator();
                    ui.label("Units:");
                    let mut use_mm = self.state.display_units.length == LengthUnit::Millimeters;
                    if ui.checkbox(&mut use_mm, "Millimeters")
                        .on_hover_text("Display lengths in millimeters instead of meters")
                        .changed()
                    {
                        self.state.display_units.length = if use_mm {
                            LengthUnit::Millimeters
                        } else {
                            LengthUnit::Meters
                        };
                    }
                    let mut use_deg = self.state.display_units.angle == AngleUnit::Degrees;
                    if ui.checkbox(&mut use_deg, "Degrees")
                        .on_hover_text("Display angles in degrees instead of radians")
                        .changed()
                    {
                        self.state.display_units.angle = if use_deg {
                            AngleUnit::Degrees
                        } else {
                            AngleUnit::Radians
                        };
                    }
                    ui.separator();
                    ui.label("Grid:");
                    ui.checkbox(&mut self.state.grid.show_grid, "Show Grid")
                        .on_hover_text("Show/hide the background grid on the canvas");
                    ui.checkbox(&mut self.state.grid.snap_enabled, "Snap to Grid")
                        .on_hover_text("Snap placed points to the nearest grid intersection");
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
                    ui.checkbox(&mut self.state.show_load_path, "Load Path (heat map)")
                        .on_hover_text("Color-code links by joint reaction force magnitude (blue=low, red=high)");
                    if ui.checkbox(&mut self.state.nathan_mode, "Nathan Mode")
                        .on_hover_text("Toggle grayscale mode")
                        .changed() && !self.state.nathan_mode
                    {
                        // Restore normal visuals when turning off.
                        restore_normal_visuals(ctx);
                    }
                });
                view_resp.response.on_hover_text("Toggle display options and visualization settings");

                // ── Image menu ──────────────────────────────────────────
                let image_resp = ui.menu_button("Image", |ui| {
                    #[cfg(feature = "native")]
                    {
                        if ui.button("Import Image...")
                            .on_hover_text("Load a photo or sketch as a canvas background for tracing")
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                                .pick_file()
                            {
                                match load_background_image(ctx, &path) {
                                    Ok(bg) => {
                                        self.state.background_image = Some(bg);
                                        self.state.status_message = Some("Background image loaded".to_string());
                                        self.state.status_message_time = 3.0;
                                    }
                                    Err(e) => {
                                        log::error!("Failed to load background image: {}", e);
                                        self.state.status_message = Some(format!("Image load failed: {}", e));
                                        self.state.status_message_time = 4.0;
                                    }
                                }
                            }
                            ui.close();
                        }
                    }
                    #[cfg(not(feature = "native"))]
                    {
                        ui.label(
                            egui::RichText::new("Drag & drop an image onto the canvas")
                                .small().weak(),
                        );
                    }
                    if self.state.background_image.is_some() {
                        if ui.button("Remove Image")
                            .on_hover_text("Remove the background image")
                            .clicked()
                        {
                            self.state.background_image = None;
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Image Settings...")
                            .on_hover_text("Open image controls (opacity, size, position)")
                            .clicked()
                        {
                            self.state.show_image_settings = true;
                            ui.close();
                        }
                    }
                });
                image_resp.response.on_hover_text("Background image import and controls");
            });
        });

        // ── Delete / Backspace shortcut ───────────────────────────────────
        if ctx.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)) {
            if !self.state.multi_selected.is_empty() {
                // Delete all multi-selected items.
                let items: Vec<_> = self.state.multi_selected.drain(..).collect();
                for entity in items {
                    match entity {
                        SelectedEntity::Body(id) => self.state.remove_body(&id),
                        SelectedEntity::Joint(id) => self.state.remove_joint(&id),
                        _ => {}
                    }
                }
                self.state.selected = None;
            } else {
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
        }

        // --- Toolbar ---
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().button_padding = egui::vec2(10.0, 5.0);

                let tool = self.state.active_tool;

                // ── Editor tools (blue accent) ──────────────────────
                let tool_color = self.state.nc(egui::Color32::from_rgb(80, 160, 255));
                let tool_active_color = self.state.nc(egui::Color32::from_rgb(40, 120, 220));

                let select_text = if tool == EditorTool::Select {
                    egui::RichText::new("Select").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("Select").color(tool_color)
                };
                if ui.add(egui::Button::new(select_text))
                    .on_hover_text("Select entities on the canvas")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::Select;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                    self.state.place_mass_body = None;
                }

                let draw_active = tool == EditorTool::DrawLink || self.state.draw_link_start.is_some();
                let draw_text = if draw_active {
                    egui::RichText::new("Draw Link").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("Draw Link").color(tool_color)
                };
                if ui.add(egui::Button::new(draw_text))
                    .on_hover_text("Click and drag to draw a link")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::DrawLink;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                    self.state.place_mass_body = None;
                }

                let body_text = egui::RichText::new("+ Body [WIP]").color(
                    self.state.nc(egui::Color32::from_rgb(120, 120, 120))
                );
                ui.add_enabled(false, egui::Button::new(body_text))
                    .on_hover_text("Multi-point body creation (coming soon — use Draw Link for bars)");
                if false {
                    // Disabled — WIP. Original handler preserved for future use.
                    self.state.active_tool = EditorTool::AddBody;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let ground_text = if tool == EditorTool::AddGroundPivot {
                    egui::RichText::new("+ Ground").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("+ Ground").color(tool_color)
                };
                if ui.add(egui::Button::new(ground_text))
                    .on_hover_text("Click canvas to place a ground pivot")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddGroundPivot;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let mass_text = if tool == EditorTool::PlaceMass {
                    egui::RichText::new("+ Mass").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("+ Mass").color(tool_color)
                };
                if ui.add(egui::Button::new(mass_text))
                    .on_hover_text("Place a point mass on a body")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::PlaceMass;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                    // If a body is already selected, skip phase 1
                    if let Some(SelectedEntity::Body(ref id)) = self.state.selected {
                        if id != GROUND_ID {
                            self.state.place_mass_body = Some(id.clone());
                        } else {
                            self.state.place_mass_body = None;
                        }
                    } else {
                        self.state.place_mass_body = None;
                    }
                }

                ui.separator();

                // ── Playback controls (green/yellow) ────────────────
                let is_playing = self.state.playing;
                let (label, color) = if is_playing {
                    ("Pause", self.state.nc(egui::Color32::from_rgb(240, 200, 60)))
                } else {
                    ("\u{25B6}  Play", self.state.nc(egui::Color32::from_rgb(60, 220, 90)))
                };
                if ui.add(egui::Button::new(
                    egui::RichText::new(label).color(color).strong().size(14.0)
                ))
                    .on_hover_text("Animate the mechanism (kinematic playback)")
                    .clicked()
                {
                    self.state.playing = !self.state.playing;
                    if self.state.playing {
                        // Stop simulation playback — only one can drive the canvas.
                        if let Some(sim) = &mut self.state.simulation {
                            sim.playing = false;
                        }
                        if !self.state.loop_mode {
                            self.state.animation_direction = 1.0;
                        }
                    }
                }

                // Speed control (compact)
                ui.add(
                    egui::Slider::new(&mut self.state.animation_speed_deg_per_sec, 10.0..=720.0)
                        .text("\u{00B0}/s")
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Always),
                ).on_hover_text("Kinematic animation speed in degrees per second");

                ui.separator();

                // ── Sample mechanism selector (purple) ──────────────
                let sample_color = self.state.nc(egui::Color32::from_rgb(180, 140, 255));
                let samples_resp = ui.menu_button(
                    egui::RichText::new("Samples v").color(sample_color),
                    |ui| {
                        draw_sample_menu(ui, &mut self.state, &self.sample_thumbnails);
                    },
                );
                samples_resp.response.on_hover_text("Load a preset sample mechanism");
            });
        });

        // --- Force element toolbar ribbon ---
        egui::TopBottomPanel::top("force_toolbar").show(ctx, |ui| {
            if let Some(force_add) = force_toolbar::draw_force_toolbar(ui, &self.state) {
                match force_add {
                    force_toolbar::PendingForceAdd::Add(force) => {
                        self.state.add_force_element(force);
                    }
                    force_toolbar::PendingForceAdd::EnterPlaceMode(template) => {
                        self.state.active_tool = EditorTool::PlaceForce;
                        self.state.place_force_state = Some(PlaceForceState {
                            force_template: template,
                            start: None,
                        });
                    }
                    force_toolbar::PendingForceAdd::EnterForceZoneMode => {
                        self.state.active_tool = EditorTool::CreateForceZone;
                        self.state.creating_force_zone = None;
                        // Actual drag state gets set on mouse press in canvas.rs
                    }
                }
            }
        });

        // --- Status bar ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let dim = self.state.nc(egui::Color32::from_rgb(140, 145, 160));
                let bright = self.state.nc(egui::Color32::from_rgb(200, 205, 220));
                let green = self.state.nc(egui::Color32::from_rgb(80, 200, 80));
                let red = self.state.nc(egui::Color32::from_rgb(220, 70, 70));
                let blue = self.state.nc(egui::Color32::from_rgb(100, 180, 255));
                let warn = self.state.nc(egui::Color32::from_rgb(255, 180, 50));

                if let Some(sample) = self.state.current_sample {
                    ui.colored_label(bright, sample.label());
                    ui.colored_label(dim, "\u{2502}");
                }

                if self.state.has_mechanism() {
                    // Solver status
                    let status = &self.state.solver_status;
                    if status.converged {
                        ui.colored_label(green, "\u{25CF}");
                    } else {
                        ui.colored_label(red, "\u{25CF} FAIL");
                    }
                    ui.colored_label(dim, "\u{2502}");

                    // Angle + torque
                    ui.colored_label(dim, "\u{03b8}");
                    ui.colored_label(bright, format!(
                        "{:.1}{}",
                        self.state.display_units.angle(self.state.driver_angle),
                        self.state.display_units.angle_suffix()
                    ));

                    if let Some(torque) = self.state.force_results.driver_torque {
                        ui.colored_label(dim, "\u{03c4}");
                        ui.colored_label(bright, format!("{:.3} N\u{00b7}m", torque));
                    }
                    ui.colored_label(dim, "\u{2502}");

                    // Mechanism info
                    if let Some(mech) = &self.state.mechanism {
                        let n_b = mech.bodies().len().saturating_sub(1);
                        let n_j = mech.joints().len();
                        let dof = mech.state().n_coords() as isize - mech.n_constraints() as isize;
                        ui.colored_label(dim, format!("{}B {}J DOF={}", n_b, n_j, dof));
                    }

                    // Playback/sim state
                    if self.state.playing {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(green, "\u{25B6} PLAYING");
                    }
                    if let Some(sim) = &self.state.simulation {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(blue, format!(
                            "SIM t={:.2}s",
                            sim.times.get(sim.time_index).unwrap_or(&0.0)
                        ));
                    }

                    // Warnings
                    let warnings = &self.state.validation_warnings;
                    if let Some(ref dof_msg) = warnings.dof_warning {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(warn, dof_msg);
                    }
                    if warnings.missing_driver {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(warn, "! No driver");
                    }
                    if !warnings.disconnected_bodies.is_empty() {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(warn, format!(
                            "! Disconnected: {}",
                            warnings.disconnected_bodies.join(", ")
                        ));
                    }

                    if !self.state.error_log.is_empty() {
                        ui.separator();
                        let label = format!("{} error(s)", self.state.error_log.len());
                        if ui
                            .colored_label(self.state.nc(egui::Color32::from_rgb(220, 80, 80)), &label)
                            .on_hover_text("Click to show error panel")
                            .clicked()
                        {
                            self.state.show_error_panel = !self.state.show_error_panel;
                        }
                    }
                } else {
                    ui.label("No mechanism loaded");
                }

                // ── Status toast (e.g. "Saved: foo.json") ──────────────
                if let Some(ref msg) = self.state.status_message {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.colored_label(self.state.nc(egui::Color32::from_rgb(100, 220, 100)), msg);
                    });
                }
            });
        });

        // --- Bottom panel: plots ---
        if self.state.show_plots {
            egui::TopBottomPanel::bottom("plot_panel")
                .resizable(true)
                .default_height(250.0)
                .show(ctx, |ui| {
                    plot_panel::draw_plot_panel(ui, &mut self.state);
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
            .resizable(true)
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
            // Determine if the workspace is effectively empty (just ground body,
            // no real mechanism content) and no tutorial is active.
            let is_empty_workspace = !self.state.tutorial.active
                && self.state.current_sample.is_none()
                && !self.demo_mode
                && self.state.blueprint.as_ref().map_or(true, |bp| {
                    bp.bodies.len() <= 1 && bp.joints.is_empty()
                });

            if is_empty_workspace {
                // ── Welcome screen ──────────────────────────────────────
                let center = ui.min_rect().center();
                egui::Area::new(egui::Id::new("welcome_screen"))
                    .fixed_pos(egui::pos2(
                        center.x - 160.0,
                        center.y - 120.0,
                    ))
                    .show(ui.ctx(), |ui| {
                        egui::Frame::popup(ui.ctx().style().as_ref()).show(ui, |ui| {
                            ui.set_min_width(300.0);
                            ui.vertical_centered(|ui| {
                                ui.heading("Linkage Simulator");
                                ui.add_space(10.0);
                                ui.label("Get started:");
                                ui.add_space(6.0);
                                if ui.button("Load a Sample Mechanism").clicked() {
                                    self.state.load_sample(SampleMechanism::FourBar);
                                    self.state.pending_fit_to_view = true;
                                }
                                if ui.button("Start Tutorial").clicked() {
                                    self.state.new_empty_mechanism();
                                    self.state.tutorial = tutorial::TutorialState::new_fourbar();
                                }
                                if ui.button("New Empty Mechanism").clicked() {
                                    self.state.new_empty_mechanism();
                                    // Switch to AddGroundPivot so the user can start placing immediately.
                                    self.state.active_tool = state::EditorTool::AddGroundPivot;
                                }
                                ui.add_space(4.0);
                                if ui.button("Watch Demo").clicked() {
                                    self.demo_mode = true;
                                    self.demo_timer = 0.0;
                                    self.demo_sample_index = 0;
                                    let sample = SampleMechanism::all()[0];
                                    self.state.load_sample(sample);
                                    self.state.playing = true;
                                    self.state.pending_fit_to_view = true;
                                }
                                ui.add_space(5.0);
                                ui.label(
                                    egui::RichText::new("Or drag & drop a JSON file to open")
                                        .small()
                                        .weak(),
                                );
                            });
                        });
                    });
            } else {
                canvas::draw_canvas(ui, &mut self.state);
            }
        });

        // ── Demo mode banner overlay ────────────────────────────────────
        if self.demo_mode {
            egui::Area::new(egui::Id::new("demo_banner"))
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 8.0))
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.ctx().style().as_ref())
                        .fill(egui::Color32::from_rgba_premultiplied(30, 30, 30, 200))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("Demo Mode -- press Escape to stop")
                                    .color(egui::Color32::from_rgb(255, 220, 100))
                                    .size(14.0),
                            );
                        });
                });
        }

        // ── Image Settings floating window ───────────────────────────
        // Shown when the user clicks "Image Settings..." in the Image menu.
        // Uses a persistent window so +/- buttons and sliders don't close on click.
        if self.state.show_image_settings && self.state.background_image.is_some() {
            egui::Window::new("Image Settings")
                .open(&mut self.state.show_image_settings)
                .resizable(false)
                .default_width(260.0)
                .show(ctx, |ui| {
                    if let Some(ref mut bg) = self.state.background_image {
                        ui.horizontal(|ui| {
                            ui.label("Opacity:");
                            ui.add(egui::Slider::new(&mut bg.opacity, 0.0..=1.0).fixed_decimals(2));
                        });
                        let img_width_m = bg.size_px[0] as f64 / bg.scale_px_per_m;
                        let mut img_width_mm = img_width_m * 1000.0;
                        ui.horizontal(|ui| {
                            ui.label("Width:");
                            if ui.add(
                                egui::DragValue::new(&mut img_width_mm)
                                    .speed(1.0)
                                    .range(1.0..=100000.0)
                                    .suffix(" mm")
                            ).changed() {
                                bg.scale_px_per_m = bg.size_px[0] as f64 / (img_width_mm / 1000.0);
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Resize:");
                            if ui.button("Shrink 10%").on_hover_text("Decrease image size by 10%").clicked() {
                                bg.scale_px_per_m *= 1.1;
                            }
                            if ui.button("Grow 10%").on_hover_text("Increase image size by 10%").clicked() {
                                bg.scale_px_per_m *= 0.9;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("X:");
                            ui.add(egui::DragValue::new(&mut bg.world_offset[0]).speed(0.001).suffix(" m"));
                            ui.label("Y:");
                            ui.add(egui::DragValue::new(&mut bg.world_offset[1]).speed(0.001).suffix(" m"));
                        });
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("Tip: Hold Ctrl and drag on canvas to move image")
                                .small()
                                .weak(),
                        );
                    }
                });
        }
        // Close image settings if the image was removed.
        if self.state.background_image.is_none() {
            self.state.show_image_settings = false;
        }

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
                        if ui.button("Recover")
                            .on_hover_text("Load the autosaved mechanism from last session")
                            .clicked()
                        {
                            load = true;
                        }
                        if ui.button("Discard")
                            .on_hover_text("Delete the autosave file and start fresh")
                            .clicked()
                        {
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

        // ── WASM autosave recovery prompt ────────────────────────────────
        #[cfg(target_arch = "wasm32")]
        if self.state.wasm_has_recovery {
            let mut dismiss = false;
            let mut load = false;
            egui::Window::new("Recover Unsaved Work?")
                .collapsible(false)
                .resizable(false)
                .default_width(340.0)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("An autosave was found from a previous session.");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Recover")
                            .on_hover_text("Load the autosaved mechanism from last session")
                            .clicked()
                        {
                            load = true;
                        }
                        if ui.button("Discard")
                            .on_hover_text("Clear the autosave and start fresh")
                            .clicked()
                        {
                            dismiss = true;
                        }
                    });
                });
            if load {
                self.state.wasm_has_recovery = false;
                if let Some(json_str) = AppState::wasm_load_autosave() {
                    if let Err(e) = self.state.load_from_json_str(&json_str) {
                        log::error!("Failed to recover WASM autosave: {}", e);
                    }
                }
                // Clean up after loading.
                AppState::wasm_clear_autosave();
            } else if dismiss {
                self.state.wasm_has_recovery = false;
                AppState::wasm_clear_autosave();
            }
        }

        // ── Keyboard shortcuts window ────────────────────────────────────
        // ── "Save as Template" name dialog ──────────────────────────────
        if self.state.show_template_name_dialog {
            let mut open = true;
            egui::Window::new("Save as Template")
                .collapsible(false)
                .resizable(false)
                .default_width(280.0)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label("Template name:");
                    let response = ui.text_edit_singleline(&mut self.state.template_name_buf);
                    // Auto-focus the text field on first frame.
                    if response.gained_focus() || self.state.template_name_buf.is_empty() {
                        response.request_focus();
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let name_valid = !self.state.template_name_buf.trim().is_empty();
                        if ui.add_enabled(name_valid, egui::Button::new("Save")).clicked()
                            || (name_valid
                                && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            let name = self.state.template_name_buf.trim().to_string();
                            self.state.save_as_template(&name);
                            self.state.show_template_name_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.state.show_template_name_dialog = false;
                        }
                    });
                });
            if !open {
                self.state.show_template_name_dialog = false;
            }
        }

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

        // ── Tutorial overlay ─────────────────────────────────────────────
        if self.state.tutorial.active {
            tutorial::draw_tutorial_overlay(ctx, &mut self.state);
        }

        // ── Autosave tick ────────────────────────────────────────────────
        #[cfg(feature = "native")]
        self.state.tick_autosave(dt);
        #[cfg(target_arch = "wasm32")]
        self.state.tick_autosave(dt);
    }
}

// ── Nathan Mode helpers ─────────────────────────────────────────────────────

/// Convert a color to grayscale using luminance weights.
fn gray(c: egui::Color32) -> egui::Color32 {
    let lum = (c.r() as f32 * 0.299 + c.g() as f32 * 0.587 + c.b() as f32 * 0.114) as u8;
    egui::Color32::from_rgba_premultiplied(lum, lum, lum, c.a())
}

/// Override egui visuals with grayscale colors (Nathan Mode).
fn apply_nathan_mode(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();

    v.widgets.noninteractive.bg_fill = gray(v.widgets.noninteractive.bg_fill);
    v.widgets.noninteractive.fg_stroke.color = gray(v.widgets.noninteractive.fg_stroke.color);
    v.widgets.inactive.bg_fill = gray(v.widgets.inactive.bg_fill);
    v.widgets.inactive.fg_stroke.color = gray(v.widgets.inactive.fg_stroke.color);
    v.widgets.hovered.bg_fill = gray(v.widgets.hovered.bg_fill);
    v.widgets.hovered.fg_stroke.color = gray(v.widgets.hovered.fg_stroke.color);
    v.widgets.active.bg_fill = gray(v.widgets.active.bg_fill);
    v.widgets.active.fg_stroke.color = gray(v.widgets.active.fg_stroke.color);
    v.widgets.open.bg_fill = gray(v.widgets.open.bg_fill);
    v.widgets.open.fg_stroke.color = gray(v.widgets.open.fg_stroke.color);
    v.selection.bg_fill = gray(v.selection.bg_fill);
    v.selection.stroke.color = gray(v.selection.stroke.color);
    v.hyperlink_color = gray(v.hyperlink_color);
    v.window_fill = gray(v.window_fill);
    v.panel_fill = gray(v.panel_fill);
    v.extreme_bg_color = gray(v.extreme_bg_color);

    ctx.set_visuals(v);
}

/// Restore the custom CAD-inspired dark visuals (mirrors `LinkageApp::new`).
fn restore_normal_visuals(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();

    v.panel_fill = egui::Color32::from_rgb(30, 32, 38);
    v.window_fill = egui::Color32::from_rgb(35, 37, 44);
    v.extreme_bg_color = egui::Color32::from_rgb(20, 22, 28);
    v.faint_bg_color = egui::Color32::from_rgb(38, 40, 48);

    v.selection.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(42, 44, 52);
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 62, 72));

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 52, 62);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(70, 72, 82));

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 65, 80);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 140, 220));

    v.widgets.active.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 58, 68));

    ctx.set_visuals(v);
}

// ── Sample thumbnail generation ─────────────────────────────────────────────

/// Generate thumbnail textures for all sample mechanisms.
///
/// Builds each sample, renders it as SVG, rasterizes to RGBA via resvg, and
/// uploads to the egui texture manager. Samples that fail to render are silently
/// skipped (the gallery will show text-only for those entries).
///
/// On WASM (no `native` feature), returns an empty map — thumbnails are not
/// available without resvg.
#[cfg(feature = "native")]
fn generate_sample_thumbnails(
    ctx: &egui::Context,
) -> HashMap<SampleMechanism, egui::TextureHandle> {
    use samples::build_sample;
    use export::{generate_svg_string, rasterize_svg_to_rgba};

    const THUMB_W: u32 = 120;
    const THUMB_H: u32 = 80;

    let mut thumbnails = HashMap::new();
    for sample in SampleMechanism::all() {
        let (mech, q0) = build_sample(*sample);
        let svg = match generate_svg_string(&mech, &q0) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rgba = match rasterize_svg_to_rgba(&svg, THUMB_W, THUMB_H) {
            Ok(buf) => buf,
            Err(_) => continue,
        };
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [THUMB_W as usize, THUMB_H as usize],
            &rgba,
        );
        let texture = ctx.load_texture(
            format!("sample_thumb_{:?}", sample),
            image,
            egui::TextureOptions::LINEAR,
        );
        thumbnails.insert(*sample, texture);
    }
    thumbnails
}

#[cfg(not(feature = "native"))]
fn generate_sample_thumbnails(
    _ctx: &egui::Context,
) -> HashMap<SampleMechanism, egui::TextureHandle> {
    HashMap::new()
}

// ── Sample gallery menu ─────────────────────────────────────────────────────

/// Draw the sample mechanism menu with categories, thumbnails, and descriptions.
///
/// Samples are grouped by category (4-bar, 6-bar, specialty) with separator
/// headers. Each sample shows a small thumbnail (when available) alongside its
/// label, with a tooltip containing a brief description.
fn draw_sample_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    thumbnails: &HashMap<SampleMechanism, egui::TextureHandle>,
) {
    let mut last_category: Option<&str> = None;
    for sample in SampleMechanism::all() {
        let cat = sample.category();
        if last_category != Some(cat) {
            if last_category.is_some() {
                ui.separator();
            }
            ui.label(egui::RichText::new(cat).small().weak());
            last_category = Some(cat);
        }
        let clicked = ui
            .horizontal(|ui| {
                if let Some(tex) = thumbnails.get(sample) {
                    ui.image(egui::load::SizedTexture::new(
                        tex.id(),
                        egui::vec2(60.0, 40.0),
                    ));
                } else {
                    // Show a colored category badge when no thumbnail is available
                    let badge_color = match cat {
                        "4-Bar Mechanisms" => egui::Color32::from_rgb(80, 160, 255),
                        "6-Bar Mechanisms" => egui::Color32::from_rgb(100, 220, 140),
                        _ => egui::Color32::from_rgb(255, 165, 80),
                    };
                    let badge_text = match cat {
                        "4-Bar Mechanisms" => "[4B]",
                        "6-Bar Mechanisms" => "[6B]",
                        _ => "[SP]",
                    };
                    ui.colored_label(badge_color, badge_text);
                }
                ui.vertical(|ui| {
                    let btn_clicked = ui.button(sample.label())
                        .on_hover_text(sample.description())
                        .clicked();
                    ui.label(
                        egui::RichText::new(sample.description())
                            .small()
                            .weak(),
                    );
                    btn_clicked
                })
                .inner
            })
            .inner;
        if clicked {
            state.load_sample(*sample);
            ui.close();
        }
    }
}

// ── Background image loading ────────────────────────────────────────────────

/// Decode an in-memory image (PNG/JPEG) and create an egui texture from it.
///
/// Uses the `image` crate to decode, converts to RGBA, and uploads to the egui
/// texture manager. Returns a `BackgroundImage` with sensible defaults
/// (centered at origin, 1000 px/m scale, 30% opacity).
///
/// This works on both native and WASM targets, since the `image` crate is
/// a pure-Rust dependency available everywhere.
fn load_background_image_from_bytes(
    ctx: &egui::Context,
    name: &str,
    bytes: &[u8],
) -> Result<state::BackgroundImage, String> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("Image decode error: {}", e))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], rgba.as_raw());
    let texture = ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR);
    Ok(state::BackgroundImage {
        texture,
        world_offset: [0.0, 0.0],
        scale_px_per_m: 1000.0, // default: 1000 px = 1 m
        opacity: 0.3,
        size_px: [w, h],
    })
}

/// Load an image file from disk and create an egui texture from it.
///
/// Reads the file and delegates to [`load_background_image_from_bytes`].
#[cfg(feature = "native")]
fn load_background_image(
    ctx: &egui::Context,
    path: &std::path::Path,
) -> Result<state::BackgroundImage, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("File read error: {}", e))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "background".to_string());
    load_background_image_from_bytes(ctx, &name, &bytes)
}
