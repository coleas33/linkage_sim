//! GUI module — egui-based visualization shell for the linkage simulator.

mod state;
mod canvas;
pub mod dxf_import;
mod error_panel;
mod export;
mod force_toolbar;
mod input_panel;
mod menu_bar;
mod parametric_panel;
mod plot_panel;
mod property_panel;
pub mod samples;
pub mod sweep;
mod theme;
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
        theme::apply_cad_theme(&cc.egui_ctx);

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
            theme::apply_nathan_mode(ctx);
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

        // ── Drag-and-drop file import (works on native + WASM) ─────────
        // Supports images (background overlay) and DXF (CAD import).
        let dropped_files = ctx.input(|i| i.raw.dropped_files.clone());
        for file in &dropped_files {
            if let Some(bytes) = &file.bytes {
                let name = file.name.as_str();
                let lower = name.to_lowercase();
                if lower.ends_with(".dxf") {
                    // DXF import
                    match dxf_import::parse_dxf_bytes(bytes, 0.001) {
                        Ok(overlay) => {
                            let n_ent = overlay.entities.len();
                            let n_circ = overlay.snap_circles.len();
                            self.state.dxf_overlay = Some(overlay);
                            self.state.status_message = Some(format!(
                                "DXF loaded: {} entities, {} circles (drag & drop). Click 'New Body' in the DXF panel to assign bodies.",
                                n_ent, n_circ
                            ));
                            self.state.status_message_time = 5.0;
                        }
                        Err(e) => {
                            log::error!("DXF import failed: {}", e);
                            self.state.status_message = Some(format!("DXF import failed: {}", e));
                            self.state.status_message_time = 4.0;
                        }
                    }
                } else {
                    // Image import
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

        // Ctrl+V — hint that image paste is not yet supported.
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V)) {
            self.state.status_message =
                Some("Image paste not supported yet \u{2014} drag & drop an image onto the canvas instead.".to_string());
            self.state.status_message_time = 4.0;
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
            let should_advance = self.demo_timer > 5.0;
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
        menu_bar::draw_menu_bar(ctx, &mut self.state, &self.sample_thumbnails);


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
                let tool_color = self.state.nc(egui::Color32::from_rgb(140, 180, 230));
                let tool_active_bg = self.state.nc(egui::Color32::from_rgb(40, 100, 200));
                let tool_active_text = egui::Color32::WHITE;

                let select_active = tool == EditorTool::Select;
                let select_btn = if select_active {
                    egui::Button::new(egui::RichText::new("Select").color(tool_active_text).strong().size(14.0))
                        .fill(tool_active_bg)
                } else {
                    egui::Button::new(egui::RichText::new("Select").color(tool_color))
                };
                if ui.add(select_btn)
                    .on_hover_text("Select mode: click a link, joint, or body to select it. Drag empty space to pan the canvas. Shift+click to multi-select. Press Delete/Backspace to remove the selected entity. (Shortcut: Escape returns here from any tool)")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::Select;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                    self.state.place_mass_body = None;

                }

                let draw_active = tool == EditorTool::DrawLink || self.state.draw_link_start.is_some();
                let draw_btn = if draw_active {
                    egui::Button::new(egui::RichText::new("Draw Link").color(tool_active_text).strong().size(14.0))
                        .fill(tool_active_bg)
                } else {
                    egui::Button::new(egui::RichText::new("Draw Link").color(tool_color))
                };
                if ui.add(draw_btn)
                    .on_hover_text("Draw Link: click an existing attachment point to start, then drag to set link length and direction. Release to place. If you click empty space on the ground, a new ground pivot is created automatically. Creates revolute joints at both ends if connecting to existing points.")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::DrawLink;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                    self.state.place_mass_body = None;

                }

                let is_adding_jp = self.state.adding_joint_point.is_some();
                let jp_btn = if is_adding_jp {
                    egui::Button::new(egui::RichText::new("+ Joint Point").color(tool_active_text).strong().size(14.0))
                        .fill(tool_active_bg)
                } else {
                    egui::Button::new(egui::RichText::new("+ Joint Point").color(tool_color))
                };
                if ui.add(jp_btn)
                    .on_hover_text("Add a new attachment point to the selected body (creates ternary/quaternary shapes)")
                    .clicked()
                {
                    // Use the link editor body, or the selected body, or show a message
                    let target_body = self.state.link_editor_body.clone()
                        .or_else(|| match &self.state.selected {
                            Some(crate::gui::state::SelectedEntity::Body(bid)) => {
                                if bid != GROUND_ID { Some(bid.clone()) } else { None }
                            }
                            _ => None,
                        });
                    if let Some(bid) = target_body {
                        self.state.adding_joint_point = Some(bid);
                        self.state.active_tool = EditorTool::Select;
                        self.state.draw_link_start = None;
                        self.state.add_body_state = None;
                        self.state.place_mass_body = None;
                    } else {
                        self.state.status_message = Some("Select a body first".to_string());
                        self.state.status_message_time = 3.0;
                    }
                }

                // + Body is disabled in the top ribbon — the correct entry
                // point for creating rigid bodies is the Link Editor, which
                // exposes mass, inertia, mount/coupler points, and geometry
                // in one place. The ribbon button is left visible (rather
                // than removed) so its hover text can redirect users who
                // look for it here.
                ui.add_enabled(
                    false,
                    egui::Button::new(egui::RichText::new("+ Body").color(tool_color)),
                )
                .on_hover_text(
                    "To add a rigid body, use the Link Editor in the property panel (right side). Draw links with + Link, then edit mass, points, and geometry there.",
                );

                let ground_active = tool == EditorTool::AddGroundPivot;
                let ground_text = if ground_active {
                    egui::RichText::new("+ Ground").color(tool_active_text).strong().size(14.0)
                } else {
                    egui::RichText::new("+ Ground").color(tool_color)
                };
                let ground_btn = if ground_active {
                    egui::Button::new(ground_text).fill(tool_active_bg)
                } else {
                    egui::Button::new(ground_text)
                };
                if ui.add(ground_btn)
                    .on_hover_text("Click canvas to place a ground pivot")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddGroundPivot;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let mass_active = tool == EditorTool::PlaceMass;
                let mass_text = if mass_active {
                    egui::RichText::new("+ Mass").color(tool_active_text).strong().size(14.0)
                } else {
                    egui::RichText::new("+ Mass").color(tool_color)
                };
                let mass_btn = if mass_active {
                    egui::Button::new(mass_text).fill(tool_active_bg)
                } else {
                    egui::Button::new(mass_text)
                };
                if ui.add(mass_btn)
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
                    egui::Slider::new(&mut self.state.animation_speed_deg_per_sec, 0.5..=720.0)
                        .text("\u{00B0}/s")
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Always),
                ).on_hover_text("Kinematic animation speed in degrees per second");

                // Frame-time diagnostics for debugging slow-speed animation.
                // Only rendered when the Debug Overlay is enabled (View menu).
                // Shows the reported dt, derived FPS, and the per-frame angle
                // step so the user can see what the animation pipeline is
                // actually doing.
                if self.state.show_debug_overlay {
                    let dt = ctx.input(|i| i.stable_dt) as f64;
                    let fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
                    let step_deg = self.state.animation_speed_deg_per_sec
                        * dt
                        * self.state.animation_direction;
                    ui.separator();
                    ui.small(
                        egui::RichText::new(format!(
                            "dt={:.1}ms  fps={:.0}  step={:.3}\u{00B0}/frame",
                            dt * 1000.0,
                            fps,
                            step_deg,
                        ))
                        .color(egui::Color32::from_rgb(150, 150, 170)),
                    ).on_hover_text(
                        "Animation pipeline diagnostics. dt is egui's stable_dt. step is per-frame driver-angle delta.",
                    );
                }

                ui.separator();

                // ── Sample mechanism selector (purple) ──────────────
                let sample_color = self.state.nc(egui::Color32::from_rgb(180, 140, 255));
                let samples_resp = ui.menu_button(
                    egui::RichText::new("Samples v").color(sample_color),
                    |ui| {
                        menu_bar::draw_sample_menu(ui, &mut self.state, &self.sample_thumbnails);
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
                    if self.state.dxf_overlay.is_some() {
                        ui.add_space(20.0);
                        dxf_import::draw_dxf_panel(ui, &mut self.state);
                    }
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
            let is_empty_workspace = !self.state.dismiss_welcome
                && !self.state.tutorial.active
                && !self.demo_mode
                && self.state.current_sample.is_none()
                && self.state.mechanism.as_ref().map_or(true, |m| {
                    m.body_order().is_empty() // no moving bodies
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
                                if ui.button("Load a Sample Mechanism")
                                    .on_hover_text("Load the classic 4-bar linkage sample. You can switch to other samples from the Samples dropdown in the toolbar.")
                                    .clicked()
                                {
                                    self.state.load_sample(SampleMechanism::FourBar);
                                    self.state.pending_fit_to_view = true;
                                }
                                if ui.button("Start Tutorial")
                                    .on_hover_text("Step-by-step interactive tutorial that walks you through building a 4-bar linkage from scratch, adding joints, links, forces, and analyzing the mechanism.")
                                    .clicked()
                                {
                                    self.state.new_empty_mechanism();
                                    self.state.tutorial = tutorial::TutorialState::new_fourbar();
                                }
                                if ui.button("New Empty Mechanism")
                                    .on_hover_text("Start with a blank canvas. You'll be placed in the 'Add Ground Pivot' tool so you can start placing ground attachment points immediately.")
                                    .clicked()
                                {
                                    self.state.new_empty_mechanism();
                                    self.state.dismiss_welcome = true;
                                    self.state.active_tool = state::EditorTool::AddGroundPivot;
                                }
                                ui.add_space(4.0);
                                if ui.button("Watch Demo")
                                    .on_hover_text("Auto-play through all sample mechanisms, cycling every few seconds. Press Escape or click anywhere to stop the demo.")
                                    .clicked()
                                {
                                    self.demo_mode = true;
                                    self.demo_sample_index = 0;
                                    // Load first sample immediately with full init
                                    let sample = SampleMechanism::all()[0];
                                    self.state.load_sample(sample);
                                    self.state.driver_angle = 0.0;
                                    self.state.solve_at_angle(0.0);
                                    self.state.last_good_q = self.state.q.clone();
                                    self.state.q_at_zero = self.state.q.clone();
                                    self.state.playing = true;
                                    self.state.loop_mode = true;
                                    self.state.animation_direction = 1.0;
                                    self.state.pending_fit_to_view = true;
                                    // Start index at 1 so first demo advance loads SliderCrank
                                    self.demo_sample_index = 1;
                                    self.demo_timer = 0.2;
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
                            ui.add(egui::Slider::new(&mut bg.opacity, 0.0..=1.0).fixed_decimals(2))
                                .on_hover_text("Transparency of the background image. 0 = fully transparent, 1 = fully opaque. Lower values make it easier to see mechanism geometry on top of the image.");
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
                            ).on_hover_text("Real-world width of the background image in millimeters. Set this to match the known dimension of an object in the image so that the mechanism overlays at the correct scale.")
                            .changed() {
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
                            ui.label("X:").on_hover_text("Horizontal offset of the image center in world coordinates (meters)");
                            ui.add(egui::DragValue::new(&mut bg.world_offset[0]).speed(0.001).suffix(" m"))
                                .on_hover_text("Horizontal position of the image center in world space. Drag to adjust, or hold Ctrl and drag on the canvas.");
                            ui.label("Y:").on_hover_text("Vertical offset of the image center in world coordinates (meters)");
                            ui.add(egui::DragValue::new(&mut bg.world_offset[1]).speed(0.001).suffix(" m"))
                                .on_hover_text("Vertical position of the image center in world space. Drag to adjust, or hold Ctrl and drag on the canvas.");
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

        // ── "Save as Sample" name dialog ────────────────────────────────
        if self.state.show_custom_sample_dialog {
            let mut open = true;
            egui::Window::new("Save as Sample")
                .collapsible(false)
                .resizable(false)
                .default_width(280.0)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label("Sample name:");
                    let response = ui.text_edit_singleline(&mut self.state.custom_sample_name_buf);
                    // Auto-focus the text field on first frame.
                    if response.gained_focus() || self.state.custom_sample_name_buf.is_empty() {
                        response.request_focus();
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let name_valid = !self.state.custom_sample_name_buf.trim().is_empty();
                        if ui.add_enabled(name_valid, egui::Button::new("Save")).clicked()
                            || (name_valid
                                && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                        {
                            let name = self.state.custom_sample_name_buf.trim().to_string();
                            self.state.save_as_custom_sample(&name);
                            self.state.show_custom_sample_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.state.show_custom_sample_dialog = false;
                        }
                    });
                });
            if !open {
                self.state.show_custom_sample_dialog = false;
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

        // ── DXF "Attach Geometry to Link" picker popup ───────────────────
        dxf_import::draw_geometry_target_dialog(ctx, &mut self.state);

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
