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
pub mod undo;

use eframe::egui;
pub use state::AppState;
pub use sweep::SweepData;
use samples::SampleMechanism;
use state::{AngleUnit, EditorTool, LengthUnit, SelectedEntity};

/// Top-level application struct for eframe.
pub struct LinkageApp {
    state: AppState,
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
                ui.menu_button("\u{1F4C1} File", |ui| {
                    if ui.button("\u{1F4C4} New  Ctrl+N").clicked() {
                        self.state.new_empty_mechanism();
                        ui.close();
                    }
                    ui.menu_button("\u{1F4C2} Load Sample", |ui| {
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
                        if ui.button("\u{1F4C2} Open JSON...").clicked() {
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
                        if ui.button("\u{1F4BE} Save  Ctrl+S").clicked() {
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
                        if ui.button("\u{1F4BE} Save As...  Ctrl+Shift+S").clicked() {
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
                ui.menu_button("\u{270F} Edit", |ui| {
                    if ui
                        .add_enabled(self.state.can_undo(), egui::Button::new("\u{21A9} Undo  Ctrl+Z"))
                        .clicked()
                    {
                        self.state.undo();
                        ui.close();
                    }
                    if ui
                        .add_enabled(self.state.can_redo(), egui::Button::new("\u{21AA} Redo  Ctrl+Y"))
                        .clicked()
                    {
                        self.state.redo();
                        ui.close();
                    }
                });
                ui.menu_button("\u{2753} Help", |ui| {
                    if ui.button("\u{2328} Keyboard Shortcuts").clicked() {
                        self.state.show_shortcuts = true;
                        ui.close();
                    }
                });
                ui.menu_button("\u{1F441} View", |ui| {
                    ui.checkbox(&mut self.state.show_debug_overlay, "Debug Overlay");
                    ui.checkbox(&mut self.state.show_plots, "Plot Panel");
                    ui.checkbox(&mut self.state.show_parametric, "Parametric Study");
                    ui.checkbox(&mut self.state.show_forces, "Force Arrows");
                    ui.checkbox(&mut self.state.show_dimensions, "Link Dimensions");
                    let enabled = self.state.gravity_magnitude > 0.0;
                    let mut check = enabled;
                    if ui.checkbox(&mut check, "Gravity").changed() {
                        self.state.gravity_magnitude = if check { 9.81 } else { 0.0 };
                        self.state.mark_sweep_dirty();
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
                ui.spacing_mut().button_padding = egui::vec2(10.0, 5.0);

                let tool = self.state.active_tool;

                // ── Editor tools (blue accent) ──────────────────────
                let tool_color = egui::Color32::from_rgb(80, 160, 255);
                let tool_active_color = egui::Color32::from_rgb(40, 120, 220);

                let select_text = if tool == EditorTool::Select {
                    egui::RichText::new("\u{1F5B1} Select").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("\u{1F5B1} Select").color(tool_color)
                };
                if ui.add(egui::Button::new(select_text))
                    .on_hover_text("Select entities on the canvas")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::Select;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let draw_active = tool == EditorTool::DrawLink || self.state.draw_link_start.is_some();
                let draw_text = if draw_active {
                    egui::RichText::new("\u{270F} Draw Link").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("\u{270F} Draw Link").color(tool_color)
                };
                if ui.add(egui::Button::new(draw_text))
                    .on_hover_text("Click and drag to draw a link")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::DrawLink;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let body_active = tool == EditorTool::AddBody || self.state.add_body_state.is_some();
                let body_text = if body_active {
                    egui::RichText::new("\u{2795} Body").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("\u{2795} Body").color(tool_color)
                };
                if ui.add(egui::Button::new(body_text))
                    .on_hover_text("Click to place points, double-click to finish")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddBody;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                let ground_text = if tool == EditorTool::AddGroundPivot {
                    egui::RichText::new("\u{2693} Ground").color(tool_active_color).strong()
                } else {
                    egui::RichText::new("\u{2693} Ground").color(tool_color)
                };
                if ui.add(egui::Button::new(ground_text))
                    .on_hover_text("Click canvas to place a ground pivot")
                    .clicked()
                {
                    self.state.active_tool = EditorTool::AddGroundPivot;
                    self.state.draw_link_start = None;
                    self.state.add_body_state = None;
                }

                ui.separator();

                // ── Playback controls (green/yellow) ────────────────
                let is_playing = self.state.playing;
                let (label, color) = if is_playing {
                    ("\u{23F8}  Pause", egui::Color32::from_rgb(240, 200, 60))
                } else {
                    ("\u{25B6}  Play", egui::Color32::from_rgb(60, 220, 90))
                };
                if ui.add(egui::Button::new(
                    egui::RichText::new(label).color(color).strong().size(14.0)
                ))
                    .on_hover_text("Animate the mechanism (kinematic playback)")
                    .clicked()
                {
                    self.state.playing = !self.state.playing;
                    if self.state.playing && !self.state.loop_mode {
                        self.state.animation_direction = 1.0;
                    }
                }

                // Speed control (compact)
                ui.add(
                    egui::Slider::new(&mut self.state.animation_speed_deg_per_sec, 10.0..=720.0)
                        .text("\u{00B0}/s")
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Always),
                );

                ui.separator();

                // ── Sample mechanism selector (purple) ──────────────
                let sample_color = egui::Color32::from_rgb(180, 140, 255);
                ui.menu_button(
                    egui::RichText::new("\u{1F4C2} Samples \u{25BC}").color(sample_color),
                    |ui| {
                        for sample in SampleMechanism::all() {
                            if ui.button(sample.label()).clicked() {
                                self.state.load_sample(*sample);
                                ui.close();
                            }
                        }
                    },
                );
            });
        });

        // --- Force element toolbar ribbon ---
        egui::TopBottomPanel::top("force_toolbar").show(ctx, |ui| {
            if let Some(force_add) = force_toolbar::draw_force_toolbar(ui, &self.state) {
                match force_add {
                    force_toolbar::PendingForceAdd::Add(force) => {
                        self.state.add_force_element(force);
                    }
                }
            }
        });

        // --- Status bar ---
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let dim = egui::Color32::from_rgb(140, 145, 160);
                let bright = egui::Color32::from_rgb(200, 205, 220);
                let green = egui::Color32::from_rgb(80, 200, 80);
                let red = egui::Color32::from_rgb(220, 70, 70);
                let blue = egui::Color32::from_rgb(100, 180, 255);
                let warn = egui::Color32::from_rgb(255, 180, 50);

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
                        ui.colored_label(warn, "\u{26A0} No driver");
                    }
                    if !warnings.disconnected_bodies.is_empty() {
                        ui.colored_label(dim, "\u{2502}");
                        ui.colored_label(warn, format!(
                            "\u{26A0} Disconnected: {}",
                            warnings.disconnected_bodies.join(", ")
                        ));
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
