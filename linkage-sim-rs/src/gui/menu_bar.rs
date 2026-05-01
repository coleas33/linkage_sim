//! Menu bar for the linkage simulator GUI.

use std::collections::HashMap;

use eframe::egui;

use super::state::{AppState, AngleUnit, LengthUnit};
use super::samples::SampleMechanism;
use super::{dxf_import, export, tutorial};

pub(crate) fn draw_menu_bar(
    ctx: &egui::Context,
    state: &mut AppState,
    sample_thumbnails: &HashMap<SampleMechanism, egui::TextureHandle>,
) {
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                let file_resp = ui.menu_button("File", |ui| {
                    if ui.button("New  Ctrl+N")
                        .on_hover_text("Create a new empty mechanism (Ctrl+N)")
                        .clicked()
                    {
                        state.new_empty_mechanism();
                        ui.close();
                    }
                    let load_sample_resp = ui.menu_button("Load Sample", |ui| {
                        draw_sample_menu(ui, state, sample_thumbnails);
                    });
                    load_sample_resp.response.on_hover_text("Load a preset sample mechanism");
                    // ── Templates ─────────────────────────────────────
                    ui.separator();
                    if ui
                        .add_enabled(
                            state.blueprint.is_some(),
                            egui::Button::new("Save as Template..."),
                        )
                        .on_hover_text("Save the current mechanism as a reusable template")
                        .clicked()
                    {
                        state.show_template_name_dialog = true;
                        state.template_name_buf = String::new();
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            state.blueprint.is_some(),
                            egui::Button::new("Save as Sample..."),
                        )
                        .on_hover_text("Add this mechanism to the Samples dropdown")
                        .clicked()
                    {
                        state.show_custom_sample_dialog = true;
                        state.custom_sample_name_buf = String::new();
                        ui.close();
                    }
                    if !state.saved_templates.is_empty() {
                        let load_tpl_resp = ui.menu_button("Load Template", |ui| {
                            let mut load_idx = None;
                            for (i, (name, _)) in state.saved_templates.iter().enumerate() {
                                if ui.button(name).clicked() {
                                    load_idx = Some(i);
                                    ui.close();
                                }
                            }
                            if let Some(idx) = load_idx {
                                state.load_template(idx);
                            }
                        });
                        load_tpl_resp.response.on_hover_text("Load a saved mechanism template");

                        let manage_tpl_resp = ui.menu_button("Manage Templates", |ui| {
                            let mut delete_idx = None;
                            for (i, (name, _)) in state.saved_templates.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.label(name);
                                    if ui.small_button("Delete").clicked() {
                                        delete_idx = Some(i);
                                    }
                                });
                            }
                            if let Some(idx) = delete_idx {
                                state.delete_template(idx);
                            }
                        });
                        manage_tpl_resp.response.on_hover_text("Delete saved templates");
                    }
                    // ── Share via URL ─────────────────────────────────
                    ui.separator();
                    if ui
                        .add_enabled(
                            state.mechanism.is_some(),
                            egui::Button::new("Share via URL"),
                        )
                        .on_hover_text("Copy a shareable URL to the clipboard (loads in the web version)")
                        .clicked()
                    {
                        match state.generate_share_url() {
                            Ok(url) => {
                                ui.ctx().copy_text(url.clone());
                                state.status_message = Some(format!("Share URL copied ({} chars)", url.len()));
                                state.status_message_time = 4.0;
                            }
                            Err(e) => {
                                log::error!("Share URL generation failed: {}", e);
                                state.status_message = Some(format!("Share failed: {}", e));
                                state.status_message_time = 4.0;
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
                                if let Err(e) = state.load_from_file(&path) {
                                    log::error!("Failed to load mechanism: {}", e);
                                }
                            }
                            ui.close();
                        }
                        if ui.button("Import DXF...")
                            .on_hover_text("Import CAD geometry from a DXF file as a snappable overlay")
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("DXF", &["dxf"])
                                .pick_file()
                            {
                                match dxf_import::parse_dxf_file(&path, 0.001) {
                                    Ok(overlay) => {
                                        let n_ent = overlay.entities.len();
                                        let n_circ = overlay.snap_circles.len();
                                        state.dxf_overlay = Some(overlay);
                                        state.status_message = Some(format!(
                                            "DXF loaded: {} entities, {} circles (snap targets). Use tools to build or assign bodies.",
                                            n_ent, n_circ
                                        ));
                                        state.status_message_time = 5.0;
                                    }
                                    Err(e) => {
                                        log::error!("DXF import failed: {}", e);
                                        state.status_message = Some(format!("DXF import failed: {}", e));
                                        state.status_message_time = 4.0;
                                    }
                                }
                            }
                            ui.close();
                        }
                        if !state.recent_files.is_empty() {
                            let recent_resp = ui.menu_button("Recent Files", |ui| {
                                let mut load_path = None;
                                for path in &state.recent_files {
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
                                    if let Err(e) = state.load_from_file(&path) {
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
                            if let Some(path) = state.last_save_path.clone() {
                                if let Err(e) = state.save_to_file(&path) {
                                    log::error!("Save failed: {}", e);
                                }
                            } else if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .set_file_name("mechanism.json")
                                .save_file()
                            {
                                if let Err(e) = state.save_to_file(&path) {
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
                                if let Err(e) = state.save_to_file(&path) {
                                    log::error!("Failed to save mechanism: {}", e);
                                }
                            }
                            ui.close();
                        }
                        ui.separator();
                        if ui
                            .add_enabled(
                                state.sweep_data.is_some(),
                                egui::Button::new("Export Sweep CSV..."),
                            )
                            .on_hover_text(
                                "Export the active sweep as CSV.\n\
                                 \u{2022} Angle/Stroke modes: per-step kinematic and dynamic columns.\n\
                                 \u{2022} Trajectory mode: time-series with target/achieved/u/u_dot/u_ddot/F_actuator/status.",
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("CSV", &["csv"])
                                .set_file_name("sweep_data.csv")
                                .save_file()
                            {
                                if let Some(ref sweep) = state.sweep_data {
                                    if let Err(e) = export::export_sweep_csv(&path, sweep) {
                                        log::error!("CSV export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                state.sweep_data.is_some(),
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
                                if let Some(ref sweep) = state.sweep_data {
                                    if let Err(e) = export::export_coupler_csv(&path, sweep) {
                                        log::error!("Coupler CSV export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        // ── Firmware export (trajectory mode only) ──────────────
                        // Always visible so the feature is discoverable; greyed
                        // out + tooltip-explained when the active sweep isn't a
                        // computed trajectory. The adapter trait is shared with
                        // planned G-code/Aerotech/Beckhoff/Galil follow-ups
                        // (see export/firmware/mod.rs).
                        let in_traj_mode = state
                            .sweep_data
                            .as_ref()
                            .map(|d| matches!(d.sweep_mode, crate::gui::sweep::SweepMode::Trajectory { .. }))
                            .unwrap_or(false);
                        let firmware_clicked = ui
                            .add_enabled(
                                in_traj_mode,
                                egui::Button::new("Export firmware (JSON)..."),
                            )
                            .on_hover_text(if in_traj_mode {
                                "Export the trajectory as a firmware-friendly JSON \
                                 document for downstream actuator controllers."
                            } else {
                                "Switch to Trajectory mode and click Compute to enable this export."
                            })
                            .clicked();
                        if firmware_clicked {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON", &["json"])
                                .set_file_name("trajectory_firmware.json")
                                .save_file()
                            {
                                export_firmware_json(state, &path);
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                state.mechanism.is_some(),
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
                                if let Some(ref mech) = state.mechanism {
                                    if let Err(e) =
                                        export::export_mechanism_svg(&path, mech, &state.q)
                                    {
                                        log::error!("SVG export failed: {}", e);
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                state.mechanism.is_some(),
                                egui::Button::new("Export labeled schematic (SVG)..."),
                            )
                            .on_hover_text(
                                "Export the current mechanism as an SVG figure with constraint \
                                 labels — suitable for design docs, papers, and lab reports.",
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("SVG", &["svg"])
                                .set_file_name("mechanism_schematic.svg")
                                .save_file()
                            {
                                if let Some(ref mech) = state.mechanism {
                                    match export::schematic::generate_schematic_svg(
                                        mech,
                                        &state.q,
                                    ) {
                                        Ok(svg) => {
                                            if let Err(e) = std::fs::write(&path, &svg) {
                                                log::error!("Schematic write failed: {}", e);
                                                state.error_log.push(format!(
                                                    "Schematic export failed: {}",
                                                    e
                                                ));
                                                state.show_error_panel = true;
                                            } else {
                                                state.status_message = Some(format!(
                                                    "Schematic exported: {}",
                                                    path.display()
                                                ));
                                                state.status_message_time = 3.0;
                                            }
                                        }
                                        Err(e) => {
                                            log::error!("Schematic generation failed: {}", e);
                                            state.error_log.push(format!(
                                                "Schematic generation failed: {}",
                                                e
                                            ));
                                            state.show_error_panel = true;
                                        }
                                    }
                                }
                            }
                            ui.close();
                        }
                        if ui
                            .add_enabled(
                                state.mechanism.is_some(),
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
                                if let Some(ref mech) = state.mechanism {
                                    if let Err(e) = export::export_mechanism_png(
                                        &path,
                                        mech,
                                        &state.q,
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
                                state.sweep_data.is_some()
                                    && state.mechanism.is_some(),
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
                                    (&state.mechanism, &state.sweep_data)
                                {
                                    if let Err(e) = export::export_mechanism_gif(
                                        &path,
                                        mech,
                                        sweep,
                                        &state.q,
                                        state.driver_omega(),
                                        state.driver_theta_0(),
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
                    #[cfg(not(feature = "native"))]
                    {
                        ui.separator();
                        ui.label("Drag & drop a .dxf file onto the canvas to import CAD geometry");
                        if state.dxf_overlay.is_some() {
                            if ui.button("Clear DXF Overlay").clicked() {
                                state.dxf_overlay = None;
                                ui.close();
                            }
                        }
                    }
                    #[cfg(feature = "native")]
                    {
                        if ui
                            .add_enabled(
                                state.mechanism.is_some(),
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
                                if let Some(ref mech) = state.mechanism {
                                    if let Err(e) =
                                        export::export_mechanism_dxf(&path, mech, &state.q)
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
                                state.sweep_data.is_some()
                                    && state.mechanism.is_some(),
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
                                    (&state.mechanism, &state.sweep_data)
                                {
                                    match export::generate_html_report(
                                        mech,
                                        &state.q,
                                        sweep,
                                        state.grashof_result.as_ref(),
                                        &state.display_units,
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
                        .add_enabled(state.can_undo(), egui::Button::new("\u{21A9} Undo  Ctrl+Z"))
                        .on_hover_text("Undo the last change (Ctrl+Z)")
                        .clicked()
                    {
                        state.undo();
                        ui.close();
                    }
                    if ui
                        .add_enabled(state.can_redo(), egui::Button::new("\u{21AA} Redo  Ctrl+Y"))
                        .on_hover_text("Redo the last undone change (Ctrl+Y)")
                        .clicked()
                    {
                        state.redo();
                        ui.close();
                    }
                });
                edit_resp.response.on_hover_text("Undo, redo, and editing operations");
                let help_resp = ui.menu_button("Help", |ui| {
                    if ui.button("Keyboard Shortcuts")
                        .on_hover_text("Show all keyboard shortcuts")
                        .clicked()
                    {
                        state.show_shortcuts = true;
                        ui.close();
                    }
                    if ui.button("Tutorial: Build a 4-Bar")
                        .on_hover_text("Interactive step-by-step tutorial for building a 4-bar linkage")
                        .clicked()
                    {
                        state.new_empty_mechanism();
                        state.tutorial = tutorial::TutorialState::new_fourbar();
                        ui.close();
                    }
                    if ui.button("Tutorial: Actuator Sizing")
                        .on_hover_text("Learn how to size a linear actuator using force, speed, and power analysis")
                        .clicked()
                    {
                        state.tutorial = tutorial::TutorialState::new_actuator_sizing();
                        ui.close();
                    }
                });
                help_resp.response.on_hover_text("Keyboard shortcuts and help");
                let view_resp = ui.menu_button("View", |ui| {
                    ui.checkbox(&mut state.show_debug_overlay, "Debug Overlay")
                        .on_hover_text("Show solver status, body IDs, and attachment point names on the canvas");
                    ui.checkbox(&mut state.show_plots, "Plot Panel")
                        .on_hover_text("Show/hide the sweep data plot panel at the bottom");
                    ui.add_enabled(false, egui::Checkbox::new(&mut state.show_parametric, "Parametric Study [WIP]"))
                        .on_hover_text("Parametric study panel (coming soon)");
                    ui.checkbox(&mut state.show_forces, "Force Arrows")
                        .on_hover_text("Show/hide joint reaction force arrows and force element visuals on the canvas");
                    ui.checkbox(&mut state.show_dimensions, "Link Dimensions")
                        .on_hover_text("Show/hide link length dimensions on the canvas");
                    ui.checkbox(&mut state.show_labels, "Show Labels")
                        .on_hover_text("Show/hide body and joint labels on the canvas");
                    ui.checkbox(&mut state.show_equation_overlay, "Show equations")
                        .on_hover_text("Overlay loop-equation tags Φ_J* and per-body q vectors on the canvas. Toggle off to hide.");
                    let enabled = state.gravity_magnitude > 0.0;
                    let mut check = enabled;
                    if ui.checkbox(&mut check, "Gravity")
                        .on_hover_text("Enable/disable gravitational acceleration (9.81 m/s\u{00b2})")
                        .changed()
                    {
                        state.gravity_magnitude = if check { 9.81 } else { 0.0 };
                        state.mark_sweep_dirty();
                    }
                    ui.separator();
                    ui.label("Units:");
                    let mut use_mm = state.display_units.length == LengthUnit::Millimeters;
                    if ui.checkbox(&mut use_mm, "Millimeters")
                        .on_hover_text("Display lengths in millimeters instead of meters")
                        .changed()
                    {
                        state.display_units.length = if use_mm {
                            LengthUnit::Millimeters
                        } else {
                            LengthUnit::Meters
                        };
                    }
                    let mut use_deg = state.display_units.angle == AngleUnit::Degrees;
                    if ui.checkbox(&mut use_deg, "Degrees")
                        .on_hover_text("Display angles in degrees instead of radians")
                        .changed()
                    {
                        state.display_units.angle = if use_deg {
                            AngleUnit::Degrees
                        } else {
                            AngleUnit::Radians
                        };
                    }
                    ui.separator();
                    ui.label("Grid:");
                    ui.checkbox(&mut state.grid.show_grid, "Show Grid")
                        .on_hover_text("Show/hide the background grid on the canvas");
                    ui.checkbox(&mut state.grid.snap_enabled, "Snap to Grid")
                        .on_hover_text("Snap placed points to the nearest grid intersection");
                    ui.horizontal(|ui| {
                        ui.label("Spacing:");
                        let mut spacing_display =
                            state.display_units.length(state.grid.spacing_m);
                        if ui
                            .add(
                                egui::DragValue::new(&mut spacing_display)
                                    .speed(0.1)
                                    .range(0.001..=100.0)
                                    .suffix(state.display_units.length_suffix()),
                            )
                            .on_hover_text("Distance between grid lines. When 'Snap to Grid' is enabled, placed points will snap to the nearest multiple of this value. Set to 0 for auto-spacing based on zoom level.")
                            .changed()
                        {
                            state.grid.spacing_m =
                                state.display_units.length_to_si(spacing_display);
                        }
                    });
                    ui.checkbox(&mut state.show_load_path, "Load Path (heat map)")
                        .on_hover_text("Color-code links by joint reaction force magnitude (blue=low, red=high)");
                    if ui.checkbox(&mut state.nathan_mode, "Nathan Mode")
                        .on_hover_text("Render all plot series and UI colors in grayscale. Useful for accessibility or when preparing figures for print.")
                        .changed() && !state.nathan_mode
                    {
                        // Restore normal visuals when turning off.
                        super::theme::restore_normal_visuals(ui.ctx());
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
                                match super::load_background_image(ctx, &path) {
                                    Ok(bg) => {
                                        state.background_image = Some(bg);
                                        state.status_message = Some("Background image loaded".to_string());
                                        state.status_message_time = 3.0;
                                    }
                                    Err(e) => {
                                        log::error!("Failed to load background image: {}", e);
                                        state.status_message = Some(format!("Image load failed: {}", e));
                                        state.status_message_time = 4.0;
                                    }
                                }
                            }
                            ui.close();
                        }
                    }
                    #[cfg(not(feature = "native"))]
                    {
                        ui.label("Drag & drop an image onto the canvas");
                    }
                    if state.background_image.is_some() {
                        if ui.button("Remove Image")
                            .on_hover_text("Remove the background image")
                            .clicked()
                        {
                            state.background_image = None;
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("Image Settings...")
                            .on_hover_text("Open image controls (opacity, size, position)")
                            .clicked()
                        {
                            state.show_image_settings = true;
                            ui.close();
                        }
                    }
                });
                image_resp.response.on_hover_text("Background image import and controls");
            });
        });
}


// ── Sample gallery menu ─────────────────────────────────────────────────────

/// Draw the sample mechanism menu with categories, thumbnails, and descriptions.
///
/// Samples are grouped by category (4-bar, 6-bar, specialty) with separator
/// headers. Each sample shows a small thumbnail (when available) alongside its
/// label, with a tooltip containing a brief description.
pub(crate) fn draw_sample_menu(
    ui: &mut egui::Ui,
    state: &mut AppState,
    thumbnails: &HashMap<SampleMechanism, egui::TextureHandle>,
) {
    egui::ScrollArea::vertical()
        .max_height(400.0)
        .show(ui, |ui| {
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

    // ── My Samples (user-saved custom samples) ──────────────────────
    if !state.custom_samples.is_empty() {
        ui.separator();
        ui.label(egui::RichText::new("My Samples").small().weak());
        let mut delete_idx = None;
        let mut load_idx = None;
        for (i, (name, _json)) in state.custom_samples.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.button(name).clicked() {
                    load_idx = Some(i);
                }
                if ui
                    .small_button("\u{2715}")
                    .on_hover_text("Remove from samples")
                    .clicked()
                {
                    delete_idx = Some(i);
                }
            });
        }
        if let Some(idx) = load_idx {
            state.load_custom_sample(idx);
            ui.close();
        }
        if let Some(idx) = delete_idx {
            state.delete_custom_sample(idx);
        }
    }
        }); // end ScrollArea
}

/// Trajectory-mode JSON firmware export. Reads the active `SweepMode::Trajectory`'s
/// target + trajectory + driver-kind units from `state` and dispatches to
/// `JsonAdapter`. Any failure is surfaced via `error_log` + `show_error_panel`;
/// success sets a transient status message.
#[cfg(feature = "native")]
fn export_firmware_json(state: &mut AppState, path: &std::path::Path) {
    use crate::gui::export::firmware::{FirmwareAdapter, JsonAdapter};
    use crate::gui::state::DriverKind;
    use crate::gui::sweep::SweepMode;

    // Pull (target, trajectory) from the active sweep_mode living in the
    // SweepData itself — that mirrors what `compute_trajectory` was last
    // invoked with and avoids assuming `state.sweep_mode` is in lockstep.
    let Some(data) = state.sweep_data.as_ref() else {
        state
            .error_log
            .push("Firmware export: no sweep data available".to_string());
        state.show_error_panel = true;
        return;
    };
    let SweepMode::Trajectory {
        target, trajectory, ..
    } = &data.sweep_mode
    else {
        state
            .error_log
            .push("Firmware export: not in Trajectory mode".to_string());
        state.show_error_panel = true;
        return;
    };

    // Input parameter units follow the active driver kind: revolute → rad,
    // linear → m. None falls back to "rad" (matches the historical default
    // when no driver is bound; the trajectory solve still uses the body-angle
    // input parameter in that case).
    let input_units = match state.driver_kind {
        DriverKind::Linear { .. } => "m",
        DriverKind::Revolute { .. } | DriverKind::None => "rad",
    };

    let adapter = JsonAdapter;
    match adapter.emit(data, target, trajectory, input_units) {
        Ok(json_str) => match std::fs::write(path, json_str) {
            Ok(()) => {
                state.status_message =
                    Some(format!("Firmware JSON exported: {}", path.display()));
                state.status_message_time = 3.0;
            }
            Err(e) => {
                state
                    .error_log
                    .push(format!("Firmware export write failed: {}", e));
                state.show_error_panel = true;
            }
        },
        Err(e) => {
            state
                .error_log
                .push(format!("Firmware export failed: {}", e));
            state.show_error_panel = true;
        }
    }
}
