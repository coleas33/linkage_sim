//! DXF import: parse DXF files into a snappable overlay with interactive body assignment.
//!
//! Two modes:
//! - **Overlay (Mode C):** DXF geometry renders on the canvas; circle centers become snap targets.
//! - **Assignment (Mode B):** Sidebar panel for grouping entities into bodies, auto-detecting joints.

use std::collections::{HashMap, HashSet};

use crate::core::body::BodyGeometry;
use crate::forces::elements::{ForceElement, LinearActuatorElement};
use crate::io::schema::{BodyJson, JointJson};

// ── Data model ──────────────────────────────────────────────────────────────

/// A parsed DXF entity in world coordinates (after scale + offset).
#[derive(Debug, Clone)]
pub enum DxfEntityKind {
    Line { x1: f64, y1: f64, x2: f64, y2: f64 },
    Circle { cx: f64, cy: f64, radius: f64 },
    Arc { cx: f64, cy: f64, radius: f64, start_deg: f64, end_deg: f64 },
    Point { x: f64, y: f64 },
}

/// A single DXF entity with its index for selection tracking.
#[derive(Debug, Clone)]
pub struct DxfEntity {
    pub kind: DxfEntityKind,
    pub index: usize,
}

/// A circle center extracted for snap targeting.
#[derive(Debug, Clone)]
pub struct DxfSnapCircle {
    pub center: [f64; 2],
    pub radius: f64,
    pub entity_index: usize,
}

/// A body assignment: user-selected group of DXF entities.
#[derive(Debug, Clone)]
pub struct BodyAssignment {
    pub name: String,
    pub entity_indices: Vec<usize>,
    pub is_ground: bool,
}

/// The full DXF overlay state.
#[derive(Debug, Clone)]
pub struct DxfOverlay {
    /// All parsed entities.
    pub entities: Vec<DxfEntity>,
    /// Circle centers for snapping.
    pub snap_circles: Vec<DxfSnapCircle>,
    /// Scale factor applied to raw DXF coordinates (e.g., 0.001 for mm->m).
    pub scale: f64,
    /// World offset [x, y] in meters.
    pub offset: [f64; 2],
    /// Whether overlay is visible.
    pub visible: bool,
    /// Rendering opacity.
    pub opacity: f32,
    /// Body assignments (Mode B).
    pub assignments: Vec<BodyAssignment>,
    /// Currently selected entity indices (during assignment).
    pub selected_entities: HashSet<usize>,
    /// Whether in body-assignment mode.
    pub assigning: bool,
    /// Name for the next body being built.
    pub next_body_name: String,
}

impl DxfOverlay {
    /// Find the nearest snap circle center within `threshold` meters of `world_pos`.
    pub fn find_snap(&self, world_x: f64, world_y: f64, threshold: f64) -> Option<[f64; 2]> {
        if !self.visible {
            return None;
        }
        let mut best: Option<(f64, [f64; 2])> = None;
        for sc in &self.snap_circles {
            let dx = sc.center[0] - world_x;
            let dy = sc.center[1] - world_y;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < threshold {
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, sc.center));
                }
            }
        }
        best.map(|(_, c)| c)
    }
}

// ── DXF parsing ─────────────────────────────────────────────────────────────

/// Parse a DXF file from a filesystem path.
///
/// `scale` converts raw DXF coordinates to meters (e.g., 0.001 for mm).
#[cfg(feature = "native")]
pub fn parse_dxf_file(path: &std::path::Path, scale: f64) -> Result<DxfOverlay, String> {
    let drawing = dxf::Drawing::load_file(path)
        .map_err(|e| format!("Failed to parse DXF: {}", e))?;
    parse_dxf_drawing(drawing, scale)
}

/// Parse a DXF file from raw bytes (works on native + WASM).
pub fn parse_dxf_bytes(bytes: &[u8], scale: f64) -> Result<DxfOverlay, String> {
    let mut reader = std::io::Cursor::new(bytes);
    let drawing = dxf::Drawing::load(&mut reader)
        .map_err(|e| format!("Failed to parse DXF: {}", e))?;
    parse_dxf_drawing(drawing, scale)
}

fn parse_dxf_drawing(drawing: dxf::Drawing, scale: f64) -> Result<DxfOverlay, String> {
    let mut entities = Vec::new();
    let mut snap_circles = Vec::new();
    let mut idx = 0usize;

    for entity in drawing.entities() {
        match &entity.specific {
            dxf::entities::EntityType::Line(line) => {
                entities.push(DxfEntity {
                    kind: DxfEntityKind::Line {
                        x1: line.p1.x * scale,
                        y1: line.p1.y * scale,
                        x2: line.p2.x * scale,
                        y2: line.p2.y * scale,
                    },
                    index: idx,
                });
                idx += 1;
            }
            dxf::entities::EntityType::Circle(circle) => {
                let cx = circle.center.x * scale;
                let cy = circle.center.y * scale;
                let r = circle.radius * scale;
                snap_circles.push(DxfSnapCircle {
                    center: [cx, cy],
                    radius: r,
                    entity_index: idx,
                });
                entities.push(DxfEntity {
                    kind: DxfEntityKind::Circle { cx, cy, radius: r },
                    index: idx,
                });
                idx += 1;
            }
            dxf::entities::EntityType::Arc(arc) => {
                let cx = arc.center.x * scale;
                let cy = arc.center.y * scale;
                let r = arc.radius * scale;
                entities.push(DxfEntity {
                    kind: DxfEntityKind::Arc {
                        cx, cy, radius: r,
                        start_deg: arc.start_angle,
                        end_deg: arc.end_angle,
                    },
                    index: idx,
                });
                // Arc centers are also snap targets
                snap_circles.push(DxfSnapCircle {
                    center: [cx, cy],
                    radius: r,
                    entity_index: idx,
                });
                idx += 1;
            }
            dxf::entities::EntityType::LwPolyline(poly) => {
                // Tessellate polyline into line segments
                let verts: Vec<_> = poly.vertices.iter()
                    .map(|v| (v.x * scale, v.y * scale))
                    .collect();
                for w in verts.windows(2) {
                    entities.push(DxfEntity {
                        kind: DxfEntityKind::Line {
                            x1: w[0].0, y1: w[0].1,
                            x2: w[1].0, y2: w[1].1,
                        },
                        index: idx,
                    });
                    idx += 1;
                }
                // Close the polyline if flagged (bit 0 of flags = closed)
                if (poly.flags & 1) != 0 && verts.len() >= 3 {
                    entities.push(DxfEntity {
                        kind: DxfEntityKind::Line {
                            x1: verts.last().unwrap().0,
                            y1: verts.last().unwrap().1,
                            x2: verts[0].0,
                            y2: verts[0].1,
                        },
                        index: idx,
                    });
                    idx += 1;
                }
            }
            dxf::entities::EntityType::ModelPoint(pt) => {
                entities.push(DxfEntity {
                    kind: DxfEntityKind::Point {
                        x: pt.location.x * scale,
                        y: pt.location.y * scale,
                    },
                    index: idx,
                });
                idx += 1;
            }
            _ => {} // skip unsupported entities
        }
    }

    if entities.is_empty() {
        return Err("DXF contains no geometry entities.".to_string());
    }

    Ok(DxfOverlay {
        entities,
        snap_circles,
        scale,
        offset: [0.0, 0.0],
        visible: true,
        opacity: 0.6,
        assignments: Vec::new(),
        selected_entities: HashSet::new(),
        assigning: false,
        next_body_name: "body_1".to_string(),
    })
}

// ── Canvas rendering ────────────────────────────────────────────────────────

use eframe::egui;

/// Render DXF overlay on the canvas.
pub fn draw_dxf_overlay(
    painter: &egui::Painter,
    canvas_rect: egui::Rect,
    state: &super::state::AppState,
) {
    let overlay = match &state.dxf_overlay {
        Some(o) if o.visible => o,
        _ => return,
    };

    let view = &state.view;
    let alpha = (overlay.opacity * 255.0) as u8;
    let _ = canvas_rect; // used for clipping checks below

    let to_screen = |wx: f64, wy: f64| -> egui::Pos2 {
        let [sx, sy] = view.world_to_screen(wx, wy);
        egui::pos2(sx, sy)
    };

    let assigned_colors = [
        egui::Color32::from_rgba_unmultiplied(100, 200, 255, alpha),
        egui::Color32::from_rgba_unmultiplied(255, 150, 80, alpha),
        egui::Color32::from_rgba_unmultiplied(120, 220, 120, alpha),
        egui::Color32::from_rgba_unmultiplied(255, 100, 100, alpha),
        egui::Color32::from_rgba_unmultiplied(200, 150, 255, alpha),
        egui::Color32::from_rgba_unmultiplied(255, 220, 100, alpha),
    ];

    // Build entity -> assignment color lookup
    let mut entity_color: HashMap<usize, egui::Color32> = HashMap::new();
    for (ai, assignment) in overlay.assignments.iter().enumerate() {
        let color = assigned_colors[ai % assigned_colors.len()];
        for &ei in &assignment.entity_indices {
            entity_color.insert(ei, color);
        }
    }

    let default_color = egui::Color32::from_rgba_unmultiplied(180, 180, 180, alpha);
    let selected_color = egui::Color32::from_rgba_unmultiplied(80, 180, 255, alpha);

    for entity in &overlay.entities {
        let color = if overlay.selected_entities.contains(&entity.index) {
            selected_color
        } else {
            entity_color.get(&entity.index).copied().unwrap_or(default_color)
        };
        let width = if overlay.selected_entities.contains(&entity.index) { 2.5 } else { 1.5 };

        match &entity.kind {
            DxfEntityKind::Line { x1, y1, x2, y2 } => {
                let ox = overlay.offset[0];
                let oy = overlay.offset[1];
                let p1 = to_screen(*x1 + ox, *y1 + oy);
                let p2 = to_screen(*x2 + ox, *y2 + oy);
                if canvas_rect.contains(p1) || canvas_rect.contains(p2) {
                    painter.line_segment([p1, p2], egui::Stroke::new(width, color));
                }
            }
            DxfEntityKind::Circle { cx, cy, radius } => {
                let ox = overlay.offset[0];
                let oy = overlay.offset[1];
                let center = to_screen(*cx + ox, *cy + oy);
                let r_px = (*radius * view.scale as f64) as f32;
                if r_px > 0.5 {
                    painter.circle_stroke(center, r_px, egui::Stroke::new(width, color));
                }
                // Draw center dot (snap target)
                let dot_color = egui::Color32::from_rgba_unmultiplied(50, 220, 50, alpha);
                painter.circle_filled(center, 3.0, dot_color);
            }
            DxfEntityKind::Arc { cx, cy, radius, start_deg, end_deg } => {
                let ox = overlay.offset[0];
                let oy = overlay.offset[1];
                let _center = to_screen(*cx + ox, *cy + oy);
                let r_px = (*radius * view.scale as f64) as f32;
                if r_px > 0.5 {
                    // Approximate arc with line segments
                    let start = start_deg.to_radians();
                    let mut end = end_deg.to_radians();
                    if end < start { end += std::f64::consts::TAU; }
                    let n_segs = ((end - start) / 0.1).ceil().max(4.0) as usize;
                    let mut pts = Vec::with_capacity(n_segs + 1);
                    for i in 0..=n_segs {
                        let angle = start + (end - start) * (i as f64 / n_segs as f64);
                        let wx = *cx + ox + *radius * angle.cos();
                        let wy = *cy + oy + *radius * angle.sin();
                        pts.push(to_screen(wx, wy));
                    }
                    for w in pts.windows(2) {
                        painter.line_segment([w[0], w[1]], egui::Stroke::new(width, color));
                    }
                }
            }
            DxfEntityKind::Point { x, y } => {
                let ox = overlay.offset[0];
                let oy = overlay.offset[1];
                let p = to_screen(*x + ox, *y + oy);
                painter.circle_filled(p, 3.0, color);
            }
        }
    }
}

// ── Assignment panel (sidebar) ──────────────────────────────────────────────

/// Deferred action from the DXF panel, executed after the mutable overlay
/// borrow ends.
enum DxfAction {
    ConvertSelectedToLinks(Vec<usize>),
    ConvertSelectedToMultiJointBody(Vec<usize>),
    /// Snapshot the DXF selection and open the "pick target link" popup;
    /// the popup calls `convert_selected_to_rigid_geometry` on apply.
    OpenRigidGeometryTargetDialog(Vec<usize>),
    ConvertSelectedToGround(Vec<usize>),
    ConvertSelectedLineToActuator(Vec<usize>),
    ConvertAllLinesToLinks,
    DeleteSelected(Vec<usize>),
    ClearOverlay,
}

/// Draw the DXF import settings and body assignment panel.
pub fn draw_dxf_panel(ui: &mut egui::Ui, state: &mut super::state::AppState) {
    // First phase: draw UI with mutable overlay borrow. Record any actions
    // that need to run after the borrow is released.
    let mut action: Option<DxfAction> = None;

    {
    let overlay = match state.dxf_overlay.as_mut() {
        Some(o) => o,
        None => return,
    };

    ui.heading("DXF Import");
    ui.separator();

    // Visibility and opacity
    ui.checkbox(&mut overlay.visible, "Show overlay");
    ui.add(egui::Slider::new(&mut overlay.opacity, 0.1..=1.0).text("Opacity"));

    // Scale
    ui.separator();
    ui.label("Units:");
    ui.horizontal(|ui| {
        if ui.selectable_label(overlay.scale == 0.001, "mm").clicked() {
            rescale_overlay(overlay, 0.001);
        }
        if ui.selectable_label(overlay.scale == 0.01, "cm").clicked() {
            rescale_overlay(overlay, 0.01);
        }
        if ui.selectable_label(overlay.scale == 1.0, "m").clicked() {
            rescale_overlay(overlay, 1.0);
        }
        if ui.selectable_label((overlay.scale - 0.0254).abs() < 1e-6, "in").clicked() {
            rescale_overlay(overlay, 0.0254);
        }
    });

    // Offset
    ui.separator();
    ui.label("Offset (mm):");
    let mut ox_mm = overlay.offset[0] * 1000.0;
    let mut oy_mm = overlay.offset[1] * 1000.0;
    ui.horizontal(|ui| {
        ui.label("X:");
        ui.add(egui::DragValue::new(&mut ox_mm).speed(1.0).suffix(" mm"));
        ui.label("Y:");
        ui.add(egui::DragValue::new(&mut oy_mm).speed(1.0).suffix(" mm"));
    });
    overlay.offset[0] = ox_mm / 1000.0;
    overlay.offset[1] = oy_mm / 1000.0;

    // Info
    ui.separator();
    let n_lines = overlay.entities.iter().filter(|e| matches!(e.kind, DxfEntityKind::Line { .. })).count();
    let n_circles = overlay.snap_circles.len();
    let n_arcs = overlay.entities.iter().filter(|e| matches!(e.kind, DxfEntityKind::Arc { .. })).count();
    ui.label(format!("{} lines, {} circles, {} arcs", n_lines, n_circles, n_arcs));

    // ── Interactive selection + conversion ──────────────────────────────
    ui.separator();
    ui.heading("Convert to Mechanism");
    ui.label("Click DXF lines/circles on the canvas to select. Shift-click for more. Then use a button below.");
    ui.label(format!("Selected: {}", overlay.selected_entities.len()));

    ui.horizontal(|ui| {
        if ui.button("Deselect All").clicked() {
            overlay.selected_entities.clear();
        }
        if ui.button("Select All Lines").clicked() {
            for e in &overlay.entities {
                if matches!(e.kind, DxfEntityKind::Line { .. }) {
                    overlay.selected_entities.insert(e.index);
                }
            }
        }
    });
    if !overlay.selected_entities.is_empty() {
        let del_label = format!("Delete Selected ({})", overlay.selected_entities.len());
        if ui.add(egui::Button::new(del_label)
            .fill(egui::Color32::from_rgb(160, 60, 60)))
            .on_hover_text("Remove the selected lines/circles from the DXF overlay. Does not affect the mechanism.")
            .clicked()
        {
            action = Some(DxfAction::DeleteSelected(
                overlay.selected_entities.iter().copied().collect(),
            ));
        }
    }

    let n_selected = overlay.selected_entities.len();
    let n_selected_lines = overlay.entities.iter()
        .filter(|e| overlay.selected_entities.contains(&e.index) && matches!(e.kind, DxfEntityKind::Line { .. }))
        .count();

    ui.separator();
    ui.label("Convert selected:");

    // Convert each selected line to its own 2-point body (link)
    let can_lines = n_selected_lines > 0;
    if ui.add_enabled(can_lines, egui::Button::new(format!("→ Links ({} lines)", n_selected_lines)))
        .on_hover_text("Each selected line becomes its own 2-point body. Shared endpoints between selected lines become revolute joints automatically.")
        .clicked()
    {
        action = Some(DxfAction::ConvertSelectedToLinks(
            overlay.selected_entities.iter().copied().collect(),
        ));
    }

    // Group all selected entities into ONE multi-point body (appears as a
    // ternary/quaternary link with joints at every endpoint).
    if ui.add_enabled(n_selected >= 1, egui::Button::new(format!("→ Multi-joint Body ({})", n_selected)))
        .on_hover_text("All selected entities become ONE rigid body with attachment points at every unique endpoint. Appears as a ternary/quaternary link — the attachment points become joint locations.")
        .clicked()
    {
        action = Some(DxfAction::ConvertSelectedToMultiJointBody(
            overlay.selected_entities.iter().copied().collect(),
        ));
    }

    // Add BodyGeometry (force zone shape) to a link chosen in a popup.
    // The popup lists every non-ground link; clicking one attaches the
    // DXF bounding-box geometry to that link rigidly. No new body, no
    // new joints — just sets the target link's `geometry` field.
    if ui.add_enabled(n_selected >= 1, egui::Button::new(format!("→ Add Geometry to Link ({})", n_selected)))
        .on_hover_text("Open a popup to pick which link the DXF bounding-box geometry attaches to. The geometry is rigidly attached to the chosen link and moves with it, just like the Link Editor's Add Geometry button but sized from the DXF.")
        .clicked()
    {
        action = Some(DxfAction::OpenRigidGeometryTargetDialog(
            overlay.selected_entities.iter().copied().collect(),
        ));
    }

    // Mark selected entity endpoints as ground
    if ui.add_enabled(n_selected >= 1, egui::Button::new("→ Ground Pivots"))
        .on_hover_text("Endpoints and circle centers of selected entities become fixed ground pivots.")
        .clicked()
    {
        action = Some(DxfAction::ConvertSelectedToGround(
            overlay.selected_entities.iter().copied().collect(),
        ));
    }

    // Convert a single selected line to a Linear Actuator
    if ui.add_enabled(n_selected_lines >= 1, egui::Button::new("→ Linear Actuator"))
        .on_hover_text("Convert one selected DXF line into a LinearActuator force element. The line endpoints snap to the nearest existing body attachment points (ground preferred). If no body point is near an endpoint, a new ground pivot is created there automatically.")
        .clicked()
    {
        action = Some(DxfAction::ConvertSelectedLineToActuator(
            overlay.selected_entities.iter().copied().collect(),
        ));
    }

    // Auto-convert everything at once
    ui.separator();
    if ui.button("Auto: All Lines → Links")
        .on_hover_text("One-click: convert every DXF line into a link. Skip the selection step.")
        .clicked()
    {
        action = Some(DxfAction::ConvertAllLinesToLinks);
    }

    // Clear overlay
    ui.separator();
    if ui.button("Clear DXF Overlay").clicked() {
        action = Some(DxfAction::ClearOverlay);
    }
    } // end of mutable overlay borrow

    // Second phase: execute deferred action with full state access
    if let Some(act) = action {
        match act {
            DxfAction::ConvertSelectedToLinks(indices) => {
                convert_selected_lines_to_links(state, &indices);
            }
            DxfAction::ConvertSelectedToMultiJointBody(indices) => {
                convert_selected_to_single_body(state, &indices);
            }
            DxfAction::OpenRigidGeometryTargetDialog(indices) => {
                state.dxf_geometry_pending_indices = indices;
                state.show_dxf_geometry_target_dialog = true;
            }
            DxfAction::ConvertSelectedToGround(indices) => {
                convert_selected_to_ground(state, &indices);
            }
            DxfAction::ConvertSelectedLineToActuator(indices) => {
                convert_selected_line_to_actuator(state, &indices);
            }
            DxfAction::ConvertAllLinesToLinks => {
                let all_line_indices: Vec<usize> = state.dxf_overlay.as_ref().map(|o|
                    o.entities.iter()
                        .filter(|e| matches!(e.kind, DxfEntityKind::Line { .. }))
                        .map(|e| e.index)
                        .collect()
                ).unwrap_or_default();
                convert_selected_lines_to_links(state, &all_line_indices);
            }
            DxfAction::DeleteSelected(indices) => {
                if let Some(overlay) = state.dxf_overlay.as_mut() {
                    let to_remove: std::collections::HashSet<usize> = indices.iter().copied().collect();
                    let n_before = overlay.entities.len();
                    overlay.entities.retain(|e| !to_remove.contains(&e.index));
                    overlay.snap_circles.retain(|sc| !to_remove.contains(&sc.entity_index));
                    overlay.selected_entities.clear();
                    let n_deleted = n_before - overlay.entities.len();
                    state.status_message = Some(format!("Deleted {} DXF entities", n_deleted));
                    state.status_message_time = 3.0;
                }
            }
            DxfAction::ClearOverlay => {
                state.dxf_overlay = None;
            }
        }
    }
}

/// Popup shown after clicking "→ Add Geometry to Link". Lists every
/// non-ground body; clicking one attaches the DXF bounding-box geometry
/// to that link rigidly and closes the dialog. Escape / window-close
/// cancels without touching the blueprint.
///
/// Must be called from the root `Context` (not a panel `Ui`) because it
/// draws a floating `egui::Window`. `LinkageApp::update` is the canonical
/// call site, alongside the other dialogs.
pub fn draw_geometry_target_dialog(ctx: &egui::Context, state: &mut super::state::AppState) {
    if !state.show_dxf_geometry_target_dialog {
        return;
    }

    // Collect non-ground body IDs from the mechanism (preserves display
    // order used by the Link Editor). If the mechanism isn't built yet
    // there's nothing to pick; close and bail.
    let body_ids: Vec<String> = match state.mechanism.as_ref() {
        Some(mech) => mech
            .body_order()
            .iter()
            .filter(|b| b.as_str() != crate::core::state::GROUND_ID)
            .cloned()
            .collect(),
        None => {
            state.show_dxf_geometry_target_dialog = false;
            state.dxf_geometry_pending_indices.clear();
            state.status_message =
                Some("Mechanism not built — try again after building".to_string());
            state.status_message_time = 3.0;
            return;
        }
    };

    // Resolve labels from the blueprint for friendlier display text.
    let labels: Vec<(String, String)> = body_ids
        .iter()
        .map(|bid| {
            let label = state
                .blueprint
                .as_ref()
                .and_then(|bp| bp.bodies.get(bid))
                .and_then(|b| b.label.clone())
                .unwrap_or_else(|| bid.clone());
            (bid.clone(), label)
        })
        .collect();

    let mut open = true;
    let mut chosen: Option<String> = None;
    let mut cancel = false;

    egui::Window::new("Attach DXF Geometry to Link")
        .collapsible(false)
        .resizable(false)
        .default_width(280.0)
        .open(&mut open)
        .show(ctx, |ui| {
            if body_ids.is_empty() {
                ui.label("No links available — create a link first.");
                ui.add_space(4.0);
                if ui.button("Close").clicked() {
                    cancel = true;
                }
                return;
            }

            ui.label("Click a link to attach the DXF geometry:");
            ui.add_space(4.0);

            for (bid, label) in &labels {
                let text = if label == bid {
                    bid.clone()
                } else {
                    format!("{}  ({})", label, bid)
                };
                if ui.selectable_label(false, text).clicked() {
                    chosen = Some(bid.clone());
                }
            }

            ui.add_space(6.0);
            if ui.button("Cancel").clicked()
                || ui.input(|i| i.key_pressed(egui::Key::Escape))
            {
                cancel = true;
            }
        });

    if let Some(bid) = chosen {
        let indices = std::mem::take(&mut state.dxf_geometry_pending_indices);
        state.show_dxf_geometry_target_dialog = false;
        convert_selected_to_rigid_geometry(state, &indices, bid);
    } else if cancel || !open {
        state.show_dxf_geometry_target_dialog = false;
        state.dxf_geometry_pending_indices.clear();
    }
}

/// Rescale all entities in the overlay when unit changes.
fn rescale_overlay(overlay: &mut DxfOverlay, new_scale: f64) {
    if (new_scale - overlay.scale).abs() < 1e-15 {
        return;
    }
    let ratio = new_scale / overlay.scale;
    for entity in &mut overlay.entities {
        match &mut entity.kind {
            DxfEntityKind::Line { x1, y1, x2, y2 } => {
                *x1 *= ratio; *y1 *= ratio; *x2 *= ratio; *y2 *= ratio;
            }
            DxfEntityKind::Circle { cx, cy, radius } => {
                *cx *= ratio; *cy *= ratio; *radius *= ratio;
            }
            DxfEntityKind::Arc { cx, cy, radius, .. } => {
                *cx *= ratio; *cy *= ratio; *radius *= ratio;
            }
            DxfEntityKind::Point { x, y } => {
                *x *= ratio; *y *= ratio;
            }
        }
    }
    for sc in &mut overlay.snap_circles {
        sc.center[0] *= ratio;
        sc.center[1] *= ratio;
        sc.radius *= ratio;
    }
    overlay.scale = new_scale;
}

// ── Build mechanism from assignments ────────────────────────────────────────

/// Convert body assignments into a MechanismJson and load it.
// ── Conversion helpers ──────────────────────────────────────────────────────

const JOINT_TOL: f64 = 0.002; // 2mm tolerance for "same point"

/// Snap a line endpoint to the nearest circle center within tolerance.
fn snap_to_circle(overlay: &DxfOverlay, pt: [f64; 2]) -> [f64; 2] {
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];
    let mut best_dist = JOINT_TOL;
    let mut best = pt;
    for sc in &overlay.snap_circles {
        let cx = sc.center[0] + ox;
        let cy = sc.center[1] + oy;
        let d = ((pt[0] - cx).powi(2) + (pt[1] - cy).powi(2)).sqrt();
        if d < best_dist {
            best_dist = d;
            best = [cx, cy];
        }
    }
    best
}

/// Convert selected DXF lines into individual 2-point bodies (links).
/// Shared endpoints between lines become revolute joints. Endpoints that
/// match nothing else are kept as free attachment points (e.g. for ground).
/// **Additive**: adds to the existing mechanism without replacing it.
fn convert_selected_lines_to_links(state: &mut super::state::AppState, line_indices: &[usize]) {
    // Phase 1: collect lines while holding an immutable overlay borrow.
    let lines: Vec<([f64; 2], [f64; 2])> = {
        let overlay = match &state.dxf_overlay {
            Some(o) => o,
            None => return,
        };
        let ox = overlay.offset[0];
        let oy = overlay.offset[1];
        let mut out = Vec::new();
        for &idx in line_indices {
            if let Some(entity) = overlay.entities.iter().find(|e| e.index == idx) {
                if let DxfEntityKind::Line { x1, y1, x2, y2 } = &entity.kind {
                    let p1 = snap_to_circle(overlay, [*x1 + ox, *y1 + oy]);
                    let p2 = snap_to_circle(overlay, [*x2 + ox, *y2 + oy]);
                    out.push((p1, p2));
                }
            }
        }
        out
    };

    if lines.is_empty() {
        state.status_message = Some("No lines selected to convert".to_string());
        state.status_message_time = 3.0;
        return;
    }

    // Phase 2: cluster endpoints to detect shared joints.
    let mut cluster_centers: Vec<[f64; 2]> = Vec::new();
    let mut point_to_cluster: Vec<usize> = Vec::new(); // 2 entries per line
    for pair in &lines {
        for pt in [&pair.0, &pair.1] {
            let mut found = None;
            for (ci, cc) in cluster_centers.iter().enumerate() {
                let d = ((pt[0] - cc[0]).powi(2) + (pt[1] - cc[1]).powi(2)).sqrt();
                if d < JOINT_TOL {
                    found = Some(ci);
                    break;
                }
            }
            if let Some(ci) = found {
                point_to_cluster.push(ci);
            } else {
                point_to_cluster.push(cluster_centers.len());
                cluster_centers.push(*pt);
            }
        }
    }
    let mut cluster_count = vec![0usize; cluster_centers.len()];
    for &ci in &point_to_cluster {
        cluster_count[ci] += 1;
    }

    // Phase 3: mutate the blueprint additively. Push undo first.
    state.push_undo();

    let bp = match state.blueprint.as_mut() {
        Some(b) => b,
        None => {
            state.status_message = Some("No mechanism blueprint — load a sample first or create a new mechanism".to_string());
            state.status_message_time = 4.0;
            return;
        }
    };

    // Find next available link_N ID
    let mut next_link_n = 1;
    while bp.bodies.contains_key(&format!("link_{}", next_link_n)) {
        next_link_n += 1;
    }

    // Create a body for each line (local frame = world frame → attachment
    // points stored in world coordinates). Body pose (0,0,0) then places
    // the points at their DXF positions.
    let mut line_body_ids: Vec<String> = Vec::with_capacity(lines.len());
    for (li, pair) in lines.iter().enumerate() {
        let body_id = format!("link_{}", next_link_n + li);
        let p1 = pair.0;
        let p2 = pair.1;
        let cg = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0];
        let length = ((p2[0] - p1[0]).powi(2) + (p2[1] - p1[1]).powi(2)).sqrt();
        let izz = (1.0 / 12.0) * length.powi(2); // uniform rod about CG

        let mut attachment_points = HashMap::new();
        attachment_points.insert("A".to_string(), p1);
        attachment_points.insert("B".to_string(), p2);

        bp.bodies.insert(body_id.clone(), BodyJson {
            attachment_points,
            mass: 1.0,
            cg_local: cg,
            izz_cg: izz,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: Some(body_id.clone()),
            geometry: None,
        });
        line_body_ids.push(body_id);
    }

    // Add revolute joints for shared endpoints (clusters with 2+ endpoints).
    let mut joint_count_before = bp.joints.len();
    let mut joints_added = 0;
    for (ci, &count) in cluster_count.iter().enumerate() {
        if count < 2 {
            continue;
        }
        let mut members: Vec<(String, &'static str)> = Vec::new();
        for li in 0..lines.len() {
            if point_to_cluster[li * 2] == ci {
                members.push((line_body_ids[li].clone(), "A"));
            }
            if point_to_cluster[li * 2 + 1] == ci {
                members.push((line_body_ids[li].clone(), "B"));
            }
        }
        if members.len() >= 2 {
            let (first_body, first_pt) = (members[0].0.clone(), members[0].1);
            for (other_body, other_pt) in members.iter().skip(1) {
                joint_count_before += 1;
                let mut joint_id = format!("J{}", joint_count_before);
                while bp.joints.contains_key(&joint_id) {
                    joint_count_before += 1;
                    joint_id = format!("J{}", joint_count_before);
                }
                bp.joints.insert(joint_id, JointJson::Revolute {
                    body_i: first_body.clone(),
                    body_j: other_body.clone(),
                    point_i: first_pt.to_string(),
                    point_j: other_pt.to_string(),
                    label: None,
                });
                joints_added += 1;
            }
        }
    }

    // Single rebuild after all mutations.
    state.rebuild();

    state.status_message = Some(format!(
        "Added {} links and {} joints to mechanism. Right-click a grounded joint to set driver.",
        line_body_ids.len(), joints_added
    ));
    state.status_message_time = 5.0;
}

/// Collect unique world points from selected DXF entities.
/// Line endpoints and circle centers are added (snapped to nearby circles).
fn collect_selected_points(state: &super::state::AppState, indices: &[usize]) -> Vec<[f64; 2]> {
    let overlay = match &state.dxf_overlay {
        Some(o) => o,
        None => return Vec::new(),
    };
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];

    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut add_pt = |pts: &mut Vec<[f64; 2]>, p: [f64; 2]| {
        let snapped = snap_to_circle(overlay, p);
        if !pts.iter().any(|q| ((q[0]-snapped[0]).powi(2)+(q[1]-snapped[1]).powi(2)).sqrt() < JOINT_TOL) {
            pts.push(snapped);
        }
    };
    for &idx in indices {
        if let Some(entity) = overlay.entities.iter().find(|e| e.index == idx) {
            match &entity.kind {
                DxfEntityKind::Line { x1, y1, x2, y2 } => {
                    add_pt(&mut pts, [*x1 + ox, *y1 + oy]);
                    add_pt(&mut pts, [*x2 + ox, *y2 + oy]);
                }
                DxfEntityKind::Circle { cx, cy, .. } => {
                    add_pt(&mut pts, [*cx + ox, *cy + oy]);
                }
                DxfEntityKind::Arc { cx, cy, .. } => {
                    add_pt(&mut pts, [*cx + ox, *cy + oy]);
                }
                DxfEntityKind::Point { x, y } => {
                    add_pt(&mut pts, [*x + ox, *y + oy]);
                }
            }
        }
    }
    pts
}

/// Convert all selected entities into a single multi-point rigid body.
/// **Additive**: the body is added to the existing mechanism without
/// replacing anything. The body has no joints — use Fixed joint tool to
/// rigidly attach it to another link.
fn convert_selected_to_single_body(state: &mut super::state::AppState, indices: &[usize]) {
    let pts_world = collect_selected_points(state, indices);

    if pts_world.len() < 2 {
        state.status_message = Some("Need at least 2 unique points to make a body".to_string());
        state.status_message_time = 3.0;
        return;
    }

    state.push_undo();

    let bp = match state.blueprint.as_mut() {
        Some(b) => b,
        None => {
            state.status_message = Some("No mechanism blueprint — load a sample first".to_string());
            state.status_message_time = 4.0;
            return;
        }
    };

    // Find next unique body ID
    let mut next_n = 1;
    while bp.bodies.contains_key(&format!("rigid_{}", next_n)) {
        next_n += 1;
    }
    let body_id = format!("rigid_{}", next_n);

    // Body local frame = world frame → attachment points are world coords
    let mut attachment_points = HashMap::new();
    let names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
    for (i, pt) in pts_world.iter().enumerate() {
        let name = if i < names.len() { names[i].to_string() } else { format!("P{}", i) };
        attachment_points.insert(name, *pt);
    }

    let n = pts_world.len() as f64;
    let cg_x: f64 = pts_world.iter().map(|p| p[0]).sum::<f64>() / n;
    let cg_y: f64 = pts_world.iter().map(|p| p[1]).sum::<f64>() / n;

    let mut max_r_sq = 0.0_f64;
    for pt in &pts_world {
        let r_sq = (pt[0] - cg_x).powi(2) + (pt[1] - cg_y).powi(2);
        if r_sq > max_r_sq { max_r_sq = r_sq; }
    }
    let izz = max_r_sq.max(0.0001);

    bp.bodies.insert(body_id.clone(), BodyJson {
        attachment_points,
        mass: 1.0,
        cg_local: [cg_x, cg_y],
        izz_cg: izz,
        mount_points: HashMap::new(),
        coupler_points: HashMap::new(),
        point_masses: Vec::new(),
        label: Some(body_id.clone()),
        geometry: None,
    });

    state.rebuild();

    state.status_message = Some(format!(
        "Added rigid body '{}' with {} attachment points. Use Fixed joint to rigidly attach it to another link.",
        body_id, pts_world.len()
    ));
    state.status_message_time = 6.0;
}

/// Compute the world-space bounding box of selected entities.
/// Includes line endpoints and circle/arc full extents.
fn compute_bounding_box(state: &super::state::AppState, indices: &[usize]) -> Option<(f64, f64, f64, f64)> {
    let overlay = state.dxf_overlay.as_ref()?;
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let mut found = false;

    for &idx in indices {
        if let Some(entity) = overlay.entities.iter().find(|e| e.index == idx) {
            match &entity.kind {
                DxfEntityKind::Line { x1, y1, x2, y2 } => {
                    let (wx1, wy1, wx2, wy2) = (*x1 + ox, *y1 + oy, *x2 + ox, *y2 + oy);
                    xmin = xmin.min(wx1.min(wx2));
                    xmax = xmax.max(wx1.max(wx2));
                    ymin = ymin.min(wy1.min(wy2));
                    ymax = ymax.max(wy1.max(wy2));
                    found = true;
                }
                DxfEntityKind::Circle { cx, cy, radius } => {
                    let (wcx, wcy) = (*cx + ox, *cy + oy);
                    xmin = xmin.min(wcx - radius);
                    xmax = xmax.max(wcx + radius);
                    ymin = ymin.min(wcy - radius);
                    ymax = ymax.max(wcy + radius);
                    found = true;
                }
                DxfEntityKind::Arc { cx, cy, radius, .. } => {
                    let (wcx, wcy) = (*cx + ox, *cy + oy);
                    xmin = xmin.min(wcx - radius);
                    xmax = xmax.max(wcx + radius);
                    ymin = ymin.min(wcy - radius);
                    ymax = ymax.max(wcy + radius);
                    found = true;
                }
                DxfEntityKind::Point { x, y } => {
                    let (wx, wy) = (*x + ox, *y + oy);
                    xmin = xmin.min(wx);
                    xmax = xmax.max(wx);
                    ymin = ymin.min(wy);
                    ymax = ymax.max(wy);
                    found = true;
                }
            }
        }
    }

    if found { Some((xmin, ymin, xmax, ymax)) } else { None }
}

/// Set BodyGeometry on an existing link from the bounding box of selected
/// DXF entities. Equivalent to the Link Editor's "Add Geometry" button,
/// but with dimensions computed from the DXF selection instead of a
/// manual rectangle draw.
///
/// `target_body` is chosen by the user in the "pick target link" popup.
/// Does NOT create a new body, add joints, or modify attachment points —
/// it just sets the target link's `geometry` field.
fn convert_selected_to_rigid_geometry(
    state: &mut super::state::AppState,
    indices: &[usize],
    target_body: String,
) {
    // 1. Compute the bounding box of the DXF selection in world coords
    let bbox = match compute_bounding_box(state, indices) {
        Some(b) => b,
        None => {
            state.status_message = Some("No geometry in selection".to_string());
            state.status_message_time = 3.0;
            return;
        }
    };
    let (xmin, ymin, xmax, ymax) = bbox;
    let width = xmax - xmin;
    let height = ymax - ymin;
    let cx_world = (xmin + xmax) / 2.0;
    let cy_world = (ymin + ymax) / 2.0;

    if width <= 0.0 || height <= 0.0 {
        state.status_message = Some("Selection has zero extent — select entities with area".to_string());
        state.status_message_time = 3.0;
        return;
    }

    // 2. Convert the bounding box center from world to the target body's
    //    local frame, so the geometry moves with the body as it articulates.
    let offset_local = {
        let mech = match state.mechanism.as_ref() {
            Some(m) => m,
            None => {
                state.status_message = Some("Mechanism not built — try again after building".to_string());
                state.status_message_time = 3.0;
                return;
            }
        };
        let (bx, by, btheta) = mech.state().get_pose(&target_body, &state.q);
        let cos_t = btheta.cos();
        let sin_t = btheta.sin();
        let dx = cx_world - bx;
        let dy = cy_world - by;
        // Inverse rotation: world → local
        nalgebra::Vector2::new(
            cos_t * dx + sin_t * dy,
            -sin_t * dx + cos_t * dy,
        )
    };

    // 3. Push undo and set the target body's geometry directly on the
    //    blueprint. No new body, no new joints.
    state.push_undo();

    if let Some(bp) = state.blueprint.as_mut() {
        if let Some(body) = bp.bodies.get_mut(&target_body) {
            body.geometry = Some(BodyGeometry {
                width,
                height,
                offset: offset_local,
            });
        }
    }

    state.rebuild();

    state.status_message = Some(format!(
        "Added {:.0}x{:.0} mm geometry to '{}' from DXF selection (rigidly attached — moves with the link).",
        width * 1000.0, height * 1000.0, target_body
    ));
    state.status_message_time = 6.0;
}

/// Mark the endpoints of selected entities as ground pivots.
/// **Additive**: adds to the existing ground body.
fn convert_selected_to_ground(state: &mut super::state::AppState, indices: &[usize]) {
    let pivot_pts = collect_selected_points(state, indices);

    if pivot_pts.is_empty() {
        state.status_message = Some("No entities selected".to_string());
        state.status_message_time = 3.0;
        return;
    }

    state.push_undo();

    let bp = match state.blueprint.as_mut() {
        Some(b) => b,
        None => {
            state.status_message = Some("No mechanism blueprint".to_string());
            state.status_message_time = 3.0;
            return;
        }
    };

    // Get or create ground body
    let ground = bp.bodies.entry("ground".to_string()).or_insert_with(|| BodyJson {
        attachment_points: HashMap::new(),
        mass: 0.0,
        cg_local: [0.0, 0.0],
        izz_cg: 0.0,
        mount_points: HashMap::new(),
        coupler_points: HashMap::new(),
        point_masses: Vec::new(),
        label: Some("ground".to_string()),
        geometry: None,
    });

    let start_idx = ground.attachment_points.len();
    let added_count = pivot_pts.len();
    for (i, pt) in pivot_pts.iter().enumerate() {
        ground.attachment_points.insert(format!("P{}", start_idx + i + 1), *pt);
    }

    state.rebuild();

    state.status_message = Some(format!("Added {} ground pivots from selected entities", added_count));
    state.status_message_time = 4.0;
}

/// Convert a single selected DXF line into a LinearActuator force element.
/// The line endpoints are matched to the nearest existing body attachment
/// points in world space. Creates the actuator with force=0 (user can set
/// force in the property panel).
fn convert_selected_line_to_actuator(state: &mut super::state::AppState, indices: &[usize]) {
    // Phase 1: find the first selected line's endpoints
    let (p1, p2) = {
        let overlay = match &state.dxf_overlay {
            Some(o) => o,
            None => return,
        };
        let ox = overlay.offset[0];
        let oy = overlay.offset[1];

        let found = indices.iter()
            .filter_map(|idx| overlay.entities.iter().find(|e| e.index == *idx))
            .find_map(|e| if let DxfEntityKind::Line { x1, y1, x2, y2 } = &e.kind {
                Some((
                    snap_to_circle(overlay, [*x1 + ox, *y1 + oy]),
                    snap_to_circle(overlay, [*x2 + ox, *y2 + oy]),
                ))
            } else { None });

        match found {
            Some(l) => l,
            None => {
                state.status_message = Some("Select a DXF line to convert to an actuator".to_string());
                state.status_message_time = 3.0;
                return;
            }
        }
    };

    // Phase 2: find nearest body attachment point to each endpoint.
    //
    // Strategy:
    //   1. Look for any body point within SNAP_TOL meters (10 mm) — prefer
    //      ground if multiple hits.
    //   2. If nothing found for an endpoint, fall back to creating a new
    //      ground pivot at that endpoint position.
    const SNAP_TOL: f64 = 0.010;

    let find_nearest = |target: [f64; 2]| -> Option<(String, String, [f64; 2])> {
        let mech = state.mechanism.as_ref()?;
        let bp = state.blueprint.as_ref()?;
        let q = &state.q;

        // Pass 1: ground body (preferred)
        if let Some(ground) = bp.bodies.get("ground") {
            let mut best: Option<(f64, String, [f64; 2])> = None;
            for (point_name, local_pt) in &ground.attachment_points {
                let local_vec = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let world = mech.state().body_point_global("ground", &local_vec, q);
                let dist = ((world.x - target[0]).powi(2) + (world.y - target[1]).powi(2)).sqrt();
                if dist < SNAP_TOL && (best.is_none() || dist < best.as_ref().unwrap().0) {
                    best = Some((dist, point_name.clone(), *local_pt));
                }
            }
            if let Some((_, name, lpt)) = best {
                return Some(("ground".to_string(), name, lpt));
            }
        }

        // Pass 2: any other body
        let mut best: Option<(f64, String, String, [f64; 2])> = None;
        for (body_id, body_json) in &bp.bodies {
            if body_id == "ground" { continue; }
            for (point_name, local_pt) in &body_json.attachment_points {
                let local_vec = nalgebra::Vector2::new(local_pt[0], local_pt[1]);
                let world = mech.state().body_point_global(body_id, &local_vec, q);
                let dist = ((world.x - target[0]).powi(2) + (world.y - target[1]).powi(2)).sqrt();
                if dist < SNAP_TOL && (best.is_none() || dist < best.as_ref().unwrap().0) {
                    best = Some((dist, body_id.clone(), point_name.clone(), *local_pt));
                }
            }
        }
        best.map(|(_, bid, pname, lpt)| (bid, pname, lpt))
    };

    let mut attach_a = find_nearest(p1);
    let mut attach_b = find_nearest(p2);

    state.push_undo();

    // Auto-create a ground pivot for any endpoint that didn't find a match.
    let mut create_ground_pivot = |world_pt: [f64; 2]| -> (String, String, [f64; 2]) {
        if let Some(bp) = state.blueprint.as_mut() {
            let ground = bp.bodies.entry("ground".to_string()).or_insert_with(|| BodyJson {
                attachment_points: HashMap::new(),
                mass: 0.0,
                cg_local: [0.0, 0.0],
                izz_cg: 0.0,
                mount_points: HashMap::new(),
                coupler_points: HashMap::new(),
                point_masses: Vec::new(),
                label: Some("ground".to_string()),
                geometry: None,
            });
            // Unique name (Pn)
            let mut n = ground.attachment_points.len() + 1;
            loop {
                let name = format!("P{}", n);
                if !ground.attachment_points.contains_key(&name) {
                    ground.attachment_points.insert(name.clone(), world_pt);
                    return ("ground".to_string(), name, world_pt);
                }
                n += 1;
            }
        }
        ("ground".to_string(), "P1".to_string(), world_pt)
    };

    if attach_a.is_none() {
        attach_a = Some(create_ground_pivot(p1));
    }
    if attach_b.is_none() {
        attach_b = Some(create_ground_pivot(p2));
    }

    let (body_a, name_a, local_a) = attach_a.unwrap();
    let (body_b, name_b, local_b) = attach_b.unwrap();

    if body_a == body_b && body_a != "ground" {
        state.status_message = Some("Actuator endpoints resolved to the same body — select a line that spans two different bodies".to_string());
        state.status_message_time = 4.0;
        return;
    }

    // Phase 3: add the LinearActuator force element
    let actuator = LinearActuatorElement {
        body_a: body_a.clone(),
        point_a: local_a,
        point_a_name: Some(name_a),
        body_b: body_b.clone(),
        point_b: local_b,
        point_b_name: Some(name_b),
        force: 0.0,
        speed_limit: 0.0,
        stroke_min: 0.0,
        stroke_max: 0.0,
        end_stop_stiffness: 10000.0,
        end_stop_damping: 10.0,
        end_stop_restitution: 0.5,
    };

    if let Some(bp) = state.blueprint.as_mut() {
        bp.forces.push(ForceElement::LinearActuator(actuator));
    }

    state.rebuild();

    state.status_message = Some(format!(
        "Added linear actuator between {} and {}. Force=0 — set it in the Property panel.",
        body_a, body_b
    ));
    state.status_message_time = 6.0;
}

// ── Hit-testing for entity selection ────────────────────────────────────────

/// Check if a world position hits a DXF entity (for click-to-select in assignment mode).
/// Returns the index of the nearest entity within threshold, if any.
pub fn hit_test_dxf_entity(overlay: &DxfOverlay, world_x: f64, world_y: f64, threshold: f64) -> Option<usize> {
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];
    let mut best: Option<(f64, usize)> = None;

    for entity in &overlay.entities {
        let dist = match &entity.kind {
            DxfEntityKind::Line { x1, y1, x2, y2 } => {
                point_to_segment_dist(world_x, world_y, *x1 + ox, *y1 + oy, *x2 + ox, *y2 + oy)
            }
            DxfEntityKind::Circle { cx, cy, radius } => {
                let d = ((world_x - *cx - ox).powi(2) + (world_y - *cy - oy).powi(2)).sqrt();
                (d - *radius).abs() // distance to circle boundary
            }
            DxfEntityKind::Arc { cx, cy, radius, .. } => {
                let d = ((world_x - *cx - ox).powi(2) + (world_y - *cy - oy).powi(2)).sqrt();
                (d - *radius).abs()
            }
            DxfEntityKind::Point { x, y } => {
                ((world_x - *x - ox).powi(2) + (world_y - *y - oy).powi(2)).sqrt()
            }
        };

        if dist < threshold {
            if best.is_none() || dist < best.unwrap().0 {
                best = Some((dist, entity.index));
            }
        }
    }

    best.map(|(_, idx)| idx)
}

fn point_to_segment_dist(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-15 {
        return ((px - x1).powi(2) + (py - y1).powi(2)).sqrt();
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}
