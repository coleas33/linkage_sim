//! DXF import: parse DXF files into a snappable overlay with interactive body assignment.
//!
//! Two modes:
//! - **Overlay (Mode C):** DXF geometry renders on the canvas; circle centers become snap targets.
//! - **Assignment (Mode B):** Sidebar panel for grouping entities into bodies, auto-detecting joints.

use std::collections::{HashMap, HashSet};

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
    ConvertSelectedToSingleBody(Vec<usize>),
    ConvertSelectedToGround(Vec<usize>),
    ConvertAllLinesToLinks,
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

    // Group all selected lines into a single multi-point body (ternary+ link)
    if ui.add_enabled(n_selected >= 1, egui::Button::new(format!("→ Single Body ({} entities)", n_selected)))
        .on_hover_text("All selected entities become ONE rigid body. Good for ternary links or irregular shapes. Unique endpoints become attachment points.")
        .clicked()
    {
        action = Some(DxfAction::ConvertSelectedToSingleBody(
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
            DxfAction::ConvertSelectedToSingleBody(indices) => {
                convert_selected_to_single_body(state, &indices);
            }
            DxfAction::ConvertSelectedToGround(indices) => {
                convert_selected_to_ground(state, &indices);
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
            DxfAction::ClearOverlay => {
                state.dxf_overlay = None;
            }
        }
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
fn convert_selected_lines_to_links(state: &mut super::state::AppState, line_indices: &[usize]) {
    let overlay = match &state.dxf_overlay {
        Some(o) => o,
        None => return,
    };

    let ox = overlay.offset[0];
    let oy = overlay.offset[1];

    // Collect each selected line's endpoints (snapped to circles if near)
    let mut lines: Vec<(usize, [f64; 2], [f64; 2])> = Vec::new();
    for &idx in line_indices {
        if let Some(entity) = overlay.entities.iter().find(|e| e.index == idx) {
            if let DxfEntityKind::Line { x1, y1, x2, y2 } = &entity.kind {
                let p1 = snap_to_circle(overlay, [*x1 + ox, *y1 + oy]);
                let p2 = snap_to_circle(overlay, [*x2 + ox, *y2 + oy]);
                lines.push((idx, p1, p2));
            }
        }
    }

    if lines.is_empty() {
        state.status_message = Some("No lines selected to convert".to_string());
        state.status_message_time = 3.0;
        return;
    }

    // Find shared endpoints: cluster all endpoints within JOINT_TOL of each other
    let mut all_points: Vec<[f64; 2]> = Vec::new();
    for (_, p1, p2) in &lines {
        all_points.push(*p1);
        all_points.push(*p2);
    }

    // Cluster into unique positions
    let mut cluster_centers: Vec<[f64; 2]> = Vec::new();
    let mut point_to_cluster: Vec<usize> = Vec::new();
    for pt in &all_points {
        let mut found_cluster = None;
        for (ci, cc) in cluster_centers.iter().enumerate() {
            let d = ((pt[0] - cc[0]).powi(2) + (pt[1] - cc[1]).powi(2)).sqrt();
            if d < JOINT_TOL {
                found_cluster = Some(ci);
                break;
            }
        }
        if let Some(ci) = found_cluster {
            point_to_cluster.push(ci);
        } else {
            point_to_cluster.push(cluster_centers.len());
            cluster_centers.push(*pt);
        }
    }

    // Count how many line endpoints land on each cluster
    let mut cluster_count = vec![0usize; cluster_centers.len()];
    for &ci in &point_to_cluster {
        cluster_count[ci] += 1;
    }

    // Determine ground pivots: clusters with only one endpoint become free ends.
    // We pick the leftmost/bottom-most single-endpoint clusters and use them
    // as ground pivots, so the mechanism has something fixed. Clusters with
    // 2+ endpoints become revolute joints between bodies.
    let mut ground_clusters: Vec<usize> = cluster_count.iter().enumerate()
        .filter(|(_, c)| **c == 1)
        .map(|(i, _)| i)
        .collect();
    // If nothing qualifies, no ground — user can assign later.
    ground_clusters.sort_by(|a, b| {
        let pa = cluster_centers[*a];
        let pb = cluster_centers[*b];
        pa[0].partial_cmp(&pb[0]).unwrap_or(std::cmp::Ordering::Equal)
            .then(pa[1].partial_cmp(&pb[1]).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Build bodies (one per line)
    let mut bodies_json = serde_json::Map::new();
    let mut line_body_ids: Vec<String> = Vec::with_capacity(lines.len());

    // Ground body with all single-endpoint clusters as attachment points
    let mut ground_pts = serde_json::Map::new();
    for (i, &ci) in ground_clusters.iter().enumerate() {
        let cc = cluster_centers[ci];
        ground_pts.insert(format!("P{}", i + 1), serde_json::json!([cc[0], cc[1]]));
    }
    // Always include a ground body (even if empty — user can add pivots)
    if !ground_pts.is_empty() {
        bodies_json.insert("ground".to_string(), serde_json::json!({
            "attachment_points": ground_pts,
            "mass": 0.0,
            "cg_local": [0.0, 0.0],
            "izz_cg": 0.0,
            "label": "ground",
        }));
    }

    // Map cluster index -> ground point name (if ground)
    let mut ground_cluster_name: HashMap<usize, String> = HashMap::new();
    for (i, &ci) in ground_clusters.iter().enumerate() {
        ground_cluster_name.insert(ci, format!("P{}", i + 1));
    }

    // Create a body for each selected line
    for (li, (_ei, p1, p2)) in lines.iter().enumerate() {
        let body_id = format!("link_{}", li + 1);
        // Local frame: p1 is origin, p2 is at B in local coords
        let local_b = [p2[0] - p1[0], p2[1] - p1[1]];
        let cg = [local_b[0] / 2.0, local_b[1] / 2.0];
        let length = (local_b[0].powi(2) + local_b[1].powi(2)).sqrt();
        let izz = (1.0 / 12.0) * length.powi(2); // uniform rod about CG

        let mut pts = serde_json::Map::new();
        pts.insert("A".to_string(), serde_json::json!([0.0, 0.0]));
        pts.insert("B".to_string(), serde_json::json!([local_b[0], local_b[1]]));

        bodies_json.insert(body_id.clone(), serde_json::json!({
            "attachment_points": pts,
            "mass": 1.0,
            "cg_local": cg,
            "izz_cg": izz,
            "label": body_id,
        }));
        line_body_ids.push(body_id);
    }

    // Build joints from clusters
    let mut joints_json = serde_json::Map::new();
    let mut joint_count = 0;

    for (ci, &count) in cluster_count.iter().enumerate() {
        // Collect all (line_body_id, point_name) that land on this cluster
        let mut members: Vec<(String, &'static str)> = Vec::new();
        for (li, _) in lines.iter().enumerate() {
            let p1_cluster = point_to_cluster[li * 2];
            let p2_cluster = point_to_cluster[li * 2 + 1];
            if p1_cluster == ci {
                members.push((line_body_ids[li].clone(), "A"));
            }
            if p2_cluster == ci {
                members.push((line_body_ids[li].clone(), "B"));
            }
        }

        if count == 1 {
            // Single endpoint → joint to ground (if ground_clusters contains ci)
            if let Some(gname) = ground_cluster_name.get(&ci) {
                let (body_id, point) = &members[0];
                joint_count += 1;
                joints_json.insert(format!("J{}", joint_count), serde_json::json!({
                    "type": "revolute",
                    "body_i": "ground",
                    "body_j": body_id,
                    "point_i": gname,
                    "point_j": point,
                }));
            }
        } else {
            // Multiple endpoints → connect them pairwise (chain all to first)
            if members.len() >= 2 {
                let (first_body, first_pt) = &members[0];
                for (other_body, other_pt) in &members[1..] {
                    joint_count += 1;
                    joints_json.insert(format!("J{}", joint_count), serde_json::json!({
                        "type": "revolute",
                        "body_i": first_body,
                        "body_j": other_body,
                        "point_i": first_pt,
                        "point_j": other_pt,
                    }));
                }
            }
        }
    }

    let mechanism_json = serde_json::json!({
        "schema_version": "1.1.0",
        "bodies": bodies_json,
        "joints": joints_json,
        "drivers": {},
        "forces": [],
    });

    let json_str = serde_json::to_string_pretty(&mechanism_json).unwrap_or_default();
    let n_bodies = line_body_ids.len();
    match state.load_from_json_str(&json_str) {
        Ok(_) => {
            state.status_message = Some(format!(
                "Converted {} lines → {} links, {} joints, {} ground pivots. Right-click a grounded joint to set driver.",
                n_bodies, n_bodies, joint_count, ground_clusters.len()
            ));
        }
        Err(e) => {
            state.status_message = Some(format!("Mechanism build failed: {}", e));
        }
    }
    state.status_message_time = 6.0;
}

/// Convert all selected entities into a single multi-point body.
fn convert_selected_to_single_body(state: &mut super::state::AppState, indices: &[usize]) {
    let overlay = match &state.dxf_overlay {
        Some(o) => o,
        None => return,
    };
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];

    // Collect all unique endpoints from selected entities
    let mut pts_world: Vec<[f64; 2]> = Vec::new();
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
                    add_pt(&mut pts_world, [*x1 + ox, *y1 + oy]);
                    add_pt(&mut pts_world, [*x2 + ox, *y2 + oy]);
                }
                DxfEntityKind::Circle { cx, cy, .. } => {
                    add_pt(&mut pts_world, [*cx + ox, *cy + oy]);
                }
                _ => {}
            }
        }
    }

    if pts_world.len() < 2 {
        state.status_message = Some("Need at least 2 unique points to make a body".to_string());
        state.status_message_time = 3.0;
        return;
    }

    // First point is body origin, others are local offsets
    let origin = pts_world[0];
    let mut attachment_points = serde_json::Map::new();
    let names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
    for (i, pt) in pts_world.iter().enumerate() {
        let name = if i < names.len() { names[i].to_string() } else { format!("P{}", i) };
        attachment_points.insert(name, serde_json::json!([pt[0] - origin[0], pt[1] - origin[1]]));
    }

    // CG = centroid of attachment points in local frame
    let cg_x: f64 = pts_world.iter().map(|p| p[0] - origin[0]).sum::<f64>() / pts_world.len() as f64;
    let cg_y: f64 = pts_world.iter().map(|p| p[1] - origin[1]).sum::<f64>() / pts_world.len() as f64;

    // Estimate characteristic length for inertia
    let mut max_r_sq = 0.0_f64;
    for pt in &pts_world {
        let r_sq = (pt[0] - origin[0] - cg_x).powi(2) + (pt[1] - origin[1] - cg_y).powi(2);
        if r_sq > max_r_sq { max_r_sq = r_sq; }
    }
    let izz = 1.0 * max_r_sq.max(0.0001);

    // Figure out a unique body name
    let next_id = state.mechanism.as_ref()
        .map(|m| m.bodies().len() + 1)
        .unwrap_or(1);
    let body_id = format!("link_{}", next_id);

    let body_json = serde_json::json!({
        "attachment_points": attachment_points,
        "mass": 1.0,
        "cg_local": [cg_x, cg_y],
        "izz_cg": izz,
        "label": body_id,
    });

    // If there's already a mechanism, add to it. Otherwise create a new one
    // with just this body plus a ground.
    let current_json = state.blueprint.as_ref().and_then(|b| serde_json::to_value(b).ok());
    let mut mechanism_json = match current_json {
        Some(v) => v,
        None => serde_json::json!({
            "schema_version": "1.1.0",
            "bodies": { "ground": {
                "attachment_points": {},
                "mass": 0.0,
                "cg_local": [0.0, 0.0],
                "izz_cg": 0.0,
                "label": "ground",
            }},
            "joints": {},
            "drivers": {},
            "forces": [],
        }),
    };

    if let Some(bodies) = mechanism_json.get_mut("bodies").and_then(|v| v.as_object_mut()) {
        bodies.insert(body_id.clone(), body_json);
    }

    let json_str = serde_json::to_string_pretty(&mechanism_json).unwrap_or_default();
    match state.load_from_json_str(&json_str) {
        Ok(_) => {
            state.status_message = Some(format!(
                "Added '{}' as a single body with {} attachment points. Use Draw Link or + Joint Point to connect.",
                body_id, pts_world.len()
            ));
        }
        Err(e) => {
            state.status_message = Some(format!("Mechanism build failed: {}", e));
        }
    }
    state.status_message_time = 6.0;
}

/// Mark the endpoints of selected entities as ground pivots.
fn convert_selected_to_ground(state: &mut super::state::AppState, indices: &[usize]) {
    let overlay = match &state.dxf_overlay {
        Some(o) => o,
        None => return,
    };
    let ox = overlay.offset[0];
    let oy = overlay.offset[1];

    let mut pivot_pts: Vec<[f64; 2]> = Vec::new();
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
                    add_pt(&mut pivot_pts, [*x1 + ox, *y1 + oy]);
                    add_pt(&mut pivot_pts, [*x2 + ox, *y2 + oy]);
                }
                DxfEntityKind::Circle { cx, cy, .. } => {
                    add_pt(&mut pivot_pts, [*cx + ox, *cy + oy]);
                }
                _ => {}
            }
        }
    }

    if pivot_pts.is_empty() {
        state.status_message = Some("No entities selected".to_string());
        state.status_message_time = 3.0;
        return;
    }

    // Add to existing ground body (or create new mechanism with just ground)
    let current_json = state.blueprint.as_ref().and_then(|b| serde_json::to_value(b).ok());
    let mut mechanism_json = match current_json {
        Some(v) => v,
        None => serde_json::json!({
            "schema_version": "1.1.0",
            "bodies": { "ground": {
                "attachment_points": {},
                "mass": 0.0,
                "cg_local": [0.0, 0.0],
                "izz_cg": 0.0,
                "label": "ground",
            }},
            "joints": {},
            "drivers": {},
            "forces": [],
        }),
    };

    let added_count = pivot_pts.len();
    if let Some(bodies) = mechanism_json.get_mut("bodies").and_then(|v| v.as_object_mut()) {
        // Get or create ground body
        if !bodies.contains_key("ground") {
            bodies.insert("ground".to_string(), serde_json::json!({
                "attachment_points": {},
                "mass": 0.0,
                "cg_local": [0.0, 0.0],
                "izz_cg": 0.0,
                "label": "ground",
            }));
        }
        if let Some(ground) = bodies.get_mut("ground").and_then(|v| v.as_object_mut()) {
            if let Some(att) = ground.get_mut("attachment_points").and_then(|v| v.as_object_mut()) {
                let start_idx = att.len();
                for (i, pt) in pivot_pts.iter().enumerate() {
                    att.insert(format!("P{}", start_idx + i + 1), serde_json::json!([pt[0], pt[1]]));
                }
            }
        }
    }

    let json_str = serde_json::to_string_pretty(&mechanism_json).unwrap_or_default();
    match state.load_from_json_str(&json_str) {
        Ok(_) => {
            state.status_message = Some(format!("Added {} ground pivots from selected entities", added_count));
        }
        Err(e) => {
            state.status_message = Some(format!("Failed to add pivots: {}", e));
        }
    }
    state.status_message_time = 4.0;
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
