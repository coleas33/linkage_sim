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

/// Draw the DXF import settings and body assignment panel.
pub fn draw_dxf_panel(ui: &mut egui::Ui, state: &mut super::state::AppState) {
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

    // ── Body assignment ─────────────────────────────────────────────────
    ui.separator();
    ui.heading("Body Assignment");

    // Show existing assignments
    for (i, assignment) in overlay.assignments.iter().enumerate() {
        let label = if assignment.is_ground {
            format!("{} (ground) - {} entities", assignment.name, assignment.entity_indices.len())
        } else {
            format!("{} - {} entities", assignment.name, assignment.entity_indices.len())
        };
        ui.label(label);
        // Placeholder for per-assignment controls (toggle ground, delete, etc.)
        let _ = i;
    }

    if overlay.assigning {
        ui.label(format!("Selecting entities for: {}", overlay.next_body_name));
        ui.label(format!("{} selected", overlay.selected_entities.len()));
        ui.horizontal(|ui| {
            if ui.button("Finish Body").clicked() && !overlay.selected_entities.is_empty() {
                let assignment = BodyAssignment {
                    name: overlay.next_body_name.clone(),
                    entity_indices: overlay.selected_entities.iter().copied().collect(),
                    is_ground: false,
                };
                overlay.assignments.push(assignment);
                overlay.selected_entities.clear();
                // Auto-increment name
                let n = overlay.assignments.len() + 1;
                overlay.next_body_name = format!("body_{}", n);
                overlay.assigning = false;
            }
            if ui.button("Cancel").clicked() {
                overlay.selected_entities.clear();
                overlay.assigning = false;
            }
        });
    } else {
        if ui.button("New Body").clicked() {
            overlay.assigning = true;
            overlay.selected_entities.clear();
        }
    }

    // Mark last assignment as ground
    if !overlay.assignments.is_empty() {
        let last = overlay.assignments.len() - 1;
        let is_ground = overlay.assignments[last].is_ground;
        let label = if is_ground { "Unmark Ground" } else { "Mark Last as Ground" };
        if ui.button(label).clicked() {
            overlay.assignments[last].is_ground = !is_ground;
        }
    }

    // Build mechanism from assignments
    ui.separator();
    let can_build = overlay.assignments.len() >= 2
        && overlay.assignments.iter().any(|a| a.is_ground);
    if ui.add_enabled(can_build, egui::Button::new("Build Mechanism"))
        .on_hover_text("Create mechanism from body assignments. Need at least 2 bodies with one marked as ground.")
        .clicked()
    {
        build_mechanism_from_assignments(state);
    }

    // Clear overlay
    ui.separator();
    if ui.button("Clear DXF Overlay").clicked() {
        state.dxf_overlay = None;
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
fn build_mechanism_from_assignments(state: &mut super::state::AppState) {
    let overlay = match &state.dxf_overlay {
        Some(o) => o,
        None => return,
    };

    let ox = overlay.offset[0];
    let oy = overlay.offset[1];

    // For each assignment, collect circle centers as attachment points
    // and determine body position (first circle = origin).
    let mut bodies_json = serde_json::Map::new();
    let mut body_circles: HashMap<String, Vec<[f64; 2]>> = HashMap::new();

    let joint_tolerance = 0.0005; // 0.5mm tolerance for matching joint positions

    for assignment in &overlay.assignments {
        let mut circles_world: Vec<[f64; 2]> = Vec::new();
        for &ei in &assignment.entity_indices {
            if let Some(entity) = overlay.entities.iter().find(|e| e.index == ei) {
                if let DxfEntityKind::Circle { cx, cy, .. } = &entity.kind {
                    circles_world.push([*cx + ox, *cy + oy]);
                }
            }
        }

        if circles_world.is_empty() {
            // No circles -> collect line endpoints as attachment points
            for &ei in &assignment.entity_indices {
                if let Some(entity) = overlay.entities.iter().find(|e| e.index == ei) {
                    if let DxfEntityKind::Line { x1, y1, x2, y2 } = &entity.kind {
                        let p1 = [*x1 + ox, *y1 + oy];
                        let p2 = [*x2 + ox, *y2 + oy];
                        if !circles_world.iter().any(|c| (c[0]-p1[0]).abs() < joint_tolerance && (c[1]-p1[1]).abs() < joint_tolerance) {
                            circles_world.push(p1);
                        }
                        if !circles_world.iter().any(|c| (c[0]-p2[0]).abs() < joint_tolerance && (c[1]-p2[1]).abs() < joint_tolerance) {
                            circles_world.push(p2);
                        }
                    }
                }
            }
        }

        if circles_world.is_empty() {
            continue;
        }

        // For ground: attachment points are in world coordinates
        // For moving bodies: first circle = body origin, rest are local offsets
        let body_id = if assignment.is_ground {
            "ground".to_string()
        } else {
            assignment.name.clone()
        };

        let mut attachment_points = serde_json::Map::new();
        let names = ["A", "B", "C", "D", "E", "F", "G", "H"];

        if assignment.is_ground {
            for (i, pt) in circles_world.iter().enumerate() {
                let name = if i < names.len() { format!("P{}", i + 1) } else { format!("P{}", i + 1) };
                attachment_points.insert(name, serde_json::json!([pt[0], pt[1]]));
            }
        } else {
            // First circle is body origin
            let origin = circles_world[0];
            for (i, pt) in circles_world.iter().enumerate() {
                let name = if i < names.len() { names[i].to_string() } else { format!("P{}", i) };
                let local = [pt[0] - origin[0], pt[1] - origin[1]];
                attachment_points.insert(name, serde_json::json!([local[0], local[1]]));
            }
        }

        // Compute CG and mass
        let n = circles_world.len() as f64;
        let cg_x: f64 = circles_world.iter().map(|c| c[0]).sum::<f64>() / n;
        let cg_y: f64 = circles_world.iter().map(|c| c[1]).sum::<f64>() / n;
        let (cg_local, mass) = if assignment.is_ground {
            ([0.0, 0.0], 0.0)
        } else {
            let origin = circles_world[0];
            ([cg_x - origin[0], cg_y - origin[1]], 1.0)
        };

        let body_json = serde_json::json!({
            "attachment_points": attachment_points,
            "mass": mass,
            "cg_local": cg_local,
            "izz_cg": if assignment.is_ground { 0.0 } else { 0.01 },
            "label": body_id,
        });
        bodies_json.insert(body_id.clone(), body_json);
        body_circles.insert(body_id, circles_world);
    }

    // Auto-detect joints: circles from different bodies at same position
    let mut joints_json = serde_json::Map::new();
    let mut joint_count = 0;
    let body_ids: Vec<String> = body_circles.keys().cloned().collect();

    for i in 0..body_ids.len() {
        for j in (i + 1)..body_ids.len() {
            let id_a = &body_ids[i];
            let id_b = &body_ids[j];
            let circles_a = &body_circles[id_a];
            let circles_b = &body_circles[id_b];

            for (ai, ca) in circles_a.iter().enumerate() {
                for (bi, cb) in circles_b.iter().enumerate() {
                    let dist = ((ca[0] - cb[0]).powi(2) + (ca[1] - cb[1]).powi(2)).sqrt();
                    if dist < joint_tolerance {
                        joint_count += 1;
                        let joint_id = format!("J{}", joint_count);

                        // Find the point names
                        let names_a = ["A", "B", "C", "D", "E", "F", "G", "H"];
                        let point_a = if id_a == "ground" {
                            format!("P{}", ai + 1)
                        } else if ai < names_a.len() {
                            names_a[ai].to_string()
                        } else {
                            format!("P{}", ai)
                        };
                        let point_b = if id_b == "ground" {
                            format!("P{}", bi + 1)
                        } else if bi < names_a.len() {
                            names_a[bi].to_string()
                        } else {
                            format!("P{}", bi)
                        };

                        joints_json.insert(joint_id, serde_json::json!({
                            "type": "revolute",
                            "body_i": id_a,
                            "body_j": id_b,
                            "point_i": point_a,
                            "point_j": point_b,
                        }));
                    }
                }
            }
        }
    }

    // Build MechanismJson
    let mechanism_json = serde_json::json!({
        "schema_version": "1.1.0",
        "bodies": bodies_json,
        "joints": joints_json,
        "drivers": {},
        "forces": [],
    });

    let json_str = serde_json::to_string_pretty(&mechanism_json).unwrap_or_default();
    let _ = state.load_from_json_str(&json_str);
    state.status_message = Some(format!(
        "Built mechanism: {} bodies, {} joints. Right-click a grounded joint to set driver.",
        body_circles.len(), joint_count
    ));
    state.status_message_time = 5.0;
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
