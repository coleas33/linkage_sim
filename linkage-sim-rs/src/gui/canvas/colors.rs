//! Color and sizing constants for the 2D canvas — CAD-inspired dark palette.

use eframe::egui::Color32;

// ── Colors — CAD-inspired dark palette ───────────────────────────────────────

// Canvas background: subtle gradient-like dark with slight blue tint
pub const BG_COLOR: Color32 = Color32::from_rgb(22, 24, 32);
pub const GRID_COLOR: Color32 = Color32::from_rgba_premultiplied(45, 50, 65, 50);
pub const GRID_MAJOR_COLOR: Color32 = Color32::from_rgba_premultiplied(55, 60, 80, 80);
pub const GROUND_LINE_COLOR: Color32 = Color32::from_rgb(80, 85, 100);

// Bodies: clean blue with warm orange selection (SolidWorks-style)
pub const BODY_COLOR: Color32 = Color32::from_rgb(70, 150, 240);
pub const BODY_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 180, 40);

// Joints: bright with clear hierarchy
pub const JOINT_COLOR: Color32 = Color32::from_rgb(220, 225, 240);
pub const JOINT_SELECTED_COLOR: Color32 = Color32::from_rgb(255, 180, 40);
pub const DRIVER_JOINT_COLOR: Color32 = Color32::from_rgb(80, 220, 130);
pub const GROUND_MARKER_COLOR: Color32 = Color32::from_rgb(160, 145, 110);
pub const ATTACHMENT_DOT_COLOR: Color32 = Color32::from_rgb(170, 185, 210);
pub const MOUNT_POINT_COLOR: Color32 = Color32::from_rgb(224, 86, 253); // #e056fd magenta

// Labels
pub const DEBUG_TEXT_COLOR: Color32 = Color32::from_rgb(150, 160, 180);
pub const DEBUG_DIM_COLOR: Color32 = Color32::from_rgb(85, 90, 105);
pub const NO_MECH_TEXT_COLOR: Color32 = Color32::from_rgb(90, 95, 115);
pub const JOINT_CREATE_HIGHLIGHT: Color32 = Color32::from_rgb(50, 230, 100);
pub const JOINT_HOVER_HIGHLIGHT: Color32 = Color32::from_rgb(100, 200, 255);
pub const DIM_LABEL_COLOR: Color32 = Color32::from_rgb(170, 195, 130);

// Canvas element labels (body names, joint IDs)
pub const LABEL_COLOR: Color32 = Color32::from_gray(136); // #888

// Force elements: semantic color coding
pub const FORCE_ARROW_COLOR: Color32 = Color32::from_rgb(255, 80, 80);
pub const SPRING_COLOR: Color32 = Color32::from_rgb(50, 200, 110);
pub const DAMPER_COLOR: Color32 = Color32::from_rgb(90, 145, 255);
pub const EXT_FORCE_COLOR: Color32 = Color32::from_rgb(255, 160, 30);
pub const GAS_SPRING_COLOR: Color32 = Color32::from_rgb(170, 95, 255);
pub const ACTUATOR_COLOR: Color32 = Color32::from_rgb(255, 115, 55);
pub const BEARING_COLOR: Color32 = Color32::from_rgb(190, 175, 95);
pub const JOINT_LIMIT_COLOR: Color32 = Color32::from_rgb(215, 75, 75);
pub const MOTOR_COLOR: Color32 = Color32::from_rgb(80, 220, 130);
pub const FORCE_ZONE_COLOR: Color32 = Color32::from_rgb(255, 80, 80);
pub const FORCE_ZONE_OVERLAP_FILL: Color32 = Color32::from_rgba_premultiplied(255, 200, 0, 50);
pub const FORCE_ZONE_OVERLAP_STROKE: Color32 = Color32::from_rgb(255, 204, 0);

// ── Sizing ──────────────────────────────────────────────────────────────────

pub const FORCE_ARROW_WIDTH: f32 = 2.5;
pub const FORCE_ARROW_MIN_PX: f32 = 3.0;
pub const FORCE_ARROW_MAX_PX: f32 = 80.0;
pub const FORCE_ARROW_SCALE: f32 = 30.0;

pub const BODY_STROKE_WIDTH: f32 = 3.5;
pub const LINK_HALF_WIDTH: f32 = 8.0;
pub const JOINT_RADIUS: f32 = 7.0;
pub const JOINT_STROKE_WIDTH: f32 = 2.0;
pub const GROUND_MARKER_SIZE: f32 = 14.0;
pub const HIT_RADIUS: f32 = 12.0;
pub const ATTACHMENT_DOT_RADIUS: f32 = 3.5;
pub const MOUNT_POINT_RADIUS: f32 = 4.0;
pub const ZOOM_FACTOR: f32 = 1.05;
pub const MIN_SCALE: f32 = 10.0;
pub const MAX_SCALE: f32 = 100_000.0;
