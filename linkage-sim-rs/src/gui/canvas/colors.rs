//! Color and sizing constants for the 2D canvas — CAD-inspired dark palette.

use eframe::egui::Color32;

use crate::analysis::gravity_breakdown::Classification;

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
pub const WEIGHT_COLOR: Color32 = Color32::from_rgb(255, 200, 50); // weight (point mass) gold

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
pub const FORCE_ARROW_COLOR: Color32 = Color32::from_rgb(255, 220, 60); // yellow — visible against red force zones
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

// Payload weights: helping / hurting / neutral (gravity_breakdown::classify).
// Brightness also differs (grayscale ~186 / ~100 / ~141) so Nathan Mode keeps
// the three classes apart.
pub const WEIGHT_HELPING_COLOR: Color32 = Color32::from_rgb(110, 235, 140);
pub const WEIGHT_HURTING_COLOR: Color32 = Color32::from_rgb(220, 50, 50);
pub const WEIGHT_NEUTRAL_COLOR: Color32 = Color32::from_rgb(140, 140, 150);

/// Colour of a weight that is helping (green), hurting (red) or neutral
/// (gray) at a sample. The single palette for the canvas weight arrows and
/// the Weight Breakdown plot lines, so the two always agree.
pub fn classification_color(class: Classification) -> Color32 {
    match class {
        Classification::Helping => WEIGHT_HELPING_COLOR,
        Classification::Hurting => WEIGHT_HURTING_COLOR,
        Classification::Neutral => WEIGHT_NEUTRAL_COLOR,
    }
}

/// Convert a color to grayscale (for Nathan Mode).
pub fn to_grayscale(c: Color32) -> Color32 {
    let lum = (c.r() as f32 * 0.299 + c.g() as f32 * 0.587 + c.b() as f32 * 0.114) as u8;
    Color32::from_rgba_premultiplied(lum, lum, lum, c.a())
}

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
/// Radius of a weight (point mass) marker, in screen pixels.
pub const WEIGHT_RADIUS: f32 = 5.0;
/// Pick radius around a weight marker's centre, in screen pixels. Weights
/// are picked before joints and pins (see `handle_click_selection`), so this
/// stays below `HIT_RADIUS`: a joint under a weight is still reachable from
/// its outer ring.
pub const WEIGHT_HIT_RADIUS: f32 = 8.0;
/// Pick radius, in screen pixels, for choosing a link by clicking or dropping
/// near its bar: Place Mass, Move to Link and dropping a dragged weight.
pub const LINK_PICK_RADIUS: f32 = 60.0;
pub const ATTACHMENT_DOT_RADIUS: f32 = 3.5;
pub const MOUNT_POINT_RADIUS: f32 = 4.0;
pub const ZOOM_FACTOR: f32 = 1.05;
pub const MIN_SCALE: f32 = 10.0;
pub const MAX_SCALE: f32 = 100_000.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::Classification::{Helping, Hurting, Neutral};

    #[test]
    fn classification_colors_are_green_red_and_gray() {
        assert_eq!(classification_color(Helping), WEIGHT_HELPING_COLOR);
        assert_eq!(classification_color(Hurting), WEIGHT_HURTING_COLOR);
        assert_eq!(classification_color(Neutral), WEIGHT_NEUTRAL_COLOR);
        let green = WEIGHT_HELPING_COLOR;
        assert!(green.g() > green.r() && green.g() > green.b(), "helping is green: {green:?}");
        let red = WEIGHT_HURTING_COLOR;
        assert!(red.r() > red.g() && red.r() > red.b(), "hurting is red: {red:?}");
        let gray = WEIGHT_NEUTRAL_COLOR;
        let (lo, hi) = (gray.r().min(gray.g()).min(gray.b()), gray.r().max(gray.g()).max(gray.b()));
        assert!(hi - lo <= 16, "neutral is gray: {gray:?}");
    }

    /// Nathan Mode draws everything in grayscale: the three classes must
    /// stay apart by brightness (helping brightest, hurting darkest).
    #[test]
    fn classification_colors_stay_distinct_in_grayscale() {
        let lum = |c: Color32| i32::from(to_grayscale(c).r());
        let (help, neutral, hurt) = (lum(WEIGHT_HELPING_COLOR), lum(WEIGHT_NEUTRAL_COLOR), lum(WEIGHT_HURTING_COLOR));
        assert!(help - neutral >= 30 && neutral - hurt >= 30, "grayscale {help} / {neutral} / {hurt}");
    }
}
