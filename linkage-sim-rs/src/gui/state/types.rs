// ── Editor tool and small types ───────────────────────────────────────────────

use std::collections::HashMap;

// ── Alignment guides ─────────────────────────────────────────────────────────

/// Axis for an alignment guide line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentAxis {
    /// Same y-value -- horizontal guide line.
    Horizontal,
    /// Same x-value -- vertical guide line.
    Vertical,
}

/// A snap-alignment guide shown when a dragged point lines up with another
/// attachment point horizontally or vertically.
#[derive(Debug, Clone)]
pub struct AlignmentGuide {
    /// Whether this guide is horizontal (same y) or vertical (same x).
    pub axis: AlignmentAxis,
    /// The world coordinate value: y for Horizontal, x for Vertical.
    pub world_value: f64,
    /// Label of the aligned attachment point (e.g. "ground/A").
    pub label: String,
}

/// Joint type for the two-click creation flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingJointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// Active editor tool — determines what happens on canvas clicks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorTool {
    /// Default: click to select entities. Drag empty space to pan.
    Select,
    /// Draw Link: click a point (or empty space for ground pivot), drag to
    /// another point → creates bar + auto-creates revolute joints at both
    /// ends if they connect to existing points.
    DrawLink,
    /// Multi-click body placement: click to place attachment points, then
    /// confirm to create a body with those points.
    AddBody,
    /// Click canvas to place a new ground pivot.
    AddGroundPivot,
    /// Two-click placement of a two-point force element.
    PlaceForce,
    /// Drag-to-define force zone creation: click and drag a rectangle.
    CreateForceZone,
    /// Click on a link to place a point mass at that position.
    PlaceMass,
}

// ── Context menu target ──────────────────────────────────────────────────────

/// Stores what was under the cursor when a right-click occurred.
///
/// egui's `context_menu()` closure runs every frame while the menu is open,
/// but `secondary_clicked()` is only true on the trigger frame. This struct
/// persists the hit-test result so the menu content stays correct.
#[derive(Debug, Clone, Default)]
pub struct ContextMenuTarget {
    /// Joint ID if right-click landed on a joint.
    pub joint_id: Option<String>,
    /// Attachment point under cursor (body_id, point_name).
    /// Takes priority over body_area.
    pub attachment_point: Option<(String, String)>,
    /// Body area under cursor (body_id) -- only set when no
    /// attachment point is within HIT_RADIUS.
    pub body_area: Option<String>,
    /// World coordinates of the right-click position.
    pub world_pos: Option<[f64; 2]>,
}

// ── Selection ─────────────────────────────────────────────────────────────────

/// Which entity in the mechanism is currently selected for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectedEntity {
    Body(String),
    Joint(String),
    Driver(String),
}

// ── Validation warnings ──────────────────────────────────────────────────────

/// Lightweight validation warnings computed after each rebuild.
#[derive(Clone, Debug, Default)]
pub struct ValidationWarnings {
    /// Grubler DOF mismatch: expected 0 for a fully-constrained driven mechanism.
    pub dof_warning: Option<String>,
    /// Bodies not connected by any joint.
    pub disconnected_bodies: Vec<String>,
    /// Mechanism has no driver.
    pub missing_driver: bool,
}

// ── Solver status ─────────────────────────────────────────────────────────────

/// Summary of the most recent solver call.
#[derive(Debug, Clone)]
pub struct SolverStatus {
    pub converged: bool,
    pub residual_norm: f64,
    pub iterations: usize,
}

impl Default for SolverStatus {
    fn default() -> Self {
        Self {
            converged: false,
            residual_norm: 0.0,
            iterations: 0,
        }
    }
}

// ── Force results ─────────────────────────────────────────────────────────────

/// Results from static/inverse-dynamics force computation at the current pose.
#[derive(Debug, Clone, Default)]
pub struct ForceResults {
    /// Driver torque (N*m) -- the required input effort from the driver constraint.
    pub driver_torque: Option<f64>,
    /// Per-joint reaction forces in global frame: joint_id -> (Fx, Fy) in Newtons.
    pub joint_reactions: HashMap<String, (f64, f64)>,
    /// Condition number of the constraint Jacobian at this pose.
    pub condition_number: Option<f64>,
    /// True if the statics solver detected an overconstrained system.
    pub is_overconstrained: bool,
    /// Mechanical advantage (output/input angular velocity ratio) at current pose.
    pub mechanical_advantage: Option<f64>,
    /// Per-element force contribution norms at the current pose.
    pub force_contributions: Vec<(String, f64)>,
    /// Virtual work cross-check result: (vw_torque, lagrange_torque, agrees).
    pub virtual_work_check: Option<(f64, f64, bool)>,
}
