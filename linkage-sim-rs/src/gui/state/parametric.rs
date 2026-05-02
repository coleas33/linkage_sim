// ── Parametric study ──────────────────────────────────────────────────────────

use crate::gui::sweep::SweepData;

/// A parameter that can be swept in a parametric study.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SweepParameter {
    /// Body mass (kg). Value: body_id.
    BodyMass(String),
    /// Body moment of inertia about CG (kg*m^2). Value: body_id.
    BodyIzz(String),
    /// Attachment point X coordinate (m). Value: (body_id, point_name).
    AttachmentX(String, String),
    /// Attachment point Y coordinate (m). Value: (body_id, point_name).
    AttachmentY(String, String),
    /// Force element scalar parameter. Value: (force_index, field_name).
    ForceParam(usize, String),
    /// Driver angular velocity (rad/s).
    DriverOmega,
}

impl SweepParameter {
    /// Human-readable label for the parameter.
    pub fn label(&self) -> String {
        match self {
            Self::BodyMass(id) => format!("{} mass (kg)", id),
            Self::BodyIzz(id) => format!("{} Izz (kg*m^2)", id),
            Self::AttachmentX(body, pt) => format!("{}.{} x (m)", body, pt),
            Self::AttachmentY(body, pt) => format!("{}.{} y (m)", body, pt),
            Self::ForceParam(idx, field) => format!("Force[{}].{}", idx, field),
            Self::DriverOmega => "Driver omega (rad/s)".to_string(),
        }
    }

    /// Short unit suffix for DragValue inputs (e.g. " kg", " m").
    pub fn unit_suffix(&self) -> &'static str {
        match self {
            Self::BodyMass(_) => " kg",
            Self::BodyIzz(_) => " kg\u{b7}m\u{b2}",
            Self::AttachmentX(_, _) | Self::AttachmentY(_, _) => " m",
            Self::ForceParam(_, field) => match field.as_str() {
                "stiffness" => " N/m",
                "free_length" | "extended_length" | "stroke" => " m",
                "free_angle" => " rad",
                "damping" => " N\u{b7}s/m",
                "initial_force" | "force" | "force_x" | "force_y" => " N",
                "torque" | "stall_torque" | "constant_drag" => " N\u{b7}m",
                "no_load_speed" | "speed_limit" => " rad/s",
                "viscous_coeff" | "coulomb_coeff" => "",
                _ => "",
            },
            Self::DriverOmega => " rad/s",
        }
    }

    /// Whether this parameter represents a physical quantity that must be
    /// strictly positive (mass, stiffness, etc.).
    pub fn requires_positive(&self) -> bool {
        match self {
            Self::BodyMass(_) | Self::BodyIzz(_) => true,
            Self::ForceParam(_, field) => matches!(
                field.as_str(),
                "stiffness"
                    | "free_length"
                    | "extended_length"
                    | "stroke"
                    | "damping"
                    | "initial_force"
                    | "no_load_speed"
                    | "speed_limit"
            ),
            _ => false,
        }
    }
}

/// Which output metric to plot on the Y-axis of a parametric study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ParametricMetric {
    PeakDriverTorque,
    RmsDriverTorque,
    MinTransmissionAngle,
    MaxTransmissionAngle,
    PeakReaction,
    PeakKineticEnergy,
    MeanMechanicalAdvantage,
}

impl ParametricMetric {
    pub fn label(&self) -> &'static str {
        match self {
            Self::PeakDriverTorque => "Peak Driver Torque (N*m)",
            Self::RmsDriverTorque => "RMS Driver Torque (N*m)",
            Self::MinTransmissionAngle => "Min Transmission Angle (deg)",
            Self::MaxTransmissionAngle => "Max Transmission Angle (deg)",
            Self::PeakReaction => "Peak Joint Reaction (N)",
            Self::PeakKineticEnergy => "Peak Kinetic Energy (J)",
            Self::MeanMechanicalAdvantage => "Mean Mechanical Advantage",
        }
    }

    pub fn all() -> &'static [ParametricMetric] {
        &[
            Self::PeakDriverTorque,
            Self::RmsDriverTorque,
            Self::MinTransmissionAngle,
            Self::MaxTransmissionAngle,
            Self::PeakReaction,
            Self::PeakKineticEnergy,
            Self::MeanMechanicalAdvantage,
        ]
    }

    /// Extract a scalar value from a sweep dataset for this metric.
    pub fn extract(&self, sweep: &SweepData) -> f64 {
        match self {
            Self::PeakDriverTorque => sweep
                .driver_torques
                .as_ref()
                .map(|v| v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max))
                .unwrap_or(0.0),
            Self::RmsDriverTorque => sweep
                .driver_torques
                .as_ref()
                .map(|v| {
                    let n = v.len() as f64;
                    if n == 0.0 { return 0.0; }
                    (v.iter().map(|x| x * x).sum::<f64>() / n).sqrt()
                })
                .unwrap_or(0.0),
            Self::MinTransmissionAngle => sweep
                .transmission_angles
                .as_ref()
                .map(|v| v.iter().copied().fold(f64::INFINITY, f64::min))
                .unwrap_or(0.0),
            Self::MaxTransmissionAngle => sweep
                .transmission_angles
                .as_ref()
                .map(|v| v.iter().copied().fold(0.0_f64, f64::max))
                .unwrap_or(0.0),
            Self::PeakReaction => sweep
                .joint_reaction_magnitudes
                .values()
                .flat_map(|v| v.iter().copied())
                .fold(0.0_f64, f64::max),
            Self::PeakKineticEnergy => sweep
                .kinetic_energy
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
            Self::MeanMechanicalAdvantage => {
                let v = &sweep.mechanical_advantage;
                if v.is_empty() { return 0.0; }
                // Filter out extreme values near toggle
                let filtered: Vec<f64> = v.iter().copied().filter(|x| x.abs() < 1e6).collect();
                if filtered.is_empty() { return 0.0; }
                filtered.iter().sum::<f64>() / filtered.len() as f64
            }
        }
    }
}

/// Configuration for a parametric study.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParametricStudyConfig {
    pub parameter: SweepParameter,
    pub min_value: f64,
    pub max_value: f64,
    pub num_steps: usize,
    pub metric: ParametricMetric,
}

/// Results of a parametric study.
#[derive(Debug, Clone)]
pub struct ParametricStudyResult {
    pub config: ParametricStudyConfig,
    /// Parameter values at each step.
    pub parameter_values: Vec<f64>,
    /// Extracted metric value at each step.
    pub metric_values: Vec<f64>,
    /// Full sweep data for the selected (hovered/clicked) parameter value, if any.
    pub selected_sweep: Option<(f64, SweepData)>,
}

// ── Counterbalance assistant ──────────────────────────────────────────────────

/// Configuration for a counterbalance spring optimization.
#[derive(Debug, Clone)]
pub struct CounterbalanceConfig {
    /// Body A for the spring attachment (e.g., ground).
    pub body_a: String,
    /// Attachment point name on body A.
    pub point_a: String,
    /// Body B for the spring attachment (e.g., coupler).
    pub body_b: String,
    /// Attachment point name on body B.
    pub point_b: String,
    /// Minimum spring stiffness to search (N/m).
    pub k_min: f64,
    /// Maximum spring stiffness to search (N/m).
    pub k_max: f64,
    /// Number of stiffness steps.
    pub k_steps: usize,
    /// Minimum free length to search (m).
    pub free_length_min: f64,
    /// Maximum free length to search (m).
    pub free_length_max: f64,
    /// Number of free length steps.
    pub free_length_steps: usize,
}

/// Results from a counterbalance optimization.
#[derive(Debug, Clone)]
pub struct CounterbalanceResult {
    /// Optimal spring stiffness (N/m).
    pub best_k: f64,
    /// Optimal free length (m).
    pub best_free_length: f64,
    /// Peak-to-peak torque with the optimal spring (N*m).
    pub best_peak_to_peak: f64,
    /// Peak-to-peak torque without any counterbalance spring (N*m).
    pub baseline_peak_to_peak: f64,
    /// Driver torques over the sweep WITHOUT the spring (baseline).
    pub baseline_torques: Vec<f64>,
    /// Driver torques over the sweep WITH the optimal spring.
    pub optimized_torques: Vec<f64>,
    /// Sweep angles in degrees (shared by both torque curves).
    pub angles_deg: Vec<f64>,
    /// Full grid of peak-to-peak values: [k_idx][fl_idx].
    pub grid: Vec<Vec<f64>>,
    /// K values searched.
    pub k_values: Vec<f64>,
    /// Free length values searched.
    pub fl_values: Vec<f64>,
}
