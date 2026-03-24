// ── Simulation state ──────────────────────────────────────────────────────────

use nalgebra::DVector;

/// Forward dynamics simulation result and playback state.
pub struct SimulationState {
    /// Time history from the simulation.
    pub times: Vec<f64>,
    /// Position history (one q vector per time step).
    pub positions: Vec<DVector<f64>>,
    /// Current playback index into the trajectory.
    pub time_index: usize,
    /// Whether simulation playback is active.
    pub playing: bool,
    /// Playback speed multiplier.
    pub speed: f64,
    /// Accumulated time for playback interpolation.
    pub elapsed: f64,
    /// Constraint drift at each step.
    pub drift: Vec<f64>,
}
