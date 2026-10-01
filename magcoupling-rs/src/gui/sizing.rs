//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward
//! calculation, or Torque → Magnets, inverse sizing by [`crate::engine::sizing::solve`].
//!
//! The mode, the free variable and the target torque are GUI state (A-2 decision A2-9): the
//! engine takes them as arguments. Design files and share links record them (decision M41-4).
//!
//! [`SizingRunner`] runs the solve: on a change, once the design has been still for
//! [`DEBOUNCE_S`] and no edit is in progress (no drag, no typing), never per frame (a solve
//! takes about 1 to 3 ms in a release build, longer on wasm, against compute_all's 17 us). Until
//! then the panel keeps showing the last solved value, marked as solving. The design the panel
//! shows in Torque → Magnets is the inputs with the free variable at the solved value, or at
//! the best value when the target is not reachable (decision M41-8); the inputs keep their own
//! value of the free variable until the user leaves the mode (decision M41-7), when a change
//! still waiting for its debounce is solved at once ([`SizingRunner::solve_now`]).

use crate::DesignInputs;
use crate::engine::meta::Value;
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve};
use crate::gui::format::{format_value, with_unit};

/// Which way the calculator runs (spec A1 "Mode switch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizingMode {
    /// The forward calculation: the inputs give the torque.
    MagnetsToTorque,
    /// Inverse sizing: the target torque gives the free variable.
    TorqueToMagnets,
}

impl SizingMode {
    /// Both modes; the first is the default.
    pub const ALL: [SizingMode; 2] = [SizingMode::MagnetsToTorque, SizingMode::TorqueToMagnets];

    /// The name a design file and a share link record.
    pub const fn key(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "magnets_to_torque",
            SizingMode::TorqueToMagnets => "torque_to_magnets",
        }
    }

    /// The text of the mode switch: ASCII arrows, since egui's default fonts have no U+2192.
    pub const fn label(self) -> &'static str {
        match self {
            SizingMode::MagnetsToTorque => "Magnets -> Torque",
            SizingMode::TorqueToMagnets => "Torque -> Magnets",
        }
    }

    /// The mode a design file names, if it names one.
    pub fn from_key(key: &str) -> Option<SizingMode> {
        SizingMode::ALL.into_iter().find(|mode| mode.key() == key)
    }
}

/// The name a design file and a share link record for a free variable.
pub const fn variable_key(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "axial_length",
        FreeVariable::MagnetsPerRing => "magnets_per_ring",
        FreeVariable::RingRadius => "ring_radius",
    }
}

/// The free variable a design file names, if it names one.
pub fn variable_from_key(key: &str) -> Option<FreeVariable> {
    FreeVariable::ALL
        .into_iter()
        .find(|&variable| variable_key(variable) == key)
}

/// The text of the free-variable picker.
pub const fn variable_label(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "Axial magnet length",
        FreeVariable::MagnetsPerRing => "Magnets per ring",
        FreeVariable::RingRadius => "Ring radius",
    }
}

/// The input whose metadata gives the target torque its unit, range and step: the hot
/// minimum requirement, the torque inverse sizing is usually asked to meet.
pub const TARGET_RANGE_INPUT: &str = "metal.required_min_Nm";

/// The sizing state: the mode, the free variable and the target torque.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub struct SizingState {
    pub mode: SizingMode,
    pub variable: FreeVariable,
    /// The hot-low torque with production variation that inverse sizing must reach [N·m].
    pub target_Nm: f64,
}

/// The default target torque [N·m]: the workbook's hot minimum (`metal.required_min_Nm`,
/// Metal design C7), decision M41-9.
pub const DEFAULT_TARGET_NM: f64 = 2.5;

impl Default for SizingState {
    fn default() -> Self {
        Self {
            mode: SizingMode::MagnetsToTorque,
            variable: FreeVariable::ALL[0],
            target_Nm: DEFAULT_TARGET_NM,
        }
    }
}

/// How long the design must stay unchanged before a solve runs [s].
pub const DEBOUNCE_S: f64 = 0.25;

/// The start of the status line of a solved design.
pub const SOLVED_PREFIX: &str = "Solved at ";

/// The start of the status line when the target is out of reach.
pub const NOT_REACHABLE_PREFIX: &str = "Not reachable";

/// The status line while a solve waits for the design to settle.
pub const SOLVING: &str = "Solving...";

/// What a solve depends on: the inputs (with the free variable cleared, since the solve sets
/// it), the free variable and the target.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
struct SolveKey {
    inputs: DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
}

impl SolveKey {
    fn new(inputs: &DesignInputs, state: &SizingState) -> Self {
        Self {
            inputs: state.variable.apply(inputs, 0.0),
            variable: state.variable,
            target_Nm: state.target_Nm,
        }
    }
}

/// Runs inverse sizing for the panel: debounced, never per frame.
#[derive(Clone, Debug, Default)]
pub struct SizingRunner {
    /// The last solve: what it was for, and its outcome.
    last: Option<(SolveKey, Result<SizingOutcome, SizingError>)>,
    /// A changed design waiting to settle, and when it was last seen changing [s, egui time].
    pending: Option<(SolveKey, f64)>,
    /// How many solves ran (the tests check that a solve never runs per frame).
    pub(crate) solves: usize,
}

impl SizingRunner {
    /// Called once per frame in Torque → Magnets mode with the frame's time and whether an
    /// edit is in progress. Solves when the design has been unchanged for [`DEBOUNCE_S`] and no
    /// edit is in progress. Returns how long to wait before the next frame must run for a
    /// pending solve [s], or `None` when nothing is pending.
    pub fn update(
        &mut self,
        inputs: &DesignInputs,
        state: &SizingState,
        now: f64,
        editing: bool,
    ) -> Option<f64> {
        let key = SolveKey::new(inputs, state);
        if self.last.as_ref().is_some_and(|(last, _)| *last == key) {
            self.pending = None;
            return None;
        }
        let since = match &self.pending {
            Some((pending, since)) if *pending == key && !editing => *since,
            _ => {
                self.pending = Some((key, now));
                return Some(DEBOUNCE_S);
            }
        };
        let waited = now - since;
        if waited < DEBOUNCE_S {
            return Some(DEBOUNCE_S - waited);
        }
        self.solve_now(inputs, state);
        None
    }

    /// Solves this design now, whatever the debounce: the panel calls it once when the user
    /// leaves Torque → Magnets before a change was solved, so the value written into the
    /// inputs is this design's solution, not the last one's (decision M41-7). Never per frame.
    pub fn solve_now(&mut self, inputs: &DesignInputs, state: &SizingState) {
        let outcome = solve(inputs, state.variable, state.target_Nm);
        self.solves += 1;
        self.pending = None;
        self.last = Some((SolveKey::new(inputs, state), outcome));
    }

    /// Whether the last solve is for this design (nothing pending).
    pub fn is_current(&self, inputs: &DesignInputs, state: &SizingState) -> bool {
        self.last
            .as_ref()
            .is_some_and(|(key, _)| *key == SolveKey::new(inputs, state))
    }

    /// The last outcome, if the last solve was for the same free variable (it may be stale:
    /// see [`SizingRunner::is_current`]).
    pub fn outcome(&self, variable: FreeVariable) -> Option<&Result<SizingOutcome, SizingError>> {
        match &self.last {
            Some((key, outcome)) if key.variable == variable => Some(outcome),
            _ => None,
        }
    }

    /// The free variable's value the panel shows: the solved value, or the best valid value
    /// when the target is out of reach; `None` before the first solve of this variable, when
    /// no value is valid, or when the solve refused.
    pub fn value(&self, variable: FreeVariable) -> Option<f64> {
        match self.outcome(variable)? {
            Ok(SizingOutcome::Solved(point)) => Some(point.value),
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => Some(best.value),
            _ => None,
        }
    }

    /// The design the panel shows in Torque → Magnets: `inputs` with the free variable at
    /// [`SizingRunner::value`], else `inputs` as they are.
    pub fn shown(&self, inputs: &DesignInputs, state: &SizingState) -> DesignInputs {
        match self.value(state.variable) {
            Some(value) => state.variable.apply(inputs, value),
            None => inputs.clone(),
        }
    }

    /// The status line under the sizing controls.
    pub fn status(&self, inputs: &DesignInputs, state: &SizingState) -> String {
        if !self.is_current(inputs, state) {
            return SOLVING.to_owned();
        }
        match self.outcome(state.variable) {
            Some(Ok(SizingOutcome::Solved(point))) => format!(
                "{SOLVED_PREFIX}{} (hot-low torque {})",
                variable_text(state.variable, point.value),
                torque_text(point.torque_hot_low_Nm)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) => format!(
                "{NOT_REACHABLE_PREFIX} (best {} at {})",
                torque_text(best.torque_hot_low_Nm),
                variable_text(state.variable, best.value)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: None })) => format!(
                "{NOT_REACHABLE_PREFIX}: no value of the free variable gives a valid design"
            ),
            Some(Err(SizingError::InvalidTarget(target))) => format!(
                "Sizing refused: the target {} is not a positive torque",
                torque_text(*target)
            ),
            Some(Err(SizingError::InvalidInputs(errors))) => {
                let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
                format!("Sizing refused: invalid inputs ({})", paths.join(", "))
            }
            None => SOLVING.to_owned(),
        }
    }
}

/// A free variable's value as the status line shows it, with its input's unit.
fn variable_text(variable: FreeVariable, value: f64) -> String {
    let shown = match variable {
        FreeVariable::MagnetsPerRing => Value::Int(value as i64),
        _ => Value::Num(value),
    };
    let unit = crate::gui::inputs::InputCatalogue::get()
        .entry(variable.path())
        .map_or("", |entry| entry.meta.unit);
    with_unit(format_value(&shown), unit)
}

/// A torque as the status line shows it.
#[allow(non_snake_case)] // unit suffix
fn torque_text(torque_Nm: f64) -> String {
    with_unit(format_value(&Value::Num(torque_Nm)), "N·m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
        for mode in SizingMode::ALL {
            assert_eq!(SizingMode::from_key(mode.key()), Some(mode));
        }
        for variable in FreeVariable::ALL {
            assert_eq!(variable_from_key(variable_key(variable)), Some(variable));
        }
        assert_eq!(SizingMode::from_key("Torque"), None);
        assert_eq!(variable_from_key(""), None);
    }

    #[test]
    fn the_default_state_is_the_forward_calculation_at_the_hot_minimum() {
        let state = SizingState::default();
        assert_eq!(state.mode, SizingMode::MagnetsToTorque);
        assert_eq!(state.variable, FreeVariable::AxialLength);
        assert_eq!(
            state.target_Nm,
            DesignInputs::default().metal.required_min_Nm
        );
    }

    #[allow(non_snake_case)] // unit suffix
    fn inverse(variable: FreeVariable, target_Nm: f64) -> SizingState {
        SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable,
            target_Nm,
        }
    }

    #[test]
    fn a_solve_waits_for_the_debounce_and_runs_once() {
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        assert_eq!(
            runner.update(&inputs, &state, 10.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.status(&inputs, &state), SOLVING);
        let wait = runner.update(&inputs, &state, 10.1, false).unwrap();
        assert!((wait - 0.15).abs() < 1e-12, "{wait}");
        assert_eq!(runner.solves, 0);
        assert_eq!(runner.update(&inputs, &state, 10.25, false), None);
        assert_eq!(runner.solves, 1);
        for frame in 0..10 {
            assert_eq!(
                runner.update(&inputs, &state, 10.3 + frame as f64, false),
                None
            );
        }
        assert_eq!(runner.solves, 1, "never per frame");
        assert!(runner.is_current(&inputs, &state));
    }

    #[test]
    fn an_edit_in_progress_or_a_new_change_restarts_the_wait() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        // A drag: still editing past the debounce, so nothing runs.
        runner.update(&inputs, &state, 1.0, true);
        assert_eq!(runner.solves, 0);
        // Released: the wait starts again from the release.
        runner.update(&inputs, &state, 1.1, false);
        runner.update(&inputs, &state, 1.2, false);
        assert_eq!(runner.solves, 0);
        inputs.metal.face_gap_mm = 1.5;
        runner.update(&inputs, &state, 1.4, false);
        runner.update(&inputs, &state, 1.5, false);
        assert_eq!(runner.solves, 0, "a new change restarts the wait");
        runner.update(&inputs, &state, 1.65, false);
        assert_eq!(runner.solves, 1);
    }

    #[test]
    fn solve_now_solves_a_pending_change_at_once() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        runner.update(&inputs, &state, 1.0, false);
        inputs.metal.face_gap_mm = 1.5;
        assert_eq!(runner.update(&inputs, &state, 1.1, false), Some(DEBOUNCE_S));
        assert!(
            !runner.is_current(&inputs, &state),
            "waiting for the debounce"
        );
        runner.solve_now(&inputs, &state);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state));
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable at a 1.5 mm gap")
        };
        assert_eq!(
            point.inputs.metal.face_gap_mm, 1.5,
            "this design's solution"
        );
        // Nothing is left pending: the next frame does not solve again.
        assert_eq!(runner.update(&inputs, &state, 1.2, false), None);
        assert_eq!(runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_own_value_does_not_trigger_a_solve() {
        // The solve sets the free variable, so its value in the inputs is not part of the key.
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, 0.0, false);
        runner.update(&inputs, &state, 1.0, false);
        inputs.coupling.magnets.axial_length_mm = Some(30.0);
        assert!(runner.is_current(&inputs, &state));
        assert_eq!(runner.update(&inputs, &state, 2.0, false), None);
        assert_eq!(runner.solves, 1);
    }

    /// The runner after a solve of `state` at the defaults.
    fn solved(state: &SizingState) -> SizingRunner {
        let mut runner = SizingRunner::default();
        let inputs = DesignInputs::default();
        runner.update(&inputs, state, 0.0, false);
        runner.update(&inputs, state, 1.0, false);
        runner
    }

    #[test]
    fn a_solved_design_shows_the_solved_value() {
        let state = inverse(FreeVariable::AxialLength, 2.5);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable by the axial length")
        };
        assert_eq!(runner.value(state.variable), Some(point.value));
        assert_eq!(runner.shown(&inputs, &state), point.inputs);
        assert_eq!(
            runner.status(&inputs, &state),
            format!(
                "Solved at {} mm (hot-low torque 2.500 N·m)",
                format_value(&Value::Num(point.value))
            )
        );
        // Another free variable has no outcome yet.
        assert_eq!(runner.value(FreeVariable::RingRadius), None);
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value() {
        let state = inverse(FreeVariable::AxialLength, 50.0);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        assert_eq!(runner.value(state.variable), Some(50.8));
        assert_eq!(
            runner
                .shown(&inputs, &state)
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8)
        );
        assert_eq!(
            runner.status(&inputs, &state),
            "Not reachable (best 9.937 N·m at 50.80 mm)"
        );
        let poles = inverse(FreeVariable::MagnetsPerRing, 2.5);
        assert_eq!(
            solved(&poles).status(&inputs, &poles),
            "Not reachable (best 2.285 N·m at 10)"
        );
    }

    #[test]
    fn a_refused_solve_shows_why_and_keeps_the_inputs() {
        let state = inverse(FreeVariable::RingRadius, -1.0);
        let runner = solved(&state);
        let inputs = DesignInputs::default();
        assert_eq!(runner.value(state.variable), None);
        assert_eq!(runner.shown(&inputs, &state), inputs);
        assert_eq!(
            runner.status(&inputs, &state),
            "Sizing refused: the target -1.000 N·m is not a positive torque"
        );
        let mut bad = DesignInputs::default();
        bad.coupling.backiron = 7;
        let state = inverse(FreeVariable::RingRadius, 2.5);
        let mut runner = SizingRunner::default();
        runner.update(&bad, &state, 0.0, false);
        runner.update(&bad, &state, 1.0, false);
        assert_eq!(
            runner.status(&bad, &state),
            "Sizing refused: invalid inputs (coupling.backiron)"
        );
    }
}
