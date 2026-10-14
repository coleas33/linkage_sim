//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward
//! calculation, or Torque → Magnets, inverse sizing by [`crate::engine::sizing::solve_in`].
//!
//! The mode and the free variable are GUI state (A-2 decision A2-9): the engine takes them as
//! arguments. Design files and share links record them (decision M41-4). The target torque is
//! not GUI state: it is the requirement, the design input [`REQUIREMENT_INPUT`] (decision G-2:
//! one torque number, so the solve and the hot minimum check read the same value).
//!
//! [`SizingRunner`] runs the solve over the search range the panel gives: on a change, once
//! the design has been still for [`DEBOUNCE_S`] and no edit is in progress (no drag, no
//! typing), never per frame (a solve takes about 1 to 3 ms in a release build, longer on wasm,
//! against compute_all's 17 us). Until then the panel keeps showing the last solved value,
//! marked as solving. The design the panel shows in Torque → Magnets is the inputs with the
//! free variable at the solved value, or at the best value when the target is not reachable
//! (decision M41-8); the inputs keep their own value of the free variable until the user
//! leaves the mode (decision M41-7), when a change still waiting for its debounce is solved at
//! once ([`SizingRunner::solve_now`]).

use crate::DesignInputs;
use crate::engine::meta::{SliderRange, Value};
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve_in};
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

/// The input inverse sizing meets: the hot minimum requirement. Decision G-2 (one torque
/// number): the sizing target is this design input, so the solve and the hot minimum check
/// read one value, and an edit of it is an ordinary design edit (undo, files, links).
pub const REQUIREMENT_INPUT: &str = "metal.required_min_Nm";

/// The sizing state: the mode and the free variable (the target is [`REQUIREMENT_INPUT`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingState {
    pub mode: SizingMode,
    pub variable: FreeVariable,
}

impl Default for SizingState {
    fn default() -> Self {
        Self {
            mode: SizingMode::MagnetsToTorque,
            variable: FreeVariable::ALL[0],
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

/// The status line of a solve refused because the requirement ([`REQUIREMENT_INPUT`], the
/// target) is not a positive torque.
pub const REQUIREMENT_NOT_POSITIVE: &str = "The requirement must be a positive torque.";

/// What a solve depends on: the inputs with the free variable cleared (the solve sets it; the
/// target, [`REQUIREMENT_INPUT`], is among them), the free variable and the search range.
#[derive(Clone, Debug, PartialEq)]
struct SolveKey {
    inputs: DesignInputs,
    variable: FreeVariable,
    /// The search range's min, max, step and log flag as bit patterns: a NaN bound (the solve
    /// refuses it) still equals itself, so it is refused once instead of awaited every frame.
    range: [u64; 4],
}

impl SolveKey {
    fn new(inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> Self {
        Self {
            inputs: state.variable.apply(inputs, 0.0),
            variable: state.variable,
            range: [
                range.min.to_bits(),
                range.max.to_bits(),
                range.step.to_bits(),
                u64::from(range.log),
            ],
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
    /// Called once per frame in Torque → Magnets mode with the search range, the frame's time
    /// and whether an edit is in progress. Solves when the design and the range have been
    /// unchanged for [`DEBOUNCE_S`] and no edit is in progress. Returns how long to wait before
    /// the next frame must run for a pending solve [s], or `None` when nothing is pending.
    pub fn update(
        &mut self,
        inputs: &DesignInputs,
        state: &SizingState,
        range: SliderRange,
        now: f64,
        editing: bool,
    ) -> Option<f64> {
        let key = SolveKey::new(inputs, state, range);
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
        self.solve_now(inputs, state, range);
        None
    }

    /// Solves this design over `range` now, whatever the debounce: the panel calls it once when
    /// the user leaves Torque → Magnets before a change was solved, so the value written into
    /// the inputs is this design's solution, not the last one's (decision M41-7). Never per
    /// frame. The target is the design's requirement ([`REQUIREMENT_INPUT`]).
    pub fn solve_now(&mut self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) {
        let outcome = solve_in(inputs, state.variable, inputs.metal.required_min_Nm, range);
        self.solves += 1;
        self.pending = None;
        self.last = Some((SolveKey::new(inputs, state, range), outcome));
    }

    /// Whether the last solve is for this design and this search range (nothing pending).
    pub fn is_current(
        &self,
        inputs: &DesignInputs,
        state: &SizingState,
        range: SliderRange,
    ) -> bool {
        self.last
            .as_ref()
            .is_some_and(|(key, _)| *key == SolveKey::new(inputs, state, range))
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
    /// [`SizingRunner::value`] when that value lies in `range` (a value from an earlier search
    /// outside the range asked for now is not shown), else `inputs` as they are.
    pub fn shown(
        &self,
        inputs: &DesignInputs,
        state: &SizingState,
        range: SliderRange,
    ) -> DesignInputs {
        match self
            .value(state.variable)
            .filter(|value| (range.min..=range.max).contains(value))
        {
            Some(value) => state.variable.apply(inputs, value),
            None => inputs.clone(),
        }
    }

    /// The status line under the sizing controls, for the search over `range`.
    pub fn status(&self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> String {
        if !self.is_current(inputs, state, range) {
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
            Some(Err(SizingError::InvalidTarget(_))) => REQUIREMENT_NOT_POSITIVE.to_owned(),
            Some(Err(SizingError::InvalidRange { .. })) => {
                "Sizing refused: the search range is empty".to_owned()
            }
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
    use crate::engine::sizing::solve;

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
    fn the_default_state_is_the_forward_calculation_with_the_axial_length_free() {
        let state = SizingState::default();
        assert_eq!(state.mode, SizingMode::MagnetsToTorque);
        assert_eq!(state.variable, FreeVariable::AxialLength);
    }

    fn inverse(variable: FreeVariable) -> SizingState {
        SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable,
        }
    }

    /// The default design with the requirement, the sizing target (decision G-2), at
    /// `required_Nm`.
    #[allow(non_snake_case)] // unit suffix
    fn requiring(required_Nm: f64) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.metal.required_min_Nm = required_Nm;
        inputs
    }

    #[test]
    fn a_solve_waits_for_the_debounce_and_runs_once() {
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        assert_eq!(
            runner.update(&inputs, &state, range, 10.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.status(&inputs, &state, range), SOLVING);
        let wait = runner.update(&inputs, &state, range, 10.1, false).unwrap();
        assert!((wait - 0.15).abs() < 1e-12, "{wait}");
        assert_eq!(runner.solves, 0);
        assert_eq!(runner.update(&inputs, &state, range, 10.25, false), None);
        assert_eq!(runner.solves, 1);
        for frame in 0..10 {
            assert_eq!(
                runner.update(&inputs, &state, range, 10.3 + frame as f64, false),
                None
            );
        }
        assert_eq!(runner.solves, 1, "never per frame");
        assert!(runner.is_current(&inputs, &state, range));
    }

    #[test]
    fn an_edit_in_progress_or_a_new_change_restarts_the_wait() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        // A drag: still editing past the debounce, so nothing runs.
        runner.update(&inputs, &state, range, 1.0, true);
        assert_eq!(runner.solves, 0);
        // Released: the wait starts again from the release.
        runner.update(&inputs, &state, range, 1.1, false);
        runner.update(&inputs, &state, range, 1.2, false);
        assert_eq!(runner.solves, 0);
        inputs.metal.face_gap_mm = 1.5;
        runner.update(&inputs, &state, range, 1.4, false);
        runner.update(&inputs, &state, range, 1.5, false);
        assert_eq!(runner.solves, 0, "a new change restarts the wait");
        runner.update(&inputs, &state, range, 1.65, false);
        assert_eq!(runner.solves, 1);
    }

    #[test]
    fn solve_now_solves_a_pending_change_at_once() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        runner.update(&inputs, &state, range, 1.0, false);
        inputs.metal.face_gap_mm = 1.5;
        assert_eq!(
            runner.update(&inputs, &state, range, 1.1, false),
            Some(DEBOUNCE_S)
        );
        assert!(
            !runner.is_current(&inputs, &state, range),
            "waiting for the debounce"
        );
        runner.solve_now(&inputs, &state, range);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state, range));
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable at a 1.5 mm gap")
        };
        assert_eq!(
            point.inputs.metal.face_gap_mm, 1.5,
            "this design's solution"
        );
        // Nothing is left pending: the next frame does not solve again.
        assert_eq!(runner.update(&inputs, &state, range, 1.2, false), None);
        assert_eq!(runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_own_value_does_not_trigger_a_solve() {
        // The solve sets the free variable, so its value in the inputs is not part of the key.
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        runner.update(&inputs, &state, range, 1.0, false);
        inputs.coupling.magnets.axial_length_mm = Some(30.0);
        assert!(runner.is_current(&inputs, &state, range));
        assert_eq!(runner.update(&inputs, &state, range, 2.0, false), None);
        assert_eq!(runner.solves, 1);
    }

    /// The runner after a solve of `state` at `inputs`, over the free variable's slider range.
    fn solved(inputs: &DesignInputs, state: &SizingState) -> SizingRunner {
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(inputs, state, range, 0.0, false);
        runner.update(inputs, state, range, 1.0, false);
        runner
    }

    #[test]
    fn a_solved_design_shows_the_solved_value() {
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = DesignInputs::default();
        let runner = solved(&inputs, &state);
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable by the axial length")
        };
        assert_eq!(runner.value(state.variable), Some(point.value));
        assert_eq!(runner.shown(&inputs, &state, range), point.inputs);
        assert_eq!(
            runner.status(&inputs, &state, range),
            format!(
                "Solved at {} mm (hot-low torque 2.500 N·m)",
                format_value(&Value::Num(point.value))
            )
        );
        // Another free variable has no outcome yet.
        assert_eq!(runner.value(FreeVariable::RingRadius), None);
    }

    #[test]
    fn the_target_is_the_requirement() {
        // Decision G-2: the runner solves for metal.required_min_Nm, and an edit of it is a
        // change of the design: the last solve is stale until the next one runs.
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = requiring(3.0);
        let mut runner = solved(&inputs, &state);
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve(&inputs, state.variable, 3.0))
        );
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("3.0 N·m is reachable by the axial length")
        };
        assert!((point.torque_hot_low_Nm - 3.0).abs() < 1e-6);
        let raised = requiring(3.5);
        assert!(!runner.is_current(&raised, &state, range));
        assert_eq!(runner.status(&raised, &state, range), SOLVING);
        assert_eq!(
            runner.update(&raised, &state, range, 2.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.update(&raised, &state, range, 3.0, false), None);
        assert_eq!(runner.solves, 2);
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve(&raised, state.variable, 3.5))
        );
    }

    #[test]
    fn the_search_range_is_part_of_the_key() {
        // The panel passes the range to search (the ring size's follows the Size tab's
        // diameter range, Task 7): another range is another solve, over that range.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let full = state.variable.range();
        let mut runner = solved(&inputs, &state);
        let narrow = SliderRange {
            min: 20.0,
            max: 30.0,
            ..full
        };
        assert!(runner.is_current(&inputs, &state, full));
        assert!(!runner.is_current(&inputs, &state, narrow));
        assert_eq!(runner.status(&inputs, &state, narrow), SOLVING);
        assert_eq!(
            runner.update(&inputs, &state, narrow, 2.0, false),
            Some(DEBOUNCE_S)
        );
        runner.solve_now(&inputs, &state, narrow);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state, narrow));
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve_in(
                &inputs,
                state.variable,
                inputs.metal.required_min_Nm,
                narrow
            ))
        );
        // 2.5 N·m needs less than 20 mm: the narrow range's smallest value meets it.
        assert_eq!(runner.value(state.variable), Some(20.0));
    }

    #[test]
    fn a_value_outside_the_search_range_is_not_shown() {
        // A value from an earlier search outside the range asked for now (a raised minimum OD
        // moves the ring size's range past it) is not shown: the inputs show as they are until
        // the new search runs. Its bounds count as inside.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let full = state.variable.range();
        let runner = solved(&inputs, &state);
        let value = runner.value(state.variable).expect("solved");
        let sized = state.variable.apply(&inputs, value);
        assert_eq!(runner.shown(&inputs, &state, full), sized);
        let above = SliderRange {
            min: value + 1.0,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, above), inputs);
        let below = SliderRange {
            max: value - 1.0,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, below), inputs);
        let exact = SliderRange {
            min: value,
            max: value,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, exact), sized);
    }

    #[test]
    fn an_empty_or_nan_search_range_is_refused_once() {
        // The engine refuses an empty range (Task 2). A NaN bound still equals itself in the
        // key: one refused solve, not a wait restarted every frame.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::RingRadius);
        let full = state.variable.range();
        for range in [
            SliderRange {
                min: 12.0,
                max: 11.0,
                ..full
            },
            SliderRange {
                min: f64::NAN,
                ..full
            },
        ] {
            let mut runner = SizingRunner::default();
            runner.update(&inputs, &state, range, 0.0, false);
            assert_eq!(runner.update(&inputs, &state, range, 1.0, false), None);
            assert_eq!(runner.solves, 1);
            for frame in 0..5 {
                assert_eq!(
                    runner.update(&inputs, &state, range, 2.0 + frame as f64, false),
                    None
                );
            }
            assert_eq!(runner.solves, 1, "{range:?}: refused once, not re-awaited");
            let status = runner.status(&inputs, &state, range);
            assert_eq!(status, "Sizing refused: the search range is empty");
            assert!(status.is_ascii(), "every glyph is in egui's default fonts");
            assert_eq!(runner.shown(&inputs, &state, range), inputs);
        }
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value() {
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = requiring(50.0);
        let runner = solved(&inputs, &state);
        assert_eq!(runner.value(state.variable), Some(50.8));
        assert_eq!(
            runner
                .shown(&inputs, &state, range)
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8)
        );
        assert_eq!(
            runner.status(&inputs, &state, range),
            "Not reachable (best 9.937 N·m at 50.80 mm)"
        );
        let poles = inverse(FreeVariable::MagnetsPerRing);
        let defaults = DesignInputs::default();
        assert_eq!(
            solved(&defaults, &poles).status(&defaults, &poles, poles.variable.range()),
            "Not reachable (best 2.285 N·m at 10)"
        );
    }

    #[test]
    fn a_refused_solve_shows_why_and_keeps_the_inputs() {
        // The requirement is the target: one that is not a positive torque (a file can hold
        // it; the slider cannot reach it) refuses the solve (spec 5), in plain words.
        let state = inverse(FreeVariable::RingRadius);
        let range = state.variable.range();
        for required in [-1.0, 0.0] {
            let inputs = requiring(required);
            let runner = solved(&inputs, &state);
            assert_eq!(runner.value(state.variable), None);
            assert_eq!(runner.shown(&inputs, &state, range), inputs);
            let status = runner.status(&inputs, &state, range);
            assert_eq!(status, REQUIREMENT_NOT_POSITIVE, "requirement {required}");
            assert!(status.is_ascii(), "every glyph is in egui's default fonts");
        }
        let mut bad = DesignInputs::default();
        bad.coupling.backiron = 7;
        let runner = solved(&bad, &state);
        assert_eq!(
            runner.status(&bad, &state, range),
            "Sizing refused: invalid inputs (coupling.backiron)"
        );
    }
}
