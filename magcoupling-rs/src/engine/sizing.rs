//! Addendum A1: inverse sizing (Torque → Magnets).
//!
//! Spec A1: inverse sizing "takes a target torque and makes the hot-low torque with
//! production variation (`metal.torque_hot_low_Nm`) meet it, by adjusting ONE free variable
//! the user picks: axial magnet length (default), magnets per ring (discrete; poles stay
//! even), or ring radius. Every other input stays fixed. ... It returns the smallest value
//! that meets the target, or 'not reachable' with the best value achieved inside the
//! variable's slider range. It fails loudly and never extrapolates past the range."
//!
//! **Method.** The torque is not guaranteed monotone in the variable (on the default design
//! it rises with the ring radius to about 14 mm, falls to about 20 mm and rises again), so a
//! continuous variable is sampled in ascending order at the [`SCAN_CELLS`] + 1 ends of equal
//! cells of its slider range, and three refinements find what lies between two samples:
//!
//! - **a crossing**: the first sample that meets is bisected against the sample before it to
//!   [`VALUE_TOLERANCE_MM`]; a range minimum that meets is the answer as it is;
//! - **a peak**: where three consecutive valid samples rise and then do not rise
//!   (T0 < T1 ≥ T2, the middle one missing the target), a golden-section search between the
//!   outer two finds the peak to [`VALUE_TOLERANCE_MM`]; a peak that meets has the crossing
//!   below it bisected (so a meeting interval inside one cell is found), and every peak is
//!   offered as the best value;
//! - **a validity edge**: where validity ([`is_valid`]) changes between two samples, the edge
//!   is bisected to [`VALUE_TOLERANCE_MM`] and its valid side sampled, so a torque largest
//!   where the blocks start to fit, or the keyway starts to leave hub wall, is seen there.
//!
//! Magnets per ring (poles per ring, one block per pole) steps through the even values of its
//! slider, with no refinement. Each value is one [`compute_all`] of the design with only the
//! free variable changed, so the answer is what the forward calculation shows, and every
//! value tried lies inside the slider range.
//!
//! **Residual limits** (stated, not handled): a hump of the torque whose rise and fall both
//! lie inside one cell (0.328 mm of ring radius, 0.778 mm of axial length) leaves no trace on
//! the samples and is not seen, and neither is a validity change that reverts inside one
//! cell. Near a peak the torque is flat, so the peak search finds the peak torque to its
//! floating-point resolution only (about 1e-15 relative): a target within that of the true
//! peak may read "not reachable".
//!
//! **What counts** ([`is_valid`], decision A2-4): the blocks fit ([`blocks_fit`]: faceted
//! blocks their polygon flats, the Calculator's C52 and C59; arcs without overlapping at the
//! magnet mid-radius), the keyway leaves hub wall (C53 > 0), the end-effect factor is in
//! range ([`end_effect_in_range`], audit M9: at short lengths f_end ≤ 0 makes the torque 0 or
//! negative) and the hot-low torque is finite. A value meets when it counts and its hot-low
//! torque is at least the target. "Not reachable" reports the best valid value the search
//! evaluated (the largest hot-low torque among the samples, the validity edges and the
//! refined peaks), or none when no value is valid.
//!
//! The space claim is not a condition: a solution may exceed it, and the housing results show
//! by how much (spec A1: a red callout, not a constraint).

use super::api::{DesignInputs, DesignResults, compute_all};
use super::meta::{SetError, SliderRange, input_rows};
use super::model::{blocks_fit, end_effect_in_range};

/// Equal cells of the coarse scan over a continuous variable's slider range.
pub const SCAN_CELLS: usize = 64;

/// Every bisection and peak search stops when its bracket is this narrow [mm]: a solved value
/// meets the target and lies within it of the crossing its bracket holds.
pub const VALUE_TOLERANCE_MM: f64 = 1e-9;

/// The free variable of inverse sizing (spec A1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeVariable {
    /// Axial magnet length of both rings (`coupling.magnets.axial_length_mm`): the default.
    AxialLength,
    /// Magnets per ring, one block per pole (`coupling.npole`): even values only.
    MagnetsPerRing,
    /// Ring radius: the inner magnet back apothem (`coupling.inner_back_apothem_mm`,
    /// Calculator C8; report 6.5: the variable starts from this input, not the pole sweep's rule).
    RingRadius,
}

impl FreeVariable {
    /// Every free variable; the first is the default.
    pub const ALL: [FreeVariable; 3] = [
        FreeVariable::AxialLength,
        FreeVariable::MagnetsPerRing,
        FreeVariable::RingRadius,
    ];

    /// The input the variable sets.
    pub const fn path(self) -> &'static str {
        match self {
            FreeVariable::AxialLength => "coupling.magnets.axial_length_mm",
            FreeVariable::MagnetsPerRing => "coupling.npole",
            FreeVariable::RingRadius => "coupling.inner_back_apothem_mm",
        }
    }

    /// The variable's slider range, read from its input's metadata (one source).
    pub fn range(self) -> SliderRange {
        input_rows(&DesignInputs::default())
            .into_iter()
            .find(|row| row.path == self.path())
            .and_then(|row| row.meta.range)
            .expect("every free variable is an input with a slider (tests/sizing.rs)")
    }

    /// The values the coarse pass evaluates, ascending: the [`SCAN_CELLS`] + 1 cell ends of a
    /// continuous variable, or every even whole number of the magnets-per-ring slider.
    pub fn grid(self) -> Vec<f64> {
        let r = self.range();
        match self {
            FreeVariable::MagnetsPerRing => (r.min as i64..=r.max as i64)
                .filter(|n| n % 2 == 0)
                .map(|n| n as f64)
                .collect(),
            FreeVariable::AxialLength | FreeVariable::RingRadius => (0..=SCAN_CELLS)
                .map(|i| {
                    if i == SCAN_CELLS {
                        r.max
                    } else {
                        r.min + (r.max - r.min) * i as f64 / SCAN_CELLS as f64
                    }
                })
                .collect(),
        }
    }

    /// The design `inputs` with this variable set to `value` and every other input kept.
    pub fn apply(self, inputs: &DesignInputs, value: f64) -> DesignInputs {
        let mut design = inputs.clone();
        match self {
            FreeVariable::AxialLength => design.coupling.magnets.axial_length_mm = Some(value),
            FreeVariable::MagnetsPerRing => design.coupling.npole = value as i64,
            FreeVariable::RingRadius => design.coupling.inner_back_apothem_mm = value,
        }
        design
    }
}

/// One value of the free variable, evaluated.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the result names
pub struct SizingPoint {
    /// The free variable's value (magnets per ring as a whole number).
    pub value: f64,
    /// The hot-low torque with production variation there (`metal.torque_hot_low_Nm`).
    pub torque_hot_low_Nm: f64,
    /// The design at that value: the inputs with only the free variable changed.
    pub inputs: DesignInputs,
}

/// What inverse sizing found.
#[derive(Clone, Debug, PartialEq)]
pub enum SizingOutcome {
    /// The smallest value in the slider range that meets the target, within the module docs'
    /// residual limits (a continuous variable to within [`VALUE_TOLERANCE_MM`] of its crossing).
    Solved(SizingPoint),
    /// No value the search evaluated meets the target: the best valid one (the largest
    /// hot-low torque, refined peaks included), or `None` when no value is valid.
    NotReachable { best: Option<SizingPoint> },
}

/// Why inverse sizing refused to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SizingError {
    /// The target is not a positive, finite torque [N·m].
    InvalidTarget(f64),
    /// The inputs fail [`DesignInputs::validate`] (every offending path, in schema order).
    InvalidInputs(Vec<SetError>),
}

/// Whether a design counts for inverse sizing (decision A2-4): its blocks fit
/// ([`blocks_fit`]), the keyway leaves hub wall (`model.hub_wall_past_key_mm` > 0, Calculator
/// C53), the end-effect factor is in range ([`end_effect_in_range`]) and the hot-low torque is
/// finite. `results` is `compute_all(design)`.
pub fn is_valid(design: &DesignInputs, results: &DesignResults) -> bool {
    blocks_fit(&design.coupling, &results.model)
        && results.model.hub_wall_past_key_mm > 0.0
        && end_effect_in_range(results.model.f_end)
        && results.metal.torque_hot_low_Nm.is_finite()
}

/// Makes the hot-low torque of `inputs` meet `target_Nm` by adjusting `variable` alone (spec
/// A1; the module docs give the method, its limits and what counts).
#[allow(non_snake_case)] // unit suffix
pub fn solve(
    inputs: &DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
) -> Result<SizingOutcome, SizingError> {
    if !(target_Nm.is_finite() && target_Nm > 0.0) {
        return Err(SizingError::InvalidTarget(target_Nm));
    }
    inputs.validate().map_err(SizingError::InvalidInputs)?;
    let mut evaluate = |value: f64| {
        let design = variable.apply(inputs, value);
        let results = compute_all(&design);
        Sample {
            value,
            torque_Nm: results.metal.torque_hot_low_Nm,
            valid: is_valid(&design, &results),
            design,
        }
    };
    let search = Search::new(
        &mut evaluate,
        target_Nm,
        variable != FreeVariable::MagnetsPerRing,
    );
    Ok(match search.over(&variable.grid()) {
        Found::Meets(sample) => SizingOutcome::Solved(sample.into_point()),
        Found::Best(best) => SizingOutcome::NotReachable {
            best: best.map(Sample::into_point),
        },
    })
}

/// One evaluated value: its torque, whether it counts ([`is_valid`]), and the design there
/// (the inputs in [`solve`]; nothing in the search's unit tests).
#[derive(Clone, Debug)]
#[allow(non_snake_case)] // unit suffix
struct Sample<P> {
    value: f64,
    torque_Nm: f64,
    valid: bool,
    design: P,
}

impl<P> Sample<P> {
    #[allow(non_snake_case)] // unit suffix
    fn meets(&self, target_Nm: f64) -> bool {
        self.valid && self.torque_Nm >= target_Nm
    }

    /// The torque the peak search compares: a value that does not count has none at all.
    fn height(&self) -> f64 {
        if self.valid {
            self.torque_Nm
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl Sample<DesignInputs> {
    fn into_point(self) -> SizingPoint {
        SizingPoint {
            value: self.value,
            torque_hot_low_Nm: self.torque_Nm,
            inputs: self.design,
        }
    }
}

/// What the search found.
#[derive(Debug)]
enum Found<P> {
    /// The smallest value that meets, within the module docs' limits.
    Meets(Sample<P>),
    /// Nothing met: the best valid sample, if any.
    Best(Option<Sample<P>>),
}

/// The search of [`solve`] over an ascending grid (the module docs' method). It takes the
/// evaluation as a function so its unit tests can drive it with plain functions.
#[allow(non_snake_case)] // unit suffix
struct Search<'a, P> {
    evaluate: &'a mut dyn FnMut(f64) -> Sample<P>,
    target_Nm: f64,
    /// A continuous variable: bisect crossings, search peaks, locate validity edges.
    refine: bool,
    /// The sample taken last: the lower end of a crossing's bisection.
    previous: Option<Sample<P>>,
    /// The last (up to) three valid samples of the current run of valid samples.
    window: Vec<Sample<P>>,
    /// The valid sample with the largest torque so far (the first of equals).
    best: Option<Sample<P>>,
}

impl<'a, P: Clone> Search<'a, P> {
    #[allow(non_snake_case)] // unit suffix
    fn new(evaluate: &'a mut dyn FnMut(f64) -> Sample<P>, target_Nm: f64, refine: bool) -> Self {
        Search {
            evaluate,
            target_Nm,
            refine,
            previous: None,
            window: Vec::new(),
            best: None,
        }
    }

    /// Runs the search over `grid` (ascending).
    fn over(mut self, grid: &[f64]) -> Found<P> {
        for &value in grid {
            let sample = (self.evaluate)(value);
            if let Some(found) = self.grid_sample(sample) {
                return Found::Meets(found);
            }
        }
        Found::Best(self.best)
    }

    /// Takes the next grid sample, after sampling the validity edge between it and the
    /// previous sample if validity changed there (a continuous variable only).
    fn grid_sample(&mut self, sample: Sample<P>) -> Option<Sample<P>> {
        let edge_before = match &self.previous {
            Some(previous) if self.refine && previous.valid != sample.valid => {
                Some(previous.clone())
            }
            _ => None,
        };
        if let Some(previous) = edge_before {
            let changed = !previous.valid;
            let (lo, hi) = self.bisect(previous.clone(), sample.clone(), |s| s.valid == changed);
            if previous.valid {
                // Valid, then not: the largest valid value, unless the bisection never moved.
                if lo.value > previous.value
                    && let Some(found) = self.take(lo)
                {
                    return Some(found);
                }
            } else {
                // Not valid, then valid: the smallest valid value. The invalid end nearest it
                // is the lower end of any crossing, so an edge that meets is the answer as is.
                self.previous = Some(lo);
                if hi.value < sample.value
                    && let Some(found) = self.take(hi)
                {
                    return Some(found);
                }
            }
        }
        self.take(sample)
    }

    /// Takes one sample, in ascending order: returns the answer when it meets (bisected
    /// against the sample before it); otherwise keeps the best and searches a peak the last
    /// three valid samples show.
    fn take(&mut self, sample: Sample<P>) -> Option<Sample<P>> {
        if sample.meets(self.target_Nm) {
            return Some(match self.previous.take() {
                Some(lo) if self.refine => self.crossing(lo, sample),
                _ => sample,
            });
        }
        if !sample.valid {
            self.window.clear();
            self.previous = Some(sample);
            return None;
        }
        self.offer(&sample);
        self.window.push(sample.clone());
        if self.window.len() > 3 {
            self.window.remove(0);
        }
        let hump = match &self.window[..] {
            [rise, top, fall]
                if self.refine
                    && rise.torque_Nm < top.torque_Nm
                    && top.torque_Nm >= fall.torque_Nm =>
            {
                Some((rise.clone(), top.clone(), fall.value))
            }
            _ => None,
        };
        if let Some((rise, top, fall)) = hump {
            let peak = self.peak(rise.value, top, fall);
            self.offer(&peak);
            if peak.meets(self.target_Nm) {
                return Some(self.crossing(rise, peak));
            }
        }
        self.previous = Some(sample);
        None
    }

    /// Keeps `sample` as the best if it counts and its torque beats the best so far.
    fn offer(&mut self, sample: &Sample<P>) {
        if sample.valid
            && self
                .best
                .as_ref()
                .is_none_or(|b| sample.torque_Nm > b.torque_Nm)
        {
            self.best = Some(sample.clone());
        }
    }

    /// The smallest value that meets between `lo` (does not meet) and `hi` (meets).
    #[allow(non_snake_case)] // unit suffix
    fn crossing(&mut self, lo: Sample<P>, hi: Sample<P>) -> Sample<P> {
        let target_Nm = self.target_Nm;
        self.bisect(lo, hi, |s| s.meets(target_Nm)).1
    }

    /// Bisects between `lo`, which lacks the property `has`, and `hi`, which has it, until the
    /// two are [`VALUE_TOLERANCE_MM`] apart or adjacent doubles; returns both ends.
    fn bisect(
        &mut self,
        mut lo: Sample<P>,
        mut hi: Sample<P>,
        has: impl Fn(&Sample<P>) -> bool,
    ) -> (Sample<P>, Sample<P>) {
        while hi.value - lo.value > VALUE_TOLERANCE_MM {
            let mid = lo.value + (hi.value - lo.value) / 2.0;
            if mid <= lo.value || mid >= hi.value {
                break; // adjacent doubles
            }
            let sample = (self.evaluate)(mid);
            if has(&sample) {
                hi = sample;
            } else {
                lo = sample;
            }
        }
        (lo, hi)
    }

    /// The golden-section search for the largest torque between `lo` and `hi`, which hold the
    /// sample `top` whose torque is at least both ends': narrows the bracket to
    /// [`VALUE_TOLERANCE_MM`] and returns the best sample it saw (`top` if none beats it).
    fn peak(&mut self, mut lo: f64, top: Sample<P>, mut hi: f64) -> Sample<P> {
        let ratio = (5.0_f64.sqrt() - 1.0) / 2.0; // 1 / golden ratio
        let mut best = top;
        let mut c = hi - (hi - lo) * ratio;
        let mut d = lo + (hi - lo) * ratio;
        let mut at_c = (self.evaluate)(c);
        let mut at_d = (self.evaluate)(d);
        loop {
            for sample in [&at_c, &at_d] {
                if sample.height() > best.height() {
                    best = sample.clone();
                }
            }
            if hi - lo <= VALUE_TOLERANCE_MM || !(lo < c && c < d && d < hi) {
                return best;
            }
            if at_c.height() >= at_d.height() {
                // The peak lies in [lo, d]: d's point becomes c's, a new c.
                hi = d;
                d = c;
                at_d = at_c;
                c = hi - (hi - lo) * ratio;
                at_c = (self.evaluate)(c);
            } else {
                // The peak lies in [c, hi]: c's point becomes d's, a new d.
                lo = c;
                c = d;
                at_c = at_d;
                d = lo + (hi - lo) * ratio;
                at_d = (self.evaluate)(d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The search alone, driven by plain functions of the value (no model): each refinement,
    //! the stated limit and the evaluation count.
    use super::*;

    /// The grid 0, 1, ..., 10 (cells of 1).
    fn grid() -> Vec<f64> {
        (0..=10).map(f64::from).collect()
    }

    /// Runs the search of `torque` (`valid` says where it counts) and records every value it
    /// evaluated.
    fn run(
        torque: impl Fn(f64) -> f64,
        valid: impl Fn(f64) -> bool,
        target: f64,
        refine: bool,
    ) -> (Found<()>, Vec<f64>) {
        let mut seen = Vec::new();
        let mut evaluate = |value: f64| {
            seen.push(value);
            Sample {
                value,
                torque_Nm: torque(value),
                valid: valid(value),
                design: (),
            }
        };
        let found = Search::new(&mut evaluate, target, refine).over(&grid());
        (found, seen)
    }

    fn met(found: Found<()>) -> Sample<()> {
        match found {
            Found::Meets(sample) => sample,
            other => panic!("expected a value that meets, got {other:?}"),
        }
    }

    fn always(_: f64) -> bool {
        true
    }

    #[test]
    fn a_crossing_is_bisected_to_the_tolerance() {
        let (found, seen) = run(|x| x, always, 4.25, true);
        let s = met(found);
        assert!(
            s.torque_Nm >= 4.25 && s.value - 4.25 <= VALUE_TOLERANCE_MM,
            "{}",
            s.value
        );
        assert!(
            seen.iter().all(|v| (0.0..=10.0).contains(v)),
            "never outside the grid"
        );
    }

    #[test]
    fn a_first_value_that_meets_is_the_answer_as_it_is() {
        let (found, seen) = run(|x| x, always, -1.0, true);
        assert_eq!(met(found).value, 0.0);
        assert_eq!(seen, [0.0]);
    }

    #[test]
    fn a_hump_inside_one_cell_is_found_at_its_rising_crossing() {
        // Peak 1 at 4.5; the samples 4 and 5 read 0.75 and miss a target of 0.9.
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (found, seen) = run(torque, always, 0.9, true);
        let s = met(found);
        let crossing = 4.5 - 0.1_f64.sqrt();
        assert!(s.torque_Nm >= 0.9, "{}", s.torque_Nm);
        assert!(
            (s.value - crossing).abs() < 1e-8,
            "{} vs {crossing}",
            s.value
        );
        assert!(seen.iter().all(|v| (0.0..=10.0).contains(v)));
    }

    #[test]
    fn not_reachable_reports_the_refined_peak() {
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        match run(torque, always, 2.0, true).0 {
            Found::Best(Some(best)) => {
                assert!((best.value - 4.5).abs() < 1e-6, "{}", best.value);
                assert!(best.torque_Nm > 1.0 - 1e-12, "{}", best.torque_Nm);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_torque_falling_from_where_values_start_to_count_is_met_at_that_edge() {
        // Counts from 3.3 on, falling: the sample 4 (6.0) misses 6.65, the edge (6.7) meets.
        let (found, _) = run(|x| 10.0 - x, |x| x >= 3.3, 6.65, true);
        let s = met(found);
        assert!(
            s.valid && s.value >= 3.3 && s.value - 3.3 <= VALUE_TOLERANCE_MM,
            "{}",
            s.value
        );
    }

    #[test]
    fn a_torque_rising_to_where_values_stop_counting_is_met_before_that_edge() {
        // Counts up to 6.7, rising: the sample 6 misses 6.65, the sample 7 does not count.
        let (found, _) = run(|x| x, |x| x <= 6.7, 6.65, true);
        let s = met(found);
        assert!(
            (s.value - 6.65).abs() < 1e-8 && s.torque_Nm >= 6.65,
            "{}",
            s.value
        );
    }

    #[test]
    fn nothing_counting_reports_no_best_value() {
        let (found, seen) = run(|x| x, |_| false, 1.0, true);
        assert!(matches!(found, Found::Best(None)), "{found:?}");
        assert_eq!(seen, grid(), "no edge, no peak: only the grid");
    }

    #[test]
    fn a_plateau_searches_no_peak() {
        // Equal torques never rise, so no peak search runs: one evaluation per grid value,
        // and the best is the first of equals.
        let (found, seen) = run(|_| 1.0, always, 2.0, true);
        assert_eq!(seen, grid());
        match found {
            Found::Best(Some(best)) => assert_eq!(best.value, 0.0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stepping_evaluates_the_grid_only() {
        // The discrete variable (magnets per ring): no bisection, peak or edge between values.
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (found, seen) = run(torque, |x| x >= 3.3, 0.9, false);
        assert_eq!(seen, grid());
        assert!(
            matches!(found, Found::Best(Some(ref b)) if b.value == 4.0),
            "{found:?}"
        );
        let (found, _) = run(|x| x, always, 4.25, false);
        assert_eq!(met(found).value, 5.0);
    }

    #[test]
    fn a_hump_whose_rise_and_fall_are_inside_one_cell_is_the_stated_limit() {
        // The module docs' residual limit, pinned: a spike between 4 and 5 over a rising line
        // leaves every sample rising, so no peak search runs and the spike is not seen.
        let torque = |x: f64| x / 10.0 + if (x - 4.4).abs() < 0.05 { 5.0 } else { 0.0 };
        let (found, seen) = run(torque, always, 3.0, true);
        assert!(
            matches!(found, Found::Best(Some(ref b)) if b.value == 10.0),
            "{found:?}"
        );
        assert_eq!(seen, grid());
    }

    #[test]
    fn a_peak_search_is_bounded() {
        // One peak search: about 45 evaluations for a bracket of 2 narrowed to 1e-9, never
        // more than the golden ratio allows (plus the grid and a crossing's bisection).
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (_, seen) = run(torque, always, 2.0, true);
        let golden_steps = (VALUE_TOLERANCE_MM / 2.0).ln() / ((5.0_f64.sqrt() - 1.0) / 2.0).ln();
        assert!(
            seen.len() <= grid().len() + 2 + golden_steps.ceil() as usize,
            "{}",
            seen.len()
        );
    }
}
