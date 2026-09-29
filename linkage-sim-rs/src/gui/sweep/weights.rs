//! Per-weight gravity breakdown of a sweep (payload weights, spec Track 2
//! section 2): which weight helps or hurts the actuator at each sample, by
//! how much in actuator force and in actuator power, and where the load
//! drives the actuator (braking).
//!
//! `compute_sweep_data_with_weights` pushes one row per sample into a
//! [`WeightBreakdownBuilder`]; the rules that need sweep-wide maxima
//! (near-reversal NaN, braking tolerance) run in
//! [`WeightBreakdownBuilder::finish`]. The physics and the named thresholds
//! live in `analysis::gravity_breakdown`.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::analysis::gravity_breakdown::{self as gb, Classification, WeightSource};
use crate::core::mechanism::Mechanism;

/// What the force shares are shares of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShareBasis {
    /// The mechanism has a `LinearActuator`: shares of the actuator's
    /// required axial force (N, positive = extension push); the rate is the
    /// actuator extension rate dL/dt (m/s).
    ActuatorForce,
    /// No actuator: shares of the driver effort (N*m for a revolute
    /// driver, N for a linear driver); the rate is the driver rate.
    DriverTorque,
}

/// Per-weight gravity breakdown over one sweep.
///
/// Every per-sample series has `SweepData::angles_deg.len()` entries (NaN
/// where the sample has no pose or no velocity, `false` in `braking`); the
/// outer index of the per-source series follows `sources`. By construction
/// `sum(force_share) + other_force == total_force` and
/// `sum(power_share) + other_power == total_power` wherever the shares are
/// finite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBreakdown {
    /// The weights, in `gravity_breakdown::weight_sources` order.
    pub sources: Vec<WeightSource>,
    /// Actuator-force shares or driver-torque shares.
    pub basis: ShareBasis,
    /// `P_g,i = m_i g . v_i` (W), `[source][sample]`; positive = helping.
    pub gravity_power: Vec<Vec<f64>>,
    /// `-P_g,i / rate`: actuator force share (N), or driver torque share
    /// for `ShareBasis::DriverTorque`. NaN near stroke reversal.
    pub force_share: Vec<Vec<f64>>,
    /// `-P_g,i` (W): actuator (driver) power spent on each weight;
    /// negative = the weight gives power back.
    pub power_share: Vec<Vec<f64>>,
    /// `total_force - sum(force_share)`: non-gravity loads (force zones,
    /// springs, external loads, end stops).
    pub other_force: Vec<f64>,
    /// `total_power - sum(power_share)`.
    pub other_power: Vec<f64>,
    /// Required actuator force (N): the sweep's `actuator_forces`, which
    /// is the required force in sizing and stored-force mode alike
    /// (BL-026). Driver torque for `DriverTorque`.
    pub total_force: Vec<f64>,
    /// Required actuator (driver) power (W); negative = braking.
    pub total_power: Vec<f64>,
    /// True where the load drives the actuator (`gravity_breakdown::is_braking`).
    pub braking: Vec<bool>,
}

impl WeightBreakdown {
    /// Helping / hurting / neutral for `source` at `sample`, against that
    /// weight's sweep-wide neutral band (`gravity_breakdown::classify`).
    /// Neutral when either index is out of range.
    pub fn classification(&self, source: usize, sample: usize) -> Classification {
        let Some(series) = self.gravity_power.get(source) else {
            return Classification::Neutral;
        };
        let Some(&p_g) = series.get(sample) else {
            return Classification::Neutral;
        };
        gb::classify(p_g, gb::max_abs_finite(series))
    }
}

/// Required totals `(total_force, total_power)` at one sample.
///
/// `driver_torque` is the pass-1 statics driver effort (NaN when statics
/// failed) and `omega` the driver rate. For `ActuatorForce`, `rate` is the
/// actuator dL/dt, `actuator_force` the sweep's required actuator force
/// (`solver::reactions::required_actuator_force`, NaN where dL/dt is near
/// zero) and `stored_force` the stored force pass-1 statics applied (0 in
/// sizing mode). The total force is `actuator_force` unchanged: since
/// BL-026 it already includes the stored force, so adding it again would
/// double count it. The total power is the same balance multiplied through
/// by dL/dt, `F_required * rate = driver_torque * omega + stored_force *
/// rate`, which stays finite through stroke reversal where the force is
/// NaN.
pub(super) fn required_totals(
    basis: ShareBasis,
    driver_torque: f64,
    omega: f64,
    rate: f64,
    actuator_force: f64,
    stored_force: f64,
) -> (f64, f64) {
    match basis {
        ShareBasis::ActuatorForce => (actuator_force, driver_torque * omega + stored_force * rate),
        ShareBasis::DriverTorque => (driver_torque, driver_torque * omega),
    }
}

/// Collects one row per sweep sample and turns the rows into a
/// [`WeightBreakdown`] once the sweep-wide maxima are known.
pub(super) struct WeightBreakdownBuilder {
    sources: Vec<WeightSource>,
    basis: ShareBasis,
    g: [f64; 2],
    gravity_power: Vec<Vec<f64>>,
    rate: Vec<f64>,
    total_force: Vec<f64>,
    total_power: Vec<f64>,
}

impl WeightBreakdownBuilder {
    /// `g` is the mechanism's gravity vector (`gravity_breakdown::gravity_vector`).
    pub(super) fn new(sources: Vec<WeightSource>, basis: ShareBasis, g: [f64; 2], capacity: usize) -> Self {
        let gravity_power = sources.iter().map(|_| Vec::with_capacity(capacity)).collect();
        Self {
            sources,
            basis,
            g,
            gravity_power,
            rate: Vec::with_capacity(capacity),
            total_force: Vec::with_capacity(capacity),
            total_power: Vec::with_capacity(capacity),
        }
    }

    /// A sample with pose `q` and velocity `q_dot`: `rate` is the actuator
    /// dL/dt or the driver rate, totals from [`required_totals`].
    pub(super) fn push_sample(
        &mut self,
        mech: &Mechanism,
        q: &DVector<f64>,
        q_dot: &DVector<f64>,
        rate: f64,
        total_force: f64,
        total_power: f64,
    ) {
        let powers = gb::gravity_powers(mech, &self.sources, q, q_dot, self.g);
        self.push_row(&powers, rate, total_force, total_power);
    }

    /// A sample with no pose or no velocity: NaN in every series.
    pub(super) fn push_nan(&mut self) {
        let nan = vec![f64::NAN; self.sources.len()];
        self.push_row(&nan, f64::NAN, f64::NAN, f64::NAN);
    }

    fn push_row(&mut self, powers: &[f64], rate: f64, total_force: f64, total_power: f64) {
        for (series, &p_g) in self.gravity_power.iter_mut().zip(powers) {
            series.push(p_g);
        }
        self.rate.push(rate);
        self.total_force.push(total_force);
        self.total_power.push(total_power);
    }

    /// Apply the sweep-wide rules: force shares (NaN near stroke reversal),
    /// power shares, the non-gravity remainder and braking.
    pub(super) fn finish(self) -> WeightBreakdown {
        let max_rate = gb::max_abs_finite(&self.rate);
        let force_share: Vec<Vec<f64>> = self
            .gravity_power
            .iter()
            .map(|series| {
                series
                    .iter()
                    .zip(&self.rate)
                    .map(|(&p_g, &rate)| gb::force_share(p_g, rate, max_rate))
                    .collect()
            })
            .collect();
        let power_share: Vec<Vec<f64>> = self
            .gravity_power
            .iter()
            .map(|series| series.iter().map(|&p_g| -p_g).collect())
            .collect();
        let other_force = remainder(&self.total_force, &force_share);
        let other_power = remainder(&self.total_power, &power_share);
        let max_power = gb::max_abs_finite(&self.total_power);
        let braking = self.total_power.iter().map(|&p| gb::is_braking(p, max_power)).collect();
        WeightBreakdown {
            sources: self.sources,
            basis: self.basis,
            gravity_power: self.gravity_power,
            force_share,
            power_share,
            other_force,
            other_power,
            total_force: self.total_force,
            total_power: self.total_power,
            braking,
        }
    }
}

/// `total[k] - sum_i shares[i][k]` for every sample `k`.
fn remainder(total: &[f64], shares: &[Vec<f64>]) -> Vec<f64> {
    total
        .iter()
        .enumerate()
        .map(|(k, &t)| t - shares.iter().map(|s| s[k]).sum::<f64>())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::gravity_breakdown::{gravity_vector, max_abs_finite, EPS_REL_LDOT};
    use crate::forces::elements::ForceElement;
    use crate::gui::samples::{build_sample, SampleMechanism};
    use crate::gui::state::AppState;
    use crate::gui::sweep::{compute_sweep_data, compute_sweep_data_with_weights, SweepData};
    use crate::gui::test_support::set_actuator_stored_force;
    use crate::solver::reactions::required_actuator_force;
    use Classification::{Helping, Hurting, Neutral};

    /// Tolerance factor for identities that hold up to round-off.
    const ROUND_OFF: f64 = 1e-9;
    /// Tolerance factor for "the weights explain the whole load" when
    /// gravity is the only load: exact by the power balance up to solver
    /// precision, amplified by at most 1 / EPS_REL_LDOT.
    const GRAVITY_ONLY: f64 = 1e-6;
    /// Tolerances of the adjacent-sample energy check, relative to each
    /// source's peak gravity power. The central difference over +-1 deg has
    /// a truncation error of ~(1 deg)^2 / 6 = 5.1e-5 times the squared
    /// harmonic order of the motion. Measured worst cases: 5.1e-5 on the
    /// parallelogram (pure circles) and 1.2e-3 on the Chebyshev (the
    /// coupler's fast return stroke). Doubling the step to 2 deg quadruples
    /// both, so they are truncation error, not a physics mismatch.
    const PARALLELOGRAM_FD_TOL: f64 = 2e-4;
    const CHEBYSHEV_FD_TOL: f64 = 2.5e-3;
    /// Tolerance factor when comparing two separate sweeps: at the
    /// parallelogram's change points (0/180/360 deg, singular Jacobian) the
    /// statics solve amplifies round-off to ~1e-8 relative.
    const CROSS_SWEEP: f64 = 1e-6;

    /// Weights to add: (body, kg, body-local position in m).
    type Weights = &'static [(&'static str, f64, [f64; 2])];

    /// Parallelogram weights: W1 on the rocker tip (rocker-local C, traced by
    /// the sweep as "rocker.C"), W2 on coupler point P (traced "coupler.P").
    const PARALLELOGRAM_WEIGHTS: Weights =
        &[("rocker", 50.0, [0.0, 0.0]), ("coupler", 20.0, [2.0, 0.0])];
    /// Chebyshev weights: W1 on coupler point M (the straight-line tracer,
    /// traced "coupler.M"), W2 off the rocker line.
    const CHEBYSHEV_WEIGHTS: Weights =
        &[("coupler", 50.0, [0.1838, 0.0]), ("rocker", 5.0, [0.0, 0.02])];
    /// FourBar (no actuator) weights.
    const FOURBAR_WEIGHTS: Weights =
        &[("coupler", 1.0, [0.02, 0.005]), ("rocker", 0.5, [0.0, 0.0])];

    fn add_weights(state: &mut AppState, weights: &[(&str, f64, [f64; 2])]) {
        for &(body, mass, pos) in weights {
            state.add_point_mass(body, mass, pos).expect("weight added");
        }
    }

    /// Load `sample`, optionally put its actuator in sizing mode (stored
    /// force 0), add `weights` (body, kg, body-local m), sweep 0..=360 deg.
    fn swept(sample: SampleMechanism, sizing: bool, weights: &[(&str, f64, [f64; 2])]) -> AppState {
        let mut state = AppState::default();
        state.load_sample(sample);
        if sizing {
            set_actuator_stored_force(&mut state, 0.0);
        }
        add_weights(&mut state, weights);
        state.compute_sweep();
        state
    }

    fn breakdown(state: &AppState) -> (&SweepData, &WeightBreakdown) {
        let data = state.sweep_data.as_ref().expect("sweep computed");
        (data, data.weight_breakdown.as_ref().expect("breakdown computed"))
    }

    fn source_ids(b: &WeightBreakdown) -> Vec<&str> {
        b.sources.iter().map(|s| s.id.as_str()).collect()
    }

    fn source_index(b: &WeightBreakdown, id: &str) -> usize {
        b.sources.iter().position(|s| s.id == id).unwrap_or_else(|| panic!("no source {id}"))
    }

    fn sample_at(data: &SweepData, deg: f64) -> usize {
        data.angles_deg
            .iter()
            .position(|&a| (a - deg).abs() < 1e-9)
            .unwrap_or_else(|| panic!("no sample at {deg} deg"))
    }

    fn assert_close(what: &str, data: &SweepData, k: usize, got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "{what} at {} deg: {got} vs {want} (tol {tol})",
            data.angles_deg[k]
        );
    }

    /// Every per-sample series has one entry per sweep sample.
    fn assert_aligned(data: &SweepData, b: &WeightBreakdown) {
        let n = data.angles_deg.len();
        assert!(n > 0, "empty sweep");
        for (name, per_source) in [
            ("gravity_power", &b.gravity_power),
            ("force_share", &b.force_share),
            ("power_share", &b.power_share),
        ] {
            assert_eq!(per_source.len(), b.sources.len(), "{name}: one series per source");
            for (i, series) in per_source.iter().enumerate() {
                assert_eq!(series.len(), n, "{name}[{}]", b.sources[i].id);
            }
        }
        for (name, len) in [
            ("other_force", b.other_force.len()),
            ("other_power", b.other_power.len()),
            ("total_force", b.total_force.len()),
            ("total_power", b.total_power.len()),
            ("braking", b.braking.len()),
        ] {
            assert_eq!(len, n, "{name}");
        }
    }

    /// Sum invariants of a sweep where gravity is the only load: shares plus
    /// remainder give the totals at every valid sample, and the weights
    /// explain the whole load (remainder ~ 0).
    fn assert_gravity_only_sum_invariants(data: &SweepData, b: &WeightBreakdown) {
        let n = data.angles_deg.len();
        let valid_force: Vec<usize> = (0..n)
            .filter(|&k| b.total_force[k].is_finite() && b.force_share.iter().all(|s| s[k].is_finite()))
            .collect();
        assert!(valid_force.len() > n / 2, "only {} of {n} samples have finite force shares", valid_force.len());
        let peak_force = valid_force.iter().fold(0.0_f64, |m, &k| m.max(b.total_force[k].abs()));
        let peak_power = max_abs_finite(&b.total_power);
        assert!(peak_force > 0.0 && peak_power > 0.0, "the weights load the driver");
        for &k in &valid_force {
            let sum: f64 = b.force_share.iter().map(|s| s[k]).sum();
            assert_close("sum(F_i) + F_other", data, k, sum + b.other_force[k], b.total_force[k], ROUND_OFF * peak_force);
            assert_close("F_other (gravity is the only load)", data, k, b.other_force[k], 0.0, GRAVITY_ONLY * peak_force);
        }
        let mut n_power = 0;
        for k in (0..n).filter(|&k| b.total_power[k].is_finite()) {
            n_power += 1;
            let sum: f64 = b.power_share.iter().map(|s| s[k]).sum();
            assert_close("sum(P_i) + P_other", data, k, sum + b.other_power[k], b.total_power[k], ROUND_OFF * peak_power);
            assert_close("P_other (gravity is the only load)", data, k, b.other_power[k], 0.0, GRAVITY_ONLY * peak_power);
        }
        assert!(n_power >= valid_force.len(), "power is defined wherever force shares are");
    }

    /// The breakdown totals are the plotted actuator series (the required
    /// force in sizing and stored-force mode alike since BL-026).
    fn assert_totals_match_actuator_plot(data: &SweepData, b: &WeightBreakdown) {
        assert_eq!(b.basis, ShareBasis::ActuatorForce);
        let forces = data.actuator_forces.as_ref().expect("actuator forces");
        let powers = data.actuator_power.as_ref().expect("actuator power");
        let peak_power = max_abs_finite(powers);
        for k in 0..data.angles_deg.len() {
            if forces[k].is_finite() {
                assert_eq!(b.total_force[k], forces[k], "total_force at {} deg", data.angles_deg[k]);
            }
            if powers[k].is_finite() {
                assert_close("total_power vs actuator_power", data, k, b.total_power[k], powers[k], ROUND_OFF * peak_power);
            }
        }
    }

    #[test]
    fn parallelogram_sizing_breakdown_sums_to_actuator_force_and_power() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W2", "W1"]);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    #[test]
    fn chebyshev_sizing_breakdown_sums_to_actuator_force_and_power() {
        let state = swept(SampleMechanism::ChebyshevLambdaActuator, true, CHEBYSHEV_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W1", "W2"]);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    /// Stored-force mode and sizing mode move the same way under the same
    /// loads, and since BL-026 the sweep reports the REQUIRED actuator force
    /// in both, so the two breakdowns must be identical: totals, shares,
    /// remainders, braking and classification. In both, the total is the
    /// plotted actuator force exactly. Adding the stored force back on top
    /// (the pre-BL-026 workaround) would double count it and fail here.
    #[test]
    fn stored_force_mode_and_sizing_mode_give_identical_breakdowns() {
        for (sample, weights) in [
            (SampleMechanism::ParallelogramActuator, PARALLELOGRAM_WEIGHTS),
            (SampleMechanism::ChebyshevLambdaActuator, CHEBYSHEV_WEIGHTS),
        ] {
            let stored = swept(sample, false, weights);
            let sizing = swept(sample, true, weights);
            let stored_force = stored
                .mechanism
                .as_ref()
                .unwrap()
                .forces()
                .iter()
                .find_map(|f| match f {
                    ForceElement::LinearActuator(la) => Some(la.force),
                    _ => None,
                })
                .expect("actuator");
            assert!(stored_force.abs() > 1.0, "{sample:?} ships with a stored force");
            let (data, b) = breakdown(&stored);
            let (sizing_data, sizing_b) = breakdown(&sizing);
            assert_eq!(data.angles_deg, sizing_data.angles_deg, "{sample:?}: same samples");
            assert_eq!(source_ids(b), source_ids(sizing_b), "{sample:?}: same sources");
            assert_aligned(data, b);
            assert_totals_match_actuator_plot(data, b);
            assert_gravity_only_sum_invariants(data, b);

            let force_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.total_force).max(stored_force.abs());
            let power_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.total_power);
            let name = |series: &str| format!("{sample:?} {series}");
            assert_series(&name("total_force"), &b.total_force, &sizing_b.total_force, force_tol);
            assert_series(&name("other_force"), &b.other_force, &sizing_b.other_force, force_tol);
            assert_series(&name("total_power"), &b.total_power, &sizing_b.total_power, power_tol);
            assert_series(&name("other_power"), &b.other_power, &sizing_b.other_power, power_tol);
            for (i, s) in b.sources.iter().enumerate() {
                let p_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.gravity_power[i]);
                let f_tol = CROSS_SWEEP * max_abs_finite(&sizing_b.force_share[i]);
                let id = &s.id;
                assert_series(&name(&format!("gravity_power[{id}]")), &b.gravity_power[i], &sizing_b.gravity_power[i], p_tol);
                assert_series(&name(&format!("force_share[{id}]")), &b.force_share[i], &sizing_b.force_share[i], f_tol);
                assert_series(&name(&format!("power_share[{id}]")), &b.power_share[i], &sizing_b.power_share[i], p_tol);
                for k in 0..data.angles_deg.len() {
                    let deg = data.angles_deg[k];
                    assert_eq!(b.classification(i, k), sizing_b.classification(i, k), "{sample:?} {id} at {deg} deg");
                }
            }
            assert_eq!(b.braking, sizing_b.braking, "{sample:?}: braking");
        }
    }

    /// Parallelogram: the rocker turns with the crank, so the rocker tip W1
    /// is at O4 + 2 (cos t, sin t) and, with omega > 0, rises while
    /// cos t > 0. At 135 deg every link and weight is coming down, so
    /// gravity drives the actuator (braking); at 45 deg the actuator lifts
    /// them all (motoring); at 90 deg they all move horizontally.
    #[test]
    fn parallelogram_classification_and_braking_at_known_poses() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        let tip = source_index(b, "W1");
        let (k45, k90, k135) = (sample_at(data, 45.0), sample_at(data, 90.0), sample_at(data, 135.0));
        for i in 0..b.sources.len() {
            let id = &b.sources[i].id;
            assert_eq!(b.classification(i, k45), Hurting, "{id} rising at 45 deg");
            assert_eq!(b.classification(i, k90), Neutral, "{id} moving horizontally at 90 deg");
            assert_eq!(b.classification(i, k135), Helping, "{id} coming down at 135 deg");
        }
        assert!(!b.braking[k45], "the actuator lifts the load at 45 deg");
        assert!(b.braking[k135], "the load drives the actuator at 135 deg");
        assert!(b.power_share[tip][k45] > 0.0, "lifting W1 costs actuator power");
        assert!(b.power_share[tip][k135] < 0.0, "lowering W1 gives power back");
    }

    /// Chebyshev: W1 rides on coupler point M, whose path the sweep traces
    /// as "coupler.M". Wherever M clearly moves up (central difference of the
    /// trace) W1 is hurting, wherever it clearly moves down W1 is helping
    /// (M runs nearly level along its straight-line stretch, so only part of
    /// the cycle qualifies). With gravity the only load, the actuator brakes
    /// exactly where the weights' net gravity power is positive.
    #[test]
    fn chebyshev_classification_follows_the_weight_motion_and_braking_follows_net_gravity_power() {
        let state = swept(SampleMechanism::ChebyshevLambdaActuator, true, CHEBYSHEV_WEIGHTS);
        let (data, b) = breakdown(&state);
        let w1 = source_index(b, "W1");
        let trace = &data.coupler_traces["coupler.M"];
        let n = trace.len();
        let rise: Vec<f64> = (0..n)
            .map(|k| if k == 0 || k + 1 == n { f64::NAN } else { trace[k + 1][1] - trace[k - 1][1] })
            .collect();
        let max_rise = max_abs_finite(&rise);
        let mut checked = 0;
        for k in (0..n).filter(|&k| rise[k].abs() > 0.05 * max_rise) {
            let expected = if rise[k] > 0.0 { Hurting } else { Helping };
            assert_eq!(b.classification(w1, k), expected, "W1 at {} deg (rise {})", data.angles_deg[k], rise[k]);
            checked += 1;
        }
        assert!(checked > n / 4, "only {checked} samples with clear vertical motion");

        let net: Vec<f64> = (0..n).map(|k| b.gravity_power.iter().map(|s| s[k]).sum()).collect();
        let max_net = max_abs_finite(&net);
        let (mut n_braking, mut n_motoring) = (0, 0);
        for k in (0..n).filter(|&k| net[k].abs() > 1e-3 * max_net) {
            assert_eq!(b.braking[k], net[k] > 0.0, "braking at {} deg, net gravity power {}", data.angles_deg[k], net[k]);
            if b.braking[k] {
                n_braking += 1;
            } else {
                n_motoring += 1;
            }
        }
        assert!(n_braking > 0 && n_motoring > 0, "a lift cycle both motors and brakes");
    }

    /// Near stroke reversal (|dL/dt| < 1 % of its sweep maximum) the force
    /// shares and the force remainder are NaN; the power shares, total power
    /// and power remainder stay defined because they do not divide by dL/dt.
    #[test]
    fn force_share_is_nan_exactly_near_stroke_reversal_and_power_share_stays_defined() {
        let state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (data, b) = breakdown(&state);
        let speeds = data.actuator_speeds.as_ref().expect("actuator speeds");
        let max_speed = max_abs_finite(speeds);
        let mut n_reversal = 0;
        for (k, &ldot) in speeds.iter().enumerate() {
            assert!(ldot.is_finite(), "every parallelogram sample solves (dL/dt at {} deg)", data.angles_deg[k]);
            let near_reversal = ldot.abs() < EPS_REL_LDOT * max_speed;
            n_reversal += usize::from(near_reversal);
            for i in 0..b.sources.len() {
                let id = &b.sources[i].id;
                let deg = data.angles_deg[k];
                assert_eq!(b.force_share[i][k].is_nan(), near_reversal, "{id} force share at {deg} deg (dL/dt {ldot})");
                assert!(b.power_share[i][k].is_finite(), "{id} power share at {deg} deg");
            }
            assert_eq!(b.other_force[k].is_nan(), near_reversal, "other_force at {} deg", data.angles_deg[k]);
            assert!(b.total_power[k].is_finite() && b.other_power[k].is_finite());
        }
        assert!(n_reversal > 0, "the actuator reverses twice per revolution");
    }

    /// Stretch the ground link (move O4) until the crank cannot pass the far
    /// side, so a band of samples fails to solve (O4 is also moved off the
    /// x axis where needed: from the parallelogram's collinear start pose a
    /// symmetric target never leaves the axis). Every breakdown series still
    /// has one entry per sample: NaN, neutral and not braking where the
    /// sample failed, finite gravity power everywhere else.
    fn assert_forced_failures_stay_aligned(
        sample: SampleMechanism,
        o4: [f64; 2],
        sizing: bool,
        weights: &[(&str, f64, [f64; 2])],
        basis: ShareBasis,
    ) {
        let mut state = AppState::default();
        state.load_sample(sample);
        if sizing {
            set_actuator_stored_force(&mut state, 0.0);
        }
        state
            .blueprint
            .as_mut()
            .unwrap()
            .bodies
            .get_mut("ground")
            .unwrap()
            .attachment_points
            .insert("O4".to_string(), o4);
        state.rebuild();
        assert!(state.solver_status.converged, "{sample:?}: 0 deg must still assemble");
        add_weights(&mut state, weights);
        state.compute_sweep();
        let (data, b) = breakdown(&state);
        assert_eq!(b.basis, basis);
        assert_eq!(data.angles_deg.len(), 361);
        assert_aligned(data, b);
        let n = data.angles_deg.len();
        let failed: Vec<usize> = (0..n).filter(|&k| data.body_angles["crank"][k].is_nan()).collect();
        assert!(failed.len() > 90, "{sample:?}: expected an unreachable band, {} samples failed", failed.len());
        for &k in &failed {
            let deg = data.angles_deg[k];
            for i in 0..b.sources.len() {
                assert!(b.gravity_power[i][k].is_nan(), "{sample:?} gravity_power at {deg} deg");
                assert!(b.force_share[i][k].is_nan(), "{sample:?} force_share at {deg} deg");
                assert!(b.power_share[i][k].is_nan(), "{sample:?} power_share at {deg} deg");
                assert_eq!(b.classification(i, k), Neutral, "{sample:?} classification at {deg} deg");
            }
            for (name, v) in [
                ("other_force", b.other_force[k]),
                ("other_power", b.other_power[k]),
                ("total_force", b.total_force[k]),
                ("total_power", b.total_power[k]),
            ] {
                assert!(v.is_nan(), "{sample:?} {name} at {deg} deg");
            }
            assert!(!b.braking[k], "{sample:?} braking at {deg} deg");
        }
        for k in (0..n).filter(|k| !failed.contains(k)) {
            assert!(
                b.gravity_power.iter().all(|s| s[k].is_finite()),
                "{sample:?}: converged sample at {} deg has a non-finite gravity power",
                data.angles_deg[k]
            );
        }
    }

    #[test]
    fn forced_solver_failures_keep_the_breakdown_aligned_actuator_basis() {
        // |O4| = 5.46: |B - O4| <= coupler 4 + rocker 2 only for crank angles
        // of about -87..104 deg, so about 105..272 deg is unreachable.
        assert_forced_failures_stay_aligned(
            SampleMechanism::ParallelogramActuator,
            [5.4, 0.8],
            true,
            PARALLELOGRAM_WEIGHTS,
            ShareBasis::ActuatorForce,
        );
    }

    #[test]
    fn forced_solver_failures_keep_the_breakdown_aligned_driver_basis() {
        // Crank 0.01 + ground 0.065 > coupler 0.04 + rocker 0.03: about 117..243 deg unreachable.
        assert_forced_failures_stay_aligned(
            SampleMechanism::FourBar,
            [0.065, 0.0],
            false,
            FOURBAR_WEIGHTS,
            ShareBasis::DriverTorque,
        );
    }

    /// No actuator: the shares are driver-torque shares `-P_g / omega`, the
    /// total is the plotted driver torque, and with the constant driver rate
    /// no share is NaN.
    #[test]
    fn fourbar_without_actuator_breaks_down_the_driver_torque() {
        let state = swept(SampleMechanism::FourBar, false, FOURBAR_WEIGHTS);
        let (data, b) = breakdown(&state);
        assert_eq!(b.basis, ShareBasis::DriverTorque);
        assert_eq!(source_ids(b), ["link:coupler", "link:crank", "link:rocker", "W1", "W2"]);
        assert_aligned(data, b);
        assert_gravity_only_sum_invariants(data, b);
        let torques = data.driver_torques.as_ref().expect("driver torques");
        let omega = state.driver_omega();
        let peak_power = max_abs_finite(&b.total_power);
        for (k, &torque) in torques.iter().enumerate() {
            assert_eq!(b.total_force[k], torque, "total = driver torque at {} deg", data.angles_deg[k]);
            assert_close("total_power = T * omega", data, k, b.total_power[k], torque * omega, ROUND_OFF * peak_power);
            for i in 0..b.sources.len() {
                let share = b.force_share[i][k];
                assert!(share.is_finite(), "{} torque share at {} deg", b.sources[i].id, data.angles_deg[k]);
                assert_eq!(share, -b.gravity_power[i][k] / omega);
            }
        }
    }

    /// Gravity comes from the mechanism's element, which carries the
    /// mounting angle: tilted 30 deg, the weights still explain the whole
    /// actuator load, and at 90 deg the rocker tip (moving in -x) is helped
    /// by the now horizontal gravity component instead of being neutral.
    #[test]
    fn breakdown_uses_the_mounting_angle_through_the_gravity_element() {
        let level = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let mut tilted = AppState::default();
        tilted.load_sample(SampleMechanism::ParallelogramActuator);
        set_actuator_stored_force(&mut tilted, 0.0);
        tilted.mounting_angle = 30.0_f64.to_radians();
        add_weights(&mut tilted, PARALLELOGRAM_WEIGHTS);
        tilted.compute_sweep();
        let g = gravity_vector(tilted.mechanism.as_ref().unwrap());
        assert!((g[0] + 9.81 * 0.5).abs() < 1e-9, "gravity tilted by the mounting angle: {g:?}");

        let (data, b) = breakdown(&tilted);
        assert_gravity_only_sum_invariants(data, b);
        let (level_data, level_b) = breakdown(&level);
        let tip = source_index(b, "W1");
        assert_eq!(level_b.classification(tip, sample_at(level_data, 90.0)), Neutral);
        assert_eq!(b.classification(tip, sample_at(data, 90.0)), Helping);
    }

    /// World position of `source` at every sweep sample, rebuilt only from
    /// positions the sweep recorded (no velocity solve): the trace of one
    /// attachment point of the source's body plus that body's angle.
    fn traced_positions(state: &AppState, data: &SweepData, source: &WeightSource) -> Vec<[f64; 2]> {
        let body = &state.blueprint.as_ref().expect("blueprint").bodies[&source.body_id];
        let (point, anchor) = body
            .attachment_points
            .iter()
            .min_by(|a, b| a.0.cmp(b.0))
            .unwrap_or_else(|| panic!("{} has no attachment point", source.body_id));
        let trace = &data.coupler_traces[&format!("{}.{point}", source.body_id)];
        let angles_deg = &data.body_angles[&source.body_id];
        let offset = [source.local_pos[0] - anchor[0], source.local_pos[1] - anchor[1]];
        trace
            .iter()
            .zip(angles_deg)
            .map(|(p, deg)| {
                let (sin, cos) = deg.to_radians().sin_cos();
                [p[0] + cos * offset[0] - sin * offset[1], p[1] + sin * offset[0] + cos * offset[1]]
            })
            .collect()
    }

    /// The sweep feeds each sample's own pose and velocity to the physics:
    /// on both actuator samples, every source (link self-weights and point
    /// masses) has gravity power equal to minus the rate of change of its
    /// potential energy between adjacent sweep samples, a central difference
    /// over +-1 deg of the positions the sweep recorded. On the
    /// parallelogram, samples within 1 deg of its change points (0/180/360
    /// deg, all links collinear) are skipped: the velocity there is not
    /// unique and Newton only reaches the pose to ~sqrt(tolerance). The
    /// Chebyshev crank-rocker has no change point.
    #[test]
    fn sweep_gravity_power_matches_energy_change_between_adjacent_samples() {
        // (sample, weights, change points to skip, tolerance relative to the
        // source's peak |P_g|).
        let cases: [(SampleMechanism, Weights, &[f64], f64); 2] = [
            (SampleMechanism::ParallelogramActuator, PARALLELOGRAM_WEIGHTS, &[0.0, 180.0, 360.0], PARALLELOGRAM_FD_TOL),
            (SampleMechanism::ChebyshevLambdaActuator, CHEBYSHEV_WEIGHTS, &[], CHEBYSHEV_FD_TOL),
        ];
        for (sample, weights, change_points, rel_tol) in cases {
            let state = swept(sample, true, weights);
            let (data, b) = breakdown(&state);
            assert_eq!(b.sources.iter().filter(|s| !s.is_link_self_weight).count(), weights.len());
            let g = gravity_vector(state.mechanism.as_ref().unwrap());
            let dt = 1.0_f64.to_radians() / state.driver_omega(); // one degree per sample
            let n = data.angles_deg.len();
            for (i, s) in b.sources.iter().enumerate() {
                let r = traced_positions(&state, data, s);
                let peak = max_abs_finite(&b.gravity_power[i]);
                assert!(peak > 0.0, "{sample:?} {}: never moves vertically", s.id);
                let mut checked = 0;
                for k in 1..n - 1 {
                    let deg = data.angles_deg[k];
                    if change_points.iter().any(|&c| (deg - c).abs() <= 1.0) {
                        continue;
                    }
                    let (before, after) = (r[k - 1], r[k + 1]);
                    let energy_fd = s.mass * (g[0] * (after[0] - before[0]) + g[1] * (after[1] - before[1])) / (2.0 * dt);
                    let what = format!("{sample:?} {}", s.id);
                    assert_close(&what, data, k, b.gravity_power[i][k], energy_fd, rel_tol * peak);
                    checked += 1;
                }
                assert!(checked > 300, "{sample:?} {}: only {checked} samples checked", s.id);
            }
        }
    }

    /// BL-024: a Mass edit on a link that carries weights re-derives the
    /// live composite body (base + point masses) without a rebuild, so the
    /// breakdown recomputed after the edit still has the weights explain
    /// the whole actuator load, with the new base mass in that link's
    /// self-weight.
    #[test]
    fn breakdown_sum_invariant_holds_after_a_mass_edit_on_a_weighted_link_without_rebuild() {
        let mut state = swept(SampleMechanism::ParallelogramActuator, true, PARALLELOGRAM_WEIGHTS);
        let (_, before) = breakdown(&state);
        let old_mass = before.sources[source_index(before, "link:rocker")].mass;
        let new_mass = old_mass + 7.5;
        assert!(before.sources.iter().any(|s| s.id == "W1" && s.body_id == "rocker"), "the rocker carries W1");

        state.set_body_mass("rocker", new_mass);
        assert!(state.sweep_dirty, "the edit marks the sweep stale");
        // What the update loop does once the debounce expires; no rebuild.
        state.compute_sweep();

        let (data, b) = breakdown(&state);
        assert_eq!(b.sources[source_index(b, "link:rocker")].mass, new_mass);
        assert_eq!(b.sources[source_index(b, "W1")].mass, 50.0);
        assert_aligned(data, b);
        assert_totals_match_actuator_plot(data, b);
        assert_gravity_only_sum_invariants(data, b);
    }

    #[test]
    fn sweep_without_weight_sources_has_no_breakdown() {
        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let omega = 2.0 * std::f64::consts::PI;
        let (plain, _) = compute_sweep_data(&mech, &q0, omega, 0.0, 9.81, None);
        assert!(plain.weight_breakdown.is_none());
        let (empty, _) = compute_sweep_data_with_weights(&mech, &q0, omega, 0.0, 9.81, None, Vec::new());
        assert!(empty.weight_breakdown.is_none());
        assert_eq!(plain.angles_deg, empty.angles_deg);
    }

    fn source(id: &str) -> WeightSource {
        WeightSource {
            id: id.to_string(),
            name: id.to_string(),
            body_id: "bar".to_string(),
            local_pos: [0.0, 0.0],
            mass: 1.0,
            is_link_self_weight: false,
        }
    }

    /// `got` equals `want` sample by sample: NaN at the same samples, finite
    /// values within the absolute tolerance `tol`.
    fn assert_series(name: &str, got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "{name} length");
        for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
            let same = (g.is_nan() && w.is_nan()) || (g - w).abs() <= tol;
            assert!(same, "{name}[{k}]: {g} vs {w} (tol {tol})");
        }
    }

    /// Hand-checkable rows: rates 2, 0.01, -1, then a failed sample. The
    /// sweep-wide max |rate| is 2, so 0.01 (< 1 % of it) is a near-reversal
    /// sample; the max |P_act| is 10, so -1e-9 is inside the braking
    /// tolerance.
    #[test]
    fn builder_applies_the_sweep_wide_rules() {
        let mut builder = WeightBreakdownBuilder::new(vec![source("A"), source("B")], ShareBasis::ActuatorForce, [0.0, -9.81], 4);
        builder.push_row(&[4.0, -2.0], 2.0, 5.0, 10.0);
        builder.push_row(&[0.03, 0.5], 0.01, 7.0, -3.0);
        builder.push_row(&[-2.0, 6.0], -1.0, 3.0, -1e-9);
        builder.push_nan();
        let b = builder.finish();
        let nan = f64::NAN;
        // Hand values of magnitude <= 12: exact up to round-off.
        const HAND: f64 = 1e-11;
        assert_eq!(b.basis, ShareBasis::ActuatorForce);
        assert_series("gravity_power[0]", &b.gravity_power[0], &[4.0, 0.03, -2.0, nan], HAND);
        assert_series("force_share[0]", &b.force_share[0], &[-2.0, nan, -2.0, nan], HAND);
        assert_series("force_share[1]", &b.force_share[1], &[1.0, nan, 6.0, nan], HAND);
        assert_series("power_share[0]", &b.power_share[0], &[-4.0, -0.03, 2.0, nan], HAND);
        assert_series("power_share[1]", &b.power_share[1], &[2.0, -0.5, -6.0, nan], HAND);
        assert_series("total_force", &b.total_force, &[5.0, 7.0, 3.0, nan], HAND);
        assert_series("total_power", &b.total_power, &[10.0, -3.0, -1e-9, nan], HAND);
        assert_series("other_force", &b.other_force, &[6.0, nan, -1.0, nan], HAND);
        assert_series("other_power", &b.other_power, &[12.0, -2.47, 4.0 - 1e-9, nan], HAND);
        assert_eq!(b.braking, vec![false, true, false, false]);
        // Source A: max |P_g| 4, neutral band 0.04. Source B: max 6, band 0.06.
        let classes = |i: usize| (0..4).map(|k| b.classification(i, k)).collect::<Vec<_>>();
        assert_eq!(classes(0), vec![Helping, Neutral, Hurting, Neutral]);
        assert_eq!(classes(1), vec![Hurting, Helping, Helping, Neutral]);
        assert_eq!(b.classification(2, 0), Neutral, "source index out of range");
        assert_eq!(b.classification(0, 4), Neutral, "sample index out of range");
    }

    #[test]
    fn required_totals_take_the_required_force_as_is_with_a_reversal_safe_power() {
        use ShareBasis::{ActuatorForce, DriverTorque};
        // Sizing: T = 10 N*m at omega = 2 rad/s over dL/dt = 4 m/s gives
        // F = 5 N and P = 20 W.
        let sizing = required_actuator_force(10.0, 2.0, 4.0, 0.0).unwrap();
        assert_eq!(required_totals(ActuatorForce, 10.0, 2.0, 4.0, sizing, 0.0), (5.0, 20.0));
        // Stored 3 N: statics already applied it, and since BL-026 the
        // sweep's required force includes it (10 * 2 / 4 + 3 = 8 N). The
        // total takes it unchanged, and P = F * dL/dt = 32 W.
        let stored = required_actuator_force(10.0, 2.0, 4.0, 3.0).unwrap();
        assert_eq!(stored, 8.0);
        assert_eq!(required_totals(ActuatorForce, 10.0, 2.0, 4.0, stored, 3.0), (8.0, 32.0));
        // Stroke reversal (dL/dt = 0): no required force, but the power is
        // still the driver's T * omega.
        assert_eq!(required_actuator_force(10.0, 2.0, 0.0, 3.0), None);
        let (force, power) = required_totals(ActuatorForce, 10.0, 2.0, 0.0, f64::NAN, 3.0);
        assert!(force.is_nan());
        assert_eq!(power, 20.0);
        // No actuator: the driver torque and its power.
        assert_eq!(required_totals(DriverTorque, 10.0, 2.0, 2.0, f64::NAN, 0.0), (10.0, 20.0));
        // Failed statics (NaN torque and force) stay NaN, never a fake zero.
        let (force, power) = required_totals(ActuatorForce, f64::NAN, 2.0, 4.0, f64::NAN, 3.0);
        assert!(force.is_nan() && power.is_nan());
    }
}
