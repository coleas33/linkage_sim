//! Result envelopes: peak, RMS, min/max extraction over a sweep.
//!
//! Given arrays of values computed at each step of a position sweep,
//! extract summary statistics useful for engineering sizing decisions.
//! Ported from `linkage_sim.analysis.envelopes`.

/// Summary statistics of a signal over a sweep.
///
/// Captures min, max, mean, RMS, and peak-to-peak range along with
/// the indices at which the extremes occur, enabling downstream code
/// to correlate extreme values with specific driver positions.
#[derive(Debug, Clone)]
pub struct SignalEnvelope {
    /// Minimum value in the signal.
    pub min_value: f64,
    /// Maximum value in the signal.
    pub max_value: f64,
    /// Index at which the minimum occurs.
    pub min_index: usize,
    /// Index at which the maximum occurs.
    pub max_index: usize,
    /// Arithmetic mean of the signal.
    pub mean: f64,
    /// Root mean square of the signal.
    pub rms: f64,
    /// Peak-to-peak range (max - min).
    pub peak_to_peak: f64,
}

/// Compute envelope statistics for a signal array.
///
/// Non-finite values (NaN, infinity) are filtered out before computing
/// statistics. Returns `None` if no finite values remain.
///
/// # Arguments
/// * `values` - Signal values at each sweep step.
///
/// # Returns
/// `Some(SignalEnvelope)` with summary statistics, or `None` if the
/// input is empty or contains only non-finite values.
pub fn compute_envelope(values: &[f64]) -> Option<SignalEnvelope> {
    // Filter out NaN/infinite values, keeping original indices
    let finite: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i, *v))
        .collect();

    if finite.is_empty() {
        return None;
    }

    let mut min_value = finite[0].1;
    let mut max_value = finite[0].1;
    let mut min_index = finite[0].0;
    let mut max_index = finite[0].0;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;

    for &(i, v) in &finite {
        if v < min_value {
            min_value = v;
            min_index = i;
        }
        if v > max_value {
            max_value = v;
            max_index = i;
        }
        sum += v;
        sum_sq += v * v;
    }

    let n = finite.len() as f64;
    let mean = sum / n;
    let rms = (sum_sq / n).sqrt();
    let peak_to_peak = max_value - min_value;

    Some(SignalEnvelope {
        min_value,
        max_value,
        min_index,
        max_index,
        mean,
        rms,
        peak_to_peak,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn envelope_empty_returns_none() {
        assert!(compute_envelope(&[]).is_none());
    }

    #[test]
    fn envelope_all_nan_returns_none() {
        assert!(compute_envelope(&[f64::NAN, f64::NAN, f64::NAN]).is_none());
    }

    #[test]
    fn envelope_all_inf_returns_none() {
        assert!(compute_envelope(&[f64::INFINITY, f64::NEG_INFINITY]).is_none());
    }

    #[test]
    fn envelope_constant_signal() {
        let values = vec![5.0; 10];
        let env = compute_envelope(&values).unwrap();

        assert_abs_diff_eq!(env.min_value, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.max_value, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.mean, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.rms, 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.peak_to_peak, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn envelope_known_values() {
        // values = [1, 2, 3, 4, 5]
        // min=1 at idx 0, max=5 at idx 4
        // mean = 3.0
        // rms = sqrt((1+4+9+16+25)/5) = sqrt(55/5) = sqrt(11)
        // peak_to_peak = 4.0
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let env = compute_envelope(&values).unwrap();

        assert_abs_diff_eq!(env.min_value, 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.max_value, 5.0, epsilon = 1e-15);
        assert_eq!(env.min_index, 0);
        assert_eq!(env.max_index, 4);
        assert_abs_diff_eq!(env.mean, 3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.rms, (11.0_f64).sqrt(), epsilon = 1e-14);
        assert_abs_diff_eq!(env.peak_to_peak, 4.0, epsilon = 1e-15);
    }

    #[test]
    fn envelope_with_nan_values_filtered() {
        // Some NaN values should be skipped
        let values = vec![f64::NAN, 2.0, 4.0, f64::NAN, 6.0];
        let env = compute_envelope(&values).unwrap();

        assert_abs_diff_eq!(env.min_value, 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.max_value, 6.0, epsilon = 1e-15);
        assert_eq!(env.min_index, 1); // index in original array
        assert_eq!(env.max_index, 4);
        assert_abs_diff_eq!(env.mean, 4.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.peak_to_peak, 4.0, epsilon = 1e-15);
    }

    #[test]
    fn envelope_negative_values() {
        let values = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let env = compute_envelope(&values).unwrap();

        assert_abs_diff_eq!(env.min_value, -3.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.max_value, 3.0, epsilon = 1e-15);
        assert_eq!(env.min_index, 0);
        assert_eq!(env.max_index, 4);
        assert_abs_diff_eq!(env.mean, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(env.peak_to_peak, 6.0, epsilon = 1e-15);
        // rms = sqrt((9+1+0+1+9)/5) = sqrt(4) = 2.0
        assert_abs_diff_eq!(env.rms, 2.0, epsilon = 1e-14);
    }
}
