//! Shared condition number computation with rank-aware filtering.

/// Compute condition number with rank-aware filtering.
///
/// Given the singular values of a matrix (in descending order) and the
/// number of constraint equations, this function:
///
/// 1. Determines the numerical rank using a tolerance of `1e-10 * sigma_max`.
/// 2. Computes the condition number as `sigma_max / sigma_min` where
///    `sigma_min` is the smallest singular value within the numerical rank
///    (ignoring near-zero trailing values that indicate rank deficiency).
/// 3. Reports whether the system is overconstrained (rank < n_equations).
///
/// This avoids the pitfall of the naive `sv[0] / sv[last]` approach,
/// which inflates the condition number from near-zero singular values
/// that represent rank deficiency rather than ill-conditioning.
///
/// Returns `(condition_number, is_overconstrained)`.
pub fn rank_aware_condition_number(singular_values: &[f64], n_equations: usize) -> (f64, bool) {
    if singular_values.is_empty() || singular_values[0] <= 0.0 {
        return (f64::INFINITY, true);
    }

    let sigma_max = singular_values[0];
    let rank_tol = 1e-10 * sigma_max;
    let rank = singular_values.iter().filter(|&&s| s > rank_tol).count();

    let sigma_min = if rank > 0 {
        singular_values[rank.min(singular_values.len()) - 1]
    } else {
        0.0
    };

    let condition_number = if sigma_min > 0.0 {
        sigma_max / sigma_min
    } else {
        f64::INFINITY
    };

    (condition_number, rank < n_equations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_conditioned_full_rank() {
        let sv = [10.0, 5.0, 2.0];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 3);
        assert!((cond - 5.0).abs() < 1e-14, "cond = {}", cond);
        assert!(!overconstrained);
    }

    #[test]
    fn rank_deficient_filters_near_zero() {
        // 3 equations but one near-zero SV: effective rank = 2
        let sv = [10.0, 5.0, 1e-15];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 3);
        // sigma_min should be sv[1]=5.0, not the near-zero sv[2]
        assert!((cond - 2.0).abs() < 1e-14, "cond = {}", cond);
        assert!(overconstrained, "rank 2 < 3 equations");
    }

    #[test]
    fn all_zero_singular_values() {
        let sv = [0.0, 0.0];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 2);
        assert!(cond.is_infinite());
        assert!(overconstrained);
    }

    #[test]
    fn empty_singular_values() {
        let sv: [f64; 0] = [];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 0);
        assert!(cond.is_infinite());
        assert!(overconstrained);
    }

    #[test]
    fn single_singular_value() {
        let sv = [7.5];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 1);
        assert!((cond - 1.0).abs() < 1e-14, "cond = {}", cond);
        assert!(!overconstrained);
    }

    #[test]
    fn overconstrained_but_well_conditioned() {
        // 4 equations but only rank 3 (one SV below tolerance)
        let sv = [100.0, 50.0, 25.0, 1e-12];
        let (cond, overconstrained) = rank_aware_condition_number(&sv, 4);
        assert!((cond - 4.0).abs() < 1e-14, "cond = {}", cond);
        assert!(overconstrained);
    }

    #[test]
    fn naive_vs_rank_aware_difference() {
        // The naive approach would give sv[0]/sv[3] = 10 / 1e-14 = 1e15
        // The rank-aware approach gives sv[0]/sv[2] = 10 / 2 = 5
        let sv = [10.0, 5.0, 2.0, 1e-14];
        let (cond, _) = rank_aware_condition_number(&sv, 4);
        assert!((cond - 5.0).abs() < 1e-14, "cond = {} (should be 5, not 1e15)", cond);
    }
}
