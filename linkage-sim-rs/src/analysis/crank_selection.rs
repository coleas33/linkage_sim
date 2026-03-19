//! Crank selection analysis: recommend which link to drive for maximum rotation.
//!
//! For 4-bar mechanisms, uses Grashof classification to determine the optimal
//! crank analytically. Each ground-adjacent link is evaluated as a candidate
//! driver, and candidates are ranked by rotation capability.

use super::grashof::{check_grashof, GrashofType};

/// A candidate link that could serve as the driven crank.
#[derive(Debug, Clone)]
pub struct CrankCandidate {
    /// Label of this link (e.g. "ground", "crank", "coupler", "rocker").
    pub link_name: String,
    /// Length of this link.
    pub link_length: f64,
    /// Whether this link can fully rotate (360 degrees).
    pub can_fully_rotate: bool,
    /// Whether this link is the shortest in the mechanism.
    pub is_shortest: bool,
    /// Estimated angular range in degrees (360.0 for full rotation).
    pub estimated_range_deg: f64,
    /// Human-readable explanation of this candidate's capability.
    pub reason: String,
}

/// Recommendation for which link to use as the crank (driver).
#[derive(Debug, Clone)]
pub struct CrankRecommendation {
    /// Recommended driver link name.
    pub recommended_link: String,
    /// Whether the recommended link can rotate fully (360 degrees).
    pub full_rotation: bool,
    /// Grashof classification of the mechanism.
    pub classification: GrashofType,
    /// All candidates with their rotation capability, sorted best-first.
    pub candidates: Vec<CrankCandidate>,
}

/// Estimate the angular range of a candidate crank by numerical probing.
///
/// Scans evenly-spaced crank angles and checks whether the diagonal distance
/// between the crank tip and the rocker pivot satisfies the triangle inequality
/// with the coupler and rocker. Returns estimated range in degrees (0 to 360).
fn estimate_driven_range(
    ground_len: f64,
    crank_len: f64,
    coupler_len: f64,
    rocker_len: f64,
) -> f64 {
    let n_samples: usize = 3600;
    let lower = (coupler_len - rocker_len).abs();
    let upper = coupler_len + rocker_len;

    let mut valid_count: usize = 0;
    for i in 0..n_samples {
        let theta = (i as f64) * 2.0 * std::f64::consts::PI / (n_samples as f64);
        let d_tip = (crank_len * crank_len + ground_len * ground_len
            - 2.0 * crank_len * ground_len * theta.cos())
        .sqrt();
        if d_tip >= lower && d_tip <= upper {
            valid_count += 1;
        }
    }

    (valid_count as f64 / n_samples as f64) * 360.0
}

/// For a 4-bar mechanism, recommend which link to drive.
///
/// Evaluates each ground-adjacent link (crank and rocker) as a potential driver
/// using Grashof's theorem. The shortest link can fully rotate if and only if
/// S + L <= P + Q (Grashof condition).
///
/// Rules:
/// - CrankRocker: drive the shortest link (it's the crank that fully rotates)
/// - DoubleCrank: either grounded link works; both fully rotate
/// - DoubleRocker: no link can fully rotate; pick the one with greater range
/// - ChangePoint: any configuration possible but may have dead points
/// - NonGrashof: no full rotation; pick the link with greater range
///
/// The four lengths are given in the conventional order: ground (fixed frame
/// distance between pivots), crank (input), coupler (floating), rocker (output).
pub fn recommend_crank(
    ground_length: f64,
    crank_length: f64,
    coupler_length: f64,
    rocker_length: f64,
) -> CrankRecommendation {
    let result = check_grashof(
        ground_length,
        crank_length,
        coupler_length,
        rocker_length,
        1e-10,
    );

    let links: [(&str, f64); 4] = [
        ("ground", ground_length),
        ("crank", crank_length),
        ("coupler", coupler_length),
        ("rocker", rocker_length),
    ];

    let shortest_len = links
        .iter()
        .map(|(_, l)| *l)
        .fold(f64::INFINITY, f64::min);

    // Build candidates for ground-adjacent links only (crank and rocker).
    // Ground and coupler cannot serve as drivers in a standard 4-bar.
    let mut candidates: Vec<CrankCandidate> = Vec::new();

    for &(name, len) in &[("crank", crank_length), ("rocker", rocker_length)] {
        let is_shortest = (len - shortest_len).abs() < 1e-10;

        // When this link is used as the driver, the "other" grounded link
        // becomes the output. Compute Grashof classification with this
        // link in the crank position.
        let other_len = if name == "crank" {
            rocker_length
        } else {
            crank_length
        };

        let trial =
            check_grashof(ground_length, len, coupler_length, other_len, 1e-10);

        let (can_fully_rotate, estimated_range_deg, reason) = match trial.classification {
            GrashofType::CrankRocker => {
                if trial.shortest_is == "crank" {
                    // This link is shortest and can fully rotate.
                    (
                        true,
                        360.0,
                        format!(
                            "Crank-rocker: '{}' is the shortest link ({:.3}) and can fully rotate.",
                            name, len
                        ),
                    )
                } else {
                    let range = estimate_driven_range(
                        ground_length,
                        len,
                        coupler_length,
                        other_len,
                    );
                    (
                        false,
                        range,
                        format!(
                            "Crank-rocker: '{}' is not the shortest link; limited to ~{:.1} deg.",
                            name, range
                        ),
                    )
                }
            }
            GrashofType::DoubleCrank => (
                true,
                360.0,
                format!(
                    "Double-crank: ground is shortest, '{}' can fully rotate.",
                    name
                ),
            ),
            GrashofType::ChangePoint => (
                true,
                360.0,
                format!(
                    "Change-point: '{}' can fully rotate (may lock at dead points).",
                    name
                ),
            ),
            _ => {
                // DoubleRocker or NonGrashof: limited range.
                let range = estimate_driven_range(
                    ground_length,
                    len,
                    coupler_length,
                    other_len,
                );
                (
                    false,
                    range,
                    format!(
                        "'{}' cannot fully rotate; estimated range ~{:.1} deg.",
                        name, range
                    ),
                )
            }
        };

        candidates.push(CrankCandidate {
            link_name: name.to_string(),
            link_length: len,
            can_fully_rotate,
            is_shortest,
            estimated_range_deg,
            reason,
        });
    }

    // Sort: full-rotation candidates first, then by range descending.
    candidates.sort_by(|a, b| {
        let key_a = (!a.can_fully_rotate, -a.estimated_range_deg as i64);
        let key_b = (!b.can_fully_rotate, -b.estimated_range_deg as i64);
        key_a.cmp(&key_b)
    });

    let recommended = candidates
        .first()
        .map(|c| c.link_name.clone())
        .unwrap_or_else(|| "crank".to_string());
    let full_rotation = candidates
        .first()
        .is_some_and(|c| c.can_fully_rotate);

    CrankRecommendation {
        recommended_link: recommended,
        full_rotation,
        classification: result.classification,
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crank_rocker_recommends_shortest_link() {
        // ground=4, crank=1, coupler=3.5, rocker=2.5
        // S=1 (crank), L=4, P=2.5, Q=3.5 => S+L=5, P+Q=6 => Grashof, crank-rocker
        let rec = recommend_crank(4.0, 1.0, 3.5, 2.5);
        assert_eq!(rec.classification, GrashofType::CrankRocker);
        assert_eq!(rec.recommended_link, "crank");
        assert!(rec.full_rotation);
        assert_eq!(rec.candidates.len(), 2);

        // The recommended candidate (crank) should have full rotation.
        let best = &rec.candidates[0];
        assert!(best.can_fully_rotate);
        assert!((best.estimated_range_deg - 360.0).abs() < 1e-6);

        // The other candidate (rocker) should have limited range.
        let other = &rec.candidates[1];
        assert!(!other.can_fully_rotate);
        assert!(other.estimated_range_deg < 360.0);
        assert!(other.estimated_range_deg > 0.0);
    }

    #[test]
    fn double_crank_both_links_can_fully_rotate() {
        // ground=1, crank=2.5, coupler=3.5, rocker=4
        // S=1 (ground), L=4, P=2.5, Q=3.5 => S+L=5, P+Q=6 => Grashof, double-crank
        let rec = recommend_crank(1.0, 2.5, 3.5, 4.0);
        assert_eq!(rec.classification, GrashofType::DoubleCrank);
        assert!(rec.full_rotation);

        // Both ground-adjacent links should fully rotate.
        for c in &rec.candidates {
            assert!(
                c.can_fully_rotate,
                "In double-crank, '{}' should fully rotate",
                c.link_name
            );
            assert!(
                (c.estimated_range_deg - 360.0).abs() < 1e-6,
                "Expected 360 deg for '{}', got {}",
                c.link_name,
                c.estimated_range_deg
            );
        }
    }

    #[test]
    fn non_grashof_no_full_rotation() {
        // ground=1, crank=5, coupler=2, rocker=2
        // S=1, L=5, P=2, Q=2 => S+L=6, P+Q=4 => NOT Grashof
        let rec = recommend_crank(1.0, 5.0, 2.0, 2.0);
        assert_eq!(rec.classification, GrashofType::NonGrashof);
        assert!(!rec.full_rotation);

        // No candidate should have full rotation.
        for c in &rec.candidates {
            assert!(
                !c.can_fully_rotate,
                "Non-Grashof '{}' should not fully rotate",
                c.link_name
            );
        }
    }

    #[test]
    fn change_point_allows_full_rotation() {
        // ground=4, crank=1, coupler=3, rocker=2
        // S=1, L=4, P=2, Q=3 => S+L=5, P+Q=5 => change-point
        let rec = recommend_crank(4.0, 1.0, 3.0, 2.0);
        assert_eq!(rec.classification, GrashofType::ChangePoint);
        assert!(rec.full_rotation);

        // At least one candidate should have full rotation.
        let any_full = rec.candidates.iter().any(|c| c.can_fully_rotate);
        assert!(any_full, "Change-point should have at least one full-rotation candidate");
    }
}
