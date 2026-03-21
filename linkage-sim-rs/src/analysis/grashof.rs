//! Grashof condition check for 4-bar linkages.
//!
//! Classifies a 4-bar linkage based on the Grashof criterion:
//!     S + L <= P + Q
//!
//! where S = shortest link, L = longest link, P and Q are the other two.

/// Classification of a 4-bar linkage by Grashof condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrashofType {
    /// Shortest link is crank or rocker; it can fully rotate.
    CrankRocker,
    /// Shortest link is the ground; both crank and rocker fully rotate.
    DoubleCrank,
    /// Shortest link is the coupler; neither crank nor rocker fully rotates,
    /// but the mechanism satisfies S + L < P + Q.
    DoubleRocker,
    /// S + L == P + Q (within tolerance); mechanism can lock.
    ChangePoint,
    /// S + L > P + Q; no link can fully rotate.
    NonGrashof,
}

/// Result of Grashof condition analysis.
#[derive(Debug, Clone)]
pub struct GrashofResult {
    /// Link lengths in order: [ground, crank, coupler, rocker].
    pub link_lengths: [f64; 4],
    /// Length of shortest link.
    pub shortest: f64,
    /// Length of longest link.
    pub longest: f64,
    /// S + L.
    pub grashof_sum: f64,
    /// P + Q (the other two).
    pub other_sum: f64,
    /// True if S + L <= P + Q.
    pub is_grashof: bool,
    /// True if S + L == P + Q (within tolerance).
    pub is_change_point: bool,
    /// Grashof classification.
    pub classification: GrashofType,
    /// Which link is shortest: "ground", "crank", "coupler", or "rocker".
    pub shortest_is: &'static str,
}

/// Classify a 4-bar linkage by the Grashof condition.
///
/// The four link lengths are: ground (fixed frame distance between pivots),
/// crank (input), coupler (connecting), rocker (output).
///
/// Classification rules:
/// - If S + L > P + Q: NonGrashof (no link can fully rotate)
/// - If S + L == P + Q: ChangePoint (special case, can lock)
/// - If S + L < P + Q:
///     - Shortest is ground -> DoubleCrank
///     - Shortest is crank or rocker -> CrankRocker
///     - Shortest is coupler -> DoubleRocker
pub fn check_grashof(
    ground_length: f64,
    crank_length: f64,
    coupler_length: f64,
    rocker_length: f64,
    tol: f64,
) -> GrashofResult {
    let named: [(&str, f64); 4] = [
        ("ground", ground_length),
        ("crank", crank_length),
        ("coupler", coupler_length),
        ("rocker", rocker_length),
    ];

    let mut sorted_lengths = [ground_length, crank_length, coupler_length, rocker_length];
    sorted_lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let s = sorted_lengths[0];
    let l = sorted_lengths[3];
    let p = sorted_lengths[1];
    let q_val = sorted_lengths[2];

    let grashof_sum = s + l;
    let other_sum = p + q_val;

    // Find which link name is shortest
    let shortest_is = named
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0;

    let is_change_point = (grashof_sum - other_sum).abs() < tol;
    let is_grashof = grashof_sum <= other_sum + tol;

    let classification = if is_change_point {
        GrashofType::ChangePoint
    } else if !is_grashof {
        GrashofType::NonGrashof
    } else {
        // Grashof: classify by which link is shortest
        match shortest_is {
            "ground" => GrashofType::DoubleCrank,
            "crank" | "rocker" => GrashofType::CrankRocker,
            _ => GrashofType::DoubleRocker, // coupler
        }
    };

    GrashofResult {
        link_lengths: [ground_length, crank_length, coupler_length, rocker_length],
        shortest: s,
        longest: l,
        grashof_sum,
        other_sum,
        is_grashof,
        is_change_point,
        classification,
        shortest_is,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crank_rocker_classification() {
        // Classic crank-rocker: crank is shortest, S+L < P+Q
        // ground=4, crank=1, coupler=3, rocker=2
        // S=1(crank), L=4(ground), P=2, Q=3
        // S+L=5, P+Q=5 -> actually change point with these numbers
        // Let's use: ground=4, crank=1, coupler=3.5, rocker=2.5
        // S=1, L=4, P=2.5, Q=3.5 => S+L=5, P+Q=6 => Grashof, shortest=crank
        let result = check_grashof(4.0, 1.0, 3.5, 2.5, 1e-10);
        assert_eq!(result.classification, GrashofType::CrankRocker);
        assert!(result.is_grashof);
        assert!(!result.is_change_point);
        assert_eq!(result.shortest_is, "crank");
    }

    #[test]
    fn double_crank_classification() {
        // Ground is shortest: ground=1, crank=2.5, coupler=3.5, rocker=4
        // S=1(ground), L=4, P=2.5, Q=3.5 => S+L=5, P+Q=6 => Grashof
        let result = check_grashof(1.0, 2.5, 3.5, 4.0, 1e-10);
        assert_eq!(result.classification, GrashofType::DoubleCrank);
        assert!(result.is_grashof);
        assert_eq!(result.shortest_is, "ground");
    }

    #[test]
    fn double_rocker_classification() {
        // Coupler is shortest: ground=3, crank=2.5, coupler=1, rocker=4
        // S=1(coupler), L=4, P=2.5, Q=3 => S+L=5, P+Q=5.5 => Grashof
        let result = check_grashof(3.0, 2.5, 1.0, 4.0, 1e-10);
        assert_eq!(result.classification, GrashofType::DoubleRocker);
        assert!(result.is_grashof);
        assert_eq!(result.shortest_is, "coupler");
    }

    #[test]
    fn non_grashof_classification() {
        // S+L > P+Q: ground=1, crank=5, coupler=2, rocker=2
        // sorted: [1,2,2,5] => S=1, L=5, P=2, Q=2
        // S+L=6, P+Q=4 => NOT Grashof
        let result = check_grashof(1.0, 5.0, 2.0, 2.0, 1e-10);
        assert_eq!(result.classification, GrashofType::NonGrashof);
        assert!(!result.is_grashof);
    }

    #[test]
    fn change_point_classification() {
        // S+L == P+Q: ground=4, crank=1, coupler=3, rocker=2
        // sorted: [1,2,3,4] => S=1, L=4, P=2, Q=3
        // S+L=5, P+Q=5 => change point
        let result = check_grashof(4.0, 1.0, 3.0, 2.0, 1e-10);
        assert_eq!(result.classification, GrashofType::ChangePoint);
        assert!(result.is_grashof); // S+L == P+Q counts as Grashof
        assert!(result.is_change_point);
    }

    #[test]
    fn change_point_with_repeated_middle_lengths() {
        // Links (1.0, 2.0, 2.0, 3.0) where S+L = P+Q exactly:
        // sorted: [1,2,2,3] => S=1, L=3, P=2, Q=2
        // S+L = 1+3 = 4, P+Q = 2+2 = 4 => change point
        let result = check_grashof(1.0, 2.0, 2.0, 3.0, 1e-10);
        assert_eq!(result.classification, GrashofType::ChangePoint);
        assert!(result.is_grashof);
        assert!(result.is_change_point);
        assert_eq!(result.shortest, 1.0);
        assert_eq!(result.longest, 3.0);
        assert_eq!(result.grashof_sum, 4.0);
        assert_eq!(result.other_sum, 4.0);
    }
}
