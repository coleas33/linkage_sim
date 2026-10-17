//! BL-059: the gearbox check. The coupling sits on the gearbox output (owner, 2026-10-14): when
//! the wheel is held it passes up to its pull-out before it slips, and while it slips the
//! gearbox carries a fully reversing swing of the same amplitude. With the minimum magnet
//! temperature at or below the operating one, the strongest pull-out expected is cold-high
//! (`metal.torque_cold_high_Nm`: the pull-out at the minimum magnet temperature with the
//! production allowance added); an operating temperature below the minimum is backlog BL-067.
//!
//! The check compares cold-high at the gearbox input, cold-high divided by the ratio and the
//! efficiency (`metal.cold_high_input_Nm`, Metal design C158), with the gearbox input torque
//! rating (`coupling.gearbox_input_rating_Nm`): the same test as cold-high at most rating ×
//! ratio × efficiency at the output, the workbook's `model.gearbox_reference_Nm` (owner
//! decision 1A). In the forward-drive half of each slip cycle the input carries T/(i·η), so
//! the input basis never checks less than the gearbox sees. Fatigue from repeated slips is not
//! checked: the repeated slip swing as a fatigue load is backlog BL-066.
//!
//! At the rating passes, as the hot minimum check passes at the requirement. The badge quotes
//! both torques to 0.01 N·m, the cold-high at least 0.01 N·m above the rating as quoted, so a
//! failing check never shows two equal numbers.

use super::compat::{fmt_fixed, py_max};
use super::meta::{out_rust_only, results};

results! {
    /// The gearbox check (BL-059). Rust-only.
    pub struct GearboxResults {
        fields {
            cold_high_check: String => out_rust_only("", "Cold-high torque at the gearbox input against its rating",
                "Compares cold-high at the gearbox input (cold-high at the gearbox output divided by ratio and efficiency) with the gearbox input torque rating. Fatigue from repeated slips is not checked."),
        }
    }
}

/// `gearbox.cold_high_check` when cold-high at the gearbox input is at or under the rating.
pub const WITHIN_THE_GEARBOX_RATING: &str = "Within the gearbox input rating";

/// The start of `gearbox.cold_high_check` when cold-high at the gearbox input is above the
/// rating; ": {cold-high at the input} N·m against {rating} N·m" follows.
pub const ABOVE_THE_GEARBOX_RATING: &str = "Above the gearbox input rating";

/// `gearbox.cold_high_check` when cold-high at the gearbox input or the rating is not a number
/// (a value set on the struct; `DesignInputs::validate` names the input).
pub const GEARBOX_CHECK_UNKNOWN: &str = "Gearbox check unknown: a torque is not a number";

/// The gearbox check of a design: cold-high at the gearbox input [N·m] against the gearbox
/// input torque rating [N·m]. Equality passes; +∞ fails; NaN in either reads unknown.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn compute(cold_high_input_Nm: f64, rating_Nm: f64) -> GearboxResults {
    let check = if cold_high_input_Nm.is_nan() || rating_Nm.is_nan() {
        GEARBOX_CHECK_UNKNOWN.to_owned()
    } else if cold_high_input_Nm > rating_Nm {
        let rating = fmt_fixed(rating_Nm, 2);
        // The rating as quoted, so the torque quoted beside it is at least 0.01 N·m above it.
        let quoted: f64 = rating.parse().unwrap_or(rating_Nm);
        format!(
            "{ABOVE_THE_GEARBOX_RATING}: {} N·m against {rating} N·m",
            fmt_fixed(py_max(cold_high_input_Nm, quoted + 0.01), 2)
        )
    } else {
        WITHIN_THE_GEARBOX_RATING.to_owned()
    };
    GearboxResults {
        cold_high_check: check,
    }
}
