//! Addendum A1: housing autofit and the space claim.
//!
//! **Autofit** (report 6.5 classes; decisions 27 and 28). Class D, the housing dimensions
//! the calculator already derives from the magnet layout (the cup OD `model.cup_od_mm`, the
//! pocket corner radius, the ring wall at the flats, the sleeve and liner diameters, the
//! endplate OD), are results: they follow every change of the layout, in both modes. Class R:
//! the cup wall at the corners stays an input, and the wall rule's value is the suggestion
//! `materials.cup_wall_suggested_mm` (decision 27); the sleeve and liner thicknesses keep
//! their clearance check; the inner back apothem is an input (inverse sizing's ring radius).
//! Class N, the dimensions with no rule (cup depth, retainer span, hub length, web, boss,
//! endplates, cap), stay inputs (decision 28); the M41 cap thread below the cup OD and the
//! 22 mm against 25 mm boss OD are M4 design questions.
//!
//! **The axial housing follows the magnets** (Addendum A decision A2-8, option B). With the
//! Rust-only axial length override (`coupling.magnets.axial_length_mm`) set, the three class N
//! dimensions a ring bounds follow that ring's length change from its own length (its part's,
//! or its manual length), so the stacks the space claim reads grow and shrink with the magnets
//! ([`axial_housing`]): the hub length (C123, the steel the inner ring sits on, priced at C112)
//! with the inner ring; the cup cavity depth (C124, the pocket the outer ring sits in, summed by
//! the axial stack C134 and the large-diameter stack C137 and priced at C111) with the outer
//! ring; the retainer span (C172, one span for both the sleeve over the inner ring and the liner
//! inside the outer ring, priced at C46) with the longer ring, since it must cover both. The
//! other class N dimensions bound no magnet: the cap (C133) and the web (C125) are thicknesses
//! at the two ends of the cavity, the boss (C126) is the shaft interface and the endplates
//! (C170, C171) are plates. Each followed dimension keeps its input's margin over its ring and
//! never ends shorter than the ring itself, the physical minimum (the workbook has no axial
//! clearance input to add: its margins, 0.3, 2.8 and 1.8 mm at the defaults, are these inputs,
//! and report 6.5 finds two identities for the cup depth, so neither is a rule). Blank, the
//! three are the inputs, bit for bit. Every result that reads them (the masses and so the heat
//! capacity, both stacks, the hybrid length) reads the values in effect, which `housing.*`
//! shows.
//!
//! **Space claim** (spec A1: "43 mm diameter × 35 mm overall length, and the 20 mm
//! large-diameter bay, from the metal-design inputs"). Each derived dimension against its
//! claim: the rotating OD (Metal design C135) against the diameter (C131), the axial stack
//! (C134) against the overall length (C130), the large-diameter stack (C137) against the bay
//! (C129). The overshoot is exceeded exactly when the dimension exceeds its claim (the
//! workbook's reserve, claim minus dimension, below 0); at the claim is inside. The badge
//! quotes each overshoot to 0.01 mm and at least 0.01 mm, so a dimension past its claim never
//! reads "0.00 mm over"; an axis that is not a number reads unknown, beside the exceeded ones.

use super::compat::{fmt_fixed, py_max};
use super::deviations::Deviations;
use super::meta::{out_rust_only, results};
use super::metal_design::{MetalDesignInputs, MetalDesignResults};
use super::model::{MagnetInputs, resolve_magnets};

results! {
    /// The space claim, per axis (Addendum A1). Rust-only.
    pub struct HousingResults {
        fields {
            diameter_overshoot_mm: f64 => out_rust_only("mm", "Diameter beyond the space claim",
                "Addendum A1: the rotating OD (Metal design C135, the larger of the cup and the cap) minus the claimed diameter (C131) when it is larger; 0 inside or at the claim."),
            length_overshoot_mm: f64 => out_rust_only("mm", "Overall length beyond the space claim",
                "The axial stack (Metal design C134) minus the claimed overall length (C130) when it is larger; 0 inside or at the claim."),
            bay_overshoot_mm: f64 => out_rust_only("mm", "Large-diameter stack beyond its bay",
                "The large-diameter stack (Metal design C137) minus the claimed bay (C129) when it is larger; 0 inside or at the claim."),
            space_claim_check: String => out_rust_only("", "Space claim",
                "The dashboard badge: 'Inside the space claim', or 'Exceeds the space claim:' and each axis it exceeds with the overshoot in mm (at least 0.01), an axis that is not a number reading 'unknown'; 'Space claim unknown' when nothing is exceeded and an axis is not a number."),
            hub_length_mm: f64 => out_rust_only("mm", "Steel inner hub axial length in effect",
                "Metal design C123; with the axial length override set, C123 plus the inner ring's length change, and at least the ring's length (decision A2-8)."),
            cup_depth_mm: f64 => out_rust_only("mm", "Cup cavity axial depth in effect",
                "Metal design C124; with the axial length override set, C124 plus the outer ring's length change, and at least the ring's length (decision A2-8). Both axial stacks sum it."),
            retainer_span_mm: f64 => out_rust_only("mm", "Retainer axial span in effect",
                "Metal design C172; with the axial length override set, C172 plus the longer ring's length change, and at least that ring's length (decision A2-8)."),
        }
    }
}

/// `housing.space_claim_check` when every derived dimension is inside or at its claim.
pub const INSIDE_THE_SPACE_CLAIM: &str = "Inside the space claim";

/// `housing.space_claim_check` when no axis is exceeded and a dimension or a claim is not a
/// number (a value set on the struct; `DesignInputs::validate` names the input).
pub const SPACE_CLAIM_UNKNOWN: &str = "Space claim unknown: a dimension or a claim is not a number";

/// How far a derived dimension passes its claim [mm], from the reserve (claim − dimension):
/// its negative when below 0, else 0; NaN when the reserve is NaN.
fn overshoot(reserve_mm: f64) -> f64 {
    if reserve_mm < 0.0 {
        -reserve_mm
    } else if reserve_mm.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// An overshoot as the badge quotes it [mm]: two decimals, and at least 0.01, so a dimension
/// past its claim by less than 0.005 mm does not read "0.00 mm over".
fn quoted_overshoot(over_mm: f64) -> String {
    fmt_fixed(py_max(over_mm, 0.01), 2)
}

/// The axial housing dimensions in effect (decision A2-8): Metal design C123, C124 and C172.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxialHousing {
    pub hub_length_mm: f64,
    pub cup_depth_mm: f64,
    pub retainer_span_mm: f64,
}

/// A dimension that bounds a ring, following the ring's length change: the input plus
/// (`length_mm` - `own_length_mm`), so it keeps the input's margin over the ring, and at least
/// `length_mm`, the ring itself (the physical minimum). NaN in, NaN out ([`py_max`]).
fn follow(input_mm: f64, length_mm: f64, own_length_mm: f64) -> f64 {
    py_max(input_mm + (length_mm - own_length_mm), length_mm)
}

/// The hub length, cup cavity depth and retainer span in effect. With the override blank, the
/// inputs, untouched. With it set, each follows the ring it bounds ([`follow`]) from the ring's
/// own length (its part's, or its manual length; the override gives both rings one length):
/// the hub the inner ring, the cup cavity the outer ring, the retainer span the longer ring.
pub fn axial_housing(
    md: &MetalDesignInputs,
    magnets: &MagnetInputs,
    dev: Deviations,
) -> AxialHousing {
    let Some(length_mm) = magnets.axial_length_mm else {
        return AxialHousing {
            hub_length_mm: md.hub_length_mm,
            cup_depth_mm: md.cup_depth_mm,
            retainer_span_mm: md.retainer_span_mm,
        };
    };
    let own = MagnetInputs {
        axial_length_mm: None,
        ..magnets.clone()
    };
    let (inner, outer) = resolve_magnets(&own, dev);
    AxialHousing {
        hub_length_mm: follow(md.hub_length_mm, length_mm, inner.length_mm),
        cup_depth_mm: follow(md.cup_depth_mm, length_mm, outer.length_mm),
        retainer_span_mm: follow(
            md.retainer_span_mm,
            length_mm,
            py_max(inner.length_mm, outer.length_mm),
        ),
    }
}

/// The space claim of a design, from its Metal design results, and the axial housing in
/// effect ([`axial_housing`]), which it reports.
pub fn compute(mdr: &MetalDesignResults, axial: &AxialHousing) -> HousingResults {
    let axes = [
        ("diameter", overshoot(mdr.diameter_reserve_mm)),
        ("overall length", overshoot(mdr.axial_reserve_mm)),
        ("large-diameter bay", overshoot(mdr.large_dia_reserve_mm)),
    ];
    let check = if axes.iter().any(|&(_, over)| over > 0.0) {
        // Every axis not inside, in order: its overshoot, or unknown when it is not a number.
        let named: Vec<String> = axes
            .iter()
            .filter_map(|&(axis, over)| {
                if over > 0.0 {
                    Some(format!("{axis} {} mm over", quoted_overshoot(over)))
                } else if over.is_nan() {
                    Some(format!("{axis} unknown"))
                } else {
                    None
                }
            })
            .collect();
        format!("Exceeds the space claim: {}", named.join(", "))
    } else if axes.iter().any(|(_, over)| over.is_nan()) {
        SPACE_CLAIM_UNKNOWN.to_owned()
    } else {
        INSIDE_THE_SPACE_CLAIM.to_owned()
    };
    HousingResults {
        diameter_overshoot_mm: axes[0].1,
        length_overshoot_mm: axes[1].1,
        bay_overshoot_mm: axes[2].1,
        space_claim_check: check,
        hub_length_mm: axial.hub_length_mm,
        cup_depth_mm: axial.cup_depth_mm,
        retainer_span_mm: axial.retainer_span_mm,
    }
}
