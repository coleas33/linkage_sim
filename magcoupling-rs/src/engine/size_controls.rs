//! Manual overall dimensions mapped to existing design inputs, with no new torque model.
//!
//! Diameter moves the inner magnet back apothem. The cap outside and thread diameters move
//! by the same cup-diameter change, preserving their existing signed offsets from the cup.
//! A negative thread offset remains a design check, not an allowance we silently repair.
//! The bore, boss, gaps, magnet cross sections and wall thicknesses remain fixed.
//!
//! Overall length sets the existing axial magnet length override. Its housing allowances
//! and ring-length floors come from [`super::housing::axial_housing`]. All baseline axial
//! housing inputs remain unchanged. Editing a dimension already displayed is an exact no-op,
//! including when a blank override has an inconsistent housing shorter than its magnets.
//!
//! These are forward controls: an in-range edit may fail block fit, clearance or another
//! design check. The ordinary engine results expose those checks alongside the torque.

use std::f64::consts::PI;

use super::api::{DesignInputs, compute_all};
use super::deviations::Deviations;
use super::housing::axial_housing;
use super::meta::{SliderRange, input_rows};

/// Current assembly dimensions and the ranges attainable with the other inputs fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeControls {
    pub diameter_mm: f64,
    pub overall_length_mm: f64,
    pub diameter_range: SliderRange,
    pub length_range: SliderRange,
}

/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
struct Geometry {
    controls: SizeControls,
    cup_od_mm: f64,
    diameter_slope: f64,
    cap_offset_mm: f64,
    radius_range: SliderRange,
    axial_range: SliderRange,
}

fn range(min: f64, max: f64) -> Option<SliderRange> {
    (min.is_finite() && max.is_finite() && min > 0.0 && min <= max).then_some(SliderRange {
        min,
        max,
        step: 0.1,
        log: false,
    })
}

fn geometry(inputs: &DesignInputs) -> Option<Geometry> {
    inputs.validate().ok()?;
    let rows = input_rows(inputs);
    let input_range = |path: &str| {
        rows.iter()
            .find(|row| row.path == path)
            .and_then(|row| row.meta.range)
    };
    // Polygon geometry is meaningful for the even pole counts offered by the existing UI.
    let pole_range = input_range("coupling.npole")?;
    let poles = inputs.coupling.npole as f64;
    if poles < pole_range.min || poles > pole_range.max || inputs.coupling.npole % 2 != 0 {
        return None;
    }
    let radius_range = input_range("coupling.inner_back_apothem_mm")?;
    let cap_range = input_range("metal.cap_od_mm")?;
    let thread_range = input_range("metal.cap_thread_dia_mm")?;
    let axial_range = input_range("coupling.magnets.axial_length_mm")?;
    let results = compute_all(inputs);
    let cup = results.model.cup_od_mm;
    let md = &inputs.metal;
    let outside = cup.max(md.cap_od_mm).max(md.boss_od_mm);
    let overall_length = results.metal.axial_stack_mm;
    if ![cup, outside, overall_length]
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return None;
    }
    // With E8 on, outer face apothem = inner back apothem + inner thickness + face gap.
    // The model's flat-block pocket divides its back apothem by cos(pi/N); arcs do not.
    // Read the current cup diameter from compute_all and use only this exact linear slope,
    // rather than duplicate its cross-section, gap, bondline and wall formula.
    let slope = if inputs.coupling.faceted == 1 {
        2.0 / (PI / inputs.coupling.npole as f64).cos()
    } else {
        2.0
    };
    let cap_offset = md.cap_od_mm - cup;
    let thread_offset = md.cap_thread_dia_mm - cup;
    let cap_overhang = cap_offset.max(0.0);
    let radius = inputs.coupling.inner_back_apothem_mm;
    let cup_min = (cup + slope * (radius_range.min - radius))
        .max(cap_range.min - cap_offset)
        .max(thread_range.min - thread_offset);
    let cup_max = (cup + slope * (radius_range.max - radius))
        .min(cap_range.max - cap_offset)
        .min(thread_range.max - thread_offset);
    if !cup_min.is_finite() || !cup_max.is_finite() || cup_min > cup_max {
        return None;
    }
    let diameter_range = range(
        (cup_min + cap_overhang).max(md.boss_od_mm),
        (cup_max + cap_overhang).max(md.boss_od_mm),
    )?;
    let at_length = |length_mm| {
        let mut magnets = inputs.coupling.magnets.clone();
        magnets.axial_length_mm = Some(length_mm);
        let axial = axial_housing(md, &magnets, Deviations::ALL);
        md.cap_axial_mm + axial.cup_depth_mm + md.web_mm + md.boss_length_mm
    };
    let length_range = range(at_length(axial_range.min), at_length(axial_range.max))?;
    Some(Geometry {
        controls: SizeControls {
            diameter_mm: outside,
            overall_length_mm: overall_length,
            diameter_range,
            length_range,
        },
        cup_od_mm: cup,
        diameter_slope: slope,
        cap_offset_mm: cap_offset,
        radius_range,
        axial_range,
    })
}

/// Current outside diameter (cup, cap or boss) and overall axial stack. `None` means the
/// geometry or the intersection of its supported input ranges cannot define these controls.
pub fn dimensions(inputs: &DesignInputs) -> Option<SizeControls> {
    geometry(inputs).map(|geometry| geometry.controls)
}

fn check_request(requested: f64, range: SliderRange, name: &str) -> Result<(), String> {
    if !requested.is_finite() {
        return Err(format!("{name} must be a finite number"));
    }
    if requested < range.min || requested > range.max {
        return Err(format!(
            "{name} must be between {:.4} and {:.4} mm with the other dimensions fixed",
            range.min, range.max
        ));
    }
    Ok(())
}

/// Select an outside diameter while preserving fixed dimensions and the cap's existing
/// signed offsets from the cup. Requests outside the supported range are refused.
pub fn resize_diameter(inputs: &DesignInputs, requested: f64) -> Result<DesignInputs, String> {
    let geometry = geometry(inputs).ok_or("The current geometry cannot define a size control")?;
    if requested == geometry.controls.diameter_mm {
        return Ok(inputs.clone());
    }
    check_request(
        requested,
        geometry.controls.diameter_range,
        "Outside diameter",
    )?;
    let requested_cup = requested - geometry.cap_offset_mm.max(0.0);
    let change = requested_cup - geometry.cup_od_mm;
    let mut changed = inputs.clone();
    // The request was checked before this clamp: it only absorbs endpoint rounding in the
    // algebraic inverse, e.g. a radius a few ulps below its metadata minimum.
    changed.coupling.inner_back_apothem_mm = (inputs.coupling.inner_back_apothem_mm
        + change / geometry.diameter_slope)
        .clamp(geometry.radius_range.min, geometry.radius_range.max);
    changed.metal.cap_od_mm += change;
    changed.metal.cap_thread_dia_mm += change;
    Ok(changed)
}

/// Select an overall axial stack by setting both rings' existing axial length override.
/// The housing module retains each ring's baseline margins and its physical length floor.
pub fn resize_length(inputs: &DesignInputs, requested: f64) -> Result<DesignInputs, String> {
    let geometry = geometry(inputs).ok_or("The current geometry cannot define a size control")?;
    if requested == geometry.controls.overall_length_mm {
        return Ok(inputs.clone());
    }
    check_request(requested, geometry.controls.length_range, "Overall length")?;
    let length = requested - geometry.controls.length_range.min + geometry.axial_range.min;
    let mut changed = inputs.clone();
    // Only floating-point endpoint rounding is clamped; out-of-range requests fail above.
    changed.coupling.magnets.axial_length_mm =
        Some(length.clamp(geometry.axial_range.min, geometry.axial_range.max));
    Ok(changed)
}
