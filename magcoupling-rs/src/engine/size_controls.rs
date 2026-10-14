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
//! An explicit minimum outside diameter can extend below the legacy 9 mm radius slider
//! floor. It still preserves a nonnegative hub wall around the bore, the cap/thread input
//! limits and the fixed boss. Changing that minimum only changes the available range.
//!
//! Read-outs for the Size tab come from the same geometry: what sets the outside diameter
//! ([`diameter_breakdown`]: the largest of the cup, the cap and the boss) and how the overall
//! length divides ([`length_breakdown`]: the outer ring the cup cavity holds, and the housing
//! around it).
//!
//! [`smallest_fitting_diameter`] finds the smallest outside diameter at which a magnet count's
//! blocks fit, and [`ring_radius_range`] the ring radii the diameter range spans (the ring-size
//! search of inverse sizing). At a fixed magnet count the blocks only fit better as the ring
//! grows: a flat block's polygon flat widens with its apothem, and an arc's share of its pole
//! pitch falls. The fit is bisected on that basis; tests/size_controls.rs sweeps it for both
//! shapes. [`with_ring_radius`] applies a ring radius found that way through the diameter
//! slider's own mapping, so the cap follows the cup instead of keeping its old size.

use std::f64::consts::PI;

use super::api::{DesignInputs, compute_all};
use super::deviations::Deviations;
use super::housing::axial_housing;
use super::meta::{SliderRange, input_rows};
use super::model::blocks_fit;

/// Current assembly dimensions and the ranges attainable with the other inputs fixed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeControls {
    pub diameter_mm: f64,
    pub overall_length_mm: f64,
    pub diameter_range: SliderRange,
    /// Absolute supported range for a user's minimum outside diameter setting.
    pub minimum_diameter_range: SliderRange,
    pub length_range: SliderRange,
}

/// What sets the outside diameter [mm]: the cup (from the magnet layout), the threaded cap
/// and the shaft boss. The outside diameter is the largest of the three.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterBreakdown {
    pub cap_mm: f64,
    pub cup_mm: f64,
    pub boss_mm: f64,
}

impl DiameterBreakdown {
    /// The outside diameter: the largest of the three (one expression for every caller).
    fn outside_mm(&self) -> f64 {
        self.cup_mm.max(self.cap_mm).max(self.boss_mm)
    }
}

/// How the overall axial stack divides [mm]: the outer ring's magnets, which the cup cavity
/// holds, and the housing around them (the cap, the cavity's margin, the web and the boss).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LengthBreakdown {
    pub magnets_mm: f64,
    pub housing_mm: f64,
}

/// Where a magnet count's blocks start to fit over the Size tab's diameter range [mm].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiameterFit {
    /// They fit from this diameter up; [`FIT_TOLERANCE_MM`] below it they do not.
    At(f64),
    /// They fit at the range minimum (this value), so at every diameter in the range.
    AtMinimum(f64),
    /// They fit at no diameter in the range; the value is its maximum.
    Nowhere(f64),
}

/// The bisection of [`smallest_fitting_diameter`] stops when its bracket is this narrow [mm].
pub const FIT_TOLERANCE_MM: f64 = 1e-6;

/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
#[derive(Clone, Copy)]
struct Geometry {
    controls: SizeControls,
    diameter: DiameterBreakdown,
    /// The outer ring's axial length: the ring the cup cavity holds ([`axial_housing`]).
    magnets_length_mm: f64,
    cup_od_mm: f64,
    diameter_slope: f64,
    cap_offset_mm: f64,
    radius_range: SliderRange,
    /// The ring radii the cup reaches with the cap and thread following it: from the radius
    /// input's floor (with a Minimum OD, the shaft's) to its ceiling, narrowed by the cap and
    /// thread input limits. Where the boss sets the outside diameter, they bound the radius.
    ring_floor_mm: f64,
    ring_ceiling_mm: f64,
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

fn geometry(inputs: &DesignInputs, minimum_diameter: Option<f64>) -> Option<Geometry> {
    if minimum_diameter.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return None;
    }
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
    let mut radius_range = input_range("coupling.inner_back_apothem_mm")?;
    let cap_range = input_range("metal.cap_od_mm")?;
    let thread_range = input_range("metal.cap_thread_dia_mm")?;
    let axial_range = input_range("coupling.magnets.axial_length_mm")?;
    let results = compute_all(inputs);
    let cup = results.model.cup_od_mm;
    let md = &inputs.metal;
    let diameter = DiameterBreakdown {
        cap_mm: md.cap_od_mm,
        cup_mm: cup,
        boss_mm: md.boss_od_mm,
    };
    let outside = diameter.outside_mm();
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
    let cup_at_radius = |radius_mm: f64| cup + slope * (radius_mm - radius);
    let radius_at_cup = |cup_mm: f64| radius + (cup_mm - cup) / slope;
    // The cup diameters at which the cap and thread, following the cup, meet their input limits.
    let cup_floor = (cap_range.min - cap_offset).max(thread_range.min - thread_offset);
    let cup_ceiling = (cap_range.max - cap_offset).min(thread_range.max - thread_offset);
    let cup_min_at_radius = |radius_mm: f64| cup_at_radius(radius_mm).max(cup_floor);
    let cup_max = cup_at_radius(radius_range.max).min(cup_ceiling);
    // The explicit floor may relax a UI bound, but cannot put the hub inside its bore.
    let mut shaft_radius = inputs.coupling.bore_mm / 2.0 + md.bond_inner_mm;
    if !shaft_radius.is_finite() || shaft_radius <= 0.0 {
        return None;
    }
    // Use the engine's subtraction order: rounding a sum down must not create a tiny
    // negative hub wall at the endpoint. Zero wall is allowed and remains a design warning.
    if shaft_radius - md.bond_inner_mm - inputs.coupling.bore_mm / 2.0 < 0.0 {
        shaft_radius = shaft_radius.next_up();
    }
    let absolute_cup_min = cup_min_at_radius(shaft_radius);
    if !absolute_cup_min.is_finite() || !cup_max.is_finite() || absolute_cup_min > cup_max {
        return None;
    }
    // The ring radii the cup reaches, the same floors and ceilings in radius space. The boss
    // is not among them: it sets the outside diameter over a plateau of radii.
    let ring_floor = if minimum_diameter.is_some() {
        shaft_radius
    } else {
        radius_range.min
    }
    .max(radius_at_cup(cup_floor));
    let ring_ceiling = radius_range.max.min(radius_at_cup(cup_ceiling));
    let minimum_diameter_range = range(
        (absolute_cup_min + cap_overhang).max(md.boss_od_mm),
        (cup_max + cap_overhang).max(md.boss_od_mm),
    )?;
    let diameter_range = if let Some(minimum) = minimum_diameter {
        // A saved view setting may outlive a change of magnets or gaps. Bound the setting,
        // rather than rewriting the design to fit its former envelope.
        let minimum = minimum.clamp(minimum_diameter_range.min, minimum_diameter_range.max);
        radius_range.min = (radius + (minimum - cap_overhang - cup) / slope)
            .max(shaft_radius)
            .min(radius_range.max);
        range(minimum, minimum_diameter_range.max)?
    } else {
        let cup_min = cup_min_at_radius(radius_range.min);
        if !cup_min.is_finite() || cup_min > cup_max {
            return None;
        }
        range(
            (cup_min + cap_overhang).max(md.boss_od_mm),
            minimum_diameter_range.max,
        )?
    };
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
            minimum_diameter_range,
            length_range,
        },
        diameter,
        magnets_length_mm: results.model.outer_length_mm,
        cup_od_mm: cup,
        diameter_slope: slope,
        cap_offset_mm: cap_offset,
        radius_range,
        ring_floor_mm: ring_floor,
        ring_ceiling_mm: ring_ceiling,
        axial_range,
    })
}

/// Current outside diameter (cup, cap or boss) and overall axial stack. `None` means the
/// geometry or the intersection of its supported input ranges cannot define these controls.
pub fn dimensions(inputs: &DesignInputs) -> Option<SizeControls> {
    geometry(inputs, None).map(|geometry| geometry.controls)
}

/// Current dimensions with a user-selected minimum outside diameter. The positive finite
/// setting is bounded to the absolute range; the inputs and their current size stay intact.
pub fn dimensions_with_minimum_diameter(
    inputs: &DesignInputs,
    minimum_od_mm: f64,
) -> Option<SizeControls> {
    geometry(inputs, Some(minimum_od_mm)).map(|geometry| geometry.controls)
}

/// What sets the current outside diameter: its largest part is [`dimensions`]'s
/// `diameter_mm`. `None` exactly when [`dimensions`] is `None`.
pub fn diameter_breakdown(inputs: &DesignInputs) -> Option<DiameterBreakdown> {
    geometry(inputs, None).map(|geometry| geometry.diameter)
}

/// How the current overall length divides: the parts sum to [`dimensions`]'s
/// `overall_length_mm` (to rounding). `None` when [`dimensions`] is `None`, or when a blank
/// axial length override leaves the housing shorter than its outer ring, which no split into
/// magnets and housing describes.
pub fn length_breakdown(inputs: &DesignInputs) -> Option<LengthBreakdown> {
    let geometry = geometry(inputs, None)?;
    let magnets_mm = geometry.magnets_length_mm;
    let housing_mm = geometry.controls.overall_length_mm - magnets_mm;
    (magnets_mm.is_finite() && magnets_mm > 0.0 && housing_mm >= 0.0).then_some(LengthBreakdown {
        magnets_mm,
        housing_mm,
    })
}

/// The smallest outside diameter at which `npole` blocks per ring fit ([`blocks_fit`]), every
/// other input kept. It searches the Size tab's diameter range of the design with `npole`
/// magnets: [`dimensions`] when `minimum_diameter_mm` is `None`, else
/// [`dimensions_with_minimum_diameter`]. Each diameter maps to the inputs exactly as the
/// diameter slider maps it. The fit only improves as the ring grows (module docs), so a
/// bisection to [`FIT_TOLERANCE_MM`] finds its edge. `None` when that design has no diameter
/// range: invalid inputs, an odd or out-of-range count, or an invalid minimum.
pub fn smallest_fitting_diameter(
    inputs: &DesignInputs,
    npole: i64,
    minimum_diameter_mm: Option<f64>,
) -> Option<DiameterFit> {
    let mut design = inputs.clone();
    design.coupling.npole = npole;
    let geometry = geometry(&design, minimum_diameter_mm)?;
    let range = geometry.controls.diameter_range;
    // Every diameter tried lies inside the range, so the resize is never refused.
    let fits = |diameter_mm: f64| {
        resize_diameter_using(&design, diameter_mm, geometry)
            .is_ok_and(|resized| blocks_fit(&resized.coupling, &compute_all(&resized).model))
    };
    if fits(range.min) {
        return Some(DiameterFit::AtMinimum(range.min));
    }
    if !fits(range.max) {
        return Some(DiameterFit::Nowhere(range.max));
    }
    // `low` never fits and `high` always does.
    let (mut low, mut high) = (range.min, range.max);
    while high - low > FIT_TOLERANCE_MM {
        let middle = low + (high - low) / 2.0;
        if middle <= low || middle >= high {
            break; // adjacent doubles
        }
        if fits(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(DiameterFit::At(high))
}

/// The ring radii (`coupling.inner_back_apothem_mm`) at the ends of the Size tab's diameter
/// range, each mapped as the diameter slider maps it: [`dimensions`] when
/// `minimum_diameter_mm` is `None`, else [`dimensions_with_minimum_diameter`], so a lowered
/// minimum reaches below the radius slider's floor. An end at the boss is different: the boss
/// sets that outside diameter over a plateau of ring radii, so the end is the plateau's edge,
/// where the cup meets its own floor (the radius input's, or with a Minimum OD the shaft's, and
/// the cap and thread input floors) or ceiling. The range therefore does not depend on where on
/// the plateau the ring sits. The step and scale are the radius input's. `None` when the design
/// has no diameter range.
pub fn ring_radius_range(
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
) -> Option<SliderRange> {
    ring_radius_span(inputs, geometry(inputs, minimum_diameter_mm)?)
}

/// [`ring_radius_range`] for a geometry already computed.
fn ring_radius_span(inputs: &DesignInputs, geometry: Geometry) -> Option<SliderRange> {
    let diameters = geometry.controls.diameter_range;
    // A diameter above the boss names one ring radius, the slider's. At the boss the slider's
    // no-op would name the current radius, one of many on the plateau.
    let radius_at = |diameter_mm: f64, plateau_edge_mm: f64| {
        if diameter_mm > geometry.diameter.boss_mm {
            resize_diameter_using(inputs, diameter_mm, geometry)
                .ok()
                .map(|resized| resized.coupling.inner_back_apothem_mm)
        } else {
            Some(plateau_edge_mm)
        }
    };
    let min = radius_at(diameters.min, geometry.ring_floor_mm)?;
    let max = radius_at(diameters.max, geometry.ring_ceiling_mm)?;
    (min.is_finite() && max.is_finite() && min <= max).then_some(SliderRange {
        min,
        max,
        ..geometry.radius_range
    })
}

/// The design with its ring radius (`coupling.inner_back_apothem_mm`) set to `apothem_mm`,
/// mapped as the diameter slider maps it: the cap OD and cap thread move by the cup's change
/// and keep their signed offsets from the cup. `None` where that mapping refuses: a radius
/// that is not a number or lies outside the widest [`ring_radius_range`], the lowest Minimum
/// OD's (every other range, with or without a Minimum OD, lies inside it), or a design without
/// a diameter range. The current radius, when inside that range, is an exact no-op.
pub fn with_ring_radius(inputs: &DesignInputs, apothem_mm: f64) -> Option<DesignInputs> {
    // Any positive minimum is bounded up to the absolute one, which gives the widest range.
    let geometry = geometry(inputs, Some(f64::MIN_POSITIVE))?;
    let reach = ring_radius_span(inputs, geometry)?;
    // A range never contains NaN.
    if !(reach.min..=reach.max).contains(&apothem_mm) {
        return None;
    }
    let change = geometry.diameter_slope * (apothem_mm - inputs.coupling.inner_back_apothem_mm);
    Some(with_cup_change(inputs, change, apothem_mm))
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
    let geometry =
        geometry(inputs, None).ok_or("The current geometry cannot define a size control")?;
    resize_diameter_using(inputs, requested, geometry)
}

/// Select an outside diameter with an explicit minimum. This can relax the inherited
/// radius slider's lower bound, while preserving the absolute housing and shaft limits.
pub fn resize_diameter_with_minimum(
    inputs: &DesignInputs,
    requested: f64,
    minimum_od_mm: f64,
) -> Result<DesignInputs, String> {
    if !minimum_od_mm.is_finite() || minimum_od_mm <= 0.0 {
        return Err("Minimum outside diameter must be a positive finite number".to_owned());
    }
    let geometry = geometry(inputs, Some(minimum_od_mm))
        .ok_or("The current geometry cannot define a size control")?;
    resize_diameter_using(inputs, requested, geometry)
}

fn resize_diameter_using(
    inputs: &DesignInputs,
    requested: f64,
    geometry: Geometry,
) -> Result<DesignInputs, String> {
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
    // The request was checked before this clamp: it only absorbs endpoint rounding in the
    // algebraic inverse, e.g. a radius a few ulps below its metadata minimum.
    let radius = (inputs.coupling.inner_back_apothem_mm + change / geometry.diameter_slope)
        .clamp(geometry.radius_range.min, geometry.radius_range.max);
    Ok(with_cup_change(inputs, change, radius))
}

/// The diameter slider's mapping of a cup change: the inner back apothem is set to
/// `radius_mm`, and the cap OD and cap thread move by the cup's change `change_mm`, so they
/// keep their signed offsets from the cup. The one place that mapping is written.
fn with_cup_change(inputs: &DesignInputs, change_mm: f64, radius_mm: f64) -> DesignInputs {
    let mut changed = inputs.clone();
    changed.coupling.inner_back_apothem_mm = radius_mm;
    changed.metal.cap_od_mm += change_mm;
    changed.metal.cap_thread_dia_mm += change_mm;
    changed
}

/// Select an overall axial stack by setting both rings' existing axial length override.
/// The housing module retains each ring's baseline margins and its physical length floor.
pub fn resize_length(inputs: &DesignInputs, requested: f64) -> Result<DesignInputs, String> {
    let geometry =
        geometry(inputs, None).ok_or("The current geometry cannot define a size control")?;
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
