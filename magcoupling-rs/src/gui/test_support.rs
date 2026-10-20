//! Helpers for the headless egui tests of the panel: one frame with injected
//! input, input events, and inspection of what the frame painted. The same
//! pattern as `linkage-sim-rs/src/gui/test_support.rs` (a separate crate, so
//! the few helpers used here are repeated, not shared).

/// A design with f_end below 0 (the engine's short-magnet test, audit M9): 2 mm manual
/// blocks with c_end 0.5.
pub(crate) fn short_magnets() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.c_end = 0.5;
    inputs.coupling.magnets.part_inner.clear();
    inputs.coupling.magnets.part_outer.clear();
    inputs.coupling.magnets.manual_inner_length_mm = 2.0;
    inputs.coupling.magnets.manual_outer_length_mm = 2.0;
    inputs
}

/// The defaults with `npole` poles per ring, the ring size kept.
pub(crate) fn with_poles(npole: i64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.npole = npole;
    inputs
}

/// A design that passes all 10 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
/// to spare: the defaults with a 1.8 mm face gap (running clearance about 0.1 mm over its
/// target), a 2.0 mm cup wall (the flux needs about 1.85 mm at that gap), a 1.6 N·m
/// requirement (hot-low torque about 2.0 N·m, a quarter over it: a hotter-rated grade, weaker
/// than N42SH, still meets it) and a 1.0 N·m gearbox input rating (cold-high at the gearbox
/// input about 0.71 N·m, 29 % under it; the rating changes no other result than the
/// workbook's gearbox reference, `model.gearbox_reference_Nm`, its only other reader). Its
/// shaft joints pass at the example's bands (BL-064: the inner hub holds about 9.48 N·m and
/// the cup 16.90 N·m against 6.72 N·m; the inner hub would need about 93.3 °C to fit, against
/// 100 °C allowed). The guidance tests break one check of it at a time, so a fix that turns
/// another check red shows.
pub(crate) fn passing_design() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.0;
    inputs.metal.required_min_Nm = 1.6;
    inputs.coupling.gearbox_input_rating_Nm = 1.0;
    inputs
}

/// A hot day: 80 °C ambient and a 60 °C driving rise start the magnets at 140 °C, above
/// both the magnet and the adhesive limit, with the hottest-rated library part in use.
pub(crate) fn hot_day(inputs: &mut crate::DesignInputs) {
    inputs.temperature.duty.hot_ambient_C = 80.0;
    inputs.temperature.duty.driving_rise_C = 60.0;
}

/// Library N42 blocks (rated 80 °C) in both rings: their magnet limit is below the hot-day
/// start, and on the example design the inner ring's single-ring onset less the cure margin is
/// the inner hub's allowed fitting temperature (BL-064).
pub(crate) fn n42_parts(inputs: &mut crate::DesignInputs) {
    inputs.coupling.magnets.part_inner = "B842".to_owned();
    inputs.coupling.magnets.part_outer = "B842".to_owned();
}

/// Mixed rings on manual dimensions (both parts cleared): N42M in the inner ring, N35 in the
/// outer. On [`passing_design`] the outer ring governs the magnets' limit, so the governing
/// single-ring onset is not the inner ring's own, and the inner hub's allowed fitting
/// temperature is still the inner ring's own onset less the cure margin (BL-064, the owner's
/// ring ruling).
pub(crate) fn mixed_rings(inputs: &mut crate::DesignInputs) {
    inputs.coupling.magnets.part_inner.clear();
    inputs.coupling.magnets.part_outer.clear();
    inputs.coupling.magnets.grade_inner = "N42M".to_owned();
    inputs.coupling.magnets.grade_outer = "N35".to_owned();
}

/// No screw fits a 12 mm clamp boss.
pub(crate) fn small_boss(inputs: &mut crate::DesignInputs) {
    inputs.clamps.boss_od_mm = 12.0;
}

/// [`passing_design`] with a [`small_boss`]: it fails the clamp alone, so the clamp section's
/// pause decides whether the design passes.
pub(crate) fn only_the_clamp_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    small_boss(&mut inputs);
    inputs
}

/// [`passing_design`] with the gearbox input rating back at the example design's: it fails the
/// gearbox check alone (cold-high at the gearbox input about 0.71 N·m against 0.6 N·m).
pub(crate) fn only_the_gearbox_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    inputs.coupling.gearbox_input_rating_Nm = crate::DesignInputs::default()
        .coupling
        .gearbox_input_rating_Nm;
    inputs
}

/// [`passing_design`] with the inner hub's minimum interference at 8 µm: smoothing and the
/// hot end take all of it (its effective interference at the loosest is floored at 0), so the
/// inner hub slips, and nothing else fails (BL-064).
pub(crate) fn only_the_joint_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    inputs.shaft_joint.hub_interference_min_um = 8.0;
    inputs
}

/// The inner hub's minimum interference at 20 µm, over its default 18 µm maximum (BL-064): its
/// band is reversed, so the inner hub is not judged.
pub(crate) fn reversed_hub_band(inputs: &mut crate::DesignInputs) {
    inputs.shaft_joint.hub_interference_min_um = 20.0;
}

/// The inner hub's maximum interference at 26 µm (BL-064): at its tightest its bore stress is
/// above its limit, and fitting it needs more than its allowed temperature.
pub(crate) fn hub_overstressed(inputs: &mut crate::DesignInputs) {
    inputs.shaft_joint.hub_interference_max_um = 26.0;
}

/// The defaults at the clamp's limit: the clamp's safety factor raised until its capacity only
/// just covers the requirement.
pub(crate) fn clamp_at_its_limit() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    let base = crate::compute_all(&inputs);
    let crate::engine::meta::NumOrText::Num(capacity) = base.clamps.capacity_Nm else {
        panic!("a screw fits the example design's clamp")
    };
    inputs.clamps.safety_factor = capacity / base.clamps.max_torque_Nm * (1.0 - 1e-9);
    inputs
}

/// Whether `path` is one of the shaft joints' inputs or results (BL-064): their group's paths.
pub(crate) fn is_shaft_joint(path: &str) -> bool {
    path.starts_with("shaft_joint.")
}

/// The shaft joints' input rows of `inputs` (BL-064), in schema order.
pub(crate) fn shaft_joint_input_rows(
    inputs: &crate::DesignInputs,
) -> Vec<crate::engine::meta::InputRow> {
    crate::engine::meta::input_rows(inputs)
        .into_iter()
        .filter(|row| is_shaft_joint(&row.path))
        .collect()
}

/// The shaft joints' result rows of `results` (BL-064), in schema order.
pub(crate) fn shaft_joint_result_rows(
    results: &crate::DesignResults,
) -> Vec<crate::engine::meta::ResultRow> {
    crate::engine::meta::result_rows(results)
        .into_iter()
        .filter(|row| is_shaft_joint(&row.path))
        .collect()
}

/// The inputs side's workflow group `id` ([`crate::gui::inputs::WORKFLOW`]).
pub(crate) fn workflow_group(id: &str) -> &'static crate::gui::inputs::WorkflowGroup {
    crate::gui::inputs::WORKFLOW
        .iter()
        .find(|group| group.id == id)
        .unwrap_or_else(|| panic!("no workflow group {id}"))
}

/// The slider range of the input at `path`.
pub(crate) fn slider_range(path: &str) -> crate::engine::meta::SliderRange {
    crate::gui::inputs::InputCatalogue::get()
        .entry(path)
        .and_then(|entry| entry.meta.range)
        .unwrap_or_else(|| panic!("{path} has a slider"))
}

/// The defaults with every shaft joint input (BL-064) one slider step off its default: up, or
/// down where up would leave the slider. Each input then reads back as itself, not its default.
pub(crate) fn shaft_joints_off_default() -> crate::DesignInputs {
    use crate::engine::meta::{InputSet, Value};
    let mut inputs = crate::DesignInputs::default();
    for row in shaft_joint_input_rows(&crate::DesignInputs::default()) {
        let (Value::Num(x), Some(range)) = (row.value, row.meta.range) else {
            panic!("{} is a number with a slider", row.path)
        };
        let step = if x + range.step <= range.max {
            range.step
        } else {
            -range.step
        };
        inputs
            .set(&row.path, Value::Num(x + step))
            .expect("a value on the slider");
    }
    inputs
}

// The torque card's shared test designs (slice 3 spec section 2.1). Each doc states the
// design's fact there, and `tests::the_shared_card_designs_read_as_section_2_1_says` pins it, so
// a changed default that moves one fails there first, not inside a card test. "Counted checks"
// are the 9 guides that count while the shaft-clamp section is paused, as shipped.

/// The mixed-length pair (slice 3 spec 2.1, "mixed"): the library's B442 (6.35 mm long) inside
/// and B842SH (12.7 mm) outside, the axial override blank. Hot-low 1.0099 N·m over an active
/// length of 6.35 mm; it fails 4 of the 9 counted checks (Hot-low torque, Running clearance, Cup
/// wall, Shaft joints) and asks for a look at Temperature limits. Apart from [`mixed_rings`],
/// which mixes grades.
pub(crate) fn mixed_lengths() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B442".to_owned();
    inputs.coupling.magnets.part_outer = "B842SH".to_owned();
    inputs
}

/// The mixed-length pair the other way round (slice 3 spec 2.1, "reverse"): B842SH inside, B442
/// outside, the axial override blank. Hot-low 1.0099 N·m, as [`mixed_lengths`]; it fails 3 of
/// the 9 counted checks (Hot-low torque, Running clearance, Cup wall) and asks for a look at
/// Temperature limits.
pub(crate) fn reverse_lengths() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B842SH".to_owned();
    inputs.coupling.magnets.part_outer = "B442".to_owned();
    inputs
}

/// No back iron (slice 3 spec 2.1, "(c)"): `coupling.backiron` 0, the example's measured
/// calibration factor (1.0495, `model.f_cal`). Hot-low 1.4428 N·m; it fails 3 of the 9 counted
/// checks (Hot-low torque, Running clearance, Shaft joints). Its Magnets per ring `-2` design
/// takes the calibration coefficient itself, 0.95.
pub(crate) fn no_back_iron() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.backiron = 0;
    inputs
}

/// [`no_back_iron`] with a design file's calibration coefficient of -0.95
/// (`calibration.f_cal_original`; `validate()` passes it). The measured factor cancels it, so the
/// design reads as [`no_back_iron`]: factor 1.0495, hot-low 1.4428 and pull-out 1.6974 N·m,
/// failing the same 3 of 9. Its Magnets per ring `-2` design (8 poles) takes the coefficient
/// itself: factor -0.95, hot-low -0.8505 and pull-out -1.0006 N·m with the end-effect factor
/// 0.870, in range (slice 3 spec 2.1, "(c), `f_cal_original` -0.95").
pub(crate) fn no_back_iron_negative_calibration() -> crate::DesignInputs {
    let mut inputs = no_back_iron();
    inputs.calibration.f_cal_original = -0.95;
    inputs
}

/// An end-effect factor near its edge (slice 3 spec 2.1, "end effect at a step"): `c_end` 0.5
/// and the axial override at 5.0 mm. Hot-low 0.1196 N·m, with the end effect in range; one
/// millimetre shorter (the override at 4.0 mm) it is not.
pub(crate) fn end_effect_at_a_step() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.c_end = 0.5;
    inputs.coupling.magnets.axial_length_mm = Some(5.0);
    inputs
}

/// An odd magnet count from a design file (slice 3 spec 2.1, "odd count"): `coupling.npole` 9,
/// which the slider (even counts) does not take. Hot-low 2.0295 N·m; the Size tab has no size
/// controls for it (`engine::size_controls::dimensions` is `None`).
pub(crate) fn odd_count() -> crate::DesignInputs {
    with_poles(9)
}

/// A keyway through the hub (slice 3 spec 2.1, "keyway through"): a 16 mm bore with a 2.5 mm
/// keyway. The hub wall past the keyway is below zero (`model.hub_wall_past_key_mm`), so the
/// shaft joints have no wall to judge and ask for a look (Caution).
pub(crate) fn keyway_through() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.bore_mm = 16.0;
    inputs.coupling.keyway_depth_mm = 2.5;
    inputs
}

/// A cup cavity input shorter than the outer ring (slice 3 spec 2.1, "short cavity"):
/// `metal.cup_depth_mm` 11.0 against B842SH's 12.7 mm, the axial override blank. The design
/// reads as the defaults (hot-low 2.2852 N·m, failing the same 4 of 9 checks); its Overall
/// length is 27.30 mm on a range of 18.30 to 67.10 mm.
pub(crate) fn short_cavity() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.cup_depth_mm = 11.0;
    inputs
}

/// A hot day just under the temperature verdict's edge (slice 3 spec 2.1, "temperature edge":
/// the edge is at a hot-day ambient of 82.0913 °C): `temperature.duty.hot_ambient_C` 82.08.
/// Hot-low 2.2852 N·m, failing the defaults' 4 of 9 checks, with Temperature limits passing.
pub(crate) fn temperature_edge() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.temperature.duty.hot_ambient_C = 82.08;
    inputs
}

/// Forty narrow magnets that fit (slice 3 spec 2.1, "npole 40"): `coupling.npole` 40, both
/// parts blank and both rings' manual blocks 1.2 mm wide. The blocks fit; hot-low 0.4857 N·m.
pub(crate) fn forty_narrow_magnets() -> crate::DesignInputs {
    let mut inputs = with_poles(40);
    let magnets = &mut inputs.coupling.magnets;
    magnets.part_inner.clear();
    magnets.part_outer.clear();
    magnets.manual_inner_width_mm = 1.2;
    magnets.manual_outer_width_mm = 1.2;
    inputs
}

/// An unmagnetized inner ring from a design file (slice 3 spec 2.1, "zero torque"): both parts
/// blank and `coupling.magnets.manual_inner_br_T` 0 (the slider stops at 0.2 T; `set()` and
/// `validate()` check only that it is finite). The model predicts zero: pull-out and hot-low
/// 0 N·m with the end-effect factor 0.896, in range. It fails 2 of the 9 counted checks (Hot-low
/// torque, Running clearance) and asks for a look at 2 (Shaft joints, Gearbox input torque).
pub(crate) fn zero_torque() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    let magnets = &mut inputs.coupling.magnets;
    magnets.part_inner.clear();
    magnets.part_outer.clear();
    magnets.manual_inner_br_T = 0.0;
    inputs
}

/// The defaults with the variation allowance at `variation` (`metal.variation`; a design file
/// can hold any finite value). At 1.0 hot-low is 0 N·m while the model predicts (pull-out
/// 2.6885 N·m, end-effect factor 0.896); at 1.5 hot-low is -1.3442 N·m (slice 3 spec 2.1,
/// "variation 1.0").
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}

/// The shaft boss setting the Outside diameter (slice 3 spec R-13, the boss plateau): 8 magnets
/// per ring on a ring of 8.0 mm, the cap at 30 mm with its thread at 29 mm, and the boss at
/// 40 mm. The magnets fit and the boss sets the Outside diameter, 40.00 mm, the slider's bottom
/// (its range 40.00 to 85.41 mm): one +1 mm step lifts the cup from its own size to the boss
/// plus the step.
pub(crate) fn boss_plateau() -> crate::DesignInputs {
    let mut inputs = with_poles(8);
    inputs.coupling.inner_back_apothem_mm = 8.0;
    inputs.metal.cap_od_mm = 30.0;
    inputs.metal.cap_thread_dia_mm = 29.0;
    inputs.metal.boss_od_mm = 40.0;
    inputs
}

/// The gui-smoke share link's design inputs (slice 3 spec 2.1, "(b)"), before Auto on the
/// magnet length solves them: the defaults with a 1.5 mm face gap.
pub(crate) fn smoke_inputs() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.5;
    inputs
}

// The torque card's one table of designs (slice 3 spec section 8, code review 4 SF-4): here,
// beside the designs it reads, so any test module reads the one table.

/// One design of the torque card's table ([`card_designs`]) as the panel shows it: its key,
/// the design shown (an Auto design solved as the panel solves it), the Size tab's Minimum OD
/// preference and the Auto state.
pub(crate) struct CardDesign {
    pub(crate) key: &'static str,
    pub(crate) inputs: crate::DesignInputs,
    pub(crate) minimum_diameter_mm: Option<f64>,
    pub(crate) auto: Option<crate::engine::sizing::FreeVariable>,
}

impl CardDesign {
    /// The design's card, at its Minimum OD preference ([`card_at`]).
    pub(crate) fn card(&self) -> crate::gui::torque_card::TorqueCard {
        card_at(&self.inputs, self.minimum_diameter_mm)
    }

    /// The side of the design's `control` row, `more` its `+` side ([`side_at`]).
    pub(crate) fn side(
        &self,
        control: crate::gui::torque_card::Control,
        more: bool,
    ) -> crate::gui::torque_card::Side {
        side_at(&self.inputs, self.minimum_diameter_mm, control, more)
    }
}

/// The card of `inputs` as the panel asks for it: with its own results, the Minimum OD
/// preference `minimum` and the shaft-clamp section's shipped pause.
pub(crate) fn card_at(
    inputs: &crate::DesignInputs,
    minimum: Option<f64>,
) -> crate::gui::torque_card::TorqueCard {
    let results = crate::compute_all(inputs);
    crate::gui::torque_card::card(inputs, &results, minimum, crate::engine::clamps::PAUSED)
}

/// The side of `inputs`' card ([`card_at`], at the Minimum OD preference `minimum`) on
/// `control`'s row, `more` its `+` side.
pub(crate) fn side_at(
    inputs: &crate::DesignInputs,
    minimum: Option<f64>,
    control: crate::gui::torque_card::Control,
    more: bool,
) -> crate::gui::torque_card::Side {
    let card = card_at(inputs, minimum);
    let row = card
        .rows()
        .iter()
        .find(|row| row.control == control)
        .unwrap_or_else(|| panic!("a {control:?} row: {card:?}"));
    if more {
        row.more.clone()
    } else {
        row.fewer.clone()
    }
}

/// The design the panel shows from `inputs` with Auto on `variable` at the Minimum OD
/// preference `minimum`: the solver's value, or its best where the requirement is out of
/// reach (`SizingRunner`), set through `gui::sizing::sized`, over the range the panel
/// searches (`gui::sizing::sizing_range`, the panel's own rule).
fn auto_shown(
    inputs: &crate::DesignInputs,
    variable: crate::engine::sizing::FreeVariable,
    minimum: Option<f64>,
) -> crate::DesignInputs {
    use crate::gui::sizing::{SizingMode, SizingRunner, SizingState, sized, sizing_range};
    let range = sizing_range(inputs, variable, minimum);
    let state = SizingState {
        mode: SizingMode::TorqueToMagnets,
        variable,
    };
    let mut runner = SizingRunner::default();
    runner.solve_now(inputs, &state, range);
    let value = runner
        .value(variable)
        .unwrap_or_else(|| panic!("Auto on {variable:?} shows a value"));
    sized(variable, inputs, value)
}

/// The torque card's designs: those of the slice 3 spec's section 2.1 and its Edges, each with
/// its Minimum OD preference and Auto state (an Auto design solved as the panel solves it).
/// The one table the card's tests read (code review 4 SF-4), by key or all of it, so a design
/// added once reaches every test. Each key is the spec's, without its Markdown, with
/// ": <what differs>" where a row holds several designs.
pub(crate) fn card_designs() -> Vec<CardDesign> {
    use crate::DesignInputs;
    use crate::engine::size_controls::{
        dimensions, dimensions_with_minimum_diameter, with_ring_radius,
    };
    use crate::engine::sizing::FreeVariable::{AxialLength, MagnetsPerRing, RingRadius};
    use crate::gui::guidance::FACE_GAP;
    use crate::gui::size_controls::{SliderMapping, slider_minimum};
    let a = DesignInputs::default;
    let e = || with_poles(8);
    let e2 = || {
        let mut inputs = with_poles(8);
        inputs.metal.required_min_Nm = 1.5;
        inputs
    };
    let size = |inputs: &DesignInputs| dimensions(inputs).expect("a design with size controls");
    let lowest = |inputs: &DesignInputs| size(inputs).minimum_diameter_range.min;
    // A design one slider mapping away (the Size tab's own, `SliderMapping`): the Outside
    // diameter at the default Minimum OD, the Overall length, the gap.
    let at_diameter = |inputs: &DesignInputs, mm: f64| {
        SliderMapping::Diameter {
            minimum: slider_minimum(&size(inputs), None),
        }
        .design(inputs, mm)
        .expect("a diameter on the slider")
    };
    let at_length = |inputs: &DesignInputs, mm: f64| {
        SliderMapping::Length
            .design(inputs, mm)
            .expect("a length on the slider")
    };
    let at_gap = |mm: f64| {
        SliderMapping::Input { path: FACE_GAP }
            .design(&a(), mm)
            .expect("a gap on the slider")
    };
    let diameter_top = |inputs: &DesignInputs| {
        let minimum = slider_minimum(&size(inputs), None);
        dimensions_with_minimum_diameter(inputs, minimum)
            .expect("a diameter range")
            .diameter_range
            .max
    };
    let lengths = size(&a()).length_range;
    let plain = |key, inputs| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: None,
        auto: None,
    };
    let at_minimum = |key, inputs, minimum: f64| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: Some(minimum),
        auto: None,
    };
    let solved = |key, inputs: DesignInputs, variable, minimum: Option<f64>| CardDesign {
        key,
        inputs: auto_shown(&inputs, variable, minimum),
        minimum_diameter_mm: minimum,
        auto: Some(variable),
    };
    let hot = {
        let mut inputs = a();
        hot_day(&mut inputs);
        inputs
    };
    // T-14's one-flip row (slice 3 spec 10.1 step 6): (e2) with Auto on the magnet length,
    // shown with Auto off and Required halfway between its Outside diameter sides' hot-low, so
    // the two sides read alike while only one crosses Required.
    let one_flip = {
        use crate::gui::torque_card::Side;
        let mut shown = auto_shown(&e2(), AxialLength, None);
        let card = card_at(&shown, None);
        let row = &card.rows()[0];
        let hot_low = |side: &Side| match side {
            Side::Change(change) => change.hot_low_Nm,
            Side::Words(words) => panic!("an Outside diameter side reads {words:?}"),
        };
        shown.metal.required_min_Nm = (hot_low(&row.fewer) + hot_low(&row.more)) / 2.0;
        shown
    };
    vec![
        plain("(a)", a()),
        solved("(b)", smoke_inputs(), AxialLength, None),
        plain("(c)", no_back_iron()),
        plain("(e)", e()),
        plain("(e2)", e2()),
        plain("mixed", mixed_lengths()),
        plain("reverse", reverse_lengths()),
        solved("(a)+Auto length", a(), AxialLength, None),
        solved("(a)+Auto count", a(), MagnetsPerRing, None),
        solved("(a)+Auto ring", a(), RingRadius, None),
        solved("(e2)+Auto length", e2(), AxialLength, None),
        solved("(e2)+Auto count", e2(), MagnetsPerRing, None),
        solved("(e2)+Auto ring", e2(), RingRadius, None),
        solved(
            "(e2)+Auto ring, low Min OD",
            e2(),
            RingRadius,
            Some(lowest(&e2())),
        ),
        plain("short magnets", short_magnets()),
        plain("end effect at a step", end_effect_at_a_step()),
        plain("odd count", odd_count()),
        at_minimum("(e), low Min OD", e(), lowest(&e())),
        at_minimum("(a), Min OD 42.8", a(), 42.8),
        at_minimum("(a), Min OD 45", a(), 45.0),
        at_minimum("(a), Min OD 45: 43.3 mm", a(), 43.3),
        plain(
            "shown does not fit",
            with_ring_radius(&a(), 9.5).expect("a ring the diameter slider reaches"),
        ),
        plain("keyway through", keyway_through()),
        plain("gap at 0.3", at_gap(0.3)),
        plain("npole 4", with_poles(4)),
        plain("hot day", hot),
        plain("only the joint fails", only_the_joint_fails()),
        plain("passing", passing_design()),
        plain("short cavity", short_cavity()),
        plain("temperature edge", temperature_edge()),
        plain("npole 40", forty_narrow_magnets()),
        plain("slider tops: gap 5.0", at_gap(5.0)),
        plain(
            "slider tops: Outside diameter 80",
            at_diameter(&a(), diameter_top(&a())),
        ),
        plain(
            "slider tops: Overall length 69.9",
            at_length(&a(), lengths.max),
        ),
        plain(
            "slider tops: Overall length 21.1",
            at_length(&a(), lengths.min),
        ),
        plain("steps to an end: (a) at length 68.9", at_length(&a(), 68.9)),
        plain("steps to an end: (a) at length 22.1", at_length(&a(), 22.1)),
        plain("steps to an end: (e) at length 68.9", at_length(&e(), 68.9)),
        plain("steps to an end: (e) at length 22.1", at_length(&e(), 22.1)),
        plain("steps to an end: (a) at gap 4.9", at_gap(4.9)),
        plain("steps to an end: (a) at gap 0.4", at_gap(0.4)),
        plain(
            "steps to an end: (a) at Outside diameter 79",
            at_diameter(&a(), diameter_top(&a()) - 1.0),
        ),
        plain(
            "steps to an end: (e) at Outside diameter 79",
            at_diameter(&e(), diameter_top(&e()) - 1.0),
        ),
        at_minimum(
            "steps to an end: (e), Min OD 41.8",
            e(),
            size(&e()).diameter_mm - 1.0,
        ),
        plain("variation 1.0", with_variation(1.0)),
        plain("variation 1.0: 1.5", with_variation(1.5)),
        plain("zero torque", zero_torque()),
        plain(
            "(c), f_cal_original -0.95",
            no_back_iron_negative_calibration(),
        ),
        plain("T-14 one flip", one_flip),
        plain("R-13 boss plateau", boss_plateau()),
        plain("count overflow: i64::MAX", with_poles(i64::MAX)),
        plain("count overflow: i64::MIN", with_poles(i64::MIN)),
    ]
}

/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
/// wall around its bore, so its holding, bore stress, needed and elastic results are not a
/// number, and its key no bearing length (the web alone, 2.5 mm, under the 3 mm key), so its
/// key pressure is +inf.
pub(crate) fn cup_without_wall_or_key_bearing() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.bore_mm = 16.0;
    inputs.metal.boss_od_mm = 12.0;
    inputs.metal.boss_length_mm = 0.0;
    inputs
}

/// The screen of the headless frames [points]: a laptop window.
pub(crate) const SCREEN: egui::Vec2 = egui::vec2(1280.0, 1024.0);

/// One headless frame of `draw` inside a central panel on a screen of `size`, with `events`
/// as the frame's input. Returns what egui painted.
pub(crate) fn sized_frame(
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
    draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    sized_frame_at(ctx, size, None, events, draw)
}

/// [`sized_frame`] at the input time `time` [s] (`None`: egui adds its predicted frame time,
/// 1/60 s, to the last frame's).
pub(crate) fn sized_frame_at(
    ctx: &egui::Context,
    size: egui::Vec2,
    time: Option<f64>,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        time,
        ..Default::default()
    };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
}

/// Shows `ctx`'s tooltips at once: no delay, and the pointer need not rest.
pub(crate) fn tooltips_at_once(ctx: &egui::Context) {
    ctx.style_mut(|s| {
        s.interaction.tooltip_delay = 0.0;
        s.interaction.show_tooltips_only_when_still = false;
    });
}

/// Frames of `draw` on a screen of `size`, `ctx`'s tooltips at once ([`tooltips_at_once`]):
/// two to lay it out, one with the pointer moved to the point `target` finds in the second, and
/// two more, so what the pointer hovers shows its tooltip. Returns the second frame (the
/// pointer not there yet) and the last.
pub(crate) fn hover_frames(
    ctx: &egui::Context,
    size: egui::Vec2,
    target: impl FnOnce(&egui::FullOutput) -> egui::Pos2,
    mut draw: impl FnMut(&mut egui::Ui),
) -> (egui::FullOutput, egui::FullOutput) {
    tooltips_at_once(ctx);
    let mut frame = |time: f64, events: Vec<egui::Event>| {
        sized_frame_at(ctx, size, Some(time), events, &mut draw)
    };
    frame(0.0, Vec::new());
    let before = frame(0.1, Vec::new());
    frame(0.2, vec![egui::Event::PointerMoved(target(&before))]);
    frame(0.3, Vec::new());
    (before, frame(0.4, Vec::new()))
}

/// Hovers the first drawn `text` of `draw` on a screen of `size` ([`hover_frames`]), with the
/// spreadsheet details off and then on (spec S2-5: off hides every provenance line), and asserts
/// that the hover drew `hover(show_details)` and not `hover(!show_details)`, and that `path` is
/// a line of the hover text only with the details. `draw` draws the view with the readouts it is
/// handed. The hover text without the details can be the label alone, which the view may draw
/// too: the hover counts as one more drawn copy of its text.
pub(crate) fn assert_hover_names_path_only_with_details(
    size: egui::Vec2,
    path: &str,
    text: &str,
    hover: impl Fn(bool) -> String,
    mut draw: impl FnMut(&mut egui::Ui, &mut crate::gui::readouts::Readouts),
) {
    use crate::gui::readouts::Readouts;
    use crate::gui::typeset::TermColors;
    for show_details in [false, true] {
        let ctx = egui::Context::default();
        let (before, after) = hover_frames(
            &ctx,
            size,
            |output| text_center(output, text),
            |ui| draw(ui, &mut Readouts::new(TermColors::default(), show_details)),
        );
        let (want, other) = (hover(show_details), hover(!show_details));
        assert_eq!(
            drawn_count(&after, &want),
            drawn_count(&before, &want) + 1,
            "details {show_details}: the hover draws {want:?}"
        );
        assert_eq!(
            drawn_count(&after, &other),
            drawn_count(&before, &other),
            "details {show_details}: the hover draws {other:?}"
        );
        assert_eq!(
            want.lines().any(|line| line == path),
            show_details,
            "details {show_details}: {want:?}"
        );
    }
}

/// One headless frame of `draw` in a child ui whose max rect is `region`, on a screen of
/// `size`, as the centre region lays a view out above the Equation panel but without its clip:
/// what the view lays out past the region's foot shows in the rect the child used. Returns what
/// egui painted and that rect.
pub(crate) fn region_frame(
    ctx: &egui::Context,
    size: egui::Vec2,
    region: egui::Rect,
    mut draw: impl FnMut(&mut egui::Ui),
) -> (egui::FullOutput, egui::Rect) {
    let mut used = egui::Rect::NOTHING;
    let output = sized_frame(ctx, size, Vec::new(), |ui| {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(region));
        draw(&mut child);
        used = child.min_rect();
    });
    (output, used)
}

/// A key event with no modifiers: a press, or (`pressed` false) its release.
pub(crate) fn key_event(key: egui::Key, pressed: bool) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

/// A key tapped with no modifiers: pressed and released in one frame, as a user taps it. egui
/// keeps a pressed key down until its release and reads another press of it as a repeat.
pub(crate) fn key_tap(key: egui::Key) -> Vec<egui::Event> {
    vec![key_event(key, true), key_event(key, false)]
}

/// Ctrl+A (Cmd+A on a Mac) tapped: select all in the focused text field.
pub(crate) fn select_all() -> Vec<egui::Event> {
    [true, false]
        .map(|pressed| egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        })
        .to_vec()
}

/// A primary-button press (`pressed`) or release at `pos`.
pub(crate) fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Every shape egui painted in a frame, the shapes nested in a `Shape::Vec`
/// flattened, in paint order: what the helpers below and the tests' own
/// filters walk.
pub(crate) fn flat_shapes(output: &egui::FullOutput) -> Vec<&egui::Shape> {
    fn walk<'a>(shape: &'a egui::Shape, shapes: &mut Vec<&'a egui::Shape>) {
        match shape {
            egui::Shape::Vec(nested) => nested.iter().for_each(|s| walk(s, shapes)),
            other => shapes.push(other),
        }
    }
    let mut shapes = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut shapes);
    }
    shapes
}

/// Every text egui drew in a frame (widgets and painter text), nested shapes
/// included, in paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) => Some(text.galley.text().to_owned()),
            _ => None,
        })
        .collect()
}

/// How many drawn texts of a frame equal `text`.
pub(crate) fn drawn_count(output: &egui::FullOutput, text: &str) -> usize {
    drawn_texts(output).iter().filter(|t| *t == text).count()
}

/// Screen rects of every drawn text equal to `needle`, in paint order.
pub(crate) fn text_rects(output: &egui::FullOutput, needle: &str) -> Vec<egui::Rect> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.galley.rect.translate(text.pos.to_vec2()))
            }
            _ => None,
        })
        .collect()
}

/// Where a click lands on the last character of every drawn text equal to `needle`, in paint
/// order: 2 points in from the right end of its last row, at that row's middle. A text in a
/// wrapped row is laid out from the row's start, so its rect ([`text_rects`]) covers the widgets
/// before it on that row, while egui senses only the text: a click at the rect's centre can land
/// on one of those widgets.
pub(crate) fn last_characters(output: &egui::FullOutput, needle: &str) -> Vec<egui::Pos2> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                let row = text
                    .galley
                    .rows
                    .last()?
                    .rect()
                    .translate(text.pos.to_vec2());
                Some(egui::pos2(row.max.x - 2.0, row.center().y))
            }
            _ => None,
        })
        .collect()
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    text_rects(output, needle).into_iter().next()
}

/// The centre of the first drawn text equal to `needle`: where a test points to hover it.
/// Panics when no such text is drawn.
pub(crate) fn text_center(output: &egui::FullOutput, needle: &str) -> egui::Pos2 {
    text_rect(output, needle)
        .unwrap_or_else(|| panic!("no {needle:?} drawn"))
        .center()
}

/// Asserts that every character of `text` but whitespace has a glyph in egui's default fonts
/// at proportional 14 (the UI's criterion): egui draws an empty box for one it lacks. `what`
/// names the text's source in the failure. The fonts load on `ctx`'s first frame.
pub(crate) fn assert_glyphs(ctx: &egui::Context, text: &str, what: &str) {
    let font = egui::FontId::proportional(14.0);
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        assert!(
            ctx.fonts(|f| f.has_glyph(&font, c)),
            "{what}: no glyph for {c:?} (U+{:04X}) in {text:?}",
            c as u32
        );
    }
}

/// The colour of the first drawn text equal to `needle`: its override, else its first section's
/// colour (a `RichText` colour or a painter text's), else the fallback colour.
pub(crate) fn text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
    flat_shapes(output)
        .into_iter()
        .find_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.override_text_color.unwrap_or_else(|| {
                    match text.galley.job.sections.first().map(|s| s.format.color) {
                        Some(color) if color != egui::Color32::PLACEHOLDER => color,
                        _ => text.fallback_color,
                    }
                }))
            }
            _ => None,
        })
}

/// Every sheet of an .xlsx file as calamine reads it: its name and its rows from A1 (calamine's
/// range starts at the first cell used; every sheet of the export uses A1), each row as wide as
/// the sheet's widest, `Data::Empty` where nothing was written.
pub(crate) fn read_xlsx(bytes: &[u8]) -> Vec<(String, Vec<Vec<calamine::Data>>)> {
    use calamine::Reader;
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook_from_rs(std::io::Cursor::new(bytes.to_vec()))
            .expect("an .xlsx file calamine reads");
    workbook
        .sheet_names()
        .into_iter()
        .map(|name| {
            let range = workbook.worksheet_range(&name).expect("the sheet reads");
            assert_eq!(
                range.start().unwrap_or((0, 0)),
                (0, 0),
                "{name} starts at A1"
            );
            let rows = range.rows().map(<[calamine::Data]>::to_vec).collect();
            (name, rows)
        })
        .collect()
}

/// The spreadsheet tests' snapshot of `inputs` and `results`: Magnets -> Torque, the workflow
/// order, a stand-in share link and export time.
pub(crate) fn snapshot<'a>(
    inputs: &'a crate::DesignInputs,
    results: &'a crate::DesignResults,
) -> crate::gui::spreadsheet::Snapshot<'a> {
    crate::gui::spreadsheet::Snapshot {
        inputs,
        results,
        sizing: crate::gui::sizing::SizingState::default(),
        sizing_status: None,
        input_order: crate::gui::inputs::InputOrder::Workflow,
        share_link: "https://example.test/magcoupling/?m=abc".to_owned(),
        exported_unix_s: 1_790_000_000,
    }
}

/// The text of the part `name` of an .xlsx file (a zip archive), e.g. `xl/styles.xml`.
pub(crate) fn xlsx_part(bytes: &[u8], name: &str) -> String {
    use std::io::Read;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).expect("an .xlsx file is a zip");
    let mut part = archive
        .by_name(name)
        .unwrap_or_else(|_| panic!("no part {name}"));
    let mut text = String::new();
    part.read_to_string(&mut text)
        .expect("an XML part is UTF-8");
    text
}

mod tests {
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::model::blocks_fit;
    use crate::engine::size_controls::dimensions;
    use crate::gui::dashboard::Level;
    use crate::gui::format::fixed;
    use crate::gui::guidance::{
        CLEARANCE, CUP_WALL, GEARBOX, HOT_MIN, SHAFT_JOINT, TEMPERATURE, counted_guides, level,
    };
    use crate::gui::size_controls::predicts;
    use crate::{DesignInputs, compute_all};

    /// The counted checks (the shaft-clamp section paused, as shipped) at `want` on `inputs`, in
    /// the guides' order.
    fn checks_at(inputs: &DesignInputs, want: Level) -> Vec<&'static str> {
        let results = compute_all(inputs);
        counted_guides(PAUSED)
            .filter(|guide| level(guide, inputs, &results) == want)
            .map(|guide| guide.check)
            .collect()
    }

    #[test]
    fn the_shared_card_designs_read_as_section_2_1_says() {
        // Slice 3 spec 2.1, each number to the decimals the section quotes it to, so a changed
        // default that moves a design fails here first, not inside a card test.
        let defaults_fail = [HOT_MIN, CLEARANCE, CUP_WALL, GEARBOX];
        let none: [&str; 0] = [];
        let hot_low = |inputs: &DesignInputs| compute_all(inputs).metal.torque_hot_low_Nm;

        // The mixed-length pair and its reverse.
        let mixed = compute_all(&mixed_lengths());
        assert_eq!(fixed(mixed.metal.torque_hot_low_Nm, 4), "1.0099");
        assert_eq!(fixed(mixed.model.active_length_mm, 2), "6.35");
        assert_eq!(
            checks_at(&mixed_lengths(), Level::Bad),
            [HOT_MIN, CLEARANCE, CUP_WALL, SHAFT_JOINT]
        );
        assert_eq!(checks_at(&mixed_lengths(), Level::Caution), [TEMPERATURE]);
        assert_eq!(fixed(hot_low(&reverse_lengths()), 4), "1.0099");
        assert_eq!(
            checks_at(&reverse_lengths(), Level::Bad),
            [HOT_MIN, CLEARANCE, CUP_WALL]
        );
        assert_eq!(checks_at(&reverse_lengths(), Level::Caution), [TEMPERATURE]);

        // No back iron, and with its calibration coefficient at -0.95: the measured factor on
        // the design shown, the coefficient itself on its Magnets per ring -2 design.
        let two_fewer = |mut inputs: DesignInputs| {
            inputs.coupling.npole -= 2;
            inputs
        };
        for (name, inputs) in [
            ("no back iron", no_back_iron()),
            (
                "its coefficient at -0.95",
                no_back_iron_negative_calibration(),
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(
                results.model.f_cal.to_bits(),
                results.calibration.f_cal_updated.to_bits(),
                "{name}: the measured factor"
            );
            assert_eq!(fixed(results.model.f_cal, 4), "1.0495", "{name}");
            assert_eq!(
                fixed(results.metal.torque_hot_low_Nm, 4),
                "1.4428",
                "{name}"
            );
            assert_eq!(fixed(results.model.pullout_Nm, 4), "1.6974", "{name}");
            assert_eq!(
                checks_at(&inputs, Level::Bad),
                [HOT_MIN, CLEARANCE, SHAFT_JOINT],
                "{name}"
            );
            let fewer = compute_all(&two_fewer(inputs.clone()));
            assert_eq!(
                fewer.model.f_cal, inputs.calibration.f_cal_original,
                "{name}: its -2 design takes the coefficient"
            );
        }
        let fewer = compute_all(&two_fewer(no_back_iron_negative_calibration()));
        assert_eq!(fewer.model.f_cal, -0.95);
        assert_eq!(fixed(fewer.metal.torque_hot_low_Nm, 4), "-0.8505");
        assert_eq!(fixed(fewer.model.pullout_Nm, 4), "-1.0006");
        assert_eq!(fixed(fewer.model.f_end, 3), "0.870");
        assert!(predicts(&fewer, fewer.model.pullout_Nm));
        assert!(predicts(&fewer, fewer.metal.torque_hot_low_Nm));

        // The end effect at a step: in range at 5.0 mm, not at 4.0 mm.
        let at_a_step = compute_all(&end_effect_at_a_step());
        assert_eq!(fixed(at_a_step.metal.torque_hot_low_Nm, 4), "0.1196");
        assert!(predicts(&at_a_step, at_a_step.metal.torque_hot_low_Nm));
        let mut shorter = end_effect_at_a_step();
        shorter.coupling.magnets.axial_length_mm = Some(4.0);
        let shorter = compute_all(&shorter);
        assert!(!predicts(&shorter, shorter.metal.torque_hot_low_Nm));

        // An odd count: no size controls.
        assert_eq!(fixed(hot_low(&odd_count()), 4), "2.0295");
        assert!(dimensions(&odd_count()).is_none());

        // The keyway through the hub: no wall past it, so the joints are not judged.
        let keyway = compute_all(&keyway_through());
        assert!(keyway.model.hub_wall_past_key_mm < 0.0);
        assert_eq!(checks_at(&keyway_through(), Level::Caution), [SHAFT_JOINT]);

        // The short cavity: the defaults' torque and checks, a shorter Overall length.
        let size = dimensions(&short_cavity()).expect("the short cavity has size controls");
        assert_eq!(fixed(hot_low(&short_cavity()), 4), "2.2852");
        assert_eq!(checks_at(&short_cavity(), Level::Bad), defaults_fail);
        assert_eq!(
            [
                size.overall_length_mm,
                size.length_range.min,
                size.length_range.max,
            ]
            .map(|mm| fixed(mm, 2)),
            ["27.30", "18.30", "67.10"]
        );

        // The temperature edge: the defaults' torque and checks, Temperature limits passing.
        assert_eq!(fixed(hot_low(&temperature_edge()), 4), "2.2852");
        assert_eq!(checks_at(&temperature_edge(), Level::Bad), defaults_fail);
        assert_eq!(checks_at(&temperature_edge(), Level::Caution), none);

        // Forty narrow magnets fit.
        let forty = compute_all(&forty_narrow_magnets());
        assert!(blocks_fit(&forty_narrow_magnets().coupling, &forty.model));
        assert_eq!(fixed(forty.metal.torque_hot_low_Nm, 4), "0.4857");

        // Zero torque: both torques 0 N·m and both shown (the end effect in range).
        let zero = compute_all(&zero_torque());
        assert_eq!(
            (zero.model.pullout_Nm, zero.metal.torque_hot_low_Nm),
            (0.0, 0.0)
        );
        assert!(predicts(&zero, zero.model.pullout_Nm));
        assert!(predicts(&zero, zero.metal.torque_hot_low_Nm));
        assert_eq!(fixed(zero.model.f_end, 3), "0.896");
        assert_eq!(checks_at(&zero_torque(), Level::Bad), [HOT_MIN, CLEARANCE]);
        assert_eq!(
            checks_at(&zero_torque(), Level::Caution),
            [SHAFT_JOINT, GEARBOX]
        );

        // The variation allowance at 1.0 and 1.5: hot-low 0 and below it, the pull-out shown.
        let one = compute_all(&with_variation(1.0));
        assert_eq!(one.metal.torque_hot_low_Nm, 0.0);
        assert_eq!(fixed(one.model.pullout_Nm, 4), "2.6885");
        assert_eq!(fixed(one.model.f_end, 3), "0.896");
        assert!(predicts(&one, one.model.pullout_Nm));
        assert_eq!(fixed(hot_low(&with_variation(1.5)), 4), "-1.3442");

        // The boss plateau (R-13): the magnets fit, and the boss sets the Outside diameter at the
        // bottom of the slider's range.
        let boss = boss_plateau();
        let size = dimensions(&boss).expect("the boss plateau has size controls");
        assert!(blocks_fit(&boss.coupling, &compute_all(&boss).model));
        assert_eq!(size.diameter_mm, boss.metal.boss_od_mm);
        assert_eq!(
            [
                size.diameter_mm,
                size.diameter_range.min,
                size.diameter_range.max
            ]
            .map(|mm| fixed(mm, 2)),
            ["40.00", "40.00", "85.41"]
        );
    }
}
