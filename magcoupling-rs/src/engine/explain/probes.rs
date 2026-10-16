//! The five probe designs (guidance slice 2, spec 3.2): where the note tests and each note's
//! physics review read every note, so a sentence true only at the defaults shows. Also the
//! move those checks make: one slider step of one input ([`step_input`]). Pure std: the engine
//! tests (`tests/explain.rs`) and the GUI tests (`gui::explorer`) both use them.

use super::notes::Step;
use crate::engine::api::{DesignInputs, OPTIONAL_SEEDS, compute_all};
use crate::engine::material_library::BACK_IRON_CHOICES;
use crate::engine::meta::{InputSet, ResultSet, Value, input_rows};
use crate::engine::picks::set_by_pick;
use crate::engine::size_controls::with_ring_radius;

/// The probe designs' names, in spec 3.2's order.
pub const PROBE_NAMES: [&str; 5] = [
    "default",
    "8 magnets per ring",
    "304 back iron",
    "Y30 ferrite, custom dimensions",
    "20 mm ring radius",
];

/// The second probe's magnets per ring (the owner's 10 to 8 case).
pub const PROBE_MAGNETS: i64 = 8;

/// The third probe's back iron, by library id: 304 stainless, non-ferromagnetic, so the
/// free-space circuit is in effect.
pub const PROBE_BACK_IRON: &str = "304_annealed";

/// The fourth probe's grade, on both rings with their manual dimensions: hard ferrite.
pub const PROBE_GRADE: &str = "Y30";

/// The fifth probe's ring radius (`coupling.inner_back_apothem_mm`) [mm]: a larger ring, at
/// the dip in torque between about 14 and 20 mm (spec section 8).
pub const PROBE_RING_RADIUS_MM: f64 = 20.0;

/// The five probe designs, named, in [`PROBE_NAMES`]' order: the default (Example) design; the
/// default at [`PROBE_MAGNETS`] magnets per ring; the default with a [`PROBE_BACK_IRON`] back
/// iron; [`PROBE_GRADE`] on both rings, on their custom (manual) dimensions; the default at a
/// [`PROBE_RING_RADIUS_MM`] ring radius, as the Size tab's Outside diameter slider sets it (the
/// cap OD and thread follow the cup: `size_controls::with_ring_radius`). Each computes, counts
/// for inverse sizing and draws on the Size tab (the tests check it).
pub fn probe_designs() -> Vec<(&'static str, DesignInputs)> {
    let default = DesignInputs::default();
    let mut magnets = default.clone();
    magnets.coupling.npole = PROBE_MAGNETS;
    let mut back_iron = default.clone();
    back_iron.materials.parts.back_iron = BACK_IRON_CHOICES
        .iter()
        .find(|&&(_, id)| id == PROBE_BACK_IRON)
        .map(|&(code, _)| code)
        .expect("304 stainless is a back-iron choice (the tests check it)");
    let mut ferrite = default.clone();
    ferrite.coupling.magnets.part_inner.clear();
    ferrite.coupling.magnets.part_outer.clear();
    ferrite.coupling.magnets.grade_inner = PROBE_GRADE.to_owned();
    ferrite.coupling.magnets.grade_outer = PROBE_GRADE.to_owned();
    let ring = with_ring_radius(&default, PROBE_RING_RADIUS_MM)
        .expect("the Size tab reaches a 20 mm ring radius from the default (the tests check it)");
    PROBE_NAMES
        .into_iter()
        .zip([default, magnets, back_iron, ferrite, ring])
        .collect()
}

/// Why [`step_input`] leaves an input where it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unmoved {
    /// The entry chooses or enters; it does not step.
    Choice,
    /// No input has the path.
    NoSuchInput,
    /// A part pick sets the input on this design ([`set_by_pick`]): the pick's input path.
    Held(&'static str),
    /// The input is no number with a slider here: a text, a selector, or an optional number
    /// left blank with no result to start from.
    NotANumber,
    /// One more step its way is past its slider (`SliderRange::step_from`).
    AtEnd,
}

/// `design` with the input at `path` one slider step up ([`Step::Raise`]) or down
/// ([`Step::Lower`]), as a check guide's fix moves it (`SliderRange::step_from`); every other
/// input keeps its value. A count stays a whole number. An optional number left blank starts
/// from the result it enters at (`api::OPTIONAL_SEEDS`, as its input row enters it).
pub fn step_input(design: &DesignInputs, path: &str, step: Step) -> Result<DesignInputs, Unmoved> {
    let up = step.up().ok_or(Unmoved::Choice)?;
    let row = input_rows(design)
        .into_iter()
        .find(|r| r.path == path)
        .ok_or(Unmoved::NoSuchInput)?;
    if let Some(pick) = set_by_pick(design, path) {
        return Err(Unmoved::Held(pick));
    }
    let range = row.meta.range.ok_or(Unmoved::NotANumber)?;
    let x = match row.value {
        Value::Num(x) => x,
        Value::Int(n) => n as f64,
        Value::None => {
            let (_, seed) = OPTIONAL_SEEDS
                .iter()
                .find(|(input, _)| *input == path)
                .ok_or(Unmoved::NotANumber)?;
            match compute_all(design).get(seed) {
                Some(Value::Num(x)) => x,
                _ => return Err(Unmoved::NotANumber),
            }
        }
        Value::Text(_) => return Err(Unmoved::NotANumber),
    };
    let next = range.step_from(x, up).ok_or(Unmoved::AtEnd)?;
    let value = match row.value {
        Value::Int(_) => Value::Int(next.round() as i64),
        _ => Value::Num(next),
    };
    let mut moved = design.clone();
    moved
        .set(path, value)
        .expect("a number on its slider is a value the input takes");
    Ok(moved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grades;
    use crate::engine::material_library::chosen;
    use crate::engine::meta::SliderRange;
    use crate::engine::size_controls::dimensions;
    use crate::engine::sizing::is_valid;

    /// The slider of the input at `path`.
    fn slider(path: &str) -> SliderRange {
        input_rows(&DesignInputs::default())
            .into_iter()
            .find(|r| r.path == path)
            .and_then(|r| r.meta.range)
            .expect("an input with a slider")
    }

    /// A finite number result.
    fn number(design: &DesignInputs, path: &str) -> f64 {
        match compute_all(design).get(path) {
            Some(Value::Num(x)) if x.is_finite() => x,
            other => panic!("{path}: {other:?}"),
        }
    }

    #[test]
    fn the_probes_are_the_five_designs_their_names_say() {
        let designs = probe_designs();
        let names: Vec<&str> = designs.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, PROBE_NAMES);
        let default = DesignInputs::default();
        let [magnets, back_iron, ferrite, ring] = [1, 2, 3, 4].map(|i| &designs[i].1);
        assert_eq!(designs[0].1, default);
        // The count alone changes.
        assert_eq!(magnets.coupling.npole, PROBE_MAGNETS);
        let mut back = magnets.clone();
        back.coupling.npole = default.coupling.npole;
        assert_eq!(back, default);
        // A non-ferromagnetic pick: the free-space circuit is in effect.
        let pick = chosen(&BACK_IRON_CHOICES, back_iron.materials.parts.back_iron).unwrap();
        assert_eq!(pick.id, PROBE_BACK_IRON);
        assert!(!pick.ferromagnetic);
        assert_eq!(
            compute_all(back_iron).get("materials.circuit_backiron"),
            Some(Value::Int(0))
        );
        // Both rings in the grade mode, with the grade's remanence, and the cold check runs.
        let y30 = grades::grade(PROBE_GRADE).unwrap();
        assert_eq!(number(ferrite, "model.inner_br_T"), y30.br_T);
        assert_eq!(number(ferrite, "model.outer_br_T"), y30.br_T);
        assert!(number(ferrite, "temperature.demag.cold_limit_C").is_finite());
        // The ring radius as the Size tab sets it: the cap follows the cup.
        assert_eq!(ring.coupling.inner_back_apothem_mm, PROBE_RING_RADIUS_MM);
        assert!(ring.metal.cap_od_mm > default.metal.cap_od_mm);
        assert_eq!(
            with_ring_radius(&default, PROBE_RING_RADIUS_MM).as_ref(),
            Some(ring)
        );
    }

    #[test]
    fn every_probe_computes_counts_for_inverse_sizing_and_draws_on_the_size_tab() {
        for (name, design) in probe_designs() {
            assert!(design.validate().is_ok(), "{name}");
            let results = compute_all(&design);
            assert!(is_valid(&design, &results), "{name}");
            assert!(dimensions(&design).is_some(), "{name}");
            for path in [
                "model.pullout_Nm",
                "metal.torque_hot_low_Nm",
                "model.cup_od_mm",
            ] {
                assert!(number(&design, path) > 0.0, "{name}: {path}");
            }
        }
    }

    #[test]
    fn a_step_moves_one_input_one_slider_step_its_way() {
        let default = DesignInputs::default();
        let gap = "metal.face_gap_mm";
        let lower = step_input(&default, gap, Step::Lower).unwrap();
        assert_eq!(
            lower.metal.face_gap_mm,
            default.metal.face_gap_mm - slider(gap).step
        );
        let mut back = lower.clone();
        back.metal.face_gap_mm = default.metal.face_gap_mm;
        assert_eq!(back, default, "nothing else moves");
        let raise = step_input(&default, gap, Step::Raise).unwrap();
        assert_eq!(
            raise.metal.face_gap_mm,
            default.metal.face_gap_mm + slider(gap).step
        );
        // A count stays whole.
        let poles = step_input(&default, "coupling.npole", Step::Raise).unwrap();
        assert_eq!(
            poles.coupling.npole,
            default.coupling.npole + slider("coupling.npole").step as i64
        );
        // A blank optional number starts where its row enters it: the axial length at the inner
        // ring's length in use, the measured drag at the model's drag.
        let length = "coupling.magnets.axial_length_mm";
        let longer = step_input(&default, length, Step::Raise).unwrap();
        assert_eq!(
            longer.coupling.magnets.axial_length_mm,
            Some(number(&default, "model.inner_length_mm") + slider(length).step)
        );
        let drag = "metal.measured_drag_Nm";
        let more = step_input(&default, drag, Step::Raise).unwrap();
        assert_eq!(
            more.metal.measured_drag_Nm,
            Some(number(&default, "temperature.slip_loss.drag_Nm") + slider(drag).step)
        );
    }

    #[test]
    fn a_step_names_why_it_cannot_move() {
        let default = DesignInputs::default();
        let gap = "metal.face_gap_mm";
        assert_eq!(
            step_input(&default, "materials.parts.back_iron", Step::Choose),
            Err(Unmoved::Choice)
        );
        // A number entered from a measurement is not stepped either.
        assert_eq!(
            step_input(&default, "metal.measured_drag_Nm", Step::Enter),
            Err(Unmoved::Choice)
        );
        assert_eq!(
            step_input(&default, "no.such.input", Step::Raise),
            Err(Unmoved::NoSuchInput)
        );
        for not_a_number in ["materials.parts.back_iron", "coupling.magnets.part_inner"] {
            assert_eq!(
                step_input(&default, not_a_number, Step::Raise),
                Err(Unmoved::NotANumber),
                "{not_a_number}"
            );
        }
        assert_eq!(
            step_input(
                &default,
                "coupling.magnets.manual_inner_length_mm",
                Step::Raise
            ),
            Err(Unmoved::Held("coupling.magnets.part_inner"))
        );
        let mut top = default.clone();
        top.metal.face_gap_mm = slider(gap).max;
        assert_eq!(step_input(&top, gap, Step::Raise), Err(Unmoved::AtEnd));
        assert!(step_input(&top, gap, Step::Lower).is_ok());
        let (_, steel_304) = &probe_designs()[2];
        assert_eq!(
            step_input(steel_304, "coupling.backiron", Step::Raise),
            Err(Unmoved::Held("materials.parts.back_iron"))
        );
    }
}
