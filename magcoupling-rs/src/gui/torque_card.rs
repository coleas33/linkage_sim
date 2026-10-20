//! The torque card (slice 3 spec, BL-073; owner decisions 1A-17A): for each Size control, what
//! one fixed step each way does to the hot-low torque of the design shown. Pure: no egui. The
//! panel draws it and the screen dump prints it.
//!
//! Each side is the design the control's own slider mapping makes at exactly one step (owner
//! 3A: Outside diameter ±1 mm at the user's Minimum OD, Overall length ±1 mm, Magnet face gap
//! ±0.1 mm, Magnets per ring ±2), and its number is the difference of two `compute_all` runs. A
//! side with no honest number reads words, by the order of tests of spec 3.3: the design shown
//! first ([`TorqueCard::ShownDoesNotFit`], [`TorqueCard::ShownNoPrediction`],
//! [`TorqueCard::ShownNoHotLow`]), then the slider's range within [`STEP_SLACK_MM`], the
//! slider's mapping, the validity rule every row shares ([`side_words`]) and, on the Overall
//! length row, the jump.

use crate::engine::meta::SliderRange;
use crate::engine::model::blocks_fit;
use crate::engine::size_controls::{SizeControls, dimensions, dimensions_with_minimum_diameter};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::Level;
use crate::gui::guidance::FACE_GAP;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::size_controls::{self, SliderMapping, predicts, slider_minimum};
use crate::{DesignInputs, DesignResults, compute_all};

/// The one slack of the card's range test and its jump rule [mm] (spec 3.3 steps 2 and 7): far
/// above the float noise of the sliders' mappings (a step that lands 1.42e-14 mm past a range's
/// end, 3.6e-15 mm in `resize_length`'s subtraction), far below the sliders' 0.01 and 0.1 mm
/// steps and any real change.
pub(crate) const STEP_SLACK_MM: f64 = 1e-6;

/// The Magnets per ring step (owner 3A): two magnets, so an even count stays even.
const COUNT_STEP: i64 = 2;

/// A Size-tab control the card steps (owner 2A), in the controls' order (6A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    OutsideDiameter,
    OverallLength,
    FaceGap,
    MagnetsPerRing,
}

impl Control {
    /// The four controls in the Size tab's order: the card's rows keep it (owner 6A).
    pub const ALL: [Control; 4] = [
        Control::OutsideDiameter,
        Control::OverallLength,
        Control::FaceGap,
        Control::MagnetsPerRing,
    ];

    /// The step of decision 3A: 1.0 mm, 1.0 mm, 0.1 mm, 2 magnets.
    pub const fn step(self) -> f64 {
        match self {
            Control::OutsideDiameter | Control::OverallLength => 1.0,
            Control::FaceGap => 0.1,
            Control::MagnetsPerRing => COUNT_STEP as f64,
        }
    }

    /// The label drawn above the control's slider: `size_controls::DIAMETER` and `LENGTH` (no
    /// input carries those two names), and the gap's and the count's input labels (owner 11A).
    pub fn label(self) -> &'static str {
        match self {
            Control::OutsideDiameter => size_controls::DIAMETER,
            Control::OverallLength => size_controls::LENGTH,
            Control::FaceGap => label_for(FACE_GAP),
            Control::MagnetsPerRing => label_for(FreeVariable::MagnetsPerRing.path()),
        }
    }

    /// The Auto variable whose slider is this control: the inverse of
    /// [`size_controls::variable_slider`]; `None` for the gap, which Auto never sets.
    pub fn held_by(self) -> Option<FreeVariable> {
        FreeVariable::ALL
            .into_iter()
            .find(|&variable| size_controls::variable_slider(variable) == self.label())
    }

    /// The step on the `+` side (`more`) or the `-` side, with its sign.
    fn signed_step(self, more: bool) -> f64 {
        if more { self.step() } else { -self.step() }
    }
}

/// Why a side shows words, not a number (owner 9A, 16A; spec T-4), in the order of the tests
/// that find them (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Words {
    /// The step leaves its slider's range (the diameter's at the user's Minimum OD, owner 9A),
    /// or the slider's mapping refuses it.
    EndOfRange,
    /// An odd magnet count from a design file, which the slider does not take.
    NotOnTheSlider,
    /// The stepped design's blocks do not fit.
    DoesNotFit,
    /// The keyway leaves no hub wall past it (not on the count row: decision D-1).
    KeywayBreaksThrough,
    /// The model has no prediction for the stepped design.
    NoPrediction,
    /// The overlapping magnet length does not change by the step (the Overall length row).
    Jump,
}

/// A dashboard check whose level differs on the stepped design (owner 8A).
#[derive(Clone, Debug, PartialEq)]
pub struct Flip {
    /// The check's guide label.
    pub check: &'static str,
    /// Its level on the stepped design.
    pub to: Level,
}

/// The change of a failing check's own quantity (owner 8A; spec 3.6). `unit` is the result's
/// own (`result_info(path).meta.unit`: mm, µm, MPa, °C or N·m for the quoted paths), drawn
/// through `with_unit` as `result_text` does; no unit is typed.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginChange {
    /// The check's guide label.
    pub check: &'static str,
    /// The quantity's name, where the label alone does not name it.
    pub name: Option<&'static str>,
    pub change: f64,
    pub unit: &'static str,
    /// The decimals the change starts from (section 5's rule).
    pub decimals: usize,
}

/// A side with a number.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub struct Change {
    /// The stepped design's hot-low torque (`metal.torque_hot_low_Nm`).
    pub hot_low_Nm: f64,
    /// That hot-low less the design shown's.
    pub change_Nm: f64,
    /// `change_Nm` over the design shown's hot-low.
    pub fraction: f64,
    /// Whether the stepped design takes another calibration factor (`model.f_cal`, the factor
    /// `select_calibration_factor` picks; owner 16A).
    pub calibration_changed: bool,
    /// The Outside diameter the slider would show, less the design shown's ([`dimensions`]);
    /// `None` where either design has no size controls.
    pub diameter_change_mm: Option<f64>,
    /// The Overall length likewise.
    pub length_change_mm: Option<f64>,
    /// The counted checks whose level differs, in `CHECK_GUIDES` order.
    pub flips: Vec<Flip>,
    /// The failing checks' quantities that move and show (spec 3.6).
    pub margins: Vec<MarginChange>,
}

/// One side of a row: words, or a number with what the step costs.
#[derive(Clone, Debug, PartialEq)]
pub enum Side {
    Words(Words),
    Change(Box<Change>),
}

/// One control's row: its `-` side, then its `+` side.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub control: Control,
    pub fewer: Side,
    pub more: Side,
}

/// The card of a design. `ShownNoHotLow`: the model predicts (the readouts show both torques),
/// but hot-low, the percent's base, is not above zero (a design file's variation of 1 or more;
/// physics review 2 MF-1), or the pull-out is not above zero (a magnet Br or calibration
/// coefficient of 0 or less; physics review 3 MF-2).
#[derive(Clone, Debug, PartialEq)]
pub enum TorqueCard {
    ShownDoesNotFit,
    ShownNoPrediction,
    ShownNoHotLow,
    Rows(Vec<Row>),
}

impl TorqueCard {
    /// The rows: empty for the three one-line forms.
    pub fn rows(&self) -> &[Row] {
        match self {
            TorqueCard::Rows(rows) => rows,
            TorqueCard::ShownDoesNotFit
            | TorqueCard::ShownNoPrediction
            | TorqueCard::ShownNoHotLow => &[],
        }
    }
}

/// The card of `inputs`, the design shown, whose results are `results` (`compute_all(inputs)`),
/// with the Size tab's Minimum OD preference (`None`: unset), as the panel holds it, and the
/// shaft-clamp pause (`clamps_paused`, the panel's field): the flips and margins run over
/// `counted_guides(clamps_paused)`.
pub fn card(
    inputs: &DesignInputs,
    results: &DesignResults,
    minimum_diameter_mm: Option<f64>,
    #[expect(
        unused_variables,
        reason = "the flips and margins read it (slice 3 Task 4, which removes this)"
    )]
    clamps_paused: bool,
) -> TorqueCard {
    // Test 0 (spec 3.3): the design shown, by the readouts' own rule.
    if !blocks_fit(&inputs.coupling, &results.model) {
        return TorqueCard::ShownDoesNotFit;
    }
    if !predicts_both(results) {
        return TorqueCard::ShownNoPrediction;
    }
    if !(results.model.pullout_Nm > 0.0 && results.metal.torque_hot_low_Nm > 0.0) {
        return TorqueCard::ShownNoHotLow;
    }
    let shown = Shown {
        inputs,
        results,
        size: dimensions(inputs),
    };
    TorqueCard::Rows(
        Control::ALL
            .into_iter()
            .filter_map(|control| {
                // Test 1: a size row only where the Size tab draws its slider (T-5).
                let slider = Slider::of(control, &shown, minimum_diameter_mm)?;
                Some(Row {
                    control,
                    fewer: shown.side(control, &slider, false),
                    more: shown.side(control, &slider, true),
                })
            })
            .collect(),
    )
}

/// Why a stepped design reads words, not a number, by sizing's own validity rule
/// ([`crate::engine::sizing::is_valid`]) in its order (spec 3.3 steps 4-6): its blocks do not
/// fit; with `keyway`, the keyway leaves no hub wall past it (`model.hub_wall_past_key_mm`, NaN
/// included); the model has no prediction, by the readouts' own rule ([`predicts`]: the end
/// effect out of range, or a torque that is not finite). `None` where the design counts.
/// `results` is `compute_all(design)`. The Magnets per ring row passes `keyway` false (decision
/// D-1): the wall past the keyway does not depend on the count, so the design shown says so.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN wall fails it
pub(crate) fn side_words(
    design: &DesignInputs,
    results: &DesignResults,
    keyway: bool,
) -> Option<Words> {
    if !blocks_fit(&design.coupling, &results.model) {
        Some(Words::DoesNotFit)
    } else if keyway && !(results.model.hub_wall_past_key_mm > 0.0) {
        Some(Words::KeywayBreaksThrough)
    } else if !predicts_both(results) {
        Some(Words::NoPrediction)
    } else {
        None
    }
}

/// One step of the Magnets per ring slider from `npole`, `more` the `+` side (owner 3A: two
/// magnets): the count, or why the slider does not take it. The slider takes the even counts
/// of its range (one block per pole): a step past the range, or one that overflows (a design
/// file's count), reads [`Words::EndOfRange`]; an odd count inside the range, which a design
/// file can hold, [`Words::NotOnTheSlider`].
pub(crate) fn count_step(npole: i64, more: bool) -> Result<i64, Words> {
    let range = FreeVariable::MagnetsPerRing.range();
    let stepped = if more {
        npole.checked_add(COUNT_STEP)
    } else {
        npole.checked_sub(COUNT_STEP)
    };
    match stepped {
        Some(count) if (range.min..=range.max).contains(&(count as f64)) => {
            if count % 2 == 0 {
                Ok(count)
            } else {
                Err(Words::NotOnTheSlider)
            }
        }
        _ => Err(Words::EndOfRange),
    }
}

/// Whether the readouts show both torques of `results` as numbers ([`predicts`]: the end
/// effect in range and the torque finite). The card's "no prediction" rule for the design shown
/// and for every side (spec 3.3 tests 0 and 6), so the readouts, the design shown and each side
/// read one rule.
fn predicts_both(results: &DesignResults) -> bool {
    predicts(results, results.model.pullout_Nm)
        && predicts(results, results.metal.torque_hot_low_Nm)
}

/// `value` on a slider's `range` (spec 3.3 step 2): `None` when it is not finite or lies more
/// than [`STEP_SLACK_MM`] past an end; else the value, taken as that end within the slack past
/// it. The mappings' own range checks are exact, and a step from a value on the slider's grid
/// can land a few ulps past the end the slider reaches exactly.
fn on_slider(value: f64, range: SliderRange) -> Option<f64> {
    (value.is_finite() && value >= range.min - STEP_SLACK_MM && value <= range.max + STEP_SLACK_MM)
        .then(|| value.clamp(range.min, range.max))
}

/// The Magnet face gap slider's range: its input's, as the Size tab draws it.
fn face_gap_range() -> SliderRange {
    InputCatalogue::get()
        .entry(FACE_GAP)
        .and_then(|entry| entry.meta.range)
        .expect("the face gap has a slider")
}

/// The design shown, its results and its size controls ([`dimensions`]: `None` where the Size
/// tab draws no size slider): what each side is compared with.
struct Shown<'a> {
    inputs: &'a DesignInputs,
    results: &'a DesignResults,
    size: Option<SizeControls>,
}

/// How a control's own slider maps a value to a design, with the range it takes (spec 3.3): the
/// range here, the mapping the controls' own ([`SliderMapping`]).
enum Slider {
    /// The Outside diameter slider at the user's Minimum OD: its range there, and its mapping at
    /// `minimum` ([`slider_minimum`]).
    Diameter { size: SizeControls, minimum: f64 },
    /// The Overall length slider: its mapping over the size controls' length range.
    Length { size: SizeControls },
    /// The Magnet face gap slider: its input set inside its input's range.
    FaceGap { range: SliderRange },
    /// The Magnets per ring slider: [`count_step`], then the count set as the slider sets it,
    /// the ring radius kept.
    MagnetsPerRing,
}

impl Slider {
    /// The slider of `control` on the design shown, at the Minimum OD preference `minimum`;
    /// `None` where the Size tab draws no such slider (T-5): a size row without size controls
    /// (an odd or out-of-range count, or a geometry they cannot define), or the diameter row
    /// without a diameter range at that Minimum OD.
    fn of(control: Control, shown: &Shown, minimum: Option<f64>) -> Option<Slider> {
        match control {
            Control::OutsideDiameter => {
                let minimum = slider_minimum(&shown.size?, minimum);
                let size = dimensions_with_minimum_diameter(shown.inputs, minimum)?;
                Some(Slider::Diameter { size, minimum })
            }
            Control::OverallLength => Some(Slider::Length { size: shown.size? }),
            Control::FaceGap => Some(Slider::FaceGap {
                range: face_gap_range(),
            }),
            Control::MagnetsPerRing => Some(Slider::MagnetsPerRing),
        }
    }

    /// The design one step of `control` from `inputs` makes, `more` its `+` side; else why the
    /// slider does not take that step (spec 3.3 steps 2 and 3): a value more than
    /// [`STEP_SLACK_MM`] past an end of its range, one the mapping refuses, or a count
    /// [`count_step`] does not take.
    fn step(
        &self,
        control: Control,
        inputs: &DesignInputs,
        more: bool,
    ) -> Result<DesignInputs, Words> {
        let step = control.signed_step(more);
        let on = |value: f64, range: SliderRange| on_slider(value, range).ok_or(Words::EndOfRange);
        match *self {
            // Each side through the slider's own mapping (`SliderMapping`, the controls' too).
            Slider::Diameter { size, minimum } => {
                let requested = on(size.diameter_mm + step, size.diameter_range)?;
                SliderMapping::Diameter { minimum }
                    .design(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::Length { size } => {
                let requested = on(size.overall_length_mm + step, size.length_range)?;
                SliderMapping::Length
                    .design(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::FaceGap { range } => {
                let gap = on(inputs.metal.face_gap_mm + step, range)?;
                SliderMapping::Input { path: FACE_GAP }
                    .design(inputs, gap)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::MagnetsPerRing => {
                let npole = count_step(inputs.coupling.npole, more)?;
                SliderMapping::Input {
                    path: FreeVariable::MagnetsPerRing.path(),
                }
                .design(inputs, npole as f64)
                .map_err(|_| Words::EndOfRange)
            }
        }
    }
}

impl Shown<'_> {
    /// The side of `control`'s row one step of its `slider` away, `more` the `+` side, by the
    /// order of tests (spec 3.3 steps 2-8).
    fn side(&self, control: Control, slider: &Slider, more: bool) -> Side {
        let design = match slider.step(control, self.inputs, more) {
            Ok(design) => design,
            Err(words) => return Side::Words(words),
        };
        let results = compute_all(&design);
        // Steps 4-6: sizing's validity rule, the keyway left out on the count row (D-1).
        let keyway = control != Control::MagnetsPerRing;
        if let Some(words) = side_words(&design, &results, keyway) {
            return Side::Words(words);
        }
        // Step 7: a length step that is not a step of the overlapping magnet length.
        if control == Control::OverallLength && !self.moves_by_the_step(&results, more) {
            return Side::Words(Words::Jump);
        }
        Side::Change(Box::new(self.change(&design, &results)))
    }

    /// Whether the overlapping magnet length (`model.active_length_mm`) of the stepped design,
    /// whose results are `stepped`, moves by exactly the Overall length step, within
    /// [`STEP_SLACK_MM`] (spec 3.3 step 7: owner 9A, widened by question O-1). A change by
    /// anything else, more or less or the other way, is a jump, and so is a length that is not
    /// a number.
    fn moves_by_the_step(&self, stepped: &DesignResults, more: bool) -> bool {
        let moved = stepped.model.active_length_mm - self.results.model.active_length_mm;
        (moved - Control::OverallLength.signed_step(more)).abs() <= STEP_SLACK_MM
    }

    /// The number side of the stepped `design`, whose results are `stepped` (spec 3.3 step 8):
    /// its hot-low against the design shown's, whatever its sign, and what the step costs.
    #[allow(non_snake_case)] // unit suffix
    fn change(&self, design: &DesignInputs, stepped: &DesignResults) -> Change {
        let shown_hot_low_Nm = self.results.metal.torque_hot_low_Nm;
        let hot_low_Nm = stepped.metal.torque_hot_low_Nm;
        let change_Nm = hot_low_Nm - shown_hot_low_Nm;
        let (diameter_change_mm, length_change_mm) = match (self.size, dimensions(design)) {
            (Some(shown), Some(after)) => (
                Some(after.diameter_mm - shown.diameter_mm),
                Some(after.overall_length_mm - shown.overall_length_mm),
            ),
            _ => (None, None),
        };
        Change {
            hot_low_Nm,
            change_Nm,
            fraction: change_Nm / shown_hot_low_Nm,
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
            diameter_change_mm,
            length_change_mm,
            flips: Vec::new(),
            margins: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::size_controls::{resize_diameter_with_minimum, resize_length};
    use crate::engine::sizing::is_valid;
    use crate::gui::format::fixed;
    use crate::gui::test_support::{
        CardDesign, card_designs, end_effect_at_a_step, keyway_through, no_back_iron,
        short_magnets, side_at, slider_range,
    };

    /// The number sides of the cards of section 2's designs ([`card_designs`]): the tests that
    /// read every one count them, so none can pass by reading none.
    const NUMBER_SIDES: usize = 263;

    /// The design keyed `key` in `designs` ([`card_designs`]).
    fn find<'a>(designs: &'a [CardDesign], key: &str) -> &'a CardDesign {
        designs
            .iter()
            .find(|design| design.key == key)
            .unwrap_or_else(|| panic!("no card design {key}"))
    }

    /// A row's two sides, each with whether it is the `+` side.
    fn sides(row: &Row) -> [(bool, &Side); 2] {
        [(false, &row.fewer), (true, &row.more)]
    }

    /// The design one step of `control` makes from `inputs` through the control's own slider,
    /// written here and not through the card: the stepped value held to the slider's range
    /// (which a number side lies within [`STEP_SLACK_MM`] of), then the mapping the Size tab's
    /// controls apply ([`SliderMapping`]).
    fn slider_step(
        control: Control,
        inputs: &DesignInputs,
        minimum: Option<f64>,
        more: bool,
    ) -> DesignInputs {
        let step = control.signed_step(more);
        let size = || dimensions(inputs).expect("a size row has size controls");
        let held = |value: f64, range: SliderRange| value.clamp(range.min, range.max);
        let (mapping, value) = match control {
            Control::OutsideDiameter => {
                let minimum = slider_minimum(&size(), minimum);
                let slider =
                    dimensions_with_minimum_diameter(inputs, minimum).expect("a diameter range");
                (
                    SliderMapping::Diameter { minimum },
                    held(slider.diameter_mm + step, slider.diameter_range),
                )
            }
            Control::OverallLength => {
                let size = size();
                (
                    SliderMapping::Length,
                    held(size.overall_length_mm + step, size.length_range),
                )
            }
            Control::FaceGap => (
                SliderMapping::Input { path: FACE_GAP },
                held(inputs.metal.face_gap_mm + step, slider_range(FACE_GAP)),
            ),
            Control::MagnetsPerRing => (
                SliderMapping::Input {
                    path: FreeVariable::MagnetsPerRing.path(),
                },
                inputs.coupling.npole as f64 + step,
            ),
        };
        mapping.design(inputs, value).expect("on the slider")
    }

    /// The words of `side`; panics on a number.
    fn words(side: &Side) -> Words {
        match side {
            Side::Words(words) => *words,
            Side::Change(change) => panic!("a number, not words: {change:?}"),
        }
    }

    /// The number of `side`; panics on words.
    fn number(side: &Side) -> Change {
        match side {
            Side::Change(change) => (**change).clone(),
            Side::Words(words) => panic!("words, not a number: {words:?}"),
        }
    }

    #[test]
    fn each_control_is_a_size_tab_slider_with_its_step() {
        // Owner 2A, 3A, 6A and 11A: the four Size controls in the tab's order, each one's step,
        // the label drawn above its slider, and the Auto variable that sets it (the inverse of
        // `size_controls::variable_slider`; the gap is never Auto's).
        use size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        assert_eq!(
            Control::ALL.map(|control| (control.label(), control.step(), control.held_by())),
            [
                (DIAMETER, 1.0, Some(FreeVariable::RingRadius)),
                (LENGTH, 1.0, Some(FreeVariable::AxialLength)),
                (GAP, 0.1, None),
                (MAGNETS, 2.0, Some(FreeVariable::MagnetsPerRing)),
            ]
        );
        for variable in FreeVariable::ALL {
            let held: Vec<Control> = Control::ALL
                .into_iter()
                .filter(|control| control.held_by() == Some(variable))
                .collect();
            assert_eq!(held.len(), 1, "{variable:?}");
            assert_eq!(held[0].label(), size_controls::variable_slider(variable));
        }
    }

    #[test]
    fn each_number_is_the_difference_of_two_compute_all_runs() {
        // Spec 8.1 test 1 (owner 1A): every number side, at every design of section 2, is the
        // stepped design's compute_all less the design shown's, bit for bit, the stepped design
        // rebuilt here through the slider's own mapping.
        let mut numbers = 0;
        for design in card_designs() {
            let before = compute_all(&design.inputs);
            let card = card(&design.inputs, &before, design.minimum_diameter_mm, PAUSED);
            for row in card.rows() {
                for (more, side) in sides(row) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let what = format!("{} {:?} {more}", design.key, row.control);
                    let stepped = slider_step(
                        row.control,
                        &design.inputs,
                        design.minimum_diameter_mm,
                        more,
                    );
                    let after = compute_all(&stepped);
                    let hot_low = after.metal.torque_hot_low_Nm;
                    let change_nm = hot_low - before.metal.torque_hot_low_Nm;
                    assert_eq!(change.hot_low_Nm.to_bits(), hot_low.to_bits(), "{what}");
                    assert_eq!(change.change_Nm.to_bits(), change_nm.to_bits(), "{what}");
                    assert_eq!(
                        change.fraction.to_bits(),
                        (change_nm / before.metal.torque_hot_low_Nm).to_bits(),
                        "{what}"
                    );
                    // The sizes the sliders would show; none exactly where either design has no
                    // size controls.
                    let sizes = match (dimensions(&design.inputs), dimensions(&stepped)) {
                        (Some(shown), Some(stepped)) => [
                            Some(stepped.diameter_mm - shown.diameter_mm),
                            Some(stepped.overall_length_mm - shown.overall_length_mm),
                        ],
                        _ => [None, None],
                    };
                    assert_eq!(
                        [change.diameter_change_mm, change.length_change_mm]
                            .map(|mm| mm.map(f64::to_bits)),
                        sizes.map(|mm| mm.map(f64::to_bits)),
                        "{what}"
                    );
                    assert_eq!(
                        change.calibration_changed,
                        after.model.f_cal != before.model.f_cal,
                        "{what}"
                    );
                    numbers += 1;
                }
            }
        }
        assert_eq!(numbers, NUMBER_SIDES);
    }

    #[test]
    fn the_words_follow_the_order_of_the_tests() {
        // Spec 8.1 test 3: one design per word, in the order of spec 3.3's tests.
        use Control::{FaceGap, MagnetsPerRing, OutsideDiameter, OverallLength};
        let designs = card_designs();
        let design = |key| find(&designs, key);
        let a = design("(a)");

        // Step 4: the blocks do not fit, a step down the ring or two magnets more.
        assert_eq!(words(&a.side(OutsideDiameter, false)), Words::DoesNotFit);
        assert_eq!(words(&a.side(MagnetsPerRing, true)), Words::DoesNotFit);

        // Step 2: off the slider's range, the diameter's at the user's Minimum OD (owner 9A). At
        // 42.8 mm, the diameter shown, the -1 mm side leaves it (at the default it reached the
        // fit test above). At 45 mm, above the diameter shown, both sides lie below the range;
        // at 43.3 mm only the -1 mm side does, and the +1 mm side is (a)'s.
        for (key, more) in [
            ("(a), Min OD 42.8", false),
            ("(a), Min OD 45", false),
            ("(a), Min OD 45", true),
            ("(a), Min OD 45: 43.3 mm", false),
        ] {
            assert_eq!(
                words(&design(key).side(OutsideDiameter, more)),
                Words::EndOfRange,
                "{key} {more}"
            );
        }
        assert_eq!(
            number(&design("(a), Min OD 45: 43.3 mm").side(OutsideDiameter, true)),
            number(&a.side(OutsideDiameter, true))
        );
        // Each slider's own ends: the gap at its bottom and top, the count at 4 and 40 (forty
        // narrow magnets, which fit), the Outside diameter and the Overall length at theirs.
        for (key, control, more) in [
            ("gap at 0.3", FaceGap, false),
            ("slider tops: gap 5.0", FaceGap, true),
            ("npole 4", MagnetsPerRing, false),
            ("npole 40", MagnetsPerRing, true),
            ("slider tops: Outside diameter 80", OutsideDiameter, true),
            ("slider tops: Overall length 69.9", OverallLength, true),
            ("slider tops: Overall length 21.1", OverallLength, false),
        ] {
            assert_eq!(
                words(&design(key).side(control, more)),
                Words::EndOfRange,
                "{key}"
            );
        }
        // A count from a design file the slider does not take: odd, or overflowing (the design
        // shown does not fit there, so no row is reached; `count_step` itself is pinned by
        // `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`).
        for more in [false, true] {
            assert_eq!(
                words(&design("odd count").side(MagnetsPerRing, more)),
                Words::NotOnTheSlider
            );
        }
        for key in ["count overflow: i64::MAX", "count overflow: i64::MIN"] {
            assert_eq!(design(key).card(), TorqueCard::ShownDoesNotFit, "{key}");
        }

        // Step 5: the keyway through the hub, on the diameter, length and gap rows; not on the
        // count row (decision D-1), whose -2 side is the number of the keyway intact.
        let keyway = design("keyway through");
        for control in [OverallLength, FaceGap] {
            for more in [false, true] {
                assert_eq!(
                    words(&keyway.side(control, more)),
                    Words::KeywayBreaksThrough,
                    "{control:?} {more}"
                );
            }
        }
        let mut intact = keyway.inputs.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert_eq!(
            number(&keyway.side(MagnetsPerRing, false))
                .change_Nm
                .to_bits(),
            number(&side_at(&intact, None, MagnetsPerRing, false))
                .change_Nm
                .to_bits()
        );

        // Step 6: no prediction, a step shorter or two magnets fewer at the end effect's edge.
        let edge = design("end effect at a step");
        assert_eq!(words(&edge.side(OverallLength, false)), Words::NoPrediction);
        assert_eq!(
            words(&edge.side(MagnetsPerRing, false)),
            Words::NoPrediction
        );

        // Step 7: the jump, words whenever the overlapping magnet length changes by anything but
        // the step (O-1). The mixed pair jumps both ways; its reverse moves by the step.
        for more in [false, true] {
            assert_eq!(
                words(&design("mixed").side(OverallLength, more)),
                Words::Jump
            );
            assert!(matches!(
                design("reverse").side(OverallLength, more),
                Side::Change(_)
            ));
        }
        // The cup cavity shorter than the outer ring: at 11.0 mm both sides jump (red first: 9A as
        // written read the +1 mm side, -0.70 mm of active length, as a number); at 12.0 and
        // 12.5 mm the +1 mm side moves it +0.30 and +0.80 mm; at 10.7 mm it moves -1.00 mm, the
        // step's size the wrong way (an unsigned comparison read it as a number). At 12.7 and
        // 15.5 mm both sides move it by exactly the step.
        let cavity = |mm: f64| {
            let mut inputs = DesignInputs::default();
            inputs.metal.cup_depth_mm = mm;
            inputs
        };
        assert_eq!(
            words(&design("short cavity").side(OverallLength, false)),
            Words::Jump
        );
        for mm in [11.0, 12.0, 12.5, 10.7] {
            assert_eq!(
                words(&side_at(&cavity(mm), None, OverallLength, true)),
                Words::Jump,
                "{mm}"
            );
        }
        for mm in [12.7, 15.5] {
            for more in [false, true] {
                assert!(
                    matches!(
                        side_at(&cavity(mm), None, OverallLength, more),
                        Side::Change(_)
                    ),
                    "{mm} {more}"
                );
            }
        }

        // Step 8: a number whatever its sign. (c) with a design file's calibration coefficient
        // of -0.95: two magnets fewer take the coefficient itself, a hot-low below zero; and (c)
        // at 8 poles with a measured torque of 0: two magnets more take the prototype's measured
        // factor, 0, a hot-low of exactly 0 (owner 16A; code review 4 MF-2, physics review 5
        // SF-2).
        let negative = design("(c), f_cal_original -0.95");
        let fewer = number(&negative.side(MagnetsPerRing, false));
        let stepped = compute_all(&FreeVariable::MagnetsPerRing.apply(&negative.inputs, 8.0));
        assert!(fewer.calibration_changed);
        assert!(fewer.hot_low_Nm < 0.0, "{fewer:?}");
        assert_eq!(
            fewer.hot_low_Nm.to_bits(),
            stepped.metal.torque_hot_low_Nm.to_bits()
        );
        let mut unmeasured = no_back_iron();
        unmeasured.coupling.npole = 8;
        unmeasured.calibration.measured_torque_Nm = 0.0;
        let more = number(&side_at(&unmeasured, None, MagnetsPerRing, true));
        assert!(more.calibration_changed);
        assert_eq!(more.hot_low_Nm, 0.0, "{more:?}");

        // A step that lands on an end (the slack of step 2; code review 2 MF-1): one slider
        // step inside each end, the side is a number, the end's own design bit for bit (red
        // first: an exact range test read the length's +1 mm side at 68.9 mm as the end of its
        // range, 1.42e-14 mm past it).
        for (key, more) in [
            ("steps to an end: (a) at length 68.9", true),
            ("steps to an end: (a) at length 22.1", false),
        ] {
            let shown = design(key);
            let lengths = dimensions(&shown.inputs)
                .expect("size controls")
                .length_range;
            let end = if more { lengths.max } else { lengths.min };
            let at_end = resize_length(&shown.inputs, end).expect("the range's end");
            assert_eq!(
                number(&shown.side(OverallLength, more))
                    .hot_low_Nm
                    .to_bits(),
                compute_all(&at_end).metal.torque_hot_low_Nm.to_bits(),
                "{key}"
            );
        }
        for (key, control, more) in [
            ("steps to an end: (a) at gap 4.9", FaceGap, true),
            ("steps to an end: (a) at gap 0.4", FaceGap, false),
            (
                "steps to an end: (a) at Outside diameter 79",
                OutsideDiameter,
                true,
            ),
            (
                "steps to an end: (e) at Outside diameter 79",
                OutsideDiameter,
                true,
            ),
        ] {
            assert!(
                matches!(design(key).side(control, more), Side::Change(_)),
                "{key}"
            );
        }
        // (e) with its Minimum OD 1 mm under the diameter shown: the -1 mm side lands on the
        // range's start, the number at the default preference.
        assert_eq!(
            number(&design("steps to an end: (e), Min OD 41.8").side(OutsideDiameter, false)),
            number(&design("(e)").side(OutsideDiameter, false))
        );
    }

    #[test]
    fn the_design_shown_decides_first() {
        // Spec 8.1 test 4: the design shown decides before any row, by the readouts' own rule
        // (physics review 3 MF-2).
        let designs = card_designs();
        let design = |key| find(&designs, key).inputs.clone();
        let mut negative_coefficient = DesignInputs::default();
        negative_coefficient.calibration.f_cal_original = -0.95;
        let mut negative_pull_out = negative_coefficient.clone();
        negative_pull_out.metal.variation = 1.5;
        for (name, inputs, shown) in [
            // The end effect out of range: the readouts show no prediction.
            (
                "short magnets",
                design("short magnets"),
                TorqueCard::ShownNoPrediction,
            ),
            // The model predicts and the readouts show both torques, but hot-low, the percent's
            // base, is not above zero (a design file's variation).
            (
                "variation 1.0",
                design("variation 1.0"),
                TorqueCard::ShownNoHotLow,
            ),
            (
                "variation 1.0: 1.5",
                design("variation 1.0: 1.5"),
                TorqueCard::ShownNoHotLow,
            ),
            // Zero torque, shown as 0 N·m by both readouts (red first: revision 2's rule read
            // "no prediction" for a pull-out that is not above zero).
            (
                "zero torque",
                design("zero torque"),
                TorqueCard::ShownNoHotLow,
            ),
            // Both torques below zero; then the pull-out below zero with hot-low above it, which
            // a rule on hot-low alone would let through to the rows.
            (
                "f_cal_original -0.95",
                negative_coefficient,
                TorqueCard::ShownNoHotLow,
            ),
            (
                "f_cal_original -0.95, variation 1.5",
                negative_pull_out,
                TorqueCard::ShownNoHotLow,
            ),
            (
                "shown does not fit",
                design("shown does not fit"),
                TorqueCard::ShownDoesNotFit,
            ),
        ] {
            let results = compute_all(&inputs);
            let torques = [results.model.pullout_Nm, results.metal.torque_hot_low_Nm];
            // What the readouts show (`size_controls::predicts`), and the torques' signs.
            let shows_both = torques.iter().all(|&torque| predicts(&results, torque));
            let both_above_zero = torques.iter().all(|&torque| torque > 0.0);
            match shown {
                TorqueCard::ShownNoPrediction => assert!(!shows_both, "{name}"),
                TorqueCard::ShownNoHotLow => {
                    assert!(shows_both && !both_above_zero, "{name}: {torques:?}")
                }
                _ => assert!(!blocks_fit(&inputs.coupling, &results.model), "{name}"),
            }
            if name == "f_cal_original -0.95, variation 1.5" {
                assert!(torques[0] < 0.0 && torques[1] > 0.0, "{torques:?}");
            }
            let card = card(&inputs, &results, None, PAUSED);
            assert_eq!(card, shown, "{name}");
            assert!(card.rows().is_empty(), "{name}");
        }
    }

    #[test]
    fn the_size_rows_follow_the_size_controls() {
        // Spec 8.1 test 5 (T-5): the Outside diameter and Overall length rows only where the
        // Size tab draws their sliders; every row in the controls' order (owner 6A).
        let mut without_size_controls = Vec::new();
        for design in card_designs() {
            let card = design.card();
            let controls: Vec<Control> = card.rows().iter().map(|row| row.control).collect();
            if !matches!(card, TorqueCard::Rows(_)) {
                assert!(controls.is_empty(), "{}", design.key);
            } else if dimensions(&design.inputs).is_some() {
                assert_eq!(controls, Control::ALL, "{}", design.key);
            } else {
                assert_eq!(
                    controls,
                    [Control::FaceGap, Control::MagnetsPerRing],
                    "{}",
                    design.key
                );
                without_size_controls.push(design.key);
            }
        }
        assert_eq!(without_size_controls, ["odd count"]);
    }

    #[test]
    fn the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum() {
        // Spec 8.1 test 6 (owner 9A: the slider's own range at the user's Minimum OD).
        let designs = card_designs();
        let design = |key| find(&designs, key);
        // At (e) the lowest Minimum OD and the default map a step each way to the same rings,
        // 9.688060 and 10.611940 mm, so the row is the same.
        let (e, lowest) = (design("(e)"), design("(e), low Min OD"));
        let size = dimensions(&e.inputs).expect("(e) has size controls");
        for (step, ring) in [(-1.0, "9.688060"), (1.0, "10.611940")] {
            for preference in [e.minimum_diameter_mm, lowest.minimum_diameter_mm] {
                let minimum = slider_minimum(&size, preference);
                let stepped =
                    resize_diameter_with_minimum(&e.inputs, size.diameter_mm + step, minimum)
                        .expect("on the slider");
                assert_eq!(fixed(stepped.coupling.inner_back_apothem_mm, 6), ring);
            }
        }
        assert_eq!(lowest.card().rows()[0], e.card().rows()[0]);
        assert_eq!(e.card().rows()[0].control, Control::OutsideDiameter);
        // At (a) with Minimum OD 42.8 mm, the diameter shown, the slider starts there: the -1 mm
        // side leaves it and the +1 mm side reads as at the default.
        let (a, at_shown) = (design("(a)"), design("(a), Min OD 42.8"));
        assert_eq!(
            slider_minimum(
                &dimensions(&a.inputs).expect("(a) has size controls"),
                at_shown.minimum_diameter_mm
            ),
            42.8
        );
        assert_eq!(
            at_shown.side(Control::OutsideDiameter, false),
            Side::Words(Words::EndOfRange)
        );
        assert_eq!(
            at_shown.side(Control::OutsideDiameter, true),
            a.side(Control::OutsideDiameter, true)
        );
        // At every design of section 2 the row reads the Outside diameter slider at the user's
        // Minimum OD: the range at `slider_minimum` (the controls' own expression) starts at it,
        // and a side reads the end of its range exactly when its step lands more than
        // STEP_SLACK_MM outside that range.
        let mut rows = 0;
        for design in &designs {
            let card = design.card();
            let Some(row) = card
                .rows()
                .iter()
                .find(|row| row.control == Control::OutsideDiameter)
            else {
                continue;
            };
            let legacy = dimensions(&design.inputs).expect("a diameter row has size controls");
            let minimum = slider_minimum(&legacy, design.minimum_diameter_mm);
            let slider = dimensions_with_minimum_diameter(&design.inputs, minimum)
                .expect("a diameter range");
            assert_eq!(slider.diameter_range.min, minimum, "{}", design.key);
            for (more, side) in sides(row) {
                let requested = slider.diameter_mm + Control::OutsideDiameter.signed_step(more);
                let range = slider.diameter_range;
                let off =
                    requested < range.min - STEP_SLACK_MM || requested > range.max + STEP_SLACK_MM;
                assert_eq!(
                    *side == Side::Words(Words::EndOfRange),
                    off,
                    "{} {more}",
                    design.key
                );
            }
            rows += 1;
        }
        // Not vacuous: every diameter row of section 2's designs.
        assert_eq!(rows, 42);
    }

    #[test]
    fn a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range() {
        // A design file or link can carry a value past a slider's end (`validate()` checks only
        // that a number is finite). A side more than STEP_SLACK_MM outside the slider's range
        // reads the end of its range (spec 3.3 step 2): on both sides where the design shown
        // itself lies past an end; a side that lands back on the slider is a number, the
        // slider's own design.
        use Control::{FaceGap, OutsideDiameter, OverallLength};
        let with = |edit: fn(&mut DesignInputs)| {
            let mut inputs = DesignInputs::default();
            edit(&mut inputs);
            inputs
        };
        let gap_above = with(|i| i.metal.face_gap_mm = 5.5);
        let gap_below = with(|i| i.metal.face_gap_mm = 0.2);
        let long = with(|i| i.coupling.magnets.axial_length_mm = Some(60.0));
        for (inputs, control) in [(&gap_above, FaceGap), (&long, OverallLength)] {
            for more in [false, true] {
                assert_eq!(
                    words(&side_at(inputs, None, control, more)),
                    Words::EndOfRange,
                    "{control:?} {more}"
                );
            }
        }
        assert_eq!(
            words(&side_at(&gap_below, None, FaceGap, false)),
            Words::EndOfRange
        );
        let up = number(&side_at(&gap_below, None, FaceGap, true));
        let stepped = compute_all(&slider_step(FaceGap, &gap_below, None, true));
        assert_eq!(
            up.hot_low_Nm.to_bits(),
            stepped.metal.torque_hot_low_Nm.to_bits()
        );
        // The Minimum OD at its slider's top: the diameter's range is that one value, above the
        // diameter shown, so both sides lie off it.
        let top = dimensions(&DesignInputs::default())
            .expect("size controls")
            .minimum_diameter_range
            .max;
        for more in [false, true] {
            assert_eq!(
                words(&side_at(
                    &DesignInputs::default(),
                    Some(top),
                    OutsideDiameter,
                    more
                )),
                Words::EndOfRange
            );
        }
    }

    #[test]
    fn count_step_keeps_even_counts_inside_the_slider_and_never_overflows() {
        // Moved with the neighbour lines' rule (`neighbour_counts`, gui/guidance.rs), one side
        // at a time: two magnets each way, an even count inside the magnet slider's range.
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        assert_eq!(count_step(fewest + 4, false), Ok(fewest + 2));
        assert_eq!(count_step(fewest + 4, true), Ok(fewest + 6));
        // At the slider's ends only the step inside it.
        assert_eq!(count_step(fewest, false), Err(Words::EndOfRange));
        assert_eq!(count_step(most, true), Err(Words::EndOfRange));
        // A count from a design file may be odd or past the slider: odd inside the range is not
        // a value the slider takes; past it, the range decides first.
        for more in [false, true] {
            assert_eq!(count_step(fewest + 3, more), Err(Words::NotOnTheSlider));
        }
        assert_eq!(count_step(most + 1, false), Err(Words::NotOnTheSlider));
        assert_eq!(count_step(most + 1, true), Err(Words::EndOfRange));
        assert_eq!(count_step(most + 2, false), Ok(most));
        assert_eq!(count_step(most + 2, true), Err(Words::EndOfRange));
        assert_eq!(count_step(fewest - 2, true), Ok(fewest));
        assert_eq!(count_step(fewest - 2, false), Err(Words::EndOfRange));
        // Never overflowing: checked arithmetic reads the end of the range.
        for more in [false, true] {
            assert_eq!(count_step(most + 4, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MAX, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MIN, more), Err(Words::EndOfRange));
        }
    }

    #[test]
    fn a_side_reads_a_number_exactly_when_sizing_counts_its_design() {
        // Spec 3.3 steps 4-6: `side_words` is sizing's validity rule (`is_valid`) in its order,
        // one rule for every row. Moved with the neighbour lines' tests of gui/guidance.rs: their
        // designs at every even count (the defaults, magnets too short for the model, the end
        // effect near its edge) and the keyway through the hub, broken and intact.
        let range = FreeVariable::MagnetsPerRing.range();
        let counts = (range.min as i64..=range.max as i64).step_by(2);
        let mut intact = keyway_through();
        intact.coupling.keyway_depth_mm = 0.0;
        let at = |inputs: &DesignInputs, npole: i64| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let results = compute_all(&design);
            (design, results)
        };
        let mut found = Vec::new();
        for base in [
            DesignInputs::default(),
            short_magnets(),
            end_effect_at_a_step(),
            keyway_through(),
            intact.clone(),
        ] {
            for npole in counts.clone() {
                let (design, results) = at(&base, npole);
                let read = side_words(&design, &results, true);
                assert_eq!(
                    read.is_none(),
                    is_valid(&design, &results),
                    "{npole}: {read:?}"
                );
                found.extend(read);
            }
        }
        // Not vacuous: every word of steps 4-6 is met.
        for word in [
            Words::DoesNotFit,
            Words::KeywayBreaksThrough,
            Words::NoPrediction,
        ] {
            assert!(found.contains(&word), "{word:?}");
        }
        // Decision D-1: with the keyway left out, a keyway through the hub changes no side's
        // words; each count reads as with the keyway intact (the wall past it does not depend on
        // the count).
        for npole in counts {
            let (broken, broken_results) = at(&keyway_through(), npole);
            let (whole, whole_results) = at(&intact, npole);
            assert_eq!(
                side_words(&broken, &broken_results, false),
                side_words(&whole, &whole_results, true),
                "{npole}"
            );
        }
        // The order: the fit before the keyway, the keyway before the prediction.
        let (crowded, crowded_results) = at(&keyway_through(), 12);
        assert_eq!(
            side_words(&crowded, &crowded_results, true),
            Some(Words::DoesNotFit)
        );
        let mut short_and_keyed = short_magnets();
        short_and_keyed.coupling.bore_mm = keyway_through().coupling.bore_mm;
        short_and_keyed.coupling.keyway_depth_mm = keyway_through().coupling.keyway_depth_mm;
        let results = compute_all(&short_and_keyed);
        assert_eq!(
            side_words(&short_and_keyed, &results, true),
            Some(Words::KeywayBreaksThrough)
        );
        assert_eq!(
            side_words(&short_and_keyed, &results, false),
            Some(Words::NoPrediction)
        );
    }
}
