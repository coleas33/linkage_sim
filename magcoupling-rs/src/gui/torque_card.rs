//! The torque card (slice 3 spec, BL-073; owner decisions 1A-17A): for each Size control, what
//! one fixed step each way does to the hot-low torque of the design shown. Pure: no egui. The
//! panel draws it and the screen dump prints it.
//!
//! Each side is the design the control's own slider mapping makes at exactly one step (owner
//! 3A: Outside diameter ±1 mm at the user's Minimum OD, Overall length ±1 mm, Magnet face gap
//! ±0.1 mm, Magnets per ring ±2), and its number is the difference of two `compute_all` runs. A
//! side with no honest number reads words, by the order of tests of spec 3.3: the design shown
//! first ([`TorqueCard::ShownCountBelowSlider`], the owner's FO-3, then
//! [`TorqueCard::ShownDoesNotFit`], [`TorqueCard::ShownNoPrediction`],
//! [`TorqueCard::ShownNoHotLow`]), then the slider's range within [`STEP_SLACK_MM`], the
//! slider's mapping, the validity rule every row shares ([`side_words`]) and, on the Overall
//! length and Outside diameter rows, the jump (the latter the owner's FO-2). A number side also
//! says what the step costs (owner 8A, spec 3.6): the change in Outside diameter and Overall
//! length, the counted dashboard checks whose level flips, and the change of each failing
//! check's own quantity ([`check_terms`]).
//!
//! [`TorqueCard::lines`] draws it: every text a named constant here (drafts until the slice's G-3
//! review sets them), every number from the results through the Size tab's one number format
//! (decision 10B with the owner's FO-4 bound, `gui::format`), a row's two N·m at one precision
//! (T-14). The Auto marking is drawn there too, outside the card the cache keeps, so one card
//! serves every Auto state.

use crate::engine::meta::SliderRange;
use crate::engine::model::blocks_fit;
use crate::engine::size_controls::{SizeControls, dimensions, dimensions_with_minimum_diameter};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::{Level, result_info};
use crate::gui::format::{
    TorquePrecision, decimals_with_a_digit, shared_torque_precision, signed_change, signed_percent,
    signed_torque_change, with_unit,
};
use crate::gui::guidance::{
    CheckGuide, FACE_GAP, check_terms, counted_guides, level, result_number,
};
use crate::gui::inputs::slider_range;
use crate::gui::plain_text::label_for;
use crate::gui::size_controls::{self, SET_BY_AUTO, SliderMapping, predicts, slider_minimum};
use crate::{DesignInputs, DesignResults, compute_all};

/// The one slack of the card's range test and its jump rules [mm] (spec 3.3 steps 2 and 7): far
/// above the float noise of the sliders' mappings (a step that lands 1.42e-14 mm past a range's
/// end, 3.6e-15 mm in `resize_length`'s subtraction), far below the sliders' 0.01 and 0.1 mm
/// steps and any real change.
pub(crate) const STEP_SLACK_MM: f64 = 1e-6;

/// The Magnets per ring step (owner 3A): two magnets, so an even count stays even.
const COUNT_STEP: i64 = 2;

/// The decimals a size change starts from (spec section 5): the sliders' 0.1 mm. A torque change
/// starts from the Size tab's own, `TORQUE_DECIMALS` (`gui::format`).
const SIZE_DECIMALS: usize = 1;

// The card's texts (spec 4.1 and 4.2): drafts until the slice's G-3 review sets the words. A
// row's label is its Size control's ([`Control::label`]), a check's its guide's.

/// The card's heading (named apart from `size_controls::HEADING` and `panel::HEADING`).
pub const CARD_HEADING: &str = "One step of each size control";
/// The first line of the rows (owner 12A): every number on the card is a change.
pub const CAPTION: &str =
    "The change in the model's hot-low torque after one step of one control alone.";
/// The Auto line with Auto on the magnet length (owner 5A): what Auto does after a change, and
/// what the rows keep.
pub const AUTO_HOLDS_LENGTH: &str =
    "Auto sets the magnet length again after a change; each row keeps the length shown.";
/// The Auto line with Auto on the ring size.
pub const AUTO_HOLDS_RING: &str =
    "Auto sizes the ring again after a change; each row keeps the ring radius shown.";
/// The Auto line with Auto on the magnet count.
pub const AUTO_HOLDS_COUNT: &str =
    "Auto sets the magnet count again after a change; each row keeps the count shown.";
/// A side off its slider's range ([`Words::EndOfRange`]; owner 9A, in the words of the owner's
/// FO-1 at the slice-3 STOP): true wherever the design shown sits, past an end included, and
/// it names no end the control would be stopped at.
pub const END_OF_RANGE: &str = "outside the slider's range";
/// An odd count's sides ([`Words::NotOnTheSlider`]; T-4).
pub const NOT_ON_THE_SLIDER: &str = "not a value the slider takes";
/// A side whose blocks do not fit ([`Words::DoesNotFit`]): the neighbour lines' shipped words,
/// moved here; the Size tab's exploration line is built from it.
pub const SIDE_DOES_NOT_FIT: &str = "magnets do not fit";
/// A side the model has no prediction for ([`Words::NoPrediction`]): the neighbour lines'
/// shipped words, moved here (named apart from `size_controls::NO_PREDICTION`).
pub const SIDE_NO_PREDICTION: &str = "no prediction";
/// A side whose keyway leaves no hub wall past it ([`Words::KeywayBreaksThrough`]; T-4): the
/// shipped words of the Size tab's exploration line, which is built from it.
pub const KEYWAY_BREAKS_THROUGH: &str = "the keyway breaks through the hub";
/// A length side that is not a step of the overlapping magnet length ([`Words::Jump`]; owner
/// 9A, widened by question O-1). It names no cause, and it names the step's amount, so "by the
/// step" never reads as the step's doing (the G-3 review's SF-1), and its direction, so it holds
/// where the overlap moves by the step's size the other way (a cup cavity of 10.7 mm; the
/// fold-in's G-3 re-review, MF-1).
pub const JUMP: &str =
    "not a step: the overlapping magnet length does not change by the step's amount and direction";
/// A diameter side that is not a step of the cup's diameter ([`Words::CupJump`]; the owner's
/// FO-2 at the slice-3 STOP, spec risk R-13): where the shaft boss sets the Outside diameter,
/// above the cup and the cap, one step lifts the cup by more than the step (on the boss
/// plateau from its own size to the boss plus the step). It mirrors [`JUMP`] and names no
/// cause.
pub const CUP_JUMP: &str =
    "not a step: the cup diameter does not change by the step's amount and direction";
/// After a number whose stepped design takes another calibration factor (owner 16A): it names
/// the change, never a cause.
pub const CALIBRATION_CHANGE: &str = "includes a calibration change";
/// The card of a design shown whose magnet count lies below the Magnets per ring slider's range
/// ([`TorqueCard::ShownCountBelowSlider`]; the owner's FO-3 at the slice-3 STOP): a design
/// file's 2 or 3 magnets per ring, or any lower count.
pub const SHOWN_COUNT_BELOW_SLIDER: &str =
    "No step is compared while the magnet count of the design shown is below the slider's range.";
/// The card of a design shown whose magnets do not fit ([`TorqueCard::ShownDoesNotFit`]).
pub const SHOWN_DOES_NOT_FIT: &str =
    "No step is compared while the magnets of the design shown do not fit.";
/// The card of a design shown with no torque prediction ([`TorqueCard::ShownNoPrediction`]: its
/// readout shows `NO_PREDICTION` or `NO_TORQUE`).
pub const SHOWN_NO_PREDICTION: &str =
    "No step is compared while the design shown has no torque prediction.";
/// The card of a design shown whose readouts show a pull-out or hot-low of zero or less
/// ([`TorqueCard::ShownNoHotLow`]): true for either torque, in plain words with no double
/// negative (the G-3 review's SF-2).
pub const SHOWN_NO_HOT_LOW: &str =
    "No step is compared while the design shown has a pull-out or hot-low torque of zero or less.";
/// A flip's words, drawn "{label}: {word}" ([`Flip`]): a phrase inside a sentence, not the
/// dashboard badge's own words (`Level::word`).
pub const FLIP_FAILS: &str = "fails";
pub const FLIP_PASSES: &str = "passes";
pub const FLIP_CAUTION: &str = "caution";

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

    /// The unit of the control's step: millimetres, or none for the magnet count.
    fn unit(self) -> &'static str {
        match self {
            Control::OutsideDiameter | Control::OverallLength | Control::FaceGap => "mm",
            Control::MagnetsPerRing => "",
        }
    }

    /// A side's step as its row writes it (spec 4.1): "-1 mm", "+0.1 mm", "-2".
    fn step_text(self, more: bool) -> String {
        with_unit(format!("{:+}", self.signed_step(more)), self.unit())
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
    /// The overlapping magnet length does not change by the step's amount and direction (the
    /// Overall length row).
    Jump,
    /// The cup's diameter does not change by the step's amount and direction (the Outside
    /// diameter row, where the shaft boss sets the Outside diameter: the owner's FO-2).
    CupJump,
}

impl Words {
    /// The words the side shows.
    pub fn text(self) -> &'static str {
        match self {
            Words::EndOfRange => END_OF_RANGE,
            Words::NotOnTheSlider => NOT_ON_THE_SLIDER,
            Words::DoesNotFit => SIDE_DOES_NOT_FIT,
            Words::KeywayBreaksThrough => KEYWAY_BREAKS_THROUGH,
            Words::NoPrediction => SIDE_NO_PREDICTION,
            Words::Jump => JUMP,
            Words::CupJump => CUP_JUMP,
        }
    }
}

/// A dashboard check whose level differs on the stepped design (owner 8A).
#[derive(Clone, Debug, PartialEq)]
pub struct Flip {
    /// The check's guide label.
    pub check: &'static str,
    /// Its level on the stepped design.
    pub to: Level,
}

impl Flip {
    /// "{label}: fails", "passes" or "caution" (owner 8A).
    fn text(&self) -> String {
        let word = match self.to {
            Level::Bad => FLIP_FAILS,
            Level::Good => FLIP_PASSES,
            Level::Caution => FLIP_CAUTION,
        };
        format!("{}: {word}", self.check)
    }
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

/// "{label} {change} {unit}" (spec 3.6): a side's size change or a failing check's quantity, the
/// change at its own decimals from `from` ([`signed_change`]) in its `unit` ([`with_unit`]).
fn labelled_change(label: &str, change: f64, from: usize, unit: &str) -> String {
    format!("{label} {}", with_unit(signed_change(change, from), unit))
}

impl MarginChange {
    /// "{label} {change} {unit}", or "{label}: {name} {change} {unit}" where it has a name
    /// (spec 3.6): [`labelled_change`] at the change's own decimals.
    fn text(&self) -> String {
        let label = match self.name {
            None => self.check.to_owned(),
            Some(name) => format!("{}: {name}", self.check),
        };
        labelled_change(&label, self.change, self.decimals, self.unit)
    }
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

impl Change {
    /// What the step costs, in the order of spec 3.6: the calibration note, the size changes
    /// (Outside diameter, then Overall length: never the row's own `control`, and only where
    /// one shows, [`shows`]), the flips, then the failing checks' quantities.
    fn costs(&self, control: Control) -> Vec<String> {
        let mut costs = Vec::new();
        if self.calibration_changed {
            costs.push(CALIBRATION_CHANGE.to_owned());
        }
        for (size, change) in [
            (Control::OutsideDiameter, self.diameter_change_mm),
            (Control::OverallLength, self.length_change_mm),
        ] {
            if let Some(change) =
                change.filter(|&change| size != control && shows(change, SIZE_DECIMALS))
            {
                costs.push(labelled_change(
                    size.label(),
                    change,
                    SIZE_DECIMALS,
                    size.unit(),
                ));
            }
        }
        costs.extend(self.flips.iter().map(Flip::text));
        costs.extend(self.margins.iter().map(MarginChange::text));
        costs
    }
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

impl Row {
    /// The precision of the row's two N·m (T-14: one for both, [`shared_torque_precision`] of
    /// its number sides beside their percents): the larger of their own decimals, or four
    /// significant digits for both where either reads zero at the cap beside a percent that
    /// reads a digit (the owner's FO-4; the fold-in's G-3 re-review, MF-2). A row of words draws
    /// none.
    fn torque_precision(&self) -> TorquePrecision {
        let changes: Vec<(f64, Option<f64>)> = [&self.fewer, &self.more]
            .into_iter()
            .filter_map(|side| match side {
                Side::Change(change) => Some((change.change_Nm, Some(change.fraction))),
                Side::Words(_) => None,
            })
            .collect();
        shared_torque_precision(&changes)
    }

    /// "{label}: {fewer}; {more}" (spec 4.1), the two sides' N·m at the row's one precision.
    fn line(&self) -> String {
        let precision = self.torque_precision();
        format!(
            "{}: {}; {}",
            self.control.label(),
            self.side_text(&self.fewer, false, precision),
            self.side_text(&self.more, true, precision)
        )
    }

    /// One side (spec 4.1): "{step} -> {words}", or "{step} -> {N·m} N·m ({percent})" with the
    /// N·m at the row's `precision` ([`signed_torque_change`]: past the owner's FO-4 bound, four
    /// significant digits), then ", " and each cost.
    fn side_text(&self, side: &Side, more: bool, precision: TorquePrecision) -> String {
        let step = self.control.step_text(more);
        match side {
            Side::Words(words) => format!("{step} -> {}", words.text()),
            Side::Change(change) => {
                let torque = with_unit(signed_torque_change(change.change_Nm, precision), "N·m");
                let number = format!("{step} -> {torque} ({})", signed_percent(change.fraction));
                std::iter::once(number)
                    .chain(change.costs(self.control))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
    }
}

/// The card of a design. `ShownCountBelowSlider`: the design shown has fewer magnets per ring
/// than the slider takes (a design file's 2 or 3, or any lower count: the owner's FO-3 at the
/// slice-3 STOP; at 2 the polygon of blocks degenerates). `ShownNoHotLow`: the model predicts
/// (the readouts show both torques), but hot-low, the percent's base, is not above zero (a
/// design file's variation of 1 or more; physics review 2 MF-1), or the pull-out is not above
/// zero (a magnet Br or calibration coefficient of 0 or less; physics review 3 MF-2).
#[derive(Clone, Debug, PartialEq)]
pub enum TorqueCard {
    ShownCountBelowSlider,
    ShownDoesNotFit,
    ShownNoPrediction,
    ShownNoHotLow,
    Rows(Vec<Row>),
}

impl TorqueCard {
    /// The lines under the heading, as drawn and as the screen dump prints them. For the rows:
    /// [`CAPTION`], the Auto line while Auto is on (`auto` is its variable), then one line per
    /// row, the row Auto holds reading [`held_line`] (owner 5A; spec 3.4: the marking is drawn
    /// here, outside the card the cache keeps). For the four one-line forms, their line alone,
    /// with no caption and no Auto line (T-12).
    pub fn lines(&self, auto: Option<FreeVariable>) -> Vec<String> {
        match self {
            TorqueCard::ShownCountBelowSlider => vec![SHOWN_COUNT_BELOW_SLIDER.to_owned()],
            TorqueCard::ShownDoesNotFit => vec![SHOWN_DOES_NOT_FIT.to_owned()],
            TorqueCard::ShownNoPrediction => vec![SHOWN_NO_PREDICTION.to_owned()],
            TorqueCard::ShownNoHotLow => vec![SHOWN_NO_HOT_LOW.to_owned()],
            TorqueCard::Rows(rows) => {
                let mut lines = vec![CAPTION.to_owned()];
                lines.extend(auto.map(|variable| auto_line(variable).to_owned()));
                lines.extend(rows.iter().map(|row| {
                    if auto.is_some() && row.control.held_by() == auto {
                        held_line(row.control)
                    } else {
                        row.line()
                    }
                }));
                lines
            }
        }
    }

    /// The rows: empty for the four one-line forms.
    pub fn rows(&self) -> &[Row] {
        match self {
            TorqueCard::Rows(rows) => rows,
            TorqueCard::ShownCountBelowSlider
            | TorqueCard::ShownDoesNotFit
            | TorqueCard::ShownNoPrediction
            | TorqueCard::ShownNoHotLow => &[],
        }
    }
}

/// The held row's line, "{label}: {SET_BY_AUTO}" (owner 5A): [`TorqueCard::lines`] draws it,
/// and `app.rs`'s smoke test asserts it in the gui-smoke prompt (`size_controls` is a private
/// module, so `app.rs` reaches the words only here).
pub fn held_line(control: Control) -> String {
    format!("{}: {SET_BY_AUTO}", control.label())
}

/// The Auto line while Auto is on `variable` (owner 5A).
fn auto_line(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => AUTO_HOLDS_LENGTH,
        FreeVariable::RingRadius => AUTO_HOLDS_RING,
        FreeVariable::MagnetsPerRing => AUTO_HOLDS_COUNT,
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
    clamps_paused: bool,
) -> TorqueCard {
    // Test 0 (spec 3.3): the design shown. Its magnet count below the slider's range first (the
    // owner's FO-3 at the slice-3 STOP), then the readouts' own rule.
    if count_below_the_slider(inputs.coupling.npole) {
        return TorqueCard::ShownCountBelowSlider;
    }
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
        clamps_paused,
    };
    TorqueCard::Rows(
        Control::ALL
            .into_iter()
            .filter_map(|control| {
                // Test 1: a size row only where the Size tab draws its slider (T-5).
                let slider = Slider::of(control, &shown, minimum_diameter_mm)?;
                Some(Row {
                    control,
                    fewer: shown.side(&slider, false),
                    more: shown.side(&slider, true),
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
fn side_words(design: &DesignInputs, results: &DesignResults, keyway: bool) -> Option<Words> {
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
/// of its range (one block per pole; the count input's [`slider_range`]): a step past the
/// range, or one that overflows (a design file's count), reads [`Words::EndOfRange`], as would
/// every step of a count input without a range; an odd count inside the range, which a design
/// file can hold, [`Words::NotOnTheSlider`].
fn count_step(npole: i64, more: bool) -> Result<i64, Words> {
    let range = slider_range(FreeVariable::MagnetsPerRing.path()).ok_or(Words::EndOfRange)?;
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

/// Whether a magnet count `npole` lies below the range of the Magnets per ring slider, the one
/// [`count_step`] reads: a design file's 2 or 3, or any lower count (the owner's FO-3 at the
/// slice-3 STOP). The card of such a design shown draws one line, checked before the magnets'
/// fit (at 0, 1 or fewer the count is the more exact reason). A count above the range is left
/// to the fit, as before.
fn count_below_the_slider(npole: i64) -> bool {
    slider_range(FreeVariable::MagnetsPerRing.path())
        .is_some_and(|range| (npole as f64) < range.min)
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

/// Whether a size or margin change is drawn (spec 3.6): it is finite and reads a digit that is
/// not zero within the cap of section 5's format, `from` + `CHANGE_EXTRA_DECIMALS` decimals
/// (`gui::format`'s cap rule, [`decimals_with_a_digit`], which the change's own decimals read
/// too; by rounding, the same as a digit at the cap itself). So float noise (3.6e-15 mm) is
/// never drawn as "+0.0 mm", nor is a NaN or an infinite change (a key with no bearing area,
/// +inf against +inf).
fn shows(change: f64, from: usize) -> bool {
    decimals_with_a_digit(change, from).is_some()
}

/// The design shown, its results and its size controls ([`dimensions`]: `None` where the Size
/// tab draws no size slider): what each side is compared with, over the guides that count at
/// the shaft-clamp pause `clamps_paused`.
struct Shown<'a> {
    inputs: &'a DesignInputs,
    results: &'a DesignResults,
    size: Option<SizeControls>,
    clamps_paused: bool,
}

/// How a control's own slider maps a value to a design, with the range it takes (spec 3.3): the
/// range here, the mapping the controls' own ([`SliderMapping`]).
enum Slider {
    /// The Outside diameter slider at the user's Minimum OD: its range there, and its mapping at
    /// `minimum` ([`slider_minimum`]).
    Diameter { size: SizeControls, minimum: f64 },
    /// The Overall length slider: its mapping over the size controls' length range.
    Length { size: SizeControls },
    /// The Magnet face gap slider: its input set inside its input's range ([`slider_range`]),
    /// each side reading [`Words::EndOfRange`] where the input has none, as the count's do.
    FaceGap { range: Option<SliderRange> },
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
                range: slider_range(FACE_GAP),
            }),
            Control::MagnetsPerRing => Some(Slider::MagnetsPerRing),
        }
    }

    /// The control whose slider this is, the one [`Slider::of`] made it for: the step and the
    /// row's rules read it, so a slider cannot be stepped as another control.
    fn control(&self) -> Control {
        match self {
            Slider::Diameter { .. } => Control::OutsideDiameter,
            Slider::Length { .. } => Control::OverallLength,
            Slider::FaceGap { .. } => Control::FaceGap,
            Slider::MagnetsPerRing => Control::MagnetsPerRing,
        }
    }

    /// The design one step of this slider's control from `inputs` makes, `more` its `+` side;
    /// else why the slider does not take that step (spec 3.3 steps 2 and 3): a value more than
    /// [`STEP_SLACK_MM`] past an end of its range, one the mapping refuses, or a count
    /// [`count_step`] does not take.
    fn step(&self, inputs: &DesignInputs, more: bool) -> Result<DesignInputs, Words> {
        let step = self.control().signed_step(more);
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
                let range = range.ok_or(Words::EndOfRange)?;
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
    /// The side of the row of `slider`'s control one step of the slider away, `more` the `+`
    /// side, by the order of tests (spec 3.3 steps 2-8).
    fn side(&self, slider: &Slider, more: bool) -> Side {
        let control = slider.control();
        let design = match slider.step(self.inputs, more) {
            Ok(design) => design,
            Err(words) => return Side::Words(words),
        };
        let results = compute_all(&design);
        // Steps 4-6: sizing's validity rule, the keyway left out on the count row (D-1).
        let keyway = control != Control::MagnetsPerRing;
        if let Some(words) = side_words(&design, &results, keyway) {
            return Side::Words(words);
        }
        // Step 7: a size step that is not a step of what it moves.
        if let Some(words) = self.jump(control, &results, more) {
            return Side::Words(words);
        }
        Side::Change(Box::new(self.change(&design, &results)))
    }

    /// The jump rules (spec 3.3 step 7): the words of a size row's side whose step does not
    /// move what it moves by exactly the step, within [`STEP_SLACK_MM`]; `None` where it does,
    /// and on the gap and count rows, which have no jump rule. `stepped` is the stepped design's
    /// results, `more` the `+` side. A change by anything else, more or less or the other way,
    /// is a jump, and so is a quantity that is not a number:
    /// - the Overall length row moves the overlapping magnet length (`model.active_length_mm`),
    ///   else [`Words::Jump`] (owner 9A, widened by question O-1);
    /// - the Outside diameter row moves the cup's diameter (`model.cup_od_mm`), else
    ///   [`Words::CupJump`] (the owner's FO-2 at the slice-3 STOP): the slider asks for the cup
    ///   at the request less the cap's overhang, so where the shaft boss sets the Outside
    ///   diameter, above the cup and the cap, one step lifts the cup by more than the step.
    fn jump(&self, control: Control, stepped: &DesignResults, more: bool) -> Option<Words> {
        let (moved_by_the_row, words): (fn(&DesignResults) -> f64, Words) = match control {
            Control::OverallLength => (|results| results.model.active_length_mm, Words::Jump),
            Control::OutsideDiameter => (|results| results.model.cup_od_mm, Words::CupJump),
            Control::FaceGap | Control::MagnetsPerRing => return None,
        };
        let moved = moved_by_the_row(stepped) - moved_by_the_row(self.results);
        // A NaN fails the comparison, so it reads the words too.
        let by_the_step = (moved - control.signed_step(more)).abs() <= STEP_SLACK_MM;
        (!by_the_step).then_some(words)
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
        // Each counted check whose level differs, and of each check that fails on both designs
        // the quantities of its failing terms (spec 3.6).
        let mut flips = Vec::new();
        let mut margins = Vec::new();
        for guide in counted_guides(self.clamps_paused) {
            let before = level(guide, self.inputs, self.results);
            let after = level(guide, design, stepped);
            if before != after {
                flips.push(Flip {
                    check: guide.label,
                    to: after,
                });
            }
            if before == Level::Bad && after == Level::Bad {
                margins.extend(self.quoted_changes(guide, design, stepped));
            }
        }
        Change {
            hot_low_Nm,
            change_Nm,
            fraction: change_Nm / shown_hot_low_Nm,
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
            diameter_change_mm,
            length_change_mm,
            flips,
            margins,
        }
    }

    /// The quoted quantities of `guide`'s check, a check that fails on the design shown and on
    /// the stepped `design`, whose results are `stepped`, each under the guide's label: of the
    /// terms of [`check_terms`], paired term by term, each that fails on either design and quotes
    /// a quantity, where its change shows ([`shows`]), in the result's own unit. "Either"
    /// catches a step that puts a second term of a failing check over its limit, which no flip
    /// shows.
    fn quoted_changes(
        &self,
        guide: &CheckGuide,
        design: &DesignInputs,
        stepped: &DesignResults,
    ) -> Vec<MarginChange> {
        check_terms(guide.check, self.inputs, self.results)
            .into_iter()
            .zip(check_terms(guide.check, design, stepped))
            .filter(|(before, after)| !(before.passes && after.passes))
            .filter_map(|(before, _)| before.quoted)
            .filter_map(|quoted| {
                let change =
                    result_number(stepped, quoted.path) - result_number(self.results, quoted.path);
                shows(change, quoted.decimals).then(|| MarginChange {
                    check: guide.label,
                    name: quoted.name,
                    change,
                    unit: result_info(quoted.path).map_or("", |info| info.meta.unit),
                    decimals: quoted.decimals,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::explain::plain::{
        TORQUE_ORDER_WORDS, mentions_gearbox_protection, plain_text_violations, ranking_words_in,
    };
    use crate::engine::size_controls::{resize_diameter_with_minimum, resize_length};
    use crate::engine::sizing::is_valid;
    use crate::gui::format::{
        TORQUE_DECIMALS, fixed, last_digit_step, reads_not_zero, signed_fixed,
    };
    use crate::gui::guidance::{
        CLAMP, CLEARANCE, COLD_HIGH, CUP_WALL, Consequences, GEARBOX, HOT_MIN, Quoted,
        RING_SIZE_STAYS, SHAFT_JOINT, SPACE_CLAIM, SPACE_CLAIM_AXIS_SIZES, TEMPERATURE,
        WALL_NEEDED, guide,
    };
    use crate::gui::test_support::{
        CardDesign, QUOTED_UNITS, assert_glyphs, auto_states, card_at, card_designs,
        clamp_at_its_limit, end_effect_at_a_step, glyph_context, keyway_through, no_back_iron,
        only_the_clamp_fails, row_line, short_magnets, side_at, slider_range, with_cup_depth,
        with_magnets,
    };

    /// The number sides of the cards of section 2's designs ([`card_designs`]): the tests that
    /// read every one count them, so none can pass by reading none.
    const NUMBER_SIDES: usize = 276;

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

    /// A number side of a card of the table ([`card_designs`]) with the two designs it compares:
    /// the design shown and its results, and the stepped design rebuilt here through the
    /// slider's own mapping ([`slider_step`]) and its results.
    struct NumberSide<'a> {
        /// "{key} {control:?} {more}", for a failure's message.
        what: String,
        design: &'a CardDesign,
        change: &'a Change,
        before: &'a DesignResults,
        stepped: DesignInputs,
        after: DesignResults,
    }

    /// Calls `check` on every number side of the cards of [`card_designs`], each design's card
    /// asked as the panel asks for it, in the table's order; returns how many it read, so a
    /// test can pin [`NUMBER_SIDES`].
    fn for_each_number_side(mut check: impl FnMut(&NumberSide)) -> usize {
        let mut numbers = 0;
        for design in &card_designs() {
            let before = compute_all(&design.inputs);
            let card = card(&design.inputs, &before, design.minimum_diameter_mm, PAUSED);
            for row in card.rows() {
                for (more, side) in sides(row) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let stepped = slider_step(
                        row.control,
                        &design.inputs,
                        design.minimum_diameter_mm,
                        more,
                    );
                    let after = compute_all(&stepped);
                    check(&NumberSide {
                        what: format!("{} {:?} {more}", design.key, row.control),
                        design,
                        change,
                        before: &before,
                        stepped,
                        after,
                    });
                    numbers += 1;
                }
            }
        }
        numbers
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
    fn each_control_s_slider_is_stepped_as_that_control() {
        // A row's slider names its own control, the one it was built for, so its step and its
        // row's rules (the keyway on the count row, the jumps) read that control and never
        // another's (the T3 review's M-8 (b)). Every control at every design of the table:
        // where the Size tab draws a slider, it is that control's. The gap's and the count's
        // are drawn at every design (spec 3.3 step 1), and each control's slider is checked at
        // some design, so the test cannot pass by checking none.
        let mut checked = [0; Control::ALL.len()];
        for design in card_designs() {
            let results = compute_all(&design.inputs);
            let shown = Shown {
                inputs: &design.inputs,
                results: &results,
                size: dimensions(&design.inputs),
                clamps_paused: PAUSED,
            };
            for (index, control) in Control::ALL.into_iter().enumerate() {
                let slider = Slider::of(control, &shown, design.minimum_diameter_mm);
                if matches!(control, Control::FaceGap | Control::MagnetsPerRing) {
                    assert!(slider.is_some(), "{}: {control:?}", design.key);
                }
                if let Some(slider) = slider {
                    assert_eq!(slider.control(), control, "{}", design.key);
                    checked[index] += 1;
                }
            }
        }
        assert!(checked.iter().all(|&sliders| sliders > 0), "{checked:?}");
    }

    #[test]
    fn each_number_is_the_difference_of_two_compute_all_runs() {
        // Spec 8.1 test 1 (owner 1A): every number side, at every design of section 2, is the
        // stepped design's compute_all less the design shown's, bit for bit, the stepped design
        // rebuilt here through the slider's own mapping.
        let numbers = for_each_number_side(|side| {
            let NumberSide {
                what,
                design,
                change,
                before,
                stepped,
                after,
            } = side;
            let hot_low = after.metal.torque_hot_low_Nm;
            let change_nm = hot_low - before.metal.torque_hot_low_Nm;
            assert_eq!(change.hot_low_Nm.to_bits(), hot_low.to_bits(), "{what}");
            assert_eq!(change.change_Nm.to_bits(), change_nm.to_bits(), "{what}");
            assert_eq!(
                change.fraction.to_bits(),
                (change_nm / before.metal.torque_hot_low_Nm).to_bits(),
                "{what}"
            );
            // The sizes the sliders would show; none exactly where either design has no size
            // controls.
            let sizes = match (dimensions(&design.inputs), dimensions(stepped)) {
                (Some(shown), Some(stepped)) => [
                    Some(stepped.diameter_mm - shown.diameter_mm),
                    Some(stepped.overall_length_mm - shown.overall_length_mm),
                ],
                _ => [None, None],
            };
            assert_eq!(
                [change.diameter_change_mm, change.length_change_mm].map(|mm| mm.map(f64::to_bits)),
                sizes.map(|mm| mm.map(f64::to_bits)),
                "{what}"
            );
            assert_eq!(
                change.calibration_changed,
                after.model.f_cal != before.model.f_cal,
                "{what}"
            );
        });
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
        // A count from a design file the slider does not take: odd, or overflowing (no row is
        // reached: above the slider the design shown does not fit, and below it the count
        // decides first, the owner's FO-3; `count_step` itself is pinned by
        // `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`).
        for more in [false, true] {
            assert_eq!(
                words(&design("odd count").side(MagnetsPerRing, more)),
                Words::NotOnTheSlider
            );
        }
        assert_eq!(
            design("count overflow: i64::MAX").card(),
            TorqueCard::ShownDoesNotFit
        );
        assert_eq!(
            design("count overflow: i64::MIN").card(),
            TorqueCard::ShownCountBelowSlider
        );

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
        // ... and a stepped torque that is not finite with the end effect in range: no back
        // iron at 8 magnets with a design file's calibration coefficient of 1e-320, whose +2
        // design (10 magnets) takes the measured factor, which overflows.
        let mut tiny = no_back_iron();
        tiny.coupling.npole = 8;
        tiny.calibration.f_cal_original = 1e-320;
        let ten = compute_all(&with_magnets(&tiny, 10));
        assert!(crate::gui::dashboard::end_effect_out_of_range(&ten).is_none());
        assert!(
            !ten.metal.torque_hot_low_Nm.is_finite(),
            "{}",
            ten.metal.torque_hot_low_Nm
        );
        assert_eq!(
            words(&side_at(&tiny, None, MagnetsPerRing, true)),
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
        assert_eq!(
            words(&design("short cavity").side(OverallLength, false)),
            Words::Jump
        );
        for mm in [11.0, 12.0, 12.5, 10.7] {
            assert_eq!(
                words(&side_at(&with_cup_depth(mm), None, OverallLength, true)),
                Words::Jump,
                "{mm}"
            );
        }
        // At 11.7 mm, a value on the cavity's slider, the +1 mm side lengthens the coupling by
        // the step and leaves the overlap where it was, 12.70 mm bit for bit: a jump whose
        // words must hold where the overlap does not move at all (G-3 round 2, SF-1).
        let still = with_cup_depth(11.7);
        assert_eq!(
            words(&side_at(&still, None, OverallLength, true)),
            Words::Jump
        );
        let stepped = slider_step(OverallLength, &still, None, true);
        let overall =
            |inputs: &DesignInputs| dimensions(inputs).expect("size controls").overall_length_mm;
        assert!((overall(&stepped) - overall(&still) - 1.0).abs() <= STEP_SLACK_MM);
        let overlap = compute_all(&still).model.active_length_mm;
        assert_eq!(fixed(overlap, 2), "12.70");
        assert_eq!(
            compute_all(&stepped).model.active_length_mm.to_bits(),
            overlap.to_bits()
        );
        // At 10.7 mm, also on the cavity's slider, the +1 mm side (a jump, above) moves the
        // overlap by the step's size the other way, 12.70 -> 11.70 mm: so the jump words name the
        // step's direction as well as its amount (the fold-in's G-3 re-review, MF-1).
        let reversed = with_cup_depth(10.7);
        let moved = compute_all(&slider_step(OverallLength, &reversed, None, true))
            .model
            .active_length_mm
            - compute_all(&reversed).model.active_length_mm;
        assert!((moved + 1.0).abs() <= STEP_SLACK_MM, "{moved}");
        // ... and the diameter row's jump (the owner's FO-2): where the shaft boss sets the
        // Outside diameter, one step up lifts the cup by more than the step, and the step down
        // asks for less than the boss, below the slider's range.
        let plateau = design("R-13 boss plateau");
        assert_eq!(words(&plateau.side(OutsideDiameter, true)), Words::CupJump);
        assert_eq!(
            words(&plateau.side(OutsideDiameter, false)),
            Words::EndOfRange
        );
        for mm in [12.7, 15.5] {
            for more in [false, true] {
                assert!(
                    matches!(
                        side_at(&with_cup_depth(mm), None, OverallLength, more),
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
        let stepped = compute_all(&with_magnets(&negative.inputs, 8));
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
        // It is drawn as every number side, as spec 8.1 test 3 writes it: its N·m with its true
        // sign at the row's precision, and -100.0 %, the hot-low gone (physics review 5 SF-2).
        assert_eq!(more.fraction, -1.0);
        let card = card_at(&unmeasured, None);
        let row = &card.rows()[3];
        assert!(
            side_texts(&card.lines(None)[4], row)[1]
                .starts_with(&format!("+2 -> -0.85 N·m (-100.0 %), {CALIBRATION_CHANGE}")),
            "{:?}",
            card.lines(None)
        );

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
        // Spec 8.1 test 4: the design shown decides before any row: its magnet count below the
        // magnet slider's range first (the owner's FO-3 at the slice-3 STOP), then the readouts'
        // own rule (physics review 3 MF-2).
        use crate::gui::test_support::with_poles;
        let designs = card_designs();
        let design = |key| find(&designs, key).inputs.clone();
        let mut negative_coefficient = DesignInputs::default();
        negative_coefficient.calibration.f_cal_original = -0.95;
        let mut negative_pull_out = negative_coefficient.clone();
        negative_pull_out.metal.variation = 1.5;
        // A torque that is not finite with the end effect in range (spec 2.1: (c) with a design
        // file's calibration coefficient of 0, whose measured factor is NaN): the readouts show
        // NO_TORQUE, so the card reads no prediction, never SHOWN_NO_HOT_LOW.
        let mut zero_coefficient = no_back_iron();
        zero_coefficient.calibration.f_cal_original = 0.0;
        let fewest = slider_range(FreeVariable::MagnetsPerRing.path()).min;
        for (name, inputs, shown) in [
            // A design file's count below the magnet slider's range (the owner's FO-3): at 3 and
            // 2 the blocks fit and the model predicts, so the count alone keeps the card off its
            // rows (at 2 blocks the polygon degenerates: the cup reads 5.86e17 mm); at 1, 0 and
            // -2 the blocks do not fit, and the count, checked first, is the more exact reason.
            ("npole 3", with_poles(3), TorqueCard::ShownCountBelowSlider),
            ("npole 2", with_poles(2), TorqueCard::ShownCountBelowSlider),
            ("npole 1", with_poles(1), TorqueCard::ShownCountBelowSlider),
            ("npole 0", with_poles(0), TorqueCard::ShownCountBelowSlider),
            (
                "npole -2",
                with_poles(-2),
                TorqueCard::ShownCountBelowSlider,
            ),
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
            (
                "(c), f_cal_original 0",
                zero_coefficient,
                TorqueCard::ShownNoPrediction,
            ),
        ] {
            let results = compute_all(&inputs);
            let torques = [results.model.pullout_Nm, results.metal.torque_hot_low_Nm];
            // What the readouts show (`size_controls::predicts`), and the torques' signs.
            let shows_both = torques.iter().all(|&torque| predicts(&results, torque));
            let both_above_zero = torques.iter().all(|&torque| torque > 0.0);
            let fits = blocks_fit(&inputs.coupling, &results.model);
            // The count decides first: exactly the counts below the slider read its line.
            let below = (inputs.coupling.npole as f64) < fewest;
            assert_eq!(below, shown == TorqueCard::ShownCountBelowSlider, "{name}");
            match shown {
                TorqueCard::ShownCountBelowSlider => {
                    // Without the count's rule 3 and 2 would draw rows, 1, 0 and -2 the fit's
                    // line.
                    let rows = fits && shows_both && both_above_zero;
                    assert_eq!(rows, ["npole 3", "npole 2"].contains(&name), "{name}");
                    if name == "npole 2" {
                        assert!(
                            results.model.cup_od_mm > 1e17,
                            "{}",
                            results.model.cup_od_mm
                        );
                    }
                }
                TorqueCard::ShownNoPrediction => assert!(!shows_both, "{name}"),
                // Its line's claim: both torques shown, and one of them zero or less.
                TorqueCard::ShownNoHotLow => assert!(
                    shows_both && torques.iter().any(|&torque| torque <= 0.0),
                    "{name}: {torques:?}"
                ),
                _ => assert!(!fits, "{name}"),
            }
            if name == "f_cal_original -0.95, variation 1.5" {
                assert!(torques[0] < 0.0 && torques[1] > 0.0, "{torques:?}");
            }
            if name == "(c), f_cal_original 0" {
                // The finiteness half of the readouts' rule decides, not the end effect.
                assert!(
                    crate::gui::dashboard::end_effect_out_of_range(&results).is_none(),
                    "{name}"
                );
                assert!(!torques[1].is_finite(), "{name}: {torques:?}");
            }
            let card = card(&inputs, &results, None, PAUSED);
            assert_eq!(card, shown, "{name}");
            assert!(card.rows().is_empty(), "{name}");
            // Its one line alone: no caption, no Auto line, whatever the Auto state (T-12).
            let line = match card {
                TorqueCard::ShownCountBelowSlider => SHOWN_COUNT_BELOW_SLIDER,
                TorqueCard::ShownDoesNotFit => SHOWN_DOES_NOT_FIT,
                TorqueCard::ShownNoPrediction => SHOWN_NO_PREDICTION,
                TorqueCard::ShownNoHotLow => SHOWN_NO_HOT_LOW,
                TorqueCard::Rows(_) => unreachable!("{name}"),
            };
            for auto in auto_states() {
                assert_eq!(card.lines(auto), [line], "{name} {auto:?}");
            }
        }
        // The hot-low line says it plainly, with no double negative ("not above zero"): the
        // same truth in every case routed to it (the G-3 review's SF-2).
        assert!(
            !SHOWN_NO_HOT_LOW.contains("not above"),
            "{SHOWN_NO_HOT_LOW}"
        );
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
        // and a side reads END_OF_RANGE exactly when its step lands more than STEP_SLACK_MM
        // outside that range.
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
        assert_eq!(rows, 44);
    }

    #[test]
    fn a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range() {
        // A design file or link can carry a value past a slider's end (`validate()` checks only
        // that a number is finite). A side more than STEP_SLACK_MM outside the slider's range
        // reads END_OF_RANGE (spec 3.3 step 2): on both sides where the design shown itself
        // lies past an end; a side that lands back on the slider is a number, the slider's own
        // design.
        use crate::gui::test_support::forty_narrow_magnets;
        use Control::{FaceGap, MagnetsPerRing, OutsideDiameter, OverallLength};
        let with = |edit: fn(&mut DesignInputs)| {
            let mut inputs = DesignInputs::default();
            edit(&mut inputs);
            inputs
        };
        // Review Focus item 3 at each slider: a gap of 5.5 or 6.0 mm and an axial override of
        // 60 mm; a ring radius of 31.0 mm, past its slider's 30.0 mm, whose Outside diameter lies
        // above its range's top; an axial override of 0.5 mm, under its slider's 2.0 mm (with no
        // end effect, c_end 0, so the model predicts there), whose Overall length lies below its
        // range's bottom; and 44 narrow magnets, which fit, past the magnet slider's 40.
        let gap_above = with(|i| i.metal.face_gap_mm = 5.5);
        let gap_far_above = with(|i| i.metal.face_gap_mm = 6.0);
        let gap_below = with(|i| i.metal.face_gap_mm = 0.2);
        let long = with(|i| i.coupling.magnets.axial_length_mm = Some(60.0));
        let wide = with(|i| i.coupling.inner_back_apothem_mm = 31.0);
        let short = with(|i| {
            i.coupling.c_end = 0.0;
            i.coupling.magnets.axial_length_mm = Some(0.5);
        });
        let with_narrow_magnets = |npole: i64| {
            let mut inputs = forty_narrow_magnets();
            inputs.coupling.npole = npole;
            inputs
        };
        let many = with_narrow_magnets(44);
        // Every side of the six rows, then every one that is not END_OF_RANGE, so a failure
        // names each case it reaches.
        let mut not_at_the_end = Vec::new();
        for (key, inputs, control) in [
            ("gap 5.5", &gap_above, FaceGap),
            ("gap 6.0", &gap_far_above, FaceGap),
            ("override 60", &long, OverallLength),
            ("ring 31.0", &wide, OutsideDiameter),
            ("override 0.5, c_end 0", &short, OverallLength),
            ("44 narrow magnets", &many, MagnetsPerRing),
        ] {
            for more in [false, true] {
                let side = side_at(inputs, None, control, more);
                if side != Side::Words(Words::EndOfRange) {
                    not_at_the_end.push((key, more, side));
                }
            }
        }
        assert!(not_at_the_end.is_empty(), "{not_at_the_end:?}");
        // Each lies past its end by more than a step, so neither side lands back on the slider.
        let size = |inputs: &DesignInputs| dimensions(inputs).expect("size controls");
        let wide_size = size(&wide);
        assert!(
            wide_size.diameter_mm - 1.0 > wide_size.diameter_range.max + STEP_SLACK_MM,
            "{wide_size:?}"
        );
        let short_size = size(&short);
        assert!(
            short_size.overall_length_mm + 1.0 < short_size.length_range.min - STEP_SLACK_MM,
            "{short_size:?}"
        );
        let most = slider_range(FreeVariable::MagnetsPerRing.path()).max as i64;
        assert!(many.coupling.npole - 2 > most);
        assert!(blocks_fit(&many.coupling, &compute_all(&many).model));
        // One step past the magnet slider's top, the -2 side lands back on it: the number of
        // forty narrow magnets, bit for bit; the +2 side does not.
        let forty_two = with_narrow_magnets(most + 2);
        assert_eq!(
            number(&side_at(&forty_two, None, MagnetsPerRing, false))
                .hot_low_Nm
                .to_bits(),
            compute_all(&forty_narrow_magnets())
                .metal
                .torque_hot_low_Nm
                .to_bits()
        );
        assert_eq!(
            words(&side_at(&forty_two, None, MagnetsPerRing, true)),
            Words::EndOfRange
        );
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
        // At (a) with the Minimum OD at 45.0 mm the slider's range starts above the diameter
        // shown, 42.80 mm: the +1 mm side asks for 43.80 mm, below that range, while the handle
        // sits at its bottom and moves up from 45.00 mm. So the words name the slider's range and
        // no end of it: "at the end of its range" named an end the control is not stopped at
        // (the owner's FO-1 at the slice-3 STOP; G-3 decision D5).
        let a = DesignInputs::default();
        let shown = dimensions(&a).expect("size controls");
        let range = dimensions_with_minimum_diameter(&a, slider_minimum(&shown, Some(45.0)))
            .expect("a diameter range")
            .diameter_range;
        assert_eq!(
            [shown.diameter_mm + 1.0, range.min].map(|mm| fixed(mm, 2)),
            ["43.80", "45.00"]
        );
        assert!(shown.diameter_mm + 1.0 < range.min - STEP_SLACK_MM);
        let card = card_at(&a, Some(45.0));
        let row = &card.rows()[0];
        assert_eq!(row.control, OutsideDiameter);
        assert_eq!(
            side_texts(&card.lines(None)[1], row)[1],
            format!("+1 mm -> {END_OF_RANGE}")
        );
        assert!(
            !END_OF_RANGE.split_whitespace().any(|word| word == "end"),
            "{END_OF_RANGE}"
        );
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
        // Never overflowing: checked arithmetic reads `EndOfRange`.
        for more in [false, true] {
            assert_eq!(count_step(most + 4, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MAX, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MIN, more), Err(Words::EndOfRange));
        }
    }

    #[test]
    fn a_side_reads_a_number_exactly_when_sizing_counts_its_design() {
        // Spec 3.3 steps 4-6: `side_words` is sizing's validity rule (`is_valid`) in its order,
        // one rule for every row, at every even count of the magnet slider over the defaults,
        // magnets too short for the model, the end effect at a step (`end_effect_at_a_step()`)
        // and the keyway through the hub, broken and intact. Modelled on the neighbour lines'
        // test of gui/guidance.rs, which moved here with the lines as this test's last part, the
        // card's count row, so the one rule is checked in one place (it was
        // with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it).
        let range = FreeVariable::MagnetsPerRing.range();
        let counts = (range.min as i64..=range.max as i64).step_by(2);
        let mut intact = keyway_through();
        intact.coupling.keyway_depth_mm = 0.0;
        let at = |inputs: &DesignInputs, npole: i64| {
            let design = with_magnets(inputs, npole);
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
        for npole in counts.clone() {
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
        // Through the card's count row, with wall past the keyway: over the slider's even counts
        // at the defaults and at the end effect at a step, whose end effect two magnets fewer take
        // out of range, a count side is a number exactly when its design counts for inverse
        // sizing; a step the slider does not take reads END_OF_RANGE.
        let (mut numbers, mut words) = (0, 0);
        for base in [DesignInputs::default(), end_effect_at_a_step()] {
            for npole in counts.clone() {
                let (inputs, _) = at(&base, npole);
                let shown = card_at(&inputs, None);
                // A design shown that does not fit or predicts nothing has no rows (test 0).
                let Some(row) = shown
                    .rows()
                    .iter()
                    .find(|row| row.control == Control::MagnetsPerRing)
                else {
                    continue;
                };
                for (more, side) in sides(row) {
                    let Ok(count) = count_step(npole, more) else {
                        assert_eq!(*side, Side::Words(Words::EndOfRange), "{npole}");
                        continue;
                    };
                    let (stepped, stepped_results) = at(&inputs, count);
                    assert!(
                        stepped_results.model.hub_wall_past_key_mm > 0.0,
                        "{npole} -> {count}"
                    );
                    let number = matches!(side, Side::Change(_));
                    assert_eq!(
                        number,
                        is_valid(&stepped, &stepped_results),
                        "{npole} -> {count}: {side:?}"
                    );
                    if number {
                        numbers += 1;
                    } else {
                        words += 1;
                    }
                }
            }
        }
        // Not vacuous: both outcomes occur.
        assert!(numbers > 0 && words > 0, "{numbers} numbers, {words} words");
    }

    // ------------------------------------------------------------------ what a step costs, drawn

    /// The label of the guide of `check`.
    fn label(check: &str) -> &'static str {
        guide(check).expect("a guide's check").label
    }

    /// The four rows of `design`'s card, in order.
    fn four_rows(design: &CardDesign) -> [Row; 4] {
        <[Row; 4]>::try_from(design.card().rows().to_vec()).expect("four rows")
    }

    /// A number side as section 2.3's tables print it: the change in N·m to 4 decimals and in
    /// percent to 2, its flips, and each quoted change to 4 decimals.
    #[derive(Debug, PartialEq)]
    struct Pinned {
        torque: String,
        percent: String,
        flips: Vec<Flip>,
        quoted: Vec<(&'static str, Option<&'static str>, String)>,
    }

    impl Pinned {
        fn of(side: &Side) -> Pinned {
            let change = number(side);
            Pinned {
                torque: signed_fixed(change.change_Nm, 4),
                percent: signed_fixed(100.0 * change.fraction, 2),
                flips: change.flips,
                quoted: change
                    .margins
                    .iter()
                    .map(|margin| (margin.check, margin.name, signed_fixed(margin.change, 4)))
                    .collect(),
            }
        }
    }

    /// Section 2.3's number side: the N·m and the percent as its tables print them, the flips
    /// (each check by its path, and its level on the stepped design) and the quoted changes
    /// (check, name, change).
    fn pin(
        torque: &str,
        percent: &str,
        flips: &[(&str, Level)],
        quoted: &[(&str, Option<&'static str>, &str)],
    ) -> Pinned {
        Pinned {
            torque: torque.to_owned(),
            percent: percent.to_owned(),
            flips: flips
                .iter()
                .map(|&(check, to)| Flip {
                    check: label(check),
                    to,
                })
                .collect(),
            quoted: quoted
                .iter()
                .map(|&(check, name, change)| (label(check), name, change.to_owned()))
                .collect(),
        }
    }

    #[test]
    fn the_card_reads_the_spec_s_numbers() {
        // Spec 8.1 test 2: section 2.3's values (C), to its decimals: N·m to 4, % to 2, mm to 4.
        let designs = card_designs();
        let rows = |key| four_rows(find(&designs, key));
        let (wall, cold) = (Some(WALL_NEEDED), Some(COLD_HIGH));
        let rotating = Some(SPACE_CLAIM_AXIS_SIZES[0].0);
        let stack = Some(SPACE_CLAIM_AXIS_SIZES[2].0);
        let does_not_fit = Side::Words(Words::DoesNotFit);

        // (a), the defaults: no size change, no calibration change (the cap sets the Outside
        // diameter at every stepped design).
        let [diameter, length, gap, count] = rows("(a)");
        assert_eq!(diameter.fewer, does_not_fit);
        assert_eq!(
            Pinned::of(&diameter.more),
            pin(
                "+0.0798",
                "+3.49",
                &[(SPACE_CLAIM, Level::Bad)],
                &[
                    (CLEARANCE, None, "+0.0125"),
                    (CUP_WALL, wall, "+0.0651"),
                    (GEARBOX, cold, "+0.0281"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&length.fewer),
            pin("-0.2008", "-8.79", &[], &[(GEARBOX, cold, "-0.0707")])
        );
        assert_eq!(
            Pinned::of(&length.more),
            pin("+0.2008", "+8.79", &[], &[(GEARBOX, cold, "+0.0707")])
        );
        assert_eq!(
            Pinned::of(&gap.fewer),
            pin(
                "+0.0759",
                "+3.32",
                &[],
                &[
                    (CLEARANCE, None, "-0.1000"),
                    (CUP_WALL, wall, "+0.0182"),
                    (GEARBOX, cold, "+0.0267"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&gap.more),
            pin(
                "-0.0733",
                "-3.21",
                &[],
                &[
                    (CLEARANCE, None, "+0.1000"),
                    (CUP_WALL, wall, "-0.0177"),
                    (GEARBOX, cold, "-0.0258"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&count.fewer),
            pin(
                "-0.6121",
                "-26.79",
                &[(GEARBOX, Level::Good)],
                &[(CUP_WALL, wall, "+0.4797")]
            )
        );
        assert_eq!(count.more, does_not_fit);

        // (b), the gui-smoke share link's design with Auto on the magnet length, solved: the
        // held row is computed (drawn as Auto's, test 8); the diameter +1 mm side puts the
        // Rotating diameter over the claim, a second axis of a check that already fails.
        let [diameter, length, gap, count] = rows("(b)");
        assert_eq!(diameter.fewer, does_not_fit);
        assert_eq!(
            Pinned::of(&diameter.more),
            pin(
                "+0.0892",
                "+3.57",
                &[],
                &[
                    (CLEARANCE, None, "+0.0125"),
                    (CUP_WALL, wall, "+0.0643"),
                    (GEARBOX, cold, "+0.0314"),
                    (SPACE_CLAIM, rotating, "+1.0000"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&length.fewer),
            pin(
                "-0.1945",
                "-7.78",
                &[(HOT_MIN, Level::Bad), (SPACE_CLAIM, Level::Good)],
                &[(GEARBOX, cold, "-0.0685")]
            )
        );
        assert_eq!(
            Pinned::of(&length.more),
            pin(
                "+0.1945",
                "+7.78",
                &[],
                &[(GEARBOX, cold, "+0.0685"), (SPACE_CLAIM, stack, "+1.0000")]
            )
        );
        assert_eq!(
            Pinned::of(&gap.fewer),
            pin(
                "+0.0827",
                "+3.31",
                &[],
                &[
                    (CLEARANCE, None, "-0.1000"),
                    (CUP_WALL, wall, "+0.0177"),
                    (GEARBOX, cold, "+0.0291"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&gap.more),
            pin(
                "-0.0798",
                "-3.19",
                &[(HOT_MIN, Level::Bad)],
                &[
                    (CLEARANCE, None, "+0.1000"),
                    (CUP_WALL, wall, "-0.0173"),
                    (GEARBOX, cold, "-0.0281"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&count.fewer),
            pin(
                "-0.6579",
                "-26.32",
                &[(HOT_MIN, Level::Bad)],
                &[(CUP_WALL, wall, "+0.4753"), (GEARBOX, cold, "-0.2317")]
            )
        );
        assert_eq!(count.more, does_not_fit);

        // (c), no back iron: the count -2 side takes the calibration coefficient, a change of
        // factor its number includes (owner 16A).
        let [.., count] = rows("(c)");
        let fewer = number(&count.fewer);
        assert_eq!(
            [
                signed_fixed(fewer.change_Nm, 4),
                signed_fixed(100.0 * fewer.fraction, 2)
            ],
            ["-0.5923", "-41.05"]
        );
        assert!(fewer.calibration_changed);

        // (e), the owner's case: the diameter row under 1 % both ways, and two magnets fewer
        // grow the Outside diameter.
        let [diameter, _, _, count] = rows("(e)");
        for (side, torque, percent) in [
            (&diameter.fewer, "-0.0063", "-0.37"),
            (&diameter.more, "-0.0046", "-0.28"),
        ] {
            let change = number(side);
            assert_eq!(signed_fixed(change.change_Nm, 4), torque);
            assert_eq!(signed_fixed(100.0 * change.fraction, 2), percent);
        }
        let grown = number(&count.fewer)
            .diameter_change_mm
            .expect("a size change");
        assert_eq!(signed_fixed(grown, 4), "+2.2307");

        // The mixed-length pair jumps both ways; its diameter +1 mm side is a number. The
        // reverse pair moves the active length by the step.
        let [diameter, length, ..] = rows("mixed");
        assert_eq!(
            [length.fewer, length.more],
            [Side::Words(Words::Jump), Side::Words(Words::Jump)]
        );
        assert_eq!(signed_fixed(number(&diameter.more).change_Nm, 4), "+0.0300");
        let [_, length, ..] = rows("reverse");
        for (side, percent) in [(&length.fewer, "-19.89"), (&length.more, "+19.89")] {
            assert_eq!(signed_fixed(100.0 * number(side).fraction, 2), percent);
        }
    }

    /// The N·m of a side as drawn: `x` at `decimals` places with its true sign, and the unit.
    fn torque(x: f64, decimals: usize) -> String {
        with_unit(signed_fixed(x, decimals), "N·m")
    }

    /// A quoted or size change as drawn: `x` at its own decimals from `from`, and `unit`.
    fn quoted(x: f64, from: usize, unit: &str) -> String {
        with_unit(signed_change(x, from), unit)
    }

    /// The unit of the result at `path`, as the card quotes it.
    fn unit_of(path: &str) -> &'static str {
        result_info(path).expect("a result").meta.unit
    }

    #[test]
    fn the_card_s_lines_read_as_drafted() {
        // Spec 8.1 test 2a (code review 2 MF-5): whole lines, built from the constants, the
        // labels, the step texts and section 2.3's numbers through the Size tab's format at the
        // row's decimals (T-14). The drafted blocks of 2.3 are these lines; a G-3 word change
        // flows through.
        let designs = card_designs();
        let lines = |key| find(&designs, key).card().lines(None);
        let [od, length, gap, count] = Control::ALL.map(Control::label);
        let (clearance, cup_wall, gearbox, space) = (
            label(CLEARANCE),
            label(CUP_WALL),
            label(GEARBOX),
            label(SPACE_CLAIM),
        );
        let mm = |x: f64| quoted(x, 2, "mm");
        let cold = |x: f64| quoted(x, 2, "N·m");
        // A Magnet face gap side as (a) and the odd count draw it: its N·m and percent, then
        // the Running clearance, the wall needed and cold-high at the gearbox input it quotes.
        let gap_side = |step: &str,
                        torque_nm: f64,
                        fraction: f64,
                        clearance_mm: f64,
                        wall_mm: f64,
                        cold_high_nm: f64| {
            format!(
                "{step} -> {} ({}), {clearance} {}, {cup_wall}: {WALL_NEEDED} {}, \
                 {gearbox}: {COLD_HIGH} {}",
                torque(torque_nm, 2),
                signed_percent(fraction),
                mm(clearance_mm),
                mm(wall_mm),
                cold(cold_high_nm)
            )
        };

        // (a): the order within a side is the number, the size changes, the flips, then the
        // quoted changes in the checks' order.
        assert_eq!(
            lines("(a)"),
            [
                CAPTION.to_owned(),
                format!(
                    "{od}: -1 mm -> {SIDE_DOES_NOT_FIT}; +1 mm -> {} ({}), {space}: {FLIP_FAILS}, \
                     {clearance} {}, {cup_wall}: {WALL_NEEDED} {}, {gearbox}: {COLD_HIGH} {}",
                    torque(0.0798, 2),
                    signed_percent(0.0349),
                    mm(0.0125),
                    mm(0.0651),
                    cold(0.0281)
                ),
                format!(
                    "{length}: -1 mm -> {} ({}), {gearbox}: {COLD_HIGH} {}; +1 mm -> {} ({}), \
                     {gearbox}: {COLD_HIGH} {}",
                    torque(-0.2008, 2),
                    signed_percent(-0.0879),
                    cold(-0.0707),
                    torque(0.2008, 2),
                    signed_percent(0.0879),
                    cold(0.0707)
                ),
                format!(
                    "{gap}: {}; {}",
                    gap_side("-0.1 mm", 0.0759, 0.0332, -0.1, 0.0182, 0.0267),
                    gap_side("+0.1 mm", -0.0733, -0.0321, 0.1, -0.0177, -0.0258)
                ),
                format!(
                    "{count}: -2 -> {} ({}), {gearbox}: {FLIP_PASSES}, {cup_wall}: {WALL_NEEDED} \
                     {}; +2 -> {SIDE_DOES_NOT_FIT}",
                    torque(-0.6121, 2),
                    signed_percent(-0.2679),
                    mm(0.4797)
                ),
            ]
        );

        // (e): the diameter row's two N·m at the row's 3 decimals (T-14: each on its own would
        // read "-0.01" and "-0.005"), and two magnets fewer grow the Outside diameter.
        let e = lines("(e)");
        assert_eq!(
            e[1],
            format!(
                "{od}: -1 mm -> {} ({}), {clearance} {}, {cup_wall}: {WALL_NEEDED} {}; \
                 +1 mm -> {} ({}), {space}: {FLIP_FAILS}, {clearance} {}, {cup_wall}: \
                 {WALL_NEEDED} {}",
                torque(-0.0063, 3),
                signed_percent(-0.0037),
                mm(-0.013),
                mm(-0.079),
                torque(-0.0046, 3),
                signed_percent(-0.0028),
                mm(0.0122),
                mm(0.079)
            )
        );
        assert_eq!(
            e[4],
            format!(
                "{count}: -2 -> {} ({}), {od} {}, {space}: {FLIP_FAILS}, {cup_wall}: \
                 {WALL_NEEDED} {}; +2 -> {} ({}), {gearbox}: {FLIP_FAILS}, {cup_wall}: \
                 {WALL_NEEDED} {}",
                torque(-0.8868, 2),
                signed_percent(-0.53),
                quoted(2.2307, 1, "mm"),
                mm(0.7995),
                torque(0.6121, 2),
                signed_percent(0.3658),
                mm(-0.4797)
            )
        );

        // (c): the calibration note right after the number, then the shaft joints' quoted
        // needs, named by their results' labels. (Its fraction to 5 decimals: -41.05 % at two
        // reads -41.0 %, while the side, -41.0516 %, reads -41.1 %.)
        let joints = label(SHAFT_JOINT);
        let need = |path: &str, x: f64| {
            format!(
                "{joints}: {} {}",
                label_for(path),
                quoted(x, 2, unit_of(path))
            )
        };
        assert_eq!(
            lines("(c)")[4],
            format!(
                "{count}: -2 -> {} ({}), {CALIBRATION_CHANGE}, {}, {}; +2 -> {SIDE_DOES_NOT_FIT}",
                torque(-0.5923, 2),
                signed_percent(-0.41052),
                need("shaft_joint.hub_needed_um", -2.43),
                need("shaft_joint.cup_needed_um", -1.9182)
            )
        );

        // The mixed pair's and the short cavity's Overall length rows: the jump both ways.
        let jumps = format!("{length}: -1 mm -> {JUMP}; +1 mm -> {JUMP}");
        for key in ["mixed", "short cavity"] {
            assert_eq!(lines(key)[2], jumps, "{key}");
        }
        // The boss plateau's Outside diameter row: outside the slider's range below the boss,
        // and the cup's jump above it (the owner's FO-1 and FO-2).
        assert_eq!(
            lines("R-13 boss plateau")[1],
            format!("{od}: -1 mm -> {END_OF_RANGE}; +1 mm -> {CUP_JUMP}")
        );
        // The jump words name the step's amount, so "by the step" never reads as the step's
        // doing; they hold where the overlap does not move at all (the cup cavity of 11.7 mm,
        // test 3), as round 1's "changes by a different amount" did not (G-3 round 2, SF-1). They
        // name its direction too, so "amount" never reads as a size alone: at a cup cavity of
        // 10.7 mm the overlap moves by the step's size the other way (test 3; the fold-in's G-3
        // re-review, MF-1). The two jump texts read alike, each naming its own quantity (FO-2
        // mirrors JUMP).
        for text in [JUMP, CUP_JUMP] {
            assert!(
                text.ends_with("by the step's amount and direction"),
                "{text}"
            );
        }
        assert_eq!(
            JUMP.replacen("the overlapping magnet length", "{}", 1),
            CUP_JUMP.replacen("the cup diameter", "{}", 1)
        );

        // The caption claims one control's step alone, never that everything else is held: at
        // the reverse pair the Overall length slider's own mapping trims the longer inner ring
        // (12.70 -> 7.35 mm on the +1 mm side), and no cost on its row shows it (G-3 round 1,
        // MF-1, decision D4).
        let reverse = &find(&designs, "reverse").inputs;
        let trimmed = compute_all(&slider_step(Control::OverallLength, reverse, None, true));
        assert!(trimmed.model.inner_length_mm < compute_all(reverse).model.inner_length_mm - 5.0);
        assert!(!CAPTION.contains("held"), "{CAPTION}");

        // The odd count: no size rows; the gap's numbers, and the count's two sides in words.
        assert_eq!(
            lines("odd count"),
            [
                CAPTION.to_owned(),
                format!(
                    "{gap}: {}; {}",
                    gap_side("-0.1 mm", 0.0653, 0.0322, -0.1, 0.0202, 0.0230),
                    gap_side("+0.1 mm", -0.0628, -0.031, 0.1, -0.0197, -0.0221)
                ),
                format!("{count}: -2 -> {NOT_ON_THE_SLIDER}; +2 -> {NOT_ON_THE_SLIDER}"),
            ]
        );

        // The side words not drawn above, each where spec 2.3's Edges show it, and the caution
        // flip: the keyway through the hub on the length and gap rows, no prediction at the end
        // effect's edge, a side outside the slider's range at the user's Minimum OD, and
        // Temperature limits turning to a caution at the temperature edge, on those three sides
        // only.
        let keyway = lines("keyway through");
        assert_eq!(
            keyway[2],
            format!("{length}: -1 mm -> {KEYWAY_BREAKS_THROUGH}; +1 mm -> {KEYWAY_BREAKS_THROUGH}")
        );
        assert_eq!(
            keyway[3],
            format!(
                "{gap}: -0.1 mm -> {KEYWAY_BREAKS_THROUGH}; +0.1 mm -> {KEYWAY_BREAKS_THROUGH}"
            )
        );
        let edge = lines("end effect at a step");
        assert!(
            edge[2].starts_with(&format!(
                "{length}: -1 mm -> {SIDE_NO_PREDICTION}; +1 mm -> "
            )),
            "{}",
            edge[2]
        );
        assert_eq!(
            edge[4],
            format!("{count}: -2 -> {SIDE_NO_PREDICTION}; +2 -> {SIDE_DOES_NOT_FIT}")
        );
        let at_minimum = lines("(a), Min OD 42.8");
        assert!(
            at_minimum[1].starts_with(&format!("{od}: -1 mm -> {END_OF_RANGE}; +1 mm -> ")),
            "{}",
            at_minimum[1]
        );
        let caution = format!("{}: {FLIP_CAUTION}", label(TEMPERATURE));
        let temperature = find(&designs, "temperature edge").card();
        let drawn = temperature.lines(None);
        // Each row's (- side, + side): the caution on the Outside diameter and Overall length
        // +1 mm sides and on the count's -2 side, none on the gap row.
        for (index, expected) in [
            (1, [false, true]),
            (2, [false, true]),
            (3, [false, false]),
            (4, [true, false]),
        ] {
            let texts = side_texts(&drawn[index], &temperature.rows()[index - 1]);
            assert_eq!(
                texts.map(|side| side.contains(&caution)),
                expected,
                "{}",
                drawn[index]
            );
        }

        // (b) with Auto on the magnet length: the caption, the Auto line, then the held row in
        // its place.
        let b = find(&designs, "(b)")
            .card()
            .lines(Some(FreeVariable::AxialLength));
        assert_eq!(b[..2], [CAPTION, AUTO_HOLDS_LENGTH]);
        assert_eq!(b[3], held_line(Control::OverallLength));
        assert_eq!(b[3], format!("{length}: {}", size_controls::SET_BY_AUTO));
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn the_flips_and_the_margins_name_the_checks_that_move() {
        // Spec 8.1 test 7 (owner 8A; spec 3.6). At every design of section 2, against each side's
        // stepped design rebuilt here: the flips are exactly the counted checks whose level
        // differs, in the guides' order; each quoted change is of a check that fails on both
        // designs (never Hot-low torque, which is the row's own number, nor Temperature limits,
        // whose terms read stored 3D fields: T-13), is drawn (`shows`), is in its result's own
        // unit, one of mm, µm, MPa, °C and N·m, and is that result on the stepped design less
        // the design shown's, bit for bit.
        let mut quotes = 0;
        let numbers = for_each_number_side(|side| {
            let NumberSide {
                what,
                design,
                change,
                before,
                stepped,
                after,
            } = side;
            let flips: Vec<Flip> = counted_guides(PAUSED)
                .filter_map(|guide| {
                    let to = level(guide, stepped, after);
                    (level(guide, &design.inputs, before) != to).then_some(Flip {
                        check: guide.label,
                        to,
                    })
                })
                .collect();
            assert_eq!(change.flips, flips, "{what}");
            for margin in &change.margins {
                let guide = counted_guides(PAUSED)
                    .find(|guide| guide.label == margin.check)
                    .unwrap_or_else(|| panic!("{what}: {margin:?}"));
                assert_eq!(level(guide, &design.inputs, before), Level::Bad, "{what}");
                assert_eq!(level(guide, stepped, after), Level::Bad, "{what}");
                assert!(![HOT_MIN, TEMPERATURE].contains(&guide.check), "{what}");
                assert!(shows(margin.change, margin.decimals), "{what}: {margin:?}");
                assert!(QUOTED_UNITS.contains(&margin.unit), "{what}: {margin:?}");
                // The unit and the decimals are those of the one term the check quotes under its
                // name, and the change is that term's result on the stepped design less the
                // design shown's, bit for bit.
                let quoted: Vec<Quoted> = check_terms(guide.check, &design.inputs, before)
                    .iter()
                    .filter_map(|term| term.quoted)
                    .filter(|q| {
                        q.name == margin.name
                            && q.decimals == margin.decimals
                            && unit_of(q.path) == margin.unit
                    })
                    .collect();
                let [quoted] = quoted[..] else {
                    panic!("{what}: {margin:?} against {quoted:?}")
                };
                assert_eq!(
                    margin.change.to_bits(),
                    (result_number(after, quoted.path) - result_number(before, quoted.path))
                        .to_bits(),
                    "{what}: {margin:?}"
                );
                quotes += 1;
            }
        });
        // Not vacuous: every number side, and every quoted change, at section 2's designs.
        assert_eq!(numbers, NUMBER_SIDES);
        assert_eq!(quotes, 515);

        let designs = card_designs();
        let design = |key| find(&designs, key);
        // (a) gap -0.1 mm quotes Running clearance -0.10 mm (owner 8A's example), and the
        // gearbox's quantity by its name (T-11).
        let gap = number(&design("(a)").side(Control::FaceGap, false));
        assert_eq!(
            (gap.margins[0].check, gap.margins[0].name),
            (label(CLEARANCE), None)
        );
        assert_eq!(signed_change(gap.margins[0].change, 2), "-0.10");
        let gearbox = gap.margins.last().expect("the gearbox's quantity");
        assert_eq!(
            (gearbox.check, gearbox.name),
            (label(GEARBOX), Some(COLD_HIGH))
        );
        // A check that fails on both designs, whose quantity the step does not move, is not
        // quoted: at (a) a length step leaves the running clearance and the wall needed as they
        // were.
        let length = number(&design("(a)").side(Control::OverallLength, true));
        assert_eq!(
            length.margins.iter().map(|m| m.check).collect::<Vec<_>>(),
            [label(GEARBOX)]
        );
        // A term that fails on the stepped design only is quoted: at (b) the diameter +1 mm side
        // puts the Rotating diameter over the space claim, whose large-diameter stack already
        // fails, so no flip shows it.
        let b = design("(b)");
        let plus = number(&b.side(Control::OutsideDiameter, true));
        let rotating = SPACE_CLAIM_AXIS_SIZES[0].0;
        assert!(plus.flips.is_empty(), "{plus:?}");
        assert!(
            plus.margins.iter().any(|m| m.name == Some(rotating)),
            "{plus:?}"
        );
        let rotating_passes = |inputs: &DesignInputs| {
            check_terms(SPACE_CLAIM, inputs, &compute_all(inputs))[0].passes
        };
        assert!(rotating_passes(&b.inputs));
        assert!(!rotating_passes(&slider_step(
            Control::OutsideDiameter,
            &b.inputs,
            None,
            true
        )));
        // Only the joint fails: every number side quotes the inner hub's needed minimum.
        let need = label_for("shaft_joint.hub_needed_um");
        for row in design("only the joint fails").card().rows() {
            for (_, side) in sides(row) {
                if let Side::Change(change) = side {
                    assert!(
                        change.margins.iter().any(|m| m.name == Some(need)),
                        "{row:?}"
                    );
                }
            }
        }
        // The temperature edge: Temperature limits turns to a caution on three sides.
        let edge = design("temperature edge").card();
        let caution = Flip {
            check: label(TEMPERATURE),
            to: Level::Caution,
        };
        let cautions: Vec<(Control, bool)> = edge
            .rows()
            .iter()
            .flat_map(|row| {
                sides(row)
                    .into_iter()
                    .filter(
                        |(_, side)| matches!(side, Side::Change(c) if c.flips.contains(&caution)),
                    )
                    .map(move |(more, _)| (row.control, more))
            })
            .collect();
        assert_eq!(
            cautions,
            [
                (Control::OutsideDiameter, true),
                (Control::OverallLength, true),
                (Control::MagnetsPerRing, false)
            ]
        );
        // Temperature limits is never quoted (T-13). Its verdict is amber at worst (`verdict_level`
        // reads OK or CHECK), so it never fails on both designs; and its terms quote nothing at
        // any design, so no level could let one through.
        let temperature = guide(TEMPERATURE).expect("the temperature guide");
        for design in &designs {
            let results = compute_all(&design.inputs);
            assert!(
                check_terms(TEMPERATURE, &design.inputs, &results)
                    .iter()
                    .all(|term| term.quoted.is_none()),
                "{}",
                design.key
            );
            assert_ne!(
                level(temperature, &design.inputs, &results),
                Level::Bad,
                "{}",
                design.key
            );
        }
    }

    #[test]
    fn the_card_names_the_clamp_only_while_unpaused() {
        // Spec 8.1 test 7a (code review M1): the flips run over the guides that count at the
        // panel's pause. Unpaused, the clamp's guide counts: at the clamp's limit the Overall
        // length +1 mm side flips it to fails, and no clamp quantity is quoted (`check_terms`
        // gives the clamp none). Paused, as shipped, no side names it.
        let at_limit = clamp_at_its_limit();
        let results = compute_all(&at_limit);
        let clamp = label(CLAMP);
        let names_the_clamp = |card: &TorqueCard| {
            card.rows().iter().any(|row| {
                sides(row).into_iter().any(|(_, side)| match side {
                    Side::Change(change) => {
                        change.flips.iter().any(|flip| flip.check == clamp)
                            || change.margins.iter().any(|margin| margin.check == clamp)
                    }
                    Side::Words(_) => false,
                })
            })
        };
        let unpaused = card(&at_limit, &results, None, false);
        let longer = number(&unpaused.rows()[1].more);
        assert!(
            longer.flips.contains(&Flip {
                check: clamp,
                to: Level::Bad
            }),
            "{longer:?}"
        );
        assert!(names_the_clamp(&unpaused));
        assert!(unpaused.rows().iter().all(|row| {
            sides(row).into_iter().all(|(_, side)| match side {
                Side::Change(change) => change.margins.iter().all(|m| m.check != clamp),
                Side::Words(_) => true,
            })
        }));
        assert!(!names_the_clamp(&card(&at_limit, &results, None, true)));
        for inputs in [at_limit, only_the_clamp_fails()] {
            let paused = card_at(&inputs, None);
            assert!(!names_the_clamp(&paused));
            for line in paused.lines(None) {
                assert!(!line.to_lowercase().contains("clamp"), "{line}");
            }
        }
    }

    #[test]
    fn the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s() {
        // Spec 8.1 test 8 (owner 5A; spec 3.4): Auto's marking is drawn by `lines`, outside the
        // card. For each Auto variable at (a) and (e2): the held row, the one `held_by` names,
        // reads `held_line` after the caption and the Auto line; the other rows read as with
        // Auto off, which shows the held row's numbers. One card serves both: `card` takes no
        // Auto argument, so the card cannot differ between them (the panel's cache test pins
        // that turning Auto on or off computes none), and `lines` alone draws the marking.
        let designs = card_designs();
        for key in ["(a)", "(e2)"] {
            let card = find(&designs, key).card();
            let plain = card.lines(None);
            assert_eq!(plain[0], CAPTION);
            for variable in FreeVariable::ALL {
                let held = Control::ALL
                    .into_iter()
                    .position(|control| control.held_by() == Some(variable))
                    .expect("a control for every variable");
                let mut expected = plain.clone();
                expected.insert(1, auto_line(variable).to_owned());
                expected[2 + held] = held_line(Control::ALL[held]);
                assert_eq!(card.lines(Some(variable)), expected, "{key} {variable:?}");
                // Auto off, the held row shows its numbers: its line is the row's two sides, and
                // each number side draws an N·m and its own percent.
                let row = &card.rows()[held];
                let mut numbers = 0;
                for ((_, side), text) in sides(row)
                    .into_iter()
                    .zip(side_texts(&plain[1 + held], row))
                {
                    if let Side::Change(change) = side {
                        let (_, percent) = drawn_number(text);
                        assert_eq!(
                            format!("{percent} %"),
                            signed_percent(change.fraction),
                            "{key} {variable:?}: {text}"
                        );
                        numbers += 1;
                    }
                }
                assert!(numbers > 0, "{key} {variable:?}: {}", plain[1 + held]);
            }
        }
        // The three Auto lines, each the variable's own.
        assert_eq!(
            FreeVariable::ALL.map(auto_line),
            [AUTO_HOLDS_LENGTH, AUTO_HOLDS_COUNT, AUTO_HOLDS_RING]
        );
        // The odd count has no size rows (T-5): with Auto on the ring size or the magnet length
        // the Auto line is drawn and no row is held; with Auto on the count, its row is.
        let odd = find(&designs, "odd count").card();
        let plain = odd.lines(None);
        assert_eq!(plain.len(), 3);
        for variable in [FreeVariable::RingRadius, FreeVariable::AxialLength] {
            assert_eq!(
                odd.lines(Some(variable)),
                [
                    CAPTION.to_owned(),
                    auto_line(variable).to_owned(),
                    plain[1].clone(),
                    plain[2].clone()
                ]
            );
        }
        assert_eq!(
            odd.lines(Some(FreeVariable::MagnetsPerRing)),
            [
                CAPTION.to_owned(),
                AUTO_HOLDS_COUNT.to_owned(),
                plain[1].clone(),
                held_line(Control::MagnetsPerRing)
            ]
        );
        // Each Auto design of section 2 draws its held row once.
        for design in &designs {
            let Some(variable) = design.auto else {
                continue;
            };
            let lines = design.card().lines(Some(variable));
            let held = lines
                .iter()
                .filter(|line| line.ends_with(size_controls::SET_BY_AUTO))
                .count();
            assert_eq!(held, 1, "{}: {lines:?}", design.key);
        }
    }

    #[test]
    fn size_changes_and_margins_are_drawn_only_when_they_show() {
        // Spec 8.1 test 9a (code review M2): section 5's cap keeps float noise off the screen.
        for from in [1, 2] {
            for x in [
                0.0,
                1e-15,
                -1e-15,
                3.6e-15,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                assert!(!shows(x, from), "{x} from {from}");
            }
            assert!(shows(0.0046, from), "from {from}");
        }
        assert!(shows(6e-5, 2));
        assert!(!shows(6e-5, 1));
        // The odd count has no size controls: its gap sides carry no size change, and its card
        // equals a second card of it (no NaN stands for "no value").
        let designs = card_designs();
        let odd = find(&designs, "odd count");
        for (_, side) in sides(&odd.card().rows()[0]) {
            let change = number(side);
            assert_eq!(
                (change.diameter_change_mm, change.length_change_mm),
                (None, None)
            );
        }
        assert_eq!(odd.card(), odd.card());
        // At (a) no side draws a size change: the cap sets the Outside diameter at every stepped
        // design, and the length moves by float noise only.
        let size_labels = [Control::OutsideDiameter, Control::OverallLength].map(Control::label);
        let draws_a_size = |change: &Change, control: Control| {
            change.costs(control).iter().any(|cost| {
                size_labels
                    .iter()
                    .any(|label| cost.starts_with(&format!("{label} ")))
            })
        };
        let a = find(&designs, "(a)");
        for row in a.card().rows() {
            for (_, side) in sides(row) {
                if let Side::Change(change) = side {
                    assert!(!draws_a_size(change, row.control), "{row:?}");
                }
            }
        }
        // A side with a planted 1e-15 mm size change draws nothing for it; 0.04 mm draws, but
        // never on the row of its own dimension.
        let mut planted = number(&a.side(Control::FaceGap, true));
        let drawn = planted.costs(Control::FaceGap);
        planted.diameter_change_mm = Some(1e-15);
        planted.length_change_mm = Some(-1e-15);
        assert_eq!(planted.costs(Control::FaceGap), drawn);
        planted.diameter_change_mm = Some(0.04);
        assert!(draws_a_size(&planted, Control::FaceGap));
        assert!(!draws_a_size(&planted, Control::OutsideDiameter));
    }

    #[test]
    fn a_change_is_drawn_exactly_where_its_own_decimals_read_a_digit() {
        // One cap rule (decision 10B, T-9): the card draws a size or margin change (`shows`)
        // exactly where the decimals the Size tab writes it at (`signed_change`) read a digit
        // that is not zero, at every start: the sliders' 1, the N·m's 2, and a start whose cap
        // lies past the 15 decimals of the guidance's search (`fewest_decimals_apart`).
        for from in [0, 1, 2, 13, 14, 15, 16] {
            for x in [
                0.0,
                -0.0,
                0.2010,
                -0.0063,
                -0.0046,
                -0.00012,
                -0.00004,
                6e-5,
                1e-10,
                3.6e-15,
                6e-17,
                -6e-17,
                1e-17,
                0.5,
                1e300,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                let drawn = reads_not_zero(&signed_change(x, from));
                assert_eq!(shows(x, from), drawn, "{x} from {from}");
            }
        }
    }

    #[test]
    fn a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required() {
        // Spec T-14 and 10.1 step 6 (the one-flip line on the STOP sheet): the table's "T-14 one
        // flip", (e2) with Auto on the magnet length shown with Auto off and Required halfway
        // between its Outside diameter sides' hot-low. Both sides draw the same N·m and percent
        // at the row's decimals, and only the side whose hot-low is below Required flips
        // Hot-low torque.
        let designs = card_designs();
        let design = find(&designs, "T-14 one flip");
        let required = design.inputs.metal.required_min_Nm;
        let card = design.card();
        let row = &card.rows()[0];
        assert_eq!(row.control, Control::OutsideDiameter);
        let hot_low = Flip {
            check: label(HOT_MIN),
            to: Level::Bad,
        };
        let sides = [number(&row.fewer), number(&row.more)];
        let flips = sides.each_ref().map(|side| side.flips.contains(&hot_low));
        let below = sides.each_ref().map(|side| side.hot_low_Nm < required);
        assert_eq!(flips, below);
        assert_eq!(flips.iter().filter(|&&flip| flip).count(), 1, "{flips:?}");
        let lines = card.lines(None);
        let [fewer, more] = side_texts(&lines[1], row);
        assert_eq!(drawn_number(fewer), drawn_number(more), "{fewer} | {more}");
    }

    #[test]
    fn where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift() {
        // Spec R-13, ruled by the owner's FO-2 at the slice-3 STOP: the table's "R-13 boss
        // plateau". The boss sets the Outside diameter at the slider's bottom: -1 mm reads
        // END_OF_RANGE, and +1 mm, which lifts the cup from its own size to the boss plus the
        // step, reads the cup's jump words: the number it drew, +7.55 % for one step that lifts
        // the cup more than three, read a jump as a slope (trap 4).
        let designs = card_designs();
        let design = find(&designs, "R-13 boss plateau");
        assert_eq!(
            design.side(Control::OutsideDiameter, false),
            Side::Words(Words::EndOfRange)
        );
        assert_eq!(
            design.side(Control::OutsideDiameter, true),
            Side::Words(Words::CupJump)
        );
        let (shown, stepped) = (
            compute_all(&design.inputs),
            compute_all(&slider_step(
                Control::OutsideDiameter,
                &design.inputs,
                design.minimum_diameter_mm,
                true,
            )),
        );
        assert_eq!(
            [shown.model.cup_od_mm, stepped.model.cup_od_mm].map(|mm| fixed(mm, 2)),
            ["37.78", "41.00"]
        );
        assert!(stepped.model.cup_od_mm - shown.model.cup_od_mm > 1.0 + STEP_SLACK_MM);
        let hot_low = |results: &DesignResults| results.metal.torque_hot_low_Nm;
        assert_eq!(
            signed_fixed(100.0 * (hot_low(&stepped) / hot_low(&shown) - 1.0), 2),
            "+7.55"
        );
    }

    #[test]
    fn the_diameter_row_reads_the_cup_s_jump_exactly_where_the_boss_sets_the_outside_diameter() {
        // The owner's FO-2 (spec 3.3 step 7, the diameter row's jump). The slider's mapping asks
        // for the cup at the request less the cap's overhang (`resize_diameter_using`), so a
        // step moves the cup by exactly the step wherever the cup or the cap sets the Outside
        // diameter, and by more wherever the shaft boss sets it, above both. So a diameter side
        // that reaches step 7 reads the cup's jump words exactly where the boss of the design
        // shown sits more than STEP_SLACK_MM above its cup and its cap (`diameter_breakdown`),
        // and is a number everywhere else: over the table's designs, the defaults at every even
        // count of the magnet slider, and the boss plateau with its boss just below and just
        // above its cup.
        use crate::engine::size_controls::diameter_breakdown;
        use crate::gui::test_support::{boss_plateau, with_poles};
        let plateau = boss_plateau();
        let cup = compute_all(&plateau).model.cup_od_mm;
        let with_boss = |mm: f64| {
            let mut inputs = plateau.clone();
            inputs.metal.boss_od_mm = mm;
            inputs
        };
        let range = FreeVariable::MagnetsPerRing.range();
        let mut designs: Vec<(String, DesignInputs, Option<f64>)> = card_designs()
            .into_iter()
            .map(|design| {
                (
                    design.key.to_owned(),
                    design.inputs,
                    design.minimum_diameter_mm,
                )
            })
            .collect();
        designs.extend(
            (range.min as i64..=range.max as i64)
                .step_by(2)
                .map(|npole| (format!("{npole} magnets"), with_poles(npole), None)),
        );
        designs.push((
            "boss just below the cup".to_owned(),
            with_boss(cup - 0.01),
            None,
        ));
        designs.push((
            "boss just above the cup".to_owned(),
            with_boss(cup + 0.01),
            None,
        ));
        let (mut jumps, mut numbers) = (Vec::new(), 0);
        for (key, inputs, minimum) in &designs {
            let card = card_at(inputs, *minimum);
            let Some(row) = card
                .rows()
                .iter()
                .find(|row| row.control == Control::OutsideDiameter)
            else {
                continue;
            };
            let parts = diameter_breakdown(inputs).expect("a diameter row has size controls");
            let boss_sets_it = parts.boss_mm - parts.cup_mm.max(parts.cap_mm) > STEP_SLACK_MM;
            for (more, side) in sides(row) {
                match side {
                    Side::Words(Words::CupJump) => {
                        assert!(boss_sets_it, "{key} {more}: {parts:?}");
                        jumps.push((key.as_str(), more));
                    }
                    Side::Change(_) => {
                        assert!(!boss_sets_it, "{key} {more}: {parts:?}");
                        numbers += 1;
                    }
                    Side::Words(_) => {}
                }
            }
        }
        // Not vacuous: the words where the boss sets the diameter, on the +1 mm side only (the
        // -1 mm side asks for less than the boss, below the slider's range), and numbers
        // elsewhere.
        assert_eq!(
            jumps,
            [
                ("R-13 boss plateau", true),
                ("boss just above the cup", true)
            ]
        );
        assert_eq!(numbers, 64);
    }

    #[test]
    fn a_step_to_a_design_without_size_controls_draws_no_size_change() {
        // Spec 2.2: a size change is the sliders' own (`dimensions`) of both designs, none where
        // either has no size controls. A design file can put the ring past its slider: at 6
        // magnets on a ring of 38.0 mm the Size tab draws its size controls, while at 4 magnets
        // (the count row's -2 side) it cannot define them, so that side draws its number and no
        // size change.
        use crate::gui::test_support::with_poles;
        let mut inputs = with_poles(6);
        inputs.coupling.inner_back_apothem_mm = 38.0;
        let mut four = inputs.clone();
        four.coupling.npole = 4;
        assert!(dimensions(&inputs).is_some());
        assert!(dimensions(&four).is_none());
        let fewer = number(&side_at(&inputs, None, Control::MagnetsPerRing, false));
        assert_eq!(
            (fewer.diameter_change_mm, fewer.length_change_mm),
            (None, None)
        );
        let card = card_at(&inputs, None);
        let line = card.lines(None).pop().expect("the count row's line");
        let [fewer_text, _] = side_texts(&line, &card.rows()[3]);
        assert!(
            !fewer_text.contains(Control::OutsideDiameter.label()),
            "{fewer_text}"
        );
    }

    #[test]
    fn the_card_never_panics_at_the_values_a_design_file_can_carry() {
        // A design file or a share link can carry any finite value (`validate()` checks only
        // that a number is finite), and the panel draws the card on the first view of every
        // design it opens, so a panic here ends the app (wasm). At each such value, at both
        // back irons and at the Minimum OD's top, the card is computed and equals a second
        // card of the same design (no field holds a NaN, spec 3.1), and every line of every
        // Auto state is drawn in plain words with the font's glyphs. At the degenerate ones the
        // owner ruled at the slice-3 STOP: a count below the slider's range draws one line
        // (FO-3), and no number on the card or in the readout's gap above it is wider than
        // WIDEST_NUMBER, nor does an N·m there read zero beside a percent that does not (FO-4: a
        // calibration coefficient of 1e-9 or 1e300, a variation just under 1), and a row with
        // such an N·m reads both its N·m in four significant digits (T-14; the fold-in's G-3
        // re-review, MF-2).
        type Edit = fn(&mut DesignInputs);
        let edits: [(&str, Edit); 47] = [
            ("npole 0", |d| d.coupling.npole = 0),
            ("npole 1", |d| d.coupling.npole = 1),
            ("npole 2", |d| d.coupling.npole = 2),
            ("npole 3", |d| d.coupling.npole = 3),
            ("npole -2", |d| d.coupling.npole = -2),
            ("npole 42", |d| d.coupling.npole = 42),
            ("npole 1e6", |d| d.coupling.npole = 1_000_000),
            ("npole i64::MAX", |d| d.coupling.npole = i64::MAX),
            ("npole i64::MIN", |d| d.coupling.npole = i64::MIN),
            ("gap 0", |d| d.metal.face_gap_mm = 0.0),
            ("gap -1", |d| d.metal.face_gap_mm = -1.0),
            ("gap 5.5", |d| d.metal.face_gap_mm = 5.5),
            ("gap 1e300", |d| d.metal.face_gap_mm = 1e300),
            ("override 0.1", |d| {
                d.coupling.magnets.axial_length_mm = Some(0.1)
            }),
            ("override 60", |d| {
                d.coupling.magnets.axial_length_mm = Some(60.0)
            }),
            ("override 1e300", |d| {
                d.coupling.magnets.axial_length_mm = Some(1e300)
            }),
            ("override -1", |d| {
                d.coupling.magnets.axial_length_mm = Some(-1.0)
            }),
            ("ring 0", |d| d.coupling.inner_back_apothem_mm = 0.0),
            ("ring 0.5", |d| d.coupling.inner_back_apothem_mm = 0.5),
            ("ring 200", |d| d.coupling.inner_back_apothem_mm = 200.0),
            ("ring 1e300", |d| d.coupling.inner_back_apothem_mm = 1e300),
            ("ring -1", |d| d.coupling.inner_back_apothem_mm = -1.0),
            ("cup depth 0", |d| d.metal.cup_depth_mm = 0.0),
            ("cup depth 1e300", |d| d.metal.cup_depth_mm = 1e300),
            ("boss 0", |d| d.metal.boss_od_mm = 0.0),
            ("boss 200", |d| d.metal.boss_od_mm = 200.0),
            ("cap 0", |d| d.metal.cap_od_mm = 0.0),
            ("cap 1e300", |d| d.metal.cap_od_mm = 1e300),
            ("bore 0", |d| d.coupling.bore_mm = 0.0),
            ("bore 1e300", |d| d.coupling.bore_mm = 1e300),
            ("variation 1 - 1e-10", |d| d.metal.variation = 1.0 - 1e-10),
            ("variation -1", |d| d.metal.variation = -1.0),
            ("variation 1e300", |d| d.metal.variation = 1e300),
            ("inner Br 1e-6 T", |d| {
                d.coupling.magnets.part_inner.clear();
                d.coupling.magnets.part_outer.clear();
                d.coupling.magnets.manual_inner_br_T = 1e-6;
            }),
            ("inner Br -1 T", |d| {
                d.coupling.magnets.part_inner.clear();
                d.coupling.magnets.part_outer.clear();
                d.coupling.magnets.manual_inner_br_T = -1.0;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-9", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-9;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-300", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-300;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-320", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-320;
            }),
            ("f_cal_original 1e300", |d| {
                d.calibration.f_cal_original = 1e300
            }),
            ("required 1e-300", |d| d.metal.required_min_Nm = 1e-300),
            ("required -1", |d| d.metal.required_min_Nm = -1.0),
            ("c_end 0", |d| d.coupling.c_end = 0.0),
            ("c_end 1e300", |d| d.coupling.c_end = 1e300),
            ("gearbox rating 0", |d| {
                d.coupling.gearbox_input_rating_Nm = 0.0
            }),
            ("residual target 1e300", |d| {
                d.metal.residual_target_mm = 1e300
            }),
            ("hot ambient 1e300", |d| {
                d.temperature.duty.hot_ambient_C = 1e300
            }),
            ("hot ambient -1e300", |d| {
                d.temperature.duty.hot_ambient_C = -1e300
            }),
        ];
        let ctx = glyph_context();
        let top = dimensions(&DesignInputs::default())
            .expect("size controls")
            .minimum_diameter_range
            .max;
        let mut rows = 0;
        for backiron in [1, 0] {
            for (name, edit) in edits {
                for minimum in [None, Some(top)] {
                    let mut inputs = DesignInputs::default();
                    inputs.coupling.backiron = backiron;
                    edit(&mut inputs);
                    let what = format!("{name}, back iron {backiron}, Minimum OD {minimum:?}");
                    let shown = card_at(&inputs, minimum);
                    assert_eq!(shown, card_at(&inputs, minimum), "{what}");
                    rows += shown.rows().len();
                    assert_one_precision_per_row(&shown, &what);
                    for auto in auto_states() {
                        let lines = shown.lines(auto);
                        assert!(!lines.is_empty(), "{what}");
                        for line in &lines {
                            assert!(!line.is_empty(), "{what}");
                            assert_eq!(
                                plain_text_violations(line),
                                Vec::<&str>::new(),
                                "{what}: {line}"
                            );
                            assert_glyphs(&ctx, line, &what);
                            // The owner's FO-4.
                            assert_numbers_fit(line, &what);
                            assert_no_zero_beside_its_percent(line, &what);
                        }
                    }
                    // The readout's gap above the card, where it draws one (the hot-low readout
                    // shows a number), keeps the same bound.
                    let results = compute_all(&inputs);
                    let hot_low = results.metal.torque_hot_low_Nm;
                    if predicts(&results, hot_low) {
                        let required = inputs.metal.required_min_Nm;
                        let (gap, _) = size_controls::torque_gap_text(hot_low, required);
                        assert_numbers_fit(&gap, &what);
                        assert_no_zero_beside_its_percent(&gap, &what);
                    }
                }
            }
        }
        // Not vacuous: every row the 188 cards draw, counted.
        assert_eq!(rows, 416);
    }

    /// The widest number a card line or the readout's gap draws (the owner's FO-4 at the slice-3
    /// STOP): a sign, six whole digits, a point and the cap's four decimals ("+999999.9999", a
    /// row's N·m at T-14's shared decimals). Four significant digits take at most 11
    /// ("-4.941e-324").
    const WIDEST_NUMBER: usize = 12;

    /// Asserts that every number `text` draws ([`numbers_in`]) is at most [`WIDEST_NUMBER`]
    /// characters wide.
    fn assert_numbers_fit(text: &str, what: &str) {
        for number in numbers_in(text).1 {
            assert!(
                number.chars().count() <= WIDEST_NUMBER,
                "{what}: {number} in {text}"
            );
        }
    }

    /// Asserts that no N·m `text` draws beside its percent, "{N·m} N·m ({percent} %)" (a card
    /// side, the readout's gap), reads zero while that percent reads a digit (trap 2; the owner's
    /// FO-4 at the slice-3 STOP).
    fn assert_no_zero_beside_its_percent(text: &str, what: &str) {
        let mut pieces = text.split(" N·m (");
        let mut before = pieces.next().unwrap_or_default();
        for after in pieces {
            let torque = before.rsplit(' ').next().unwrap_or_default();
            let (percent, _) = after.split_once(" %)").expect("a percent after the N·m");
            assert!(
                reads_not_zero(torque) || !reads_not_zero(percent),
                "{what}: {torque} N·m ({percent} %) in {text}"
            );
            before = after;
        }
    }

    /// Asserts T-14 under the owner's FO-4 on every row of `card` (the fold-in's G-3 re-review,
    /// MF-2): where either number side's change reads zero at the cap (no
    /// [`decimals_with_a_digit`] from [`TORQUE_DECIMALS`]) while its drawn percent reads a
    /// digit, both drawn N·m are their changes to within half a unit of the fourth significant
    /// digit ([`last_digit_step`]), so the row's two N·m keep their true ratio; a change of
    /// exactly zero reads zero.
    fn assert_one_precision_per_row(card: &TorqueCard, what: &str) {
        let lines = card.lines(None);
        for (row, line) in card.rows().iter().zip(&lines[1..]) {
            let numbers: Vec<(f64, &str, &str)> = sides(row)
                .into_iter()
                .zip(side_texts(line, row))
                .filter_map(|((_, side), text)| match side {
                    Side::Change(change) => {
                        let (torque, percent) = drawn_number(text);
                        Some((change.change_Nm, torque, percent))
                    }
                    Side::Words(_) => None,
                })
                .collect();
            let zero_beside_a_digit = numbers.iter().any(|&(change, _, percent)| {
                decimals_with_a_digit(change, TORQUE_DECIMALS).is_none() && reads_not_zero(percent)
            });
            if !zero_beside_a_digit {
                continue;
            }
            for (change, torque, _) in numbers {
                let drawn = torque.parse::<f64>().unwrap_or(f64::NAN);
                let true_to_four_digits = if change == 0.0 {
                    drawn == 0.0
                } else {
                    (drawn - change).abs() <= 0.5 * last_digit_step(change.abs())
                };
                assert!(
                    true_to_four_digits,
                    "{what}: {torque} N·m for a change of {change:e} in {line}"
                );
            }
        }
    }

    #[test]
    fn at_a_design_file_s_magnitudes_a_number_reads_four_significant_digits() {
        // The owner's FO-4 at the slice-3 STOP, at the G-3 record's probe 13: a calibration
        // coefficient of 1e-9 with no back iron at 8 magnets puts the sides' changes near
        // 1e-11 N·m beside true percents, which read "+0.00" and "-0.00", and one of 1e300 or a
        // Required of 1e-300 put numbers of about 300 digits on the card and in the readout's
        // gap. Each reads four significant digits with its sign, as the record writes the lines.
        let [od, _, _, count] = Control::ALL.map(Control::label);
        let gap = |inputs: &DesignInputs| {
            let hot_low = compute_all(inputs).metal.torque_hot_low_Nm;
            size_controls::torque_gap_text(hot_low, inputs.metal.required_min_Nm).0
        };
        let mut small = DesignInputs::default();
        small.coupling.backiron = 0;
        small.coupling.npole = 8;
        small.calibration.f_cal_original = 1e-9;
        let lines = card_at(&small, None).lines(None);
        let start = format!("{od}: -1 mm -> +1.690e-11 N·m (+1.9 %), ");
        assert!(lines[1].starts_with(&start), "{}", lines[1]);
        // The row's numbers, each read whole, its exponent included, as the width pin of
        // the_card_never_panics_at_the_values_a_design_file_can_carry reads them.
        assert_eq!(
            numbers_in(&lines[1]).1,
            ["+1.690e-11", "+1.9", "-0.01", "-2.412e-11", "-2.7", "+0.01"]
        );
        // Its -2 side reads zero at the cap beside -57.9 %, so the row's +2 side reads four
        // significant digits too (T-14; the fold-in's G-3 re-review, MF-2).
        let two_more = format!("; +2 -> +1.443 N·m (+1.612e11 %), {CALIBRATION_CHANGE}, ");
        assert!(
            lines[4].starts_with(&format!("{count}: ")) && lines[4].contains(&two_more),
            "{}",
            lines[4]
        );
        let mut large = DesignInputs::default();
        large.calibration.f_cal_original = 1e300;
        let lines = card_at(&large, None).lines(None);
        let start = format!("{od}: -1 mm -> {SIDE_DOES_NOT_FIT}; +1 mm -> +8.401e298 N·m (+3.5 %)");
        assert!(lines[1].starts_with(&start), "{}", lines[1]);
        assert_eq!(gap(&large), "margin +2.405e300 N·m (+9.622e301 %)");
        let mut low_required = DesignInputs::default();
        low_required.metal.required_min_Nm = 1e-300;
        assert_eq!(gap(&low_required), "margin +2.29 N·m (+2.285e302 %)");
    }

    #[test]
    fn at_slider_values_a_row_s_two_n_m_read_four_significant_digits_together() {
        // T-14 under the owner's FO-4 (the fold-in's G-3 re-review, MF-2): FO-4's second clause
        // reaches slider values. With both parts blank, manual blocks of 0.2 T, a gap of 3.0 mm,
        // 4 magnets and a ring of 15.0 mm, each on its slider, the Outside diameter row's +1 mm
        // side changes hot-low by +4.599e-5 N·m (+0.7 %), zero at the cap, and its -1 mm side by
        // -5.033e-5 N·m (-0.7 %), which reads "-0.0001" there: 2.17:1 drawn for a true 1.09:1.
        // So the row draws both its N·m in four significant digits.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.coupling.magnets.manual_inner_br_T = 0.2;
        inputs.coupling.magnets.manual_outer_br_T = 0.2;
        inputs.metal.face_gap_mm = 3.0;
        inputs.coupling.npole = 4;
        inputs.coupling.inner_back_apothem_mm = 15.0;
        for (path, value) in [
            ("coupling.magnets.manual_inner_br_T", 0.2),
            ("coupling.magnets.manual_outer_br_T", 0.2),
            (FACE_GAP, 3.0),
            (FreeVariable::MagnetsPerRing.path(), 4.0),
            ("coupling.inner_back_apothem_mm", 15.0),
        ] {
            let range = slider_range(path);
            let steps = (value - range.min) / range.step;
            assert!(
                (range.min..=range.max).contains(&value) && (steps - steps.round()).abs() < 1e-6,
                "{path} {value} on its slider"
            );
        }
        let [od, ..] = Control::ALL.map(Control::label);
        let (space, rotating) = (label(SPACE_CLAIM), SPACE_CLAIM_AXIS_SIZES[0].0);
        assert_eq!(
            card_at(&inputs, None).lines(None)[1],
            format!(
                "{od}: -1 mm -> -5.033e-5 N·m (-0.7 %), {space}: {rotating} -1.00 mm; +1 mm -> \
                 +4.599e-5 N·m (+0.7 %), {space}: {rotating} +1.00 mm"
            )
        );
    }

    /// The side texts of a row's line, "{label}: {fewer}; {more}": the two sides.
    fn side_texts<'a>(line: &'a str, row: &Row) -> [&'a str; 2] {
        let texts = line
            .strip_prefix(&format!("{}: ", row.control.label()))
            .unwrap_or_else(|| panic!("a row's line: {line}"));
        <[&str; 2]>::try_from(texts.split("; ").collect::<Vec<_>>())
            .unwrap_or_else(|_| panic!("two sides: {line}"))
    }

    /// The N·m and the percent a number side's text draws: "{step} -> {N·m} N·m ({percent} %)".
    fn drawn_number(side: &str) -> (&str, &str) {
        let (_, number) = side.split_once(" -> ").expect("a step");
        let (torque, rest) = number.split_once(" N·m (").expect("a number side");
        let (percent, _) = rest.split_once(" %)").expect("a percent");
        (torque, percent)
    }

    #[test]
    fn no_card_number_reads_zero_or_disagrees_with_its_percent() {
        // Spec 8.3 test 15 (decision 10B): at every design of section 2, each number side's N·m
        // reads a digit that is not zero unless the change is under the format's cap (T-9),
        // and its sign is its percent's.
        let designs = card_designs();
        let mut numbers = 0;
        for design in &designs {
            let card = design.card();
            let lines = card.lines(None);
            for (row, line) in card.rows().iter().zip(&lines[1..]) {
                for ((_, side), text) in sides(row).into_iter().zip(side_texts(line, row)) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let (torque, percent) = drawn_number(text);
                    assert!(
                        reads_not_zero(torque) || !shows(change.change_Nm, TORQUE_DECIMALS),
                        "{}: {text}",
                        design.key
                    );
                    assert_eq!(
                        torque.chars().next(),
                        percent.chars().next(),
                        "{}: {text}",
                        design.key
                    );
                    numbers += 1;
                }
            }
        }
        assert_eq!(numbers, NUMBER_SIDES);
        // Below zero: (c) with a design file's calibration coefficient of -0.95, two magnets
        // fewer.
        let card = find(&designs, "(c), f_cal_original -0.95").card();
        let count = &card.rows()[3];
        let lines = card.lines(None);
        let (torque, percent) = drawn_number(side_texts(&lines[4], count)[0]);
        assert!(number(&count.fewer).hot_low_Nm < 0.0);
        assert!(
            torque.starts_with('-') && percent.starts_with('-'),
            "{torque} {percent}"
        );
        // The one-line forms draw no number.
        for key in [
            "count overflow: i64::MIN",
            "short magnets",
            "shown does not fit",
            "variation 1.0",
            "zero torque",
        ] {
            for line in find(&designs, key).card().lines(None) {
                assert!(!line.chars().any(|c| c.is_ascii_digit()), "{key}: {line}");
            }
        }
    }

    /// Every line the card draws at every design of section 2, with each Auto state, and its
    /// heading: what the word and glyph tests scan.
    fn every_card_text() -> Vec<(String, String)> {
        let mut texts = vec![("heading".to_owned(), CARD_HEADING.to_owned())];
        for design in card_designs() {
            let card = design.card();
            for auto in auto_states() {
                for line in card.lines(auto) {
                    texts.push((format!("{} {auto:?}", design.key), line));
                }
            }
        }
        texts
    }

    #[test]
    fn the_card_texts_never_rank_the_controls() {
        // Spec 8.4 test 16 (owner 14A; spec 4.4): no card text ranks the controls
        // (`CARD_RANKING_WORDS`; its matcher's own examples are in `engine::explain::plain`),
        // orders the torques, says the coupling protects the gearbox, names the paused clamp
        // section, or says what the plain-words guard bans.
        let texts = every_card_text();
        // Not vacuous: the heading, and every line at every design with each Auto state.
        assert_eq!(texts.len(), 1056);
        for (what, text) in &texts {
            assert_eq!(ranking_words_in(text), Vec::<&str>::new(), "{what}: {text}");
            let lower = text.to_lowercase();
            for word in TORQUE_ORDER_WORDS {
                assert!(!lower.contains(word), "{what}: {text}");
            }
            assert!(!mentions_gearbox_protection(text), "{what}: {text}");
            assert!(!lower.contains("clamp"), "{what}: {text}");
            assert_eq!(
                plain_text_violations(text),
                Vec::<&str>::new(),
                "{what}: {text}"
            );
        }
    }

    #[test]
    fn every_card_text_has_its_glyphs() {
        // Spec 8.4 test 17.
        let ctx = glyph_context();
        for (what, text) in every_card_text() {
            assert_glyphs(&ctx, &text, &what);
        }
    }

    /// `line` with each number it draws replaced by '#' (its skeleton), and those numbers. A
    /// side's step ("-1", "+0.1", "-2" before " mm -> " or " -> ") is the template's, so it
    /// stays in the skeleton. A number in four significant digits keeps its exponent
    /// ("+1.690e-11", "+1.000e6": `format_value`'s form, the owner's FO-4).
    fn numbers_in(line: &str) -> (String, Vec<String>) {
        let chars: Vec<char> = line.chars().collect();
        let (mut skeleton, mut numbers) = (String::new(), Vec::new());
        let mut i = 0;
        while i < chars.len() {
            let signed =
                matches!(chars[i], '+' | '-') && chars.get(i + 1).is_some_and(char::is_ascii_digit);
            if !(signed || chars[i].is_ascii_digit()) {
                skeleton.push(chars[i]);
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
                end += 1;
            }
            // An exponent: "e", a minus sign or none, then digits.
            if chars.get(end) == Some(&'e') {
                let digits = end + 1 + usize::from(chars.get(end + 1) == Some(&'-'));
                if chars.get(digits).is_some_and(char::is_ascii_digit) {
                    end = digits;
                    while end < chars.len() && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                }
            }
            let number: String = chars[i..end].iter().collect();
            let after: String = chars[end..].iter().collect();
            if after.starts_with(" mm -> ") || after.starts_with(" -> ") {
                skeleton.push_str(&number);
            } else {
                skeleton.push('#');
                numbers.push(number);
            }
            i = end;
        }
        (skeleton, numbers)
    }

    #[test]
    fn the_card_s_strings_are_constants_and_its_numbers_come_from_results() {
        // Spec 8.4 test 18. Each design's card against a copy whose every number (each change in
        // N·m and as a fraction, each size change and each quoted change) is tripled: every line
        // keeps its skeleton, so its words are the template's and the constants', and differs at
        // every number it draws, so each number is read from the card (whose fields test 1 holds
        // to compute_all, and test 7 each quoted change). Three times a drawn value moves its
        // first drawn digit; two designs could not show it, since a gap step's Running clearance
        // change is the step itself at every design, and a torque that changes in proportion
        // keeps every percent.
        let tripled = |card: &TorqueCard| -> TorqueCard {
            let mut card = card.clone();
            if let TorqueCard::Rows(rows) = &mut card {
                for row in rows {
                    for side in [&mut row.fewer, &mut row.more] {
                        if let Side::Change(change) = side {
                            change.change_Nm *= 3.0;
                            change.fraction *= 3.0;
                            change.diameter_change_mm =
                                change.diameter_change_mm.map(|mm| 3.0 * mm);
                            change.length_change_mm = change.length_change_mm.map(|mm| 3.0 * mm);
                            for margin in &mut change.margins {
                                margin.change *= 3.0;
                            }
                        }
                    }
                }
            }
            card
        };
        let mut numbers = 0;
        for design in card_designs() {
            let card = design.card();
            let moved = tripled(&card);
            for (line, moved_line) in card.lines(design.auto).iter().zip(moved.lines(design.auto)) {
                let (skeleton, drawn) = numbers_in(line);
                let (moved_skeleton, moved_drawn) = numbers_in(&moved_line);
                assert_eq!(skeleton, moved_skeleton, "{}", design.key);
                for (number, moved_number) in drawn.iter().zip(&moved_drawn) {
                    assert_ne!(number, moved_number, "{}: {line}", design.key);
                    numbers += 1;
                }
            }
        }
        // Not vacuous: every number drawn at section 2's designs.
        assert_eq!(numbers, 1045);
    }

    // The count row: the neighbour lines' tests, moved from `gui/guidance.rs` with the lines
    // (slice 3 T5, spec 6.3; decision 7A), rewritten to the card's types and hot-low.

    #[test]
    fn the_count_row_s_sides_are_the_design_with_only_the_magnet_count_changed() {
        // Was each_neighbour_is_the_design_with_only_the_magnet_count_changed: each side is
        // compute_all of the design with only `coupling.npole` changed, the ring radius kept
        // (FreeVariable::MagnetsPerRing.apply, as the neighbour lines were), now in hot-low, or
        // words where its blocks do not fit.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let [fewer, more] =
            [false, true].map(|more| side_at(&inputs, None, Control::MagnetsPerRing, more));
        for (count, side) in [(npole - 2, &fewer), (npole + 2, &more)] {
            let design = with_magnets(&inputs, count);
            assert_eq!(
                design.coupling.inner_back_apothem_mm,
                inputs.coupling.inner_back_apothem_mm
            );
            let r = compute_all(&design);
            match side {
                Side::Change(change) => {
                    assert!(is_valid(&design, &r), "{count}");
                    assert_eq!(change.hot_low_Nm, r.metal.torque_hot_low_Nm, "{count}");
                    assert_eq!(
                        change.change_Nm,
                        r.metal.torque_hot_low_Nm - results.metal.torque_hot_low_Nm,
                        "{count}"
                    );
                }
                Side::Words(Words::DoesNotFit) => {
                    assert!(!blocks_fit(&design.coupling, &r.model), "{count}")
                }
                other => panic!("{count}: {other:?}"),
            }
        }
        // The owner's case (spec 1): two fewer magnets fit and lose hot-low torque, the cup
        // growing under the cap; two more do not fit at this ring size.
        let fewer = number(&fewer);
        assert!(fewer.change_Nm < 0.0, "{fewer:?}");
        let fewer_cup = compute_all(&with_magnets(&inputs, npole - 2))
            .model
            .cup_od_mm;
        assert!(fewer_cup > results.model.cup_od_mm, "{fewer_cup}");
        assert_eq!(words(&more), Words::DoesNotFit);
    }

    #[test]
    fn the_count_row_says_each_side_s_hot_low_change_or_why_not() {
        // Was the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not: the count's
        // numbers are the card's count row, in hot-low and the Size tab's number format, and the
        // lines under Magnets per ring keep the ring-size line and the fit line only (7A).
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let fewer = compute_all(&with_magnets(&inputs, npole - 2));
        let change = fewer.metal.torque_hot_low_Nm - results.metal.torque_hot_low_Nm;
        let label = Control::MagnetsPerRing.label();
        let line = row_line(
            &card(&inputs, &results, None, PAUSED),
            Control::MagnetsPerRing,
            None,
        );
        // The row's one number side, at its own decimals (T-14).
        let fewer_text = format!(
            "{label}: -2 -> {} N·m ({})",
            signed_change(change, TORQUE_DECIMALS),
            signed_percent(change / results.metal.torque_hot_low_Nm)
        );
        assert!(line.starts_with(&fewer_text), "{line}");
        assert!(
            line.ends_with(&format!("; +2 -> {SIDE_DOES_NOT_FIT}")),
            "{line}"
        );
        let lines = Consequences::of(&inputs, &results, None, PAUSED).magnet_lines(npole, None);
        assert_eq!(
            lines.len(),
            2,
            "the ring-size line and the fit line: {lines:?}"
        );
        assert_eq!(lines[0], RING_SIZE_STAYS);
    }

    #[test]
    fn at_either_end_of_the_magnet_slider_the_count_side_past_it_reads_its_end() {
        // Was at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown: at 4
        // magnets the -2 side, at 40 the +2 side reads END_OF_RANGE (9A, in the words of the
        // owner's FO-1), and the side inside the slider is the design with that count. 40
        // blocks fit only when narrow (the default parts do not fit 40: that card is
        // ShownDoesNotFit). An odd count (a design file's) draws only the ring-size line under
        // Magnets per ring: no count's line, and no fit line, since an odd count has no
        // diameter range to search (`smallest_fitting_diameter` is `None`). Its count row reads
        // NOT_ON_THE_SLIDER both ways (the_words_follow_the_order_of_the_tests).
        use crate::gui::test_support::{forty_narrow_magnets, odd_count, with_poles};
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        for (inputs, past_is_more) in [(with_poles(fewest), false), (forty_narrow_magnets(), true)]
        {
            let npole = inputs.coupling.npole;
            assert_eq!(npole, if past_is_more { most } else { fewest });
            let [fewer, more] =
                [false, true].map(|more| side_at(&inputs, None, Control::MagnetsPerRing, more));
            let (past, inside, inside_count) = if past_is_more {
                (&more, &fewer, npole - 2)
            } else {
                (&fewer, &more, npole + 2)
            };
            assert_eq!(*past, Side::Words(Words::EndOfRange), "{npole}");
            let at = compute_all(&with_magnets(&inputs, inside_count));
            match inside {
                Side::Change(change) => {
                    assert_eq!(
                        change.hot_low_Nm, at.metal.torque_hot_low_Nm,
                        "{inside_count}"
                    )
                }
                Side::Words(words) => assert_ne!(*words, Words::EndOfRange, "{inside_count}"),
            }
            let step = if past_is_more { "+2" } else { "-2" };
            let line = card_at(&inputs, None)
                .lines(None)
                .pop()
                .expect("the count row is the last line");
            assert!(
                line.contains(&format!("{step} -> {END_OF_RANGE}")),
                "{line}"
            );
        }
        let odd = odd_count();
        let lines = Consequences::of(&odd, &compute_all(&odd), None, PAUSED)
            .magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines, [RING_SIZE_STAYS]);
    }

    #[test]
    fn a_keyway_through_the_hub_changes_no_count_side_s_hot_low() {
        // Was a_keyway_through_the_hub_changes_no_neighbour_s_reading: the hub wall past the
        // keyway does not depend on the magnet count, so the count row skips the keyway (D-1)
        // and its numbers are the same with the keyway through the hub or not (the Size tab's
        // exploration line says it breaks through; the other rows say so on each side).
        let mut broken = DesignInputs::default();
        broken.coupling.bore_mm = 16.0; // the bore and keyway sliders' maxima
        broken.coupling.keyway_depth_mm = 4.0;
        let mut intact = broken.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert!(compute_all(&broken).model.hub_wall_past_key_mm <= 0.0);
        assert!(compute_all(&intact).model.hub_wall_past_key_mm > 0.0);
        // Each side's hot-low numbers, or its words.
        let read = |inputs: &DesignInputs| -> Vec<Result<(f64, f64, f64), Words>> {
            [false, true]
                .map(|more| side_at(inputs, None, Control::MagnetsPerRing, more))
                .into_iter()
                .map(|side| match side {
                    Side::Change(change) => {
                        Ok((change.hot_low_Nm, change.change_Nm, change.fraction))
                    }
                    Side::Words(words) => Err(words),
                })
                .collect()
        };
        assert_eq!(read(&broken), read(&intact));
        assert!(read(&broken)[0].is_ok(), "{:?}", read(&broken));
    }

    #[test]
    fn with_no_prediction_on_the_design_shown_the_card_invents_no_change() {
        // Was a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented: the
        // design shown has no prediction (its end effect just out of range) while two more
        // magnets would have one. The neighbour line showed that count's torque with no change;
        // the card compares nothing there (test 0, owner 9A): its one line, no row.
        use crate::gui::test_support::end_effect_edge;
        let inputs = end_effect_edge();
        let results = compute_all(&inputs);
        assert!(
            !crate::engine::model::end_effect_in_range(results.model.f_end),
            "{}",
            results.model.f_end
        );
        let more = with_magnets(&inputs, inputs.coupling.npole + 2);
        assert!(
            is_valid(&more, &compute_all(&more)),
            "two more magnets predict"
        );
        let shown = card(&inputs, &results, None, PAUSED);
        assert_eq!(shown, TorqueCard::ShownNoPrediction);
        assert!(shown.rows().is_empty());
        for auto in auto_states() {
            assert_eq!(shown.lines(auto), [SHOWN_NO_PREDICTION], "{auto:?}");
        }
    }

    #[test]
    fn with_magnets_too_short_for_the_model_the_card_names_no_count() {
        // Was with_magnets_too_short_for_the_model_every_neighbour_reads_words: at short magnets
        // (f_end below 0) the card draws its one line and no row, so no count's number shows,
        // and the lines under Magnets per ring are the ring-size line and the fit line only:
        // none for two magnets fewer or two more.
        let inputs = short_magnets();
        let results = compute_all(&inputs);
        let shown = card(&inputs, &results, None, PAUSED);
        assert_eq!(shown, TorqueCard::ShownNoPrediction);
        assert_eq!(shown.lines(None), [SHOWN_NO_PREDICTION]);
        let lines = Consequences::of(&inputs, &results, None, PAUSED)
            .magnet_lines(inputs.coupling.npole, None);
        assert_eq!(
            lines.len(),
            2,
            "the ring-size line and the fit line: {lines:?}"
        );
        assert_eq!(lines[0], RING_SIZE_STAYS);
    }
}
