//! Check guides (guidance slice 1, spec 3.4 and 4.1): under a dashboard check that fails or
//! asks for a look, what is wrong in this design's numbers (the Problem), why the check
//! matters (the Why, one plain sentence) and which inputs to move which way (the Fix); where
//! each dashboard row's "Why:" link leads ([`Learn`], [`row_learn`]); and the design status
//! line that counts the failing checks ([`design_status`]).
//!
//! [`CHECK_GUIDES`] holds one guide per check of the dashboard's rows
//! ([`crate::gui::dashboard::DASHBOARD`]) and one for the Magnets fit row ([`FIT_CHECK`]),
//! which no result holds: the blocks fit is [`blocks_fit`] of the model's results. A guide's
//! level is the dashboard's own badge ([`check_level`]); its Problem quotes the live results
//! and inputs, each number as the readouts format it, and is `None` while the check passes.
//!
//! A fix is an input path and a direction ([`fixes`]): a slider input raised or lowered, or a
//! choice or picker input changed. The temperature guide picks its fixes by the terms of its
//! verdict that fail ([`Fixes::PerTerm`]); every other guide has one list. A fix whose slider
//! is already at its end that way is left out. A fix's text takes the input's label from the
//! [`InputCatalogue`], or the Size tab's name for it ([`fix_text`]), so no guide types a
//! label. The tests hold the fixes to helping: on each test design that fails a check, every
//! failing term of the check has a fix that moves it toward passing by one slider step or one
//! choice, and no fix turns another check red.
//!
//! The engine's verdict texts, the results table and the exports keep their wording (locked by
//! `tests/python_schema.rs`): this is a GUI layer over them, in plain words
//! ([`plain_text_violations`] lists what it must not say). Pure functions and data: no egui.

use crate::engine::clamps::SCREW_SIZES;
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::model::{blocks_fit, pitch_share};
use crate::engine::temperature::CURE_MARGIN_MIN_C;
use crate::gui::dashboard::{DASHBOARD, Level, check_level, result_info};
use crate::gui::format::{format_value, with_unit};
use crate::gui::inputs::{InputCatalogue, optional_seed, section_label};
use crate::gui::pickers::PART_PATHS;
use crate::gui::size_controls;
use crate::{DesignInputs, DesignResults};

/// Which way a fix moves its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Up its slider.
    Raise,
    /// Down its slider.
    Lower,
    /// To another choice: a selector's code, or a picker's part.
    Change,
}

impl Direction {
    /// The verb a fix's text starts with.
    const fn word(self) -> &'static str {
        match self {
            Direction::Raise => "Raise",
            Direction::Lower => "Lower",
            Direction::Change => "Change",
        }
    }
}

/// Where a dashboard row's "Why:" link leads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Learn {
    /// A teaching note by id; every one named here has passed the accuracy gate
    /// ([`crate::gui::explorer::reviewed_note`], a test checks).
    Note(&'static str),
    /// An assumption of the A3 registry by id ([`crate::engine::assumptions::ASSUMPTIONS`]):
    /// its rationale.
    Assumption(&'static str),
    /// No note covers the row yet; the text says so.
    None(&'static str),
}

/// A guide's fixes, each an input path and a direction ([`fixes`]).
#[derive(Clone, Copy, Debug)]
pub enum Fixes {
    /// One list, whatever fails.
    Static(&'static [(&'static str, Direction)]),
    /// A list built from the terms of the check's verdict that fail (the temperature guide).
    PerTerm(fn(&DesignInputs, &DesignResults) -> Vec<(&'static str, Direction)>),
}

/// One check's guide.
#[derive(Clone, Copy, Debug)]
pub struct CheckGuide {
    /// The check's result path, a check of a [`DASHBOARD`] row, or [`FIT_CHECK`].
    pub check: &'static str,
    /// The guide's plain name: the heading of its block, and the Magnets fit row's label.
    pub label: &'static str,
    /// What is wrong in this design, from its inputs and results, every number live; `None`
    /// while the check passes.
    pub problem: fn(&DesignInputs, &DesignResults) -> Option<String>,
    /// Why the check matters: one plain sentence of at most 30 words.
    pub why: &'static str,
    /// The inputs to move and which way; [`fixes`] gives the ones to show for a design.
    pub fixes: Fixes,
    /// Where the row's "Why:" link leads.
    pub learn: Learn,
}

/// The Magnets fit row's check: no result holds it, so the GUI names it (a path no result
/// has, which [`crate::gui::dashboard::result_info`] does not find).
pub const FIT_CHECK: &str = "guidance.blocks_fit";

// The dashboard checks the guides cover (`DASHBOARD`'s check column; a test compares).
const HOT_MIN: &str = "metal.hot_min_check";
const CLEARANCE: &str = "metal.clearance_check";
const CUP_WALL: &str = "materials.cup_wall_check";
const TEMPERATURE: &str = "temperature.summary.verdict";
const CLAMP: &str = "clamps.recommended";
const SPACE_CLAIM: &str = "housing.space_claim_check";
const END_EFFECT: &str = "model.end_effect_check";

// Inputs several guides move.
const MAGNET_LENGTH: &str = "coupling.magnets.axial_length_mm";
const RING: &str = "coupling.inner_back_apothem_mm";
const POLES: &str = "coupling.npole";

/// Inputs the Size tab shows under its own name, by path: a fix on one names the Size-tab
/// control a user moves it with ("Raise Outside diameter", not the ring's radius).
pub const SIZE_TAB_ALIASES: [(&str, &str); 1] = [(RING, size_controls::DIAMETER)];

const HOT_MIN_GUIDE: CheckGuide = CheckGuide {
    check: HOT_MIN,
    label: "Hot-low torque",
    problem: hot_min_problem,
    why: "Magnets weaken as they warm, and production scatter makes some couplings weaker than nominal; hot-low is the pull-out torque after both, and it must reach the requirement.",
    // Not the gap: a smaller gap raises the torque but takes running clearance, whose own fix
    // opens the gap (both fail on the default design).
    fixes: Fixes::Static(&[(MAGNET_LENGTH, Direction::Raise)]),
    learn: Learn::Assumption("production_variation"),
};

const CLEARANCE_GUIDE: CheckGuide = CheckGuide {
    check: CLEARANCE,
    label: "Running clearance",
    problem: clearance_problem,
    why: "The sleeve and liner spin past each other across a small radial gap that must absorb runout, deflection, thermal growth and tolerances, or they rub.",
    fixes: Fixes::Static(&[
        ("metal.face_gap_mm", Direction::Raise),
        ("metal.sleeve_mm", Direction::Lower),
        ("metal.liner_mm", Direction::Lower),
    ]),
    learn: Learn::None("No note covers the running-clearance stack yet."),
};

const CUP_WALL_GUIDE: CheckGuide = CheckGuide {
    check: CUP_WALL,
    label: "Cup wall",
    problem: cup_wall_problem,
    why: "The steel ring behind the outer magnets carries each pole's flux to its neighbours; too thin a wall saturates, flux leaks, and torque falls below the prediction.",
    fixes: Fixes::Static(&[("metal.cup_wall_corner_mm", Direction::Raise)]),
    learn: Learn::Note("a5.low_saturation"),
};

const TEMPERATURE_GUIDE: CheckGuide = CheckGuide {
    check: TEMPERATURE,
    label: "Temperature limits",
    problem: temperature_problem,
    why: "Too hot, and the magnets can lose strength for good or the adhesive can weaken; ferrite is the exception, at risk when too cold.",
    fixes: Fixes::PerTerm(temperature_fixes),
    learn: Learn::Note("demagnetization"),
};

const CLAMP_GUIDE: CheckGuide = CheckGuide {
    check: CLAMP,
    label: "Shaft clamp",
    problem: clamp_problem,
    why: "The clamp holds by screw preload and friction, and must carry the most torque the coupling can pass, times a safety factor, with room for its screws.",
    fixes: Fixes::Static(&[
        ("clamps.boss_od_mm", Direction::Raise),
        ("clamps.clamp_length_mm", Direction::Raise),
    ]),
    learn: Learn::Note("clamp_preload"),
};

const SPACE_CLAIM_GUIDE: CheckGuide = CheckGuide {
    check: SPACE_CLAIM,
    label: "Space claim",
    problem: space_claim_problem,
    why: "The coupling must fit the envelope reserved for it: its diameter, its overall length and the length of the large-diameter bay.",
    // Which one helps depends on what sets the axis; the Problem names the axis.
    fixes: Fixes::Static(&[
        (MAGNET_LENGTH, Direction::Lower),
        ("metal.cap_od_mm", Direction::Lower),
        (RING, Direction::Lower),
    ]),
    learn: Learn::None("No note covers the space claim yet."),
};

const END_EFFECT_GUIDE: CheckGuide = CheckGuide {
    check: END_EFFECT,
    label: "End-effect model",
    problem: end_effect_problem,
    why: "Field fringes at the magnet ends, so the model trims a length set by the pole pitch; magnets that short leave no usable torque prediction.",
    // Not more poles: they shorten the pitch, but at the default ring 12 poles already do not
    // fit.
    fixes: Fixes::Static(&[(MAGNET_LENGTH, Direction::Raise)]),
    learn: Learn::Note("end_effect"),
};

const FIT_GUIDE: CheckGuide = CheckGuide {
    check: FIT_CHECK,
    label: "Magnets fit",
    problem: fit_problem,
    why: "Each ring holds one block per pole; at a given ring size, more blocks leave each a narrower place, and a block wider than its place does not fit.",
    fixes: Fixes::Static(&[(RING, Direction::Raise), (POLES, Direction::Lower)]),
    learn: Learn::None("No note covers block fit yet."),
};

/// The guides: the seven dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 8] = [
    HOT_MIN_GUIDE,
    CLEARANCE_GUIDE,
    CUP_WALL_GUIDE,
    TEMPERATURE_GUIDE,
    CLAMP_GUIDE,
    SPACE_CLAIM_GUIDE,
    END_EFFECT_GUIDE,
    FIT_GUIDE,
];

/// Where each dashboard row's "Why:" link leads, in [`DASHBOARD`]'s order (spec 3.4: every
/// row has one). A row with a check leads where its guide does; a value row to the reviewed
/// note that explains its equation, an assumption's rationale, or the reason there is none.
pub static ROW_LEARN: [(&str, Learn); DASHBOARD.len()] = [
    ("model.pullout_Nm", Learn::Note("pullout_angle")),
    ("model.pullout_20C_Nm", Learn::Note("br_temperature")),
    ("metal.torque_hot_low_Nm", HOT_MIN_GUIDE.learn),
    ("metal.hot_min_check", HOT_MIN_GUIDE.learn),
    // The cold pull-out with the variation allowance added, as the hot-low torque has it taken
    // off: the allowance's rationale states both.
    (
        "metal.torque_cold_high_Nm",
        Learn::Assumption("production_variation"),
    ),
    // The pull-out through the gearbox: the torque-angle curve's amplitude.
    (
        "model.gearbox_input_ripple_Nm",
        Learn::Note("pullout_angle"),
    ),
    (
        "model.cup_od_mm",
        Learn::None("No note covers the cup's outside diameter yet."),
    ),
    (
        "mass.total_g",
        Learn::None("No note covers the rotating mass yet."),
    ),
    ("metal.min_running_clearance_mm", CLEARANCE_GUIDE.learn),
    ("metal.clearance_check", CLEARANCE_GUIDE.learn),
    ("materials.cup_wall_check", CUP_WALL_GUIDE.learn),
    (
        "temperature.summary.governing_limit_C",
        TEMPERATURE_GUIDE.learn,
    ),
    (
        "temperature.summary.margin_hot_day_C",
        TEMPERATURE_GUIDE.learn,
    ),
    (TEMPERATURE, TEMPERATURE_GUIDE.learn),
    (CLAMP, CLAMP_GUIDE.learn),
    (SPACE_CLAIM, SPACE_CLAIM_GUIDE.learn),
    (END_EFFECT, END_EFFECT_GUIDE.learn),
];

/// Where the "Why:" link of the dashboard row at `path` leads ([`ROW_LEARN`]); `None` for a
/// path that is no dashboard row.
pub fn row_learn(path: &str) -> Option<Learn> {
    ROW_LEARN
        .iter()
        .find(|&&(row, _)| row == path)
        .map(|&(_, learn)| learn)
}

/// The guide of `check` (a dashboard check or [`FIT_CHECK`]); `None` for any other path.
pub fn guide(check: &str) -> Option<&'static CheckGuide> {
    CHECK_GUIDES.iter().find(|guide| guide.check == check)
}

/// The level of `guide`'s check in this design: the dashboard's badge of the check
/// ([`check_level`]), green where it gives none (a check that does not apply, such as the cup
/// wall without back iron, or one greyed while the end-effect model is out of range: the
/// end-effect guide carries that); for [`FIT_CHECK`] red when the blocks do not fit.
pub fn level(guide: &CheckGuide, inputs: &DesignInputs, results: &DesignResults) -> Level {
    level_of(guide.check, inputs, results)
}

/// [`level`] by check path.
fn level_of(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Level {
    if check == FIT_CHECK {
        if blocks_fit(&inputs.coupling, &results.model) {
            Level::Good
        } else {
            Level::Bad
        }
    } else {
        check_level(results, check).unwrap_or(Level::Good)
    }
}

/// Whether `check` fails or asks for a look in this design: a Problem is built only then.
fn failing(check: &str, inputs: &DesignInputs, results: &DesignResults) -> bool {
    level_of(check, inputs, results) != Level::Good
}

// ------------------------------------------------------------------ the fixes

/// The fixes of `guide` to show for this design, in order: its list, or for the temperature
/// guide the fixes of each failing term (none while the check passes), less any slider input
/// already at its slider's end that way, which could not help.
pub fn fixes(
    guide: &CheckGuide,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<(&'static str, Direction)> {
    let listed = match guide.fixes {
        Fixes::Static(list) => list.to_vec(),
        Fixes::PerTerm(build) => build(inputs, results),
    };
    listed
        .into_iter()
        .filter(|&(path, direction)| can_move(inputs, results, path, direction))
        .collect()
}

/// A number of a value: a number or an integer; `None` for a text or no value.
fn number(value: Value) -> Option<f64> {
    match value {
        Value::Num(x) => Some(x),
        Value::Int(i) => Some(i as f64),
        Value::Text(_) | Value::None => None,
    }
}

/// The number the input at `path` holds, or for an optional input left blank the result it
/// starts from ([`optional_seed`]: magnets of their parts' own length); `None` for a text.
fn current_number(inputs: &DesignInputs, results: &DesignResults, path: &str) -> Option<f64> {
    match inputs.get(path)? {
        Value::None => number(results.get(optional_seed(path)?)?),
        value => number(value),
    }
}

/// Whether the input at `path` can move in `direction`: a slider input not yet at its
/// slider's end that way (a typed value past the end can still come back); a choice always
/// can, and so can a number that is not one.
fn can_move(
    inputs: &DesignInputs,
    results: &DesignResults,
    path: &str,
    direction: Direction,
) -> bool {
    let range = InputCatalogue::get()
        .entry(path)
        .and_then(|entry| entry.meta.range);
    match (direction, range, current_number(inputs, results, path)) {
        (Direction::Raise, Some(range), Some(x)) => x.is_nan() || x < range.max,
        (Direction::Lower, Some(range), Some(x)) => x.is_nan() || x > range.min,
        _ => true,
    }
}

/// A fix's text: its direction, then the input's label as the inputs side shows it ("Raise
/// Minimum outer return ring wall"), or the Size tab's name for it ([`SIZE_TAB_ALIASES`]). A
/// label that another input's label equals or ends with ("Boss outside diameter" against
/// "Shaft boss outside diameter") is followed by its group's heading, so the fix names one
/// input: "Raise Boss outside diameter (Shaft clamps)". A path that is no input reads as
/// itself (a test keeps every guide's fixes on real inputs).
pub fn fix_text(path: &str, direction: Direction) -> String {
    format!("{} {}", direction.word(), fix_label(path))
}

/// The input's name in a fix ([`fix_text`]).
fn fix_label(path: &str) -> String {
    if let Some(&(_, alias)) = SIZE_TAB_ALIASES
        .iter()
        .find(|&&(aliased, _)| aliased == path)
    {
        return alias.to_owned();
    }
    let catalogue = InputCatalogue::get();
    let Some(entry) = catalogue.entry(path) else {
        return path.to_owned();
    };
    let label = entry.meta.label;
    let shared = catalogue
        .all()
        .any(|other| other.path != path && ends_with_label(other.meta.label, label));
    let heading = path
        .rsplit_once('.')
        .and_then(|(group, _)| section_label(group));
    match heading {
        Some(heading) if shared => format!("{label} ({heading})"),
        _ => label.to_owned(),
    }
}

/// Whether `other` is `label` or ends with it as whole words, ignoring case.
fn ends_with_label(other: &str, label: &str) -> bool {
    let Some(start) = other.len().checked_sub(label.len()) else {
        return false;
    };
    other.is_char_boundary(start)
        && other[start..].eq_ignore_ascii_case(label)
        && (start == 0 || other[..start].ends_with(' '))
}

// ------------------------------------------------------------------ the temperature terms

/// The magnet part of each ring: the weaker ring sets the magnet limit, so a hotter-rated
/// magnet must go in both.
const MAGNET_PARTS: [(&str, Direction); 2] = [
    (PART_PATHS[0], Direction::Change),
    (PART_PATHS[1], Direction::Change),
];

/// The adhesive: its limit and its cure temperature.
const ADHESIVE: [(&str, Direction); 1] = [("temperature.adhesive.selected", Direction::Change)];

/// Less heat at the slip-fault peak: a shorter fault, a better path out for the heat.
const COOLING: [(&str, Direction); 2] = [
    ("temperature.duty.fault_trip_s", Direction::Lower),
    ("temperature.thermal.conductance_W_K", Direction::Raise),
];

/// A term of the temperature verdict, in the verdict's order; the verdict passes when none
/// fails (a test checks). The hot-day margin is split by limit: the start must be below the
/// magnet limit and below the adhesive limit, and each has its own fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemperatureTerm {
    /// The hot-day start is not below the magnet limit.
    HotDayMagnet,
    /// The hot-day start is not below the adhesive limit.
    HotDayAdhesive,
    /// The slip-fault peak reaches the magnet limit.
    MagnetPeak,
    /// The slip-fault peak reaches the adhesive limit.
    AdhesivePeak,
    /// The adhesive cures less than [`CURE_MARGIN_MIN_C`] below the single-ring
    /// demagnetization onset.
    Cure,
    /// A ring whose coercivity falls on cooling (ferrite) is below its cold limit.
    Cold,
}

impl TemperatureTerm {
    const ALL: [TemperatureTerm; 6] = [
        TemperatureTerm::HotDayMagnet,
        TemperatureTerm::HotDayAdhesive,
        TemperatureTerm::MagnetPeak,
        TemperatureTerm::AdhesivePeak,
        TemperatureTerm::Cure,
        TemperatureTerm::Cold,
    ];

    /// Whether the term fails, by the verdict's own comparisons (a margin that is not a
    /// number fails too).
    fn fails(self, results: &DesignResults) -> bool {
        let t = &results.temperature;
        let start = t.summary.hot_day_start_C;
        match self {
            TemperatureTerm::HotDayMagnet => not_positive(t.summary.magnet_limit_C - start),
            TemperatureTerm::HotDayAdhesive => not_positive(t.summary.adhesive_limit_C - start),
            TemperatureTerm::MagnetPeak => not_positive(t.magnet_life.margin_limit_C),
            TemperatureTerm::AdhesivePeak => not_positive(t.adhesive_life.margin_C),
            TemperatureTerm::Cure => {
                let cure = t.summary.cure_margin_C;
                cure.is_nan() || cure < CURE_MARGIN_MIN_C
            }
            TemperatureTerm::Cold => {
                check_level(results, "temperature.demag.cold_check") == Some(Level::Bad)
            }
        }
    }

    /// The term's sentence, its numbers from the results and inputs.
    fn sentence(self, inputs: &DesignInputs, results: &DesignResults) -> String {
        let start = result_text(results, "temperature.summary.hot_day_start_C");
        let magnet_limit = result_text(results, "temperature.summary.magnet_limit_C");
        let adhesive_limit = result_text(results, "temperature.summary.adhesive_limit_C");
        let magnet_peak = result_text(results, "temperature.magnet_life.peak_C");
        let bond_peak = result_text(results, "temperature.adhesive_life.peak_C");
        match self {
            TemperatureTerm::HotDayMagnet => {
                format!("Hot-day start {start} is not below the {magnet_limit} magnet limit.")
            }
            TemperatureTerm::HotDayAdhesive => {
                format!("Hot-day start {start} is not below the {adhesive_limit} adhesive limit.")
            }
            TemperatureTerm::MagnetPeak => {
                format!(
                    "The slip-fault peak of {magnet_peak} reaches the {magnet_limit} magnet limit."
                )
            }
            TemperatureTerm::AdhesivePeak => format!(
                "The slip-fault peak of {bond_peak} reaches the {adhesive_limit} adhesive limit."
            ),
            TemperatureTerm::Cure => format!(
                "The adhesive cures at {}, {} below the {} single-ring demagnetization onset; the margin must be at least {}.",
                result_text(results, "temperature.adhesive.cure_C"),
                result_text(results, "temperature.summary.cure_margin_C"),
                result_text(results, "temperature.demag.onset_single_ring_C"),
                number_text(CURE_MARGIN_MIN_C, "temperature.summary.cure_margin_C"),
            ),
            // The cold limit comes from the stored reverse fields, which do not follow the
            // grade (spec section 8): the sentence says so rather than quote it as ferrite's.
            TemperatureTerm::Cold => format!(
                "The minimum magnet temperature {} is below the {} cold demagnetization limit (from stored 3D fields that do not yet follow the magnet grade).",
                input_text(inputs, "metal.min_temp_C"),
                result_text(results, "temperature.demag.cold_limit_C"),
            ),
        }
    }

    /// The fixes that move the term toward passing.
    const fn fixes(self) -> &'static [(&'static str, Direction)] {
        match self {
            TemperatureTerm::HotDayMagnet | TemperatureTerm::Cold => &MAGNET_PARTS,
            TemperatureTerm::HotDayAdhesive | TemperatureTerm::Cure => &ADHESIVE,
            TemperatureTerm::MagnetPeak | TemperatureTerm::AdhesivePeak => &COOLING,
        }
    }
}

/// Whether a temperature margin fails the verdict's `margin > 0` (NaN fails it too).
fn not_positive(margin: f64) -> bool {
    margin.is_nan() || margin <= 0.0
}

/// The temperature terms this design fails, in the verdict's order.
fn failing_terms(results: &DesignResults) -> Vec<TemperatureTerm> {
    TemperatureTerm::ALL
        .into_iter()
        .filter(|term| term.fails(results))
        .collect()
}

/// The fixes of `terms`, in their order, each input once.
fn term_fixes(terms: impl IntoIterator<Item = TemperatureTerm>) -> Vec<(&'static str, Direction)> {
    let mut fixes: Vec<(&'static str, Direction)> = Vec::new();
    for term in terms {
        for &fix in term.fixes() {
            if !fixes.contains(&fix) {
                fixes.push(fix);
            }
        }
    }
    fixes
}

/// The temperature guide's fixes ([`Fixes::PerTerm`]): the failing terms' ([`term_fixes`]).
/// The terms read the results only.
fn temperature_fixes(
    _inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<(&'static str, Direction)> {
    term_fixes(failing_terms(results))
}

// ------------------------------------------------------------------ the design status

/// The design status line's start for the defaults (spec G-6: they stay the workbook's example).
pub const WORKBOOK_EXAMPLE: &str = "Workbook example design";

/// The design status line's start for any other design.
pub const YOUR_DESIGN: &str = "Your design";

/// How the design stands against the guides' checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesignStatus {
    /// The checks that fail (red).
    pub bad: usize,
    /// The checks that ask for a look (amber), counted apart from the failures.
    pub caution: usize,
    /// Every guide's check: [`CHECK_GUIDES`]'s length.
    pub total: usize,
    /// Whether the inputs are the defaults (the workbook's example design).
    pub is_default: bool,
}

/// The design status of `inputs` and their `results`, counted live from the guides' levels.
pub fn design_status(inputs: &DesignInputs, results: &DesignResults) -> DesignStatus {
    let count = |wanted: Level| {
        CHECK_GUIDES
            .iter()
            .filter(|guide| level(guide, inputs, results) == wanted)
            .count()
    };
    DesignStatus {
        bad: count(Level::Bad),
        caution: count(Level::Caution),
        total: CHECK_GUIDES.len(),
        is_default: *inputs == DesignInputs::default(),
    }
}

/// The design status line: "Workbook example design: fails 3 of 8 checks", "Your design: fails
/// 2 of 8 checks", "Your design: passes all 8 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look.
pub fn design_status_text(status: &DesignStatus) -> String {
    let design = if status.is_default {
        WORKBOOK_EXAMPLE
    } else {
        YOUR_DESIGN
    };
    let checks = if status.bad == 0 {
        format!("passes all {} checks", status.total)
    } else {
        format!("fails {} of {} checks", status.bad, status.total)
    };
    let cautions = match status.caution {
        0 => String::new(),
        1 => ", 1 caution".to_owned(),
        n => format!(", {n} cautions"),
    };
    format!("{design}: {checks}{cautions}")
}

// ------------------------------------------------------------------ plain words

/// What [`plain_text_violations`] looks for, in its order: a workbook cell reference
/// (`C\d{2,3}`), a correction id (`E\d{1,2}` as a whole id), the word "workbook" in any case
/// (but in [`WORKBOOK_EXAMPLE`]'s exact phrase), and three names of the engine's provenance.
pub const BANNED: [&str; 6] = [
    "C\\d{2,3}",
    "E\\d{1,2}",
    "workbook",
    "Addendum",
    "Rust-only",
    "Calculator!",
];

/// The phrase that may say "workbook": the design status line's.
const ALLOWED_WORKBOOK: &str = "Workbook example";

/// The [`BANNED`] patterns `text` contains, in [`BANNED`]'s order, each once. The guidance
/// text is for engineers who are not reading the workbook: the tests of every task of the
/// slice hold its strings to an empty list.
pub fn plain_text_violations(text: &str) -> Vec<&'static str> {
    let chars: Vec<char> = text.chars().collect();
    let digits_after = |i: usize| {
        chars[i + 1..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count()
    };
    // C followed by two or more digits holds C\d{2,3}; E with one or two digits and then no
    // letter, digit or underscore is E\d{1,2}\b.
    let cell = chars
        .iter()
        .enumerate()
        .any(|(i, &c)| c == 'C' && digits_after(i) >= 2);
    let correction = chars.iter().enumerate().any(|(i, &c)| {
        let digits = digits_after(i);
        c == 'E'
            && (1..=2).contains(&digits)
            && chars
                .get(i + 1 + digits)
                .is_none_or(|&next| !next.is_alphanumeric() && next != '_')
    });
    let workbook =
        text.to_lowercase().matches("workbook").count() > text.matches(ALLOWED_WORKBOOK).count();
    let hits = [
        cell,
        correction,
        workbook,
        text.contains("Addendum"),
        text.contains("Rust-only"),
        text.contains("Calculator!"),
    ];
    BANNED
        .iter()
        .zip(hits)
        .filter_map(|(&pattern, hit)| hit.then_some(pattern))
        .collect()
}

// ------------------------------------------------------------------ the Problem builders

/// The result at `path` as the readouts show it: four significant digits, then its unit.
fn result_text(results: &DesignResults, path: &str) -> String {
    let unit = result_info(path).map_or("", |info| info.meta.unit);
    with_unit(
        format_value(&results.get(path).unwrap_or(Value::None)),
        unit,
    )
}

/// The input at `path` as a readout shows it: four significant digits, then its unit.
fn input_text(inputs: &DesignInputs, path: &str) -> String {
    let unit = InputCatalogue::get()
        .entry(path)
        .map_or("", |entry| entry.meta.unit);
    with_unit(format_value(&inputs.get(path).unwrap_or(Value::None)), unit)
}

/// A number worked out here from results and inputs (no result holds it), in the unit of the
/// result at `unit_of`.
fn number_text(x: f64, unit_of: &str) -> String {
    let unit = result_info(unit_of).map_or("", |info| info.meta.unit);
    with_unit(format_value(&Value::Num(x)), unit)
}

/// A fraction as a percentage: four significant digits without trailing zeros (0.15 gives
/// "15 %", 3.4449 gives "344.5 %").
fn percent_text(fraction: f64) -> String {
    let text = format_value(&Value::Num(fraction * 100.0));
    let digits = if text.contains('.') && !text.contains('e') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text.as_str()
    };
    format!("{digits} %")
}

/// `sentences` as one text, or `fallback` when there are none: a failing check always names
/// at least one, and the fallback keeps its block from going blank if a verdict and its numbers
/// ever disagree.
fn joined_or(sentences: Vec<String>, fallback: &str) -> String {
    if sentences.is_empty() {
        fallback.to_owned()
    } else {
        sentences.join(" ")
    }
}

fn hot_min_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(HOT_MIN, inputs, results).then(|| {
        format!(
            "Hot-low torque {} is below the required {}: the pull-out of {} at {}, less the {} variation allowance.",
            result_text(results, "metal.torque_hot_low_Nm"),
            input_text(inputs, "metal.required_min_Nm"),
            result_text(results, "model.pullout_Nm"),
            input_text(inputs, "coupling.op_temp_C"),
            percent_text(inputs.metal.variation),
        )
    })
}

fn clearance_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(CLEARANCE, inputs, results).then(|| {
        format!(
            "Sleeve-to-liner gap {} less {} of movement allowances leaves {}, below the {} target.",
            result_text(results, "metal.sleeve_liner_clearance_mm"),
            result_text(results, "metal.adverse_movement_mm"),
            result_text(results, "metal.min_running_clearance_mm"),
            input_text(inputs, "metal.residual_target_mm"),
        )
    })
}

fn cup_wall_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(CUP_WALL, inputs, results).then(|| {
        format!(
            "Cup wall at the pocket corners is {}; the flux it carries needs {}.",
            result_text(results, "materials.cup_wall_corner_mm"),
            result_text(results, "materials.backiron_thickness_needed_mm"),
        )
    })
}

fn temperature_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(TEMPERATURE, inputs, results).then(|| {
        let sentences = failing_terms(results)
            .into_iter()
            .map(|term| term.sentence(inputs, results))
            .collect();
        joined_or(
            sentences,
            "A temperature limit is not met: see the temperature results.",
        )
    })
}

fn clamp_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(CLAMP, inputs, results).then(|| {
        format!(
            "No screw size from {} to {} fits the boss and the clamp length with enough screws to hold {}: the cold-high torque {} times a safety factor of {}.",
            SCREW_SIZES[0].name,
            SCREW_SIZES[SCREW_SIZES.len() - 1].name,
            result_text(results, "clamps.required_Nm"),
            result_text(results, "clamps.max_torque_Nm"),
            input_text(inputs, "clamps.safety_factor"),
        )
    })
}

fn space_claim_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(SPACE_CLAIM, inputs, results).then(|| {
        let h = &results.housing;
        // (axis, its overshoot, the overshoot's path, the dimension's, the claim's, the claim's name)
        let axes = [
            (
                "Rotating diameter",
                h.diameter_overshoot_mm,
                "housing.diameter_overshoot_mm",
                "metal.rotating_od_mm",
                "metal.max_diameter_mm",
                "claim",
            ),
            (
                "Overall length",
                h.length_overshoot_mm,
                "housing.length_overshoot_mm",
                "metal.axial_stack_mm",
                "metal.max_overall_axial_mm",
                "claim",
            ),
            (
                "Large-diameter stack",
                h.bay_overshoot_mm,
                "housing.bay_overshoot_mm",
                "metal.large_dia_stack_mm",
                "metal.max_large_dia_axial_mm",
                "bay",
            ),
        ];
        let sentences = axes
            .iter()
            .filter_map(|&(axis, over, over_path, dimension, claim, name)| {
                if over > 0.0 {
                    Some(format!(
                        "{axis} {} is {} over the {} {name}.",
                        result_text(results, dimension),
                        result_text(results, over_path),
                        input_text(inputs, claim),
                    ))
                } else if over.is_nan() {
                    Some(format!(
                        "{axis} cannot be checked: it or its {name} is not a number."
                    ))
                } else {
                    None
                }
            })
            .collect();
        joined_or(sentences, "The coupling is outside its space claim.")
    })
}

fn end_effect_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(END_EFFECT, inputs, results).then(|| {
        format!(
            "Magnets {} long with a pole pitch of {} give an end-effect factor of {}; the model gives no torque until the magnets are longer than {} and is rough just above that.",
            result_text(results, "model.active_length_mm"),
            result_text(results, "model.pole_pitch_mm"),
            result_text(results, "model.f_end"),
            // f_end = 1 - c_end * pitch / L is positive exactly when L > c_end * pitch.
            number_text(
                inputs.coupling.c_end * results.model.pole_pitch_mm,
                "model.pole_pitch_mm"
            ),
        )
    })
}

fn fit_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(FIT_CHECK, inputs, results).then(|| {
        let m = &results.model;
        let n = inputs.coupling.npole;
        let mut sentences = vec![format!("{n} blocks per ring do not fit at this ring size.")];
        // The comparisons of `blocks_fit`: flat blocks against their flats, arcs by their
        // share of the pitch circle at the magnet mid-radius.
        if inputs.coupling.faceted == 1 {
            let rings = [
                (
                    "Inner",
                    m.inner_flat_width_mm < m.inner_width_mm,
                    "model.inner_width_mm",
                    "model.inner_flat_width_mm",
                ),
                (
                    "Outer",
                    m.outer_flat_width_mm < m.outer_width_mm,
                    "model.outer_width_mm",
                    "model.outer_flat_width_mm",
                ),
            ];
            for (ring, narrow, width, flat) in rings {
                if narrow {
                    sentences.push(format!(
                        "{ring} blocks are {} wide on {} flats.",
                        result_text(results, width),
                        result_text(results, flat),
                    ));
                }
            }
        } else {
            let shares = [
                (
                    "inner",
                    pitch_share(
                        m.inner_width_mm,
                        inputs.coupling.inner_back_apothem_mm,
                        m.inner_thickness_mm,
                        n as f64,
                    ),
                ),
                (
                    "outer",
                    pitch_share(
                        m.outer_width_mm,
                        m.outer_face_apothem_mm,
                        m.outer_thickness_mm,
                        n as f64,
                    ),
                ),
            ];
            for (ring, share) in shares {
                if share > 1.0 {
                    sentences.push(format!(
                        "The {n} {ring} blocks together need {} of the circle at their mid-radius.",
                        percent_text(share)
                    ));
                }
            }
        }
        sentences.join(" ")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::explain::notes;
    use crate::engine::grades::GRADES;
    use crate::engine::library::MAGNET_LIBRARY;
    use crate::engine::meta::NumOrText;
    use crate::gui::dashboard::{CHECKS, DASHBOARD};
    use crate::gui::explorer::reviewed_note;
    use crate::gui::pickers::GRADE_PATHS;
    use crate::gui::test_support::{assert_glyphs, passing_design, short_magnets, with_poles};
    use std::collections::BTreeSet;

    // ------------------------------------------------------------------ the test designs

    /// `inputs` with `edit` applied.
    fn with(mut inputs: DesignInputs, edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        edit(&mut inputs);
        inputs
    }

    /// The defaults with `edit` applied.
    fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        with(DesignInputs::default(), edit)
    }

    /// [`passing_design`] with `edit` applied: it fails what `edit` breaks, so a fix that turns
    /// another check red shows.
    fn breaking(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        with(passing_design(), edit)
    }

    /// A hot day: 80 °C ambient and a 60 °C driving rise start the magnets at 140 °C, above
    /// both the magnet and the adhesive limit, with the hottest-rated library part in use.
    fn hot_day(i: &mut DesignInputs) {
        i.temperature.duty.hot_ambient_C = 80.0;
        i.temperature.duty.driving_rise_C = 60.0;
    }

    /// Library N42 blocks (rated 80 °C) in both rings: their magnet limit is below the hot-day
    /// start.
    fn n42_parts(i: &mut DesignInputs) {
        i.coupling.magnets.part_inner = "B842".to_owned();
        i.coupling.magnets.part_outer = "B842".to_owned();
    }

    /// A long unbroken slip at a high slip speed: only the fault peak passes the magnet limit.
    fn slip_fault(i: &mut DesignInputs) {
        i.temperature.duty.fault_trip_s = 60.0;
        i.metal.slip_rpm = 6000.0;
    }

    /// The fourth adhesive: its 60 °C limit is below the hot-day start.
    fn cool_adhesive(i: &mut DesignInputs) {
        i.temperature.adhesive.selected = 4;
    }

    /// The second adhesive (cured at 120 °C) with a stored single-ring reverse field that puts
    /// the single-ring onset about 6 °C above the cure: only the cure margin fails.
    fn hot_cure(i: &mut DesignInputs) {
        i.temperature.adhesive.selected = 2;
        i.temperature.demag.h_rev_single_ring_kA_m = 700.0;
    }

    /// Manual Y30 ferrite blocks: the cold demagnetization check fails.
    fn ferrite(i: &mut DesignInputs) {
        i.coupling.magnets.part_inner.clear();
        i.coupling.magnets.part_outer.clear();
        i.coupling.magnets.grade_inner = "Y30".to_owned();
        i.coupling.magnets.grade_outer = "Y30".to_owned();
    }

    /// No screw fits a 12 mm boss.
    fn small_boss(i: &mut DesignInputs) {
        i.clamps.boss_od_mm = 12.0;
    }

    /// No screw fits a 3 mm clamp.
    fn short_clamp(i: &mut DesignInputs) {
        i.clamps.clamp_length_mm = 3.0;
    }

    /// 50.8 mm magnets: past the overall length and the large-diameter bay.
    fn long_magnets(i: &mut DesignInputs) {
        i.coupling.magnets.axial_length_mm = Some(50.8);
    }

    /// A 44 mm cap: past the diameter claim, the cap setting the diameter.
    fn large_cap(i: &mut DesignInputs) {
        i.metal.cap_od_mm = 44.0;
    }

    /// A 12 mm ring: past the diameter claim, the cup setting the diameter.
    fn large_ring(i: &mut DesignInputs) {
        i.coupling.inner_back_apothem_mm = 12.0;
    }

    /// A diameter claim that is not a number: the space claim asks for a look.
    fn unknown_claim(i: &mut DesignInputs) {
        i.metal.max_diameter_mm = f64::NAN;
    }

    /// True arcs instead of flat blocks.
    fn arcs(i: &mut DesignInputs) {
        i.coupling.faceted = 0;
    }

    /// No back iron: the cup wall check does not apply.
    fn no_back_iron(i: &mut DesignInputs) {
        i.coupling.backiron = 0;
    }

    /// No sleeve and no liner: of the running clearance's fixes only the gap can still move.
    fn bare_clearance(i: &mut DesignInputs) {
        i.metal.sleeve_mm = 0.0;
        i.metal.liner_mm = 0.0;
    }

    /// Every design the tests use: the defaults (they fail three checks), a passing design,
    /// and for every check designs that fail it, most of them the passing design with one
    /// check broken.
    fn designs() -> Vec<(&'static str, DesignInputs)> {
        vec![
            ("defaults", DesignInputs::default()),
            ("passing", passing_design()),
            ("hot_day", breaking(hot_day)),
            ("n42_parts", breaking(n42_parts)),
            ("slip_fault", breaking(slip_fault)),
            ("cool_adhesive", breaking(cool_adhesive)),
            ("hot_cure", breaking(hot_cure)),
            ("ferrite", breaking(ferrite)),
            ("small_boss", breaking(small_boss)),
            ("short_clamp", breaking(short_clamp)),
            ("long_magnets", breaking(long_magnets)),
            ("large_cap", breaking(large_cap)),
            ("large_ring", breaking(large_ring)),
            ("unknown_claim", breaking(unknown_claim)),
            ("short_magnets", short_magnets()),
            ("twelve_poles", breaking(|i| i.coupling.npole = 12)),
            ("crowded", breaking(|i| i.coupling.npole = 40)),
            (
                "crowded_arcs",
                breaking(|i| {
                    i.coupling.npole = 40;
                    arcs(i);
                }),
            ),
            ("no_back_iron", breaking(no_back_iron)),
            ("bare_clearance", design(bare_clearance)),
        ]
    }

    /// The Problem of `check` for `inputs`.
    fn problem(check: &str, inputs: &DesignInputs) -> Option<String> {
        (guide(check).unwrap().problem)(inputs, &compute_all(inputs))
    }

    /// Every fix `g` can show: its list, or for the temperature guide every term's fixes.
    fn possible_fixes(g: &CheckGuide) -> Vec<(&'static str, Direction)> {
        match g.fixes {
            Fixes::Static(list) => list.to_vec(),
            Fixes::PerTerm(_) => {
                assert_eq!(
                    g.check, TEMPERATURE,
                    "only the temperature guide fixes per term"
                );
                term_fixes(TemperatureTerm::ALL)
            }
        }
    }

    /// Every text the guides draw: labels, Whys, the reasons of the links without a note, every
    /// fix, every Problem of every test design and its status line.
    fn every_text() -> Vec<String> {
        let mut texts = Vec::new();
        for g in &CHECK_GUIDES {
            texts.push(g.label.to_owned());
            texts.push(g.why.to_owned());
            for (path, direction) in possible_fixes(g) {
                texts.push(fix_text(path, direction));
            }
        }
        for (_, learn) in &ROW_LEARN {
            if let Learn::None(reason) = learn {
                texts.push((*reason).to_owned());
            }
        }
        for (_, inputs) in designs() {
            let results = compute_all(&inputs);
            for g in &CHECK_GUIDES {
                texts.extend((g.problem)(&inputs, &results));
            }
            texts.push(design_status_text(&design_status(&inputs, &results)));
        }
        texts
    }

    // ------------------------------------------------------------------ the guides and links

    #[test]
    fn the_guides_cover_every_dashboard_check_and_the_fit_row() {
        let dashboard: BTreeSet<&str> = DASHBOARD.iter().filter_map(|&(_, check)| check).collect();
        let mut guided: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.check).collect();
        assert_eq!(guided.len(), CHECK_GUIDES.len(), "one guide per check");
        assert!(guided.remove(FIT_CHECK), "the Magnets fit row has a guide");
        assert_eq!(guided, dashboard);
        for g in &CHECK_GUIDES {
            // Every guide names a real check: a design check of the results, or the fit row.
            assert!(
                g.check == FIT_CHECK || CHECKS.contains(&g.check),
                "{}",
                g.check
            );
            assert_eq!(guide(g.check).map(|found| found.check), Some(g.check));
            assert!(!g.label.is_empty() && !g.why.is_empty(), "{}", g.check);
        }
        // The fit row's path is no result, and nothing but a check has a guide.
        assert!(result_info(FIT_CHECK).is_none());
        assert!(guide("model.pullout_Nm").is_none());
        assert!(guide("").is_none());
        let labels: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.label).collect();
        assert_eq!(labels.len(), CHECK_GUIDES.len(), "labels are distinct");
        assert_eq!(guide(FIT_CHECK).unwrap().label, "Magnets fit");
    }

    #[test]
    fn every_why_is_one_short_sentence() {
        for g in &CHECK_GUIDES {
            let words = g.why.split_whitespace().count();
            assert!(words <= 30, "{}: {words} words", g.check);
            let body = g.why.strip_suffix('.').unwrap_or_else(|| {
                panic!("{}: a Why ends with a full stop", g.check);
            });
            assert!(!body.contains(". "), "{}: one sentence", g.check);
        }
    }

    /// The fix inputs that are model assumptions, each a design lever all the same: the heat
    /// path to the housing (fins, a more conductive housing) sets the conductance, whose value
    /// stays a placeholder until it is measured.
    const ASSUMPTION_LEVERS: [&str; 1] = ["temperature.thermal.conductance_W_K"];

    #[test]
    fn every_fix_is_a_design_input_with_a_slider_or_choices() {
        let catalogue = InputCatalogue::get();
        for g in &CHECK_GUIDES {
            let possible = possible_fixes(g);
            assert!(!possible.is_empty(), "{}: a guide names a fix", g.check);
            let paths: BTreeSet<&str> = possible.iter().map(|&(path, _)| path).collect();
            assert_eq!(paths.len(), possible.len(), "{}: a fix once", g.check);
            for (path, direction) in possible {
                let entry = catalogue
                    .entry(path)
                    .unwrap_or_else(|| panic!("{}: {path} is no input", g.check));
                let meta = entry.meta;
                match direction {
                    Direction::Raise | Direction::Lower => assert!(
                        meta.range.is_some(),
                        "{}: {path} has no slider to raise or lower",
                        g.check
                    ),
                    Direction::Change => assert!(
                        !meta.choices.is_empty()
                            || PART_PATHS.contains(&path)
                            || GRADE_PATHS.contains(&path),
                        "{}: {path} has no choices to change",
                        g.check
                    ),
                }
                assert!(
                    !meta.assumption || ASSUMPTION_LEVERS.contains(&path),
                    "{}: {path} is a model assumption, not a design change",
                    g.check
                );
            }
        }
    }

    #[test]
    fn every_learn_link_resolves() {
        let guides = CHECK_GUIDES.iter().map(|g| (g.check, g.learn));
        for (what, learn) in guides.chain(ROW_LEARN.iter().copied()) {
            match learn {
                Learn::Note(id) => assert!(
                    reviewed_note(id).is_some(),
                    "{what}: note {id} is missing or not reviewed"
                ),
                Learn::Assumption(id) => assert!(
                    ASSUMPTIONS.iter().any(|a| a.id == id),
                    "{what}: no assumption {id}"
                ),
                Learn::None(reason) => assert!(!reason.trim().is_empty(), "{what}: a reason"),
            }
        }
    }

    #[test]
    fn every_dashboard_row_has_a_why_link() {
        let rows: Vec<&str> = ROW_LEARN.iter().map(|&(path, _)| path).collect();
        let dashboard: Vec<&str> = DASHBOARD.iter().map(|&(path, _)| path).collect();
        assert_eq!(rows, dashboard, "one link per dashboard row, in its order");
        for &(path, check) in &DASHBOARD {
            let learn = row_learn(path).unwrap_or_else(|| panic!("{path}: no Why link"));
            match check {
                // A row with a check leads where its guide does.
                Some(check) => assert_eq!(learn, guide(check).unwrap().learn, "{path}"),
                // A value row leads to the reviewed note that explains its equation, if any.
                None => {
                    if let Some(note) = notes::note_for(path) {
                        assert_eq!(learn, Learn::Note(note.id), "{path}");
                    }
                }
            }
        }
        assert_eq!(row_learn("no.such"), None);
    }

    // ------------------------------------------------------------------ the fix texts

    #[test]
    fn fix_text_names_the_input_by_its_label() {
        assert_eq!(
            fix_text("metal.cup_wall_corner_mm", Direction::Raise),
            "Raise Minimum outer return ring wall"
        );
        assert_eq!(
            fix_text("coupling.npole", Direction::Lower),
            "Lower Number of poles per ring"
        );
        assert_eq!(
            fix_text("coupling.magnets.part_inner", Direction::Change),
            "Change Inner magnet part"
        );
        // The Size tab's name for the ring radius: the control a user moves it with.
        assert_eq!(
            fix_text("coupling.inner_back_apothem_mm", Direction::Raise),
            "Raise Outside diameter"
        );
        assert_eq!(
            fix_text("coupling.inner_back_apothem_mm", Direction::Lower),
            "Lower Outside diameter"
        );
        // A label another input's label equals or ends with names its group: the clamp's boss
        // is not the drawn shaft boss.
        assert_eq!(
            fix_text("clamps.boss_od_mm", Direction::Raise),
            "Raise Boss outside diameter (Shaft clamps)"
        );
        assert_eq!(
            fix_text("metal.boss_od_mm", Direction::Raise),
            "Raise Shaft boss outside diameter"
        );
        assert_eq!(
            fix_text("coupling.mu0", Direction::Lower),
            "Lower Vacuum permeability (Coupling)"
        );
        assert_eq!(
            fix_text("calibration.mu0", Direction::Lower),
            "Lower Vacuum permeability (Calibration)"
        );
        // Every fix: its direction, then its input's label (the alias, or with its group).
        let catalogue = InputCatalogue::get();
        for g in &CHECK_GUIDES {
            for (path, direction) in possible_fixes(g) {
                let label = SIZE_TAB_ALIASES
                    .iter()
                    .find(|&&(aliased, _)| aliased == path)
                    .map_or(catalogue.entry(path).unwrap().meta.label, |&(_, alias)| {
                        alias
                    });
                let text = fix_text(path, direction);
                assert!(
                    text.starts_with(&format!("{} {label}", direction.word())),
                    "{text}"
                );
            }
        }
        // A path that is no input reads as itself rather than failing the frame.
        assert_eq!(fix_text("no.such", Direction::Raise), "Raise no.such");
    }

    #[test]
    fn every_input_has_its_own_fix_text() {
        // Two inputs never read alike in a fix, so a fix names one input: checked over every
        // input, not only today's fixes.
        let catalogue = InputCatalogue::get();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for entry in catalogue.all() {
            let text = fix_text(&entry.path, Direction::Raise);
            assert!(seen.insert(text.clone()), "{}: {text} twice", entry.path);
        }
    }

    // ------------------------------------------------------------------ the fixes help

    /// One term of a check's verdict: its margin, larger is better, and whether it passes as
    /// the verdict compares it.
    #[derive(Clone, Copy, Debug)]
    struct Term {
        name: &'static str,
        margin: f64,
        passes: bool,
    }

    /// The clamp's margin for the fix tests: minus the screw sizes' total shortfall on the two
    /// slacks its fixes act on, the wall outside each screw hole (the boss) and the axial room
    /// for the screws each size needs (the clamp length). The verdict itself, whether some
    /// size works, moves with no single slider step.
    fn clamp_margin(inputs: &DesignInputs, results: &DesignResults) -> f64 {
        let ci = &inputs.clamps;
        let shortfall: f64 = results
            .clamps
            .table
            .iter()
            .zip(SCREW_SIZES)
            .map(|(row, size)| {
                let wall = (ci.wall_out_mm - row.wall_out_mm).max(0.0);
                let room = ci.clamp_length_mm - 2.0 * ci.axial_margin_mm - (size.head_mm + 0.5);
                let length = ((row.screws_needed - 1) as f64 * row.pitch_axial_mm - room).max(0.0);
                wall + length
            })
            .sum();
        -shortfall
    }

    /// The terms of `check`'s verdict in a fixed order (two designs' lists pair up).
    fn terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<Term> {
        let term = |name, margin: f64, passes| Term {
            name,
            margin,
            passes,
        };
        let at_least_zero = |name, margin: f64| term(name, margin, margin >= 0.0);
        let (m, md, t) = (&results.model, &results.metal, &results.temperature);
        match check {
            HOT_MIN => vec![at_least_zero(
                "hot-low torque",
                md.torque_hot_low_Nm - inputs.metal.required_min_Nm,
            )],
            CLEARANCE => vec![at_least_zero(
                "running clearance",
                md.min_running_clearance_mm - inputs.metal.residual_target_mm,
            )],
            CUP_WALL => vec![at_least_zero(
                "cup wall",
                results.materials.cup_wall_corner_mm
                    - results.materials.backiron_thickness_needed_mm,
            )],
            TEMPERATURE => {
                let above_zero = |name, margin: f64| term(name, margin, margin > 0.0);
                let start = t.summary.hot_day_start_C;
                let cold = match t.demag.cold_limit_C {
                    NumOrText::Num(limit) => inputs.metal.min_temp_C - limit,
                    NumOrText::Text(_) => f64::INFINITY,
                };
                vec![
                    above_zero("hot day, magnets", t.summary.magnet_limit_C - start),
                    above_zero("hot day, adhesive", t.summary.adhesive_limit_C - start),
                    above_zero("peak, magnets", t.magnet_life.margin_limit_C),
                    above_zero("peak, adhesive", t.adhesive_life.margin_C),
                    at_least_zero("cure", t.summary.cure_margin_C - CURE_MARGIN_MIN_C),
                    at_least_zero("cold", cold),
                ]
            }
            CLAMP => vec![term(
                "screw size",
                clamp_margin(inputs, results),
                results.clamps.index > 0,
            )],
            SPACE_CLAIM => vec![
                at_least_zero("diameter", md.diameter_reserve_mm),
                at_least_zero("overall length", md.axial_reserve_mm),
                at_least_zero("large-diameter bay", md.large_dia_reserve_mm),
            ],
            END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0)],
            FIT_CHECK if inputs.coupling.faceted == 1 => vec![
                at_least_zero("inner ring", m.inner_flat_width_mm - m.inner_width_mm),
                at_least_zero("outer ring", m.outer_flat_width_mm - m.outer_width_mm),
            ],
            FIT_CHECK => {
                let n = inputs.coupling.npole as f64;
                let inner = pitch_share(
                    m.inner_width_mm,
                    inputs.coupling.inner_back_apothem_mm,
                    m.inner_thickness_mm,
                    n,
                );
                let outer = pitch_share(
                    m.outer_width_mm,
                    m.outer_face_apothem_mm,
                    m.outer_thickness_mm,
                    n,
                );
                vec![
                    at_least_zero("inner ring", 1.0 - inner),
                    at_least_zero("outer ring", 1.0 - outer),
                ]
            }
            _ => panic!("no terms for {check}"),
        }
    }

    /// The margin of the term `name` of `check`.
    fn margin(check: &str, name: &str, inputs: &DesignInputs, results: &DesignResults) -> f64 {
        terms(check, inputs, results)
            .into_iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{check} has no term {name}"))
            .margin
    }

    /// `inputs` with the input at `path` set to the number `x`, as an integer for an integer
    /// input.
    fn set_number(inputs: &DesignInputs, path: &str, x: f64) -> DesignInputs {
        let entry = InputCatalogue::get().entry(path).unwrap();
        let value = match entry.default {
            Value::Int(_) => Value::Int(x.round() as i64),
            _ => Value::Num(x),
        };
        let mut moved = inputs.clone();
        moved.set(path, value).unwrap();
        moved
    }

    /// The designs one move of a fix reaches from `inputs`: one slider step in its direction
    /// (none at the slider's end), or each other choice of a selector. A magnet part takes one
    /// magnet for both rings, a library part or the custom block in a grade (the part
    /// picker's "Custom dimensions" opens the grade picker): the weaker ring sets a limit both
    /// rings share, so the Fix names both parts and one ring alone cannot move it.
    fn moves(
        inputs: &DesignInputs,
        results: &DesignResults,
        path: &str,
        direction: Direction,
    ) -> Vec<DesignInputs> {
        let entry = InputCatalogue::get().entry(path).unwrap();
        match direction {
            Direction::Raise | Direction::Lower => {
                let range = entry.meta.range.unwrap();
                let x = current_number(inputs, results, path).unwrap();
                let step = match direction {
                    Direction::Raise => range.step,
                    _ => -range.step,
                };
                let next = (x + step).clamp(range.min, range.max);
                if next == x {
                    Vec::new()
                } else {
                    vec![set_number(inputs, path, next)]
                }
            }
            Direction::Change if PART_PATHS.contains(&path) => {
                let m = &inputs.coupling.magnets;
                let both = |parts: [&str; 2], grades: [&str; 2]| {
                    let mut moved = inputs.clone();
                    for (path, text) in PART_PATHS
                        .iter()
                        .zip(parts)
                        .chain(GRADE_PATHS.iter().zip(grades))
                    {
                        moved.set(path, Value::Text(text.to_owned())).unwrap();
                    }
                    moved
                };
                let grades = [m.grade_inner.as_str(), m.grade_outer.as_str()];
                let parts = MAGNET_LIBRARY
                    .iter()
                    .filter(|spec| [&m.part_inner, &m.part_outer] != [spec.part, spec.part])
                    .map(|spec| both([spec.part; 2], grades));
                let custom = GRADES
                    .iter()
                    .filter(|g| {
                        !(m.part_inner.is_empty() && m.part_outer.is_empty() && grades == [g.id; 2])
                    })
                    .map(|g| both([""; 2], [g.id; 2]));
                parts.chain(custom).collect()
            }
            Direction::Change => {
                let current = inputs.get(path);
                entry
                    .meta
                    .choices
                    .iter()
                    .filter(|&&(code, _)| current != Some(Value::Int(code)))
                    .map(|&(code, _)| {
                        let mut moved = inputs.clone();
                        moved.set(path, Value::Int(code)).unwrap();
                        moved
                    })
                    .collect()
            }
        }
    }

    #[test]
    fn each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red() {
        // Spec review focus 3. For every test design and every check it fails: each fix shown
        // moves a failing term of the check toward passing by one slider step or one choice
        // (the space claim excepted: it lists the fixes of all three axes and its Problem names
        // the axis that fails); every failing term has such a fix; and no fix turns another
        // guided check from green to red (a choice that would is not the one to take).
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let before: Vec<Level> = CHECK_GUIDES
                .iter()
                .map(|g| level(g, &inputs, &results))
                .collect();
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                if before[index] == Level::Good {
                    continue;
                }
                let terms_before = terms(g.check, &inputs, &results);
                // A margin that is not a number (a claim left blank) has no fix to move it:
                // its Problem asks for the number.
                let failing: BTreeSet<&str> = terms_before
                    .iter()
                    .filter(|t| !t.passes && !t.margin.is_nan())
                    .map(|t| t.name)
                    .collect();
                let shown = fixes(g, &inputs, &results);
                assert!(!shown.is_empty(), "{name}: {} shows no fix", g.check);
                let mut moved: BTreeSet<&str> = BTreeSet::new();
                for &(path, direction) in &shown {
                    let text = fix_text(path, direction);
                    let mut helps = false;
                    for design in moves(&inputs, &results, path, direction) {
                        let after = compute_all(&design);
                        let reddens = CHECK_GUIDES.iter().enumerate().any(|(other, h)| {
                            other != index
                                && before[other] == Level::Good
                                && level(h, &design, &after) == Level::Bad
                        });
                        if direction != Direction::Change {
                            assert!(!reddens, "{name}: {text} turns another check red");
                        } else if reddens {
                            continue;
                        }
                        let terms_after = terms(g.check, &design, &after);
                        for (b, a) in terms_before.iter().zip(&terms_after) {
                            if failing.contains(b.name) && a.margin > b.margin {
                                moved.insert(b.name);
                                helps = true;
                            }
                        }
                    }
                    if g.check != SPACE_CLAIM {
                        assert!(helps, "{name}: {text} moves no failing term of {}", g.check);
                    }
                }
                assert_eq!(
                    moved, failing,
                    "{name}: {}: a fix for every failing term",
                    g.check
                );
            }
        }
    }

    #[test]
    fn every_fix_moves_its_check_toward_passing_over_its_slider() {
        // (check, a design that fails it, a slider fix, the term it moves): the fix's input at
        // its slider's end in the fix's direction raises the term's margin. The plan's engine
        // probes found each fix monotone over its slider; the inputs left out are not (the ring
        // radius against the hot-low torque, the pole count against it) or are assumptions.
        // The cap goes down only to the cup it threads onto: below that the cup sets the
        // diameter, and the cap would be smaller than its cup.
        let cases: Vec<(&str, DesignInputs, &str, Direction, &str)> = vec![
            (
                HOT_MIN,
                DesignInputs::default(),
                MAGNET_LENGTH,
                Direction::Raise,
                "hot-low torque",
            ),
            (
                CLEARANCE,
                DesignInputs::default(),
                "metal.face_gap_mm",
                Direction::Raise,
                "running clearance",
            ),
            (
                CLEARANCE,
                DesignInputs::default(),
                "metal.sleeve_mm",
                Direction::Lower,
                "running clearance",
            ),
            (
                CLEARANCE,
                DesignInputs::default(),
                "metal.liner_mm",
                Direction::Lower,
                "running clearance",
            ),
            (
                CUP_WALL,
                DesignInputs::default(),
                "metal.cup_wall_corner_mm",
                Direction::Raise,
                "cup wall",
            ),
            (
                TEMPERATURE,
                breaking(slip_fault),
                "temperature.duty.fault_trip_s",
                Direction::Lower,
                "peak, magnets",
            ),
            (
                TEMPERATURE,
                breaking(slip_fault),
                "temperature.thermal.conductance_W_K",
                Direction::Raise,
                "peak, magnets",
            ),
            (
                CLAMP,
                breaking(small_boss),
                "clamps.boss_od_mm",
                Direction::Raise,
                "screw size",
            ),
            (
                CLAMP,
                breaking(short_clamp),
                "clamps.clamp_length_mm",
                Direction::Raise,
                "screw size",
            ),
            (
                SPACE_CLAIM,
                breaking(long_magnets),
                MAGNET_LENGTH,
                Direction::Lower,
                "overall length",
            ),
            (
                SPACE_CLAIM,
                breaking(large_cap),
                "metal.cap_od_mm",
                Direction::Lower,
                "diameter",
            ),
            (
                SPACE_CLAIM,
                breaking(large_ring),
                RING,
                Direction::Lower,
                "diameter",
            ),
            (
                END_EFFECT,
                short_magnets(),
                MAGNET_LENGTH,
                Direction::Raise,
                "end-effect factor",
            ),
            (
                FIT_CHECK,
                with_poles(40),
                RING,
                Direction::Raise,
                "inner ring",
            ),
            (
                FIT_CHECK,
                with_poles(40),
                POLES,
                Direction::Lower,
                "inner ring",
            ),
        ];
        // The table holds every slider fix of every guide, once.
        let key = |check: &'static str, path: &'static str, direction: Direction| {
            (check, path, direction == Direction::Raise)
        };
        let listed: BTreeSet<_> = cases.iter().map(|&(c, _, p, d, _)| key(c, p, d)).collect();
        let slider_fixes: BTreeSet<_> = CHECK_GUIDES
            .iter()
            .flat_map(|g| {
                possible_fixes(g)
                    .into_iter()
                    .filter(|&(_, d)| d != Direction::Change)
                    .map(move |(p, d)| key(g.check, p, d))
            })
            .collect();
        assert_eq!(listed, slider_fixes);
        assert_eq!(listed.len(), cases.len());
        let catalogue = InputCatalogue::get();
        for (check, failing_design, path, direction, term) in cases {
            let before = compute_all(&failing_design);
            assert_ne!(
                level_of(check, &failing_design, &before),
                Level::Good,
                "{check}: the case's design fails it"
            );
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            let end = match (path, direction) {
                ("metal.cap_od_mm", _) => before.model.cup_od_mm,
                (_, Direction::Raise) => range.max,
                _ => range.min,
            };
            let moved = set_number(&failing_design, path, end);
            let after = compute_all(&moved);
            let (from, to) = (
                margin(check, term, &failing_design, &before),
                margin(check, term, &moved, &after),
            );
            assert!(
                to > from,
                "{check}: {} moves the {term} margin {from} -> {to}",
                fix_text(path, direction)
            );
        }
    }

    #[test]
    fn the_temperature_fixes_follow_the_failing_terms() {
        use Direction::{Change, Lower, Raise};
        let parts = [
            ("coupling.magnets.part_inner", Change),
            ("coupling.magnets.part_outer", Change),
        ];
        let adhesive = ("temperature.adhesive.selected", Change);
        let cooling = [
            ("temperature.duty.fault_trip_s", Lower),
            ("temperature.thermal.conductance_W_K", Raise),
        ];
        let temperature = guide(TEMPERATURE).unwrap();
        let shown = |inputs: DesignInputs| fixes(temperature, &inputs, &compute_all(&inputs));
        // Above both limits: a hotter-rated magnet, a hotter adhesive, and less heat at the peak.
        assert_eq!(
            shown(breaking(hot_day)),
            [parts[0], parts[1], adhesive, cooling[0], cooling[1]]
        );
        assert_eq!(
            shown(breaking(n42_parts)),
            [parts[0], parts[1], cooling[0], cooling[1]]
        );
        assert_eq!(shown(breaking(slip_fault)), cooling);
        assert_eq!(
            shown(breaking(cool_adhesive)),
            [adhesive, cooling[0], cooling[1]]
        );
        assert_eq!(shown(breaking(hot_cure)), [adhesive]);
        assert_eq!(shown(breaking(ferrite)), parts);
        assert_eq!(shown(passing_design()), []);
        // A trip time already at its slider's end is not offered: a 92 °C start and the
        // shortest trip still peak past the magnet limit, and only the conductance can help.
        let quick_trip = breaking(|i| {
            slip_fault(i);
            i.temperature.duty.fault_trip_s = 0.1;
            i.temperature.duty.hot_ambient_C = 82.0;
        });
        let results = compute_all(&quick_trip);
        assert_eq!(failing_terms(&results), [TemperatureTerm::MagnetPeak]);
        assert_eq!(shown(quick_trip), [cooling[1]]);
    }

    #[test]
    fn a_fix_at_its_slider_end_is_not_offered() {
        let shown = |check: &str, inputs: &DesignInputs| {
            fixes(guide(check).unwrap(), inputs, &compute_all(inputs))
        };
        // No sleeve and no liner to thin: only the gap can still open.
        assert_eq!(
            shown(CLEARANCE, &design(bare_clearance)),
            [("metal.face_gap_mm", Direction::Raise)]
        );
        // Magnets left at their part's length can still grow; at the slider's top they cannot.
        assert_eq!(
            shown(HOT_MIN, &DesignInputs::default()),
            [(MAGNET_LENGTH, Direction::Raise)]
        );
        let longest = design(|i| {
            i.coupling.magnets.axial_length_mm = Some(50.8);
            i.metal.required_min_Nm = 20.0;
        });
        assert_eq!(
            level_of(HOT_MIN, &longest, &compute_all(&longest)),
            Level::Bad
        );
        assert_eq!(shown(HOT_MIN, &longest), []);
        // A cap typed above its slider's top can still come down.
        let typed = design(|i| i.metal.cap_od_mm = 90.0);
        assert!(shown(SPACE_CLAIM, &typed).contains(&("metal.cap_od_mm", Direction::Lower)));
    }

    #[test]
    fn no_design_shows_one_input_raised_and_lowered() {
        // Spec review focus 4: the default design's first screen once said "lower the gap" for
        // the hot-low torque and "raise the gap" for the clearance.
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let shown: Vec<(&str, Direction)> = CHECK_GUIDES
                .iter()
                .filter(|g| level(g, &inputs, &results) != Level::Good)
                .flat_map(|g| fixes(g, &inputs, &results))
                .collect();
            for &(path, direction) in &shown {
                if direction == Direction::Raise {
                    assert!(
                        !shown.contains(&(path, Direction::Lower)),
                        "{name}: {path} both ways"
                    );
                }
            }
        }
    }

    // ------------------------------------------------------------------ plain words

    #[test]
    fn plain_text_violations_names_each_banned_pattern() {
        assert_eq!(
            plain_text_violations("Too thin: raise Metal design C122 to at least 2.0 mm"),
            ["C\\d{2,3}"]
        );
        assert_eq!(plain_text_violations("E3 E7 E8"), ["E\\d{1,2}"]);
        assert_eq!(plain_text_violations("corrections E20."), ["E\\d{1,2}"]);
        assert_eq!(
            plain_text_violations("Calculator!C93"),
            ["C\\d{2,3}", "Calculator!"]
        );
        assert_eq!(
            plain_text_violations("Rust-only result (no workbook cell)"),
            ["workbook", "Rust-only"]
        );
        assert_eq!(plain_text_violations("Spec Addendum A5"), ["Addendum"]);
        assert_eq!(plain_text_violations("WORKBOOK"), ["workbook"]);
        // The status line's phrase excuses itself only, and only as written.
        assert_eq!(
            plain_text_violations("Workbook example design from the workbook"),
            ["workbook"]
        );
        assert_eq!(plain_text_violations("workbook example"), ["workbook"]);
        for clean in [
            "Workbook example design: fails 3 of 8 checks",
            "2.000e7 mm",
            "E123",
            "C1",
            "50.00 °C",
            "ISO 4762 M4 x 14, class 12.9",
            "Raise Minimum outer return ring wall",
            "",
        ] {
            assert!(plain_text_violations(clean).is_empty(), "{clean:?}");
        }
    }

    #[test]
    fn no_guide_text_carries_workbook_jargon() {
        for text in every_text() {
            assert_eq!(plain_text_violations(&text), Vec::<&str>::new(), "{text:?}");
        }
    }

    #[test]
    fn every_guide_text_has_glyphs_in_the_default_fonts() {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for text in every_text() {
            assert_glyphs(&ctx, &text, "guidance");
        }
    }

    // ------------------------------------------------------------------ the Problems

    #[test]
    fn each_problem_is_some_exactly_when_its_check_fails() {
        use Level::{Bad, Caution, Good};
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            for g in &CHECK_GUIDES {
                let badge = level(g, &inputs, &results);
                assert_eq!(
                    (g.problem)(&inputs, &results).is_some(),
                    badge != Good,
                    "{name}: {}",
                    g.check
                );
                // The level is the dashboard's badge of the check (green where it gives none).
                if g.check != FIT_CHECK {
                    assert_eq!(
                        check_level(&results, g.check).unwrap_or(Good),
                        badge,
                        "{name}: {}",
                        g.check
                    );
                }
            }
        }
        // The passing design passes every guide; the defaults fail three, all red.
        let inputs = passing_design();
        let results = compute_all(&inputs);
        for g in &CHECK_GUIDES {
            assert_eq!(level(g, &inputs, &results), Good, "{}", g.check);
        }
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let failing: Vec<(&str, Level)> = CHECK_GUIDES
            .iter()
            .map(|g| (g.check, level(g, &inputs, &results)))
            .filter(|&(_, badge)| badge != Good)
            .collect();
        assert_eq!(failing, [(HOT_MIN, Bad), (CLEARANCE, Bad), (CUP_WALL, Bad)]);
        // Each failing design fails the check it was built for, at the level the dashboard gives.
        for (inputs, check, want) in [
            (breaking(hot_day), TEMPERATURE, Caution),
            (breaking(n42_parts), TEMPERATURE, Caution),
            (breaking(slip_fault), TEMPERATURE, Caution),
            (breaking(cool_adhesive), TEMPERATURE, Caution),
            (breaking(hot_cure), TEMPERATURE, Caution),
            (breaking(ferrite), TEMPERATURE, Caution),
            (breaking(small_boss), CLAMP, Bad),
            (breaking(short_clamp), CLAMP, Bad),
            (breaking(long_magnets), SPACE_CLAIM, Bad),
            (breaking(large_cap), SPACE_CLAIM, Bad),
            (breaking(large_ring), SPACE_CLAIM, Bad),
            (breaking(unknown_claim), SPACE_CLAIM, Caution),
            (short_magnets(), END_EFFECT, Bad),
            (with_poles(12), FIT_CHECK, Bad),
            (with_poles(40), FIT_CHECK, Bad),
            (with(with_poles(40), arcs), FIT_CHECK, Bad),
            (design(bare_clearance), CLEARANCE, Bad),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(level_of(check, &inputs, &results), want, "{check}");
        }
    }

    #[test]
    fn the_problems_quote_numbers_from_the_results() {
        // Spec G-3: a Problem's numbers come from the results. For each guide, two designs that
        // fail it with different results: the two Problems differ, and each quotes the
        // results at the guide's key paths as the readouts show them.
        let cases: Vec<(&str, DesignInputs, DesignInputs, &[&str])> = vec![
            (
                HOT_MIN,
                DesignInputs::default(),
                design(|i| i.metal.face_gap_mm = 1.6),
                &["metal.torque_hot_low_Nm", "model.pullout_Nm"],
            ),
            (
                CLEARANCE,
                DesignInputs::default(),
                design(|i| i.metal.liner_mm = 0.1),
                &[
                    "metal.sleeve_liner_clearance_mm",
                    "metal.adverse_movement_mm",
                    "metal.min_running_clearance_mm",
                ],
            ),
            (
                CUP_WALL,
                DesignInputs::default(),
                design(|i| i.metal.cup_wall_corner_mm = 1.6),
                &[
                    "materials.cup_wall_corner_mm",
                    "materials.backiron_thickness_needed_mm",
                ],
            ),
            (
                TEMPERATURE,
                breaking(slip_fault),
                breaking(|i| {
                    slip_fault(i);
                    i.temperature.duty.fault_trip_s = 45.0;
                }),
                &[
                    "temperature.magnet_life.peak_C",
                    "temperature.summary.magnet_limit_C",
                ],
            ),
            (
                CLAMP,
                breaking(small_boss),
                breaking(|i| {
                    small_boss(i);
                    i.metal.face_gap_mm = 1.0;
                }),
                &["clamps.required_Nm", "clamps.max_torque_Nm"],
            ),
            (
                SPACE_CLAIM,
                breaking(large_cap),
                breaking(|i| i.metal.cap_od_mm = 45.0),
                &["metal.rotating_od_mm", "housing.diameter_overshoot_mm"],
            ),
            (
                END_EFFECT,
                short_magnets(),
                with(short_magnets(), |i| i.coupling.c_end = 0.4),
                &[
                    "model.active_length_mm",
                    "model.pole_pitch_mm",
                    "model.f_end",
                ],
            ),
            (
                FIT_CHECK,
                with_poles(40),
                with_poles(36),
                &["model.inner_width_mm", "model.inner_flat_width_mm"],
            ),
        ];
        let checks: BTreeSet<&str> = cases.iter().map(|&(check, ..)| check).collect();
        let guided: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.check).collect();
        assert_eq!(checks, guided, "every guide");
        for (check, first, second, keys) in cases {
            let (a, b) = (compute_all(&first), compute_all(&second));
            let quoted = |results: &DesignResults, path: &str| {
                format_value(&results.get(path).unwrap_or(Value::None))
            };
            assert!(
                keys.iter().any(|path| quoted(&a, path) != quoted(&b, path)),
                "{check}: the two designs' results differ"
            );
            let texts = [
                (guide(check).unwrap().problem)(&first, &a).expect("the first design fails"),
                (guide(check).unwrap().problem)(&second, &b).expect("the second design fails"),
            ];
            assert_ne!(texts[0], texts[1], "{check}");
            for (text, results) in texts.iter().zip([&a, &b]) {
                for path in keys {
                    let number = quoted(results, path);
                    assert!(
                        text.contains(&number),
                        "{check}: {number} ({path}) in {text}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_problems_quote_the_live_numbers() {
        let defaults = DesignInputs::default();
        assert_eq!(
            problem(HOT_MIN, &defaults).unwrap(),
            "Hot-low torque 2.285 N·m is below the required 2.500 N·m: the pull-out of 2.688 N·m at 50.00 °C, less the 15 % variation allowance."
        );
        assert_eq!(
            problem(CLEARANCE, &defaults).unwrap(),
            "Sleeve-to-liner gap 0.6768 mm less 0.7800 mm of movement allowances leaves -0.1032 mm, below the 0.2000 mm target."
        );
        assert_eq!(
            problem(CUP_WALL, &defaults).unwrap(),
            "Cup wall at the pocket corners is 1.800 mm; the flux it carries needs 1.919 mm."
        );
        // The numbers follow an edit: a 2.4 N·m requirement still fails, quoted as such.
        let edited = design(|i| i.metal.required_min_Nm = 2.4);
        let text = problem(HOT_MIN, &edited).unwrap();
        assert!(text.contains("below the required 2.400 N·m"), "{text}");
        // A 0.1 mm liner: the gap and the result after the allowances move by its 0.1 mm.
        let thinner = design(|i| i.metal.liner_mm = 0.1);
        let text = problem(CLEARANCE, &thinner).unwrap();
        assert!(text.starts_with("Sleeve-to-liner gap 0.7768 mm"), "{text}");
        assert!(text.contains("leaves -0.003174 mm"), "{text}");
    }

    #[test]
    fn the_temperature_problem_names_each_failing_term() {
        assert_eq!(
            problem(TEMPERATURE, &design(hot_day)).unwrap(),
            "Hot-day start 140.0 °C is not below the 93.06 °C magnet limit. Hot-day start 140.0 °C is not below the 120.0 °C adhesive limit. The slip-fault peak of 141.0 °C reaches the 93.06 °C magnet limit. The slip-fault peak of 141.0 °C reaches the 120.0 °C adhesive limit."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(slip_fault)).unwrap(),
            "The slip-fault peak of 104.0 °C reaches the 93.06 °C magnet limit."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(cool_adhesive)).unwrap(),
            "Hot-day start 65.00 °C is not below the 60.00 °C adhesive limit. The slip-fault peak of 65.96 °C reaches the 60.00 °C adhesive limit."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(hot_cure)).unwrap(),
            "The adhesive cures at 120.0 °C, 5.955 °C below the 126.0 °C single-ring demagnetization onset; the margin must be at least 10.00 °C."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(ferrite)).unwrap(),
            "The minimum magnet temperature -40.00 °C is below the 335.7 °C cold demagnetization limit (from stored 3D fields that do not yet follow the magnet grade)."
        );
        assert_eq!(problem(TEMPERATURE, &DesignInputs::default()), None);
    }

    #[test]
    fn the_temperature_terms_reproduce_the_engine_verdict() {
        // The terms are read from the margins with the verdict's own comparisons (and its
        // cure-margin minimum): over designs on both sides of every term, they are empty
        // exactly when the verdict passes. The single-ring fields step the cure margin across
        // its minimum (666.36 kA/m puts it at 10 °C, as the engine's own test finds), and an
        // adhesive code outside the table has no limits at all.
        let mut cases = designs();
        for field in [640.0, 650.0, 660.0, 666.3586306326292, 667.0, 680.0, 700.0] {
            cases.push((
                "single-ring field",
                design(|i| {
                    i.temperature.adhesive.selected = 2;
                    i.temperature.demag.h_rev_single_ring_kA_m = field;
                }),
            ));
        }
        for start in [83.0, 93.0, 93.06, 94.0, 119.0, 121.0] {
            cases.push((
                "hot-day start",
                design(|i| i.temperature.duty.hot_ambient_C = start - 10.0),
            ));
        }
        cases.push((
            "no adhesive",
            design(|i| i.temperature.adhesive.selected = 0),
        ));
        for (name, inputs) in cases {
            let results = compute_all(&inputs);
            let passes = check_level(&results, TEMPERATURE) == Some(Level::Good);
            assert_eq!(
                failing_terms(&results).is_empty(),
                passes,
                "{name}: {}",
                results.temperature.summary.verdict
            );
            assert_eq!(
                temperature_fixes(&inputs, &results).is_empty(),
                passes,
                "{name}"
            );
        }
    }

    #[test]
    fn the_clamp_problem_names_the_screw_range_and_the_torque() {
        assert_eq!(
            problem(CLAMP, &design(small_boss)).unwrap(),
            "No screw size from M2.5 to M6 fits the boss and the clamp length with enough screws to hold 7.647 N·m: the cold-high torque 3.823 N·m times a safety factor of 2.000."
        );
        assert_eq!(problem(CLAMP, &DesignInputs::default()), None);
    }

    #[test]
    fn the_space_claim_problem_names_each_axis_past_its_claim() {
        assert_eq!(
            problem(SPACE_CLAIM, &design(long_magnets)).unwrap(),
            "Overall length 69.90 mm is 34.90 mm over the 35.00 mm claim. Large-diameter stack 56.90 mm is 36.90 mm over the 20.00 mm bay."
        );
        assert_eq!(
            problem(SPACE_CLAIM, &design(large_cap)).unwrap(),
            "Rotating diameter 44.00 mm is 1.000 mm over the 43.00 mm claim."
        );
        assert_eq!(
            problem(SPACE_CLAIM, &design(unknown_claim)).unwrap(),
            "Rotating diameter cannot be checked: it or its claim is not a number."
        );
        assert_eq!(problem(SPACE_CLAIM, &DesignInputs::default()), None);
    }

    #[test]
    fn out_of_range_end_effect_gives_its_own_problem_and_greys_the_others() {
        // Spec 5: with the end-effect model out of range the hot-low torque has no prediction,
        // so the hot minimum and the clamp (computed from it) give no Problem; the end-effect
        // guide says what to change.
        let inputs = short_magnets();
        let results = compute_all(&inputs);
        assert_eq!(
            problem(END_EFFECT, &inputs).unwrap(),
            "Magnets 2.000 mm long with a pole pitch of 8.809 mm give an end-effect factor of -1.202; the model gives no torque until the magnets are longer than 4.405 mm and is rough just above that."
        );
        for check in [HOT_MIN, CLAMP] {
            assert_eq!(level(guide(check).unwrap(), &inputs, &results), Level::Good);
            assert_eq!(problem(check, &inputs), None, "{check}");
        }
        assert_eq!(problem(END_EFFECT, &DesignInputs::default()), None);
    }

    #[test]
    fn the_fit_problem_names_the_rings_that_do_not_fit() {
        assert_eq!(
            problem(FIT_CHECK, &with_poles(40)).unwrap(),
            "40 blocks per ring do not fit at this ring size. Inner blocks are 6.350 mm wide on 1.598 mm flats. Outer blocks are 6.350 mm wide on 2.317 mm flats."
        );
        assert_eq!(
            problem(FIT_CHECK, &with(with_poles(40), arcs)).unwrap(),
            "40 blocks per ring do not fit at this ring size. The 40 inner blocks together need 344.5 % of the circle at their mid-radius. The 40 outer blocks together need 247.9 % of the circle at their mid-radius."
        );
        // 12 flat blocks: the inner flats are too narrow, the outer ones are not.
        assert_eq!(
            problem(FIT_CHECK, &with_poles(12)).unwrap(),
            "12 blocks per ring do not fit at this ring size. Inner blocks are 6.350 mm wide on 5.439 mm flats."
        );
        // The level follows the engine's fit rule, the one inverse sizing counts.
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let fits = blocks_fit(&inputs.coupling, &results.model);
            assert_eq!(
                level_of(FIT_CHECK, &inputs, &results) == Level::Good,
                fits,
                "{name}"
            );
        }
        assert_eq!(problem(FIT_CHECK, &DesignInputs::default()), None);
    }

    #[test]
    fn a_check_that_does_not_apply_gives_no_problem() {
        // Without back iron no magnetic rule sizes the cup wall: no badge, no Problem.
        let inputs = breaking(no_back_iron);
        let results = compute_all(&inputs);
        assert_eq!(results.materials.cup_wall_check, "No back iron");
        assert_eq!(
            level(guide(CUP_WALL).unwrap(), &inputs, &results),
            Level::Good
        );
        assert_eq!(problem(CUP_WALL, &inputs), None);
    }

    // ------------------------------------------------------------------ the design status

    #[test]
    fn the_design_status_counts_the_failing_checks() {
        let status = |inputs: &DesignInputs| design_status(inputs, &compute_all(inputs));
        let defaults = status(&DesignInputs::default());
        assert_eq!(
            defaults,
            DesignStatus {
                bad: 3,
                caution: 0,
                total: 8,
                is_default: true
            }
        );
        assert_eq!(
            design_status_text(&defaults),
            "Workbook example design: fails 3 of 8 checks"
        );
        // One fix: the cup wall at its suggested 2.0 mm.
        let fixed = status(&design(|i| i.metal.cup_wall_corner_mm = 2.0));
        assert_eq!(
            design_status_text(&fixed),
            "Your design: fails 2 of 8 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 8 checks"
        );
        // Cautions are counted apart from the failures.
        assert_eq!(
            design_status_text(&status(&design(hot_day))),
            "Your design: fails 3 of 8 checks, 1 caution"
        );
        let two_cautions = design(|i| {
            hot_day(i);
            unknown_claim(i);
        });
        assert_eq!(
            design_status_text(&status(&two_cautions)),
            "Your design: fails 3 of 8 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 8 checks"
        );
        // Any edit leaves the example, even one that keeps the counts.
        let edited = status(&design(|i| i.metal.hardware_g = 7.0));
        assert_eq!((edited.bad, edited.is_default), (3, false));
    }

    #[test]
    fn the_status_text_reads_each_form() {
        let text = |bad, caution, is_default| {
            design_status_text(&DesignStatus {
                bad,
                caution,
                total: 8,
                is_default,
            })
        };
        assert_eq!(
            text(0, 0, true),
            "Workbook example design: passes all 8 checks"
        );
        assert_eq!(
            text(0, 1, false),
            "Your design: passes all 8 checks, 1 caution"
        );
        assert_eq!(text(1, 0, false), "Your design: fails 1 of 8 checks");
        assert_eq!(
            text(8, 3, true),
            "Workbook example design: fails 8 of 8 checks, 3 cautions"
        );
        assert!(text(1, 0, true).starts_with(WORKBOOK_EXAMPLE));
        assert!(text(1, 0, false).starts_with(YOUR_DESIGN));
    }
}
