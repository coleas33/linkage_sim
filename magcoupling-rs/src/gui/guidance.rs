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
//! A fix is an input path and a direction ([`fixes`]): a slider input raised or lowered, a
//! choice or picker input changed, or a ring's magnet part taken to Custom dimensions with a
//! grade rated hotter than the part ([`Direction::CustomHotterGrade`], the way past the
//! hottest-rated library part). The temperature, space-claim and shaft joints guides pick their
//! fixes by the terms of their verdicts that fail ([`Fixes::PerTerm`]: each temperature limit,
//! ring by ring; each axis of the claim; each term of each shaft joint); every other guide has
//! one list. A fix is shown only where its move helps: one slider step, or one other choice of
//! its input alone (those two picks count as one), that turns no passing guided check red; a
//! fix picked per term must also raise that term's margin, while each fix of a static list
//! raises its check from a failing design to its slider's end (a test moves each there). No
//! move leaves a ring's magnet without a rating, since the engine does not calibrate such a
//! magnet's limit. These checks compute the design each move reaches, so the panel asks through
//! a [`FixCache`]. A failing check left without a fix says so ([`NO_FIX`]), and two failing
//! checks that pull one input both ways each say so ([`trade_offs`]), as does a fix that sets
//! back another failing check ([`side_effects`]). A fix's text takes the input's label from the
//! [`InputCatalogue`], or the Size tab's name for it ([`fix_text`]), so no guide types a label.
//! The tests hold the fixes to helping: on each test design that fails a check, every failing
//! term of the check has a shown fix that moves it toward passing, and no shown fix turns
//! another check red.
//!
//! What moves what (spec 3.3): [`Consequences`] are the lines under the Size tab's controls
//! (the smallest outside diameter at which the count fits, and what sets the outside diameter
//! and the overall length, [`crate::engine::size_controls`]) and the torque card the Size tab
//! draws between its readouts and its drawing ([`crate::gui::torque_card`]: one step of each
//! size control each way, its count row the lines for two magnets fewer and two more that
//! stood under Magnets per ring until slice 3, decision 7A). Every number comes from the
//! engine. [`ConsequenceCache`] keeps them until the inputs, the minimum diameter or the
//! shaft-clamp pause change.
//!
//! The engine's verdict texts, the results table and the exports keep their wording (locked by
//! `tests/python_schema.rs`): this is a GUI layer over them, in plain words
//! ([`plain_text_violations`] lists what it must not say). No egui.

use crate::engine::api::shaft_joint_links;
use crate::engine::clamps::SCREW_SIZES;
use crate::engine::deviations::Deviations;
use crate::engine::grades::{GRADES, GradeFamily};
use crate::engine::library::{MAGNET_LIBRARY, lookup};
use crate::engine::meta::{InputSet, NumOrText, ResultSet, SliderRange, Value};
use crate::engine::model::{ResolvedMagnet, RingFit, blocks_fit, resolve_magnets, ring_fits};
use crate::engine::shaft_joint::{
    self, CAUSES, CannotBeMade, Cause, Floor, HubFitLimit, ImpossibleValue, JointTerms, NoWall,
};
use crate::engine::size_controls::{
    DiameterBreakdown, DiameterFit, FIT_TOLERANCE_MM, LengthBreakdown, diameter_breakdown,
    dimensions, dimensions_with_minimum_diameter, length_breakdown, resize_diameter,
    smallest_fitting_diameter,
};
use crate::engine::sizing::FreeVariable;
use crate::engine::temperature::{CURE_MARGIN_MIN_C, DemagInputs};
use crate::gui::dashboard::{DASHBOARD, Level, check_level, result_info};
use crate::gui::format::{
    SIGNIFICANT_DIGITS, fewest_decimals_apart, fixed, format_value, last_digit_step, with_unit,
};
use crate::gui::inputs::{InputCatalogue, optional_seed, section_label, slider_range};
use crate::gui::pickers::{CUSTOM, GRADE_PATHS, PART_PATHS};
use crate::gui::plain_text::{display_label, label_for};
use crate::gui::size_controls;
use crate::gui::torque_card::{TorqueCard, card};
use crate::{DesignInputs, DesignResults, compute_all};

/// Which way a fix moves its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Up its slider.
    Raise,
    /// Down its slider.
    Lower,
    /// To another choice: a selector's code, or a picker's part.
    Change,
    /// A ring's magnet part to "Custom dimensions" and its grade to one rated hotter than the
    /// part: two picks scored as one move, the way past the hottest-rated library part. A fix
    /// on a part input only; its text names both inputs ([`fix_text`]).
    CustomHotterGrade,
}

impl Direction {
    /// The verb a fix's text starts with.
    const fn word(self) -> &'static str {
        match self {
            Direction::Raise => "Raise",
            Direction::Lower => "Lower",
            Direction::Change | Direction::CustomHotterGrade => "Change",
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
    /// A list built from the terms of the check's verdict that fail, each fix checked against
    /// the term it serves (the temperature, space-claim and shaft joints guides), with the
    /// shaft-clamp section's pause ([`helps`]).
    PerTerm(fn(&DesignInputs, &DesignResults, bool) -> Vec<(&'static str, Direction)>),
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

// The dashboard checks the guides cover (`DASHBOARD`'s check column; a test compares). The
// torque card's tests name the checks it flips and quotes by them.
pub(crate) const HOT_MIN: &str = "metal.hot_min_check";
pub(crate) const CLEARANCE: &str = "metal.clearance_check";
pub(crate) const CUP_WALL: &str = "materials.cup_wall_check";
pub(crate) const TEMPERATURE: &str = "temperature.summary.verdict";
pub(crate) const CLAMP: &str = "clamps.recommended";
pub(crate) const SHAFT_JOINT: &str = "shaft_joint.check";
pub(crate) const GEARBOX: &str = "gearbox.cold_high_check";
pub(crate) const SPACE_CLAIM: &str = "housing.space_claim_check";
const END_EFFECT: &str = "model.end_effect_check";

// Inputs several guides move.
const MAGNET_LENGTH: &str = "coupling.magnets.axial_length_mm";
const RING: &str = "coupling.inner_back_apothem_mm";
const POLES: &str = "coupling.npole";
pub(crate) const FACE_GAP: &str = "metal.face_gap_mm";

/// Inputs the Size tab shows under its own name, by path: a fix on one names the Size-tab
/// control a user moves it with ("Raise Outside diameter", not the ring's radius).
pub const SIZE_TAB_ALIASES: [(&str, &str); 1] = [(RING, size_controls::DIAMETER)];

const HOT_MIN_GUIDE: CheckGuide = CheckGuide {
    check: HOT_MIN,
    label: "Hot-low torque",
    problem: hot_min_problem,
    why: "Magnets weaken as they warm, and production scatter makes some couplings weaker than nominal; hot-low is the pull-out at the operating temperature less that scatter, and must reach the requirement.",
    // Not the gap: a smaller gap raises the torque but takes running clearance, whose own fix
    // opens the gap (both fail on the default design).
    fixes: Fixes::Static(&[(MAGNET_LENGTH, Direction::Raise)]),
    learn: Learn::Note("hot_low_cold_high"),
};

const CLEARANCE_GUIDE: CheckGuide = CheckGuide {
    check: CLEARANCE,
    label: "Running clearance",
    problem: clearance_problem,
    why: "Only a small radial gap separates the inner and outer rotors; it must absorb shaft offset, runout, deflection, thermal growth and tolerances, or they rub.",
    // Thinner sleeve and liner first: they open the gap at no cost in torque, while a larger
    // face gap lowers the pull-out ([`SIDE_EFFECTS`] says so while the hot-low torque fails).
    fixes: Fixes::Static(&[
        ("metal.sleeve_mm", Direction::Lower),
        ("metal.liner_mm", Direction::Lower),
        (FACE_GAP, Direction::Raise),
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
    why: "Too hot, and the magnets can lose strength for good or the adhesive can weaken; ferrite magnets are also at risk when too cold.",
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

const SHAFT_JOINT_GUIDE: CheckGuide = CheckGuide {
    check: SHAFT_JOINT,
    label: "Shaft joints",
    problem: shaft_joint_problem,
    why: "A shrink fit's grip holds each rotor on its shaft; each fit must carry cold-high with margin when loosest, stay elastic when tightest, and be fitted within its allowed temperature.",
    fixes: Fixes::PerTerm(shaft_joint_fixes),
    learn: Learn::Note("shaft_joint"),
};

// The shaft joints' inputs the guide names (BL-064).
const HUB_MIN: &str = "shaft_joint.hub_interference_min_um";
const HUB_MAX: &str = "shaft_joint.hub_interference_max_um";
const CUP_MIN: &str = "shaft_joint.cup_interference_min_um";
const CUP_MAX: &str = "shaft_joint.cup_interference_max_um";
const KEY_CONTACT: &str = "shaft_joint.key_contact_mm";
const WHEEL_SHAFT: &str = "shaft_joint.wheel_shaft_fit_C";
const GEARBOX_SHAFT: &str = "shaft_joint.output_shaft_fit_C";
const HUB_LENGTH: &str = "metal.hub_length_mm";
const BOSS_LENGTH: &str = "metal.boss_length_mm";
const SAFETY_FACTOR: &str = "shaft_joint.safety_factor";
const KEY_WIDTH: &str = "shaft_joint.key_width_mm";
const WEB: &str = "metal.web_mm";
// The shaft joints' results the guide quotes (BL-064): the torque each joint must hold, the
// stress limit and the part's yield it comes from, and the inner hub's length in effect (the
// magnets' length under the axial override).
const REQUIRED: &str = "shaft_joint.required_Nm";
const STRESS_LIMIT: &str = "shaft_joint.stress_limit_MPa";
const PART_YIELD: &str = "shaft_joint.part_yield_MPa";
const HUB_LENGTH_IN_EFFECT: &str = "housing.hub_length_mm";

/// One shaft joint as its guide reads and names it (BL-064), in the engine's order
/// ([`shaft_joint::joints`]: the inner hub, then the cup; a test holds the two to it).
struct JointSide {
    /// The joint's name in the engine's texts ([`shaft_joint::INNER_HUB`], [`shaft_joint::CUP`]).
    name: &'static str,
    /// The shaft it is fitted on.
    shaft: &'static str,
    /// Its band: the minimum and the maximum measured interference (inputs).
    min: &'static str,
    max: &'static str,
    /// Its shaft's temperature at fitting (an input).
    shaft_fit: &'static str,
    /// Its results: the holding torque, the bore stress, the key pressure, the window's edges
    /// (the smallest minimum that holds, the largest maximum within the stress limit and the
    /// largest that can be fitted), the fitting temperature needed and the one allowed.
    holding: &'static str,
    stress: &'static str,
    key: &'static str,
    needed: &'static str,
    elastic: &'static str,
    fittable: &'static str,
    fit_temp: &'static str,
    fit_allowed: &'static str,
    /// The fixes of each of its terms ([`JointSide::fixes`]; spec 5.2).
    slip_fixes: &'static [(&'static str, Direction)],
    stress_fixes: &'static [(&'static str, Direction)],
    fitting_fixes: &'static [(&'static str, Direction)],
    key_fixes: &'static [(&'static str, Direction)],
}

impl JointSide {
    /// The fixes of the joint's `cause` term.
    const fn fixes(&self, cause: Cause) -> &'static [(&'static str, Direction)] {
        match cause {
            Cause::Slips => self.slip_fixes,
            Cause::BoreStress => self.stress_fixes,
            Cause::CannotBeFitted => self.fitting_fixes,
            Cause::Key => self.key_fixes,
        }
    }

    /// The result the torque card quotes for the joint's `cause` term, and its decimals (slice 3
    /// spec 3.6): the smallest measured interference that holds [µm], the bore stress [MPa], the
    /// fitting temperature needed [°C] and the key pressure [MPa].
    const fn quoted(&self, cause: Cause) -> (&'static str, usize) {
        match cause {
            Cause::Slips => (self.needed, 2),
            Cause::BoreStress => (self.stress, 1),
            Cause::CannotBeFitted => (self.fit_temp, 1),
            Cause::Key => (self.key, 1),
        }
    }
}

/// The two shaft joints. A larger face gap lowers cold-high, and with it the torque each fit
/// must hold; shorter magnets do too, but only the cup is offered them, since the inner hub
/// lengthens with its magnets. Not offered (spec 5.2): the assumptions (friction, smoothing,
/// the rotor-over-shaft allowance, the joining clearance), the safety factor, the allowed
/// fitting temperatures, the back iron, the keyway depth and the magnets' part or grade.
const JOINT_SIDES: [JointSide; 2] = [
    JointSide {
        name: shaft_joint::INNER_HUB,
        shaft: "wheel shaft",
        min: HUB_MIN,
        max: HUB_MAX,
        shaft_fit: WHEEL_SHAFT,
        holding: "shaft_joint.hub_holding_Nm",
        stress: "shaft_joint.hub_bore_stress_MPa",
        key: "shaft_joint.hub_key_pressure_MPa",
        needed: "shaft_joint.hub_needed_um",
        elastic: "shaft_joint.hub_elastic_max_um",
        fittable: "shaft_joint.hub_fittable_max_um",
        fit_temp: "shaft_joint.hub_fit_temp_C",
        fit_allowed: "shaft_joint.hub_fit_allowed_C",
        slip_fixes: &[
            (HUB_MIN, Direction::Raise),
            (HUB_LENGTH, Direction::Raise),
            (FACE_GAP, Direction::Raise),
        ],
        stress_fixes: &[(HUB_MAX, Direction::Lower)],
        fitting_fixes: &[(HUB_MAX, Direction::Lower), (WHEEL_SHAFT, Direction::Lower)],
        key_fixes: &[
            (KEY_CONTACT, Direction::Raise),
            (HUB_LENGTH, Direction::Raise),
        ],
    },
    JointSide {
        name: shaft_joint::CUP,
        shaft: "gearbox shaft",
        min: CUP_MIN,
        max: CUP_MAX,
        shaft_fit: GEARBOX_SHAFT,
        holding: "shaft_joint.cup_holding_Nm",
        stress: "shaft_joint.cup_bore_stress_MPa",
        key: "shaft_joint.cup_key_pressure_MPa",
        needed: "shaft_joint.cup_needed_um",
        elastic: "shaft_joint.cup_elastic_max_um",
        fittable: "shaft_joint.cup_fittable_max_um",
        fit_temp: "shaft_joint.cup_fit_temp_C",
        fit_allowed: "shaft_joint.cup_fit_allowed_C",
        slip_fixes: &[
            (CUP_MIN, Direction::Raise),
            (BOSS_LENGTH, Direction::Raise),
            (MAGNET_LENGTH, Direction::Lower),
            (FACE_GAP, Direction::Raise),
        ],
        stress_fixes: &[(CUP_MAX, Direction::Lower)],
        fitting_fixes: &[
            (CUP_MAX, Direction::Lower),
            (GEARBOX_SHAFT, Direction::Lower),
        ],
        key_fixes: &[
            (KEY_CONTACT, Direction::Raise),
            (BOSS_LENGTH, Direction::Raise),
        ],
    },
];

/// The shaft joints of a design as the engine judges them, inner hub first, rebuilt from its
/// inputs and results ([`shaft_joint_links`]); while the check cannot be made, why, by the
/// engine's own test ([`shaft_joint::cannot_be_made`]: a value it reads or works out is not a
/// number, the safety factor is not above zero, an input is typed below what it can physically
/// be, or the gearbox shaft is typed below its slider's bottom), where no term is judged.
fn shaft_joints(
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Result<[JointTerms; 2], CannotBeMade> {
    let links = shaft_joint_links(inputs, results);
    let joints = shaft_joint::joints(&inputs.shaft_joint, &links);
    match shaft_joint::cannot_be_made(&inputs.shaft_joint, &links, &joints) {
        Some(reason) => Err(reason),
        None => Ok(joints),
    }
}

/// The index in [`JOINT_SIDES`] of the joint named `name`, and its side.
fn joint_side(name: &str) -> Option<(usize, &'static JointSide)> {
    JOINT_SIDES
        .iter()
        .enumerate()
        .find(|(_, side)| side.name == name)
}

/// The shaft joints' fixes ([`Fixes::PerTerm`]): for each term a judged joint fails, in the
/// badge's order ([`shaft_joint::failing`]), each of that joint's fixes for the term whose move
/// helps ([`helps`] against the term's margin, [`JointTerms::margin`]), each fix once; none
/// while the check cannot be made. Three are held back by rule (spec 5.2): raising a band's
/// minimum or lowering its maximum where one step puts the minimum above the maximum (the row
/// would read amber, and the margin's rise would count as help), and raising the key's contact
/// height for a joint whose key has no bearing length while its contact height is above zero
/// (no contact height gives that key any bearing area).
fn shaft_joint_fixes(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<(&'static str, Direction)> {
    let Ok(joints) = shaft_joints(inputs, results) else {
        return Vec::new();
    };
    let mut shown: Vec<(&'static str, Direction)> = Vec::new();
    for (cause, joint) in shaft_joint::failing(&joints) {
        let Some((index, side)) = joint_side(joint.name) else {
            continue;
        };
        let margin = |design: &DesignInputs, after: &DesignResults| {
            shaft_joint::joints(&design.shaft_joint, &shaft_joint_links(design, after))[index]
                .margin(cause)
        };
        for &fix in side.fixes(cause) {
            let contact_without_length = fix == (KEY_CONTACT, Direction::Raise)
                && joint.key_bearing_mm <= 0.0
                && joint.key_contact_mm > 0.0;
            if !shown.contains(&fix)
                && !contact_without_length
                && !reverses_the_band(inputs, results, side, fix)
                && helps(inputs, results, fix, Some(&margin), clamps_paused)
            {
                shown.push(fix);
            }
        }
    }
    shown
}

/// Whether one move of `fix`, an input of the band of `side`, puts the band's minimum above its
/// maximum; `false` for any other input.
fn reverses_the_band(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    (path, direction): (&str, Direction),
) -> bool {
    if path != side.min && path != side.max {
        return false;
    }
    moves(inputs, results, path, direction)
        .iter()
        .any(|design| {
            let band = |path: &str| design.get(path).and_then(number).unwrap_or(f64::NAN);
            band(side.min) > band(side.max)
        })
}

const GEARBOX_GUIDE: CheckGuide = CheckGuide {
    check: GEARBOX,
    label: "Gearbox input torque",
    problem: gearbox_problem,
    why: "Cold-high torque can reach the gearbox output before the coupling slips, and in each swing while it slips; at the gearbox input it must stay within the rating.",
    // Geometry only (owner decision 6A): both lower cold-high, and hot-low with it, which the
    // trade-off and side-effect lines say where the hot minimum fails. Not the variation (an
    // assumption), the minimum temperature (an environment fact), or the rating or ratio (no
    // real gearbox changes one alone): the Problem names them where no geometry passes both.
    fixes: Fixes::Static(&[
        (MAGNET_LENGTH, Direction::Lower),
        (FACE_GAP, Direction::Raise),
    ]),
    learn: Learn::Note(TORQUE_BAND_NOTE),
};

const SPACE_CLAIM_GUIDE: CheckGuide = CheckGuide {
    check: SPACE_CLAIM,
    label: "Space claim",
    problem: space_claim_problem,
    why: "The coupling must fit the envelope reserved for it: its diameter, its overall length and the length of the large-diameter bay.",
    fixes: Fixes::PerTerm(space_claim_fixes),
    learn: Learn::None("No note covers the space claim yet."),
};

/// The inputs that shrink each space-claim axis, in the claim's order (diameter, overall
/// length, large-diameter bay): the larger of the cap and the cup sets the diameter, and the
/// Outside diameter's slider moves both; the magnets lengthen both stacks.
const SPACE_CLAIM_AXES: [&[(&str, Direction)]; 3] = [
    &[
        ("metal.cap_od_mm", Direction::Lower),
        (RING, Direction::Lower),
    ],
    &[(MAGNET_LENGTH, Direction::Lower)],
    &[(MAGNET_LENGTH, Direction::Lower)],
];

/// Each space-claim axis's reserve, in [`SPACE_CLAIM_AXES`]' order: the claim less the size,
/// larger is better, below zero when the axis is over its claim.
fn axis_reserves(results: &DesignResults) -> [f64; 3] {
    let m = &results.metal;
    [
        m.diameter_reserve_mm,
        m.axial_reserve_mm,
        m.large_dia_reserve_mm,
    ]
}

/// Each space-claim axis's name and the result path of the size its reserve is the claim less,
/// in [`axis_reserves`]' order: the one table of the axes' names and sizes (slice 3 spec 3.6).
/// The space claim's Problem names an axis past its claim by it, and [`check_terms`] quotes the
/// size on the torque card.
pub(crate) const SPACE_CLAIM_AXIS_SIZES: [(&str, &str); 3] = [
    ("Rotating diameter", "metal.rotating_od_mm"),
    ("Overall length", "metal.axial_stack_mm"),
    ("Large-diameter stack", "metal.large_dia_stack_mm"),
];

/// The space claim's fixes ([`Fixes::PerTerm`]): for each axis over its claim, each of its
/// inputs whose step helps ([`helps`] against the axis's reserve), each input once. An axis
/// whose size or claim is not a number shows none: its Problem asks for the number.
fn space_claim_fixes(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<(&'static str, Direction)> {
    let mut shown: Vec<(&'static str, Direction)> = Vec::new();
    for (axis, axis_fixes) in SPACE_CLAIM_AXES.iter().enumerate() {
        let over = axis_reserves(results)[axis] < 0.0;
        if !over {
            continue;
        }
        let reserve = |_: &DesignInputs, after: &DesignResults| axis_reserves(after)[axis];
        for &fix in *axis_fixes {
            if !shown.contains(&fix) && helps(inputs, results, fix, Some(&reserve), clamps_paused) {
                shown.push(fix);
            }
        }
    }
    shown
}

const END_EFFECT_GUIDE: CheckGuide = CheckGuide {
    check: END_EFFECT,
    label: "End-effect model",
    problem: end_effect_problem,
    why: "Field fringes at the magnet ends, so the model takes off a length set by the pole pitch; magnets no longer than that leave no usable torque prediction.",
    // Not more poles: they shorten the pitch, but at the default ring 12 poles already do not
    // fit.
    fixes: Fixes::Static(&[(MAGNET_LENGTH, Direction::Raise)]),
    learn: Learn::Note("end_effect"),
};

const FIT_GUIDE: CheckGuide = CheckGuide {
    check: FIT_CHECK,
    label: "Magnets fit",
    problem: fit_problem,
    why: "Each ring holds one block per pole; at a given ring radius, more blocks leave each a narrower place, and a block wider than its place does not fit.",
    fixes: Fixes::Static(&[(RING, Direction::Raise), (POLES, Direction::Lower)]),
    learn: Learn::None("No note covers block fit yet."),
};

/// The guides: the nine dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 10] = [
    HOT_MIN_GUIDE,
    CLEARANCE_GUIDE,
    CUP_WALL_GUIDE,
    TEMPERATURE_GUIDE,
    CLAMP_GUIDE,
    SHAFT_JOINT_GUIDE,
    GEARBOX_GUIDE,
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
    // The cold pull-out with the variation allowance added: the torque-band note explains it
    // beside hot-low (a value row a reviewed note covers links to that note).
    (
        "metal.torque_cold_high_Nm",
        Learn::Note("hot_low_cold_high"),
    ),
    // The torque swings that reach the gearbox input while the coupling slips.
    (
        "model.gearbox_input_ripple_Nm",
        Learn::Note("when_slipping"),
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
    (SHAFT_JOINT, SHAFT_JOINT_GUIDE.learn),
    (GEARBOX, GEARBOX_GUIDE.learn),
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

/// Where the Size tab's Why links under Magnets per ring and Outside diameter (spec 3.7) and on
/// the torque card's heading (slice 3 spec 6.1) lead: how the coupling makes torque, Start
/// here's first step.
pub const SIZE_TAB_NOTE: &str = "torque_from_shear";

/// Where the Why links at the end of the Size tab's Required and Hot-low rows lead: the hot
/// minimum check's note (a test keeps it [`HOT_MIN_GUIDE`]'s).
pub const TORQUE_BAND_NOTE: &str = "hot_low_cold_high";

/// Where the Why link under the end-effect banner leads: the end-effect check's note (a test
/// keeps it [`END_EFFECT_GUIDE`]'s).
pub const BANNER_NOTE: &str = "end_effect";

/// The guide of `check` (a dashboard check or [`FIT_CHECK`]); `None` for any other path.
pub fn guide(check: &str) -> Option<&'static CheckGuide> {
    CHECK_GUIDES.iter().find(|guide| guide.check == check)
}

/// Whether `guide`'s check counts while the shaft-clamp section is paused (`clamps_paused`) or
/// not: every guide but the clamp's while paused ([`crate::engine::clamps::hides`] its check).
/// A guide that does not count is not drawn, shows no fix, holds back no other guide's fix and
/// is not in the design status.
pub fn counts(guide: &CheckGuide, clamps_paused: bool) -> bool {
    !crate::engine::clamps::hides(guide.check, clamps_paused)
}

/// The guides that count while the shaft-clamp section is paused (`clamps_paused`) or not
/// ([`counts`]), in [`CHECK_GUIDES`]' order.
pub fn counted_guides(clamps_paused: bool) -> impl Iterator<Item = &'static CheckGuide> {
    CHECK_GUIDES
        .iter()
        .filter(move |guide| counts(guide, clamps_paused))
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

// ------------------------------------------------------------------ the check terms

/// The name the torque card gives the cup wall's quoted quantity, the wall the flux needs
/// (`materials.backiron_thickness_needed_mm`): "Cup wall: wall needed +0.07 mm" (slice 3 spec
/// 4.1; a draft until the slice's G-3 review sets it).
pub const WALL_NEEDED: &str = "wall needed";

/// The name the torque card gives the gearbox check's quoted quantity, cold-high at the gearbox
/// input (`metal.cold_high_input_Nm`): the torque it is (slice 3 spec T-11), "Gearbox input
/// torque: cold-high +0.03 N·m"; a draft until the slice's G-3 review sets it.
pub const COLD_HIGH: &str = "cold-high";

/// The quantity the torque card quotes for a term of a check that fails (slice 3 spec 3.6): the
/// result at `path`, named `name` after the check's label where the label alone does not name
/// it, to `decimals` places of the result's own unit (its metadata's, never typed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Quoted {
    pub path: &'static str,
    pub name: Option<&'static str>,
    pub decimals: usize,
}

impl Quoted {
    /// The result at `path`, named `name`, to `decimals` places: the one constructor of
    /// [`check_terms`]' table and of its test's expected rows.
    pub(crate) const fn new(
        path: &'static str,
        name: Option<&'static str>,
        decimals: usize,
    ) -> Self {
        Quoted {
            path,
            name,
            decimals,
        }
    }
}

/// One term of a counted check's verdict (slice 3 spec 3.6): its name, its margin (larger is
/// better), whether it passes as the verdict compares it, and the quantity the torque card
/// quotes for it, if any.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CheckTerm {
    pub name: String,
    pub margin: f64,
    pub passes: bool,
    pub quoted: Option<Quoted>,
}

/// The terms of `check`'s verdict, in a fixed order so two designs' lists pair up: the one table
/// the fix tests' margins and the torque card's read (slice 3 spec 3.6). The temperature terms
/// are the guide's own ([`TemperatureTerm`]), named as they print, and quote nothing (T-13: they
/// read stored 3D fields that do not follow a step). The shaft joints' are the engine's four
/// causes of each joint, inner hub first ([`shaft_joint::joints`]), always eight: a joint not
/// judged (no wall, a reversed band) or a check that cannot be made passes them. The clamp has
/// none here (the fix tests add their own; its verdict moves with no single slider step), nor
/// has a path that is no guide's check.
pub(crate) fn check_terms(
    check: &str,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<CheckTerm> {
    let term = |name: &str, margin: f64, passes, quoted| CheckTerm {
        name: name.to_owned(),
        margin,
        passes,
        quoted,
    };
    let at_least_zero = |name, margin: f64, quoted| term(name, margin, margin >= 0.0, quoted);
    let (m, md) = (&results.model, &results.metal);
    match check {
        // The row's own number is this term's quantity: never quoted again.
        HOT_MIN => vec![at_least_zero(
            "hot-low torque",
            md.torque_hot_low_Nm - inputs.metal.required_min_Nm,
            None,
        )],
        CLEARANCE => vec![at_least_zero(
            "running clearance",
            md.min_running_clearance_mm - inputs.metal.residual_target_mm,
            Some(Quoted::new("metal.min_running_clearance_mm", None, 2)),
        )],
        CUP_WALL => vec![at_least_zero(
            "cup wall",
            results.materials.cup_wall_corner_mm - results.materials.backiron_thickness_needed_mm,
            Some(Quoted::new(
                "materials.backiron_thickness_needed_mm",
                Some(WALL_NEEDED),
                2,
            )),
        )],
        TEMPERATURE => TemperatureTerm::ALL
            .into_iter()
            .map(|t| {
                let margin = t.margin(inputs, results);
                term(&format!("{t:?}"), margin, !t.fails(inputs, results), None)
            })
            .collect(),
        SPACE_CLAIM => ["diameter", "overall length", "large-diameter bay"]
            .into_iter()
            .zip(axis_reserves(results))
            .zip(SPACE_CLAIM_AXIS_SIZES)
            .map(|((name, reserve), (axis, size))| {
                at_least_zero(name, reserve, Some(Quoted::new(size, Some(axis), 2)))
            })
            .collect(),
        GEARBOX => vec![at_least_zero(
            "gearbox input torque",
            inputs.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
            Some(Quoted::new("metal.cold_high_input_Nm", Some(COLD_HIGH), 2)),
        )],
        // The engine's four terms of each joint, with the guide's margins (BL-064 spec 5.2). A
        // joint that is not judged (no wall, a reversed band, or the check cannot be made) passes
        // them; with no keyway the key terms pass with a margin of +inf.
        SHAFT_JOINT => {
            // Not judged while the check cannot be made (`shaft_joints` is an `Err`).
            let judged = shaft_joints(inputs, results).is_ok();
            let links = shaft_joint_links(inputs, results);
            shaft_joint::joints(&inputs.shaft_joint, &links)
                .iter()
                .flat_map(|joint| {
                    CAUSES.map(|cause| {
                        let passes = !(judged && joint.determinate() && joint.fails(cause));
                        let quoted = joint_side(joint.name).map(|(_, side)| {
                            let (path, decimals) = side.quoted(cause);
                            Quoted::new(path, Some(label_for(path)), decimals)
                        });
                        term(
                            &joint_term(joint.name, cause),
                            joint.margin(cause),
                            passes,
                            quoted,
                        )
                    })
                })
                .collect()
        }
        END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0, None)],
        // The engine's rule ring by ring; the margin is the flat's slack, or what is left of the
        // arcs' share of the circle.
        FIT_CHECK => ring_fits(&inputs.coupling, m)
            .into_iter()
            .zip(["inner ring", "outer ring"])
            .map(|(ring, name)| {
                let margin = match ring {
                    RingFit::Flat { width_mm, flat_mm } => flat_mm - width_mm,
                    RingFit::Arc { share } => 1.0 - share,
                };
                term(name, margin, ring.fits(), None)
            })
            .collect(),
        // CLAMP, and any path that is no guide's check.
        _ => Vec::new(),
    }
}

/// The name of the shaft joint `joint`'s term `cause` in [`check_terms`]: "inner hub slips",
/// "cup bore stress", "inner hub fitting", "cup key".
fn joint_term(joint: &str, cause: Cause) -> String {
    let term = match cause {
        Cause::Slips => "slips",
        Cause::BoreStress => "bore stress",
        Cause::CannotBeFitted => "fitting",
        Cause::Key => "key",
    };
    format!("{joint} {term}")
}

// ------------------------------------------------------------------ the fixes

/// The fixes of `guide` to show for this design, in order; none while its check passes, and
/// none for a guide that does not count while the shaft-clamp section is paused
/// (`clamps_paused`, [`counts`]). Of a static list, each fix that can still move and whose
/// move turns no passing guided check red ([`helps`]); a list built per failing term
/// ([`Fixes::PerTerm`]) is checked by its builder, against the term each fix serves. Each
/// check runs `compute_all` on the design a move reaches, so the panel asks through a
/// [`FixCache`]. The fit guide also leaves out "Raise Outside diameter" where no diameter on
/// its slider fits the magnets ([`fits_on_the_slider`]).
pub fn fixes(
    guide: &CheckGuide,
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<(&'static str, Direction)> {
    if !counts(guide, clamps_paused) || level(guide, inputs, results) == Level::Good {
        return Vec::new();
    }
    match guide.fixes {
        // Each fix of a static list moves its check toward passing from a failing design to
        // its slider's end (a test checks each one), so only the step itself is checked.
        Fixes::Static(list) => list
            .iter()
            .copied()
            .filter(|&fix| {
                let raises_the_fit_diameter =
                    guide.check == FIT_CHECK && fix == (RING, Direction::Raise);
                !raises_the_fit_diameter || fits_on_the_slider(inputs)
            })
            .filter(|&fix| helps(inputs, results, fix, None, clamps_paused))
            .collect(),
        Fixes::PerTerm(build) => build(inputs, results, clamps_paused),
    }
}

/// Whether the design's magnet count fits at some diameter on the Outside diameter slider
/// (without the panel's Minimum OD, as [`moves`] steps it). Where it does not, "Raise Outside
/// diameter" cannot make the blocks fit, and the Size tab says so beside it ("Does not fit at
/// any diameter up to ..."), so the fit guide leaves that fix out (physics review 3, N3): one
/// step does widen each block's place, but no step on the slider makes it wide enough. A
/// design without size controls keeps the fix: [`helps`] still judges it.
fn fits_on_the_slider(inputs: &DesignInputs) -> bool {
    !matches!(
        smallest_fitting_diameter(inputs, inputs.coupling.npole, None),
        Some(DiameterFit::Nowhere(_))
    )
}

/// The Fix line of a failing check whose [`fixes`] are empty (spec 3.4 gives every block a
/// Fix): its inputs are at their sliders' ends, have no better choice or would turn another
/// check red, or a value the check needs is not a number or is outside what it can be, which
/// the Problem names.
pub const NO_FIX: &str = "No listed change helps from here: each is at its slider's end, has no better choice or would fail another check, or a value the check needs is not a number or is outside what it can be.";

/// The notes to draw under the fixes of the guide at `index`, where another failing check's
/// fix moves one of its inputs the other way (spec review focus 4). `all` holds every guide's
/// [`fixes`] in [`CHECK_GUIDES`]' order, as [`FixCache::get`] gives them. Both fixes stay, since
/// each helps its own check; the note, the same in both blocks, says that moving the input
/// settles one check and sets the other back. Two checks already failing are the only way
/// there: a fix whose step turns a passing check red is not offered ([`helps`]).
pub fn trade_offs(all: &[Vec<(&'static str, Direction)>], index: usize) -> Vec<String> {
    let (Some(own), Some(guide)) = (all.get(index), CHECK_GUIDES.get(index)) else {
        return Vec::new();
    };
    let mut notes = Vec::new();
    for &(path, direction) in own {
        let opposite = match direction {
            Direction::Raise => Direction::Lower,
            Direction::Lower => Direction::Raise,
            Direction::Change | Direction::CustomHotterGrade => continue,
        };
        for (other, other_fixes) in CHECK_GUIDES.iter().zip(all) {
            if other.check == guide.check || !other_fixes.contains(&(path, opposite)) {
                continue;
            }
            let (raising, lowering) = match direction {
                Direction::Raise => (guide, other),
                _ => (other, guide),
            };
            notes.push(format!(
                "One input pulls two ways ({}): {} needs it raised and {} needs it lowered, so change another input to settle one of them.",
                fix_label(path),
                raising.label,
                lowering.label
            ));
        }
    }
    notes
}

/// A fix that also sets back another guide's check, by physics the fix's own check does not
/// see. [`helps`] blocks only a step that turns a passing check red, so a check that already
/// fails could be pushed further without a word; [`side_effects`] says so instead.
struct SideEffect {
    /// The fix.
    fix: (&'static str, Direction),
    /// The check it sets back.
    check: &'static str,
    /// What the move does, after "Raising {input} also ".
    effect: &'static str,
    /// What follows for the check, after its guide's label.
    outcome: &'static str,
}

/// The fixes with a side effect (physics review 2, N2): a larger face gap opens the running
/// clearance, but lowers the gap flux density and so the pull-out the hot-low torque is
/// taken from.
const SIDE_EFFECTS: [SideEffect; 1] = [SideEffect {
    fix: (FACE_GAP, Direction::Raise),
    check: HOT_MIN,
    effect: "lowers the pull-out",
    outcome: "falls further",
}];

/// The notes to draw under the fixes of the guide at `index` (in `all`, every guide's
/// [`fixes`] as [`FixCache::get`] gives them) for each shown fix that sets back another check
/// that is not passing in this design ([`SIDE_EFFECTS`]), for example "Raising Magnet
/// face gap also lowers the pull-out, so Hot-low torque falls further."
pub fn side_effects(
    all: &[Vec<(&'static str, Direction)>],
    index: usize,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<String> {
    let Some(own) = all.get(index) else {
        return Vec::new();
    };
    SIDE_EFFECTS
        .iter()
        .filter(|effect| own.contains(&effect.fix))
        .filter_map(|effect| {
            let other = guide(effect.check)?;
            if level(other, inputs, results) == Level::Good {
                return None;
            }
            let (path, direction) = effect.fix;
            let verb = match direction {
                Direction::Raise => "Raising",
                Direction::Lower => "Lowering",
                Direction::Change | Direction::CustomHotterGrade => "Changing",
            };
            Some(format!(
                "{verb} {} also {}, so {} {}.",
                fix_label(path),
                effect.effect,
                other.label,
                effect.outcome
            ))
        })
        .collect()
}

/// The fixes whose cost no guided check sees, each with the note drawn while one of them is
/// shown (physics review 3, S14): the retaining sleeve and the keeper liner go down to 0 mm on
/// their sliders, which opens the running clearance but takes away what holds the magnets,
/// and their hoop stress, buckling and bond-loss capture are validation items only.
const UNCHECKED_COSTS: [(&[(&str, Direction)], &str); 1] = [(
    &[
        ("metal.sleeve_mm", Direction::Lower),
        ("metal.liner_mm", Direction::Lower),
    ],
    "A thinner sleeve or liner holds the magnets less securely; no check here covers that, so confirm it separately.",
)];

/// The [`UNCHECKED_COSTS`] notes of the fixes shown in `own`, each note once.
fn unchecked_costs(own: &[(&'static str, Direction)]) -> Vec<String> {
    UNCHECKED_COSTS
        .iter()
        .filter(|(costly, _)| costly.iter().any(|fix| own.contains(fix)))
        .map(|&(_, note)| note.to_owned())
        .collect()
}

/// Every note drawn under the fixes of the guide at `index`: its [`trade_offs`], its
/// [`side_effects`], then the costs no check sees ([`unchecked_costs`]).
pub fn guide_notes(
    all: &[Vec<(&'static str, Direction)>],
    index: usize,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<String> {
    let mut notes = trade_offs(all, index);
    notes.extend(side_effects(all, index, inputs, results));
    if let Some(own) = all.get(index) {
        notes.extend(unchecked_costs(own));
    }
    notes
}

/// Every guide's [`fixes`] for the design they were built for (spec 4.1, as for the
/// consequences): the panel asks every frame, and each fix runs `compute_all` on the design
/// its move reaches. They are computed again only when the inputs or the shaft-clamp
/// section's pause change.
#[derive(Default)]
pub struct FixCache {
    last: Option<CachedFixes>,
    /// How many times the fixes were computed (the tests check that a repeat computes
    /// nothing).
    pub(crate) computed: usize,
}

/// What a [`FixCache`] holds: the inputs, the shaft-clamp section's pause and every guide's
/// fixes for them.
struct CachedFixes {
    inputs: DesignInputs,
    clamps_paused: bool,
    fixes: Vec<Vec<(&'static str, Direction)>>,
}

impl FixCache {
    /// Each guide's [`fixes`] for `inputs` (whose results are `results`) with the shaft-clamp
    /// section's pause (`clamps_paused`), in [`CHECK_GUIDES`]' order, none for a check that
    /// passes or does not count; computed only when the inputs or the pause differ from the
    /// last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        clamps_paused: bool,
    ) -> &[Vec<(&'static str, Direction)>] {
        if self
            .last
            .as_ref()
            .is_none_or(|last| last.inputs != *inputs || last.clamps_paused != clamps_paused)
        {
            self.computed += 1;
            self.last = Some(CachedFixes {
                inputs: inputs.clone(),
                clamps_paused,
                fixes: all_fixes(inputs, results, clamps_paused),
            });
        }
        &self.last.as_ref().expect("filled above").fixes
    }
}

/// Each guide's [`fixes`] for `inputs` (whose results are `results`) with the shaft-clamp
/// section's pause (`clamps_paused`), in [`CHECK_GUIDES`]' order, none for a check that passes
/// or does not count: what [`FixCache::get`] holds, computed every call.
pub fn all_fixes(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<Vec<(&'static str, Direction)>> {
    CHECK_GUIDES
        .iter()
        .map(|guide| fixes(guide, inputs, results, clamps_paused))
        .collect()
}

/// `all` (every guide's [`fixes`], as [`FixCache::get`] gives them) as the panel shows them
/// with the Size tab's Minimum OD at `minimum_diameter_mm`: [`moves`] steps the Outside
/// diameter over its slider without that floor, so where the floor stops the slider (the
/// diameter at or below it) "Lower Outside diameter" is left out. Every other fix stays; a
/// guide left with none shows [`NO_FIX`], whose slider's end this is.
pub fn within_minimum_diameter(
    all: &[Vec<(&'static str, Direction)>],
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
) -> Vec<Vec<(&'static str, Direction)>> {
    let floored = minimum_diameter_mm
        .and_then(|minimum| dimensions_with_minimum_diameter(inputs, minimum))
        .is_some_and(|size| size.diameter_mm <= size.diameter_range.min);
    all.iter()
        .map(|list| {
            list.iter()
                .copied()
                .filter(|&fix| !(floored && fix == (RING, Direction::Lower)))
                .collect()
        })
        .collect()
}

/// A margin of a design, larger is better: what a fix must raise to help a term.
type Margin<'a> = &'a dyn Fn(&DesignInputs, &DesignResults) -> f64;

/// Whether `fix` is worth showing for `inputs` (whose results are `results`): one of the
/// designs its move reaches ([`moves`]) turns no guided check that passes here red (review
/// focus 3) and, given a `margin`, raises it. Only the checks that count with the shaft-clamp
/// section's pause (`clamps_paused`, [`counts`]) hold a fix back: a hidden check never does.
fn helps(
    inputs: &DesignInputs,
    results: &DesignResults,
    (path, direction): (&str, Direction),
    margin: Option<Margin>,
    clamps_paused: bool,
) -> bool {
    let counted: Vec<&CheckGuide> = counted_guides(clamps_paused).collect();
    let before: Vec<Level> = counted
        .iter()
        .map(|guide| level(guide, inputs, results))
        .collect();
    moves(inputs, results, path, direction)
        .iter()
        .any(|design| {
            let after = compute_all(design);
            let reddens = counted.iter().zip(&before).any(|(guide, &was)| {
                was == Level::Good && level(guide, design, &after) == Level::Bad
            });
            let raises = margin.is_none_or(|margin| {
                let (to, from) = (margin(design, &after), margin(inputs, results));
                // A margin that is not a number fails whatever it is: any number is better.
                to > from || (from.is_nan() && !to.is_nan())
            });
            !reddens && raises
        })
}

/// The designs one move of a fix reaches from `inputs`, each changing the fix's input only:
/// - **Raise or Lower:** one step of its slider that way ([`step`]). The ring radius steps the
///   Size tab's Outside diameter slider instead: its fix bears that name
///   ([`SIZE_TAB_ALIASES`]) and its link opens that tab, and the slider's mapping moves the
///   cap and its thread with the cup (`engine::size_controls`). The panel's Minimum OD is not
///   seen here: the slider's range is the one without it.
/// - **Change:** each other choice: a selector's codes, a part picker's "Custom dimensions" (a
///   blank part) and its library parts, a grade picker's "Blank" and its grades, each kept
///   only where it leaves the ring's magnet rated ([`rated`]). So "Custom dimensions" counts
///   only with a grade already picked, and a grade only on manual dimensions (its label says
///   so: with a library part it changes nothing).
/// - **CustomHotterGrade:** the ring's part to "Custom dimensions" with each grade rated
///   hotter than its library part ([`custom_hotter_grade`]): this changes the part and the
///   grade of one ring.
fn moves(
    inputs: &DesignInputs,
    results: &DesignResults,
    path: &str,
    direction: Direction,
) -> Vec<DesignInputs> {
    let Some(entry) = InputCatalogue::get().entry(path) else {
        return Vec::new();
    };
    if direction == Direction::CustomHotterGrade {
        return custom_hotter_grade(inputs, path);
    }
    if direction == Direction::Change {
        // A picker's choices: its blank one ("Custom dimensions", or "Blank"), then its table.
        let picker = |table: Vec<&str>| -> Vec<Value> {
            std::iter::once("")
                .chain(table)
                .map(|text| Value::Text(text.to_owned()))
                .collect()
        };
        let choices = if PART_PATHS.contains(&path) {
            picker(MAGNET_LIBRARY.iter().map(|spec| spec.part).collect())
        } else if GRADE_PATHS.contains(&path) {
            picker(GRADES.iter().map(|grade| grade.id).collect())
        } else {
            entry
                .meta
                .choices
                .iter()
                .map(|&(code, _)| Value::Int(code))
                .collect()
        };
        let current = inputs.get(path);
        return choices
            .into_iter()
            .filter(|choice| current.as_ref() != Some(choice))
            .filter_map(|choice| with_value(inputs, path, choice))
            .filter(|design| rated(path, design))
            .collect();
    }
    if path == RING {
        return dimensions(inputs)
            .and_then(|size| step(size.diameter_mm, size.diameter_range, direction))
            .and_then(|diameter| resize_diameter(inputs, diameter).ok())
            .into_iter()
            .collect();
    }
    let (Some(range), Some(x)) = (entry.meta.range, current_number(inputs, results, path)) else {
        return Vec::new();
    };
    step(x, range, direction)
        .and_then(|next| with_number(inputs, path, next))
        .into_iter()
        .collect()
}

/// Whether `design`, reached by moving the input at `path`, leaves the ring that input sets
/// with a rated magnet: a library part, or a grade picked for manual dimensions; for a grade
/// input, that grade in use. A magnet with neither ("Custom dimensions" with a blank grade) has
/// no rating, so the engine leaves its demagnetization limit uncalibrated: the default block
/// reads about 10 °C above the same block from the library. A move there would pass the
/// temperature check with no change of magnet (review N-1). Any other input's move counts.
fn rated(path: &str, design: &DesignInputs) -> bool {
    let Some(ring) = magnet_ring(path) else {
        return true;
    };
    let magnet = ring.resolved(design);
    if path == ring.grade() {
        magnet.from_grade
    } else {
        matches!(magnet.tmax_C, NumOrText::Num(_))
    }
}

/// The designs a [`Direction::CustomHotterGrade`] fix on the part input at `path` reaches:
/// the ring's part set to "Custom dimensions" (a blank part) and its grade to each grade rated
/// hotter than the part. None for a ring that is not on a library part: there a grade applies
/// already, and the grade's own fix offers it.
fn custom_hotter_grade(inputs: &DesignInputs, path: &str) -> Vec<DesignInputs> {
    let Some(ring) = magnet_ring(path).filter(|ring| ring.part() == path) else {
        return Vec::new();
    };
    let on_library_part =
        matches!(inputs.get(path), Some(Value::Text(part)) if lookup(&part).is_some());
    if !on_library_part {
        return Vec::new();
    }
    // A library part always has a rating.
    let NumOrText::Num(rating) = ring.resolved(inputs).tmax_C else {
        return Vec::new();
    };
    GRADES
        .iter()
        .filter(|grade| grade.tmax_C > rating)
        .filter_map(|grade| {
            let custom = with_value(inputs, path, Value::Text(String::new()))?;
            with_value(&custom, ring.grade(), Value::Text(grade.id.to_owned()))
        })
        .collect()
}

/// `x` one step of the slider `range` up (Raise) or down (Lower), kept on the slider: `None`
/// at the slider's end that way, for a number that is not one, and for a choice. A value
/// typed past the slider's end comes back onto it ([`SliderRange::step_from`], the step a
/// teaching note's change entry takes too).
fn step(x: f64, range: SliderRange, direction: Direction) -> Option<f64> {
    match direction {
        Direction::Raise => range.step_from(x, true),
        Direction::Lower => range.step_from(x, false),
        Direction::Change | Direction::CustomHotterGrade => None,
    }
}

/// `inputs` with the input at `path` set to `value`; `None` where the inputs refuse it.
fn with_value(inputs: &DesignInputs, path: &str, value: Value) -> Option<DesignInputs> {
    let mut moved = inputs.clone();
    moved.set(path, value).ok()?;
    Some(moved)
}

/// [`with_value`] with the number `x`, as an integer for an integer input.
fn with_number(inputs: &DesignInputs, path: &str, x: f64) -> Option<DesignInputs> {
    let value = match InputCatalogue::get().entry(path)?.default {
        Value::Int(_) => Value::Int(x.round() as i64),
        _ => Value::Num(x),
    };
    with_value(inputs, path, value)
}

/// A number of a value: a number or an integer; `None` for a text or no value.
fn number(value: Value) -> Option<f64> {
    match value {
        Value::Num(x) => Some(x),
        Value::Int(i) => Some(i as f64),
        Value::Text(_) | Value::None => None,
    }
}

/// The result at `path` as a number ([`number`]); NaN for a text, no value or an unknown path.
/// The torque card reads its quoted quantities with it.
pub(crate) fn result_number(results: &DesignResults, path: &str) -> f64 {
    number(results.get(path).unwrap_or(Value::None)).unwrap_or(f64::NAN)
}

/// The number the input at `path` holds, or for an optional input left blank the result it
/// starts from ([`optional_seed`]: magnets of their parts' own length); `None` for a text.
fn current_number(inputs: &DesignInputs, results: &DesignResults, path: &str) -> Option<f64> {
    match inputs.get(path)? {
        Value::None => number(results.get(optional_seed(path)?)?),
        value => number(value),
    }
}

/// A fix's text: its direction, then the input's label as the inputs side shows it (its display
/// label, `gui::plain_text`: "Raise Magnets per ring"), or the Size tab's name for it
/// ([`SIZE_TAB_ALIASES`], which wins over the display label in fix text only). A
/// label that another input's label equals or ends with ("Boss outside diameter" against
/// "Shaft boss outside diameter") is followed by its group's heading, so the fix names one
/// input: "Raise Boss outside diameter (Shaft clamps)". A [`Direction::CustomHotterGrade`] fix
/// names its two picks: "Change Inner magnet part to Custom dimensions (manual), enter the
/// part's dimensions, then set Inner magnet grade (manual dimensions) to one rated hotter than
/// the part". A path that is no input reads as itself (a test keeps every guide's fixes on
/// real inputs).
pub fn fix_text(path: &str, direction: Direction) -> String {
    let text = format!("{} {}", direction.word(), fix_label(path));
    match (direction, magnet_ring(path)) {
        (Direction::CustomHotterGrade, Some(ring)) => format!(
            "{text} to {CUSTOM}, enter the part's dimensions, then set {} to one rated hotter than the part",
            fix_label(ring.grade())
        ),
        _ => text,
    }
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
    // C-3 compares the labels drawn on screen: the display labels (spec 3.5).
    let label = display_label(path, entry.meta.label);
    let shared = catalogue.all().any(|other| {
        other.path != path && ends_with_label(display_label(&other.path, other.meta.label), label)
    });
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

/// A magnet ring. Each ring is checked against its own magnet's limits; the weaker ring sets
/// the design's magnet limit, and the cold check passes only if both rings pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ring {
    Inner,
    Outer,
}

impl Ring {
    const BOTH: [Ring; 2] = [Ring::Inner, Ring::Outer];

    /// The ring's name in a sentence.
    const fn name(self) -> &'static str {
        match self {
            Ring::Inner => "inner",
            Ring::Outer => "outer",
        }
    }

    /// The ring's own magnet limit: its value and the result that holds it.
    fn magnet_limit(self, results: &DesignResults) -> (f64, &'static str) {
        let d = &results.temperature.demag;
        match self {
            Ring::Inner => (
                d.inner_magnet_limit_C,
                "temperature.demag.inner_magnet_limit_C",
            ),
            Ring::Outer => (
                d.outer_magnet_limit_C,
                "temperature.demag.outer_magnet_limit_C",
            ),
        }
    }

    /// The ring's own cold limit ("n/a" for a magnet whose coercivity rises as it cools): its
    /// value and the result that holds it.
    fn cold_limit(self, results: &DesignResults) -> (NumOrText, &'static str) {
        let d = &results.temperature.demag;
        match self {
            Ring::Inner => (d.inner_cold_limit_C, "temperature.demag.inner_cold_limit_C"),
            Ring::Outer => (d.outer_cold_limit_C, "temperature.demag.outer_cold_limit_C"),
        }
    }

    /// The ring's magnet fixes: its part, and its grade, which applies to manual dimensions.
    const fn magnet(self) -> &'static [(&'static str, Direction)] {
        match self {
            Ring::Inner => &INNER_MAGNET,
            Ring::Outer => &OUTER_MAGNET,
        }
    }

    /// The ring's magnet fixes against its hot-day limit: [`Ring::magnet`]'s, then the two
    /// picks past the hottest-rated library part ([`Direction::CustomHotterGrade`]).
    const fn hot_magnet(self) -> &'static [(&'static str, Direction)] {
        match self {
            Ring::Inner => &INNER_HOT_MAGNET,
            Ring::Outer => &OUTER_HOT_MAGNET,
        }
    }

    /// The ring's part input.
    const fn part(self) -> &'static str {
        match self {
            Ring::Inner => PART_PATHS[0],
            Ring::Outer => PART_PATHS[1],
        }
    }

    /// The ring's grade input (manual dimensions).
    const fn grade(self) -> &'static str {
        match self {
            Ring::Inner => GRADE_PATHS[0],
            Ring::Outer => GRADE_PATHS[1],
        }
    }

    /// The ring's magnet in `inputs` as the engine resolves it, with every approved correction
    /// as [`compute_all`] has them: its rating, and whether a grade picked for manual
    /// dimensions is in use.
    fn resolved(self, inputs: &DesignInputs) -> ResolvedMagnet {
        let (inner, outer) = resolve_magnets(&inputs.coupling.magnets, Deviations::ALL);
        match self {
            Ring::Inner => inner,
            Ring::Outer => outer,
        }
    }
}

/// The ring whose magnet the part or grade input at `path` sets; `None` for any other input.
fn magnet_ring(path: &str) -> Option<Ring> {
    Ring::BOTH
        .into_iter()
        .find(|ring| ring.part() == path || ring.grade() == path)
}

/// The inner ring's magnet inputs ([`Ring::magnet`]).
const INNER_MAGNET: [(&str, Direction); 2] = [
    (Ring::Inner.part(), Direction::Change),
    (Ring::Inner.grade(), Direction::Change),
];

/// The outer ring's magnet inputs ([`Ring::magnet`]).
const OUTER_MAGNET: [(&str, Direction); 2] = [
    (Ring::Outer.part(), Direction::Change),
    (Ring::Outer.grade(), Direction::Change),
];

/// The inner ring's magnet fixes against its hot-day limit ([`Ring::hot_magnet`]).
const INNER_HOT_MAGNET: [(&str, Direction); 3] = [
    INNER_MAGNET[0],
    INNER_MAGNET[1],
    (Ring::Inner.part(), Direction::CustomHotterGrade),
];

/// The outer ring's magnet fixes against its hot-day limit ([`Ring::hot_magnet`]).
const OUTER_HOT_MAGNET: [(&str, Direction); 3] = [
    OUTER_MAGNET[0],
    OUTER_MAGNET[1],
    (Ring::Outer.part(), Direction::CustomHotterGrade),
];

/// The adhesive: its limit and its cure temperature.
const ADHESIVE: [(&str, Direction); 1] = [("temperature.adhesive.selected", Direction::Change)];

/// Less heat at the slip-fault peak: a shorter fault, a better path out for the heat. Neither
/// moves the hot-day start (the ambient plus the driving rise), so neither is a hot-day fix,
/// and neither can bring the peak below a limit that start already reaches
/// ([`TemperatureTerm::peak_start_and_limit`]).
const COOLING: [(&str, Direction); 2] = [
    ("temperature.duty.fault_trip_s", Direction::Lower),
    ("temperature.thermal.conductance_W_K", Direction::Raise),
];

/// A term of the temperature verdict, in the verdict's order; the verdict passes when none
/// fails (a test checks). The hot-day margin is split by limit, and the magnet limit and the
/// cold check by ring, so that each fix is tested against the one term it can move: a hotter
/// magnet in one ring does not raise a limit the other ring also sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemperatureTerm {
    /// The hot-day start is not below this ring's magnet limit.
    HotDayMagnet(Ring),
    /// The hot-day start is not below the adhesive limit.
    HotDayAdhesive,
    /// The slip-fault peak reaches the magnet limit.
    MagnetPeak,
    /// The slip-fault peak reaches the adhesive limit.
    AdhesivePeak,
    /// The adhesive cures less than [`CURE_MARGIN_MIN_C`] below the single-ring
    /// demagnetization onset.
    Cure,
    /// This ring's coercivity falls on cooling (ferrite), and the minimum magnet temperature
    /// is below its cold limit.
    Cold(Ring),
}

impl TemperatureTerm {
    const ALL: [TemperatureTerm; 8] = [
        TemperatureTerm::HotDayMagnet(Ring::Inner),
        TemperatureTerm::HotDayMagnet(Ring::Outer),
        TemperatureTerm::HotDayAdhesive,
        TemperatureTerm::MagnetPeak,
        TemperatureTerm::AdhesivePeak,
        TemperatureTerm::Cure,
        TemperatureTerm::Cold(Ring::Inner),
        TemperatureTerm::Cold(Ring::Outer),
    ];

    /// How far the term is from failing, larger is better: what a fix must raise.
    fn margin(self, inputs: &DesignInputs, results: &DesignResults) -> f64 {
        let t = &results.temperature;
        let start = t.summary.hot_day_start_C;
        match self {
            TemperatureTerm::HotDayMagnet(ring) => ring.magnet_limit(results).0 - start,
            TemperatureTerm::HotDayAdhesive => t.summary.adhesive_limit_C - start,
            TemperatureTerm::MagnetPeak => t.magnet_life.margin_limit_C,
            TemperatureTerm::AdhesivePeak => t.adhesive_life.margin_C,
            TemperatureTerm::Cure => t.summary.cure_margin_C - CURE_MARGIN_MIN_C,
            TemperatureTerm::Cold(ring) => match ring.cold_limit(results).0 {
                NumOrText::Num(limit) => inputs.metal.min_temp_C - limit,
                NumOrText::Text(_) => f64::INFINITY,
            },
        }
    }

    /// Whether the term fails, by the verdict's own comparisons: a hot-day or peak margin must
    /// be above zero, the cure and cold margins at least zero, and a margin that is not a
    /// number fails. A ring's hot-day term fails only while the design's does: the engine sets
    /// the design's limit from the other ring when one ring's limit is not a number.
    fn fails(self, inputs: &DesignInputs, results: &DesignResults) -> bool {
        let margin = self.margin(inputs, results);
        match self {
            TemperatureTerm::HotDayMagnet(_) => {
                let s = &results.temperature.summary;
                not_positive(margin) && not_positive(s.magnet_limit_C - s.hot_day_start_C)
            }
            TemperatureTerm::HotDayAdhesive
            | TemperatureTerm::MagnetPeak
            | TemperatureTerm::AdhesivePeak => not_positive(margin),
            TemperatureTerm::Cure | TemperatureTerm::Cold(_) => margin.is_nan() || margin < 0.0,
        }
    }

    /// For a slip-fault peak, the hot-day start (the ambient plus the driving rise) and the
    /// peak's limit; `None` for any other term. The engine builds the peak from that start by
    /// adding heat (`peak = max(T_fault, T0 + rise per event) + duty x steady rise`, no added
    /// term below zero), so the peak is never below the start: once the start reaches the
    /// limit, no shorter fault and no better heat path brings the peak under it, and the
    /// hot-day term's own fixes are the ones that clear both (physics review 2, N1).
    fn peak_start_and_limit(self, results: &DesignResults) -> Option<(f64, f64)> {
        let s = &results.temperature.summary;
        match self {
            TemperatureTerm::MagnetPeak => Some((s.hot_day_start_C, s.magnet_limit_C)),
            TemperatureTerm::AdhesivePeak => Some((s.hot_day_start_C, s.adhesive_limit_C)),
            _ => None,
        }
    }

    /// Whether the term is a slip-fault peak whose limit the hot-day start already reaches
    /// ([`TemperatureTerm::peak_start_and_limit`]).
    fn start_reaches_peak_limit(self, results: &DesignResults) -> bool {
        self.peak_start_and_limit(results)
            .is_some_and(|(start, limit)| start >= limit)
    }

    /// The inputs that can move the term toward passing; [`temperature_fixes`] shows each only
    /// where one of its moves does.
    const fn fixes(self) -> &'static [(&'static str, Direction)] {
        match self {
            TemperatureTerm::HotDayMagnet(ring) => ring.hot_magnet(),
            TemperatureTerm::Cold(ring) => ring.magnet(),
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
fn failing_terms(inputs: &DesignInputs, results: &DesignResults) -> Vec<TemperatureTerm> {
    TemperatureTerm::ALL
        .into_iter()
        .filter(|term| term.fails(inputs, results))
        .collect()
}

/// The temperature Problem's sentences, numbers from the results, in the verdict's order: the
/// hot-day start against each limit it is not below, each slip-fault peak that reaches its
/// limit, the cure margin, and the cold limits the minimum temperature is below. A ring's
/// limit is named with its ring, since each ring needs its own fix.
fn temperature_sentences(inputs: &DesignInputs, results: &DesignResults) -> Vec<String> {
    use TemperatureTerm::{AdhesivePeak, Cold, Cure, HotDayAdhesive, HotDayMagnet, MagnetPeak};
    let failing = failing_terms(inputs, results);
    let fails = |term| failing.contains(&term);
    let mut sentences = Vec::new();
    // A ring whose limit is not a number fails its term but is not quoted as a number.
    let (unknown_rings, hot_rings): (Vec<Ring>, Vec<Ring>) = Ring::BOTH
        .into_iter()
        .filter(|&ring| fails(HotDayMagnet(ring)))
        .partition(|&ring| ring.magnet_limit(results).0.is_nan());
    let mut hot_limits: Vec<String> = hot_rings
        .iter()
        .copied()
        .map(|ring| {
            let (_, limit) = ring.magnet_limit(results);
            format!(
                "the {} magnets' {} limit",
                ring.name(),
                result_text(results, limit)
            )
        })
        .collect();
    if fails(HotDayAdhesive) {
        hot_limits.push(format!(
            "the {} adhesive limit",
            result_text(results, "temperature.summary.adhesive_limit_C")
        ));
    }
    if !hot_limits.is_empty() {
        sentences.push(format!(
            "Hot-day magnet temperature {} (hot ambient plus the driving rise) is at or above {}.",
            result_text(results, "temperature.summary.hot_day_start_C"),
            listed(&hot_limits, "and")
        ));
    }
    for ring in unknown_rings {
        sentences.push(format!(
            "The {} magnets' limit is not a number, so the hot-day magnet temperature cannot be checked against it.",
            ring.name()
        ));
    }
    // A peak whose limit the hot-day start already reaches says nothing the hot-day sentence
    // does not: the peak is never below the start ([`TemperatureTerm::start_reaches_peak_limit`]).
    let magnet_peak_quoted = fails(MagnetPeak) && !MagnetPeak.start_reaches_peak_limit(results);
    if magnet_peak_quoted {
        sentences.push(format!(
            "The slip-fault peak of {} is at or above the {} magnet limit.",
            result_text(results, "temperature.magnet_life.peak_C"),
            result_text(results, "temperature.summary.magnet_limit_C"),
        ));
    }
    if fails(AdhesivePeak) && !AdhesivePeak.start_reaches_peak_limit(results) {
        sentences.push(format!(
            "The slip-fault peak of {} is at or above the {} adhesive limit.",
            result_text(results, "temperature.adhesive_life.peak_C"),
            result_text(results, "temperature.summary.adhesive_limit_C"),
        ));
    }
    // The peak sentence quotes the design's magnet limit: the lower ring's.
    let peak_rings = Ring::BOTH.into_iter().filter(|&ring| {
        magnet_peak_quoted
            && ring.magnet_limit(results).0 == results.temperature.summary.magnet_limit_C
    });
    let quoted: Vec<Ring> = hot_rings.into_iter().chain(peak_rings).collect();
    sentences.extend(magnet_limit_sentence(inputs, results, &quoted));
    if fails(Cure) {
        let (margin, minimum) = cure_margin_texts(results);
        sentences.push(format!(
            "The adhesive cures at {}, {margin} below the {} at which a ring alone on its carrier, as during the cure, starts to demagnetize; the margin must be at least {minimum}.",
            result_text(results, "temperature.adhesive.cure_C"),
            result_text(results, "temperature.demag.onset_single_ring_C"),
        ));
    }
    let cold_limits: Vec<String> = Ring::BOTH
        .into_iter()
        .filter(|&ring| fails(Cold(ring)))
        .map(|ring| {
            let (_, limit) = ring.cold_limit(results);
            format!(
                "the {} magnets' {} cold demagnetization limit",
                ring.name(),
                result_text(results, limit)
            )
        })
        .collect();
    // The cold limit comes from the reverse-field inputs. Their defaults are 3D results for the
    // default NdFeB design and follow neither this design's geometry nor its grade (spec
    // section 8, trap 6): while they are in use the sentence says so and asks for the design's
    // own; once a user has entered their own, it says nothing about where they came from.
    if !cold_limits.is_empty() {
        let source = if default_reverse_fields(inputs) {
            " (from the default reverse-field inputs: 3D results for the default NdFeB design, not for this design's geometry or grade). Enter this design's own reverse fields to check the cold limit."
        } else {
            "."
        };
        sentences.push(format!(
            "The minimum magnet temperature {} is below {}{source}",
            input_text(inputs, "metal.min_temp_C"),
            listed(&cold_limits, "and")
        ));
    }
    sentences
}

/// What a quoted hot magnet limit is (physics review 2, S2): for NdFeB, the like-pole
/// (skipping) onset less the design margin, the temperature at which the field of a slip
/// starts to demagnetize the ring, not a running rating; without it an 80 °C-rated part with
/// a limit below zero reads as a bug. `quoted` holds the rings whose limit the Problem quotes
/// (a ring may be listed twice); a ferrite ring's hot limit is its rating, which needs no
/// sentence, so only the others are named. `None` when none is left.
fn magnet_limit_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    quoted: &[Ring],
) -> Option<String> {
    let onset_rings: Vec<Ring> = Ring::BOTH
        .into_iter()
        .filter(|ring| quoted.contains(ring))
        .filter(|ring| matches!(ring.cold_limit(results).0, NumOrText::Text(_)))
        .collect();
    let margin = input_text(inputs, "temperature.demag.design_margin_C");
    // The onset comes from the stored reverse fields, as the cold limit does (physics review
    // 3, S16): while they hold the default design's, the sentence says so.
    let source = if default_reverse_fields(inputs) {
        " (slip fields from the default design; enter this design's own to confirm)"
    } else {
        ""
    };
    match onset_rings[..] {
        [] => None,
        [ring] => Some(format!(
            "The {} magnets' limit is the temperature at which the field of a slip starts to demagnetize them, less the {margin} design margin{source}.",
            ring.name()
        )),
        _ => Some(format!(
            "Each ring's magnet limit is the temperature at which the field of a slip starts to demagnetize its magnets, less the {margin} design margin{source}."
        )),
    }
}

/// Whether all four reverse-field inputs of the demagnetization check hold their defaults: the
/// 3D results for the default NdFeB design.
fn default_reverse_fields(inputs: &DesignInputs) -> bool {
    let fields = |d: &DemagInputs| {
        [
            d.h_rev_aligned_kA_m,
            d.h_rev_pullout_kA_m,
            d.h_rev_likepole_kA_m,
            d.h_rev_single_ring_kA_m,
        ]
    };
    fields(&inputs.temperature.demag) == fields(&DemagInputs::default())
}

/// The cure margin and the minimum it must reach, each with as many decimals as the onset the
/// margin is measured from shows, so the numbers printed beside it subtract ("6.0 °C below
/// the 126.0 °C ... at least 10.0 °C"). Both read as any result does where the onset's text
/// has no fixed decimals (an exponent, or not a number), and where that rounding would carry a
/// failing margin up to the minimum (9.96 read as "10.0" beside "at least 10.0 °C").
fn cure_margin_texts(results: &DesignResults) -> (String, String) {
    const MARGIN: &str = "temperature.summary.cure_margin_C";
    let margin = results.temperature.summary.cure_margin_C;
    let as_results = || {
        (
            result_text(results, MARGIN),
            number_text(CURE_MARGIN_MIN_C, MARGIN),
        )
    };
    match shown_decimals(results.temperature.demag.onset_single_ring_C) {
        Some(decimals) if margin.is_finite() => {
            let shown = fixed(margin, decimals);
            let rounded_up_to_pass = margin < CURE_MARGIN_MIN_C
                && shown.parse::<f64>().is_ok_and(|x| x >= CURE_MARGIN_MIN_C);
            if rounded_up_to_pass {
                return as_results();
            }
            let unit = result_info(MARGIN).map_or("", |info| info.meta.unit);
            (
                with_unit(shown, unit),
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals), unit),
            )
        }
        _ => as_results(),
    }
}

/// `items` as one list joined by `conjunction` ("and" or "or"): "a", "a and b", "a, b and c".
fn listed(items: &[String], conjunction: &str) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [rest @ .., last] => format!("{} {conjunction} {last}", rest.join(", ")),
    }
}

/// The temperature guide's fixes ([`Fixes::PerTerm`]): for each failing term in order, each
/// of its fixes that one of its moves helps ([`helps`] against the term's margin), each fix
/// once. A slip-fault peak gets its cooling only while the hot-day start is below the same
/// limit ([`TemperatureTerm::peak_start_and_limit`]): otherwise a step lowers the peak by a
/// fraction of a degree, which [`helps`] would accept, but no move brings it under. A term
/// none of whose fixes helps from this design shows none. The two picks past the library
/// ([`Direction::CustomHotterGrade`]) show only where the part's own fix does not: a library
/// part that helps is the simpler change.
fn temperature_fixes(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<(&'static str, Direction)> {
    let mut shown: Vec<(&'static str, Direction)> = Vec::new();
    for term in failing_terms(inputs, results) {
        let margin = |design: &DesignInputs, after: &DesignResults| term.margin(design, after);
        // Cooling only while the start is below the peak's limit; a start or limit that is
        // not a number gives none either.
        if term
            .peak_start_and_limit(results)
            .is_some_and(|(start, limit)| start >= limit || start.is_nan() || limit.is_nan())
        {
            continue;
        }
        for &fix in term.fixes() {
            let (path, direction) = fix;
            let part_shown = direction == Direction::CustomHotterGrade
                && shown.contains(&(path, Direction::Change));
            if !shown.contains(&fix)
                && !part_shown
                && helps(inputs, results, fix, Some(&margin), clamps_paused)
            {
                shown.push(fix);
            }
        }
    }
    shown
}

// ------------------------------------------------------------------ the design status

/// The design status line's start for the defaults (spec G-6 as slice 2 amends it, S2-5).
pub const EXAMPLE_DESIGN: &str = "Example design";

/// The hover text of the status line at the defaults: where the example's values come from.
pub const EXAMPLE_DESIGN_HOVER: &str = "The starting values from the original spreadsheet.";

/// The design status line's start for any other design.
pub const YOUR_DESIGN: &str = "Your design";

/// How the design stands against the guides' checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DesignStatus {
    /// The checks that fail (red).
    pub bad: usize,
    /// The checks that ask for a look (amber), counted apart from the failures.
    pub caution: usize,
    /// Every guide's check that counts ([`counts`]): [`CHECK_GUIDES`]'s length, less the
    /// clamp's while the shaft-clamp section is paused.
    pub total: usize,
    /// Whether the inputs are the defaults (the example design).
    pub is_default: bool,
}

/// The design status of `inputs` and their `results`, counted live from the levels of the
/// guides that count while the shaft-clamp section is paused (`clamps_paused`) or not
/// ([`counts`]): a failing clamp moves no count while paused.
pub fn design_status(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> DesignStatus {
    let counted: Vec<&CheckGuide> = counted_guides(clamps_paused).collect();
    let count = |wanted: Level| {
        counted
            .iter()
            .filter(|guide| level(guide, inputs, results) == wanted)
            .count()
    };
    DesignStatus {
        bad: count(Level::Bad),
        caution: count(Level::Caution),
        total: counted.len(),
        is_default: *inputs == DesignInputs::default(),
    }
}

/// The design status line: "Example design: fails 4 of 10 checks", "Your design: fails
/// 3 of 10 checks", "Your design: passes all 10 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 9 while the shaft-clamp section is paused.
pub fn design_status_text(status: &DesignStatus) -> String {
    let design = if status.is_default {
        EXAMPLE_DESIGN
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

// The plain-words guard lives in the engine (engine::explain::plain, pure std) so the engine's
// tests hold the teaching notes to it; the GUI's texts use it from here.
pub use crate::engine::explain::plain::{BANNED, plain_text_violations};

// ------------------------------------------------------------------ the Problem builders

/// The result at `path` as the readouts show it: four significant digits, then its unit.
pub(crate) fn result_text(results: &DesignResults, path: &str) -> String {
    let unit = result_info(path).map_or("", |info| info.meta.unit);
    with_unit(
        format_value(&results.get(path).unwrap_or(Value::None)),
        unit,
    )
}

/// The input at `path` as a readout shows it: four significant digits, then its unit.
fn input_text(inputs: &DesignInputs, path: &str) -> String {
    with_unit(
        format_value(&inputs.get(path).unwrap_or(Value::None)),
        input_unit(path),
    )
}

/// The unit of the input at `path` (none for a path that is not an input).
fn input_unit(path: &str) -> &'static str {
    InputCatalogue::get()
        .entry(path)
        .map_or("", |entry| entry.meta.unit)
}

/// A number worked out here from results and inputs (no result holds it), in the unit of the
/// result at `unit_of`.
fn number_text(x: f64, unit_of: &str) -> String {
    let unit = result_info(unit_of).map_or("", |info| info.meta.unit);
    with_unit(format_value(&Value::Num(x)), unit)
}

/// The value a number reads as: `x` to four significant digits, as [`format_value`] shows it.
fn quoted_value(x: f64) -> f64 {
    format_value(&Value::Num(x)).parse().unwrap_or(x)
}

/// Float noise in last shown digits: 0.6 × 4.75 / 2.5 is not 1.14 in binary but reads 1.140, and
/// is quoted so when rounded down or up at its last digit.
const LAST_DIGIT_NOISE: f64 = 1e-9;

/// A limit `x` as gearbox Problem sentence 2 quotes it (review I-2): rounded down at its last
/// shown digit, so a requirement or band set to the quoted limit meets it, and one digit further
/// down where it would still read equal to `limited`, the number it limits, as that is quoted.
/// No rounding makes a limit read equal to what it limits. A number that is not finite or not
/// positive (a value from a file) is quoted as it is.
fn quoted_limit(x: f64, limited: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    let down = rounded_down(x);
    if quoted_value(down) < quoted_value(limited) {
        down
    } else {
        down - last_digit_step(down)
    }
}

/// A torque `x` on the gearbox input, above the `rating`, as gearbox Problem sentence 2 quotes it
/// (review I-2): rounded up at its last shown digit, so a gearbox rated for the quoted torque
/// takes it, and one digit further up where it would still read equal to the rating as that is
/// quoted (as the gearbox verdict quotes cold-high past the rating). A number that is not
/// finite or not positive (a value from a file) is quoted as it is.
fn quoted_load(x: f64, rating: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    let up = rounded_up(x);
    if quoted_value(up) > quoted_value(rating) {
        up
    } else {
        up + last_digit_step(up)
    }
}

/// `x` rounded down to a multiple of `step`, float noise aside ([`LAST_DIGIT_NOISE`]): the one
/// rounding-down rule of the quotes.
fn floor_at(x: f64, step: f64) -> f64 {
    (x / step + LAST_DIGIT_NOISE).floor() * step
}

/// `x` rounded up to a multiple of `step`, float noise aside ([`LAST_DIGIT_NOISE`]): the one
/// rounding-up rule of the quotes.
fn ceil_at(x: f64, step: f64) -> f64 {
    (x / step - LAST_DIGIT_NOISE).ceil() * step
}

/// `x` rounded down at its last shown digit ([`floor_at`]); a number that is not finite or not
/// positive as it is.
fn rounded_down(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    floor_at(x, last_digit_step(x))
}

/// `x` rounded up at its last shown digit ([`ceil_at`]); a number that is not finite or not
/// positive as it is.
fn rounded_up(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    ceil_at(x, last_digit_step(x))
}

/// How many decimals the readouts show for `x` ([`format_value`]'s text): `None` where that
/// text has no fixed decimals (an exponent, a whole number, or not a number).
fn shown_decimals(x: f64) -> Option<usize> {
    let text = format_value(&Value::Num(x));
    match text.split_once('.') {
        Some((_, decimals)) if !text.contains('e') => Some(decimals.len()),
        _ => None,
    }
}

/// The band and the widest band the rating allows, as gearbox Problem sentence 2's first remedy
/// quotes them: the band as the readouts show it, the widest band rounded down past it
/// ([`quoted_limit`]). With the minimum temperature at or below the operating one no band is
/// below one, so a widest band under one by float noise is one; and where the band reads 1.000
/// with the widest band from one up to it, the readouts' digits cannot quote the widest band
/// below the band without quoting a band below one, which no change reaches (review NB-1). Both
/// then get more decimals, as few as tell them apart, the widest band still rounded down and
/// never below one: "1.0004" and "1.0000".
fn band_quotes(band: f64, widest: f64) -> (String, String) {
    let widest = widest.max(1.0);
    let limit = quoted_limit(widest, band);
    if limit >= 1.0 {
        return (
            format_value(&Value::Num(band)),
            format_value(&Value::Num(limit)),
        );
    }
    // Both are from 1 to 1.0005, which the readouts show to three decimals: from four.
    let at = |decimals: usize| {
        let step = 10f64.powi(-(decimals as i32));
        let down = floor_at(widest, step).max(1.0);
        (fixed(band, decimals), fixed(down, decimals))
    };
    let apart = |(here, limit): &(String, String)| {
        let value = |text: &str| text.parse().unwrap_or(f64::NAN);
        value(limit) < value(here)
    };
    fewest_decimals_apart(SIGNIFICANT_DIGITS as usize, at, apart).1
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

/// The temperature Problem when no term reads as failing although the verdict does
/// ([`joined_or`]).
const TEMPERATURE_FALLBACK: &str = "A temperature limit is not met: see the temperature results.";

/// The space claim's Problem when no axis reads as over its claim although the check fails
/// ([`joined_or`]).
const SPACE_CLAIM_FALLBACK: &str = "The coupling is outside its space claim.";

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
            "Radial running gap {} less {} of movement allowances leaves {}, below the {} target.",
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
            "Cup wall at the pocket corners is {}; the flux it carries needs {} at the {} design flux density.",
            result_text(results, "materials.cup_wall_corner_mm"),
            result_text(results, "materials.backiron_thickness_needed_mm"),
            result_text(results, "model.bsat_T"),
        )
    })
}

fn temperature_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(TEMPERATURE, inputs, results)
        .then(|| joined_or(temperature_sentences(inputs, results), TEMPERATURE_FALLBACK))
}

fn clamp_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(CLAMP, inputs, results).then(|| {
        format!(
            "No screw size from {} to {} fits the boss and the clamp length with enough screws to hold {}: the highest torque the coupling can pass (cold, plus the {} variation allowance), {}, times a safety factor of {}.",
            SCREW_SIZES[0].name,
            SCREW_SIZES[SCREW_SIZES.len() - 1].name,
            result_text(results, "clamps.required_Nm"),
            percent_text(inputs.metal.variation),
            result_text(results, "clamps.max_torque_Nm"),
            input_text(inputs, "clamps.safety_factor"),
        )
    })
}

/// The shaft joints' Problem when the check cannot be made: a value it reads or works out is not
/// a number (a bore typed at 0 reads only numbers, and its holding works out as NaN).
const SHAFT_JOINT_NOT_A_NUMBER: &str =
    "The shaft joint check cannot be made: a value it reads or works out is not a number.";

/// The shaft joints' Problem when no sentence reads as failing although the check does
/// ([`joined_or`]).
const SHAFT_JOINT_FALLBACK: &str = "A shaft joint does not pass: see the shaft joint results.";

/// The shaft joints' Problem (spec 5.2): while the check cannot be made (a value that is not a
/// number, a safety factor not above zero, ruling C-4, an input typed below what it can
/// physically be, its extension, or a gearbox shaft typed below its slider's bottom, build
/// ruling B-5), its sentences alone ([`cannot_be_made_problem`]); else a
/// sentence per term a judged joint fails, in the badge's order ([`shaft_joint::failing`]),
/// with its live numbers; then, for each judged joint failing a slip, stress or fitting term,
/// the window of measured interference that would work, or why there is none
/// ([`window_sentence`]); then each joint that is not judged, and why
/// ([`not_judged_sentences`]).
fn shaft_joint_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    if !failing(SHAFT_JOINT, inputs, results) {
        return None;
    }
    let joints = match shaft_joints(inputs, results) {
        Ok(joints) => joints,
        Err(reason) => return Some(cannot_be_made_problem(inputs, &reason)),
    };
    let mut sentences: Vec<String> = shaft_joint::failing(&joints)
        .into_iter()
        .filter_map(|(cause, joint)| {
            let (_, side) = joint_side(joint.name)?;
            Some(term_sentence(inputs, results, side, joint, cause))
        })
        .collect();
    for (side, joint) in JOINT_SIDES.iter().zip(&joints) {
        let windowed = joint.determinate()
            && [Cause::Slips, Cause::BoreStress, Cause::CannotBeFitted]
                .into_iter()
                .any(|cause| joint.fails(cause));
        if windowed {
            sentences.extend(window_sentence(inputs, results, side));
        }
    }
    for (side, joint) in JOINT_SIDES.iter().zip(&joints) {
        sentences.extend(not_judged_sentences(inputs, side, joint));
    }
    Some(joined_or(sentences, SHAFT_JOINT_FALLBACK))
}

/// The shaft joints' Problem while the check cannot be made, with no fix: the not-a-number
/// sentence; the safety factor's (ruling C-4); a sentence for each input typed below what it
/// can physically be, in the inputs' order ([`impossible_value_sentence`]); or the gearbox
/// shaft's below its slider's bottom, the coldest the check takes it since the gearbox's rated
/// range is not checked (build ruling B-5), its value quoted so that it reads below that bottom
/// ([`input_text_below`]).
fn cannot_be_made_problem(inputs: &DesignInputs, reason: &CannotBeMade) -> String {
    match reason {
        CannotBeMade::NotANumber => SHAFT_JOINT_NOT_A_NUMBER.to_owned(),
        CannotBeMade::SafetyFactor => format!(
            "The safety factor, {}, is not above zero: enter at least 1.",
            input_text(inputs, SAFETY_FACTOR)
        ),
        CannotBeMade::ImpossibleValues(values) => values
            .iter()
            .map(|value| impossible_value_sentence(inputs, value))
            .collect::<Vec<_>>()
            .join(" "),
        CannotBeMade::GearboxShaftTooCold => {
            let bottom = slider_bottom_text(GEARBOX_SHAFT);
            format!(
                "{}, {}, is below {bottom}, the coldest the gearbox shaft is taken, since its rated range is not checked: enter at least {bottom}, its slider's bottom.",
                fix_label(GEARBOX_SHAFT),
                input_text_below(inputs, GEARBOX_SHAFT, slider_bottom(GEARBOX_SHAFT)),
            )
        }
    }
}

/// The bottom of the slider of the input at `path` ([`slider_range`]). Panics where `path` is
/// not an input with a slider: the guide asks only for those.
fn slider_bottom(path: &str) -> f64 {
    slider_range(path)
        .map(|range| range.min)
        .unwrap_or_else(|| panic!("{path} has a slider"))
}

/// The bottom of the slider of the input at `path` ([`slider_bottom`]), with its unit, as a
/// readout shows it.
fn slider_bottom_text(path: &str) -> String {
    with_unit(
        format_value(&Value::Num(slider_bottom(path))),
        input_unit(path),
    )
}

/// The input at `path`, typed below `bottom`, as a Problem quotes it (G-3 re-review of the
/// review leftovers, MF-1): as its readout shows it where that reads below `bottom` as a readout
/// shows it; else with the fewest more decimals that read it below ([`fewest_decimals_apart`]),
/// rounded to the nearest as the readouts round, so it never reads as the bottom it is below,
/// as [`apart_from`] keeps a failing result off its bound. A value not below `bottom`, or not a
/// number, reads as its readout shows it.
fn input_text_below(inputs: &DesignInputs, path: &str, bottom: f64) -> String {
    let readout = input_text(inputs, path);
    let Some(x) = inputs.get(path).and_then(number) else {
        return readout;
    };
    let below_bottom = x < bottom;
    if !below_bottom || quoted_value(x) < quoted_value(bottom) {
        return readout;
    }
    let first = shown_decimals(x).map_or(SIGNIFICANT_DIGITS as usize, |decimals| decimals + 1);
    let reads_below = |text: &String| text.parse().is_ok_and(|q: f64| q < quoted_value(bottom));
    let (_, text) = fewest_decimals_apart(first, |decimals| fixed(x, decimals), reads_below);
    with_unit(text, input_unit(path))
}

/// The sentence of an input typed below what it can physically be (the whole-branch review's
/// MF-1): its name as the fixes give it ([`fix_label`]), its value, its floor, and its slider's
/// bottom to enter instead, as the safety factor's sentence asks for at least its own. The
/// sentence calls that number its slider's bottom: it is not the floor the sentence names, and
/// for the two shafts it is a fitting temperature limit, which names its source (the text
/// re-review, MF-1).
fn impossible_value_sentence(inputs: &DesignInputs, value: &ImpossibleValue) -> String {
    let path = format!("shaft_joint.{}", value.field);
    let floor = match value.floor {
        Floor::Zero => "below zero",
        Floor::AboveZero => "zero or below",
        Floor::AbsoluteZero => "below absolute zero",
    };
    format!(
        "{}, {}, cannot physically be {floor}: enter at least {}, its slider's bottom.",
        fix_label(&path),
        input_text(inputs, &path),
        slider_bottom_text(&path),
    )
}

/// The sentence of a term `cause` the joint of `side` fails, its numbers live, a failing value
/// never read equal to its bound ([`apart_from`]).
fn term_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    joint: &JointTerms,
    cause: Cause,
) -> String {
    let name = side.name;
    match cause {
        Cause::Slips => format!(
            "At its loosest the {name} holds {}, less than the {} it must hold: cold-high, {}, times the safety factor of {}.",
            apart_from(results, side.holding, joint.required_Nm, false),
            result_text(results, REQUIRED),
            result_text(results, "metal.torque_cold_high_Nm"),
            input_text(inputs, SAFETY_FACTOR),
        ),
        Cause::BoreStress => format!(
            "At its tightest the {name}'s bore stress is {}, above the {} limit ({} yield with a margin).",
            apart_from(results, side.stress, joint.stress_limit_MPa, true),
            result_text(results, STRESS_LIMIT),
            result_text(results, PART_YIELD),
        ),
        Cause::CannotBeFitted => format!(
            "Fitting the {name} would need it at {} with the {} at {}, above the {} allowed: {}.",
            apart_from(results, side.fit_temp, joint.fit_allowed_C, true),
            side.shaft,
            input_text(inputs, side.shaft_fit),
            result_text(results, side.fit_allowed),
            fitting_source(inputs, results, side),
        ),
        Cause::Key => key_sentence(inputs, results, side, joint),
    }
}

/// What sets the allowed fitting temperature of the joint of `side` (trap 9): the cup's is your
/// limit for the bare cup, whose number the sentence already quotes as the allowed temperature;
/// the bonded inner hub's is the lowest of three (owner R-15), each with its live number, named
/// by the engine's own rule ([`shaft_joint::hub_fit_allowed`]) on the engine's own links
/// ([`shaft_joint_links`]), so the number and its name come from one place.
fn fitting_source(inputs: &DesignInputs, results: &DesignResults, side: &JointSide) -> String {
    if side.name != shaft_joint::INNER_HUB {
        return "your limit for the bare cup".to_owned();
    }
    let links = shaft_joint_links(inputs, results);
    let (_, limit) = shaft_joint::hub_fit_allowed(
        inputs.shaft_joint.hub_fit_max_C,
        links.adhesive_limit_C,
        links.inner_onset_single_ring_C,
    );
    match limit {
        Some(HubFitLimit::Yours) => format!(
            "your limit for the bonded inner hub, {}",
            input_text(inputs, "shaft_joint.hub_fit_max_C")
        ),
        Some(HubFitLimit::Adhesive) => format!(
            "the selected adhesive's design limit, {}",
            result_text(results, "temperature.adhesive.design_limit_C")
        ),
        Some(HubFitLimit::MagnetOnset) => {
            let (onset, margin) = onset_texts(results);
            format!("the magnets' single-ring onset, {onset}, less {margin}")
        }
        // Not reached: the check cannot be made while one of the three is not a number.
        None => "a limit that is not a number".to_owned(),
    }
}

/// The inner ring's single-ring onset and the cure margin as the bonded inner hub's fitting
/// source quotes them (owner R-15), each with as many decimals as the allowed temperature
/// shows, so the three numbers subtract ("above the 59.83 °C allowed: the magnets' single-ring
/// onset, 69.83 °C, less 10.00 °C"), as [`cure_margin_texts`] quotes the cure margin. Both read
/// as any result does where the allowed temperature's text has no fixed decimals.
fn onset_texts(results: &DesignResults) -> (String, String) {
    const ONSET: &str = "temperature.demag.inner_onset_single_ring_C";
    match shown_decimals(results.shaft_joint.hub_fit_allowed_C) {
        Some(decimals) => {
            let unit = result_info(ONSET).map_or("", |info| info.meta.unit);
            let onset = results.temperature.demag.inner_onset_single_ring_C;
            (
                with_unit(fixed(onset, decimals), unit),
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals), unit),
            )
        }
        None => (
            result_text(results, ONSET),
            number_text(CURE_MARGIN_MIN_C, ONSET),
        ),
    }
}

/// The key term's sentence of the joint of `side` (spec 5.2): no contact height (it alone, where
/// the key has no bearing length either: no length gives it bearing area without contact); no
/// bearing length; else the key's pressure, alone, above the limit.
fn key_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    joint: &JointTerms,
) -> String {
    let name = side.name;
    let key_width = input_text(inputs, KEY_WIDTH);
    if joint.key_contact_mm <= 0.0 {
        format!(
            "The key has no bearing area in the {name}: its contact height, {}, is not above zero.",
            input_text(inputs, KEY_CONTACT)
        )
    } else if joint.key_bearing_mm <= 0.0 && name == shaft_joint::INNER_HUB {
        format!(
            "The key has no bearing area in the inner hub: the hub, {} long, is no longer than the key is wide, {key_width}.",
            result_text(results, HUB_LENGTH_IN_EFFECT)
        )
    } else if joint.key_bearing_mm <= 0.0 {
        format!(
            "The key has no bearing area in the cup: its boss and web, {} and {} long, together are no longer than the key is wide, {key_width}.",
            input_text(inputs, BOSS_LENGTH),
            input_text(inputs, WEB)
        )
    } else {
        format!(
            "Alone, the key would bear {} on the {name}, above the {} limit.",
            apart_from(results, side.key, joint.stress_limit_MPa, true),
            result_text(results, STRESS_LIMIT)
        )
    }
}

/// The window of measured interference that would work on the joint of `side` (spec 5.2):
/// from the smallest minimum that holds to the smaller of the largest maximum within the stress
/// limit and the largest that can be fitted, its clause naming the edge that binds, beside the
/// joint's band; or, where the smallest minimum is above that, why no interference works, each
/// edge below it named. The ends are quoted inward ([`window_quotes`], [`rounded_up`],
/// [`rounded_down`]), so a band typed at them passes. `None` where the smallest minimum is not
/// finite (a key wider than the bore's circumference leaves no contact): the slip sentence's
/// holding of 0 says it.
fn window_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
) -> Option<String> {
    let (needed, elastic, fittable) = (
        result_number(results, side.needed),
        result_number(results, side.elastic),
        result_number(results, side.fittable),
    );
    if !needed.is_finite() {
        return None;
    }
    let name = side.name;
    let (upper, clause) = if elastic <= fittable {
        (elastic, "stay within the stress limit")
    } else {
        (fittable, "fit at the allowed temperature")
    };
    if needed <= upper {
        let (low, high) = window_quotes(needed, upper, side.needed);
        return Some(format!(
            "On the {name}, a measured interference from {low} to {high} would hold and {clause} (its band is {} to {}).",
            input_text(inputs, side.min),
            input_text(inputs, side.max)
        ));
    }
    let mut edges = Vec::new();
    if elastic < needed {
        edges.push(format!(
            "its bore stress stays within the limit only up to {}",
            number_text(rounded_down(elastic), side.elastic)
        ));
    }
    // Strictly below 0 (review B5 I-1): at exactly 0 the bore at its allowed temperature opens
    // the joining clearance on the shaft as set, no more, so the fitting edge is named.
    if fittable < needed && fittable < 0.0 {
        edges.push(format!(
            "at its allowed temperature its bore does not even open the {} joining clearance on the {} as set",
            input_text(inputs, "shaft_joint.joining_clearance_um"),
            side.shaft
        ));
    } else if fittable < needed {
        edges.push(format!(
            "the {name} can be fitted only up to {}",
            number_text(rounded_down(fittable), side.fittable)
        ));
    }
    Some(format!(
        "No measured interference works on the {name} here: holding needs at least {}, but {}.",
        number_text(rounded_up(needed), side.needed),
        listed(&edges, "and")
    ))
}

/// A window's ends as the shaft joints' Problem quotes them, in the unit of the result at
/// `unit_of` (review SF-11, as BL-059 quotes its band): `needed` rounded up and `upper` rounded
/// down at each one's last shown digit, so a band typed at the quoted ends passes; where those
/// read equal or crossed (a window narrower than the readouts' last digit), both with more
/// decimals, as few as read them apart in order (BL-059 NB-1).
fn window_quotes(needed: f64, upper: f64, unit_of: &str) -> (String, String) {
    let (low, high) = (rounded_up(needed), rounded_down(upper));
    if quoted_value(low) < quoted_value(high) {
        return (number_text(low, unit_of), number_text(high, unit_of));
    }
    let at = |decimals: usize| {
        let step = 10f64.powi(-(decimals as i32));
        (ceil_at(needed, step), floor_at(upper, step))
    };
    let first = [low, high]
        .into_iter()
        .filter_map(shown_decimals)
        .max()
        .map_or(SIGNIFICANT_DIGITS as usize, |decimals| decimals + 1);
    let (decimals, (low, high)) = fewest_decimals_apart(first, at, |&(low, high)| low < high);
    let unit = result_info(unit_of).map_or("", |info| info.meta.unit);
    (
        with_unit(fixed(low, decimals), unit),
        with_unit(fixed(high, decimals), unit),
    )
}

/// Why the joint of `side` is not judged, if it is not (spec 4.3): no wall around its bore (the
/// bore at least as wide as the part, or the keyway through the wall), its band reversed, or
/// both. The cup's part is its boss, whose diameter the check takes for the web too, so the
/// cup's no-wall sentences name the boss and say the web alone is not judged.
fn not_judged_sentences(
    inputs: &DesignInputs,
    side: &JointSide,
    joint: &JointTerms,
) -> Vec<String> {
    let name = side.name;
    let bore = input_text(inputs, "coupling.bore_mm");
    let keyway = input_text(inputs, "coupling.keyway_depth_mm");
    let mut sentences = Vec::new();
    match joint.no_wall {
        Some(NoWall::BoreAsWideAsPart) if name == shaft_joint::INNER_HUB => sentences.push(format!(
            "The inner hub's bore, {bore}, reaches the flats its magnets are bonded on, so it leaves no wall to grip the wheel shaft."
        )),
        Some(NoWall::BoreAsWideAsPart) => sentences.push(format!(
            "The cup's bore, {bore}, is at least as wide as its boss, {}, so the boss leaves no wall to grip the gearbox shaft; the check does not judge the web alone.",
            input_text(inputs, "metal.boss_od_mm")
        )),
        Some(NoWall::KeywayThroughWall) if name == shaft_joint::INNER_HUB => {
            sentences.push(format!(
                "The inner hub's keyway, {keyway} deep, cuts through the wall around its {bore} bore, so the fit has no closed ring to grip the wheel shaft."
            ))
        }
        Some(NoWall::KeywayThroughWall) => sentences.push(format!(
            "The cup's keyway, {keyway} deep, cuts through the boss's wall around its {bore} bore, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web alone."
        )),
        None => {}
    }
    if joint.band_reversed {
        sentences.push(format!(
            "The {name}'s minimum interference, {}, is above its maximum, {}: enter the band's smaller value as the minimum.",
            input_text(inputs, side.min),
            input_text(inputs, side.max)
        ));
    }
    sentences
}

/// A failing value at `path` as a Problem quotes it beside its `bound` (BL-059 MF-1): as its
/// readout shows it, or, where it would read the same as the bound, one last digit past it: up
/// above a limit (`above`, [`quoted_load`]), down under a need ([`quoted_limit`]).
fn apart_from(results: &DesignResults, path: &str, bound: f64, above: bool) -> String {
    let x = result_number(results, path);
    let reads_apart = if above {
        quoted_value(x) > quoted_value(bound)
    } else {
        quoted_value(x) < quoted_value(bound)
    };
    match (reads_apart, above) {
        (true, _) => result_text(results, path),
        (false, true) => number_text(quoted_load(x, bound), path),
        (false, false) => number_text(quoted_limit(x, bound), path),
    }
}

/// The gearbox check's Problem when it cannot be made (it asks for a look): a value it
/// compares is not a number.
const GEARBOX_UNKNOWN: &str = "The gearbox check cannot be made: cold-high at the gearbox input or the rating is not a number.";

fn gearbox_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    match level_of(GEARBOX, inputs, results) {
        Level::Good => None,
        Level::Caution => Some(GEARBOX_UNKNOWN.to_owned()),
        Level::Bad => {
            let rating = inputs.coupling.gearbox_input_rating_Nm;
            // Where both read the same at four significant digits, cold-high is quoted one last
            // digit past the rating, as the badge and sentence 2 quote a torque on the gearbox
            // input (whole-branch review MF-1).
            let torque_text = apart_from(results, "metal.cold_high_input_Nm", rating, true);
            let overload = format!(
                "Cold-high at the gearbox input is {torque_text}, above its {} input rating.",
                input_text(inputs, "coupling.gearbox_input_rating_Nm"),
            );
            // Owner decision 4A: where no geometry passes both checks, the Problem says so.
            Some(match gearbox_conflict(inputs, results) {
                Some(conflict) => format!("{overload} {conflict}"),
                None => overload,
            })
        }
    }
}

/// The changes sentence 2 of the gearbox Problem can name as narrowing the band, in its order:
/// the variation allowance last, so that no "or" follows "backed by measurement" (it would
/// read as "backed by measurement or SmCo magnets").
const WARMER_MINIMUM: &str = "a warmer minimum temperature";
const SMCO_MAGNETS: &str = "SmCo magnets";
const SMALLER_VARIATION: &str = "a smaller variation allowance backed by measurement";

/// What narrows the band (cold-high over hot-low) of a design with the minimum temperature at
/// or below the operating one, as sentence 2 says it in brackets ("a warmer minimum
/// temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows
/// it"), each change named only where it does narrow it (ruling R-3: sentence 2 is true
/// wherever it shows).
///
/// The band is (1 + v) / (1 − v) times each ring's th(minimum) / th(operating), with
/// th(T) = 1 + α (T − 20 °C) and α the ring's Br temperature coefficient (its grade's for
/// manual dimensions with a grade, else the calibration's; at or below zero on every slider and
/// grade). A ring's share is above one only with the minimum below the operating temperature
/// and α below zero, and then falls as the minimum warms and as α rises toward zero. So a
/// warmer minimum narrows the band only below the operating temperature with a coefficient
/// below zero, SmCo magnets only there with every ring's coefficient below every SmCo grade's
/// (each ring's share then falls whichever SmCo grade it takes), and a smaller variation
/// allowance only above zero. `None` where nothing does, as at a band of one (review NB-2):
/// on the sliders a band above one always has a change that narrows it, but values from a file
/// can make one without.
fn band_levers(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    let alphas = [
        results.model.inner_alpha_br_per_C,
        results.model.outer_alpha_br_per_C,
    ];
    let spread = inputs.metal.min_temp_C < inputs.coupling.op_temp_C;
    let below_every_smco = |alpha: f64| {
        GRADES
            .iter()
            .filter(|g| matches!(g.family, GradeFamily::SmCo2_17 | GradeFamily::SmCo1_5))
            .all(|g| alpha < g.alpha_br_per_C)
    };
    let names: Vec<String> = [
        (WARMER_MINIMUM, spread && alphas.iter().any(|&a| a < 0.0)),
        (
            SMCO_MAGNETS,
            spread && alphas.iter().all(|&a| below_every_smco(a)),
        ),
        (SMALLER_VARIATION, inputs.metal.variation > 0.0),
    ]
    .into_iter()
    .filter(|&(_, narrows)| narrows)
    .map(|(change, _)| change.to_owned())
    .collect();
    // After "or" the verb agrees with the change named last; only SmCo magnets is plural.
    let verb = match names.last() {
        None => return None,
        Some(last) if last == SMCO_MAGNETS => "narrow",
        Some(_) => "narrows",
    };
    Some(format!("{} {verb} it", listed(&names, "or")))
}

/// Relative float noise in a torque gearbox Problem sentence 2 sets against the rating (review
/// NB-2). The band is a ratio of two engine torques, 1.0000000000000002 where it is exactly one
/// (no variation allowance, the minimum at the operating temperature), and the requirement over
/// ratio × efficiency can miss a rating it equals in decimals by an ulp (0.84 N·m × 5 × 0.95 is
/// 3.99 N·m). A torque only this far above the rating is at it, and at the rating passes.
const RATING_NOISE: f64 = 1e-12;

/// Problem sentence 2 of the gearbox check (owner decision 4A): where no magnet size, gap or
/// count can pass both it and the hot minimum check, the numbers and what would have to change
/// instead. Cold-high over hot-low (the band) does not depend on the geometry, the magnet count
/// or the back iron, only on the two temperatures, each ring's Br temperature coefficient (its
/// grade's, else the calibration's) and the variation allowance ([`band_levers`]). So a design
/// that just meets the requirement has cold-high the requirement times the band at the gearbox
/// output, and that over ratio × efficiency at its input: above the rating, no geometry passes
/// both checks. Above the rating means beyond float noise ([`RATING_NOISE`]): where meeting the
/// requirement puts exactly the rating on the gearbox input (a tie: the requirement times the
/// band is the rating times ratio × efficiency, at a band of one or above it), a design passes
/// both. Each limit is quoted rounded down and each torque on the gearbox input rounded up, each
/// past what it is set against ([`quoted_limit`], [`quoted_load`], [`band_quotes`]). Where
/// nothing narrows the band ([`band_levers`]) no band is offered. `None` where hot-low is not
/// above zero (the end-effect model is out of range, and the check is greyed) or a value is not
/// a number (a comparison with NaN is false). `None` also with the operating temperature below
/// the minimum (BL-067).
fn gearbox_conflict(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    let (c, md) = (&inputs.coupling, &results.metal);
    let required = inputs.metal.required_min_Nm;
    let rating = c.gearbox_input_rating_Nm;
    // Output torque per input torque in forward drive: ratio × efficiency, never zero on the
    // sliders (ratio from 1, efficiency from 0.5).
    let gear = c.gear_ratio * c.gear_efficiency;
    let band = md.torque_cold_high_Nm / md.torque_hot_low_Nm;
    let input_at_required = required * band / gear;
    // What the requirement alone, at a band of one, puts on the gearbox input.
    let alone = required / gear;
    let above_rating = |torque: f64| torque > rating * (1.0 + RATING_NOISE);
    // With the operating temperature below the minimum, hot-low is taken colder than
    // cold-high: the band's levers and its floor of one no longer hold, so no sentence 2.
    let temperatures_in_order = inputs.metal.min_temp_C <= inputs.coupling.op_temp_C;
    let holds =
        temperatures_in_order && md.torque_hot_low_Nm > 0.0 && above_rating(input_at_required);
    holds.then(|| {
        // The requirement is a hot-low torque, at the gearbox output.
        let largest_required = number_text(
            quoted_limit(rating * gear / band, required),
            "metal.torque_hot_low_Nm",
        );
        // The band is at least one (cold-high is never below hot-low with the temperatures in
        // order): where the requirement alone is above the rating, a band under one is out of the
        // levers' reach, and only the gearbox or the requirement can settle it.
        let (band_text, remedy) = if above_rating(alone) {
            (
                format_value(&Value::Num(band)),
                format!(
                    "It takes a gearbox rated for that or a requirement of at most {largest_required}: the requirement alone puts {} on the gearbox input, so no narrower band settles it.",
                    number_text(quoted_load(alone, rating), "metal.cold_high_input_Nm"),
                ),
            )
        } else if let Some(levers) = band_levers(inputs, results) {
            let (band_text, widest_band) = band_quotes(band, rating * gear / required);
            (
                band_text,
                format!(
                    "It takes a gearbox rated for that, a requirement of at most {largest_required}, or a band of at most {widest_band} times ({levers})."
                ),
            )
        } else {
            // Nothing narrows the band (no input on the sliders reaches this): no band.
            (
                format_value(&Value::Num(band)),
                format!(
                    "It takes a gearbox rated for that or a requirement of at most {largest_required}."
                ),
            )
        };
        format!(
            "No magnet size, gap or count passes both this and the hot minimum: cold-high is {band_text} times hot-low here, so even just meeting the {} requirement at the gearbox output puts {} on the gearbox input. {remedy}",
            input_text(inputs, "metal.required_min_Nm"),
            number_text(
                quoted_load(input_at_required, rating),
                "metal.cold_high_input_Nm"
            ),
        )
    })
}

fn space_claim_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(SPACE_CLAIM, inputs, results).then(|| {
        let h = &results.housing;
        // Each axis's overshoot, the overshoot's path, the claim's and the claim's name, in
        // SPACE_CLAIM_AXIS_SIZES' order (the axis's name and its dimension's path).
        let overshoots = [
            (
                h.diameter_overshoot_mm,
                "housing.diameter_overshoot_mm",
                "metal.max_diameter_mm",
                "claim",
            ),
            (
                h.length_overshoot_mm,
                "housing.length_overshoot_mm",
                "metal.max_overall_axial_mm",
                "claim",
            ),
            (
                h.bay_overshoot_mm,
                "housing.bay_overshoot_mm",
                "metal.max_large_dia_axial_mm",
                "bay",
            ),
        ];
        let sentences = SPACE_CLAIM_AXIS_SIZES
            .iter()
            .zip(overshoots)
            .filter_map(|(&(axis, dimension), (over, over_path, claim, name))| {
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
        joined_or(sentences, SPACE_CLAIM_FALLBACK)
    })
}

fn end_effect_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(END_EFFECT, inputs, results).then(|| {
        format!(
            "Magnet overlap {} long with a pole pitch of {} gives an end-effect factor of {}; the model gives no usable torque until the overlap is longer than {} and is rough just above that.",
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

/// The rings in [`ring_fits`]' order: the name a sentence starts with, the name inside one,
/// and the results that hold the block's width and its flat.
const FIT_RINGS: [(&str, &str, &str, &str); 2] = [
    (
        "Inner",
        "inner",
        "model.inner_width_mm",
        "model.inner_flat_width_mm",
    ),
    (
        "Outer",
        "outer",
        "model.outer_width_mm",
        "model.outer_flat_width_mm",
    ),
];

fn fit_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    failing(FIT_CHECK, inputs, results).then(|| {
        let n = inputs.coupling.npole;
        let mut sentences = vec![format!(
            "{n} blocks per ring do not fit at this ring radius."
        )];
        // The engine's rule, ring by ring: the rings it fails, each in its own terms.
        let rings = ring_fits(&inputs.coupling, &results.model);
        for (ring, (start, name, width, flat)) in rings.into_iter().zip(FIT_RINGS) {
            if ring.fits() {
                continue;
            }
            sentences.push(match ring {
                RingFit::Flat { .. } => format!(
                    "{start} blocks are {} wide on {} flats.",
                    result_text(results, width),
                    result_text(results, flat),
                ),
                RingFit::Arc { share } => format!(
                    "The {n} {name} blocks together need {} of the circle at their mid-radius.",
                    percent_text(share)
                ),
            });
        }
        sentences.join(" ")
    })
}

// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision).
// Two magnets fewer and two more are the torque card's count row (slice 3, decision 7A), in
// the Size tab's one number format (decision 10B).

/// The first line under Magnets per ring with Auto off or on the magnet length: the ring radius
/// (`coupling.inner_back_apothem_mm`) is its own input, the magnet count only sets how many
/// blocks go around it, and Auto on the magnet length moves no ring. The cup can still change
/// size: what a count step does to it is the torque card's count row now (slice 3 spec 4.3;
/// decision 7A moved the neighbour lines' "cup OD" there).
pub const RING_SIZE_STAYS: &str = "The ring radius stays the same when you change the magnet count; the cup can still change size.";

/// The first line under Magnets per ring for the Auto state `auto` (`None`: Auto off): what a
/// new magnet count does to the ring (slice 3 spec 4.3). [`RING_SIZE_STAYS`] with Auto off and
/// with Auto on the magnet length, which moves no ring. `None` with Auto on the ring size, which
/// sizes the ring again after a count change (the torque card's Auto line says so, decision
/// 5A), and with Auto on the magnet count itself, whose slider is locked.
pub fn magnet_count_line(auto: Option<FreeVariable>) -> Option<&'static str> {
    match auto {
        None | Some(FreeVariable::AxialLength) => Some(RING_SIZE_STAYS),
        Some(FreeVariable::RingRadius | FreeVariable::MagnetsPerRing) => None,
    }
}

/// The "what moves what" readings of a design (spec 3.3), all from the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Consequences {
    /// The torque card of the design (slice 3, [`card`]): one step of each size control each
    /// way, drawn between the Size tab's readouts and its drawing.
    pub card: TorqueCard,
    /// The smallest outside diameter at which the design's magnets fit, over the Size tab's
    /// diameter range.
    pub fit: Option<DiameterFit>,
    /// The same search with the Minimum OD preference at its lowest, kept only where `fit` is
    /// stopped by the diameter slider's floor ([`DiameterFit::AtMinimum`]): how far the
    /// coupling could go if the preference were lowered.
    pub fit_unlocked: Option<DiameterFit>,
    /// The three diameters the outside diameter is the largest of.
    pub diameter: Option<DiameterBreakdown>,
    /// The magnets and the housing that make up the overall length.
    pub length: Option<LengthBreakdown>,
}

impl Consequences {
    /// The consequences of `inputs`, whose results are `results` (`compute_all(inputs)`: each
    /// step of the torque card compares with this design's hot-low), with the Size tab's minimum
    /// diameter preference (`None`: the slider's own floor) and the shaft-clamp pause
    /// (`clamps_paused`, the panel's: the card's flips and margins run over the guides that
    /// count, slice 3 spec 3.2).
    pub fn of(
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
        clamps_paused: bool,
    ) -> Self {
        let npole = inputs.coupling.npole;
        let fit = smallest_fitting_diameter(inputs, npole, minimum_diameter_mm);
        let fit_unlocked = match fit {
            Some(DiameterFit::AtMinimum(_)) => dimensions(inputs).and_then(|size| {
                smallest_fitting_diameter(inputs, npole, Some(size.minimum_diameter_range.min))
            }),
            _ => None,
        };
        Self {
            card: card(inputs, results, minimum_diameter_mm, clamps_paused),
            fit,
            fit_unlocked,
            diameter: diameter_breakdown(inputs),
            length: length_breakdown(inputs),
        }
    }

    /// The lines under Magnets per ring (decision 7A): the line for the Auto state `auto`
    /// ([`magnet_count_line`]; `None`: Auto off), then the smallest diameter that fits. `npole`
    /// is the count the consequences were built for (`inputs.coupling.npole`). Two magnets fewer
    /// and two more are the torque card's count row.
    pub fn magnet_lines(&self, npole: i64, auto: Option<FreeVariable>) -> Vec<String> {
        let mut lines: Vec<String> = magnet_count_line(auto)
            .map(str::to_owned)
            .into_iter()
            .collect();
        lines.extend(self.fit.map(|fit| fit_line(fit, self.fit_unlocked, npole)));
        lines
    }

    /// The line under Outside diameter: the three diameters it is the largest of.
    pub fn diameter_line(&self) -> Option<String> {
        self.diameter.map(|d| {
            format!(
                "Set by: cap {}, cup {}, boss {}",
                fixed(d.cap_mm, 1),
                fixed(d.cup_mm, 1),
                with_unit(fixed(d.boss_mm, 1), "mm")
            )
        })
    }

    /// The line under Overall length: the magnets and the housing around them.
    pub fn length_line(&self) -> Option<String> {
        self.length.map(|l| {
            format!(
                "Magnets {} + housing {}",
                with_unit(fixed(l.magnets_mm, 1), "mm"),
                with_unit(fixed(l.housing_mm, 1), "mm")
            )
        })
    }
}

/// The fit line under Magnets per ring (spec 3.3). A diameter named as fitting is rounded up,
/// so the number read fits; the range ends read as the diameter slider shows them. Where the
/// slider's floor stops a fit that lowering Minimum OD would take lower (`unlocked`), the line
/// says how far and what unlocks it, rather than implying the coupling cannot get smaller.
fn fit_line(fit: DiameterFit, unlocked: Option<DiameterFit>, npole: i64) -> String {
    match fit {
        DiameterFit::At(mm) => format!(
            "Smallest outside diameter that fits {npole} magnets: {}",
            with_unit(fixed(up_to_tenth(mm), 1), "mm")
        ),
        DiameterFit::AtMinimum(mm) => match unlocked {
            Some(DiameterFit::At(lower)) if lower < mm - FIT_TOLERANCE_MM => format!(
                "Fits down to {}; set Minimum OD lower to reach it.",
                with_unit(fixed(up_to_tenth(lower), 1), "mm")
            ),
            Some(DiameterFit::AtMinimum(lower)) if lower < mm - FIT_TOLERANCE_MM => format!(
                "Fits down to {}; set Minimum OD lower to reach it.",
                with_unit(fixed(lower, 1), "mm")
            ),
            _ => format!(
                "Fits at every diameter down to the slider's lowest, {}",
                with_unit(fixed(mm, 1), "mm")
            ),
        },
        DiameterFit::Nowhere(mm) => format!(
            "Does not fit at any diameter up to {}",
            with_unit(fixed(mm, 1), "mm")
        ),
    }
}

/// Slack of [`up_to_tenth`] [mm]: far below the fit search's 1e-6 mm, far above the
/// floating-point noise on a diameter that is a whole tenth.
const TENTH_SLACK_MM: f64 = 1e-9;

/// `mm` rounded up to the lines' 0.1 mm, so a diameter they name as fitting fits: 42.0043
/// reads 42.1, since 42.0 does not fit. A value within [`TENTH_SLACK_MM`] above a whole tenth
/// stays at that tenth.
fn up_to_tenth(mm: f64) -> f64 {
    ((mm - TENTH_SLACK_MM) * 10.0).ceil() / 10.0
}

/// [`Consequences`] kept for the design they were built for (spec 4.1): the panel asks every
/// frame, and the torque card's stepped `compute_all` runs and the fit search are not per-frame
/// work. They are computed again only when the inputs, the minimum diameter or the shaft-clamp
/// pause change: one key in three parts (slice 3 spec 3.2), as [`FixCache`] keys on the pause.
#[derive(Default)]
pub struct ConsequenceCache {
    last: Option<CachedConsequences>,
    /// How many times consequences were computed (the tests check that a repeat computes
    /// nothing).
    pub(crate) computed: usize,
}

/// What a [`ConsequenceCache`] holds: the question (the design, the Minimum OD and the
/// shaft-clamp pause) and its answer.
struct CachedConsequences {
    inputs: DesignInputs,
    minimum_diameter_mm: Option<f64>,
    clamps_paused: bool,
    consequences: Consequences,
}

impl ConsequenceCache {
    /// [`Consequences::of`] `inputs` (whose results are `results`) with the minimum diameter
    /// preference and the shaft-clamp pause, computed only when the inputs, the minimum or the
    /// pause differ from the last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
        clamps_paused: bool,
    ) -> &Consequences {
        let current = self.last.as_ref().is_some_and(|last| {
            last.inputs == *inputs
                && last.minimum_diameter_mm == minimum_diameter_mm
                && last.clamps_paused == clamps_paused
        });
        if !current {
            self.computed += 1;
            self.last = Some(CachedConsequences {
                inputs: inputs.clone(),
                minimum_diameter_mm,
                clamps_paused,
                consequences: Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused),
            });
        }
        &self.last.as_ref().expect("filled above").consequences
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::explain::notes;
    use crate::engine::grades::Grade;
    use crate::engine::model::ring_fits;
    use crate::engine::shaft_joint::CAUSES;
    use crate::engine::sizing::{SizingOutcome, solve};
    use crate::gui::dashboard::{CHECKS, DASHBOARD, FIX, dashboard_ui};
    use crate::gui::explorer::reviewed_note;
    use crate::gui::plain_text::label_for;
    use crate::gui::readouts::Readouts;
    use crate::gui::test_support::{
        QUOTED_UNITS, assert_glyphs, auto_states, clamp_at_its_limit, drawn_texts, end_effect_edge,
        glyph_context, hot_day, hub_overstressed, is_shaft_joint, keyway_through, mixed_rings,
        n42_parts, only_the_clamp_fails, only_the_gearbox_fails, only_the_joint_fails,
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
        short_magnets, sized_frame, slider_range, small_boss, with_magnets, with_poles,
    };
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

    /// Every value on the slider of the input at `path`, bottom to top, one step apart, each the
    /// decimal it reads (k / 100 on a slider of step 0.01): the number a typed or loaded value
    /// is, not the bottom plus k steps, which can miss it by an ulp.
    fn slider_values(path: &str) -> Vec<f64> {
        let range = slider_range(path);
        let per_unit = (1.0 / range.step).round();
        let (first, last) = (
            (range.min * per_unit).round() as i64,
            (range.max * per_unit).round() as i64,
        );
        (first..=last).map(|k| k as f64 / per_unit).collect()
    }

    /// The defaults with the gearbox ratio, then the efficiency, at the bottom and at the top of
    /// its slider, each named for its input and value. At the two bottoms the requirement alone
    /// is above the rating at the gearbox input, which no test design reaches.
    fn gear_slider_ends() -> Vec<(String, DesignInputs)> {
        let mut ends = Vec::new();
        for path in ["coupling.gear_ratio", "coupling.gear_efficiency"] {
            let range = slider_range(path);
            for end in [range.min, range.max] {
                let inputs = with_number(&DesignInputs::default(), path, end).unwrap();
                ends.push((format!("{path} at {end}"), inputs));
            }
        }
        ends
    }

    /// [`passing_design`] with `edit` applied: it fails what `edit` breaks, so a fix that turns
    /// another check red shows.
    fn breaking(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
        with(passing_design(), edit)
    }

    /// A hot day with the inner ring on manual dimensions and a blank grade: no library part
    /// beats it any more, a hotter grade does.
    fn custom_hot_day(i: &mut DesignInputs) {
        hot_day(i);
        i.coupling.magnets.part_inner.clear();
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

    /// Both rings on manual dimensions of the grade `id`, which then sets each ring's Br
    /// temperature coefficient.
    fn graded(i: &mut DesignInputs, id: &str) {
        i.coupling.magnets.part_inner.clear();
        i.coupling.magnets.part_outer.clear();
        i.coupling.magnets.grade_inner = id.to_owned();
        i.coupling.magnets.grade_outer = id.to_owned();
    }

    /// Manual Y30 ferrite blocks: the cold demagnetization check fails.
    fn ferrite(i: &mut DesignInputs) {
        graded(i, "Y30");
    }

    /// The SmCo grades of the grade table, in its order.
    fn smco_grades() -> Vec<&'static Grade> {
        GRADES
            .iter()
            .filter(|g| matches!(g.family, GradeFamily::SmCo2_17 | GradeFamily::SmCo1_5))
            .collect()
    }

    /// Designs where gearbox Problem sentence 2 shows its band and names only some of what
    /// narrows it (review I-1), each with whether it names a warmer minimum temperature, a
    /// smaller variation and SmCo magnets: the defaults, the Br coefficient at its slider ends,
    /// rings of a SmCo grade, one ring just below every SmCo grade's coefficient beside one of a
    /// SmCo grade, no variation allowance and the minimum at the operating temperature.
    fn band_lever_designs() -> Vec<(&'static str, DesignInputs, [bool; 3])> {
        let smco = smco_grades();
        let lowest_smco = smco
            .iter()
            .map(|g| g.alpha_br_per_C)
            .fold(f64::INFINITY, f64::min);
        let alpha = slider_range("calibration.alpha_br_per_C");
        let at_alpha = |x: f64| design(|i| i.calibration.alpha_br_per_C = x);
        vec![
            ("defaults", DesignInputs::default(), [true, true, true]),
            (
                "Br coefficient at its bottom",
                at_alpha(alpha.min),
                [true, true, true],
            ),
            (
                "Br coefficient at its top",
                at_alpha(alpha.max),
                [false, true, false],
            ),
            (
                "SmCo rings",
                design(|i| graded(i, smco[0].id)),
                [true, true, false],
            ),
            // The inner ring (a library part, at the calibration's coefficient) one slider step
            // below every SmCo grade's; the outer of the first SmCo grade. Both rings of the
            // lowest-coefficient SmCo grade widen the band: the outer ring's share rises more
            // than the inner's falls.
            (
                "one ring below every SmCo grade, one of a SmCo grade",
                design(|i| {
                    i.calibration.alpha_br_per_C = lowest_smco - alpha.step;
                    i.coupling.magnets.part_outer.clear();
                    i.coupling.magnets.grade_outer = smco[0].id.to_owned();
                }),
                [true, true, false],
            ),
            (
                "no variation allowance",
                design(|i| i.metal.variation = slider_range("metal.variation").min),
                [true, false, true],
            ),
            (
                "minimum at the operating temperature",
                design(|i| i.coupling.op_temp_C = i.metal.min_temp_C),
                [false, true, false],
            ),
        ]
    }

    /// A design whose band reads 1.000 but is above one (review NB-1): the Br coefficient one
    /// slider step below zero, no variation allowance, the minimum (-40 °C) 20 °C below the
    /// operating temperature. Its band is about 1.0004.
    fn band_reading_one() -> DesignInputs {
        let alpha = slider_range("calibration.alpha_br_per_C");
        design(|i| {
            i.calibration.alpha_br_per_C = alpha.max - alpha.step;
            i.metal.variation = slider_range("metal.variation").min;
            i.coupling.op_temp_C = -20.0;
        })
    }

    /// The designs with a band of one (review NB-2), each named: no variation allowance, with the
    /// minimum at the operating temperature (both -20 °C; the engine's band there is one ulp
    /// above one) or with no Br temperature coefficient (its slider's top; the band exactly one).
    fn bands_of_one() -> [(&'static str, DesignInputs); 2] {
        let no_variation = slider_range("metal.variation").min;
        [
            (
                "minimum at the operating temperature",
                design(|i| {
                    i.metal.variation = no_variation;
                    i.metal.min_temp_C = -20.0;
                    i.coupling.op_temp_C = -20.0;
                }),
            ),
            (
                "no Br temperature coefficient",
                design(|i| {
                    i.metal.variation = no_variation;
                    i.calibration.alpha_br_per_C = slider_range("calibration.alpha_br_per_C").max;
                }),
            ),
        ]
    }

    /// Designs whose band is above one and a ratio of small whole numbers, so that a requirement
    /// on its slider times the band can be, in decimals, a rating times ratio times efficiency
    /// (review I-3, round 3), each named: no Br temperature coefficient (its slider's top) at the
    /// defaults' variation allowance, a band of 1.15/0.85 = 23/17; and every lever of the band
    /// live, a Br coefficient of -0.002 1/°C, a variation allowance of 0.10, the minimum
    /// temperature at -20 °C and the operating at 80 °C, a band of (1.08/0.88)² × 1.10/0.90 =
    /// 81/44.
    fn short_fraction_bands() -> [(&'static str, DesignInputs); 2] {
        [
            (
                "no Br temperature coefficient, the defaults' allowance",
                design(|i| {
                    i.calibration.alpha_br_per_C = slider_range("calibration.alpha_br_per_C").max;
                }),
            ),
            (
                "every band lever live",
                design(|i| {
                    i.calibration.alpha_br_per_C = -0.002;
                    i.metal.variation = 0.10;
                    i.metal.min_temp_C = -20.0;
                    i.coupling.op_temp_C = 80.0;
                }),
            ),
        ]
    }

    /// `base` at its gearbox efficiency and at an efficiency of one (the slider's top), each with
    /// every rating on its slider whose product with the ratio and efficiency is, in decimals, a
    /// requirement on its slider times `band`, and that requirement: a design that just meets
    /// it, at that band, puts exactly the rating on the gearbox input, though float arithmetic
    /// can put it an ulp either side. At a `band` of one, what the requirement alone puts there
    /// (review NB-2); at the base's own band, what meeting the requirement puts there, a tie
    /// (review I-3).
    fn exact_products(base: &DesignInputs, band: f64) -> Vec<(String, DesignInputs)> {
        let requirements = slider_values("metal.required_min_Nm");
        let mut found = Vec::new();
        for efficiency in [
            base.coupling.gear_efficiency,
            slider_range("coupling.gear_efficiency").max,
        ] {
            let gear = base.coupling.gear_ratio * efficiency;
            for rating in slider_values("coupling.gearbox_input_rating_Nm") {
                // Ratio, efficiency and rating have five decimals between them, so where `band`
                // is one, or a ratio p/q of whole numbers, a requirement on the slider that is
                // not such a product misses one by 1e-5 / q or more.
                if let Some(&required) = requirements
                    .iter()
                    .find(|&&required| (required * band - rating * gear).abs() < 1e-9)
                {
                    let inputs = with(base.clone(), |i| {
                        i.coupling.gear_efficiency = efficiency;
                        i.coupling.gearbox_input_rating_Nm = rating;
                        i.metal.required_min_Nm = required;
                    });
                    let name =
                        format!("efficiency {efficiency}, rating {rating}, required {required}");
                    found.push((name, inputs));
                }
            }
        }
        found
    }

    /// A design and results whose band is above one with nothing that narrows it, which no input
    /// on the sliders reaches (review NB-2): the results of the defaults with no variation
    /// allowance and the minimum at the operating temperature (a band of one), cold-high then
    /// edited by hand to 1.5 times hot-low. The gearbox check fails, and the requirement alone
    /// is under the rating at the gearbox input.
    fn nothing_narrows_the_band() -> (DesignInputs, DesignResults) {
        let inputs = design(|i| {
            i.metal.variation = slider_range("metal.variation").min;
            i.coupling.op_temp_C = i.metal.min_temp_C;
        });
        let mut results = compute_all(&inputs);
        results.metal.torque_cold_high_Nm = 1.5 * results.metal.torque_hot_low_Nm;
        (inputs, results)
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

    /// A gearbox input rating that is not a number (set on the struct): the gearbox check asks
    /// for a look.
    fn unknown_rating(i: &mut DesignInputs) {
        i.coupling.gearbox_input_rating_Nm = f64::NAN;
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

    /// The cup's minimum interference at 10 µm: at its loosest the cup slips.
    fn cup_slips(i: &mut DesignInputs) {
        i.shaft_joint.cup_interference_min_um = 10.0;
    }

    /// The cup's maximum interference at 26 µm: at its tightest its bore stress is above its
    /// limit.
    fn cup_overstressed(i: &mut DesignInputs) {
        i.shaft_joint.cup_interference_max_um = 26.0;
    }

    /// The cup's maximum at 24 µm: its bore stress is above the limit (348.3 against 345.8 MPa)
    /// and it still fits (296.4 against 300 °C).
    fn cup_stress_alone(i: &mut DesignInputs) {
        i.shaft_joint.cup_interference_max_um = 24.0;
    }

    /// The wheel shaft at 20 °C for fitting, not cooled: the inner hub cannot be fitted.
    fn uncooled_wheel_shaft(i: &mut DesignInputs) {
        i.shaft_joint.wheel_shaft_fit_C = 20.0;
    }

    /// The bare cup allowed only 250 °C for fitting: the cup cannot be fitted.
    fn low_cup_fit_limit(i: &mut DesignInputs) {
        i.shaft_joint.cup_fit_max_C = 250.0;
    }

    /// No boss length: the cup's boss and web are no longer than the key is wide, so the cup's
    /// key has no bearing length (and the cup slips).
    fn no_boss(i: &mut DesignInputs) {
        i.metal.boss_length_mm = 0.0;
    }

    /// A 3 mm inner hub: no longer than the key is wide, so its key has no bearing length (and
    /// it slips).
    fn short_hub(i: &mut DesignInputs) {
        i.metal.hub_length_mm = 3.0;
    }

    /// Every design the tests use: the defaults (they fail four checks), a passing design,
    /// and for every check designs that fail it, most of them the passing design with one
    /// check broken.
    fn designs() -> Vec<(&'static str, DesignInputs)> {
        vec![
            ("defaults", DesignInputs::default()),
            ("passing", passing_design()),
            ("hot_day", breaking(hot_day)),
            ("custom_hot_day", breaking(custom_hot_day)),
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
            ("twelve_poles", with_poles(12)),
            (
                "twelve_poles_with_room",
                breaking(|i| i.coupling.npole = 12),
            ),
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
            ("gearbox_only", only_the_gearbox_fails()),
            // BL-064 (spec 7.9): the shaft joints, each design failing them alone. The inner hub
            // slips; its bore stress and fitting temperature are over (a 26 µm maximum); it
            // cannot be fitted on an uncooled wheel shaft; the cup's boss and web, and then the
            // inner hub, are no longer than the key is wide (each also slips); the cup's bore
            // stress alone is over (a 24 µm maximum).
            ("joint_slips", only_the_joint_fails()),
            ("joint_overstressed", breaking(hub_overstressed)),
            ("joint_cannot_be_fitted", breaking(uncooled_wheel_shaft)),
            ("cup_key_no_bearing_area", breaking(no_boss)),
            ("inner_hub_key_no_bearing_area", breaking(short_hub)),
            ("cup_stress_alone", breaking(cup_stress_alone)),
            // Two failing checks that pull one input apart (review focus 4).
            (
                "length_conflict",
                design(|i| i.metal.max_overall_axial_mm = 30.0),
            ),
            (
                "diameter_conflict",
                with(with_poles(12), |i| i.metal.max_diameter_mm = 42.0),
            ),
        ]
    }

    /// A value typed below what its input can physically be: (an edit that types it so, the one
    /// sentence the shaft joints' Problem then reads).
    type ImpossibleValueCase = (fn(&mut DesignInputs), &'static str);

    /// Each input a value can be typed below what it can physically be (the whole-branch
    /// review's MF-1), in the inputs' order.
    fn impossible_value_cases() -> [ImpossibleValueCase; 6] {
        [
            (
                |i| i.shaft_joint.friction = -0.14,
                "Shrink-fit friction coefficient, -0.1400, cannot physically be below zero: enter at least 0.05000, its slider's bottom.",
            ),
            (
                |i| i.shaft_joint.smoothing_um = -3.0,
                "Surface smoothing of the fit, -3.000 µm, cannot physically be below zero: enter at least 0 µm, its slider's bottom.",
            ),
            (
                |i| i.shaft_joint.key_width_mm = 0.0,
                "Shaft key width, 0 mm, cannot physically be zero or below: enter at least 1.000 mm, its slider's bottom.",
            ),
            (
                |i| i.shaft_joint.wheel_shaft_fit_C = -300.0,
                "Wheel shaft temperature at fitting, -300.0 °C, cannot physically be below absolute zero: enter at least -196.0 °C, its slider's bottom.",
            ),
            (
                |i| i.shaft_joint.output_shaft_fit_C = -300.0,
                "Gearbox shaft temperature at fitting, -300.0 °C, cannot physically be below absolute zero: enter at least -20.00 °C, its slider's bottom.",
            ),
            (
                |i| i.shaft_joint.joining_clearance_um = -2.0,
                "Joining clearance, -2.000 µm, cannot physically be below zero: enter at least 0 µm, its slider's bottom.",
            ),
        ]
    }

    /// Two values typed below what they can physically be at once ([`impossible_value_cases`]):
    /// the gearbox shaft, then the friction, the other order than the inputs'.
    fn two_impossible_values(i: &mut DesignInputs) {
        let [friction, .., gearbox_shaft, _] = impossible_value_cases();
        (gearbox_shaft.0)(i);
        (friction.0)(i);
    }

    /// One design per sentence form of the shaft joints' Problem (spec 5.2), each the passing
    /// design with joint inputs moved: (its name, the design, words its Problem holds, the
    /// form's own). Few of them are in [`designs`]; their Problems join [`every_text`].
    fn shaft_joint_problem_designs() -> Vec<(&'static str, DesignInputs, &'static str)> {
        // Each value below what its input can physically be, its words its whole sentence.
        let [
            friction,
            smoothing,
            key_width,
            wheel_shaft,
            gearbox_shaft,
            clearance,
        ] = impossible_value_cases();
        let typed = |name, (edit, sentence): ImpossibleValueCase| (name, breaking(edit), sentence);
        vec![
            (
                "inner hub slips",
                only_the_joint_fails(),
                "At its loosest the inner hub holds",
            ),
            (
                "cup slips",
                breaking(cup_slips),
                "At its loosest the cup holds",
            ),
            (
                "inner hub overstressed",
                breaking(hub_overstressed),
                "the inner hub's bore stress is",
            ),
            (
                "cup overstressed",
                breaking(cup_overstressed),
                "the cup's bore stress is",
            ),
            (
                "uncooled wheel shaft",
                breaking(uncooled_wheel_shaft),
                "your limit for the bonded inner hub",
            ),
            (
                "the adhesive's limit",
                breaking(cool_adhesive),
                "the selected adhesive's design limit",
            ),
            (
                "the magnets' onset",
                breaking(n42_parts),
                "the magnets' single-ring onset",
            ),
            (
                "cup cannot be fitted",
                breaking(low_cup_fit_limit),
                "your limit for the bare cup",
            ),
            (
                "key above its limit",
                breaking(|i| i.coupling.keyway_depth_mm = 0.3),
                "Alone, the key would bear",
            ),
            (
                "cup key without length",
                breaking(no_boss),
                "its boss and web",
            ),
            (
                "inner hub key without length",
                breaking(short_hub),
                "is no longer than the key is wide",
            ),
            (
                "key without contact",
                breaking(|i| i.shaft_joint.key_contact_mm = 0.0),
                "is not above zero",
            ),
            (
                "no window under the stress limit",
                breaking(|i| i.materials.parts.back_iron = 7),
                "its bore stress stays within the limit only up to",
            ),
            (
                "no window that can be fitted",
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = -78.0),
                "the inner hub can be fitted only up to",
            ),
            (
                "no joining clearance",
                breaking(uncooled_wheel_shaft),
                "does not even open the",
            ),
            (
                "not a number",
                breaking(|i| i.temperature.adhesive.selected = 0),
                "cannot be made",
            ),
            (
                "safety factor not above zero",
                breaking(|i| i.shaft_joint.safety_factor = 0.0),
                "The safety factor,",
            ),
            typed("a friction below zero", friction),
            typed("a smoothing below zero", smoothing),
            typed("a key width of zero", key_width),
            typed("a wheel shaft below absolute zero", wheel_shaft),
            typed("a gearbox shaft below absolute zero", gearbox_shaft),
            typed("a joining clearance below zero", clearance),
            (
                "two values it cannot physically be",
                breaking(two_impossible_values),
                "its slider's bottom. Gearbox shaft temperature at fitting",
            ),
            (
                "gearbox shaft below its slider's bottom",
                breaking(|i| i.shaft_joint.output_shaft_fit_C = -21.0),
                "the coldest the gearbox shaft is taken",
            ),
            (
                "cup without a wall",
                breaking(|i| {
                    i.coupling.bore_mm = 16.0;
                    i.metal.boss_od_mm = 12.0;
                }),
                "is at least as wide as its boss",
            ),
            (
                "inner hub keyway through its wall",
                breaking(|i| {
                    i.coupling.bore_mm = 16.0;
                    i.coupling.keyway_depth_mm = 2.1;
                }),
                "cuts through the wall",
            ),
            (
                "cup keyway through its boss's wall",
                breaking(|i| i.metal.boss_od_mm = 12.0),
                "cuts through the boss's wall",
            ),
            (
                "inner hub without a wall",
                breaking(|i| i.coupling.bore_mm = 21.0),
                "reaches the flats",
            ),
            (
                "reversed band",
                breaking(reversed_hub_band),
                "is above its maximum",
            ),
        ]
    }

    /// The Problem of `check` for `inputs`.
    fn problem(check: &str, inputs: &DesignInputs) -> Option<String> {
        (guide(check).unwrap().problem)(inputs, &compute_all(inputs))
    }

    /// The index in [`CHECK_GUIDES`] of the guide of `check`.
    fn guide_index(check: &str) -> usize {
        CHECK_GUIDES
            .iter()
            .position(|g| g.check == check)
            .unwrap_or_else(|| panic!("{check} has a guide"))
    }

    /// Every fix `g` can show: its list, or for a guide that picks its fixes per failing term
    /// every term's, each once.
    fn possible_fixes(g: &CheckGuide) -> Vec<(&'static str, Direction)> {
        let lists: Vec<&[(&str, Direction)]> = match (g.fixes, g.check) {
            (Fixes::Static(list), _) => vec![list],
            (Fixes::PerTerm(_), TEMPERATURE) => {
                TemperatureTerm::ALL.iter().map(|t| t.fixes()).collect()
            }
            (Fixes::PerTerm(_), SPACE_CLAIM) => SPACE_CLAIM_AXES.to_vec(),
            (Fixes::PerTerm(_), SHAFT_JOINT) => JOINT_SIDES
                .iter()
                .flat_map(|side| CAUSES.map(|cause| side.fixes(cause)))
                .collect(),
            (Fixes::PerTerm(_), check) => panic!("{check}: the test lists no terms"),
        };
        let mut all = Vec::new();
        for &fix in lists.into_iter().flatten() {
            if !all.contains(&fix) {
                all.push(fix);
            }
        }
        all
    }

    /// Every text the guides draw: labels, Whys, the reasons of the links without a note (the
    /// guides' and the dashboard rows'), every fix, the fallbacks, and for every test design
    /// every Problem, every trade-off note and its status line.
    fn every_text() -> Vec<String> {
        let mut texts = Vec::new();
        for g in &CHECK_GUIDES {
            texts.push(g.label.to_owned());
            texts.push(g.why.to_owned());
            if let Learn::None(reason) = g.learn {
                texts.push(reason.to_owned());
            }
            for (path, direction) in possible_fixes(g) {
                texts.push(fix_text(path, direction));
            }
        }
        for fallback in [
            NO_FIX,
            TEMPERATURE_FALLBACK,
            SPACE_CLAIM_FALLBACK,
            GEARBOX_UNKNOWN,
            SHAFT_JOINT_NOT_A_NUMBER,
            SHAFT_JOINT_FALLBACK,
        ] {
            texts.push(fallback.to_owned());
        }
        // Every sentence form of the shaft joints' Problem, most of which no design of
        // `designs()` reaches.
        for (_, inputs, _) in shaft_joint_problem_designs() {
            texts.extend(shaft_joint_problem(&inputs, &compute_all(&inputs)));
        }
        for (_, learn) in &ROW_LEARN {
            if let Learn::None(reason) = learn {
                texts.push((*reason).to_owned());
            }
        }
        for (_, inputs) in designs() {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                texts.extend((g.problem)(&inputs, &results));
                texts.extend(guide_notes(&all, index, &inputs, &results));
            }
            texts.push(design_status_text(&design_status(&inputs, &results, false)));
        }
        // The gearbox Problem's other remedy (no band under one settles it) shows only at the
        // gearbox ratio's and efficiency's slider bottoms, which no test design reaches.
        for (_, inputs) in gear_slider_ends() {
            texts.extend(gearbox_problem(&inputs, &compute_all(&inputs)));
        }
        // Its first remedy names only what narrows the band, so the bracket's other wordings
        // show only at the band-lever designs.
        for (_, inputs, _) in band_lever_designs() {
            texts.extend(gearbox_problem(&inputs, &compute_all(&inputs)));
        }
        // Where nothing narrows the band it offers no band, which only hand-edited results reach;
        // where the band and the widest band both read 1.000 (an exact product at a band that
        // reads 1.000: the widest band is one), it quotes both with more decimals.
        let (inputs, results) = nothing_narrows_the_band();
        texts.extend(gearbox_problem(&inputs, &results));
        for (_, inputs) in exact_products(&band_reading_one(), 1.0).into_iter().take(1) {
            texts.extend(gearbox_problem(&inputs, &compute_all(&inputs)));
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

    /// The gearbox guide's Why is a static text, so it holds at every reachable input (spec
    /// section 6), and cold-high is an expectation, not a bound. With the operating temperature
    /// below the minimum temperature, each inside its slider (the corner of backlog BL-067),
    /// hot-low is above cold-high: more than cold-high reaches the gearbox output while the
    /// coupling slips there. The Why may say cold-high "can reach" the output, never that "up to"
    /// cold-high does.
    #[test]
    fn the_gearbox_why_never_bounds_the_torque_by_cold_high() {
        let mut corner = DesignInputs::default();
        for (path, x) in [
            ("coupling.op_temp_C", -40.0),
            ("metal.min_temp_C", 20.0),
            ("metal.variation", 0.0),
        ] {
            let range = slider_range(path);
            assert!(
                range.min <= x && x <= range.max,
                "{path}: {x} on its slider"
            );
            corner = with_number(&corner, path, x).unwrap_or_else(|| panic!("{path}: {x}"));
        }
        let m = &compute_all(&corner).metal;
        assert!(
            m.torque_hot_low_Nm > m.torque_cold_high_Nm,
            "hot-low {} above cold-high {}",
            m.torque_hot_low_Nm,
            m.torque_cold_high_Nm
        );
        assert!(
            !GEARBOX_GUIDE.why.to_lowercase().contains("up to cold-high"),
            "{}",
            GEARBOX_GUIDE.why
        );
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
            // One guide never moves an input both ways. A part may carry two choice fixes:
            // another part, or Custom dimensions with a hotter grade.
            for &(path, _) in &possible {
                let both_ways = [Direction::Raise, Direction::Lower]
                    .iter()
                    .all(|&way| possible.contains(&(path, way)));
                assert!(!both_ways, "{}: {path} both ways", g.check);
            }
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
                    Direction::CustomHotterGrade => assert!(
                        PART_PATHS.contains(&path),
                        "{}: {path} is no magnet part",
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

    #[test]
    fn the_size_tab_and_banner_links_lead_where_their_checks_do() {
        // Spec 3.7: the Size tab's Required and Hot-low links lead where the hot minimum
        // check's Why does, the banner's where the end-effect check's does, and the links
        // under Magnets per ring and Outside diameter and on the torque card's heading (slice 3
        // spec 6.1) to Start here's first step.
        assert_eq!(Learn::Note(TORQUE_BAND_NOTE), guide(HOT_MIN).unwrap().learn);
        assert_eq!(Learn::Note(BANNER_NOTE), guide(END_EFFECT).unwrap().learn);
        assert_eq!(notes::START_HERE[0].0, SIZE_TAB_NOTE);
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE, BANNER_NOTE] {
            assert!(
                reviewed_note(id).is_some(),
                "{id} is missing or not reviewed"
            );
            assert!(notes::shown(id), "{id} is hidden");
        }
    }

    #[test]
    fn the_hot_low_hot_minimum_and_cold_high_rows_lead_to_the_torque_band_note() {
        // Spec 6 test 6: the hot-minimum guide and the three rows of the band link the note
        // that explains their equations, and it has passed review.
        let band = Learn::Note("hot_low_cold_high");
        assert_eq!(guide(HOT_MIN).map(|g| g.learn), Some(band));
        for path in [
            "metal.torque_hot_low_Nm",
            "metal.hot_min_check",
            "metal.torque_cold_high_Nm",
        ] {
            assert_eq!(row_learn(path), Some(band), "{path}");
            assert_eq!(
                notes::note_for(path).map(|n| n.id),
                Some("hot_low_cold_high"),
                "{path}"
            );
        }
        assert!(reviewed_note("hot_low_cold_high").is_some());
    }

    #[test]
    fn the_ripple_row_leads_to_the_slip_note() {
        // Spec 3.3: the gearbox ripple row's Why link moves from the pull-out note to the note
        // that owns its equation, what happens when the coupling slips.
        assert_eq!(
            row_learn("model.gearbox_input_ripple_Nm"),
            Some(Learn::Note("when_slipping"))
        );
        assert_eq!(
            notes::note_for("model.gearbox_input_ripple_Nm").map(|n| n.id),
            Some("when_slipping")
        );
    }

    /// The gearbox check (BL-059, replacing the test that said nothing checks a gearbox torque):
    /// one dashboard row of its own, after the headline's rows and the shaft joints row and
    /// before the space claim, a design check of the results, and a guide, between the shaft
    /// joints' and the space claim's, whose Why link and its row's lead to the torque-band note,
    /// which owns cold-high at the gearbox input. The ripple row stays a value with no check.
    #[test]
    fn the_gearbox_check_is_a_dashboard_row_with_its_guide() {
        let gearbox_rows: Vec<(usize, (&str, Option<&str>))> = DASHBOARD
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, (row, check))| {
                row.contains("gearbox") || check.is_some_and(|c| c.contains("gearbox"))
            })
            .collect();
        assert_eq!(
            gearbox_rows,
            [
                (5, ("model.gearbox_input_ripple_Nm", None)),
                (16, (GEARBOX, Some(GEARBOX))),
            ]
        );
        let checks: Vec<&str> = CHECKS
            .iter()
            .copied()
            .filter(|c| c.contains("gearbox"))
            .collect();
        assert_eq!(checks, [GEARBOX]);
        let gearbox = guide(GEARBOX).expect("the gearbox check has a guide");
        assert_eq!(gearbox.learn, Learn::Note(TORQUE_BAND_NOTE));
        assert_eq!(row_learn(GEARBOX), Some(gearbox.learn));
        let guided = CHECK_GUIDES
            .iter()
            .filter(|g| g.check.contains("gearbox"))
            .count();
        assert_eq!(guided, 1);
        assert_eq!(guide_index(GEARBOX), guide_index(SHAFT_JOINT) + 1);
        assert_eq!(guide_index(SPACE_CLAIM), guide_index(GEARBOX) + 1);
    }

    /// The shaft joints check (BL-064): one dashboard row of its own at index 15, after the
    /// headline's rows (the hidden clamp row last among them) and before the gearbox row; a
    /// design check of the results, listed last (its group follows the gearbox's in the
    /// schema); a guide between the clamp's and the gearbox's, labelled apart from the clamp's,
    /// counted while the clamp section is paused, whose Why link is its row's.
    #[test]
    fn the_shaft_joint_check_is_a_dashboard_row_with_its_guide() {
        let rows: Vec<(usize, (&str, Option<&str>))> = DASHBOARD
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, (row, check))| is_shaft_joint(row) || check.is_some_and(is_shaft_joint))
            .collect();
        assert_eq!(rows, [(15, (SHAFT_JOINT, Some(SHAFT_JOINT)))]);
        assert_eq!(DASHBOARD[14].0, CLAMP);
        assert_eq!(DASHBOARD[16].0, GEARBOX);
        assert_eq!(CHECKS.last(), Some(&SHAFT_JOINT));
        assert_eq!(CHECKS.iter().filter(|c| is_shaft_joint(c)).count(), 1);
        let joint = guide(SHAFT_JOINT).expect("the shaft joints check has a guide");
        assert_eq!(joint.label, "Shaft joints");
        // The Why's subject is each fit, not its grip, and each fit has one allowed temperature
        // (T5 text re-review SF-4).
        assert!(
            joint.why.contains("each fit must carry")
                && joint.why.ends_with("within its allowed temperature."),
            "{}",
            joint.why
        );
        assert_eq!(row_learn(SHAFT_JOINT), Some(joint.learn));
        assert!(counts(joint, true) && counts(joint, false));
        assert_eq!(guide_index(SHAFT_JOINT), guide_index(CLAMP) + 1);
        assert_eq!(guide_index(GEARBOX), guide_index(SHAFT_JOINT) + 1);
    }

    /// The guide's two joints are the engine's, in its order, and each path it reads is the
    /// result or input of that joint: at the defaults and at a design where every number
    /// differs between the joints, each result equals the engine's term bit for bit, and each
    /// input (the band and the shaft's temperature at fitting) is the value the engine takes
    /// for that joint.
    #[test]
    fn the_shaft_joint_sides_are_the_engine_s_joints() {
        for inputs in [DesignInputs::default(), only_the_joint_fails()] {
            let results = compute_all(&inputs);
            let joints = shaft_joints(&inputs, &results).expect("a judged check");
            let (_, specs) = shaft_joint::joint_specs(
                &inputs.shaft_joint,
                &shaft_joint_links(&inputs, &results),
            );
            let inputs_of = |spec: &shaft_joint::JointSpec| {
                [
                    spec.interference_min_um,
                    spec.interference_max_um,
                    spec.shaft_fit_C,
                ]
            };
            // Every input differs between the joints, so a side reading the other's shows.
            for (hub, cup) in inputs_of(&specs[0]).into_iter().zip(inputs_of(&specs[1])) {
                assert_ne!(hub, cup);
            }
            for ((side, joint), spec) in JOINT_SIDES.iter().zip(&joints).zip(&specs) {
                assert_eq!(side.name, joint.name);
                assert_eq!(joint_side(joint.name).map(|(_, s)| s.name), Some(side.name));
                let result = |path: &str| match results.get(path) {
                    Some(Value::Num(x)) => x.to_bits(),
                    other => panic!("{path}: {other:?}"),
                };
                for (path, term) in [
                    (side.holding, joint.holding_Nm),
                    (side.stress, joint.bore_stress_MPa),
                    (side.key, joint.key_pressure_MPa),
                    (side.needed, joint.needed_um),
                    (side.elastic, joint.elastic_max_um),
                    (side.fittable, joint.fittable_max_um),
                    (side.fit_temp, joint.fit_temp_C),
                    (side.fit_allowed, joint.fit_allowed_C),
                ] {
                    assert_eq!(result(path), term.to_bits(), "{}: {path}", side.name);
                }
                for (path, value) in [side.min, side.max, side.shaft_fit]
                    .into_iter()
                    .zip(inputs_of(spec))
                {
                    assert_eq!(
                        inputs.get(path),
                        Some(Value::Num(value)),
                        "{}: {path}",
                        side.name
                    );
                }
            }
        }
        assert_eq!(joint_side("no such joint").map(|(i, _)| i), None);
    }

    /// BL-064 (spec 6.1 and 7.4): the shaft-joint row's Why link, its guide's and its
    /// `ROW_LEARN` entry's, opens the shaft-joint note, which has passed its physics review and
    /// explains the joint's holding torque, the number Start here opens it on.
    #[test]
    fn the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note() {
        let shaft_joint = guide(SHAFT_JOINT).expect("the shaft joint check has a guide");
        assert_eq!(shaft_joint.learn, Learn::Note("shaft_joint"));
        assert_eq!(row_learn(SHAFT_JOINT), Some(shaft_joint.learn));
        let note = reviewed_note("shaft_joint").expect("the shaft-joint note is reviewed");
        assert!(
            note.equations.contains(&"shaft_joint.hub_holding_Nm"),
            "{:?}",
            note.equations
        );
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
            "Lower Magnets per ring"
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
        for g in &CHECK_GUIDES {
            for (path, direction) in possible_fixes(g) {
                let label = SIZE_TAB_ALIASES
                    .iter()
                    .find(|&&(aliased, _)| aliased == path)
                    .map_or(label_for(path), |&(_, alias)| alias);
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

    #[test]
    fn the_fix_label_is_the_display_label_and_still_names_one_input() {
        // Spec 3.5: the overlay's names reach the fix text ...
        assert_eq!(
            fix_text(FACE_GAP, Direction::Raise),
            format!("Raise {}", label_for(FACE_GAP))
        );
        // ... but a fix on the ring radius names the Size-tab control a user moves it with.
        assert_eq!(
            fix_text(RING, Direction::Raise),
            format!("Raise {}", size_controls::DIAMETER)
        );
        assert_ne!(label_for(RING), size_controls::DIAMETER);
        // C-3 runs on the display labels: the inputs whose fix text adds a group heading are
        // the eight whose label another input's equals or ends with (map-help 3.5): BL-064's
        // "Shaft key width" ends with the paused clamp's "Key width".
        let headed: Vec<&str> = InputCatalogue::get()
            .all()
            .map(|e| e.path.as_str())
            .filter(|p| !SIZE_TAB_ALIASES.iter().any(|&(aliased, _)| aliased == *p))
            .filter(|p| fix_text(p, Direction::Raise) != format!("Raise {}", label_for(p)))
            .collect();
        assert_eq!(
            headed,
            [
                "coupling.c_end",
                "coupling.mu0",
                "calibration.mu0",
                "materials.steel.modulus_GPa",
                "materials.steel.density_g_cm3",
                "clamps.alloy",
                "clamps.boss_od_mm",
                "clamps.key_width_mm",
            ]
        );
    }

    // ------------------------------------------------------------------ the fixes help

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

    /// The terms of `check`'s verdict in a fixed order (two designs' lists pair up): the
    /// production table ([`check_terms`]: the temperature terms are the guide's own,
    /// [`TemperatureTerm`], which a test holds to the verdict), and the clamp's own term, whose
    /// margin is the fix tests' ([`clamp_margin`]). Panics for a path that is no guide's check.
    fn terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<CheckTerm> {
        assert!(guide(check).is_some(), "no terms for {check}");
        let mut terms = check_terms(check, inputs, results);
        if check == CLAMP {
            terms.push(CheckTerm {
                name: "screw size".to_owned(),
                margin: clamp_margin(inputs, results),
                passes: results.clamps.index > 0,
                quoted: None,
            });
        }
        terms
    }

    /// The margin of the term `name` of `check`.
    fn margin(check: &str, name: &str, inputs: &DesignInputs, results: &DesignResults) -> f64 {
        terms(check, inputs, results)
            .into_iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{check} has no term {name}"))
            .margin
    }

    /// The names of the terms of `check` that fail, in [`terms`]' order.
    fn failing_term_names(
        check: &str,
        inputs: &DesignInputs,
        results: &DesignResults,
    ) -> Vec<String> {
        terms(check, inputs, results)
            .into_iter()
            .filter(|t| !t.passes)
            .map(|t| t.name)
            .collect()
    }

    #[test]
    fn check_terms_keeps_the_fix_tests_margins() {
        // Slice 3 spec 3.6 (test 9): the counted checks' terms are one production table,
        // `check_terms`, which the torque card reads; the fix tests read it through `terms()`,
        // which adds only the clamp's own term, and pass unchanged.
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            for guide in &CHECK_GUIDES {
                let shared = check_terms(guide.check, &inputs, &results);
                let tested = terms(guide.check, &inputs, &results);
                if guide.check == CLAMP {
                    assert!(shared.is_empty(), "{name}: the clamp has no shared term");
                    let [screw] = tested.as_slice() else {
                        panic!("{name}: the clamp's one term: {tested:?}")
                    };
                    assert_eq!(screw.name, "screw size");
                    assert!(same(screw.margin, clamp_margin(&inputs, &results)));
                    assert_eq!(screw.passes, results.clamps.index > 0, "{name}");
                    assert_eq!(screw.quoted, None);
                    continue;
                }
                assert!(!shared.is_empty(), "{name}: {} has terms", guide.check);
                assert_eq!(tested.len(), shared.len(), "{name}: {}", guide.check);
                for (t, s) in tested.iter().zip(&shared) {
                    assert!(
                        t.name == s.name
                            && same(t.margin, s.margin)
                            && t.passes == s.passes
                            && t.quoted == s.quoted,
                        "{name}: {t:?} against {s:?}"
                    );
                }
                // Every quoted quantity is a number result in a unit of spec 3.6's table, drawn
                // through its metadata (never typed).
                for quoted in shared.iter().filter_map(|t| t.quoted) {
                    let unit = result_info(quoted.path)
                        .unwrap_or_else(|| panic!("{} is a result", quoted.path))
                        .meta
                        .unit;
                    assert!(QUOTED_UNITS.contains(&unit), "{}: {unit}", quoted.path);
                    assert!(
                        matches!(results.get(quoted.path), Some(Value::Num(_))),
                        "{name}: {}",
                        quoted.path
                    );
                }
            }
            assert!(check_terms("no.such.check", &inputs, &results).is_empty());
        }

        // Spec 3.6's table at every test design, not the defaults alone: each check's terms,
        // margins (larger is better) and quoted quantities, each margin the results' own fields
        // bit for bit. Hot-low torque, Temperature limits (T-13), the end effect and the magnets'
        // fit quote none.
        type Row = (String, u64, bool, Option<Quoted>);
        let row = |name: &str, margin: f64, quoted| -> Row {
            (name.to_owned(), margin.to_bits(), margin >= 0.0, quoted)
        };
        let axis = |index: usize| {
            let (name, size) = SPACE_CLAIM_AXIS_SIZES[index];
            Some(Quoted::new(size, Some(name), 2))
        };
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let (md, materials) = (&results.metal, &results.materials);
            let read = |check: &str| -> Vec<Row> {
                check_terms(check, &inputs, &results)
                    .into_iter()
                    .map(|t| (t.name, t.margin.to_bits(), t.passes, t.quoted))
                    .collect()
            };
            assert_eq!(
                read(HOT_MIN),
                [row(
                    "hot-low torque",
                    md.torque_hot_low_Nm - inputs.metal.required_min_Nm,
                    None
                )],
                "{name}"
            );
            assert_eq!(
                read(CLEARANCE),
                [row(
                    "running clearance",
                    md.min_running_clearance_mm - inputs.metal.residual_target_mm,
                    Some(Quoted::new("metal.min_running_clearance_mm", None, 2))
                )],
                "{name}"
            );
            assert_eq!(
                read(CUP_WALL),
                [row(
                    "cup wall",
                    materials.cup_wall_corner_mm - materials.backiron_thickness_needed_mm,
                    Some(Quoted::new(
                        "materials.backiron_thickness_needed_mm",
                        Some(WALL_NEEDED),
                        2
                    ))
                )],
                "{name}"
            );
            assert_eq!(
                read(GEARBOX),
                [row(
                    "gearbox input torque",
                    inputs.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
                    Some(Quoted::new("metal.cold_high_input_Nm", Some(COLD_HIGH), 2))
                )],
                "{name}"
            );
            assert_eq!(
                read(SPACE_CLAIM),
                [
                    row("diameter", md.diameter_reserve_mm, axis(0)),
                    row("overall length", md.axial_reserve_mm, axis(1)),
                    row("large-diameter bay", md.large_dia_reserve_mm, axis(2)),
                ],
                "{name}"
            );
            for check in [HOT_MIN, TEMPERATURE, END_EFFECT, FIT_CHECK] {
                assert!(
                    read(check).iter().all(|(.., quoted)| quoted.is_none()),
                    "{name}: {check}"
                );
            }
            assert_eq!(
                read(TEMPERATURE).len(),
                TemperatureTerm::ALL.len(),
                "{name}"
            );
        }
        let defaults = DesignInputs::default();
        let before = compute_all(&defaults);

        // The shaft joints: eight terms in one order whether both joints are judged (the
        // defaults; the inner hub slipping), one is not (no wall past a keyway through the hub,
        // a reversed band: each of the inner hub's four terms passes there) or the check cannot
        // be made (a safety factor of 0), where every term passes. Each quotes its joint's result
        // for its cause, named by its label, in that cause's unit.
        let names: Vec<String> = [shaft_joint::INNER_HUB, shaft_joint::CUP]
            .into_iter()
            .flat_map(|joint| CAUSES.map(|cause| joint_term(joint, cause)))
            .collect();
        assert_eq!(
            names,
            [
                "inner hub slips",
                "inner hub bore stress",
                "inner hub fitting",
                "inner hub key",
                "cup slips",
                "cup bore stress",
                "cup fitting",
                "cup key",
            ]
        );
        let unmade = design(|i| i.shaft_joint.safety_factor = 0.0);
        for (name, inputs) in [
            ("defaults", defaults.clone()),
            ("joint_slips", only_the_joint_fails()),
            ("keyway through", keyway_through()),
            ("reversed band", breaking(reversed_hub_band)),
            ("cannot be made", unmade.clone()),
        ] {
            let joint_terms = check_terms(SHAFT_JOINT, &inputs, &compute_all(&inputs));
            let read_names: Vec<&str> = joint_terms.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(read_names, names, "{name}");
            if ["keyway through", "reversed band"].contains(&name) {
                let inner_hub: Vec<&CheckTerm> = joint_terms
                    .iter()
                    .filter(|t| t.name.starts_with(shaft_joint::INNER_HUB))
                    .collect();
                assert_eq!(inner_hub.len(), CAUSES.len(), "{name}");
                assert!(inner_hub.iter().all(|t| t.passes), "{name}: {inner_hub:?}");
            }
        }
        let unmade_results = compute_all(&unmade);
        assert!(shaft_joints(&unmade, &unmade_results).is_err());
        assert!(
            check_terms(SHAFT_JOINT, &unmade, &unmade_results)
                .iter()
                .all(|t| t.passes)
        );
        let joints_quoted: Vec<Option<Quoted>> = check_terms(SHAFT_JOINT, &defaults, &before)
            .into_iter()
            .map(|t| t.quoted)
            .collect();
        let per_cause: Vec<Option<Quoted>> = JOINT_SIDES
            .iter()
            .flat_map(|side| {
                [
                    (side.needed, 2),
                    (side.stress, 1),
                    (side.fit_temp, 1),
                    (side.key, 1),
                ]
            })
            .map(|(path, decimals)| Some(Quoted::new(path, Some(label_for(path)), decimals)))
            .collect();
        assert_eq!(joints_quoted, per_cause);
        // Each cause's unit, as the card quotes it, joint by joint: the needed interference in
        // µm, the bore stress in MPa, the fitting temperature in °C and the key's pressure in
        // MPa.
        let units: Vec<&str> = joints_quoted
            .iter()
            .map(|quoted| {
                let path = quoted.expect("every joint term quotes its result").path;
                result_info(path).expect("a joint's result").meta.unit
            })
            .collect();
        assert_eq!(units, ["µm", "MPa", "°C", "MPa"].repeat(2));

        // SPACE_CLAIM_AXES, axis_reserves and SPACE_CLAIM_AXIS_SIZES name the three axes in one
        // order: each reserve is its claim less the size SPACE_CLAIM_AXIS_SIZES names, bit for
        // bit, at every test design; and one step of each fix of an axis, from the defaults,
        // shrinks that size and raises that reserve.
        let claims = [
            "metal.max_diameter_mm",
            "metal.max_overall_axial_mm",
            "metal.max_large_dia_axial_mm",
        ];
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let reserves = axis_reserves(&results);
            for (index, (&(_, size), claim)) in
                SPACE_CLAIM_AXIS_SIZES.iter().zip(claims).enumerate()
            {
                let claim = inputs.get(claim).and_then(number).expect("a claim input");
                let size = result_number(&results, size);
                assert!(same(reserves[index], claim - size), "{name}: axis {index}");
            }
        }
        for (index, axis_fixes) in SPACE_CLAIM_AXES.iter().enumerate() {
            let size = SPACE_CLAIM_AXIS_SIZES[index].1;
            for &(path, direction) in *axis_fixes {
                let moved = moves(&defaults, &before, path, direction);
                assert!(!moved.is_empty(), "{path} moves from the defaults");
                for design in moved {
                    let after = compute_all(&design);
                    assert!(
                        result_number(&after, size) < result_number(&before, size),
                        "{path}: {size}"
                    );
                    assert!(
                        axis_reserves(&after)[index] > axis_reserves(&before)[index],
                        "{path}: axis {index}"
                    );
                }
            }
        }
    }

    /// A reading of one shaft joint's terms that holds still wherever a sweep of
    /// [`shaft_joint_sweeps`] finds its margin flat.
    type JointReading = fn(&JointTerms) -> f64;

    /// Every slider fix of the shaft joints' guide, with what to sweep it on (spec 5.2): (a design
    /// that fails the check, the fix's input and direction, the term it moves, the reading that
    /// holds still where that term's margin is flat). The reading is `None` where the margin
    /// rises at every step. A margin is flat only where nothing the term reads moves: the bore
    /// stress against the band's maximum where smoothing takes all of it (the effective
    /// interference is floored at zero), the key against the contact height past the keyway depth
    /// (the contact is capped there), and the inner hub's slip against the hub length where the
    /// axial override sets the hub's length (BL-064). The hub's minimum goes past the band's
    /// maximum (to the slider's 40 µm): the margin keeps rising, though the band then reads
    /// reversed.
    fn shaft_joint_sweeps() -> Vec<(
        DesignInputs,
        &'static str,
        Direction,
        &'static str,
        Option<JointReading>,
    )> {
        use Direction::{Lower, Raise};
        let effective_max: JointReading = |j| j.effective_max_um;
        let contact: JointReading = |j| j.key_contact_mm;
        let length: JointReading = |j| j.key_bearing_mm;
        vec![
            (
                only_the_joint_fails(),
                HUB_MIN,
                Raise,
                "inner hub slips",
                None,
            ),
            (
                only_the_joint_fails(),
                HUB_LENGTH,
                Raise,
                "inner hub slips",
                None,
            ),
            (
                only_the_joint_fails(),
                FACE_GAP,
                Raise,
                "inner hub slips",
                None,
            ),
            (breaking(cup_slips), CUP_MIN, Raise, "cup slips", None),
            (breaking(cup_slips), BOSS_LENGTH, Raise, "cup slips", None),
            (breaking(cup_slips), MAGNET_LENGTH, Lower, "cup slips", None),
            (
                breaking(hub_overstressed),
                HUB_MAX,
                Lower,
                "inner hub bore stress",
                Some(effective_max),
            ),
            (
                breaking(uncooled_wheel_shaft),
                HUB_MAX,
                Lower,
                "inner hub fitting",
                None,
            ),
            (
                breaking(cup_overstressed),
                CUP_MAX,
                Lower,
                "cup bore stress",
                Some(effective_max),
            ),
            (
                breaking(low_cup_fit_limit),
                CUP_MAX,
                Lower,
                "cup fitting",
                None,
            ),
            (
                breaking(uncooled_wheel_shaft),
                WHEEL_SHAFT,
                Lower,
                "inner hub fitting",
                None,
            ),
            (
                breaking(low_cup_fit_limit),
                GEARBOX_SHAFT,
                Lower,
                "cup fitting",
                None,
            ),
            // 50.8 mm magnets: the cup's key alone bears about 390 MPa at a 1.2 mm contact,
            // under the 1.7 mm keyway depth that caps it.
            (
                breaking(long_magnets),
                KEY_CONTACT,
                Raise,
                "cup key",
                Some(contact),
            ),
            (breaking(no_boss), BOSS_LENGTH, Raise, "cup key", None),
            (
                breaking(short_hub),
                HUB_LENGTH,
                Raise,
                "inner hub key",
                None,
            ),
            (
                breaking(|i| i.coupling.magnets.axial_length_mm = Some(40.0)),
                HUB_LENGTH,
                Raise,
                "inner hub slips",
                Some(length),
            ),
        ]
    }

    #[test]
    fn each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red() {
        // Spec review focus 3. For every test design and every check it fails: each fix shown
        // moves a failing term of the check toward passing by one slider step or one choice of
        // its input alone; every failing term has such a fix; and no fix turns another guided
        // check from green to red (a choice that would is not the one to take).
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
                // its Problem asks for the number. A slip-fault peak whose limit the hot-day
                // start already reaches has no fix of its own either (review 2, N1): the
                // hot-day term at the same limit, which fails too, carries the fixes that
                // clear both (a test checks it fails).
                let reached: Vec<String> = TemperatureTerm::ALL
                    .iter()
                    .filter(|t| g.check == TEMPERATURE && t.start_reaches_peak_limit(&results))
                    .map(|t| format!("{t:?}"))
                    .collect();
                let failing: BTreeSet<&str> = terms_before
                    .iter()
                    .filter(|t| !t.passes && !t.margin.is_nan() && !reached.contains(&t.name))
                    .map(|t| t.name.as_str())
                    .collect();
                let shown = fixes(g, &inputs, &results, false);
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
                        if matches!(direction, Direction::Raise | Direction::Lower) {
                            assert!(!reddens, "{name}: {text} turns another check red");
                        } else if reddens {
                            continue;
                        }
                        let terms_after = terms(g.check, &design, &after);
                        for (b, a) in terms_before.iter().zip(&terms_after) {
                            if failing.contains(b.name.as_str()) && a.margin > b.margin {
                                moved.insert(b.name.as_str());
                                helps = true;
                            }
                        }
                    }
                    assert!(helps, "{name}: {text} moves no failing term of {}", g.check);
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
                "MagnetPeak",
            ),
            (
                TEMPERATURE,
                breaking(slip_fault),
                "temperature.thermal.conductance_W_K",
                Direction::Raise,
                "MagnetPeak",
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
                GEARBOX,
                DesignInputs::default(),
                MAGNET_LENGTH,
                Direction::Lower,
                "gearbox input torque",
            ),
            (
                GEARBOX,
                DesignInputs::default(),
                FACE_GAP,
                Direction::Raise,
                "gearbox input torque",
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
        // The table holds every slider fix of every guide once, the shaft joints' apart: those
        // are in `shaft_joint_sweeps`, whose test asks the same of each over its whole slider.
        let key = |check: &'static str, path: &'static str, direction: Direction| {
            (check, path, direction == Direction::Raise)
        };
        let mut listed: BTreeSet<_> = cases.iter().map(|&(c, _, p, d, _)| key(c, p, d)).collect();
        assert_eq!(listed.len(), cases.len());
        listed.extend(
            shaft_joint_sweeps()
                .into_iter()
                .map(|(_, path, direction, _, _)| key(SHAFT_JOINT, path, direction)),
        );
        let slider_fixes: BTreeSet<_> = CHECK_GUIDES
            .iter()
            .flat_map(|g| {
                possible_fixes(g)
                    .into_iter()
                    .filter(|&(_, d)| matches!(d, Direction::Raise | Direction::Lower))
                    .map(move |(p, d)| key(g.check, p, d))
            })
            .collect();
        assert_eq!(listed, slider_fixes);
        for (check, failing_design, path, direction, term) in cases {
            let before = compute_all(&failing_design);
            assert_ne!(
                level_of(check, &failing_design, &before),
                Level::Good,
                "{check}: the case's design fails it"
            );
            let to_end = |range: SliderRange| match direction {
                Direction::Raise => range.max,
                _ => range.min,
            };
            // The Outside diameter moves as the Size tab's slider moves it (cap following).
            let moved = if path == RING {
                let size = dimensions(&failing_design).unwrap();
                resize_diameter(&failing_design, to_end(size.diameter_range)).unwrap()
            } else {
                let range = slider_range(path);
                let end = match path {
                    "metal.cap_od_mm" => before.model.cup_od_mm,
                    _ => to_end(range),
                };
                with_number(&failing_design, path, end).unwrap()
            };
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
        // Each failing term's fixes, each shown only where one of its moves (one step, or one
        // choice of that input alone) moves that term toward passing without turning another
        // check red. A ring's magnet is fixed in that ring: each ring has its own limits.
        use Direction::{Change, CustomHotterGrade, Lower, Raise};
        let [part_inner, part_outer] = PART_PATHS.map(|path| (path, Change));
        let [grade_inner, grade_outer] = GRADE_PATHS.map(|path| (path, Change));
        let [hotter_inner, hotter_outer] = PART_PATHS.map(|path| (path, CustomHotterGrade));
        let adhesive = ("temperature.adhesive.selected", Change);
        let trip = ("temperature.duty.fault_trip_s", Lower);
        let conductance = ("temperature.thermal.conductance_W_K", Raise);
        let temperature = guide(TEMPERATURE).unwrap();
        let shown =
            |inputs: DesignInputs| fixes(temperature, &inputs, &compute_all(&inputs), false);
        // Above both rings' limits and the adhesive's. No library part is rated above the
        // N42SH parts in use, and a grade does nothing to a ring with a library part. Custom
        // dimensions with a blank grade has no rating, so its limit is not calibrated: no fix.
        // Each ring's way is Custom dimensions with a grade rated hotter than the part. The
        // start is above both limits, so no cooling can bring the peak under them (review 2,
        // N1): the hot-day fixes clear both.
        assert_eq!(
            shown(breaking(hot_day)),
            [hotter_inner, hotter_outer, adhesive]
        );
        // The inner ring already on manual dimensions with a blank grade: no library part beats
        // it, a grade does. The ring's own limit is the uncalibrated one, so a grade shows only
        // where its rated limit beats that. The outer ring is still on its library part.
        assert_eq!(
            shown(breaking(custom_hot_day)),
            [grade_inner, hotter_outer, adhesive]
        );
        assert_eq!(shown(breaking(n42_parts)), [part_inner, part_outer]);
        assert_eq!(shown(breaking(slip_fault)), [trip, conductance]);
        assert_eq!(shown(breaking(cool_adhesive)), [adhesive]);
        assert_eq!(shown(breaking(hot_cure)), [adhesive]);
        // Ferrite in both rings, each below its own cold limit: either ring's part or grade.
        assert_eq!(
            shown(breaking(ferrite)),
            [part_inner, grade_inner, part_outer, grade_outer]
        );
        assert_eq!(shown(passing_design()), []);
        // A trip time already at its slider's end is not offered: a 92 °C start and the
        // shortest trip still peak past the magnet limit, and only the conductance can help.
        let quick_trip = breaking(|i| {
            slip_fault(i);
            i.temperature.duty.fault_trip_s = 0.1;
            i.temperature.duty.hot_ambient_C = 82.0;
        });
        let results = compute_all(&quick_trip);
        assert_eq!(
            failing_terms(&quick_trip, &results),
            [TemperatureTerm::MagnetPeak]
        );
        assert_eq!(shown(quick_trip), [conductance]);
        // One slip event longer than the fault trip sets the peak: a shorter trip would not
        // lower it, so only the conductance is offered.
        let long_event = breaking(|i| {
            slip_fault(i);
            i.temperature.duty.fault_trip_s = 0.5;
            i.metal.slip_event_s = 5.0;
        });
        assert_eq!(shown(long_event), [conductance]);
    }

    #[test]
    fn every_move_of_a_fix_changes_its_own_input_only() {
        // Review I-2: a fix's move changes the inputs the fix names and nothing else, so the
        // test of what it does is the test of what the screen says. The Outside diameter is
        // the exception the Size tab makes: its slider moves the cap and its thread with the
        // cup. Custom dimensions with a hotter grade names its ring's part and grade.
        let catalogue = InputCatalogue::get();
        let same = |a: Option<Value>, b: Option<Value>| match (a, b) {
            (Some(Value::Num(x)), Some(Value::Num(y))) => x.to_bits() == y.to_bits(),
            (a, b) => a == b,
        };
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            for g in &CHECK_GUIDES {
                for (path, direction) in possible_fixes(g) {
                    let ring_grade = PART_PATHS
                        .iter()
                        .position(|&part| part == path)
                        .map(|index| GRADE_PATHS[index]);
                    let allowed: Vec<&str> = match (direction, ring_grade) {
                        _ if path == RING => {
                            vec![RING, "metal.cap_od_mm", "metal.cap_thread_dia_mm"]
                        }
                        (Direction::CustomHotterGrade, Some(grade)) => vec![path, grade],
                        _ => vec![path],
                    };
                    for moved in moves(&inputs, &results, path, direction) {
                        let changed: Vec<&str> = catalogue
                            .all()
                            .map(|entry| entry.path.as_str())
                            .filter(|&other| !same(inputs.get(other), moved.get(other)))
                            .collect();
                        assert!(
                            !changed.is_empty() && changed.iter().all(|p| allowed.contains(p)),
                            "{name}: {} changes {changed:?}",
                            fix_text(path, direction)
                        );
                    }
                }
            }
        }
    }

    /// The magnet of the ring whose part or grade input is at `path`, as the engine resolves it
    /// for `inputs`; `None` for any other input.
    fn ring_magnet(path: &str, inputs: &DesignInputs) -> Option<ResolvedMagnet> {
        let index = PART_PATHS
            .iter()
            .position(|&part| part == path)
            .or_else(|| GRADE_PATHS.iter().position(|&grade| grade == path))?;
        let (inner, outer) = resolve_magnets(&inputs.coupling.magnets, Deviations::ALL);
        Some([inner, outer][index])
    }

    #[test]
    fn no_magnet_move_lands_on_a_magnet_without_a_rating() {
        // Review N-1. A magnet with neither a library part nor a grade ("Custom dimensions" with
        // a blank grade) has no rating, so the engine leaves its demagnetization limit
        // uncalibrated: the same block reads about 10 °C above the library part's limit. A move
        // there would pass the temperature check with no change of magnet, so no magnet fix's
        // move goes there, and a grade's move puts the grade in use (manual dimensions).
        let mut checked = 0;
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            for g in &CHECK_GUIDES {
                for (path, direction) in possible_fixes(g) {
                    for moved in moves(&inputs, &results, path, direction) {
                        let Some(magnet) = ring_magnet(path, &moved) else {
                            continue;
                        };
                        let text = fix_text(path, direction);
                        assert!(
                            matches!(magnet.tmax_C, NumOrText::Num(_)),
                            "{name}: {text} reaches a magnet without a rating"
                        );
                        if GRADE_PATHS.contains(&path) {
                            assert!(
                                magnet.from_grade,
                                "{name}: {text} reaches a grade not in use"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "the designs reach some magnet moves");
    }

    #[test]
    fn a_warm_day_past_the_hottest_library_part_offers_custom_dimensions_with_a_hotter_grade() {
        // Review N-1, its probe: the passing design on a 98 °C hot day, above the 150 °C-rated
        // N42SH parts' limit in both rings. No library part is rated hotter. "Custom dimensions"
        // with a blank grade is the same block without a rating: its uncalibrated limit would
        // read the temperature check as passing with the hot-low torque unchanged. So that move
        // is no fix; the real one is two picks, the part to Custom dimensions and a grade rated
        // hotter than the part, scored as one move.
        use Direction::{Change, CustomHotterGrade};
        let warm = breaking(|i| i.temperature.duty.hot_ambient_C = 88.0);
        let results = compute_all(&warm);
        assert_eq!(results.temperature.summary.hot_day_start_C, 98.0);
        let rating = |r: &DesignResults| (r.model.inner_tmax_C, r.model.outer_tmax_C);
        assert_eq!(
            rating(&results),
            (NumOrText::Num(150.0), NumOrText::Num(150.0))
        );
        for ring in Ring::BOTH {
            assert!(
                failing_terms(&warm, &results).contains(&TemperatureTerm::HotDayMagnet(ring)),
                "{ring:?}"
            );
        }
        // The false pass the guide must not lead to.
        let unrated = with(warm.clone(), |i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        });
        let unrated_results = compute_all(&unrated);
        assert_eq!(check_level(&results, TEMPERATURE), Some(Level::Caution));
        assert_eq!(
            check_level(&unrated_results, TEMPERATURE),
            Some(Level::Good)
        );
        let hot_low = |r: &DesignResults| format_value(&Value::Num(r.metal.torque_hot_low_Nm));
        assert_eq!(hot_low(&unrated_results), hot_low(&results));
        // What the guide shows: each ring's two picks. The start is above the magnet limit,
        // so no cooling can bring the peak under it (review 2, N1).
        let shown = fixes(guide(TEMPERATURE).unwrap(), &warm, &results, false);
        assert_eq!(
            shown,
            [
                (PART_PATHS[0], CustomHotterGrade),
                (PART_PATHS[1], CustomHotterGrade),
            ]
        );
        assert!(!shown.contains(&(PART_PATHS[0], Change)));
        // Each of the two picks' moves puts a grade rated hotter than the part in use.
        for (index, part) in PART_PATHS.into_iter().enumerate() {
            let reached = moves(&warm, &results, part, CustomHotterGrade);
            assert!(!reached.is_empty(), "{part}");
            for moved in reached {
                let after = compute_all(&moved);
                let magnet = ring_magnet(part, &moved).unwrap();
                assert!(magnet.from_grade, "{part}");
                let ring_rating = [rating(&after).0, rating(&after).1][index];
                match ring_rating {
                    NumOrText::Num(t) => assert!(t > 150.0, "{part}: {t}"),
                    NumOrText::Text(text) => panic!("{part}: {text}"),
                }
            }
        }
        assert_eq!(
            fix_text(PART_PATHS[0], CustomHotterGrade),
            "Change Inner magnet part to Custom dimensions (manual), enter the part's dimensions, then set Inner magnet grade (manual dimensions) to one rated hotter than the part"
        );
    }

    #[test]
    fn the_space_claim_fixes_follow_the_axis_over_its_claim() {
        // Review I-3: each axis over its claim brings the inputs that shrink it, each shown
        // where one step does: the magnets for the two lengths; for the diameter the cap, and
        // the Outside diameter, whose slider moves the cap with the cup. An axis whose claim is
        // not a number has no fix: its Problem asks for the number.
        let space = guide(SPACE_CLAIM).unwrap();
        let shown = |inputs: DesignInputs| fixes(space, &inputs, &compute_all(&inputs), false);
        let magnets = (MAGNET_LENGTH, Direction::Lower);
        let cap = ("metal.cap_od_mm", Direction::Lower);
        let diameter = (RING, Direction::Lower);
        // Over the overall length and the bay: the magnets only.
        assert_eq!(shown(breaking(long_magnets)), [magnets]);
        // The cap sets the diameter: the cap, or the Outside diameter with the cap following.
        assert_eq!(shown(breaking(large_cap)), [cap, diameter]);
        // The cup sets the diameter: a smaller cap leaves it as it is.
        let ring = breaking(large_ring);
        let results = compute_all(&ring);
        assert!(results.model.cup_od_mm > ring.metal.cap_od_mm);
        assert_eq!(shown(ring), [diameter]);
        assert_eq!(shown(breaking(unknown_claim)), []);
        assert_eq!(shown(passing_design()), []);
    }

    #[test]
    fn a_slider_fix_whose_step_turns_another_check_red_is_not_offered() {
        // Review focus 3. The defaults at 12 poles: the blocks do not fit, while the hot-low
        // torque and the cup wall pass. One step down to 10 poles turns both red, so only the
        // larger ring is offered. On the passing design at 12 poles the same step only returns
        // to a passing design, so both fixes are offered.
        let fit = guide(FIT_CHECK).unwrap();
        let shown = |inputs: &DesignInputs| fixes(fit, inputs, &compute_all(inputs), false);
        let crowded = with_poles(12);
        let results = compute_all(&crowded);
        for check in [HOT_MIN, CUP_WALL] {
            assert_eq!(level_of(check, &crowded, &results), Level::Good, "{check}");
            let ten = with_poles(10);
            assert_eq!(
                level_of(check, &ten, &compute_all(&ten)),
                Level::Bad,
                "{check}"
            );
        }
        assert_eq!(shown(&crowded), [(RING, Direction::Raise)]);
        assert_eq!(
            shown(&breaking(|i| i.coupling.npole = 12)),
            [(RING, Direction::Raise), (POLES, Direction::Lower)]
        );
    }

    #[test]
    fn raising_the_diameter_is_not_offered_where_no_diameter_on_its_slider_fits() {
        // Physics review 3, N3: 40 blocks need about four times the ring's flats, past the
        // slider's top, and the Size tab says so beside the fix ("Does not fit at any diameter
        // up to ..."). Only fewer poles is offered. Where a larger diameter does fit, the
        // larger diameter is offered.
        let fit = guide(FIT_CHECK).unwrap();
        let shown = |inputs: &DesignInputs| fixes(fit, inputs, &compute_all(inputs), false);
        let nowhere = |inputs: &DesignInputs| {
            matches!(
                smallest_fitting_diameter(inputs, inputs.coupling.npole, None),
                Some(DiameterFit::Nowhere(_))
            )
        };
        let crowded = breaking(|i| i.coupling.npole = 40);
        let crowded_arcs = breaking(|i| {
            i.coupling.npole = 40;
            arcs(i);
        });
        for (name, inputs) in [("crowded", &crowded), ("crowded_arcs", &crowded_arcs)] {
            assert!(nowhere(inputs), "{name}");
            assert!(!fits_on_the_slider(inputs), "{name}");
            // One step up would still widen each place and turn nothing red: the slider's
            // reach, not the step, rules the fix out.
            assert!(
                helps(
                    inputs,
                    &compute_all(inputs),
                    (RING, Direction::Raise),
                    None,
                    false
                ),
                "{name}"
            );
            assert_eq!(shown(inputs), [(POLES, Direction::Lower)], "{name}");
        }
        let twelve_poles = with_poles(12);
        let twelve_poles_with_room = breaking(|i| i.coupling.npole = 12);
        for (name, inputs) in [
            ("twelve_poles", &twelve_poles),
            ("twelve_poles_with_room", &twelve_poles_with_room),
        ] {
            assert!(!nowhere(inputs), "{name}");
            assert!(shown(inputs).contains(&(RING, Direction::Raise)), "{name}");
        }
        // Only the fit guide drops it: the space claim lowers the diameter, never raises it.
        assert!(fits_on_the_slider(&DesignInputs::default()));
    }

    #[test]
    fn a_fix_at_its_slider_end_is_not_offered() {
        let shown = |check: &str, inputs: &DesignInputs| {
            fixes(guide(check).unwrap(), inputs, &compute_all(inputs), false)
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

    /// The defaults with the gearbox input rating just above cold-high at the gearbox input:
    /// halfway into the rise one step of the magnet-length slider gives (spec 7, item 7). The
    /// rating feeds no other result, so the step and its rise are the defaults' own.
    fn near_the_gearbox_limit() -> DesignInputs {
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let [longer] = &moves(&defaults, &results, MAGNET_LENGTH, Direction::Raise)[..] else {
            panic!("one step of longer magnets from the defaults")
        };
        let at = results.metal.cold_high_input_Nm;
        let rise = compute_all(longer).metal.cold_high_input_Nm - at;
        assert!(rise > 0.0, "longer magnets raise cold-high: {rise}");
        design(|i| i.coupling.gearbox_input_rating_Nm = at + rise / 2.0)
    }

    /// Asserts that the guide at `guide_index` of [`CHECK_GUIDES`] has no fix to show in this
    /// design (`inputs`, whose results are `results`) with the shaft-clamp section paused or
    /// shown (`paused`), and that the dashboard, fed the fixes as the panel hands them to it (no
    /// Minimum OD set), draws [`NO_FIX`] right after the Fix label of the guide's open block,
    /// found by its `problem`: `dashboard::fix_line` draws it for an open block whose fixes are
    /// empty (ruling B-4).
    fn assert_open_block_draws_no_fix(
        inputs: &DesignInputs,
        results: &DesignResults,
        guide_index: usize,
        problem: &str,
        paused: bool,
    ) {
        let all = within_minimum_diameter(&all_fixes(inputs, results, paused), inputs, None);
        assert_eq!(all[guide_index], [], "paused {paused}");
        let output = sized_frame(
            &egui::Context::default(),
            egui::vec2(1280.0, 4000.0),
            Vec::new(),
            |ui| dashboard_ui(ui, inputs, results, &all, paused, &mut Readouts::default()),
        );
        let texts = drawn_texts(&output);
        let check = CHECK_GUIDES[guide_index].check;
        let open = texts
            .iter()
            .position(|t| t == problem)
            .unwrap_or_else(|| panic!("the block of {check} is open, paused {paused}"));
        let fix = open + texts[open..].iter().position(|t| t == FIX).unwrap();
        assert_eq!(texts[fix + 1], NO_FIX, "paused {paused}");
    }

    #[test]
    fn near_the_gearbox_limit_the_hot_minimum_guide_shows_no_fix() {
        // Spec 7, item 7, a test of its own rather than a `designs()` entry: there the hot
        // minimum guide's only fix is held back by construction. The gearbox check passes just
        // under its rating and the hot minimum check fails; one step of longer magnets, the
        // hot minimum's only fix, would turn the gearbox check red, so `helps` holds it back
        // and the hot minimum's open block draws the no-fix line (`dashboard::fix_line` draws
        // `NO_FIX` for an open block whose fixes are empty), with the shaft-clamp section
        // shown and paused.
        let inputs = near_the_gearbox_limit();
        let results = compute_all(&inputs);
        assert_eq!(level_of(GEARBOX, &inputs, &results), Level::Good);
        assert_eq!(problem(GEARBOX, &inputs), None);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Bad);
        let longer = (MAGNET_LENGTH, Direction::Raise);
        let [stepped] = &moves(&inputs, &results, longer.0, longer.1)[..] else {
            panic!("one step of longer magnets")
        };
        assert_eq!(
            level_of(GEARBOX, stepped, &compute_all(stepped)),
            Level::Bad
        );
        let hot_min = guide_index(HOT_MIN);
        let problem = hot_min_problem(&inputs, &results).unwrap();
        for paused in [false, true] {
            assert!(
                !helps(&inputs, &results, longer, None, paused),
                "paused {paused}"
            );
            assert_open_block_draws_no_fix(&inputs, &results, hot_min, &problem, paused);
        }
        // With the rating a whole rise further up, the step stays within it and the fix shows.
        let rise =
            2.0 * (inputs.coupling.gearbox_input_rating_Nm - results.metal.cold_high_input_Nm);
        let roomier = with(inputs, |i| i.coupling.gearbox_input_rating_Nm += rise);
        let results = compute_all(&roomier);
        assert_eq!(fixes(&HOT_MIN_GUIDE, &roomier, &results, false), [longer]);
    }

    #[test]
    fn at_the_hot_minimum_s_limit_the_gearbox_guide_shows_no_fix_and_its_problem_says_why() {
        // Spec 5 (5A) and 4A: the length Auto solves, set by hand, puts hot-low at the
        // requirement; cold-high over hot-low does not depend on the geometry, so the gearbox
        // check fails there. Both its fixes lower hot-low, so `helps` holds each back (one step
        // would turn the hot minimum red), the open gearbox block draws NO_FIX right after its
        // Fix label, and its Problem's sentence 2 says why, with the clamp section shown and
        // paused.
        let (_, inputs) = auto_on_length(&DesignInputs::default());
        let results = compute_all(&inputs);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Good);
        assert_eq!(level_of(GEARBOX, &inputs, &results), Level::Bad);
        let gearbox = guide_index(GEARBOX);
        let problem = gearbox_problem(&inputs, &results).unwrap();
        let conflict = gearbox_conflict(&inputs, &results).expect("no geometry passes both");
        assert!(problem.ends_with(&format!(" {conflict}")), "{problem}");
        for paused in [false, true] {
            for fix in [
                (MAGNET_LENGTH, Direction::Lower),
                (FACE_GAP, Direction::Raise),
            ] {
                assert!(
                    !helps(&inputs, &results, fix, None, paused),
                    "{fix:?} paused {paused}"
                );
            }
            assert_open_block_draws_no_fix(&inputs, &results, gearbox, &problem, paused);
        }
    }

    #[test]
    fn each_gearbox_fix_moves_the_check_toward_passing_all_along_its_slider() {
        // Spec 5: both gearbox fixes are monotone over their sliders from the example design,
        // so a step anywhere along the way, not only the slider's end, moves cold-high at the
        // gearbox input down (200 points from the default to the end the fix points to).
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        for (path, direction) in [
            (MAGNET_LENGTH, Direction::Lower),
            (FACE_GAP, Direction::Raise),
        ] {
            let range = slider_range(path);
            let start = current_number(&defaults, &results, path).unwrap();
            let end = match direction {
                Direction::Lower => range.min,
                _ => range.max,
            };
            let mut last = results.metal.cold_high_input_Nm;
            for k in 1..=200 {
                let x = start + (end - start) * f64::from(k) / 200.0;
                let inputs = with_number(&defaults, path, x).unwrap();
                let now = compute_all(&inputs).metal.cold_high_input_Nm;
                assert!(now < last, "{path} at {x}: {now} after {last}");
                last = now;
            }
        }
    }

    #[test]
    fn the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other() {
        // Spec 7.9, the test designs re-probed: without back iron both aluminium parts slip;
        // 50.8 mm magnets make the cup slip and its key alone bear too much (the cup does not
        // lengthen with the magnets); DP460's design limit, and the inner ring's single-ring onset
        // of library N42 blocks (owner R-15), keep the inner hub from being fitted; and the six
        // designs that break the joint alone. Every other test design passes, its check reading
        // the pass, never unknown (a joint that is not judged fails no term), the short magnets'
        // greyed row too (a cold-high below zero asks nothing of a fit). The two key designs
        // move a `metal.*` input, which other checks read too, and still fail the joint alone.
        let key_alone = ["cup_key_no_bearing_area", "inner_hub_key_no_bearing_area"];
        let want: Vec<(&str, Vec<String>)> = [
            ("n42_parts", &["inner hub fitting"][..]),
            ("cool_adhesive", &["inner hub fitting"]),
            ("long_magnets", &["cup slips", "cup key"]),
            ("no_back_iron", &["inner hub slips", "cup slips"]),
            ("joint_slips", &["inner hub slips"]),
            (
                "joint_overstressed",
                &["inner hub bore stress", "inner hub fitting"],
            ),
            ("joint_cannot_be_fitted", &["inner hub fitting"]),
            ("cup_key_no_bearing_area", &["cup slips", "cup key"]),
            (
                "inner_hub_key_no_bearing_area",
                &["inner hub slips", "inner hub key"],
            ),
            ("cup_stress_alone", &["cup bore stress"]),
        ]
        .into_iter()
        .map(|(name, terms)| (name, terms.iter().map(|t| (*t).to_owned()).collect()))
        .collect();
        let failing: Vec<(&str, Vec<String>)> = designs()
            .into_iter()
            .filter_map(|(name, inputs)| {
                let results = compute_all(&inputs);
                let failing = failing_term_names(SHAFT_JOINT, &inputs, &results);
                if failing.is_empty() {
                    assert_eq!(
                        results.shaft_joint.check,
                        shaft_joint::SHAFT_JOINTS_PASS,
                        "{name}"
                    );
                }
                if key_alone.contains(&name) {
                    let red: Vec<&str> = CHECK_GUIDES
                        .iter()
                        .filter(|g| level(g, &inputs, &results) == Level::Bad)
                        .map(|g| g.check)
                        .collect();
                    assert_eq!(red, [SHAFT_JOINT], "{name}");
                }
                (!failing.is_empty()).then_some((name, failing))
            })
            .collect();
        assert_eq!(failing, want);
    }

    /// The inner hub's slip fixes, as the guide shows them where each helps (spec 5.2): its
    /// minimum up, the hub longer, the face gap wider.
    const HUB_SLIP_FIXES: [(&str, Direction); 3] = [
        (HUB_MIN, Direction::Raise),
        (HUB_LENGTH, Direction::Raise),
        (FACE_GAP, Direction::Raise),
    ];

    /// The cup's slip fixes, as the guide shows them where each helps (spec 5.2): its minimum
    /// up, the boss longer, the magnets shorter, the face gap wider.
    const CUP_SLIP_FIXES: [(&str, Direction); 4] = [
        (CUP_MIN, Direction::Raise),
        (BOSS_LENGTH, Direction::Raise),
        (MAGNET_LENGTH, Direction::Lower),
        (FACE_GAP, Direction::Raise),
    ];

    #[test]
    fn the_shaft_joint_fixes_follow_the_failing_terms() {
        // Spec 5.2: each failing term's fixes, joint by joint, each shown only where one step
        // moves that term toward passing and turns no passing check red, each once, in the
        // badge's order (the slips first, the inner hub's before the cup's).
        use Direction::{Lower, Raise};
        let joint = guide(SHAFT_JOINT).unwrap();
        let shown = |inputs: DesignInputs| fixes(joint, &inputs, &compute_all(&inputs), false);
        assert_eq!(shown(only_the_joint_fails()), HUB_SLIP_FIXES);
        assert_eq!(shown(breaking(cup_slips)), CUP_SLIP_FIXES);
        // The inner hub's bore stress and fitting temperature: its maximum down; its wheel shaft
        // is at the slider's bottom already.
        assert_eq!(shown(breaking(hub_overstressed)), [(HUB_MAX, Lower)]);
        // The cup's bore stress alone: its maximum down.
        assert_eq!(shown(breaking(cup_stress_alone)), [(CUP_MAX, Lower)]);
        // An uncooled wheel shaft: the maximum down, or the shaft colder.
        assert_eq!(
            shown(breaking(uncooled_wheel_shaft)),
            [(HUB_MAX, Lower), (WHEEL_SHAFT, Lower)]
        );
        // The bare cup allowed 250 °C: its maximum down, or the gearbox shaft colder (to the
        // slider's -20 °C).
        assert_eq!(
            shown(breaking(low_cup_fit_limit)),
            [(CUP_MAX, Lower), (GEARBOX_SHAFT, Lower)]
        );
        // No back iron: both aluminium parts slip; the face gap once, under the inner hub.
        let mut both = HUB_SLIP_FIXES.to_vec();
        both.extend([
            (CUP_MIN, Raise),
            (BOSS_LENGTH, Raise),
            (MAGNET_LENGTH, Lower),
        ]);
        assert_eq!(shown(breaking(no_back_iron)), both);
        // 50.8 mm magnets: the cup slips and its key alone bears too much.
        let mut long = CUP_SLIP_FIXES.to_vec();
        long.push((KEY_CONTACT, Raise));
        assert_eq!(shown(breaking(long_magnets)), long);
        assert_eq!(shown(passing_design()), []);
    }

    /// Asserts that one step of `fix` on `inputs` (whose results are `results`) reverses the
    /// shaft joints' band: the stepped design reads the reversed band, amber.
    fn assert_one_step_reverses_the_band(
        inputs: &DesignInputs,
        results: &DesignResults,
        fix: (&str, Direction),
    ) {
        let [stepped] = &moves(inputs, results, fix.0, fix.1)[..] else {
            panic!("one step of {}", fix.0)
        };
        let after = compute_all(stepped);
        assert!(
            after
                .shaft_joint
                .check
                .starts_with(shaft_joint::UNKNOWN_BAND_REVERSED),
            "{}",
            after.shaft_joint.check
        );
        assert_eq!(level_of(SHAFT_JOINT, stepped, &after), Level::Caution);
    }

    #[test]
    fn raising_a_minimum_is_not_offered_where_one_step_reverses_the_band() {
        // Review SF-10. The passing design with the inner hub's band at 10-10 µm slips. One step
        // up on the minimum (10.5 µm) raises the slip margin, so `helps` alone would show it, but
        // it puts the minimum above the maximum, and the row would read amber, not green. The
        // hub length and the face gap are shown; the window sentence names the band to enter.
        // With a step of room in the band, the minimum is offered.
        let joint = guide(SHAFT_JOINT).unwrap();
        let inputs = breaking(|i| {
            i.shaft_joint.hub_interference_min_um = 10.0;
            i.shaft_joint.hub_interference_max_um = 10.0;
        });
        let results = compute_all(&inputs);
        let raise_min = (HUB_MIN, Direction::Raise);
        let slip =
            |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, "inner hub slips", d, r);
        assert!(helps(&inputs, &results, raise_min, Some(&slip), false));
        assert_one_step_reverses_the_band(&inputs, &results, raise_min);
        assert_eq!(
            fixes(joint, &inputs, &results, false),
            [(HUB_LENGTH, Direction::Raise), (FACE_GAP, Direction::Raise)]
        );
        let text = shaft_joint_problem(&inputs, &results).unwrap();
        let band = format!(
            "(its band is {} to {})",
            input_text(&inputs, HUB_MIN),
            input_text(&inputs, HUB_MAX)
        );
        assert!(text.contains(&band), "{band} in {text}");
        let roomy = with(inputs, |i| i.shaft_joint.hub_interference_max_um = 10.5);
        let shown = fixes(joint, &roomy, &compute_all(&roomy), false);
        assert!(shown.contains(&raise_min), "{shown:?}");
    }

    #[test]
    fn lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band() {
        // Review 2, SF-2, the mirror. The passing design with DP460 and the inner hub's band at
        // 18-18 µm fails on fitting alone (93.3 against 60.0 °C). One step down on the maximum
        // raises the fitting margin but reverses the band, and the wheel shaft is at its slider's
        // bottom: no fix, and the open block draws the no-fix line with the clamp section shown
        // and paused; the window sentence names the band to enter (11.14 to 13.90 µm). With a
        // step of room in the band, the maximum is offered.
        let joint = guide(SHAFT_JOINT).unwrap();
        let inputs = breaking(|i| {
            cool_adhesive(i);
            i.shaft_joint.hub_interference_min_um = 18.0;
        });
        let results = compute_all(&inputs);
        let red: Vec<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| level(g, &inputs, &results) == Level::Bad)
            .map(|g| g.check)
            .collect();
        assert_eq!(red, [SHAFT_JOINT]);
        let lower_max = (HUB_MAX, Direction::Lower);
        let fitting =
            |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, "inner hub fitting", d, r);
        assert!(helps(&inputs, &results, lower_max, Some(&fitting), false));
        assert_one_step_reverses_the_band(&inputs, &results, lower_max);
        assert!(moves(&inputs, &results, WHEEL_SHAFT, Direction::Lower).is_empty());
        let problem = shaft_joint_problem(&inputs, &results).unwrap();
        assert_in_order(
            &problem,
            &[
                "from 11.14 µm to 13.90 µm",
                "(its band is 18.00 µm to 18.00 µm)",
            ],
        );
        let index = guide_index(SHAFT_JOINT);
        for paused in [false, true] {
            assert_open_block_draws_no_fix(&inputs, &results, index, &problem, paused);
        }
        let roomy = with(inputs, |i| i.shaft_joint.hub_interference_min_um = 17.5);
        assert_eq!(
            fixes(joint, &roomy, &compute_all(&roomy), false),
            [lower_max]
        );
    }

    #[test]
    fn raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length() {
        // Spec 5.2 (review MF-2, SF-2): a key no longer than its joint's engaged length less its
        // width has no bearing length, and no contact height gives it bearing area, though the
        // key margin rises with the height. Only the length fix is shown: the boss for the cup,
        // the hub for the inner hub. With the shared contact height typed 0, the contact is what
        // each key lacks: its fix is shown, and each key sentence is the contact one, never the
        // length one beside it (review 2, SF-5); a contact typed below zero reads as typed, and
        // as spec 5.2 says (build ruling B-5), it is judged and its fix shown (one step up lands
        // on the slider's bottom, where both keys bear), while under an unknown no fix shows.
        let joint = guide(SHAFT_JOINT).unwrap();
        let contact = (KEY_CONTACT, Direction::Raise);
        for (name, inputs, term, length) in [
            (
                "cup",
                breaking(no_boss),
                "cup key",
                (BOSS_LENGTH, Direction::Raise),
            ),
            (
                "inner hub",
                breaking(short_hub),
                "inner hub key",
                (HUB_LENGTH, Direction::Raise),
            ),
        ] {
            let results = compute_all(&inputs);
            let key = |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, term, d, r);
            assert!(key(&inputs, &results).is_finite(), "{name}");
            assert!(
                helps(&inputs, &results, contact, Some(&key), false),
                "{name}"
            );
            let shown = fixes(joint, &inputs, &results, false);
            assert!(!shown.contains(&contact), "{name}: {shown:?}");
            assert!(shown.contains(&length), "{name}: {shown:?}");
        }
        let no_contact = breaking(|i| {
            no_boss(i);
            i.shaft_joint.key_contact_mm = 0.0;
        });
        let results = compute_all(&no_contact);
        assert!(fixes(joint, &no_contact, &results, false).contains(&contact));
        let text = shaft_joint_problem(&no_contact, &results).unwrap();
        for name in [shaft_joint::INNER_HUB, shaft_joint::CUP] {
            let sentence = format!(
                "The key has no bearing area in the {name}: its contact height, {}, is not above zero.",
                input_text(&no_contact, KEY_CONTACT)
            );
            assert!(text.contains(&sentence), "{sentence} in {text}");
        }
        assert!(!text.contains("its boss and web"), "{text}");
        let negative = breaking(|i| i.shaft_joint.key_contact_mm = -0.5);
        let text = problem(SHAFT_JOINT, &negative).unwrap();
        let typed = format!(
            "its contact height, {}, is not above zero",
            input_text(&negative, KEY_CONTACT)
        );
        assert!(text.contains(&typed), "{typed} in {text}");
        let results = compute_all(&negative);
        assert_eq!(level_of(SHAFT_JOINT, &negative, &results), Level::Bad);
        assert_eq!(fixes(joint, &negative, &results, false), [contact]);
        let unknown = with(negative, |i| i.shaft_joint.safety_factor = 0.0);
        let results = compute_all(&unknown);
        assert_eq!(level_of(SHAFT_JOINT, &unknown, &results), Level::Caution);
        assert_eq!(fixes(joint, &unknown, &results, false), []);
    }

    #[test]
    fn with_no_keyway_both_key_terms_pass_with_an_infinite_margin() {
        // Review 2, SF-5c: without a keyway the key term does not apply; its margin is +inf,
        // never the -inf a contact capped at a zero depth would give, so no fix chases it.
        let keyless = breaking(|i| i.coupling.keyway_depth_mm = 0.0);
        let results = compute_all(&keyless);
        let terms = terms(SHAFT_JOINT, &keyless, &results);
        assert_eq!(terms.len(), 8);
        for name in ["inner hub key", "cup key"] {
            let t = terms.iter().find(|t| t.name == name).unwrap();
            assert!(t.passes && t.margin == f64::INFINITY, "{name}: {t:?}");
        }
        assert_eq!(results.shaft_joint.check, shaft_joint::SHAFT_JOINTS_PASS);
    }

    #[test]
    fn each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider() {
        // Spec 5.2: each margin is monotone over its fix's whole slider, every value at the
        // slider's own step from bottom to top, rising the way the fix moves, and flat only where
        // its case's reading holds still (`shaft_joint_sweeps`). As
        // `every_fix_moves_its_check_toward_passing_over_its_slider` asks of the other guides'
        // fixes, each case's design fails the check, and the fix's input at its slider's end in
        // the fix's direction gives a larger margin than the design's own.
        let cases = shaft_joint_sweeps();
        // Every slider fix of the guide is swept.
        let swept: BTreeSet<(&str, bool)> = cases
            .iter()
            .map(|&(_, path, direction, _, _)| (path, direction == Direction::Raise))
            .collect();
        let listed: BTreeSet<(&str, bool)> = possible_fixes(guide(SHAFT_JOINT).unwrap())
            .into_iter()
            .map(|(path, direction)| (path, direction == Direction::Raise))
            .collect();
        assert_eq!(swept, listed);
        for (base, path, direction, term, flat_reading) in cases {
            let before = compute_all(&base);
            assert_ne!(
                level_of(SHAFT_JOINT, &base, &before),
                Level::Good,
                "{term} against {path}: the case's design fails the check"
            );
            let from = margin(SHAFT_JOINT, term, &base, &before);
            let index = usize::from(term.starts_with(shaft_joint::CUP));
            let mut values = slider_values(path);
            if direction == Direction::Lower {
                values.reverse();
            }
            let at = |x: f64| {
                let design = with_number(&base, path, x).unwrap();
                let results = compute_all(&design);
                let joints =
                    shaft_joint::joints(&design.shaft_joint, &shaft_joint_links(&design, &results));
                (margin(SHAFT_JOINT, term, &design, &results), joints[index])
            };
            let (mut last, mut last_joint) = at(values[0]);
            let mut case_flats = 0;
            for &x in &values[1..] {
                let (now, joint) = at(x);
                let what = format!("{term} against {path} at {x}: {last} -> {now}");
                assert!(now >= last, "{what}");
                if now == last {
                    let reading = flat_reading.unwrap_or_else(|| panic!("flat: {what}"));
                    assert_eq!(reading(&joint), reading(&last_joint), "{what}");
                    case_flats += 1;
                }
                (last, last_joint) = (now, joint);
            }
            assert!(
                last > from,
                "{}: moves the {term} margin {from} -> {last}",
                fix_text(path, direction)
            );
            // Each case with a reading reaches its own flat (the floor, the cap or the override).
            assert_eq!(
                case_flats > 0,
                flat_reading.is_some(),
                "{term} against {path}"
            );
        }
    }

    /// The defaults with the cup's minimum interference 0.001 µm above the smallest that holds:
    /// one step of longer magnets raises cold-high, and with it the torque the cup must hold,
    /// past what it holds (spec 7.9, near the joint limit; the cup does not lengthen with the
    /// magnets).
    fn near_the_joint_limit() -> DesignInputs {
        let needed = compute_all(&DesignInputs::default())
            .shaft_joint
            .cup_needed_um;
        design(|i| i.shaft_joint.cup_interference_min_um = needed + 0.001)
    }

    #[test]
    fn near_the_joint_limit_the_hot_minimum_guide_shows_no_fix() {
        // Spec 7.9 and R-7, a test of its own as BL-059's: the cup holds just above what it
        // must and the hot minimum fails. One step of longer magnets, the hot minimum's only fix,
        // makes the cup slip (its badge reads two numbers apart), so `helps` holds the fix back
        // and the hot minimum's open block draws the no-fix line, with the shaft-clamp section
        // shown and paused. The joint stays green. With half a micrometre more on the cup's
        // minimum, the fix shows again.
        let inputs = near_the_joint_limit();
        let results = compute_all(&inputs);
        assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Good);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Bad);
        let longer = (MAGNET_LENGTH, Direction::Raise);
        let [stepped] = &moves(&inputs, &results, longer.0, longer.1)[..] else {
            panic!("one step of longer magnets")
        };
        let after = compute_all(stepped);
        assert_eq!(level_of(SHAFT_JOINT, stepped, &after), Level::Bad);
        assert_eq!(
            after.shaft_joint.check,
            format!(
                "{} ({}): holds 7.64 N·m against 7.65 N·m",
                shaft_joint::FIT_SLIPS,
                shaft_joint::CUP
            )
        );
        let hot_min = guide_index(HOT_MIN);
        let problem = hot_min_problem(&inputs, &results).unwrap();
        for paused in [false, true] {
            assert!(
                !helps(&inputs, &results, longer, None, paused),
                "paused {paused}"
            );
            assert_open_block_draws_no_fix(&inputs, &results, hot_min, &problem, paused);
        }
        let roomier = with(inputs, |i| i.shaft_joint.cup_interference_min_um += 0.5);
        let results = compute_all(&roomier);
        assert_eq!(fixes(&HOT_MIN_GUIDE, &roomier, &results, false), [longer]);
    }

    #[test]
    fn where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade() {
        // Owner R-15 and its ring ruling (spec 5.2, "Where the onset binds"; 7.9). Library N42
        // blocks in both rings cap the bonded inner hub's allowed fitting temperature at the
        // inner ring's own single-ring onset, 69.83 °C, less the cure margin: 59.83 °C against
        // the 93.27 °C its fit needs. The joint fails on that term alone; the temperature check
        // stays amber. One step down on the hub's maximum raises the fitting margin and turns
        // nothing red, so it is shown; the wheel shaft is at its slider's bottom. The magnets'
        // part or grade is the temperature guide's fix, never the joint's. N50M on manual
        // dimensions too: its onset, 98.80 °C, is above the need; only the onset less the margin,
        // 88.80 °C, is below it (the cup wall fails there too: the joint alone is asserted). The
        // step down raises each fitting margin as spec 5.2 and 7.9 give it: -33.43 to -29.37 °C
        // and -4.46 to -0.40 °C.
        let joint = guide(SHAFT_JOINT).unwrap();
        for (name, inputs, want_onset, want_margins) in [
            ("n42_parts", breaking(n42_parts), 69.8326, [-33.43, -29.37]),
            (
                "N50M",
                breaking(|i| graded(i, "N50M")),
                98.8029,
                [-4.46, -0.40],
            ),
        ] {
            let results = compute_all(&inputs);
            let s = &results.shaft_joint;
            let onset = results.temperature.demag.inner_onset_single_ring_C;
            assert!((onset - want_onset).abs() < 1e-4, "{name}: {onset}");
            assert_eq!(s.hub_fit_allowed_C, onset - CURE_MARGIN_MIN_C, "{name}");
            assert!((s.hub_fit_temp_C - 93.2674).abs() < 1e-4, "{name}");
            assert_eq!(
                s.check,
                format!(
                    "{} ({}): needs 93.3 °C against {} °C",
                    shaft_joint::CANNOT_BE_FITTED,
                    shaft_joint::INNER_HUB,
                    fixed(onset - CURE_MARGIN_MIN_C, 1)
                ),
                "{name}"
            );
            let failing = failing_term_names(SHAFT_JOINT, &inputs, &results);
            assert_eq!(failing, ["inner hub fitting"], "{name}");
            assert_eq!(
                level_of(TEMPERATURE, &inputs, &results),
                Level::Caution,
                "{name}"
            );
            let shown = fixes(joint, &inputs, &results, false);
            assert_eq!(shown, [(HUB_MAX, Direction::Lower)], "{name}");
            let fitting =
                |d: &DesignInputs| margin(SHAFT_JOINT, "inner hub fitting", d, &compute_all(d));
            let [stepped] = &moves(&inputs, &results, HUB_MAX, Direction::Lower)[..] else {
                panic!("{name}: one step down")
            };
            let margins = [fitting(&inputs), fitting(stepped)];
            for (margin, want) in margins.into_iter().zip(want_margins) {
                assert!((margin - want).abs() < 5e-3, "{name}: {margins:?}");
            }
            assert!(
                moves(&inputs, &results, WHEEL_SHAFT, Direction::Lower).is_empty(),
                "{name}"
            );
        }
        for path in PART_PATHS.into_iter().chain(GRADE_PATHS) {
            assert!(
                possible_fixes(joint).iter().all(|&(p, _)| p != path),
                "{path}"
            );
        }
    }

    #[test]
    fn with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down() {
        // Review MF-5 (spec 7.9): DP460's 60 °C design limit caps the bonded inner hub's allowed
        // fitting temperature below the 93.27 °C its fit needs, the only failing term; one step
        // down on its maximum takes the need to 89.20 °C, and that fix is shown; the wheel shaft
        // is at its slider's bottom.
        let inputs = breaking(cool_adhesive);
        let results = compute_all(&inputs);
        let failing = failing_term_names(SHAFT_JOINT, &inputs, &results);
        assert_eq!(failing, ["inner hub fitting"]);
        assert_eq!(
            fixes(guide(SHAFT_JOINT).unwrap(), &inputs, &results, false),
            [(HUB_MAX, Direction::Lower)]
        );
        let [stepped] = &moves(&inputs, &results, HUB_MAX, Direction::Lower)[..] else {
            panic!("one step down")
        };
        let need = |r: &DesignResults| format_value(&Value::Num(r.shaft_joint.hub_fit_temp_C));
        assert_eq!(
            (need(&results), need(&compute_all(stepped))),
            ("93.27".to_owned(), "89.20".to_owned())
        );
    }

    #[test]
    fn the_default_inner_hub_band_is_two_steps_from_failing_on_fitting() {
        // Review focus (spec R-9): the example's inner hub band sits 0.83 µm under its fitting
        // edge. One step up on its maximum (18.5 µm) still passes; two (19 µm) fail on fitting,
        // 101.4 against 100.0 °C, and the guide offers the band back down: not the wheel shaft,
        // already at its slider's bottom, nor the allowed temperature (a limit, not a lever).
        let one = design(|i| i.shaft_joint.hub_interference_max_um = 18.5);
        assert_eq!(
            compute_all(&one).shaft_joint.check,
            shaft_joint::SHAFT_JOINTS_PASS
        );
        let two = design(|i| i.shaft_joint.hub_interference_max_um = 19.0);
        let results = compute_all(&two);
        assert_eq!(
            results.shaft_joint.check,
            format!(
                "{} ({}): needs 101.4 °C against 100.0 °C",
                shaft_joint::CANNOT_BE_FITTED,
                shaft_joint::INNER_HUB
            )
        );
        assert_eq!(
            fixes(guide(SHAFT_JOINT).unwrap(), &two, &results, false),
            [(HUB_MAX, Direction::Lower)]
        );
        let text = problem(SHAFT_JOINT, &two).unwrap();
        assert!(
            text.contains("your limit for the bonded inner hub"),
            "{text}"
        );
    }

    #[test]
    fn a_joint_that_is_not_judged_keeps_out_of_the_other_joint_s_badge_and_fixes() {
        // Spec 4.3 (review SF-8): beside a joint that fails, a joint that is not judged turns
        // nothing red and offers no fix; the badge and the fixes are the failing joint's, and its
        // Problem gives the failing joint's causes and window first, then why the other is not
        // judged. A reversed inner hub band with the cup slipping; the cup without a wall (bore
        // 16 mm, boss 12 mm) with the inner hub slipping. Without the failing joint (the hub's
        // minimum at 15 µm holds on the 16 mm bore) each reads amber with no fix.
        fn no_wall(i: &mut DesignInputs) {
            i.coupling.bore_mm = 16.0;
            i.metal.boss_od_mm = 12.0;
        }
        fn no_wall_alone(i: &mut DesignInputs) {
            no_wall(i);
            i.shaft_joint.hub_interference_min_um = 15.0;
        }
        let joint = guide(SHAFT_JOINT).unwrap();
        let cases = [
            (
                breaking(|i| {
                    reversed_hub_band(i);
                    cup_slips(i);
                }),
                shaft_joint::CUP,
                CUP_SLIP_FIXES.to_vec(),
                [
                    "At its loosest the cup holds",
                    "On the cup, a measured interference",
                    "The inner hub's minimum interference",
                ],
            ),
            (
                breaking(no_wall),
                shaft_joint::INNER_HUB,
                HUB_SLIP_FIXES.to_vec(),
                [
                    "At its loosest the inner hub holds",
                    "On the inner hub, a measured interference",
                    "The cup's bore",
                ],
            ),
        ];
        for (inputs, failing, want, order) in cases {
            let results = compute_all(&inputs);
            let badge = &results.shaft_joint.check;
            assert!(
                badge.starts_with(&format!("{} ({failing})", shaft_joint::FIT_SLIPS)),
                "{badge}"
            );
            assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Bad);
            assert_eq!(fixes(joint, &inputs, &results, false), want, "{badge}");
            assert_in_order(&shaft_joint_problem(&inputs, &results).unwrap(), &order);
        }
        for edit in [reversed_hub_band as fn(&mut DesignInputs), no_wall_alone] {
            let inputs = breaking(edit);
            let results = compute_all(&inputs);
            assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Caution);
            assert_eq!(fixes(joint, &inputs, &results, false), []);
        }
        // Neither judged (the whole-branch review's suggestion 1): the inner hub's band reversed
        // and the cup without a wall. The row reads amber with no fix, and the Problem is the two
        // not-judged sentences alone, the inner hub's first.
        let neither = breaking(|i| {
            reversed_hub_band(i);
            no_wall(i);
        });
        let results = compute_all(&neither);
        assert_eq!(level_of(SHAFT_JOINT, &neither, &results), Level::Caution);
        assert_eq!(fixes(joint, &neither, &results, false), []);
        let text = shaft_joint_problem(&neither, &results).unwrap();
        assert_in_order(
            &text,
            &["The inner hub's minimum interference", "The cup's bore"],
        );
        assert!(!text.contains("At its loosest"), "{text}");
        let joints = shaft_joints(&neither, &results).expect("the check can be made");
        let sentences: Vec<String> = JOINT_SIDES
            .iter()
            .zip(&joints)
            .flat_map(|(side, joint)| not_judged_sentences(&neither, side, joint))
            .collect();
        assert_eq!(text, sentences.join(" "));
    }

    #[test]
    fn two_failing_checks_that_pull_one_input_apart_say_so() {
        // Spec review focus 4: the default design's first screen once said "lower the gap" for
        // the hot-low torque and "raise the gap" for the clearance. A fix whose step turns a
        // passing check red is not offered, so that pair cannot show. One input still shows
        // both ways when two checks that pull it apart both fail already: here the hot-low
        // torque against the gearbox check, on the example design and wherever both fail (magnet
        // length: no magnet size passes both), and against an overall-length claim the stack is
        // over (magnet length), the hot-low torque against the cup's slip without back iron
        // (magnet length: the cup does not lengthen with the magnets, BL-064), and the blocks
        // against a diameter claim (Outside diameter). Each fix helps its own check, so both
        // stay, and each of the two blocks says, in the same words, that the input pulls two ways
        // and another input has to settle one of them (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 5] = [
            ("defaults", &[MAGNET_LENGTH]),
            ("no_back_iron", &[MAGNET_LENGTH]),
            ("bare_clearance", &[MAGNET_LENGTH]),
            ("length_conflict", &[MAGNET_LENGTH]),
            ("diameter_conflict", &[RING]),
        ];
        let note = |path: &str, raising: &str, lowering: &str| {
            format!(
                "One input pulls two ways ({}): {raising} needs it raised and {lowering} needs it lowered, so change another input to settle one of them.",
                fix_label(path)
            )
        };
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            let shown: Vec<(&str, Direction)> = all.iter().flatten().copied().collect();
            let both_ways: Vec<&str> = shown
                .iter()
                .filter(|&&(path, direction)| {
                    direction == Direction::Raise && shown.contains(&(path, Direction::Lower))
                })
                .map(|&(path, _)| path)
                .collect();
            let expected = pulled_apart
                .iter()
                .find(|&&(fixture, _)| fixture == name)
                .map_or(&[][..], |&(_, paths)| paths);
            assert_eq!(both_ways, expected, "{name}");
            // The guides that raise and lower each such input carry its note, one for each guide
            // that pulls it the other way; no other guide does.
            let pulling = |path: &str, direction: Direction| -> Vec<&str> {
                CHECK_GUIDES
                    .iter()
                    .zip(&all)
                    .filter(|(_, fixes)| fixes.contains(&(path, direction)))
                    .map(|(g, _)| g.label)
                    .collect()
            };
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                let mut wanted = Vec::new();
                for &(path, direction) in &all[index] {
                    if !both_ways.contains(&path) {
                        continue;
                    }
                    match direction {
                        Direction::Raise => {
                            for lowering in pulling(path, Direction::Lower) {
                                wanted.push(note(path, g.label, lowering));
                            }
                        }
                        Direction::Lower => {
                            for raising in pulling(path, Direction::Raise) {
                                wanted.push(note(path, raising, g.label));
                            }
                        }
                        Direction::Change | Direction::CustomHotterGrade => {}
                    }
                }
                assert_eq!(trade_offs(&all, index), wanted, "{name}: {}", g.check);
            }
        }
        // The cases in words. The overall-length claim: the space claim's note in its block and
        // in the hot-low torque's, which also carries the gearbox check's (it lowers the
        // magnets too).
        let length = design(|i| i.metal.max_overall_axial_mm = 30.0);
        let results = compute_all(&length);
        let all = all_fixes(&length, &results, false);
        let space = "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Space claim needs it lowered, so change another input to settle one of them.";
        let gearbox = note(MAGNET_LENGTH, HOT_MIN_GUIDE.label, GEARBOX_GUIDE.label);
        assert_eq!(
            trade_offs(&all, guide_index(HOT_MIN)),
            [gearbox.as_str(), space]
        );
        assert_eq!(trade_offs(&all, guide_index(GEARBOX)), [gearbox.as_str()]);
        assert_eq!(trade_offs(&all, guide_index(SPACE_CLAIM)), [space]);
        // The example design: the hot-low torque against the gearbox check alone, the same note
        // in both blocks.
        let defaults = DesignInputs::default();
        let all = all_fixes(&defaults, &compute_all(&defaults), false);
        assert_eq!(trade_offs(&all, guide_index(HOT_MIN)), [gearbox.as_str()]);
        assert_eq!(trade_offs(&all, guide_index(GEARBOX)), [gearbox.as_str()]);
        // No back iron: the cup slips (BL-064) beside the failing hot-low torque, the same note
        // in both blocks.
        let aluminium = breaking(no_back_iron);
        let all = all_fixes(&aluminium, &compute_all(&aluminium), false);
        let joint = note(MAGNET_LENGTH, HOT_MIN_GUIDE.label, SHAFT_JOINT_GUIDE.label);
        assert_eq!(trade_offs(&all, guide_index(HOT_MIN)), [joint.as_str()]);
        assert_eq!(trade_offs(&all, guide_index(SHAFT_JOINT)), [joint.as_str()]);
        let crowded = with(with_poles(12), |i| i.metal.max_diameter_mm = 42.0);
        let results = compute_all(&crowded);
        let all = all_fixes(&crowded, &results, false);
        let expected = [
            "One input pulls two ways (Outside diameter): Magnets fit needs it raised and Space claim needs it lowered, so change another input to settle one of them.",
        ];
        assert_eq!(trade_offs(&all, guide_index(FIT_CHECK)), expected);
        assert_eq!(trade_offs(&all, guide_index(SPACE_CLAIM)), expected);
        // An index past the guides, or lists shorter than the guides, read as no note.
        assert!(trade_offs(&all, CHECK_GUIDES.len()).is_empty());
        assert!(trade_offs(&all[..1], guide_index(SPACE_CLAIM)).is_empty());
    }

    #[test]
    fn all_fixes_lists_each_guide_s_fixes_in_the_guides_order() {
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            assert_eq!(all.len(), CHECK_GUIDES.len(), "{name}");
            for (guide, shown) in CHECK_GUIDES.iter().zip(&all) {
                assert_eq!(
                    *shown,
                    fixes(guide, &inputs, &results, false),
                    "{name}: {}",
                    guide.check
                );
            }
        }
        let passing = passing_design();
        let all = all_fixes(&passing, &compute_all(&passing), false);
        assert!(all.iter().all(Vec::is_empty), "{all:?}");
    }

    #[test]
    fn the_fix_cache_computes_once_per_design_and_follows_an_edit() {
        // Every guide's fixes, in the guides' order (none for a passing check), computed once
        // per design: each fix runs `compute_all` on the design its move reaches.
        let all = |inputs: &DesignInputs| all_fixes(inputs, &compute_all(inputs), false);
        let mut cache = FixCache::default();
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        for _ in 0..3 {
            assert_eq!(
                cache.get(&defaults, &results, false),
                all(&defaults).as_slice()
            );
        }
        assert_eq!(
            cache.computed, 1,
            "a design asked about again is not computed again"
        );
        let crowded = with_poles(12);
        let crowded_results = compute_all(&crowded);
        assert_eq!(
            cache.get(&crowded, &crowded_results, false),
            all(&crowded).as_slice()
        );
        assert_eq!(cache.computed, 2);
        let passing = passing_design();
        let passing_results = compute_all(&passing);
        assert!(
            cache
                .get(&passing, &passing_results, false)
                .iter()
                .all(Vec::is_empty)
        );
        assert_eq!(cache.computed, 3);
    }

    // ------------------------------------------------------------------ plain words
    // (The guard's own examples are in engine::explain::plain, where gate 4 runs them.)

    #[test]
    fn no_guide_text_carries_workbook_jargon() {
        for text in every_text() {
            assert_eq!(plain_text_violations(&text), Vec::<&str>::new(), "{text:?}");
        }
    }

    #[test]
    fn every_guide_text_has_glyphs_in_the_default_fonts() {
        let ctx = glyph_context();
        for text in every_text() {
            assert_glyphs(&ctx, &text, "guidance");
        }
    }

    #[test]
    fn no_guide_text_calls_the_ring_radius_ring_size() {
        // One name per quantity (spec 3.5): the input row says Ring radius, so no guide, fix,
        // problem or consequence line says "ring size", nor the torque card's Auto lines, which
        // took over the rule of the two Auto lines that stood under Magnets per ring (slice 3
        // spec 4.3).
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
        let mut texts = every_text();
        texts.extend(
            [
                RING_SIZE_STAYS,
                AUTO_HOLDS_LENGTH,
                AUTO_HOLDS_RING,
                AUTO_HOLDS_COUNT,
            ]
            .map(str::to_owned),
        );
        for text in texts {
            assert!(!text.to_lowercase().contains("ring size"), "{text:?}");
        }
    }

    #[test]
    fn no_guide_text_says_the_coupling_protects_the_gearbox() {
        // BL-059 and slice-2 spec section 8: the gearbox check's Problem, Why and fixes, and
        // every other guide text, say what the check compares; a pass is no protection.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        for text in every_text() {
            assert!(!mentions_gearbox_protection(&text), "{text:?}");
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
        // The passing design passes every guide; the defaults fail four, all red.
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
        assert_eq!(
            failing,
            [
                (HOT_MIN, Bad),
                (CLEARANCE, Bad),
                (CUP_WALL, Bad),
                (GEARBOX, Bad)
            ]
        );
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
            (only_the_gearbox_fails(), GEARBOX, Bad),
            (only_the_joint_fails(), SHAFT_JOINT, Bad),
            (breaking(no_back_iron), SHAFT_JOINT, Bad),
            (breaking(cool_adhesive), SHAFT_JOINT, Bad),
            (breaking(n42_parts), SHAFT_JOINT, Bad),
            (breaking(reversed_hub_band), SHAFT_JOINT, Caution),
            (
                breaking(|i| i.temperature.adhesive.selected = 0),
                SHAFT_JOINT,
                Caution,
            ),
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
                GEARBOX,
                DesignInputs::default(),
                design(|i| i.metal.min_temp_C = 0.0),
                &["metal.cold_high_input_Nm"],
            ),
            (
                SHAFT_JOINT,
                design(|i| i.shaft_joint.hub_interference_min_um = 10.0),
                design(|i| i.shaft_joint.hub_interference_min_um = 9.0),
                &[
                    "shaft_joint.hub_holding_Nm",
                    "shaft_joint.required_Nm",
                    "metal.torque_cold_high_Nm",
                ],
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
        // The shaft joints' window is worked out from the results: its lower end the smallest
        // minimum that holds, rounded up, its upper end the smaller of the two upper edges,
        // rounded down (spec 5.2), at two designs whose windows differ.
        let edits: [fn(&mut DesignInputs); 2] = [
            |i| i.shaft_joint.hub_interference_min_um = 10.0,
            |i| {
                i.shaft_joint.hub_interference_min_um = 10.0;
                i.shaft_joint.wheel_shaft_fit_C = -150.0;
            },
        ];
        let mut windows = Vec::new();
        for edit in edits {
            let inputs = design(edit);
            let results = compute_all(&inputs);
            let text = problem(SHAFT_JOINT, &inputs).expect("the inner hub slips");
            let s = &results.shaft_joint;
            let upper = s.hub_elastic_max_um.min(s.hub_fittable_max_um);
            let window = [
                number_text(rounded_up(s.hub_needed_um), "shaft_joint.hub_needed_um"),
                number_text(rounded_down(upper), "shaft_joint.hub_needed_um"),
            ];
            for quoted in &window {
                assert!(text.contains(quoted.as_str()), "{quoted} in {text}");
            }
            windows.push(window);
        }
        assert_ne!(windows[0], windows[1], "the two windows differ");
        // The gearbox Problem's sentence 2 (owner decision 4A) quotes numbers worked out from
        // the results: two designs with different bands (a warmer minimum temperature narrows
        // it) quote different numbers, each its own.
        let sentences: Vec<String> = [
            ("defaults", DesignInputs::default()),
            ("minimum -20 °C", design(|i| i.metal.min_temp_C = -20.0)),
        ]
        .iter()
        .map(|(name, inputs)| {
            let results = compute_all(inputs);
            let conflict = gearbox_conflict(inputs, &results).expect("the conflict holds");
            assert_conflict_quotes(name, inputs, &results, &conflict);
            conflict
        })
        .collect();
        assert_ne!(sentences[0], sentences[1]);
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
            "Radial running gap 0.6768 mm less 0.7800 mm of movement allowances leaves -0.1032 mm, below the 0.2000 mm target."
        );
        assert_eq!(
            problem(CUP_WALL, &defaults).unwrap(),
            "Cup wall at the pocket corners is 1.800 mm; the flux it carries needs 1.919 mm at the 1.500 T design flux density."
        );
        // The numbers follow an edit: a 2.4 N·m requirement still fails, quoted as such.
        let edited = design(|i| i.metal.required_min_Nm = 2.4);
        let text = problem(HOT_MIN, &edited).unwrap();
        assert!(text.contains("below the required 2.400 N·m"), "{text}");
        // A 0.1 mm liner: the gap and the result after the allowances move by its 0.1 mm.
        let thinner = design(|i| i.metal.liner_mm = 0.1);
        let text = problem(CLEARANCE, &thinner).unwrap();
        assert!(text.starts_with("Radial running gap 0.7768 mm"), "{text}");
        assert!(text.contains("leaves -0.003174 mm"), "{text}");
        // The gearbox check: cold-high at the gearbox input and the rating, each as its readout
        // shows it; the rating follows an edit.
        let results = compute_all(&defaults);
        let quoted = [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&defaults, "coupling.gearbox_input_rating_Nm"),
        ];
        assert_eq!(quoted, ["0.8049 N·m", "0.6000 N·m"]);
        let text = problem(GEARBOX, &defaults).unwrap();
        for number in &quoted {
            assert!(text.contains(number.as_str()), "{number} in {text}");
        }
        let rated = design(|i| i.coupling.gearbox_input_rating_Nm = 0.7);
        let text = problem(GEARBOX, &rated).unwrap();
        assert!(text.contains("0.7000 N·m"), "{text}");
    }

    #[test]
    fn sentence_1_never_reads_two_equal_numbers() {
        // Whole-branch review MF-1. Sentence 1 quotes cold-high at the gearbox input and the
        // rating as the readouts do, to four significant digits. Where cold-high is above the
        // rating by less than half a last digit both would read the same, so cold-high is quoted
        // one last digit past the rating, as the verdict and sentence 2 quote a torque on the
        // gearbox input. On the sliders, an 8.01 mm magnet length, a 1.00 mm face gap and a
        // 0.54 N·m rating put about 0.54004 N·m on the gearbox input.
        let mut close = DesignInputs::default();
        for (path, x) in [
            ("coupling.magnets.axial_length_mm", 8.01),
            ("metal.face_gap_mm", 1.0),
            ("coupling.gearbox_input_rating_Nm", 0.54),
        ] {
            assert!(
                slider_values(path).contains(&x),
                "{path}: {x} on a step of its slider"
            );
            close = with_number(&close, path, x).unwrap_or_else(|| panic!("{path}: {x}"));
        }
        let results = compute_all(&close);
        assert_eq!(level_of(GEARBOX, &close, &results), Level::Bad);
        // The readouts show the two the same.
        assert_eq!(
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&close, "coupling.gearbox_input_rating_Nm")
        );
        let text = problem(GEARBOX, &close).unwrap();
        assert!(
            text.starts_with(
                "Cold-high at the gearbox input is 0.5401 N·m, above its 0.5400 N·m input rating."
            ),
            "{text}"
        );
        // Where the two read apart, cold-high reads as its readout: the defaults are unchanged.
        let text = problem(GEARBOX, &DesignInputs::default()).unwrap();
        assert!(
            text.starts_with(
                "Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating."
            ),
            "{text}"
        );
    }

    #[test]
    fn off_the_sliders_sentence_1_can_read_two_equal_numbers_as_the_readme_says() {
        // Whole-branch review fix wave 2, m-1 (G-3 SF-2). Sentence 1 quotes cold-high one last
        // digit past the rating only for a positive torque: [`quoted_load`] quotes any other as
        // it is. On the sliders a failing check has cold-high above a rating of at least the
        // rating slider's bottom, so above zero. A design file can carry values off the
        // sliders: here a negative gearbox ratio, and a rating a hair below the negative
        // cold-high at the gearbox input that ratio gives, and the two read the same. So the
        // README says sentence 1 never reads two equal numbers on the sliders; a change that
        // quotes these apart too can drop that scope from the README and this test.
        let reversed = design(|i| i.coupling.gear_ratio = -5.0);
        let cold_high_input = compute_all(&reversed).metal.cold_high_input_Nm;
        let file = with(reversed, |i| {
            i.coupling.gearbox_input_rating_Nm = cold_high_input - 1e-7;
        });
        for (path, x) in [
            ("coupling.gear_ratio", file.coupling.gear_ratio),
            (
                "coupling.gearbox_input_rating_Nm",
                file.coupling.gearbox_input_rating_Nm,
            ),
        ] {
            assert!(x < slider_range(path).min, "{path}: {x} below its slider");
        }
        let results = compute_all(&file);
        assert_eq!(level_of(GEARBOX, &file, &results), Level::Bad);
        let text = problem(GEARBOX, &file).unwrap();
        assert!(
            text.starts_with(
                "Cold-high at the gearbox input is -0.8049 N·m, above its -0.8049 N·m input rating."
            ),
            "{text}"
        );
        let readme = include_str!("../../README.md");
        let claim =
            "The gearbox Problem's first sentence never reads two equal numbers on the sliders:";
        assert!(readme.contains(claim), "README.md should say {claim:?}");
    }

    // ------------------------------------------------------------------ the gearbox conflict

    /// Problem sentence 2's numbers (owner decision 4A), worked out here from a design's
    /// results and inputs as the sentence's own check: the band (cold-high over hot-low), the
    /// requirement, cold-high at the gearbox input of a design that just meets the
    /// requirement, what the requirement alone puts there, the largest requirement and the
    /// widest band the rating allows, whether no band is offered (the requirement alone above
    /// the rating) and whether the conflict holds (the minimum temperature at or below the
    /// operating one, hot-low above zero, and the requirement times the band, at the gearbox
    /// input, above the rating). Above the rating means beyond float noise ([`RATING_NOISE`],
    /// review NB-2): at the rating passes.
    struct ConflictNumbers {
        band: f64,
        required: f64,
        input_at_required: f64,
        alone: f64,
        largest_required: f64,
        widest_band: f64,
        no_band: bool,
        holds: bool,
    }

    impl ConflictNumbers {
        fn of(inputs: &DesignInputs, results: &DesignResults) -> Self {
            let (c, md) = (&inputs.coupling, &results.metal);
            let gear = c.gear_ratio * c.gear_efficiency;
            let rating = c.gearbox_input_rating_Nm;
            let above_rating = |torque: f64| torque > rating * (1.0 + RATING_NOISE);
            let band = md.torque_cold_high_Nm / md.torque_hot_low_Nm;
            let required = inputs.metal.required_min_Nm;
            let input_at_required = required * band / gear;
            let alone = required / gear;
            Self {
                band,
                required,
                input_at_required,
                alone,
                largest_required: rating * gear / band,
                widest_band: rating * gear / required,
                no_band: above_rating(alone),
                holds: inputs.metal.min_temp_C <= inputs.coupling.op_temp_C
                    && md.torque_hot_low_Nm > 0.0
                    && above_rating(input_at_required),
            }
        }

        /// Sentence 2's three numbers that it sets against another, in its order (review I-2):
        /// cold-high at the gearbox input of a design that just meets the requirement, against
        /// the rating; the largest requirement, against the requirement; and the widest band,
        /// against the band, or where no band is offered the requirement alone at the gearbox
        /// input, against the rating.
        fn bounds(&self, rating: f64) -> [Bound; 3] {
            let reads = |x: f64| format_value(&Value::Num(x));
            let load = |what, value| Bound {
                what,
                value,
                text: number_text(quoted_load(value, rating), "metal.cold_high_input_Nm"),
                against: reads(rating),
                below: false,
                floor: 0.0,
            };
            [
                load("cold-high at the input", self.input_at_required),
                Bound {
                    what: "largest requirement",
                    value: self.largest_required,
                    text: number_text(
                        quoted_limit(self.largest_required, self.required),
                        "metal.torque_hot_low_Nm",
                    ),
                    against: reads(self.required),
                    below: true,
                    floor: 0.0,
                },
                if self.no_band {
                    load("requirement alone at the input", self.alone)
                } else {
                    let (band, widest) = band_quotes(self.band, self.widest_band);
                    Bound {
                        what: "widest band",
                        value: self.widest_band,
                        text: widest,
                        against: band,
                        below: true,
                        // With the temperatures in order no band is below one (review NB-1).
                        floor: 1.0,
                    }
                },
            ]
        }

        /// Each number as sentence 2 quotes it (the readouts' format; its [`bounds`] rounded
        /// toward passing), in the sentence's order.
        fn quoted(&self, inputs: &DesignInputs) -> [String; 5] {
            let [input, largest, last] = self.bounds(inputs.coupling.gearbox_input_rating_Nm);
            // The band as the sentence quotes it: where it offers a band, as set against it.
            let band = if self.no_band {
                format_value(&Value::Num(self.band))
            } else {
                last.against.clone()
            };
            [
                band,
                input_text(inputs, "metal.required_min_Nm"),
                input.text,
                largest.text,
                last.text,
            ]
        }
    }

    /// A number sentence 2 sets against another ([`ConflictNumbers::bounds`]): its value, its
    /// text as the sentence quotes it (with its unit), the number it must read past as the
    /// sentence quotes that, below it (a limit) or above it (a torque on the gearbox input,
    /// against the rating), and the least it can read.
    struct Bound {
        what: &'static str,
        value: f64,
        text: String,
        against: String,
        below: bool,
        floor: f64,
    }

    /// Asserts that `conflict`, Problem sentence 2 of `inputs` (whose results are `results`),
    /// quotes its five numbers from the results in the sentence's order, and that each number it
    /// sets against another reads past it as quoted: no rounding makes it read against itself
    /// (review I-2). Each of those is also quoted on its passing side of its own value, within
    /// two of its last shown digits, and not below the least it can be: a requirement or band set
    /// to a quoted limit meets it, a band quoted is one a change can reach (review NB-1), and a
    /// gearbox rated for a quoted torque takes it.
    fn assert_conflict_quotes(
        name: &str,
        inputs: &DesignInputs,
        results: &DesignResults,
        conflict: &str,
    ) {
        let numbers = ConflictNumbers::of(inputs, results);
        let mut from = 0;
        for text in numbers.quoted(inputs) {
            let at = conflict[from..]
                .find(&text)
                .unwrap_or_else(|| panic!("{name}: {text} after byte {from} of {conflict}"));
            from += at + text.len();
        }
        for b in numbers.bounds(inputs.coupling.gearbox_input_rating_Nm) {
            let (shown, against) = (leading_number(&b.text), leading_number(&b.against));
            // How far the quoted value is from the value, in last shown digits: the coarser of
            // the two, as at a decade (0.99999 quoted 1.001 is one digit of 1.001 past 1.000).
            let step = last_digit_step(b.value).max(last_digit_step(shown));
            let digits = (shown - b.value) / step;
            let message = format!(
                "{name}: {} {} quoted {} against {}: {conflict}",
                b.what, b.value, b.text, b.against
            );
            assert!(shown >= b.floor, "{message}");
            if b.below {
                assert!(shown < against, "{message}");
                assert!(-2.0 < digits && digits <= 1e-6, "{message}");
            } else {
                assert!(shown > against, "{message}");
                assert!((-1e-6..2.0).contains(&digits), "{message}");
            }
        }
    }

    /// The two numbers gearbox Problem sentence 1 quotes, as they read: cold-high at the gearbox
    /// input and the rating.
    fn sentence_1_numbers(problem: &str) -> (f64, f64) {
        let between = |start: &str, end: &str| -> f64 {
            let from = problem
                .find(start)
                .unwrap_or_else(|| panic!("{start:?} in {problem}"))
                + start.len();
            let to = from
                + problem[from..]
                    .find(end)
                    .unwrap_or_else(|| panic!("{end:?} after {start:?} in {problem}"));
            problem[from..to]
                .parse()
                .unwrap_or_else(|_| panic!("a number between {start:?} and {end:?} in {problem}"))
        };
        (
            between("Cold-high at the gearbox input is ", " N·m, above its "),
            between("above its ", " N·m input rating."),
        )
    }

    /// Holds the gearbox Problem of `inputs` to owner decision 4A: it is present exactly when
    /// the check fails or asks for a look; where the check fails, sentence 1 quotes cold-high at
    /// the gearbox input above the rating as it reads, never two equal numbers (whole-branch
    /// review MF-1); sentence 2 ([`gearbox_conflict`]) exists exactly when the conflict holds by
    /// the test's own numbers ([`ConflictNumbers`]), quotes them ([`assert_conflict_quotes`]),
    /// names the requirement's side of the gearbox, and follows sentence 1 after one space
    /// exactly when the check fails as well; and a design that meets the requirement puts at
    /// least what sentence 2 says even just meeting it puts on the gearbox input (G-3 review
    /// M-1). Returns the check's level and whether sentence 2 is shown.
    fn assert_gearbox_problem_consistent(name: &str, inputs: &DesignInputs) -> (Level, bool) {
        let results = compute_all(inputs);
        let badge = level_of(GEARBOX, inputs, &results);
        let problem = gearbox_problem(inputs, &results);
        assert_eq!(
            problem.is_some(),
            badge != Level::Good,
            "{name}: {problem:?}"
        );
        if let (Level::Bad, Some(problem)) = (badge, &problem) {
            let (torque, rating) = sentence_1_numbers(problem);
            assert!(torque > rating, "{name}: {problem}");
        }
        let conflict = gearbox_conflict(inputs, &results);
        let numbers = ConflictNumbers::of(inputs, &results);
        assert_eq!(conflict.is_some(), numbers.holds, "{name}: {conflict:?}");
        let shown = match (&problem, &conflict) {
            (Some(problem), Some(conflict)) => problem.ends_with(&format!(" {conflict}")),
            _ => false,
        };
        assert_eq!(
            shown,
            badge == Level::Bad && numbers.holds,
            "{name}: {problem:?}"
        );
        if let Some(conflict) = &conflict {
            assert_conflict_quotes(name, inputs, &results, conflict);
            assert!(
                conflict.contains("requirement at the gearbox output"),
                "{name}: {conflict}"
            );
            // The band does not depend on the geometry, so a design with hot-low at or above the
            // requirement has cold-high, at the gearbox input too, at or above what just meeting
            // it puts there, up to float noise. The band here is this design's own, so this holds
            // by construction; tests/gearbox.rs
            // `the_band_is_the_same_at_every_magnet_size_gap_count_and_back_iron` pins the premise
            // across the geometry sliders (whole-branch review MF-3).
            if results.metal.torque_hot_low_Nm >= numbers.required {
                assert!(
                    results.metal.cold_high_input_Nm >= numbers.input_at_required * (1.0 - 1e-12),
                    "{name}: {} against {}",
                    results.metal.cold_high_input_Nm,
                    numbers.input_at_required
                );
            }
        }
        (badge, shown)
    }

    /// The Size tab's Auto on `variable` for `base`: the solve over the variable's slider (the
    /// panel's search range for every free variable but ring size) and the design it shows, as
    /// the solved value and that design.
    fn auto_on(variable: FreeVariable, base: &DesignInputs) -> (f64, DesignInputs) {
        match solve(base, variable, base.metal.required_min_Nm) {
            Ok(SizingOutcome::Solved(point)) => (
                point.value,
                crate::gui::sizing::sized(variable, base, point.value),
            ),
            other => panic!("Auto on {variable:?} does not solve: {other:?}"),
        }
    }

    /// [`auto_on`] the axial magnet length (ruling B-5): the solved length and the design.
    fn auto_on_length(base: &DesignInputs) -> (f64, DesignInputs) {
        auto_on(FreeVariable::AxialLength, base)
    }

    #[test]
    fn at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks() {
        // Owner decision 4A. Cold-high over hot-low does not depend on the geometry, so every
        // design that meets the default requirement puts the requirement times that band on
        // the gearbox output, above the rating at its input: sentence 2 follows sentence 1,
        // every number of both from the results and inputs.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        assert_eq!(
            assert_gearbox_problem_consistent("defaults", &inputs),
            (Level::Bad, true)
        );
        let problem = gearbox_problem(&inputs, &results).unwrap();
        let conflict = gearbox_conflict(&inputs, &results).unwrap();
        let first = problem.strip_suffix(&format!(" {conflict}")).unwrap();
        for text in [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
        ] {
            assert!(first.contains(&text), "{text} in {first}");
        }
        // Spec 7.6's numbers, typed as the readouts print them, so a formula error the oracle
        // shares with the code still shows: the band, the requirement, cold-high at the input
        // of a design that just meets it, the largest requirement and the widest band.
        for text in ["1.673", "2.500 N·m", "0.8806 N·m", "1.703 N·m", "1.140"] {
            assert!(conflict.contains(text), "{text} in {conflict}");
        }
        // G-3 review M-1: the requirement named at the gearbox output, and the torque on the
        // gearbox input what even just meeting it puts there (a design with room puts more).
        let load = "so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806 N·m on the gearbox input.";
        assert!(conflict.contains(load), "{conflict}");
        // The second remedy form, at a gearbox ratio of 1: the largest requirement and what the
        // requirement alone puts on the gearbox input. The largest requirement, 0.34069 N·m, is
        // quoted rounded down (review I-2): a requirement of 0.3407 N·m would not pass.
        let ratio_one = design(|i| i.coupling.gear_ratio = 1.0);
        let conflict = gearbox_conflict(&ratio_one, &compute_all(&ratio_one)).unwrap();
        for text in ["0.3406 N·m", "2.632 N·m"] {
            assert!(conflict.contains(text), "{text} in {conflict}");
        }
    }

    #[test]
    fn a_design_failing_the_gearbox_check_alone_has_no_conflict_sentence() {
        // The passing design with the rating back at its default fails the gearbox check
        // alone. Its requirement times the band, at the gearbox input, is under the rating, so
        // a design that just meets its requirement passes both checks: no sentence 2.
        let inputs = only_the_gearbox_fails();
        let results = compute_all(&inputs);
        assert_eq!(
            assert_gearbox_problem_consistent("gearbox_only", &inputs),
            (Level::Bad, false)
        );
        let problem = gearbox_problem(&inputs, &results).unwrap();
        for text in [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
        ] {
            assert!(problem.contains(&text), "{text} in {problem}");
        }
        // Not a word of the defaults' sentence 2 either: its lead, the words before its first
        // number, is nowhere in this Problem.
        let defaults = DesignInputs::default();
        let conflict = gearbox_conflict(&defaults, &compute_all(&defaults)).unwrap();
        let lead = conflict.split(|c: char| c.is_ascii_digit()).next().unwrap();
        assert!(!lead.is_empty() && !problem.contains(lead), "{problem}");
    }

    #[test]
    fn under_auto_on_length_the_gearbox_check_fails_exactly_when_the_conflict_holds() {
        // Spec 5 (Auto, owner decision 5A): Auto sizes for the hot minimum check alone, so where
        // the solve lands on the requirement (above the length slider's bottom it bisects to
        // it), a solved design's hot-low is the requirement and its cold-high at the gearbox
        // input the requirement times the band over ratio and efficiency: with the minimum
        // temperature at or below the operating one, the check fails exactly where the conflict
        // holds, with sentence 2, except at a tie, where that is the rating itself (the
        // requirement times the band is the rating times ratio times efficiency): there the
        // solve's last digits above the requirement fail the check without sentence 2
        // (`under_auto_on_length_…_only_at_a_tie`). With the operating temperature below the
        // minimum sentence 2 never shows, though the check can fail
        // (`an_operating_temperature_below_the_minimum_…`). Where Auto overshoots the
        // requirement the two need not agree either
        // (`where_auto_overshoots_the_requirement_…`). At the defaults the check fails with
        // sentence 2. With the minimum temperature at its slider's top and no variation
        // allowance the band narrows enough: Auto solves about 11.90 mm, the check passes and
        // there is no Problem.
        let narrow_band = design(|i| {
            i.metal.min_temp_C = slider_range("metal.min_temp_C").max;
            i.metal.variation = slider_range("metal.variation").min;
        });
        for (name, base, fails) in [
            ("defaults", DesignInputs::default(), true),
            ("narrow band", narrow_band, false),
        ] {
            let (length, inputs) = auto_on_length(&base);
            assert!(
                length > slider_range("coupling.magnets.axial_length_mm").min,
                "{name}: {length}"
            );
            let results = compute_all(&inputs);
            assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Good, "{name}");
            let (badge, shown) = assert_gearbox_problem_consistent(name, &inputs);
            assert_eq!((badge == Level::Bad, shown), (fails, fails), "{name}");
            if !fails {
                assert!((length - 11.902).abs() < 0.005, "{length}");
                assert_eq!(gearbox_problem(&inputs, &results), None);
            }
        }
    }

    #[test]
    fn where_auto_overshoots_the_requirement_the_gearbox_check_can_fail_without_the_conflict() {
        // Review I-3, why the README says the check can fail without sentence 2 where Auto
        // overshoots the requirement: the overshoot reaches the gearbox input too, and a rating
        // on its slider between what meeting the requirement puts on the gearbox input and what
        // the solved design puts there fails the check with no sentence 2. Auto on Magnets per
        // ring steps by two: with a 2 N·m requirement (the review's probe; Auto on the count
        // does not reach the defaults' 2.5 N·m) it solves 10 magnets. Auto on length stops at
        // its slider's bottom where that already meets the requirement: with no end effect and
        // the lowest requirement it does. Where the solve lands on the requirement instead, see
        // `under_auto_on_length_the_gearbox_check_fails_without_sentence_2_only_at_a_tie`.
        let length_bottom = slider_range("coupling.magnets.axial_length_mm").min;
        for (variable, base) in [
            (
                FreeVariable::MagnetsPerRing,
                design(|i| i.metal.required_min_Nm = 2.0),
            ),
            (
                FreeVariable::AxialLength,
                design(|i| {
                    i.coupling.c_end = slider_range("coupling.c_end").min;
                    i.metal.required_min_Nm = slider_range("metal.required_min_Nm").min;
                }),
            ),
        ] {
            let (value, solved) = auto_on(variable, &base);
            let name = format!("Auto on {variable:?} at {value}");
            let results = compute_all(&solved);
            if variable == FreeVariable::AxialLength {
                assert_eq!(value, length_bottom, "{name}");
            }
            // It overshoots the requirement.
            assert!(
                results.metal.torque_hot_low_Nm > base.metal.required_min_Nm * 1.01,
                "{name}"
            );
            let at_requirement = ConflictNumbers::of(&solved, &results).input_at_required;
            let at_solved = results.metal.cold_high_input_Nm;
            let rating = *slider_values("coupling.gearbox_input_rating_Nm")
                .iter()
                .find(|&&rating| at_requirement < rating && rating < at_solved)
                .unwrap_or_else(|| {
                    panic!("{name}: a rating between {at_requirement} and {at_solved}")
                });
            let inputs = with(solved, |i| i.coupling.gearbox_input_rating_Nm = rating);
            let results = compute_all(&inputs);
            assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Good, "{name}");
            assert_eq!(
                assert_gearbox_problem_consistent(&name, &inputs),
                (Level::Bad, false),
                "{name}"
            );
        }
    }

    #[test]
    fn under_auto_on_length_the_gearbox_check_fails_without_sentence_2_only_at_a_tie() {
        // Review I-3, the README's claim in general. Every Auto solve meets the requirement, so
        // where the conflict holds the solved design fails the check and sentence 2 shows. Where
        // Auto on length lands on the requirement (above its slider's bottom it bisects to it),
        // the check fails without sentence 2 only at a tie, where meeting the requirement puts
        // the rating itself on the gearbox input: the requirement times the band is the rating
        // times ratio times efficiency. A design exactly at the requirement then passes both
        // checks, so sentence 2 is held back ([`RATING_NOISE`]), but the solve stops within its
        // length tolerance above the requirement and the check, strict at the rating, fails.
        // The sliders reach a tie at a band of one and at bands above one that are ratios of
        // small whole numbers. At each band of one and two such bands above it, Auto on length
        // solves the tie whose requirement is nearest the defaults' (no typed numbers), and
        // over every rating on its slider the solved design fails the check with sentence 2
        // below the tie's rating, without it at that rating, and passes above it.
        let length_bottom = slider_range("coupling.magnets.axial_length_mm").min;
        let bases = bands_of_one()
            .map(|(name, base)| (name, base, false))
            .into_iter()
            .chain(short_fraction_bands().map(|(name, base)| (name, base, true)));
        for (base_name, base, above_one) in bases {
            let band = ConflictNumbers::of(&base, &compute_all(&base)).band;
            // A band of one is one up to float noise.
            assert_eq!(band > 1.0 + 1e-9, above_one, "{base_name}: {band}");
            let off =
                |i: &DesignInputs| (i.metal.required_min_Nm - base.metal.required_min_Nm).abs();
            let (tie_name, tie) = exact_products(&base, band)
                .into_iter()
                .min_by(|(_, a), (_, b)| off(a).total_cmp(&off(b)))
                .unwrap_or_else(|| panic!("{base_name}: a tie on the sliders"));
            let (length, solved) = auto_on_length(&tie);
            let name = format!("{base_name}, {tie_name}, Auto on length at {length}");
            // It lands on the requirement: its hot-low reads as the requirement, its last digits
            // above it.
            assert!(length > length_bottom, "{name}");
            let results = compute_all(&solved);
            let hot_low = results.metal.torque_hot_low_Nm;
            assert!(hot_low > solved.metal.required_min_Nm, "{name}: {hot_low}");
            assert_eq!(
                result_text(&results, "metal.torque_hot_low_Nm"),
                input_text(&solved, "metal.required_min_Nm"),
                "{name}"
            );
            assert_eq!(level_of(HOT_MIN, &solved, &results), Level::Good, "{name}");
            let at_tie = tie.coupling.gearbox_input_rating_Nm;
            for rating in slider_values("coupling.gearbox_input_rating_Nm") {
                let inputs = with(solved.clone(), |i| {
                    i.coupling.gearbox_input_rating_Nm = rating
                });
                let expected = if rating < at_tie {
                    (Level::Bad, true)
                } else if rating == at_tie {
                    (Level::Bad, false)
                } else {
                    (Level::Good, false)
                };
                let name = format!("{name}, rating {rating}");
                assert_eq!(
                    assert_gearbox_problem_consistent(&name, &inputs),
                    expected,
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks() {
        // Review focus (owner decision 4A): over every test design, and the defaults with the
        // gearbox ratio and efficiency at each end of their sliders, sentence 2 shows exactly
        // when the check fails and the conflict holds, and every limit it quotes is below
        // what it limits.
        let mut cases: Vec<(String, DesignInputs)> = designs()
            .into_iter()
            .map(|(name, inputs)| (name.to_owned(), inputs))
            .collect();
        cases.extend(gear_slider_ends());
        let (mut with_sentence, mut without_sentence) = (0, 0);
        // The designs that show sentence 2 while meeting the requirement with room.
        let mut with_room = Vec::new();
        for (name, inputs) in &cases {
            let (badge, shown) = assert_gearbox_problem_consistent(name, inputs);
            match (badge, shown) {
                (Level::Bad, true) => with_sentence += 1,
                (Level::Bad, false) => without_sentence += 1,
                _ => {}
            }
            if shown && compute_all(inputs).metal.torque_hot_low_Nm > inputs.metal.required_min_Nm {
                with_room.push(name.as_str());
            }
        }
        // Both forms of a failing check's Problem are reached.
        assert!(
            with_sentence > 0 && without_sentence > 0,
            "{with_sentence} with sentence 2, {without_sentence} without"
        );
        // G-3 review M-1: twelve poles meets the requirement with room and puts more on the
        // gearbox input than sentence 2 quotes, so the helper's "even just meeting" bound is
        // checked where it is not an equality.
        assert!(with_room.contains(&"twelve_poles"), "{with_room:?}");
    }

    #[test]
    fn sentence_2_rounds_its_limits_toward_passing_and_past_what_they_limit() {
        // Review I-2. A limit is quoted rounded down at its last shown digit and a torque on the
        // gearbox input rounded up, so a value set to the quoted number passes; float noise is
        // not a digit; where the quoted number would still read equal to the one it is set
        // against, it goes one digit further.
        let reads = |x: f64| format_value(&Value::Num(x));
        for (limit, limited, quoted) in [
            // The defaults' largest requirement: down is also the nearest.
            (1.703451, 2.5, "1.703"),
            // The ratio-1 design's: the nearest, 0.3407, would not pass.
            (0.340690, 2.5, "0.3406"),
            // 1.14 to the last digit, though not in binary.
            (0.6 * 4.75 / 2.5, 1.673, "1.140"),
            (0.99996, 2.0, "0.9999"),
            // Down reads 1.590, as the 1.59004 it limits does: one digit further.
            (1.59001, 1.59004, "1.589"),
            // A requirement can be below one; a band cannot ([`band_quotes`], below).
            (1.00001, 1.00004, "0.9990"),
        ] {
            assert_eq!(
                reads(quoted_limit(limit, limited)),
                quoted,
                "{limit} below {limited}"
            );
        }
        for (load, rating, quoted) in [
            (0.880564, 0.6, "0.8806"),
            (0.88051, 0.6, "0.8806"),
            (2.5 / 0.95, 0.6, "2.632"),
            (2.85 / 4.75, 0.5, "0.6000"),
            (0.99994, 0.5, "1.000"),
            // Up reads 0.5600, as the 0.55998 rating does: one digit further.
            (0.55999, 0.55998, "0.5601"),
            (0.99999, 0.99996, "1.001"),
        ] {
            assert_eq!(
                reads(quoted_load(load, rating)),
                quoted,
                "{load} above {rating}"
            );
        }
        // A number that is not finite or not positive (a value from a file) is quoted as it is.
        for x in [f64::INFINITY, 0.0, -0.5] {
            assert_eq!(reads(quoted_limit(x, 1.0)), reads(x));
            assert_eq!(reads(quoted_load(x, 1.0)), reads(x));
        }
        // The band and the widest band (review NB-1): the widest band is a limit, but no band is
        // below one, so one under one by float noise is one, and where both read 1.000 both get
        // more decimals, as few as tell them apart, the widest band never below one (at 13
        // decimals one rounded down is 0.9999999999999 in binary).
        let under_one = 1.0 - RATING_NOISE / 2.0;
        for (band, widest, quoted) in [
            (1.673, 1.14, ("1.673", "1.140")),
            (1.67303, 1.6730295, ("1.673", "1.672")),
            (1.673, under_one, ("1.673", "1.000")),
            (1.0012, 1.0, ("1.001", "1.000")),
            (1.0004, 1.0, ("1.0004", "1.0000")),
            (1.0004, under_one, ("1.0004", "1.0000")),
            (1.00004, 1.00001, ("1.00004", "1.00001")),
            (1.0000000000002, 1.0, ("1.0000000000002", "1.0000000000000")),
        ] {
            assert_eq!(
                band_quotes(band, widest),
                (quoted.0.to_owned(), quoted.1.to_owned()),
                "{widest} below {band}"
            );
        }
    }

    #[test]
    fn at_a_rounding_boundary_sentence_2_reads_each_limit_past_what_it_limits() {
        // Review I-2. Where the requirement times the band, at the gearbox input, is only just
        // above the rating, each number sentence 2 sets against another is only just past it,
        // and quoted to the nearest it reads equal to it ("a requirement of at most 1.590 N·m"
        // for a 1.590 N·m requirement). Two searches find such designs, with no typed numbers:
        // on the sliders, the defaults with each requirement on its slider and the highest
        // rating on its slider below what meeting that requirement puts on the gearbox input;
        // and at each gearbox ratio and efficiency slider end, a rating (from a file) a hair
        // below what the requirement alone puts there. Every such tie is found, and at each
        // design that shows one, sentence 2 reads each number past the other. The slider search
        // also runs at a band that reads 1.000 (review NB-1), where it finds the widest band
        // from one up to the band: no band below one is offered there.
        let ratings = slider_values("coupling.gearbox_input_rating_Nm");
        let mut boundary = Vec::new();
        for (base_name, base) in [
            ("defaults", DesignInputs::default()),
            ("band reading 1.000", band_reading_one()),
        ] {
            // The band depends on neither the requirement nor the rating.
            let results = compute_all(&base);
            for required in slider_values("metal.required_min_Nm") {
                let inputs = with(base.clone(), |i| i.metal.required_min_Nm = required);
                let target = ConflictNumbers::of(&inputs, &results).input_at_required;
                if let Some(&rating) = ratings.iter().rev().find(|&&rating| rating < target) {
                    let inputs = with(inputs, |i| i.coupling.gearbox_input_rating_Nm = rating);
                    let name = format!("{base_name}, required {required}, rating {rating}");
                    boundary.push((name, inputs));
                }
            }
        }
        for (name, inputs) in gear_slider_ends() {
            let numbers = ConflictNumbers::of(&inputs, &compute_all(&inputs));
            let rating = numbers.alone * (1.0 - 1e-9);
            let inputs = with(inputs, |i| i.coupling.gearbox_input_rating_Nm = rating);
            boundary.push((format!("{name}, rating {rating}"), inputs));
        }
        let reads = |x: f64| format_value(&Value::Num(x));
        let (mut ties, mut widest_under_a_band_reading_one) = (BTreeSet::new(), 0);
        for (name, inputs) in &boundary {
            let numbers = ConflictNumbers::of(inputs, &compute_all(inputs));
            assert!(numbers.holds, "{name}");
            // Tied: the number and the one it is set against read equal at the readouts' format.
            let tied: Vec<&str> = numbers
                .bounds(inputs.coupling.gearbox_input_rating_Nm)
                .iter()
                .filter(|b| reads(b.value) == reads(b.against.parse().unwrap()))
                .map(|b| b.what)
                .collect();
            if !numbers.no_band && reads(numbers.band) == reads(1.0) {
                widest_under_a_band_reading_one += 1;
            }
            if !tied.is_empty() {
                assert_gearbox_problem_consistent(name, inputs);
                ties.extend(tied);
            }
        }
        assert_eq!(
            ties,
            BTreeSet::from([
                "cold-high at the input",
                "largest requirement",
                "widest band",
                "requirement alone at the input",
            ])
        );
        assert!(widest_under_a_band_reading_one > 0);
    }

    #[test]
    fn at_a_band_of_one_meeting_the_requirement_exactly_at_the_rating_shows_no_sentence_2() {
        // Review NB-2. With no variation allowance and the minimum at the operating temperature,
        // or no Br temperature coefficient, the band is exactly one, so where the requirement is
        // exactly the rating times ratio times efficiency a design that just meets it puts
        // exactly the rating on the gearbox input, and at the rating passes: no sentence 2, so
        // no "no magnet size ... passes both" and no empty bracket. Float arithmetic puts the
        // band (1.0000000000000002 at -20 °C) or the requirement over ratio times efficiency an
        // ulp past it; the search finds designs where it does, at the default ratio and
        // efficiency and at an efficiency of one.
        let mut past_by_noise = 0;
        for (base_name, base) in bands_of_one() {
            for (name, inputs) in exact_products(&base, 1.0) {
                let name = format!("{base_name}, {name}");
                let results = compute_all(&inputs);
                let numbers = ConflictNumbers::of(&inputs, &results);
                if numbers.input_at_required > inputs.coupling.gearbox_input_rating_Nm {
                    past_by_noise += 1;
                }
                assert_eq!(band_levers(&inputs, &results), None, "{name}");
                assert_eq!(gearbox_conflict(&inputs, &results), None, "{name}");
                let (_, shown) = assert_gearbox_problem_consistent(&name, &inputs);
                assert!(!shown, "{name}");
            }
        }
        assert!(past_by_noise > 0);
    }

    #[test]
    fn where_the_requirement_alone_meets_the_rating_exactly_sentence_2_offers_a_band_of_one() {
        // Review NB-2, the same float noise in the choice of remedy. Where the requirement is
        // exactly the rating times ratio times efficiency, a band of one puts exactly the rating
        // on the gearbox input and passes, so the requirement alone is not above the rating and
        // a narrower band settles it: sentence 2 offers a band, of at most one, and never says
        // "no narrower band settles it", though float arithmetic puts the requirement over
        // ratio times efficiency an ulp above the rating at some of these designs.
        let mut above_by_noise = 0;
        let one = format!("a band of at most {} times", format_value(&Value::Num(1.0)));
        for (name, inputs) in exact_products(&DesignInputs::default(), 1.0) {
            let results = compute_all(&inputs);
            if ConflictNumbers::of(&inputs, &results).alone
                > inputs.coupling.gearbox_input_rating_Nm
            {
                above_by_noise += 1;
            }
            // Sentence 2 holds at every one of them (the band is 1.673); the Problem shows it
            // where the check fails too.
            assert_gearbox_problem_consistent(&name, &inputs);
            let conflict = gearbox_conflict(&inputs, &results)
                .unwrap_or_else(|| panic!("{name}: the conflict holds"));
            assert!(conflict.contains(&one), "{name}: {conflict}");
        }
        assert!(above_by_noise > 0);
    }

    #[test]
    fn where_nothing_narrows_the_band_sentence_2_offers_only_the_gearbox_or_the_requirement() {
        // Review NB-2's guard. The bracket names what narrows the band; where nothing does,
        // which no input on the sliders reaches (hand-edited results), sentence 2 offers no band
        // and no bracket, only the gearbox and the largest requirement, rounded down.
        let (inputs, results) = nothing_narrows_the_band();
        assert_eq!(band_levers(&inputs, &results), None);
        let numbers = ConflictNumbers::of(&inputs, &results);
        assert!(numbers.holds && !numbers.no_band);
        let problem = gearbox_problem(&inputs, &results).unwrap();
        let largest = number_text(
            quoted_limit(numbers.largest_required, numbers.required),
            "metal.torque_hot_low_Nm",
        );
        let remedy =
            format!("It takes a gearbox rated for that or a requirement of at most {largest}.");
        assert!(problem.ends_with(&format!(" {remedy}")), "{problem}");
        assert!(
            !problem.contains('(') && !problem.contains("band of"),
            "{problem}"
        );
    }

    #[test]
    fn an_operating_temperature_below_the_minimum_keeps_the_gearbox_problem_to_sentence_1() {
        // Review focus: the operating-temperature slider reaches below the minimum
        // temperature. There hot-low is taken colder than cold-high, the band can fall below
        // one and SmCo widens it, so sentence 2's levers would be false: the Problem is
        // sentence 1 alone wherever the check fails, and nothing panics.
        let coldest = slider_range("coupling.op_temp_C").min;
        let mut forms = BTreeSet::new();
        for minimum in [-20.0, slider_range("metal.min_temp_C").max] {
            for variation in [
                DesignInputs::default().metal.variation,
                slider_range("metal.variation").min,
            ] {
                let inputs = design(|i| {
                    i.coupling.op_temp_C = coldest;
                    i.metal.min_temp_C = minimum;
                    i.metal.variation = variation;
                });
                assert!(inputs.coupling.op_temp_C < inputs.metal.min_temp_C);
                let name = format!("minimum {minimum} °C, variation {variation}");
                forms.insert(assert_gearbox_problem_consistent(&name, &inputs));
                for required in [
                    inputs.metal.required_min_Nm,
                    slider_range("metal.required_min_Nm").max,
                ] {
                    let demanding = with(inputs.clone(), |i| i.metal.required_min_Nm = required);
                    let results = compute_all(&demanding);
                    assert_eq!(
                        gearbox_conflict(&demanding, &results),
                        None,
                        "{name}, {required}"
                    );
                    forms.insert(assert_gearbox_problem_consistent(&name, &demanding));
                }
            }
        }
        // Red in every case, never with sentence 2.
        assert_eq!(forms, BTreeSet::from([(Level::Bad, false)]));
    }

    #[test]
    fn sentence_2_names_only_what_narrows_that_design_s_band() {
        // Review I-1 (ruling R-3: sentence 2 is true wherever it shows). The band is
        // (1 + v) / (1 - v) times each ring's th(minimum) / th(operating), th(T) = 1 + α (T - 20 °C)
        // with α the ring's Br temperature coefficient. A warmer minimum narrows it only below the
        // operating temperature and with a coefficient below zero, SmCo magnets only there and
        // with every ring's coefficient below every SmCo grade's, a smaller variation allowance
        // only above zero. Over the Br coefficient's slider ends (its top, zero, leaves nothing
        // that changes with temperature), rings of a SmCo grade, one ring just below every SmCo
        // grade's coefficient beside one of a SmCo grade, no variation allowance and the minimum
        // at the operating temperature: sentence 2 names each change exactly where every design
        // one step of it away has a narrower band.
        let smco = smco_grades();
        let band = |inputs: &DesignInputs| ConflictNumbers::of(inputs, &compute_all(inputs)).band;
        for (name, inputs, named) in band_lever_designs() {
            let conflict = gearbox_conflict(&inputs, &compute_all(&inputs))
                .unwrap_or_else(|| panic!("{name}: the conflict holds"));
            let here = band(&inputs);
            let (minimum, operating) = (inputs.metal.min_temp_C, inputs.coupling.op_temp_C);
            let variation = inputs.metal.variation;
            // Each change and the designs one step of it away: the minimum one step warmer (not
            // past the operating temperature), the allowance one step smaller (not below zero),
            // both rings of each SmCo grade.
            let changes = [
                (
                    WARMER_MINIMUM,
                    vec![with(inputs.clone(), |i| {
                        i.metal.min_temp_C =
                            (minimum + slider_range("metal.min_temp_C").step).min(operating)
                    })],
                ),
                (
                    SMALLER_VARIATION,
                    vec![with(inputs.clone(), |i| {
                        i.metal.variation =
                            (variation - slider_range("metal.variation").step).max(0.0)
                    })],
                ),
                (
                    SMCO_MAGNETS,
                    smco.iter()
                        .map(|g| with(inputs.clone(), |i| graded(i, g.id)))
                        .collect(),
                ),
            ];
            for ((change, moved), named) in changes.into_iter().zip(named) {
                assert_eq!(
                    conflict.contains(change),
                    named,
                    "{name}: {change} in {conflict}"
                );
                let bands: Vec<f64> = moved.iter().map(band).collect();
                assert_eq!(
                    bands.iter().all(|&after| after < here * (1.0 - 1e-12)),
                    named,
                    "{name}: {change} moves the band from {here} to {bands:?}"
                );
            }
        }
    }

    #[test]
    fn sentence_2_names_the_variation_by_its_input() {
        // BL-059 G-3 round 1 S-2: the bracket names the input a user moves, the production
        // variation allowance ("a smaller measured variation" named none), and says the
        // smaller allowance needs measurement behind it.
        let input = label_for("metal.variation").to_lowercase();
        assert!(input.contains("variation allowance"), "{input}");
        for words in ["variation allowance", "measurement"] {
            assert!(
                SMALLER_VARIATION.contains(words),
                "{words:?} in {SMALLER_VARIATION:?}"
            );
        }
    }

    #[test]
    fn sentence_2_names_the_variation_allowance_last() {
        // BL-059 G-3 re-review of the review leftovers, M-1: an "or" after "backed by
        // measurement" would read as "backed by measurement or SmCo magnets", which is false
        // (SmCo narrows the band through its Br temperature coefficient, not by backing the
        // allowance). So the allowance is named last wherever it is named, and its verb is
        // "narrows". At the defaults, the first gearbox Problem a user sees, all three are named.
        let text = problem(GEARBOX, &DesignInputs::default()).unwrap();
        assert!(
            text.contains("(a warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it)"),
            "{text}"
        );
        for (name, inputs, _) in band_lever_designs() {
            let conflict = gearbox_conflict(&inputs, &compute_all(&inputs))
                .unwrap_or_else(|| panic!("{name}: the conflict holds"));
            assert!(
                !conflict.contains(&format!("{SMALLER_VARIATION} or ")),
                "{name}: {conflict}"
            );
            if conflict.contains(SMALLER_VARIATION) {
                assert!(
                    conflict.contains(&format!("{SMALLER_VARIATION} narrows it)")),
                    "{name}: {conflict}"
                );
            }
        }
    }

    #[test]
    fn the_temperature_problem_names_each_failing_term() {
        // The hot-day start against every limit it is not below, each ring's by name: each
        // ring needs its own fix. The peaks reach those limits too, which the start already
        // says (review 2, S1), and a quoted magnet limit says what it is (S2).
        assert_eq!(
            problem(TEMPERATURE, &design(hot_day)).unwrap(),
            "Hot-day magnet temperature 140.0 °C (hot ambient plus the driving rise) is at or above the inner magnets' 93.06 °C limit, the outer magnets' 93.06 °C limit and the 120.0 °C adhesive limit. Each ring's magnet limit is the temperature at which the field of a slip starts to demagnetize its magnets, less the 10.00 °C design margin (slip fields from the default design; enter this design's own to confirm)."
        );
        // Only the inner ring's library N42 block (rated 80 °C) is below the start.
        let text = problem(
            TEMPERATURE,
            &design(|i| i.coupling.magnets.part_inner = "B842".to_owned()),
        )
        .unwrap();
        assert!(
            text.starts_with(
                "Hot-day magnet temperature 65.00 °C (hot ambient plus the driving rise) is at or above the inner magnets' -3.517 °C limit. The inner magnets' limit is the temperature at which the field of a slip starts to demagnetize them, less the 10.00 °C design margin (slip fields from the default design; enter this design's own to confirm)."
            ),
            "{text}"
        );
        assert_eq!(
            problem(TEMPERATURE, &design(slip_fault)).unwrap(),
            "The slip-fault peak of 104.0 °C is at or above the 93.06 °C magnet limit. Each ring's magnet limit is the temperature at which the field of a slip starts to demagnetize its magnets, less the 10.00 °C design margin (slip fields from the default design; enter this design's own to confirm)."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(cool_adhesive)).unwrap(),
            "Hot-day magnet temperature 65.00 °C (hot ambient plus the driving rise) is at or above the 60.00 °C adhesive limit."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(hot_cure)).unwrap(),
            "The adhesive cures at 120.0 °C, 6.0 °C below the 126.0 °C at which a ring alone on its carrier, as during the cure, starts to demagnetize; the margin must be at least 10.0 °C."
        );
        assert_eq!(
            problem(TEMPERATURE, &design(ferrite)).unwrap(),
            "The minimum magnet temperature -40.00 °C is below the inner magnets' 335.7 °C cold demagnetization limit and the outer magnets' 335.7 °C cold demagnetization limit (from the default reverse-field inputs: 3D results for the default NdFeB design, not for this design's geometry or grade). Enter this design's own reverse fields to check the cold limit."
        );
        assert_eq!(problem(TEMPERATURE, &DesignInputs::default()), None);
    }

    #[test]
    fn the_cold_limit_names_its_source_only_while_the_reverse_fields_are_the_defaults() {
        // Spec section 8, trap 6: the default reverse fields are 3D results for the default
        // NdFeB design. Once a user enters this design's own, the caveat would be false.
        let caveat = "(from the default reverse-field inputs";
        let with_default_fields = problem(TEMPERATURE, &design(ferrite)).unwrap();
        assert!(
            with_default_fields.contains(caveat),
            "{with_default_fields}"
        );
        let edits: [fn(&mut DesignInputs); 4] = [
            |i| i.temperature.demag.h_rev_aligned_kA_m += 1.0,
            |i| i.temperature.demag.h_rev_pullout_kA_m += 1.0,
            |i| i.temperature.demag.h_rev_likepole_kA_m += 1.0,
            |i| i.temperature.demag.h_rev_single_ring_kA_m += 1.0,
        ];
        for edit in edits {
            let own = design(|i| {
                ferrite(i);
                edit(i);
            });
            assert!(!default_reverse_fields(&own));
            // A 1 kA/m step leaves the cold check failing; its sentence now ends at the limits,
            // with neither the caveat nor the request to enter the fields.
            let text = problem(TEMPERATURE, &own).expect("the cold check still fails");
            assert!(text.contains("The minimum magnet temperature"), "{text}");
            assert!(text.ends_with("cold demagnetization limit."), "{text}");
            assert!(!text.contains(caveat), "{text}");
            assert!(!text.contains("reverse fields"), "{text}");
        }
        assert!(default_reverse_fields(&DesignInputs::default()));
    }

    #[test]
    fn the_cure_margin_has_the_onset_s_decimals_so_the_printed_numbers_subtract() {
        let inputs = design(hot_cure);
        let results = compute_all(&inputs);
        // The minimum too (physics review 2, S3): "6.0 °C below ... at least 10.0 °C".
        let texts = |r: &DesignResults| cure_margin_texts(r);
        let pair = |a: &str, b: &str| (a.to_owned(), b.to_owned());
        assert_eq!(texts(&results), pair("6.0 °C", "10.0 °C"));
        // A margin that is not a number reads as the result does, and so does the minimum.
        let mut broken = results.clone();
        broken.temperature.summary.cure_margin_C = f64::NAN;
        assert_eq!(
            texts(&broken),
            (
                result_text(&broken, "temperature.summary.cure_margin_C"),
                "10.00 °C".to_owned()
            )
        );
        // An onset with no fixed decimals: both read as results do.
        let mut no_decimals = results.clone();
        no_decimals.temperature.demag.onset_single_ring_C = f64::INFINITY;
        assert_eq!(
            texts(&no_decimals),
            (
                result_text(&results, "temperature.summary.cure_margin_C"),
                "10.00 °C".to_owned()
            )
        );
        // A failing margin the onset's decimals would round up to the required 10 keeps its
        // own digits, so the sentence never reads "10.0 °C below ... at least 10.0 °C".
        let mut close = results.clone();
        close.temperature.summary.cure_margin_C = 9.96;
        assert_eq!(texts(&close), pair("9.960 °C", "10.00 °C"));
        // One that rounds down stays at the onset's decimals.
        close.temperature.summary.cure_margin_C = 9.94;
        assert_eq!(texts(&close), pair("9.9 °C", "10.0 °C"));
    }

    #[test]
    fn a_ring_limit_that_is_not_a_number_is_not_quoted_as_one() {
        // A ring's hot-day term fails on a limit that is not a number, so the Problem says
        // that limit cannot be checked rather than quote it, ring by ring.
        let inputs = design(hot_day);
        let temperature = guide(TEMPERATURE).unwrap();
        let mut outer_unknown = compute_all(&inputs);
        outer_unknown.temperature.demag.outer_magnet_limit_C = f64::NAN;
        assert_eq!(
            (temperature.problem)(&inputs, &outer_unknown).unwrap(),
            "Hot-day magnet temperature 140.0 °C (hot ambient plus the driving rise) is at or above the inner magnets' 93.06 °C limit and the 120.0 °C adhesive limit. The outer magnets' limit is not a number, so the hot-day magnet temperature cannot be checked against it. The inner magnets' limit is the temperature at which the field of a slip starts to demagnetize them, less the 10.00 °C design margin (slip fields from the default design; enter this design's own to confirm)."
        );
        let mut both_unknown = outer_unknown.clone();
        both_unknown.temperature.demag.inner_magnet_limit_C = f64::NAN;
        let text = (temperature.problem)(&inputs, &both_unknown).unwrap();
        assert!(
            text.starts_with(
                "Hot-day magnet temperature 140.0 °C (hot ambient plus the driving rise) is at or above the 120.0 °C adhesive limit. The inner magnets' limit is not a number, so the hot-day magnet temperature cannot be checked against it. The outer magnets' limit is not a number, so the hot-day magnet temperature cannot be checked against it."
            ),
            "{text}"
        );
        assert!(!text.contains("NaN"), "{text}");
    }

    #[test]
    fn the_no_fix_line_covers_a_claim_that_is_not_a_number() {
        // The space claim of a claim that is not a number fails with no fix: the Fix line
        // must name that case as well as the sliders' ends.
        let inputs = breaking(unknown_claim);
        let results = compute_all(&inputs);
        let space = guide(SPACE_CLAIM).unwrap();
        assert!(
            (space.problem)(&inputs, &results)
                .unwrap()
                .contains("is not a number")
        );
        assert_eq!(fixes(space, &inputs, &results, false), []);
        assert!(NO_FIX.contains("not a number"), "{NO_FIX}");
    }

    #[test]
    fn the_no_fix_line_covers_a_value_outside_what_it_can_be() {
        // The shaft joints' check cannot be made with a friction typed below zero (the
        // whole-branch review's MF-1), a safety factor not above zero (ruling C-4) or a gearbox
        // shaft typed below its slider's bottom (build ruling B-5); its Problem says so and it
        // has no fix, so the Fix line must name that case as well (the fix wave's text
        // re-review, MF-2).
        let joint = guide(SHAFT_JOINT).unwrap();
        type Edit = fn(&mut DesignInputs);
        let edits: [Edit; 3] = [
            |i| i.shaft_joint.friction = -0.14,
            |i| i.shaft_joint.safety_factor = 0.0,
            |i| i.shaft_joint.output_shaft_fit_C = -21.0,
        ];
        for edit in edits {
            let inputs = breaking(edit);
            let results = compute_all(&inputs);
            assert!((joint.problem)(&inputs, &results).is_some());
            assert_eq!(fixes(joint, &inputs, &results, false), []);
        }
        assert!(NO_FIX.contains("outside what it can be"), "{NO_FIX}");
    }

    #[test]
    fn the_outside_diameter_is_not_lowered_past_the_minimum_od() {
        // `fixes` steps the Outside diameter over its slider without the panel's Minimum OD;
        // the panel's raised floor stops that slider, so "Lower Outside diameter" goes there.
        // A raise of it (the fit's) and every other fix stay.
        let lower_ring = (RING, Direction::Lower);
        let raise_ring = (RING, Direction::Raise);
        for (inputs, check, fix) in [
            (breaking(large_ring), SPACE_CLAIM, lower_ring),
            (with_poles(12), FIT_CHECK, raise_ring),
        ] {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            assert!(all[guide_index(check)].contains(&fix), "{check}: {all:?}");
            let diameter = dimensions(&inputs).unwrap().diameter_mm;
            // No floor, or one below the diameter: the slider can still go down.
            for minimum in [None, Some(diameter - 1.0)] {
                assert_eq!(within_minimum_diameter(&all, &inputs, minimum), all);
            }
            // A floor at the diameter or above it.
            for minimum in [diameter, diameter + 1.0] {
                let mut want = all.clone();
                for list in &mut want {
                    list.retain(|&f| f != lower_ring);
                }
                assert_eq!(
                    within_minimum_diameter(&all, &inputs, Some(minimum)),
                    want,
                    "{check} at {minimum}"
                );
            }
        }
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
            let failing = failing_terms(&inputs, &results);
            assert_eq!(
                failing.is_empty(),
                passes,
                "{name}: {}",
                results.temperature.summary.verdict
            );
            assert_eq!(
                temperature_fixes(&inputs, &results, false).is_empty(),
                passes,
                "{name}"
            );
            // The peak is never below the hot-day start, so a start at a ring's magnet limit
            // always fails the magnet peak too; but no cooling can bring that peak under a
            // limit the start already reaches, so the cooling is not offered (review 2, N1).
            // Each peak whose limit the start reaches fails beside its hot-day term, whose
            // fixes clear both.
            let shown = temperature_fixes(&inputs, &results, false);
            let conductance = ("temperature.thermal.conductance_W_K", Direction::Raise);
            let trip = ("temperature.duty.fault_trip_s", Direction::Lower);
            for ring in Ring::BOTH {
                if failing.contains(&TemperatureTerm::HotDayMagnet(ring)) {
                    assert!(
                        failing.contains(&TemperatureTerm::MagnetPeak),
                        "{name}: {ring:?}"
                    );
                }
            }
            let reached = |term: TemperatureTerm| {
                failing.contains(&term) && term.start_reaches_peak_limit(&results)
            };
            if reached(TemperatureTerm::MagnetPeak) {
                let hot_rings = Ring::BOTH
                    .into_iter()
                    .filter(|&ring| failing.contains(&TemperatureTerm::HotDayMagnet(ring)));
                assert!(hot_rings.count() > 0, "{name}");
            }
            if reached(TemperatureTerm::AdhesivePeak) {
                assert!(failing.contains(&TemperatureTerm::HotDayAdhesive), "{name}");
            }
            let any_cooling = [TemperatureTerm::MagnetPeak, TemperatureTerm::AdhesivePeak]
                .into_iter()
                .any(|t| failing.contains(&t) && !t.start_reaches_peak_limit(&results));
            if !any_cooling {
                assert!(!shown.contains(&conductance), "{name}");
                assert!(!shown.contains(&trip), "{name}");
            }
        }
    }

    #[test]
    fn the_clamp_problem_names_the_screw_range_and_the_torque() {
        assert_eq!(
            problem(CLAMP, &design(small_boss)).unwrap(),
            "No screw size from M2.5 to M6 fits the boss and the clamp length with enough screws to hold 7.647 N·m: the highest torque the coupling can pass (cold, plus the 15 % variation allowance), 3.823 N·m, times a safety factor of 2.000."
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
            "Magnet overlap 2.000 mm long with a pole pitch of 8.809 mm gives an end-effect factor of -1.202; the model gives no usable torque until the overlap is longer than 4.405 mm and is rough just above that."
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
            "40 blocks per ring do not fit at this ring radius. Inner blocks are 6.350 mm wide on 1.598 mm flats. Outer blocks are 6.350 mm wide on 2.317 mm flats."
        );
        assert_eq!(
            problem(FIT_CHECK, &with(with_poles(40), arcs)).unwrap(),
            "40 blocks per ring do not fit at this ring radius. The 40 inner blocks together need 344.5 % of the circle at their mid-radius. The 40 outer blocks together need 247.9 % of the circle at their mid-radius."
        );
        // 12 flat blocks: the inner flats are too narrow, the outer ones are not.
        assert_eq!(
            problem(FIT_CHECK, &with_poles(12)).unwrap(),
            "12 blocks per ring do not fit at this ring radius. Inner blocks are 6.350 mm wide on 5.439 mm flats."
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
    fn a_failing_fit_problem_names_each_ring_that_does_not_fit() {
        // The fit rule is the engine's, ring by ring (`ring_fits`): the Problem's first sentence
        // says the blocks do not fit, and one more names each ring that fails the rule, also
        // when a width is not a number (the rule fails that ring, as it fails the design).
        let mut cases = designs();
        for faceted in [1, 0] {
            cases.push((
                "a width that is not a number",
                breaking(move |i| {
                    i.coupling.faceted = faceted;
                    i.coupling.magnets.part_inner.clear();
                    i.coupling.magnets.manual_inner_width_mm = f64::NAN;
                }),
            ));
        }
        for (name, inputs) in cases {
            let results = compute_all(&inputs);
            let failing_rings = ring_fits(&inputs.coupling, &results.model)
                .iter()
                .filter(|ring| !ring.fits())
                .count();
            match (guide(FIT_CHECK).unwrap().problem)(&inputs, &results) {
                Some(text) => {
                    assert!(failing_rings > 0, "{name}: a failing fit fails a ring");
                    let sentences = text.matches(". ").count() + 1;
                    assert_eq!(sentences, 1 + failing_rings, "{name}: {text}");
                }
                None => assert_eq!(failing_rings, 0, "{name}"),
            }
        }
    }

    #[test]
    fn the_gearbox_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made() {
        // Red above the rating; amber when a torque it compares is not a number (the rating,
        // or cold-high at the gearbox input through a ratio set on the struct), with the text
        // that says so; green at or under the rating and while the end-effect model is out of
        // range (the row is greyed).
        use Level::{Bad, Caution, Good};
        let gearbox = guide(GEARBOX).unwrap();
        for (name, inputs, want) in [
            ("defaults", DesignInputs::default(), Bad),
            ("passing", passing_design(), Good),
            ("gearbox only", only_the_gearbox_fails(), Bad),
            ("unknown rating", breaking(unknown_rating), Caution),
            (
                "unknown ratio",
                breaking(|i| i.coupling.gear_ratio = f64::NAN),
                Caution,
            ),
            ("short magnets", short_magnets(), Good),
            ("40 poles", with_poles(40), Good),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(level(gearbox, &inputs, &results), want, "{name}");
            let text = (gearbox.problem)(&inputs, &results);
            match want {
                Good => assert_eq!(text, None, "{name}"),
                Caution => assert_eq!(text.as_deref(), Some(GEARBOX_UNKNOWN), "{name}"),
                Bad => {
                    let text = text.unwrap_or_else(|| panic!("{name}: no Problem"));
                    for number in [
                        result_text(&results, "metal.cold_high_input_Nm"),
                        input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
                    ] {
                        assert!(text.contains(&number), "{name}: {number} in {text}");
                    }
                }
            }
        }
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

    #[test]
    fn every_shaft_joint_problem_form_is_reached() {
        // Spec 5.2 and 4.3: a design for each sentence form; each fails the check (red, or amber
        // for the forms of a joint that is not judged and of a value that is not a number), and
        // its Problem holds the form's words, never the fallback.
        for (name, inputs, words) in shaft_joint_problem_designs() {
            let results = compute_all(&inputs);
            assert_ne!(
                level_of(SHAFT_JOINT, &inputs, &results),
                Level::Good,
                "{name}"
            );
            let text = shaft_joint_problem(&inputs, &results)
                .unwrap_or_else(|| panic!("{name}: no Problem"));
            assert!(text.contains(words), "{name}: {words:?} in {text}");
            assert_ne!(text, SHAFT_JOINT_FALLBACK, "{name}");
        }
    }

    #[test]
    fn the_shaft_joint_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made() {
        // Red with a joint that fails; amber while a joint is not judged, or while the check
        // cannot be made (then that sentence alone, and no fix); green, with no Problem, while it
        // passes and while the end-effect model is out of range (the row is greyed, whatever its
        // text: a bore stress above its limit reads red there).
        use Level::{Bad, Caution, Good};
        let joint = guide(SHAFT_JOINT).unwrap();
        let not_a_number = breaking(|i| i.temperature.adhesive.selected = 0);
        let bore_0 = breaking(|i| i.coupling.bore_mm = 0.0);
        let no_factor = breaking(|i| i.shaft_joint.safety_factor = 0.0);
        let negative_friction = breaking(|i| i.shaft_joint.friction = -0.14);
        let frictionless = breaking(|i| i.shaft_joint.friction = 0.0);
        let overstressed_short = with(short_magnets(), hub_overstressed);
        for (name, inputs, want) in [
            ("defaults", DesignInputs::default(), Good),
            ("passing", passing_design(), Good),
            ("joint only", only_the_joint_fails(), Bad),
            ("not a number", not_a_number.clone(), Caution),
            ("bore 0", bore_0.clone(), Caution),
            ("safety factor 0", no_factor.clone(), Caution),
            ("friction below zero", negative_friction.clone(), Caution),
            ("friction 0", frictionless.clone(), Bad),
            ("reversed band", breaking(reversed_hub_band), Caution),
            ("short magnets", short_magnets(), Good),
            (
                "short magnets overstressed",
                overstressed_short.clone(),
                Good,
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(level(joint, &inputs, &results), want, "{name}");
            assert_eq!(
                (joint.problem)(&inputs, &results).is_some(),
                want != Good,
                "{name}"
            );
        }
        let results = compute_all(&overstressed_short);
        assert!(
            results
                .shaft_joint
                .check
                .starts_with(shaft_joint::BORE_OVER_ITS_LIMIT),
            "{}",
            results.shaft_joint.check
        );
        let results = compute_all(&not_a_number);
        assert_eq!(results.shaft_joint.check, shaft_joint::UNKNOWN_NOT_A_NUMBER);
        assert_eq!(
            shaft_joint_problem(&not_a_number, &results).as_deref(),
            Some(SHAFT_JOINT_NOT_A_NUMBER)
        );
        assert_eq!(fixes(joint, &not_a_number, &results, false), []);
        // A bore typed at 0 (review focus 1): every input and engine value the check reads is a
        // number; a term it works out is not (the holding: an infinite contact pressure per µm
        // times a contact arc of 0), so the sentence names both.
        let results = compute_all(&bore_0);
        let mut input_nan = false;
        bore_0.shaft_joint.visit("", &mut |_, _, value| {
            input_nan |= matches!(value, Value::Num(x) if x.is_nan());
        });
        assert!(!input_nan);
        let links = shaft_joint_links(&bore_0, &results);
        assert!(links.values().iter().all(|x| !x.is_nan()), "{links:?}");
        assert_eq!(results.shaft_joint.check, shaft_joint::UNKNOWN_NOT_A_NUMBER);
        let text = shaft_joint_problem(&bore_0, &results).unwrap();
        assert_eq!(text, SHAFT_JOINT_NOT_A_NUMBER);
        assert!(text.contains("a value it reads or works out"), "{text}");
        assert_eq!(fixes(joint, &bore_0, &results, false), []);
        // The cup's boss typed at or below zero (plan deviation PD-1): no wall, and the sentence
        // names the boss, never the web, which the check does not judge alone.
        for boss in [0.0, -22.0] {
            let inputs = breaking(|i| i.metal.boss_od_mm = boss);
            let results = compute_all(&inputs);
            assert_eq!(
                results.shaft_joint.check,
                format!("{} ({})", shaft_joint::UNKNOWN_NO_WALL, shaft_joint::CUP)
            );
            assert_eq!(level(joint, &inputs, &results), Caution, "{boss}");
            assert_eq!(
                shaft_joint_problem(&inputs, &results),
                Some(format!(
                    "The cup's bore, {}, is at least as wide as its boss, {}, so the boss leaves no wall to grip the gearbox shaft; the check does not judge the web alone.",
                    input_text(&inputs, "coupling.bore_mm"),
                    input_text(&inputs, "metal.boss_od_mm")
                ))
            );
            assert_eq!(fixes(joint, &inputs, &results, false), [], "{boss}");
        }
        // A safety factor typed at or below zero (ruling C-4): its sentence alone, and no fix.
        let results = compute_all(&no_factor);
        assert_eq!(
            results.shaft_joint.check,
            shaft_joint::UNKNOWN_SAFETY_FACTOR
        );
        assert_eq!(
            shaft_joint_problem(&no_factor, &results),
            Some(format!(
                "The safety factor, {}, is not above zero: enter at least 1.",
                input_text(&no_factor, "shaft_joint.safety_factor")
            ))
        );
        assert_eq!(fixes(joint, &no_factor, &results, false), []);
        // A friction typed below zero (the whole-branch review's MF-1, ruling C-4 extended): its
        // sentence alone, and no fix. Judged, it quoted a holding below zero, a window that no
        // interference gives, and raising either band's minimum, which holds less still.
        let results = compute_all(&negative_friction);
        assert_eq!(
            results.shaft_joint.check,
            shaft_joint::UNKNOWN_IMPOSSIBLE_VALUE
        );
        let text = shaft_joint_problem(&negative_friction, &results).unwrap();
        assert_eq!(
            text,
            "Shrink-fit friction coefficient, -0.1400, cannot physically be below zero: enter at least 0.05000, its slider's bottom."
        );
        for words in ["would hold", "holds", "measured interference"] {
            assert!(!text.contains(words), "{words:?} in {text}");
        }
        assert_eq!(fixes(joint, &negative_friction, &results, false), []);
        // A friction of exactly 0 is judged: each joint holds 0 N·m, which its sentence says;
        // no interference holds without friction, so no window shows and no fix helps.
        let results = compute_all(&frictionless);
        assert_eq!(
            (
                results.shaft_joint.hub_holding_Nm,
                results.shaft_joint.cup_holding_Nm
            ),
            (0.0, 0.0)
        );
        let text = shaft_joint_problem(&frictionless, &results).unwrap();
        assert_in_order(
            &text,
            &[
                "At its loosest the inner hub holds 0 N·m",
                "At its loosest the cup holds 0 N·m",
            ],
        );
        for words in ["would hold", "No measured interference works"] {
            assert!(!text.contains(words), "{words:?} in {text}");
        }
        assert_eq!(fixes(joint, &frictionless, &results, false), []);
    }

    #[test]
    fn each_value_below_what_its_input_can_physically_be_has_its_own_sentence() {
        // The whole-branch review's MF-1 (ruling C-4 extended): one sentence for each input
        // typed below what it can physically be, in the inputs' order, each with its label, its
        // value, its floor and its slider's bottom; nothing else, and no fix.
        let joint = guide(SHAFT_JOINT).unwrap();
        let cases = impossible_value_cases();
        for (edit, sentence) in cases {
            let inputs = breaking(edit);
            let results = compute_all(&inputs);
            assert_eq!(
                results.shaft_joint.check,
                shaft_joint::UNKNOWN_IMPOSSIBLE_VALUE,
                "{sentence}"
            );
            assert_eq!(
                level(joint, &inputs, &results),
                Level::Caution,
                "{sentence}"
            );
            assert_eq!(
                shaft_joint_problem(&inputs, &results).as_deref(),
                Some(sentence)
            );
            assert_eq!(fixes(joint, &inputs, &results, false), [], "{sentence}");
        }
        // Two at once, typed in the other order: both sentences, in the inputs' order.
        let inputs = breaking(two_impossible_values);
        let results = compute_all(&inputs);
        assert_eq!(
            shaft_joint_problem(&inputs, &results).as_deref(),
            Some(format!("{} {}", cases[0].1, cases[4].1).as_str())
        );
        assert_eq!(fixes(joint, &inputs, &results, false), []);
    }

    #[test]
    fn a_gearbox_shaft_below_its_slider_s_bottom_has_its_own_sentence() {
        // Build ruling B-5: the gearbox's rated range is not checked, so the check takes its
        // output shaft no colder than its slider's bottom. Typed below it, the check cannot be
        // made: one sentence that names the slider's bottom as the coldest the shaft is taken,
        // and no fix. At its slider's bottom the joint is judged.
        let joint = guide(SHAFT_JOINT).unwrap();
        let inputs = breaking(|i| i.shaft_joint.output_shaft_fit_C = -21.0);
        let results = compute_all(&inputs);
        assert_eq!(
            results.shaft_joint.check,
            shaft_joint::UNKNOWN_GEARBOX_SHAFT_TOO_COLD
        );
        assert_eq!(level(joint, &inputs, &results), Level::Caution);
        assert_eq!(
            shaft_joint_problem(&inputs, &results).as_deref(),
            Some(
                "Gearbox shaft temperature at fitting, -21.00 °C, is below -20.00 °C, the coldest the gearbox shaft is taken, since its rated range is not checked: enter at least -20.00 °C, its slider's bottom."
            )
        );
        assert_eq!(fixes(joint, &inputs, &results, false), []);
        let at_bottom = breaking(|i| {
            i.shaft_joint.output_shaft_fit_C = shaft_joint::OUTPUT_SHAFT_COLDEST_C;
        });
        let results = compute_all(&at_bottom);
        assert_eq!(results.shaft_joint.check, shaft_joint::SHAFT_JOINTS_PASS);
        assert_eq!(shaft_joint_problem(&at_bottom, &results), None);
    }

    #[test]
    fn a_gearbox_shaft_just_below_its_slider_s_bottom_is_quoted_below_it() {
        // G-3 re-review of the review leftovers, MF-1: typed in (-20.005, -20) °C, the gearbox
        // shaft reads "-20.00 °C" at the readout's four significant digits, the bottom it is
        // below. The Problem quotes it as the readout shows it where that reads below the
        // bottom, else with the fewest more decimals that do, rounded as the readouts round,
        // never onto the bottom.
        let coldest = shaft_joint::OUTPUT_SHAFT_COLDEST_C;
        let problem_at = |celsius: f64| {
            let inputs = breaking(|i| i.shaft_joint.output_shaft_fit_C = celsius);
            shaft_joint_problem(&inputs, &compute_all(&inputs)).expect("it cannot be made")
        };
        assert_eq!(
            problem_at(coldest - 0.001),
            "Gearbox shaft temperature at fitting, -20.001 °C, is below -20.00 °C, the coldest the gearbox shaft is taken, since its rated range is not checked: enter at least -20.00 °C, its slider's bottom."
        );
        // The value as the sentence quotes it: between the label's comma and its unit.
        let opening = format!("{}, ", fix_label(GEARBOX_SHAFT));
        let quoted = |problem: &str| {
            problem
                .strip_prefix(opening.as_str())
                .and_then(|rest| rest.split_once(" °C, is below "))
                .map(|(value, _)| value.to_owned())
                .unwrap_or_else(|| panic!("{problem}"))
        };
        // The review's probes, and two that the readout already reads below.
        for under in [0.001, 0.004, 0.005, 1e-9, 0.5, 1.0] {
            let celsius = coldest - under;
            let problem = problem_at(celsius);
            let text = quoted(&problem);
            let value: f64 = text.parse().unwrap_or_else(|_| panic!("{problem}"));
            assert!(value < coldest, "{problem}");
            let readout = format_value(&Value::Num(celsius));
            if quoted_value(celsius) < coldest {
                assert_eq!(text, readout, "{problem}");
            } else {
                // As few decimals as read it below: one fewer reads the bottom.
                let decimals = text.split_once('.').map_or(0, |(_, digits)| digits.len());
                assert_eq!(text, fixed(celsius, decimals), "{problem}");
                assert_eq!(
                    fixed(celsius, decimals - 1).parse::<f64>(),
                    Ok(coldest),
                    "{problem}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "has a slider")]
    fn the_slider_bottom_of_a_path_without_a_slider_panics() {
        // Re-review note 5: a sentence quotes a slider's bottom only for an input with a slider;
        // any other path is a mistake in the guide, never a "NaN" on screen.
        slider_bottom_text("shaft_joint.check");
    }

    /// Asserts that `text` holds each of `parts`, in their order.
    fn assert_in_order(text: &str, parts: &[&str]) {
        let mut rest = text;
        for part in parts {
            let at = rest
                .find(part)
                .unwrap_or_else(|| panic!("{part:?}, in order, in {text}"));
            rest = &rest[at + part.len()..];
        }
    }

    /// The number a quoted value starts with, its unit after a space dropped ("12.09 µm" reads
    /// 12.09).
    fn leading_number(text: &str) -> f64 {
        text.split(' ')
            .next()
            .and_then(|number| number.parse().ok())
            .unwrap_or_else(|| panic!("{text:?} starts with a number"))
    }

    #[test]
    fn the_shaft_joint_problem_quotes_the_live_numbers() {
        // Spec 5.2 and 7.8: each number as the readouts show it (four significant digits); the
        // window's ends quoted inward (review SF-11: the results 12.0815 and 18.8281 µm read
        // 12.09 and 18.82 µm), the edge that binds named; the source of the inner hub's allowed
        // fitting temperature named with its live number, the onset's margin written from
        // CURE_MARGIN_MIN_C with the allowed temperature's decimals (owner R-15).
        let margin = |decimals: usize| with_unit(fixed(CURE_MARGIN_MIN_C, decimals), "°C");
        let text = |inputs: DesignInputs| problem(SHAFT_JOINT, &inputs).expect("it fails");
        let hub_min_10 = text(design(|i| i.shaft_joint.hub_interference_min_um = 10.0));
        assert_in_order(
            &hub_min_10,
            &[
                "At its loosest the inner hub holds 3.482 N·m",
                "7.647 N·m",
                "cold-high, 3.823 N·m",
                "safety factor of 2.000",
                "On the inner hub, a measured interference from 12.09 µm to 18.82 µm",
                "fit at the allowed temperature",
                "(its band is 10.00 µm to 18.00 µm)",
            ],
        );
        // With a window there is no empty-window sentence (spec 7.8).
        assert!(
            !hub_min_10.contains("No measured interference works"),
            "{hub_min_10}"
        );
        // 304 back iron: no window on either joint, each edge below the need named.
        assert_in_order(
            &text(design(|i| i.materials.parts.back_iron = 7)),
            &[
                "No measured interference works on the inner hub here: holding needs at least 16.31 µm, but its bore stress stays within the limit only up to 11.72 µm.",
                "No measured interference works on the cup here: holding needs at least 15.80 µm, but its bore stress stays within the limit only up to 11.74 µm.",
            ],
        );
        // An uncooled wheel shaft (review MF-6, SF-12, review 2 SF-6b).
        assert_in_order(
            &text(design(uncooled_wheel_shaft)),
            &[
                "Fitting the inner hub would need it at 247.6 °C with the wheel shaft at 20.00 °C, above the 100.0 °C allowed: your limit for the bonded inner hub, 100.0 °C.",
                "holding needs at least 12.09 µm, but at its allowed temperature its bore does not even open the 10.00 µm joining clearance on the wheel shaft as set.",
            ],
        );
        // The cup's clearance form, its limit at its slider's bottom (G-3 SF-5; review B5 M-2).
        let bottom = |path: &str| slider_range(path).min;
        let cup_at_bottom = |i: &mut DesignInputs| {
            i.shaft_joint.cup_fit_max_C = bottom("shaft_joint.cup_fit_max_C");
        };
        assert_in_order(
            &text(design(cup_at_bottom)),
            &[
                "holding needs at least 11.31 µm, but at its allowed temperature its bore does not even open the 10.00 µm joining clearance on the gearbox shaft as set.",
            ],
        );
        // A fittable interference of exactly 0, each joint on its sliders (review B5 I-1): with
        // the joining clearance at its slider's bottom, 0 µm, the bore at its allowed temperature
        // on the shaft as set is line to line, which that clearance needs, so the fitting edge is
        // named, never the clearance.
        let no_clearance = |i: &mut DesignInputs| {
            i.shaft_joint.joining_clearance_um = bottom("shaft_joint.joining_clearance_um");
        };
        let cup_line_to_line = design(|i| {
            cup_at_bottom(i);
            no_clearance(i);
        });
        assert_eq!(
            compute_all(&cup_line_to_line)
                .shaft_joint
                .cup_fittable_max_um,
            0.0
        );
        assert_in_order(
            &text(cup_line_to_line),
            &["holding needs at least 11.31 µm, but the cup can be fitted only up to 0 µm."],
        );
        let hub_line_to_line = design(|i| {
            uncooled_wheel_shaft(i);
            i.shaft_joint.hub_fit_max_C = bottom("shaft_joint.hub_fit_max_C");
            no_clearance(i);
        });
        assert_eq!(
            compute_all(&hub_line_to_line)
                .shaft_joint
                .hub_fittable_max_um,
            0.0
        );
        assert_in_order(
            &text(hub_line_to_line),
            &["holding needs at least 12.09 µm, but the inner hub can be fitted only up to 0 µm."],
        );
        // DP460: the adhesive's design limit binds.
        assert_in_order(
            &text(breaking(cool_adhesive)),
            &[
                "would need it at 93.27 °C with the wheel shaft at -196.0 °C, above the 60.00 °C allowed: the selected adhesive's design limit, 60.00 °C.",
                "from 11.14 µm to 13.90 µm would hold and fit at the allowed temperature (its band is 13.00 µm to 18.00 µm)",
            ],
        );
        // Library N42 parts (owner R-15): the inner ring's onset less the margin binds.
        assert_in_order(
            &text(breaking(n42_parts)),
            &[
                &format!(
                    "above the 59.83 °C allowed: the magnets' single-ring onset, 69.83 °C, less {}.",
                    margin(2)
                ),
                "from 11.62 µm to 13.88 µm",
            ],
        );
        // N50M on manual dimensions: its onset is above the need, the onset less the margin
        // below it.
        assert_in_order(
            &text(breaking(|i| graded(i, "N50M"))),
            &[
                &format!(
                    "above the 88.80 °C allowed: the magnets' single-ring onset, 98.80 °C, less {}.",
                    margin(2)
                ),
                "from 12.21 µm to 17.45 µm",
            ],
        );
        // 2214 Hi-Temp (177 °C) and the hub's limit typed at 150 °C: the onset less the margin,
        // 133.4 °C, binds; the margin takes the allowed temperature's one decimal (review 4,
        // SF-5).
        assert_in_order(
            &text(breaking(|i| {
                i.temperature.adhesive.selected = 3;
                i.shaft_joint.hub_fit_max_C = 150.0;
                hub_overstressed(i);
            })),
            &[&format!(
                "above the 133.4 °C allowed: the magnets' single-ring onset, 143.4 °C, less {}.",
                margin(1)
            )],
        );
        // Mixed rings (owner R-15 ring ruling): the inner N42M's own onset, never the outer
        // N35's 59.93 °C.
        let mixed = text(breaking(mixed_rings));
        assert_in_order(
            &mixed,
            &[&format!(
                "above the 81.35 °C allowed: the magnets' single-ring onset, 91.35 °C, less {}.",
                margin(2)
            )],
        );
        assert!(!mixed.contains("59.93"), "{mixed}");
        // Spec 7.2 (review 2, SF-4): a key typed wider than the bore's circumference (40 mm on Ø10)
        // leaves no contact arc: each slip sentence says the joint holds 0, and no window sentence
        // follows (its smallest minimum is +inf).
        let wide = text(design(|i| i.shaft_joint.key_width_mm = 40.0));
        assert!(
            wide.starts_with("At its loosest the inner hub holds 0 N·m"),
            "{wide}"
        );
        assert!(!wide.contains("measured interference"), "{wide}");
        // No wall around a bore, on the sliders (spec 4.3). The cup's sentences name its boss,
        // whose diameter the check takes for the boss and the web alike, and say the web alone
        // is not judged; the inner hub's diameter is its own along its length.
        assert_in_order(
            &text(breaking(|i| {
                i.coupling.bore_mm = 16.0;
                i.metal.boss_od_mm = 12.0;
            })),
            &[
                "The cup's bore, 16.00 mm, is at least as wide as its boss, 12.00 mm, so the boss leaves no wall to grip the gearbox shaft; the check does not judge the web alone.",
            ],
        );
        assert_eq!(
            text(breaking(|i| i.metal.boss_od_mm = 12.0)),
            "The cup's keyway, 1.700 mm deep, cuts through the boss's wall around its 10.00 mm bore, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web alone."
        );
        assert_eq!(
            text(breaking(|i| {
                i.coupling.bore_mm = 16.0;
                i.coupling.keyway_depth_mm = 2.1;
            })),
            "The inner hub's keyway, 2.100 mm deep, cuts through the wall around its 16.00 mm bore, so the fit has no closed ring to grip the wheel shaft."
        );
        // Spec 7.1: at a 26 µm hub maximum the badge is the bore stress; the fitting cause follows it
        // in the Problem, then the window.
        assert_in_order(
            &text(design(hub_overstressed)),
            &[
                "At its tightest the inner hub's bore stress is 385.1 MPa",
                "Fitting the inner hub would need it at 158.3 °C",
                "(its band is 13.00 µm to 26.00 µm)",
            ],
        );
        // Review T5 5: the stress sentence's limit and yield, and the numbers of both no-length
        // key sentences, pinned, as `result_text` and `input_text` show a mistyped path as a
        // dash without failing.
        assert_in_order(
            &text(breaking(cup_overstressed)),
            &[
                "At its tightest the cup's bore stress is 384.0 MPa, above the 345.8 MPa limit (415.0 MPa yield with a margin).",
            ],
        );
        assert_in_order(
            &text(breaking(no_boss)),
            &[
                "The key has no bearing area in the cup: its boss and web, 0 mm and 2.500 mm long, together are no longer than the key is wide, 3.000 mm.",
            ],
        );
        assert_in_order(
            &text(breaking(short_hub)),
            &[
                "The key has no bearing area in the inner hub: the hub, 3.000 mm long, is no longer than the key is wide, 3.000 mm.",
            ],
        );
    }

    #[test]
    fn a_shaft_joint_window_typed_at_its_quoted_ends_passes() {
        // Review SF-11: the window is quoted inward, so a band typed at its quoted ends holds,
        // stays within the stress limit and fits. At the defaults: 12.09 to 18.82 µm on the
        // inner hub, 11.31 to 23.86 µm on the cup.
        let defaults = DesignInputs::default();
        let s = compute_all(&defaults).shaft_joint;
        let ends = |needed: f64, elastic: f64, fittable: f64| {
            let (low, high) =
                window_quotes(needed, elastic.min(fittable), "shaft_joint.hub_needed_um");
            (leading_number(&low), leading_number(&high))
        };
        let hub = ends(s.hub_needed_um, s.hub_elastic_max_um, s.hub_fittable_max_um);
        let cup = ends(s.cup_needed_um, s.cup_elastic_max_um, s.cup_fittable_max_um);
        assert_eq!((hub, cup), ((12.09, 18.82), (11.31, 23.86)));
        let typed = design(|i| {
            i.shaft_joint.hub_interference_min_um = hub.0;
            i.shaft_joint.hub_interference_max_um = hub.1;
            i.shaft_joint.cup_interference_min_um = cup.0;
            i.shaft_joint.cup_interference_max_um = cup.1;
        });
        assert_eq!(
            compute_all(&typed).shaft_joint.check,
            shaft_joint::SHAFT_JOINTS_PASS
        );
    }

    #[test]
    fn a_window_narrower_than_the_last_shown_digit_quotes_its_ends_with_more_decimals() {
        // BL-059 NB-1, spec 5.2: where the ends rounded inward at the readouts' digits would read
        // equal or crossed, both take more decimals, as few as read them apart in order.
        let unit = "shaft_joint.hub_needed_um";
        let quotes = |needed: f64, upper: f64| window_quotes(needed, upper, unit);
        assert_eq!(
            quotes(12.0815, 18.8281),
            ("12.09 µm".into(), "18.82 µm".into())
        );
        assert_eq!(
            quotes(12.0815, 12.0850),
            ("12.082 µm".into(), "12.085 µm".into())
        );
        assert_eq!(
            quotes(12.081512, 12.081588),
            ("12.08152 µm".into(), "12.08158 µm".into())
        );
        // Below 10 µm the readouts show three decimals: from four.
        assert_eq!(
            quotes(9.62012, 9.62088),
            ("9.6202 µm".into(), "9.6208 µm".into())
        );
    }

    #[test]
    fn a_failing_shaft_joint_number_never_reads_equal_to_its_bound() {
        // BL-059 MF-1, spec 5.2: a holding read equal to the need is quoted one last digit under
        // it, a stress, a fitting temperature or a key pressure read equal to its limit one last
        // digit over it. The results are set by hand: no design on the sliders lands there.
        let inputs = only_the_joint_fails();
        let mut results = compute_all(&inputs);
        let s = &mut results.shaft_joint;
        s.hub_holding_Nm = s.required_Nm - 1e-6;
        s.hub_bore_stress_MPa = s.stress_limit_MPa + 1e-6;
        s.hub_fit_temp_C = s.hub_fit_allowed_C + 1e-6;
        s.hub_key_pressure_MPa = s.stress_limit_MPa + 1e-6;
        let s = &results.shaft_joint;
        let read = |path: &str| result_text(&results, path);
        for (path, bound, above, bound_path) in [
            (
                "shaft_joint.hub_holding_Nm",
                s.required_Nm,
                false,
                "shaft_joint.required_Nm",
            ),
            (
                "shaft_joint.hub_bore_stress_MPa",
                s.stress_limit_MPa,
                true,
                "shaft_joint.stress_limit_MPa",
            ),
            (
                "shaft_joint.hub_fit_temp_C",
                s.hub_fit_allowed_C,
                true,
                "shaft_joint.hub_fit_allowed_C",
            ),
            (
                "shaft_joint.hub_key_pressure_MPa",
                s.stress_limit_MPa,
                true,
                "shaft_joint.stress_limit_MPa",
            ),
        ] {
            assert_eq!(
                read(path),
                read(bound_path),
                "{path}: the readouts read equal"
            );
            let quoted = apart_from(&results, path, bound, above);
            assert_ne!(quoted, read(bound_path), "{path}");
            let (q, b) = (leading_number(&quoted), leading_number(&read(bound_path)));
            assert!(
                if above { q > b } else { q < b },
                "{path}: {quoted} against {}",
                read(bound_path)
            );
        }
        // Apart already: as the readout shows it.
        assert_eq!(
            apart_from(
                &compute_all(&inputs),
                "shaft_joint.hub_holding_Nm",
                6.72,
                false
            ),
            result_text(&compute_all(&inputs), "shaft_joint.hub_holding_Nm")
        );
    }

    /// The shaft joints' texts are static where they can be, so they hold at every reachable
    /// input (review 2, MF-1; BL-067): cold-high "can reach" a joint, it is no bound, and with
    /// the operating temperature below the minimum (each on its slider) the operating pull-out
    /// with its allowance is above it. No text the check shows may call any torque the
    /// strongest or the most expected (`plain::TORQUE_ORDER_WORDS`).
    #[test]
    fn the_shaft_joint_texts_never_order_the_torques() {
        use crate::engine::explain::plain::TORQUE_ORDER_WORDS;
        let mut corner = DesignInputs::default();
        for (path, x) in [
            ("coupling.op_temp_C", -40.0),
            ("metal.min_temp_C", 20.0),
            ("metal.variation", 0.1),
        ] {
            let range = slider_range(path);
            assert!(
                range.min <= x && x <= range.max,
                "{path}: {x} on its slider"
            );
            corner = with_number(&corner, path, x).unwrap_or_else(|| panic!("{path}: {x}"));
        }
        let results = compute_all(&corner);
        let allowance = 1.0 + corner.metal.variation;
        assert!(
            results.model.pullout_Nm * allowance > results.metal.torque_cold_high_Nm,
            "the operating pull-out with its allowance, {}, above cold-high, {}",
            results.model.pullout_Nm * allowance,
            results.metal.torque_cold_high_Nm
        );
        let mut texts: Vec<String> = Vec::new();
        for row in shaft_joint_input_rows(&corner) {
            texts.extend([row.meta.label.to_owned(), row.meta.help.to_owned()]);
        }
        for row in shaft_joint_result_rows(&results) {
            texts.extend([row.meta.label.to_owned(), row.meta.help.to_owned()]);
        }
        texts.extend(
            [
                shaft_joint::SHAFT_JOINTS_PASS,
                shaft_joint::UNKNOWN_NOT_A_NUMBER,
                shaft_joint::UNKNOWN_NO_WALL,
                shaft_joint::UNKNOWN_BAND_REVERSED,
                shaft_joint::UNKNOWN_SAFETY_FACTOR,
                shaft_joint::UNKNOWN_IMPOSSIBLE_VALUE,
                shaft_joint::UNKNOWN_GEARBOX_SHAFT_TOO_COLD,
                shaft_joint::FIT_SLIPS,
                shaft_joint::BORE_OVER_ITS_LIMIT,
                shaft_joint::CANNOT_BE_FITTED,
                shaft_joint::KEY_OVER_ITS_LIMIT,
                shaft_joint::KEY_HAS_NO_BEARING_AREA,
                SHAFT_JOINT_GUIDE.label,
                SHAFT_JOINT_GUIDE.why,
                SHAFT_JOINT_NOT_A_NUMBER,
                SHAFT_JOINT_FALLBACK,
            ]
            .map(str::to_owned),
        );
        let corner_failing = with(corner, |i| i.shaft_joint.hub_interference_min_um = 8.0);
        let designs = designs()
            .into_iter()
            .map(|(_, inputs)| inputs)
            .chain(
                shaft_joint_problem_designs()
                    .into_iter()
                    .map(|(_, inputs, _)| inputs),
            )
            .chain([corner_failing]);
        for inputs in designs {
            texts.extend(problem(SHAFT_JOINT, &inputs));
        }
        assert!(texts.len() > 50, "{}", texts.len());
        for text in texts {
            let lower = text.to_lowercase();
            for word in TORQUE_ORDER_WORDS {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }
        }
    }

    // ------------------------------------------------------------------ the design status

    #[test]
    fn the_design_status_counts_the_failing_checks() {
        let status = |inputs: &DesignInputs| design_status(inputs, &compute_all(inputs), false);
        let defaults = status(&DesignInputs::default());
        assert_eq!(
            defaults,
            DesignStatus {
                bad: 4,
                caution: 0,
                total: 10,
                is_default: true
            }
        );
        assert_eq!(
            design_status_text(&defaults),
            "Example design: fails 4 of 10 checks"
        );
        // One fix: the cup wall at its suggested 2.0 mm.
        let fixed = status(&design(|i| i.metal.cup_wall_corner_mm = 2.0));
        assert_eq!(
            design_status_text(&fixed),
            "Your design: fails 3 of 10 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 10 checks"
        );
        // Cautions are counted apart from the failures.
        assert_eq!(
            design_status_text(&status(&design(hot_day))),
            "Your design: fails 4 of 10 checks, 1 caution"
        );
        let two_cautions = design(|i| {
            hot_day(i);
            unknown_claim(i);
        });
        assert_eq!(
            design_status_text(&status(&two_cautions)),
            "Your design: fails 4 of 10 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall, the shaft joints and the gearbox check.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 10 checks"
        );
        // Any edit leaves the example, even one that keeps the counts.
        let edited = status(&design(|i| i.metal.hardware_g = 7.0));
        assert_eq!((edited.bad, edited.is_default), (4, false));
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
        assert_eq!(text(0, 0, true), "Example design: passes all 8 checks");
        assert_eq!(
            text(0, 1, false),
            "Your design: passes all 8 checks, 1 caution"
        );
        assert_eq!(text(1, 0, false), "Your design: fails 1 of 8 checks");
        assert_eq!(
            text(8, 3, true),
            "Example design: fails 8 of 8 checks, 3 cautions"
        );
        assert!(text(1, 0, true).starts_with(EXAMPLE_DESIGN));
        assert!(text(1, 0, false).starts_with(YOUR_DESIGN));
    }

    // ------------------------------------------------------------------ the paused clamp section

    #[test]
    fn only_the_clamp_guide_stops_counting_and_only_while_the_section_is_paused() {
        for guide in CHECK_GUIDES.iter() {
            assert!(counts(guide, false), "{}", guide.check);
            assert_eq!(counts(guide, true), guide.check != CLAMP, "{}", guide.check);
        }
    }

    #[test]
    fn a_design_failing_only_the_clamp_passes_while_the_section_is_paused() {
        // Ruling 11: the clamp counts toward no verdict while it is paused.
        let inputs = only_the_clamp_fails();
        let results = compute_all(&inputs);
        assert_eq!(level(guide(CLAMP).unwrap(), &inputs, &results), Level::Bad);
        let shown = design_status(&inputs, &results, false);
        assert_eq!((shown.bad, shown.caution, shown.total), (1, 0, 10));
        assert_eq!(
            design_status_text(&shown),
            "Your design: fails 1 of 10 checks"
        );
        let paused = design_status(&inputs, &results, true);
        assert_eq!((paused.bad, paused.caution, paused.total), (0, 0, 9));
        assert_eq!(
            design_status_text(&paused),
            "Your design: passes all 9 checks"
        );
        // The example design fails the same four checks either way.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 4 of 9 checks"
        );
    }

    #[test]
    fn the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not() {
        // Owner decision 3A: counted, and red above the rating. The example design fails it
        // beside the hot minimum, the clearance and the cup wall; the passing design passes it,
        // and with the example's rating back fails it alone; a rating that is not a number asks
        // for a look. The clamp section's pause takes only the clamp out of the count.
        let status = |inputs: &DesignInputs, paused: bool| {
            design_status_text(&design_status(inputs, &compute_all(inputs), paused))
        };
        let defaults = DesignInputs::default();
        assert_eq!(
            status(&defaults, true),
            "Example design: fails 4 of 9 checks"
        );
        assert_eq!(
            status(&defaults, false),
            "Example design: fails 4 of 10 checks"
        );
        let passing = passing_design();
        assert_eq!(status(&passing, true), "Your design: passes all 9 checks");
        assert_eq!(status(&passing, false), "Your design: passes all 10 checks");
        let gearbox_only = only_the_gearbox_fails();
        let results = compute_all(&gearbox_only);
        let failing: Vec<(&str, Level)> = CHECK_GUIDES
            .iter()
            .map(|g| (g.check, level(g, &gearbox_only, &results)))
            .filter(|&(_, badge)| badge != Level::Good)
            .collect();
        assert_eq!(failing, [(GEARBOX, Level::Bad)]);
        assert_eq!(
            status(&gearbox_only, true),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&gearbox_only, false),
            "Your design: fails 1 of 10 checks"
        );
        assert_eq!(
            status(&breaking(unknown_rating), true),
            "Your design: passes all 9 checks, 1 caution"
        );
    }

    #[test]
    fn the_shaft_joint_check_counts_in_the_status_whether_the_clamp_section_shows_or_not() {
        // BL-064 (spec 7.7): counted, and outside the clamp section, so the pause takes only the
        // clamp out of the count. The example design passes it; with the passing design's inner
        // hub minimum at 8 µm it fails alone, red; a reversed band asks for a look; with the
        // clamp also failing, the clamp still moves no count while paused.
        let status = |inputs: &DesignInputs, paused: bool| {
            design_status_text(&design_status(inputs, &compute_all(inputs), paused))
        };
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(level_of(SHAFT_JOINT, &defaults, &results), Level::Good);
        let joint_only = only_the_joint_fails();
        let results = compute_all(&joint_only);
        let failing: Vec<(&str, Level)> = CHECK_GUIDES
            .iter()
            .map(|g| (g.check, level(g, &joint_only, &results)))
            .filter(|&(_, badge)| badge != Level::Good)
            .collect();
        assert_eq!(failing, [(SHAFT_JOINT, Level::Bad)]);
        assert_eq!(
            status(&joint_only, true),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&joint_only, false),
            "Your design: fails 1 of 10 checks"
        );
        let reversed = breaking(reversed_hub_band);
        assert_eq!(
            status(&reversed, true),
            "Your design: passes all 9 checks, 1 caution"
        );
        let with_the_clamp = with(only_the_joint_fails(), small_boss);
        assert_eq!(
            status(&with_the_clamp, true),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&with_the_clamp, false),
            "Your design: fails 2 of 10 checks"
        );
    }

    #[test]
    fn a_paused_clamp_offers_no_fix_and_holds_back_no_other_guide_s_fix() {
        let (clamp, hot) = (guide(CLAMP).unwrap(), guide(HOT_MIN).unwrap());
        let index = guide_index(CLAMP);
        // Its own fixes: shown while the section shows, none while it is paused.
        let inputs = only_the_clamp_fails();
        let results = compute_all(&inputs);
        let shown = fixes(clamp, &inputs, &results, false);
        assert!(!shown.is_empty(), "the clamp's fixes show with the section");
        assert_eq!(fixes(clamp, &inputs, &results, true), []);
        assert!(all_fixes(&inputs, &results, true)[index].is_empty());
        // At the clamp's limit one step of longer magnets fails the clamp: the hot-low guide's
        // one fix is held back while the clamp shows, and offered while it is paused.
        let at_limit = clamp_at_its_limit();
        let results = compute_all(&at_limit);
        assert_eq!(level(clamp, &at_limit, &results), Level::Good);
        assert_eq!(level(hot, &at_limit, &results), Level::Bad);
        assert_eq!(fixes(hot, &at_limit, &results, false), []);
        assert_eq!(
            fixes(hot, &at_limit, &results, true),
            [(MAGNET_LENGTH, Direction::Raise)]
        );
    }

    #[test]
    fn the_fix_cache_computes_again_when_the_pause_changes() {
        let inputs = only_the_clamp_fails();
        let results = compute_all(&inputs);
        let mut cache = FixCache::default();
        let index = guide_index(CLAMP);
        assert!(!cache.get(&inputs, &results, false)[index].is_empty());
        assert!(cache.get(&inputs, &results, true)[index].is_empty());
        assert_eq!(cache.computed, 2);
        cache.get(&inputs, &results, true);
        assert_eq!(
            cache.computed, 2,
            "the same inputs and pause: nothing computed"
        );
    }

    // What moves what (spec 3.3, test 5): every number is recomputed here with compute_all.

    /// The consequences of `inputs` as the panel builds them: with the design's own results, at
    /// the shipped shaft-clamp pause.
    fn consequences_of(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Consequences {
        Consequences::of(
            inputs,
            &compute_all(inputs),
            minimum_diameter_mm,
            crate::engine::clamps::PAUSED,
        )
    }

    #[test]
    fn the_fit_line_names_a_diameter_that_fits_and_one_step_below_does_not() {
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        assert_eq!(
            consequences.fit,
            smallest_fitting_diameter(&inputs, npole, None)
        );
        let Some(DiameterFit::At(smallest)) = consequences.fit else {
            panic!(
                "the default magnets fit above the slider's minimum: {:?}",
                consequences.fit
            )
        };
        let line = consequences
            .magnet_lines(npole, None)
            .pop()
            .expect("a fit line");
        let prefix = format!("Smallest outside diameter that fits {npole} magnets: ");
        let shown: f64 = line
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(" mm"))
            .and_then(|number| number.parse().ok())
            .unwrap_or_else(|| panic!("{line}"));
        // Rounded up to the tenth (less the rounding's 1e-9 mm slack).
        assert!(
            shown >= smallest - 1e-9 && shown - smallest < 0.1,
            "{shown} for {smallest}"
        );
        let fits = |diameter: f64| {
            let design = crate::engine::size_controls::resize_diameter(&inputs, diameter)
                .expect("inside the diameter slider");
            blocks_fit(&design.coupling, &compute_all(&design).model)
        };
        assert!(fits(shown), "{line}");
        assert!(!fits(shown - 0.1), "{line}: one 0.1 mm step below fits too");
    }

    #[test]
    fn the_fit_line_s_edge_forms_name_the_ends_of_the_diameter_range() {
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        // The fewest magnets fit at the slider's smallest diameter, with the slider's own floor
        // and with the lowest minimum preference.
        let inputs = with_magnets(&DesignInputs::default(), fewest);
        let size = crate::engine::size_controls::dimensions(&inputs).expect("a size");
        let lowest = size.minimum_diameter_range.min;
        let lowered =
            crate::engine::size_controls::dimensions_with_minimum_diameter(&inputs, lowest)
                .expect("a size at the lowest minimum");
        for (minimum, floor) in [
            (None, size.diameter_range.min),
            (Some(lowest), lowered.diameter_range.min),
        ] {
            let consequences = consequences_of(&inputs, minimum);
            assert_eq!(
                consequences.fit,
                smallest_fitting_diameter(&inputs, fewest, minimum)
            );
            assert_eq!(
                consequences.fit,
                Some(DiameterFit::AtMinimum(floor)),
                "{minimum:?}"
            );
            // Where the slider's own floor is the only thing in the way, the line says how far
            // it could go; at the lowest minimum nothing is in the way.
            let expected = if lowered.diameter_range.min < floor - FIT_TOLERANCE_MM {
                format!(
                    "Fits down to {:.1} mm; set Minimum OD lower to reach it.",
                    lowered.diameter_range.min
                )
            } else {
                format!("Fits at every diameter down to the slider's lowest, {floor:.1} mm")
            };
            assert_eq!(
                consequences.magnet_lines(fewest, None).last(),
                Some(&expected)
            );
        }
        // The most magnets fit nowhere on the slider.
        let inputs = with_magnets(&DesignInputs::default(), most);
        let top = crate::engine::size_controls::dimensions(&inputs)
            .expect("a size")
            .diameter_range
            .max;
        let consequences = consequences_of(&inputs, None);
        assert_eq!(consequences.fit, Some(DiameterFit::Nowhere(top)));
        assert_eq!(
            consequences.magnet_lines(most, None).last(),
            Some(&format!("Does not fit at any diameter up to {top:.1} mm"))
        );
    }

    #[test]
    fn with_minimum_od_unset_the_fit_line_says_how_far_it_could_go_and_what_unlocks_it() {
        // Review Focus 1, the owner's 8-magnet case: the slider's floor, not the blocks, stops
        // the diameter. The line must not say the coupling cannot get smaller.
        let default = DesignInputs::default();
        let fewer = default.coupling.npole - 2;
        let inputs = with_magnets(&default, fewer);
        let size = crate::engine::size_controls::dimensions(&inputs).expect("a size");
        let lowest = size.minimum_diameter_range.min;
        let user = smallest_fitting_diameter(&inputs, fewer, None);
        let absolute = smallest_fitting_diameter(&inputs, fewer, Some(lowest));
        let Some(DiameterFit::AtMinimum(floor)) = user else {
            panic!("8 magnets fit down to the slider's floor with no Minimum OD: {user:?}")
        };
        let Some(DiameterFit::At(reach)) = absolute else {
            panic!("8 magnets have an edge above the lowest minimum: {absolute:?}")
        };
        assert!(reach < floor - FIT_TOLERANCE_MM, "{reach} vs {floor}");
        let line = consequences_of(&inputs, None)
            .magnet_lines(fewer, None)
            .pop()
            .expect("a fit line");
        assert_eq!(
            line,
            format!(
                "Fits down to {} mm; set Minimum OD lower to reach it.",
                fixed(up_to_tenth(reach), 1)
            )
        );
        assert!(!line.contains("the slider's lowest"), "{line}");
        // With Minimum OD at its lowest, the edge is a plain smallest diameter.
        let unlocked = consequences_of(&inputs, Some(lowest))
            .magnet_lines(fewer, None)
            .pop()
            .expect("a fit line");
        assert_eq!(
            unlocked,
            format!(
                "Smallest outside diameter that fits {fewer} magnets: {}",
                with_unit(fixed(up_to_tenth(reach), 1), "mm")
            )
        );
    }

    #[test]
    fn a_fitting_diameter_rounds_up_to_the_tenth_and_a_whole_tenth_stays() {
        assert_eq!(up_to_tenth(40.43), 40.5);
        assert_eq!(up_to_tenth(40.5), 40.5);
        assert_eq!(up_to_tenth(40.4000001), 40.5);
        // Floating-point noise a hair above a whole tenth stays at that tenth.
        assert_eq!(up_to_tenth(37.300000000000004), 37.3);
    }

    #[test]
    fn the_diameter_and_length_lines_read_the_engine_breakdowns() {
        // The default (the cap is the largest) and the fewest magnets (the cup is).
        let default = DesignInputs::default();
        let fewest = FreeVariable::MagnetsPerRing.range().min as i64;
        for inputs in [default.clone(), with_magnets(&default, fewest)] {
            let consequences = consequences_of(&inputs, None);
            let diameter = diameter_breakdown(&inputs).expect("a diameter breakdown");
            let length = length_breakdown(&inputs).expect("a length breakdown");
            assert_eq!(consequences.diameter, Some(diameter));
            assert_eq!(consequences.length, Some(length));
            assert_eq!(
                consequences.diameter_line(),
                Some(format!(
                    "Set by: cap {:.1}, cup {:.1}, boss {:.1} mm",
                    diameter.cap_mm, diameter.cup_mm, diameter.boss_mm
                ))
            );
            assert_eq!(
                consequences.length_line(),
                Some(format!(
                    "Magnets {:.1} mm + housing {:.1} mm",
                    length.magnets_mm, length.housing_mm
                ))
            );
        }
    }

    #[test]
    fn at_8_magnets_the_ring_size_stays_and_the_cap_sets_the_diameter() {
        // Spec test 10, the owner's case (10 -> 8 magnets): the ring radius is unchanged and
        // the cap is the largest of the three, which is why the diameter did not shrink.
        let default = DesignInputs::default();
        let fewer = default.coupling.npole - 2;
        let inputs = with_magnets(&default, fewer);
        assert_eq!(
            inputs.coupling.inner_back_apothem_mm,
            default.coupling.inner_back_apothem_mm
        );
        let consequences = consequences_of(&inputs, None);
        assert_eq!(consequences.magnet_lines(fewer, None)[0], RING_SIZE_STAYS);
        let d = consequences.diameter.expect("a diameter breakdown");
        assert!(d.cap_mm > d.cup_mm && d.cap_mm > d.boss_mm, "{d:?}");
        assert_eq!(
            d.cap_mm,
            crate::engine::size_controls::dimensions(&inputs)
                .expect("a size")
                .diameter_mm
        );
        assert!(
            consequences
                .diameter_line()
                .expect("a line")
                .starts_with("Set by: cap "),
            "{:?}",
            consequences.diameter_line()
        );
    }

    #[test]
    fn without_a_breakdown_or_a_fit_there_is_no_line_for_it() {
        let empty = Consequences {
            card: TorqueCard::Rows(Vec::new()),
            fit: None,
            fit_unlocked: None,
            diameter: None,
            length: None,
        };
        assert_eq!(empty.magnet_lines(10, None), [RING_SIZE_STAYS]);
        assert_eq!(empty.diameter_line(), None);
        assert_eq!(empty.length_line(), None);
    }

    #[test]
    fn every_consequence_line_has_its_glyphs_and_plain_words() {
        // Spec tests 3 and 11, over every line form under the controls: the ring-size line, the
        // three fit forms and both breakdowns (the neighbour lines' torque and word forms are the
        // torque card's count row since slice 3: its own tests hold its lines).
        let ctx = glyph_context();
        let range = FreeVariable::MagnetsPerRing.range();
        let default = DesignInputs::default();
        let designs = [
            default.clone(),
            with_magnets(&default, range.min as i64),
            with_magnets(&default, default.coupling.npole - 2),
            with_magnets(&default, range.max as i64),
            crate::gui::test_support::short_magnets(),
            end_effect_edge(),
        ];
        for inputs in &designs {
            let consequences = consequences_of(inputs, None);
            let mut lines = consequences.magnet_lines(inputs.coupling.npole, None);
            lines.extend(consequences.diameter_line());
            lines.extend(consequences.length_line());
            for line in &lines {
                crate::gui::test_support::assert_glyphs(&ctx, line, "a consequence line");
                assert_eq!(plain_text_violations(line), Vec::<&str>::new(), "{line}");
            }
        }
    }

    #[test]
    fn the_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): the ring keeps its
        // radius with Auto off and with Auto on the magnet length (that Auto moves no ring), so
        // both draw the ring-size line. With Auto on the ring size a new count sizes the ring
        // again, which the torque card's Auto line says (decision 5A), so no line under Magnets
        // per ring speaks of it; with Auto on the count its slider is locked. The fit line
        // follows in every state, and no line names a neighbouring count (decision 7A: the card's
        // count row): 2 lines with the ring-size line, 1 without.
        let ctx = glyph_context();
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        let off = consequences.magnet_lines(npole, None);
        let states: [(Option<FreeVariable>, Option<&str>); 4] = [
            (None, Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::AxialLength), Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::RingRadius), None),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            assert_eq!(magnet_count_line(auto), first, "{auto:?}");
            let lines = consequences.magnet_lines(npole, auto);
            assert_eq!(
                lines.len(),
                1 + usize::from(first.is_some()),
                "{auto:?}: {lines:?}"
            );
            // The first line alone follows the state: the rest is the Auto-off fit line.
            match first {
                Some(first) => {
                    assert_eq!(lines[0], first, "{auto:?}");
                    assert_eq!(lines[1..], off[1..], "{auto:?}");
                }
                None => assert_eq!(lines[..], off[1..], "{auto:?}"),
            }
        }
        // The ring-size line's words (its clause is new in slice 3): glyphs and plain words.
        crate::gui::test_support::assert_glyphs(&ctx, RING_SIZE_STAYS, "a magnet count line");
        assert_eq!(plain_text_violations(RING_SIZE_STAYS), Vec::<&str>::new());
    }

    #[test]
    fn the_cache_computes_once_per_design_and_minimum_and_follows_an_edit() {
        // Spec 4.1 and test 5: computed again only when the inputs, the minimum or the
        // shaft-clamp pause change (slice 3 spec 3.2: one key in three parts); what it returns is
        // always Consequences::of the design asked about.
        use crate::gui::torque_card::{Control, Side};
        let paused = crate::engine::clamps::PAUSED;
        let mut cache = ConsequenceCache::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let first = cache.get(&inputs, &results, None, paused).clone();
        assert_eq!(first, Consequences::of(&inputs, &results, None, paused));
        for _ in 0..5 {
            assert_eq!(cache.get(&inputs, &results, None, paused), &first);
        }
        assert_eq!(
            cache.computed, 1,
            "a design asked about again is not computed again"
        );
        // An edit: the torque card's numbers follow it (its count row is the old neighbour
        // lines, in hot-low).
        let mut hot = inputs.clone();
        hot.coupling.op_temp_C += 30.0;
        let hot_results = compute_all(&hot);
        let edited = cache.get(&hot, &hot_results, None, paused).clone();
        assert_eq!(cache.computed, 2);
        assert_eq!(edited, Consequences::of(&hot, &hot_results, None, paused));
        assert_ne!(edited.card, first.card);
        let fewer = compute_all(&with_magnets(&hot, hot.coupling.npole - 2));
        let count = edited
            .card
            .rows()
            .iter()
            .find(|row| row.control == Control::MagnetsPerRing)
            .expect("the count row");
        assert!(
            matches!(
                &count.fewer,
                Side::Change(change) if change.hot_low_Nm == fewer.metal.torque_hot_low_Nm
            ),
            "{count:?}"
        );
        // A new minimum diameter is a new question too.
        let lowest = crate::engine::size_controls::dimensions(&hot)
            .expect("a size")
            .minimum_diameter_range
            .min;
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest), paused),
            &Consequences::of(&hot, &hot_results, Some(lowest), paused)
        );
        assert_eq!(cache.computed, 3);
        cache.get(&hot, &hot_results, Some(lowest), paused);
        assert_eq!(cache.computed, 3);
        // And so is the other pause: the card's flips and margins run over the guides that count
        // (an unpaused test panel counts the shaft clamp's).
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest), !paused),
            &Consequences::of(&hot, &hot_results, Some(lowest), !paused)
        );
        assert_eq!(cache.computed, 4);
        cache.get(&hot, &hot_results, Some(lowest), !paused);
        assert_eq!(cache.computed, 4);
        // The pause reaches the card: at the clamp's limit one step of length flips the shaft
        // clamp, which counts only while its section is not paused (spec 3.2, test 7a).
        let limit = clamp_at_its_limit();
        let limit_results = compute_all(&limit);
        assert_ne!(
            Consequences::of(&limit, &limit_results, None, false).card,
            Consequences::of(&limit, &limit_results, None, true).card
        );
    }

    #[test]
    fn cooling_is_offered_only_while_the_hot_day_start_is_below_the_peak_s_limit() {
        // Physics review 2, N1: the engine's slip-fault peak is the hot-day start plus heat,
        // never below it, so once the start reaches a limit no trip time or conductance can
        // bring the peak under that limit, though each step lowers the peak a little.
        let temperature = guide(TEMPERATURE).unwrap();
        let trip = ("temperature.duty.fault_trip_s", Direction::Lower);
        let conductance = ("temperature.thermal.conductance_W_K", Direction::Raise);
        let names = ["hot_day", "custom_hot_day", "n42_parts", "cool_adhesive"];
        let fixtures: Vec<(&str, DesignInputs)> = designs()
            .into_iter()
            .filter(|(name, _)| names.contains(name))
            .collect();
        assert_eq!(fixtures.len(), names.len());
        for (name, inputs) in fixtures {
            let results = compute_all(&inputs);
            let peaks: Vec<TemperatureTerm> = failing_terms(&inputs, &results)
                .into_iter()
                .filter(|t| t.peak_start_and_limit(&results).is_some())
                .collect();
            assert!(!peaks.is_empty(), "{name}: a peak fails");
            for peak in peaks {
                let (start, limit) = peak.peak_start_and_limit(&results).unwrap();
                assert!(start >= limit, "{name}: {peak:?} {start} < {limit}");
                assert!(peak.start_reaches_peak_limit(&results), "{name}");
            }
            // The engine's invariant the rule rests on.
            let t = &results.temperature;
            assert!(t.magnet_life.peak_C >= t.summary.hot_day_start_C, "{name}");
            assert!(
                t.adhesive_life.peak_C >= t.summary.hot_day_start_C,
                "{name}"
            );
            let shown = fixes(temperature, &inputs, &results, false);
            assert!(!shown.contains(&trip), "{name}: {shown:?}");
            assert!(!shown.contains(&conductance), "{name}: {shown:?}");
            assert!(!shown.is_empty(), "{name}: the hot-day fixes stay");
            // Nor does the Problem repeat a peak the hot-day sentence already covers (S1).
            let text = problem(TEMPERATURE, &inputs).unwrap();
            assert!(!text.contains("The slip-fault peak"), "{name}: {text}");
        }
        // A start below the magnet limit: the cooling clears the peak, and both are offered.
        let inputs = breaking(slip_fault);
        let results = compute_all(&inputs);
        let (start, limit) = TemperatureTerm::MagnetPeak
            .peak_start_and_limit(&results)
            .unwrap();
        assert!(start < limit);
        assert_eq!(
            fixes(temperature, &inputs, &results, false),
            [trip, conductance]
        );
        // A limit that is not a number is not "reached": its peak sentence stays.
        let mut unknown = results.clone();
        unknown.temperature.summary.magnet_limit_C = f64::NAN;
        assert!(!TemperatureTerm::MagnetPeak.start_reaches_peak_limit(&unknown));
        assert_eq!(TemperatureTerm::Cure.peak_start_and_limit(&results), None);
    }

    #[test]
    fn a_quoted_magnet_limit_says_what_it_is() {
        // Physics review 2, S2: an NdFeB magnet limit is the slip (like-pole) onset less the
        // design margin, the margin read from its input; a ferrite ring's is its rating.
        let both = problem(TEMPERATURE, &breaking(n42_parts)).unwrap();
        assert!(
            both.contains("Each ring's magnet limit is the temperature at which the field of a slip starts to demagnetize its magnets, less the 10.00 °C design margin (slip fields from the default design; enter this design's own to confirm)."),
            "{both}"
        );
        let inner_only = problem(
            TEMPERATURE,
            &design(|i| {
                i.coupling.magnets.part_inner = "B842".to_owned();
                i.temperature.demag.design_margin_C = 12.5;
            }),
        )
        .unwrap();
        assert!(
            inner_only.contains("The inner magnets' limit is the temperature at which the field of a slip starts to demagnetize them, less the 12.50 °C design margin (slip fields from the default design; enter this design's own to confirm)."),
            "{inner_only}"
        );
        // Physics review 3, S16: the onset comes from the stored slip fields. Once this
        // design's own are entered, the sentence no longer says where they came from.
        let own_fields = problem(
            TEMPERATURE,
            &breaking(|i| {
                n42_parts(i);
                i.temperature.demag.h_rev_aligned_kA_m += 1.0;
            }),
        )
        .unwrap();
        assert!(
            own_fields.contains("less the 10.00 °C design margin."),
            "{own_fields}"
        );
        assert!(!own_fields.contains("slip fields from"), "{own_fields}");
        // No magnet limit quoted, or only a ferrite rating's: no sentence.
        let quiet: [fn(&mut DesignInputs); 3] = [cool_adhesive, hot_cure, ferrite];
        for edit in quiet {
            let text = problem(TEMPERATURE, &breaking(edit)).unwrap();
            assert!(!text.contains("design margin"), "{text}");
        }
        // The peak sentence alone quotes the design's limit: its ring is named.
        let text = problem(TEMPERATURE, &breaking(slip_fault)).unwrap();
        assert!(text.contains("design margin"), "{text}");
    }

    #[test]
    fn raising_the_face_gap_says_it_lowers_the_failing_hot_low_torque() {
        // Physics review 2, N2: a larger face gap opens the running clearance but lowers the
        // pull-out. The thinner sleeve and liner come first, at no torque cost, and while the
        // gap fix is shown beside a hot-low torque that is not passing, a note says so.
        let note =
            "Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further.";
        let clearance = guide_index(CLEARANCE);
        let gap = (FACE_GAP, Direction::Raise);
        let notes_of = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            let all = all_fixes(inputs, &results, false);
            let notes = side_effects(&all, clearance, inputs, &results);
            (all[clearance].clone(), notes)
        };
        let defaults = DesignInputs::default();
        let (shown, notes) = notes_of(&defaults);
        assert_eq!(
            shown,
            [
                ("metal.sleeve_mm", Direction::Lower),
                ("metal.liner_mm", Direction::Lower),
                gap
            ]
        );
        assert_eq!(notes, [note]);
        // Only the gap is left to open, and the note still says what it costs.
        let (shown, notes) = notes_of(&design(bare_clearance));
        assert_eq!(shown, [gap]);
        assert_eq!(notes, [note]);
        // A clearance-only failure: the gap fix is shown, the hot-low torque passes, no note.
        let only = breaking(|i| i.metal.sleeve_mm += 0.5);
        let results = compute_all(&only);
        let failing: Vec<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| level(g, &only, &results) != Level::Good)
            .map(|g| g.check)
            .collect();
        assert_eq!(failing, [CLEARANCE]);
        let (shown, notes) = notes_of(&only);
        assert!(shown.contains(&gap), "{shown:?}");
        assert!(notes.is_empty(), "{notes:?}");
        // The gearbox guide's gap fix, shown at the defaults beside the failing hot-low torque,
        // brings the same note; no other guide gets it, and an index past the guides gets none.
        let gearbox = guide_index(GEARBOX);
        let results = compute_all(&defaults);
        let all = all_fixes(&defaults, &results, false);
        assert!(all[gearbox].contains(&gap), "{:?}", all[gearbox]);
        assert_eq!(side_effects(&all, gearbox, &defaults, &results), [note]);
        for index in (0..=CHECK_GUIDES.len()).filter(|&i| i != clearance && i != gearbox) {
            assert!(side_effects(&all, index, &defaults, &results).is_empty());
        }
    }

    #[test]
    fn thinning_the_sleeve_or_liner_says_no_check_covers_magnet_retention() {
        // Physics review 3, S14: the sleeve and liner fixes go down to 0 mm, which takes away
        // what holds the magnets, and no guided check covers retention. While either fix is
        // shown the note says so, once, after the side-effect note; with neither shown (no
        // sleeve or liner left to thin) it is gone.
        let retention = "A thinner sleeve or liner holds the magnets less securely; no check here covers that, so confirm it separately.";
        let gap_note =
            "Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further.";
        let clearance = guide_index(CLEARANCE);
        let notes_of = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            let all = all_fixes(inputs, &results, false);
            guide_notes(&all, clearance, inputs, &results)
        };
        assert_eq!(notes_of(&DesignInputs::default()), [gap_note, retention]);
        assert_eq!(notes_of(&design(bare_clearance)), [gap_note]);
        // The liner alone left to thin still brings it.
        let liner_only = design(|i| i.metal.sleeve_mm = 0.0);
        let results = compute_all(&liner_only);
        let shown = &all_fixes(&liner_only, &results, false)[clearance];
        assert!(
            !shown.contains(&("metal.sleeve_mm", Direction::Lower)),
            "{shown:?}"
        );
        assert!(
            shown.contains(&("metal.liner_mm", Direction::Lower)),
            "{shown:?}"
        );
        assert!(notes_of(&liner_only).contains(&retention.to_owned()));
        // Only the clearance guide shows those fixes, so no other guide gets the note, in any
        // design.
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            for index in (0..CHECK_GUIDES.len()).filter(|&i| i != clearance) {
                let notes = guide_notes(&all, index, &inputs, &results);
                assert!(!notes.contains(&retention.to_owned()), "{name}: {notes:?}");
            }
        }
        assert_eq!(unchecked_costs(&[]), Vec::<String>::new());
    }

    /// Prints every user-visible string of the guidance slice, each tagged with where it
    /// appears and for which design, for the physics review (plan Task 11, step 1). Run:
    /// `cargo test --features app --lib guidance_strings_for_review -- --ignored --nocapture`.
    #[test]
    #[ignore = "prints the strings for the physics review; run by hand"]
    fn guidance_strings_for_review() {
        use crate::gui::size_controls as sc;
        let mut count = 0_usize;
        let mut emit = |tag: &str, text: &str| {
            count += 1;
            println!("[{tag}] {text}");
        };

        // Each check guide: its Why, its link reason, and its fixes for every design that
        // shows them, rendered through fix_text.
        for g in &CHECK_GUIDES {
            emit(&format!("why/{}", g.label), g.why);
            if let Learn::None(reason) = g.learn {
                emit(&format!("learn/{}", g.label), reason);
            }
        }
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                for &(path, direction) in &all[index] {
                    emit(
                        &format!("fix/{}/{name}", g.label),
                        &fix_text(path, direction),
                    );
                }
                // The notes under the fixes: trade-offs and side effects (review 2, S9).
                for note in guide_notes(&all, index, &inputs, &results) {
                    emit(&format!("note/{}/{name}", g.label), &note);
                }
                if let Some(text) = (g.problem)(&inputs, &results) {
                    emit(&format!("problem/{}/{name}", g.label), &text);
                }
            }
        }
        emit("dashboard/no-fix line", NO_FIX);

        // The dashboard rows' link reasons.
        for (path, learn) in &ROW_LEARN {
            if let Learn::None(reason) = learn {
                emit(&format!("learn/row {path}"), reason);
            }
        }

        // What moves what, under Magnets per ring, Outside diameter and Overall length.
        // The first line under Magnets per ring, in each Auto state (physics review 3, N4).
        for auto in auto_states() {
            if let Some(line) = magnet_count_line(auto) {
                emit(
                    &format!("size tab/under Magnets per ring/Auto {auto:?}"),
                    line,
                );
            }
        }
        let default = DesignInputs::default();
        for npole in [4, 8, 10, 40] {
            let inputs = with_magnets(&default, npole);
            for (label, minimum) in [("Minimum OD unset", None), ("Minimum OD 37 mm", Some(37.0))] {
                let consequences = consequences_of(&inputs, minimum);
                for line in consequences.magnet_lines(inputs.coupling.npole, None) {
                    emit(&format!("size tab/magnets/{npole} magnets/{label}"), &line);
                }
                if let Some(line) = consequences.diameter_line() {
                    emit(&format!("size tab/diameter/{npole} magnets/{label}"), &line);
                }
                if let Some(line) = consequences.length_line() {
                    emit(&format!("size tab/length/{npole} magnets/{label}"), &line);
                }
            }
        }

        // The torque card (slice 3; spec sections 4 and 10.1 step 2), for its G-3 review: the
        // heading, then the lines `TorqueCard::lines` draws at every design of `card_designs()`,
        // each in its own Auto state as the panel draws it, at the shipped clamp pause; then every
        // Auto state at the five designs whose card takes a form of its own under Auto (the odd
        // count has no size rows, so Auto on ring size or magnet length holds no row; the four
        // one-line forms draw no Auto line: T-12, test 8; the count below the slider's range,
        // the owner's FO-3, by the table's count overflow at i64::MIN). Tag:
        // "card/<key>/<Auto state>/<line>", the line numbered from 00.
        {
            use crate::gui::test_support::card_designs;
            use crate::gui::torque_card::CARD_HEADING;
            let state = |auto: Option<FreeVariable>| {
                auto.map_or_else(|| "Auto off".to_owned(), |v| format!("Auto {v:?}"))
            };
            emit("card/heading", CARD_HEADING);
            // Each design's card as the panel asks for it (`CardDesign::card`).
            let cards: Vec<_> = card_designs()
                .into_iter()
                .map(|d| {
                    let shown = d.card();
                    (d, shown)
                })
                .collect();
            for (d, shown) in &cards {
                for (n, line) in shown.lines(d.auto).iter().enumerate() {
                    emit(&format!("card/{}/{}/{n:02}", d.key, state(d.auto)), line);
                }
            }
            let design = |key: &str| {
                cards
                    .iter()
                    .find(|(d, _)| d.key == key)
                    .unwrap_or_else(|| panic!("card_designs() has no design {key:?}"))
            };
            for key in [
                "odd count",
                "count overflow: i64::MIN",
                "short magnets",
                "shown does not fit",
                "variation 1.0",
            ] {
                let (_, shown) = design(key);
                for variable in FreeVariable::ALL {
                    for (n, line) in shown.lines(Some(variable)).iter().enumerate() {
                        emit(
                            &format!("card/{key}/{}/{n:02}", state(Some(variable))),
                            line,
                        );
                    }
                }
            }
            // The readout's gap above the card, in the card's number format (decision 10B),
            // for the review to read in context.
            for key in ["(a)", "(b)", "(c)", "(e)", "(e2)"] {
                let (d, _) = design(key);
                let results = compute_all(&d.inputs);
                let (gap, _) = sc::torque_gap_text(
                    results.metal.torque_hot_low_Nm,
                    d.inputs.metal.required_min_Nm,
                );
                emit(&format!("size tab/torque gap/{key}"), &gap);
            }
        }

        // The Size tab constants.
        emit("size tab/no prediction", sc::NO_PREDICTION);
        emit("size tab/no torque", sc::NO_TORQUE);
        emit("size tab/locked slider", sc::SET_BY_AUTO);
        emit("size tab/Auto label", sc::AUTO);
        emit("size tab/Auto suffix", sc::AUTO_SUFFIX);
        emit("size tab/Auto hover", sc::AUTO_HOVER);
        for variable in FreeVariable::ALL {
            emit(
                &format!("size tab/Auto choice/{variable:?}"),
                crate::gui::sizing::auto_choice(variable),
            );
        }
        emit("size tab/Required label", sc::REQUIRED);
        emit("size tab/Hot-low label", sc::HOT_LOW);
        emit("size tab/Outside diameter hover", sc::DIAMETER_HOVER);
        emit("size tab/Overall length hover", sc::LENGTH_HOVER);
        emit("size tab/footnote", sc::LINKED_HOUSING);
        {
            use crate::gui::plots as pl;
            for (tag, text) in [
                ("pull-out", pl::PULL_OUT),
                ("torque-rotation curve", pl::TORQUE_CURVE),
                ("pole sweep line", pl::PULL_OUT_SMALLEST_APOTHEM),
                ("low band", pl::LOW_BAND),
                ("high band", pl::HIGH_BAND),
                ("required", pl::REQUIRED),
                ("below the floor", pl::BELOW_MINIMUM),
                ("sweep y axis", pl::SWEEP_Y_AXIS),
            ] {
                emit(&format!("plots/{tag}"), text);
            }
        }
        emit(
            "assumptions/production variation",
            ASSUMPTIONS
                .iter()
                .find(|a| a.id == "production_variation")
                .unwrap()
                .label,
        );
        emit("panel/welcome strip", crate::gui::panel::WELCOME);

        // The solved change of each free variable, from the default to each end of its slider,
        // built as the panel builds the design shown (the ring radius moves the cap with the
        // cup: review 2, S9).
        for variable in FreeVariable::ALL {
            let range = variable.range();
            for (end, value) in [("min", range.min), ("max", range.max)] {
                let to = crate::gui::sizing::sized(variable, &default, value);
                emit(
                    &format!("size tab/solved change/{variable:?}/to slider {end}"),
                    &sc::change_text(variable, &default, &to),
                );
            }
        }

        // The design status line.
        for (name, inputs) in [("defaults", default.clone()), ("passing", passing_design())] {
            let results = compute_all(&inputs);
            emit(
                &format!("dashboard/status/{name}"),
                &design_status_text(&design_status(&inputs, &results, false)),
            );
        }

        emit(
            "file load/target mismatch notice",
            &crate::gui::session::target_mismatch_notice(3.0, 2.5),
        );
        emit(
            "size tab/requirement refusal",
            crate::gui::sizing::REQUIREMENT_NOT_POSITIVE,
        );

        // Reachability (slice 2, spec 3.7): the Equation panel's own lines, the Why links of the
        // Size tab and the banner, and what Explain offers each dashboard row without a note
        // (a row the paused shaft-clamp section hides is not drawn: none of its offers).
        use crate::gui::explorer as ex;
        for (tag, text) in [
            ("explorer/empty text", ex::EMPTY_TEXT),
            ("explorer/no note", ex::NO_NOTE),
            ("explorer/start here hint", ex::START_HERE_HINT),
            ("explorer/closest note lead", ex::NEAREST_NOTE),
            ("explorer/assumption lead", ex::NEAREST_ASSUMPTION),
        ] {
            emit(tag, text);
        }
        for (place, id) in [
            (
                "size tab/under Magnets per ring and Outside diameter",
                SIZE_TAB_NOTE,
            ),
            ("size tab/the torque card's heading", SIZE_TAB_NOTE),
            (
                "size tab/end of the Required and Hot-low rows",
                TORQUE_BAND_NOTE,
            ),
            ("end-effect banner", BANNER_NOTE),
        ] {
            let note = reviewed_note(id).expect("a linked note is reviewed");
            emit(
                &format!("why link/{place}"),
                &crate::gui::dashboard::why_note_text(note),
            );
        }
        for &(path, _) in DASHBOARD
            .iter()
            .filter(|row| !crate::engine::clamps::hides(row.0, crate::engine::clamps::PAUSED))
        {
            if notes::note_for(path).is_some() {
                continue;
            }
            let text = match ex::note_fallback(path) {
                ex::NoteFallback::Note(note) => format!("{} {}", ex::NEAREST_NOTE, note.title),
                ex::NoteFallback::Assumption(a) => {
                    format!("{} {}: {}", ex::NEAREST_ASSUMPTION, a.label, a.rationale)
                }
                ex::NoteFallback::Reason(reason) => format!("{reason} {}", ex::START_HERE_HINT),
            };
            emit(&format!("explain fallback/{path}"), &text);
        }
        println!("TOTAL STRINGS: {count}");
    }
}
