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
//! hottest-rated library part). The temperature and space-claim guides pick their fixes by the
//! terms of their verdicts that fail ([`Fixes::PerTerm`]: each temperature limit, ring by
//! ring; each axis of the claim); every other guide has one list. A fix is shown only where
//! its move helps: one slider step, or one other choice of its input alone (those two picks
//! count as one), that turns no passing guided check red; a fix picked per term must also
//! raise that term's margin, while each fix of a static list raises its check from a failing
//! design to its slider's end (a test moves each there). No move leaves a ring's magnet
//! without a rating, since the engine does not calibrate such a magnet's limit. These checks
//! compute the design each move reaches, so the panel asks through a [`FixCache`]. A failing
//! check left without a fix says so ([`NO_FIX`]), and two failing checks that pull one input
//! both ways each say so ([`trade_offs`]), as does a fix that sets back another failing check
//! ([`side_effects`]). A fix's text takes the input's label from the
//! [`InputCatalogue`], or the Size tab's name for it ([`fix_text`]), so no guide types a
//! label. The tests hold the fixes to helping: on each test design that fails a check, every
//! failing term of the check has a shown fix that moves it toward passing, and no shown fix
//! turns another check red.
//!
//! What moves what (spec 3.3): [`Consequences`] are the lines under the Size tab's controls:
//! two magnets fewer and two more (each one `compute_all` of the design with only the count
//! changed; words, not numbers, where the blocks do not fit or the model has no prediction),
//! the smallest outside diameter at which the count fits, and what sets the outside diameter
//! and the overall length ([`crate::engine::size_controls`]). Every number comes from the
//! engine. [`ConsequenceCache`] keeps them until the inputs or the minimum diameter change.
//!
//! The engine's verdict texts, the results table and the exports keep their wording (locked by
//! `tests/python_schema.rs`): this is a GUI layer over them, in plain words
//! ([`plain_text_violations`] lists what it must not say). No egui.

use crate::engine::clamps::SCREW_SIZES;
use crate::engine::deviations::Deviations;
use crate::engine::grades::GRADES;
use crate::engine::library::{MAGNET_LIBRARY, lookup};
use crate::engine::meta::{InputSet, NumOrText, ResultSet, SliderRange, Value};
use crate::engine::model::{
    ResolvedMagnet, RingFit, blocks_fit, end_effect_in_range, resolve_magnets, ring_fits,
};
use crate::engine::size_controls::{
    DiameterBreakdown, DiameterFit, FIT_TOLERANCE_MM, LengthBreakdown, diameter_breakdown,
    dimensions, dimensions_with_minimum_diameter, length_breakdown, resize_diameter,
    smallest_fitting_diameter,
};
use crate::engine::sizing::FreeVariable;
use crate::engine::temperature::{CURE_MARGIN_MIN_C, DemagInputs};
use crate::gui::dashboard::{DASHBOARD, Level, check_level, result_info};
use crate::gui::format::{fixed, format_value, signed_percent, with_unit};
use crate::gui::inputs::{InputCatalogue, optional_seed, section_label};
use crate::gui::pickers::{CUSTOM, GRADE_PATHS, PART_PATHS};
use crate::gui::plain_text::display_label;
use crate::gui::size_controls;
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
    /// the term it serves (the temperature and space-claim guides), with the shaft-clamp
    /// section's pause ([`helps`]).
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
const FACE_GAP: &str = "metal.face_gap_mm";

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
    // The cold pull-out with the variation allowance added: the torque-band note explains it
    // beside hot-low (a value row a reviewed note covers links to that note).
    (
        "metal.torque_cold_high_Nm",
        Learn::Note("hot_low_cold_high"),
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
/// check red, or a value the check needs is not a number, which the Problem names.
pub const NO_FIX: &str = "No listed change helps from here: each is at its slider's end, has no better choice or would fail another check, or a value the check needs is not a number.";

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
            listed(&hot_limits)
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
            listed(&cold_limits)
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
    let onset = format_value(&Value::Num(results.temperature.demag.onset_single_ring_C));
    let margin = results.temperature.summary.cure_margin_C;
    let as_results = || {
        (
            result_text(results, MARGIN),
            number_text(CURE_MARGIN_MIN_C, MARGIN),
        )
    };
    match onset.split_once('.') {
        Some((_, decimals)) if !onset.contains('e') && margin.is_finite() => {
            let shown = fixed(margin, decimals.len());
            let rounded_up_to_pass = margin < CURE_MARGIN_MIN_C
                && shown.parse::<f64>().is_ok_and(|x| x >= CURE_MARGIN_MIN_C);
            if rounded_up_to_pass {
                return as_results();
            }
            let unit = result_info(MARGIN).map_or("", |info| info.meta.unit);
            (
                with_unit(shown, unit),
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals.len()), unit),
            )
        }
        _ => as_results(),
    }
}

/// `items` as one list: "a", "a and b", "a, b and c".
fn listed(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
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

/// The design status line: "Example design: fails 3 of 8 checks", "Your design: fails
/// 2 of 8 checks", "Your design: passes all 8 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 7 while the shaft-clamp section is paused.
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

// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision),
// torques to 0.01 N·m and changes to whole percent.

/// The first line under Magnets per ring with Auto off: the ring radius
/// (`coupling.inner_back_apothem_mm`) is its own input; the magnet count only sets how many
/// blocks go around it.
pub const RING_SIZE_STAYS: &str = "The ring radius stays the same when you change the magnet count; the cup OD below can still change.";

/// The first line under Magnets per ring with Auto on the ring size (physics review 3, N4): a
/// new count changes the solve, so Auto solves the ring radius again, while the neighbour
/// lines are built at the present ring.
pub const AUTO_RESIZES_RING: &str = "Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.";

/// The first line under Magnets per ring with Auto on the magnet length (physics review 3,
/// N4): the ring stays, but Auto solves the length again, while the neighbour lines are built
/// at the present length.
pub const AUTO_RESETS_LENGTH: &str = "Auto resets the magnet length when you change the magnet count; the lines below keep the present length.";

/// The first line under Magnets per ring for the Auto state `auto` (`None`: Auto off): what a
/// new magnet count does to the size Auto sets. `None` with Auto on the magnet count itself:
/// its slider is locked, so no one changes the count there.
pub fn magnet_count_line(auto: Option<FreeVariable>) -> Option<&'static str> {
    match auto {
        None => Some(RING_SIZE_STAYS),
        Some(FreeVariable::RingRadius) => Some(AUTO_RESIZES_RING),
        Some(FreeVariable::AxialLength) => Some(AUTO_RESETS_LENGTH),
        Some(FreeVariable::MagnetsPerRing) => None,
    }
}

/// A neighbour line's words when its blocks do not fit.
const NEIGHBOUR_DOES_NOT_FIT: &str = "magnets do not fit";

/// A neighbour line's words when the model has no number there.
const NEIGHBOUR_NO_PREDICTION: &str = "no prediction";

/// What two magnets fewer or two more would do (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub enum NeighbourOutcome {
    /// The blocks fit and the model predicts a torque.
    Torque {
        /// The neighbour's pull-out torque at the operating temperature (`model.pullout_Nm`).
        pullout_Nm: f64,
        /// Its change as a fraction of the pull-out of the design shown; NaN when that design
        /// has no prediction itself, so there is nothing to compare with.
        change: f64,
        /// The neighbour's cup outer diameter (`model.cup_od_mm`).
        cup_od_mm: f64,
    },
    /// The blocks do not fit at this ring size ([`blocks_fit`]).
    DoesNotFit,
    /// The blocks fit but the model has no number: the end effect is out of range
    /// ([`end_effect_in_range`]) or a torque is not finite.
    NoPrediction,
}

/// A magnet count next to the design's, and what it would do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbour {
    pub npole: i64,
    pub outcome: NeighbourOutcome,
}

impl Neighbour {
    /// "At N: ..." (spec 3.3): the torque and cup, or why there is no number.
    fn line(&self) -> String {
        let at = format!("At {}:", self.npole);
        match self.outcome {
            NeighbourOutcome::Torque {
                pullout_Nm: pullout,
                change,
                cup_od_mm,
            } => {
                let percent = if change.is_finite() {
                    format!(" ({})", signed_percent(change))
                } else {
                    String::new()
                };
                format!(
                    "{at} pull-out {}{percent}, cup OD {}",
                    with_unit(fixed(pullout, 2), "N·m"),
                    with_unit(fixed(cup_od_mm, 1), "mm")
                )
            }
            NeighbourOutcome::DoesNotFit => format!("{at} {NEIGHBOUR_DOES_NOT_FIT}"),
            NeighbourOutcome::NoPrediction => format!("{at} {NEIGHBOUR_NO_PREDICTION}"),
        }
    }
}

/// The "what moves what" readings of a design (spec 3.3), all from the engine.
#[derive(Clone, Debug, PartialEq)]
pub struct Consequences {
    /// Two magnets fewer and two more, ascending, where the count stays an even count inside
    /// the magnet slider.
    pub neighbours: Vec<Neighbour>,
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
    /// neighbour's change compares with this design's pull-out), with the Size tab's minimum
    /// diameter preference (`None`: the slider's own floor).
    pub fn of(
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
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
            neighbours: neighbours(inputs, results),
            fit,
            fit_unlocked,
            diameter: diameter_breakdown(inputs),
            length: length_breakdown(inputs),
        }
    }

    /// The lines under Magnets per ring: the line for the Auto state `auto`
    /// ([`magnet_count_line`]; `None`: Auto off), one line per neighbour, then the smallest
    /// diameter that fits. `npole` is the count the consequences were built for
    /// (`inputs.coupling.npole`).
    pub fn magnet_lines(&self, npole: i64, auto: Option<FreeVariable>) -> Vec<String> {
        let mut lines: Vec<String> = magnet_count_line(auto)
            .map(str::to_owned)
            .into_iter()
            .collect();
        lines.extend(self.neighbours.iter().map(Neighbour::line));
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

/// The neighbours of `inputs` (`results` are its own), each one `compute_all` of the design
/// with only the magnet count changed.
fn neighbours(inputs: &DesignInputs, results: &DesignResults) -> Vec<Neighbour> {
    let baseline = prediction(results);
    neighbour_counts(inputs.coupling.npole)
        .into_iter()
        .map(|npole| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let outcome = neighbour_outcome(&design, &compute_all(&design), baseline);
            Neighbour { npole, outcome }
        })
        .collect()
}

/// Two fewer and two more than `npole`, ascending, keeping the even counts inside the magnet
/// slider (one block per pole; the slider steps by 2). A count from a design file may be odd
/// or past the slider, and none overflows.
fn neighbour_counts(npole: i64) -> Vec<i64> {
    let range = FreeVariable::MagnetsPerRing.range();
    [npole.checked_sub(2), npole.checked_add(2)]
        .into_iter()
        .flatten()
        .filter(|&n| n % 2 == 0 && (range.min..=range.max).contains(&(n as f64)))
        .collect()
}

/// The pull-out of the design shown when the model predicts one: the end effect in range and
/// a positive, finite torque. Without it a neighbour's change has nothing to compare with.
fn prediction(results: &DesignResults) -> Option<f64> {
    let pullout = results.model.pullout_Nm;
    (end_effect_in_range(results.model.f_end) && pullout.is_finite() && pullout > 0.0)
        .then_some(pullout)
}

/// What a neighbour design does (`results` are its own; `baseline` is [`prediction`] of the
/// design shown). The conditions of [`crate::engine::sizing::is_valid`] but the keyway's: the
/// hub wall past the keyway (apothem - bondline - bore / 2 - keyway depth) does not depend on
/// the magnet count, so a neighbour fails it exactly when the design shown does, the Size tab
/// already says so, and the torques still compare.
fn neighbour_outcome(
    design: &DesignInputs,
    results: &DesignResults,
    baseline: Option<f64>,
) -> NeighbourOutcome {
    if !blocks_fit(&design.coupling, &results.model) {
        return NeighbourOutcome::DoesNotFit;
    }
    let pullout = results.model.pullout_Nm;
    if !end_effect_in_range(results.model.f_end)
        || !pullout.is_finite()
        || !results.metal.torque_hot_low_Nm.is_finite()
    {
        return NeighbourOutcome::NoPrediction;
    }
    NeighbourOutcome::Torque {
        pullout_Nm: pullout,
        change: baseline.map_or(f64::NAN, |base| pullout / base - 1.0),
        cup_od_mm: results.model.cup_od_mm,
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
/// frame, and the neighbours' `compute_all` runs and the fit search are not per-frame work. They
/// are computed again only when the inputs or the minimum diameter change.
#[derive(Default)]
pub struct ConsequenceCache {
    last: Option<CachedConsequences>,
    /// How many times consequences were computed (the tests check that a repeat computes
    /// nothing).
    pub(crate) computed: usize,
}

/// What a [`ConsequenceCache`] holds: the question and its answer.
struct CachedConsequences {
    inputs: DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: Consequences,
}

impl ConsequenceCache {
    /// [`Consequences::of`] `inputs` (whose results are `results`) with the minimum diameter
    /// preference, computed only when the inputs or the minimum differ from the last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
    ) -> &Consequences {
        let current = self.last.as_ref().is_some_and(|last| {
            last.inputs == *inputs && last.minimum_diameter_mm == minimum_diameter_mm
        });
        if !current {
            self.computed += 1;
            self.last = Some(CachedConsequences {
                inputs: inputs.clone(),
                minimum_diameter_mm,
                consequences: Consequences::of(inputs, results, minimum_diameter_mm),
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
    use crate::engine::model::ring_fits;
    use crate::gui::dashboard::{CHECKS, DASHBOARD};
    use crate::gui::explorer::reviewed_note;
    use crate::gui::plain_text::label_for;
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, hot_day, only_the_clamp_fails, passing_design,
        short_magnets, small_boss, with_poles,
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

    /// The Problem of `check` for `inputs`.
    fn problem(check: &str, inputs: &DesignInputs) -> Option<String> {
        (guide(check).unwrap().problem)(inputs, &compute_all(inputs))
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
        for fallback in [NO_FIX, TEMPERATURE_FALLBACK, SPACE_CLAIM_FALLBACK] {
            texts.push(fallback.to_owned());
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

    /// The torque-band note's second watch-out says nothing checks cold-high at the gearbox
    /// input against the gearbox's rating. When such a check lands (BL-059) this fails, and
    /// that watch-out changes in the same commit.
    #[test]
    fn nothing_checks_a_gearbox_torque_against_its_rating_yet() {
        let gearbox_rows: Vec<(&str, Option<&str>)> = DASHBOARD
            .iter()
            .copied()
            .filter(|&(row, check)| {
                row.contains("gearbox") || check.is_some_and(|c| c.contains("gearbox"))
            })
            .collect();
        // The ripple row: a value with no check.
        assert_eq!(gearbox_rows, [("model.gearbox_input_ripple_Nm", None)]);
        assert!(CHECKS.iter().all(|c| !c.contains("gearbox")));
        assert!(CHECK_GUIDES.iter().all(|g| !g.check.contains("gearbox")));
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
        // still the seven whose label another input's equals or ends with (map-help 3.5).
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
            ]
        );
    }

    // ------------------------------------------------------------------ the fixes help

    /// One term of a check's verdict: its margin, larger is better, and whether it passes as
    /// the verdict compares it.
    #[derive(Clone, Debug)]
    struct Term {
        name: String,
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

    /// The terms of `check`'s verdict in a fixed order (two designs' lists pair up). The
    /// temperature terms are the guide's own ([`TemperatureTerm`]: a test holds them to the
    /// verdict), named as they print.
    fn terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<Term> {
        let term = |name: &str, margin: f64, passes| Term {
            name: name.to_owned(),
            margin,
            passes,
        };
        let at_least_zero = |name, margin: f64| term(name, margin, margin >= 0.0);
        let (m, md) = (&results.model, &results.metal);
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
            TEMPERATURE => TemperatureTerm::ALL
                .into_iter()
                .map(|t| {
                    let margin = t.margin(inputs, results);
                    term(&format!("{t:?}"), margin, !t.fails(inputs, results))
                })
                .collect(),
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
            // The engine's rule ring by ring; the margin is the flat's slack, or what is left
            // of the arcs' share of the circle.
            FIT_CHECK => ring_fits(&inputs.coupling, m)
                .into_iter()
                .zip(["inner ring", "outer ring"])
                .map(|(ring, name)| {
                    let margin = match ring {
                        RingFit::Flat { width_mm, flat_mm } => flat_mm - width_mm,
                        RingFit::Arc { share } => 1.0 - share,
                    };
                    term(name, margin, ring.fits())
                })
                .collect(),
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
                    .filter(|&(_, d)| matches!(d, Direction::Raise | Direction::Lower))
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
            let to_end = |range: SliderRange| match direction {
                Direction::Raise => range.max,
                _ => range.min,
            };
            // The Outside diameter moves as the Size tab's slider moves it (cap following).
            let moved = if path == RING {
                let size = dimensions(&failing_design).unwrap();
                resize_diameter(&failing_design, to_end(size.diameter_range)).unwrap()
            } else {
                let range = catalogue.entry(path).unwrap().meta.range.unwrap();
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

    #[test]
    fn two_failing_checks_that_pull_one_input_apart_say_so() {
        // Spec review focus 4: the default design's first screen once said "lower the gap" for
        // the hot-low torque and "raise the gap" for the clearance. A fix whose step turns a
        // passing check red is not offered, so that pair cannot show. One input still shows
        // both ways when two checks that pull it apart both fail already: here the hot-low
        // torque against an overall-length claim the stack is over (magnet length), and the
        // blocks against a diameter claim (Outside diameter). Each fix helps its own check, so
        // both stay, and each of the two blocks says, in the same words, that the input pulls
        // two ways and another input has to settle one of them (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 2] = [
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
            // The guides that raise and lower each such input carry its note; no other does.
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                let mut wanted = Vec::new();
                for &path in &both_ways {
                    let by = |direction| {
                        let at = all.iter().position(|f| f.contains(&(path, direction)));
                        CHECK_GUIDES[at.unwrap()].label
                    };
                    if all[index].iter().any(|&(fixed, _)| fixed == path) {
                        wanted.push(note(path, by(Direction::Raise), by(Direction::Lower)));
                    }
                }
                assert_eq!(trade_offs(&all, index), wanted, "{name}: {}", g.check);
            }
        }
        // The two cases in words: the same note in both blocks.
        let length = design(|i| i.metal.max_overall_axial_mm = 30.0);
        let results = compute_all(&length);
        let all = all_fixes(&length, &results, false);
        let expected = [
            "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Space claim needs it lowered, so change another input to settle one of them.",
        ];
        let at = |check| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        assert_eq!(trade_offs(&all, at(HOT_MIN)), expected);
        assert_eq!(trade_offs(&all, at(SPACE_CLAIM)), expected);
        let crowded = with(with_poles(12), |i| i.metal.max_diameter_mm = 42.0);
        let results = compute_all(&crowded);
        let all = all_fixes(&crowded, &results, false);
        let expected = [
            "One input pulls two ways (Outside diameter): Magnets fit needs it raised and Space claim needs it lowered, so change another input to settle one of them.",
        ];
        assert_eq!(trade_offs(&all, at(FIT_CHECK)), expected);
        assert_eq!(trade_offs(&all, at(SPACE_CLAIM)), expected);
        // An index past the guides, or lists shorter than the guides, read as no note.
        assert!(trade_offs(&all, CHECK_GUIDES.len()).is_empty());
        assert!(trade_offs(&all[..1], at(SPACE_CLAIM)).is_empty());
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
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for text in every_text() {
            assert_glyphs(&ctx, &text, "guidance");
        }
    }

    #[test]
    fn no_guide_text_calls_the_ring_radius_ring_size() {
        // One name per quantity (spec 3.5): the input row says Ring radius, so no guide, fix,
        // problem or consequence line says "ring size".
        let mut texts = every_text();
        texts.extend([RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH].map(str::to_owned));
        for text in texts {
            assert!(!text.to_lowercase().contains("ring size"), "{text:?}");
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
    fn the_outside_diameter_is_not_lowered_past_the_minimum_od() {
        // `fixes` steps the Outside diameter over its slider without the panel's Minimum OD;
        // the panel's raised floor stops that slider, so "Lower Outside diameter" goes there.
        // A raise of it (the fit's) and every other fix stay.
        let lower_ring = (RING, Direction::Lower);
        let raise_ring = (RING, Direction::Raise);
        let index = |check: &str| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        for (inputs, check, fix) in [
            (breaking(large_ring), SPACE_CLAIM, lower_ring),
            (with_poles(12), FIT_CHECK, raise_ring),
        ] {
            let results = compute_all(&inputs);
            let all = all_fixes(&inputs, &results, false);
            assert!(all[index(check)].contains(&fix), "{check}: {all:?}");
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
        let status = |inputs: &DesignInputs| design_status(inputs, &compute_all(inputs), false);
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
            "Example design: fails 3 of 8 checks"
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
        assert_eq!((shown.bad, shown.caution, shown.total), (1, 0, 8));
        assert_eq!(
            design_status_text(&shown),
            "Your design: fails 1 of 8 checks"
        );
        let paused = design_status(&inputs, &results, true);
        assert_eq!((paused.bad, paused.caution, paused.total), (0, 0, 7));
        assert_eq!(
            design_status_text(&paused),
            "Your design: passes all 7 checks"
        );
        // The example design fails the same three checks either way.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 3 of 7 checks"
        );
    }

    #[test]
    fn a_paused_clamp_offers_no_fix_and_holds_back_no_other_guide_s_fix() {
        let (clamp, hot) = (guide(CLAMP).unwrap(), guide(HOT_MIN).unwrap());
        let index = CHECK_GUIDES.iter().position(|g| g.check == CLAMP).unwrap();
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
        let index = CHECK_GUIDES.iter().position(|g| g.check == CLAMP).unwrap();
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

    /// The consequences of `inputs` as the panel builds them: with the design's own results.
    fn consequences_of(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Consequences {
        Consequences::of(inputs, &compute_all(inputs), minimum_diameter_mm)
    }

    /// `inputs` with `npole` magnets per ring and nothing else changed.
    fn with_magnets(inputs: &DesignInputs, npole: i64) -> DesignInputs {
        FreeVariable::MagnetsPerRing.apply(inputs, npole as f64)
    }

    /// A design whose own end effect is just out of range while two more magnets bring it back
    /// in and fit: the first ring radius of the solver's grid where two more magnets fit, the
    /// magnets cut halfway between the two counts' end-effect limits (f_end = 1 - c_end * pole
    /// pitch / L is zero at L = c_end * pole pitch, and the pitch falls as the count rises).
    fn end_effect_edge() -> DesignInputs {
        let base = DesignInputs::default();
        let more = base.coupling.npole + 2;
        let radius = FreeVariable::RingRadius
            .grid()
            .into_iter()
            .find(|&radius| {
                let design = with_magnets(&FreeVariable::RingRadius.apply(&base, radius), more);
                blocks_fit(&design.coupling, &compute_all(&design).model)
            })
            .expect("two more magnets fit somewhere on the ring-radius slider");
        let mut design = FreeVariable::RingRadius.apply(&base, radius);
        // The end-effect coefficient's slider maximum: the length lands inside its slider.
        design.coupling.c_end = 0.5;
        let pitch = |npole: i64| {
            compute_all(&with_magnets(&design, npole))
                .model
                .pole_pitch_mm
        };
        let length = design.coupling.c_end * (pitch(base.coupling.npole) + pitch(more)) / 2.0;
        design.coupling.magnets.axial_length_mm = Some(length);
        design
    }

    #[test]
    fn each_neighbour_is_the_design_with_only_the_magnet_count_changed() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
        assert_eq!(counts, [npole - 2, npole + 2]);
        for neighbour in &consequences.neighbours {
            let design = with_magnets(&inputs, neighbour.npole);
            let r = compute_all(&design);
            match neighbour.outcome {
                NeighbourOutcome::Torque {
                    pullout_Nm: pullout,
                    change,
                    cup_od_mm,
                } => {
                    assert!(
                        crate::engine::sizing::is_valid(&design, &r),
                        "{neighbour:?}"
                    );
                    assert_eq!(pullout, r.model.pullout_Nm);
                    assert_eq!(change, r.model.pullout_Nm / results.model.pullout_Nm - 1.0);
                    assert_eq!(cup_od_mm, r.model.cup_od_mm);
                }
                NeighbourOutcome::DoesNotFit => {
                    assert!(!blocks_fit(&design.coupling, &r.model), "{neighbour:?}")
                }
                NeighbourOutcome::NoPrediction => {
                    assert!(!end_effect_in_range(r.model.f_end), "{neighbour:?}")
                }
            }
        }
        // The owner's case (spec 1): two fewer magnets fit, lose torque and grow the cup; two
        // more do not fit at this ring size.
        match consequences.neighbours[0].outcome {
            NeighbourOutcome::Torque {
                change, cup_od_mm, ..
            } => {
                assert!(change < 0.0, "{change}");
                assert!(cup_od_mm > results.model.cup_od_mm, "{cup_od_mm}");
            }
            other => panic!("{} magnets fit at the default ring: {other:?}", npole - 2),
        }
        assert_eq!(
            consequences.neighbours[1].outcome,
            NeighbourOutcome::DoesNotFit
        );
    }

    #[test]
    fn the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let fewer = compute_all(&with_magnets(&inputs, npole - 2));
        let lines = Consequences::of(&inputs, &results, None).magnet_lines(npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert_eq!(
            lines[1],
            format!(
                "At {}: pull-out {:.2} N·m ({:+.0} %), cup OD {:.1} mm",
                npole - 2,
                fewer.model.pullout_Nm,
                (100.0 * (fewer.model.pullout_Nm / results.model.pullout_Nm - 1.0)).round(),
                fewer.model.cup_od_mm
            )
        );
        assert_eq!(lines[2], format!("At {}: magnets do not fit", npole + 2));
        assert_eq!(lines.len(), 4, "the fit line closes the list: {lines:?}");
    }

    #[test]
    fn at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown() {
        // Spec 5: at 4 or 40 magnets, only the neighbour inside the slider range.
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        for (npole, inside, outside) in
            [(fewest, fewest + 2, fewest - 2), (most, most - 2, most + 2)]
        {
            let inputs = with_magnets(&DesignInputs::default(), npole);
            let consequences = consequences_of(&inputs, None);
            let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(counts, [inside], "{npole} magnets");
            let lines = consequences.magnet_lines(npole, None);
            assert!(
                lines
                    .iter()
                    .any(|l| l.starts_with(&format!("At {inside}:"))),
                "{lines:?}"
            );
            assert!(
                !lines
                    .iter()
                    .any(|l| l.starts_with(&format!("At {outside}:"))),
                "{lines:?}"
            );
        }
    }

    #[test]
    fn neighbours_are_even_counts_inside_the_slider_and_never_overflow() {
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        assert_eq!(neighbour_counts(fewest + 4), [fewest + 2, fewest + 6]);
        // A count from a design file may be odd or past the slider.
        assert!(
            neighbour_counts(fewest + 3).is_empty(),
            "odd: no even neighbour"
        );
        assert_eq!(neighbour_counts(most + 2), [most]);
        assert_eq!(neighbour_counts(fewest - 2), [fewest]);
        assert!(neighbour_counts(most + 4).is_empty());
        assert!(neighbour_counts(i64::MAX).is_empty());
        assert!(neighbour_counts(i64::MIN).is_empty());
        // An odd count shows the ring-size line and no neighbour.
        let odd = with_magnets(&DesignInputs::default(), fewest + 5);
        let lines = consequences_of(&odd, None).magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(!lines.iter().any(|l| l.starts_with("At ")), "{lines:?}");
    }

    #[test]
    fn a_keyway_through_the_hub_changes_no_neighbour_s_reading() {
        // The hub wall past the keyway does not depend on the magnet count, so a neighbour
        // fails it exactly when the design shown does (the Size tab says so already).
        let mut broken = DesignInputs::default();
        broken.coupling.bore_mm = 16.0; // the bore and keyway sliders' maxima
        broken.coupling.keyway_depth_mm = 4.0;
        let mut intact = broken.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert!(compute_all(&broken).model.hub_wall_past_key_mm <= 0.0);
        assert!(compute_all(&intact).model.hub_wall_past_key_mm > 0.0);
        let read = |inputs: &DesignInputs| neighbours(inputs, &compute_all(inputs));
        assert_eq!(read(&broken), read(&intact));
        assert!(
            matches!(read(&broken)[0].outcome, NeighbourOutcome::Torque { .. }),
            "{:?}",
            read(&broken)
        );
    }

    #[test]
    fn with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it() {
        let range = FreeVariable::MagnetsPerRing.range();
        for design in [
            DesignInputs::default(),
            crate::gui::test_support::short_magnets(),
            end_effect_edge(),
        ] {
            for npole in (range.min as i64..=range.max as i64).step_by(2) {
                let inputs = with_magnets(&design, npole);
                for neighbour in neighbours(&inputs, &compute_all(&inputs)) {
                    let at = with_magnets(&design, neighbour.npole);
                    let r = compute_all(&at);
                    assert!(r.model.hub_wall_past_key_mm > 0.0);
                    assert_eq!(
                        matches!(neighbour.outcome, NeighbourOutcome::Torque { .. }),
                        crate::engine::sizing::is_valid(&at, &r),
                        "{npole} -> {neighbour:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented() {
        // Spec 5: words, not numbers. The design shown has no prediction either, so the
        // neighbour that has one shows its torque without a change against a meaningless one.
        let inputs = end_effect_edge();
        let results = compute_all(&inputs);
        assert!(
            !end_effect_in_range(results.model.f_end),
            "{}",
            results.model.f_end
        );
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        assert_eq!(
            consequences.neighbours[0],
            Neighbour {
                npole: npole - 2,
                outcome: NeighbourOutcome::NoPrediction
            }
        );
        let more = compute_all(&with_magnets(&inputs, npole + 2));
        match consequences.neighbours[1].outcome {
            NeighbourOutcome::Torque {
                pullout_Nm: pullout,
                change,
                cup_od_mm,
            } => {
                assert_eq!(pullout, more.model.pullout_Nm);
                assert!(change.is_nan(), "{change}");
                assert_eq!(cup_od_mm, more.model.cup_od_mm);
            }
            other => panic!("{other:?}"),
        }
        let lines = consequences.magnet_lines(npole, None);
        assert_eq!(lines[1], format!("At {}: no prediction", npole - 2));
        assert_eq!(
            lines[2],
            format!(
                "At {}: pull-out {:.2} N·m, cup OD {:.1} mm",
                npole + 2,
                more.model.pullout_Nm,
                more.model.cup_od_mm
            )
        );
    }

    #[test]
    fn with_magnets_too_short_for_the_model_every_neighbour_reads_words() {
        let inputs = crate::gui::test_support::short_magnets();
        let npole = inputs.coupling.npole;
        let lines = consequences_of(&inputs, None).magnet_lines(npole, None);
        let neighbour_lines: Vec<&String> = lines.iter().filter(|l| l.starts_with("At ")).collect();
        assert_eq!(neighbour_lines.len(), 2, "{lines:?}");
        for (line, count) in neighbour_lines.into_iter().zip([npole - 2, npole + 2]) {
            let r = compute_all(&with_magnets(&inputs, count));
            let words = if !blocks_fit(&with_magnets(&inputs, count).coupling, &r.model) {
                "magnets do not fit"
            } else if !end_effect_in_range(r.model.f_end) {
                "no prediction"
            } else {
                panic!("{count} magnets give a torque on short magnets: {line}")
            };
            assert_eq!(*line, format!("At {count}: {words}"));
        }
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
            neighbours: Vec::new(),
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
        // Spec tests 3 and 11, over every line form: a torque, a torque without a change, both
        // word forms, the three fit forms and both breakdowns.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
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
        // Physics review 3, N4: the ring keeps its radius only with Auto off. With Auto on the
        // ring size a new count is solved again, and on the magnet length the length is; the
        // neighbour lines keep the present size, and the first line says so. With Auto on the
        // count its slider is locked, so no line speaks of changing it.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        let off = consequences.magnet_lines(npole, None);
        let states: [(Option<FreeVariable>, Option<&str>); 4] = [
            (None, Some(RING_SIZE_STAYS)),
            (
                Some(FreeVariable::RingRadius),
                Some(
                    "Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.",
                ),
            ),
            (
                Some(FreeVariable::AxialLength),
                Some(
                    "Auto resets the magnet length when you change the magnet count; the lines below keep the present length.",
                ),
            ),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            assert_eq!(magnet_count_line(auto), first, "{auto:?}");
            let lines = consequences.magnet_lines(npole, auto);
            // The first line alone follows the state: the rest are the Auto-off lines.
            match first {
                Some(first) => {
                    assert_eq!(lines[0], first, "{auto:?}");
                    assert_eq!(lines[1..], off[1..], "{auto:?}");
                    crate::gui::test_support::assert_glyphs(&ctx, first, "a magnet count line");
                    assert_eq!(plain_text_violations(first), Vec::<&str>::new(), "{first}");
                }
                None => assert_eq!(lines[..], off[1..], "{auto:?}"),
            }
        }
    }

    #[test]
    fn the_cache_computes_once_per_design_and_minimum_and_follows_an_edit() {
        // Spec 4.1 and test 5: computed again only when the inputs or the minimum change; what
        // it returns is always Consequences::of the design asked about.
        let mut cache = ConsequenceCache::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let first = cache.get(&inputs, &results, None).clone();
        assert_eq!(first, Consequences::of(&inputs, &results, None));
        for _ in 0..5 {
            assert_eq!(cache.get(&inputs, &results, None), &first);
        }
        assert_eq!(
            cache.computed, 1,
            "a design asked about again is not computed again"
        );
        // An edit: the neighbours' numbers follow it.
        let mut hot = inputs.clone();
        hot.coupling.op_temp_C += 30.0;
        let hot_results = compute_all(&hot);
        let edited = cache.get(&hot, &hot_results, None).clone();
        assert_eq!(cache.computed, 2);
        assert_eq!(edited, Consequences::of(&hot, &hot_results, None));
        assert_ne!(edited.neighbours, first.neighbours);
        let fewer = compute_all(&with_magnets(&hot, hot.coupling.npole - 2));
        assert!(
            matches!(
                edited.neighbours[0].outcome,
                NeighbourOutcome::Torque { pullout_Nm: pullout, .. } if pullout == fewer.model.pullout_Nm
            ),
            "{:?}",
            edited.neighbours
        );
        // A new minimum diameter is a new question too.
        let lowest = crate::engine::size_controls::dimensions(&hot)
            .expect("a size")
            .minimum_diameter_range
            .min;
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest)),
            &Consequences::of(&hot, &hot_results, Some(lowest))
        );
        assert_eq!(cache.computed, 3);
        cache.get(&hot, &hot_results, Some(lowest));
        assert_eq!(cache.computed, 3);
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
        let clearance = CHECK_GUIDES
            .iter()
            .position(|g| g.check == CLEARANCE)
            .unwrap();
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
        // No other guide gets the note, and an index past the guides gets none.
        let results = compute_all(&defaults);
        let all = all_fixes(&defaults, &results, false);
        for index in (0..=CHECK_GUIDES.len()).filter(|&i| i != clearance) {
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
        let clearance = CHECK_GUIDES
            .iter()
            .position(|g| g.check == CLEARANCE)
            .unwrap();
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
        for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
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
        println!("TOTAL STRINGS: {count}");
    }
}
