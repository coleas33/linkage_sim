//! BL-064: the shaft joints. Both rotors are held on their shafts by a keyed shrink fit (owner,
//! 2026-10-17: "Hub is shrink fit with key only"): the inner hub, its magnets bonded on its
//! flats, on the wheel shaft, and the cup, its magnets in its pocket, its rear web and integral
//! boss, on the gearbox output shaft. Both sit on the one bore (`coupling.bore_mm`). Cold-high
//! (`metal.torque_cold_high_Nm`: the pull-out at the minimum magnet temperature with the
//! production allowance) can reach each joint before the coupling slips, and while it slips
//! each joint carries a fully reversing swing.
//!
//! The rule (owner decision 2A): the fit alone must hold the safety factor times cold-high at
//! its loosest; the part must stay within its stress limit, yield ÷ [`YIELD_FACTOR`], at its
//! tightest; the key alone, a backup, is checked at the same torque against the same limit. The
//! fit's and the key's capacities are never added. Each part must also be fitted within its
//! allowed temperature (owner decision 7A) in the owner's order: the bare cup is shrunk onto the
//! gearbox shaft before its magnets are bonded; the inner hub is bonded first, then shrunk onto
//! the cooled wheel shaft, so its allowed temperature is also capped by the adhesive's design
//! limit and by the inner magnets' own single-ring onset less the cure margin (owner R-15 and
//! its ring ruling, [`hub_fit_allowed`]).
//!
//! Each joint is judged on four terms, in this order ([`CAUSES`]): the fit slips, the bore
//! stress is above its limit, it cannot be fitted at the allowed temperature, the key backup
//! (no bearing area, or above the limit). The badge shows the first failing term, naming the
//! first failing joint (inner hub, then cup). A joint with no wall around its bore or with its
//! band reversed is not judged; it reads unknown only when no judged joint fails. A value that
//! is not a number reads unknown before all of them, then a safety factor not above zero
//! ([`verdict`]). Equality passes; a failing badge never quotes two equal numbers (the BL-059
//! rounding).
//!
//! The temperature model: the interference moves with temperature by the expansion mismatch of
//! shaft and part. The loosest fit takes the end that loosens it more: the minimum magnet
//! temperature, or the selected adhesive's design limit with the rotor the rotor-over-shaft
//! allowance hotter than its shaft. The tightest fit takes the end that tightens it more. The
//! shafts' shrink at fitting comes from NIST's expansion fit for a nickel steel below 265 K.
//!
//! Not checked: fretting and fatigue of the joints under the reversing swing (BL-070), the
//! gearbox's own shaft, bearings and seal, the keyway corner's stress concentration, the bond
//! strain from fitting a bonded hub, and the key's shear.

use std::f64::consts::PI;

use super::compat::{fmt_fixed, py_max, py_min};
use super::meta::{InputSet, Value, inputs, out_rust_only, param_rust_only, results};
use super::temperature::CURE_MARGIN_MIN_C;

inputs! {
    /// The shaft joints' inputs (BL-064). Rust-only: no workbook cell.
    pub struct ShaftJointInputs {
        fields {
            hub_interference_min_um: f64 = 13.0 => param_rust_only("µm", "Inner hub interference, minimum",
                "The smallest measured diametral interference of the inner hub on the wheel shaft: shaft diameter less bore, both measured. Holding is checked at it.")
                .range(0.0, 40.0, 0.5),
            hub_interference_max_um: f64 = 18.0 => param_rust_only("µm", "Inner hub interference, maximum",
                "The largest measured diametral interference of the inner hub. Its bore stress and fitting temperature are checked at it.")
                .range(0.0, 40.0, 0.5),
            cup_interference_min_um: f64 = 15.0 => param_rust_only("µm", "Cup interference, minimum",
                "The smallest measured diametral interference of the cup's boss on the gearbox shaft. Holding is checked at it.")
                .range(0.0, 40.0, 0.5),
            cup_interference_max_um: f64 = 22.0 => param_rust_only("µm", "Cup interference, maximum",
                "The largest measured diametral interference of the cup. Its bore stress and fitting temperature are checked at it.")
                .range(0.0, 40.0, 0.5),
            safety_factor: f64 = 2.0 => param_rust_only("-", "Shaft joint safety factor",
                "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. The torque reverses many times in each slip.")
                .range(1.0, 4.0, 0.1),
            friction: f64 = 0.14 => param_rust_only("-", "Shrink-fit friction coefficient",
                "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it.")
                .range(0.05, 0.30, 0.01),
            smoothing_um: f64 = 5.0 => param_rust_only("µm", "Surface smoothing of the fit",
                "Interference lost as surface peaks flatten when fitted. It lowers both the grip and the bore stress; a ground shaft and honed bore lose less.")
                .range(0.0, 15.0, 0.1),
            rotor_over_shaft_C: f64 = 20.0 => param_rust_only("°C", "Rotor hotter than its shaft, running",
                "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls.")
                .range(0.0, 60.0, 1.0),
            key_width_mm: f64 = 3.0 => param_rust_only("mm", "Shaft key width",
                "Width of the parallel key in both joints (DIN 6885-1 gives 3 mm for a 10 mm shaft). Its slot takes this much of the fit's contact.")
                .range(1.0, 6.0, 0.5),
            key_contact_mm: f64 = 1.2 => param_rust_only("mm", "Shaft key contact height",
                "Height of the key face bearing on the inner hub and the cup, capped at the keyway depth (DIN 6885-1 gives 1.2 mm for 10 mm).")
                .range(0.5, 3.0, 0.1),
            hub_fit_max_C: f64 = 100.0 => param_rust_only("°C", "Hottest the bonded inner hub may be for fitting",
                "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and at the magnets' single-ring onset less 10 °C.")
                .range(20.0, 130.0, 1.0),
            cup_fit_max_C: f64 = 300.0 => param_rust_only("°C", "Hottest the bare cup may be for fitting",
                "The cup is fitted before its magnets are bonded. Stay below the steel's and its finish's treatment temperatures, and keep the heat off the gearbox seal.")
                .range(20.0, 400.0, 5.0),
            wheel_shaft_fit_C: f64 = -196.0 => param_rust_only("°C", "Wheel shaft temperature at fitting",
                "A cooled shaft shrinks, so the inner hub needs less heat. Dry ice is about -78 °C; liquid nitrogen is -196 °C.")
                .range(-196.0, 40.0, 1.0),
            output_shaft_fit_C: f64 = 20.0 => param_rust_only("°C", "Gearbox shaft temperature at fitting",
                "The gearbox output shaft cannot be immersed. Cold spray can take it to about -20 °C, the slider's bottom; never below the gearbox's rated range.")
                .range(-20.0, 40.0, 1.0),
            joining_clearance_um: f64 = 10.0 => param_rust_only("µm", "Joining clearance",
                "Clearance each part needs over its shaft to slide on before it grips; one value for both joints. Less needs a guided, single fast push to a stop.")
                .range(0.0, 30.0, 0.5),
        }
    }
}

results! {
    /// The shaft joint check (BL-064). Rust-only.
    pub struct ShaftJointResults {
        fields {
            required_Nm: f64 => out_rust_only("N·m", "Torque each shaft joint must hold",
                "Cold-high (the pull-out at the minimum magnet temperature, with the production allowance) times the safety factor. Each fit alone must hold it; the key alone is checked at it."),
            part_modulus_GPa: f64 => out_rust_only("GPa", "Inner hub and cup elastic modulus",
                "Of the inner hub and cup material in effect: the back-iron steel with a steel circuit, else the hub body's material. The fit's contact pressure rises with it."),
            part_cte_per_C: f64 => out_rust_only("1/°C", "Inner hub and cup expansion coefficient",
                "Of the inner hub and cup material in effect. Against the shafts' steel, a higher value loosens the fit hot and a lower one loosens it cold."),
            part_yield_MPa: f64 => out_rust_only("MPa", "Inner hub and cup yield strength",
                "The library's yield strength of the inner hub and cup material in effect; for 4140, the annealed value, conservative while the condition is open."),
            stress_limit_MPa: f64 => out_rust_only("MPa", "Bore stress limit",
                "The part's yield strength with a margin. At the tightest fit the bore stress must not exceed it, so the part stays elastic and keeps its grip."),
            hub_holding_Nm: f64 => out_rust_only("N·m", "Inner hub holding torque at its loosest",
                "Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."),
            cup_holding_Nm: f64 => out_rust_only("N·m", "Cup holding torque at its loosest",
                "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."),
            hub_bore_stress_MPa: f64 => out_rust_only("MPa", "Inner hub bore stress at its tightest",
                "Von Mises stress at the inner hub's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included."),
            cup_bore_stress_MPa: f64 => out_rust_only("MPa", "Cup bore stress at its tightest",
                "Von Mises stress at the cup's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included."),
            hub_key_pressure_MPa: f64 => out_rust_only("MPa", "Key pressure on the inner hub, key alone",
                "Pressure on the key's face in the inner hub were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area."),
            cup_key_pressure_MPa: f64 => out_rust_only("MPa", "Key pressure on the cup, key alone",
                "Pressure on the key's face in the cup were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area."),
            hub_needed_um: f64 => out_rust_only("µm", "Inner hub smallest measured interference that holds",
                "The lowest the band's minimum may be for the inner hub to hold the torque it must: the window's lower end, with smoothing and temperature included."),
            cup_needed_um: f64 => out_rust_only("µm", "Cup smallest measured interference that holds",
                "The lowest the band's minimum may be for the cup to hold the torque it must: the window's lower end, with smoothing and temperature included."),
            hub_elastic_max_um: f64 => out_rust_only("µm", "Inner hub largest measured interference within the stress limit",
                "The highest the band's maximum may be for the inner hub's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window."),
            cup_elastic_max_um: f64 => out_rust_only("µm", "Cup largest measured interference within the stress limit",
                "The highest the band's maximum may be for the cup's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window."),
            wheel_shaft_shrink_um: f64 => out_rust_only("µm", "Wheel shaft shrink at its fitting temperature",
                "How much the wheel shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows."),
            output_shaft_shrink_um: f64 => out_rust_only("µm", "Gearbox shaft shrink at its fitting temperature",
                "How much the gearbox shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows."),
            hub_fittable_max_um: f64 => out_rust_only("µm", "Inner hub largest measured interference that can be fitted",
                "The highest the band's maximum may be for the inner hub to be fitted: what its allowed temperature and the wheel shaft's shrink open, less the joining clearance."),
            cup_fittable_max_um: f64 => out_rust_only("µm", "Cup largest measured interference that can be fitted",
                "The highest the band's maximum may be for the cup to be fitted: what its allowed temperature and the gearbox shaft's shrink open, less the joining clearance."),
            hub_fit_temp_C: f64 => out_rust_only("°C", "Inner hub temperature the fit would need",
                "What the inner hub would need on its wheel shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further."),
            cup_fit_temp_C: f64 => out_rust_only("°C", "Cup temperature the fit would need",
                "What the bare cup would need on the gearbox shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further."),
            hub_fit_allowed_C: f64 => out_rust_only("°C", "Inner hub fitting temperature allowed",
                "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the magnets' single-ring onset less 10 °C; not a number if any is."),
            cup_fit_allowed_C: f64 => out_rust_only("°C", "Cup fitting temperature allowed",
                "Your limit for the bare cup, which is fitted before its magnets are bonded."),
            check: String => out_rust_only("", "Shaft joints: keyed shrink fits",
                "Each rotor's shrink fit must hold cold-high times the safety factor when loosest, stay elastic when tightest and be fitted within the allowed temperature; the key alone is a backup."),
        }
    }
}

/// The shafts' elastic modulus [GPa]: a generic alloy steel (the shafts' steels are not known).
pub const SHAFT_MODULUS_GPA: f64 = 205.0;

/// Poisson's ratio of the shafts and the parts.
pub const POISSON: f64 = 0.30;

/// The shafts' expansion coefficient [1/°C], from [`NIST_FIT_TOP_K`] up.
pub const SHAFT_CTE_PER_C: f64 = 11.5e-6;

/// Room temperature [°C]: the measured interference's temperature, and where a shaft's shrink
/// is zero.
pub const ROOM_C: f64 = 20.0;

/// The part's stress limit is its yield strength divided by this (DIN 7190 practice is 1.1 to
/// 1.2; the physics review confirms it). The key flank is held to the same limit.
pub const YIELD_FACTOR: f64 = 1.2;

/// NIST's fit of the linear expansion of nickel steel Fe-2.25Ni, the nearest ferritic steel
/// NIST lists: y = (L − L293) / L293 × 10⁵ = a0 + a1·T + a2·T² + a3·T³ + a4·T⁴, T in K, for
/// 4 to [`NIST_FIT_TOP_K`] (1.5 % fit error;
/// trc.nist.gov/cryogenics/materials/NickelSteel/NickelSteel_Fe2.25Ni)_rev.htm).
pub const NIST_FE_2_25NI: [f64; 5] = [-210.4, -0.05699, 5.072e-3, -1.381e-5, 1.897e-8];

/// The top of the NIST fit's range [K]. The shrink is the fit below it, anchored here, and
/// [`SHAFT_CTE_PER_C`] above it.
pub const NIST_FIT_TOP_K: f64 = 265.0;

/// 0 °C in kelvin.
const ZERO_C_IN_K: f64 = 273.15;

/// The inner hub's name in the check's texts.
pub const INNER_HUB: &str = "inner hub";

/// The cup's name in the check's texts.
pub const CUP: &str = "cup";

/// `shaft_joint.check` when both joints hold, stay within the stress limit, can be fitted and
/// their keys pass.
pub const SHAFT_JOINTS_PASS: &str = "Both shaft joints hold and can be fitted";

/// The unknown verdicts, each the prefix [`SHAFT_JOINT_UNKNOWN`] with its reason.
macro_rules! unknown {
    ($reason:literal) => {
        concat!("Shaft joint check unknown", $reason)
    };
}

/// The start of every unknown `shaft_joint.check`.
pub const SHAFT_JOINT_UNKNOWN: &str = unknown!("");

/// `shaft_joint.check` when a value it reads, or a term a judged joint compares, is not a
/// number.
pub const UNKNOWN_NOT_A_NUMBER: &str = unknown!(": a value is not a number");

/// The start of `shaft_joint.check` when a joint has no wall around its bore and no judged
/// joint fails; " ({joint})" follows.
pub const UNKNOWN_NO_WALL: &str = unknown!(": no wall around the bore");

/// The start of `shaft_joint.check` when a joint's minimum interference is above its maximum
/// and no judged joint fails; " ({joint})" follows.
pub const UNKNOWN_BAND_REVERSED: &str = unknown!(": the minimum interference is above the maximum");

/// `shaft_joint.check` when the safety factor is not above zero, a typed value no slider reaches
/// (ruling C-4): the torque each joint must hold would be zero or below, and a key pressure below
/// zero would pass.
pub const UNKNOWN_SAFETY_FACTOR: &str = unknown!(": the safety factor is not above zero");

/// The start of `shaft_joint.check` when a fit slips; " ({joint}): holds {holding} N·m against
/// {required} N·m" follows.
pub const FIT_SLIPS: &str = "The fit slips";

/// The start of `shaft_joint.check` when a bore stress is above its limit; " ({joint}): {stress}
/// MPa against {limit} MPa" follows.
pub const BORE_OVER_ITS_LIMIT: &str = "Bore stress above its limit";

/// The start of `shaft_joint.check` when a part cannot be fitted at its allowed temperature;
/// " ({joint}): needs {needed} °C against {allowed} °C" follows.
pub const CANNOT_BE_FITTED: &str = "Cannot be fitted at the allowed temperature";

/// The start of `shaft_joint.check` when the key alone would bear more than the limit;
/// " ({joint}): {pressure} MPa against {limit} MPa" follows.
pub const KEY_OVER_ITS_LIMIT: &str = "Key backup above its limit";

/// The start of `shaft_joint.check` when the key has no bearing length or no contact height;
/// " ({joint})" follows.
pub const KEY_HAS_NO_BEARING_AREA: &str = "The key has no bearing area";

/// The four terms each judged joint must pass, in the badge's order (spec section 4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The fit alone holds less than the required torque at its loosest.
    Slips,
    /// The bore stress at the tightest fit is above the stress limit.
    BoreStress,
    /// The temperature the fit would need is above the allowed one.
    CannotBeFitted,
    /// The key alone has no bearing area, or bears more than the stress limit.
    Key,
}

/// [`Cause`] in the badge's order.
pub const CAUSES: [Cause; 4] = [
    Cause::Slips,
    Cause::BoreStress,
    Cause::CannotBeFitted,
    Cause::Key,
];

/// Why a joint has no wall around its bore: its Lamé terms then mean nothing and it is not
/// judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoWall {
    /// The bore is as wide as the part (bore over outer diameter at least 1), or the part's
    /// diameter is typed at or below zero (ruling C-3).
    BoreAsWideAsPart,
    /// The keyway is at least as deep as the wall around the bore is thick.
    KeywayThroughWall,
}

/// Which of the three values caps the bonded inner hub's allowed fitting temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HubFitLimit {
    /// The owner's input, `shaft_joint.hub_fit_max_C`.
    Yours,
    /// The selected adhesive's design limit, `temperature.adhesive.design_limit_C`.
    Adhesive,
    /// The inner ring's own single-ring onset less [`CURE_MARGIN_MIN_C`],
    /// `temperature.demag.inner_onset_single_ring_C`.
    MagnetOnset,
}

/// The bonded inner hub's allowed fitting temperature [°C] (owner R-15): the lowest of the
/// owner's input, the selected adhesive's design limit and the inner ring's own single-ring
/// onset less [`CURE_MARGIN_MIN_C`], with the limit that binds (ties name the earlier: yours,
/// the adhesive's, the onset's). NaN and no limit when any of the three is NaN: `py_min` drops
/// a NaN in its second argument and `f64::min` in either, so no order makes a bare min safe.
/// An onset of +inf (ferrite) never binds.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn hub_fit_allowed(input_C: f64, adhesive_C: f64, onset_C: f64) -> (f64, Option<HubFitLimit>) {
    if input_C.is_nan() || adhesive_C.is_nan() || onset_C.is_nan() {
        return (f64::NAN, None);
    }
    // py_min returns one of its arguments bit for bit, so the equalities below are exact.
    let allowed = py_min(py_min(input_C, adhesive_C), onset_C - CURE_MARGIN_MIN_C);
    let limit = if allowed == input_C {
        HubFitLimit::Yours
    } else if allowed == adhesive_C {
        HubFitLimit::Adhesive
    } else {
        HubFitLimit::MagnetOnset
    };
    (allowed, Some(limit))
}

/// NIST's Fe-2.25Ni fit at `kelvin`: (L − L293) / L293 × 10⁵.
fn nist_y(kelvin: f64) -> f64 {
    let [a0, a1, a2, a3, a4] = NIST_FE_2_25NI;
    a0 + a1 * kelvin + a2 * kelvin.powi(2) + a3 * kelvin.powi(3) + a4 * kelvin.powi(4)
}

/// The diametral shrink [µm] of a steel shaft `bore_mm` across, taken from [`ROOM_C`] to
/// `shaft_C`: positive when colder, negative above 20 °C (it grows). [`SHAFT_CTE_PER_C`] from
/// [`NIST_FIT_TOP_K`] up; below it NIST's Fe-2.25Ni fit, anchored at [`NIST_FIT_TOP_K`], so the
/// shrink is continuous there (a constant coefficient overstates the shrink in liquid nitrogen
/// by about 31 %).
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn shaft_shrink_um(shaft_C: f64, bore_mm: f64) -> f64 {
    let kelvin = shaft_C + ZERO_C_IN_K;
    let strain = if kelvin >= NIST_FIT_TOP_K {
        SHAFT_CTE_PER_C * (ROOM_C - shaft_C)
    } else {
        SHAFT_CTE_PER_C * (ROOM_C - (NIST_FIT_TOP_K - ZERO_C_IN_K))
            + (nist_y(NIST_FIT_TOP_K) - nist_y(kelvin)) * 1e-5
    };
    strain * bore_mm * 1000.0
}

/// The engine values the check reads (spec section 4.4's fifteen), filled by `api::compute`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct ShaftJointLinks {
    /// Cold-high [N·m] (`metal.torque_cold_high_Nm`).
    pub cold_high_Nm: f64,
    /// Both joints' bore [mm] (`coupling.bore_mm`).
    pub bore_mm: f64,
    /// Both joints' keyway depth [mm] (`coupling.keyway_depth_mm`); 0 for no keyway.
    pub keyway_depth_mm: f64,
    /// The inner hub's length in effect [mm] (`housing.hub_length_mm`: the magnets' length
    /// under the axial override).
    pub hub_length_mm: f64,
    /// The inner magnets' back apothem, bondline included [mm] (`coupling.inner_back_apothem_mm`).
    pub inner_back_apothem_mm: f64,
    /// The inner bondline [mm] (`metal.bond_inner_mm`).
    pub bond_inner_mm: f64,
    /// The cup's boss length [mm] (`metal.boss_length_mm`).
    pub boss_length_mm: f64,
    /// The cup's rear web thickness [mm] (`metal.web_mm`): the bore runs through it too.
    pub web_mm: f64,
    /// The cup's boss outside diameter [mm] (`metal.boss_od_mm`).
    pub boss_od_mm: f64,
    /// The minimum magnet temperature [°C] (`metal.min_temp_C`): the cold end.
    pub min_temp_C: f64,
    /// The selected adhesive's design limit [°C] (`temperature.adhesive.design_limit_C`): the
    /// hot end, and a cap on the bonded inner hub's fitting temperature.
    pub adhesive_limit_C: f64,
    /// The inner ring's own single-ring onset [°C]
    /// (`temperature.demag.inner_onset_single_ring_C`): a cap on the bonded inner hub's fitting
    /// temperature, less [`CURE_MARGIN_MIN_C`].
    pub inner_onset_single_ring_C: f64,
    /// The inner hub and cup material's elastic modulus [GPa] (`PartProperties::hub`).
    pub part_modulus_GPa: f64,
    /// The inner hub and cup material's expansion coefficient [1/°C] (`PartProperties::hub`).
    pub part_cte_per_C: f64,
    /// The inner hub and cup material's library yield strength [MPa] (`PartProperties::hub`).
    pub part_yield_MPa: f64,
}

impl ShaftJointLinks {
    /// The fifteen values, for the unknown test.
    pub fn values(&self) -> [f64; 15] {
        [
            self.cold_high_Nm,
            self.bore_mm,
            self.keyway_depth_mm,
            self.hub_length_mm,
            self.inner_back_apothem_mm,
            self.bond_inner_mm,
            self.boss_length_mm,
            self.web_mm,
            self.boss_od_mm,
            self.min_temp_C,
            self.adhesive_limit_C,
            self.inner_onset_single_ring_C,
            self.part_modulus_GPa,
            self.part_cte_per_C,
            self.part_yield_MPa,
        ]
    }
}

/// One joint's own values; [`JointCommon`] holds what both joints share.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointSpec {
    /// [`INNER_HUB`] or [`CUP`].
    pub name: &'static str,
    /// The part's outer diameter around the bore [mm]: the inner hub's flats' incircle, the
    /// cup's boss (both conservative).
    pub outer_diameter_mm: f64,
    /// The engaged length [mm]: the inner hub's length, the cup's boss and web.
    pub length_mm: f64,
    /// The band's minimum measured interference [µm].
    pub interference_min_um: f64,
    /// The band's maximum measured interference [µm].
    pub interference_max_um: f64,
    /// The hottest the part may be at fitting [°C].
    pub fit_allowed_C: f64,
    /// Its shaft's temperature at fitting [°C].
    pub shaft_fit_C: f64,
}

/// What both joints share.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointCommon {
    /// The bore [mm].
    pub bore_mm: f64,
    /// The keyway depth [mm]; 0 for no keyway (no key term and no key arc).
    pub keyway_depth_mm: f64,
    /// The key's width [mm].
    pub key_width_mm: f64,
    /// The key face's contact height [mm], before the cap at the keyway depth.
    pub key_contact_mm: f64,
    /// The fit's friction coefficient.
    pub friction: f64,
    /// The interference lost to smoothing at fitting [µm].
    pub smoothing_um: f64,
    /// How much hotter the rotor runs than its shaft at the hot end [°C].
    pub rotor_over_shaft_C: f64,
    /// The clearance each part needs over its shaft to slide on [µm].
    pub joining_clearance_um: f64,
    /// The parts' elastic modulus [GPa].
    pub part_modulus_GPa: f64,
    /// The parts' expansion coefficient [1/°C].
    pub part_cte_per_C: f64,
    /// The cold end [°C]: the minimum magnet temperature.
    pub cold_C: f64,
    /// The hot end [°C]: the selected adhesive's design limit.
    pub hot_C: f64,
    /// The torque each fit, and each key alone, must hold [N·m].
    pub required_Nm: f64,
    /// The bore stress limit, and the key flank's [MPa].
    pub stress_limit_MPa: f64,
}

/// One joint's terms (spec sections 3.1 and 3.3), computed whether or not the joint is judged.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointTerms {
    /// [`INNER_HUB`] or [`CUP`].
    pub name: &'static str,
    /// Why the joint has no wall around its bore; `None` when it has one.
    pub no_wall: Option<NoWall>,
    /// The band's minimum is above its maximum.
    pub band_reversed: bool,
    /// Whether there is a keyway (depth above 0): the key term applies and its arc is taken off.
    pub keyed: bool,
    /// Contact pressure per µm of effective diametral interference [MPa/µm].
    pub pressure_per_um: f64,
    /// Holding torque per µm of effective interference [N·m/µm].
    pub holding_per_um: f64,
    /// Bore von Mises stress per µm of effective interference [MPa/µm].
    pub stress_per_um: f64,
    /// Effective interference at the loosest [µm]: the minimum less smoothing at the end that
    /// loosens the fit more, floored at 0.
    pub effective_min_um: f64,
    /// Effective interference at the tightest [µm]: the maximum less smoothing at the end that
    /// tightens the fit more, floored at 0.
    pub effective_max_um: f64,
    /// The fit's holding torque at its loosest [N·m]; NaN with no wall.
    pub holding_Nm: f64,
    /// The bore's von Mises stress at its tightest [MPa]; NaN with no wall.
    pub bore_stress_MPa: f64,
    /// The smallest measured minimum that holds the required torque [µm]; NaN with no wall.
    pub needed_um: f64,
    /// The largest measured maximum within the stress limit [µm]; NaN with no wall.
    pub elastic_max_um: f64,
    /// The shaft's diametral shrink at its fitting temperature [µm].
    pub shaft_shrink_um: f64,
    /// The largest measured maximum that can be fitted at the allowed temperature [µm].
    pub fittable_max_um: f64,
    /// The part's temperature the fit would need [°C].
    pub fit_temp_C: f64,
    /// The hottest the part may be at fitting [°C].
    pub fit_allowed_C: f64,
    /// The key's bearing length, the engaged length less the key width [mm].
    pub key_bearing_mm: f64,
    /// The key face's contact height, capped at the keyway depth [mm].
    pub key_contact_mm: f64,
    /// The key flank's pressure were the key to carry the required torque alone [MPa]: 0 with no
    /// keyway, +inf with no bearing area.
    pub key_pressure_MPa: f64,
    /// The torque each fit, and each key alone, must hold [N·m].
    pub required_Nm: f64,
    /// The bore stress limit, and the key flank's [MPa].
    pub stress_limit_MPa: f64,
    /// The slip term's margin [µm]: the measured minimum less the needed one.
    pub slip_margin_um: f64,
    /// The key term's margin [mm]: the bearing length less the length needed at the limit;
    /// −inf with no contact height, +inf with no keyway.
    pub key_margin_mm: f64,
}

impl JointTerms {
    /// Whether the joint is judged: it has a wall and its band is not reversed.
    pub fn determinate(&self) -> bool {
        self.no_wall.is_none() && !self.band_reversed
    }

    /// Whether the key has a bearing area: a positive bearing length and contact height.
    pub fn key_has_bearing_area(&self) -> bool {
        self.key_bearing_mm > 0.0 && self.key_contact_mm > 0.0
    }

    /// The terms the four comparisons read; a judged joint with any of them NaN reads unknown.
    pub fn compared(&self) -> [f64; 7] {
        [
            self.holding_Nm,
            self.required_Nm,
            self.bore_stress_MPa,
            self.stress_limit_MPa,
            self.fit_temp_C,
            self.fit_allowed_C,
            self.key_pressure_MPa,
        ]
    }

    /// Whether the joint fails `cause`'s comparison. Each is written so that a NaN fails it;
    /// equality passes. The key term does not apply without a keyway.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails
    pub fn fails(&self, cause: Cause) -> bool {
        match cause {
            Cause::Slips => !(self.holding_Nm >= self.required_Nm),
            Cause::BoreStress => !(self.bore_stress_MPa <= self.stress_limit_MPa),
            Cause::CannotBeFitted => !(self.fit_temp_C <= self.fit_allowed_C),
            Cause::Key => self.keyed && !(self.key_pressure_MPa <= self.stress_limit_MPa),
        }
    }

    /// `cause`'s margin, positive when it passes (spec section 5.2): the slip term in µm of the
    /// band's minimum, the stress term in MPa, the fitting term in °C, the key term in mm of
    /// bearing length.
    pub fn margin(&self, cause: Cause) -> f64 {
        match cause {
            Cause::Slips => self.slip_margin_um,
            Cause::BoreStress => self.stress_limit_MPa - self.bore_stress_MPa,
            Cause::CannotBeFitted => self.fit_allowed_C - self.fit_temp_C,
            Cause::Key => self.key_margin_mm,
        }
    }

    /// The badge's text for `cause` on this joint, its numbers never equal: the limit quoted
    /// half-up to its last decimal, the failing value pushed one last digit past it.
    pub fn text(&self, cause: Cause) -> String {
        let name = self.name;
        match cause {
            Cause::Slips => {
                let (holding, required) = quoted_below(self.holding_Nm, self.required_Nm, 2);
                format!("{FIT_SLIPS} ({name}): holds {holding} N·m against {required} N·m")
            }
            Cause::BoreStress => {
                let (stress, limit) = quoted_above(self.bore_stress_MPa, self.stress_limit_MPa, 1);
                format!("{BORE_OVER_ITS_LIMIT} ({name}): {stress} MPa against {limit} MPa")
            }
            Cause::CannotBeFitted => {
                let (needed, allowed) = quoted_above(self.fit_temp_C, self.fit_allowed_C, 1);
                format!("{CANNOT_BE_FITTED} ({name}): needs {needed} °C against {allowed} °C")
            }
            Cause::Key if !self.key_has_bearing_area() => {
                format!("{KEY_HAS_NO_BEARING_AREA} ({name})")
            }
            Cause::Key => {
                let (pressure, limit) =
                    quoted_above(self.key_pressure_MPa, self.stress_limit_MPa, 1);
                format!("{KEY_OVER_ITS_LIMIT} ({name}): {pressure} MPa against {limit} MPa")
            }
        }
    }
}

/// A value above its limit as the badge quotes them: (the value, the limit), the limit half-up
/// to `decimals`, the value at least one last digit above the limit as quoted.
fn quoted_above(value: f64, limit: f64, decimals: usize) -> (String, String) {
    let step = 10f64.powi(-(decimals as i32));
    let limit_text = fmt_fixed(limit, decimals);
    let quoted: f64 = limit_text.parse().unwrap_or(limit);
    (
        fmt_fixed(py_max(value, quoted + step), decimals),
        limit_text,
    )
}

/// A value below its need as the badge quotes them: (the value, the need), the need half-up to
/// `decimals`, the value at least one last digit below the need as quoted and not below 0.
/// Where the need itself quotes as 0, the value is quoted 0 and the need one last digit.
fn quoted_below(value: f64, need: f64, decimals: usize) -> (String, String) {
    let step = 10f64.powi(-(decimals as i32));
    let need_text = fmt_fixed(need, decimals);
    let quoted: f64 = need_text.parse().unwrap_or(need);
    if quoted <= 0.0 {
        return (fmt_fixed(0.0, decimals), fmt_fixed(step, decimals));
    }
    (
        fmt_fixed(py_max(0.0, py_min(value, quoted - step)), decimals),
        need_text,
    )
}

/// One joint's terms (spec sections 3.1 and 3.3): plane-stress Lamé thick cylinders on a solid
/// shaft, the key's arc taken off the contact, the temperature ends, the fitting temperature
/// and the key alone.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn joint(spec: &JointSpec, common: &JointCommon) -> JointTerms {
    let d = common.bore_mm;
    let keyed = common.keyway_depth_mm > 0.0;
    // Lamé: contact pressure and bore von Mises stress per µm of effective interference.
    let q = d / spec.outer_diameter_mm;
    let hoop = (1.0 + q * q) / (1.0 - q * q);
    let part_E_MPa = common.part_modulus_GPa * 1000.0;
    let shaft_E_MPa = SHAFT_MODULUS_GPA * 1000.0;
    let pressure_per_um = 1e-3
        / (d * ((1.0 / part_E_MPa) * (hoop + POISSON) + (1.0 / shaft_E_MPa) * (1.0 - POISSON)));
    // The key's slot takes its width off the contact arc (none without a keyway), floored at 0.
    let key_arc_mm = if keyed { common.key_width_mm } else { 0.0 };
    let contact_arc_mm = py_max(0.0, PI * d - key_arc_mm);
    let holding_per_um =
        common.friction * pressure_per_um * contact_arc_mm * spec.length_mm * d / 2.0 / 1000.0;
    let stress_per_um = (hoop * hoop + hoop + 1.0).sqrt() * pressure_per_um;
    // The interference's change at a uniform temperature; at the hot end the rotor also runs the
    // allowance warmer than its shaft.
    let alpha = common.part_cte_per_C;
    let mismatch_um = |t: f64| (SHAFT_CTE_PER_C - alpha) * (t - ROOM_C) * d * 1000.0;
    let rotor_warmer_um = alpha * common.rotor_over_shaft_C * d * 1000.0;
    let loosest_um = py_min(
        mismatch_um(common.cold_C),
        mismatch_um(common.hot_C) - rotor_warmer_um,
    );
    let tightest_um = py_max(mismatch_um(common.cold_C), mismatch_um(common.hot_C));
    let effective_min_um = py_max(
        0.0,
        spec.interference_min_um - common.smoothing_um + loosest_um,
    );
    let effective_max_um = py_max(
        0.0,
        spec.interference_max_um - common.smoothing_um + tightest_um,
    );
    // No wall: the bore is as wide as the part, or the keyway cuts through the wall. The Lamé
    // rows then mean nothing: NaN.
    let wall_mm = (spec.outer_diameter_mm - d) / 2.0;
    // the bore at least as wide as the part (Q >= 1), a part typed at or below zero included
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails it
    let no_wall = if !(spec.outer_diameter_mm > d) {
        Some(NoWall::BoreAsWideAsPart)
    } else if keyed && wall_mm <= common.keyway_depth_mm {
        Some(NoWall::KeywayThroughWall)
    } else {
        None
    };
    let lame = |x: f64| if no_wall.is_some() { f64::NAN } else { x };
    let holding_Nm = lame(effective_min_um * holding_per_um);
    let bore_stress_MPa = lame(effective_max_um * stress_per_um);
    let needed_um = lame(common.required_Nm / holding_per_um + common.smoothing_um - loosest_um);
    let elastic_max_um =
        lame(common.stress_limit_MPa / stress_per_um + common.smoothing_um - tightest_um);
    // Fitting: the part warmed, its shaft at its fitting temperature, the joining clearance
    // open. Measured interference: no smoothing is taken off.
    let shaft_shrink_um = shaft_shrink_um(spec.shaft_fit_C, d);
    let fittable_max_um = alpha * (spec.fit_allowed_C - ROOM_C) * d * 1000.0 + shaft_shrink_um
        - common.joining_clearance_um;
    let fit_temp_C = ROOM_C
        + (spec.interference_max_um + common.joining_clearance_um - shaft_shrink_um)
            / (alpha * d * 1000.0);
    // The key alone (form A: bearing length L − b), its contact capped at the keyway depth.
    let key_bearing_mm = spec.length_mm - common.key_width_mm;
    let key_contact_mm = py_min(common.key_contact_mm, common.keyway_depth_mm);
    let bearing = key_bearing_mm > 0.0 && key_contact_mm > 0.0;
    let key_pressure_MPa = if !keyed {
        0.0
    } else if bearing {
        2.0 * common.required_Nm * 1000.0 / (d * key_contact_mm * key_bearing_mm)
    } else {
        f64::INFINITY
    };
    let key_needed_mm = if key_contact_mm > 0.0 {
        2.0 * common.required_Nm * 1000.0 / (d * key_contact_mm * common.stress_limit_MPa)
    } else {
        f64::INFINITY
    };
    JointTerms {
        name: spec.name,
        no_wall,
        band_reversed: spec.interference_min_um > spec.interference_max_um,
        keyed,
        pressure_per_um,
        holding_per_um,
        stress_per_um,
        effective_min_um,
        effective_max_um,
        holding_Nm,
        bore_stress_MPa,
        needed_um,
        elastic_max_um,
        shaft_shrink_um,
        fittable_max_um,
        fit_temp_C,
        fit_allowed_C: spec.fit_allowed_C,
        key_bearing_mm,
        key_contact_mm,
        key_pressure_MPa,
        required_Nm: common.required_Nm,
        stress_limit_MPa: common.stress_limit_MPa,
        slip_margin_um: spec.interference_min_um - needed_um,
        key_margin_mm: if keyed {
            key_bearing_mm - key_needed_mm
        } else {
            f64::INFINITY
        },
    }
}

/// What the two joints are: what both share, and each joint's own values, inner hub first.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn joint_specs(
    inputs: &ShaftJointInputs,
    links: &ShaftJointLinks,
) -> (JointCommon, [JointSpec; 2]) {
    let common = JointCommon {
        bore_mm: links.bore_mm,
        keyway_depth_mm: links.keyway_depth_mm,
        key_width_mm: inputs.key_width_mm,
        key_contact_mm: inputs.key_contact_mm,
        friction: inputs.friction,
        smoothing_um: inputs.smoothing_um,
        rotor_over_shaft_C: inputs.rotor_over_shaft_C,
        joining_clearance_um: inputs.joining_clearance_um,
        part_modulus_GPa: links.part_modulus_GPa,
        part_cte_per_C: links.part_cte_per_C,
        cold_C: links.min_temp_C,
        hot_C: links.adhesive_limit_C,
        required_Nm: inputs.safety_factor * links.cold_high_Nm,
        stress_limit_MPa: links.part_yield_MPa / YIELD_FACTOR,
    };
    let (hub_allowed_C, _) = hub_fit_allowed(
        inputs.hub_fit_max_C,
        links.adhesive_limit_C,
        links.inner_onset_single_ring_C,
    );
    let hub = JointSpec {
        name: INNER_HUB,
        // The flats' incircle, under the bondline (conservative).
        outer_diameter_mm: 2.0 * (links.inner_back_apothem_mm - links.bond_inner_mm),
        length_mm: links.hub_length_mm,
        interference_min_um: inputs.hub_interference_min_um,
        interference_max_um: inputs.hub_interference_max_um,
        fit_allowed_C: hub_allowed_C,
        shaft_fit_C: inputs.wheel_shaft_fit_C,
    };
    let cup = JointSpec {
        name: CUP,
        // The boss (conservative for the web); the bore runs through the boss and the web.
        outer_diameter_mm: links.boss_od_mm,
        length_mm: links.boss_length_mm + links.web_mm,
        interference_min_um: inputs.cup_interference_min_um,
        interference_max_um: inputs.cup_interference_max_um,
        fit_allowed_C: inputs.cup_fit_max_C,
        shaft_fit_C: inputs.output_shaft_fit_C,
    };
    (common, [hub, cup])
}

/// Both joints' terms, inner hub first.
pub fn joints(inputs: &ShaftJointInputs, links: &ShaftJointLinks) -> [JointTerms; 2] {
    let (common, [hub, cup]) = joint_specs(inputs, links);
    [joint(&hub, &common), joint(&cup, &common)]
}

/// Every term the judged joints fail, in the badge's order: by cause ([`CAUSES`]), then inner
/// hub before cup.
pub fn failing(joints: &[JointTerms; 2]) -> Vec<(Cause, &JointTerms)> {
    CAUSES
        .iter()
        .flat_map(|&cause| {
            joints
                .iter()
                .filter(move |j| j.determinate() && j.fails(cause))
                .map(move |j| (cause, j))
        })
        .collect()
}

/// `shaft_joint.check` (spec section 4.3): unknown when any input or engine value it reads,
/// or any term a judged joint compares, is not a number; then unknown when the safety factor
/// is not above zero (ruling C-4); else the first failing term of a judged joint; else a joint
/// with no wall, then a reversed band, as unknown; else the pass.
pub fn verdict(
    inputs: &ShaftJointInputs,
    links: &ShaftJointLinks,
    joints: &[JointTerms; 2],
) -> String {
    let mut input_nan = false;
    inputs.visit("", &mut |_, _, value| {
        input_nan |= matches!(value, Value::Num(x) if x.is_nan());
    });
    let link_nan = links.values().iter().any(|x| x.is_nan());
    let term_nan = joints
        .iter()
        .filter(|j| j.determinate())
        .any(|j| j.compared().iter().any(|x| x.is_nan()));
    if input_nan || link_nan || term_nan {
        return UNKNOWN_NOT_A_NUMBER.to_owned();
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails it
    if !(inputs.safety_factor > 0.0) {
        return UNKNOWN_SAFETY_FACTOR.to_owned();
    }
    if let Some((cause, joint)) = failing(joints).first() {
        return joint.text(*cause);
    }
    if let Some(joint) = joints.iter().find(|j| j.no_wall.is_some()) {
        return format!("{UNKNOWN_NO_WALL} ({})", joint.name);
    }
    if let Some(joint) = joints.iter().find(|j| j.band_reversed) {
        return format!("{UNKNOWN_BAND_REVERSED} ({})", joint.name);
    }
    SHAFT_JOINTS_PASS.to_owned()
}

/// The shaft joint check of a design.
pub fn compute(inputs: &ShaftJointInputs, links: &ShaftJointLinks) -> ShaftJointResults {
    let [hub, cup] = joints(inputs, links);
    let check = verdict(inputs, links, &[hub, cup]);
    ShaftJointResults {
        required_Nm: hub.required_Nm,
        part_modulus_GPa: links.part_modulus_GPa,
        part_cte_per_C: links.part_cte_per_C,
        part_yield_MPa: links.part_yield_MPa,
        stress_limit_MPa: hub.stress_limit_MPa,
        hub_holding_Nm: hub.holding_Nm,
        cup_holding_Nm: cup.holding_Nm,
        hub_bore_stress_MPa: hub.bore_stress_MPa,
        cup_bore_stress_MPa: cup.bore_stress_MPa,
        hub_key_pressure_MPa: hub.key_pressure_MPa,
        cup_key_pressure_MPa: cup.key_pressure_MPa,
        hub_needed_um: hub.needed_um,
        cup_needed_um: cup.needed_um,
        hub_elastic_max_um: hub.elastic_max_um,
        cup_elastic_max_um: cup.elastic_max_um,
        wheel_shaft_shrink_um: hub.shaft_shrink_um,
        output_shaft_shrink_um: cup.shaft_shrink_um,
        hub_fittable_max_um: hub.fittable_max_um,
        cup_fittable_max_um: cup.fittable_max_um,
        hub_fit_temp_C: hub.fit_temp_C,
        cup_fit_temp_C: cup.fit_temp_C,
        hub_fit_allowed_C: hub.fit_allowed_C,
        cup_fit_allowed_C: cup.fit_allowed_C,
        check,
    }
}
