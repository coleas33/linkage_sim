//! Material consequence warnings (spec Addendum A5): six rules, each an engine
//! output with plain-language text, a severity and a teaching-note id (the
//! notes are Addendum A4's; the ids are placeholders until then).
//!
//! Each rule reads the materials IN EFFECT (`material_library::PartProperties`
//! and the inputs), so a value typed into an input can fire a rule as a library
//! pick does. A result is the rule's text when it fires and empty otherwise;
//! [`WARNING_RULES`] carries the severity and note id of each, in result order.
//! The three thresholds are model choices (no source; decision table of the
//! A-1 plan): [`SLEEVE_SIGMA_BASELINE_S_M`], [`LOW_SATURATION_T`],
//! [`CTE_MISMATCH_LIMIT_PER_C`].

use super::meta::{out_rust_only, results};

/// How serious a warning is (the GUI's badge colour).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Red: the coupling does not work as designed (torque).
    Warning,
    /// Amber: a design consequence to check (heat, wall, corrosion, bond stress).
    Caution,
}

/// One rule: its result field, severity, teaching note and text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarningRule {
    /// The result field (`warnings.<id>`).
    pub id: &'static str,
    pub severity: Severity,
    /// Addendum A4 teaching-note id (placeholder until the notes exist).
    pub note_id: &'static str,
    /// What the result reads when the rule fires.
    pub text: &'static str,
}

/// The sleeve conductivity above which slip heating exceeds the workbook design's:
/// 316L, Temperature design!C111's default [S/m]. No listed sleeve exceeds it; a
/// typed conductivity can.
pub const SLEEVE_SIGMA_BASELINE_S_M: f64 = 1.35e6;

/// Saturation below which a back iron is flagged [T]: 1.7 T, the highest design
/// flux density the workbook names for a back-iron steel (Materials!C13's comment,
/// "1018 would be about 1.7 T", which the library keeps as 1018's
/// `design_flux_density_T`, decision 20). It is a design flux density, not a
/// saturation figure: no source gives a saturation threshold, so the rule is a
/// model choice (A-1 plan, decision A10). A steel that saturates below it cannot
/// carry the flux the best-sourced back iron is designed for.
pub const LOW_SATURATION_T: f64 = 1.7;

/// Expansion mismatch between the magnets and the part they bond to above which
/// the bond stress is "higher" [1/°C]: above the default design's 4140 hub
/// (12.3e-6 - (-0.8e-6) = 13.1e-6), below 304 stainless (17.7e-6).
pub const CTE_MISMATCH_LIMIT_PER_C: f64 = 15e-6;

/// The six rules, in the order of [`WarningResults`].
pub const WARNING_RULES: [WarningRule; 6] = [
    WarningRule {
        id: "non_ferromagnetic_back_iron",
        severity: Severity::Warning,
        note_id: "a5.non_ferromagnetic_back_iron",
        text: "Non-ferromagnetic back iron: the magnetic circuit is open, so torque drops, a strong stray field extends outside the coupling, and the part collects ferrous chips and debris.",
    },
    WarningRule {
        id: "ferromagnetic_sleeve_or_liner",
        severity: Severity::Warning,
        note_id: "a5.ferromagnetic_sleeve_or_liner",
        text: "Ferromagnetic sleeve or liner: it short-circuits part of the gap flux, so torque drops, more for a thicker shell.",
    },
    WarningRule {
        id: "high_conductivity_sleeve_or_liner",
        severity: Severity::Caution,
        note_id: "a5.high_conductivity_sleeve_or_liner",
        text: "High-conductivity sleeve or liner: more eddy current than 316L, so more slip heating.",
    },
    WarningRule {
        id: "low_saturation",
        severity: Severity::Caution,
        note_id: "a5.low_saturation",
        text: "Low or unsourced saturation: the wall check assumes a design flux density this back iron may not reach, so its walls need to be thicker.",
    },
    WarningRule {
        id: "uncoated_low_alloy_steel",
        severity: Severity::Caution,
        note_id: "a5.uncoated_low_alloy_steel",
        text: "Uncoated low-alloy steel: it corrodes, so it needs plating; the example design uses electroless nickel.",
    },
    WarningRule {
        id: "cte_mismatch_with_magnets",
        severity: Severity::Caution,
        note_id: "a5.cte_mismatch_with_magnets",
        text: "Large expansion mismatch between the magnets and the hub they are bonded to: higher bond stress over temperature swings.",
    },
];

results! {
    /// The material warnings (Rust-only): each the rule's text when it fires, else empty.
    pub struct WarningResults {
        fields {
            non_ferromagnetic_back_iron: String => out_rust_only("", "Non-ferromagnetic back iron",
                "Fires when the circuit is free space: a non-ferromagnetic back-iron material, or back iron switched off."),
            ferromagnetic_sleeve_or_liner: String => out_rust_only("", "Ferromagnetic sleeve or liner",
                "Fires when the sleeve and liner material is ferromagnetic."),
            high_conductivity_sleeve_or_liner: String => out_rust_only("", "High-conductivity sleeve or liner",
                "Fires when the sleeve conductivity in effect exceeds 316L's 1.35e6 S/m."),
            low_saturation: String => out_rust_only("", "Low saturation",
                "Fires for a steel back iron that saturates below 1.7 T, or has no design flux density of its own: the wall check then uses the design flux density input."),
            uncoated_low_alloy_steel: String => out_rust_only("", "Uncoated low-alloy steel",
                "Fires for a plain or low-alloy steel back iron with a plating thickness of zero."),
            cte_mismatch_with_magnets: String => out_rust_only("", "Expansion mismatch with the magnets",
                "Fires when the hub's expansion coefficient differs from the magnets' by more than 15e-6 /°C."),
        }
    }
}

/// What the rules read: the materials in effect.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct WarningInputs {
    /// Calculator C6 in effect (1 steel circuit; anything else is free space, as the engine takes it).
    pub circuit_backiron: i64,
    /// The sleeve and liner material is ferromagnetic.
    pub sleeve_ferromagnetic: bool,
    /// The sleeve conductivity in effect [S/m].
    pub sleeve_sigma_S_m: f64,
    /// The back-iron code names a library material. False for a code outside its
    /// choices: that is no material (decision D3), so rules 4 and 5 stay silent.
    pub back_iron_known: bool,
    /// The back iron's saturation flux density, where sourced [T].
    pub back_iron_bsat_T: Option<f64>,
    /// The back iron has its own design flux density (the workbook's for 4140, the library's).
    pub back_iron_design_flux_sourced: bool,
    /// The back iron is plain or low-alloy steel.
    pub back_iron_needs_plating: bool,
    /// Electroless nickel thickness (Materials C26) [mm].
    pub plating_mm: f64,
    /// The hub's expansion coefficient in effect [1/°C] (the steel circuit's, or the
    /// body's with no back iron).
    pub hub_cte_per_C: f64,
    /// The magnets' expansion coefficient in the bond plane (Temperature design C95) [1/°C].
    pub magnet_cte_per_C: f64,
}

/// Evaluates the six rules.
pub fn compute(w: &WarningInputs) -> WarningResults {
    let steel = w.circuit_backiron == 1;
    let fired = [
        !steel,
        w.sleeve_ferromagnetic,
        w.sleeve_sigma_S_m > SLEEVE_SIGMA_BASELINE_S_M,
        steel
            && w.back_iron_known
            && (w.back_iron_bsat_T.is_some_and(|b| b < LOW_SATURATION_T)
                || !w.back_iron_design_flux_sourced),
        steel && w.back_iron_known && w.back_iron_needs_plating && w.plating_mm <= 0.0,
        (w.hub_cte_per_C - w.magnet_cte_per_C).abs() > CTE_MISMATCH_LIMIT_PER_C,
    ];
    let text = |i: usize| {
        if fired[i] {
            WARNING_RULES[i].text.to_owned()
        } else {
            String::new()
        }
    };
    WarningResults {
        non_ferromagnetic_back_iron: text(0),
        ferromagnetic_sleeve_or_liner: text(1),
        high_conductivity_sleeve_or_liner: text(2),
        low_saturation: text(3),
        uncoated_low_alloy_steel: text(4),
        cte_mismatch_with_magnets: text(5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::api::{DesignInputs, compute_all};
    use crate::engine::explain::notes::note;
    use crate::engine::material_library::{BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, material};
    use crate::engine::meta::{ResultSet, Value};

    /// The default design's inputs: no rule fires.
    fn quiet() -> WarningInputs {
        WarningInputs {
            circuit_backiron: 1,
            sleeve_ferromagnetic: false,
            sleeve_sigma_S_m: 1.35e6,
            back_iron_known: true,
            back_iron_bsat_T: None,
            back_iron_design_flux_sourced: true,
            back_iron_needs_plating: true,
            plating_mm: 0.015,
            hub_cte_per_C: 12.3e-6,
            magnet_cte_per_C: -0.8e-6,
        }
    }

    /// A change to [`quiet`] that meets one rule's condition.
    type Meet = fn(&mut WarningInputs);

    /// Which rules fired, by index.
    fn fired(w: &WarningInputs) -> Vec<usize> {
        let r = compute(w);
        WARNING_RULES
            .iter()
            .enumerate()
            .filter(|(_, rule)| r.get(rule.id) != Some(Value::Text(String::new())))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn the_rules_are_the_result_fields_in_order() {
        let names: Vec<&str> = WarningResults::FIELDS.iter().map(|m| m.name).collect();
        let ids: Vec<&str> = WARNING_RULES.iter().map(|r| r.id).collect();
        assert_eq!(names, ids);
        for rule in &WARNING_RULES {
            assert!(!rule.text.is_empty() && rule.note_id == format!("a5.{}", rule.id));
        }
        assert!(WarningResults::FIELDS.iter().all(|m| m.rust_only));
    }

    #[test]
    fn each_rule_fires_exactly_on_its_condition() {
        // Spec, Addendum testing: "each warning rule fires exactly on its condition".
        assert_eq!(fired(&quiet()), Vec::<usize>::new());
        let cases: [(usize, Meet); 6] = [
            (0, |w| w.circuit_backiron = 0),
            (1, |w| w.sleeve_ferromagnetic = true),
            (2, |w| w.sleeve_sigma_S_m = 1.3500001e6),
            (3, |w| w.back_iron_bsat_T = Some(1.6)),
            (4, |w| w.plating_mm = 0.0),
            (5, |w| w.hub_cte_per_C = 16.9e-6),
        ];
        for (rule, set) in cases {
            let mut w = quiet();
            set(&mut w);
            assert_eq!(fired(&w), vec![rule], "{}", WARNING_RULES[rule].id);
            assert_eq!(
                compute(&w).get(WARNING_RULES[rule].id),
                Some(Value::Text(WARNING_RULES[rule].text.to_owned()))
            );
        }
        // Low saturation also fires for a back iron with no design flux density of its own.
        let mut w = quiet();
        w.back_iron_design_flux_sourced = false;
        assert_eq!(fired(&w), vec![3]);
        // Plating is needed for plain and low-alloy steel only: unplated stainless is quiet.
        let mut w = quiet();
        w.back_iron_needs_plating = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn an_unknown_back_iron_fires_no_material_rule() {
        // Decision D3: a back-iron code outside its choices is no material, so its missing
        // saturation, design flux density and plating need fire nothing.
        let mut w = quiet();
        w.back_iron_known = false;
        w.back_iron_bsat_T = Some(1.0);
        w.back_iron_design_flux_sourced = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn thresholds_are_strict_at_equality() {
        // Each comparison at exact equality does not fire (the threshold is the last quiet value).
        let mut w = quiet();
        w.sleeve_sigma_S_m = SLEEVE_SIGMA_BASELINE_S_M;
        w.back_iron_bsat_T = Some(LOW_SATURATION_T);
        w.hub_cte_per_C = CTE_MISMATCH_LIMIT_PER_C;
        w.magnet_cte_per_C = 0.0;
        assert_eq!(
            w.hub_cte_per_C - w.magnet_cte_per_C,
            CTE_MISMATCH_LIMIT_PER_C
        );
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn steel_rules_need_the_steel_circuit() {
        // With free space in effect there is no steel back iron to saturate or to plate.
        let mut w = quiet();
        w.circuit_backiron = 0;
        w.back_iron_bsat_T = Some(1.0);
        w.back_iron_design_flux_sourced = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), vec![0]);
        w.circuit_backiron = 7; // an invalid code is free space, as the engine takes it
        assert_eq!(fired(&w), vec![0]);
    }

    // ---- R11 (slice 2): which library picks fire which rule, through the whole engine. The
    // ---- warning notes name these picks, so these tests pin what the notes say.

    /// The rule ids that fire on `inputs`, through the whole engine (`api::compute` feeds the rules
    /// the materials in effect).
    fn fired_on(inputs: &DesignInputs) -> Vec<&'static str> {
        let results = compute_all(inputs);
        WARNING_RULES
            .iter()
            .filter(|rule| {
                results.get(&format!("warnings.{}", rule.id)) != Some(Value::Text(String::new()))
            })
            .map(|rule| rule.id)
            .collect()
    }

    /// The default design with the back iron `code`, the circuit selector `circuit` (1 steel, 0
    /// none) and `plating_mm` of plating per surface.
    fn with_back_iron(code: i64, circuit: i64, plating_mm: f64) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.back_iron = code;
        inputs.coupling.backiron = circuit;
        inputs.materials.nickel.thickness_mm = plating_mm;
        inputs
    }

    /// The back-iron codes whose pick fires `rule` on the circuit `circuit` at `plating_mm`.
    fn back_irons_firing(rule: &str, circuit: i64, plating_mm: f64) -> Vec<i64> {
        BACK_IRON_CHOICES
            .iter()
            .map(|&(code, _)| code)
            .filter(|&code| fired_on(&with_back_iron(code, circuit, plating_mm)).contains(&rule))
            .collect()
    }

    /// The name a note gives a library back iron: its id's leading token ("12L14_cold_drawn" ->
    /// "12L14", "17-4PH_H900" -> "17-4PH").
    fn short_name(code: i64) -> &'static str {
        let id = BACK_IRON_CHOICES
            .iter()
            .find(|&&(c, _)| c == code)
            .map(|&(_, id)| id)
            .unwrap();
        id.split('_').next().unwrap()
    }

    /// The default plating per surface (above zero).
    fn plated() -> f64 {
        DesignInputs::default().materials.nickel.thickness_mm
    }

    #[test]
    fn low_saturation_fires_for_12l14_416_and_both_17_4ph_conditions_only() {
        // R11: on the steel circuit, the back irons with no design flux density of their own or a
        // sourced saturation below LOW_SATURATION_T fire it; with the circuit off nothing can.
        let fires = back_irons_firing("low_saturation", 1, plated());
        assert_eq!(fires, [3, 4, 5, 6]);
        // The same codes from the library's own data: the rule and the data agree.
        let from_data: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .filter(|&&(code, id)| {
                let m = material(id).unwrap();
                m.ferromagnetic
                    && (m.bsat_T.value.is_some_and(|b| b < LOW_SATURATION_T)
                        || (code != BACK_IRON_CHOICES[0].0 && m.design_flux_density_T.is_none()))
            })
            .map(|&(code, _)| code)
            .collect();
        assert_eq!(fires, from_data);
        let names: Vec<&str> = fires.iter().map(|&code| short_name(code)).collect();
        assert_eq!(names, ["12L14", "416", "17-4PH", "17-4PH"]);
        assert!(back_irons_firing("low_saturation", 0, plated()).is_empty());
    }

    #[test]
    fn uncoated_fires_for_4140_1018_and_12l14_at_zero_plating_only() {
        // R11: the plain and low-alloy steels, unplated, on the steel circuit.
        assert!(plated() > 0.0);
        let bare = back_irons_firing("uncoated_low_alloy_steel", 1, 0.0);
        assert_eq!(bare, [1, 2, 3]);
        let from_data: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .filter(|&&(_, id)| {
                let m = material(id).unwrap();
                m.ferromagnetic && m.needs_plating
            })
            .map(|&(code, _)| code)
            .collect();
        assert_eq!(bare, from_data);
        let names: Vec<&str> = bare.iter().map(|&code| short_name(code)).collect();
        assert_eq!(names, ["4140", "1018", "12L14"]);
        assert!(back_irons_firing("uncoated_low_alloy_steel", 1, plated()).is_empty());
        assert!(back_irons_firing("uncoated_low_alloy_steel", 0, 0.0).is_empty());
    }

    #[test]
    fn the_open_circuit_and_expansion_rules_fire_for_304_and_6061_or_the_circuit_switched_off() {
        // R11: a non-magnetic pick selects free space; so does the circuit selector at 0 with any
        // pick. Either way the hub is non-magnetic metal, and its expansion is past the limit.
        let non_magnetic: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .filter(|&&(_, id)| !material(id).unwrap().ferromagnetic)
            .map(|&(code, _)| code)
            .collect();
        assert_eq!(non_magnetic, [7, 8]);
        let names: Vec<&str> = non_magnetic.iter().map(|&code| short_name(code)).collect();
        assert_eq!(names, ["304", "6061"]);
        let every: Vec<i64> = BACK_IRON_CHOICES.iter().map(|&(code, _)| code).collect();
        for rule in ["non_ferromagnetic_back_iron", "cte_mismatch_with_magnets"] {
            assert_eq!(back_irons_firing(rule, 1, plated()), non_magnetic, "{rule}");
            assert_eq!(
                back_irons_firing(rule, 0, plated()),
                every,
                "{rule}: circuit off"
            );
        }
    }

    #[test]
    fn with_plating_only_4140_and_1018_fire_no_warning() {
        // R11: 416, 17-4PH and 12L14 trade the plating caution for the low-saturation one.
        let quiet: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .map(|&(code, _)| code)
            .filter(|&code| fired_on(&with_back_iron(code, 1, plated())).is_empty())
            .collect();
        assert_eq!(quiet, [1, 2]);
    }

    #[test]
    fn no_sleeve_choice_fires_a_sleeve_warning_but_a_316l_conductivity_typed_above_the_threshold_does()
     {
        // R11: every listed sleeve and liner is non-magnetic and conducts no better than 316L.
        for &(code, id) in &SLEEVE_LINER_CHOICES {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.sleeve_liner = code;
            let fired = fired_on(&inputs);
            assert!(fired.is_empty(), "{id}: {fired:?}");
            assert!(!material(id).unwrap().ferromagnetic, "{id}");
        }
        // The threshold is a constant, not the 316L entry: 316L typed above it fires the caution;
        // another pick, which overrides that entry, does not.
        let mut typed = DesignInputs::default();
        typed.temperature.slip_loss.sigma_316_S_m = SLEEVE_SIGMA_BASELINE_S_M * 1.01;
        assert_eq!(fired_on(&typed), ["high_conductivity_sleeve_or_liner"]);
        typed.materials.parts.sleeve_liner = SLEEVE_LINER_CHOICES[1].0;
        assert!(fired_on(&typed).is_empty());
    }

    #[test]
    fn each_back_iron_warning_note_names_the_library_picks_that_fire_it() {
        // The notes list the steels by name; the library decides which fire. A library change that
        // moves a pick in or out of a rule fails here until the note follows.
        let cases = [
            (
                "low_saturation",
                back_irons_firing("low_saturation", 1, plated()),
            ),
            (
                "uncoated_low_alloy_steel",
                back_irons_firing("uncoated_low_alloy_steel", 1, 0.0),
            ),
            (
                "non_ferromagnetic_back_iron",
                back_irons_firing("non_ferromagnetic_back_iron", 1, plated()),
            ),
            (
                "cte_mismatch_with_magnets",
                back_irons_firing("cte_mismatch_with_magnets", 1, plated()),
            ),
        ];
        for (rule_id, codes) in cases {
            let rule = WARNING_RULES.iter().find(|r| r.id == rule_id).unwrap();
            let text = note(rule.note_id).unwrap().sentences.join(" ");
            assert!(!codes.is_empty(), "{rule_id} fires for some pick");
            for code in codes {
                let name = short_name(code);
                assert!(
                    text.contains(name),
                    "{}: the note names {name}",
                    rule.note_id
                );
            }
        }
    }
}
