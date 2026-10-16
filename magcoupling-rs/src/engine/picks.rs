//! Which inputs a part pick sets in place of the user's value (guidance slice 2, spec 3.1 and
//! section 5): a teaching note's "What to change" entry whose input a pick sets is drawn as
//! plain text with the reason, never as a link to a row that changes nothing, and the
//! change-direction test skips it there.
//!
//! The rules are the engine's own: `material_library::resolve` for the back-iron and the
//! sleeve-and-liner picks, `model::resolve_magnets` for each ring's part and grade. The tests
//! nudge every input [`set_by_pick`] names, on every pick, and require that no result moves;
//! and they require every other input a pick leaves without effect to be named here, or listed
//! with the reason it is unused.

use super::api::DesignInputs;
use super::grades;
use super::library;
use super::material_library::{BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, chosen};

/// The back-iron pick: the hub, cup and boss material.
pub const BACK_IRON_PICK: &str = "materials.parts.back_iron";

/// The sleeve-and-liner pick.
pub const SLEEVE_LINER_PICK: &str = "materials.parts.sleeve_liner";

/// The steel values a ferromagnetic back-iron pick other than the default supplies from the
/// library in place of these inputs: density, conductivity, specific heat, expansion and
/// modulus.
const STEEL_INPUTS: [&str; 5] = [
    "metal.steel_density_g_mm3",
    "materials.steel.conductivity_S_m",
    "materials.steel.specific_heat_J_kgK",
    "materials.steel.cte_per_C",
    "materials.steel.modulus_GPa",
];

/// The wall check's design flux density: a back-iron pick with a design flux density of its
/// own (in the library, 1018) supplies it.
const DESIGN_FLUX_INPUT: &str = "materials.steel.bsat_T";

/// The circuit selector: a non-ferromagnetic back-iron pick selects the free-space circuit
/// whatever it says.
const CIRCUIT_INPUT: &str = "coupling.backiron";

/// The sleeve and liner values a pick other than the default supplies: density, conductivity
/// and specific heat.
const SLEEVE_LINER_INPUTS: [&str; 3] = [
    "metal.sleeve_density_g_mm3",
    "temperature.slip_loss.sigma_316_S_m",
    "temperature.thermal.c_316",
];

/// One magnet ring's inputs: its part and its grade, and the manual fields a library part
/// replaces (length, width, thickness, then Br).
struct Ring {
    part: &'static str,
    grade: &'static str,
    manual: [&'static str; 4],
}

/// The inner ring, then the outer ring.
const RINGS: [Ring; 2] = [
    Ring {
        part: "coupling.magnets.part_inner",
        grade: "coupling.magnets.grade_inner",
        manual: [
            "coupling.magnets.manual_inner_length_mm",
            "coupling.magnets.manual_inner_width_mm",
            "coupling.magnets.manual_inner_thickness_mm",
            "coupling.magnets.manual_inner_br_T",
        ],
    },
    Ring {
        part: "coupling.magnets.part_outer",
        grade: "coupling.magnets.grade_outer",
        manual: [
            "coupling.magnets.manual_outer_length_mm",
            "coupling.magnets.manual_outer_width_mm",
            "coupling.magnets.manual_outer_thickness_mm",
            "coupling.magnets.manual_outer_br_T",
        ],
    },
];

/// The pick input whose current choice on `inputs` sets the input at `path` in place of its
/// own value, or `None` when no pick does:
///
/// - a back-iron pick other than the default (4140): a ferromagnetic one supplies the steel's
///   density, conductivity, specific heat, expansion and modulus; a non-ferromagnetic one (304,
///   6061) selects the free-space circuit, so the Back iron selector has no effect; either
///   supplies the wall check's design flux density where the library has one (1018);
/// - a sleeve-and-liner pick other than the default (316L): their density, conductivity and
///   specific heat;
/// - a ring's library part: its manual dimensions, its manual Br and its grade; with no
///   library part, a grade of the grade table: its manual Br.
///
/// The cap pick sets no input on its own (the hub, cup and boss still read the aluminium
/// values it replaces for the cap), and a selector code outside its choices sets none (the
/// results are then not numbers, and validation names the code).
pub fn set_by_pick(inputs: &DesignInputs, path: &str) -> Option<&'static str> {
    let parts = &inputs.materials.parts;
    if parts.back_iron != BACK_IRON_CHOICES[0].0
        && let Some(material) = chosen(&BACK_IRON_CHOICES, parts.back_iron)
    {
        let own = if material.ferromagnetic {
            STEEL_INPUTS.contains(&path)
        } else {
            path == CIRCUIT_INPUT
        };
        let flux = path == DESIGN_FLUX_INPUT && material.design_flux_density_T.is_some();
        if own || flux {
            return Some(BACK_IRON_PICK);
        }
    }
    if parts.sleeve_liner != SLEEVE_LINER_CHOICES[0].0
        && chosen(&SLEEVE_LINER_CHOICES, parts.sleeve_liner).is_some()
        && SLEEVE_LINER_INPUTS.contains(&path)
    {
        return Some(SLEEVE_LINER_PICK);
    }
    let magnets = &inputs.coupling.magnets;
    let picked = [
        (&magnets.part_inner, &magnets.grade_inner),
        (&magnets.part_outer, &magnets.grade_outer),
    ];
    for (ring, (part, grade)) in RINGS.iter().zip(picked) {
        let [.., br] = ring.manual;
        if library::lookup(part).is_some() {
            if path == ring.grade || ring.manual.contains(&path) {
                return Some(ring.part);
            }
        } else if grades::grade(grade).is_some() && path == br {
            return Some(ring.grade);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::engine::api::compute_all;
    use crate::engine::material_library::CAP_HOUSING_CHOICES;
    use crate::engine::meta::{InputSet, ResultRow, Value, input_rows, result_rows};

    /// The cap pick.
    const CAP_PICK: &str = "materials.parts.cap_housing";

    /// Inputs a non-ferromagnetic back iron leaves without effect that no pick sets: the steel
    /// circuit's stored slip-loss fields, unused under the free-space circuit (as Back iron =
    /// no back iron leaves them).
    const UNUSED_WITHOUT_STEEL: [&str; 3] = [
        "temperature.slip_loss.b_hub_T",
        "temperature.slip_loss.b_cup_T",
        "temperature.slip_loss.web_integral_T2m2",
    ];

    /// Any change at all; two NaNs are no change.
    fn changed(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Num(x), Value::Num(y)) => {
                x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan())
            }
            _ => a != b,
        }
    }

    /// The values the input at `path` is nudged to on `inputs`: a slider step both ways and the
    /// slider's ends for a number, a step both ways for a count, every other choice for a
    /// selector, two grades and a blank for a grade; none for a part name or an unset number
    /// (no pick sets either).
    fn nudges(inputs: &DesignInputs, path: &str) -> Vec<Value> {
        let row = input_rows(inputs)
            .into_iter()
            .find(|r| r.path == path)
            .expect("an input");
        match row.value {
            Value::Int(code) if !row.meta.choices.is_empty() => row
                .meta
                .choices
                .iter()
                .map(|&(c, _)| c)
                .filter(|&c| c != code)
                .map(Value::Int)
                .collect(),
            Value::Int(n) => {
                let step = row.meta.range.expect("a count has a slider").step as i64;
                vec![Value::Int(n + step), Value::Int(n - step)]
            }
            Value::Num(x) => {
                let r = row.meta.range.expect("a number has a slider");
                [x + r.step, x - r.step, r.min, r.max]
                    .into_iter()
                    .filter(|&y| y != x)
                    .map(Value::Num)
                    .collect()
            }
            Value::Text(_) if row.meta.name.starts_with("grade_") => ["N52", "Y30", ""]
                .map(|g| Value::Text(g.to_owned()))
                .to_vec(),
            Value::Text(_) | Value::None => Vec::new(),
        }
    }

    /// Whether some nudge of the input at `path` changes some result of `inputs`, whose
    /// results are `base`.
    fn moves_a_result(inputs: &DesignInputs, base: &[ResultRow], path: &str) -> bool {
        nudges(inputs, path).into_iter().any(|value| {
            let mut nudged = inputs.clone();
            nudged.set(path, value).is_ok()
                && result_rows(&compute_all(&nudged))
                    .iter()
                    .zip(base)
                    .any(|(a, b)| changed(&a.value, &b.value))
        })
    }

    /// Every pick off its default, one at a time: each other back-iron, sleeve-and-liner and
    /// cap choice; then the inner ring, the outer ring and both on custom dimensions, with a
    /// grade and without.
    fn pick_designs() -> Vec<(String, DesignInputs)> {
        let mut designs = Vec::new();
        for (pick, choices) in [
            (BACK_IRON_PICK, &BACK_IRON_CHOICES[..]),
            (SLEEVE_LINER_PICK, &SLEEVE_LINER_CHOICES[..]),
            (CAP_PICK, &CAP_HOUSING_CHOICES[..]),
        ] {
            for &(code, id) in &choices[1..] {
                let mut inputs = DesignInputs::default();
                inputs.set(pick, Value::Int(code)).unwrap();
                designs.push((format!("{pick} = {id}"), inputs));
            }
        }
        for grade in ["Y30", ""] {
            for rings in [&RINGS[..1], &RINGS[1..], &RINGS[..]] {
                let mut inputs = DesignInputs::default();
                for ring in rings {
                    inputs.set(ring.part, Value::Text(String::new())).unwrap();
                    inputs
                        .set(ring.grade, Value::Text(grade.to_owned()))
                        .unwrap();
                }
                let parts: Vec<&str> = rings.iter().map(|r| r.part).collect();
                designs.push((format!("custom {parts:?}, grade {grade:?}"), inputs));
            }
        }
        designs
    }

    /// The inputs a pick sets on `inputs`, with the pick.
    fn held(inputs: &DesignInputs) -> BTreeSet<(String, &'static str)> {
        input_rows(inputs)
            .into_iter()
            .filter_map(|r| set_by_pick(inputs, &r.path).map(|pick| (r.path, pick)))
            .collect()
    }

    #[test]
    fn every_path_named_here_is_an_input() {
        let paths: BTreeSet<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|r| r.path)
            .collect();
        let named = STEEL_INPUTS
            .iter()
            .chain(&SLEEVE_LINER_INPUTS)
            .chain(&[
                DESIGN_FLUX_INPUT,
                CIRCUIT_INPUT,
                BACK_IRON_PICK,
                SLEEVE_LINER_PICK,
            ])
            .copied()
            .chain(
                RINGS
                    .iter()
                    .flat_map(|r| [r.part, r.grade].into_iter().chain(r.manual)),
            );
        for path in named {
            assert!(paths.contains(path), "{path}");
        }
    }

    #[test]
    fn at_the_default_design_only_the_library_parts_set_inputs() {
        let (inner, outer) = ("coupling.magnets.part_inner", "coupling.magnets.part_outer");
        let want: BTreeSet<(String, &str)> = [
            ("coupling.magnets.manual_inner_length_mm", inner),
            ("coupling.magnets.manual_inner_width_mm", inner),
            ("coupling.magnets.manual_inner_thickness_mm", inner),
            ("coupling.magnets.manual_inner_br_T", inner),
            ("coupling.magnets.grade_inner", inner),
            ("coupling.magnets.manual_outer_length_mm", outer),
            ("coupling.magnets.manual_outer_width_mm", outer),
            ("coupling.magnets.manual_outer_thickness_mm", outer),
            ("coupling.magnets.manual_outer_br_T", outer),
            ("coupling.magnets.grade_outer", outer),
        ]
        .into_iter()
        .map(|(path, pick)| (path.to_owned(), pick))
        .collect();
        assert_eq!(held(&DesignInputs::default()), want);
    }

    #[test]
    fn each_material_pick_names_what_it_sets() {
        let back_iron = |code: i64| {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.back_iron = code;
            inputs
        };
        let set = |inputs: &DesignInputs, pick: &str| -> BTreeSet<String> {
            held(inputs)
                .into_iter()
                .filter(|(_, p)| *p == pick)
                .map(|(path, _)| path)
                .collect()
        };
        let strings = |paths: &[&str]| -> BTreeSet<String> {
            paths.iter().map(|p| (*p).to_owned()).collect()
        };
        for &(code, id) in &BACK_IRON_CHOICES[1..] {
            let material = chosen(&BACK_IRON_CHOICES, code).unwrap();
            let mut want = if material.ferromagnetic {
                strings(&STEEL_INPUTS)
            } else {
                strings(&[CIRCUIT_INPUT])
            };
            if material.design_flux_density_T.is_some() {
                want.insert(DESIGN_FLUX_INPUT.to_owned());
            }
            assert_eq!(set(&back_iron(code), BACK_IRON_PICK), want, "{id}");
        }
        // 1018 is the one steel with a design flux density of its own; 304 and 6061 are the two
        // non-ferromagnetic picks.
        assert!(set(&back_iron(2), BACK_IRON_PICK).contains(DESIGN_FLUX_INPUT));
        assert_eq!(
            set(&back_iron(7), BACK_IRON_PICK),
            strings(&[CIRCUIT_INPUT])
        );
        for &(code, id) in &SLEEVE_LINER_CHOICES[1..] {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.sleeve_liner = code;
            assert_eq!(
                set(&inputs, SLEEVE_LINER_PICK),
                strings(&SLEEVE_LINER_INPUTS),
                "{id}"
            );
        }
        for &(code, _) in &CAP_HOUSING_CHOICES[1..] {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.cap_housing = code;
            assert_eq!(
                held(&inputs),
                held(&DesignInputs::default()),
                "the cap sets nothing"
            );
        }
        // A grade on custom dimensions supplies the Br; the dimensions stay the user's.
        let mut y30 = DesignInputs::default();
        y30.coupling.magnets.part_inner.clear();
        y30.coupling.magnets.grade_inner = "Y30".to_owned();
        assert_eq!(
            set(&y30, "coupling.magnets.grade_inner"),
            strings(&["coupling.magnets.manual_inner_br_T"])
        );
        assert!(set(&y30, "coupling.magnets.part_inner").is_empty());
    }

    #[test]
    fn a_code_outside_the_choices_sets_nothing() {
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.back_iron = 99;
        inputs.materials.parts.sleeve_liner = 99;
        for path in STEEL_INPUTS
            .iter()
            .chain(&SLEEVE_LINER_INPUTS)
            .chain(&[DESIGN_FLUX_INPUT, CIRCUIT_INPUT])
        {
            assert_eq!(set_by_pick(&inputs, path), None, "{path}");
        }
    }

    #[test]
    fn no_input_a_pick_sets_moves_a_result() {
        let mut designs = pick_designs();
        designs.push(("defaults".to_owned(), DesignInputs::default()));
        let failures: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = designs
                .iter()
                .map(|(label, inputs)| {
                    s.spawn(move || {
                        let base = result_rows(&compute_all(inputs));
                        held(inputs)
                            .into_iter()
                            .filter(|(path, _)| moves_a_result(inputs, &base, path))
                            .map(|(path, pick)| {
                                format!("{label}: {pick} sets {path}, yet it moves a result")
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a design's thread"))
                .collect()
        });
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn every_input_a_pick_leaves_without_effect_is_named() {
        let defaults = DesignInputs::default();
        let base = result_rows(&compute_all(&defaults));
        let live: Vec<String> = input_rows(&defaults)
            .into_iter()
            .map(|r| r.path)
            .filter(|path| moves_a_result(&defaults, &base, path))
            .collect();
        for path in UNUSED_WITHOUT_STEEL {
            assert!(
                live.iter().any(|p| p == path),
                "{path} moves a result at the defaults"
            );
        }
        let designs = pick_designs();
        let failures: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = designs
                .iter()
                .map(|(label, inputs)| {
                    let live = &live;
                    s.spawn(move || {
                        let base = result_rows(&compute_all(inputs));
                        let free_space =
                            chosen(&BACK_IRON_CHOICES, inputs.materials.parts.back_iron)
                                .is_some_and(|m| !m.ferromagnetic);
                        let mut failures = Vec::new();
                        for path in live {
                            let unused =
                                free_space && UNUSED_WITHOUT_STEEL.contains(&path.as_str());
                            let named = set_by_pick(inputs, path).is_some();
                            let moves = moves_a_result(inputs, &base, path);
                            if !moves && !named && !unused {
                                failures.push(format!(
                                    "{label}: {path} moves nothing, and no pick is named for it"
                                ));
                            }
                            if unused && moves {
                                failures.push(format!(
                                    "{label}: {path} is listed unused, yet it moves a result"
                                ));
                            }
                        }
                        failures
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("a design's thread"))
                .collect()
        });
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
