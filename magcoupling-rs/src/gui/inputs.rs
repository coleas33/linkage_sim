//! The input list of the panel's left side (spec M4 "Layout": "inputs, generated from
//! metadata, grouped as the package groups them ... A Key design group on top").
//!
//! [`InputCatalogue`] is built once from the engine's input metadata: the Key design group
//! ([`KEY_DESIGN`]), then every input in its package group (`coupling`, `metal`,
//! `calibration`, `materials`, `temperature`, `clamps`), each group split into sections by
//! its nested input groups (`coupling.magnets`, `temperature.demag`, ...). Every input is in
//! exactly one section; the Key design inputs are also in their group (decision M41-13), and
//! both rows edit the same value.

use std::sync::OnceLock;

use crate::DesignInputs;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value, input_rows};

/// The Key design group, in order (spec M4 "Layout": face gap, pole count, magnet part, axial
/// length, operating temperature, back iron, cup wall, conductance, measured drag). The axial
/// length is the A-2 override of both rings' length, blank by default (it moves the torque at
/// the default library part; the manual lengths do not); the sizing mode switch sits above
/// the group.
pub const KEY_DESIGN: [&str; 10] = [
    "metal.face_gap_mm",
    "coupling.npole",
    "coupling.magnets.part_inner",
    "coupling.magnets.part_outer",
    "coupling.magnets.axial_length_mm",
    "coupling.op_temp_C",
    "coupling.backiron",
    "metal.cup_wall_corner_mm",
    "temperature.thermal.conductance_W_K",
    "metal.measured_drag_Nm",
];

/// The heading of every input group and nested group, by path prefix, in the package's order.
pub const SECTION_LABELS: [(&str, &str); 18] = [
    ("coupling", "Coupling"),
    ("coupling.magnets", "Magnets"),
    ("metal", "Metal design"),
    ("calibration", "Calibration"),
    ("materials", "Materials"),
    ("materials.steel", "Back-iron steel"),
    ("materials.nickel", "Nickel plating"),
    ("materials.screws", "Screw classes"),
    ("materials.parts", "Part materials"),
    ("temperature", "Temperature design"),
    ("temperature.duty", "Duty"),
    ("temperature.demag", "Demagnetization"),
    ("temperature.adhesive", "Adhesive"),
    ("temperature.mismatch", "Thermal mismatch"),
    ("temperature.slip_loss", "Slip loss"),
    ("temperature.thermal", "Thermal network"),
    ("temperature.adhesive_life", "Adhesive life"),
    ("clamps", "Shaft clamps"),
];

/// Where an optional input starts when the user enters a value (decision M41-12): the result
/// it overrides, unrounded. The axial length override enters at the inner ring's length in
/// use (both rings take it), so nothing moves. The measured drag enters at the model's
/// equivalent mean drag torque; entering any measured drag switches the thermal summary from
/// the not-measured high estimate to the measured branch, so the slip loss and the steady
/// temperatures move (the drag torque itself is the model's).
pub const OPTIONAL_SEEDS: [(&str, &str); 2] = [
    ("coupling.magnets.axial_length_mm", "model.inner_length_mm"),
    ("metal.measured_drag_Nm", "temperature.slip_loss.drag_Nm"),
];

/// One input: its path, metadata and default value.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEntry {
    pub path: String,
    pub meta: &'static InputMeta,
    pub default: Value,
}

/// A run of inputs under one heading: a group's own fields, or one of its nested groups.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSection {
    /// The path prefix (`coupling`, `coupling.magnets`).
    pub prefix: String,
    pub label: &'static str,
    pub entries: Vec<InputEntry>,
}

/// A package group (`coupling`, ...) and its sections, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct InputGroup {
    pub name: String,
    pub label: &'static str,
    pub sections: Vec<InputSection>,
}

/// Every input, arranged for the left side of the panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InputCatalogue {
    /// [`KEY_DESIGN`], in order.
    pub key_design: Vec<InputEntry>,
    /// Every input, by package group and section, in schema order.
    pub groups: Vec<InputGroup>,
}

/// The heading of a path prefix; `None` if [`SECTION_LABELS`] lacks it (a test checks none
/// does).
pub fn section_label(prefix: &str) -> Option<&'static str> {
    SECTION_LABELS
        .iter()
        .find(|(p, _)| *p == prefix)
        .map(|(_, label)| *label)
}

impl InputCatalogue {
    /// The catalogue of the engine's inputs, with the defaults of [`DesignInputs::default`].
    pub fn new() -> Self {
        let rows = input_rows(&DesignInputs::default());
        let entry = |row: &crate::engine::meta::InputRow| InputEntry {
            path: row.path.clone(),
            meta: row.meta,
            default: row.value.clone(),
        };
        let key_design = KEY_DESIGN
            .iter()
            .map(|&path| {
                let row = rows.iter().find(|row| row.path == path);
                entry(row.unwrap_or_else(|| panic!("KEY_DESIGN: no input {path}")))
            })
            .collect();
        let mut groups: Vec<InputGroup> = Vec::new();
        for row in &rows {
            let (prefix, _) = row
                .path
                .rsplit_once('.')
                .expect("every input sits in a group");
            let name = prefix.split('.').next().unwrap_or(prefix);
            if groups.last().is_none_or(|g| g.name != name) {
                groups.push(InputGroup {
                    name: name.to_owned(),
                    label: section_label(name).unwrap_or("Inputs"),
                    sections: Vec::new(),
                });
            }
            let group = groups.last_mut().expect("pushed above");
            if group.sections.last().is_none_or(|s| s.prefix != prefix) {
                group.sections.push(InputSection {
                    prefix: prefix.to_owned(),
                    label: section_label(prefix).unwrap_or("Inputs"),
                    entries: Vec::new(),
                });
            }
            let section = group.sections.last_mut().expect("pushed above");
            section.entries.push(entry(row));
        }
        Self { key_design, groups }
    }

    /// The catalogue, built once: it depends only on the engine's metadata.
    pub fn get() -> &'static InputCatalogue {
        static CATALOGUE: OnceLock<InputCatalogue> = OnceLock::new();
        CATALOGUE.get_or_init(InputCatalogue::new)
    }

    /// Every entry of the package groups, in schema order.
    pub fn all(&self) -> impl Iterator<Item = &InputEntry> {
        self.groups
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
    }

    /// The entry of an input path.
    pub fn entry(&self, path: &str) -> Option<&InputEntry> {
        self.all().find(|entry| entry.path == path)
    }
}

impl Default for InputCatalogue {
    fn default() -> Self {
        Self::new()
    }
}

/// Decimal places of a slider step: the fewest that write the step exactly (0.01 → 2,
/// 0.005 → 3, 2 → 0, 1e-13 → 13). Slider values are rounded to them (decision M41-1), so a
/// value the slider sets is the decimal the user reads (1.41, not 1.4100000000000001), and
/// stepping back to a default lands on it exactly, for every default on its step grid (all
/// but the vacuum permeability's two, a test lists them).
pub fn step_decimals(step: f64) -> usize {
    (0..=15)
        .find(|&decimals| {
            let scaled = step * 10f64.powi(decimals as i32);
            scaled.round() >= 1.0 && (scaled - scaled.round()).abs() <= 1e-9 * scaled
        })
        .unwrap_or(15)
}

/// Whether a number lies outside its slider range (a value from a design file, a share link
/// or the struct: the slider keeps it until edited, decision M41-2, and the row flags it).
pub fn outside_range(range: Option<SliderRange>, value: &Value) -> bool {
    let x = match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        _ => return false,
    };
    range.is_some_and(|r| !(r.min..=r.max).contains(&x))
}

/// The result an optional input starts from ([`OPTIONAL_SEEDS`]).
pub fn optional_seed(path: &str) -> Option<&'static str> {
    OPTIONAL_SEEDS
        .iter()
        .find(|(input, _)| *input == path)
        .map(|(_, result)| *result)
}

/// A short note under a text input saying what the engine makes of the text: whether a part
/// name is a library part, whether a grade name is in the grade table.
pub fn text_hint(path: &str, text: &str) -> Option<&'static str> {
    match path {
        "coupling.magnets.part_inner" | "coupling.magnets.part_outer" => {
            Some(if library::lookup(text).is_some() {
                "Library part"
            } else {
                "Not a library part: the manual dimensions are used"
            })
        }
        "coupling.magnets.grade_inner" | "coupling.magnets.grade_outer" => {
            Some(if text.is_empty() {
                "Blank: the manual Br, no rating"
            } else if grades::grade(text).is_some() {
                "Grade table entry (used with manual dimensions)"
            } else {
                "Not in the grade table: the manual Br, no rating"
            })
        }
        _ => None,
    }
}

/// The hover text of an input: help, path, workbook cell, slider range, default, and whether
/// it is a model assumption.
pub fn input_tooltip(entry: &InputEntry) -> String {
    let meta = entry.meta;
    let mut lines = Vec::new();
    if !meta.help.is_empty() {
        lines.push(meta.help.to_owned());
    }
    lines.push(entry.path.clone());
    lines.push(meta.cell.map_or_else(
        || "Rust-only input (no workbook cell)".to_owned(),
        str::to_owned,
    ));
    if let Some(r) = meta.range {
        let unit = crate::gui::format::with_unit(String::new(), meta.unit);
        let scale = if r.log { ", logarithmic" } else { "" };
        lines.push(format!(
            "Slider {} to {}{unit}, step {}{scale}",
            r.min, r.max, r.step
        ));
    }
    let default = match (&entry.default, meta.ty) {
        (Value::None, FieldType::OptF64) => "blank".to_owned(),
        (Value::Text(text), _) if text.is_empty() => "blank".to_owned(),
        (Value::Int(code), _) if !meta.choices.is_empty() => {
            let choice = meta.choices.iter().find(|(c, _)| c == code);
            choice.map_or_else(|| code.to_string(), |(c, text)| format!("{c} = {text}"))
        }
        (value, _) => crate::gui::format::format_value(value),
    };
    lines.push(format!("Default: {default}"));
    if meta.assumption {
        lines.push("Model assumption (Addendum A3)".to_owned());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::{InputSet, ResultSet};

    #[test]
    fn every_input_is_in_exactly_one_section_in_schema_order() {
        let catalogue = InputCatalogue::new();
        let rows = input_rows(&DesignInputs::default());
        let listed: Vec<&str> = catalogue.all().map(|e| e.path.as_str()).collect();
        let schema: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(listed, schema);
        let names: Vec<&str> = catalogue.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "coupling",
                "metal",
                "calibration",
                "materials",
                "temperature",
                "clamps"
            ]
        );
        for group in &catalogue.groups {
            for section in &group.sections {
                assert!(
                    section.prefix.starts_with(&group.name),
                    "{}",
                    section.prefix
                );
                assert!(!section.entries.is_empty());
                for entry in &section.entries {
                    assert_eq!(entry.path.rsplit_once('.').unwrap().0, section.prefix);
                }
            }
        }
    }

    #[test]
    fn every_section_has_a_heading_and_every_heading_a_section() {
        let catalogue = InputCatalogue::new();
        let mut prefixes: Vec<&str> = catalogue
            .groups
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.sections.iter().map(|s| s.prefix.as_str()))
            })
            .collect();
        prefixes.dedup();
        for prefix in &prefixes {
            assert!(section_label(prefix).is_some(), "no heading for {prefix}");
        }
        for (prefix, _) in SECTION_LABELS {
            assert!(prefixes.contains(&prefix), "unused heading {prefix}");
        }
        assert_eq!(catalogue.groups[0].sections[1].label, "Magnets");
        assert_eq!(InputCatalogue::get(), &catalogue);
    }

    #[test]
    fn the_key_design_group_is_the_spec_list_with_the_axial_length_override() {
        let catalogue = InputCatalogue::new();
        let paths: Vec<&str> = catalogue
            .key_design
            .iter()
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(paths, KEY_DESIGN);
        let axial = catalogue.entry("coupling.magnets.axial_length_mm").unwrap();
        assert_eq!(axial.meta.ty, FieldType::OptF64);
        assert_eq!(axial.default, Value::None, "blank by default");
        // Every Key design entry is the same entry as in its group.
        for key in &catalogue.key_design {
            assert_eq!(Some(key), catalogue.entry(&key.path));
        }
    }

    #[test]
    fn the_inputs_cover_every_field_type_the_panel_draws() {
        let catalogue = InputCatalogue::new();
        let has = |pred: &dyn Fn(&InputEntry) -> bool| catalogue.all().any(pred);
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some_and(|r| r.log)
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && e.meta.choices.is_empty()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && !e.meta.choices.is_empty()
        ));
        assert!(has(&|e| e.meta.ty == FieldType::OptF64));
        assert!(has(&|e| e.meta.ty == FieldType::Text));
        assert!(has(&|e| e.meta.rust_only));
        // No input is of a type the panel has no widget for.
        assert!(!has(&|e| e.meta.ty == FieldType::NumOrText));
        // Every number without choices has a slider range.
        assert!(!has(&|e| matches!(
            e.meta.ty,
            FieldType::F64 | FieldType::I64 | FieldType::OptF64
        ) && e.meta.choices.is_empty()
            && e.meta.range.is_none()));
    }

    #[test]
    fn step_decimals_write_each_step_exactly() {
        for (step, decimals) in [
            (2.0, 0),
            (1000.0, 0),
            (0.5, 1),
            (0.1, 1),
            (0.05, 2),
            (0.01, 2),
            (0.005, 3),
            (0.0001, 4),
            (1e-5, 5),
            (1e-7, 7),
            (1e-8, 8),
            (1e-11, 11),
            (1e-13, 13),
        ] {
            assert_eq!(step_decimals(step), decimals, "{step}");
        }
        for entry in InputCatalogue::new().all() {
            if let Some(r) = entry.meta.range {
                let d = step_decimals(r.step);
                assert!(d < 15, "{}: step {}", entry.path, r.step);
            }
        }
    }

    /// The inputs whose default is off its slider's step grid: the engine's step for the
    /// vacuum permeability (1e-11) is coarser than its default's last digit (1.256637e-6), so
    /// once nudged it never returns to the default (an open item of `04-memory.yaml`: the
    /// engine step should be 1e-12).
    const OFF_GRID_DEFAULTS: [&str; 2] = ["coupling.mu0", "calibration.mu0"];

    #[test]
    fn every_slider_default_is_on_its_step_grid_except_the_listed() {
        // Decision M41-1: a slider stores min + k * step rounded to the step's decimals, so
        // stepping back lands on a default exactly only if the default is such a value.
        let on_grid = |r: SliderRange, x: f64| {
            let k = (x - r.min) / r.step;
            let decimals = step_decimals(r.step);
            (k - k.round()).abs() < 1e-6 && format!("{x:.decimals$}").parse() == Ok(x)
        };
        let catalogue = InputCatalogue::new();
        let mut off_grid = Vec::new();
        for entry in catalogue.all() {
            let x = match entry.default {
                Value::Num(x) => x,
                Value::Int(i) => i as f64,
                _ => continue,
            };
            if let Some(r) = entry.meta.range
                && !on_grid(r, x)
            {
                off_grid.push(entry.path.as_str());
            }
        }
        assert_eq!(off_grid, OFF_GRID_DEFAULTS);
    }

    #[test]
    fn values_outside_the_slider_range_are_flagged() {
        let face_gap = InputCatalogue::new()
            .entry("metal.face_gap_mm")
            .unwrap()
            .meta
            .range;
        assert!(!outside_range(face_gap, &Value::Num(0.3)));
        assert!(!outside_range(face_gap, &Value::Num(5.0)));
        assert!(outside_range(face_gap, &Value::Num(5.0000001)));
        assert!(outside_range(face_gap, &Value::Num(0.29)));
        assert!(outside_range(face_gap, &Value::Int(7)));
        assert!(!outside_range(face_gap, &Value::None));
        assert!(!outside_range(None, &Value::Num(1e9)));
    }

    #[test]
    fn every_optional_input_starts_from_the_result_it_overrides() {
        let catalogue = InputCatalogue::new();
        let results = compute_all(&DesignInputs::default());
        let optional: Vec<&str> = catalogue
            .all()
            .filter(|e| e.meta.ty == FieldType::OptF64)
            .map(|e| e.path.as_str())
            .collect();
        let seeded: Vec<&str> = OPTIONAL_SEEDS.iter().map(|(input, _)| *input).collect();
        assert_eq!(optional, seeded);
        for (input, result) in OPTIONAL_SEEDS {
            assert_eq!(optional_seed(input), Some(result));
            let Some(Value::Num(seed)) = results.get(result) else {
                panic!("{result} is not a number")
            };
            let range = catalogue.entry(input).unwrap().meta.range.unwrap();
            assert!((range.min..=range.max).contains(&seed), "{input}: {seed}");
        }
    }

    #[test]
    fn entering_the_axial_length_at_its_seed_moves_nothing() {
        // Decision M41-12: the override starts at the inner ring's length (both default rings
        // are 12.7 mm B842SH), so the design is unchanged until the slider moves.
        let base = DesignInputs::default();
        let results = compute_all(&base);
        let Some(Value::Num(seed)) = results.get("model.inner_length_mm") else {
            panic!("a number")
        };
        let mut seeded = base.clone();
        seeded
            .set("coupling.magnets.axial_length_mm", Value::Num(seed))
            .unwrap();
        assert_eq!(compute_all(&seeded), results);
    }

    #[test]
    fn text_hints_say_what_the_engine_makes_of_the_text() {
        assert_eq!(
            text_hint("coupling.magnets.part_inner", "B842SH"),
            Some("Library part")
        );
        assert_eq!(
            text_hint("coupling.magnets.part_outer", "b842sh"),
            Some("Not a library part: the manual dimensions are used")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_inner", ""),
            Some("Blank: the manual Br, no rating")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "Y30"),
            Some("Grade table entry (used with manual dimensions)")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "N99"),
            Some("Not in the grade table: the manual Br, no rating")
        );
        assert_eq!(text_hint("metal.face_gap_mm", "1"), None);
        // Every text input has a hint.
        for entry in InputCatalogue::new().all() {
            if entry.meta.ty == FieldType::Text {
                assert!(text_hint(&entry.path, "").is_some(), "{}", entry.path);
            }
        }
    }

    #[test]
    fn tooltips_carry_help_path_cell_range_and_default() {
        let catalogue = InputCatalogue::new();
        assert_eq!(
            input_tooltip(catalogue.entry("metal.face_gap_mm").unwrap()),
            "Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119\n\
             Slider 0.3 to 5 mm, step 0.01\nDefault: 1.400"
        );
        let backiron = input_tooltip(catalogue.entry("coupling.backiron").unwrap());
        assert!(
            backiron.ends_with("Default: 1 = steel circuit"),
            "{backiron}"
        );
        let drag = input_tooltip(catalogue.entry("metal.measured_drag_Nm").unwrap());
        assert!(
            drag.contains("logarithmic") && drag.ends_with("Default: blank"),
            "{drag}"
        );
        let harmonic = input_tooltip(catalogue.entry("coupling.max_harmonic").unwrap());
        assert!(
            harmonic.contains("Rust-only input (no workbook cell)")
                && harmonic.ends_with("Model assumption (Addendum A3)"),
            "{harmonic}"
        );
        let npole = input_tooltip(catalogue.entry("coupling.npole").unwrap());
        assert!(npole.contains("Slider 4 to 40, step 2"), "{npole}");
    }
}
