//! The material, magnet-part and grade pickers (spec Addendum A5 "Per-part material pickers
//! backed by a small library", A6 "Two tables replace the single part list ... Custom
//! dimensions stay available: pick any grade with manual dimensions").
//!
//! The rows stay the inputs' own (M4-1's: a part or a grade is a text the engine looks up, a
//! material a selector code), so a design file or a typed name keeps working; the pickers add
//! what the library knows (decision M43-4):
//!
//! - **Materials** (`materials.parts.*`): each choice of the drop-down shows, on hover, every
//!   property the library holds for it ([`material_properties`]), sourced values only, with
//!   "not sourced" where the data file has none; a line under the row sums up the material
//!   picked ([`material_summary`]).
//! - **Magnet parts** (`coupling.magnets.part_*`): a "pick" drop-down under the text field lists
//!   "Custom dimensions (manual)" (a blank part: the manual dimensions) and the 15 library
//!   parts, each with vendor, shape, dimensions, grade, remanence, rating, coating and
//!   magnetization on hover ([`part_properties`], every approved correction on: E3's N42SH
//!   remanence, E19's vendor rating and grade); a line under it sums up the part in use.
//! - **Grades** (`coupling.magnets.grade_*`): a drop-down of "Blank" and the 17 grades, each
//!   with its table row on hover ([`grade_properties`]); the grade applies to a ring with manual
//!   dimensions, which "Custom dimensions" gives.

use egui::ComboBox;

use crate::engine::deviations::Deviations;
use crate::engine::grades::{GRADES, Grade, GradeFamily, grade};
use crate::engine::library::{self, MAGNET_LIBRARY, MagnetSpec};
use crate::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, Material, SLEEVE_LINER_CHOICES, Sourced, chosen,
};
use crate::engine::meta::Value;
use crate::gui::format::format_value;
use crate::gui::input_ui::{RowEdit, current_text};

/// Each material selector and the library choices behind its codes.
pub const MATERIAL_PICKERS: [(&str, &[(i64, &str)]); 3] = [
    ("materials.parts.back_iron", &BACK_IRON_CHOICES),
    ("materials.parts.sleeve_liner", &SLEEVE_LINER_CHOICES),
    ("materials.parts.cap_housing", &CAP_HOUSING_CHOICES),
];

/// The magnet part inputs.
pub const PART_PATHS: [&str; 2] = ["coupling.magnets.part_inner", "coupling.magnets.part_outer"];

/// The magnet grade inputs.
pub const GRADE_PATHS: [&str; 2] = [
    "coupling.magnets.grade_inner",
    "coupling.magnets.grade_outer",
];

/// The part picker's choice of a blank part (the manual dimensions).
pub const CUSTOM: &str = "Custom dimensions (manual)";

/// The grade picker's choice of a blank grade.
pub const BLANK_GRADE: &str = "Blank: the manual Br, no rating";

/// The part picker's label.
pub const PICK_PART: &str = "Pick a part";

/// The grade picker's label.
pub const PICK_GRADE: &str = "Pick a grade";

/// A number to four significant digits.
fn num(x: f64) -> String {
    format_value(&Value::Num(x))
}

/// A sourced property times `scale` with its unit (`unit` empty: none), or "not sourced".
fn sourced(value: &Sourced, scale: f64, unit: &str) -> String {
    value.value.map_or_else(
        || "not sourced".to_owned(),
        |x| format!("{}{unit}", num(x * scale)),
    )
}

/// The library material a selector `code` picks at the material input `path`.
pub fn material_of(path: &str, code: i64) -> Option<&'static Material> {
    MATERIAL_PICKERS
        .iter()
        .find(|(p, _)| *p == path)
        .and_then(|(_, choices)| chosen(choices, code))
}

/// The line under a material picker: what the material is and the properties the engine reads.
pub fn material_summary(m: &Material) -> String {
    format!(
        "{}; conductivity {}; density {}; expansion {}; modulus {}",
        if m.ferromagnetic {
            "ferromagnetic"
        } else {
            "non-ferromagnetic"
        },
        sourced(&m.sigma_S_m, 1e-6, " MS/m"),
        sourced(&m.density_g_cm3, 1.0, " g/cm³"),
        sourced(&m.cte_1e6_per_K, 1.0, "e-6/K"),
        sourced(&m.modulus_GPa, 1.0, " GPa"),
    )
}

/// Every library property of a material, one per line (a material choice's hover text).
pub fn material_properties(m: &Material) -> String {
    let mut lines = vec![
        m.name.to_owned(),
        m.condition.to_owned(),
        format!(
            "Ferromagnetic: {}",
            if m.ferromagnetic { "yes" } else { "no" }
        ),
        format!(
            "Relative permeability: {} (reference only)",
            sourced(&m.mu_r, 1.0, "")
        ),
        format!("Saturation flux density: {}", sourced(&m.bsat_T, 1.0, " T")),
        format!(
            "Electrical conductivity: {}",
            sourced(&m.sigma_S_m, 1e-6, " MS/m")
        ),
        format!("Density: {}", sourced(&m.density_g_cm3, 1.0, " g/cm³")),
        format!(
            "Expansion coefficient: {}",
            sourced(&m.cte_1e6_per_K, 1.0, "e-6/K")
        ),
        format!("Young's modulus: {}", sourced(&m.modulus_GPa, 1.0, " GPa")),
        format!("Yield strength: {}", sourced(&m.yield_MPa, 1.0, " MPa")),
        format!("Specific heat: {}", sourced(&m.cp_J_kgK, 1.0, " J/(kg·K)")),
    ];
    if let Some(b) = m.design_flux_density_T {
        lines.push(format!("Wall check design flux density: {} T", num(b)));
    }
    if m.needs_plating {
        lines.push("Plain or low-alloy steel: needs plating".to_owned());
    }
    lines.join("\n")
}

/// A part's picker label: `B842SH: K&J block 12.7 × 6.35 × 3.17 mm, N42SH`.
pub fn part_label(spec: &MagnetSpec) -> String {
    format!(
        "{}: {} {} {} × {} × {} mm, {}",
        spec.part,
        spec.vendor,
        spec.shape,
        num(spec.length_mm),
        num(spec.width_mm),
        num(spec.thickness_mm),
        library::grade_id(spec, Deviations::ALL)
    )
}

/// What the calculator uses of a library part, with every approved correction on (a part
/// choice's hover text and the line under the part picker).
pub fn part_properties(spec: &MagnetSpec) -> String {
    let mut lines = vec![
        part_label(spec),
        format!(
            "Remanence at 20 °C: {} T; maximum operating temperature: {} °C",
            num(library::br_T(spec, Deviations::ALL)),
            num(library::tmax_C(spec, Deviations::ALL))
        ),
    ];
    if !spec.coating.is_empty() {
        lines.push(format!("Coating: {}", spec.coating));
    }
    if !spec.magnetization.is_empty() {
        lines.push(format!("Magnetization: {}", spec.magnetization));
    }
    lines.join("\n")
}

fn family(f: GradeFamily) -> &'static str {
    match f {
        GradeFamily::NdFeB => "sintered NdFeB",
        GradeFamily::SmCo2_17 => "sintered Sm2Co17",
        GradeFamily::SmCo1_5 => "sintered SmCo5",
        GradeFamily::Ferrite => "hard ferrite",
        GradeFamily::BondedNdFeB => "bonded NdFeB",
    }
}

/// A grade's picker label: `N42SH: Br 1.300 T, Hcj 1592 kA/m, 150.0 °C`.
pub fn grade_label(g: &Grade) -> String {
    format!(
        "{}: Br {} T, Hcj {} kA/m, {} °C",
        g.name,
        num(g.br_T),
        num(g.hcj20_kA_m),
        num(g.tmax_C)
    )
}

/// A grade's table row, one property per line (a grade choice's hover text).
pub fn grade_properties(g: &Grade) -> String {
    let mut lines = vec![
        format!("{} ({})", g.name, family(g.family)),
        format!("Remanence Br at 20 °C: {} T", num(g.br_T)),
        format!(
            "Intrinsic coercivity Hcj at 20 °C: {} kA/m",
            num(g.hcj20_kA_m)
        ),
        format!("Normal coercivity Hcb: {} kA/m", num(g.hcb_kA_m)),
        format!("Maximum energy product: {} kJ/m³", num(g.bhmax_kJ_m3)),
        format!(
            "Br temperature coefficient: {} %/°C",
            num(g.alpha_br_per_C * 100.0)
        ),
        format!(
            "Hcj temperature coefficient: {} %/°C",
            num(g.beta_hcj_per_C * 100.0)
        ),
        format!("Maximum operating temperature: {} °C", num(g.tmax_C)),
        format!("Density: {} g/cm³", num(g.density_g_mm3 * 1000.0)),
    ];
    if let Some(mu) = g.mu_rec {
        lines.push(format!("Recoil permeability: {}", num(mu)));
    }
    if g.beta_hcj_per_C > 0.0 {
        lines.push(
            "Positive Hcj coefficient: the demagnetization risk is at the cold end".to_owned(),
        );
    }
    lines.join("\n")
}

/// The hover text of a selector choice that a picker describes (a material's properties).
pub fn choice_hover(path: &str, code: i64) -> Option<String> {
    material_of(path, code).map(material_properties)
}

/// Draws the picker of the input at `path` under its row (nothing for an input without one) and
/// returns the edit picked.
pub fn picker_ui(ui: &mut egui::Ui, path: &str, current: &Value) -> Option<RowEdit> {
    if MATERIAL_PICKERS.iter().any(|(p, _)| *p == path) {
        if let Value::Int(code) = current
            && let Some(m) = material_of(path, *code)
        {
            ui.add(
                egui::Label::new(egui::RichText::new(material_summary(m)).small().weak()).wrap(),
            );
        }
        return None;
    }
    if PART_PATHS.contains(&path) {
        return part_picker(ui, current_text(current));
    }
    if GRADE_PATHS.contains(&path) {
        return grade_picker(ui, current_text(current));
    }
    None
}

fn part_picker(ui: &mut egui::Ui, current: &str) -> Option<RowEdit> {
    let spec = library::lookup(current);
    let shown = match spec {
        Some(spec) => part_label(spec),
        None if current.is_empty() => CUSTOM.to_owned(),
        None => format!("{current} (not a library part: manual dimensions)"),
    };
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.weak(PICK_PART);
        ComboBox::from_id_salt("part_picker")
            .selected_text(shown)
            .width(ui.available_width())
            .truncate()
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current.is_empty(), CUSTOM)
                    .on_hover_text("The manual dimensions below, with a grade or a manual Br")
                    .clicked()
                {
                    picked = Some(String::new());
                }
                for spec in &MAGNET_LIBRARY {
                    if ui
                        .selectable_label(current == spec.part, part_label(spec))
                        .on_hover_text(part_properties(spec))
                        .clicked()
                    {
                        picked = Some(spec.part.to_owned());
                    }
                }
            });
    });
    if let Some(spec) = spec {
        ui.add(egui::Label::new(egui::RichText::new(part_properties(spec)).small().weak()).wrap());
    }
    picked
        .filter(|p| p != current)
        .map(|p| RowEdit::Set(Value::Text(p)))
}

fn grade_picker(ui: &mut egui::Ui, current: &str) -> Option<RowEdit> {
    let shown = match grade(current) {
        Some(g) => grade_label(g),
        None if current.is_empty() => BLANK_GRADE.to_owned(),
        None => format!("{current} (not in the grade table)"),
    };
    let mut picked = None;
    ui.horizontal(|ui| {
        ui.weak(PICK_GRADE);
        ComboBox::from_id_salt("grade_picker")
            .selected_text(shown)
            .width(ui.available_width())
            .truncate()
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(current.is_empty(), BLANK_GRADE)
                    .clicked()
                {
                    picked = Some(String::new());
                }
                for g in &GRADES {
                    if ui
                        .selectable_label(current == g.id, grade_label(g))
                        .on_hover_text(grade_properties(g))
                        .clicked()
                    {
                        picked = Some(g.id.to_owned());
                    }
                }
            });
    });
    if let Some(g) = grade(current) {
        ui.add(egui::Label::new(egui::RichText::new(grade_properties(g)).small().weak()).wrap());
    }
    picked
        .filter(|p| p != current)
        .map(|p| RowEdit::Set(Value::Text(p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::inputs::InputCatalogue;

    #[test]
    fn the_material_pickers_are_the_three_selectors_and_their_library_choices() {
        for (path, choices) in MATERIAL_PICKERS {
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("a material selector");
            let codes: Vec<i64> = entry.meta.choices.iter().map(|(c, _)| *c).collect();
            let library: Vec<i64> = choices.iter().map(|(c, _)| *c).collect();
            assert_eq!(codes, library, "{path}");
            for &(code, id) in choices.iter() {
                let m = material_of(path, code).unwrap();
                assert_eq!(m.id, id);
                assert!(choice_hover(path, code).unwrap().starts_with(m.name));
            }
            assert_eq!(material_of(path, 0), None);
        }
        assert_eq!(material_of("coupling.backiron", 1), None);
    }

    #[test]
    fn a_material_s_summary_and_properties_say_what_the_library_holds() {
        let steel = material_of("materials.parts.back_iron", 1).unwrap();
        assert_eq!(
            material_summary(steel),
            "ferromagnetic; conductivity 4.330 MS/m; density 7.850 g/cm³; expansion 12.20e-6/K; modulus 205.0 GPa"
        );
        let props = material_properties(steel);
        assert!(
            props.contains("Saturation flux density: not sourced"),
            "{props}"
        );
        assert!(
            props.contains("Wall check design flux density: 1.500 T"),
            "{props}"
        );
        assert!(props.contains("needs plating"), "{props}");
        let al = material_of("materials.parts.back_iron", 8).unwrap();
        assert!(material_summary(al).starts_with("non-ferromagnetic"));
    }

    #[test]
    fn parts_and_grades_read_with_every_correction_on() {
        let b842sh = library::lookup("B842SH").unwrap();
        assert_eq!(
            part_label(b842sh),
            "B842SH: K&J block 12.70 × 6.350 × 3.170 mm, N42SH"
        );
        // E3: the N42SH parts take the grade's 1.30 T, not the workbook row's 1.29 T.
        assert!(part_properties(b842sh).contains("Remanence at 20 °C: 1.300 T"));
        let y30 = grade("Y30").unwrap();
        assert!(grade_properties(y30).contains("demagnetization risk is at the cold end"));
        assert!(grade_label(grade("N42SH").unwrap()).starts_with("N42SH: Br 1.300 T"));
    }

    #[test]
    fn every_picker_text_has_glyphs_in_the_default_fonts() {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut texts = vec![CUSTOM.to_owned(), BLANK_GRADE.to_owned()];
        for (path, choices) in MATERIAL_PICKERS {
            for &(code, _) in choices {
                let m = material_of(path, code).unwrap();
                texts.push(material_summary(m));
                texts.push(material_properties(m));
            }
        }
        for spec in &MAGNET_LIBRARY {
            texts.push(part_properties(spec));
        }
        for g in &GRADES {
            texts.push(grade_label(g));
            texts.push(grade_properties(g));
        }
        for text in &texts {
            crate::gui::test_support::assert_glyphs(&ctx, text, "a picker");
        }
    }
}
