//! The static engine tables a formula can read with `table("name", key, "field")`: the
//! magnet library, the grade table, the part materials (back iron, sleeve and liner, cap and
//! housing), the aluminium alloys, the adhesives and the clamp screw sizes, each field as the
//! engine uses it with every approved correction on (the explorer describes what users see).
//!
//! A lookup returns `none` when the key is not in the table (a part name that is not a
//! library part, a blank grade, a selector code outside the choices), which is how the
//! engine's own selections branch. A material or adhesive is keyed by its selector code, a
//! screw size by its row of the screw table (0 for M2.5), an alloy by its name.

use crate::engine::clamps::SCREW_SIZES;
use crate::engine::deviations::Deviations;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, SLEEVE_LINER_CHOICES, chosen,
};
use crate::engine::materials::{AL6061, AL7075};
use crate::engine::meta::Value;
use crate::engine::temperature::ADHESIVES;

/// One readable field: its table, name, display symbol and unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableField {
    pub table: &'static str,
    pub field: &'static str,
    /// Symbol markup; the typesetter writes the key after it in parentheses.
    pub symbol: &'static str,
    pub unit: &'static str,
    pub label: &'static str,
}

/// Every field a formula may read. The registry refuses any other `table(...)`.
#[rustfmt::skip]
pub const TABLE_FIELDS: &[TableField] = &[
    TableField { table: "magnets", field: "length_mm", symbol: "L_{lib}", unit: "mm", label: "Library part axial length" },
    TableField { table: "magnets", field: "width_mm", symbol: "w_{lib}", unit: "mm", label: "Library part tangential width" },
    TableField { table: "magnets", field: "thickness_mm", symbol: "t_{lib}", unit: "mm", label: "Library part radial thickness" },
    TableField { table: "magnets", field: "br_T", symbol: "B_{r,lib}", unit: "T", label: "Library part remanence at 20 °C (E3: the N42SH grade's)" },
    TableField { table: "magnets", field: "tmax_C", symbol: "ϑ_{max,lib}", unit: "°C", label: "Library part maximum operating temperature (E19: the vendor's where it differs)" },
    TableField { table: "magnets", field: "grade", symbol: "grade_{lib}", unit: "", label: "Library part grade (E19: the vendor grid's where it differs)" },
    TableField { table: "grades", field: "id", symbol: "grade_{tab}", unit: "", label: "The grade's key, when the table has it" },
    TableField { table: "grades", field: "br_T", symbol: "B_{r,grade}", unit: "T", label: "Grade remanence at 20 °C" },
    TableField { table: "grades", field: "alpha_br_per_C", symbol: "α_{grade}", unit: "1/°C", label: "Grade Br temperature coefficient" },
    TableField { table: "grades", field: "hcj20_kA_m", symbol: "H_{cj,grade}", unit: "kA/m", label: "Grade intrinsic coercivity at 20 °C" },
    TableField { table: "grades", field: "beta_hcj_per_C", symbol: "β_{grade}", unit: "1/°C", label: "Grade Hcj temperature coefficient (positive for hard ferrite)" },
    TableField { table: "grades", field: "tmax_C", symbol: "ϑ_{max,grade}", unit: "°C", label: "Grade maximum operating temperature" },
    TableField { table: "grades", field: "density_g_mm3", symbol: "ρ_{grade}", unit: "g/mm³", label: "Grade density" },
    TableField { table: "back_iron", field: "ferromagnetic", symbol: "ferro", unit: "-", label: "Back-iron material is ferromagnetic (1) or not (0)" },
    TableField { table: "back_iron", field: "sigma_S_m", symbol: "σ_{BI}", unit: "S/m", label: "Back-iron material conductivity (library)" },
    TableField { table: "back_iron", field: "density_g_mm3", symbol: "ρ_{BI}", unit: "g/mm³", label: "Back-iron material density (library)" },
    TableField { table: "back_iron", field: "cp_J_kgK", symbol: "c_{BI}", unit: "J/(kg·K)", label: "Back-iron material specific heat (library)" },
    TableField { table: "back_iron", field: "design_flux_density_T", symbol: "B_{des,BI}", unit: "T", label: "Back-iron material design flux density (decision 20; none where the workbook gives none)" },
    TableField { table: "sleeve_liner", field: "sigma_S_m", symbol: "σ_{SL}", unit: "S/m", label: "Sleeve and liner material conductivity (library)" },
    TableField { table: "sleeve_liner", field: "density_g_mm3", symbol: "ρ_{SL}", unit: "g/mm³", label: "Sleeve and liner material density (library)" },
    TableField { table: "sleeve_liner", field: "cp_J_kgK", symbol: "c_{SL}", unit: "J/(kg·K)", label: "Sleeve and liner material specific heat (library)" },
    TableField { table: "cap_housing", field: "sigma_S_m", symbol: "σ_{cap}", unit: "S/m", label: "Cap material conductivity (library)" },
    TableField { table: "cap_housing", field: "density_g_mm3", symbol: "ρ_{cap}", unit: "g/mm³", label: "Cap material density (library)" },
    TableField { table: "cap_housing", field: "cp_J_kgK", symbol: "c_{cap}", unit: "J/(kg·K)", label: "Cap material specific heat (library)" },
    TableField { table: "aluminium", field: "conductivity_S_m", symbol: "σ_{Al}", unit: "S/m", label: "Aluminium alloy conductivity (Materials C38, C43)" },
    TableField { table: "aluminium", field: "shear_MPa", symbol: "τ_{Al}", unit: "MPa", label: "Aluminium alloy shear strength (Materials C35, C40)" },
    TableField { table: "aluminium", field: "head_pressure_limit_MPa", symbol: "p_{head,Al}", unit: "MPa", label: "Aluminium alloy limiting pressure under a screw head (Materials C36, C41)" },
    TableField { table: "aluminium", field: "key_bearing_allow_MPa", symbol: "p_{key,Al}", unit: "MPa", label: "Aluminium alloy key bearing allowable (Materials C37, C42)" },
    TableField { table: "adhesives", field: "design_limit_C", symbol: "ϑ_{adh}", unit: "°C", label: "Adhesive design limit (Temperature design C65:C74)" },
    TableField { table: "adhesives", field: "cure_C", symbol: "ϑ_{cure}", unit: "°C", label: "Adhesive cure (stress-free) temperature" },
    TableField { table: "adhesives", field: "lap_shear_MPa", symbol: "τ_{lap}", unit: "MPa", label: "Adhesive lap shear at 22 °C (TDS)" },
    TableField { table: "screw_sizes", field: "name", symbol: "size", unit: "", label: "Screw size (Clamp screw sizes row 5)" },
    TableField { table: "screw_sizes", field: "d_mm", symbol: "d", unit: "mm", label: "Screw nominal diameter" },
    TableField { table: "screw_sizes", field: "As_mm2", symbol: "A_s", unit: "mm²", label: "Screw tensile stress area" },
    TableField { table: "screw_sizes", field: "hole_mm", symbol: "d_h", unit: "mm", label: "Clearance hole (ISO 273 medium)" },
    TableField { table: "screw_sizes", field: "head_mm", symbol: "d_k", unit: "mm", label: "Head diameter (ISO 4762 maximum)" },
    TableField { table: "screw_sizes", field: "hex_mm", symbol: "s_{hex}", unit: "mm", label: "Hex key" },
];

/// The field's metadata, or `None` if a formula may not read it.
pub fn field(table: &str, field: &str) -> Option<&'static TableField> {
    TABLE_FIELDS
        .iter()
        .find(|f| f.table == table && f.field == field)
}

/// The value of `field` in `table` at `key`: `Ok(None)` when the key is not in the table,
/// `Err` for a table or field no formula may read, or a key of the wrong type.
pub fn lookup(table: &str, key: &Value, field_name: &str) -> Result<Option<Value>, String> {
    if field(table, field_name).is_none() {
        return Err(if TABLE_FIELDS.iter().any(|f| f.table == table) {
            format!("table {table} has no field {field_name}")
        } else {
            format!("no table {table}")
        });
    }
    let text = |k: &Value| match k {
        Value::Text(s) => Ok(s.clone()),
        other => Err(format!("table {table}: key {other:?} is not a text")),
    };
    // A code: an integer input, or the same number after arithmetic promoted it.
    let code = |k: &Value| match k {
        Value::Int(c) => Ok(*c),
        Value::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => Ok(*x as i64),
        other => Err(format!("table {table}: key {other:?} is not a code")),
    };
    let num = Value::Num;
    Ok(match table {
        "magnets" => library::lookup(&text(key)?).map(|spec| match field_name {
            "length_mm" => num(spec.length_mm),
            "width_mm" => num(spec.width_mm),
            "thickness_mm" => num(spec.thickness_mm),
            "br_T" => num(library::br_T(spec, Deviations::ALL)),
            "tmax_C" => num(library::tmax_C(spec, Deviations::ALL)),
            "grade" => Value::Text(library::grade_id(spec, Deviations::ALL).to_owned()),
            _ => unreachable!("checked against TABLE_FIELDS"),
        }),
        "grades" => grades::grade(&text(key)?).map(|g| match field_name {
            "id" => Value::Text(g.id.to_owned()),
            "br_T" => num(g.br_T),
            "alpha_br_per_C" => num(g.alpha_br_per_C),
            "hcj20_kA_m" => num(g.hcj20_kA_m),
            "beta_hcj_per_C" => num(g.beta_hcj_per_C),
            "tmax_C" => num(g.tmax_C),
            "density_g_mm3" => num(g.density_g_mm3),
            _ => unreachable!("checked against TABLE_FIELDS"),
        }),
        "back_iron" | "sleeve_liner" | "cap_housing" => {
            let choices: &[(i64, &'static str)] = match table {
                "back_iron" => &BACK_IRON_CHOICES,
                "sleeve_liner" => &SLEEVE_LINER_CHOICES,
                _ => &CAP_HOUSING_CHOICES,
            };
            chosen(choices, code(key)?).map(|m| match field_name {
                "ferromagnetic" => Value::Int(i64::from(m.ferromagnetic)),
                "sigma_S_m" => num(m.engine.sigma_S_m),
                "density_g_mm3" => num(m.engine.density_g_mm3),
                "cp_J_kgK" => num(m.engine.cp_J_kgK),
                "design_flux_density_T" => m.design_flux_density_T.map_or(Value::None, num),
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }
        "aluminium" => {
            let name = text(key)?;
            [AL7075, AL6061]
                .into_iter()
                .find(|a| a.name == name)
                .map(|a| {
                    num(match field_name {
                        "conductivity_S_m" => a.conductivity_S_m,
                        "shear_MPa" => a.shear_MPa,
                        "head_pressure_limit_MPa" => a.head_pressure_limit_MPa,
                        "key_bearing_allow_MPa" => a.key_bearing_allow_MPa,
                        _ => unreachable!("checked against TABLE_FIELDS"),
                    })
                })
        }
        "adhesives" => {
            let c = code(key)?;
            (1..=ADHESIVES.len() as i64).contains(&c).then(|| {
                let a = &ADHESIVES[(c - 1) as usize];
                num(match field_name {
                    "design_limit_C" => a.design_limit_C,
                    "cure_C" => a.cure_C,
                    "lap_shear_MPa" => a.lap_shear_MPa,
                    _ => unreachable!("checked against TABLE_FIELDS"),
                })
            })
        }
        "screw_sizes" => {
            let row = code(key)?;
            usize::try_from(row)
                .ok()
                .and_then(|r| SCREW_SIZES.get(r))
                .map(|s| match field_name {
                    "name" => Value::Text(s.name.to_owned()),
                    "d_mm" => num(s.d_mm),
                    "As_mm2" => num(s.As_mm2),
                    "hole_mm" => num(s.hole_mm),
                    "head_mm" => num(s.head_mm),
                    "hex_mm" => num(s.hex_mm),
                    _ => unreachable!("checked against TABLE_FIELDS"),
                })
        }
        _ => unreachable!("checked against TABLE_FIELDS"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key each table holds, and one it does not.
    fn keys(table: &str) -> (Value, Value) {
        match table {
            "magnets" => (Value::Text("B842SH".into()), Value::Text(String::new())),
            "grades" => (Value::Text("N42SH".into()), Value::Text(String::new())),
            "aluminium" => (Value::Text("6061-T6".into()), Value::Text("2024-T3".into())),
            "screw_sizes" => (Value::Int(0), Value::Int(5)),
            _ => (Value::Int(1), Value::Int(99)),
        }
    }

    #[test]
    fn every_field_reads_and_a_missing_key_is_none() {
        for f in TABLE_FIELDS {
            let (held, missing) = keys(f.table);
            assert!(
                lookup(f.table, &held, f.field).unwrap().is_some(),
                "{}.{}",
                f.table,
                f.field
            );
            assert_eq!(
                lookup(f.table, &missing, f.field).unwrap(),
                None,
                "{}.{}",
                f.table,
                f.field
            );
        }
        assert!(lookup("magnets", &Value::Int(1), "br_T").is_err());
        assert!(lookup("magnets", &keys("magnets").0, "density").is_err());
        assert!(lookup("nope", &keys("magnets").0, "br_T").is_err());
        assert!(
            lookup("adhesives", &Value::Text("1".into()), "cure_C").is_err(),
            "a code key"
        );
    }

    #[test]
    fn the_back_iron_table_knows_the_non_ferromagnetic_choices() {
        let ferro = |c| lookup("back_iron", &Value::Int(c), "ferromagnetic").unwrap();
        assert_eq!(ferro(1), Some(Value::Int(1)), "4140");
        let non: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .map(|&(c, _)| c)
            .filter(|&c| ferro(c) == Some(Value::Int(0)))
            .collect();
        assert_eq!(non.len(), 2, "304 and 6061 (spec A5)");
    }

    #[test]
    fn corrected_library_fields_read_as_the_engine_uses_them() {
        // E19: M5045's vendor grid (N50) and every SuperMagnetMan arc's 60 °C.
        let m = |part: &str, f: &str| lookup("magnets", &Value::Text(part.into()), f).unwrap();
        assert_eq!(m("M5045", "grade"), Some(Value::Text("N50".into())));
        assert_eq!(m("M5044", "tmax_C"), Some(Value::Num(60.0)));
        assert_eq!(m("B842SH", "br_T"), Some(Value::Num(1.30)), "E3");
        let g = |f: &str| lookup("grades", &Value::Text("Y30".into()), f).unwrap();
        assert_eq!(
            g("beta_hcj_per_C"),
            Some(Value::Num(0.0035)),
            "ferrite: positive beta"
        );
        assert_eq!(
            lookup("screw_sizes", &Value::Num(2.0), "name").unwrap(),
            Some(Value::Text("M4".into()))
        );
        assert_eq!(
            lookup("adhesives", &Value::Int(2), "cure_C").unwrap(),
            Some(Value::Num(120.0))
        );
    }
}
