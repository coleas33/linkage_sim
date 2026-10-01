//! The static engine tables a formula can read with `table("name", key, "field")`: the
//! magnet library, the grade table and the back-iron materials, each field as the engine
//! uses it with every approved correction on (the explorer describes what users see).
//!
//! A lookup returns `none` when the key is not in the table (a part name that is not a
//! library part, a blank grade), which is how the engine's own selections branch.

use crate::engine::deviations::Deviations;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::material_library::{BACK_IRON_CHOICES, chosen};
use crate::engine::meta::Value;

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
    TableField { table: "grades", field: "br_T", symbol: "B_{r,grade}", unit: "T", label: "Grade remanence at 20 °C" },
    TableField { table: "grades", field: "alpha_br_per_C", symbol: "α_{grade}", unit: "1/°C", label: "Grade Br temperature coefficient" },
    TableField { table: "back_iron", field: "ferromagnetic", symbol: "ferro", unit: "-", label: "Back-iron material is ferromagnetic (1) or not (0)" },
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
    Ok(match table {
        "magnets" => library::lookup(&text(key)?).map(|spec| {
            Value::Num(match field_name {
                "length_mm" => spec.length_mm,
                "width_mm" => spec.width_mm,
                "thickness_mm" => spec.thickness_mm,
                "br_T" => library::br_T(spec, Deviations::ALL),
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "grades" => grades::grade(&text(key)?).map(|g| {
            Value::Num(match field_name {
                "br_T" => g.br_T,
                "alpha_br_per_C" => g.alpha_br_per_C,
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "back_iron" => {
            // A code: an integer input, or the same number after arithmetic promoted it.
            let code = match key {
                Value::Int(c) => *c,
                Value::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => *x as i64,
                other => return Err(format!("table back_iron: key {other:?} is not a code")),
            };
            chosen(&BACK_IRON_CHOICES, code).map(|m| Value::Int(i64::from(m.ferromagnetic)))
        }
        _ => unreachable!("checked against TABLE_FIELDS"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_reads_and_a_missing_key_is_none() {
        let key = |t: &str| match t {
            "magnets" => Value::Text("B842SH".into()),
            "grades" => Value::Text("N42SH".into()),
            _ => Value::Int(1),
        };
        for f in TABLE_FIELDS {
            assert!(
                lookup(f.table, &key(f.table), f.field).unwrap().is_some(),
                "{}.{}",
                f.table,
                f.field
            );
            let missing = if f.table == "back_iron" {
                Value::Int(99)
            } else {
                Value::Text(String::new())
            };
            assert_eq!(
                lookup(f.table, &missing, f.field).unwrap(),
                None,
                "{}.{}",
                f.table,
                f.field
            );
        }
        assert!(lookup("magnets", &Value::Int(1), "br_T").is_err());
        assert!(lookup("magnets", &key("magnets"), "density").is_err());
        assert!(lookup("nope", &key("magnets"), "br_T").is_err());
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
}
