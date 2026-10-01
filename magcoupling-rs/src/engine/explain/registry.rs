//! The equation registry: every record parsed and checked once, indexed by target, with the
//! dependency graph the explorer and the traceability test read.
//!
//! Built explicitly ([`Registry::build`]) and owned by the caller (the GUI builds it once at
//! start-up), so the engine keeps no global state. Everything a frame needs is a map lookup
//! plus `ResultSet::get`/`InputSet::get` per term.

use std::collections::{BTreeMap, BTreeSet};

use super::eval::{self, EvalError, TermSource, Trace};
use super::markup::{self, BinOp, Expr, Formula, Func, IndexSet, Symbol};
use super::record::{Eval, Family, Record};
use super::records;
use super::symbols::SYMBOLS;
use super::tables;
use crate::engine::api::{DesignInputs, DesignResults, compute_all};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::{
    InputMeta, InputSet, ResultMeta, ResultSet, Value, input_rows, result_rows,
};
use crate::engine::model::{ODD_HARMONICS, harmonic_count};

/// Unit conversions a term reference may ask for (`{path|unit}`): from, to, factor.
pub const CONVERSIONS: &[(&str, &str, f64)] = &[
    ("mm", "m", 1e-3),
    ("kA/m", "A/m", 1e3),
    ("MPa", "Pa", 1e6),
    ("GPa", "Pa", 1e9),
    ("g", "kg", 1e-3),
];

/// The factor that converts a value in `from` to `to`.
pub fn conversion(from: &str, to: &str) -> Option<f64> {
    CONVERSIONS
        .iter()
        .find(|&&(f, t, _)| f == from && t == to)
        .map(|&(_, _, k)| k)
}

/// A design's values as term values: a result path, else an input path (the two sets of
/// paths are disjoint: `tests/explain.rs`).
pub struct Design<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
}

impl TermSource for Design<'_> {
    fn value(&self, path: &str) -> Option<Value> {
        self.results.get(path).or_else(|| self.inputs.get(path))
    }
}

/// A source restricted to one record's terms (what a custom eval sees).
struct Restricted<'a> {
    inner: &'a dyn TermSource,
    allowed: &'a [String],
}

impl TermSource for Restricted<'_> {
    fn value(&self, path: &str) -> Option<Value> {
        if self.allowed.iter().any(|a| a == path) {
            self.inner.value(path)
        } else {
            None
        }
    }
}

/// One explained result, parsed and checked.
#[derive(Clone, Debug)]
pub struct Equation {
    /// The result path it explains.
    pub target: String,
    /// Its display symbol (symbol markup, index substituted).
    pub symbol: String,
    /// The formula markup as authored (a family's template; see `index`).
    pub source: &'static str,
    /// The family member's index, for a family record.
    pub index: Option<u32>,
    /// The parsed formula: the typesetter's input.
    pub formula: Formula,
    pub eval: Eval,
    /// Every term the formula can read, in first-appearance order: the static dependencies
    /// (every `cases` branch, every harmonic a Σ may include, a Σ's set selector, a table
    /// key). Each is an input or result path.
    pub terms: Vec<String>,
    /// From the result's metadata.
    pub label: &'static str,
    pub unit: &'static str,
    /// Its workbook cell (`None` for a Rust-only result).
    pub cell: Option<String>,
    pub corrections: &'static [DeviationId],
}

/// One row of the equation panel's term list.
#[derive(Clone, Debug, PartialEq)]
pub struct TermRow {
    pub path: String,
    /// Symbol markup.
    pub symbol: String,
    /// The value in `unit` (converted when the formula asks for another unit, `{path|m}`).
    pub value: Option<Value>,
    pub unit: String,
    pub kind: TermKind,
}

/// What a path is to the explorer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    /// A design input or an assumption: a leaf; its slider is highlighted.
    Input { assumption: bool },
    /// A result with an equation record: a click drills into it.
    Explained,
    /// A result without a record: shows its label, value and workbook cell only.
    CellOnly,
}

/// The registry.
pub struct Registry {
    equations: Vec<Equation>,
    by_target: BTreeMap<String, usize>,
    used_by: BTreeMap<String, Vec<String>>,
    symbols: BTreeMap<String, String>,
    family_symbols: BTreeMap<String, String>,
    inputs: BTreeMap<String, &'static InputMeta>,
    results: BTreeMap<String, &'static ResultMeta>,
}

impl Registry {
    /// Builds the registry from every records file. Panics listing every problem;
    /// `tests/explain.rs` keeps the records clean, so a shipped build never does.
    pub fn build() -> Registry {
        match Self::try_build() {
            Ok(r) => r,
            Err(errors) => panic!("equation records:\n{}", errors.join("\n")),
        }
    }

    /// [`Registry::build`], returning every problem instead of panicking.
    pub fn try_build() -> Result<Registry, Vec<String>> {
        let records: Vec<Record> = records::RECORDS
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let families: Vec<Family> = records::FAMILIES
            .iter()
            .flat_map(|f| f.iter().copied())
            .collect();
        Self::from_parts(&records, &families, SYMBOLS)
    }

    /// Builds a registry from the given records and leaf symbols (tests build small ones).
    pub fn from_parts(
        plain: &[Record],
        families: &[Family],
        leaf_symbols: &[(&'static str, &'static str)],
    ) -> Result<Registry, Vec<String>> {
        let defaults = DesignInputs::default();
        let schema_results = compute_all(&defaults);
        let inputs: BTreeMap<String, &'static InputMeta> = input_rows(&defaults)
            .into_iter()
            .map(|r| (r.path, r.meta))
            .collect();
        let result_rows = result_rows(&schema_results);
        let cells: BTreeMap<String, Option<String>> = result_rows
            .iter()
            .map(|r| (r.path.clone(), r.cell.clone()))
            .collect();
        let results: BTreeMap<String, &'static ResultMeta> =
            result_rows.into_iter().map(|r| (r.path, r.meta)).collect();
        let mut errors = Vec::new();

        // Expand the families into one record per index.
        let mut authored: Vec<(Record, Option<u32>)> = plain.iter().map(|r| (*r, None)).collect();
        let mut family_symbols = BTreeMap::new();
        for f in families {
            if matches!(f.record.eval, Eval::Custom(_)) {
                errors.push(format!(
                    "{}: a family record cannot have a custom eval",
                    f.record.target
                ));
            }
            // Each member's symbol is parsed below, which catches an unbraced index (k_11).
            if !f.record.target.contains('#') || !f.record.symbol.contains('#') {
                errors.push(format!(
                    "{}: a family's target and symbol need '#'",
                    f.record.target
                ));
            }
            family_symbols.insert(f.record.target.to_owned(), f.record.symbol.to_owned());
            authored.extend(f.indices.iter().map(|&n| (f.record, Some(n))));
        }

        let unit_of = |path: &str| -> Option<&'static str> {
            results
                .get(path)
                .map(|m| m.unit)
                .or_else(|| inputs.get(path).map(|m| m.unit))
        };
        let exists = |path: &str| results.contains_key(path) || inputs.contains_key(path);

        let mut equations = Vec::new();
        for (rec, index) in authored {
            let sub = |s: &str| match index {
                Some(n) => s.replace('#', &n.to_string()),
                None => s.to_owned(),
            };
            let target = sub(rec.target);
            let symbol = sub(rec.symbol);
            let at = |m: String| format!("{target}: {m}");
            if let Err(m) = Symbol::parse(&symbol) {
                errors.push(at(m));
            }
            let Some(meta) = results.get(&target) else {
                errors.push(at("not a result path".into()));
                continue;
            };
            let mut formula = match markup::parse(rec.formula, index) {
                Ok(f) => f,
                Err(e) => {
                    errors.push(at(format!("markup {e}")));
                    continue;
                }
            };
            // Resolve unit conversions and check every term, family member and table read.
            let mut terms: Vec<String> = Vec::new();
            let add = |p: String, terms: &mut Vec<String>| {
                if !terms.contains(&p) {
                    terms.push(p);
                }
            };
            let mut problems = Vec::new();
            // The unit each path is read in: one per formula, so the term list (one row per
            // term, in that unit) satisfies the equation shown.
            let mut read_in: BTreeMap<String, Option<String>> = BTreeMap::new();
            let mut two_units: BTreeSet<String> = BTreeSet::new();
            formula.visit_mut(&mut |e| {
                let (r, family) = match e {
                    Expr::Term(r) => (r, false),
                    Expr::FamilyTerm(r) => (r, true),
                    Expr::Sum(set, _) | Expr::Peak(set, _) => {
                        add(set.selector().to_owned(), &mut terms);
                        return;
                    }
                    Expr::Table { table, field, .. } => {
                        if tables::field(table, field).is_none() {
                            problems.push(format!("table {table} has no readable field {field}"));
                        }
                        return;
                    }
                    _ => return,
                };
                {
                    let members: Vec<String> = if family {
                        ODD_HARMONICS
                            .iter()
                            .map(|n| r.path.replace('#', &n.to_string()))
                            .collect()
                    } else {
                        vec![r.path.clone()]
                    };
                    for m in &members {
                        if !exists(m) {
                            problems.push(format!("term {m} is not an input or result path"));
                        } else if m == &target {
                            problems.push("the record reads its own target".into());
                        }
                        match read_in.get(m) {
                            Some(u) if *u != r.unit => {
                                two_units.insert(m.clone());
                            }
                            Some(_) => {}
                            None => {
                                read_in.insert(m.clone(), r.unit.clone());
                            }
                        }
                        add(m.clone(), &mut terms);
                    }
                    if let Some(u) = &r.unit {
                        match unit_of(&members[0]).and_then(|from| conversion(from, u)) {
                            Some(k) => r.scale = k,
                            None => problems.push(format!(
                                "no conversion from {} ({:?}) to {u}",
                                members[0],
                                unit_of(&members[0])
                            )),
                        }
                    }
                }
            });
            problems.extend(
                two_units
                    .into_iter()
                    .map(|m| format!("{m} is read in two units")),
            );
            errors.extend(problems.into_iter().map(&at));
            equations.push(Equation {
                target: target.clone(),
                symbol,
                source: rec.formula,
                index,
                formula,
                eval: rec.eval,
                terms,
                label: meta.label,
                unit: meta.unit,
                cell: cells.get(&target).cloned().flatten(),
                corrections: rec.corrections,
            });
        }

        let mut by_target = BTreeMap::new();
        for (i, eq) in equations.iter().enumerate() {
            if by_target.insert(eq.target.clone(), i).is_some() {
                errors.push(format!("{}: two records", eq.target));
            }
        }
        let mut used_by: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for eq in &equations {
            for t in &eq.terms {
                used_by
                    .entry(t.clone())
                    .or_default()
                    .push(eq.target.clone());
            }
        }

        // Symbols: a record's own, else the leaf table; one symbol per path, every term has
        // one, no leaf entry is dead, and no two paths share a symbol.
        let mut symbols: BTreeMap<String, String> = equations
            .iter()
            .map(|e| (e.target.clone(), e.symbol.clone()))
            .collect();
        for &(path, symbol) in leaf_symbols {
            if by_target.contains_key(path) {
                errors.push(format!(
                    "{path}: has a record; its symbol is the record's (remove it from SYMBOLS)"
                ));
            } else if !used_by.contains_key(path) {
                errors.push(format!("{path}: in SYMBOLS but no record reads it"));
            } else if !exists(path) {
                errors.push(format!(
                    "{path}: in SYMBOLS but not an input or result path"
                ));
            }
            if let Err(m) = Symbol::parse(symbol) {
                errors.push(format!("{path}: {m}"));
            }
            if symbols.insert(path.to_owned(), symbol.to_owned()).is_some() {
                errors.push(format!("{path}: listed twice"));
            }
        }
        for term in used_by.keys() {
            if !symbols.contains_key(term) {
                errors.push(format!("{term}: a term with no symbol (add it to SYMBOLS)"));
            }
        }
        let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
        for (path, symbol) in &symbols {
            if let Some(other) = owners.insert(symbol.as_str(), path.as_str()) {
                errors.push(format!(
                    "symbol {symbol} is used by both {other} and {path}"
                ));
            }
        }

        // What the panel draws must read as what is evaluated.
        for eq in &equations {
            let symbol_of = |e: &Expr| -> Option<String> {
                match e {
                    Expr::Term(r) => symbols.get(&r.path).cloned(),
                    Expr::FamilyTerm(r) => family_symbols.get(&r.path).map(|s| s.replace('#', "n")),
                    Expr::Local(k) => eq.formula.bindings.get(*k).map(|b| b.symbol.clone()),
                    _ => None,
                }
            };
            for m in typesetting_problems(&eq.formula, &symbol_of) {
                errors.push(format!("{}: {m}", eq.target));
            }
        }

        // Cycles would make drill-down loop; the graph must be a DAG.
        let graph: BTreeMap<&str, &[String]> = equations
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
            .collect();
        for eq in &equations {
            if closure(&graph, &eq.target).1 {
                errors.push(format!(
                    "{}: its terms lead back to it (a cycle)",
                    eq.target
                ));
            }
        }

        if errors.is_empty() {
            Ok(Registry {
                equations,
                by_target,
                used_by,
                symbols,
                family_symbols,
                inputs,
                results,
            })
        } else {
            Err(errors)
        }
    }

    /// Every equation, in authoring order (plain records, then family members).
    pub fn equations(&self) -> &[Equation] {
        &self.equations
    }

    /// The equation of a result path, if it has one.
    pub fn equation_for(&self, path: &str) -> Option<&Equation> {
        self.by_target.get(path).map(|&i| &self.equations[i])
    }

    /// The results whose equation reads `path` (the panel's "used by" list), sorted by
    /// authoring order; empty when none does.
    pub fn used_by(&self, path: &str) -> &[String] {
        self.used_by.get(path).map_or(&[], Vec::as_slice)
    }

    /// Whether `path` is an input: a leaf of every drill-down.
    pub fn is_leaf_input(&self, path: &str) -> bool {
        self.inputs.contains_key(path)
    }

    /// What `path` is to the explorer; `None` for a path that is neither input nor result.
    pub fn term_kind(&self, path: &str) -> Option<TermKind> {
        if let Some(m) = self.inputs.get(path) {
            Some(TermKind::Input {
                assumption: m.assumption,
            })
        } else if self.by_target.contains_key(path) {
            Some(TermKind::Explained)
        } else if self.results.contains_key(path) {
            Some(TermKind::CellOnly)
        } else {
            None
        }
    }

    /// The display symbol (symbol markup) of a record's target or of a term.
    pub fn symbol(&self, path: &str) -> Option<&str> {
        self.symbols.get(path).map(String::as_str)
    }

    /// The generic symbol of a family term inside a Σ (`model.b_i#` gives `B_{i,n}`).
    pub fn family_symbol(&self, template: &str) -> Option<String> {
        self.family_symbols
            .get(template)
            .map(|s| s.replace('#', "n"))
    }

    /// The members of a family term inside a Σ or `peak` for a design: the paths of the
    /// harmonics summed (`model.b_i#` gives `model.b_i1`, `model.b_i3`, `model.b_i5` for the
    /// workbook's set), so the panel colours and lists exactly the terms in play.
    pub fn family_members(&self, template: &str, src: &dyn TermSource) -> Vec<String> {
        let count = match src.value(IndexSet::Harmonics.selector()) {
            Some(Value::Int(code)) => harmonic_count(code).unwrap_or(0),
            _ => 0,
        };
        ODD_HARMONICS[..count]
            .iter()
            .map(|n| template.replace('#', &n.to_string()))
            .collect()
    }

    /// The dependency graph: each explained result and its terms.
    pub fn graph(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.equations
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
    }

    /// Every path `path` depends on, transitively (terms of terms), itself excluded.
    pub fn upstream(&self, path: &str) -> BTreeSet<String> {
        let graph: BTreeMap<&str, &[String]> = self.graph().collect();
        closure(&graph, path).0
    }

    /// The value of every term of `eq`, in `eq.terms` order, in each term's own unit.
    pub fn term_values(&self, eq: &Equation, src: &dyn TermSource) -> Vec<(String, Option<Value>)> {
        eq.terms.iter().map(|t| (t.clone(), src.value(t))).collect()
    }

    /// The panel's term list for `eq`: symbol, value and unit of every term, in `eq.terms`
    /// order, each in the unit the formula reads it in (`{model.gap_radius_mm|m}` lists R_g
    /// in m), so the numbers shown satisfy the equation shown.
    pub fn term_rows(&self, eq: &Equation, src: &dyn TermSource) -> Vec<TermRow> {
        let mut wanted: BTreeMap<String, (String, f64)> = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Term(r) | Expr::FamilyTerm(r) = e
                && let Some(u) = &r.unit
            {
                for n in ODD_HARMONICS {
                    wanted
                        .entry(r.path.replace('#', &n.to_string()))
                        .or_insert((u.clone(), r.scale));
                }
            }
        });
        eq.terms
            .iter()
            .map(|t| {
                let own = self
                    .results
                    .get(t)
                    .map(|m| m.unit)
                    .or_else(|| self.inputs.get(t).map(|m| m.unit))
                    .unwrap_or("");
                let (unit, scale) = wanted.get(t).cloned().unwrap_or((own.to_owned(), 1.0));
                let value = src.value(t).map(|v| match v {
                    Value::Num(x) if scale != 1.0 => Value::Num(x * scale),
                    other => other,
                });
                TermRow {
                    path: t.clone(),
                    symbol: self.symbols.get(t).cloned().unwrap_or_default(),
                    value,
                    unit,
                    kind: self
                        .term_kind(t)
                        .expect("every term is an input or result (build)"),
                }
            })
            .collect()
    }

    /// Evaluates `eq` over `src` (the drift guard; the GUI never needs it per frame).
    pub fn evaluate(
        &self,
        eq: &Equation,
        src: &dyn TermSource,
        trace: Option<&mut Trace>,
    ) -> Result<Value, EvalError> {
        match eq.eval {
            Eval::Markup => eval::evaluate(&eq.formula, src, trace),
            Eval::Custom(f) => {
                let restricted = Restricted {
                    inner: src,
                    allowed: &eq.terms,
                };
                let v = f(&restricted)?;
                if let Some(t) = trace {
                    // A custom eval's sensitivity is unknown: its terms count as read, not as moving it.
                    t.condition_terms.extend(eq.terms.iter().cloned());
                }
                Ok(v)
            }
        }
    }
}

/// What the typesetter would draw ambiguously, so that the formula shown would not read as
/// the one evaluated (the markup table in [`super::markup`] draws `a / b` inline and only the
/// parentheses the markup writes): an inline `a / b` as an operand of a product or of another
/// inline quotient (`a/b c` reads as a/(b c)); a Σ or `peak` as an operand of a product,
/// quotient or power, or left of a sum (where it ends is unclear); a power of a symbol, local
/// or `exp` that already carries a superscript (`R_g^{cal}^2`). Parentheses resolve each.
fn typesetting_problems(
    formula: &Formula,
    symbol_of: &dyn Fn(&Expr) -> Option<String>,
) -> Vec<String> {
    let inline_div = |x: &Expr| matches!(x, Expr::Bin(BinOp::Div, ..));
    let big = |x: &Expr| matches!(x, Expr::Sum(..) | Expr::Peak(..));
    let mut out = Vec::new();
    formula.visit(&mut |e| {
        let Expr::Bin(op, a, b) = e else { return };
        match op {
            BinOp::Mul | BinOp::Dot | BinOp::Div => {
                if inline_div(a) || inline_div(b) {
                    out.push("an inline a / b as an operand of a product or quotient: write frac(a, b) or (a / b)".to_owned());
                }
                if big(a) || big(b) {
                    out.push("a Σ or peak as an operand of a product or quotient: wrap it in parentheses".to_owned());
                }
            }
            BinOp::Pow => {
                let base = match &**a {
                    Expr::Call(Func::Exp, _) => Some("exp".to_owned()),
                    other => symbol_of(other)
                        .filter(|s| Symbol::parse(s).is_ok_and(|s| s.sup.is_some())),
                };
                if let Some(s) = base {
                    out.push(format!("a power of {s}, which has a superscript: wrap it in parentheses"));
                }
                if big(a) {
                    out.push("a Σ or peak raised to a power: wrap it in parentheses".to_owned());
                }
            }
            BinOp::Add | BinOp::Sub => {
                if big(a) {
                    out.push("a Σ or peak left of + or −: wrap it in parentheses".to_owned());
                }
            }
        }
    });
    out
}

/// Every node reachable from `start` along `graph`'s edges, `start` excluded, and whether
/// `start` is reachable from itself (a cycle).
fn closure(graph: &BTreeMap<&str, &[String]>, start: &str) -> (BTreeSet<String>, bool) {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<&str> = vec![start];
    let mut cycle = false;
    while let Some(p) = stack.pop() {
        for t in graph.get(p).copied().unwrap_or(&[]) {
            if t == start {
                cycle = true;
            }
            if seen.insert(t.clone()) {
                stack.push(t);
            }
        }
    }
    (seen, cycle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::record::{family, record};

    #[test]
    fn the_shipped_records_build() {
        let r = Registry::build();
        assert!(!r.equations().is_empty());
    }

    #[test]
    fn a_bad_record_is_refused_with_every_reason() {
        let errors = Registry::from_parts(
            &[
                record("model.nope", "X", "1"),
                record("model.f_end", "f_{end}", "{model.nope} + 1"),
                record("model.tau_Pa", "σ", "{model.gap_radius_mm|K}"),
                record("model.torque_2d_Nm", "T_{2D}", "1 +"),
                record(
                    "model.pullout_Nm",
                    "T_{pull}",
                    "{model.gap_radius_mm|m} * {model.gap_radius_mm}",
                ),
                record(
                    "model.k3",
                    "k_3",
                    "{coupling.npole} / 2 * {model.gap_radius_mm}",
                ),
                record("model.f_cal", "f^{cal}", "2"),
                record("model.pole_pitch_mm", "τ_p", "{model.f_cal}^2"),
                record(
                    "model.area_lever_m3",
                    "A",
                    "sum(n in H: {model.tau#_Pa}) * 2",
                ),
            ],
            &[
                family(record("model.k#", "k_#", "n"), &[1, 11]),
                family(record("model.k", "k_{#}", "n"), &[1]),
            ],
            &[],
        )
        .err()
        .expect("refused");
        let all = errors.join("\n");
        for want in [
            "model.nope: not a result path",
            "term model.nope is not",
            "no conversion from model.gap_radius_mm",
            "model.torque_2d_Nm: markup",
            "a family's target and symbol need",
            "model.k11: symbol 'k_11': unexpected text",
            "model.pullout_Nm: model.gap_radius_mm is read in two units",
            "model.k3: an inline a / b as an operand of a product",
            "model.pole_pitch_mm: a power of f^{cal}, which has a superscript",
            "model.area_lever_m3: a Σ or peak as an operand of a product",
        ] {
            assert!(all.contains(want), "missing '{want}' in:\n{all}");
        }
    }

    #[test]
    fn a_cycle_and_a_missing_symbol_are_refused() {
        let errors = Registry::from_parts(
            &[
                record("model.f_end", "f_{end}", "{model.pullout_Nm}"),
                record("model.pullout_Nm", "T_{pull}", "{model.f_end}"),
            ],
            &[],
            &[],
        )
        .err()
        .expect("refused");
        assert!(errors.iter().any(|e| e.contains("a cycle")), "{errors:?}");
    }
}
