//! Evaluates a parsed formula over term values: what the drift guard compares with the
//! engine's result, and (traced) what the traceability test reads to know which terms a
//! result actually depends on at a design point.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fmt;

use super::markup::{BinOp, Cond, Expr, Formula, Func, IndexSet, RelOp};
use super::tables;
use crate::engine::compat::{py_max, py_min};
use crate::engine::meta::Value;
use crate::engine::model::{HALF_PITCH_RAD, ODD_HARMONICS, harmonic_count, peak_off_half_pitch};

/// Where term values come from: an input or result path to its value, `None` for an
/// unknown path.
pub trait TermSource {
    fn value(&self, path: &str) -> Option<Value>;
}

/// Why a formula could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalError(pub String);

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an evaluation read, for the traceability test and branch coverage.
///
/// A term is a **value term** when its value flows into the result along the path taken
/// (arithmetically, through the chosen `cases` arm, through the winner of a `min`/`max`),
/// so nudging it moves the result. It is a **condition term** when it was read only to
/// decide something piecewise-constant: a `cases` condition, the loser of a `min`/`max`, the
/// argument of `ceil`/`floor`, the selector of a Σ's index set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub value_terms: BTreeSet<String>,
    pub condition_terms: BTreeSet<String>,
    /// For each `cases` evaluated, its id and the arm taken (`arms.len()` for `else`).
    pub arms: BTreeSet<(usize, usize)>,
}

impl Trace {
    fn absorb(&mut self, other: Trace, as_condition: bool) {
        if as_condition {
            self.condition_terms.extend(other.value_terms);
        } else {
            self.value_terms.extend(other.value_terms);
        }
        self.condition_terms.extend(other.condition_terms);
        self.arms.extend(other.arms);
    }
}

/// An evaluated value: a number (inputs' integers become numbers), a text or none.
#[derive(Clone, Debug, PartialEq)]
enum V {
    Num(f64),
    Text(String),
    None,
}

impl V {
    fn from_value(v: Value) -> V {
        match v {
            Value::Num(x) => V::Num(x),
            Value::Int(i) => V::Num(i as f64),
            Value::Text(s) => V::Text(s),
            Value::None => V::None,
        }
    }

    fn into_value(self) -> Value {
        match self {
            V::Num(x) => Value::Num(x),
            V::Text(s) => Value::Text(s),
            V::None => Value::None,
        }
    }

    fn num(&self, what: &str) -> Result<f64, EvalError> {
        match self {
            V::Num(x) => Ok(*x),
            other => Err(EvalError(format!(
                "{what}: expected a number, got {other:?}"
            ))),
        }
    }
}

/// Evaluates `formula` over `src`. With `trace`, records what was read ([`Trace`]).
/// `index` is the Σ variable's value when evaluating inside a Σ (callers pass `None`).
pub fn evaluate(
    formula: &Formula,
    src: &dyn TermSource,
    trace: Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut ev = Evaluator {
        formula,
        src,
        locals: vec![None; formula.bindings.len()],
    };
    let mut local_trace = Trace::default();
    let v = ev.expr(&formula.body, None, &mut local_trace)?;
    if let Some(t) = trace {
        t.absorb(local_trace, false);
    }
    Ok(v.into_value())
}

struct Evaluator<'a> {
    formula: &'a Formula,
    src: &'a dyn TermSource,
    /// Memoized bindings: the value and what evaluating it read.
    locals: Vec<Option<(V, Trace)>>,
}

impl Evaluator<'_> {
    fn term(&self, path: &str, scale: f64, t: &mut Trace) -> Result<V, EvalError> {
        let v = self
            .src
            .value(path)
            .ok_or_else(|| EvalError(format!("unknown term {path}")))?;
        t.value_terms.insert(path.to_owned());
        let v = V::from_value(v);
        Ok(match v {
            V::Num(x) if scale != 1.0 => V::Num(x * scale),
            other => other,
        })
    }

    fn num(&mut self, e: &Expr, n: Option<u32>, t: &mut Trace) -> Result<f64, EvalError> {
        self.expr(e, n, t)?.num("operand")
    }

    fn expr(&mut self, e: &Expr, n: Option<u32>, t: &mut Trace) -> Result<V, EvalError> {
        Ok(match e {
            Expr::Num { value, .. } => V::Num(*value),
            Expr::Text(s) => V::Text(s.clone()),
            Expr::NoneLit => V::None,
            Expr::Pi => V::Num(PI),
            Expr::Term(r) => self.term(&r.path, r.scale, t)?,
            Expr::FamilyTerm(r) => {
                let n =
                    n.ok_or_else(|| EvalError(format!("family term {} outside a Σ", r.path)))?;
                self.term(&r.path.replace('#', &n.to_string()), r.scale, t)?
            }
            Expr::Index => V::Num(f64::from(
                n.ok_or_else(|| EvalError("n outside a Σ".into()))?,
            )),
            Expr::Peak(set, body) => {
                // The amplitudes over the set; the engine's E7 search picks the angle. Their
                // terms move the angle only where it leaves half a pitch.
                let mut at = Trace::default();
                let amplitudes = match self.members(*set, &mut at)? {
                    None => Vec::new(),
                    Some(members) => members
                        .iter()
                        .map(|&k| self.num(body, Some(k), &mut at))
                        .collect::<Result<Vec<f64>, EvalError>>()?,
                };
                let peak = peak_off_half_pitch(&amplitudes);
                t.absorb(at, peak.is_none());
                V::Num(peak.unwrap_or(HALF_PITCH_RAD))
            }
            Expr::Table { table, key, field } => {
                let mut kt = Trace::default();
                let key = self.expr(key, n, &mut kt)?.into_value();
                t.absorb(kt, true); // the key selects a row: piecewise constant
                match tables::lookup(table, &key, field) {
                    Ok(Some(v)) => V::from_value(v),
                    Ok(None) => V::None,
                    Err(e) => return Err(EvalError(e)),
                }
            }
            Expr::Local(k) => {
                if self.locals[*k].is_none() {
                    let binding = &self.formula.bindings[*k].expr;
                    let mut lt = Trace::default();
                    let v = self.expr(binding, n, &mut lt)?;
                    self.locals[*k] = Some((v, lt));
                }
                let (v, lt) = self.locals[*k].clone().expect("just evaluated");
                t.absorb(lt, false);
                v
            }
            Expr::Neg(a) => V::Num(-self.num(a, n, t)?),
            Expr::Paren(a) => self.expr(a, n, t)?,
            Expr::Bin(op, a, b) => {
                let (x, y) = (self.num(a, n, t)?, self.num(b, n, t)?);
                V::Num(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul | BinOp::Dot => x * y,
                    BinOp::Div => x / y,
                    BinOp::Pow => pow(x, y),
                })
            }
            Expr::Frac(a, b) => {
                let (x, y) = (self.num(a, n, t)?, self.num(b, n, t)?);
                V::Num(x / y)
            }
            Expr::Call(func, args) => self.call(*func, args, n, t)?,
            Expr::Sum(set, body) => {
                match self.members(*set, t)? {
                    None => V::Num(f64::NAN), // an invalid set: every harmonic sum is NaN (D3)
                    Some(members) => {
                        let mut acc = 0.0; // Python sum(): a left fold from 0
                        for &k in members {
                            acc += self.num(body, Some(k), t)?;
                        }
                        V::Num(acc)
                    }
                }
            }
            Expr::Cases {
                id,
                arms,
                otherwise,
            } => {
                for (i, (cond, value)) in arms.iter().enumerate() {
                    let mut ct = Trace::default();
                    let holds = self.cond(cond, n, &mut ct)?;
                    t.absorb(ct, true);
                    if holds {
                        t.arms.insert((*id, i));
                        return self.expr(value, n, t);
                    }
                }
                t.arms.insert((*id, arms.len()));
                self.expr(otherwise, n, t)?
            }
        })
    }

    /// The members of an index set, as the engine picks them (`model::harmonic_count`);
    /// `None` for an invalid set. Reads the set's selector as a condition term.
    fn members(
        &mut self,
        set: IndexSet,
        t: &mut Trace,
    ) -> Result<Option<&'static [u32]>, EvalError> {
        let selector = set.selector();
        let code = self
            .src
            .value(selector)
            .ok_or_else(|| EvalError(format!("unknown term {selector}")))?;
        t.condition_terms.insert(selector.to_owned());
        match (set, code) {
            (IndexSet::Harmonics, Value::Int(c)) => {
                Ok(harmonic_count(c).map(|k| &ODD_HARMONICS[..k]))
            }
            (_, other) => Err(EvalError(format!(
                "{selector}: expected a code, got {other:?}"
            ))),
        }
    }

    fn call(
        &mut self,
        func: Func,
        args: &[Expr],
        n: Option<u32>,
        t: &mut Trace,
    ) -> Result<V, EvalError> {
        if matches!(func, Func::Min | Func::Max) {
            // Python's min/max fold; only the winner's terms are value terms.
            let mut best: Option<(f64, Trace)> = None;
            for a in args {
                let mut at = Trace::default();
                let x = self.num(a, n, &mut at)?;
                best = Some(match best {
                    None => (x, at),
                    Some((b, bt)) => {
                        // compat::py_min(b, x) is x exactly when x < b (py_max: x > b).
                        let pick = if func == Func::Min {
                            py_min(b, x)
                        } else {
                            py_max(b, x)
                        };
                        let x_wins = if func == Func::Min { x < b } else { x > b };
                        debug_assert!(pick.to_bits() == if x_wins { x } else { b }.to_bits());
                        if x_wins {
                            t.absorb(bt, true);
                            (x, at)
                        } else {
                            t.absorb(at, true);
                            (b, bt)
                        }
                    }
                });
            }
            let (x, bt) = best.expect("min/max has at least two arguments (parser)");
            t.absorb(bt, false);
            return Ok(V::Num(x));
        }
        let mut at = Trace::default();
        let x = self.num(&args[0], n, &mut at)?;
        let piecewise = matches!(func, Func::Ceil | Func::Floor);
        t.absorb(at, piecewise);
        Ok(V::Num(match func {
            Func::Sqrt => x.sqrt(),
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Tanh => x.tanh(),
            Func::Exp => x.exp(),
            Func::Ln => x.ln(),
            Func::Abs => x.abs(),
            Func::Ceil => x.ceil(),
            Func::Floor => x.floor(),
            Func::Min | Func::Max => unreachable!("handled above"),
        }))
    }

    fn cond(&mut self, c: &Cond, n: Option<u32>, t: &mut Trace) -> Result<bool, EvalError> {
        Ok(match c {
            Cond::And(a, b) => self.cond(a, n, t)? && self.cond(b, n, t)?,
            Cond::Or(a, b) => self.cond(a, n, t)? || self.cond(b, n, t)?,
            Cond::Rel(op, a, b) => {
                let (x, y) = (self.expr(a, n, t)?, self.expr(b, n, t)?);
                match (&x, &y) {
                    (V::Num(x), V::Num(y)) => match op {
                        RelOp::Lt => x < y,
                        RelOp::Le => x <= y,
                        RelOp::Gt => x > y,
                        RelOp::Ge => x >= y,
                        RelOp::Eq => x == y,
                        RelOp::Ne => x != y,
                    },
                    _ => match op {
                        RelOp::Eq => x == y,
                        RelOp::Ne => x != y,
                        _ => return Err(EvalError(format!("cannot order {x:?} and {y:?}"))),
                    },
                }
            }
        })
    }
}

/// `x ^ y`: an integer power by `powi` (as the engine writes squares), else `powf`.
fn pow(x: f64, y: f64) -> f64 {
    if y.fract() == 0.0 && y.abs() <= 64.0 {
        x.powi(y as i32)
    } else {
        x.powf(y)
    }
}

/// A [`TermSource`] over a fixed map (unit tests, documentation examples).
impl TermSource for BTreeMap<String, Value> {
    fn value(&self, path: &str) -> Option<Value> {
        self.get(path).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::markup::parse;

    fn src(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(p, v)| ((*p).to_owned(), v.clone()))
            .collect()
    }

    fn eval(markup: &str, s: &BTreeMap<String, Value>) -> (Value, Trace) {
        let f = parse(markup, None).unwrap();
        let mut t = Trace::default();
        (evaluate(&f, s, Some(&mut t)).unwrap(), t)
    }

    #[test]
    fn arithmetic_and_functions() {
        let s = src(&[("a.x", Value::Num(3.0)), ("a.y", Value::Int(4))]);
        assert_eq!(eval("sqrt({a.x}^2 + {a.y}^2)", &s).0, Value::Num(5.0));
        assert_eq!(eval("frac({a.x}, 2) · 2 - -1", &s).0, Value::Num(4.0));
        assert_eq!(
            eval("exp(0) + ln(1) + abs(-2) + ceil(0.2) + floor(1.8)", &s).0,
            Value::Num(5.0)
        );
        assert_eq!(eval("2^0.5", &s).0, Value::Num(2f64.powf(0.5)));
    }

    #[test]
    fn unit_scale_applies_to_numbers() {
        let s = src(&[("a.len_mm", Value::Num(12.0))]);
        let mut f = parse("{a.len_mm|m}", None).unwrap();
        f.visit_mut(&mut |e| {
            if let Expr::Term(r) = e {
                r.scale = 1e-3;
            }
        });
        assert_eq!(evaluate(&f, &s, None).unwrap(), Value::Num(12.0 * 1e-3));
    }

    #[test]
    fn cases_trace_value_and_condition_terms() {
        let s = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.v", Value::Num(7.0)),
            ("a.w", Value::Num(9.0)),
        ]);
        let (v, t) = eval(r#"cases({a.x} < {a.y} => {a.v}; else => {a.w})"#, &s);
        assert_eq!(v, Value::Num(7.0));
        assert_eq!(t.value_terms, ["a.v".to_owned()].into());
        assert_eq!(
            t.condition_terms,
            ["a.x".to_owned(), "a.y".to_owned()].into()
        );
        assert_eq!(t.arms, [(0, 0)].into());
        let (v, t) = eval(r#"cases({a.x} > {a.y} => {a.v}; else => "no")"#, &s);
        assert_eq!(v, Value::Text("no".into()));
        assert_eq!(t.arms, [(0, 1)].into());
        // NaN comparisons are false, as the engine's `if`.
        let nan = src(&[("a.x", Value::Num(f64::NAN))]);
        assert_eq!(
            eval(r#"cases({a.x} < 1 => 1; else => 2)"#, &nan).0,
            Value::Num(2.0)
        );
    }

    #[test]
    fn text_and_none_compare_exactly() {
        let s = src(&[("a.p", Value::Text("B842SH".into())), ("a.d", Value::None)]);
        assert_eq!(
            eval(
                r#"cases({a.p} = "B842SH" and {a.d} = none => 1; else => 0)"#,
                &s
            )
            .0,
            Value::Num(1.0)
        );
        assert_eq!(
            eval(r#"cases({a.p} != "B842" => 1; else => 0)"#, &s).0,
            Value::Num(1.0)
        );
        let f = parse(r#"cases({a.p} < "C" => 1; else => 0)"#, None).unwrap();
        assert!(evaluate(&f, &s, None).is_err(), "text has no order");
    }

    #[test]
    fn min_max_follow_python_and_trace_the_winner() {
        let s = src(&[("a.x", Value::Num(1.0)), ("a.y", Value::Num(2.0))]);
        let (v, t) = eval("min({a.y}, {a.x})", &s);
        assert_eq!(v, Value::Num(1.0));
        assert_eq!(t.value_terms, ["a.x".to_owned()].into());
        assert_eq!(t.condition_terms, ["a.y".to_owned()].into());
        let (v, t) = eval("max(1, {a.x})", &s);
        assert_eq!(v, Value::Num(1.0), "a tie keeps the first, as Python");
        assert!(t.value_terms.is_empty());
        // py_min(NaN, 1) is NaN: the first argument is kept when the second does not win.
        let nan = src(&[("a.x", Value::Num(f64::NAN))]);
        assert!(matches!(eval("min({a.x}, 1)", &nan).0, Value::Num(x) if x.is_nan()));
        assert_eq!(eval("min(1, {a.x})", &nan).0, Value::Num(1.0));
    }

    #[test]
    fn a_sum_runs_over_the_harmonic_set() {
        let mut s = src(&[("coupling.max_harmonic", Value::Int(5))]);
        for n in ODD_HARMONICS {
            s.insert(format!("m.t{n}"), Value::Num(f64::from(n)));
        }
        let (v, t) = eval("sum(n in H: {m.t#} * n)", &s);
        assert_eq!(v, Value::Num(1.0 + 9.0 + 25.0));
        assert!(t.condition_terms.contains("coupling.max_harmonic"));
        assert!(t.value_terms.contains("m.t5") && !t.value_terms.contains("m.t7"));
        s.insert("coupling.max_harmonic".into(), Value::Int(4));
        assert!(
            matches!(eval("sum(n in H: {m.t#})", &s).0, Value::Num(x) if x.is_nan()),
            "invalid set"
        );
    }

    #[test]
    fn peak_is_the_e7_angle_of_the_amplitudes() {
        let mut s = src(&[("coupling.max_harmonic", Value::Int(3))]);
        // A small third harmonic: half a pitch stays the peak; a large one moves it.
        s.insert("m.a1".into(), Value::Num(1.0));
        s.insert("m.a3".into(), Value::Num(0.05));
        let (v, t) = eval("peak(n in H: {m.a#})", &s);
        assert_eq!(v, Value::Num(HALF_PITCH_RAD));
        assert!(
            t.value_terms.is_empty(),
            "the angle does not move with the amplitudes here"
        );
        s.insert("m.a3".into(), Value::Num(0.5));
        let (v, t) = eval("peak(n in H: {m.a#})", &s);
        assert_eq!(v, Value::Num(peak_off_half_pitch(&[1.0, 0.5]).unwrap()));
        assert!(t.value_terms.contains("m.a1") && t.value_terms.contains("m.a3"));
    }

    #[test]
    fn tables_read_engine_data_by_key() {
        let s = src(&[
            ("c.part", Value::Text("B842SH".into())),
            ("c.none", Value::Text("".into())),
        ]);
        let (v, t) = eval(r#"table("magnets", {c.part}, "length_mm")"#, &s);
        assert_eq!(
            v,
            Value::Num(crate::engine::library::lookup("B842SH").unwrap().length_mm)
        );
        assert!(t.value_terms.is_empty() && t.condition_terms.contains("c.part"));
        assert_eq!(
            eval(r#"table("magnets", {c.none}, "length_mm")"#, &s).0,
            Value::None
        );
    }

    #[test]
    fn bindings_are_lazy_and_traced_once_used() {
        let s = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.b", Value::Int(1)),
        ]);
        let (v, t) = eval(
            "cases({a.b} = 1 => [S] * 10; else => 0) where [S] = {a.x} + {a.y}",
            &s,
        );
        assert_eq!(v, Value::Num(30.0));
        assert!(t.value_terms.contains("a.x") && t.value_terms.contains("a.y"));
        let s0 = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.b", Value::Int(0)),
        ]);
        let (_, t) = eval(
            "cases({a.b} = 1 => [S] * 10; else => 0) where [S] = {a.x} + {a.y}",
            &s0,
        );
        assert!(t.value_terms.is_empty(), "the binding was never needed");
    }

    #[test]
    fn errors_name_the_problem() {
        let s = src(&[("a.t", Value::Text("x".into()))]);
        for (markup, want) in [
            ("{a.missing}", "unknown term"),
            ("{a.t} + 1", "expected a number"),
            (r#"table("nope", 1, "x")"#, "no table"),
        ] {
            let err = evaluate(&parse(markup, None).unwrap(), &s, None).unwrap_err();
            assert!(err.0.contains(want), "{markup}: {err}");
        }
    }
}
