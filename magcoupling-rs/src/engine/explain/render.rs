//! A plain-text rendering of a formula: the results-table tooltip fallback, CSV/JSON export
//! and test output. The M4 typesetter draws the same tree ([`super::markup::Expr`]) with
//! egui (stacked fractions, scripts, radicals, Σ, braces); this is its one-line shadow.

use super::markup::{BinOp, Cond, Expr, Formula, Func, RelOp, Symbol};
use super::registry::Registry;
use super::tables;

/// `symbol = formula [, where ...]` in plain text, symbols from the registry.
pub fn plain(registry: &Registry, target_symbol: &str, formula: &Formula) -> String {
    let mut r = Renderer { registry, formula };
    let mut out = format!("{} = {}", sym(target_symbol), r.expr(&formula.body));
    for (i, b) in formula.bindings.iter().enumerate() {
        let lead = if i == 0 { ", where" } else { "," };
        out.push_str(&format!("{lead} {} = {}", sym(&b.symbol), r.expr(&b.expr)));
    }
    out
}

fn sym(markup: &str) -> String {
    Symbol::parse(markup).map_or_else(|_| markup.to_owned(), |s| s.plain())
}

struct Renderer<'a> {
    registry: &'a Registry,
    formula: &'a Formula,
}

impl Renderer<'_> {
    fn term(&self, path: &str, family: bool) -> String {
        let symbol = if family {
            self.registry.family_symbol(path)
        } else {
            self.registry.symbol(path).map(str::to_owned)
        };
        symbol.map_or_else(|| format!("[{path}]"), |s| sym(&s))
    }

    /// An operand of a product, quotient or power base: a stacked fraction, drawn inline here
    /// as `a/b`, is wrapped so the line reads as the tree does (`(π/4) (D² − d²)`).
    fn factor(&mut self, e: &Expr) -> String {
        let s = self.expr(e);
        if matches!(e, Expr::Frac(..)) {
            format!("({s})")
        } else {
            s
        }
    }

    /// Wraps compound operands of an inline `/` so the text reads as the fraction does.
    fn grouped(&mut self, e: &Expr) -> String {
        let s = self.expr(e);
        let atomic = matches!(
            e,
            Expr::Num { .. }
                | Expr::Text(_)
                | Expr::NoneLit
                | Expr::Pi
                | Expr::Term(_)
                | Expr::FamilyTerm(_)
                | Expr::Index
                | Expr::Local(_)
                | Expr::Paren(_)
                | Expr::Call(..)
                | Expr::Table { .. }
        );
        if atomic { s } else { format!("({s})") }
    }

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::Num { text, .. } => text.clone(),
            Expr::Text(t) => format!("\"{t}\""),
            Expr::NoneLit => "none".into(),
            Expr::Pi => "π".into(),
            Expr::Term(r) => self.term(&r.path, false),
            Expr::FamilyTerm(r) => self.term(&r.path, true),
            Expr::Index => "n".into(),
            Expr::Local(k) => sym(&self.formula.bindings[*k].symbol),
            Expr::Neg(a) => format!("−{}", self.expr(a)),
            Expr::Paren(a) => format!("({})", self.expr(a)),
            Expr::Bin(op, a, b) => {
                let (x, y) = match op {
                    BinOp::Add | BinOp::Sub => (self.expr(a), self.expr(b)),
                    BinOp::Pow => (self.factor(a), self.expr(b)),
                    BinOp::Mul | BinOp::Dot | BinOp::Div => (self.factor(a), self.factor(b)),
                };
                match op {
                    BinOp::Add => format!("{x} + {y}"),
                    BinOp::Sub => format!("{x} − {y}"),
                    BinOp::Mul => {
                        let numbers =
                            matches!(**a, Expr::Num { .. }) && matches!(**b, Expr::Num { .. });
                        // A family member's index 1 as a factor is not drawn (k_1 = N/(2 R_g)).
                        let unit_factor = matches!(**a, Expr::Num { value, .. } if value == 1.0);
                        if unit_factor && !numbers {
                            y
                        } else if numbers {
                            format!("{x} × {y}")
                        } else {
                            format!("{x} {y}")
                        }
                    }
                    BinOp::Dot => format!("{x} · {y}"),
                    BinOp::Div => format!("{x}/{y}"),
                    BinOp::Pow => {
                        let y = match &**b {
                            Expr::Paren(inner) => self.expr(inner),
                            _ => y,
                        };
                        if y.chars().count() == 1 {
                            format!("{x}^{y}")
                        } else {
                            format!("{x}^{{{y}}}")
                        }
                    }
                }
            }
            Expr::Frac(a, b) => format!("{}/{}", self.grouped(a), self.grouped(b)),
            Expr::Call(f, args) => {
                let a: Vec<String> = args.iter().map(|x| self.expr(x)).collect();
                match f {
                    Func::Sqrt => format!("√({})", a[0]),
                    Func::Exp => format!("e^{{{}}}", a[0]),
                    Func::Abs => format!("|{}|", a[0]),
                    Func::Ceil => format!("⌈{}⌉", a[0]),
                    Func::Floor => format!("⌊{}⌋", a[0]),
                    _ => format!("{}({})", f.name(), a.join(", ")),
                }
            }
            Expr::Sum(_, body) => format!("Σ_{{n∈H}} {}", self.expr(body)),
            Expr::Peak(_, body) => format!(
                "argmax_{{0≤φ≤π/2}} Σ_{{n∈H}} {} sin(nφ)",
                self.grouped(body)
            ),
            Expr::Table { table, key, field } => {
                let s = tables::field(table, field)
                    .map_or_else(|| format!("{table}.{field}"), |f| sym(f.symbol));
                format!("{s}({})", self.expr(key))
            }
            Expr::Cases {
                arms, otherwise, ..
            } => {
                let mut rows: Vec<String> = arms
                    .iter()
                    .map(|(c, v)| format!("{} if {}", self.expr(v), self.cond(c)))
                    .collect();
                rows.push(format!("{} otherwise", self.expr(otherwise)));
                format!("{{ {} }}", rows.join("; "))
            }
        }
    }

    fn cond(&mut self, c: &Cond) -> String {
        match c {
            Cond::And(a, b) => format!("{} and {}", self.cond(a), self.cond(b)),
            Cond::Or(a, b) => format!("{} or {}", self.cond(a), self.cond(b)),
            Cond::Rel(op, a, b) => {
                let op = match op {
                    RelOp::Lt => "<",
                    RelOp::Le => "≤",
                    RelOp::Gt => ">",
                    RelOp::Ge => "≥",
                    RelOp::Eq => "=",
                    RelOp::Ne => "≠",
                };
                format!("{} {op} {}", self.expr(a), self.expr(b))
            }
        }
    }
}
