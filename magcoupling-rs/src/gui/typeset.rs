//! A small typesetter for the equation markup (spec Addendum A2 "Rendering": "a small egui
//! typesetter for the markup: inline fractions, sub/superscripts, Σ and √. No LaTeX
//! dependency").
//!
//! [`layout`] turns a parsed formula ([`crate::engine::explain::markup::Formula`], the tree the
//! drift guard evaluates) into a [`Laid`] box: text runs and strokes placed around a baseline,
//! the TeX box model in miniature (a width, the ascent above the baseline, the descent below
//! it). [`equation_ui`] paints it and says which term the user clicked. Every term is drawn
//! with its display symbol ([`Registry::symbol`]) in its colour ([`TermColors`]), so the
//! equation panel, the hover tooltips and the term list agree.
//!
//! What it draws, by markup (the table in `engine::explain::markup`): a stacked fraction for
//! `frac`, an inline `a/b` for `/`, scripts for symbols and powers (a power of a symbol
//! stacks the exponent over its subscript), a radical drawn with strokes for `sqrt`, a large Σ
//! with `n ∈ H` beneath for `sum`, `arg max` for `peak`, delimiters scaled to their contents,
//! a left brace with one row per arm for `cases`, the `where` bindings on lines of their own.
//! A selector compared for equality (= or ≠) with a code shows the choice's label
//! (`backiron = "steel circuit"`, [`Registry::choices`]); an ordering keeps its numbers (σ_n's
//! `5 ≤ N_h` compares the harmonic index with the highest harmonic, not a code). A screw-size
//! row index shows the size's name (`d_h(M4)`). Factors sit side by side as in
//! `render::plain` (`n_slip 2 π/60`), with × only between two numbers.
//!
//! egui's default fonts lack a few characters the markup and the teaching notes use (decision
//! M43-6): ϑ (U+03D1) is drawn as θ, the same letter, the superscript minus as ¯ and ∝ as ~
//! ([`glyph_safe`]); ∈ and the ceiling and floor brackets are drawn with strokes; `concat` sets
//! its texts side by side (no ⧺). A test lays out every equation of the registry and checks
//! every character drawn against the fonts.

use std::collections::BTreeMap;
use std::sync::Arc;

use egui::text::{Fonts, Galley};
use egui::{Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2, vec2};

use crate::engine::explain::markup::{BinOp, Cond, Expr, Formula, Func, RelOp, Symbol};
use crate::engine::explain::{Equation, Registry, TermSource, tables};
use crate::engine::meta::Value;

/// The term colours, in the order the formula tree first reads the terms (decision M43-3; a
/// `cases` arm's condition before its value): distinct on the dark theme, none of them the
/// theme's text colour; a ninth term takes the first again.
pub const TERM_PALETTE: [Color32; 8] = [
    Color32::from_rgb(240, 160, 40),
    Color32::from_rgb(90, 180, 240),
    Color32::from_rgb(80, 200, 140),
    Color32::from_rgb(230, 210, 80),
    Color32::from_rgb(200, 130, 230),
    Color32::from_rgb(240, 110, 90),
    Color32::from_rgb(120, 150, 250),
    Color32::from_rgb(230, 120, 180),
];

/// How much smaller a script is than its base.
const SCRIPT_SCALE: f32 = 0.7;

/// The smallest text drawn [points].
const MIN_SIZE: f32 = 8.0;

/// The colour of each term an equation shows: a term by its path, a family term inside a Σ by
/// its template (`model.b_i#`) and, once [`TermColors::with_members`] has run, by the paths of
/// the harmonics summed, so a mark on screen finds the members' values too.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TermColors {
    colors: BTreeMap<String, Color32>,
}

impl TermColors {
    /// No colours: every term in the text colour.
    pub fn none() -> Self {
        Self::default()
    }

    /// One palette colour per term of `eq` in the formula tree's order ([`Formula::visit`]: the
    /// body, a `cases` arm's condition before its value, then each `where` binding): a path,
    /// or a family template inside a Σ.
    pub fn of(eq: &Equation) -> Self {
        let mut colors = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Term(r) | Expr::FamilyTerm(r) = e {
                let next = TERM_PALETTE[colors.len() % TERM_PALETTE.len()];
                colors.entry(r.path.clone()).or_insert(next);
            }
        });
        Self { colors }
    }

    /// Adds, for each family template, the paths of the harmonics the design sums
    /// ([`Registry::family_members`]) in the template's colour.
    pub fn with_members(mut self, registry: &Registry, src: &dyn TermSource) -> Self {
        let templates: Vec<(String, Color32)> = self
            .colors
            .iter()
            .filter(|(path, _)| path.contains('#'))
            .map(|(path, color)| (path.clone(), *color))
            .collect();
        for (template, color) in templates {
            for member in registry.family_members(&template, src) {
                self.colors.entry(member).or_insert(color);
            }
        }
        self
    }

    /// The colour of a path or template, if the equation shows it.
    pub fn get(&self, path: &str) -> Option<Color32> {
        self.colors.get(path).copied()
    }

    /// Whether no term has a colour.
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

/// One thing the typesetter draws, placed relative to its box's top-left corner.
#[derive(Clone, Debug)]
pub enum Ink {
    /// A text run; `term` is the path (or template) it shows, for clicks.
    Text {
        at: Vec2,
        galley: Arc<Galley>,
        term: Option<String>,
    },
    /// A straight stroke `width` points wide (a fraction bar, a radical's overline).
    Line { from: Vec2, to: Vec2, width: f32 },
    /// A polyline (a radical sign, ∈, a bracket).
    Path { points: Vec<Vec2>, width: f32 },
}

impl Ink {
    fn shifted(self, by: Vec2) -> Ink {
        match self {
            Ink::Text { at, galley, term } => Ink::Text {
                at: at + by,
                galley,
                term,
            },
            Ink::Line { from, to, width } => Ink::Line {
                from: from + by,
                to: to + by,
                width,
            },
            Ink::Path { points, width } => Ink::Path {
                points: points.into_iter().map(|p| p + by).collect(),
                width,
            },
        }
    }
}

/// A laid-out box: `width`, `ascent` above the baseline and `descent` below it, and what it
/// draws (the baseline is at `y = ascent`).
#[derive(Clone, Debug, Default)]
pub struct Laid {
    pub width: f32,
    pub ascent: f32,
    pub descent: f32,
    pub inks: Vec<Ink>,
}

impl Laid {
    /// The box's size.
    pub fn size(&self) -> Vec2 {
        vec2(self.width, self.ascent + self.descent)
    }

    /// Every text run in drawing order, with its rect in the box.
    pub fn texts(&self) -> Vec<(String, Rect)> {
        self.inks
            .iter()
            .filter_map(|ink| match ink {
                Ink::Text { at, galley, .. } => Some((
                    galley.text().to_owned(),
                    Rect::from_min_size(Pos2::ZERO + *at, galley.size()),
                )),
                _ => None,
            })
            .collect()
    }

    /// The rect of each term's text runs in the box, by path (a symbol's base and scripts are
    /// separate runs of one term).
    pub fn term_rects(&self) -> Vec<(String, Rect)> {
        self.inks
            .iter()
            .filter_map(|ink| match ink {
                Ink::Text {
                    at,
                    galley,
                    term: Some(term),
                } => Some((
                    term.clone(),
                    Rect::from_min_size(Pos2::ZERO + *at, galley.size()),
                )),
                _ => None,
            })
            .collect()
    }

    fn gap(width: f32) -> Laid {
        Laid {
            width,
            ..Laid::default()
        }
    }

    /// The boxes side by side on one baseline.
    fn row(parts: Vec<Laid>) -> Laid {
        let ascent = parts.iter().map(|p| p.ascent).fold(0.0, f32::max);
        let descent = parts.iter().map(|p| p.descent).fold(0.0, f32::max);
        let mut x = 0.0;
        let mut inks = Vec::new();
        for p in parts {
            let by = vec2(x, ascent - p.ascent);
            inks.extend(p.inks.into_iter().map(|i| i.shifted(by)));
            x += p.width;
        }
        Laid {
            width: x,
            ascent,
            descent,
            inks,
        }
    }

    /// The boxes one under another, left-aligned, `gap` points apart; the baseline is the
    /// first box's.
    fn column(parts: Vec<Laid>, gap: f32) -> Laid {
        let mut y = 0.0;
        let mut inks = Vec::new();
        let mut width: f32 = 0.0;
        let ascent = parts.first().map_or(0.0, |p| p.ascent);
        let count = parts.len();
        for (i, p) in parts.into_iter().enumerate() {
            width = width.max(p.width);
            let h = p.ascent + p.descent;
            inks.extend(p.inks.into_iter().map(|ink| ink.shifted(vec2(0.0, y))));
            y += h;
            if i + 1 < count {
                y += gap;
            }
        }
        Laid {
            width,
            ascent,
            descent: y - ascent,
            inks,
        }
    }
}

/// The text of `s` as the default fonts can draw it (decision M43-6): ϑ (U+03D1) as θ, the same
/// letter; the superscript minus (U+207B) as ¯, the raised bar it looks like (10¯⁶); ∝
/// (U+221D) as ~.
pub fn glyph_safe(s: &str) -> String {
    s.replace('\u{3d1}', "\u{3b8}")
        .replace('\u{207b}', "\u{af}")
        .replace('\u{221d}', "~")
}

/// The typesetter's state for one formula.
struct Setter<'a> {
    fonts: &'a Fonts,
    registry: &'a Registry,
    formula: &'a Formula,
    colors: &'a TermColors,
    ink: Color32,
}

impl Setter<'_> {
    fn text(&self, s: &str, size: f32, color: Color32, term: Option<&str>) -> Laid {
        let galley = self.fonts.layout_no_wrap(
            glyph_safe(s),
            FontId::proportional(size.max(MIN_SIZE)),
            color,
        );
        let height = galley.size().y;
        let baseline = galley
            .rows
            .first()
            .and_then(|r| r.row.glyphs.first().map(|g| r.pos.y + g.pos.y))
            .unwrap_or(height * 0.8);
        Laid {
            width: galley.size().x,
            ascent: baseline,
            descent: height - baseline,
            inks: vec![Ink::Text {
                at: Vec2::ZERO,
                galley,
                term: term.map(str::to_owned),
            }],
        }
    }

    /// Plain text in the ink colour.
    fn plain(&self, s: &str, size: f32) -> Laid {
        self.text(s, size, self.ink, None)
    }

    fn script_size(size: f32) -> f32 {
        (size * SCRIPT_SCALE).max(MIN_SIZE)
    }

    /// `base` with a subscript and a superscript stacked after it.
    fn scripts(base: Laid, sub: Option<Laid>, sup: Option<Laid>, size: f32) -> Laid {
        let raise = 0.42 * size;
        let lower = 0.28 * size;
        let mut ascent = base.ascent;
        let mut descent = base.descent;
        if let Some(s) = &sup {
            ascent = ascent.max(raise + s.ascent);
        }
        if let Some(s) = &sub {
            descent = descent.max(lower + s.descent);
        }
        let x = base.width;
        let mut width = x;
        let mut inks: Vec<Ink> = base
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2(0.0, ascent - base.ascent)))
            .collect();
        if let Some(s) = sub {
            width = width.max(x + s.width);
            let by = vec2(x, ascent + lower - s.ascent);
            inks.extend(s.inks.into_iter().map(|i| i.shifted(by)));
        }
        if let Some(s) = sup {
            width = width.max(x + s.width);
            let by = vec2(x, ascent - raise - s.ascent);
            inks.extend(s.inks.into_iter().map(|i| i.shifted(by)));
        }
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// A display symbol (symbol markup) in `color`, with `power` stacked over its subscript.
    fn symbol(
        &self,
        markup: &str,
        size: f32,
        color: Color32,
        term: Option<&str>,
        power: Option<Laid>,
    ) -> Laid {
        let symbol = Symbol::parse(markup).unwrap_or(Symbol {
            base: markup.to_owned(),
            sub: None,
            sup: None,
        });
        let small = Self::script_size(size);
        let base = self.text(&symbol.base, size, color, term);
        let sub = symbol.sub.map(|s| self.text(&s, small, color, term));
        let own_sup = symbol.sup.map(|s| self.text(&s, small, color, term));
        // A symbol with its own superscript is never raised to a power (the registry refuses
        // it), so at most one of the two is set.
        let sup = power.or(own_sup);
        Self::scripts(base, sub, sup, size)
    }

    /// `content` between `open` and `close` (either may be empty), the delimiters scaled to
    /// its height and centred on it.
    fn fenced(&self, open: &str, content: Laid, close: &str, size: f32) -> Laid {
        let height = content.ascent + content.descent;
        let dsize = size.max(height * 0.85);
        let centre = (content.descent - content.ascent) / 2.0;
        let delim = |s: &str| {
            let mut d = self.plain(s, dsize);
            let h = d.ascent + d.descent;
            d.ascent = h / 2.0 - centre;
            d.descent = h / 2.0 + centre;
            d
        };
        let mut parts = Vec::new();
        if !open.is_empty() {
            parts.push(delim(open));
        }
        parts.push(content);
        if !close.is_empty() {
            parts.push(delim(close));
        }
        Laid::row(parts)
    }

    /// `content` between ceiling (`ceil`) or floor brackets, drawn with strokes.
    fn brackets(&self, content: Laid, ceil: bool, size: f32) -> Laid {
        let w = 0.3 * size;
        let pad = 0.1 * size;
        let top = 0.0;
        let bottom = content.ascent + content.descent;
        let stroke = (size / 14.0).max(1.0);
        let serif = if ceil { top } else { bottom };
        let width = w + pad + content.width + pad + w;
        let right = width - 0.1 * size;
        let left = 0.1 * size;
        let mut inks = vec![
            Ink::Path {
                points: vec![
                    vec2(left + w * 0.6, serif),
                    vec2(left, serif),
                    vec2(left, if ceil { bottom } else { top }),
                ],
                width: stroke,
            },
            Ink::Path {
                points: vec![
                    vec2(right - w * 0.6, serif),
                    vec2(right, serif),
                    vec2(right, if ceil { bottom } else { top }),
                ],
                width: stroke,
            },
        ];
        let ascent = content.ascent;
        let descent = content.descent;
        inks.extend(
            content
                .inks
                .into_iter()
                .map(|i| i.shifted(vec2(w + pad, 0.0))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// A stacked fraction, its bar on the math axis.
    fn frac(num: Laid, den: Laid, size: f32) -> Laid {
        let pad = 0.15 * size;
        let gap = 0.12 * size;
        let axis = 0.28 * size;
        let width = num.width.max(den.width) + 2.0 * pad;
        let num_h = num.ascent + num.descent;
        let den_h = den.ascent + den.descent;
        let ascent = axis + gap + num_h;
        let descent = (den_h + gap - axis).max(0.0);
        let bar = ascent - axis;
        let mut inks: Vec<Ink> = num
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2((width - num.width) / 2.0, bar - gap - num_h)))
            .collect();
        inks.push(Ink::Line {
            from: vec2(0.0, bar),
            to: vec2(width, bar),
            width: (size / 14.0).max(1.0),
        });
        inks.extend(
            den.inks
                .into_iter()
                .map(|i| i.shifted(vec2((width - den.width) / 2.0, bar + gap))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// √ drawn with strokes: a tick, the sign down and up to the overline over `body`.
    fn sqrt(body: Laid, size: f32) -> Laid {
        let gap = 0.12 * size;
        let stroke = (size / 14.0).max(1.0);
        let sign = 0.55 * size;
        let over = body.ascent + gap;
        let ascent = over + stroke;
        let descent = body.descent;
        let base = ascent;
        let width = sign + body.width + 0.15 * size;
        let mut inks = vec![Ink::Path {
            points: vec![
                vec2(0.0, base - 0.3 * size),
                vec2(0.15 * size, base - 0.38 * size),
                vec2(0.3 * size, base + descent),
                vec2(sign, base - over),
                vec2(width, base - over),
            ],
            width: stroke,
        }];
        inks.extend(
            body.inks
                .into_iter()
                .map(|i| i.shifted(vec2(sign + 0.05 * size, ascent - body.ascent))),
        );
        Laid {
            width,
            ascent,
            descent,
            inks,
        }
    }

    /// ∈ drawn with strokes (the default fonts lack U+2208), the height of a small letter: an
    /// arc open to the right and a bar through its middle.
    fn element(size: f32) -> Laid {
        let r = 0.25 * size;
        let cx = 0.1 * size + r;
        let tip = cx + 0.15 * size;
        let stroke = (size / 14.0).max(1.0);
        let mut points = vec![vec2(tip, 0.0)];
        points.extend((0..=12).map(|i| {
            let a = std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / 12.0;
            vec2(cx + r * a.cos(), r - r * a.sin())
        }));
        points.push(vec2(tip, 2.0 * r));
        Laid {
            width: tip + 0.1 * size,
            ascent: 2.0 * r,
            descent: 0.0,
            inks: vec![
                Ink::Path {
                    points,
                    width: stroke,
                },
                Ink::Line {
                    from: vec2(cx - r, r),
                    to: vec2(tip, r),
                    width: stroke,
                },
            ],
        }
    }

    /// `op` with `under` centred beneath it; the baseline is `op`'s.
    fn under(op: Laid, under: Laid, size: f32) -> Laid {
        let width = op.width.max(under.width);
        let gap = 0.05 * size;
        let op_h = op.ascent + op.descent;
        let mut inks: Vec<Ink> = op
            .inks
            .into_iter()
            .map(|i| i.shifted(vec2((width - op.width) / 2.0, 0.0)))
            .collect();
        inks.extend(
            under
                .inks
                .into_iter()
                .map(|i| i.shifted(vec2((width - under.width) / 2.0, op_h + gap))),
        );
        Laid {
            width,
            ascent: op.ascent,
            descent: op.descent + gap + under.ascent + under.descent,
            inks,
        }
    }

    /// Σ with `n ∈ H` beneath, then the body.
    fn sum(&self, body: Laid, size: f32) -> Laid {
        let small = (size * 0.6).max(MIN_SIZE);
        let sigma = self.plain("Σ", size * 1.5);
        let set = Laid::row(vec![
            self.plain("n", small),
            Self::element(small),
            self.plain("H", small),
        ]);
        let op = Self::under(sigma, set, size);
        Laid::row(vec![op, Laid::gap(0.15 * size), body])
    }

    /// The amplitude of a `peak`, wrapped when compound so that `sin(nφ)` multiplies all of it
    /// (as `render::plain`). Elsewhere the typesetter draws only the parentheses the markup
    /// writes (the registry refuses markup that would read two ways).
    fn grouped(&self, e: &Expr, size: f32) -> Laid {
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
                | Expr::Frac(..)
        );
        let laid = self.expr(e, size);
        if atomic {
            laid
        } else {
            self.fenced("(", laid, ")", size)
        }
    }

    fn term(&self, path: &str, family: bool, size: f32, power: Option<Laid>) -> Laid {
        let symbol = if family {
            self.registry.family_symbol(path)
        } else {
            self.registry.symbol(path).map(str::to_owned)
        };
        let color = self.colors.get(path).unwrap_or(self.ink);
        match symbol {
            Some(s) => self.symbol(&s, size, color, Some(path), power),
            None => {
                let laid = self.text(&format!("[{path}]"), size, color, Some(path));
                match power {
                    Some(p) => Self::scripts(laid, None, Some(p), size),
                    None => laid,
                }
            }
        }
    }

    fn call(&self, f: Func, args: &[Expr], size: f32) -> Laid {
        let arg = |i: usize| self.expr(&args[i], size);
        match f {
            Func::Sqrt => Self::sqrt(arg(0), size),
            Func::Exp => {
                let e = self.plain("e", size);
                Self::scripts(
                    e,
                    None,
                    Some(self.expr(&args[0], Self::script_size(size))),
                    size,
                )
            }
            Func::Abs => self.fenced("|", arg(0), "|", size),
            Func::Ceil => self.brackets(arg(0), true, size),
            Func::Floor => self.brackets(arg(0), false, size),
            Func::CeilTo | Func::FloorTo => {
                let b = self.brackets(arg(0), f == Func::CeilTo, size);
                let step = self.expr(&args[1], Self::script_size(size));
                Self::scripts(b, Some(step), None, size)
            }
            Func::Fmt => Laid::row(vec![
                arg(0),
                Laid::gap(0.2 * size),
                self.text(
                    &format!("(to {} decimals)", render_number(&args[1])),
                    Self::script_size(size),
                    self.ink.gamma_multiply(0.7),
                    None,
                ),
            ]),
            Func::FmtNum => arg(0),
            Func::Concat => {
                let mut parts = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        parts.push(Laid::gap(0.25 * size));
                    }
                    parts.push(self.expr(a, size));
                }
                Laid::row(parts)
            }
            _ => {
                let mut inner = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        inner.push(self.plain(", ", size));
                    }
                    inner.push(self.expr(a, size));
                }
                let name = self.plain(f.name(), size);
                let gap = Laid::gap(0.1 * size);
                Laid::row(vec![
                    name,
                    gap,
                    self.fenced("(", Laid::row(inner), ")", size),
                ])
            }
        }
    }

    /// A condition of a `cases` arm; a selector compared for equality with a code shows the
    /// choice label.
    fn cond(&self, c: &Cond, size: f32) -> Laid {
        match c {
            Cond::And(a, b) | Cond::Or(a, b) => {
                let word = if matches!(c, Cond::And(..)) {
                    " and "
                } else {
                    " or "
                };
                Laid::row(vec![
                    self.cond(a, size),
                    self.plain(word, size),
                    self.cond(b, size),
                ])
            }
            Cond::Rel(op, a, b) => {
                // A label reads only in an equality: σ_n's n ≤ N_h compares the harmonic index
                // with the highest harmonic, not a code.
                let labelled = matches!(op, RelOp::Eq | RelOp::Ne);
                let op = match op {
                    RelOp::Lt => " < ",
                    RelOp::Le => " ≤ ",
                    RelOp::Gt => " > ",
                    RelOp::Ge => " ≥ ",
                    RelOp::Eq => " = ",
                    RelOp::Ne => " ≠ ",
                };
                let left = if labelled {
                    self.choice_or_expr(a, b, size)
                } else {
                    self.expr(a, size)
                };
                let right = if labelled {
                    self.choice_or_expr(b, a, size)
                } else {
                    self.expr(b, size)
                };
                Laid::row(vec![left, self.plain(op, size), right])
            }
        }
    }

    /// `e`, or when `e` is a code compared for equality with a selector (`other`), the choice's
    /// label.
    fn choice_or_expr(&self, e: &Expr, other: &Expr, size: f32) -> Laid {
        if let (Expr::Num { value, .. }, Expr::Term(r)) = (e, other) {
            let choices = self.registry.choices(&r.path);
            if let Some((_, label)) = choices.iter().find(|(code, _)| *code as f64 == *value) {
                return self.plain(&format!("\"{label}\""), size);
            }
        }
        self.expr(e, size)
    }

    fn expr(&self, e: &Expr, size: f32) -> Laid {
        match e {
            Expr::Num { text, .. } => self.plain(text, size),
            Expr::Text(t) => self.plain(&format!("\"{t}\""), size),
            Expr::NoneLit => self.plain("none", size),
            Expr::Pi => self.plain("π", size),
            Expr::Term(r) => self.term(&r.path, false, size, None),
            Expr::FamilyTerm(r) => self.term(&r.path, true, size, None),
            Expr::Index => self.plain("n", size),
            Expr::Local(k) => {
                let symbol = &self.formula.bindings[*k].symbol;
                self.symbol(symbol, size, self.ink, None, None)
            }
            Expr::Neg(a) => Laid::row(vec![self.plain("−", size), self.expr(a, size)]),
            Expr::Paren(a) => self.fenced("(", self.expr(a, size), ")", size),
            Expr::Bin(op, a, b) => self.binary(*op, a, b, size),
            Expr::Frac(a, b) => {
                let inner = (size * 0.9).max(MIN_SIZE + 1.0).min(size);
                Self::frac(self.expr(a, inner), self.expr(b, inner), size)
            }
            Expr::Call(f, args) => self.call(*f, args, size),
            Expr::Sum(_, body) => self.sum(self.expr(body, size), size),
            Expr::Peak(_, body) => {
                let small = (size * 0.6).max(MIN_SIZE);
                let argmax = Self::under(
                    self.plain("arg max", size),
                    self.plain("0 ≤ φ ≤ π/2", small),
                    size,
                );
                let body = Laid::row(vec![
                    self.grouped(body, size),
                    Laid::gap(0.15 * size),
                    self.plain("sin(nφ)", size),
                ]);
                Laid::row(vec![argmax, Laid::gap(0.2 * size), self.sum(body, size)])
            }
            Expr::Table { table, key, field } => {
                let symbol = tables::field(table, field).map_or(field.as_str(), |f| f.symbol);
                let name = self.symbol(symbol, size, self.ink, None, None);
                let key = match (table.as_str(), &**key) {
                    ("screw_sizes", Expr::Num { value, .. }) => {
                        match tables::lookup("screw_sizes", &Value::Num(*value), "name") {
                            Ok(Some(Value::Text(size_name))) => self.plain(&size_name, size),
                            _ => self.expr(key, size),
                        }
                    }
                    _ => self.expr(key, size),
                };
                Laid::row(vec![name, self.fenced("(", key, ")", size)])
            }
            Expr::Cases {
                arms, otherwise, ..
            } => self.cases(arms, otherwise, size),
        }
    }

    fn binary(&self, op: BinOp, a: &Expr, b: &Expr, size: f32) -> Laid {
        let thin = Laid::gap(0.17 * size);
        match op {
            BinOp::Add => Laid::row(vec![
                self.expr(a, size),
                self.plain(" + ", size),
                self.expr(b, size),
            ]),
            BinOp::Sub => Laid::row(vec![
                self.expr(a, size),
                self.plain(" − ", size),
                self.expr(b, size),
            ]),
            BinOp::Mul => {
                let numbers = matches!(a, Expr::Num { .. }) && matches!(b, Expr::Num { .. });
                // A family member's index 1 as a factor is not drawn (k_1 = N/(2 R_g)).
                let unit_factor = matches!(a, Expr::Num { value, .. } if *value == 1.0);
                if unit_factor && !numbers {
                    self.expr(b, size)
                } else if numbers {
                    Laid::row(vec![
                        self.expr(a, size),
                        self.plain(" × ", size),
                        self.expr(b, size),
                    ])
                } else {
                    Laid::row(vec![self.expr(a, size), thin, self.expr(b, size)])
                }
            }
            BinOp::Dot => Laid::row(vec![
                self.expr(a, size),
                self.plain(" · ", size),
                self.expr(b, size),
            ]),
            BinOp::Div => Laid::row(vec![
                self.expr(a, size),
                self.plain("/", size),
                self.expr(b, size),
            ]),
            BinOp::Pow => {
                let exponent = match b {
                    Expr::Paren(inner) => &**inner,
                    other => other,
                };
                let sup = self.expr(exponent, Self::script_size(size));
                match a {
                    Expr::Term(r) => self.term(&r.path, false, size, Some(sup)),
                    Expr::FamilyTerm(r) => self.term(&r.path, true, size, Some(sup)),
                    _ => Self::scripts(self.expr(a, size), None, Some(sup), size),
                }
            }
        }
    }

    /// A left brace and one row per arm: the value, then `if` and the condition; the last row
    /// `otherwise`. The block is centred on the math axis.
    fn cases(&self, arms: &[(Cond, Expr)], otherwise: &Expr, size: f32) -> Laid {
        let mut values: Vec<Laid> = arms.iter().map(|(_, v)| self.expr(v, size)).collect();
        values.push(self.expr(otherwise, size));
        let column = values.iter().map(|v| v.width).fold(0.0, f32::max);
        let mut rows = Vec::new();
        let count = values.len();
        for (i, value) in values.into_iter().enumerate() {
            let pad = Laid::gap(column - value.width + 0.8 * size);
            let tail = if i + 1 < count {
                Laid::row(vec![self.plain("if ", size), self.cond(&arms[i].0, size)])
            } else {
                self.plain("otherwise", size)
            };
            rows.push(Laid::row(vec![value, pad, tail]));
        }
        let mut block = Laid::column(rows, 0.25 * size);
        let height = block.ascent + block.descent;
        let axis = 0.28 * size;
        block.ascent = height / 2.0 + axis;
        block.descent = height / 2.0 - axis;
        let brace = self.fenced("{", block, "", size);
        Laid::row(vec![brace, Laid::gap(0.1 * size)])
    }
}

/// The text of a number literal, for `fmt`'s decimals.
fn render_number(e: &Expr) -> String {
    match e {
        Expr::Num { text, .. } => text.clone(),
        _ => "d".to_owned(),
    }
}

/// Lays out `symbol = formula` (then one line per `where` binding) at text size `size`, every
/// term in its colour from `colors` (the text colour `ink` otherwise).
pub fn layout(
    fonts: &Fonts,
    registry: &Registry,
    symbol: &str,
    formula: &Formula,
    colors: &TermColors,
    size: f32,
    ink: Color32,
) -> Laid {
    let setter = Setter {
        fonts,
        registry,
        formula,
        colors,
        ink,
    };
    let target = setter.symbol(symbol, size, ink, None, None);
    let mut lines = vec![Laid::row(vec![
        target,
        setter.plain(" = ", size),
        setter.expr(&formula.body, size),
    ])];
    for (i, b) in formula.bindings.iter().enumerate() {
        let lead = if i == 0 { "where " } else { "and " };
        lines.push(Laid::row(vec![
            Laid::gap(size),
            setter.text(lead, size, ink.gamma_multiply(0.7), None),
            setter.symbol(&b.symbol, size, ink, None, None),
            setter.plain(" = ", size),
            setter.expr(&b.expr, size),
        ]));
    }
    Laid::column(lines, 0.35 * size)
}

/// Lays out the equation of `eq` ([`layout`] with its symbol and formula).
pub fn layout_equation(
    fonts: &Fonts,
    registry: &Registry,
    eq: &Equation,
    colors: &TermColors,
    size: f32,
    ink: Color32,
) -> Laid {
    layout(fonts, registry, &eq.symbol, &eq.formula, colors, size, ink)
}

/// Paints `laid` with its top-left corner at `origin`; strokes in `ink`.
pub fn paint(painter: &egui::Painter, origin: Pos2, laid: &Laid, ink: Color32) {
    for item in &laid.inks {
        match item {
            Ink::Text { at, galley, .. } => {
                painter.galley(origin + *at, galley.clone(), ink);
            }
            Ink::Line { from, to, width } => {
                painter.line_segment([origin + *from, origin + *to], Stroke::new(*width, ink));
            }
            Ink::Path { points, width } => {
                let points: Vec<Pos2> = points.iter().map(|p| origin + *p).collect();
                painter.add(egui::Shape::line(points, Stroke::new(*width, ink)));
            }
        }
    }
}

/// Draws `eq` at text size `size` with its terms in `colors` (a tooltip's equation: it takes
/// no clicks).
pub fn equation_ui(
    ui: &mut egui::Ui,
    registry: &Registry,
    eq: &Equation,
    colors: &TermColors,
    size: f32,
) -> egui::Response {
    let ink = ui.visuals().text_color();
    let laid = ui.fonts(|f| layout_equation(f, registry, eq, colors, size, ink));
    laid_ui(ui, &laid, ink, Sense::hover())
}

/// Allocates room for `laid` with `sense` and paints it there.
pub fn laid_ui(ui: &mut egui::Ui, laid: &Laid, ink: Color32, sense: Sense) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(laid.size(), sense);
    paint(ui.painter(), rect.min, laid, ink);
    response
}

/// The term drawn at `pos` by `laid` painted at `origin` (its path, or a family template).
pub fn term_at(laid: &Laid, origin: Pos2, pos: Pos2) -> Option<String> {
    laid.term_rects()
        .into_iter()
        .find(|(_, r)| r.translate(origin.to_vec2()).contains(pos))
        .map(|(path, _)| path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DesignInputs;
    use crate::compute_all;
    use crate::engine::explain::markup::parse;
    use crate::engine::explain::{Design, render};
    use crate::gui::test_support::assert_glyphs;
    use std::sync::OnceLock;

    fn registry() -> &'static Registry {
        static REGISTRY: OnceLock<Registry> = OnceLock::new();
        REGISTRY.get_or_init(Registry::build)
    }

    /// Runs `f` with the default fonts loaded (they load on the first frame).
    fn with_fonts<R>(f: impl FnOnce(&Fonts) -> R) -> R {
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        ctx.fonts(f)
    }

    fn rect_of<'a>(texts: &'a [(String, Rect)], text: &str) -> &'a Rect {
        &texts
            .iter()
            .find(|(t, _)| t == text)
            .unwrap_or_else(|| panic!("no {text:?} in {texts:?}"))
            .1
    }

    #[test]
    fn a_fraction_a_subscript_a_sum_and_a_root_are_laid_out_in_reading_order() {
        let formula = parse(
            "frac({coupling.c_end}, 2) + sqrt({model.pole_pitch_mm}) + sum(n in H: {model.tau#_Pa})",
            None,
        )
        .unwrap();
        let laid = with_fonts(|f| {
            layout(
                f,
                registry(),
                "x_{a}",
                &formula,
                &TermColors::none(),
                20.0,
                Color32::WHITE,
            )
        });
        let texts = laid.texts();
        let order: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            order,
            [
                "x", "a", " = ", "c", "end", "2", " + ", "τ", "p", " + ", "Σ", "n", "H", "σ", "n"
            ]
        );
        let (x, a) = (rect_of(&texts, "x"), rect_of(&texts, "a"));
        assert!(
            a.left() >= x.right() - 0.5,
            "the subscript follows its base"
        );
        assert!(a.center().y > x.center().y, "and sits lower");
        assert!(a.height() < x.height(), "and is smaller");
        // The fraction: c_end over the bar over 2.
        let (c, two) = (rect_of(&texts, "c"), rect_of(&texts, "2"));
        assert!(
            c.bottom() <= two.top(),
            "the numerator is above the denominator"
        );
        let bar = laid
            .inks
            .iter()
            .find_map(|i| match i {
                Ink::Line { from, to, .. } if from.y == to.y => Some((*from, *to)),
                _ => None,
            })
            .expect("a fraction bar");
        assert!(c.bottom() <= bar.0.y && bar.0.y <= two.top());
        assert!(bar.0.x <= c.left() && c.right() <= bar.1.x);
        // The radical's overline runs over τ_p.
        let tau = rect_of(&texts, "τ");
        let radical = laid
            .inks
            .iter()
            .find_map(|i| match i {
                Ink::Path { points, .. } if points.len() == 5 => Some(points.clone()),
                _ => None,
            })
            .expect("a radical");
        let overline_y = radical[3].y;
        assert!(
            overline_y <= tau.top(),
            "the overline is above the radicand"
        );
        assert!(radical[3].x <= tau.left() && tau.right() <= radical[4].x);
        // Σ is larger than the text, with n and H beneath it.
        let sigma = rect_of(&texts, "Σ");
        assert!(sigma.height() > x.height());
        let n = rect_of(&texts, "n");
        assert!(n.top() >= sigma.bottom() - 2.0, "n ∈ H is beneath the Σ");
        assert!(laid.width > 0.0 && laid.ascent > 0.0 && laid.descent > 0.0);
    }

    #[test]
    fn terms_carry_their_colours_and_paths() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let colors = TermColors::of(eq);
        // In the order the formula shows them: T_2D, f_end, f_cal.
        assert_eq!(colors.get("model.torque_2d_Nm"), Some(TERM_PALETTE[0]));
        assert_eq!(colors.get("model.f_end"), Some(TERM_PALETTE[1]));
        assert_eq!(colors.get("model.f_cal"), Some(TERM_PALETTE[2]));
        let laid =
            with_fonts(|f| layout_equation(f, registry(), eq, &colors, 16.0, Color32::WHITE));
        let paths: Vec<String> = laid.term_rects().into_iter().map(|(p, _)| p).collect();
        assert!(paths.contains(&"model.f_end".to_owned()), "{paths:?}");
        let f_end_color = laid.inks.iter().find_map(|i| match i {
            Ink::Text {
                galley,
                term: Some(t),
                ..
            } if t == "model.f_end" => galley.job.sections.first().map(|s| s.format.color),
            _ => None,
        });
        assert_eq!(f_end_color, Some(TERM_PALETTE[1]));
    }

    #[test]
    fn a_ninth_term_takes_the_first_colour_again() {
        // C_th reads 14 terms: eight colours, then the first again (decision M43-3), in the
        // formula tree's order, where a cases arm's condition (the circuit in effect) comes
        // before its value.
        let eq = registry()
            .equation_for("temperature.thermal.heat_capacity_J_K")
            .unwrap();
        let colors = TermColors::of(eq);
        assert_eq!(
            colors.get("materials.circuit_backiron"),
            Some(TERM_PALETTE[0])
        );
        assert_eq!(colors.get("mass.magnets_g"), Some(TERM_PALETTE[1]));
        assert_eq!(
            colors.get("temperature.thermal.steel_c"),
            Some(TERM_PALETTE[0]),
            "the ninth term"
        );
        assert_eq!(
            colors.get("retainers.retainers_g"),
            Some(TERM_PALETTE[1]),
            "the tenth"
        );
    }

    #[test]
    fn only_the_parentheses_the_markup_writes_are_drawn() {
        // T_pull = T_2D f_end f_cal: a product of products, no parentheses.
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(!texts.iter().any(|t| t == "(" || t == ")"), "{texts:?}");
        // C_cup's markup writes them: 2 (r_corner + t_wall).
        let eq = registry().equation_for("model.cup_od_mm").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert_eq!(texts.iter().filter(|t| *t == "(").count(), 1, "{texts:?}");
        assert_eq!(texts.iter().filter(|t| *t == ")").count(), 1);
    }

    #[test]
    fn a_family_term_is_coloured_by_template_and_marks_only_the_harmonics_summed() {
        let eq = registry().equation_for("model.tau_Pa").unwrap();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let design = Design {
            inputs: &inputs,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        let template = colors.get("model.tau#_Pa").expect("the template");
        for n in [1, 3, 5] {
            assert_eq!(colors.get(&format!("model.tau{n}_Pa")), Some(template));
        }
        // The workbook sums 1, 3, 5: 7 is not marked.
        assert_eq!(colors.get("model.tau7_Pa"), None);
        assert!(TermColors::none().is_empty());
        // Every odd harmonic up to 11: all six are marked.
        let mut all = DesignInputs::default();
        all.coupling.max_harmonic = 11;
        let results = compute_all(&all);
        let design = Design {
            inputs: &all,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        for n in [1, 3, 5, 7, 9, 11] {
            assert_eq!(colors.get(&format!("model.tau{n}_Pa")), Some(template));
        }
        // A code outside the choices sums nothing: only the template keeps its colour.
        let mut odd = DesignInputs::default();
        odd.coupling.max_harmonic = 4;
        let results = compute_all(&odd);
        let design = Design {
            inputs: &odd,
            results: &results,
        };
        let colors = TermColors::of(eq).with_members(registry(), &design);
        assert_eq!(colors.get("model.tau1_Pa"), None);
        assert_eq!(colors.get("model.tau#_Pa"), Some(template));
    }

    #[test]
    fn a_selector_code_shows_its_choice_and_a_screw_row_its_size() {
        // A_1 compares the circuit in effect with code 1, the steel circuit.
        let eq = registry().equation_for("model.amp1_Pa").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(texts.iter().any(|t| t == "\"steel circuit\""), "{texts:?}");
        // σ_5 = A_5 · sin(5 φ_pull) if 5 ≤ N_h: an ordering keeps its number, though N_h is a
        // selector whose code 5 reads "1, 3, 5 (workbook)" (the index 5 is no code). The tau
        // records hold no text literal, so no run is quoted.
        let tau = registry().equation_for("model.tau5_Pa").unwrap();
        let laid = with_fonts(|f| {
            layout_equation(
                f,
                registry(),
                tau,
                &TermColors::none(),
                14.0,
                Color32::WHITE,
            )
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(!texts.iter().any(|t| t.starts_with('"')), "{texts:?}");
        let screw = registry()
            .equation_for("clamps.table[2].engagement_req_mm")
            .expect("a screw-size row");
        let laid = with_fonts(|f| {
            layout_equation(
                f,
                registry(),
                screw,
                &TermColors::none(),
                14.0,
                Color32::WHITE,
            )
        });
        let texts: Vec<String> = laid.texts().into_iter().map(|(t, _)| t).collect();
        assert!(texts.iter().any(|t| t == "M4"), "{texts:?}");
    }

    /// The quoted labels of the selectors `cond` compares for equality with a code, as the
    /// typesetter draws them.
    fn equality_labels(cond: &Cond, out: &mut Vec<String>) {
        match cond {
            Cond::And(a, b) | Cond::Or(a, b) => {
                equality_labels(a, out);
                equality_labels(b, out);
            }
            Cond::Rel(RelOp::Eq | RelOp::Ne, a, b) => {
                for side in [a, b] {
                    if let Expr::Term(r) = side {
                        out.extend(
                            registry()
                                .choices(&r.path)
                                .iter()
                                .map(|(_, label)| glyph_safe(&format!("\"{label}\""))),
                        );
                    }
                }
            }
            Cond::Rel(..) => {}
        }
    }

    #[test]
    fn every_equation_draws_only_what_its_record_says() {
        // The glyph test checks characters, not meaning. Here every text run the typesetter
        // draws must come from the record's plain rendering (`render::plain`, with θ for ϑ as
        // drawn), from a choice label of a selector the formula compares for equality, from
        // the size name of a `screw_sizes` row it reads, or from the words only the typesetter
        // writes. A stray substitution is a run with no source.
        const TYPESET_ONLY: [&str; 3] = ["and", "arg max", "0 ≤ φ ≤ π/2"];
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for eq in registry().equations() {
            let mut sources = vec![glyph_safe(&render::plain(
                registry(),
                &eq.symbol,
                &eq.formula,
            ))];
            eq.formula.visit(&mut |e| match e {
                Expr::Cases { arms, .. } => {
                    for (cond, _) in arms {
                        equality_labels(cond, &mut sources);
                    }
                }
                Expr::Table { table, key, .. } if table == "screw_sizes" => {
                    if let Expr::Num { value, .. } = &**key
                        && let Ok(Some(Value::Text(name))) =
                            tables::lookup("screw_sizes", &Value::Num(*value), "name")
                    {
                        sources.push(name);
                    }
                }
                _ => {}
            });
            let laid = ctx.fonts(|f| {
                layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
            });
            for (text, _) in laid.texts() {
                let run = text.trim();
                assert!(
                    run.is_empty()
                        || TYPESET_ONLY.contains(&run)
                        || sources.iter().any(|s| s.contains(run)),
                    "{}: {run:?} is in none of {sources:?}",
                    eq.target
                );
            }
        }
    }

    /// A text run of a drawn equation: the term path it carries, if any, and its text
    /// (trimmed).
    type Run = (Option<String>, String);

    /// Every text run of `laid` in drawing order: its term path, its trimmed text and its
    /// colour.
    fn runs_of(laid: &Laid) -> Vec<(Option<String>, String, Color32)> {
        laid.inks
            .iter()
            .filter_map(|ink| match ink {
                Ink::Text { galley, term, .. } => Some((
                    term.clone(),
                    galley.text().trim().to_owned(),
                    galley
                        .job
                        .sections
                        .first()
                        .map_or(Color32::PLACEHOLDER, |s| s.format.color),
                )),
                _ => None,
            })
            .collect()
    }

    fn counted(runs: impl IntoIterator<Item = Run>) -> BTreeMap<Run, usize> {
        let mut counts = BTreeMap::new();
        for run in runs {
            *counts.entry(run).or_insert(0) += 1;
        }
        counts
    }

    /// What a formula's tree says must be drawn, as the text runs it takes, read off the
    /// markup's table in `engine::explain::markup` and not off the typesetter: a number's
    /// text, a term's symbol (base, subscript and superscript, each carrying the term's
    /// path), a symbol of a `where` binding or a table field, the operators, the words. A
    /// number the typesetter replaces is wanted as its replacement: a choice label of an
    /// equality, a screw size's name, `fmt`'s decimals, the factor 1 not drawn. Structure
    /// without a text run (the stacked fraction's bar, a radical, a bracket's strokes) is
    /// pinned by the tests of its geometry.
    struct Wanted<'a> {
        formula: &'a Formula,
        runs: Vec<Run>,
    }

    impl<'a> Wanted<'a> {
        fn of(eq: &'a Equation) -> Self {
            let mut wanted = Wanted {
                formula: &eq.formula,
                runs: Vec::new(),
            };
            wanted.symbol(&eq.symbol, None);
            wanted.push(None, "=");
            wanted.expr(&eq.formula.body);
            for (i, b) in eq.formula.bindings.iter().enumerate() {
                wanted.push(None, if i == 0 { "where" } else { "and" });
                wanted.symbol(&b.symbol, None);
                wanted.push(None, "=");
                wanted.expr(&b.expr);
            }
            wanted
        }

        fn push(&mut self, term: Option<&str>, text: &str) {
            self.runs
                .push((term.map(str::to_owned), glyph_safe(text).trim().to_owned()));
        }

        /// A symbol: its base, subscript and superscript, one run each.
        fn symbol(&mut self, markup: &str, term: Option<&str>) {
            let parts = match Symbol::parse(markup) {
                Ok(s) => vec![Some(s.base), s.sub, s.sup],
                Err(_) => vec![Some(markup.to_owned())],
            };
            for part in parts.into_iter().flatten() {
                self.push(term, &part);
            }
        }

        fn term(&mut self, path: &str, family: bool) {
            let symbol = if family {
                registry().family_symbol(path)
            } else {
                registry().symbol(path).map(str::to_owned)
            };
            match symbol {
                Some(s) => self.symbol(&s, Some(path)),
                None => self.push(Some(path), &format!("[{path}]")),
            }
        }

        /// `e`, or the quoted label of the choice when `e` is a code compared with the
        /// selector `other`.
        fn choice_or_expr(&mut self, e: &Expr, other: &Expr) {
            if let (Expr::Num { value, .. }, Expr::Term(r)) = (e, other)
                && let Some((_, label)) = registry()
                    .choices(&r.path)
                    .iter()
                    .find(|(code, _)| *code as f64 == *value)
            {
                self.push(None, &format!("\"{label}\""));
            } else {
                self.expr(e);
            }
        }

        fn cond(&mut self, c: &Cond) {
            match c {
                Cond::And(a, b) | Cond::Or(a, b) => {
                    self.cond(a);
                    self.push(
                        None,
                        if matches!(c, Cond::And(..)) {
                            "and"
                        } else {
                            "or"
                        },
                    );
                    self.cond(b);
                }
                Cond::Rel(op, a, b) => {
                    let labelled = matches!(op, RelOp::Eq | RelOp::Ne);
                    if labelled {
                        self.choice_or_expr(a, b);
                    } else {
                        self.expr(a);
                    }
                    self.push(
                        None,
                        match op {
                            RelOp::Lt => "<",
                            RelOp::Le => "≤",
                            RelOp::Gt => ">",
                            RelOp::Ge => "≥",
                            RelOp::Eq => "=",
                            RelOp::Ne => "≠",
                        },
                    );
                    if labelled {
                        self.choice_or_expr(b, a);
                    } else {
                        self.expr(b);
                    }
                }
            }
        }

        fn expr(&mut self, e: &Expr) {
            match e {
                Expr::Num { text, .. } => self.push(None, text),
                Expr::Text(t) => self.push(None, &format!("\"{t}\"")),
                Expr::NoneLit => self.push(None, "none"),
                Expr::Pi => self.push(None, "π"),
                Expr::Index => self.push(None, "n"),
                Expr::Term(r) => self.term(&r.path, false),
                Expr::FamilyTerm(r) => self.term(&r.path, true),
                Expr::Local(k) => {
                    let formula = self.formula;
                    self.symbol(&formula.bindings[*k].symbol, None);
                }
                Expr::Neg(a) => {
                    self.push(None, "−");
                    self.expr(a);
                }
                Expr::Paren(a) => self.expr(a),
                Expr::Frac(a, b) => {
                    self.expr(a);
                    self.expr(b);
                }
                Expr::Bin(op, a, b) => match op {
                    BinOp::Mul => {
                        let numbers =
                            matches!(**a, Expr::Num { .. }) && matches!(**b, Expr::Num { .. });
                        let unit = matches!(**a, Expr::Num { value, .. } if value == 1.0);
                        if unit && !numbers {
                            self.expr(b);
                        } else {
                            self.expr(a);
                            if numbers {
                                self.push(None, "×");
                            }
                            self.expr(b);
                        }
                    }
                    BinOp::Add | BinOp::Sub | BinOp::Dot | BinOp::Div => {
                        self.expr(a);
                        self.push(
                            None,
                            match op {
                                BinOp::Add => "+",
                                BinOp::Sub => "−",
                                BinOp::Dot => "·",
                                _ => "/",
                            },
                        );
                        self.expr(b);
                    }
                    BinOp::Pow => {
                        self.expr(a);
                        self.expr(b);
                    }
                },
                Expr::Call(f, args) => match f {
                    Func::Exp => {
                        self.push(None, "e");
                        self.expr(&args[0]);
                    }
                    Func::Abs => {
                        self.push(None, "|");
                        self.push(None, "|");
                        self.expr(&args[0]);
                    }
                    Func::Fmt => {
                        self.expr(&args[0]);
                        let decimals = match &args[1] {
                            Expr::Num { text, .. } => text.as_str(),
                            _ => "d",
                        };
                        self.push(None, &format!("(to {decimals} decimals)"));
                    }
                    Func::Sqrt
                    | Func::Ceil
                    | Func::Floor
                    | Func::CeilTo
                    | Func::FloorTo
                    | Func::FmtNum
                    | Func::Concat => args.iter().for_each(|a| self.expr(a)),
                    _ => {
                        self.push(None, f.name());
                        args.iter().for_each(|a| self.expr(a));
                    }
                },
                Expr::Sum(_, body) => {
                    ["Σ", "n", "H"].iter().for_each(|s| self.push(None, s));
                    self.expr(body);
                }
                Expr::Peak(_, body) => {
                    ["arg max", "0 ≤ φ ≤ π/2", "sin(nφ)", "Σ", "n", "H"]
                        .iter()
                        .for_each(|s| self.push(None, s));
                    self.expr(body);
                }
                Expr::Table { table, key, field } => {
                    let symbol = tables::field(table, field).map_or(field.as_str(), |f| f.symbol);
                    self.symbol(symbol, None);
                    match (table.as_str(), &**key) {
                        ("screw_sizes", Expr::Num { value, .. }) => {
                            match tables::lookup("screw_sizes", &Value::Num(*value), "name") {
                                Ok(Some(Value::Text(name))) => self.push(None, &name),
                                _ => self.expr(key),
                            }
                        }
                        _ => self.expr(key),
                    }
                }
                Expr::Cases {
                    arms, otherwise, ..
                } => {
                    for (cond, value) in arms {
                        self.expr(value);
                        self.push(None, "if");
                        self.cond(cond);
                    }
                    self.push(None, "otherwise");
                    self.expr(otherwise);
                }
            }
        }
    }

    /// The operator runs the typesetter draws as text, each for one markup node only (a
    /// stroke or a delimiter is not among them).
    const OPERATORS: [&str; 12] = ["+", "−", "×", "·", "/", "=", "≠", "<", "≤", ">", "≥", "|"];

    /// Whether a run with no term is a number literal or an operator: the runs whose count is
    /// exact.
    fn counted_exactly(text: &str) -> bool {
        OPERATORS.contains(&text)
            || (text.starts_with(|c: char| c.is_ascii_digit()) && text.parse::<f64>().is_ok())
    }

    #[test]
    fn every_equation_draws_everything_its_record_says() {
        // The reverse of `every_equation_draws_only_what_its_record_says`: that one fails a run
        // with no source, this one a source with no run (a dropped exponent, minus sign,
        // `where` line, rounding step or bar). Every run the tree wants (`Wanted`) is drawn at
        // least as many times as the tree has it, by the term path it carries: counted, not
        // looked up, since "2" is drawn somewhere in most equations. Numbers and operators are
        // counted both ways: one more than the tree has (a unit factor 1 drawn, a sign for a
        // product) is as wrong as one fewer.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut wanted_runs = 0;
        for eq in registry().equations() {
            let wanted = Wanted::of(eq);
            wanted_runs += wanted.runs.len();
            let wanted = counted(wanted.runs);
            let laid = ctx.fonts(|f| {
                layout_equation(f, registry(), eq, &TermColors::none(), 14.0, Color32::WHITE)
            });
            let drawn = counted(
                runs_of(&laid)
                    .into_iter()
                    .map(|(term, text, _)| (term, text)),
            );
            for (run, count) in &wanted {
                let have = drawn.get(run).copied().unwrap_or(0);
                assert!(
                    have >= *count,
                    "{}: {count} x {run:?} wanted, {have} drawn, of {drawn:?}",
                    eq.target
                );
            }
            for ((term, text), have) in &drawn {
                if term.is_none() && counted_exactly(text) {
                    let count = wanted.get(&(None, text.clone())).copied().unwrap_or(0);
                    assert_eq!(
                        *have, count,
                        "{}: {text:?} drawn {have} times, wanted {count}",
                        eq.target
                    );
                }
            }
        }
        assert!(wanted_runs > 4000, "{wanted_runs}");
    }

    #[test]
    fn every_run_carries_its_terms_path_and_colour_and_the_target_is_plain() {
        // M43-3: a term is drawn, all of its runs (base, subscript, superscript), in its
        // colour and with its path, so a click on any of them finds it; the target symbol is in
        // the text colour with no path; nothing but a term takes a palette colour.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let ink = Color32::WHITE;
        let mut term_runs = 0;
        for eq in registry().equations() {
            let colors = TermColors::of(eq);
            let laid = ctx.fonts(|f| layout_equation(f, registry(), eq, &colors, 14.0, ink));
            let runs = runs_of(&laid);
            let mut target = Wanted {
                formula: &eq.formula,
                runs: Vec::new(),
            };
            target.symbol(&eq.symbol, None);
            assert!(runs.len() >= target.runs.len(), "{}", eq.target);
            for ((term, text, color), (_, want)) in runs.iter().zip(&target.runs) {
                assert_eq!(text, want, "{}: the target symbol comes first", eq.target);
                assert_eq!((term, color), (&None, &ink), "{}: {text:?}", eq.target);
            }
            for (term, text, color) in &runs {
                match term {
                    Some(path) => {
                        term_runs += 1;
                        assert_eq!(
                            Some(*color),
                            colors.get(path),
                            "{}: {text:?} of {path}",
                            eq.target
                        );
                    }
                    None => assert!(
                        !TERM_PALETTE.contains(color),
                        "{}: {text:?} has no term but the colour {color:?}",
                        eq.target
                    ),
                }
            }
        }
        assert!(term_runs > 1000, "{term_runs}");
    }

    #[test]
    fn every_equation_typesets_with_glyphs_in_the_default_fonts() {
        // Every Expr variant the records use, at a tooltip's size and the panel's: nothing
        // panics, every box is finite, and every character drawn has a glyph (egui draws an
        // empty box for one it lacks).
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let mut checked = 0;
        for eq in registry().equations() {
            for size in [14.0, 22.0] {
                let colors = TermColors::of(eq);
                let laid = ctx
                    .fonts(|f| layout_equation(f, registry(), eq, &colors, size, Color32::WHITE));
                assert!(
                    laid.width.is_finite() && laid.width > 0.0,
                    "{}: width {}",
                    eq.target,
                    laid.width
                );
                assert!(laid.ascent.is_finite() && laid.descent.is_finite());
                for (text, rect) in laid.texts() {
                    assert!(rect.is_finite(), "{}: {text:?} at {rect:?}", eq.target);
                    assert_glyphs(&ctx, &text, &eq.target);
                }
            }
            checked += 1;
        }
        assert_eq!(checked, registry().equations().len());
        assert!(checked > 300, "{checked}");
    }

    #[test]
    fn theta_is_drawn_for_the_vartheta_the_fonts_lack() {
        assert_eq!(glyph_safe("ϑ_{op}"), "θ_{op}");
        assert_eq!(glyph_safe("T_{pull}"), "T_{pull}");
        assert_eq!(glyph_safe("13 × 10⁻⁶ per °C"), "13 × 10¯⁶ per °C");
        assert_eq!(glyph_safe("torque ∝ Br²"), "torque ~ Br²");
    }

    #[test]
    fn clicking_a_term_of_a_drawn_equation_names_it() {
        let ctx = egui::Context::default();
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let colors = TermColors::of(eq);
        // One frame drawing the equation: where it was drawn, its layout, the term clicked.
        let run = |events: Vec<egui::Event>| {
            let mut drawn = (Pos2::ZERO, Laid::default(), None);
            let input = egui::RawInput {
                events,
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(800.0, 400.0))),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let origin = ui.cursor().min;
                    let laid = ui.fonts(|f| {
                        layout_equation(f, registry(), eq, &colors, 20.0, Color32::WHITE)
                    });
                    let response = laid_ui(ui, &laid, Color32::WHITE, Sense::click());
                    let clicked = response
                        .interact_pointer_pos()
                        .filter(|_| response.clicked())
                        .and_then(|at| term_at(&laid, response.rect.min, at));
                    drawn = (origin, laid, clicked);
                });
            });
            drawn
        };
        let (origin, laid, _) = run(Vec::new());
        let (_, rect) = laid
            .term_rects()
            .into_iter()
            .find(|(p, _)| p == "model.f_end")
            .unwrap();
        let at = rect.translate(origin.to_vec2()).center();
        run(vec![egui::Event::PointerMoved(at)]);
        run(vec![crate::gui::test_support::primary_button(at, true)]);
        let (_, _, clicked) = run(vec![crate::gui::test_support::primary_button(at, false)]);
        assert_eq!(clicked.as_deref(), Some("model.f_end"));
    }

    /// `symbol = src` at 20 points in white, no term colours.
    fn laid_formula(symbol: &str, src: &str) -> Laid {
        let formula = parse(src, None).unwrap();
        with_fonts(|f| {
            layout(
                f,
                registry(),
                symbol,
                &formula,
                &TermColors::none(),
                20.0,
                Color32::WHITE,
            )
        })
    }

    fn run_order(laid: &Laid) -> Vec<String> {
        laid.texts().into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn a_power_raises_its_exponent_over_the_subscript() {
        // c_end^2: the exponent is the superscript over the subscript, both after the base.
        let laid = laid_formula("x", "{coupling.c_end}^2");
        assert_eq!(run_order(&laid), ["x", " = ", "c", "end", "2"]);
        let texts = laid.texts();
        let (c, end, two) = (
            rect_of(&texts, "c"),
            rect_of(&texts, "end"),
            rect_of(&texts, "2"),
        );
        assert!(two.center().y < c.center().y, "raised above the base");
        assert!(
            c.center().y < end.center().y,
            "the subscript stays below it"
        );
        assert!(
            (two.left() - end.left()).abs() < 0.5,
            "stacked over the subscript"
        );
        assert!(two.left() >= c.right() - 0.5, "after the base");
        assert!(two.height() < c.height(), "and smaller");
    }

    #[test]
    fn a_unary_minus_is_drawn_before_its_operand_and_apart_from_a_subtraction() {
        // e^(-τ_p): the sign leads the exponent, in the script size.
        let laid = laid_formula("x", "exp(-{model.pole_pitch_mm})");
        assert_eq!(run_order(&laid), ["x", " = ", "e", "−", "τ", "p"]);
        let texts = laid.texts();
        let (e, minus, tau) = (
            rect_of(&texts, "e"),
            rect_of(&texts, "−"),
            rect_of(&texts, "τ"),
        );
        assert!(minus.right() <= tau.left() + 0.5, "before its operand");
        assert!(minus.center().y < e.center().y, "in the exponent");
        assert!(minus.height() < e.height(), "at the script size");
        // A subtraction sets its sign between the operands, in spaces.
        let laid = laid_formula("x", "{coupling.c_end} - {model.pole_pitch_mm}");
        assert_eq!(
            run_order(&laid),
            ["x", " = ", "c", "end", " − ", "τ", "p"],
            "a subtraction"
        );
    }

    #[test]
    fn where_lines_follow_the_formula_one_binding_each() {
        let laid = laid_formula(
            "x",
            "{coupling.c_end} + [S_a] where [S_a] = 2 * {model.pole_pitch_mm}",
        );
        assert_eq!(
            run_order(&laid),
            [
                "x", " = ", "c", "end", " + ", "S", "a", "where ", "S", "a", " = ", "2", "τ", "p"
            ]
        );
        let texts = laid.texts();
        let (x, lead) = (rect_of(&texts, "x"), rect_of(&texts, "where "));
        assert!(lead.top() > x.bottom(), "the line is under the formula");
        assert!(lead.left() >= 20.0 - 0.01, "and indented by one em");
        // Two bindings: the second line leads with "and".
        let laid = laid_formula("x", "[A] + [B] where [A] = {coupling.c_end}, [B] = [A] * 2");
        assert_eq!(
            run_order(&laid),
            [
                "x", " = ", "A", " + ", "B", "where ", "A", " = ", "c", "end", "and ", "B", " = ",
                "A", "2"
            ]
        );
        let texts = laid.texts();
        assert!(rect_of(&texts, "and ").top() > rect_of(&texts, "where ").bottom());
    }

    #[test]
    fn factors_sit_side_by_side_and_only_two_numbers_take_a_times_sign() {
        // a * b: juxtaposition, × only between two numbers; a · b: a dot; a / b: inline; a
        // factor 1 is not drawn (a family member's index, k_1 = N/(2 R_g)), unless it is
        // the first of two numbers.
        let order = |src: &str| run_order(&laid_formula("x", src));
        assert_eq!(order("2 * 3"), ["x", " = ", "2", " × ", "3"]);
        assert_eq!(order("1 * 2"), ["x", " = ", "1", " × ", "2"]);
        assert_eq!(order("2 * {coupling.c_end}"), ["x", " = ", "2", "c", "end"]);
        assert_eq!(order("{coupling.c_end} * 2"), ["x", " = ", "c", "end", "2"]);
        assert_eq!(order("1 * {coupling.c_end}"), ["x", " = ", "c", "end"]);
        assert_eq!(order("2 · 3"), ["x", " = ", "2", " · ", "3"]);
        assert_eq!(order("2 / 3"), ["x", " = ", "2", "/", "3"]);
    }

    /// The 3-point strokes of the brackets of `laid` (a ceiling or floor's two sides, each
    /// serif tip, corner, end), in drawing order.
    fn bracket_strokes(laid: &Laid) -> Vec<Vec<Vec2>> {
        laid.inks
            .iter()
            .filter_map(|i| match i {
                Ink::Path { points, .. } if points.len() == 3 => Some(points.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_ceiling_has_its_serifs_on_top_and_a_floor_beneath() {
        let laid = laid_formula("x", "ceil(2.5) + floor(2.5)");
        let strokes = bracket_strokes(&laid);
        assert_eq!(strokes.len(), 4, "two sides each");
        let texts = laid.texts();
        let nums: Vec<Rect> = texts
            .iter()
            .filter(|(t, _)| t == "2.5")
            .map(|(_, r)| *r)
            .collect();
        assert_eq!(nums.len(), 2);
        let near = |a: f32, b: f32| (a - b).abs() < 0.5;
        // Each side: corner at the serif's end, the stroke running the height of the number.
        for side in &strokes[0..2] {
            assert!(near(side[1].y, nums[0].top()), "ceil: serif at the top");
            assert!(near(side[2].y, nums[0].bottom()), "and the stroke down");
        }
        for side in &strokes[2..4] {
            assert!(
                near(side[1].y, nums[1].bottom()),
                "floor: serif at the bottom"
            );
            assert!(near(side[2].y, nums[1].top()), "and the stroke up");
        }
        // The pair hugs its number, the serifs pointing in.
        for (pair, num) in [(&strokes[0..2], nums[0]), (&strokes[2..4], nums[1])] {
            let (left, right) = (&pair[0], &pair[1]);
            assert!(left[1].x < num.left() && num.right() < right[1].x);
            assert!(left[0].x > left[1].x && right[0].x < right[1].x);
        }
    }

    #[test]
    fn a_rounding_step_is_drawn_beneath_the_brackets_it_rounds_to() {
        for (call, ceil) in [("ceilto", true), ("floorto", false)] {
            let laid = laid_formula("x", &format!("{call}({{model.pole_pitch_mm}}, 0.5)"));
            assert_eq!(
                run_order(&laid),
                ["x", " = ", "τ", "p", "0.5"],
                "{call}: the step is drawn"
            );
            let texts = laid.texts();
            let (tau, step) = (rect_of(&texts, "τ"), rect_of(&texts, "0.5"));
            assert!(step.center().y > tau.center().y, "{call}: beneath");
            assert!(step.left() > rect_of(&texts, "p").right(), "{call}: after");
            assert!(step.height() < tau.height(), "{call}: smaller");
            let strokes = bracket_strokes(&laid);
            assert_eq!(strokes.len(), 2, "{call}");
            for side in &strokes {
                assert_eq!(
                    side[1].y < side[2].y,
                    ceil,
                    "{call}: serif on top only for ceil"
                );
            }
        }
    }
}
