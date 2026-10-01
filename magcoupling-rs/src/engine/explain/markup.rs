//! The equation markup (Addendum A2): one text per explained result that is both what the
//! equation panel typesets and what the drift guard evaluates, so the equation shown is the
//! one that produced the number (the spec's A2 guarantee).
//!
//! # Formula grammar (EBNF)
//!
//! ```text
//! formula  = expr , [ "where" , binding , { "," , binding } ] ;
//! binding  = local , "=" , expr ;                 (* a named sub-expression *)
//! local    = "[" , symbol , "]" ;                 (* its name, in symbol markup *)
//! expr     = additive ;
//! cases    = "cases" , "(" , arm , { ";" , arm } , ";" , "else" , "=>" , expr , ")" ;
//! arm      = cond , "=>" , expr ;
//! cond     = conj , { "or" , conj } ;
//! conj     = rel , { "and" , rel } ;
//! rel      = additive , ( "<" | "<=" | ">" | ">=" | "=" | "!=" ) , additive ;
//! additive = product , { ( "+" | "-" ) , product } ;
//! product  = unary , { ( "*" | "·" | "/" ) , unary } ;
//! unary    = "-" , unary | power ;
//! power    = atom , [ "^" , unary ] ;             (* right-associative *)
//! atom     = number | text | term | local | "n" | "pi" | "π" | "none" | "inf" | "nan"
//!          | call | cases | sum | peak | table | "(" , expr , ")" ;
//! number   = digit , { digit } , [ "." , digit , { digit } ] , [ ( "e" | "E" ) , [ "-" | "+" ] , digit , { digit } ] ;
//! text     = '"' , { char - '"' } , '"' ;
//! term     = "{" , path , [ "|" , unit ] , "}" ;   (* an input or result path; "#" = the index *)
//! call     = func , "(" , expr , { "," , expr } , ")" ;
//! func     = "frac" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp"
//!          | "ln" | "abs" | "min" | "max" | "ceil" | "floor" | "ceilto" | "floorto"
//!          | "fmt" | "fmtnum" | "concat" ;
//! sum      = "sum" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! peak     = "peak" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! table    = "table" , "(" , text , "," , expr , "," , text , ")" ;
//! ```
//!
//! Whitespace separates tokens and is otherwise ignored. There is no implicit
//! multiplication: `2*{coupling.mu0}`, never `2{coupling.mu0}`.
//!
//! # Meaning (what the evaluator computes and the typesetter draws)
//!
//! | Markup | Evaluates to | Typeset as |
//! |---|---|---|
//! | `{model.gap_radius_mm}` | the value at that path (a result, else an input) | the term's display symbol, in the term's colour |
//! | `{model.gap_radius_mm\|m}` | the value converted to the unit after `\|` (the registry resolves the factor; unknown pairs are refused) | the same symbol; the term list shows the value in that unit |
//! | `a * b` | a × b | a thin space (juxtaposition); `×` between two numbers |
//! | `a · b` | a × b | a centred dot |
//! | `a / b` | a ÷ b | inline `a/b` (never a factor: see below) |
//! | `frac(a, b)` | a ÷ b | a stacked fraction |
//! | `a ^ b` | a to the power b | b as a superscript (outer parentheses of b dropped) |
//! | `sqrt(a)` | √a | a radical |
//! | `exp(a)` | e to the a | `e` with a as a superscript |
//! | `abs(a)`, `ceil(a)`, `floor(a)` | \|a\|, ⌈a⌉, ⌊a⌋ | those brackets |
//! | `ceilto(a, s)`, `floorto(a, s)` | a rounded up (down) to a multiple of s, as Excel's CEILING (FLOOR) with the engine's 1e-12 guard (`compat::ceiling`, `compat::floor_`) | ⌈a⌉ with s beneath (⌊a⌋ likewise) |
//! | `fmt(a, d)` | the text of a with d decimals (`compat::fmt_fixed`, Python's `f"{a:.{d}f}"`); d a whole number 0 to 12 | a, with d decimals |
//! | `fmtnum(a)` | the text of a as the workbook prints a number (`compat::fmt_num`: a whole number without ".0") | a |
//! | `concat(t, ...)` | the texts joined (every argument a text) | the texts side by side |
//! | `inf` | +∞ (an onset the knee is never reached at) | ∞ |
//! | `nan` | not a number: a quantity the engine leaves undefined (a torque at a limit that does not exist) | "undefined" |
//! | `sin(a)` .. `ln(a)` | the function | upright name, argument in parentheses |
//! | `min(a, b, ..)`, `max(..)` | Python's `min`/`max`, folded left to right (`compat::py_min`) | upright name |
//! | `(a)` | a | parentheses (always shown) |
//! | `sum(n in H: e)` | Σ over n in the harmonic set H (1, 3, ... up to `coupling.max_harmonic`), a left fold from 0 as Python's `sum()`; NaN for an invalid set | Σ with `n ∈ H` beneath |
//! | `cases(c1 => e1; ...; else => e)` | the first arm whose condition holds, else `e` (a NaN comparison is false, as in the engine) | a left brace, one row per arm: `e1   if c1` |
//! | `a < b` ... `a != b`, `and`, `or` | comparisons (numbers, or exact text; `none` matches an unset optional input) | `<`, `≤`, `>`, `≥`, `=`, `≠`, `and`, `or` |
//! | `"text"` | that text (a verdict) | the text, quoted |
//! | `n` | the index: the family member's index, or the Σ variable | italic n (a family member shows its digit) |
//! | `peak(n in H: A)` | the electrical angle φ in [0, π/2] at which Σ_{n∈H} A_n sin(nφ) is largest, A_n the expression: π/2 unless another maximum beats it by more than 1e-12 relative, exactly as E7 decides (`model::peak_off_half_pitch`); π/2 for an invalid set | `arg max` over φ of Σ_{n∈H} A sin(nφ) |
//! | `table("magnets", k, "br_T")` | a field of a static engine table, by key; `none` when the key is not in it ([`super::tables`] lists the tables and fields) | the field's symbol with the key: `B_{r,lib}(part_i)` |
//! | `... where [S_n] = e` | `[S_n]` stands for e (evaluated when first used) | the formula, then one line per binding: `S_n = e` |
//!
//! The typesetter draws `a / b` inline and only the parentheses the markup writes, so the
//! registry refuses markup that would read two ways: an inline `a / b` as an operand of `*`,
//! `·` or `/` (`a/b c` reads as a/(b c): write `frac(a, b)` or `(a / b)`); a `sum` or `peak`
//! as an operand of `*`, `·`, `/` or `^`, or left of `+` or `-` (write `(sum(...))`); and a
//! power of a term, local or `exp` whose symbol already has a superscript (write
//! `({calibration.gap_radius_mm})^2`, never `R_g^{cal}^2`). It also refuses a formula that
//! reads one path in two units: the term list shows each term once, in the unit it is read in.
//!
//! # Symbol markup
//!
//! A display symbol (a record's, a leaf term's, a local's) is `base [_ script] [^ script]`,
//! where `base` is one or more characters other than `_ ^ { }` and space, and `script` is
//! one character or `{...}`: `τ_p`, `B_{i,3}`, `S_{3}^{iron}`, `T_{pull,20}`. A `#` in a
//! family's symbol is its index (`B_{i,#}` is `B_{i,3}` for harmonic 3 and `B_{i,n}` under a Σ).
//! [`Symbol::parse`] splits it for the typesetter.

use std::fmt;

/// A parse error, at a character offset of the markup.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub at: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at {}: {}", self.at, self.message)
    }
}

/// A reference to an input or result path.
#[derive(Clone, Debug, PartialEq)]
pub struct TermRef {
    /// The path; inside a Σ a family path, with `#` for the index.
    pub path: String,
    /// The unit the formula wants the value in, when not the term's own.
    pub unit: Option<String>,
    /// The factor from the term's unit to `unit`, set by the registry (1 without `unit`).
    pub scale: f64,
}

/// Binary operators. `Mul` and `Dot` both multiply; they differ only in how they are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    /// `*`: drawn as juxtaposition.
    Mul,
    /// `·`: drawn as a centred dot.
    Dot,
    /// `/`: drawn inline.
    Div,
    Pow,
}

/// Functions of the markup (`frac` has its own node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    Sqrt,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Abs,
    Min,
    Max,
    Ceil,
    Floor,
    CeilTo,
    FloorTo,
    Fmt,
    FmtNum,
    Concat,
}

impl Func {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sqrt" => Func::Sqrt,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "tanh" => Func::Tanh,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "abs" => Func::Abs,
            "min" => Func::Min,
            "max" => Func::Max,
            "ceil" => Func::Ceil,
            "floor" => Func::Floor,
            "ceilto" => Func::CeilTo,
            "floorto" => Func::FloorTo,
            "fmt" => Func::Fmt,
            "fmtnum" => Func::FmtNum,
            "concat" => Func::Concat,
            _ => return None,
        })
    }

    /// The markup name.
    pub const fn name(self) -> &'static str {
        match self {
            Func::Sqrt => "sqrt",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Sinh => "sinh",
            Func::Cosh => "cosh",
            Func::Tanh => "tanh",
            Func::Exp => "exp",
            Func::Ln => "ln",
            Func::Abs => "abs",
            Func::Min => "min",
            Func::Max => "max",
            Func::Ceil => "ceil",
            Func::Floor => "floor",
            Func::CeilTo => "ceilto",
            Func::FloorTo => "floorto",
            Func::Fmt => "fmt",
            Func::FmtNum => "fmtnum",
            Func::Concat => "concat",
        }
    }

    /// How many arguments the function takes: `(least, most)`, `None` for no upper bound.
    const fn arity(self) -> (usize, Option<usize>) {
        match self {
            Func::Min | Func::Max | Func::Concat => (2, None),
            Func::CeilTo | Func::FloorTo | Func::Fmt => (2, Some(2)),
            _ => (1, Some(1)),
        }
    }
}

/// The index sets a Σ can run over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexSet {
    /// The harmonics summed: 1, 3, ... up to `coupling.max_harmonic` (Addendum A3).
    Harmonics,
}

impl IndexSet {
    /// The input that chooses the set: an implicit term of every Σ over it.
    pub const fn selector(self) -> &'static str {
        match self {
            IndexSet::Harmonics => "coupling.max_harmonic",
        }
    }
}

/// Comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A condition of a `cases` arm.
#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    Rel(RelOp, Expr, Expr),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

/// A formula expression: the typesetter's input and the evaluator's.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A number; `text` is how the markup wrote it (the typesetter shows that).
    Num {
        value: f64,
        text: String,
    },
    /// A text literal (a verdict).
    Text(String),
    /// `none`: an unset optional input.
    NoneLit,
    /// π.
    Pi,
    /// An input or result path.
    Term(TermRef),
    /// A family term inside a Σ: `path` holds `#` for the index.
    FamilyTerm(TermRef),
    /// The Σ index `n`.
    Index,
    /// A `where` binding, by position.
    Local(usize),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Frac(Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
    /// Parentheses the markup wrote (drawn; transparent to evaluation).
    Paren(Box<Expr>),
    Sum(IndexSet, Box<Expr>),
    /// `id` numbers the `cases` of a formula in parse order (the branch-coverage key).
    Cases {
        id: usize,
        arms: Vec<(Cond, Expr)>,
        otherwise: Box<Expr>,
    },
    /// The E7 pull-out angle of the amplitudes A_n (the expression, per index).
    Peak(IndexSet, Box<Expr>),
    /// A field of a static engine table ([`super::tables`]), by key.
    Table {
        table: String,
        key: Box<Expr>,
        field: String,
    },
}

/// A named sub-expression (`where [S_n] = ...`).
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// Its display name, in symbol markup, the index substituted.
    pub symbol: String,
    pub expr: Expr,
}

/// A parsed formula.
#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    pub body: Expr,
    pub bindings: Vec<Binding>,
    /// How many `cases` the formula holds (their ids are 0 .. this).
    pub cases_count: usize,
}

impl Formula {
    /// Calls `f` on every expression node: the body, then each binding, depth first.
    pub fn visit(&self, f: &mut dyn FnMut(&Expr)) {
        visit_expr(&self.body, f);
        for b in &self.bindings {
            visit_expr(&b.expr, f);
        }
    }

    /// Mutable [`Formula::visit`].
    pub fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        visit_expr_mut(&mut self.body, f);
        for b in &mut self.bindings {
            visit_expr_mut(&mut b.expr, f);
        }
    }
}

fn visit_cond(c: &Cond, f: &mut dyn FnMut(&Expr)) {
    match c {
        Cond::Rel(_, a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        Cond::And(a, b) | Cond::Or(a, b) => {
            visit_cond(a, f);
            visit_cond(b, f);
        }
    }
}

fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Neg(a) | Expr::Paren(a) | Expr::Sum(_, a) | Expr::Peak(_, a) => visit_expr(a, f),
        Expr::Table { key, .. } => visit_expr(key, f),
        Expr::Bin(_, a, b) | Expr::Frac(a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| visit_expr(a, f)),
        Expr::Cases {
            arms, otherwise, ..
        } => {
            for (c, a) in arms {
                visit_cond(c, f);
                visit_expr(a, f);
            }
            visit_expr(otherwise, f);
        }
        Expr::Num { .. }
        | Expr::Text(_)
        | Expr::NoneLit
        | Expr::Pi
        | Expr::Term(_)
        | Expr::FamilyTerm(_)
        | Expr::Index
        | Expr::Local(_) => {}
    }
}

fn visit_cond_mut(c: &mut Cond, f: &mut dyn FnMut(&mut Expr)) {
    match c {
        Cond::Rel(_, a, b) => {
            visit_expr_mut(a, f);
            visit_expr_mut(b, f);
        }
        Cond::And(a, b) | Cond::Or(a, b) => {
            visit_cond_mut(a, f);
            visit_cond_mut(b, f);
        }
    }
}

fn visit_expr_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    f(e);
    match e {
        Expr::Neg(a) | Expr::Paren(a) | Expr::Sum(_, a) | Expr::Peak(_, a) => visit_expr_mut(a, f),
        Expr::Table { key, .. } => visit_expr_mut(key, f),
        Expr::Bin(_, a, b) | Expr::Frac(a, b) => {
            visit_expr_mut(a, f);
            visit_expr_mut(b, f);
        }
        Expr::Call(_, args) => args.iter_mut().for_each(|a| visit_expr_mut(a, f)),
        Expr::Cases {
            arms, otherwise, ..
        } => {
            for (c, a) in arms {
                visit_cond_mut(c, f);
                visit_expr_mut(a, f);
            }
            visit_expr_mut(otherwise, f);
        }
        Expr::Num { .. }
        | Expr::Text(_)
        | Expr::NoneLit
        | Expr::Pi
        | Expr::Term(_)
        | Expr::FamilyTerm(_)
        | Expr::Index
        | Expr::Local(_) => {}
    }
}

/// Parses a formula. `index` is the family member's index (`Some(3)` for harmonic 3): each
/// `#` in a term path or a local's symbol becomes its digits and `n` becomes that number;
/// `None` for a plain record, where `n` and `#` are allowed only inside a Σ.
pub fn parse(src: &str, index: Option<u32>) -> Result<Formula, ParseError> {
    let mut p = Parser {
        chars: src.chars().collect(),
        pos: 0,
        index,
        in_sum: false,
        cases_count: 0,
        local_names: Vec::new(),
    };
    let body = p.expr()?;
    let mut bindings = Vec::new();
    if p.eat_word("where") {
        loop {
            p.skip_ws();
            let at = p.pos;
            let symbol = p.local_symbol()?;
            if bindings.iter().any(|b: &Binding| b.symbol == symbol) {
                return Err(p.error_at(at, format!("[{symbol}] is bound twice")));
            }
            p.expect('=')?;
            let expr = p.expr()?;
            bindings.push(Binding { symbol, expr });
            if !p.eat(',') {
                break;
            }
        }
    }
    p.skip_ws();
    if p.pos < p.chars.len() {
        return Err(p.error(format!("unexpected '{}'", p.chars[p.pos])));
    }
    // Resolve every local reference to its binding; a binding may use only earlier ones.
    let names: Vec<String> = bindings.iter().map(|b| b.symbol.clone()).collect();
    let mut formula = Formula {
        body,
        bindings,
        cases_count: p.cases_count,
    };
    let mut used = vec![false; names.len()];
    let resolve = |e: &mut Expr, limit: usize, used: &mut Vec<bool>| -> Result<(), String> {
        let mut result = Ok(());
        visit_expr_mut(e, &mut |node| {
            if let Expr::Local(i) = node {
                let name = &p.local_names[*i];
                match names.iter().position(|n| n == name) {
                    Some(k) if k < limit => {
                        used[k] = true;
                        *node = Expr::Local(k);
                    }
                    Some(_) => result = Err(format!("[{name}] is used before it is bound")),
                    None => result = Err(format!("[{name}] is not bound")),
                }
            }
        });
        result
    };
    resolve(&mut formula.body, names.len(), &mut used).map_err(|m| p.error_at(0, m))?;
    for k in 0..formula.bindings.len() {
        let mut expr = std::mem::replace(&mut formula.bindings[k].expr, Expr::NoneLit);
        resolve(&mut expr, k, &mut used).map_err(|m| p.error_at(0, m))?;
        formula.bindings[k].expr = expr;
    }
    if let Some(k) = used.iter().position(|u| !u) {
        return Err(p.error_at(0, format!("[{}] is bound but never used", names[k])));
    }
    Ok(formula)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    index: Option<u32>,
    in_sum: bool,
    cases_count: usize,
    /// Local names in order of first reference (resolved to bindings after the parse).
    local_names: Vec<String>,
}

const KEYWORDS: &[&str] = &[
    "cases", "else", "and", "or", "where", "sum", "peak", "table", "in", "n", "pi", "none", "H",
    "frac", "inf", "nan",
];

impl Parser {
    fn error(&self, message: String) -> ParseError {
        self.error_at(self.pos, message)
    }

    fn error_at(&self, at: usize, message: String) -> ParseError {
        ParseError { at, message }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn peek2(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos + 1).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, s: &str) -> bool {
        self.skip_ws();
        let n = s.chars().count();
        if self.chars.len() >= self.pos + n
            && self.chars[self.pos..self.pos + n]
                .iter()
                .copied()
                .eq(s.chars())
        {
            self.pos += n;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), ParseError> {
        if self.eat(c) {
            Ok(())
        } else {
            let got = self
                .peek()
                .map_or("the end".to_owned(), |g| format!("'{g}'"));
            Err(self.error(format!("expected '{c}', got {got}")))
        }
    }

    fn is_word_char(c: char) -> bool {
        c.is_alphabetic() || c == '_'
    }

    /// The identifier at the cursor, without consuming it.
    fn peek_word(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        let mut end = start;
        while end < self.chars.len() && Self::is_word_char(self.chars[end]) {
            end += 1;
        }
        (end > start).then(|| self.chars[start..end].iter().collect())
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if self.peek_word().as_deref() == Some(word) {
            self.pos += word.chars().count();
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, word: &str) -> Result<(), ParseError> {
        if self.eat_word(word) {
            Ok(())
        } else {
            Err(self.error(format!("expected '{word}'")))
        }
    }

    fn substitute(&self, text: &str) -> Result<String, String> {
        match self.index {
            Some(n) => Ok(text.replace('#', &n.to_string())),
            None if text.contains('#') && !self.in_sum => {
                Err(format!("'#' outside a Σ in a plain record: {text}"))
            }
            None => Ok(text.to_owned()),
        }
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.additive()
    }

    /// `cases(...)`, after the keyword.
    fn cases(&mut self) -> Result<Expr, ParseError> {
        self.expect('(')?;
        let id = self.cases_count;
        self.cases_count += 1;
        let mut arms = Vec::new();
        loop {
            if self.eat_word("else") {
                if arms.is_empty() {
                    return Err(self.error("cases needs at least one arm before else".into()));
                }
                if !self.eat_str("=>") {
                    return Err(self.error("expected '=>' after else".into()));
                }
                let otherwise = self.expr()?;
                self.expect(')')?;
                return Ok(Expr::Cases {
                    id,
                    arms,
                    otherwise: Box::new(otherwise),
                });
            }
            let cond = self.cond()?;
            if !self.eat_str("=>") {
                return Err(self.error("expected '=>' after a condition".into()));
            }
            let value = self.expr()?;
            arms.push((cond, value));
            self.expect(';')?;
        }
    }

    fn cond(&mut self) -> Result<Cond, ParseError> {
        let mut left = self.conj()?;
        while self.eat_word("or") {
            let right = self.conj()?;
            left = Cond::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn conj(&mut self) -> Result<Cond, ParseError> {
        let mut left = self.rel()?;
        while self.eat_word("and") {
            let right = self.rel()?;
            left = Cond::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn rel(&mut self) -> Result<Cond, ParseError> {
        let a = self.additive()?;
        let op = if self.eat_str("<=") {
            RelOp::Le
        } else if self.eat_str(">=") {
            RelOp::Ge
        } else if self.eat_str("!=") {
            RelOp::Ne
        } else if self.peek() == Some('=') && self.peek2() != Some('>') {
            self.pos += 1;
            RelOp::Eq
        } else if self.eat('<') {
            RelOp::Lt
        } else if self.eat('>') {
            RelOp::Gt
        } else {
            return Err(self.error("expected a comparison".into()));
        };
        let b = self.additive()?;
        Ok(Cond::Rel(op, a, b))
    }

    fn additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.product()?;
        loop {
            let op = match self.peek() {
                Some('+') => BinOp::Add,
                Some('-') => BinOp::Sub,
                _ => return Ok(left),
            };
            self.pos += 1;
            let right = self.product()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    fn product(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some('*') => BinOp::Mul,
                Some('·') => BinOp::Dot,
                Some('/') => BinOp::Div,
                _ => return Ok(left),
            };
            self.pos += 1;
            let right = self.unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.eat('-') {
            Ok(Expr::Neg(Box::new(self.unary()?)))
        } else {
            self.power()
        }
    }

    fn power(&mut self) -> Result<Expr, ParseError> {
        let base = self.atom()?;
        if self.eat('^') {
            let exponent = self.unary()?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exponent)))
        } else {
            Ok(base)
        }
    }

    fn number(&mut self) -> Result<Expr, ParseError> {
        let start = self.pos;
        let digits = |p: &mut Parser| {
            let s = p.pos;
            while p.pos < p.chars.len() && p.chars[p.pos].is_ascii_digit() {
                p.pos += 1;
            }
            p.pos > s
        };
        digits(self);
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            self.pos += 1;
            if !digits(self) {
                return Err(self.error("expected digits after '.'".into()));
            }
        }
        if self.pos < self.chars.len() && matches!(self.chars[self.pos], 'e' | 'E') {
            let save = self.pos;
            self.pos += 1;
            if self.pos < self.chars.len() && matches!(self.chars[self.pos], '+' | '-') {
                self.pos += 1;
            }
            if !digits(self) {
                self.pos = save; // not an exponent
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let value = text
            .parse::<f64>()
            .map_err(|e| self.error_at(start, format!("bad number {text}: {e}")))?;
        Ok(Expr::Num { value, text })
    }

    fn delimited(&mut self, close: char, what: &str) -> Result<String, ParseError> {
        let start = self.pos;
        let mut text = String::new();
        loop {
            match self.chars.get(self.pos) {
                None => return Err(self.error_at(start, format!("unterminated {what}"))),
                Some(&c) if c == close => {
                    self.pos += 1;
                    return Ok(text);
                }
                Some(&c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
    }

    fn quoted(&mut self, what: &str) -> Result<String, ParseError> {
        if self.eat('"') {
            self.delimited('"', what)
        } else {
            Err(self.error(format!("expected the {what} as a quoted text")))
        }
    }

    fn local_symbol(&mut self) -> Result<String, ParseError> {
        self.expect('[')?;
        let at = self.pos;
        let raw = self.delimited(']', "local name")?;
        let symbol = self
            .substitute(raw.trim())
            .map_err(|m| self.error_at(at, m))?;
        Symbol::parse(&symbol).map_err(|m| self.error_at(at, m))?;
        Ok(symbol)
    }

    fn atom(&mut self) -> Result<Expr, ParseError> {
        let at = self.pos;
        match self.peek() {
            None => Err(self.error("unexpected end of formula".into())),
            Some(c) if c.is_ascii_digit() => self.number(),
            Some('"') => {
                self.pos += 1;
                Ok(Expr::Text(self.delimited('"', "text")?))
            }
            Some('{') => {
                self.pos += 1;
                let raw = self.delimited('}', "term")?;
                let (path, unit) = match raw.split_once('|') {
                    Some((p, u)) => (p.trim().to_owned(), Some(u.trim().to_owned())),
                    None => (raw.trim().to_owned(), None),
                };
                if path.is_empty() || path.contains(char::is_whitespace) {
                    return Err(self.error_at(at, format!("bad term path '{path}'")));
                }
                let family = path.contains('#') && self.index.is_none();
                let path = self.substitute(&path).map_err(|m| self.error_at(at, m))?;
                let term = TermRef {
                    path,
                    unit,
                    scale: 1.0,
                };
                Ok(if family {
                    Expr::FamilyTerm(term)
                } else {
                    Expr::Term(term)
                })
            }
            Some('[') => {
                let symbol = self.local_symbol()?;
                let i = match self.local_names.iter().position(|n| *n == symbol) {
                    Some(i) => i,
                    None => {
                        self.local_names.push(symbol);
                        self.local_names.len() - 1
                    }
                };
                Ok(Expr::Local(i))
            }
            Some('(') => {
                self.pos += 1;
                let inner = self.expr()?;
                self.expect(')')?;
                Ok(Expr::Paren(Box::new(inner)))
            }
            Some('π') => {
                self.pos += 1;
                Ok(Expr::Pi)
            }
            Some(_) => {
                let word = self
                    .peek_word()
                    .ok_or_else(|| self.error(format!("unexpected '{}'", self.chars[self.pos])))?;
                self.pos += word.chars().count();
                match word.as_str() {
                    "pi" => Ok(Expr::Pi),
                    "none" => Ok(Expr::NoneLit),
                    "inf" => Ok(Expr::Num {
                        value: f64::INFINITY,
                        text: "∞".into(),
                    }),
                    "nan" => Ok(Expr::Num {
                        value: f64::NAN,
                        text: "undefined".into(),
                    }),
                    "cases" => self.cases(),
                    "n" => match self.index {
                        Some(n) => Ok(Expr::Num {
                            value: f64::from(n),
                            text: n.to_string(),
                        }),
                        None if self.in_sum => Ok(Expr::Index),
                        None => Err(self.error_at(at, "'n' outside a Σ in a plain record".into())),
                    },
                    "frac" => {
                        self.expect('(')?;
                        let num = self.expr()?;
                        self.expect(',')?;
                        let den = self.expr()?;
                        self.expect(')')?;
                        Ok(Expr::Frac(Box::new(num), Box::new(den)))
                    }
                    "sum" | "peak" => {
                        if self.index.is_some() || self.in_sum {
                            return Err(self
                                .error_at(at, format!("'{word}' inside a family record or a Σ")));
                        }
                        self.expect('(')?;
                        self.expect_word("n")?;
                        self.expect_word("in")?;
                        self.expect_word("H")?;
                        self.expect(':')?;
                        self.in_sum = true;
                        let body = self.expr();
                        self.in_sum = false;
                        let body = Box::new(body?);
                        self.expect(')')?;
                        Ok(if word == "sum" {
                            Expr::Sum(IndexSet::Harmonics, body)
                        } else {
                            Expr::Peak(IndexSet::Harmonics, body)
                        })
                    }
                    "table" => {
                        self.expect('(')?;
                        let table = self.quoted("table name")?;
                        self.expect(',')?;
                        let key = self.expr()?;
                        self.expect(',')?;
                        let field = self.quoted("field name")?;
                        self.expect(')')?;
                        Ok(Expr::Table {
                            table,
                            key: Box::new(key),
                            field,
                        })
                    }
                    name => {
                        let func = Func::from_name(name).ok_or_else(|| {
                            if KEYWORDS.contains(&name) {
                                self.error_at(at, format!("'{name}' is not allowed here"))
                            } else {
                                self.error_at(at, format!("unknown name '{name}'"))
                            }
                        })?;
                        self.expect('(')?;
                        let mut args = vec![self.expr()?];
                        while self.eat(',') {
                            args.push(self.expr()?);
                        }
                        self.expect(')')?;
                        let (least, most) = func.arity();
                        if args.len() < least || most.is_some_and(|m| args.len() > m) {
                            let wanted = match (least, most) {
                                (1, Some(1)) => "one".to_owned(),
                                (l, Some(m)) if l == m => format!("{l}"),
                                (l, _) => format!("{l} or more"),
                            };
                            return Err(self.error_at(
                                at,
                                format!("{name} takes {wanted} argument(s), got {}", args.len()),
                            ));
                        }
                        Ok(Expr::Call(func, args))
                    }
                }
            }
        }
    }
}

/// A display symbol split for the typesetter: `B_{i,3}` is base `B`, subscript `i,3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub base: String,
    pub sub: Option<String>,
    pub sup: Option<String>,
}

impl Symbol {
    /// Parses symbol markup (module docs); `#` must already be substituted by the caller
    /// when it stands for an index.
    pub fn parse(text: &str) -> Result<Symbol, String> {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        let mut base = String::new();
        while i < chars.len() && !matches!(chars[i], '_' | '^' | '{' | '}' | ' ') {
            base.push(chars[i]);
            i += 1;
        }
        if base.is_empty() {
            return Err(format!("symbol '{text}' has no base"));
        }
        let script = |i: &mut usize| -> Result<String, String> {
            match chars.get(*i) {
                Some('{') => {
                    let start = *i + 1;
                    let end = chars[start..]
                        .iter()
                        .position(|&c| c == '}')
                        .map(|k| start + k)
                        .ok_or_else(|| format!("symbol '{text}': unclosed '{{'"))?;
                    *i = end + 1;
                    let s: String = chars[start..end].iter().collect();
                    if s.is_empty() {
                        return Err(format!("symbol '{text}': empty script"));
                    }
                    Ok(s)
                }
                Some(&c) if !matches!(c, '_' | '^' | '}' | ' ') => {
                    *i += 1;
                    Ok(c.to_string())
                }
                _ => Err(format!("symbol '{text}': missing script")),
            }
        };
        let mut sub = None;
        let mut sup = None;
        if chars.get(i) == Some(&'_') {
            i += 1;
            sub = Some(script(&mut i)?);
        }
        if chars.get(i) == Some(&'^') {
            i += 1;
            sup = Some(script(&mut i)?);
        }
        if i != chars.len() {
            return Err(format!(
                "symbol '{text}': unexpected text after the scripts"
            ));
        }
        Ok(Symbol { base, sub, sup })
    }

    /// Plain-text form: `B_{i,3}` stays `B_i,3`-free of braces for one-character scripts.
    pub fn plain(&self) -> String {
        let script = |s: &str| {
            if s.chars().count() == 1 {
                s.to_owned()
            } else {
                format!("{{{s}}}")
            }
        };
        let mut out = self.base.clone();
        if let Some(s) = &self.sub {
            out.push('_');
            out.push_str(&script(s));
        }
        if let Some(s) = &self.sup {
            out.push('^');
            out.push_str(&script(s));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(v: f64) -> Expr {
        Expr::Num {
            value: v,
            text: v.to_string(),
        }
    }

    fn term(p: &str) -> Expr {
        Expr::Term(TermRef {
            path: p.into(),
            unit: None,
            scale: 1.0,
        })
    }

    #[test]
    fn precedence_and_associativity() {
        // a + b * c ^ d ^ e: power right-assoc and tighter than product, product tighter than sum.
        let f = parse("{a.x} + {a.y} * {a.z} ^ 2 ^ 3", None).unwrap();
        let expected = Expr::Bin(
            BinOp::Add,
            Box::new(term("a.x")),
            Box::new(Expr::Bin(
                BinOp::Mul,
                Box::new(term("a.y")),
                Box::new(Expr::Bin(
                    BinOp::Pow,
                    Box::new(term("a.z")),
                    Box::new(Expr::Bin(
                        BinOp::Pow,
                        Box::new(num(2.0)),
                        Box::new(num(3.0)),
                    )),
                )),
            )),
        );
        assert_eq!(f.body, expected);
        // -x^2 is -(x^2); a - b - c is (a - b) - c; a / b · c is (a / b) · c.
        let f = parse("-{a.x}^2", None).unwrap();
        assert!(
            matches!(f.body, Expr::Neg(ref inner) if matches!(**inner, Expr::Bin(BinOp::Pow, ..)))
        );
        let f = parse("{a.x} - {a.y} - {a.z}", None).unwrap();
        assert!(
            matches!(f.body, Expr::Bin(BinOp::Sub, ref l, _) if matches!(**l, Expr::Bin(BinOp::Sub, ..)))
        );
        let f = parse("{a.x} / {a.y} · {a.z}", None).unwrap();
        assert!(
            matches!(f.body, Expr::Bin(BinOp::Dot, ref l, _) if matches!(**l, Expr::Bin(BinOp::Div, ..)))
        );
    }

    #[test]
    fn numbers_keep_their_text() {
        let f = parse("1e-3 + 0.5 + 20 + 2.5E+2", None).unwrap();
        let mut texts = Vec::new();
        f.visit(&mut |e| {
            if let Expr::Num { text, value } = e {
                texts.push((text.clone(), *value));
            }
        });
        assert_eq!(
            texts,
            vec![
                ("1e-3".into(), 1e-3),
                ("0.5".into(), 0.5),
                ("20".into(), 20.0),
                ("2.5E+2".into(), 250.0)
            ]
        );
    }

    #[test]
    fn family_members_substitute_the_index() {
        let f = parse("frac(n * {m.b_i#|m}, 2) where [S_#] = 1", Some(3));
        // An unused binding is refused.
        assert!(f.is_err());
        let f = parse(
            "frac(n * {m.b_i#|m}, 2) · [S_#] where [S_#] = {m.s#}",
            Some(3),
        )
        .unwrap();
        assert_eq!(f.bindings[0].symbol, "S_3");
        let mut paths = Vec::new();
        f.visit(&mut |e| {
            if let Expr::Term(t) = e {
                paths.push((t.path.clone(), t.unit.clone()));
            }
        });
        assert_eq!(
            paths,
            vec![("m.b_i3".into(), Some("m".into())), ("m.s3".into(), None)]
        );
        assert!(matches!(f.body, Expr::Bin(BinOp::Dot, ref l, _)
            if matches!(**l, Expr::Frac(ref a, _) if matches!(**a, Expr::Bin(BinOp::Mul, ref n, _) if **n == Expr::Num { value: 3.0, text: "3".into() }))));
    }

    #[test]
    fn a_sum_binds_n_and_family_terms() {
        let f = parse("sum(n in H: {m.tau#} * n)", None).unwrap();
        match &f.body {
            Expr::Sum(IndexSet::Harmonics, body) => {
                assert!(matches!(**body, Expr::Bin(BinOp::Mul, ref a, ref b)
                    if matches!(**a, Expr::FamilyTerm(ref t) if t.path == "m.tau#") && **b == Expr::Index));
            }
            other => panic!("{other:?}"),
        }
        assert!(parse("n + 1", None).is_err(), "n outside a sum");
        assert!(parse("{m.x#}", None).is_err(), "# outside a sum");
        assert!(parse("sum(n in H: sum(n in H: n))", None).is_err());
        assert!(
            parse("sum(n in H: n)", Some(1)).is_err(),
            "a sum inside a family"
        );
    }

    #[test]
    fn cases_and_conditions() {
        let f = parse(
            r#"cases({a.x} < {a.y} and {a.p} = "B842SH" => "Below"; {a.z} != none or 1 >= 2 => 1; else => cases(1 <= 2 => 0; else => 1))"#,
            None,
        )
        .unwrap();
        assert_eq!(f.cases_count, 2);
        match &f.body {
            Expr::Cases {
                id,
                arms,
                otherwise,
            } => {
                assert_eq!(*id, 0);
                assert_eq!(arms.len(), 2);
                assert!(matches!(arms[0].0, Cond::And(..)));
                assert!(matches!(arms[1].0, Cond::Or(..)));
                assert!(matches!(**otherwise, Expr::Cases { id: 1, .. }));
            }
            other => panic!("{other:?}"),
        }
        assert!(parse("cases(else => 1)", None).is_err(), "needs an arm");
        assert!(parse("cases(1 < 2 => 1)", None).is_err(), "needs else");
    }

    #[test]
    fn where_bindings_resolve_in_order() {
        let f = parse("[A] + [B] where [A] = 1, [B] = [A] * 2", None).unwrap();
        assert_eq!(f.bindings.len(), 2);
        assert!(
            matches!(f.bindings[1].expr, Expr::Bin(BinOp::Mul, ref a, _) if **a == Expr::Local(0))
        );
        assert!(
            parse("[A] where [A] = [B], [B] = 1", None).is_err(),
            "used before bound"
        );
        assert!(parse("[A]", None).is_err(), "not bound");
        assert!(
            parse("[A] where [A] = 1, [A] = 2", None).is_err(),
            "bound twice"
        );
    }

    #[test]
    fn calls_check_their_arity_and_names() {
        assert!(parse("min(1)", None).is_err());
        assert!(parse("sin(1, 2)", None).is_err());
        assert!(parse("foo(1)", None).is_err());
        assert!(parse("max(1, 2, 3)", None).is_ok());
        assert!(parse("2{a.x}", None).is_err(), "no implicit multiplication");
        assert!(matches!(
            parse("peak(n in H: {m.a#})", None).unwrap().body,
            Expr::Peak(IndexSet::Harmonics, _)
        ));
        assert!(
            parse("peak(n in H: n)", Some(3)).is_err(),
            "a peak inside a family"
        );
        let t = parse(r#"table("magnets", {c.part}, "br_T")"#, None).unwrap();
        assert!(
            matches!(t.body, Expr::Table { ref table, ref field, .. } if table == "magnets" && field == "br_T")
        );
        assert!(
            parse("table(magnets, {c.part}, br_T)", None).is_err(),
            "names are quoted"
        );
        assert!(parse("{a b}", None).is_err());
        assert!(parse("(1 + 2", None).is_err());
    }

    #[test]
    fn the_text_and_rounding_functions_check_their_arity() {
        assert!(parse("ceilto({a.x}, 0.1) + floorto({a.x}, 1)", None).is_ok());
        assert!(parse("ceilto({a.x})", None).is_err(), "ceilto takes 2");
        assert!(parse("fmt({a.x}, 1, 2)", None).is_err(), "fmt takes 2");
        assert!(
            parse(r#"concat("a")"#, None).is_err(),
            "concat takes 2 or more"
        );
        assert!(parse(r#"concat("a", fmt({a.x}, 1), fmtnum({a.y}), " mm")"#, None).is_ok());
        let f = parse("inf", None).unwrap();
        assert!(
            matches!(f.body, Expr::Num { value, ref text } if value == f64::INFINITY && text == "∞")
        );
        let f = parse("nan", None).unwrap();
        assert!(
            matches!(f.body, Expr::Num { value, ref text } if value.is_nan() && text == "undefined")
        );
        assert!(
            parse("[inf] where [inf] = 1", None).is_ok(),
            "a local's name is a symbol, not a keyword"
        );
    }

    #[test]
    fn symbols_split_into_base_and_scripts() {
        assert_eq!(
            Symbol::parse("S_{3}^{iron}").unwrap(),
            Symbol {
                base: "S".into(),
                sub: Some("3".into()),
                sup: Some("iron".into())
            }
        );
        assert_eq!(Symbol::parse("τ_p").unwrap().plain(), "τ_p");
        assert_eq!(Symbol::parse("T_{pull,20}").unwrap().plain(), "T_{pull,20}");
        assert_eq!(
            Symbol::parse("N").unwrap(),
            Symbol {
                base: "N".into(),
                sub: None,
                sup: None
            }
        );
        for bad in ["", "_i", "B_", "B_{}", "B_{i", "B_i x", "B^2_i"] {
            assert!(Symbol::parse(bad).is_err(), "{bad}");
        }
    }
}
