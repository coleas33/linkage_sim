//! The Equation panel (spec Addendum A2 "Equation panel (docked, toggleable). It shows the
//! open equation large, its terms with values and units, a breadcrumb trail, and a 'used by'
//! list. Clicking a term drills into that term's own equation. Leaf terms (inputs) highlight
//! their slider. One equation at a time: never a page of every formula").
//!
//! [`Explorer`] is the panel's state: whether it is shown, the breadcrumb trail of the paths
//! opened (the last one shown), the readout hovered in the last frame and the input row to
//! highlight. A click on a readout opens its value here (a new trail); a click on a term of the
//! open equation, in the equation or in the term list, drills into it (a step on the trail) or,
//! for an input, highlights its row on the inputs side, opening its group and scrolling to it
//! (decision M43-12); a crumb goes back to its step; "used by" goes up the chain. A result
//! without an equation record shows its label, value and workbook cell. The panel docks at the
//! bottom of the centre region, closed until a value is clicked or its header button is
//! pressed (decisions M43-1, M43-2).
//!
//! The terms the panel or the hovered readout shows are marked on screen in their colours
//! ([`Explorer::marks`]): the hovered readout's equation wins, else the open one's. While the
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].
//!
//! Teaching notes (spec Addendum A4): the Explain toggle, off by default, shows under the open
//! equation the reviewed note that explains it ([`notes::note_for`]: only notes a physics
//! reviewer has checked), with its watch-out line and its diagram ([`crate::gui::diagrams`]),
//! so it follows what the user investigates. "Start here" walks the suggested order
//! ([`notes::START_HERE`]): each step opens its equation with Explain on. A warning's link
//! opens its note on its own ([`Explorer::open_note`]; five of the six warning notes explain
//! no single equation), with links to the equations it does explain. Every note the panel
//! names or draws by id goes through [`reviewed_note`], so a note sent back to Draft never
//! shows.
//!
//! Assumptions (spec Addendum A3 "Traceability"): an assumption term is tagged as one in the
//! term list, with the changed-from-default dot when it differs from its workbook default; a
//! result some modified assumption flows into is tagged too, and the open equation says which
//! modified assumptions it depends on ([`Registry::term_style`],
//! [`Registry::modified_assumptions_upstream`]).
//!
//! [`Registry::term_style`]: crate::engine::explain::Registry::term_style
//! [`Registry::modified_assumptions_upstream`]: crate::engine::explain::Registry::modified_assumptions_upstream

use egui::{Color32, Sense, Vec2};

use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::{self, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{ResultSet, Value};
use crate::gui::dashboard::result_info;
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::{format_value, with_unit};
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
    TermColors, glyph_safe, laid_ui, layout_equation, layout_symbol, term_at,
};
use crate::{DesignInputs, DesignResults};

/// The header button that shows or hides the panel.
pub const EQUATION_PANEL: &str = "Equation panel";

/// The panel's close button.
pub const CLOSE: &str = "Close";

/// What the panel says before anything is opened.
pub const EMPTY_TEXT: &str =
    "Hover a value to see its equation; click it to open it here, then click a term to follow it.";

/// The heading of the term list.
pub const TERMS: &str = "Terms";

/// The heading of the "used by" list.
pub const USED_BY: &str = "Used by";

/// What a result without an equation record shows under its value.
pub const NO_EQUATION: &str =
    "No equation record: the value, its label and its workbook cell are what the explorer knows.";

/// The start of the line naming the corrections an equation embodies.
pub const CORRECTIONS: &str = "Embodies corrections";

/// The start of the line naming the modified assumptions an equation depends on.
pub const DEPENDS_ON_MODIFIED: &str = "Depends on modified assumptions";

/// The term list's tag of an assumption.
pub const ASSUMPTION_TAG: &str = "assumption: click to find its row";

/// The toggle of the teaching note under the open equation.
pub const EXPLAIN: &str = "Explain";

/// The button that starts the suggested reading order.
pub const START_HERE_BUTTON: &str = "Start here";

/// The start-here steps' buttons.
pub const PREVIOUS: &str = "Previous";
pub const NEXT: &str = "Next";
pub const STOP: &str = "Stop";

/// What Explain shows for an equation without a reviewed note.
pub const NO_NOTE: &str = "No teaching note for this equation.";

/// The start of a note's watch-out line.
pub const WATCH_OUT: &str = "Watch out";

/// The start of a note's sources line.
pub const SOURCES: &str = "Sources";

/// The start of the links under a note opened on its own.
pub const EXPLAINS: &str = "Explains";

/// The note `id` if a physics reviewer has checked it (spec A4 "Accuracy gate"), as
/// [`notes::note_for`] gives an equation's: every note the panel or the dashboard shows by id.
pub fn reviewed_note(id: &str) -> Option<&'static Note> {
    notes::note(id).filter(|n| matches!(n.review, Review::Reviewed { .. }))
}

/// The start of the start-here bar.
pub fn start_here_text(step: usize) -> String {
    let title = reviewed_note(START_HERE[step].0).map_or_else(String::new, |n| glyph_safe(n.title));
    format!(
        "{START_HERE_BUTTON} {} of {}: {title}",
        step + 1,
        START_HERE.len()
    )
}

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;

/// The text size of a term's symbol in the term list [points].
pub const TERM_SIZE: f32 = 15.0;

/// The panel's starting height [points].
pub const PANEL_HEIGHT: f32 = 320.0;

/// The longest breadcrumb trail; a longer walk drops its oldest steps.
pub const MAX_TRAIL: usize = 32;

/// The crumbs the header shows: a longer trail shows its last ones after [`ELIDED`] (32 crumbs
/// would wrap over most of a short panel).
pub const SHOWN_CRUMBS: usize = 8;

/// What stands for the crumbs a long trail does not show (its hover counts them).
pub const ELIDED: &str = "\u{2026}";

/// The width of the frame around the input row a leaf term highlights [points].
pub const FOCUS_WIDTH: f32 = 3.0;

/// How strongly the term list's swatches fade while a value of another equation is hovered:
/// the marks on screen are then that equation's colours, not these (decision M43-3).
pub const SWATCH_DIM: f32 = 0.35;

/// The input row a leaf term highlights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Focus {
    /// The input path.
    pub path: String,
    /// The row still has to be scrolled into view (once, when it is first drawn).
    pub scroll: bool,
}

/// The Equation panel's state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Explorer {
    /// Whether the panel is shown.
    pub open: bool,
    /// The paths opened, oldest first; the last one is shown.
    trail: Vec<String>,
    /// The readout hovered in the last frame.
    hovered: Option<String>,
    /// The input row a leaf term highlights.
    focus: Option<Focus>,
    /// Whether the teaching note of the open equation is shown (off by default: spec A4).
    pub explain: bool,
    /// The start-here step shown, if the user is walking the suggested order.
    start_here: Option<usize>,
    /// A teaching note opened on its own (a warning's link), by id.
    note: Option<&'static str>,
}

impl Explorer {
    /// The path shown, if any.
    pub fn current(&self) -> Option<&str> {
        self.trail.last().map(String::as_str)
    }

    /// The breadcrumb trail, oldest first.
    pub fn trail(&self) -> &[String] {
        &self.trail
    }

    /// The readout hovered in the last frame.
    pub fn hovered(&self) -> Option<&str> {
        self.hovered.as_deref()
    }

    /// Opens `path` (a readout clicked): a new trail, the panel shown.
    pub fn open_path(&mut self, path: &str) {
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
        self.note = None;
        self.start_here = None;
    }

    /// Opens the teaching note `id` on its own (a warning's link).
    pub fn open_note(&mut self, id: &'static str) {
        self.note = Some(id);
        self.trail.clear();
        self.open = true;
        self.focus = None;
        self.start_here = None;
    }

    /// Hides the panel (its Close button, the header's toggle). The input row a leaf term
    /// framed loses its frame (decision M43-12: the frame goes with the equation shown); the
    /// trail stays, so the panel shown again opens where it was.
    pub fn close(&mut self) {
        self.open = false;
        self.focus = None;
    }

    /// The note opened on its own, if any.
    pub fn note(&self) -> Option<&'static str> {
        self.note
    }

    /// Opens start-here step `step` (its equation, Explain on); a step past the last one
    /// changes nothing.
    pub fn start(&mut self, step: usize) {
        if let Some(&(_, path)) = START_HERE.get(step) {
            self.trail = vec![path.to_owned()];
            self.open = true;
            self.focus = None;
            self.note = None;
            self.explain = true;
            self.start_here = Some(step);
        }
    }

    /// The start-here step shown, if any.
    pub fn start_here(&self) -> Option<usize> {
        self.start_here
    }

    /// Leaves the start-here walk (the equation stays open).
    pub fn stop(&mut self) {
        self.start_here = None;
    }

    /// Follows a term or a "used by" link to `path`: one more step on the trail.
    pub fn drill(&mut self, path: &str) {
        if self.current() == Some(path) {
            return;
        }
        self.trail.push(path.to_owned());
        if self.trail.len() > MAX_TRAIL {
            self.trail.remove(0);
        }
        self.focus = None;
    }

    /// Goes back to the crumb at `index` (dropping the steps after it).
    pub fn back_to(&mut self, index: usize) {
        self.trail.truncate(index + 1);
        self.focus = None;
    }

    /// Highlights the row of the input at `path` (a leaf term clicked).
    pub fn focus_input(&mut self, path: &str) {
        self.focus = Some(Focus {
            path: path.to_owned(),
            scroll: true,
        });
    }

    /// The input row highlighted, if any.
    pub fn focus(&self) -> Option<&Focus> {
        self.focus.as_ref()
    }

    /// Whether the highlighted row at `path` has to be scrolled into view now: true once.
    pub fn take_scroll(&mut self, path: &str) -> bool {
        match &mut self.focus {
            Some(focus) if focus.path == path && focus.scroll => {
                focus.scroll = false;
                true
            }
            _ => false,
        }
    }

    /// What a click on the term `path` of the open equation does: an input highlights its row;
    /// a result is drilled into; a family template (a Σ's term) drills into the first harmonic
    /// the design sums.
    pub fn follow(&mut self, path: &str, terms: &Design<'_>) {
        match registry().term_kind(path) {
            Some(TermKind::Input { .. }) => self.focus_input(path),
            Some(TermKind::Explained | TermKind::CellOnly) => self.drill(path),
            None => {
                if let Some(first) = registry().family_members(path, terms).first() {
                    self.drill(first);
                }
            }
        }
    }

    /// The term colours this frame marks: the equation of the readout hovered in the last
    /// frame, else the one open (when the panel is shown), a family's by the harmonics the
    /// design sums; none without either.
    pub fn marks(&self, inputs: &DesignInputs, results: &DesignResults) -> TermColors {
        let shown = self.current().filter(|_| self.open);
        let Some(eq) = self
            .hovered
            .as_deref()
            .or(shown)
            .and_then(|path| registry().equation_for(path))
        else {
            return TermColors::none();
        };
        TermColors::of(eq).with_members(registry(), &Design { inputs, results })
    }

    /// Takes in what the user did with the readouts this frame: the value hovered marks its
    /// terms from the next frame (asked for at once); a value clicked opens here.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
        if events.hovered != self.hovered {
            self.hovered = events.hovered;
            ctx.request_repaint();
        }
        if let Some(path) = events.clicked {
            self.open_path(&path);
            ctx.request_repaint();
        }
        if let Some(id) = events.note {
            self.open_note(id);
            ctx.request_repaint();
        }
    }
}

/// The label of an input or result path (the path itself for neither).
pub fn term_label(path: &str) -> &str {
    result_info(path)
        .map(|info| info.meta.label)
        .or_else(|| InputCatalogue::get().entry(path).map(|e| e.meta.label))
        .unwrap_or(path)
}

/// A symbol as one line of plain text the default fonts draw (`T_pull`, `θ_op`, `S_3^iron`),
/// for the breadcrumb and the "used by" links.
pub fn plain_symbol(markup: &str) -> String {
    let text = match Symbol::parse(markup) {
        Ok(s) => {
            let mut text = s.base;
            if let Some(sub) = s.sub {
                text.push('_');
                text.push_str(&sub);
            }
            if let Some(sup) = s.sup {
                text.push('^');
                text.push_str(&sup);
            }
            text
        }
        Err(_) => markup.to_owned(),
    };
    glyph_safe(&text)
}

/// The plain symbol of `path`, its label for a path without one.
fn crumb(path: &str) -> String {
    registry()
        .symbol(path)
        .map_or_else(|| term_label(path).to_owned(), plain_symbol)
}

/// A value with its unit; a selector code with its choice's label after a colon
/// (`1: steel circuit`, and `5: 1, 3, 5 (workbook)` without nested parentheses).
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = with_unit(format_value(value), unit);
    match value {
        Value::Int(code) => registry()
            .choices(path)
            .iter()
            .find(|(c, _)| c == code)
            .map_or(text.clone(), |(_, label)| format!("{text}: {label}")),
        _ => text,
    }
}

/// The paths of family members the formula lists but the design does not sum (the harmonics
/// past the set): the term list leaves them out.
fn unsummed(eq: &Equation, terms: &Design<'_>) -> Vec<String> {
    let mut templates = Vec::new();
    eq.formula.visit(&mut |e| {
        if let Expr::FamilyTerm(r) = e
            && !templates.contains(&r.path)
        {
            templates.push(r.path.clone());
        }
    });
    let summed: Vec<String> = templates
        .iter()
        .flat_map(|t| registry().family_members(t, terms))
        .collect();
    eq.terms
        .iter()
        .filter(|path| templates.iter().any(|t| covers(t, path)) && !summed.contains(path))
        .cloned()
        .collect()
}

/// Draws the Equation panel: its header row (heading, breadcrumb, close), then the path shown.
pub fn explorer_ui(
    ui: &mut egui::Ui,
    explorer: &mut Explorer,
    inputs: &DesignInputs,
    results: &DesignResults,
) {
    let terms = Design { inputs, results };
    ui.horizontal_wrapped(|ui| {
        // Whole crumbs move to the next row (a wrapped text would start mid-row).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        ui.strong(EQUATION_PANEL);
        ui.separator();
        if let Some(note) = explorer.note.and_then(reviewed_note) {
            let _ = ui.selectable_label(true, glyph_safe(note.title));
        }
        let mut back = None;
        let last = explorer.trail.len().saturating_sub(1);
        let first = explorer.trail.len().saturating_sub(SHOWN_CRUMBS);
        if first > 0 {
            ui.weak(ELIDED)
                .on_hover_text(format!("{first} earlier steps"));
        }
        for (i, path) in explorer.trail.iter().enumerate().skip(first) {
            if i > 0 {
                ui.weak(">");
            }
            if ui
                .selectable_label(i == last, crumb(path))
                .on_hover_text(term_label(path))
                .clicked()
                && i != last
            {
                back = Some(i);
            }
        }
        if let Some(i) = back {
            explorer.back_to(i);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(CLOSE).clicked() {
                explorer.close();
            }
            if ui
                .button(START_HERE_BUTTON)
                .on_hover_text("A suggested order: torque, back iron, temperature, demagnetization, slip heating, clamps")
                .clicked()
            {
                explorer.start(0);
            }
            ui.checkbox(&mut explorer.explain, EXPLAIN)
                .on_hover_text("The teaching note of the open equation");
        });
    });
    if let Some(step) = explorer.start_here {
        ui.horizontal_wrapped(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.weak(start_here_text(step));
            if ui
                .add_enabled(step > 0, egui::Button::new(PREVIOUS))
                .clicked()
            {
                explorer.start(step - 1);
            }
            if ui
                .add_enabled(step + 1 < START_HERE.len(), egui::Button::new(NEXT))
                .clicked()
            {
                explorer.start(step + 1);
            }
            if ui.button(STOP).clicked() {
                explorer.stop();
            }
        });
    }
    ui.separator();
    egui::ScrollArea::vertical()
        .id_salt("magcoupling_equation_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if let Some(note) = explorer.note.and_then(reviewed_note) {
                if let Some(path) = note_alone_ui(ui, note) {
                    explorer.open_path(&path);
                }
                return;
            }
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
            };
            // The marks on screen are another equation's while its value is hovered.
            let others_marked = explorer.hovered().is_some_and(|h| h != path);
            let follow = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, others_marked, explorer.explain),
                None => {
                    no_equation_body(ui, &path, results);
                    if explorer.explain {
                        ui.weak(NO_NOTE);
                    }
                    None
                }
            };
            ui.add_space(6.0);
            let users = registry().used_by(&path);
            let mut up = None;
            if !users.is_empty() {
                ui.horizontal_wrapped(|ui| {
                    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                    ui.strong(USED_BY);
                    for user in users {
                        if ui
                            .link(crumb(user))
                            .on_hover_text(term_label(user))
                            .clicked()
                        {
                            up = Some(user.clone());
                        }
                    }
                });
            }
            if let Some(term) = follow {
                explorer.follow(&term, &terms);
            } else if let Some(user) = up {
                explorer.drill(&user);
            }
        });
}

/// A teaching note: its title, its sentences, its watch-out line, its diagram and its sources
/// (every text drawn through `glyph_safe`).
pub fn note_ui(ui: &mut egui::Ui, note: &Note) {
    ui.strong(glyph_safe(note.title));
    for sentence in note.sentences {
        ui.add(egui::Label::new(glyph_safe(sentence)).wrap());
    }
    if let Some(watch_out) = note.watch_out {
        ui.add(
            egui::Label::new(
                egui::RichText::new(format!("{WATCH_OUT}: {}", glyph_safe(watch_out)))
                    .color(ui.visuals().warn_fg_color),
            )
            .wrap(),
        );
    }
    if let Some(diagram) = note.diagram {
        diagram_ui(ui, diagram);
    }
    ui.add(
        egui::Label::new(
            egui::RichText::new(format!(
                "{SOURCES}: {}",
                glyph_safe(&note.sources.join("; "))
            ))
            .small()
            .weak(),
        )
        .wrap(),
    );
}

/// A note opened on its own: links to the equations it explains (a family's template is left
/// out), then the note. Returns the equation clicked.
fn note_alone_ui(ui: &mut egui::Ui, note: &Note) -> Option<String> {
    let paths: Vec<&str> = note
        .equations
        .iter()
        .copied()
        .filter(|p| !p.contains('#') && registry().equation_for(p).is_some())
        .collect();
    let mut clicked = None;
    if !paths.is_empty() {
        ui.horizontal_wrapped(|ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.strong(EXPLAINS);
            for path in paths {
                if ui
                    .link(crumb(path))
                    .on_hover_text(term_label(path))
                    .clicked()
                {
                    clicked = Some(path.to_owned());
                }
            }
        });
    }
    note_ui(ui, note);
    clicked
}

/// The open equation: its label and cell, the equation large (a term clicked is followed), the
/// value, the corrections it embodies, with `explain` its teaching note, then the term list,
/// its swatches faded while `others_marked`. Returns the term clicked.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    others_marked: bool,
    explain: bool,
) -> Option<String> {
    let mut clicked = None;
    ui.weak(format!(
        "{} ({})",
        eq.label,
        eq.cell.as_deref().unwrap_or("Rust-only result")
    ));
    let colors = TermColors::of(eq);
    let ink = ui.visuals().text_color();
    let laid = ui.fonts(|f| layout_equation(f, registry(), eq, &colors, PANEL_SIZE, ink));
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_equation_formula")
        .show(ui, |ui| {
            let response = laid_ui(ui, &laid, ink, Sense::click());
            if response.clicked()
                && let Some(at) = response.interact_pointer_pos()
            {
                clicked = term_at(&laid, response.rect.min, at);
            }
        });
    let value = terms.results.get(&eq.target).unwrap_or(Value::None);
    ui.strong(format!(
        "{} = {}",
        plain_symbol(&eq.symbol),
        value_text(&eq.target, &value, eq.unit)
    ));
    let modified = registry().modified_assumptions_upstream(&eq.target, terms.inputs);
    if !modified.is_empty() {
        let names: Vec<&str> = modified.iter().map(|a| a.label).collect();
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("{DEPENDS_ON_MODIFIED}: {}", names.join(", ")),
        );
    }
    let corrections = registry().corrections_upstream(&eq.target);
    if !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        ui.weak(format!("{CORRECTIONS}: {}", ids.join(", ")));
    }
    if explain {
        ui.separator();
        match notes::note_for(&eq.target) {
            Some(note) => note_ui(ui, note),
            None => {
                ui.weak(NO_NOTE);
            }
        }
        ui.separator();
    }
    ui.add_space(6.0);
    ui.strong(TERMS);
    let hidden = unsummed(eq, terms);
    // The list scrolls sideways when the region is narrower than its rows (the tag, its last
    // column, would be cut off), as the equation above it does.
    egui::ScrollArea::horizontal()
        .id_salt("magcoupling_equation_terms_scroll")
        .show(ui, |ui| {
            egui::Grid::new("magcoupling_equation_terms")
                .striped(true)
                .show(ui, |ui| {
                    for row in registry().term_rows(eq, terms) {
                        if hidden.contains(&row.path) {
                            continue;
                        }
                        let color = colors
                            .get(&row.path)
                            .or_else(|| {
                                // A family member takes its template's colour.
                                colors_of_template(eq, &colors, &row.path)
                            })
                            .unwrap_or(ink);
                        let key = if others_marked {
                            color.gamma_multiply(SWATCH_DIM)
                        } else {
                            color
                        };
                        swatch(ui, key);
                        let symbol = ui
                            .fonts(|f| layout_symbol(f, registry(), &row.symbol, TERM_SIZE, color));
                        laid_ui(ui, &symbol, ink, Sense::hover());
                        let value = row.value.clone().unwrap_or(Value::None);
                        ui.label(value_text(&row.path, &value, &row.unit));
                        if ui
                            .add(egui::Label::new(term_label(&row.path)).sense(Sense::click()))
                            .on_hover_text(&row.path)
                            .clicked()
                        {
                            clicked = Some(row.path.clone());
                        }
                        let style = registry()
                            .term_style(&row.path, terms.inputs)
                            .expect("every term is an input or a result (registry build)");
                        let tag = term_tag(style);
                        if style.affected_by_modified_assumption
                            || matches!(style.kind, TermKind::Input { assumption: true })
                        {
                            ui.colored_label(ui.visuals().warn_fg_color, tag);
                        } else {
                            ui.weak(tag);
                        }
                        ui.end_row();
                    }
                });
        });
    clicked
}

/// The colour of the family template of `eq` that covers `path`.
fn colors_of_template(eq: &Equation, colors: &TermColors, path: &str) -> Option<Color32> {
    let mut found = None;
    eq.formula.visit(&mut |e| {
        if let Expr::FamilyTerm(r) = e
            && found.is_none()
            && covers(&r.path, path)
        {
            found = colors.get(&r.path);
        }
    });
    found
}

/// What a term is, in the term list: an assumption, an input, a result with or without an
/// equation; the changed-from-default dot on an input that differs from its default; a result
/// a modified assumption flows into.
pub fn term_tag(style: TermStyle) -> String {
    let kind = match style.kind {
        TermKind::Input { assumption: true } => ASSUMPTION_TAG,
        TermKind::Input { assumption: false } => "input: click to find its row",
        TermKind::Explained => "click to open its equation",
        TermKind::CellOnly => "no equation record",
    };
    let mut tag = if style.changed_from_default {
        format!("{CHANGED_DOT} changed; {kind}")
    } else {
        kind.to_owned()
    };
    if style.affected_by_modified_assumption && !matches!(style.kind, TermKind::Input { .. }) {
        tag.push_str("; depends on a modified assumption");
    }
    tag
}

/// A small filled square in a term's colour.
fn swatch(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
    ui.painter().rect_filled(rect, 2.0, color);
}

/// A result without an equation record: its label, value and cell.
fn no_equation_body(ui: &mut egui::Ui, path: &str, results: &DesignResults) {
    let label = term_label(path);
    let value = results.get(path).unwrap_or(Value::None);
    let unit = result_info(path).map_or("", |info| info.meta.unit);
    ui.strong(format!("{label} = {}", value_text(path, &value, unit)));
    let cell = result_info(path)
        .and_then(|info| info.cell.clone())
        .unwrap_or_else(|| "Rust-only result".to_owned());
    ui.weak(cell);
    ui.weak(NO_EQUATION);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::ResultSet;
    use crate::gui::results_table::table_entries;

    fn design() -> (DesignInputs, DesignResults) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        (inputs, results)
    }

    #[test]
    fn opening_drilling_and_going_back_walk_the_trail() {
        let mut explorer = Explorer::default();
        assert_eq!(explorer.current(), None);
        explorer.open_path("model.pullout_Nm");
        assert!(explorer.open);
        explorer.drill("model.f_end");
        explorer.drill("model.f_end");
        assert_eq!(explorer.trail(), ["model.pullout_Nm", "model.f_end"]);
        explorer.back_to(0);
        assert_eq!(explorer.current(), Some("model.pullout_Nm"));
        // A readout clicked starts a new trail.
        explorer.drill("model.f_cal");
        explorer.open_path("mass.total_g");
        assert_eq!(explorer.trail(), ["mass.total_g"]);
    }

    #[test]
    fn closing_hides_the_panel_and_clears_the_focus_but_keeps_the_trail() {
        // Decision M43-12: the frame of a leaf term's row goes with the panel.
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        explorer.focus_input("coupling.c_end");
        explorer.close();
        assert!(!explorer.open);
        assert_eq!(explorer.focus(), None);
        assert_eq!(explorer.trail(), ["model.f_end"]);
    }

    #[test]
    fn a_long_walk_keeps_the_last_steps() {
        let mut explorer = Explorer::default();
        explorer.open_path("p0");
        for i in 1..=MAX_TRAIL + 5 {
            explorer.drill(&format!("p{i}"));
        }
        assert_eq!(explorer.trail().len(), MAX_TRAIL);
        assert_eq!(
            explorer.current(),
            Some(format!("p{}", MAX_TRAIL + 5).as_str())
        );
    }

    #[test]
    fn following_a_term_drills_into_a_result_and_focuses_an_input() {
        let (inputs, results) = design();
        let terms = Design {
            inputs: &inputs,
            results: &results,
        };
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        explorer.follow("coupling.c_end", &terms);
        assert_eq!(
            explorer.current(),
            Some("model.f_end"),
            "an input is a leaf"
        );
        assert_eq!(
            explorer.focus().map(|f| f.path.as_str()),
            Some("coupling.c_end")
        );
        assert!(explorer.take_scroll("coupling.c_end"));
        assert!(!explorer.take_scroll("coupling.c_end"), "once");
        explorer.follow("model.pole_pitch_mm", &terms);
        assert_eq!(explorer.current(), Some("model.pole_pitch_mm"));
        assert_eq!(explorer.focus(), None);
        // A Σ's family term drills into the first harmonic summed.
        explorer.open_path("model.tau_Pa");
        explorer.follow("model.tau#_Pa", &terms);
        assert_eq!(explorer.current(), Some("model.tau1_Pa"));
    }

    #[test]
    fn the_marks_follow_the_hovered_readout_else_the_open_equation() {
        let (inputs, results) = design();
        let mut explorer = Explorer::default();
        assert!(explorer.marks(&inputs, &results).is_empty());
        explorer.open_path("model.f_end");
        assert!(
            explorer
                .marks(&inputs, &results)
                .get("coupling.c_end")
                .is_some()
        );
        explorer.open = false;
        assert!(
            explorer.marks(&inputs, &results).is_empty(),
            "a closed panel marks nothing"
        );
        explorer.open = true;
        let ctx = egui::Context::default();
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                hovered: Some("model.pullout_Nm".to_owned()),
                ..ReadoutEvents::default()
            },
        );
        let marks = explorer.marks(&inputs, &results);
        assert!(
            marks.get("model.f_end").is_some(),
            "the hovered equation's terms"
        );
        assert!(marks.get("coupling.c_end").is_none());
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                clicked: Some("mass.total_g".to_owned()),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(explorer.trail(), ["mass.total_g"]);
        assert_eq!(explorer.hovered(), None);
    }

    /// The colours of the term list's swatches (its 10-point squares), as one frame of
    /// `explorer_ui` on the default design draws them.
    fn swatch_colors(ctx: &egui::Context, explorer: &mut Explorer) -> Vec<Color32> {
        use crate::gui::test_support::{SCREEN, flat_shapes, sized_frame};
        let (inputs, results) = design();
        let output = sized_frame(ctx, SCREEN, Vec::new(), |ui| {
            explorer_ui(ui, explorer, &inputs, &results);
        });
        flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(r) if r.rect.size() == Vec2::splat(10.0) => Some(r.fill),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_sum_s_harmonic_set_has_its_swatch_in_the_term_list() {
        // σ = Σ_{n ∈ H} σ_n: the set's selector N_h is σ's first term (the registry's order), in
        // the first colour, not the text colour; the harmonics summed, 1, 3 and 5, take their
        // template's colour.
        use crate::gui::typeset::TERM_PALETTE;
        let ctx = egui::Context::default();
        let mut explorer = Explorer::default();
        explorer.open_path("model.tau_Pa");
        let [first, second, ..] = TERM_PALETTE;
        assert_eq!(
            swatch_colors(&ctx, &mut explorer),
            [first, second, second, second]
        );
    }

    #[test]
    fn the_swatches_fade_while_another_equation_s_value_is_hovered() {
        // The marks on screen follow the hovered value's equation (decision M43-3): while it
        // is not the open one, the term list's swatches fade, so a colour on screen is never
        // read against the open equation's key.
        use crate::gui::typeset::TERM_PALETTE;
        let ctx = egui::Context::default();
        let swatches = |explorer: &mut Explorer| swatch_colors(&ctx, explorer);
        let hover = |explorer: &mut Explorer, path: &str| {
            explorer.end_frame(
                &ctx,
                ReadoutEvents {
                    hovered: Some(path.to_owned()),
                    ..ReadoutEvents::default()
                },
            );
        };
        let mut explorer = Explorer::default();
        explorer.open_path("model.pullout_Nm");
        // T_pull = T_2D f_end f_cal: three swatches in the first three colours.
        let full = TERM_PALETTE[..3].to_vec();
        assert_eq!(swatches(&mut explorer), full);
        // The open equation's own value hovered: the marks are its colours.
        hover(&mut explorer, "model.pullout_Nm");
        assert_eq!(swatches(&mut explorer), full);
        // The cup OD hovered: the marks on screen are D_cup's.
        hover(&mut explorer, "model.cup_od_mm");
        let faded: Vec<Color32> = full.iter().map(|c| c.gamma_multiply(SWATCH_DIM)).collect();
        assert_eq!(swatches(&mut explorer), faded);
        explorer.end_frame(&ctx, ReadoutEvents::default());
        assert_eq!(swatches(&mut explorer), full, "the pointer left");
    }

    #[test]
    fn the_term_list_leaves_out_the_harmonics_past_the_set() {
        let (inputs, results) = design();
        let terms = Design {
            inputs: &inputs,
            results: &results,
        };
        let eq = registry().equation_for("model.tau_Pa").unwrap();
        let hidden = unsummed(eq, &terms);
        assert_eq!(hidden, ["model.tau7_Pa", "model.tau9_Pa", "model.tau11_Pa"]);
        assert!(unsummed(registry().equation_for("model.f_end").unwrap(), &terms).is_empty());
    }

    #[test]
    fn start_here_walks_the_suggested_order_and_a_click_leaves_it() {
        let mut explorer = Explorer::default();
        assert!(!explorer.explain, "Explain is off by default");
        explorer.start(0);
        assert_eq!(explorer.start_here(), Some(0));
        assert!(explorer.explain && explorer.open);
        assert_eq!(explorer.current(), Some(START_HERE[0].1));
        let last = START_HERE.len() - 1;
        explorer.start(last);
        explorer.start(last + 1);
        assert_eq!(
            explorer.start_here(),
            Some(last),
            "past the last step: nothing"
        );
        assert_eq!(explorer.current(), Some(START_HERE[last].1));
        // Drilling keeps the walk; a readout clicked leaves it.
        explorer.drill("clamps.preload_N");
        assert_eq!(explorer.start_here(), Some(last));
        explorer.open_path("mass.total_g");
        assert_eq!(explorer.start_here(), None);
        explorer.start(2);
        explorer.stop();
        assert_eq!(explorer.start_here(), None);
        assert_eq!(explorer.current(), Some(START_HERE[2].1));
        assert_eq!(
            start_here_text(0),
            format!(
                "{START_HERE_BUTTON} 1 of {}: Harmonic decomposition",
                START_HERE.len()
            )
        );
    }

    #[test]
    fn every_start_here_step_opens_an_equation_with_its_reviewed_note() {
        for &(id, path) in START_HERE {
            assert!(registry().equation_for(path).is_some(), "{path}");
            assert_eq!(notes::note_for(path).map(|n| n.id), Some(id), "{path}");
        }
    }

    #[test]
    fn a_note_by_id_shows_only_once_reviewed() {
        use crate::engine::explain::notes::NOTES;
        for note in NOTES {
            let reviewed = matches!(note.review, Review::Reviewed { .. });
            assert_eq!(
                reviewed_note(note.id).map(|n| n.id),
                reviewed.then_some(note.id)
            );
        }
        assert_eq!(reviewed_note("no.such.note").map(|n| n.id), None);
    }

    #[test]
    fn a_note_opened_alone_replaces_the_trail_until_a_value_is_opened() {
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        explorer.open_note("a5.low_saturation");
        assert_eq!(explorer.note(), Some("a5.low_saturation"));
        assert_eq!(explorer.current(), None);
        assert!(explorer.open);
        explorer.open_path("materials.circuit_backiron");
        assert_eq!(explorer.note(), None);
        explorer.open_note("a5.low_saturation");
        explorer.start(0);
        assert_eq!(explorer.note(), None);
    }

    #[test]
    fn a_term_s_tag_says_assumption_changed_and_affected() {
        let mut inputs = DesignInputs::default();
        let tag = |path: &str, inputs: &DesignInputs| {
            term_tag(registry().term_style(path, inputs).unwrap())
        };
        assert_eq!(tag("coupling.c_end", &inputs), ASSUMPTION_TAG);
        assert_eq!(
            tag("coupling.npole", &inputs),
            "input: click to find its row"
        );
        assert_eq!(tag("model.f_end", &inputs), "click to open its equation");
        inputs.coupling.c_end = 0.2;
        assert_eq!(
            tag("coupling.c_end", &inputs),
            format!("{CHANGED_DOT} changed; {ASSUMPTION_TAG}")
        );
        assert_eq!(
            tag("model.f_end", &inputs),
            "click to open its equation; depends on a modified assumption"
        );
        inputs.coupling.npole = 12;
        assert_eq!(
            tag("coupling.npole", &inputs),
            format!("{CHANGED_DOT} changed; input: click to find its row")
        );
    }

    #[test]
    fn labels_symbols_and_choices_read_as_the_panel_shows_them() {
        assert_eq!(term_label("model.f_end"), "End-effect factor");
        assert_eq!(term_label("coupling.c_end"), "End-effect coefficient");
        assert_eq!(term_label("no.such"), "no.such");
        assert_eq!(plain_symbol("ϑ_{op}"), "θ_op");
        assert_eq!(plain_symbol("T_{pull}"), "T_pull");
        assert_eq!(plain_symbol("S_{3}^{iron}"), "S_3^iron");
        // A selector code with its choice's label, after a colon: a label holding its own
        // parentheses (the workbook's harmonic set) reads without nesting them.
        assert_eq!(
            value_text("coupling.backiron", &Value::Int(1), "-"),
            "1: steel circuit"
        );
        assert_eq!(
            value_text("coupling.max_harmonic", &Value::Int(5), "-"),
            "5: 1, 3, 5 (workbook)"
        );
        // A code outside the choices has no label to add.
        assert_eq!(
            value_text("coupling.max_harmonic", &Value::Int(4), "-"),
            "4"
        );
        let (_, results) = design();
        let cell_only = table_entries()
            .iter()
            .find(|e| registry().term_kind(&e.path) == Some(TermKind::CellOnly))
            .expect("a result without a record");
        assert!(results.get(&cell_only.path).is_some());
    }
}
