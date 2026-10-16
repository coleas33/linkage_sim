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
//! without an equation record shows its label, value and (with the spreadsheet details) its
//! workbook cell. The panel docks at the
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
//! reviewer has checked), with its diagram ([`crate::gui::diagrams`]), its "What to change"
//! entries, its "Watch out" lines and its sources under a "Where this comes from" fold that
//! starts closed ([`note_ui`]), so it follows what the user investigates. A `{result.path}` in a
//! note's text is drawn as that result's number on the design shown ([`live_value`]), or as
//! "no prediction" where it has none. An entry's input is a link to its row, or plain text with
//! the reason where the input is not free on the design shown ([`held_reason`]: Auto holds it
//! or a part pick sets it; or the slider is at its end that way). "Start here" walks the
//! suggested order ([`notes::START_HERE`]): each step opens its equation with Explain on. A warning's link
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

use crate::engine::clamps;
use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::note_text;
use crate::engine::explain::notes::{self, Change, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::picks::set_by_pick;
use crate::gui::corrections::correction_title;
use crate::gui::dashboard::{
    end_effect_out_of_range, greyed_by_end_effect, result_info, shown_text,
};
use crate::gui::diagrams::diagram_ui;
use crate::gui::format::with_unit;
use crate::gui::guidance::result_text;
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::sizing::{SizingState, held_by_auto};
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
pub const NO_EQUATION: &str = "This value has no equation to show; its name and value are above.";

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

/// The Start here walk's topics before its last, the clamp note's: typed once, for
/// [`start_here_hover`]'s two forms.
macro_rules! start_here_topics {
    () => {
        "A suggested order: how the coupling makes torque, the magnet pattern, back iron, pull-out, magnet ends, calibration, temperature, slipping and its heat, demagnetization"
    };
}

/// The Start here button's hover: the walk's topics in [`notes::start_here_for`]'s order
/// (slice-2 spec 3.3), in plain words; the clamp step's topic only while the shaft-clamp section
/// is not paused (`clamps_paused`).
pub const fn start_here_hover(clamps_paused: bool) -> &'static str {
    if clamps_paused {
        start_here_topics!()
    } else {
        concat!(start_here_topics!(), ", clamps")
    }
}

/// The hover the Start here button shows: [`start_here_hover`] the shaft-clamp section's pause
/// (`clamps::PAUSED`).
pub const START_HERE_HOVER: &str = start_here_hover(clamps::PAUSED);

/// The start-here steps' buttons.
pub const PREVIOUS: &str = "Previous";
pub const NEXT: &str = "Next";
pub const STOP: &str = "Stop";

/// What Explain shows for an equation without a reviewed note.
pub const NO_NOTE: &str = "No teaching note for this equation.";

/// The heading of a note's watch-out lines.
pub const WATCH_OUT: &str = "Watch out";

/// The heading of a note's "What to change" entries (slice-2 spec 3.1).
pub const WHAT_TO_CHANGE: &str = "What to change";

/// The fold under a note that holds its sources, closed until clicked (slice-2 spec 3.1).
pub const WHERE_FROM: &str = "Where this comes from";

/// Why a change entry's input is plain text, not a link: Auto sets it (Torque -> Magnets).
pub const HELD_BY_AUTO: &str = "held by Auto";

/// Why a Raise entry's input is plain text: it is at the top of its slider.
pub const AT_TOP: &str = "at the top of its range";

/// Why a Lower entry's input is plain text: it is at the bottom of its slider.
pub const AT_BOTTOM: &str = "at the bottom of its range";

/// The start of the links under a note opened on its own.
pub const EXPLAINS: &str = "Explains";

/// The note `id` if a physics reviewer has checked it (spec A4 "Accuracy gate"), as
/// [`notes::note_for`] gives an equation's: every note the panel or the dashboard shows by id.
pub fn reviewed_note(id: &str) -> Option<&'static Note> {
    notes::note(id).filter(|n| matches!(n.review, Review::Reviewed { .. }))
}

/// The results that use `path`, its "Used by" links: the registry's, less those the paused
/// shaft-clamp section hides while `clamps_paused` ([`clamps::hides`]), so no link leads into
/// it (the cold-high torque's clamp demand).
pub fn used_by_shown(path: &str, clamps_paused: bool) -> Vec<&'static String> {
    registry()
        .used_by(path)
        .iter()
        .filter(|user| !clamps::hides(user, clamps_paused))
        .collect()
}

/// What the start-here bar adds on the ferrite step: its check reads n/a on NdFeB rings (spec
/// 3.3, TC-32).
pub const FERRITE_ONLY: &str = "(ferrite rings only)";

/// The start of the start-here bar: "Start here 13 of 13 (ferrite rings only): <title>" on the
/// ferrite step while the shaft-clamp section is paused (13 of 14 with it), the step and its
/// title alone on every other one.
pub fn start_here_text(step: usize) -> String {
    let (id, _) = START_HERE[step];
    let title = reviewed_note(id).map_or_else(String::new, |n| glyph_safe(n.title));
    let only = if id == "ferrite_cold_demag" {
        format!(" {FERRITE_ONLY}")
    } else {
        String::new()
    };
    format!(
        "{START_HERE_BUTTON} {} of {}{only}: {title}",
        step + 1,
        START_HERE.len()
    )
}

/// The text size of the open equation [points].
pub const PANEL_SIZE: f32 = 22.0;

/// The text size of a term's symbol in the term list [points].
pub const TERM_SIZE: f32 = 15.0;

/// The Equation panel's starting height as a share of the centre region's height.
pub const PANEL_SHARE: f32 = 0.45;

/// The Equation panel's smallest height [points], starting or dragged: its header row wrapped
/// to two rows and a few lines of the equation. In a centre region shorter than this plus
/// [`VIEW_STRIP`] the panel keeps this height (the whole region, when that is shorter) and the
/// tab row and the view get the rest.
pub const MIN_PANEL_HEIGHT: f32 = 120.0;

/// The strip the centre region keeps above the Equation panel for its tab row and a small view
/// [points]: dragging the panel's top edge up stops there.
pub const VIEW_STRIP: f32 = 160.0;

/// The Equation panel's drag range and starting height [points] in a centre region `centre`
/// points tall: it starts at [`PANEL_SHARE`] of the region and never below
/// [`MIN_PANEL_HEIGHT`], and its top edge stops [`VIEW_STRIP`] below the region's top, except
/// in a region too short for both, where the panel keeps [`MIN_PANEL_HEIGHT`] (the whole
/// region, when that is shorter). The starting height lies in the range.
pub fn panel_heights(centre: f32) -> (egui::Rangef, f32) {
    // f32::max passes over NaN: a region that is not a number is no region.
    let centre = centre.max(0.0);
    let min = MIN_PANEL_HEIGHT.min(centre);
    let max = (centre - VIEW_STRIP).max(min);
    let start = (centre * PANEL_SHARE).max(min).min(max);
    (egui::Rangef::new(min, max), start)
}

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

    /// Follows a link clicked in a teaching note: the input's row is highlighted and scrolled
    /// to, as a leaf term's is; the note stays open.
    pub fn follow_note_link(&mut self, link: NoteLink) {
        match link {
            NoteLink::Input(path) => self.focus_input(path),
        }
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
    /// terms from the next frame (asked for at once); a value clicked opens here; a check
    /// guide's link highlights its input's row, as a leaf term does.
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
        if let Some(path) = events.focus {
            self.focus_input(path);
            ctx.request_repaint();
        }
    }
}

/// The on-screen name of an input or result path ([`label_for`]: the overlay's display label,
/// else the metadata's); the path itself for neither. The term list, the breadcrumb, the "used
/// by" links and the trace banner all read it.
pub fn term_label(path: &str) -> &str {
    match label_for(path) {
        "" => path,
        label => label,
    }
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
/// (`1: steel circuit`, and `5: 1, 3, 5 (default)` without nested parentheses); the cup wall's
/// failing verdict with its input's label ([`shown_text`], ruling R9).
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = with_unit(shown_text(path, value), unit);
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

/// Draws the Equation panel: its header row (heading, breadcrumb, close), then the path shown;
/// with `show_details` (the header's Show spreadsheet details, spec S2-5) also the value's
/// workbook cell, the corrections its equation embodies and each term's internal name on hover.
/// `held` says why an input is not free on the design shown ([`held_reason`]), for the notes'
/// change entries.
pub fn explorer_ui(
    ui: &mut egui::Ui,
    explorer: &mut Explorer,
    inputs: &DesignInputs,
    results: &DesignResults,
    show_details: bool,
    held: &dyn Fn(&str) -> Option<String>,
) {
    let terms = Design { inputs, results };
    let view = NoteView {
        inputs,
        results,
        held,
    };
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
                .on_hover_text(START_HERE_HOVER)
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
                let (equation, link) = note_alone_ui(ui, note, &view);
                if let Some(link) = link {
                    explorer.follow_note_link(link);
                }
                if let Some(path) = equation {
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
            let (follow, link) = match registry().equation_for(&path) {
                Some(eq) => equation_body(
                    ui,
                    eq,
                    &terms,
                    others_marked,
                    explorer.explain,
                    show_details,
                    &view,
                ),
                None => {
                    no_equation_body(ui, &path, results, show_details);
                    if explorer.explain {
                        ui.weak(NO_NOTE);
                    }
                    (None, None)
                }
            };
            if let Some(link) = link {
                explorer.follow_note_link(link);
            }
            ui.add_space(6.0);
            let users = used_by_shown(&path, clamps::PAUSED);
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

/// What a note's change entries are drawn against: the design shown, its results, and why an
/// input is not free on it (`held`: Auto holds it or a part pick sets it, [`held_reason`];
/// `None` for a free input).
pub struct NoteView<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub held: &'a dyn Fn(&str) -> Option<String>,
}

/// What a click in a teaching note asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteLink {
    /// Highlight this input's row, as a leaf term does ([`Explorer::follow_note_link`]).
    Input(&'static str),
}

/// Why the input at `path` is not free on the design `inputs` shown with `sizing`, for a note's
/// change entry ([`NoteView::held`]): Auto sets it in Torque -> Magnets ([`HELD_BY_AUTO`]), or
/// a part pick sets it in place of its own value ("set by the `<pick's label>` pick",
/// [`set_by_pick`]); `None` when it is free.
pub fn held_reason(sizing: &SizingState, inputs: &DesignInputs, path: &str) -> Option<String> {
    if held_by_auto(sizing, path) {
        return Some(HELD_BY_AUTO.to_owned());
    }
    set_by_pick(inputs, path).map(|pick| format!("set by the {} pick", label_for(pick)))
}

/// Why a change entry draws its input as plain text on `view`'s design: the input is not free
/// there (`view.held`), or one more slider step its way is past its slider ([`AT_TOP`],
/// [`AT_BOTTOM`]; the step a check guide's fix takes, `SliderRange::step_from`). `None` when the
/// input is a link: a choice, an entered number and an optional number left blank (it enters at
/// a result) are never at an end.
fn unlinked_reason(change: &Change, view: &NoteView<'_>) -> Option<String> {
    if let Some(reason) = (view.held)(change.input) {
        return Some(reason);
    }
    let up = change.step.up()?;
    let range = InputCatalogue::get().entry(change.input)?.meta.range?;
    let x = match view.inputs.get(change.input)? {
        Value::Num(x) if x.is_finite() => x,
        Value::Int(n) => n as f64,
        _ => return None,
    };
    if range.step_from(x, up).is_some() {
        return None;
    }
    Some(if up { AT_TOP } else { AT_BOTTOM }.to_owned())
}

/// What a `{path}` in a note shows on the design whose results are `results` (slice-2 spec
/// 3.2): the result's number with its unit, as the readouts show it
/// (`guidance::result_text`); `None`, drawn as [`note_text::NO_VALUE`], for a
/// path no result has, a number that is not finite, a text (a result that reads "n/a"
/// where it has no number, or a verdict: a live value is a number), and a result computed
/// from an out-of-range end effect (the pull-out and everything after it: the model gives no
/// prediction there, spec 5).
pub fn live_value(results: &DesignResults, path: &str) -> Option<String> {
    if end_effect_out_of_range(results).is_some()
        && (after_the_end_factor().contains(path) || greyed_by_end_effect(results, path))
    {
        return None;
    }
    match results.get(path)? {
        Value::Num(x) if x.is_finite() => Some(result_text(results, path)),
        Value::Int(_) => Some(result_text(results, path)),
        Value::Num(_) | Value::Text(_) | Value::None => None,
    }
}

/// Every result computed from the end-effect factor (the registry is fixed: computed once).
fn after_the_end_factor() -> &'static std::collections::BTreeSet<String> {
    static AFTER: std::sync::OnceLock<std::collections::BTreeSet<String>> =
        std::sync::OnceLock::new();
    AFTER.get_or_init(|| registry().downstream("model.f_end"))
}

/// A note's text as drawn on `view`'s design: each `{path}` replaced by its live value
/// ([`live_value`]), then through `glyph_safe`.
fn note_text_on(text: &str, view: &NoteView<'_>) -> String {
    glyph_safe(&note_text::render(text, |path| {
        live_value(view.results, path)
    }))
}

/// A teaching note (slice-2 spec 3.1): its title; what it is (its sentences) and its diagram;
/// "What to change", each entry "`<Step> <input's label>: <text>`", the label a link to the
/// input's row unless `unlinked_reason` gives a reason (then plain text with the reason);
/// "Watch out" and its lines; and the "Where this comes from" fold, closed until clicked,
/// holding its sources. The sentences, change texts and watch-outs show live values
/// (`note_text_on`); every text is drawn through `glyph_safe`. Returns the link clicked.
pub fn note_ui(ui: &mut egui::Ui, note: &Note, view: &NoteView<'_>) -> Option<NoteLink> {
    let mut clicked = None;
    ui.strong(glyph_safe(note.title));
    for sentence in note.sentences {
        ui.add(egui::Label::new(note_text_on(sentence, view)).wrap());
    }
    if let Some(diagram) = note.diagram {
        diagram_ui(ui, diagram);
    }
    if !note.change.is_empty() {
        ui.strong(WHAT_TO_CHANGE);
        for change in note.change {
            if change_ui(ui, change, view) {
                clicked = Some(NoteLink::Input(change.input));
            }
        }
    }
    if !note.watch_out.is_empty() {
        let warn = ui.visuals().warn_fg_color;
        ui.label(egui::RichText::new(WATCH_OUT).strong().color(warn));
        for line in note.watch_out {
            let text = egui::RichText::new(note_text_on(line, view)).color(warn);
            ui.add(egui::Label::new(text).wrap());
        }
    }
    egui::CollapsingHeader::new(WHERE_FROM)
        .id_salt(("magcoupling_note_sources", note.id))
        .default_open(false)
        .show(ui, |ui| {
            for source in note.sources {
                ui.add(
                    egui::Label::new(egui::RichText::new(glyph_safe(source)).small().weak()).wrap(),
                );
            }
        });
    clicked
}

/// One change entry on one wrapped line: its step word, its input's display label (a link, or
/// plain text and the reason), then ": " and its text. True when the link was clicked.
fn change_ui(ui: &mut egui::Ui, change: &Change, view: &NoteView<'_>) -> bool {
    let label = glyph_safe(label_for(change.input));
    let mut clicked = false;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label(format!("{} ", change.step.word()));
        match unlinked_reason(change, view) {
            Some(reason) => {
                ui.label(label);
                ui.weak(format!(" ({reason})"));
            }
            None => clicked = ui.link(label).clicked(),
        }
        ui.add(egui::Label::new(format!(": {}", note_text_on(change.text, view))).wrap());
    });
    clicked
}

/// A note opened on its own: links to the equations it explains (a family's template is left
/// out), then the note. Returns the equation clicked and the note's link clicked.
fn note_alone_ui(
    ui: &mut egui::Ui,
    note: &Note,
    view: &NoteView<'_>,
) -> (Option<String>, Option<NoteLink>) {
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
    let link = note_ui(ui, note, view);
    (clicked, link)
}

/// The open equation: its label (and with `show_details` its cell), the equation large (a term
/// clicked is followed), the value, with `show_details` the corrections it embodies (their
/// titles on hover), with `explain` its teaching note (drawn against `view`), then the term
/// list, its swatches faded while `others_marked`, a term's internal name on hover with
/// `show_details`. Returns the term clicked and the note's link clicked.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    others_marked: bool,
    explain: bool,
    show_details: bool,
    view: &NoteView<'_>,
) -> (Option<String>, Option<NoteLink>) {
    let mut clicked = None;
    let mut link = None;
    if show_details {
        ui.weak(format!(
            "{} ({})",
            label_for(&eq.target),
            eq.cell
                .as_deref()
                .unwrap_or(crate::gui::plain_text::NO_SPREADSHEET_CELL)
        ));
    } else {
        ui.weak(label_for(&eq.target));
    }
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
    if show_details && !corrections.is_empty() {
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        let titles: Vec<String> = corrections.iter().map(|&id| correction_title(id)).collect();
        ui.weak(format!("{CORRECTIONS}: {}", ids.join(", ")))
            .on_hover_text(titles.join("\n"));
    }
    if explain {
        ui.separator();
        match notes::note_for(&eq.target) {
            Some(note) => link = note_ui(ui, note, view),
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
                        let term =
                            ui.add(egui::Label::new(term_label(&row.path)).sense(Sense::click()));
                        // Its internal name only with the spreadsheet details.
                        let term = if show_details {
                            term.on_hover_text(&row.path)
                        } else {
                            term
                        };
                        if term.clicked() {
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
    (clicked, link)
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
        TermKind::CellOnly => "no equation",
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

/// A result without an equation record: its label, value and, with `show_details`, its cell.
fn no_equation_body(ui: &mut egui::Ui, path: &str, results: &DesignResults, show_details: bool) {
    let label = term_label(path);
    let value = results.get(path).unwrap_or(Value::None);
    let unit = result_info(path).map_or("", |info| info.meta.unit);
    ui.strong(format!("{label} = {}", value_text(path, &value, unit)));
    if show_details {
        let cell = result_info(path)
            .and_then(|info| info.cell.clone())
            .unwrap_or_else(|| crate::gui::plain_text::NO_SPREADSHEET_CELL.to_owned());
        ui.weak(cell);
    }
    ui.weak(NO_EQUATION);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::ResultSet;
    use crate::gui::format::format_value;
    use crate::gui::results_table::table_entries;

    fn design() -> (DesignInputs, DesignResults) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        (inputs, results)
    }

    #[test]
    fn the_panel_starts_at_45_percent_and_its_drag_leaves_the_view_its_strip() {
        let heights = |centre: f32| {
            let (range, start) = panel_heights(centre);
            (range.min, range.max, start)
        };
        // A roomy region: 45 % to start, the drag stopping VIEW_STRIP below the region's top.
        assert_eq!(
            heights(600.0),
            (MIN_PANEL_HEIGHT, 600.0 - VIEW_STRIP, 270.0)
        );
        assert_eq!(
            heights(1000.0),
            (MIN_PANEL_HEIGHT, 1000.0 - VIEW_STRIP, 450.0)
        );
        assert_eq!(heights(300.0), (MIN_PANEL_HEIGHT, 140.0, 135.0));
        // Too short for the floor and the strip: the panel keeps its floor, the view the rest.
        for centre in [
            MIN_PANEL_HEIGHT + VIEW_STRIP,
            250.0,
            200.0,
            MIN_PANEL_HEIGHT,
        ] {
            let floor = MIN_PANEL_HEIGHT;
            assert_eq!(heights(centre), (floor, floor, floor), "{centre}");
        }
        // Shorter than the floor: the whole region. No region, or not a number: none.
        assert_eq!(heights(80.0), (80.0, 80.0, 80.0));
        for centre in [0.0, -5.0, f32::NAN] {
            assert_eq!(heights(centre), (0.0, 0.0, 0.0), "{centre}");
        }
        // Everywhere: a range (min <= max) holding the start; the strip kept whenever the
        // region holds it and the floor.
        for step in 0..=300 {
            let centre = step as f32 * 5.0;
            let (min, max, start) = heights(centre);
            assert!(
                min <= start && start <= max,
                "{centre}: {min} {start} {max}"
            );
            assert!(max <= centre, "{centre}");
            if centre >= MIN_PANEL_HEIGHT + VIEW_STRIP {
                assert_eq!(centre - max, VIEW_STRIP, "{centre}");
            }
        }
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
            explorer_ui(ui, explorer, &inputs, &results, false, &nothing_held);
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
                "{START_HERE_BUTTON} 1 of {}: How the coupling makes torque",
                START_HERE.len()
            )
        );
        // The button's hover names the walk's topics in plain words, step 1's first.
        assert!(crate::gui::guidance::plain_text_violations(START_HERE_HOVER).is_empty());
        assert!(START_HERE_HOVER.starts_with("A suggested order: how the coupling makes torque"));
    }

    #[test]
    fn the_ferrite_step_says_it_is_for_ferrite_rings_only() {
        // Spec 3.3 and 5 (TC-32): the ferrite step opens a check that reads n/a on NdFeB rings.
        let ferrite = START_HERE
            .iter()
            .position(|&(id, _)| id == "ferrite_cold_demag")
            .expect("the ferrite step");
        let title =
            reviewed_note("ferrite_cold_demag").map_or_else(String::new, |n| glyph_safe(n.title));
        assert_eq!(
            start_here_text(ferrite),
            format!(
                "{START_HERE_BUTTON} {} of {} {FERRITE_ONLY}: {title}",
                ferrite + 1,
                START_HERE.len()
            )
        );
        for step in (0..START_HERE.len()).filter(|&s| s != ferrite) {
            assert!(!start_here_text(step).contains(FERRITE_ONLY), "step {step}");
        }
    }

    #[test]
    fn the_start_here_hover_names_the_clamp_step_only_while_the_section_shows() {
        assert_eq!(
            start_here_hover(false),
            format!("{}, clamps", start_here_hover(true))
        );
        assert!(!start_here_hover(true).to_lowercase().contains("clamp"));
        assert_eq!(START_HERE_HOVER, start_here_hover(clamps::PAUSED));
        for paused in [false, true] {
            assert!(
                crate::gui::guidance::plain_text_violations(start_here_hover(paused)).is_empty()
            );
        }
    }

    #[test]
    fn no_equation_shown_leads_into_the_paused_clamp_section() {
        // Ruling 11: no route into the section while it is paused. A shown equation reads none
        // of its terms (a click on a term drills only into what it reads), and its Used by
        // leaves the section's results out.
        for eq in registry().equations() {
            if clamps::hides(&eq.target, true) {
                continue;
            }
            for term in &eq.terms {
                assert!(!clamps::hides(term, true), "{} reads {term}", eq.target);
            }
            for user in used_by_shown(&eq.target, true) {
                assert!(!clamps::hides(user, true), "{} -> {user}", eq.target);
            }
        }
        // The cold-high torque is the clamp's demand: its Used by shows the clamp only while
        // the section shows.
        let users = |paused| -> Vec<&str> {
            used_by_shown("metal.torque_cold_high_Nm", paused)
                .into_iter()
                .map(String::as_str)
                .collect()
        };
        assert!(users(false).contains(&"clamps.max_torque_Nm"));
        assert!(!users(true).contains(&"clamps.max_torque_Nm"));
        assert_eq!(
            users(false).len(),
            registry().used_by("metal.torque_cold_high_Nm").len()
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
        // An input or result the overlay renames reads by its display label (gui::plain_text).
        for path in ["metal.face_gap_mm", "model.pullout_Nm"] {
            assert_eq!(term_label(path), crate::gui::plain_text::label_for(path));
        }
        assert_eq!(
            term_label("metal.face_gap_mm"),
            crate::gui::size_controls::GAP
        );
        assert_eq!(plain_symbol("ϑ_{op}"), "θ_op");
        assert_eq!(plain_symbol("T_{pull}"), "T_pull");
        assert_eq!(plain_symbol("S_{3}^{iron}"), "S_3^iron");
        // A selector code with its choice's label, after a colon: a label holding its own
        // parentheses (the default harmonic set) reads without nesting them.
        assert_eq!(
            value_text("coupling.backiron", &Value::Int(1), "-"),
            "1: steel circuit"
        );
        assert_eq!(
            value_text("coupling.max_harmonic", &Value::Int(5), "-"),
            "5: 1, 3, 5 (default)"
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

    /// Frames of `explorer_ui` on the default design at `show_details` with the first drawn
    /// `text` hovered ([`hover_frames`](crate::gui::test_support::hover_frames)): one before the
    /// pointer is there, and one that shows its tooltip.
    fn hover_in_panel(
        explorer: &mut Explorer,
        show_details: bool,
        text: &str,
    ) -> (egui::FullOutput, egui::FullOutput) {
        use crate::gui::test_support::{SCREEN, hover_frames, text_center};
        let (inputs, results) = design();
        hover_frames(
            &egui::Context::default(),
            SCREEN,
            |output| text_center(output, text),
            |ui| explorer_ui(ui, explorer, &inputs, &results, show_details, &nothing_held),
        )
    }

    #[test]
    fn the_cell_and_corrections_lines_show_only_with_the_details_and_name_each_title_on_hover() {
        use crate::gui::corrections::correction_title;
        use crate::gui::test_support::drawn_texts;
        let path = "model.pullout_Nm";
        let eq = registry()
            .equation_for(path)
            .expect("the pull-out's equation");
        let cell = eq.cell.clone().expect("the pull-out has a cell");
        let corrections = registry().corrections_upstream(path);
        assert!(!corrections.is_empty());
        let ids: Vec<String> = corrections.iter().map(ToString::to_string).collect();
        let line = format!("{CORRECTIONS}: {}", ids.join(", "));
        let titles: Vec<String> = corrections.iter().map(|&id| correction_title(id)).collect();
        let titles = titles.join("\n");
        let label = crate::gui::plain_text::label_for(path);
        let with_cell = format!("{label} ({cell})");
        for show_details in [false, true] {
            let mut explorer = Explorer::default();
            explorer.open_path(path);
            // The label line is drawn either way; hovered, the corrections line names each
            // correction's title.
            let hovered = if show_details { line.as_str() } else { label };
            let (before, output) = hover_in_panel(&mut explorer, show_details, hovered);
            let texts = drawn_texts(&before);
            assert_eq!(texts.contains(&line), show_details);
            assert_eq!(texts.contains(&with_cell), show_details);
            assert_eq!(texts.contains(&label.to_owned()), !show_details);
            assert_eq!(drawn_texts(&output).contains(&titles), show_details);
        }
    }

    #[test]
    fn the_equation_panel_s_value_line_names_the_cup_wall_input_by_its_label() {
        // Ruling R9 (amendment A-9): the value under the cup wall's equation reads as the
        // dashboard draws it; a passing wall reads as the engine gives it.
        use crate::gui::dashboard::verdict_display;
        use crate::gui::test_support::{SCREEN, drawn_texts, sized_frame};
        let path = "materials.cup_wall_check";
        let symbol = plain_symbol(
            &registry()
                .equation_for(path)
                .expect("the cup wall's equation")
                .symbol,
        );
        let value_lines = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            let mut explorer = Explorer::default();
            explorer.open_path(path);
            let output = sized_frame(&egui::Context::default(), SCREEN, Vec::new(), |ui| {
                explorer_ui(ui, &mut explorer, inputs, &results, false, &nothing_held);
            });
            let engine = results.materials.cup_wall_check.clone();
            let lines: Vec<String> = drawn_texts(&output)
                .into_iter()
                .filter(|t| t.starts_with(&format!("{symbol} = ")))
                .collect();
            (engine, lines)
        };
        let (engine, lines) = value_lines(&DesignInputs::default());
        let shown = verdict_display(path, &engine).expect("the default cup wall fails");
        assert_eq!(lines, [format!("{symbol} = {shown}")]);
        let (engine, lines) = value_lines(&crate::gui::test_support::passing_design());
        assert_eq!(engine, "OK");
        assert_eq!(lines, [format!("{symbol} = OK")]);
    }

    #[test]
    fn a_term_s_internal_name_shows_on_hover_only_with_the_details() {
        use crate::gui::test_support::drawn_texts;
        // The term list of the pull-out's equation: the end-effect factor's label, hovered.
        let label = term_label("model.f_end");
        for show_details in [false, true] {
            let mut explorer = Explorer::default();
            explorer.open_path("model.pullout_Nm");
            let (_, output) = hover_in_panel(&mut explorer, show_details, label);
            let shown = drawn_texts(&output).iter().any(|t| t == "model.f_end");
            assert_eq!(shown, show_details);
        }
    }

    #[test]
    fn a_check_guide_link_highlights_its_input_s_row_and_leaves_the_panel_as_it_was() {
        let ctx = egui::Context::default();
        let mut explorer = Explorer::default();
        explorer.open_path("model.pullout_Nm");
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                focus: Some("metal.cup_wall_corner_mm"),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(
            explorer.focus(),
            Some(&Focus {
                path: "metal.cup_wall_corner_mm".to_owned(),
                scroll: true,
            })
        );
        assert_eq!(
            explorer.trail(),
            ["model.pullout_Nm"],
            "the equation stays open"
        );
        assert!(explorer.open);
        // With the panel shut, it stays shut.
        let mut explorer = Explorer::default();
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                focus: Some("metal.variation"),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(
            explorer.focus().map(|f| f.path.as_str()),
            Some("metal.variation")
        );
        assert!(!explorer.open);
    }

    /// The view of a design on which no input is held.
    fn nothing_held(_: &str) -> Option<String> {
        None
    }

    /// A note with a change entry of each kind, for the renderer's own tests (no shipped note
    /// has change entries before the note tasks rewrite them).
    const RENDER_NOTE: Note = Note {
        id: "test.render",
        title: "A note to draw",
        equations: &[],
        sentences: &["The first sentence of the note.", "The second sentence."],
        change: &[
            Change {
                input: "metal.face_gap_mm",
                step: notes::Step::Lower,
                aim: Some(("model.pullout_Nm", notes::Sense::Up)),
                text: "the entry on the face gap.",
            },
            Change {
                input: "coupling.op_temp_C",
                step: notes::Step::Raise,
                aim: Some(("model.pullout_Nm", notes::Sense::Down)),
                text: "the entry on the operating temperature.",
            },
            Change {
                input: "materials.parts.back_iron",
                step: notes::Step::Choose,
                aim: None,
                text: "the entry on the back iron.",
            },
        ],
        watch_out: &["The first watch-out.", "The second watch-out."],
        diagram: None,
        sources: &["The first source.", "The second source."],
        review: Review::Draft,
    };

    /// One frame of `note` drawn against `view` with `events`: what egui painted and the link a
    /// click asked for.
    fn note_frame(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Option<NoteLink>) {
        use crate::gui::test_support::{SCREEN, sized_frame};
        let mut link = None;
        let output = sized_frame(ctx, SCREEN, events, |ui| {
            if let Some(clicked) = note_ui(ui, note, view) {
                link = Some(clicked);
            }
        });
        (output, link)
    }

    /// Clicks the first drawn text equal to `text` (move, press, release): the link the click
    /// asked for, and a frame drawn once a fold the click opened has settled.
    fn click_in_note(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        text: &str,
    ) -> (Option<NoteLink>, egui::FullOutput) {
        use crate::gui::test_support::{primary_button, text_rect};
        let (output, _) = note_frame(ctx, note, view, Vec::new());
        // A text in a wrapped row is laid out from the row's start, so the left part of its rect
        // covers the widgets before it on that row: click its last character.
        let rect = text_rect(&output, text).unwrap_or_else(|| panic!("no text {text:?}"));
        let at = egui::pos2(rect.max.x - 2.0, rect.center().y);
        note_frame(ctx, note, view, vec![egui::Event::PointerMoved(at)]);
        note_frame(ctx, note, view, vec![primary_button(at, true)]);
        let (_, link) = note_frame(ctx, note, view, vec![primary_button(at, false)]);
        let mut output = note_frame(ctx, note, view, Vec::new()).0;
        for _ in 0..15 {
            output = note_frame(ctx, note, view, Vec::new()).0;
        }
        (link, output)
    }

    #[test]
    fn a_note_draws_its_parts_in_order_with_its_sources_folded_until_clicked() {
        use crate::gui::test_support::{drawn_texts, text_color, text_rect};
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let (output, _) = note_frame(&ctx, &RENDER_NOTE, &view, Vec::new());
        let texts = drawn_texts(&output);
        let count = |text: &str| texts.iter().filter(|t| *t == text).count();
        let once = [RENDER_NOTE.title, WHAT_TO_CHANGE, WATCH_OUT, WHERE_FROM]
            .into_iter()
            .chain(RENDER_NOTE.sentences.iter().copied())
            .chain(RENDER_NOTE.watch_out.iter().copied());
        for text in once {
            assert_eq!(count(text), 1, "{text}");
        }
        for change in RENDER_NOTE.change {
            assert_eq!(
                count(&format!("{} ", change.step.word())),
                1,
                "{}",
                change.input
            );
            assert_eq!(count(label_for(change.input)), 1, "{}", change.input);
            assert_eq!(count(&format!(": {}", change.text)), 1, "{}", change.input);
        }
        for source in RENDER_NOTE.sources {
            assert_eq!(count(source), 0, "the fold starts closed: {source}");
        }
        // Top to bottom: what it is, what to change, watch out, where this comes from.
        let top = |text: &str| text_rect(&output, text).unwrap().min.y;
        let order = [
            RENDER_NOTE.sentences[1],
            WHAT_TO_CHANGE,
            label_for(RENDER_NOTE.change[2].input),
            WATCH_OUT,
            RENDER_NOTE.watch_out[1],
            WHERE_FROM,
        ];
        for pair in order.windows(2) {
            assert!(top(pair[0]) < top(pair[1]), "{pair:?}");
        }
        let warn = ctx.style().visuals.warn_fg_color;
        for line in RENDER_NOTE.watch_out {
            assert_eq!(text_color(&output, line), Some(warn), "{line}");
        }
        let (link, output) = click_in_note(&ctx, &RENDER_NOTE, &view, WHERE_FROM);
        assert_eq!(link, None, "the fold is no link to an input");
        let texts = drawn_texts(&output);
        for source in RENDER_NOTE.sources {
            assert!(texts.iter().any(|t| t == source), "{source}");
        }
    }

    #[test]
    fn a_free_input_s_label_is_a_link_to_its_row_and_the_text_is_not() {
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        for change in RENDER_NOTE.change {
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, label_for(change.input));
            assert_eq!(
                link,
                Some(NoteLink::Input(change.input)),
                "{}",
                change.input
            );
            let text = format!(": {}", change.text);
            let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, &text);
            assert_eq!(link, None, "{text}");
        }
    }

    #[test]
    fn an_input_that_is_not_free_is_plain_text_with_the_reason() {
        use crate::gui::test_support::drawn_texts;
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let held_entry = &RENDER_NOTE.change[0];
        let held = |path: &str| (path == held_entry.input).then(|| HELD_BY_AUTO.to_owned());
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let (link, output) = click_in_note(&ctx, &RENDER_NOTE, &view, label_for(held_entry.input));
        assert_eq!(link, None, "a held input is no link");
        let reason = format!(" ({HELD_BY_AUTO})");
        assert_eq!(
            drawn_texts(&output)
                .iter()
                .filter(|t| **t == reason)
                .count(),
            1
        );
        let free = &RENDER_NOTE.change[1];
        let (link, _) = click_in_note(&ctx, &RENDER_NOTE, &view, label_for(free.input));
        assert_eq!(
            link,
            Some(NoteLink::Input(free.input)),
            "the others stay links"
        );
    }

    #[test]
    fn an_entry_whose_input_cannot_move_its_way_names_why() {
        use crate::engine::explain::notes::Step;
        let range = |path: &str| {
            InputCatalogue::get()
                .entry(path)
                .unwrap()
                .meta
                .range
                .unwrap()
        };
        let entry = |input, step| Change {
            input,
            step,
            aim: None,
            text: "",
        };
        let reason = |inputs: &DesignInputs, change: &Change| {
            let results = compute_all(inputs);
            let view = NoteView {
                inputs,
                results: &results,
                held: &nothing_held,
            };
            unlinked_reason(change, &view)
        };
        let gap = "metal.face_gap_mm";
        let (lowest, highest) = (range(gap).min, range(gap).max);
        let mut inputs = DesignInputs::default();
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(highest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Lower)), None);
        inputs.set(gap, Value::Num(lowest)).unwrap();
        assert_eq!(
            reason(&inputs, &entry(gap, Step::Lower)).as_deref(),
            Some(AT_BOTTOM)
        );
        assert_eq!(reason(&inputs, &entry(gap, Step::Raise)), None);
        // A count at its end; a choice and a blank optional number are never at an end.
        let mut poles = DesignInputs::default();
        poles.coupling.npole = range("coupling.npole").max as i64;
        assert_eq!(
            reason(&poles, &entry("coupling.npole", Step::Raise)).as_deref(),
            Some(AT_TOP)
        );
        let defaults = DesignInputs::default();
        assert_eq!(
            reason(&defaults, &entry("materials.parts.back_iron", Step::Choose)),
            None
        );
        assert_eq!(
            reason(
                &defaults,
                &entry("coupling.magnets.axial_length_mm", Step::Raise)
            ),
            None
        );
        // A held input names why it is held, whatever its slider says.
        let held = |_: &str| Some(HELD_BY_AUTO.to_owned());
        let results = compute_all(&inputs);
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        assert_eq!(
            unlinked_reason(&entry(gap, Step::Lower), &view).as_deref(),
            Some(HELD_BY_AUTO)
        );
    }

    #[test]
    fn held_reason_names_auto_and_the_pick_that_sets_an_input() {
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::picks::BACK_IRON_PICK;
        use crate::engine::sizing::FreeVariable;
        use crate::gui::sizing::SizingMode;
        let inputs = DesignInputs::default();
        let off = SizingState::default();
        let auto = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
        };
        let length = FreeVariable::AxialLength.path();
        assert_eq!(held_reason(&off, &inputs, length), None);
        assert_eq!(
            held_reason(&auto, &inputs, length).as_deref(),
            Some(HELD_BY_AUTO)
        );
        assert_eq!(held_reason(&auto, &inputs, "metal.face_gap_mm"), None);
        let part_pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_inner")
        );
        assert_eq!(
            held_reason(&off, &inputs, "coupling.magnets.manual_inner_br_T"),
            Some(part_pick.clone())
        );
        let steel = "materials.steel.cte_per_C";
        assert_eq!(held_reason(&off, &inputs, steel), None);
        let mut picked = inputs.clone();
        picked.materials.parts.back_iron = 2;
        let steel_pick = format!("set by the {} pick", label_for(BACK_IRON_PICK));
        assert_eq!(held_reason(&off, &picked, steel), Some(steel_pick.clone()));
        for reason in [
            HELD_BY_AUTO.to_owned(),
            part_pick,
            steel_pick,
            AT_TOP.into(),
            AT_BOTTOM.into(),
        ] {
            assert_eq!(
                plain_text_violations(&reason),
                Vec::<&str>::new(),
                "{reason}"
            );
        }
    }

    #[test]
    fn the_outer_block_entries_name_the_part_pick_that_sets_them() {
        // Review Focus RF-4 (amendment A-25 T-4, the harmonics part): at the defaults the outer
        // library part sets the block's width and thickness, so the harmonics and back-iron
        // entries on them are plain text naming the pick; with custom dimensions (the Y30 probe
        // design, both parts blank) the same entries are links.
        use crate::engine::explain::probes::probe_designs;
        let entry = |id: &str, input: &str| {
            *notes::note(id)
                .and_then(|n| n.change.iter().find(|c| c.input == input))
                .unwrap_or_else(|| panic!("{id}: no entry on {input}"))
        };
        let width = entry("harmonics", "coupling.magnets.manual_outer_width_mm");
        let thickness = entry(
            "back_iron_factor",
            "coupling.magnets.manual_outer_thickness_mm",
        );
        let reason = |inputs: &DesignInputs, change: &Change| {
            let results = compute_all(inputs);
            let held = |path: &str| held_reason(&SizingState::default(), inputs, path);
            let view = NoteView {
                inputs,
                results: &results,
                held: &held,
            };
            unlinked_reason(change, &view)
        };
        let pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_outer")
        );
        let defaults = DesignInputs::default();
        assert_eq!(reason(&defaults, &width), Some(pick.clone()));
        assert_eq!(reason(&defaults, &thickness), Some(pick));
        let (_, custom) = probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the custom-dimensions probe design");
        assert_eq!(reason(&custom, &width), None);
        assert_eq!(reason(&custom, &thickness), None);
    }

    #[test]
    fn the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it() {
        // Review Focus RF-4 (amendment A-25 T-4): 1018 brings its own design flux density, so the
        // wall note's entry on that input is plain text naming the back-iron pick; 12L14 has none
        // of its own and leaves the input in effect, so the same entry is a link.
        use crate::engine::material_library::BACK_IRON_CHOICES;
        use crate::engine::picks::BACK_IRON_PICK;
        assert_eq!(BACK_IRON_CHOICES[1], (2, "1018_hot_rolled"));
        assert_eq!(BACK_IRON_CHOICES[2], (3, "12L14_cold_drawn"));
        let entry = *notes::note("a5.low_saturation")
            .and_then(|n| {
                n.change
                    .iter()
                    .find(|c| c.input == "materials.steel.bsat_T")
            })
            .expect("the wall note has an entry on the design flux density");
        let reason = |back_iron: i64| {
            let mut inputs = DesignInputs::default();
            inputs.materials.parts.back_iron = back_iron;
            let results = compute_all(&inputs);
            let held = |path: &str| held_reason(&SizingState::default(), &inputs, path);
            let view = NoteView {
                inputs: &inputs,
                results: &results,
                held: &held,
            };
            unlinked_reason(&entry, &view)
        };
        let set = reason(2).expect("1018 sets the design flux density");
        assert!(set.starts_with("set by the"), "{set}");
        assert_eq!(
            set,
            format!("set by the {} pick", label_for(BACK_IRON_PICK))
        );
        assert_eq!(reason(3), None);
    }

    #[test]
    fn the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason() {
        // Amendment A-25 T-7: at the sleeve's slider minimum one step thinner is past the slider,
        // so the sleeve note's entry draws its label as plain text with the reason and no link;
        // the liner entry beside it stays a link.
        use crate::gui::test_support::drawn_texts;
        let sleeve = "metal.sleeve_mm";
        let note = notes::note("a5.high_conductivity_sleeve_or_liner").unwrap();
        let entry = note
            .change
            .iter()
            .find(|c| c.input == sleeve)
            .expect("the sleeve note has an entry on the sleeve thickness");
        assert_eq!(entry.step, notes::Step::Lower);
        let mut inputs = DesignInputs::default();
        let lowest = InputCatalogue::get()
            .entry(sleeve)
            .unwrap()
            .meta
            .range
            .unwrap()
            .min;
        inputs.set(sleeve, Value::Num(lowest)).unwrap();
        let results = compute_all(&inputs);
        let held = |path: &str| held_reason(&SizingState::default(), &inputs, path);
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let ctx = egui::Context::default();
        let (link, output) = click_in_note(&ctx, note, &view, label_for(sleeve));
        assert_eq!(link, None, "an input at its slider's end is no link");
        let reason = format!(" ({AT_BOTTOM})");
        assert_eq!(
            drawn_texts(&output)
                .iter()
                .filter(|t| **t == reason)
                .count(),
            1
        );
        let liner = "metal.liner_mm";
        let (link, _) = click_in_note(&ctx, note, &view, label_for(liner));
        assert_eq!(link, Some(NoteLink::Input(liner)), "the liner stays a link");
    }

    #[test]
    fn a_live_value_reads_as_the_readouts_and_none_where_there_is_no_prediction() {
        use crate::engine::explain::probes::probe_designs;
        let designs = probe_designs();
        let default = compute_all(&designs[0].1);
        let pullout = "model.pullout_Nm";
        let unit = result_info(pullout).unwrap().meta.unit;
        assert_eq!(
            live_value(&default, pullout),
            Some(with_unit(
                format_value(&default.get(pullout).unwrap()),
                unit
            ))
        );
        // A text where a number belongs: the cold limit of NdFeB rings reads "n/a".
        let cold = "temperature.demag.cold_limit_C";
        assert!(matches!(default.get(cold), Some(Value::Text(_))));
        assert_eq!(live_value(&default, cold), None);
        // On the ferrite probe the cold limit is a number and the hot onsets are not finite.
        let ferrite = compute_all(&designs[3].1);
        let cold_unit = result_info(cold).unwrap().meta.unit;
        assert_eq!(
            live_value(&ferrite, cold),
            Some(with_unit(
                format_value(&ferrite.get(cold).unwrap()),
                cold_unit
            ))
        );
        let onset = "temperature.demag.onset_skipping_C";
        assert!(matches!(ferrite.get(onset), Some(Value::Num(x)) if !x.is_finite()));
        assert_eq!(live_value(&ferrite, onset), None);
        // A verdict, an input and a path no result has: none.
        assert_eq!(live_value(&default, "materials.cup_wall_check"), None);
        assert_eq!(live_value(&default, "metal.face_gap_mm"), None);
        assert_eq!(live_value(&default, "no.such.result"), None);
    }

    #[test]
    fn a_value_computed_from_an_out_of_range_pull_out_reads_no_prediction() {
        let results = compute_all(&crate::gui::test_support::short_magnets());
        assert!(crate::gui::dashboard::end_effect_out_of_range(&results).is_some());
        for path in [
            "model.pullout_Nm",
            "metal.torque_hot_low_Nm",
            "metal.cold_high_input_Nm",
            "model.gearbox_input_ripple_Nm",
        ] {
            assert_eq!(live_value(&results, path), None, "{path}");
        }
        for path in [
            "model.f_end",
            "model.pole_pitch_mm",
            "model.active_length_mm",
        ] {
            assert!(live_value(&results, path).is_some(), "{path}");
        }
    }

    /// A note with live values in a sentence, a change entry and a watch-out.
    const LIVE_NOTE: Note = Note {
        id: "test.live",
        title: "A note with live values",
        equations: &[],
        sentences: &[
            "The first sentence has none.",
            "Here the pull-out is {model.pullout_Nm}.",
        ],
        change: &[Change {
            input: "metal.face_gap_mm",
            step: notes::Step::Lower,
            aim: Some(("model.pullout_Nm", notes::Sense::Up)),
            text: "the skipping onset is {temperature.demag.onset_skipping_C} here.",
        }],
        watch_out: &["The end factor is {model.f_end} on this design."],
        diagram: None,
        sources: &["A source."],
        review: Review::Draft,
    };

    #[test]
    fn a_note_draws_its_live_values_from_the_design_shown() {
        use crate::engine::explain::note_text::NO_VALUE;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::drawn_texts;
        let ctx = egui::Context::default();
        let designs = probe_designs();
        for (name, inputs) in [&designs[0], &designs[3]] {
            let results = compute_all(inputs);
            let view = NoteView {
                inputs,
                results: &results,
                held: &nothing_held,
            };
            note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let (output, _) = note_frame(&ctx, &LIVE_NOTE, &view, Vec::new());
            let texts = drawn_texts(&output);
            let value = |path: &str| live_value(&results, path).unwrap_or(NO_VALUE.to_owned());
            for want in [
                LIVE_NOTE.sentences[0].to_owned(),
                format!("Here the pull-out is {}.", value("model.pullout_Nm")),
                format!(
                    ": the skipping onset is {} here.",
                    value("temperature.demag.onset_skipping_C")
                ),
                format!("The end factor is {} on this design.", value("model.f_end")),
            ] {
                assert!(texts.contains(&want), "{name}: {want:?} in {texts:?}");
            }
        }
        // The ferrite rings' hot onset has no prediction; the NdFeB default's has one.
        let onset = "temperature.demag.onset_skipping_C";
        assert!(live_value(&compute_all(&designs[0].1), onset).is_some());
        assert_eq!(live_value(&compute_all(&designs[3].1), onset), None);
    }

    #[test]
    fn every_note_draws_at_every_probe_design_with_glyphs_and_no_braces() {
        use crate::engine::explain::note_text::rendered_texts;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::assert_glyphs;
        let ctx = egui::Context::default();
        // The fonts load on the context's first frame.
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        for (name, inputs) in probe_designs() {
            let results = compute_all(&inputs);
            let view = NoteView {
                inputs: &inputs,
                results: &results,
                held: &nothing_held,
            };
            for note in notes::NOTES {
                for text in rendered_texts(note) {
                    let drawn = note_text_on(text, &view);
                    assert!(
                        !drawn.contains(['{', '}']),
                        "{} at {name}: {drawn}",
                        note.id
                    );
                    assert_glyphs(&ctx, &drawn, note.id);
                }
            }
        }
    }

    /// The per-note physics review's sheet (slice-2 spec section 4): each note, or each named
    /// in `NOTES_FOR_REVIEW` (ids, comma-separated), its title and sources once, then at each
    /// probe design its sentences, change entries and watch-outs as the panel draws them; a
    /// change entry with its input's display label, the reason it is not a link there, and
    /// its aim's value before and after one step. A text the parser refuses or the plain-words
    /// guard flags is marked "<<<". Written to `NOTES_REVIEW_OUT` (a path from `magcoupling-rs/`;
    /// default `target/notes-for-review.txt` there).
    #[test]
    #[ignore = "a review tool, not a check: NOTES_FOR_REVIEW=id1,id2 NOTES_REVIEW_OUT=<file> cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture"]
    fn notes_for_review() {
        use std::fmt::Write as _;

        use crate::engine::explain::note_text::segments;
        use crate::engine::explain::plain::plain_text_violations;
        use crate::engine::explain::probes::{probe_designs, step_input};
        let wanted: Vec<String> = std::env::var("NOTES_FOR_REVIEW")
            .map(|ids| {
                ids.split(',')
                    .map(|id| id.trim().to_owned())
                    .filter(|id| !id.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        let unknown: Vec<&String> = wanted
            .iter()
            .filter(|id| notes::note(id).is_none())
            .collect();
        assert!(
            unknown.is_empty(),
            "NOTES_FOR_REVIEW names no note: {unknown:?}"
        );
        let chosen: Vec<&Note> = notes::NOTES
            .iter()
            .filter(|n| wanted.is_empty() || wanted.iter().any(|id| id == n.id))
            .collect();
        let designs: Vec<(&str, DesignInputs, DesignResults)> = probe_designs()
            .into_iter()
            .map(|(name, inputs)| {
                let results = compute_all(&inputs);
                (name, inputs, results)
            })
            .collect();
        // A raw text the parser refuses, or a drawn text the guard flags, is marked.
        let marks = |raw: &str, drawn: &str| {
            let mut found = Vec::new();
            if let Err(fault) = segments(raw) {
                found.push(fault);
            }
            let banned = plain_text_violations(drawn);
            if !banned.is_empty() {
                found.push(format!("plain words: {banned:?}"));
            }
            if found.is_empty() {
                String::new()
            } else {
                format!("   <<< {}", found.join("; "))
            }
        };
        let sizing = SizingState::default();
        let mut sheet = String::new();
        for note in &chosen {
            let status = match note.review {
                Review::Draft => "Draft",
                Review::Reviewed { .. } => "Reviewed",
            };
            writeln!(sheet, "==== {} ({status})", note.id).unwrap();
            writeln!(
                sheet,
                "Title: {}{}",
                note.title,
                marks(note.title, note.title)
            )
            .unwrap();
            writeln!(sheet, "{WHERE_FROM}:").unwrap();
            for source in note.sources {
                writeln!(sheet, "  - {source}{}", marks(source, source)).unwrap();
            }
            for (name, inputs, results) in &designs {
                let held = |path: &str| held_reason(&sizing, inputs, path);
                let view = NoteView {
                    inputs,
                    results,
                    held: &held,
                };
                writeln!(sheet, "\n-- at the {name} design").unwrap();
                writeln!(sheet, "What it is:").unwrap();
                for (i, sentence) in note.sentences.iter().enumerate() {
                    let drawn = note_text_on(sentence, &view);
                    writeln!(sheet, "  {}. {drawn}{}", i + 1, marks(sentence, &drawn)).unwrap();
                }
                if !note.change.is_empty() {
                    writeln!(sheet, "{WHAT_TO_CHANGE}:").unwrap();
                }
                for change in note.change {
                    let reason =
                        unlinked_reason(change, &view).map_or(String::new(), |r| format!(" ({r})"));
                    let drawn = note_text_on(change.text, &view);
                    writeln!(
                        sheet,
                        "  - {} {}{reason}: {drawn}{}",
                        change.step.word(),
                        label_for(change.input),
                        marks(change.text, &drawn)
                    )
                    .unwrap();
                    let Some((aim, sense)) = change.aim else {
                        continue;
                    };
                    let outcome = match step_input(inputs, change.input, change.step) {
                        Err(why) => format!("not stepped here: {why:?}"),
                        Ok(stepped) => match (results.get(aim), compute_all(&stepped).get(aim)) {
                            (Some(Value::Num(before)), Some(Value::Num(after)))
                                if before.is_finite() && after.is_finite() =>
                            {
                                let verdict = if sense.holds(before, after) {
                                    "moves as it says"
                                } else {
                                    "DOES NOT MOVE AS IT SAYS"
                                };
                                format!("{before} -> {after}: {verdict}")
                            }
                            (before, after) => {
                                format!("no numbers to compare: {before:?} -> {after:?}")
                            }
                        },
                    };
                    writeln!(sheet, "      aim {aim} {sense:?}, one step: {outcome}").unwrap();
                }
                if !note.watch_out.is_empty() {
                    writeln!(sheet, "{WATCH_OUT}:").unwrap();
                }
                for line in note.watch_out {
                    let drawn = note_text_on(line, &view);
                    writeln!(sheet, "  - {drawn}{}", marks(line, &drawn)).unwrap();
                }
            }
            writeln!(sheet).unwrap();
        }
        let out = std::env::var("NOTES_REVIEW_OUT").map_or_else(
            |_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/notes-for-review.txt")
            },
            std::path::PathBuf::from,
        );
        if let Some(dir) = out.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).expect("the sheet's folder is made");
        }
        std::fs::write(&out, &sheet).expect("the sheet is written");
        println!(
            "{} notes at {} designs: {}",
            chosen.len(),
            designs.len(),
            out.display()
        );
    }

    #[test]
    fn a_note_s_link_highlights_its_input_s_row_and_keeps_the_note_open() {
        let mut explorer = Explorer::default();
        explorer.open_note("a5.low_saturation");
        explorer.follow_note_link(NoteLink::Input("metal.cup_wall_corner_mm"));
        assert_eq!(
            explorer.focus(),
            Some(&Focus {
                path: "metal.cup_wall_corner_mm".to_owned(),
                scroll: true,
            })
        );
        assert_eq!(explorer.note(), Some("a5.low_saturation"));
        assert!(explorer.open);
    }
}
