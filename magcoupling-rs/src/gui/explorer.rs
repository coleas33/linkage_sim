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
//! bottom of the centre region, closed until a value is clicked or an Explain entry's Equation
//! link is followed (decisions M43-1, M43-2; UX overhaul decision C-11 took its button off the
//! top bar).
//!
//! The terms the panel or the hovered readout shows are marked on screen in their colours
//! ([`Explorer::marks`]): the hovered readout's equation wins, else the open one's. While the
//! marks are another equation's, the term list's swatches fade ([`SWATCH_DIM`]). A long trail
//! shows its last [`SHOWN_CRUMBS`] crumbs after one [`ELIDED`].
//!
//! Teaching notes (spec Addendum A4; UX overhaul decision C-11): the Explain panel draws them
//! ([`crate::gui::notes_view`]), not this panel. Under the open equation the lesson line names
//! the reviewed note explaining it ([`notes::note_for`]: only notes a physics reviewer has
//! checked) after "Full lesson:", a link that shows the note in the Explain panel
//! ([`lesson_ui`]). Under an equation without one it offers the next best read
//! ([`note_fallback`], slice-2 spec 3.7): the closest reviewed note upstream as a link, an
//! assumption's rationale, or why there is none and where the notes are; never "No teaching
//! note" on a dashboard row. "Start here" walks the suggested order ([`notes::START_HERE`]):
//! each step opens its equation here and its note in the Explain panel (UX overhaul spec 5.5).
//! Every note named or drawn by id goes through [`reviewed_note`], so a note sent back to
//! Draft never shows.
//!
//! Assumptions (spec Addendum A3 "Traceability"): an assumption term is tagged as one in the
//! term list, with the changed-from-default dot when it differs from its workbook default; a
//! result some modified assumption flows into is tagged too, and the open equation says which
//! modified assumptions it depends on ([`Registry::term_style`],
//! [`Registry::modified_assumptions_upstream`]).
//!
//! [`Registry::term_style`]: crate::engine::explain::Registry::term_style
//! [`Registry::modified_assumptions_upstream`]: crate::engine::explain::Registry::modified_assumptions_upstream

use std::collections::{BTreeSet, VecDeque};

use egui::{Color32, Sense, Vec2};

use crate::engine::assumptions::{ASSUMPTIONS, Assumption};
use crate::engine::clamps;
use crate::engine::explain::markup::{Expr, Symbol};
use crate::engine::explain::notes::{self, Note, Review, START_HERE, covers};
use crate::engine::explain::{Design, Equation, TermKind, TermStyle};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::gui::corrections::correction_title;
use crate::gui::dashboard::{result_info, shown_text};
use crate::gui::explain::FULL_LESSON;
use crate::gui::format::value_with_unit;
use crate::gui::guidance::{Learn, result_unit, row_learn};
use crate::gui::input_ui::CHANGED_DOT;
use crate::gui::notes_view::NoteLink;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::{ReadoutEvents, registry};
use crate::gui::typeset::{
    TermColors, glyph_safe, laid_ui, layout_equation, layout_symbol, term_at,
};
use crate::{DesignInputs, DesignResults};

/// The panel's heading.
pub const EQUATION_PANEL: &str = "Equation panel";

/// The panel's close button.
pub const CLOSE: &str = "Close";

/// What the panel says before anything is opened (spec 3.7, TC-29b): how to open an equation,
/// and the walk through the teaching notes, by its button's name (a test checks). Since the
/// panel's toggle went (part 2a), every way the app opens the panel sets a trail first, so this
/// is only a fallback for a panel opened with none; BL-089 decides whether it goes.
pub const EMPTY_TEXT: &str = "Hover a value to see its equation; the Equation link in its entry opens it here, then click a term to follow it. Start here walks the main teaching notes in a suggested order.";

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

/// The button that starts the suggested reading order.
pub const START_HERE_BUTTON: &str = "Start here";

/// The Start here walk's topics before its last, the clamp note's: typed once, for
/// [`start_here_hover`]'s two forms.
macro_rules! start_here_topics {
    () => {
        "A suggested order: how the coupling makes torque, the magnet pattern, back iron, pull-out, magnet ends, calibration, temperature, slipping and its heat, demagnetization, shaft joints"
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

/// The start-here steps' buttons.
pub const PREVIOUS: &str = "Previous";
pub const NEXT: &str = "Next";
pub const STOP: &str = "Stop";

/// What Explain shows when nothing better is known ([`note_fallback`]): under an equation, no
/// reviewed note is attached to it or to any equation upstream, and no assumption is upstream;
/// under a value with no equation record, nothing is known of what it is computed from.
/// Never for a dashboard row (spec section 6 test 11).
pub const NO_NOTE: &str = "No teaching note is shown here for this value.";

/// The line under a dashboard row's reason or [`NO_NOTE`]: where the notes are (TC-29a).
pub const START_HERE_HINT: &str = "Start here walks the main teaching notes in a suggested order.";

/// The start of Explain's line naming the closest note to an equation that has none (TC-29a).
pub const NEAREST_NOTE: &str = "Closest teaching note:";

/// The start of Explain's line naming the assumption behind an equation that has no note
/// (TC-30); the assumption's rationale follows it.
pub const NEAREST_ASSUMPTION: &str = "Assumption behind this value:";

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

/// The start of the start-here bar: "Start here 13 of 14 (ferrite rings only): <title>" on the
/// ferrite step while the shaft-clamp section is paused (13 of 15 with it), the step and its
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

/// What Explain offers under an equation no reviewed note explains (spec 3.7, TC-29a, TC-30).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteFallback {
    /// A reviewed note to read instead.
    Note(&'static Note),
    /// An assumption whose rationale explains the value.
    Assumption(&'static Assumption),
    /// Neither: why (a dashboard row's own reason, or [`NO_NOTE`]).
    Reason(&'static str),
}

/// [`NoteFallback`] for `path`, in this order: the dashboard row's Why link, when it leads to a
/// reviewed note or an assumption (what the row itself says); else the reviewed note of the
/// nearest result upstream (what `path` is computed from, fewest steps first, ties to the term
/// its equation lists first); else the nearest assumption upstream; else the dashboard row's
/// reason for having no note; else [`NO_NOTE`]. A reviewed note anywhere upstream outranks any
/// assumption: a note explains more than a rationale. The walk runs on every frame Explain
/// shows it: a few hundred paths at most.
pub fn note_fallback(path: &str) -> NoteFallback {
    let row = row_learn(path);
    if let Some(Learn::Note(id)) = row
        && let Some(note) = reviewed_note(id)
    {
        return NoteFallback::Note(note);
    }
    if let Some(Learn::Assumption(id)) = row
        && let Some(assumption) = ASSUMPTIONS.iter().find(|a| a.id == id)
    {
        return NoteFallback::Assumption(assumption);
    }
    let mut nearest_assumption = None;
    for term in upstream_nearest_first(path) {
        if let Some(note) = notes::note_for(&term) {
            return NoteFallback::Note(note);
        }
        if nearest_assumption.is_none() {
            nearest_assumption = ASSUMPTIONS
                .iter()
                .find(|a| a.paths.contains(&term.as_str()));
        }
    }
    match (nearest_assumption, row) {
        (Some(assumption), _) => NoteFallback::Assumption(assumption),
        (None, Some(Learn::None(reason))) => NoteFallback::Reason(reason),
        (None, _) => NoteFallback::Reason(NO_NOTE),
    }
}

/// Every path `path`'s equation reads, transitively, nearest first: breadth first over the
/// registry's terms, each equation's terms in its own order, each path once. Empty for a path
/// without an equation record.
fn upstream_nearest_first(path: &str) -> Vec<String> {
    let mut seen = BTreeSet::from([path.to_owned()]);
    let mut order = Vec::new();
    let mut queue = VecDeque::from([path.to_owned()]);
    while let Some(at) = queue.pop_front() {
        let Some(eq) = registry().equation_for(&at) else {
            continue;
        };
        for term in &eq.terms {
            if seen.insert(term.clone()) {
                order.push(term.clone());
                queue.push_back(term.clone());
            }
        }
    }
    order
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
    /// The start-here step shown, if the user is walking the suggested order.
    start_here: Option<usize>,
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
        self.start_here = None;
    }

    /// Hides the panel (its Close button, the header's toggle). The input row a leaf term
    /// framed loses its frame (decision M43-12: the frame goes with the equation shown); the
    /// trail stays, so the panel shown again opens where it was.
    pub fn close(&mut self) {
        self.open = false;
        self.focus = None;
    }

    /// Opens start-here step `step`'s equation and returns its note, which the panel shows in
    /// the Explain panel (UX overhaul spec 5.5); a step past the last one changes nothing and
    /// returns none.
    pub fn start(&mut self, step: usize) -> Option<&'static str> {
        let &(id, path) = START_HERE.get(step)?;
        self.trail = vec![path.to_owned()];
        self.open = true;
        self.focus = None;
        self.start_here = Some(step);
        Some(id)
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

    /// Follows a link clicked in a teaching note or in the lesson line under an equation: an
    /// input's row is highlighted and scrolled to, as a leaf term's is; a note is returned, for
    /// the panel to show in the Explain panel (decision C-11).
    pub fn follow_note_link(&mut self, link: NoteLink) -> Option<&'static str> {
        match link {
            NoteLink::Input(path) => {
                self.focus_input(path);
                None
            }
            NoteLink::Note(id) => Some(id),
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
    /// guide's link highlights its input's row, as a leaf term does. A note asked for is the
    /// Explain panel's: the panel takes it before this.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
        if events.hovered != self.hovered {
            self.hovered = events.hovered;
            ctx.request_repaint();
        }
        if let Some(path) = events.clicked {
            self.open_path(&path);
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
pub(crate) fn crumb(path: &str) -> String {
    registry()
        .symbol(path)
        .map_or_else(|| term_label(path).to_owned(), plain_symbol)
}

/// A value with its unit; a selector code with its choice's label after a colon
/// (`1: steel circuit`, and `5: 1, 3, 5 (default)` without nested parentheses); the cup wall's
/// failing verdict with its input's label ([`shown_text`], ruling R9).
fn value_text(path: &str, value: &Value, unit: &str) -> String {
    let text = value_with_unit(value, shown_text(path, value), unit);
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

/// Draws the Equation panel: its header row (heading, breadcrumb, Start here, close), then the
/// path shown; with `show_details` (the Design menu's Show spreadsheet details, spec S2-5) also
/// the value's workbook cell, the corrections its equation embodies and each term's internal
/// name on hover. `baseline` is the session's baseline (UX overhaul spec 7.1, C-22): an input
/// term carries the changed tag while it differs from the baseline's value. `clamps_paused` is
/// the panel's pause of the shaft-clamp section: the Start here hover and the Used by links
/// leave the clamp out while it holds. Returns the note asked for, which the Explain panel
/// shows (decision C-11): a lesson link's, or a Start here step's (spec 5.5).
pub fn explorer_ui(
    ui: &mut egui::Ui,
    explorer: &mut Explorer,
    inputs: &DesignInputs,
    results: &DesignResults,
    baseline: &DesignInputs,
    show_details: bool,
    clamps_paused: bool,
) -> Option<&'static str> {
    let terms = Design { inputs, results };
    let mut asked = None;
    ui.horizontal_wrapped(|ui| {
        // Whole crumbs move to the next row (a wrapped text would start mid-row).
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
        ui.strong(EQUATION_PANEL);
        ui.separator();
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
                .on_hover_text(start_here_hover(clamps_paused))
                .clicked()
            {
                asked = explorer.start(0);
            }
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
                asked = explorer.start(step - 1);
            }
            if ui
                .add_enabled(step + 1 < START_HERE.len(), egui::Button::new(NEXT))
                .clicked()
            {
                asked = explorer.start(step + 1);
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
            let Some(path) = explorer.current().map(str::to_owned) else {
                ui.weak(EMPTY_TEXT);
                return;
            };
            // The marks on screen are another equation's while its value is hovered.
            let others_marked = explorer.hovered().is_some_and(|h| h != path);
            let (follow, link) = match registry().equation_for(&path) {
                Some(eq) => equation_body(ui, eq, &terms, baseline, others_marked, show_details),
                None => {
                    no_equation_body(ui, &path, results, show_details);
                    (None, note_fallback_ui(ui, &path))
                }
            };
            if let Some(link) = link {
                asked = asked.or(explorer.follow_note_link(link));
            }
            ui.add_space(6.0);
            let users = used_by_shown(&path, clamps_paused);
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
    asked
}

/// The open equation: its label (and with `show_details` its cell), the equation large (a term
/// clicked is followed), the value, with `show_details` the corrections it embodies (their
/// titles on hover), its lesson line ([`lesson_ui`]), then the term list, its swatches faded
/// while `others_marked`, a term's internal name on hover with `show_details`. An input term
/// carries the changed tag while it differs from `baseline`'s value. Returns the term clicked
/// and the link clicked in the lesson line.
fn equation_body(
    ui: &mut egui::Ui,
    eq: &Equation,
    terms: &Design<'_>,
    baseline: &DesignInputs,
    others_marked: bool,
    show_details: bool,
) -> (Option<String>, Option<NoteLink>) {
    let mut clicked = None;
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
    ui.separator();
    let link = lesson_ui(ui, &eq.target);
    ui.separator();
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
                        let mut style = registry()
                            .term_style(&row.path, terms.inputs)
                            .expect("every term is an input or a result (registry build)");
                        // The registry compares with the spreadsheet's values; on screen
                        // "changed" is against the session's baseline.
                        if matches!(style.kind, TermKind::Input { .. }) {
                            style.changed_from_default =
                                terms.inputs.get(&row.path) != baseline.get(&row.path);
                        }
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
    let unit = result_unit(path);
    ui.strong(format!("{label} = {}", value_text(path, &value, unit)));
    if show_details {
        let cell = result_info(path)
            .and_then(|info| info.cell.clone())
            .unwrap_or_else(|| crate::gui::plain_text::NO_SPREADSHEET_CELL.to_owned());
        ui.weak(cell);
    }
    ui.weak(NO_EQUATION);
}

/// The lesson line under the equation of `path` (decision C-11): [`FULL_LESSON`], then on the
/// row under it the title of the reviewed note that explains it, a link that shows the note in
/// the Explain panel (its own row, so a long title wraps inside the panel: plan review UXP-1);
/// for an equation no note explains, the next best read ([`note_fallback_ui`]). Returns the
/// link clicked.
fn lesson_ui(ui: &mut egui::Ui, path: &str) -> Option<NoteLink> {
    let Some(note) = notes::note_for(path) else {
        return note_fallback_ui(ui, path);
    };
    ui.label(FULL_LESSON);
    ui.link(glyph_safe(note.title))
        .clicked()
        .then_some(NoteLink::Note(note.id))
}

/// The next best read under an equation no reviewed note explains ([`note_fallback`]): the
/// closest note, a link; an assumption's rationale; or why there is none.
fn note_fallback_ui(ui: &mut egui::Ui, path: &str) -> Option<NoteLink> {
    match note_fallback(path) {
        NoteFallback::Note(note) => {
            let mut asked = None;
            ui.horizontal_wrapped(|ui| {
                ui.weak(NEAREST_NOTE);
                if ui.link(glyph_safe(note.title)).clicked() {
                    asked = Some(NoteLink::Note(note.id));
                }
            });
            asked
        }
        NoteFallback::Assumption(assumption) => {
            ui.weak(format!("{NEAREST_ASSUMPTION} {}", assumption.label));
            ui.add(egui::Label::new(glyph_safe(assumption.rationale)).wrap());
            None
        }
        NoteFallback::Reason(reason) => {
            ui.weak(reason);
            ui.weak(START_HERE_HINT);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::explain::note_text;
    use crate::engine::meta::ResultSet;
    use crate::gui::notes_view::{
        NoteView, WATCH_OUT, WHAT_TO_CHANGE, WHERE_FROM, live_value, note_text_on,
    };
    use crate::gui::results_table::table_entries;
    use crate::gui::test_support::{design, nothing_held};

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
            explorer_ui(
                ui,
                explorer,
                &inputs,
                &results,
                &DesignInputs::default(),
                false,
                clamps::PAUSED,
            );
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
        // Each step opens its equation and names its note, which the panel shows in the
        // Explain panel (UX overhaul spec 5.5).
        let mut explorer = Explorer::default();
        assert_eq!(explorer.start(0), Some(START_HERE[0].0));
        assert_eq!(explorer.start_here(), Some(0));
        assert!(explorer.open);
        assert_eq!(explorer.current(), Some(START_HERE[0].1));
        let last = START_HERE.len() - 1;
        assert_eq!(explorer.start(last), Some(START_HERE[last].0));
        assert_eq!(explorer.start(last + 1), None);
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
        // The button's hover names the walk's topics in plain words, step 1's first.
        for paused in [false, true] {
            let hover = start_here_hover(paused);
            assert!(crate::gui::guidance::plain_text_violations(hover).is_empty());
            assert!(hover.starts_with("A suggested order: how the coupling makes torque"));
        }
    }

    #[test]
    fn the_start_here_hover_names_the_clamp_step_only_while_the_section_shows() {
        assert_eq!(
            start_here_hover(false),
            format!("{}, clamps", start_here_hover(true))
        );
        // BL-064: the shaft-joint step is the last shown while the section is paused, so its
        // topic ends the hover there, after demagnetization's.
        assert!(
            start_here_hover(true).ends_with("demagnetization, shaft joints"),
            "{}",
            start_here_hover(true)
        );
        assert!(!start_here_hover(true).to_lowercase().contains("clamp"));
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
    fn the_panel_s_pause_decides_the_drawn_used_by_links() {
        // MGV-04: the Equation panel follows the `clamps_paused` it is given, not the constant.
        use crate::gui::test_support::{SCREEN, drawn_texts, sized_frame};
        let path = "metal.torque_cold_high_Nm";
        let clamp_user = crumb("clamps.max_torque_Nm");
        let drawn = |paused: bool| {
            let (inputs, results) = design();
            let mut explorer = Explorer::default();
            explorer.open_path(path);
            let output = sized_frame(&egui::Context::default(), SCREEN, Vec::new(), |ui| {
                explorer_ui(
                    ui,
                    &mut explorer,
                    &inputs,
                    &results,
                    &DesignInputs::default(),
                    false,
                    paused,
                );
            });
            drawn_texts(&output)
        };
        assert!(drawn(false).contains(&clamp_user.to_string()));
        assert!(!drawn(true).contains(&clamp_user.to_string()));
    }

    #[test]
    fn every_start_here_step_opens_an_equation_with_its_reviewed_note() {
        for &(id, path) in START_HERE {
            assert!(registry().equation_for(path).is_some(), "{path}");
            assert_eq!(notes::note_for(path).map(|n| n.id), Some(id), "{path}");
        }
    }

    #[test]
    fn the_start_here_bar_names_every_step_it_shows() {
        // Spec 3.3, ruling 11 and BL-064: the bar walks START_HERE, the full order of fifteen
        // steps or, while the shaft-clamp section is paused, its first fourteen (the clamp step,
        // the last, hidden); the torque note first; each bar names its step of the steps shown
        // and its note, and only the ferrite step says it is for ferrite rings only (TC-32).
        assert_eq!(notes::start_here_for(false).len(), 15);
        assert_eq!(notes::start_here_for(true).len(), 14);
        assert_eq!(START_HERE[0].0, "torque_from_shear");
        // Three bars pinned as drawn: the first step, the ferrite step, the 13th in both orders,
        // and the shaft-joint step, the 14th in both (BL-064). The count is the steps shown, so
        // the pins hold in both switch states.
        let steps = START_HERE.len();
        assert_eq!(
            start_here_text(0),
            format!("Start here 1 of {steps}: How the coupling makes torque")
        );
        let ferrite = reviewed_note("ferrite_cold_demag").expect("the ferrite note is reviewed");
        assert_eq!(
            start_here_text(12),
            format!(
                "Start here 13 of {steps} (ferrite rings only): {}",
                ferrite.title
            )
        );
        let joints = reviewed_note("shaft_joint").expect("the shaft-joint note is reviewed");
        assert_eq!(
            start_here_text(13),
            format!("Start here 14 of {steps}: {}", joints.title)
        );
        // Every step's note is reviewed, so its bar draws its title; no other bar has the
        // ferrite label.
        for (step, &(id, _)) in START_HERE.iter().enumerate() {
            assert!(reviewed_note(id).is_some(), "{id} is not reviewed");
            assert_eq!(
                start_here_text(step).contains(FERRITE_ONLY),
                id == "ferrite_cold_demag",
                "step {}",
                step + 1
            );
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
    fn the_lesson_line_names_the_open_equation_s_note_and_asks_for_it() {
        // Decision C-11: the note itself is the Explain panel's; under the equation the lesson
        // line names it after "Full lesson:", a link that returns it, and nothing of the note is
        // drawn here.
        use crate::engine::explain::notes::note_for;
        use crate::gui::test_support::{
            SCREEN, drawn_texts, primary_button, sized_frame, text_rect,
        };
        let path = "model.pullout_Nm";
        let note = note_for(path).expect("the pull-out's note");
        let (inputs, results) = design();
        let ctx = egui::Context::default();
        let mut explorer = Explorer::default();
        explorer.open_path(path);
        let frame = |events: Vec<egui::Event>, explorer: &mut Explorer| {
            let mut asked = None;
            let output = sized_frame(&ctx, SCREEN, events, |ui| {
                asked = explorer_ui(
                    ui,
                    explorer,
                    &inputs,
                    &results,
                    &DesignInputs::default(),
                    false,
                    clamps::PAUSED,
                );
            });
            (output, asked)
        };
        frame(Vec::new(), &mut explorer);
        let (output, asked) = frame(Vec::new(), &mut explorer);
        assert_eq!(asked, None);
        let texts = drawn_texts(&output);
        let title = glyph_safe(note.title);
        assert_eq!(
            texts.iter().filter(|t| **t == title).count(),
            1,
            "{texts:?}"
        );
        assert!(texts.iter().any(|t| t == FULL_LESSON), "{texts:?}");
        assert!(
            !texts.contains(&note_text_on(
                note.sentences[0],
                &NoteView {
                    inputs: &inputs,
                    results: &results,
                    held: &nothing_held,
                }
            )),
            "the note's sentences are the Explain panel's"
        );
        let at = text_rect(&output, &title).unwrap().center();
        frame(vec![egui::Event::PointerMoved(at)], &mut explorer);
        frame(vec![primary_button(at, true)], &mut explorer);
        let (_, asked) = frame(vec![primary_button(at, false)], &mut explorer);
        assert_eq!(asked, Some(note.id));
        assert_eq!(explorer.current(), Some(path), "the equation stays open");
    }

    #[test]
    fn a_note_link_is_the_explain_panel_s_and_leaves_the_open_equation() {
        // Decision C-11: a note asked for under an equation is returned for the Explain panel;
        // the equation stays open (decision C-25).
        let mut explorer = Explorer::default();
        explorer.open_path("model.f_end");
        assert_eq!(
            explorer.follow_note_link(NoteLink::Note("a5.low_saturation")),
            Some("a5.low_saturation")
        );
        assert_eq!(explorer.current(), Some("model.f_end"));
        assert!(explorer.open);
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
        // A text result (the time to the limit with no drag) takes no unit after it.
        let never = crate::engine::temperature::NEVER;
        assert_eq!(
            value_text(
                "temperature.thermal.time_to_limit_high",
                &Value::Text(never.to_owned()),
                "s"
            ),
            never
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
            |ui| {
                explorer_ui(
                    ui,
                    explorer,
                    &inputs,
                    &results,
                    &DesignInputs::default(),
                    show_details,
                    clamps::PAUSED,
                );
            },
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
                explorer_ui(
                    ui,
                    &mut explorer,
                    inputs,
                    &results,
                    &DesignInputs::default(),
                    false,
                    clamps::PAUSED,
                );
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

    #[test]
    fn a_note_s_link_highlights_its_input_s_row_and_asks_for_no_note() {
        let mut explorer = Explorer::default();
        assert_eq!(
            explorer.follow_note_link(NoteLink::Input("metal.cup_wall_corner_mm")),
            None
        );
        assert_eq!(
            explorer.focus(),
            Some(&Focus {
                path: "metal.cup_wall_corner_mm".to_owned(),
                scroll: true,
            })
        );
    }

    #[test]
    fn the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words() {
        // TC-29b: the empty panel names the walk through the notes by its button's own name
        // (the Explain toggle it also named went with decision C-11); the fallback lines read
        // as plain words in egui's default fonts.
        assert!(EMPTY_TEXT.contains(START_HERE_BUTTON));
        assert!(!EMPTY_TEXT.contains("Explain"), "{EMPTY_TEXT}");
        assert!(START_HERE_HINT.contains(START_HERE_BUTTON));
        let ctx = egui::Context::default();
        crate::gui::test_support::sized_frame(
            &ctx,
            crate::gui::test_support::SCREEN,
            Vec::new(),
            |_| {},
        );
        for text in [
            EMPTY_TEXT,
            NO_NOTE,
            START_HERE_HINT,
            NEAREST_NOTE,
            NEAREST_ASSUMPTION,
        ] {
            assert_eq!(
                crate::gui::guidance::plain_text_violations(text),
                Vec::<&str>::new(),
                "{text:?}"
            );
            crate::gui::test_support::assert_glyphs(&ctx, text, "an Equation panel line");
        }
    }

    #[test]
    fn every_dashboard_row_without_a_note_offers_a_note_an_assumption_or_its_own_reason() {
        // Spec section 6 test 11: "No teaching note" never shows for a dashboard row (paths the
        // panel cannot reach while the clamp section is paused are skipped).
        use crate::gui::dashboard::DASHBOARD;
        let mut without = 0;
        for &(path, _) in DASHBOARD
            .iter()
            .filter(|row| !clamps::hides(row.0, clamps::PAUSED))
        {
            if notes::note_for(path).is_some() {
                continue;
            }
            without += 1;
            match note_fallback(path) {
                NoteFallback::Note(note) => {
                    assert_eq!(
                        reviewed_note(note.id).map(|n| n.id),
                        Some(note.id),
                        "{path}"
                    );
                    assert!(
                        notes::shown(note.id),
                        "{path}: offers {}, a note not shown",
                        note.id
                    );
                }
                NoteFallback::Assumption(assumption) => {
                    assert!(ASSUMPTIONS.iter().any(|a| a.id == assumption.id), "{path}");
                }
                NoteFallback::Reason(reason) => {
                    assert_ne!(reason, NO_NOTE, "{path}: a dead end on a dashboard row");
                    assert_eq!(row_learn(path), Some(Learn::None(reason)), "{path}");
                }
            }
        }
        assert!(without > 0, "some dashboard rows have no note of their own");
    }

    /// The fewest steps from `path` down the registry's terms to every path upstream of it,
    /// level by level (the test's own walk, apart from `upstream_nearest_first`'s queue).
    fn depths(path: &str) -> std::collections::BTreeMap<String, usize> {
        let mut depth = std::collections::BTreeMap::new();
        let mut level = BTreeSet::from([path.to_owned()]);
        let mut steps = 0;
        while !level.is_empty() {
            steps += 1;
            let next: BTreeSet<String> = level
                .iter()
                .filter_map(|p| registry().equation_for(p))
                .flat_map(|eq| eq.terms.iter().cloned())
                .filter(|t| t != path && !depth.contains_key(t))
                .collect();
            for t in &next {
                depth.insert(t.clone(), steps);
            }
            level = next;
        }
        depth
    }

    #[test]
    fn the_offered_note_is_the_nearest_upstream_and_a_note_outranks_an_assumption() {
        // TC-29a, TC-30, over every explained result without a note of its own and without a
        // dashboard Why link (those lead first: the next test); paths the panel cannot reach
        // while the clamp section is paused are skipped.
        let assumption_paths = |p: &str| ASSUMPTIONS.iter().any(|a| a.paths.contains(&p));
        let mut seen = [0_usize; 3];
        for eq in registry().equations() {
            let path = eq.target.as_str();
            if clamps::hides(path, clamps::PAUSED)
                || notes::note_for(path).is_some()
                || row_learn(path).is_some()
            {
                continue;
            }
            let depth = depths(path);
            let nearest_note = depth
                .iter()
                .filter(|(p, _)| notes::note_for(p).is_some())
                .map(|(_, &d)| d)
                .min();
            let nearest_assumption = depth
                .iter()
                .filter(|(p, _)| assumption_paths(p))
                .map(|(_, &d)| d)
                .min();
            match note_fallback(path) {
                NoteFallback::Note(note) => {
                    seen[0] += 1;
                    assert!(
                        notes::shown(note.id),
                        "{path}: offers {}, a note not shown",
                        note.id
                    );
                    let at = depth
                        .iter()
                        .filter(|(p, _)| notes::note_for(p).map(|n| n.id) == Some(note.id))
                        .map(|(_, &d)| d)
                        .min();
                    assert_eq!(at, nearest_note, "{path}: {}", note.id);
                }
                NoteFallback::Assumption(assumption) => {
                    seen[1] += 1;
                    assert_eq!(
                        nearest_note, None,
                        "{path}: a note upstream outranks {}",
                        assumption.id
                    );
                    let at = depth
                        .iter()
                        .filter(|(p, _)| assumption.paths.contains(&p.as_str()))
                        .map(|(_, &d)| d)
                        .min();
                    assert_eq!(at, nearest_assumption, "{path}: {}", assumption.id);
                }
                NoteFallback::Reason(reason) => {
                    seen[2] += 1;
                    assert_eq!(reason, NO_NOTE, "{path}");
                    assert_eq!((nearest_note, nearest_assumption), (None, None), "{path}");
                }
            }
        }
        assert!(
            seen.iter().all(|&n| n > 0),
            "each kind of offer occurs: {seen:?}"
        );
    }

    #[test]
    fn a_dashboard_row_s_own_link_comes_first_and_a_value_with_no_record_has_nothing_upstream() {
        // The end-effect check has no equation record: its row's Why note is what Explain offers.
        let check = "model.end_effect_check";
        assert_eq!(registry().term_kind(check), Some(TermKind::CellOnly));
        let Some(Learn::Note(id)) = row_learn(check) else {
            panic!("the end-effect row links a note");
        };
        assert_eq!(
            note_fallback(check),
            NoteFallback::Note(reviewed_note(id).unwrap())
        );
        // A result off the dashboard without a record: nothing is known upstream of it.
        let cell_only = crate::gui::results_table::table_entries()
            .iter()
            .map(|e| e.path.as_str())
            .find(|p| registry().term_kind(p) == Some(TermKind::CellOnly) && row_learn(p).is_none())
            .expect("a result without a record, off the dashboard");
        assert_eq!(note_fallback(cell_only), NoteFallback::Reason(NO_NOTE));
    }

    // The sign-off sheet's text before slice 2: T4's snapshot and its reader.
    use crate::engine::explain::notes::before::{SNAPSHOT, parse};

    /// The notes slice 2 added: there is no text of theirs before it.
    const NEW_NOTES: [&str; 3] = ["torque_from_shear", "hot_low_cold_high", "when_slipping"];

    /// What the sheet says in place of a new note's old text.
    const NEW_IN_SLICE: &str = "(new in slice 2)";

    /// The notes added after slice 2 (BL-064's shaft joints): no text of theirs before the slice
    /// either, and not the slice's own. The sheet is not regenerated for them (BL-064 spec 6.1):
    /// the owner's review of the diff is the new note's sign-off.
    const AFTER_THE_SLICE: [&str; 1] = ["shaft_joint"];

    /// What the sheet says in place of the old text of a note added after the slice.
    const ADDED_AFTER_THE_SLICE: &str =
        "(added after slice 2; its own physics review is its record)";

    /// What the sheet says while the shaft-clamp section is paused (ruling 11).
    const CLAMPS_PAUSED_LINE: &str = "The shaft-clamp section is paused (owner, 2026-10-15): its note, clamp_preload, keeps its earlier text, is not shown on screen and is not part of this sign-off; the shafts' keyed shrink fits are checked instead (owner, 2026-10-17).";

    /// Where `notes_signoff_sheet` writes the sheet.
    const SIGNOFF_SHEET: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/analyses/2026-10-14-magcoupler-notes-signoff.md"
    );

    /// A note as it ships, rendered at `results` as the panel draws it: the sentences, each
    /// "What to change" entry as "<Step word> <display label>: <text>", the watch-outs and the
    /// sources, every placeholder filled ([`live_value`]).
    fn note_now(note: &Note, results: &DesignResults) -> String {
        use crate::engine::explain::note_text::render;
        let value = |path: &str| live_value(results, path);
        let mut text = String::from("**What it is**\n\n");
        for (i, sentence) in note.sentences.iter().enumerate() {
            text.push_str(&format!("{}. {}\n", i + 1, render(sentence, value)));
        }
        text.push_str(&format!("\n**{WHAT_TO_CHANGE}**\n\n"));
        if note.change.is_empty() {
            text.push_str("(none)\n");
        }
        for entry in note.change {
            text.push_str(&format!(
                "- {} {}: {}\n",
                entry.step.word(),
                label_for(entry.input),
                render(entry.text, value)
            ));
        }
        text.push_str(&format!("\n**{WATCH_OUT}**\n\n"));
        if note.watch_out.is_empty() {
            text.push_str("(none)\n");
        }
        for line in note.watch_out {
            text.push_str(&format!("- {}\n", render(line, value)));
        }
        text.push_str(&format!("\n**{WHERE_FROM}**\n\n"));
        for source in note.sources {
            text.push_str(&format!("- {source}\n"));
        }
        text
    }

    /// The owner's sign-off sheet (spec section 4, S2-6): for each note shown, its text before
    /// slice 2 and the text it ships with at the default design, and its review record. While
    /// the shaft-clamp section is paused (`clamps_paused`) its note is not on the sheet, and a
    /// line says why.
    fn signoff_sheet(clamps_paused: bool) -> String {
        let results = crate::compute_all(&DesignInputs::default());
        let shown: Vec<&Note> = notes::NOTES
            .iter()
            .filter(|n| notes::shown_when(n.id, clamps_paused))
            .collect();
        let before = parse(SNAPSHOT);
        let mut sheet = format!(
            "# Magcoupler teaching notes: owner sign-off sheet (guidance slice 2)\n\n\
             Generated by `notes_signoff_sheet` (magcoupling-rs/src/gui/explorer.rs). Regenerate with\n\
             `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_signoff_sheet -- --ignored`.\n\n\
             For each of the {} notes shown: **Before** is the note at the start of slice 2 (commit b0166ad, from\n\
             `docs/analyses/2026-10-14-magcoupler-notes-before.txt`); **Now** is the note as it ships, rendered at\n\
             the default design: live values filled in, each \"{WHAT_TO_CHANGE}\" entry with the input's name as the\n\
             screen shows it. On screen \"{WHERE_FROM}\" starts closed. Each note passed its physics review;\n\
             \"Review\" names its section of `docs/analyses/2026-10-14-magcoupler-notes-review.md`.\n\n\
             Owner sign-off (spec S2-6, S2-10). Signed: ____________________  Date: ____________\n",
            shown.len()
        );
        if clamps_paused {
            sheet.push_str(&format!("\n{CLAMPS_PAUSED_LINE}\n"));
        }
        for note in shown {
            let record = match note.review {
                Review::Reviewed { record, .. } => record,
                Review::Draft => "NOT REVIEWED",
            };
            sheet.push_str(&format!(
                "\n## {}: {}\n\nReview: {record}\n\n### Before\n\n",
                note.id, note.title
            ));
            match before.iter().find(|(id, _)| *id == note.id) {
                Some((_, lines)) => {
                    let text: Vec<String> = lines
                        .iter()
                        .map(|(field, text)| format!("{field}: {text}"))
                        .collect();
                    sheet.push_str(&format!("```text\n{}\n```\n", text.join("\n")));
                }
                None if AFTER_THE_SLICE.contains(&note.id) => {
                    sheet.push_str(&format!("{ADDED_AFTER_THE_SLICE}\n"));
                }
                None => sheet.push_str(&format!("{NEW_IN_SLICE}\n")),
            }
            sheet.push_str(&format!(
                "\n### Now (default design)\n\n{}",
                note_now(note, &results)
            ));
        }
        sheet
    }

    #[test]
    #[ignore = "writes the owner's sign-off sheet; run by hand before release"]
    fn notes_signoff_sheet() {
        std::fs::write(SIGNOFF_SHEET, signoff_sheet(clamps::PAUSED))
            .expect("write the sign-off sheet");
        println!("wrote {SIGNOFF_SHEET}");
    }

    #[test]
    fn the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice() {
        // T4's snapshot holds the 17 notes there were, each once and with text; the three
        // notes slice 2 added and the one added after it (BL-064) have none.
        let before = parse(SNAPSHOT);
        let ids: Vec<&str> = before.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids.len(),
            notes::NOTES.len() - NEW_NOTES.len() - AFTER_THE_SLICE.len(),
            "{ids:?}"
        );
        for (id, lines) in &before {
            assert!(notes::note(id).is_some(), "{id}: no such note now");
            assert!(!lines.is_empty(), "{id}: no text");
        }
        let mut without: Vec<&str> = notes::NOTES
            .iter()
            .map(|n| n.id)
            .filter(|id| !ids.contains(id))
            .collect();
        without.sort_unstable();
        let mut new_notes = [NEW_NOTES.as_slice(), AFTER_THE_SLICE.as_slice()].concat();
        new_notes.sort_unstable();
        assert_eq!(without, new_notes);
        // One section per note shown, each with its text before (or "new") and its text now:
        // while the clamp section is paused its note has none, and a line says why (ruling
        // 11); with the section back every note has one. Both switch states.
        for paused in [true, false] {
            let sheet = signoff_sheet(paused);
            for note in notes::NOTES {
                let heading = format!("\n## {}: ", note.id);
                assert_eq!(
                    sheet.matches(heading.as_str()).count(),
                    usize::from(notes::shown_when(note.id, paused)),
                    "{} (paused: {paused})",
                    note.id
                );
            }
            assert_eq!(
                sheet.matches(NEW_IN_SLICE).count(),
                NEW_NOTES.len(),
                "paused: {paused}"
            );
            // The note added after the slice is shown in both switch states, so its line shows
            // once in both.
            assert_eq!(
                sheet.matches(ADDED_AFTER_THE_SLICE).count(),
                AFTER_THE_SLICE.len(),
                "paused: {paused}"
            );
            assert_eq!(sheet.contains(CLAMPS_PAUSED_LINE), paused);
        }
        // What ships has every placeholder filled, each with a value at the default design
        // (never "no prediction"), and each sentence as the panel draws it.
        let results = crate::compute_all(&DesignInputs::default());
        for note in notes::NOTES {
            let now = note_now(note, &results);
            assert!(
                !now.contains('{') && !now.contains('}'),
                "{}: {now}",
                note.id
            );
            for path in note_text::placeholders(note) {
                assert!(
                    live_value(&results, path).is_some(),
                    "{}: {path} has no value at the default design",
                    note.id
                );
            }
            for sentence in note.sentences {
                let shown = note_text::render(sentence, |path: &str| live_value(&results, path));
                assert!(now.contains(&shown), "{}: {shown:?}", note.id);
            }
        }
    }
}
