//! The Explain panel (UX overhaul spec 5): the right column under the check list. It shows one
//! thing, the [`Selection`]: what was clicked last, or the orientation while nothing is. A
//! click selects; a hover never changes the panel (decision C-2).
//!
//! Every entry has the same headings in the same order and leaves out a heading it has nothing
//! for (spec 5.3): its title, [`WHAT_IT_IS`], [`WHAT_SETS_IT`], "What to change",
//! [`WHAT_A_CHANGE_DOES`], [`CHECKS_IT_TOUCHES`], "Watch out", then its foot links
//! ([`FULL_LESSON`], [`EQUATION`]). A check's entry names its rows [`NUMBERS`] and draws them
//! after "Watch out", the notes on its fixes staying beside the fixes (the plan's decision
//! D-18). The texts under the headings are ones the screen drew before this panel: a
//! check's entry holds its guide's Problem, its Why sentence, its fixes and their notes, each
//! with the links it had; a tile's entry holds its result's help text and, for the hot-low
//! torque, the torque card (decision C-10). A dimension tag, a callout row and a results-table
//! row select their result too (spec 5.2): its value and its name as the table shows them, its
//! help, the check of its row, the stored-3D line where its row reads those fields, its row's
//! lesson and its Equation link; a check's own result selects the check.
//!
//! A row of the Size column selects its entry when touched (spec 4.3, 5.2): what the control is
//! (its label's hover), the lines the column drew under it (its hint, what sets it, the floor,
//! the footnote on what the controls move), its torque-card row, the checks whose fixes name
//! it, where two of them pull it opposite ways the sentence that says so, its Why link's note
//! and, with the spreadsheet details, its input's cell. A row of the detailed inputs or the
//! assumptions selects the same form, keyed by its input's path: its value and label, its help,
//! its hint, its checks, its "Watch out" and its cell. An input's entry leads with the link that
//! traces it ([`TRACE_LINK`]), and a check's fix link selects its input (spec 5.2). With the
//! spreadsheet details a result's and a check's entry end with their cell too.
//!
//! A teaching note shows here too (decision C-11): the top bar's [`LEARN`] lists every note
//! shown by its title, and a title, a lesson link anywhere on the page or a step of the Start
//! here walk (spec 5.5) selects the note. Its entry is the links to the equations it explains,
//! then the note as the Equation panel drew it ([`note_ui`]).
//!
//! The selection is view state: in no design file, share link or undo step.

use crate::engine::explain::notes::Note;
use crate::engine::explain::notes::{NOTES, shown_when};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::size_controls::dimensions;
use crate::engine::sizing::FreeVariable;
use crate::gui::checks::{mark_level, state_mark};
use crate::gui::dashboard::why_note_link;
use crate::gui::dashboard::{
    DOES_NOT_FIT, FITS, Level, MARKER_LEGEND, STORED_3D_LABEL, STORED_3D_NOTE, STORED_3D_ROWS,
    badging_check, learn_ui, line_ui, result_info, shown_lines, shown_text,
};
use crate::gui::explorer::{START_HERE_BUTTON, reviewed_note, start_here_hover};
use crate::gui::format::{format_value, value_with_unit};
use crate::gui::guidance::{
    self, CHECK_GUIDES, CheckGuide, Consequences, Direction, FIT_CHECK, Learn, NO_FIX,
    SIZE_TAB_ALIASES, SIZE_TAB_NOTE, fix_text, guide_notes, input_trade_offs, row_learn,
};
use crate::gui::input_ui::weak_line;
use crate::gui::inputs::{InputCatalogue, value_text};
use crate::gui::look;
use crate::gui::notes_view::{NoteLink, NoteView, WATCH_OUT, WHAT_TO_CHANGE, explains_ui, note_ui};
use crate::gui::plain_text::{
    DIAMETER_HINT_KEY, LENGTH_HINT_KEY, MINIMUM_OD_HINT_KEY, NO_SPREADSHEET_CELL, display_label,
    help_for, hint_for, label_for,
};
use crate::gui::readouts::{Readouts, registry};
use crate::gui::size_controls::{
    LINKED_HOUSING, SIZE_ROWS, SizeRow, floor_line, row_help, size_row, slider_minimum,
};
use crate::gui::tiles::Tile;
use crate::gui::torque_card::CARD_HEADING;
use crate::gui::typeset::glyph_safe;
use crate::gui::wrapping_line;
use crate::{DesignInputs, DesignResults};

/// The heading of the Explain panel.
pub const HEADING: &str = "EXPLAIN";

/// The orientation's first paragraph (spec 5.4): what the thing is.
pub const ORIENTATION: &str = "Two rings of magnets, one inside the other, pass torque across the gap between them without touching. Up to the pull-out torque the rings turn together; past it they slip.";

/// The orientation's three moves (spec 5.4).
pub const STEPS: [&str; 3] = [
    "1. Set the size on the left.",
    "2. Read the torque above the drawing.",
    "3. Open any check that needs attention.",
];

/// The link at the head of an entry that returns to the orientation (spec 5.1).
pub const ABOUT: &str = "About this tool";

/// An entry's headings (spec 5.3), with [`WHAT_TO_CHANGE`] and [`WATCH_OUT`], which the
/// teaching notes use too.
pub const WHAT_IT_IS: &str = "What it is";
pub const WHAT_SETS_IT: &str = "What sets it";
pub const WHAT_A_CHANGE_DOES: &str = "What a change does";
pub const NUMBERS: &str = "Numbers behind this check";
pub const CHECKS_IT_TOUCHES: &str = "Checks it touches";

/// The head of an entry's lesson link: its teaching note, its assumption, or the line that
/// says no note covers it yet.
pub const FULL_LESSON: &str = "Full lesson:";

/// The foot link that opens the entry's result in the Equation panel (decision C-11).
pub const EQUATION: &str = "Equation";

/// The head of an entry's foot line with the spreadsheet details on (spec 5.3): its workbook
/// cell, or the line that says the spreadsheet has none.
pub const SPREADSHEET_CELL: &str = "Spreadsheet cell:";

/// An input's link that traces it (spec 5.2): it frames the results the input drives, and a
/// second click ends the trace (decision O-8).
pub const TRACE_LINK: &str = "Highlight the results this drives";

/// The top bar's button that lists the teaching notes here, and the list's heading (UX
/// overhaul spec 4.2).
pub const LEARN: &str = "Learn";

/// What the Explain panel shows (spec 5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Selection {
    /// Nothing is selected: the orientation.
    #[default]
    Nothing,
    /// A check, by its guide's check path ([`CheckGuide::check`]).
    Check(&'static str),
    /// A result, by its path: one of the four tiles ([`Tile::path`]), a dimension tag or a
    /// callout row of the drawing, a row of the results table.
    Result(&'static str),
    /// A teaching note, by its id.
    Note(&'static str),
    /// The list of the teaching notes ([`LEARN`]).
    NoteList,
    /// An input: a row of the Size column, by its key ([`SizeRow::key`]), or a row of the
    /// detailed inputs or the assumptions, by its input's path.
    Input(&'static str),
}

impl Selection {
    /// The check selected, if one is.
    pub fn check(self) -> Option<&'static str> {
        match self {
            Selection::Check(check) => Some(check),
            Selection::Nothing
            | Selection::Result(_)
            | Selection::Note(_)
            | Selection::NoteList
            | Selection::Input(_) => None,
        }
    }

    /// The tile selected, if one is.
    pub fn tile(self) -> Option<Tile> {
        match self {
            Selection::Result(path) => Tile::ALL.into_iter().find(|tile| tile.path() == path),
            Selection::Nothing
            | Selection::Check(_)
            | Selection::Note(_)
            | Selection::NoteList
            | Selection::Input(_) => None,
        }
    }

    /// The key of the input selected, if one is.
    pub fn input(self) -> Option<&'static str> {
        match self {
            Selection::Input(key) => Some(key),
            Selection::Nothing
            | Selection::Check(_)
            | Selection::Result(_)
            | Selection::Note(_)
            | Selection::NoteList => None,
        }
    }

    /// Whether the entry reads the Size tab's consequences ([`Shown::consequences`]): the
    /// panel asks its cache for them only then. A Size column row's entry draws its
    /// what-moves-what lines and its torque-card row from them; another input's entry reads
    /// none.
    pub fn reads_consequences(self) -> bool {
        self.tile() == Some(Tile::HotLow) || self.input().and_then(size_row).is_some()
    }
}

/// What a click in the Explain panel asks of the panel that owns it. A note and an input's row
/// are asked for through the readouts, as every other link asks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Asked {
    /// Show this instead.
    Select(Selection),
    /// Open this result in the Equation panel. The selection stays (decision C-25).
    Equation(&'static str),
    /// Start the Start here walk (spec 5.4).
    StartHere,
    /// Trace the input at this path, or end its trace if it is the one traced (spec 5.2,
    /// decision O-8).
    Trace(&'static str),
}

/// What the entries are drawn from: the design shown, its results, every guide's fixes in
/// [`CHECK_GUIDES`]' order and the shaft-clamp section's pause; for an entry that reads them
/// ([`Selection::reads_consequences`]), the Size tab's consequences of the design, with the
/// size Auto sets while it is on; why an input is not free on the design shown, for a note's
/// change entries ([`NoteView::held`]); and the Size column's Minimum OD preference.
pub(crate) struct Shown<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub fixes: &'a [Vec<(&'static str, Direction)>],
    pub clamps_paused: bool,
    pub consequences: Option<&'a Consequences>,
    pub auto: Option<FreeVariable>,
    pub held: &'a dyn Fn(&str) -> Option<String>,
    pub minimum_diameter_mm: Option<f64>,
}

/// Draws the Explain panel: [`HEADING`], with [`ABOUT`] beside it while something is selected,
/// then the entry of `selection`. Returns what a click asked for.
pub(crate) fn explain_ui(
    ui: &mut egui::Ui,
    selection: Selection,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let mut asked = None;
    ui.horizontal(|ui| {
        ui.colored_label(look::quiet_text(ui.visuals()), HEADING);
        if selection != Selection::Nothing {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.link(ABOUT).clicked() {
                    asked = Some(Asked::Select(Selection::Nothing));
                }
            });
        }
    });
    let entry = match selection {
        Selection::Nothing => orientation_ui(ui, shown.clamps_paused),
        Selection::Check(check) => check_ui(ui, check, shown, readouts),
        Selection::Result(path) => result_ui(ui, path, shown, readouts),
        Selection::Note(id) => note_entry_ui(ui, id, shown, readouts),
        Selection::NoteList => note_list_ui(ui, shown.clamps_paused),
        Selection::Input(key) => input_ui(ui, key, shown, readouts),
    };
    asked.or(entry)
}

/// The entry of the input `key` (spec 5.3's input column): its value and its label; for an
/// input the trace knows, the link that traces it ([`TRACE_LINK`]); what it is (its label's
/// hover); what sets it (its hint, its what-moves-what lines, for Minimum OD the floor line,
/// for Outside diameter and Overall length the footnote on what the controls move); what a
/// change does (its torque-card row); the checks whose fixes name its input; where two of them
/// move it opposite ways, the sentence that says so ("Watch out"); and the foot (its Why
/// link's note, and with the spreadsheet details the cell of a row that is one input). `key`
/// is a Size column row's ([`size_row`]) or an input's path ([`input_row`]); nothing for
/// another.
fn input_ui(
    ui: &mut egui::Ui,
    key: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let row = size_row(key).copied().or_else(|| input_row(key))?;
    wrapping_line(ui, |ui| {
        if let Some(value) = row_value(&row, shown) {
            ui.strong(value);
        }
        ui.colored_label(look::quiet_text(ui.visuals()), row.label);
    });
    let mut asked = None;
    if let Some(input) = row.input.filter(|input| registry().is_leaf_input(input))
        && ui.link(TRACE_LINK).clicked()
    {
        asked = Some(Asked::Trace(input));
    }
    let help = row_help(key);
    if !help.is_empty() {
        heading(ui, WHAT_IT_IS);
        ui.add(egui::Label::new(help).wrap());
    }
    let lines = sets_it_lines(&row, shown);
    if !lines.is_empty() {
        heading(ui, WHAT_SETS_IT);
        for line in &lines {
            weak_line(ui, line);
        }
    }
    let card = match (row.card, shown.consequences) {
        (Some(control), Some(consequences)) => consequences.card.control_lines(control, shown.auto),
        _ => Vec::new(),
    };
    if !card.is_empty() {
        heading(ui, WHAT_A_CHANGE_DOES);
        ui.label(CARD_HEADING);
        for line in &card {
            weak_line(ui, line);
        }
    }
    let touched: Vec<&CheckGuide> = row.input.map_or_else(Vec::new, |input| {
        CHECK_GUIDES
            .iter()
            .zip(shown.fixes)
            .filter(|(_, fixes)| fixes.iter().any(|&(path, _)| path == input))
            .map(|(guide, _)| guide)
            .collect()
    });
    if !touched.is_empty() {
        heading(ui, CHECKS_IT_TOUCHES);
        for guide in touched {
            asked = asked.or(check_link_ui(ui, guide, shown));
        }
    }
    let pulls = row
        .input
        .map_or_else(Vec::new, |input| input_trade_offs(shown.fixes, input));
    if !pulls.is_empty() {
        heading(ui, WATCH_OUT);
        for note in pulls {
            ui.add(egui::Label::new(note).wrap());
        }
    }
    // The cell of a row that is one input; Outside diameter and Overall length are sizes the
    // spreadsheet has no cell for, though their sliders write an input.
    let cell = (readouts.show_details() && row.input == Some(row.key)).then(|| {
        cell_line(
            InputCatalogue::get()
                .entry(row.key)
                .and_then(|entry| entry.meta.cell),
        )
    });
    asked.or(foot_ui(
        ui,
        row.lesson.map(Learn::Note),
        None,
        cell,
        readouts,
    ))
}

/// The entry of a row of the detailed inputs or the assumptions, drawn as a Size column row of
/// its own: keyed by its input's path, its label the inputs side's ([`label_for`]), no
/// torque-card row and no lesson. `None` for a path that is no input.
fn input_row(path: &str) -> Option<SizeRow> {
    let path = InputCatalogue::get().entry(path)?.path.as_str();
    Some(SizeRow {
        key: path,
        label: label_for(path),
        input: Some(path),
        card: None,
        lesson: None,
    })
}

/// What an input's fix link selects (spec 5.2): the input, by its path, or for an input the
/// Size column shows under its own name ([`SIZE_TAB_ALIASES`]: the ring radius) that control's
/// row, whose slider the link shows.
fn fix_selection(path: &'static str) -> Selection {
    let key = SIZE_TAB_ALIASES
        .iter()
        .find(|&&(aliased, _)| aliased == path)
        .and_then(|&(_, label)| SIZE_ROWS.iter().find(|row| row.label == label))
        .map_or(path, |row| row.key);
    Selection::Input(key)
}

/// An entry's foot line with the spreadsheet details on: [`SPREADSHEET_CELL`] and `cell`, or
/// the line that says the spreadsheet has none.
fn cell_line(cell: Option<&str>) -> String {
    format!("{SPREADSHEET_CELL} {}", cell.unwrap_or(NO_SPREADSHEET_CELL))
}

/// The row's value as its slider shows it, with its unit: the input's value (a blank, a
/// selector's choice or a text in words: [`value_text`]), or for the controls that are no one
/// input the size the slider shows (`dimensions`); none where the design has no size controls.
fn row_value(row: &SizeRow, shown: &Shown<'_>) -> Option<String> {
    let mm = |x: f64| value_with_unit(&Value::Num(x), format_value(&Value::Num(x)), "mm");
    match row.key {
        MINIMUM_OD_HINT_KEY | DIAMETER_HINT_KEY | LENGTH_HINT_KEY => {
            let size = dimensions(shown.inputs)?;
            Some(mm(match row.key {
                MINIMUM_OD_HINT_KEY => slider_minimum(&size, shown.minimum_diameter_mm),
                DIAMETER_HINT_KEY => size.diameter_mm,
                _ => size.overall_length_mm,
            }))
        }
        path => {
            let value = shown.inputs.get(path)?;
            let meta = InputCatalogue::get().entry(path)?.meta;
            let text = value_text(&value, meta);
            Some(match value {
                Value::Num(_) | Value::Int(_) if meta.choices.is_empty() => {
                    value_with_unit(&value, text, meta.unit)
                }
                _ => text,
            })
        }
    }
}

/// The lines the Size column drew under the row (spec 4.3): its hint; its what-moves-what lines
/// ([`Consequences`]); for Minimum OD the diameter range's floor; for Outside diameter and
/// Overall length, [`LINKED_HOUSING`] unless Auto sets the magnet count.
fn sets_it_lines(row: &SizeRow, shown: &Shown<'_>) -> Vec<String> {
    let mut lines: Vec<String> = hint_for(row.key).map(str::to_owned).into_iter().collect();
    if let Some(consequences) = shown.consequences {
        match row.key {
            DIAMETER_HINT_KEY => lines.extend(consequences.diameter_line()),
            LENGTH_HINT_KEY => lines.extend(consequences.length_line()),
            key if key == FreeVariable::MagnetsPerRing.path() => {
                lines.extend(consequences.magnet_lines(shown.inputs.coupling.npole, shown.auto))
            }
            _ => {}
        }
    }
    if row.key == MINIMUM_OD_HINT_KEY
        && let Some(size) = dimensions(shown.inputs)
    {
        lines.push(floor_line(size.minimum_diameter_range.min));
    }
    // Its last clause says the magnet count stays, which is false while Auto sets the count;
    // the gap's and the count's own entries learn nothing from it (owner question 52).
    if matches!(row.key, DIAMETER_HINT_KEY | LENGTH_HINT_KEY)
        && shown.auto != Some(FreeVariable::MagnetsPerRing)
    {
        lines.push(LINKED_HOUSING.to_owned());
    }
    lines
}

/// A check named in an entry: its state's mark, its label, a link that selects it, and its
/// state's word. Returns the selection a click asked for.
fn check_link_ui(
    ui: &mut egui::Ui,
    guide: &'static CheckGuide,
    shown: &Shown<'_>,
) -> Option<Asked> {
    let mut asked = None;
    wrapping_line(ui, |ui| {
        let mark = mark_level(guide, shown.inputs, shown.results);
        state_mark(ui, mark);
        if ui.link(guide.label).clicked() {
            asked = Some(Asked::Select(Selection::Check(guide.check)));
        }
        if let Some(mark) = mark {
            ui.colored_label(mark.color(ui.visuals()), mark.word());
        }
    });
    asked
}

/// The teaching notes shown, in their order ([`NOTES`]): every reviewed note but the one the
/// paused shaft-clamp section takes with it.
pub fn shown_notes(clamps_paused: bool) -> impl Iterator<Item = &'static Note> {
    NOTES
        .iter()
        .filter(move |note| shown_when(note.id, clamps_paused) && reviewed_note(note.id).is_some())
}

/// The note list ([`LEARN`]): each note shown, by its title, a link that selects it.
fn note_list_ui(ui: &mut egui::Ui, clamps_paused: bool) -> Option<Asked> {
    heading(ui, LEARN);
    let mut asked = None;
    for note in shown_notes(clamps_paused) {
        if ui.link(glyph_safe(note.title)).clicked() {
            asked = Some(Asked::Select(Selection::Note(note.id)));
        }
    }
    asked
}

/// A teaching note's entry: the links to the equations it explains ([`explains_ui`]), then the
/// note ([`note_ui`]) on the design shown. A change entry's input link shows its row, as a
/// check's fix does. Nothing for a note that is not reviewed.
fn note_entry_ui(
    ui: &mut egui::Ui,
    id: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let note = reviewed_note(id)?;
    let equation = explains_ui(ui, note);
    let view = NoteView {
        inputs: shown.inputs,
        results: shown.results,
        held: shown.held,
    };
    match note_ui(ui, note, &view) {
        Some(NoteLink::Input(path)) => readouts.focus_input(path),
        Some(NoteLink::Note(id)) => return Some(Asked::Select(Selection::Note(id))),
        None => {}
    }
    equation.map(Asked::Equation)
}

/// The orientation (spec 5.4): what the thing is, the three moves, and the link to the Start
/// here walk.
fn orientation_ui(ui: &mut egui::Ui, clamps_paused: bool) -> Option<Asked> {
    ui.add(egui::Label::new(ORIENTATION).wrap());
    ui.add_space(4.0);
    for step in STEPS {
        ui.add(egui::Label::new(step).wrap());
    }
    ui.add_space(4.0);
    ui.link(START_HERE_BUTTON)
        .on_hover_text(start_here_hover(clamps_paused))
        .clicked()
        .then_some(Asked::StartHere)
}

/// One of an entry's headings.
fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(4.0);
    ui.colored_label(look::quiet_text(ui.visuals()), text);
}

/// What a check's entry is drawn with: its guide, the state its row shows
/// ([`mark_level`]), its level, the fixes to show (empty: the no-fix line) and the notes on
/// them ([`guide_notes`]), read only where the check does not pass.
struct Entry<'a> {
    guide: &'a CheckGuide,
    mark: Option<Level>,
    level: Level,
    fixes: &'a [(&'static str, Direction)],
    notes: &'a dyn Fn() -> Vec<String>,
}

/// The entry of the check `check` in the design shown ([`entry_ui`]); nothing for a path that
/// is no guide's check.
fn check_ui(
    ui: &mut egui::Ui,
    check: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let index = CHECK_GUIDES.iter().position(|g| g.check == check)?;
    let guide = &CHECK_GUIDES[index];
    let notes = || guide_notes(shown.fixes, index, shown.inputs, shown.results);
    let entry = Entry {
        guide,
        mark: mark_level(guide, shown.inputs, shown.results),
        level: guidance::level(guide, shown.inputs, shown.results),
        fixes: shown.fixes.get(index).map_or(&[], Vec::as_slice),
        notes: &notes,
    };
    entry_ui(ui, &entry, shown, readouts)
}

/// A check's entry: its state and name; why it matters; then, where it does not pass, what is
/// wrong in this design (in the level's colour), the fixes, each a link that selects its input
/// and shows its row ([`fix_selection`]), or the line saying no listed change helps, and the
/// notes on those fixes; the numbers the check reads, each a readout; and the foot links, with
/// the spreadsheet details the cell of a check that is a result.
fn entry_ui(
    ui: &mut egui::Ui,
    entry: &Entry<'_>,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let guide = entry.guide;
    let mut asked = None;
    wrapping_line(ui, |ui| {
        state_mark(ui, entry.mark);
        ui.strong(guide.label);
        if let Some(mark) = entry.mark {
            ui.colored_label(mark.color(ui.visuals()), mark.word());
        }
    });
    heading(ui, WHAT_IT_IS);
    ui.add(egui::Label::new(guide.why).wrap());
    if entry.level != Level::Good {
        if let Some(problem) = (guide.problem)(shown.inputs, shown.results) {
            heading(ui, WHAT_SETS_IT);
            let text = egui::RichText::new(problem).color(entry.level.color(ui.visuals()));
            ui.add(egui::Label::new(text).wrap());
        }
        heading(ui, WHAT_TO_CHANGE);
        if entry.fixes.is_empty() {
            ui.add(egui::Label::new(NO_FIX).wrap());
        }
        for &(path, direction) in entry.fixes {
            if ui.link(fix_text(path, direction)).clicked() {
                readouts.focus_input(path);
                asked = Some(Asked::Select(fix_selection(path)));
            }
        }
        let notes = (entry.notes)();
        if !notes.is_empty() {
            heading(ui, WATCH_OUT);
            for note in notes {
                ui.add(egui::Label::new(note).wrap());
            }
        }
    }
    heading(ui, NUMBERS);
    numbers_ui(ui, guide, entry.level, shown, readouts);
    let info = result_info(guide.check);
    let equation = info.map(|_| guide.check);
    let cell = info
        .filter(|_| readouts.show_details())
        .map(|info| cell_line(info.cell.as_deref()));
    asked.or(foot_ui(ui, Some(guide.learn), equation, cell, readouts))
}

/// The numbers behind `guide`'s check: the dashboard rows its verdict badges, drawn as the
/// dashboard drew them (each a readout), with the stored-3D line under the rows that read the
/// stored fields; for the Magnets fit check, which no result holds, whether the blocks fit.
fn numbers_ui(
    ui: &mut egui::Ui,
    guide: &CheckGuide,
    level: Level,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) {
    if guide.check == FIT_CHECK {
        let value = if level == Level::Bad {
            DOES_NOT_FIT
        } else {
            FITS
        };
        ui.add(egui::Label::new(egui::RichText::new(value).strong()).wrap());
        return;
    }
    if readouts.show_details() {
        ui.add(egui::Label::new(egui::RichText::new(MARKER_LEGEND).small().weak()).wrap());
    }
    let mut stored_3d = false;
    for line in shown_lines(shown.results, shown.clamps_paused) {
        if badging_check(line.path) == Some(guide.check) {
            line_ui(ui, &line, readouts);
            stored_3d |= line.stored_3d;
        }
    }
    if stored_3d {
        ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    }
}

/// The entry of the result at `path` (spec 5.3): a tile's, a dimension's, a callout's or a
/// results-table row's. Its number and its name (a tile's as the tile draws them, another's as
/// its results-table row does); what it is (the result's help text); for the hot-low torque,
/// what a change does (the torque card, [`card_ui`]); the check that badges its row, a link
/// that selects it; for a row that reads the stored 3D fields, the line that says so ("Watch
/// out"); and the foot links (its row's lesson where it has one, the Equation link where it
/// has an equation), with the spreadsheet details its cell. Nothing for a path that is no
/// result.
fn result_ui(
    ui: &mut egui::Ui,
    path: &'static str,
    shown: &Shown<'_>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    let info = result_info(path)?;
    let tile = Selection::Result(path).tile();
    wrapping_line(ui, |ui| {
        let quiet = look::quiet_text(ui.visuals());
        match tile {
            Some(tile) => {
                match tile.number(shown.inputs, shown.results) {
                    Ok(number) => ui.strong(number),
                    Err(words) => ui.weak(words),
                };
                ui.colored_label(quiet, tile.label(shown.results));
            }
            None => {
                let value = shown.results.get(path).unwrap_or(Value::None);
                let text = shown_text(path, &value);
                ui.strong(value_with_unit(&value, text, info.meta.unit));
                ui.colored_label(quiet, display_label(path, info.meta.label));
            }
        }
    });
    let help = help_for(path);
    if !help.is_empty() {
        heading(ui, WHAT_IT_IS);
        ui.add(egui::Label::new(help).wrap());
    }
    if tile == Some(Tile::HotLow)
        && let Some(consequences) = shown.consequences
    {
        heading(ui, WHAT_A_CHANGE_DOES);
        card_ui(ui, consequences, shown.auto, readouts);
    }
    let mut asked = None;
    if let Some(guide) = badging_check(path).and_then(guidance::guide) {
        heading(ui, CHECKS_IT_TOUCHES);
        asked = check_link_ui(ui, guide, shown);
    }
    if STORED_3D_ROWS.contains(&path) {
        heading(ui, WATCH_OUT);
        ui.weak(STORED_3D_LABEL).on_hover_text(STORED_3D_NOTE);
    }
    let cell = readouts
        .show_details()
        .then(|| cell_line(info.cell.as_deref()));
    let equation = registry().equation_for(path).map(|_| path);
    asked.or(foot_ui(ui, row_learn(path), equation, cell, readouts))
}

/// What a click on the result at `path` selects (spec 5.2): the check whose own result it is
/// ([`guidance::guide`]), else the result.
pub(crate) fn result_selection(path: &'static str) -> Selection {
    match guidance::guide(path) {
        Some(guide) => Selection::Check(guide.check),
        None => Selection::Result(path),
    }
}

/// The torque card (slice 3 spec 6.1; decision C-10 puts it here): its heading with the Why
/// link to how the coupling makes torque, then its lines for the design shown, weak, as the
/// Size tab drew them.
fn card_ui(
    ui: &mut egui::Ui,
    consequences: &Consequences,
    auto: Option<FreeVariable>,
    readouts: &mut Readouts,
) {
    wrapping_line(ui, |ui| {
        ui.label(CARD_HEADING);
        if why_note_link(ui, SIZE_TAB_NOTE) {
            readouts.open_note(SIZE_TAB_NOTE);
        }
    });
    for line in consequences.card.lines(auto) {
        weak_line(ui, &line);
    }
}

/// An entry's foot: [`FULL_LESSON`] and where `learn` leads ([`learn_ui`]), for an entry
/// that has a lesson; for an entry with a result (`equation`), the [`EQUATION`] link; then its
/// `cell` line, weak. Returns what the Equation link asked.
fn foot_ui(
    ui: &mut egui::Ui,
    learn: Option<Learn>,
    equation: Option<&'static str>,
    cell: Option<String>,
    readouts: &mut Readouts,
) -> Option<Asked> {
    ui.separator();
    if let Some(learn) = learn {
        wrapping_line(ui, |ui| {
            ui.label(FULL_LESSON);
            learn_ui(ui, learn, readouts);
        });
    }
    let asked =
        equation.and_then(|path| ui.link(EQUATION).clicked().then_some(Asked::Equation(path)));
    if let Some(cell) = cell {
        ui.add(egui::Label::new(egui::RichText::new(cell).small().weak()).wrap());
    }
    asked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::explain::plain::mentions_gearbox_protection;
    use crate::gui::dashboard::{
        DASHBOARD, NO_NOTE_YET, NOTE_NOT_REVIEWED, dashboard_lines, verdict_display,
    };
    use crate::gui::explorer::reviewed_note;
    use crate::gui::guidance::FACE_GAP;
    use crate::gui::guidance::{Fixes, all_fixes, counted_guides, plain_text_violations};
    use crate::gui::plain_text::label_for;
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::size_controls::{MINIMUM_HOVER, SIZE_ROWS};
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, flat_shapes, glyph_context, passing_design, primary_button,
        short_magnets, sized_frame, text_color, text_rect, tooltips_at_once, with_poles,
    };
    use crate::gui::typeset::{TermColors, glyph_safe};

    /// The defaults with `edit` applied.
    fn design(edit: impl FnOnce(&mut DesignInputs)) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        edit(&mut inputs);
        inputs
    }

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(crate::gui::test_support::hot_day)
    }

    /// The defaults under a 30 mm overall-length claim: the hot-low torque needs longer magnets
    /// and the space claim shorter ones, both failing (a note on each).
    fn length_conflict() -> DesignInputs {
        design(|i| i.metal.max_overall_axial_mm = 30.0)
    }

    /// Designs that between them fail or caution every check.
    fn designs() -> Vec<DesignInputs> {
        vec![
            DesignInputs::default(),
            short_magnets(),
            passing_design(),
            hot_day(),
            with_poles(40),
            length_conflict(),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
        ]
    }

    /// Frames of the Explain panel showing `selection` on `inputs` in `ctx`, a first one with
    /// no input, then one per entry of `events`: what the last drew, what it asked for and
    /// what the readouts heard in it.
    fn frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        selection: Selection,
        show_details: bool,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>, ReadoutEvents) {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, false);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::new(TermColors::default(), show_details);
            let mut asked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 4000.0), input, |ui| {
                let shown = Shown {
                    inputs,
                    results: &results,
                    fixes: &fixes,
                    clamps_paused: false,
                    consequences: None,
                    auto: None,
                    held: &|_| None,
                    minimum_diameter_mm: None,
                };
                asked = explain_ui(ui, selection, &shown, &mut readouts);
            });
            last = Some((output, asked, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Frames of the entry of the Size column's row `key` on `inputs` in `ctx`, with the Size
    /// tab's consequences at the Minimum OD `minimum`, Auto on `auto` (`None`: off) and the
    /// spreadsheet details `show_details`: a first one with no input, then one per entry of
    /// `events`. What the last drew and what it asked for.
    fn input_frames(
        ctx: &egui::Context,
        inputs: &DesignInputs,
        key: &'static str,
        show_details: bool,
        minimum: Option<f64>,
        auto: Option<FreeVariable>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>) {
        let results = compute_all(inputs);
        let fixes = all_fixes(inputs, &results, false);
        let consequences = Consequences::of(inputs, &results, minimum, false);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::new(TermColors::default(), show_details);
            let mut asked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 4000.0), input, |ui| {
                let shown = Shown {
                    inputs,
                    results: &results,
                    fixes: &fixes,
                    clamps_paused: false,
                    consequences: Some(&consequences),
                    auto,
                    held: &|_| None,
                    minimum_diameter_mm: minimum,
                };
                asked = explain_ui(ui, Selection::Input(key), &shown, &mut readouts);
            });
            last = Some((output, asked));
        }
        last.expect("one frame at least")
    }

    /// What the entry of the Size column's row `key` draws on `inputs` ([`input_frames`]).
    fn input_texts(inputs: &DesignInputs, key: &'static str, show_details: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        drawn_texts(&input_frames(&ctx, inputs, key, show_details, None, None, vec![]).0)
    }

    /// What the entry of `check` draws on `inputs`.
    fn entry_texts(inputs: &DesignInputs, check: &'static str, show_details: bool) -> Vec<String> {
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, inputs, Selection::Check(check), show_details, vec![]);
        drawn_texts(&output)
    }

    /// The frames of a click at `at` (move, press, release).
    fn click_at(at: egui::Pos2) -> Vec<Vec<egui::Event>> {
        vec![
            vec![egui::Event::PointerMoved(at)],
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ]
    }

    /// A click on `text` in the Explain panel showing `selection` on `inputs`: what it asked
    /// for and what the readouts heard.
    fn click(
        inputs: &DesignInputs,
        selection: Selection,
        text: &str,
    ) -> (Option<Asked>, ReadoutEvents) {
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, inputs, selection, false, vec![]);
        let at = text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center();
        let (_, asked, events) = frames(&ctx, inputs, selection, false, click_at(at));
        (asked, events)
    }

    fn count(texts: &[String], text: &str) -> usize {
        texts.iter().filter(|t| *t == text).count()
    }

    // ------------------------------------------------------------------ the orientation

    #[test]
    fn nothing_selected_shows_the_orientation_in_at_most_sixty_words() {
        // Spec 5.4: what the thing is, then the three moves, then the link to Start here.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, asked, _) = frames(&ctx, &inputs, Selection::Nothing, false, vec![]);
        assert_eq!(asked, None);
        let mut want = vec![HEADING, ORIENTATION];
        want.extend(STEPS);
        want.push(START_HERE_BUTTON);
        assert_eq!(drawn_texts(&output), want);
        let words: usize = std::iter::once(ORIENTATION)
            .chain(STEPS)
            .map(|text| text.split_whitespace().count())
            .sum();
        assert!(words <= 60, "{words} words");
        // The link starts the walk; no About link shows while nothing is selected.
        let (asked, events) = click(&inputs, Selection::Nothing, START_HERE_BUTTON);
        assert_eq!(asked, Some(Asked::StartHere));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn the_panel_s_own_strings_keep_the_text_rules() {
        // Spec 8.3: plain words, glyphs in the app's font, nothing about protecting the
        // gearbox, no pressure-boundary word (the tool models none) and no ring named by a
        // colour.
        let ctx = glyph_context();
        let mut texts = vec![
            HEADING,
            ORIENTATION,
            ABOUT,
            WHAT_IT_IS,
            WHAT_SETS_IT,
            WHAT_A_CHANGE_DOES,
            CHECKS_IT_TOUCHES,
            NUMBERS,
            FULL_LESSON,
            EQUATION,
            LEARN,
            TRACE_LINK,
            crate::gui::checks::HEADING,
            crate::gui::results_table::TRACED_ONLY_HELP,
        ];
        texts.extend(STEPS);
        for text in texts {
            assert_glyphs(&ctx, text, "the Explain panel");
            assert!(plain_text_violations(text).is_empty(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            let lower = text.to_lowercase();
            for banned in ["sealed", "containment", "leak", "red ring", "blue ring"] {
                assert!(!lower.contains(banned), "{text:?} says {banned:?}");
            }
        }
        // A label's click selects an input; only its entry's link traces it (spec 5.2).
        assert!(
            !crate::gui::results_table::TRACED_ONLY_HELP.contains("label"),
            "the trace filter's hover names the link, not the label"
        );
        // The orientation says what the notes say: torque across a gap without touching, and
        // slip past the pull-out torque.
        for words in ["without touching", "pull-out torque", "slip"] {
            assert!(ORIENTATION.contains(words), "{words}");
        }
    }

    // ------------------------------------------------------------------ the notes

    #[test]
    fn learn_lists_every_shown_note_by_title_and_a_title_selects_it() {
        // UX overhaul spec 4.2: Learn lists the shown notes by title here; a title opens the
        // note here. The paused shaft-clamp section takes its note with it.
        use crate::engine::explain::notes::PAUSED_NOTE;
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let (output, asked, _) = frames(&ctx, &inputs, Selection::NoteList, false, vec![]);
        assert_eq!(asked, None);
        let mut want = vec![HEADING.to_owned(), ABOUT.to_owned(), LEARN.to_owned()];
        want.extend(shown_notes(false).map(|note| glyph_safe(note.title)));
        assert_eq!(drawn_texts(&output), want);
        // Every reviewed note, in the notes' order; while paused, all but the clamp's.
        let reviewed: Vec<&str> = NOTES
            .iter()
            .filter(|n| reviewed_note(n.id).is_some())
            .map(|n| n.id)
            .collect();
        assert_eq!(
            shown_notes(false).map(|n| n.id).collect::<Vec<_>>(),
            reviewed
        );
        let paused: Vec<&str> = shown_notes(true).map(|n| n.id).collect();
        assert!(!paused.contains(&PAUSED_NOTE));
        assert_eq!(paused.len(), reviewed.len() - 1);
        // A title selects its note.
        let first = shown_notes(false).next().unwrap();
        let (asked, events) = click(&inputs, Selection::NoteList, &glyph_safe(first.title));
        assert_eq!(asked, Some(Asked::Select(Selection::Note(first.id))));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn a_note_s_entry_is_the_note_with_the_links_to_its_equations() {
        // Decision C-11: the note as the Equation panel drew it, after the links to the
        // equations it explains; an equation link opens it in the Equation panel and leaves
        // the selection; a change entry's input shows its row, as a check's fix does.
        use crate::gui::explorer::crumb;
        use crate::gui::notes_view::{EXPLAINS, WHERE_FROM};
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let note = reviewed_note("pullout_angle").unwrap();
        let selection = Selection::Note(note.id);
        let (output, asked, _) = frames(&ctx, &inputs, selection, false, vec![]);
        assert_eq!(asked, None);
        let texts = drawn_texts(&output);
        for text in [EXPLAINS, WHAT_TO_CHANGE, WHERE_FROM] {
            assert_eq!(count(&texts, text), 1, "{text} in {texts:?}");
        }
        assert_eq!(count(&texts, &glyph_safe(note.title)), 1, "{texts:?}");
        // The first equation it explains that has a record (a family's template has none).
        let equation = *note
            .equations
            .iter()
            .find(|p| crate::gui::readouts::registry().equation_for(p).is_some())
            .unwrap();
        let (asked, events) = click(&inputs, selection, &crumb(equation));
        assert_eq!(asked, Some(Asked::Equation(equation)));
        assert_eq!(events, ReadoutEvents::default());
        let input = note.change[0].input;
        let (asked, events) = click(&inputs, selection, &glyph_safe(label_for(input)));
        assert_eq!(asked, None);
        assert_eq!(events.focus, Some(input));
        // A note that is not reviewed, or no note: the heading and the way back alone.
        let (output, _, _) = frames(
            &ctx,
            &inputs,
            Selection::Note("no.such.note"),
            false,
            vec![],
        );
        assert_eq!(drawn_texts(&output), [HEADING, ABOUT]);
    }

    #[test]
    fn about_this_tool_returns_to_the_orientation() {
        let inputs = DesignInputs::default();
        let selection = Selection::Check(CHECK_GUIDES[0].check);
        let (asked, events) = click(&inputs, selection, ABOUT);
        assert_eq!(asked, Some(Asked::Select(Selection::Nothing)));
        assert_eq!(events, ReadoutEvents::default());
        // A path that is no guide's check draws the heading and the link alone.
        let texts = entry_texts(&inputs, "model.pullout_Nm", false);
        assert_eq!(texts, [HEADING, ABOUT]);
    }

    // ------------------------------------------------------------------ a check's entry

    /// The cup wall fix the test guide offers.
    const CUP_WALL_FIX: [(&str, Direction); 1] = [("metal.cup_wall_corner_mm", Direction::Raise)];

    /// A guide of the cup wall check, the test's own, linking to `learn`.
    fn guide_with(learn: Learn) -> CheckGuide {
        CheckGuide {
            check: "materials.cup_wall_check",
            label: "Cup wall",
            problem: |_, results| Some(results.materials.cup_wall_check.clone()),
            why: "A test sentence.",
            fixes: Fixes::Static(&CUP_WALL_FIX),
            learn,
        }
    }

    /// The test guide's entry at `level`, with `fixes` and `notes`.
    fn entry_at<'a>(
        guide: &'a CheckGuide,
        level: Level,
        fixes: &'a [(&'static str, Direction)],
        notes: &'a dyn Fn() -> Vec<String>,
    ) -> Entry<'a> {
        Entry {
            guide,
            mark: Some(level),
            level,
            fixes,
            notes,
        }
    }

    /// Frames of `entry` on the defaults in `ctx`, a first one with no input, then one per
    /// entry of `events`: what the last drew, what it asked for and what the readouts heard.
    fn entry_frames(
        ctx: &egui::Context,
        entry: &Entry<'_>,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, Option<Asked>, ReadoutEvents) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::default();
            let mut asked = None;
            let output = sized_frame(ctx, egui::vec2(320.0, 1200.0), input, |ui| {
                let shown = Shown {
                    inputs: &inputs,
                    results: &results,
                    fixes: &[],
                    clamps_paused: false,
                    consequences: None,
                    auto: None,
                    held: &|_| None,
                    minimum_diameter_mm: None,
                };
                asked = entry_ui(ui, entry, &shown, &mut readouts);
            });
            last = Some((output, asked, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Where `entry` draws `text`.
    fn entry_text_at(ctx: &egui::Context, entry: &Entry<'_>, text: &str) -> egui::Pos2 {
        let (output, _, _) = entry_frames(ctx, entry, Vec::new());
        text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center()
    }

    /// A click on `text` drawn by `entry`: what it asked for and what the readouts heard.
    fn click_in_entry(entry: &Entry<'_>, text: &str) -> (Option<Asked>, ReadoutEvents) {
        let ctx = egui::Context::default();
        let at = entry_text_at(&ctx, entry, text);
        let (_, asked, events) = entry_frames(&ctx, entry, click_at(at));
        (asked, events)
    }

    /// The texts drawn while `text` of `entry` is hovered, tooltips shown at once.
    fn hover_in_entry(entry: &Entry<'_>, text: &str) -> Vec<String> {
        let ctx = egui::Context::default();
        tooltips_at_once(&ctx);
        let at = entry_text_at(&ctx, entry, text);
        let hover = vec![vec![egui::Event::PointerMoved(at)], Vec::new()];
        drawn_texts(&entry_frames(&ctx, entry, hover).0)
    }

    #[test]
    fn a_lesson_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row() {
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        // A reviewed note: its title (glyph-safe) asks for the note.
        let note = reviewed_note("end_effect").expect("a reviewed note");
        let guide = guide_with(Learn::Note(note.id));
        let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (asked, events) = click_in_entry(&green, &glyph_safe(note.title));
        assert_eq!(
            (asked, events.note, events.focus),
            (None, Some(note.id), None)
        );
        // An assumption: its label asks for its first input's row; its hover text is the
        // rationale.
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        let guide = guide_with(Learn::Assumption(assumption.id));
        let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (_, events) = click_in_entry(&green, assumption.label);
        assert_eq!(
            (events.note, events.focus),
            (None, Some(assumption.paths[0]))
        );
        let texts = hover_in_entry(&green, assumption.label);
        assert!(texts.iter().any(|t| t == assumption.rationale));
        // No note, or a note that has not passed review (no note has this id): "no note yet",
        // which asks for nothing; its hover text says why.
        for (learn, reason) in [
            (Learn::None("A test reason."), "A test reason."),
            (Learn::Note("no_such_note"), NOTE_NOT_REVIEWED),
        ] {
            let guide = guide_with(learn);
            let green = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
            let (asked, events) = click_in_entry(&green, NO_NOTE_YET);
            assert_eq!((asked, events), (None, ReadoutEvents::default()));
            assert!(hover_in_entry(&green, NO_NOTE_YET).contains(&reason.to_owned()));
        }
        // A failing check's fix asks for its input's row; a two-pick magnet fix for its part's.
        let guide = guide_with(Learn::None("A test reason."));
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let red = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, none);
        let (_, events) = click_in_entry(&red, &fix);
        assert_eq!(
            (events.note, events.focus),
            (None, Some("metal.cup_wall_corner_mm"))
        );
        let part = [("coupling.magnets.part_inner", Direction::CustomHotterGrade)];
        let two_picks = entry_at(&guide, Level::Bad, &part, none);
        let (_, events) = click_in_entry(&two_picks, &fix_text(part[0].0, part[0].1));
        assert_eq!(events.focus, Some("coupling.magnets.part_inner"));
        // The Equation link asks for the check's result, and nothing of the readouts.
        let (asked, events) = click_in_entry(&red, EQUATION);
        assert_eq!(asked, Some(Asked::Equation("materials.cup_wall_check")));
        assert_eq!(events, ReadoutEvents::default());
    }

    #[test]
    fn an_entry_shows_its_problem_and_fixes_where_the_check_does_not_pass() {
        let guide = guide_with(Learn::None("A test reason."));
        let problem = compute_all(&DesignInputs::default())
            .materials
            .cup_wall_check;
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let y = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text)
                .unwrap_or_else(|| panic!("no {text:?}"))
                .top()
        };
        // Red and amber alike (an amber check's entry is open: its row was clicked): the
        // headings in the plan's order (decision D-18), the problem in the level's colour, the
        // state word.
        for level in [Level::Bad, Level::Caution] {
            let entry = entry_at(&guide, level, &CUP_WALL_FIX, none);
            let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
            let order = [
                guide.label,
                WHAT_IT_IS,
                guide.why,
                WHAT_SETS_IT,
                problem.as_str(),
                WHAT_TO_CHANGE,
                fix.as_str(),
                NUMBERS,
                FULL_LESSON,
                EQUATION,
            ];
            for pair in order.windows(2) {
                assert!(
                    y(&output, pair[0]) < y(&output, pair[1]),
                    "{level:?}: {:?} is not above {:?}",
                    pair[0],
                    pair[1]
                );
            }
            assert_eq!(text_color(&output, &problem), Some(level.color(&visuals)));
            assert_eq!(
                text_color(&output, level.word()),
                Some(level.color(&visuals))
            );
            assert_eq!(count(&drawn_texts(&output), WATCH_OUT), 0);
        }
        // Green: why it matters, its numbers and its foot; no problem and no fix.
        let entry = entry_at(&guide, Level::Good, &CUP_WALL_FIX, none);
        let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
        let texts = drawn_texts(&output);
        for want in [
            guide.label,
            Level::Good.word(),
            WHAT_IT_IS,
            guide.why,
            NUMBERS,
        ] {
            assert_eq!(count(&texts, want), 1, "{want:?}");
        }
        for absent in [WHAT_SETS_IT, WHAT_TO_CHANGE, WATCH_OUT, fix.as_str()] {
            assert_eq!(count(&texts, absent), 0, "{absent:?}");
        }
    }

    #[test]
    fn a_failing_entry_without_a_fix_says_so_and_its_notes_follow_under_watch_out() {
        let guide = guide_with(Learn::None("A test reason."));
        let ctx = egui::Context::default();
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let (output, _, _) = entry_frames(&ctx, &entry_at(&guide, Level::Bad, &[], none), vec![]);
        let texts = drawn_texts(&output);
        assert_eq!(count(&texts, WHAT_TO_CHANGE), 1);
        assert_eq!(count(&texts, NO_FIX), 1);
        let note = "A test trade-off.".to_owned();
        let notes = || vec![note.clone()];
        let noted = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, &notes);
        let (output, _, _) = entry_frames(&ctx, &noted, Vec::new());
        assert_eq!(count(&drawn_texts(&output), NO_FIX), 0);
        let fix = fix_text(CUP_WALL_FIX[0].0, CUP_WALL_FIX[0].1);
        let y = |text: &str| text_rect(&output, text).unwrap().top();
        assert!(y(&fix) < y(WATCH_OUT) && y(WATCH_OUT) < y(&note));
    }

    #[test]
    fn an_entry_reads_its_notes_only_where_its_check_does_not_pass() {
        let guide = guide_with(Learn::None("A test reason."));
        let reads = std::cell::Cell::new(0);
        let counting = || {
            reads.set(reads.get() + 1);
            vec!["A test trade-off.".to_owned()]
        };
        for (level, want) in [(Level::Good, 0), (Level::Caution, 1), (Level::Bad, 1)] {
            reads.set(0);
            let entry = entry_at(&guide, level, &CUP_WALL_FIX, &counting);
            let ctx = egui::Context::default();
            let (output, _, _) = entry_frames(&ctx, &entry, Vec::new());
            assert_eq!(reads.get(), want, "{level:?}");
            let shown = count(&drawn_texts(&output), "A test trade-off.");
            assert_eq!(shown, want, "{level:?}");
        }
    }

    /// The text of the link where `learn` leads.
    fn learn_link_text(learn: Learn) -> String {
        match learn {
            Learn::Note(id) => {
                reviewed_note(id).map_or(NO_NOTE_YET.to_owned(), |n| glyph_safe(n.title))
            }
            Learn::Assumption(id) => ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .label
                .to_owned(),
            Learn::None(_) => NO_NOTE_YET.to_owned(),
        }
    }

    #[test]
    fn each_check_s_entry_holds_the_problem_why_and_fixes_its_dashboard_block_held() {
        // Spec 8.2, test 11: the same strings and the same links, for every guide on designs
        // that between them fail or caution every check. An amber check's entry is open too:
        // its row was clicked.
        for inputs in designs() {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            for (index, guide) in CHECK_GUIDES.iter().enumerate() {
                let texts = entry_texts(&inputs, guide.check, false);
                let what = guide.check;
                let level = guidance::level(guide, &inputs, &results);
                let failing = usize::from(level != Level::Good);
                // (The hot minimum's label is also its torque row's.)
                assert!(count(&texts, guide.label) >= 1, "{what}");
                assert_eq!(count(&texts, guide.why), 1, "{what}");
                let problem = (guide.problem)(&inputs, &results);
                assert_eq!(problem.is_some(), level != Level::Good, "{what}");
                assert_eq!(count(&texts, WHAT_SETS_IT), failing, "{what}");
                if let Some(problem) = &problem {
                    assert_eq!(count(&texts, problem), 1, "{what}: {problem:?}");
                }
                assert_eq!(count(&texts, WHAT_TO_CHANGE), failing, "{what}");
                for &(path, direction) in &fixes[index] {
                    assert_eq!(count(&texts, &fix_text(path, direction)), 1, "{what}");
                }
                let no_fix = usize::from(level != Level::Good && fixes[index].is_empty());
                assert_eq!(count(&texts, NO_FIX), no_fix, "{what}");
                let notes = if level == Level::Good {
                    Vec::new()
                } else {
                    guide_notes(&fixes, index, &inputs, &results)
                };
                assert_eq!(count(&texts, WATCH_OUT), usize::from(!notes.is_empty()));
                for note in &notes {
                    let wanted = notes.iter().filter(|n| *n == note).count();
                    assert_eq!(count(&texts, note), wanted, "{what}: {note:?}");
                }
                assert_eq!(count(&texts, FULL_LESSON), 1, "{what}");
                assert_eq!(count(&texts, &learn_link_text(guide.learn)), 1, "{what}");
                let equation = usize::from(guide.check != FIT_CHECK);
                assert_eq!(count(&texts, EQUATION), equation, "{what}");
            }
        }
        // The designs above reach a note and an amber check.
        let conflict = length_conflict();
        let results = compute_all(&conflict);
        let fixes = all_fixes(&conflict, &results, false);
        assert!(
            (0..CHECK_GUIDES.len())
                .any(|i| !guide_notes(&fixes, i, &conflict, &results).is_empty())
        );
        let hot = hot_day();
        let results = compute_all(&hot);
        assert!(
            CHECK_GUIDES
                .iter()
                .any(|g| guidance::level(g, &hot, &results) == Level::Caution)
        );
    }

    #[test]
    fn the_numbers_behind_a_check_are_the_dashboard_rows_its_verdict_badges() {
        // Every dashboard row with a check is in that check's entry, under its display label
        // with its value as the dashboard drew it; a row no check badges is in none.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let lines = dashboard_lines(&results);
        for &(path, check) in &DASHBOARD {
            let line = lines.iter().find(|line| line.path == path).unwrap();
            for guide in &CHECK_GUIDES {
                let texts = entry_texts(&inputs, guide.check, false);
                let drawn = count(&texts, &line.shown_value());
                if check == Some(guide.check) {
                    assert!(drawn >= 1, "{path} in {}", guide.check);
                    assert!(count(&texts, label_for(path)) >= 1, "{path}");
                } else if check.is_none() {
                    assert_eq!(drawn, 0, "{path} in {}", guide.check);
                }
            }
        }
        // The cup wall's failing verdict reads in plain words (ruling R9), not the engine's.
        let shown = verdict_display(
            "materials.cup_wall_check",
            &results.materials.cup_wall_check,
        )
        .expect("the default cup wall fails");
        let texts = entry_texts(&inputs, "materials.cup_wall_check", false);
        assert_eq!(count(&texts, &shown), 1, "{shown}");
        let engine_words = crate::engine::materials::CUP_WALL_TOO_THIN.trim_end();
        assert!(!texts.iter().any(|t| t.contains(engine_words)));
        // The stored-3D line closes the numbers of the two checks that read the stored fields
        // (owner R-15), and of no other.
        for guide in &CHECK_GUIDES {
            let stored = ["temperature.summary.verdict", "shaft_joint.check"];
            let texts = entry_texts(&inputs, guide.check, false);
            assert_eq!(
                count(&texts, STORED_3D_LABEL),
                usize::from(stored.contains(&guide.check)),
                "{}",
                guide.check
            );
        }
        // The correction ids and their legend show only with the spreadsheet details.
        let hot_min = "metal.hot_min_check";
        let markers: Vec<String> = lines
            .iter()
            .filter(|line| !line.marker.is_empty())
            .filter(|line| DASHBOARD.contains(&(line.path, Some(hot_min))))
            .map(|line| line.marker.clone())
            .collect();
        assert!(!markers.is_empty(), "the hot minimum's rows carry ids");
        for show_details in [false, true] {
            let texts = entry_texts(&inputs, hot_min, show_details);
            for marker in &markers {
                assert_eq!(texts.contains(marker), show_details, "{marker}");
            }
            assert_eq!(count(&texts, MARKER_LEGEND), usize::from(show_details));
        }
    }

    #[test]
    fn the_fit_entry_says_whether_the_blocks_fit_and_guides_a_misfit() {
        let fit = guidance::guide(FIT_CHECK).unwrap();
        let texts = entry_texts(&DesignInputs::default(), FIT_CHECK, false);
        assert_eq!(count(&texts, fit.label), 1);
        assert_eq!((count(&texts, FITS), count(&texts, DOES_NOT_FIT)), (1, 0));
        assert_eq!(count(&texts, Level::Good.word()), 1);
        // 40 poles: the blocks do not fit; the fail mark, the problem and the guide's fixes.
        let inputs = with_poles(40);
        let results = compute_all(&inputs);
        assert_eq!(guidance::level(fit, &inputs, &results), Level::Bad);
        let ctx = egui::Context::default();
        let (output, _, _) = frames(&ctx, &inputs, Selection::Check(FIT_CHECK), false, vec![]);
        let texts = drawn_texts(&output);
        assert_eq!((count(&texts, FITS), count(&texts, DOES_NOT_FIT)), (0, 1));
        let problem = (fit.problem)(&inputs, &results).expect("a misfit has a Problem");
        assert_eq!(count(&texts, &problem), 1);
        // The entry's mark is the fail shape: a filled disc in the failing colour.
        let visuals = ctx.style().visuals.clone();
        let disc = flat_shapes(&output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == 5.0 => Some(c.fill),
                _ => None,
            });
        assert_eq!(disc, Some(Level::Bad.color(&visuals)));
    }

    #[test]
    fn the_numbers_of_a_passing_check_carry_the_pass_shape_and_never_the_fail_shape() {
        // Decision C-4 (plan review E-1): a filled disc means fails, a ring passes. A passing
        // check's entry draws a ring by its name and one by each row it badges, and no disc.
        let ctx = egui::Context::default();
        let good = Level::Good.color(&ctx.style().visuals);
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let check = "metal.hot_min_check";
        let rows = dashboard_lines(&results)
            .into_iter()
            .filter(|line| DASHBOARD.contains(&(line.path, Some(check))))
            .inspect(|line| assert_eq!(line.level, Some(Level::Good), "{}", line.path))
            .count();
        assert_eq!(rows, 2, "the hot-low torque and its check");
        let (output, _, _) = frames(&ctx, &inputs, Selection::Check(check), false, vec![]);
        let shapes = flat_shapes(&output);
        let circles = |filled: bool| {
            shapes
                .iter()
                .filter(|shape| match shape {
                    egui::Shape::Circle(c) if filled => c.fill == good,
                    egui::Shape::Circle(c) => {
                        c.fill == egui::Color32::TRANSPARENT && c.stroke.color == good
                    }
                    _ => false,
                })
                .count()
        };
        assert_eq!((circles(true), circles(false)), (0, 1 + rows));
    }

    #[test]
    fn a_tile_s_entry_names_the_check_of_its_row_and_a_click_selects_it() {
        // Spec 5.3, "Checks it touches" for a result: the check of its row (plan review C-3,
        // E-5). The Hot-low tile's entry shows the hot minimum check's state and name; a click
        // on the name selects the check. A tile no check badges has no such heading.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let check = "metal.hot_min_check";
        let guide = guidance::guide(check).unwrap();
        let hot_low = Selection::Result(Tile::HotLow.path());
        let (output, _, _) = frames(&ctx, &inputs, hot_low, false, vec![]);
        let texts = drawn_texts(&output);
        assert_eq!(count(&texts, CHECKS_IT_TOUCHES), 1, "{texts:?}");
        assert_eq!(count(&texts, Level::Bad.word()), 1, "{texts:?}");
        let heading = text_rect(&output, CHECKS_IT_TOUCHES).unwrap();
        // The tile's own label is the check's name too: the link is the one under the heading.
        let link = crate::gui::test_support::text_rects(&output, guide.label)
            .into_iter()
            .find(|r| r.top() >= heading.bottom())
            .expect("the check's name under the heading");
        let (_, asked, events) = frames(&ctx, &inputs, hot_low, false, click_at(link.center()));
        assert_eq!(asked, Some(Asked::Select(Selection::Check(check))));
        assert_eq!(events, ReadoutEvents::default());
        for tile in [Tile::Pullout, Tile::ColdHigh, Tile::Mass] {
            let (output, _, _) =
                frames(&ctx, &inputs, Selection::Result(tile.path()), false, vec![]);
            assert_eq!(
                count(&drawn_texts(&output), CHECKS_IT_TOUCHES),
                0,
                "{tile:?}"
            );
        }
    }

    #[test]
    fn every_entry_text_has_glyphs_in_the_app_s_font_and_plain_words() {
        // On designs where each check fails: every guide's entry, and each guide at red
        // whatever the design with its static fixes; the fixed texts, and the hover texts.
        let ctx = egui::Context::default();
        let mut plain: Vec<String> = Vec::new();
        let mut all = designs();
        all.push(design(|i| {
            i.clamps.boss_od_mm = 12.0;
            i.clamps.clamp_length_mm = 3.0;
        }));
        all.push(design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)));
        for inputs in all {
            let results = compute_all(&inputs);
            let fixes = all_fixes(&inputs, &results, false);
            for (index, guide) in CHECK_GUIDES.iter().enumerate() {
                plain.extend(entry_texts(&inputs, guide.check, false));
                let notes = || guide_notes(&fixes, index, &inputs, &results);
                let listed = match guide.fixes {
                    Fixes::Static(list) => list,
                    Fixes::PerTerm(_) => &fixes[index][..],
                };
                let red = Entry {
                    guide,
                    mark: Some(Level::Bad),
                    level: Level::Bad,
                    fixes: listed,
                    notes: &notes,
                };
                let output = sized_frame(&ctx, egui::vec2(320.0, 4000.0), vec![], |ui| {
                    let shown = Shown {
                        inputs: &inputs,
                        results: &results,
                        fixes: &fixes,
                        clamps_paused: false,
                        consequences: None,
                        auto: None,
                        held: &|_| None,
                        minimum_diameter_mm: None,
                    };
                    entry_ui(ui, &red, &shown, &mut Readouts::default());
                });
                plain.extend(drawn_texts(&output));
            }
        }
        plain.extend([NOTE_NOT_REVIEWED, FITS, DOES_NOT_FIT, NO_FIX].map(str::to_owned));
        // An assumption's rationale is the hover text of its lesson link: plain words too.
        for guide in counted_guides(false) {
            match guide.learn {
                Learn::None(reason) => plain.push(reason.to_owned()),
                Learn::Assumption(id) => plain.push(
                    ASSUMPTIONS
                        .iter()
                        .find(|a| a.id == id)
                        .unwrap()
                        .rationale
                        .to_owned(),
                ),
                Learn::Note(_) => {}
            }
        }
        for text in &plain {
            assert_glyphs(&ctx, text, "the Explain panel");
            assert!(
                plain_text_violations(text).is_empty(),
                "{text:?}: {:?}",
                plain_text_violations(text)
            );
        }
    }

    #[test]
    fn a_size_row_s_selection_reads_the_consequences_and_is_no_check_or_tile() {
        // UX overhaul spec 5.1: an input is one more kind of selection. Its entry draws its
        // what-moves-what lines and its torque-card row, so the panel asks its cache for them.
        for row in SIZE_ROWS {
            let selection = Selection::Input(row.key);
            assert_eq!(selection.input(), Some(row.key));
            assert!(selection.reads_consequences(), "{}", row.label);
            assert_eq!((selection.check(), selection.tile()), (None, None));
        }
        assert_eq!(Selection::Nothing.input(), None);
    }

    #[test]
    fn each_size_row_s_entry_is_its_help_its_lines_its_card_row_and_its_lesson() {
        // UX overhaul spec 4.3 and 5.3, test 11 of 8.2 for the size controls: a row's entry is
        // its title, "What it is" (its label's hover), "What sets it" (the lines the column
        // drew under it: its hint, its what-moves-what lines, Minimum OD's floor, for Outside
        // diameter and Overall length the footnote on what the controls move), "What a change
        // does" (its torque-card row,
        // under the card's heading and caption), then "Full lesson:" and the note its Why link
        // opened, in that order; a heading with nothing under it is left out ("Checks it
        // touches" and "Watch out" have their own test). At the failing example design and at
        // a passing one.
        use crate::gui::torque_card::CAPTION;
        for inputs in [DesignInputs::default(), passing_design()] {
            let results = compute_all(&inputs);
            let consequences = Consequences::of(&inputs, &results, None, false);
            let size = dimensions(&inputs).expect("size controls");
            for row in SIZE_ROWS {
                let ctx = egui::Context::default();
                let (output, _) = input_frames(&ctx, &inputs, row.key, false, None, None, vec![]);
                let texts = drawn_texts(&output);
                let has = |text: &str| texts.iter().any(|t| t == text);
                let label = row.label;
                assert!(has(label), "{label}: {texts:?}");
                assert!(has(row_help(row.key)), "{label}: {texts:?}");
                let mut lines: Vec<String> =
                    hint_for(row.key).map(str::to_owned).into_iter().collect();
                match row.key {
                    DIAMETER_HINT_KEY => lines.extend(consequences.diameter_line()),
                    LENGTH_HINT_KEY => lines.extend(consequences.length_line()),
                    MINIMUM_OD_HINT_KEY => lines.push(floor_line(size.minimum_diameter_range.min)),
                    key if key == FreeVariable::MagnetsPerRing.path() => {
                        lines.extend(consequences.magnet_lines(inputs.coupling.npole, None))
                    }
                    _ => {}
                }
                if matches!(row.key, DIAMETER_HINT_KEY | LENGTH_HINT_KEY) {
                    lines.push(LINKED_HOUSING.to_owned());
                }
                assert_eq!(has(WHAT_SETS_IT), !lines.is_empty(), "{label}");
                for line in &lines {
                    assert!(has(line), "{label}: no {line:?} in {texts:?}");
                }
                let card = row
                    .card
                    .map(|control| consequences.card.control_lines(control, None))
                    .unwrap_or_default();
                assert_eq!(has(WHAT_A_CHANGE_DOES), !card.is_empty(), "{label}");
                assert_eq!(has(CARD_HEADING), !card.is_empty(), "{label}");
                if row.card.is_some() {
                    assert_eq!(card.len(), 2, "{label}: the caption and the row: {card:?}");
                    assert_eq!(card[0], CAPTION);
                }
                for line in &card {
                    assert!(has(line), "{label}: no {line:?}");
                }
                assert_eq!(has(FULL_LESSON), row.lesson.is_some(), "{label}");
                if let Some(id) = row.lesson {
                    let note = reviewed_note(id).expect("a Why link's note is reviewed");
                    assert!(has(&glyph_safe(note.title)), "{label}: {texts:?}");
                }
                let tops: Vec<f32> = [WHAT_IT_IS, WHAT_SETS_IT, WHAT_A_CHANGE_DOES, FULL_LESSON]
                    .into_iter()
                    .filter_map(|heading| text_rect(&output, heading))
                    .map(|rect| rect.top())
                    .collect();
                assert!(
                    tops.is_sorted(),
                    "{label}: the headings out of order: {tops:?}"
                );
                assert!(
                    text_rect(&output, label).expect("its title").top() < tops[0],
                    "{label}: the title leads"
                );
            }
        }
    }

    #[test]
    fn the_footnote_on_what_the_controls_move_is_in_the_diameter_and_length_entries_alone() {
        // PT-1 of the 2026-11-23 review: LINKED_HOUSING says nothing new for the gap and the
        // magnet count, and its "magnet count stay as you set them" is false while Auto sets
        // the count, so no entry draws it beside the card's Auto line then.
        let inputs = DesignInputs::default();
        for auto in [
            None,
            Some(FreeVariable::AxialLength),
            Some(FreeVariable::RingRadius),
            Some(FreeVariable::MagnetsPerRing),
        ] {
            for row in SIZE_ROWS {
                let ctx = egui::Context::default();
                let (output, _) = input_frames(&ctx, &inputs, row.key, false, None, auto, vec![]);
                let texts = drawn_texts(&output);
                let want = matches!(row.key, DIAMETER_HINT_KEY | LENGTH_HINT_KEY)
                    && auto != Some(FreeVariable::MagnetsPerRing);
                assert_eq!(
                    texts.contains(&LINKED_HOUSING.to_owned()),
                    want,
                    "{} under {auto:?}: {texts:?}",
                    row.label
                );
                if auto.is_some() && row.card.is_some() {
                    assert!(
                        texts.iter().any(|t| t.contains("Auto")),
                        "{} under {auto:?}: the card says what Auto does: {texts:?}",
                        row.label
                    );
                }
            }
        }
    }

    #[test]
    fn a_size_row_s_title_leads_with_its_value() {
        // Spec 5.3: "Title, with the live value leading", in the panel's number format: an
        // input's value with its unit; Outside diameter, Overall length and Minimum OD the sizes
        // their sliders set, Minimum OD at the person's preference where one is set. An odd
        // magnet count has no size controls: Outside diameter's entry has no value and no card
        // row, and keeps its help and lines.
        let inputs = DesignInputs::default();
        let size = dimensions(&inputs).expect("size controls");
        let mm = |x: f64| value_with_unit(&Value::Num(x), format_value(&Value::Num(x)), "mm");
        let gap = Value::Num(inputs.metal.face_gap_mm);
        for (key, minimum, want) in [
            (DIAMETER_HINT_KEY, None, mm(size.diameter_mm)),
            (LENGTH_HINT_KEY, None, mm(size.overall_length_mm)),
            (MINIMUM_OD_HINT_KEY, None, mm(slider_minimum(&size, None))),
            (MINIMUM_OD_HINT_KEY, Some(38.0), mm(38.0)),
            (
                FACE_GAP,
                None,
                value_with_unit(&gap, format_value(&gap), "mm"),
            ),
        ] {
            let ctx = egui::Context::default();
            let (output, _) = input_frames(&ctx, &inputs, key, false, minimum, None, vec![]);
            let label = size_row(key).expect("a row").label;
            let value = text_rect(&output, &want)
                .unwrap_or_else(|| panic!("{key}: no {want:?} in {:?}", drawn_texts(&output)));
            // The label follows the value on its row: a label in a wrapping row is laid out
            // from the row's start, so its last character marks where it is.
            let end = crate::gui::test_support::last_characters(&output, label)[0];
            assert!(value.right() < end.x, "{key}: the value leads");
            assert!(value.y_range().contains(end.y), "{key}: one line");
        }
        let odd = with_poles(7);
        assert!(dimensions(&odd).is_none());
        let texts = input_texts(&odd, DIAMETER_HINT_KEY, false);
        let value = |t: &String| t.ends_with(" mm") && t.starts_with(|c: char| c.is_ascii_digit());
        assert!(!texts.iter().any(value), "no value: {texts:?}");
        assert!(!texts.contains(&WHAT_A_CHANGE_DOES.to_owned()), "{texts:?}");
        assert!(
            texts.contains(&row_help(DIAMETER_HINT_KEY).to_owned()),
            "{texts:?}"
        );
        assert!(texts.contains(&LINKED_HOUSING.to_owned()), "{texts:?}");
    }

    #[test]
    fn a_size_row_s_entry_names_the_checks_whose_fixes_name_its_input_and_a_click_selects_one() {
        // Spec 5.3, "Checks it touches" for an input: the guides whose fixes name it, on the
        // design shown, each with its state's mark and word, a link that selects the check. At
        // the example design the running clearance's and the gearbox input's fixes name the face
        // gap; a passing design has no fixes, so no entry draws the heading. Minimum OD writes
        // no input and names none.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let named = |input: Option<&str>| -> Vec<&'static CheckGuide> {
            CHECK_GUIDES
                .iter()
                .zip(&fixes)
                .filter(|(_, list)| input.is_some_and(|i| list.iter().any(|&(p, _)| p == i)))
                .map(|(guide, _)| guide)
                .collect()
        };
        for row in SIZE_ROWS {
            let texts = input_texts(&inputs, row.key, false);
            let guides = named(row.input);
            assert_eq!(
                texts.contains(&CHECKS_IT_TOUCHES.to_owned()),
                !guides.is_empty(),
                "{}",
                row.label
            );
            for guide in guides {
                assert!(
                    texts.contains(&guide.label.to_owned()),
                    "{}: {}",
                    row.label,
                    guide.label
                );
            }
        }
        assert!(named(None).is_empty());
        let gap = named(Some(FACE_GAP));
        let first = gap
            .first()
            .expect("a failing check's fix names the face gap");
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, FACE_GAP, false, None, None, vec![]);
        let at = text_rect(&output, first.label)
            .expect("the check's link")
            .center();
        let (_, asked) = input_frames(&ctx, &inputs, FACE_GAP, false, None, None, click_at(at));
        assert_eq!(asked, Some(Asked::Select(Selection::Check(first.check))));
        for row in SIZE_ROWS {
            let texts = input_texts(&passing_design(), row.key, false);
            assert!(
                !texts.contains(&CHECKS_IT_TOUCHES.to_owned()),
                "{}",
                row.label
            );
        }
    }

    #[test]
    fn a_size_row_s_entry_warns_when_two_failing_checks_pull_its_input_opposite_ways() {
        // Spec 5.3, "Watch out" for an input: each "pulls two ways" sentence the two checks'
        // blocks draw, once. Under a 30 mm overall-length claim the hot-low torque needs longer
        // magnets and the space claim (and the gearbox input) shorter ones: Overall length's
        // entry says so, after its checks; the face gap, which no two checks pull apart, has no
        // "Watch out".
        let inputs = length_conflict();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        let length = size_row(LENGTH_HINT_KEY).expect("a row");
        let notes = input_trade_offs(&fixes, length.input.expect("an input"));
        assert!(!notes.is_empty());
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, LENGTH_HINT_KEY, false, None, None, vec![]);
        let texts = drawn_texts(&output);
        for note in &notes {
            assert_eq!(
                texts.iter().filter(|t| *t == note).count(),
                1,
                "{note}: {texts:?}"
            );
        }
        let rect = |text: &str| text_rect(&output, text).expect(text);
        assert!(rect(CHECKS_IT_TOUCHES).top() < rect(WATCH_OUT).top());
        assert!(rect(WATCH_OUT).bottom() <= rect(&notes[0]).top() + 0.5);
        assert!(input_trade_offs(&fixes, FACE_GAP).is_empty());
        assert!(!input_texts(&inputs, FACE_GAP, false).contains(&WATCH_OUT.to_owned()));
    }

    #[test]
    fn a_size_row_s_entry_shows_its_input_s_cell_only_with_the_spreadsheet_details() {
        // Spec 5.3's "Spreadsheet cell" foot, shown only with the details on: the cell of a row
        // that is one input (Required torque, Magnet face gap, Magnets per ring). Minimum OD,
        // Outside diameter and Overall length are sizes the spreadsheet has no cell for.
        for row in SIZE_ROWS {
            let cell = |texts: &[String]| -> Vec<String> {
                texts
                    .iter()
                    .filter(|t| t.starts_with(SPREADSHEET_CELL))
                    .cloned()
                    .collect()
            };
            assert!(cell(&input_texts(&DesignInputs::default(), row.key, false)).is_empty());
            let on = cell(&input_texts(&DesignInputs::default(), row.key, true));
            if row.input == Some(row.key) {
                let meta = crate::gui::inputs::InputCatalogue::get()
                    .entry(row.key)
                    .expect("an input")
                    .meta;
                let want = format!(
                    "{SPREADSHEET_CELL} {}",
                    meta.cell.unwrap_or(NO_SPREADSHEET_CELL)
                );
                assert_eq!(on, [want], "{}", row.label);
            } else {
                assert!(on.is_empty(), "{}: {on:?}", row.label);
            }
        }
    }

    #[test]
    fn a_size_row_s_card_row_keeps_the_auto_line_while_auto_is_on() {
        // Owner ruling 5A, as the Hot-low tile's card draws it: while Auto is on, the Auto line
        // stands between the caption and a row, since an edit of the row makes Auto re-solve.
        // With Auto on the magnet length, Outside diameter's entry draws that line once,
        // between the caption and its row; with Auto off, none.
        use crate::gui::torque_card::{AUTO_HOLDS_LENGTH, CAPTION, Control};
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let card = Consequences::of(&inputs, &results, None, false).card;
        let auto = Some(FreeVariable::AxialLength);
        let lines = card.control_lines(Control::OutsideDiameter, auto);
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(lines[..2], [CAPTION, AUTO_HOLDS_LENGTH]);
        let ctx = egui::Context::default();
        let (output, _) = input_frames(&ctx, &inputs, DIAMETER_HINT_KEY, false, None, auto, vec![]);
        let texts = drawn_texts(&output);
        let tops: Vec<f32> = lines
            .iter()
            .map(|line| {
                let n = texts.iter().filter(|t| *t == line).count();
                assert_eq!(n, 1, "{line:?} in {texts:?}");
                text_rect(&output, line).expect("drawn").top()
            })
            .collect();
        assert!(tops.is_sorted(), "out of order: {tops:?}");
        let off = input_texts(&inputs, DIAMETER_HINT_KEY, false);
        assert!(!off.contains(&AUTO_HOLDS_LENGTH.to_owned()), "{off:?}");
    }

    #[test]
    fn the_size_rows_new_texts_have_glyphs_and_plain_words() {
        // Spec 8.3: the entry's new words, the foot line's head and Minimum OD's hover (a
        // string the column drew before, now a constant), pass the plain-words guard, say
        // nothing of protecting the gearbox and have glyphs in the default fonts; so does every
        // text each row's entry draws, with the details on.
        let ctx = glyph_context();
        let floor = floor_line(
            dimensions(&DesignInputs::default())
                .unwrap()
                .minimum_diameter_range
                .min,
        );
        for text in [SPREADSHEET_CELL, MINIMUM_HOVER, floor.as_str()] {
            assert!(plain_text_violations(text).is_empty(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            assert_glyphs(&ctx, text, "a Size row's entry");
        }
        for row in SIZE_ROWS {
            for text in input_texts(&DesignInputs::default(), row.key, true) {
                assert_glyphs(&ctx, &text, row.label);
            }
        }
    }

    // ------------------------------------------------------------------ any input's entry

    /// The inputs of the detailed inputs that the tests below select: every Key design input
    /// that is no Size column row's key, and each assumption's first input.
    fn detailed_inputs() -> Vec<&'static str> {
        let catalogue = crate::gui::inputs::InputCatalogue::get();
        catalogue
            .key_design
            .iter()
            .map(|entry| entry.path.as_str())
            .filter(|path| size_row(path).is_none())
            .chain(ASSUMPTIONS.iter().map(|a| a.paths[0]))
            .collect()
    }

    #[test]
    fn a_detailed_input_s_entry_is_its_value_label_help_hint_checks_and_cell() {
        // Spec 5.1 to 5.3 for a row of the detailed inputs or the assumptions: the input's
        // value leading its label, "What it is" (its help, as its row's hover shows it), "What
        // sets it" (its hint, as the Key design row draws it), the checks whose fixes name it
        // and, with the spreadsheet details, its cell. It has no torque-card row and no lesson,
        // and its selection reads no consequences. A selector's value is its choice in words.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        for path in detailed_inputs() {
            let meta = crate::gui::inputs::InputCatalogue::get()
                .entry(path)
                .expect("an input")
                .meta;
            let texts = input_texts(&inputs, path, false);
            let has = |text: &str| texts.iter().any(|t| t == text);
            assert!(has(label_for(path)), "{path}: {texts:?}");
            let help = help_for(path);
            assert_eq!(has(WHAT_IT_IS), !help.is_empty(), "{path}");
            assert!(help.is_empty() || has(help), "{path}: {texts:?}");
            let hint = hint_for(path);
            assert_eq!(has(WHAT_SETS_IT), hint.is_some(), "{path}");
            assert!(hint.is_none_or(has), "{path}: {texts:?}");
            for absent in [WHAT_A_CHANGE_DOES, CARD_HEADING, FULL_LESSON] {
                assert!(!has(absent), "{path}: {absent}");
            }
            let named = CHECK_GUIDES
                .iter()
                .zip(&fixes)
                .filter(|(_, list)| list.iter().any(|&(p, _)| p == path))
                .count();
            assert_eq!(has(CHECKS_IT_TOUCHES), named > 0, "{path}");
            let cells: Vec<String> = input_texts(&inputs, path, true)
                .into_iter()
                .filter(|t| t.starts_with(SPREADSHEET_CELL))
                .collect();
            assert_eq!(cells, [cell_line(meta.cell)], "{path}");
            assert!(!Selection::Input(path).reads_consequences(), "{path}");
        }
        let op_temp = "coupling.op_temp_C";
        let value = inputs.get(op_temp).unwrap();
        let unit = crate::gui::inputs::InputCatalogue::get()
            .entry(op_temp)
            .unwrap()
            .meta
            .unit;
        let want = value_with_unit(&value, format_value(&value), unit);
        assert!(
            input_texts(&inputs, op_temp, false).contains(&want),
            "{want}"
        );
        let backiron = "coupling.backiron";
        let meta = crate::gui::inputs::InputCatalogue::get()
            .entry(backiron)
            .unwrap()
            .meta;
        let choice = value_text(&inputs.get(backiron).unwrap(), meta);
        assert!(choice.contains(" = "), "{choice}");
        assert!(input_texts(&inputs, backiron, false).contains(&choice));
        // A blank optional input's value is the word the tooltip's Default line uses.
        let drag = "metal.measured_drag_Nm";
        assert_eq!(inputs.get(drag), Some(Value::None));
        assert!(input_texts(&inputs, drag, false).contains(&"blank".to_owned()));
        // A path that is no input draws the heading and the way back alone.
        assert_eq!(
            input_texts(&inputs, "no.such.input", false),
            [HEADING, ABOUT]
        );
    }

    #[test]
    fn an_input_s_entry_leads_with_the_link_that_traces_it() {
        // Spec 5.2: tracing stays, as a link in the input's entry, right under its title. A
        // click asks the panel to trace the input (a second click on it ends the trace, decision
        // O-8); Outside diameter and Overall length trace the input their slider writes, and
        // Minimum OD, which writes none, has no link.
        let inputs = DesignInputs::default();
        let mut keys: Vec<&'static str> = SIZE_ROWS.iter().map(|row| row.key).collect();
        keys.extend(detailed_inputs());
        for key in keys {
            let input = size_row(key).map_or(Some(key), |row| row.input);
            let ctx = egui::Context::default();
            let (output, asked) = input_frames(&ctx, &inputs, key, false, None, None, vec![]);
            assert_eq!(asked, None);
            let texts = drawn_texts(&output);
            let links = texts.iter().filter(|t| *t == TRACE_LINK).count();
            assert_eq!(links, usize::from(input.is_some()), "{key}: {texts:?}");
            let Some(input) = input else { continue };
            let link = text_rect(&output, TRACE_LINK).unwrap();
            let label = size_row(key).map_or_else(|| label_for(key), |row| row.label);
            let title = crate::gui::test_support::last_characters(&output, label)[0];
            assert!(title.y < link.top(), "{key}: the title leads");
            if let Some(what) = text_rect(&output, WHAT_IT_IS) {
                assert!(
                    link.bottom() <= what.top(),
                    "{key}: the link before What it is"
                );
            }
            let (_, asked) = input_frames(
                &ctx,
                &inputs,
                key,
                false,
                None,
                None,
                click_at(link.center()),
            );
            assert_eq!(asked, Some(Asked::Trace(input)), "{key}");
        }
    }

    #[test]
    fn a_fix_link_selects_its_input_and_asks_for_its_row() {
        // Spec 5.2: a fix link selects and focuses its input. The input the Size column shows
        // under its own name (the ring radius, under Outside diameter) selects that row.
        let none: &dyn Fn() -> Vec<String> = &Vec::new;
        let guide = guide_with(Learn::None("A test reason."));
        let red = entry_at(&guide, Level::Bad, &CUP_WALL_FIX, none);
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let (asked, events) = click_in_entry(&red, &fix);
        assert_eq!(
            asked,
            Some(Asked::Select(Selection::Input("metal.cup_wall_corner_mm")))
        );
        assert_eq!(events.focus, Some("metal.cup_wall_corner_mm"));
        let (ring, _) = SIZE_TAB_ALIASES[0];
        let ring_fix = [(ring, Direction::Raise)];
        let red = entry_at(&guide, Level::Bad, &ring_fix, none);
        let (asked, events) = click_in_entry(&red, &fix_text(ring, Direction::Raise));
        assert_eq!(
            asked,
            Some(Asked::Select(Selection::Input(DIAMETER_HINT_KEY)))
        );
        assert_eq!(events.focus, Some(ring));
        assert_eq!(fix_selection(FACE_GAP), Selection::Input(FACE_GAP));
    }

    #[test]
    fn a_result_s_and_a_check_s_entry_show_their_cell_only_with_the_spreadsheet_details() {
        // Spec 5.3's "Spreadsheet cell" foot for a result and a check, shown only with the
        // details on: the result's cell, or the line that says the spreadsheet has none; a
        // check that is no result (the magnets fit) has no cell line, as it has no Equation
        // link.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let cells = |selection: Selection, details: bool| -> Vec<String> {
            let (output, _, _) = frames(&ctx, &inputs, selection, details, vec![]);
            drawn_texts(&output)
                .into_iter()
                .filter(|t| t.starts_with(SPREADSHEET_CELL))
                .collect()
        };
        for tile in Tile::ALL {
            let selection = Selection::Result(tile.path());
            assert!(cells(selection, false).is_empty(), "{tile:?}");
            let info = result_info(tile.path()).expect("a result");
            assert_eq!(
                cells(selection, true),
                [cell_line(info.cell.as_deref())],
                "{tile:?}"
            );
        }
        for guide in counted_guides(false) {
            let selection = Selection::Check(guide.check);
            assert!(cells(selection, false).is_empty(), "{}", guide.check);
            let want: Vec<String> = result_info(guide.check)
                .map(|info| cell_line(info.cell.as_deref()))
                .into_iter()
                .collect();
            assert_eq!(cells(selection, true), want, "{}", guide.check);
        }
        assert!(result_info(FIT_CHECK).is_none());
        let pullout = result_info(Tile::Pullout.path()).and_then(|info| info.cell.clone());
        assert!(pullout.is_some(), "the pull-out has a cell");
    }

    // ------------------------------------------------------------------ any result's entry

    /// The results the tests below select: every dashboard row (the four tiles among them),
    /// every dimension of the drawing of `inputs` and the first results-table row with no
    /// equation record, each one the screen shows while the shaft-clamp section is paused.
    fn selected_results(inputs: &DesignInputs) -> Vec<&'static str> {
        use crate::engine::clamps;
        let results = compute_all(inputs);
        let shown = |path: &&str| !clamps::hides(path, clamps::PAUSED);
        let recordless = crate::gui::results_table::table_entries()
            .iter()
            .map(|entry| entry.path.as_str())
            .filter(shown)
            .find(|path| registry().equation_for(path).is_none())
            .expect("a result with no equation record");
        let geometry = crate::gui::geometry::geometry(inputs, &results, clamps::PAUSED);
        DASHBOARD
            .iter()
            .map(|&(path, _)| path)
            .chain(geometry.callouts().map(|callout| callout.path))
            .chain(std::iter::once(recordless))
            .filter(shown)
            .collect()
    }

    #[test]
    fn a_result_s_entry_is_its_value_label_help_check_watch_out_lesson_and_equation() {
        // Spec 5.1 to 5.3 for any result (a dimension tag, a callout row, a results-table row):
        // its value as its results-table row shows it, leading its label; "What it is" (its
        // help); the check of its dashboard row; for a row that reads the stored 3D fields, the
        // line that says so under "Watch out"; its row's lesson where it has one; the Equation
        // link where it has an equation record. A tile's keeps the tile's number and label.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        for path in selected_results(&inputs) {
            let (output, _, _) = frames(&ctx, &inputs, Selection::Result(path), false, vec![]);
            let texts = drawn_texts(&output);
            let has = |text: &str| texts.iter().any(|t| t == text);
            let info = result_info(path).expect("a result");
            let label = match Selection::Result(path).tile() {
                Some(tile) => tile.label(&results),
                None => {
                    let value = results.get(path).unwrap_or(Value::None);
                    let shown = value_with_unit(&value, shown_text(path, &value), info.meta.unit);
                    assert!(has(&shown), "{path}: {shown:?} in {texts:?}");
                    display_label(path, info.meta.label).to_owned()
                }
            };
            assert!(has(&label), "{path}: {label:?} in {texts:?}");
            let help = help_for(path);
            assert_eq!(has(WHAT_IT_IS), !help.is_empty(), "{path}");
            assert!(help.is_empty() || has(help), "{path}: {texts:?}");
            assert_eq!(
                has(CHECKS_IT_TOUCHES),
                badging_check(path).is_some(),
                "{path}"
            );
            let stored_3d = STORED_3D_ROWS.contains(&path);
            assert_eq!(has(WATCH_OUT), stored_3d, "{path}");
            assert_eq!(has(STORED_3D_LABEL), stored_3d, "{path}");
            assert_eq!(has(FULL_LESSON), row_learn(path).is_some(), "{path}");
            let equation = registry().equation_for(path).is_some();
            assert_eq!(has(EQUATION), equation, "{path}");
            assert!(
                !has(WHAT_A_CHANGE_DOES),
                "{path}: no card without the consequences"
            );
        }
    }

    #[test]
    fn a_result_s_equation_link_asks_for_its_equation_and_its_lesson_for_its_note() {
        // Spec 5.5: an entry's Equation link opens the Equation panel at its result, and the
        // selection stays. Part 1's decision D-17: the three dashboard rows no check holds
        // (the pull-out at 20 °C, the slip swing at the gearbox input, the cup's outside
        // diameter) lead from their entry's lesson line where the dashboard's Why link did,
        // one click from their results-table row.
        let inputs = DesignInputs::default();
        const FACE_GAP_RESULT: &str = "model.face_gap_mm";
        let (asked, events) = click(&inputs, Selection::Result(FACE_GAP_RESULT), EQUATION);
        assert_eq!(asked, Some(Asked::Equation(FACE_GAP_RESULT)));
        assert_eq!(events, ReadoutEvents::default());
        for path in [
            "model.pullout_20C_Nm",
            "model.gearbox_input_ripple_Nm",
            "model.cup_od_mm",
        ] {
            assert_eq!(badging_check(path), None, "{path}");
            let learn = row_learn(path).expect("a dashboard row");
            let (asked, events) = click(&inputs, Selection::Result(path), &learn_link_text(learn));
            assert_eq!(asked, None, "{path}");
            let want = match learn {
                Learn::Note(id) => Some(id),
                Learn::Assumption(_) | Learn::None(_) => None,
            };
            assert_eq!(events.note, want, "{path}");
        }
    }

    #[test]
    fn a_result_that_is_a_check_s_own_selects_the_check_and_any_other_the_result() {
        // Spec 5.2: a click on a results-table row "selects that result or check". The row of
        // a check's own result (its verdict) selects the check, whose entry holds its Problem,
        // Why and fixes; any other row selects the result.
        let checks: Vec<&CheckGuide> = CHECK_GUIDES
            .iter()
            .filter(|guide| result_info(guide.check).is_some())
            .collect();
        assert!(!checks.is_empty());
        for guide in checks {
            assert_eq!(result_selection(guide.check), Selection::Check(guide.check));
        }
        for entry in crate::gui::results_table::table_entries() {
            let path = entry.path.as_str();
            if guidance::guide(path).is_none() {
                assert_eq!(result_selection(path), Selection::Result(path), "{path}");
            }
        }
    }

    #[test]
    fn a_result_s_entry_text_has_glyphs_and_names_no_paused_clamp() {
        // Spec 8.3: every text a result's entry draws, with the spreadsheet details on designs
        // where each check fails, has its glyphs in the app's font, and none names the paused
        // clamp.
        let ctx = glyph_context();
        for inputs in designs() {
            for path in selected_results(&inputs) {
                let (output, _, _) = frames(&ctx, &inputs, Selection::Result(path), true, vec![]);
                for text in drawn_texts(&output) {
                    assert_glyphs(&ctx, &text, path);
                    assert!(!text.to_lowercase().contains("clamp"), "{path}: {text}");
                }
            }
        }
    }
}
