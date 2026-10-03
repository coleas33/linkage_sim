//! The results table (spec M4 "Layout": "every computed value with label, unit, cell;
//! searchable; CSV and JSON export").
//!
//! Every result, scalars and table rows. The layout of the results does not depend on the
//! inputs, so the rows (path, metadata, cell, marker, search text) are built once in the Python
//! schema order ([`result_rows`]); a frame reads only the values of the rows on screen. The
//! table shows them by group (decision O-6, [`crate::gui::result_groups`]: the headline open,
//! every other group a heading that opens it) or in the engine's order ([`ResultOrder`]); a
//! check's row carries the dashboard's badge, and the failing filter lists the failing checks
//! alone, the red first (decision O-7). The exports write every result in schema order at full
//! precision: CSV for spreadsheets, JSON with the design that produced it. JSON has no infinity
//! or NaN, so both write a non-finite number as `+inf`, `-inf` or `NaN` (decision M41-15).

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use serde_json::{Map, Value as Json};

use crate::engine::meta::{ResultSet, Value, result_rows};
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{
    CHECKS, Level, ResultInfo, badge, check_level, failing_checks, hover_text, result_info,
};
use crate::gui::format::{
    format_value, non_finite_text, search_haystack, search_needle, with_unit,
};
use crate::gui::readouts::Readouts;
use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
use crate::gui::session::{Design, design_json, json_value};
use crate::gui::trace::{Trace, TraceKind};
use crate::{DesignInputs, DesignResults, compute_all};

/// The `format` of a results export.
pub const RESULTS_FORMAT: &str = "magcoupling-results";

/// The version of the results export.
pub const RESULTS_VERSION: u64 = 1;

/// The file names the exports suggest.
pub const CSV_FILE_NAME: &str = "magcoupling-results.csv";
pub const JSON_FILE_NAME: &str = "magcoupling-results.json";

/// The CSV header.
pub const CSV_HEADER: &str = "path,label,value,unit,cell";

/// The button labels.
pub const EXPORT_CSV: &str = "Export CSV";
pub const EXPORT_JSON: &str = "Export JSON";

/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// The workbook cell column of a Rust-only result (no workbook cell).
pub const RUST_ONLY: &str = "Rust-only";

/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";

/// The trace filter's checkbox (decision O-8): the rows an input's trace marks alone.
pub const TRACED_ONLY: &str = "Traced only";

/// What the table says when the trace itself marks none of the rows the failing filter lets
/// through ([`empty_text`]): the trace reaches only results with an equation record, and an input
/// may reach none of them.
pub const NOTHING_TRACED: &str =
    "The trace marks no result shown here (only results with an equation record are traced).";

/// What the table says when the failing filter finds no check to show.
pub const NOTHING_FAILS: &str = "No check fails or asks for a look.";

/// What the table says when the search matches no result.
pub const NO_RESULT: &str = "No result matches the search.";

/// A group heading's hover text while the search holds more than blanks: every group is open
/// then, and a click on a heading does nothing.
pub const CLEAR_TO_CLOSE: &str = "Clear the search to close a group";

/// How far a package group's heading sits in under the "Other results" heading [points].
pub const OTHER_INDENT: f32 = 12.0;

/// The value, cell and marker columns' widths [points], M4-1's; the label column takes the rest.
pub const VALUE_WIDTH: f32 = 130.0;
pub const CELL_WIDTH: f32 = 130.0;
pub const MARKER_WIDTH: f32 = 60.0;

/// The narrowest label column [points]: below it the rows scroll sideways.
pub const LABEL_MIN_WIDTH: f32 = 120.0;

/// The widths of the label, value, cell and marker columns of a table `available` points wide
/// with `spacing` points between columns (decision M42-8): the label column flexes, at least
/// [`LABEL_MIN_WIDTH`], so a narrow centre region (a ~930 px window, where the M4-1 table showed
/// only its labels) still shows the label and the value without scrolling.
pub fn column_widths(available: f32, spacing: f32) -> [f32; 4] {
    let fixed = VALUE_WIDTH + CELL_WIDTH + MARKER_WIDTH + 3.0 * spacing;
    [
        (available - fixed).max(LABEL_MIN_WIDTH),
        VALUE_WIDTH,
        CELL_WIDTH,
        MARKER_WIDTH,
    ]
}

/// One row of the table: what does not change with the inputs.
#[derive(Clone, Debug)]
pub struct TableEntry {
    pub path: String,
    pub info: &'static ResultInfo,
    /// The corrections' marker text, empty for none.
    pub marker: String,
    /// One of the design's checks ([`CHECKS`]): its row carries a badge.
    pub check: bool,
    /// Label, path and cell, lowercase: what the search matches.
    haystack: String,
}

/// Every result's row, in schema order, built once.
pub fn table_entries() -> &'static [TableEntry] {
    static ENTRIES: OnceLock<Vec<TableEntry>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        result_rows(&compute_all(&DesignInputs::default()))
            .into_iter()
            .map(|row| {
                let info = result_info(&row.path).expect("every result has its info");
                let haystack = search_haystack(info.meta.label, &row.path, info.cell.as_deref());
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
                    check: CHECKS.contains(&row.path.as_str()),
                    path: row.path,
                    info,
                    haystack,
                }
            })
            .collect()
    })
}

/// The index of the row of the result at `path` in [`table_entries`].
pub fn entry_index(path: &str) -> Option<usize> {
    static INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    INDEX
        .get_or_init(|| {
            table_entries()
                .iter()
                .enumerate()
                .map(|(index, entry)| (entry.path.as_str(), index))
                .collect()
        })
        .get(path)
        .copied()
}

/// How the table orders its rows (decision O-6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResultOrder {
    /// By group: the headline, the physics chains, then the other results by package.
    #[default]
    Grouped,
    /// The engine's (Python's schema) order, as the exports write it.
    Engine,
}

impl ResultOrder {
    /// Both orders, in toggle order.
    pub const ALL: [ResultOrder; 2] = [ResultOrder::Grouped, ResultOrder::Engine];

    /// The toggle's text.
    pub const fn label(self) -> &'static str {
        match self {
            ResultOrder::Grouped => "By physics chain",
            ResultOrder::Engine => "Engine order",
        }
    }
}

/// One line the table draws, all of one height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// The heading over the package groups.
    OtherResults,
    /// A group's heading: the group (an index into [`result_groups`]), the rows of it the
    /// filters let through, and whether it is open (its rows follow).
    Group {
        group: usize,
        shown: usize,
        open: bool,
    },
    /// A result's row (an index into [`table_entries`]).
    Row(usize),
}

/// The lines of the table: the rows of `matches` (the search's, ascending). With `failing`
/// (the failing filter on: the failing checks' rows, the red first) those of them the search
/// matches, flat; else in `order`: flat in the engine's, or by group, each group with a match
/// under its heading (the package groups under "Other results"), its rows when `open` says so.
pub fn table_lines(
    matches: &[usize],
    order: ResultOrder,
    failing: Option<&[usize]>,
    open: &dyn Fn(usize) -> bool,
) -> Vec<Line> {
    let mut admitted = vec![false; table_entries().len()];
    for &index in matches {
        admitted[index] = true;
    }
    if let Some(failing) = failing {
        return failing
            .iter()
            .copied()
            .filter(|&index| admitted[index])
            .map(Line::Row)
            .collect();
    }
    match order {
        ResultOrder::Engine => matches.iter().copied().map(Line::Row).collect(),
        ResultOrder::Grouped => {
            let mut lines = Vec::new();
            // The "Other results" heading, before the first package group with a match.
            let mut other_started = false;
            for (group, entry) in result_groups().iter().enumerate() {
                let rows: Vec<usize> = entry
                    .rows
                    .iter()
                    .copied()
                    .filter(|&index| admitted[index])
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                if entry.other && !other_started {
                    lines.push(Line::OtherResults);
                    other_started = true;
                }
                let open = open(group);
                lines.push(Line::Group {
                    group,
                    shown: rows.len(),
                    open,
                });
                if open {
                    lines.extend(rows.into_iter().map(Line::Row));
                }
            }
            lines
        }
    }
}

/// The level of the badge `entry`'s row carries for `results`, in the table and the
/// spreadsheet: its check's level ([`check_level`]); `None` for a row that is no check and for
/// a check without a level (one the end effect greys).
pub fn entry_level(results: &DesignResults, entry: &TableEntry) -> Option<Level> {
    entry
        .check
        .then(|| check_level(results, &entry.path))
        .flatten()
}

/// The worst level of the checks among `rows` (indices into [`table_entries`]) for `results`: a
/// group heading's badge in the table and the spreadsheet; `None` when none of them is a check
/// with a level.
pub fn worst_level(results: &DesignResults, rows: &[usize]) -> Option<Level> {
    let entries = table_entries();
    rows.iter()
        .filter_map(|&index| entry_level(results, &entries[index]))
        .max()
}

/// What the table says when its filters leave no line: [`NOTHING_TRACED`] when the trace filter
/// is on and the trace itself marks none of the rows the failing filter lets through (every row
/// while it is off), whatever the search; [`NOTHING_FAILS`] when the failing filter alone leaves
/// nothing (no search narrows it); else [`NO_RESULT`], the search's doing.
pub fn empty_text(trace_marks_none: bool, failing_only: bool, searching: bool) -> &'static str {
    match (trace_marks_none, failing_only, searching) {
        (true, _, _) => NOTHING_TRACED,
        (false, true, false) => NOTHING_FAILS,
        _ => NO_RESULT,
    }
}

/// The indices of the rows whose label, path or cell contains `query`, ignoring case and the
/// surrounding blanks; every row for a blank query.
pub fn search(entries: &[TableEntry], query: &str) -> Vec<usize> {
    let needle = search_needle(query);
    (0..entries.len())
        .filter(|&i| needle.is_empty() || entries[i].haystack.contains(&needle))
        .collect()
}

/// A number at full precision: the shortest text that reads back as the same number
/// (`2.6884762950539796`, `5.27e-10`); `+inf`, `-inf` or `NaN` when not finite.
pub fn exact_number(x: f64) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None => format!("{x:?}"),
    }
}

/// A value at full precision; text as it is, an empty string for none.
fn exact_value(value: &Value) -> String {
    match value {
        Value::Num(x) => exact_number(*x),
        Value::Int(i) => i.to_string(),
        Value::Text(text) => text.clone(),
        Value::None => String::new(),
    }
}

/// A CSV field (RFC 4180): quoted, with quotes doubled, when it holds a comma, a quote or a
/// line break.
fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_owned()
    }
}

/// The CSV export: [`CSV_HEADER`], then one line per result in schema order.
pub fn results_csv(results: &DesignResults) -> String {
    let mut csv = String::from(CSV_HEADER);
    csv.push_str("\r\n");
    for row in result_rows(results) {
        let fields = [
            row.path.as_str(),
            row.meta.label,
            &exact_value(&row.value),
            row.meta.unit,
            row.cell.as_deref().unwrap_or(""),
        ];
        let line: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
        csv.push_str(&line.join(","));
        csv.push_str("\r\n");
    }
    csv
}

/// The JSON export: the design and every result with its label, unit and cell.
pub fn results_json(design: &Design, results: &DesignResults) -> String {
    let rows: Vec<Json> = result_rows(results)
        .into_iter()
        .map(|row| {
            let mut entry = Map::new();
            entry.insert("path".to_owned(), Json::String(row.path));
            entry.insert("label".to_owned(), Json::from(row.meta.label));
            entry.insert("value".to_owned(), json_value(&row.value));
            entry.insert("unit".to_owned(), Json::from(row.meta.unit));
            entry.insert("cell".to_owned(), row.cell.map_or(Json::Null, Json::String));
            Json::Object(entry)
        })
        .collect();
    let mut top = Map::new();
    top.insert("format".to_owned(), Json::from(RESULTS_FORMAT));
    top.insert("version".to_owned(), Json::from(RESULTS_VERSION));
    top.insert("design".to_owned(), design_json(design));
    top.insert("results".to_owned(), Json::Array(rows));
    let mut text = serde_json::to_string_pretty(&Json::Object(top))
        .expect("a JSON value of maps, strings and numbers always serializes");
    text.push('\n');
    text
}

/// The table's state: the search text and the rows it matches, the order, the failing filter
/// and the groups opened or closed.
#[derive(Clone, Debug, Default)]
pub struct ResultsTable {
    query: String,
    /// The rows matching `matched_query`; `None` until the first frame.
    matches: Option<Vec<usize>>,
    matched_query: String,
    order: ResultOrder,
    /// The failing checks alone (decision O-7).
    failing_only: bool,
    /// The rows the trace marks alone, while an input is traced (decision O-8).
    traced_only: bool,
    /// The groups (indices into [`result_groups`]) the user opened or closed: each starts as
    /// [`ResultsTable::opens_by_default`] says.
    toggled: BTreeSet<usize>,
}

/// Whether `trace` is an input's: the only trace whose marked results the table can filter to.
fn is_input_trace(trace: Option<&Trace>) -> bool {
    trace.is_some_and(|t| t.kind == TraceKind::Input)
}

/// What the user asked of the table this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableAction {
    ExportCsv,
    ExportJson,
}

impl ResultsTable {
    /// The search text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Whether a group starts open: the headline does, every other group starts closed.
    pub fn opens_by_default(group: usize) -> bool {
        group == 0
    }

    /// Whether a group is open: as it starts, unless the user toggled it, and every group while
    /// the search holds more than blanks (its matches are what the user is after).
    pub fn is_open(&self, group: usize) -> bool {
        self.is_open_while(group, self.searching())
    }

    /// Whether the search holds more than blanks.
    fn searching(&self) -> bool {
        !search_needle(&self.query).is_empty()
    }

    /// [`ResultsTable::is_open`] given [`ResultsTable::searching`], which a frame works out
    /// once for every group.
    fn is_open_while(&self, group: usize, searching: bool) -> bool {
        let default = Self::opens_by_default(group);
        let toggled = self.toggled.contains(&group);
        let chosen = if toggled { !default } else { default };
        chosen || searching
    }

    /// Turns the trace filter off unless an input is traced: it keeps the results an input's
    /// trace marks, goes off with the trace, and a result's trace (which marks inputs) cannot
    /// turn it on. [`ResultsTable::ui`] calls it, and the panel calls it after every frame's
    /// trace changes whichever view the centre region shows, so a trace ended or replaced while
    /// the table is off screen cannot leave the box ticked for a later trace (decision O-8).
    pub(crate) fn sync_trace_filter(&mut self, trace: Option<&Trace>) {
        if !is_input_trace(trace) {
            self.traced_only = false;
        }
    }

    /// Draws the table: the search box, the order toggle, the failing and trace filters, the
    /// count and the export buttons, then the lines on screen (the end-effect banner is the
    /// centre region's, over every view: decision M42-1), each row a readout (`readouts`, which
    /// frame the rows `trace` marks), each group heading a button that opens or closes it (not
    /// while the search holds more than blanks: every group is open then), with the worst level
    /// of its checks and the rows the trace marks among those shown. Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        trace: Option<&Trace>,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
        self.sync_trace_filter(trace);
        let input_traced = is_input_trace(trace);
        let entries = table_entries();
        let mut action = None;
        // The failing checks' rows, the red first, when the filter is on (after its checkbox).
        let mut failing: Option<Vec<usize>> = None;
        // The search's rows the trace marks, when the trace filter is on (after its checkbox).
        let mut traced: Option<Vec<usize>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text(SEARCH_HINT)
                    .desired_width(260.0),
            );
            for order in ResultOrder::ALL {
                ui.selectable_value(&mut self.order, order, order.label());
            }
            ui.checkbox(&mut self.failing_only, FAILING_ONLY)
                .on_hover_text(
                    "The checks that fail (red) or ask for a look (amber), the red first",
                );
            ui.add_enabled(
                input_traced,
                egui::Checkbox::new(&mut self.traced_only, TRACED_ONLY),
            )
            .on_hover_text("The results an input's trace marks: click the input's label");
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            traced = self.traced(trace);
            let matches = self.shown(&traced);
            failing = self.failing_only.then(|| {
                failing_checks(results)
                    .iter()
                    .filter_map(|(path, _)| entry_index(path))
                    .collect()
            });
            let shown = match &failing {
                Some(rows) => rows
                    .iter()
                    .filter(|index| matches.binary_search(index).is_ok())
                    .count(),
                None => matches.len(),
            };
            ui.weak(format!("{shown} of {} results", entries.len()));
            if ui.button(EXPORT_CSV).clicked() {
                action = Some(TableAction::ExportCsv);
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
        let matches = self.shown(&traced);
        let searching = self.searching();
        let lines = table_lines(matches, self.order, failing.as_deref(), &|group| {
            self.is_open_while(group, searching)
        });
        if lines.is_empty() {
            // The trace's text only when the trace itself marks none of the rows the failing
            // filter lets through (an input's trace may reach no result); a search that hides
            // the traced rows is the search's doing.
            let marked = |index: usize| trace.is_some_and(|t| t.marks(&entries[index].path));
            let trace_marks_none = self.traced_only
                && match &failing {
                    Some(rows) => !rows.iter().any(|&index| marked(index)),
                    None => !(0..entries.len()).any(marked),
                };
            ui.weak(empty_text(trace_marks_none, self.failing_only, searching));
        }
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let spacing = ui.spacing().item_spacing.x;
        let widths = column_widths(ui.available_width(), spacing);
        let row_width = widths.iter().sum::<f32>() + 3.0 * spacing;
        let mut clicked = None;
        // In the height left, however short (egui's 64-point floor lowered to none).
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .min_scrolled_height(0.0)
            .show_rows(ui, row_height, lines.len(), |ui, range| {
                for line in &lines[range] {
                    match *line {
                        Line::OtherResults => {
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(row_width, row_height),
                                egui::Sense::hover(),
                            );
                            ui.painter().text(
                                rect.left_center(),
                                egui::Align2::LEFT_CENTER,
                                OTHER_RESULTS,
                                egui::TextStyle::Body.resolve(ui.style()),
                                ui.visuals().strong_text_color(),
                            );
                        }
                        Line::Group { group, shown, open } => {
                            let heading = &result_groups()[group];
                            // A trace counts the rows it frames among the group's rows the
                            // filters let through (`matches`, ascending), as `shown` counts
                            // them, so a search never shows more traced rows than rows.
                            let traced = trace.map_or(0, |trace| {
                                trace.count_in(
                                    heading
                                        .rows
                                        .iter()
                                        .filter(|index| matches.binary_search(index).is_ok())
                                        .map(|&index| entries[index].path.as_str()),
                                )
                            });
                            let text = if traced > 0 {
                                format!("{} ({shown}, {traced} traced)", heading.label)
                            } else {
                                format!("{} ({shown})", heading.label)
                            };
                            // The worst level of the group's checks: a closed group shows that
                            // one of them fails.
                            let level = worst_level(results, &heading.rows);
                            let line = HeadingLine {
                                text: &text,
                                open,
                                level,
                                indent: if heading.other { OTHER_INDENT } else { 0.0 },
                                searching,
                            };
                            if group_heading_ui(ui, &line, egui::vec2(row_width, row_height)) {
                                clicked = Some(group);
                            }
                        }
                        Line::Row(index) => {
                            let entry = &entries[index];
                            let level = entry_level(results, entry);
                            row_ui(ui, entry, results, level, row_height, widths, readouts);
                        }
                    }
                }
            });
        // A click while searching would change nothing on screen: it is ignored, so a group is
        // as the user left it once the search is cleared.
        if let Some(group) = clicked
            && !searching
            && !self.toggled.remove(&group)
        {
            self.toggled.insert(group);
        }
        action
    }

    /// The rows the search and the trace filter let through: `traced` while the filter is on,
    /// else every row the search matches (the failing filter cuts them further).
    fn shown<'a>(&'a self, traced: &'a Option<Vec<usize>>) -> &'a [usize] {
        match traced {
            Some(rows) => rows,
            None => self.matches.as_deref().unwrap_or(&[]),
        }
    }

    /// The search's rows the trace marks, while the trace filter is on (an input is traced);
    /// `None` while it is off: every row the search matches is shown.
    fn traced(&self, trace: Option<&Trace>) -> Option<Vec<usize>> {
        let trace = trace.filter(|_| self.traced_only)?;
        let entries = table_entries();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        Some(
            matches
                .iter()
                .copied()
                .filter(|&index| trace.marks(&entries[index].path))
                .collect(),
        )
    }
}

/// What a group's heading line shows.
struct HeadingLine<'a> {
    /// The group's label and its count.
    text: &'a str,
    open: bool,
    /// The worst level of the group's checks, a badge after the text; `None` for no check.
    level: Option<Level>,
    /// Points in from the left (a package group's, under "Other results").
    indent: f32,
    /// The search holds more than blanks: every group is open, and a click does nothing.
    searching: bool,
}

/// A group's heading, `size` points: the open or closed triangle of egui's collapsing header,
/// then the text and the badge of `line`. Returns whether it was clicked.
fn group_heading_ui(ui: &mut egui::Ui, line: &HeadingLine, size: egui::Vec2) -> bool {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    // The triangle where egui's collapsing header puts it: its inner icon square, centred in
    // the indent.
    let indent_width = ui.spacing().indent;
    let (mut icon, _) = ui.spacing().icon_rectangles(rect);
    icon.set_center(egui::pos2(
        rect.left() + line.indent + indent_width / 2.0,
        rect.center().y,
    ));
    let openness = if line.open { 1.0 } else { 0.0 };
    egui::collapsing_header::paint_default_icon(
        ui,
        openness,
        &response.clone().with_new_rect(icon),
    );
    let text = ui.painter().text(
        egui::pos2(rect.left() + line.indent + indent_width, rect.center().y),
        egui::Align2::LEFT_CENTER,
        line.text,
        egui::TextStyle::Body.resolve(ui.style()),
        ui.visuals().strong_text_color(),
    );
    if line.level.is_some() {
        let at = egui::pos2(
            text.right() + ui.spacing().item_spacing.x,
            rect.center().y - 6.0,
        );
        // In a child Ui: the badge takes no room from the table's lines, all of one height.
        let mut badge_ui = ui.new_child(
            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(at, egui::vec2(12.0, 12.0))),
        );
        badge(&mut badge_ui, line.level);
    }
    response
        .on_hover_text(if line.searching {
            CLEAR_TO_CLOSE
        } else if line.open {
            "Close the group"
        } else {
            "Open the group"
        })
        .clicked()
}

/// A row's hover text: the hover hook's ([`hover_text`]) with the exact value.
pub fn row_tooltip(entry: &TableEntry, value: &Value) -> String {
    let mut tooltip = hover_text(&entry.path).expect("every table row is a result");
    if let Value::Num(x) = value {
        tooltip.push_str(&format!("\nExact value: {}", exact_number(*x)));
    }
    tooltip
}

/// One table row: label, value with unit (after the badge of a check's `level`), workbook
/// cell, marker, in columns `widths` wide ([`column_widths`]; the path is in the hover text,
/// which is built only while the row is hovered). The whole row is a readout: hover it for its
/// equation, click it to open it.
fn row_ui(
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
    level: Option<Level>,
    height: f32,
    widths: [f32; 4],
    readouts: &mut Readouts,
) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
        let cell = |ui: &mut egui::Ui, width: f32, text: &str, level: Option<Level>| {
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            ui.allocate_ui_with_layout(egui::vec2(width, height), layout, |ui| {
                ui.set_min_width(width);
                if level.is_some() {
                    badge(ui, level);
                }
                ui.add(egui::Label::new(text).truncate());
            });
        };
        let [label, number, workbook, marker] = widths;
        cell(ui, label, entry.info.meta.label, None);
        cell(
            ui,
            number,
            &with_unit(format_value(&value), entry.info.meta.unit),
            level,
        );
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or(RUST_ONLY),
            None,
        );
        cell(ui, marker, &entry.marker, None);
    });
    readouts.show_over(ui, row.response.rect, &entry.path, || {
        row_tooltip(entry, &value)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A positive beta typed in for manual magnets with no grade (E20): no hot limit, +inf, and
    /// no torque at it, NaN (`tests/robustness.rs`).
    fn non_finite_design() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.0035;
        inputs
    }

    #[test]
    fn the_table_lists_every_result_once_in_schema_order() {
        let entries = table_entries();
        let rows = result_rows(&compute_all(&DesignInputs::default()));
        assert_eq!(entries.len(), rows.len());
        for (entry, row) in entries.iter().zip(&rows) {
            assert_eq!(entry.path, row.path);
            assert_eq!(entry.info.cell, row.cell);
        }
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout.marker, "E3 E7 E8");
    }

    #[test]
    fn the_label_column_flexes_down_to_its_minimum() {
        // The 1280 x 800 default window: the label column widens past the M4-1 260 points.
        assert_eq!(column_widths(644.0, 8.0), [300.0, 130.0, 130.0, 60.0]);
        // A ~930 px window leaves about 294 points: the label shrinks to its minimum, so the
        // value column ends at 120 + 8 + 130 = 258 points, on screen.
        assert_eq!(column_widths(294.0, 8.0), [120.0, 130.0, 130.0, 60.0]);
        assert_eq!(column_widths(0.0, 8.0)[0], LABEL_MIN_WIDTH);
        assert_eq!(column_widths(f32::NAN, 8.0)[0], LABEL_MIN_WIDTH);
    }

    #[test]
    fn the_search_matches_label_path_and_cell_ignoring_case() {
        let entries = table_entries();
        let paths = |query: &str| -> Vec<&str> {
            search(entries, query)
                .into_iter()
                .map(|i| entries[i].path.as_str())
                .collect()
        };
        assert_eq!(paths("calculator!c93"), ["model.pullout_Nm"]);
        assert_eq!(paths("  MODEL.PULLOUT_20C_NM "), ["model.pullout_20C_Nm"]);
        assert!(paths("pull-out torque").contains(&"model.pullout_Nm"));
        let row3: Vec<&str> = paths("gap_sweep[3].");
        let columns = result_rows(&compute_all(&DesignInputs::default()))
            .iter()
            .filter(|r| r.path.starts_with("gap_sweep[3]."))
            .count();
        assert_eq!(row3.len(), columns);
        assert_eq!(search(entries, "").len(), entries.len());
        assert_eq!(search(entries, "   ").len(), entries.len());
        assert!(search(entries, "no such result anywhere").is_empty());
    }

    #[test]
    fn a_row_tooltip_is_the_hover_hook_s_with_the_exact_value() {
        let entries = table_entries();
        let pullout = entries
            .iter()
            .find(|e| e.path == "model.pullout_Nm")
            .unwrap();
        let value = Value::Num(2.6884762950539796);
        let tooltip = row_tooltip(pullout, &value);
        assert!(
            tooltip.starts_with("Pull-out torque at operating temperature\n"),
            "{tooltip}"
        );
        assert!(tooltip.contains("Corrected vs workbook:"), "{tooltip}");
        assert!(
            tooltip.ends_with("\nExact value: 2.6884762950539796"),
            "{tooltip}"
        );
        let claim = entries
            .iter()
            .find(|e| e.path == "housing.space_claim_check")
            .unwrap();
        let text = row_tooltip(claim, &Value::Text("Inside the space claim".to_owned()));
        assert!(!text.contains("Exact value"), "{text}");
    }

    #[test]
    fn exact_numbers_read_back_bit_for_bit() {
        for x in [
            2.6884762950539796,
            5.27e-10,
            -0.1031743945660739,
            50.0,
            1e300,
            0.0,
        ] {
            assert_eq!(exact_number(x).parse::<f64>().unwrap(), x, "{x}");
        }
        assert_eq!(exact_number(5.27e-10), "5.27e-10");
        assert_eq!(exact_number(f64::INFINITY), "+inf");
        assert_eq!(exact_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(exact_number(f64::NAN), "NaN");
    }

    #[test]
    fn the_csv_has_a_line_per_result_with_quoted_text() {
        let results = compute_all(&DesignInputs::default());
        let csv = results_csv(&results);
        let lines: Vec<&str> = csv.split("\r\n").collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(
            lines.len(),
            result_rows(&results).len() + 2,
            "header, rows, final break"
        );
        assert_eq!(lines.last(), Some(&""));
        let pullout = lines
            .iter()
            .find(|l| l.starts_with("model.pullout_Nm,"))
            .unwrap();
        assert_eq!(
            *pullout,
            format!(
                "model.pullout_Nm,Pull-out torque at operating temperature,{},N·m,Calculator!C93",
                exact_number(results.model.pullout_Nm)
            )
        );
        let mass = lines
            .iter()
            .find(|l| l.starts_with("mass.total_g,"))
            .unwrap();
        assert!(
            mass.starts_with("mass.total_g,\"Preliminary rotating mass, including retainers\","),
            "{mass}"
        );
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_field("a\nb"), "\"a\nb\"");
        // A Rust-only result has an empty cell field.
        let claim = lines
            .iter()
            .find(|l| l.starts_with("housing.space_claim_check,"))
            .unwrap();
        assert!(claim.ends_with(",Inside the space claim,,"), "{claim}");
    }

    #[test]
    fn non_finite_results_export_as_text_in_csv_and_json() {
        let design = Design {
            inputs: non_finite_design(),
            ..Design::default()
        };
        let results = compute_all(&design.inputs);
        assert_eq!(results.temperature.demag.magnet_limit_C, f64::INFINITY);
        assert!(results.temperature.demag.torque_at_limit_Nm.is_nan());
        let csv = results_csv(&results);
        let line = |path: &str| {
            csv.split("\r\n")
                .find(|l| l.starts_with(&format!("{path},")))
                .unwrap()
                .to_owned()
        };
        assert!(line("temperature.demag.magnet_limit_C").contains(",+inf,"));
        assert!(line("temperature.demag.torque_at_limit_Nm").contains(",NaN,"));
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        let value = |path: &str| {
            json["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["path"] == path)
                .unwrap()["value"]
                .clone()
        };
        assert_eq!(
            value("temperature.demag.magnet_limit_C"),
            Json::from("+inf")
        );
        assert_eq!(
            value("temperature.demag.torque_at_limit_Nm"),
            Json::from("NaN")
        );
    }

    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
        let design = Design::default();
        let results = compute_all(&design.inputs);
        let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
        assert_eq!(json["format"], RESULTS_FORMAT);
        assert_eq!(json["version"], RESULTS_VERSION);
        assert_eq!(json["design"], design_json(&design));
        let rows = json["results"].as_array().unwrap();
        assert_eq!(rows.len(), result_rows(&results).len());
        let pullout = rows
            .iter()
            .find(|r| r["path"] == "model.pullout_Nm")
            .unwrap();
        assert_eq!(pullout["value"].as_f64(), Some(results.model.pullout_Nm));
        assert_eq!(pullout["unit"], "N·m");
        assert_eq!(pullout["cell"], "Calculator!C93");
        let claim = rows
            .iter()
            .find(|r| r["path"] == "housing.space_claim_check")
            .unwrap();
        assert_eq!(claim["cell"], Json::Null);
    }

    /// The row indices of `lines`, headings left out.
    fn rows_of(lines: &[Line]) -> Vec<usize> {
        lines
            .iter()
            .filter_map(|line| match line {
                Line::Row(index) => Some(*index),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_lines_group_the_rows_with_only_the_headline_open_at_first() {
        let entries = table_entries();
        let groups = result_groups();
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(
            &all,
            ResultOrder::Grouped,
            None,
            &ResultsTable::opens_by_default,
        );
        // The headline's heading and its rows, then every other group's closed heading, the
        // package groups under "Other results".
        let headline = groups[0].rows.len();
        assert_eq!(
            lines[0],
            Line::Group {
                group: 0,
                shown: headline,
                open: true
            }
        );
        assert_eq!(rows_of(&lines[1..=headline]), groups[0].rows);
        let mut want = Vec::new();
        for (group, entry) in groups.iter().enumerate().skip(1) {
            if entry.other && !want.contains(&Line::OtherResults) {
                want.push(Line::OtherResults);
            }
            want.push(Line::Group {
                group,
                shown: entry.rows.len(),
                open: false,
            });
        }
        assert_eq!(lines[headline + 1..], want[..]);
        // Every group open: every row once.
        let lines = table_lines(&all, ResultOrder::Grouped, None, &|_| true);
        let mut rows = rows_of(&lines);
        assert_eq!(rows.len(), entries.len());
        rows.sort_unstable();
        assert_eq!(rows, all);
    }

    #[test]
    fn a_search_shows_only_the_groups_it_matches() {
        let entries = table_entries();
        let groups = result_groups();
        let pullout = entry_index("model.pullout_Nm").unwrap();
        assert_eq!(entries[pullout].path, "model.pullout_Nm");
        let matches = search(entries, "calculator!c93");
        let lines = table_lines(&matches, ResultOrder::Grouped, None, &|_| true);
        assert_eq!(
            lines,
            [
                Line::Group {
                    group: 0,
                    shown: 1,
                    open: true
                },
                Line::Row(pullout)
            ]
        );
        // A sweep row's columns: the Other results heading, the gap sweep's heading, the rows.
        let matches = search(entries, "gap_sweep[3].");
        let lines = table_lines(&matches, ResultOrder::Grouped, None, &|_| true);
        let gap_sweep = groups.iter().position(|g| g.id == "gap_sweep").unwrap();
        assert_eq!(lines[0], Line::OtherResults);
        assert_eq!(
            lines[1],
            Line::Group {
                group: gap_sweep,
                shown: matches.len(),
                open: true
            }
        );
        assert_eq!(rows_of(&lines), matches);
        assert!(table_lines(&[], ResultOrder::Grouped, None, &|_| true).is_empty());
        // The search opens every group it matches, whatever the user toggled.
        let mut table = ResultsTable::default();
        assert!(table.is_open(0) && !table.is_open(1));
        table.toggled.insert(0);
        assert!(!table.is_open(0));
        table.query = " f_end ".to_owned();
        assert!(table.is_open(0) && table.is_open(1));
        assert!(!ResultsTable::opens_by_default(1));
    }

    #[test]
    fn the_engine_order_and_the_failing_filter_are_flat() {
        let entries = table_entries();
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(&all, ResultOrder::Engine, None, &|_| false);
        assert_eq!(rows_of(&lines), all);
        assert_eq!(lines.len(), all.len(), "no headings");
        // The failing filter: exactly the failing checks, the red first, in either order.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        let results = compute_all(&inputs);
        let failing: Vec<usize> = failing_checks(&results)
            .iter()
            .map(|(path, _)| entry_index(path).unwrap())
            .collect();
        assert_eq!(failing.len(), 6);
        for order in ResultOrder::ALL {
            let lines = table_lines(&all, order, Some(&failing), &|_| true);
            assert_eq!(
                lines,
                failing.iter().map(|&i| Line::Row(i)).collect::<Vec<_>>()
            );
        }
        // And only those the search matches: the two amber rating checks.
        let matches = search(entries, "temperature check");
        let lines = table_lines(&matches, ResultOrder::Grouped, Some(&failing), &|_| true);
        assert_eq!(
            lines,
            [
                Line::Row(entry_index("model.inner_temp_check").unwrap()),
                Line::Row(entry_index("model.outer_temp_check").unwrap())
            ]
        );
        // Each check's row knows it is one.
        for check in CHECKS {
            assert!(entries[entry_index(check).unwrap()].check, "{check}");
        }
        assert!(!entries[entry_index("model.pullout_Nm").unwrap()].check);
        assert_eq!(entry_index("no.such.result"), None);
    }

    #[test]
    fn a_row_carries_its_check_s_level_and_a_heading_the_worst_of_its_rows() {
        use crate::gui::test_support::short_magnets;
        let entries = table_entries();
        let row = |path: &str| &entries[entry_index(path).unwrap()];
        // The default design: the hot minimum fails (red); a result that is no check has none.
        let results = compute_all(&DesignInputs::default());
        assert_eq!(
            entry_level(&results, row("metal.hot_min_check")),
            Some(Level::Bad)
        );
        assert_eq!(entry_level(&results, row("model.pullout_Nm")), None);
        // Short magnets: the end effect greys the hot minimum, so its row has no level.
        let short = compute_all(&short_magnets());
        assert_eq!(entry_level(&short, row("metal.hot_min_check")), None);
        // Every row: its check's level, none for a row that is no check.
        for results in [&results, &short] {
            for entry in entries {
                let want = CHECKS
                    .contains(&entry.path.as_str())
                    .then(|| check_level(results, &entry.path))
                    .flatten();
                assert_eq!(entry_level(results, entry), want, "{}", entry.path);
            }
            // A heading's level: the worst of its rows' levels.
            for group in result_groups() {
                let levels = group
                    .rows
                    .iter()
                    .filter_map(|&i| entry_level(results, &entries[i]));
                assert_eq!(
                    worst_level(results, &group.rows),
                    levels.max(),
                    "{}",
                    group.label
                );
            }
        }
        let pair = [
            entry_index("model.pullout_Nm").unwrap(),
            entry_index("metal.hot_min_check").unwrap(),
        ];
        assert_eq!(worst_level(&results, &pair), Some(Level::Bad));
        assert_eq!(worst_level(&results, &pair[..1]), None);
        assert_eq!(worst_level(&results, &[]), None);
    }

    #[test]
    fn the_empty_table_names_the_filter_that_empties_it() {
        // The trace's text only when the trace itself marks none of the rows the failing filter
        // lets through, whatever the search; a search that hides the traced rows is the search's.
        for failing_only in [false, true] {
            for searching in [false, true] {
                assert_eq!(empty_text(true, failing_only, searching), NOTHING_TRACED);
            }
        }
        assert_eq!(empty_text(false, false, true), NO_RESULT);
        assert_eq!(empty_text(false, true, true), NO_RESULT);
        // Nothing fails only when no search narrows the failing checks.
        assert_eq!(empty_text(false, true, false), NOTHING_FAILS);
        assert_eq!(empty_text(false, false, false), NO_RESULT);
    }

    #[test]
    fn the_table_stays_in_a_short_region() {
        // The space left above the Equation panel: under its search and filter rows the table
        // scrolls in exactly what is left (no 64-point scroll area floor).
        let ctx = egui::Context::default();
        let results = compute_all(&DesignInputs::default());
        let size = egui::vec2(1000.0, 700.0);
        let mut table = ResultsTable::default();
        for height in [400.0, 120.0, 80.0] {
            let region =
                egui::Rect::from_min_size(egui::pos2(20.0, 30.0), egui::vec2(700.0, height));
            for _ in 0..2 {
                let (_, used) = crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                    table.ui(ui, &results, None, &mut Readouts::default());
                });
                assert!(
                    used.bottom() <= region.bottom() + 0.01,
                    "{height}: the table runs {} points past the region",
                    used.bottom() - region.bottom()
                );
            }
        }
    }
}
