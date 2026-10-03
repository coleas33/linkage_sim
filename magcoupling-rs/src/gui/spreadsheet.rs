//! The spreadsheet export's layout (user request 2026-10-03: "a download button for the
//! spreadsheet equivalent of the current layout of the website"): the design shown and its
//! results laid out as the page shows them, as values (no formulas, and nothing reads a
//! spreadsheet back into the site: decision X-1). [`sheets`] lays the workbook out as rows of
//! cells, one [`Sheet`] per sheet, in order (decision X-2): the Summary (the export, the share
//! link, the sizing state, the banners, the dashboard, the material warnings and the corrections
//! applied), the Inputs in the order the inputs side shows them (decision X-3), the Results by
//! physics chain as the results table groups them (decision X-5) and the Assumptions.
//! `gui::xlsx` writes the sheets as an .xlsx file.
//!
//! Every input and every result is on exactly one row: the Key design inputs are not repeated
//! above their groups, a column flags them instead (decision X-4). A number is a number cell; a
//! number that is not finite is the text the CSV export writes (`+inf`, `-inf`, `NaN`); a check's
//! level is its cell's style, which the writer fills with the badge's colour.

use crate::engine::assumptions;
use crate::engine::deviations::REGISTRY;
use crate::engine::explain::render;
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::warnings::Severity;
use crate::gui::corrections::marks_values;
use crate::gui::dashboard::{
    Level, STORED_3D_LABEL, WARNINGS_HEADING, dashboard_lines, end_effect_banner, result_info,
    severity_level, warning_lines,
};
use crate::gui::input_ui::BLANK_TEXT;
use crate::gui::inputs::{
    ADVANCED_HEADING, InputCatalogue, InputEntry, InputGroup, InputOrder, InputSection, KEY_DESIGN,
};
use crate::gui::panel::{
    FREE_VARIABLE_LABEL, HEADING, SIZED_NOTE, TARGET_LABEL, assumptions_banner,
};
use crate::gui::readouts::registry;
use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
use crate::gui::results_table::{
    Line, RUST_ONLY, ResultOrder, TableEntry, entry_level, exact_number, table_entries,
    table_lines, worst_level,
};
use crate::gui::sizing::{SizingMode, SizingState, TARGET_RANGE_INPUT, variable_label};
use crate::{DesignInputs, DesignResults};

/// The sheets' names, in order (decision X-2).
pub const SUMMARY_SHEET: &str = "Summary";
pub const INPUTS_SHEET: &str = "Inputs";
pub const RESULTS_SHEET: &str = "Results";
pub const ASSUMPTIONS_SHEET: &str = "Assumptions";

/// The Inputs sheet's column headers.
pub const INPUT_COLUMNS: [&str; 10] = [
    "Input",
    "Value",
    "Unit",
    "Workbook cell",
    "Path",
    "Changed from default",
    "Assumption",
    "Advanced",
    "Key design",
    "Note",
];

/// The Results sheet's column headers.
pub const RESULT_COLUMNS: [&str; 8] = [
    "Result",
    "Value",
    "Unit",
    "Check",
    "Workbook cell",
    "Path",
    "Corrected vs workbook",
    "Equation",
];

/// The column headers of the Summary's dashboard rows.
pub const DASHBOARD_COLUMNS: [&str; 8] = [
    "Dashboard",
    "Value",
    "Unit",
    "Check",
    "Workbook cell",
    "Path",
    "Corrected vs workbook",
    "Note",
];

/// The Assumptions sheet's column headers.
pub const ASSUMPTION_COLUMNS: [&str; 8] = [
    "Assumption",
    "Path",
    "Value",
    "Unit",
    "Workbook default",
    "Changed from default",
    "Rationale",
    "Source",
];

/// The Summary's row labels.
pub const EXPORTED: &str = "Exported (UTC)";
pub const APP: &str = "App";
pub const SHARE_LINK: &str = "Share link (paste into a browser)";
pub const SIZING_MODE: &str = "Sizing mode";
pub const SIZING_OUTCOME: &str = "Sizing outcome";

/// The Summary's heading over the corrections applied.
pub const CORRECTIONS_HEADING: &str = "Corrections applied (every approved correction is on)";

/// The Summary's line when no material warning fires.
pub const NO_WARNING: &str = "No material warning fires.";

/// The note of a dashboard row the page greys (audit M9).
pub const GREYED_NOTE: &str = "Greyed on the page: computed from the pull-out (see the banner)";

/// A flag column's text when the flag is set (blank when not).
pub const YES: &str = "yes";

/// The columns' widths [characters].
const SUMMARY_WIDTHS: [f64; 8] = [44.0, 24.0, 10.0, 8.0, 24.0, 44.0, 12.0, 44.0];
const INPUT_WIDTHS: [f64; 10] = [44.0, 16.0, 10.0, 24.0, 44.0, 11.0, 11.0, 10.0, 10.0, 30.0];
const RESULT_WIDTHS: [f64; 8] = [44.0, 16.0, 10.0, 8.0, 24.0, 44.0, 12.0, 70.0];
const ASSUMPTION_WIDTHS: [f64; 8] = [32.0, 40.0, 14.0, 10.0, 14.0, 11.0, 70.0, 50.0];

/// What a cell holds.
#[derive(Clone, Debug, PartialEq)]
pub enum CellValue {
    Empty,
    /// A finite number (a number that is not finite is text: [`value_cell`]).
    Number(f64),
    /// Never empty: an empty text is [`CellValue::Empty`] ([`Cell::text`]).
    Text(String),
    /// A UTC date and time [s since the Unix epoch].
    Time(i64),
}

/// How a cell looks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CellStyle {
    #[default]
    Plain,
    /// The Summary's title.
    Title,
    /// A column header (the frozen row).
    Header,
    /// A group heading: an input group, a result chain, "Other results", a Summary block.
    Group,
    /// A heading inside a group: a section, "Advanced", a package group of the other results.
    Section,
    /// A check's level: the cell is filled with the badge's colour.
    Level(Level),
}

/// One cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub value: CellValue,
    pub style: CellStyle,
}

impl Cell {
    /// An empty cell.
    pub fn empty() -> Self {
        Self {
            value: CellValue::Empty,
            style: CellStyle::Plain,
        }
    }

    /// A number cell.
    pub fn number(x: f64) -> Self {
        Self {
            value: CellValue::Number(x),
            style: CellStyle::Plain,
        }
    }

    /// A text cell; an empty cell for an empty text.
    pub fn text(text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            value: if text.is_empty() {
                CellValue::Empty
            } else {
                CellValue::Text(text)
            },
            style: CellStyle::Plain,
        }
    }

    /// The cell with `style`.
    pub fn styled(mut self, style: CellStyle) -> Self {
        self.style = style;
        self
    }

    /// The cell's text; empty for a cell that holds no text.
    pub fn as_text(&self) -> &str {
        match &self.value {
            CellValue::Text(text) => text,
            _ => "",
        }
    }
}

/// One sheet: its name, its columns' widths, the rows frozen at its top, whether its first row
/// carries Excel's filter buttons, and its rows (a row may be shorter than the sheet is wide).
#[derive(Clone, Debug, PartialEq)]
pub struct Sheet {
    pub name: &'static str,
    /// Each column's width, left to right [characters].
    pub widths: &'static [f64],
    pub frozen_rows: u32,
    /// The first row (the column headers) filters every column below it.
    pub autofilter: bool,
    pub rows: Vec<Vec<Cell>>,
}

/// What the spreadsheet shows: the design on the page when the button was pressed.
#[derive(Clone, Debug)]
pub struct Snapshot<'a> {
    /// The design shown: in Torque -> Magnets the inputs with the free variable at the value
    /// shown.
    pub inputs: &'a DesignInputs,
    /// Its results.
    pub results: &'a DesignResults,
    pub sizing: SizingState,
    /// The sizing outcome line, in Torque -> Magnets.
    pub sizing_status: Option<String>,
    /// The order of the inputs side (decision O-1).
    pub input_order: InputOrder,
    /// The link that reopens the design on the site.
    pub share_link: String,
    /// When it was exported [s since the Unix epoch].
    pub exported_unix_s: i64,
}

/// The cell of an engine value: a number cell for a finite number or an integer, the CSV
/// export's text for a number that is not finite (`+inf`, `-inf`, `NaN`), text as it is, and an
/// empty cell for none.
pub fn value_cell(value: &Value) -> Cell {
    match value {
        Value::Num(x) if x.is_finite() => Cell::number(*x),
        Value::Num(x) => Cell::text(exact_number(*x)),
        Value::Int(i) => Cell::number(*i as f64),
        Value::Text(text) => Cell::text(text.clone()),
        Value::None => Cell::empty(),
    }
}

/// The cell of a check's level: the badge's word, filled with its colour; empty for none.
pub fn level_cell(level: Option<Level>) -> Cell {
    level.map_or_else(Cell::empty, |level| {
        Cell::text(level.word()).styled(CellStyle::Level(level))
    })
}

/// The workbook, sheet by sheet, in order (decision X-2).
pub fn sheets(snapshot: &Snapshot) -> Vec<Sheet> {
    vec![
        summary_sheet(snapshot),
        inputs_sheet(snapshot),
        results_sheet(snapshot.results),
        assumptions_sheet(snapshot.inputs),
    ]
}

/// A row of one heading cell.
fn heading_row(text: &str, style: CellStyle) -> Vec<Cell> {
    vec![Cell::text(text).styled(style)]
}

/// A row of column headers.
fn header_row(columns: &[&str]) -> Vec<Cell> {
    columns
        .iter()
        .map(|column| Cell::text(*column).styled(CellStyle::Header))
        .collect()
}

/// A flag column's cell: [`YES`] when set, empty when not.
fn flag(on: bool) -> Cell {
    if on { Cell::text(YES) } else { Cell::empty() }
}

/// A workbook cell column's cell: the cell, or [`RUST_ONLY`] for none.
fn workbook_cell(cell: Option<&str>) -> Cell {
    Cell::text(cell.unwrap_or(RUST_ONLY))
}

/// The Summary: the title, when and by what it was exported, the share link, the sizing state,
/// the end-effect and assumptions banners when they show, then the dashboard's rows, the material
/// warnings and the corrections applied. Nothing is frozen: its blocks are read top to bottom.
fn summary_sheet(snapshot: &Snapshot) -> Sheet {
    let results = snapshot.results;
    let sizing = snapshot.sizing;
    let mut rows = vec![
        heading_row(HEADING, CellStyle::Title),
        vec![
            Cell::text(EXPORTED),
            Cell {
                value: CellValue::Time(snapshot.exported_unix_s),
                style: CellStyle::Plain,
            },
        ],
        vec![
            Cell::text(APP),
            Cell::text(format!("magcoupling-rs {}", env!("CARGO_PKG_VERSION"))),
        ],
        vec![
            Cell::text(SHARE_LINK),
            Cell::text(snapshot.share_link.clone()),
        ],
        vec![Cell::text(SIZING_MODE), Cell::text(sizing.mode.label())],
    ];
    if sizing.mode == SizingMode::TorqueToMagnets {
        let target_unit = InputCatalogue::get()
            .entry(TARGET_RANGE_INPUT)
            .map_or("", |entry| entry.meta.unit);
        rows.push(vec![
            Cell::text(FREE_VARIABLE_LABEL),
            Cell::text(variable_label(sizing.variable)),
        ]);
        rows.push(vec![
            Cell::text(TARGET_LABEL),
            value_cell(&Value::Num(sizing.target_Nm)),
            Cell::text(target_unit),
        ]);
        if let Some(status) = &snapshot.sizing_status {
            rows.push(vec![Cell::text(SIZING_OUTCOME), Cell::text(status.clone())]);
        }
    }
    if let Some(banner) = end_effect_banner(results) {
        rows.push(heading_row(&banner, CellStyle::Level(Level::Bad)));
    }
    if let Some(banner) = assumptions_banner(snapshot.inputs) {
        rows.push(heading_row(&banner, CellStyle::Level(Level::Caution)));
    }
    rows.push(Vec::new());
    rows.push(header_row(&DASHBOARD_COLUMNS));
    for line in dashboard_lines(results) {
        let info = result_info(line.path).expect("a dashboard row is a result");
        let note = if line.greyed {
            GREYED_NOTE
        } else if line.stored_3d {
            STORED_3D_LABEL
        } else {
            ""
        };
        rows.push(vec![
            Cell::text(line.label),
            value_cell(&results.get(line.path).unwrap_or(Value::None)),
            Cell::text(info.meta.unit),
            level_cell(line.level),
            workbook_cell(info.cell.as_deref()),
            Cell::text(line.path),
            Cell::text(line.marker.clone()),
            Cell::text(note),
        ]);
    }
    rows.push(Vec::new());
    rows.push(heading_row(WARNINGS_HEADING, CellStyle::Group));
    let warnings = warning_lines(results);
    if warnings.is_empty() {
        rows.push(vec![Cell::text(NO_WARNING)]);
    }
    for (rule, text) in warnings {
        let severity = match rule.severity {
            Severity::Warning => "Warning",
            Severity::Caution => "Caution",
        };
        rows.push(vec![
            Cell::text(severity).styled(CellStyle::Level(severity_level(rule.severity))),
            Cell::text(text),
        ]);
    }
    rows.push(Vec::new());
    rows.push(heading_row(CORRECTIONS_HEADING, CellStyle::Group));
    for deviation in REGISTRY.iter().filter(|deviation| marks_values(deviation)) {
        rows.push(vec![
            Cell::text(deviation.id.to_string()),
            Cell::text(deviation.title),
        ]);
    }
    Sheet {
        name: SUMMARY_SHEET,
        widths: &SUMMARY_WIDTHS,
        frozen_rows: 0,
        autofilter: false,
        rows,
    }
}

/// The Inputs, in the order the inputs side shows them (decision X-3), drawn by the page's own
/// rules: each group's heading, its plain sections ([`InputGroup::plain_sections`]), then in the
/// workflow order its Advanced heading and the advanced sections
/// ([`InputGroup::advanced_sections`]).
fn inputs_sheet(snapshot: &Snapshot) -> Sheet {
    let sized = (snapshot.sizing.mode == SizingMode::TorqueToMagnets)
        .then(|| snapshot.sizing.variable.path());
    let mut rows = vec![header_row(&INPUT_COLUMNS)];
    for group in InputCatalogue::get().groups_in(snapshot.input_order) {
        rows.push(heading_row(group.label, CellStyle::Group));
        for section in group.plain_sections() {
            section_rows(&mut rows, group, section, snapshot.inputs, sized);
        }
        if group.has_advanced() {
            rows.push(heading_row(ADVANCED_HEADING, CellStyle::Section));
            for section in group.advanced_sections() {
                section_rows(&mut rows, group, section, snapshot.inputs, sized);
            }
        }
    }
    Sheet {
        name: INPUTS_SHEET,
        widths: &INPUT_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

/// One section's rows: its heading ([`InputSection::heading_in`]: none for the group's own
/// section), then its inputs.
fn section_rows(
    rows: &mut Vec<Vec<Cell>>,
    group: &InputGroup,
    section: &InputSection,
    inputs: &DesignInputs,
    sized: Option<&str>,
) {
    if let Some(heading) = section.heading_in(group) {
        rows.push(heading_row(heading, CellStyle::Section));
    }
    for entry in &section.entries {
        rows.push(input_row(entry, section.advanced, inputs, sized));
    }
}

/// One input's row: label, value, unit, workbook cell, path, its flags, and a note naming a
/// selector's choice, a blank optional input or empty text (a part name left out for manual
/// magnets), or the free variable Torque -> Magnets set (`sized`).
fn input_row(
    entry: &InputEntry,
    advanced: bool,
    inputs: &DesignInputs,
    sized: Option<&str>,
) -> Vec<Cell> {
    let meta = entry.meta;
    let value = inputs.get(&entry.path).unwrap_or(Value::None);
    let mut notes = Vec::new();
    if sized == Some(entry.path.as_str()) {
        notes.push(SIZED_NOTE.to_owned());
    }
    match &value {
        Value::Int(code) if !meta.choices.is_empty() => {
            if let Some((_, text)) = meta.choices.iter().find(|(c, _)| c == code) {
                notes.push(format!("{code} = {text}"));
            }
        }
        Value::None => notes.push(BLANK_TEXT.to_owned()),
        Value::Text(text) if text.is_empty() => notes.push(BLANK_TEXT.to_owned()),
        _ => {}
    }
    vec![
        Cell::text(meta.label),
        value_cell(&value),
        Cell::text(meta.unit),
        workbook_cell(meta.cell),
        Cell::text(entry.path.clone()),
        flag(value != entry.default),
        flag(meta.assumption),
        flag(advanced),
        flag(KEY_DESIGN.contains(&entry.path.as_str())),
        Cell::text(notes.join("; ")),
    ]
}

/// The Results by physics chain, as the results table groups them (decision X-5): the table's
/// own grouped lines ([`table_lines`], every row, every group open), so the headings are the
/// table's, in its order (the package groups under "Other results"; a group with no row is left
/// out, as the table leaves it out). Each group's heading carries the worst level of its checks,
/// each result its row ([`result_row`]).
fn results_sheet(results: &DesignResults) -> Sheet {
    let entries = table_entries();
    let all: Vec<usize> = (0..entries.len()).collect();
    let mut rows = vec![header_row(&RESULT_COLUMNS)];
    for line in table_lines(&all, ResultOrder::Grouped, None, &|_| true) {
        match line {
            Line::OtherResults => rows.push(heading_row(OTHER_RESULTS, CellStyle::Group)),
            Line::Group { group: index, .. } => {
                let group = &result_groups()[index];
                let style = if group.other {
                    CellStyle::Section
                } else {
                    CellStyle::Group
                };
                let mut heading = heading_row(group.label, style);
                heading.extend([
                    Cell::empty(),
                    Cell::empty(),
                    level_cell(worst_level(results, &group.rows)),
                ]);
                rows.push(heading);
            }
            Line::Row(index) => rows.push(result_row(&entries[index], results)),
        }
    }
    Sheet {
        name: RESULTS_SHEET,
        widths: &RESULT_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

/// One result's row: label, value, unit, the level of its row's badge ([`entry_level`]),
/// workbook cell, path, corrections, and the equation as plain text where a record exists.
fn result_row(entry: &TableEntry, results: &DesignResults) -> Vec<Cell> {
    let equation = registry()
        .equation_for(&entry.path)
        .map_or_else(String::new, |eq| {
            render::plain(registry(), &eq.symbol, &eq.formula)
        });
    vec![
        Cell::text(entry.info.meta.label),
        value_cell(&results.get(&entry.path).unwrap_or(Value::None)),
        Cell::text(entry.info.meta.unit),
        level_cell(entry_level(results, entry)),
        workbook_cell(entry.info.cell.as_deref()),
        Cell::text(entry.path.clone()),
        Cell::text(entry.marker.clone()),
        Cell::text(equation),
    ]
}

/// The Assumptions, as the Assumptions view lists them: one row per input of each assumption,
/// with its workbook default, whether it differs, the rationale and the source.
fn assumptions_sheet(inputs: &DesignInputs) -> Sheet {
    let mut rows = vec![header_row(&ASSUMPTION_COLUMNS)];
    for state in assumptions::states(inputs) {
        let assumption = state.assumption;
        let values = state.values.iter().zip(&state.defaults);
        for (path, (value, default)) in assumption.paths.iter().zip(values) {
            rows.push(vec![
                Cell::text(assumption.label),
                Cell::text(*path),
                value_cell(value),
                Cell::text(state.unit),
                value_cell(default),
                flag(value != default),
                Cell::text(assumption.rationale),
                Cell::text(assumption.source),
            ]);
        }
    }
    Sheet {
        name: ASSUMPTIONS_SHEET,
        widths: &ASSUMPTION_WIDTHS,
        frozen_rows: 1,
        autofilter: true,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::assumptions::ASSUMPTIONS;
    use crate::engine::sizing::FreeVariable;
    use crate::gui::dashboard::{CHECKS, DASHBOARD, check_level};
    use crate::gui::test_support::short_magnets;

    /// The snapshot of `inputs` and `results` in Magnets -> Torque, the workflow order.
    fn snapshot<'a>(inputs: &'a DesignInputs, results: &'a DesignResults) -> Snapshot<'a> {
        Snapshot {
            inputs,
            results,
            sizing: SizingState::default(),
            sizing_status: None,
            input_order: InputOrder::Workflow,
            share_link: "https://example.test/magcoupling/?m=abc".to_owned(),
            exported_unix_s: 1_790_000_000,
        }
    }

    fn sheet<'s>(book: &'s [Sheet], name: &str) -> &'s Sheet {
        book.iter()
            .find(|sheet| sheet.name == name)
            .unwrap_or_else(|| panic!("no sheet {name}"))
    }

    /// The index of the column headed `name`.
    fn column(columns: &[&str], name: &str) -> usize {
        columns
            .iter()
            .position(|column| *column == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    }

    /// The rows below the header that name a path in column `path`: one per input or result.
    fn path_rows(sheet: &Sheet, path: usize) -> Vec<&Vec<Cell>> {
        sheet.rows[sheet.frozen_rows as usize..]
            .iter()
            .filter(|row| row.len() > path && !row[path].as_text().is_empty())
            .collect()
    }

    /// The row of `sheet` whose column `path` holds `wanted`.
    fn row_of<'s>(sheet: &'s Sheet, path: usize, wanted: &str) -> &'s Vec<Cell> {
        path_rows(sheet, path)
            .into_iter()
            .find(|row| row[path].as_text() == wanted)
            .unwrap_or_else(|| panic!("no row {wanted}"))
    }

    /// The Summary row whose first cell reads `label`.
    fn summary_row<'s>(book: &'s [Sheet], label: &str) -> Option<&'s Vec<Cell>> {
        sheet(book, SUMMARY_SHEET)
            .rows
            .iter()
            .find(|row| row.first().map(Cell::as_text) == Some(label))
    }

    #[test]
    fn the_workbook_has_the_summary_inputs_results_and_assumptions_sheets() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let names: Vec<&str> = book.iter().map(|sheet| sheet.name).collect();
        assert_eq!(
            names,
            [
                SUMMARY_SHEET,
                INPUTS_SHEET,
                RESULTS_SHEET,
                ASSUMPTIONS_SHEET
            ]
        );
        for (sheet, columns) in [
            (&book[1], &INPUT_COLUMNS[..]),
            (&book[2], &RESULT_COLUMNS[..]),
            (&book[3], &ASSUMPTION_COLUMNS[..]),
        ] {
            assert_eq!(sheet.frozen_rows, 1, "{}", sheet.name);
            let header: Vec<&str> = sheet.rows[0].iter().map(Cell::as_text).collect();
            assert_eq!(header, columns, "{}", sheet.name);
            assert!(
                sheet.rows[0].iter().all(|c| c.style == CellStyle::Header),
                "{}",
                sheet.name
            );
        }
        // The Summary is read top to bottom: nothing frozen, no filter; each table sheet's
        // header row is frozen and filters its columns.
        let frozen: Vec<(u32, bool)> = book.iter().map(|s| (s.frozen_rows, s.autofilter)).collect();
        assert_eq!(frozen, [(0, false), (1, true), (1, true), (1, true)]);
        assert_eq!(
            book[0].rows[0],
            [Cell::text(HEADING).styled(CellStyle::Title)]
        );
        for sheet in &book {
            assert_eq!(
                sheet.widths.len(),
                sheet.rows.iter().map(Vec::len).max().unwrap()
            );
        }
    }

    #[test]
    fn every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order() {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41;
        let results = compute_all(&inputs);
        let catalogue = InputCatalogue::get();
        let path = column(&INPUT_COLUMNS, "Path");
        for order in InputOrder::ALL {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            let book = sheets(&shot);
            let rows = path_rows(sheet(&book, INPUTS_SHEET), path);
            for row in &rows {
                let entry = catalogue.entry(row[path].as_text()).expect("an input");
                assert_eq!(row[0], Cell::text(entry.meta.label));
                assert_eq!(row[1], value_cell(&inputs.get(&entry.path).unwrap()));
                assert_eq!(row[2], Cell::text(entry.meta.unit));
                assert_eq!(row[3], Cell::text(entry.meta.cell.unwrap_or(RUST_ONLY)));
            }
            let mut paths: Vec<&str> = rows.iter().map(|row| row[path].as_text()).collect();
            assert_eq!(paths.len(), catalogue.all().count(), "{order:?}");
            paths.sort_unstable();
            paths.dedup();
            assert_eq!(
                paths.len(),
                catalogue.all().count(),
                "each input once ({order:?})"
            );
        }
    }

    #[test]
    fn the_inputs_follow_the_order_shown_under_their_group_section_and_advanced_headings() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let catalogue = InputCatalogue::get();
        let path = column(&INPUT_COLUMNS, "Path");
        let advanced = column(&INPUT_COLUMNS, "Advanced");
        for order in InputOrder::ALL {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            let book = sheets(&shot);
            let sheet = sheet(&book, INPUTS_SHEET);
            let groups = catalogue.groups_in(order);
            let headings: Vec<&str> = sheet
                .rows
                .iter()
                .filter(|row| row[0].style == CellStyle::Group)
                .map(|row| row[0].as_text())
                .collect();
            let want: Vec<&str> = groups.iter().map(|group| group.label).collect();
            assert_eq!(headings, want, "{order:?}");
            // The inputs in exactly the order the page draws them: group by group, the plain
            // sections, then the advanced ones, each section's inputs in order.
            let got: Vec<&str> = path_rows(sheet, path)
                .iter()
                .map(|row| row[path].as_text())
                .collect();
            let drawn: Vec<&str> = groups
                .iter()
                .flat_map(|g| g.plain_sections().chain(g.advanced_sections()))
                .flat_map(|s| s.entries.iter().map(|e| e.path.as_str()))
                .collect();
            assert_eq!(got, drawn, "{order:?}");
            // Each input under its own group, after its section's heading, and an advanced input
            // only after its group's Advanced heading.
            let mut group: Option<&InputGroup> = None;
            let mut section: Option<&str> = None;
            let mut in_advanced = false;
            let mut advanced_headings = 0;
            for row in &sheet.rows[1..] {
                match row[0].style {
                    CellStyle::Group => {
                        group = groups.iter().find(|g| g.label == row[0].as_text());
                        section = None;
                        in_advanced = false;
                    }
                    CellStyle::Section if row[0].as_text() == ADVANCED_HEADING => {
                        in_advanced = true;
                        advanced_headings += 1;
                    }
                    CellStyle::Section => section = Some(row[0].as_text()),
                    _ => {
                        let input = row[path].as_text();
                        let (holder, held) = catalogue.section_of(order, input).expect(input);
                        assert_eq!(group.map(|g| &g.name), Some(&holder.name), "{input}");
                        assert_eq!(held.advanced, in_advanced, "{input}");
                        assert_eq!(row[advanced], flag(held.advanced), "{input}");
                        if held.id != holder.name {
                            assert_eq!(section, Some(held.label), "{input}");
                        }
                    }
                }
            }
            let want_advanced = groups
                .iter()
                .filter(|g| g.sections.iter().any(|s| s.advanced))
                .count();
            assert_eq!(advanced_headings, want_advanced, "{order:?}");
            // A group's own section has no heading (`InputSection`'s rule): no section heading
            // repeats its group's, and there is one per other section, plus the Advanced ones.
            let mut current = "";
            let mut section_headings = 0;
            for row in &sheet.rows[1..] {
                match row[0].style {
                    CellStyle::Group => current = row[0].as_text(),
                    CellStyle::Section => {
                        assert_ne!(row[0].as_text(), current, "{order:?}");
                        section_headings += 1;
                    }
                    _ => {}
                }
            }
            let other_sections = groups
                .iter()
                .flat_map(|g| g.sections.iter().filter(|s| s.id != g.name))
                .count();
            assert_eq!(
                section_headings,
                other_sections + want_advanced,
                "{order:?}"
            );
        }
        // The workbook order has no Advanced heading; the workflow order has some.
        let count = |order: InputOrder| {
            let mut shot = snapshot(&inputs, &results);
            shot.input_order = order;
            sheets(&shot)[1]
                .rows
                .iter()
                .filter(|row| row[0].as_text() == ADVANCED_HEADING)
                .count()
        };
        assert_eq!(count(InputOrder::Workbook), 0);
        assert!(count(InputOrder::Workflow) > 0);
    }

    #[test]
    fn an_input_row_flags_changes_assumptions_key_design_and_notes_choices_and_blanks() {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, INPUTS_SHEET);
        let path = column(&INPUT_COLUMNS, "Path");
        let at = |name: &str| column(&INPUT_COLUMNS, name);
        let row = |input: &str| row_of(sheet, path, input);
        let face_gap = row("metal.face_gap_mm");
        assert_eq!(face_gap[1], Cell::number(1.41));
        assert_eq!(face_gap[at("Changed from default")], Cell::text(YES));
        assert_eq!(face_gap[at("Key design")], Cell::text(YES));
        assert_eq!(face_gap[at("Assumption")], Cell::empty());
        assert_eq!(face_gap[at("Note")], Cell::empty());
        // An assumption at its default.
        let variation = row("metal.variation");
        assert_eq!(variation[at("Assumption")], Cell::text(YES));
        assert_eq!(variation[at("Changed from default")], Cell::empty());
        assert_eq!(variation[at("Key design")], Cell::empty());
        // A selector: its code a number, its choice in the note.
        let entry = InputCatalogue::get().entry("coupling.backiron").unwrap();
        let Some(Value::Int(code)) = inputs.get("coupling.backiron") else {
            panic!("a selector code")
        };
        let (_, choice) = entry.meta.choices.iter().find(|(c, _)| *c == code).unwrap();
        let backiron = row("coupling.backiron");
        assert_eq!(backiron[1], Cell::number(code as f64));
        assert_eq!(
            backiron[at("Note")],
            Cell::text(format!("{code} = {choice}"))
        );
        // An optional input left blank: no value, noted.
        let drag = row("metal.measured_drag_Nm");
        assert_eq!(drag[1], Cell::empty());
        assert_eq!(drag[at("Note")], Cell::text(BLANK_TEXT));
        // A text input: the part name as it is, with no note.
        let part = row("coupling.magnets.part_inner");
        assert_eq!(
            part[1],
            Cell::text(inputs.coupling.magnets.part_inner.clone())
        );
        assert_eq!(part[at("Note")], Cell::empty());
        // A Rust-only input names no cell; an advanced input is flagged.
        let catalogue = InputCatalogue::get();
        let rust_only = catalogue.all().find(|e| e.meta.cell.is_none()).unwrap();
        assert_eq!(
            row(&rust_only.path)[at("Workbook cell")],
            Cell::text(RUST_ONLY)
        );
        let advanced = catalogue
            .workflow
            .iter()
            .flat_map(|g| g.sections.iter())
            .find(|s| s.advanced)
            .unwrap();
        assert_eq!(
            row(&advanced.entries[0].path)[at("Advanced")],
            Cell::text(YES)
        );
        // An empty text input (manual magnets: no part name): no value, noted blank.
        let mut manual = inputs.clone();
        manual.coupling.magnets.part_inner.clear();
        let results = compute_all(&manual);
        let book = sheets(&snapshot(&manual, &results));
        let manual_inputs = &book[1];
        assert_eq!(manual_inputs.name, INPUTS_SHEET);
        let part = row_of(manual_inputs, path, "coupling.magnets.part_inner");
        assert_eq!(part[1], Cell::empty());
        assert_eq!(part[at("Note")], Cell::text(BLANK_TEXT));
    }

    #[test]
    fn every_result_is_on_one_row_by_physics_chain_with_its_value() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        let rows = path_rows(sheet, path);
        let entries = table_entries();
        // The results table's grouped lines, every row matched and every group open.
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(&all, ResultOrder::Grouped, None, &|_| true);
        // Exactly the table's rows, each once, in its grouped order.
        let want: Vec<&str> = lines
            .iter()
            .filter_map(|line| match line {
                Line::Row(index) => Some(entries[*index].path.as_str()),
                _ => None,
            })
            .collect();
        let got: Vec<&str> = rows.iter().map(|row| row[path].as_text()).collect();
        assert_eq!(got, want);
        assert_eq!(got.len(), entries.len());
        for row in &rows {
            let result = row[path].as_text();
            let info = result_info(result).unwrap();
            assert_eq!(row[0], Cell::text(info.meta.label), "{result}");
            assert_eq!(
                row[1],
                value_cell(&results.get(result).unwrap()),
                "{result}"
            );
            assert_eq!(row[2], Cell::text(info.meta.unit), "{result}");
            assert_eq!(
                row[4],
                Cell::text(info.cell.as_deref().unwrap_or(RUST_ONLY)),
                "{result}"
            );
        }
        // The headings: the table's, in its order, the package groups under "Other results"
        // (once, a group heading) as headings inside it.
        let headings: Vec<(&str, CellStyle)> = sheet
            .rows
            .iter()
            .filter(|row| matches!(row[0].style, CellStyle::Group | CellStyle::Section))
            .map(|row| (row[0].as_text(), row[0].style))
            .collect();
        let want: Vec<(&str, CellStyle)> = lines
            .iter()
            .filter_map(|line| match line {
                Line::OtherResults => Some((OTHER_RESULTS, CellStyle::Group)),
                Line::Group { group, .. } => {
                    let group = &result_groups()[*group];
                    let style = if group.other {
                        CellStyle::Section
                    } else {
                        CellStyle::Group
                    };
                    Some((group.label, style))
                }
                Line::Row(_) => None,
            })
            .collect();
        assert_eq!(headings, want);
        assert_eq!(
            headings[0].0,
            result_groups()[0].label,
            "the headline first"
        );
        let other = headings.iter().filter(|(text, _)| *text == OTHER_RESULTS);
        assert_eq!(other.count(), 1);
        // The pull-out: its number at full precision and its corrections.
        let pullout = row_of(sheet, path, "model.pullout_Nm");
        assert_eq!(pullout[1], Cell::number(results.model.pullout_Nm));
        assert_eq!(
            pullout[column(&RESULT_COLUMNS, "Corrected vs workbook")],
            Cell::text("E3 E7 E8")
        );
    }

    #[test]
    fn a_check_carries_its_level_and_a_greyed_check_none() {
        let check = column(&RESULT_COLUMNS, "Check");
        let path = column(&RESULT_COLUMNS, "Path");
        // The default design, one with six failing checks (manual magnets) and one out of the
        // end-effect range (short magnets).
        let mut manual = DesignInputs::default();
        manual.coupling.magnets.part_inner.clear();
        manual.coupling.magnets.part_outer.clear();
        for inputs in [DesignInputs::default(), manual.clone(), short_magnets()] {
            let results = compute_all(&inputs);
            let book = sheets(&snapshot(&inputs, &results));
            let sheet = sheet(&book, RESULTS_SHEET);
            for row in path_rows(sheet, path) {
                let result = row[path].as_text();
                let want = if CHECKS.contains(&result) {
                    level_cell(check_level(&results, result))
                } else {
                    Cell::empty()
                };
                assert_eq!(row[check], want, "{result}");
            }
            // Each group heading shows the worst level of its checks.
            for group in result_groups() {
                let heading = sheet
                    .rows
                    .iter()
                    .find(|row| row[0].as_text() == group.label && row.len() > check)
                    .unwrap();
                assert_eq!(
                    heading[check],
                    level_cell(worst_level(&results, &group.rows))
                );
            }
        }
        // Manual magnets: red and amber cells.
        let results = compute_all(&manual);
        let book = sheets(&snapshot(&manual, &results));
        let styles: Vec<CellStyle> = path_rows(sheet(&book, RESULTS_SHEET), path)
            .iter()
            .map(|row| row[check].style)
            .collect();
        assert!(styles.contains(&CellStyle::Level(Level::Bad)));
        assert!(styles.contains(&CellStyle::Level(Level::Caution)));
        // Short magnets: the hot minimum (red by its text) is greyed, so it has no level; the
        // Summary carries the end-effect banner in red and notes the greyed dashboard rows.
        let inputs = short_magnets();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let hot_min = row_of(sheet(&book, RESULTS_SHEET), path, "metal.hot_min_check");
        assert_eq!(hot_min[check], Cell::empty());
        let banner = end_effect_banner(&results).expect("out of range");
        let summary = &sheet(&book, SUMMARY_SHEET).rows;
        assert!(summary.contains(&vec![
            Cell::text(banner).styled(CellStyle::Level(Level::Bad))
        ]));
        let pullout = summary
            .iter()
            .find(|row| row.get(5).map(Cell::as_text) == Some("model.pullout_Nm"))
            .unwrap();
        assert_eq!(pullout[7], Cell::text(GREYED_NOTE));
        assert_eq!(pullout[3], Cell::empty());
    }

    #[test]
    fn a_number_that_is_not_finite_is_the_csv_s_text() {
        assert_eq!(value_cell(&Value::Num(f64::INFINITY)), Cell::text("+inf"));
        assert_eq!(
            value_cell(&Value::Num(f64::NEG_INFINITY)),
            Cell::text("-inf")
        );
        assert_eq!(value_cell(&Value::Num(f64::NAN)), Cell::text("NaN"));
        assert_eq!(value_cell(&Value::Int(10)), Cell::number(10.0));
        assert_eq!(value_cell(&Value::Text(String::new())), Cell::empty());
        assert_eq!(value_cell(&Value::None), Cell::empty());
        // A positive beta typed in for manual magnets with no grade (E20): no hot limit, +inf,
        // and no torque at it, NaN.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.temperature.demag.coercivity_source = 0;
        inputs.temperature.demag.beta_hcj_per_C = 0.0035;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        assert_eq!(
            row_of(sheet, path, "temperature.demag.magnet_limit_C")[1],
            Cell::text("+inf")
        );
        assert_eq!(
            row_of(sheet, path, "temperature.demag.torque_at_limit_Nm")[1],
            Cell::text("NaN")
        );
        // No number cell on any sheet holds a number that is not finite.
        for sheet in &book {
            for cell in sheet.rows.iter().flatten() {
                if let CellValue::Number(x) = cell.value {
                    assert!(x.is_finite(), "{}: {x}", sheet.name);
                }
            }
        }
    }

    #[test]
    fn the_equation_column_holds_the_record_s_plain_formula() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        let path = column(&RESULT_COLUMNS, "Path");
        let equation = column(&RESULT_COLUMNS, "Equation");
        assert_eq!(
            row_of(sheet, path, "model.f_end")[equation],
            Cell::text("f_{end} = 1 − c_{end} · (τ_p/L)")
        );
        // Exactly the results with an equation record have one (the space claim has none,
        // decision G2).
        let rows = path_rows(sheet, path);
        for row in &rows {
            let result = row[path].as_text();
            let explained = registry().equation_for(result).is_some();
            assert_eq!(!row[equation].as_text().is_empty(), explained, "{result}");
        }
        assert_eq!(
            row_of(sheet, path, "housing.space_claim_check")[equation],
            Cell::empty()
        );
        let explained = rows
            .iter()
            .filter(|r| !r[equation].as_text().is_empty())
            .count();
        assert!(explained > 300, "{explained}");
    }

    #[test]
    fn the_summary_holds_the_export_the_link_the_dashboard_the_warnings_and_the_corrections() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let shot = snapshot(&inputs, &results);
        let book = sheets(&shot);
        let row = |label: &str| summary_row(&book, label).unwrap_or_else(|| panic!("{label}"));
        assert_eq!(row(EXPORTED)[1].value, CellValue::Time(1_790_000_000));
        assert_eq!(
            row(APP)[1],
            Cell::text(format!("magcoupling-rs {}", env!("CARGO_PKG_VERSION")))
        );
        assert_eq!(row(SHARE_LINK)[1], Cell::text(shot.share_link.clone()));
        assert_eq!(
            row(SIZING_MODE)[1],
            Cell::text(SizingMode::MagnetsToTorque.label())
        );
        assert!(summary_row(&book, FREE_VARIABLE_LABEL).is_none());
        // No banner at the default design.
        let rows = &sheet(&book, SUMMARY_SHEET).rows;
        assert!(
            !rows
                .iter()
                .flatten()
                .any(|c| matches!(c.style, CellStyle::Level(_))
                    && c.as_text().starts_with("Assumptions modified"))
        );
        // The dashboard: its header, then each of its rows with value, unit, badge and path.
        let header = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(DASHBOARD_COLUMNS[0]))
            .unwrap();
        let lines = dashboard_lines(&results);
        assert_eq!(lines.len(), DASHBOARD.len());
        for (row, line) in rows[header + 1..].iter().zip(&lines) {
            assert_eq!(row[0], Cell::text(line.label));
            assert_eq!(row[1], value_cell(&results.get(line.path).unwrap()));
            assert_eq!(row[3], level_cell(line.level));
            assert_eq!(row[5], Cell::text(line.path));
            assert_eq!(row[6], Cell::text(line.marker.clone()));
        }
        assert_eq!(rows[header + 1 + lines.len()], Vec::<Cell>::new());
        // No warning fires at the default design.
        assert!(summary_row(&book, WARNINGS_HEADING).is_some());
        assert!(summary_row(&book, NO_WARNING).is_some());
        // The corrections applied, one row each, last.
        let corrections = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(CORRECTIONS_HEADING))
            .unwrap();
        let applied: Vec<_> = REGISTRY.iter().filter(|d| marks_values(d)).collect();
        assert!(applied.len() >= 19, "{}", applied.len());
        assert_eq!(
            rows.len(),
            corrections + 1 + applied.len(),
            "the corrections are last"
        );
        for (row, deviation) in rows[corrections + 1..].iter().zip(&applied) {
            assert_eq!(row[0], Cell::text(deviation.id.to_string()));
            assert_eq!(row[1], Cell::text(deviation.title));
        }
    }

    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
        // 304 stainless back iron: a warning (red) and a caution (amber) fire.
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.back_iron = 7;
        inputs.metal.variation = 0.2;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let rows = &sheet(&book, SUMMARY_SHEET).rows;
        let heading = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(WARNINGS_HEADING))
            .unwrap();
        let lines = warning_lines(&results);
        assert_eq!(lines.len(), 2);
        for (row, (rule, text)) in rows[heading + 1..].iter().zip(&lines) {
            assert_eq!(
                row[0].style,
                CellStyle::Level(severity_level(rule.severity))
            );
            assert_eq!(row[1], Cell::text(text.clone()));
        }
        assert_eq!(rows[heading + 1][0].as_text(), "Warning");
        assert!(summary_row(&book, NO_WARNING).is_none());
        // The assumptions banner, amber, as the header shows it.
        let banner = assumptions_banner(&inputs).expect("the variation is an assumption");
        assert!(rows.contains(&vec![
            Cell::text(banner).styled(CellStyle::Level(Level::Caution))
        ]));
    }

    #[test]
    fn torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row() {
        // The design shown: the solved axial length in the override.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.axial_length_mm = Some(14.2);
        let results = compute_all(&inputs);
        let mut shot = snapshot(&inputs, &results);
        shot.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
            target_Nm: 2.5,
        };
        shot.sizing_status = Some("Solved: 14.2 mm".to_owned());
        let book = sheets(&shot);
        let row = |label: &str| summary_row(&book, label).unwrap_or_else(|| panic!("{label}"));
        assert_eq!(
            row(SIZING_MODE)[1],
            Cell::text(SizingMode::TorqueToMagnets.label())
        );
        assert_eq!(
            row(FREE_VARIABLE_LABEL)[1],
            Cell::text(variable_label(FreeVariable::AxialLength))
        );
        assert_eq!(row(TARGET_LABEL)[1], Cell::number(2.5));
        assert_eq!(row(TARGET_LABEL)[2], Cell::text("N·m"));
        assert_eq!(row(SIZING_OUTCOME)[1], Cell::text("Solved: 14.2 mm"));
        // The free variable's row: the value shown, noted; no other row carries the note.
        let sheet = sheet(&book, INPUTS_SHEET);
        let path = column(&INPUT_COLUMNS, "Path");
        let note = column(&INPUT_COLUMNS, "Note");
        let length = row_of(sheet, path, FreeVariable::AxialLength.path());
        assert_eq!(length[1], Cell::number(14.2));
        assert_eq!(length[note], Cell::text(SIZED_NOTE));
        let noted = path_rows(sheet, path)
            .iter()
            .filter(|r| r[note].as_text().contains(SIZED_NOTE))
            .count();
        assert_eq!(noted, 1);
    }

    #[test]
    fn the_assumptions_sheet_lists_each_assumption_s_inputs_with_their_workbook_defaults() {
        let mut inputs = DesignInputs::default();
        inputs.metal.variation = 0.2;
        let results = compute_all(&inputs);
        let book = sheets(&snapshot(&inputs, &results));
        let sheet = sheet(&book, ASSUMPTIONS_SHEET);
        let defaults = DesignInputs::default();
        let rows = &sheet.rows[1..];
        let want: Vec<(&str, &str)> = ASSUMPTIONS
            .iter()
            .flat_map(|a| a.paths.iter().map(move |p| (a.label, *p)))
            .collect();
        assert_eq!(rows.len(), want.len());
        for (row, (label, path)) in rows.iter().zip(want) {
            assert_eq!(row[0], Cell::text(label));
            assert_eq!(row[1], Cell::text(path));
            assert_eq!(row[2], value_cell(&inputs.get(path).unwrap()), "{path}");
            assert_eq!(row[4], value_cell(&defaults.get(path).unwrap()), "{path}");
            let changed = path == "metal.variation";
            assert_eq!(row[5], flag(changed), "{path}");
        }
        assert!(
            rows.iter().all(|row| !row[6].as_text().is_empty()),
            "every rationale"
        );
    }
}
