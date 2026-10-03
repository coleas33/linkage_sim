//! Writes the spreadsheet export ([`crate::gui::spreadsheet`]) as an .xlsx workbook with
//! rust_xlsxwriter (decision X-9): a number as a number cell, a text as a text cell (cut to
//! Excel's 32,767 characters, as Excel counts them, ending with how long it was), a time as a
//! date, each sheet's header rows frozen, its header's filter buttons and its columns' widths as
//! the sheet asks, the headings bold (a group heading on a light grey), and a check's level
//! filled with the standalone page's badge colour, a fixed dark-theme palette (decision X-11;
//! [`fill`]). Values only: no cell holds a formula (decision X-1).

use rust_xlsxwriter::{
    Color, DocProperties, ExcelDateTime, Format, FormatBorder, Workbook, Worksheet, XlsxError,
};

use crate::gui::dashboard::Level;
use crate::gui::panel::HEADING;
use crate::gui::spreadsheet::{Cell, CellStyle, CellValue, Sheet, Snapshot, sheets};

/// The file name the export suggests (decision X-7): the stem of the CSV and JSON exports.
pub const XLSX_FILE_NAME: &str = "magcoupling-results.xlsx";

/// The media type of an .xlsx file (the web download's Blob type).
pub const XLSX_MIME: &str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

/// The most characters an Excel cell holds, as Excel counts them: UTF-16 code units (a character
/// outside the Basic Multilingual Plane, an emoji, counts two).
pub const MAX_CELL_CHARS: usize = 32_767;

/// The fill of a group heading (an input group, a result chain, "Other results", a Summary
/// block): a light grey that sets it apart from the section headings inside it.
pub const GROUP_FILL: Color = Color::RGB(0xE7_E6_E6);

/// The number format of a time cell.
pub const TIME_FORMAT: &str = "yyyy-mm-dd hh:mm:ss";

/// The spreadsheet of `snapshot` as the bytes of an .xlsx file, or why it could not be written.
pub fn spreadsheet_bytes(snapshot: &Snapshot) -> Result<Vec<u8>, String> {
    xlsx_bytes(&sheets(snapshot), snapshot.exported_unix_s).map_err(|error| error.to_string())
}

/// `sheets` as the bytes of an .xlsx file whose document properties say it was created at
/// `created_unix_s` [s since the Unix epoch].
pub fn xlsx_bytes(sheets: &[Sheet], created_unix_s: i64) -> Result<Vec<u8>, XlsxError> {
    let mut workbook = Workbook::new();
    let created = ExcelDateTime::from_timestamp(created_unix_s)?;
    workbook.set_properties(
        &DocProperties::new()
            .set_title(HEADING)
            .set_creation_datetime(&created),
    );
    for sheet in sheets {
        let worksheet = workbook.add_worksheet();
        worksheet.set_name(sheet.name)?;
        for (col, width) in sheet.widths.iter().enumerate() {
            worksheet.set_column_width(column(col)?, *width)?;
        }
        if sheet.frozen_rows > 0 {
            worksheet.set_freeze_panes(sheet.frozen_rows, 0)?;
        }
        if sheet.autofilter && !sheet.rows.is_empty() && !sheet.widths.is_empty() {
            let last_row =
                u32::try_from(sheet.rows.len() - 1).map_err(|_| XlsxError::RowColumnLimitError)?;
            worksheet.autofilter(0, 0, last_row, column(sheet.widths.len() - 1)?)?;
        }
        for (row, cells) in sheet.rows.iter().enumerate() {
            let row = u32::try_from(row).map_err(|_| XlsxError::RowColumnLimitError)?;
            for (col, cell) in cells.iter().enumerate() {
                write_cell(worksheet, row, column(col)?, cell)?;
            }
        }
    }
    workbook.save_to_buffer()
}

/// A column index as rust_xlsxwriter takes it.
fn column(col: usize) -> Result<u16, XlsxError> {
    u16::try_from(col).map_err(|_| XlsxError::RowColumnLimitError)
}

/// Writes `cell` at (`row`, `col`); an empty cell is not written.
fn write_cell(worksheet: &mut Worksheet, row: u32, col: u16, cell: &Cell) -> Result<(), XlsxError> {
    let format = format_of(cell.style);
    match &cell.value {
        CellValue::Empty => {}
        CellValue::Number(x) => {
            worksheet.write_number_with_format(row, col, *x, &format)?;
        }
        CellValue::Text(text) => {
            worksheet.write_string_with_format(row, col, cell_text(text), &format)?;
        }
        CellValue::Time(unix_s) => {
            let time = ExcelDateTime::from_timestamp(*unix_s)?;
            let format = format.set_num_format(TIME_FORMAT);
            worksheet.write_datetime_with_format(row, col, &time, &format)?;
        }
    }
    Ok(())
}

/// The format of a cell style: the title, headers and headings bold (the column headers
/// underlined, a group heading on [`GROUP_FILL`]), a check's level bold on the dark theme's
/// badge colour ([`fill`]).
fn format_of(style: CellStyle) -> Format {
    match style {
        CellStyle::Plain => Format::new(),
        CellStyle::Title => Format::new().set_bold().set_font_size(14),
        CellStyle::Header => Format::new()
            .set_bold()
            .set_border_bottom(FormatBorder::Thin),
        CellStyle::Group => Format::new()
            .set_bold()
            .set_font_size(12)
            .set_background_color(GROUP_FILL),
        CellStyle::Section => Format::new().set_bold(),
        CellStyle::Level(level) => Format::new().set_bold().set_background_color(fill(level)),
    }
}

/// The badge colour of `level` in egui's dark visuals, which the standalone page forces
/// (decision X-11): green, the warning amber, the error red. Black text reads on each. A fixed
/// palette: the linkage app's calculator window may draw in light visuals, whose badges differ
/// (green 14823C, amber FF6400), and the fills do not follow them.
pub fn fill(level: Level) -> Color {
    let color = level.color(&egui::Visuals::dark());
    Color::RGB((u32::from(color.r()) << 16) | (u32::from(color.g()) << 8) | u32::from(color.b()))
}

/// `text` as a cell holds it: as it is while it fits [`MAX_CELL_CHARS`] UTF-16 code units (as
/// Excel counts); a longer text cut after the last whole character that leaves room for a note
/// of how many characters it had.
pub fn cell_text(text: &str) -> String {
    if text.encode_utf16().count() <= MAX_CELL_CHARS {
        return text.to_owned();
    }
    let note = format!(" [cut: {} characters]", text.chars().count());
    let mut room = MAX_CELL_CHARS - note.encode_utf16().count();
    let mut cut = String::new();
    for c in text.chars() {
        if c.len_utf16() > room {
            break;
        }
        room -= c.len_utf16();
        cut.push(c);
    }
    cut.push_str(&note);
    cut
}

/// Now [s since the Unix epoch]: the time an export is stamped with (decision X-12).
pub fn now_unix_s() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        (js_sys::Date::now() / 1000.0) as i64
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use calamine::Data;

    use super::*;
    use crate::gui::spreadsheet::{INPUT_COLUMNS, INPUTS_SHEET};
    use crate::gui::test_support::{read_xlsx, snapshot, xlsx_part};
    use crate::{DesignInputs, compute_all};

    /// The Excel serial date of a Unix time (days since 1899-12-30).
    fn serial(unix_s: i64) -> f64 {
        unix_s as f64 / 86_400.0 + 25_569.0
    }

    /// Asserts the cell calamine read (`read`, `Data::Empty` past the end of a row) is `cell`.
    fn assert_cell(cell: &Cell, read: &Data, at: &str) {
        match (&cell.value, read) {
            (CellValue::Empty, Data::Empty) => {}
            (CellValue::Number(x), Data::Float(y)) => assert_eq!(x.to_bits(), y.to_bits(), "{at}"),
            (CellValue::Text(text), Data::String(read)) => {
                assert_eq!(&cell_text(text), read, "{at}")
            }
            (CellValue::Time(unix_s), Data::DateTime(time)) => {
                assert!((time.as_f64() - serial(*unix_s)).abs() < 1e-6, "{at}");
            }
            (want, got) => panic!("{at}: {want:?} read back as {got:?}"),
        }
    }

    #[test]
    fn the_workbook_reads_back_cell_for_cell() {
        // The default design, one with failing checks (manual magnets) and one with numbers
        // that are not finite (a positive beta without a grade, E20).
        let mut manual = DesignInputs::default();
        manual.coupling.magnets.part_inner.clear();
        manual.coupling.magnets.part_outer.clear();
        let mut non_finite = manual.clone();
        non_finite.temperature.demag.coercivity_source = 0;
        non_finite.temperature.demag.beta_hcj_per_C = 0.0035;
        for inputs in [DesignInputs::default(), manual, non_finite] {
            let results = compute_all(&inputs);
            let shot = snapshot(&inputs, &results);
            let book = sheets(&shot);
            let read = read_xlsx(&spreadsheet_bytes(&shot).expect("written"));
            let names: Vec<&str> = read.iter().map(|(name, _)| name.as_str()).collect();
            let want: Vec<&str> = book.iter().map(|sheet| sheet.name).collect();
            assert_eq!(names, want);
            for (sheet, (_, rows)) in book.iter().zip(&read) {
                assert_eq!(rows.len(), sheet.rows.len(), "{}", sheet.name);
                for (r, (cells, read_row)) in sheet.rows.iter().zip(rows).enumerate() {
                    for (c, read_cell) in read_row.iter().enumerate() {
                        let at = format!("{} row {} column {}", sheet.name, r + 1, c + 1);
                        assert_cell(cells.get(c).unwrap_or(&Cell::empty()), read_cell, &at);
                    }
                    assert!(cells.len() <= read_row.len().max(1), "{}", sheet.name);
                }
            }
        }
    }

    #[test]
    fn the_file_has_frozen_headers_filters_widths_bold_headings_fills_and_no_formula() {
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        let results = compute_all(&inputs);
        let shot = snapshot(&inputs, &results);
        let bytes = spreadsheet_bytes(&shot).unwrap();
        let styles = xlsx_part(&bytes, "xl/styles.xml");
        assert!(styles.contains("<b/>"), "bold headings");
        // The three badge colours as fills (ARGB): green, amber, red; and the group headings'
        // grey.
        for rgb in ["FF5AC878", "FFFF8F00", "FFFF0000", "FFE7E6E6"] {
            assert!(
                styles.contains(&format!("rgb=\"{rgb}\"")),
                "{rgb} in {styles}"
            );
        }
        for (index, sheet) in sheets(&shot).iter().enumerate() {
            let xml = xlsx_part(&bytes, &format!("xl/worksheets/sheet{}.xml", index + 1));
            // The header row frozen where the sheet asks (not the Summary).
            assert_eq!(
                xml.contains("ySplit=\"1\"") && xml.contains("state=\"frozen\""),
                sheet.frozen_rows == 1,
                "{}: frozen",
                sheet.name
            );
            assert_eq!(
                xml.contains("<pane"),
                sheet.frozen_rows > 0,
                "{}",
                sheet.name
            );
            // The header's filter buttons over every column and row, where the sheet asks.
            let last_column = char::from(b'A' + u8::try_from(sheet.widths.len() - 1).unwrap());
            let filter = format!("<autoFilter ref=\"A1:{last_column}{}\"", sheet.rows.len());
            assert_eq!(xml.contains(&filter), sheet.autofilter, "{}", sheet.name);
            assert_eq!(
                xml.contains("<autoFilter"),
                sheet.autofilter,
                "{}",
                sheet.name
            );
            // Every column's width, set (rust_xlsxwriter writes a run of equal widths as one
            // <col min max>).
            let set: usize = xml
                .split("<col ")
                .skip(1)
                .map(|col| {
                    let attribute = |name: &str| -> usize {
                        let start = col.find(&format!("{name}=\"")).expect(name) + name.len() + 2;
                        col[start..].split('"').next().unwrap().parse().unwrap()
                    };
                    attribute("max") - attribute("min") + 1
                })
                .sum();
            assert_eq!(set, sheet.widths.len(), "{}", sheet.name);
            assert!(
                !xml.contains("<f>") && !xml.contains("<f "),
                "{}: no formula",
                sheet.name
            );
        }
    }

    #[test]
    fn a_text_longer_than_a_cell_holds_is_cut_to_excel_s_limit_saying_how_long_it_was() {
        let fits = "a".repeat(MAX_CELL_CHARS);
        assert_eq!(cell_text(&fits), fits);
        let long = "é".repeat(40_000);
        let cut = cell_text(&long);
        assert_eq!(cut.chars().count(), MAX_CELL_CHARS);
        assert!(
            cut.ends_with(" [cut: 40000 characters]"),
            "{}",
            &cut[cut.len() - 40..]
        );
        assert!(cut.starts_with("éé"));
        // Excel counts UTF-16 code units: 20,000 emoji are 40,000 of them, over the limit though
        // only 20,000 characters. The cut keeps whole characters, so it may stop one unit short.
        let emoji = "\u{1F600}".repeat(20_000);
        let cut_emoji = cell_text(&emoji);
        let units = cut_emoji.encode_utf16().count();
        assert!(
            (MAX_CELL_CHARS - 1..=MAX_CELL_CHARS).contains(&units),
            "{units}"
        );
        let kept = cut_emoji
            .strip_suffix(" [cut: 20000 characters]")
            .expect("the note");
        assert!(kept.chars().all(|c| c == '\u{1F600}'), "whole characters");
        // At the limit a text is as it is; one unit over, it is cut.
        let at_limit = format!("{}\u{1F600}", "a".repeat(MAX_CELL_CHARS - 2));
        assert_eq!(cell_text(&at_limit), at_limit);
        let over = format!("{}\u{1F600}", "a".repeat(MAX_CELL_CHARS - 1));
        let cut_over = cell_text(&over);
        assert!(cut_over.ends_with(" [cut: 32767 characters]"), "{cut_over}");
        assert!(cut_over.encode_utf16().count() <= MAX_CELL_CHARS);
        // A part name that long and the share link (about 2,500 characters, over Excel's 2,080
        // for a hyperlink: decision X-6) both write.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner = long.clone();
        inputs.coupling.magnets.part_outer = emoji.clone();
        let results = compute_all(&inputs);
        let mut shot = snapshot(&inputs, &results);
        shot.share_link = format!("https://example.test/magcoupling/?m={}", "A".repeat(2_500));
        let read = read_xlsx(&spreadsheet_bytes(&shot).expect("written"));
        let (_, summary) = &read[0];
        assert!(
            summary
                .iter()
                .any(|row| row.get(1) == Some(&Data::String(shot.share_link.clone())))
        );
        let (_, rows) = read.iter().find(|(name, _)| name == INPUTS_SHEET).unwrap();
        let path = INPUT_COLUMNS.iter().position(|c| *c == "Path").unwrap();
        let part = |input: &str| {
            rows.iter()
                .find(|row| row.get(path) == Some(&Data::String(input.into())))
                .unwrap()
        };
        assert_eq!(part("coupling.magnets.part_inner")[1], Data::String(cut));
        assert_eq!(
            part("coupling.magnets.part_outer")[1],
            Data::String(cut_emoji)
        );
    }

    #[test]
    fn the_level_fills_are_the_page_s_badge_colours() {
        assert_eq!(fill(Level::Good), Color::RGB(0x5A_C8_78));
        assert_eq!(fill(Level::Caution), Color::RGB(0xFF_8F_00));
        assert_eq!(fill(Level::Bad), Color::RGB(0xFF_00_00));
    }

    #[test]
    fn the_export_is_stamped_with_the_time_now() {
        // 2023-11-14: any clock this code runs on is past it.
        assert!(now_unix_s() > 1_700_000_000);
        assert!(
            xlsx_bytes(&[], 1_790_000_000).is_ok(),
            "an empty workbook still writes"
        );
    }
}
