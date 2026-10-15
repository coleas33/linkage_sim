//! The plain-words guard (guidance slice 2, spec 3.4, ruling R2): what text on screen must not
//! say. The calculator is for engineers who are not reading the original spreadsheet, so its
//! texts name no spreadsheet cell or sheet, no correction, audit, plan or decision id and no code
//! or file name. Pure std: the engine's own tests (gate 4) hold the teaching notes to it, and
//! `gui::guidance` re-exports both items for the GUI's texts.
//!
//! The patterns are targeted, never a blanket `M\d`: the clamp's screw sizes (M2.5 to M6), a
//! size followed by its pitch or length (M41 x 0.5) and the magnet part and grade names (M5045,
//! N42SH, B842SH, Y30) are plain words.

use crate::engine::clamps::SCREW_SIZES;

/// What [`plain_text_violations`] looks for, in its order. Each entry names its pattern:
///
/// ```text
/// C\d{2,3}             a cell: C and two or more digits
/// E\d{1,2}             a correction id: E, one or two digits, then no letter, digit or _
/// workbook             the word, in any case
/// Addendum             the spec's addenda
/// Rust-only            the engine's provenance
/// Calculator!          a Calculator sheet cell
/// M1 audit             the math audit
/// skeptic              an audit role: the word or "skeptics", in any case
/// Plan A-              a plan id, "plan" in any case
/// decision \d          a numbered decision: "decision" or "decisions" in any case, a space
///                      and a digit
/// drawing.py           the original drawing script
/// \.rs\b               a Rust source file: ".rs", then no letter, digit or _
/// Metal design         three sheet names, each only where it points into the spreadsheet:
/// Temperature design   followed by "!", by " C" or " (C" and a digit, or by " sheet"; on
/// Shaft clamps         its own a sheet name is an input group's heading and plain
/// Calibration!         a Calibration sheet cell
/// correction E\d       a correction by id, "correction" in any case
/// \bM\d{1,3}\b         an audit or milestone id (M1, M9, M43-13): M and one to three digits
///                      as a whole word, unless a decimal size (M2.5), a screw size of the
///                      clamp's table (SCREW_SIZES) or followed by "x" or "×" and a digit
///                      (M41 x 0.5); a part name has four digits (M5045) and never matches
/// ```
pub const BANNED: [&str; 18] = [
    "C\\d{2,3}",
    "E\\d{1,2}",
    "workbook",
    "Addendum",
    "Rust-only",
    "Calculator!",
    "M1 audit",
    "skeptic",
    "Plan A-",
    "decision \\d",
    "drawing.py",
    "\\.rs\\b",
    "Metal design",
    "Temperature design",
    "Shaft clamps",
    "Calibration!",
    "correction E\\d",
    "\\bM\\d{1,3}\\b",
];

/// The [`BANNED`] patterns `text` contains, in [`BANNED`]'s order, each once. Every text the
/// calculator draws for its users (guides, notes, help, hover texts) is held to an empty list.
pub fn plain_text_violations(text: &str) -> Vec<&'static str> {
    let chars: Vec<char> = text.chars().collect();
    let lower = text.to_lowercase();
    let hits = [
        cell(&chars),
        correction_id(&chars),
        lower.contains("workbook"),
        text.contains("Addendum"),
        text.contains("Rust-only"),
        text.contains("Calculator!"),
        text.contains("M1 audit"),
        words(&lower).any(|word| word == "skeptic" || word == "skeptics"),
        lower.contains("plan a-"),
        followed_by_digit(&lower, "decision ") || followed_by_digit(&lower, "decisions "),
        text.contains("drawing.py"),
        rust_file(text),
        sheet_reference(text, "Metal design"),
        sheet_reference(text, "Temperature design"),
        sheet_reference(text, "Shaft clamps"),
        text.contains("Calibration!"),
        followed_by_digit(&lower, "correction e"),
        audit_id(&chars),
    ];
    BANNED
        .iter()
        .zip(hits)
        .filter_map(|(&pattern, hit)| hit.then_some(pattern))
        .collect()
}

/// The words of `text` as the caps count them: the runs between whitespace.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// The sentences of `text`: each '.', '?' or '!' that ends the text or is followed by
/// whitespace (so neither "0.1 mm" nor "1.35e6" ends one).
pub fn sentence_count(text: &str) -> usize {
    let chars: Vec<char> = text.chars().collect();
    chars
        .iter()
        .enumerate()
        .filter(|&(i, c)| ".!?".contains(*c) && chars.get(i + 1).is_none_or(|n| n.is_whitespace()))
        .count()
}

/// The audit-id prefixes `text` contains (`TC-`, `PA-`, `CG-`, case and dash as written: the
/// guidance audits' finding ids such as TC-12 or PA-8), in that order, each once. Not in
/// [`BANNED`]: the notes' texts and sources are held to an empty list (slice-2 spec 3.4, A-13;
/// the audit cross-references stay in the committed review report), by the notes' plain-words
/// rule and by the all-notes test in `tests/explain.rs`, which both call this.
pub fn audit_ids(text: &str) -> Vec<&'static str> {
    ["TC-", "PA-", "CG-"]
        .into_iter()
        .filter(|id| text.contains(id))
        .collect()
}

/// Whether `c` continues a word: a letter, a digit or an underscore.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The number of ASCII digits in `chars` from index `start` on.
fn digits_from(chars: &[char], start: usize) -> usize {
    chars.get(start..).map_or(0, |rest| {
        rest.iter().take_while(|c| c.is_ascii_digit()).count()
    })
}

/// `C\d{2,3}`: a C followed by two or more digits.
fn cell(chars: &[char]) -> bool {
    (0..chars.len()).any(|i| chars[i] == 'C' && digits_from(chars, i + 1) >= 2)
}

/// `E\d{1,2}\b`: an E, one or two digits, then no word character.
fn correction_id(chars: &[char]) -> bool {
    (0..chars.len()).any(|i| {
        let digits = digits_from(chars, i + 1);
        chars[i] == 'E'
            && (1..=2).contains(&digits)
            && chars.get(i + 1 + digits).is_none_or(|&next| !is_word(next))
    })
}

/// The words of `text`: its runs of letters and digits.
fn words(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
}

/// Whether `needle` occurs in `text` with an ASCII digit right after it.
fn followed_by_digit(text: &str, needle: &str) -> bool {
    text.match_indices(needle)
        .any(|(at, _)| text[at + needle.len()..].starts_with(|c: char| c.is_ascii_digit()))
}

/// `\.rs\b`: ".rs" with no word character after it, a Rust source file's name.
fn rust_file(text: &str) -> bool {
    text.match_indices(".rs").any(|(at, _)| {
        text[at + 3..]
            .chars()
            .next()
            .is_none_or(|next| !is_word(next))
    })
}

/// Whether the sheet name `sheet` occurs in `text` pointing into the spreadsheet: followed by
/// "!" (a cell), by " C" or " (C" and a digit (a cell after the sheet's name), or by " sheet".
fn sheet_reference(text: &str, sheet: &str) -> bool {
    text.match_indices(sheet).any(|(at, _)| {
        let rest = &text[at + sheet.len()..];
        let cell_after = |prefix: &str| {
            rest.strip_prefix(prefix)
                .is_some_and(|after| after.starts_with(|c: char| c.is_ascii_digit()))
        };
        rest.starts_with('!') || cell_after(" C") || cell_after(" (C") || rest.starts_with(" sheet")
    })
}

/// `\bM\d{1,3}\b`: an audit or milestone id (M1, M9, M43-13), M and one to three digits as a
/// whole word. A decimal size (M2.5), a screw size of the clamp's table ([`SCREW_SIZES`]) and a
/// size followed by "x" or "×" and a digit (M41 x 0.5, M4 x 14) are plain; a part name has four
/// digits (M5045), so it never matches.
fn audit_id(chars: &[char]) -> bool {
    (0..chars.len()).any(|i| {
        let digits = digits_from(chars, i + 1);
        let word_start = i == 0 || !is_word(chars[i - 1]);
        if chars[i] != 'M' || !word_start || !(1..=3).contains(&digits) {
            return false;
        }
        let end = i + 1 + digits;
        let decimal = chars.get(end) == Some(&'.') && digits_from(chars, end + 1) > 0;
        let whole_word = chars.get(end).is_none_or(|&next| !is_word(next));
        if decimal || !whole_word {
            return false;
        }
        let token: String = chars[i..end].iter().collect();
        let screw = SCREW_SIZES.iter().any(|size| size.name == token);
        let mut after = chars[end..].iter().copied().filter(|&c| c != ' ');
        let designation = matches!(after.next(), Some('x' | '×'))
            && after.next().is_some_and(|c| c.is_ascii_digit());
        !screw && !designation
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::clamps::SCREW_SIZES;
    use crate::engine::grades::GRADES;
    use crate::engine::library::MAGNET_LIBRARY;
    use std::collections::BTreeSet;

    /// Texts the guard flags, each with every pattern it holds, in [`BANNED`]'s order.
    const FLAGGED: &[(&str, &[&str])] = &[
        // The slice-1 six.
        (
            "Too thin: raise Metal design C122 to at least 2.0 mm",
            &["C\\d{2,3}", "Metal design"],
        ),
        ("E3 E7 E8", &["E\\d{1,2}"]),
        ("corrections E20.", &["E\\d{1,2}"]),
        ("Calculator!C93", &["C\\d{2,3}", "Calculator!"]),
        (
            "Rust-only result (no workbook cell)",
            &["workbook", "Rust-only"],
        ),
        ("Spec Addendum A5", &["Addendum"]),
        ("WORKBOOK", &["workbook"]),
        // No phrase is excused: the status line says "Example design" (S2-5).
        (
            "Workbook example design: fails 3 of 8 checks",
            &["workbook"],
        ),
        ("Workbook example design from the workbook", &["workbook"]),
        ("workbook example", &["workbook"]),
        // Slice 2 (spec 3.4): each text reports its patterns once, in BANNED's order.
        (
            "Workbook Calculator!C41 and Calibration!C23 (0.15); M1 audit M8 (error band against 3D) and M9 (negative for short magnets).",
            &[
                "C\\d{2,3}",
                "workbook",
                "Calculator!",
                "M1 audit",
                "Calibration!",
                "\\bM\\d{1,3}\\b",
            ],
        ),
        ("M1 audit M9 (T3-RC1)", &["M1 audit", "\\bM\\d{1,3}\\b"]),
        (
            "one M1 skeptic found it 11 % high",
            &["skeptic", "\\bM\\d{1,3}\\b"],
        ),
        ("Two skeptics disagreed.", &["skeptic"]),
        ("Plan A-3 (a term of the equation explorer)", &["Plan A-"]),
        ("plan A-3 Task 16 physics review", &["Plan A-"]),
        (
            "Addendum A decision 29 (odd harmonics up to 11)",
            &["Addendum", "decision \\d"],
        ),
        ("Decisions 18 and 19 hold.", &["decision \\d"]),
        (
            "the one-piece slotted clamp (drawing.py's only layout)",
            &["drawing.py"],
        ),
        ("tests/explain.rs CANCELLATIONS", &["\\.rs\\b"]),
        ("see notes.rs.", &["\\.rs\\b"]),
        ("Metal design!C119", &["C\\d{2,3}", "Metal design"]),
        ("the Temperature design sheet", &["Temperature design"]),
        (
            "22.00 mm in Metal design (C127) and 25.00 mm in Shaft clamps (C35)",
            &["C\\d{2,3}", "Metal design", "Shaft clamps"],
        ),
        ("Shaft clamps C66", &["C\\d{2,3}", "Shaft clamps"]),
        ("Calibration!C9", &["Calibration!"]),
        ("(correction E7)", &["E\\d{1,2}", "correction E\\d"]),
        (
            "Correction E17 rewords it",
            &["E\\d{1,2}", "correction E\\d"],
        ),
        ("decision M43-13", &["\\bM\\d{1,3}\\b"]),
        ("M2 decision D7", &["\\bM\\d{1,3}\\b"]),
        ("(M41-5)", &["\\bM\\d{1,3}\\b"]),
        ("M13.", &["\\bM\\d{1,3}\\b"]),
    ];

    /// Texts the guard leaves alone: plain words the patterns must not catch.
    const CLEAN: &[&str] = &[
        "Example design: fails 3 of 8 checks",
        "2.000e7 mm",
        "E123",
        "C1",
        "50.00 °C",
        "ISO 4762 M4 x 14, class 12.9",
        "Raise Minimum outer return ring wall",
        "",
        // A sheet name on its own is an input group's heading (the C-3 fix text).
        "Raise Boss outside diameter (Shaft clamps)",
        "Metal design",
        "Temperature design",
        "Lower Vacuum permeability (Calibration)",
        "Calibration and model",
        // Screw and thread sizes, part and grade names.
        "M41 × 0.5 concept.",
        "M41 x 0.5",
        "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3.",
        "Pilot + 3 × M3 + dowel.",
        "Through the M6 vent port",
        "No screw size from M2.5 to M6 fits the boss",
        "M5045's grade",
        "N42SH, B842SH and Y30",
        "120 MPa",
        // Words that only contain a pattern's letters.
        "a skeptical reader",
        "the decisions are yours",
        "a planar model",
        "rsync and .rst files",
    ];

    #[test]
    fn plain_text_violations_names_each_banned_pattern() {
        for &(text, want) in FLAGGED {
            assert_eq!(plain_text_violations(text), want, "{text:?}");
        }
        for clean in CLEAN {
            assert_eq!(
                plain_text_violations(clean),
                Vec::<&str>::new(),
                "{clean:?}"
            );
        }
    }

    #[test]
    fn every_pattern_has_an_example_and_the_slice_1_six_come_first() {
        let reported: BTreeSet<&str> = FLAGGED
            .iter()
            .flat_map(|&(text, _)| plain_text_violations(text))
            .collect();
        let banned: BTreeSet<&str> = BANNED.into_iter().collect();
        assert_eq!(reported, banned);
        assert_eq!(banned.len(), BANNED.len(), "each pattern once");
        assert_eq!(
            BANNED[..6],
            [
                "C\\d{2,3}",
                "E\\d{1,2}",
                "workbook",
                "Addendum",
                "Rust-only",
                "Calculator!"
            ]
        );
    }

    #[test]
    fn screw_sizes_part_names_and_grade_names_are_plain_words() {
        // Read from the engine's own tables, so a part or size added later is checked too.
        let mut names: Vec<&'static str> = SCREW_SIZES.iter().map(|size| size.name).collect();
        names.extend(
            MAGNET_LIBRARY
                .iter()
                .flat_map(|magnet| [magnet.part, magnet.grade]),
        );
        names.extend(GRADES.iter().flat_map(|grade| [grade.id, grade.name]));
        for name in names {
            assert_eq!(plain_text_violations(name), Vec::<&str>::new(), "{name:?}");
        }
    }

    #[test]
    fn the_counters_count_words_between_whitespace_and_stops_that_end_a_sentence() {
        // Words: the runs between whitespace (the caps' rule), whatever they hold.
        assert_eq!(word_count("Blank: the manual Br and no rating."), 7);
        assert_eq!(word_count("  two   words\n"), 2);
        assert_eq!(word_count(""), 0);
        // Sentences: a '.', '?' or '!' that ends the text or is followed by whitespace.
        assert_eq!(sentence_count("One. Two."), 2);
        assert_eq!(sentence_count("One. Two at 0.1 mm and 1.35e6 S/m. "), 2);
        assert_eq!(sentence_count("A gap: more torque; less clearance."), 1);
        assert_eq!(sentence_count("A 0.5 mm gap."), 1);
        assert_eq!(sentence_count("Why? Because! Yes."), 3);
        assert_eq!(sentence_count("No stop"), 0);
        assert_eq!(sentence_count(""), 0);
    }

    #[test]
    fn audit_ids_names_each_prefix_once_in_its_order() {
        assert_eq!(
            audit_ids("Guidance audit 2026-10-13, TC-1, CG-3 and CG-4: probes"),
            ["TC-", "CG-"]
        );
        assert_eq!(audit_ids("PA-8 and TC-12"), ["TC-", "PA-"]);
        assert_eq!(audit_ids("A source (TC-RC2)"), ["TC-"]);
        // Not ids: a date, the letters without the dash, or in lower case.
        for clean in [
            "Math audit of 2026-09-29: the planar harmonic model",
            "TC 1, a PA66 nylon and a cg-shift",
        ] {
            assert_eq!(audit_ids(clean), Vec::<&str>::new(), "{clean:?}");
        }
        // The plain-words guard does not know them: a note's texts are held to both lists.
        assert_eq!(
            plain_text_violations("TC-1, PA-8 and CG-3"),
            Vec::<&str>::new()
        );
    }
}
