//! Live values in teaching-note text (guidance slice 2, spec 3.2): a `{result.path}` in a
//! note's sentences, change texts and watch-outs stands for that result's value on the design
//! shown. The GUI formats the value with its unit as the readouts do; a result with no
//! prediction there reads [`NO_VALUE`]. A placeholder never stands in a title, a sentence 0 or a
//! source, and names a number result (the note tests check both). Pure std: the renderer takes
//! the formatting as a closure.

use super::notes::Note;

/// What a placeholder reads where its result has no prediction on the design shown.
pub const NO_VALUE: &str = "no prediction";

/// A piece of note text: literal text, or the path of the result whose value stands there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Value(&'a str),
}

/// `text` split into literal text and placeholders, in order (no empty literal piece). The
/// error names the first fault: a `}` with no `{` before it, a `{` with no `}` after it or with
/// another `{` inside, an empty `{}`, or a placeholder that is not a path (ASCII letters,
/// digits, `_`, `.`, `[` and `]` only, so a formula such as `{= a / b}` is refused).
pub fn segments(text: &str) -> Result<Vec<Segment<'_>>, String> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(['{', '}']) {
        if rest[open..].starts_with('}') {
            return Err(format!("a '}}' with no '{{' before it in {text:?}"));
        }
        if open > 0 {
            pieces.push(Segment::Text(&rest[..open]));
        }
        let after = &rest[open + 1..];
        let close = match after.find(['{', '}']) {
            Some(i) if after[i..].starts_with('}') => i,
            Some(_) => return Err(format!("a '{{' inside a placeholder in {text:?}")),
            None => return Err(format!("a '{{' with no '}}' after it in {text:?}")),
        };
        let path = &after[..close];
        if path.is_empty() {
            return Err(format!("an empty placeholder in {text:?}"));
        }
        let path_byte = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'[' | b']');
        if !path.bytes().all(path_byte) {
            return Err(format!("{{{path}}} is not a result path, in {text:?}"));
        }
        pieces.push(Segment::Value(path));
        rest = &after[close + 1..];
    }
    if !rest.is_empty() {
        pieces.push(Segment::Text(rest));
    }
    Ok(pieces)
}

/// `text` with each placeholder replaced by `value(path)`, or by [`NO_VALUE`] where that is
/// `None`. A text [`segments`] refuses comes back as it is (the note tests refuse it first).
pub fn render(text: &str, value: impl Fn(&str) -> Option<String>) -> String {
    match segments(text) {
        Ok(pieces) => pieces
            .into_iter()
            .map(|piece| match piece {
                Segment::Text(literal) => literal.to_owned(),
                Segment::Value(path) => value(path).unwrap_or_else(|| NO_VALUE.to_owned()),
            })
            .collect(),
        Err(_) => text.to_owned(),
    }
}

/// The texts of `note` drawn with live values: its sentences, its change entries' texts, then
/// its watch-outs, in that order.
pub fn rendered_texts(note: &Note) -> Vec<&'static str> {
    let (sentences, change, watch_out) = (note.sentences, note.change, note.watch_out);
    sentences
        .iter()
        .copied()
        .chain(change.iter().map(|c| c.text))
        .chain(watch_out.iter().copied())
        .collect()
}

/// Every placeholder's path in `note`'s [`rendered_texts`], in order, as often as it appears
/// (a text [`segments`] refuses adds none; the note tests refuse it).
pub fn placeholders(note: &Note) -> Vec<&'static str> {
    rendered_texts(note)
        .into_iter()
        .flat_map(|text| segments(text).unwrap_or_default())
        .filter_map(|piece| match piece {
            Segment::Value(path) => Some(path),
            Segment::Text(_) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::notes::{Change, Review, Sense, Step};

    #[test]
    fn text_splits_into_literals_and_placeholders() {
        use Segment::{Text, Value};
        assert_eq!(segments("Plain text."), Ok(vec![Text("Plain text.")]));
        assert_eq!(segments(""), Ok(vec![]));
        assert_eq!(
            segments("Here: {model.pullout_Nm}."),
            Ok(vec![Text("Here: "), Value("model.pullout_Nm"), Text(".")])
        );
        assert_eq!(
            segments("{model.f_end}{model.f_cal}"),
            Ok(vec![Value("model.f_end"), Value("model.f_cal")])
        );
        assert_eq!(
            segments("row {clamps.table[2].preload_N} end"),
            Ok(vec![
                Text("row "),
                Value("clamps.table[2].preload_N"),
                Text(" end")
            ])
        );
    }

    #[test]
    fn a_brace_out_of_place_is_refused_with_its_fault() {
        for (text, fault) in [
            ("a } alone", "with no '{' before it"),
            ("a { alone", "with no '}' after it"),
            ("{}", "an empty placeholder"),
            ("{a{b}}", "a '{' inside a placeholder"),
            ("{= model.a / model.b}", "is not a result path"),
            ("{ model.f_end }", "is not a result path"),
        ] {
            let error = segments(text).expect_err(text);
            assert!(error.contains(fault), "{text}: {error}");
        }
    }

    #[test]
    fn render_puts_each_value_in_place_and_no_value_where_there_is_none() {
        let value = |path: &str| (path == "model.f_end").then(|| "0.896".to_owned());
        assert_eq!(
            render("f is {model.f_end} here.", value),
            "f is 0.896 here."
        );
        assert_eq!(
            render("{model.f_end}, {model.nothing}", value),
            format!("0.896, {NO_VALUE}")
        );
        assert_eq!(render("No braces.", value), "No braces.");
        assert_eq!(
            render("Broken {model.f_end", value),
            "Broken {model.f_end",
            "a refused text comes back as it is"
        );
    }

    #[test]
    fn placeholders_are_listed_in_drawing_order() {
        const NOTE: Note = Note {
            id: "test.placeholders",
            title: "A title",
            equations: &[],
            sentences: &[
                "A plain first sentence.",
                "Here {model.pullout_Nm}.",
                "Then {model.f_end} and {model.f_cal}.",
            ],
            change: &[Change {
                input: "metal.face_gap_mm",
                step: Step::Lower,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "on this design {model.face_gap_mm}.",
            }],
            watch_out: &["A watch-out with {model.pullout_Nm} again."],
            diagram: None,
            sources: &["A source."],
            review: Review::Draft,
        };
        assert_eq!(
            placeholders(&NOTE),
            [
                "model.pullout_Nm",
                "model.f_end",
                "model.f_cal",
                "model.face_gap_mm",
                "model.pullout_Nm"
            ]
        );
        assert_eq!(rendered_texts(&NOTE).len(), 5);
    }
}
