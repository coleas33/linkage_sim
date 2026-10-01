//! Design files and share links (spec M4 "Session": "save/load design JSON; share link (as
//! the linkage tool's `?m=`)").
//!
//! One format for both. A design file is JSON:
//!
//! ```json
//! {
//!   "format": "magcoupling-design",
//!   "version": 1,
//!   "inputs": { "coupling.npole": 10, "metal.measured_drag_Nm": null, ... },
//!   "sizing": { "mode": "magnets_to_torque", "free_variable": "axial_length", "target_torque_Nm": 2.5 }
//! }
//! ```
//!
//! `inputs` maps every input's dotted path (the Python `input_schema()` paths) to its value: a
//! number, an integer (counts and selector codes), a string, or `null` for an optional input
//! left blank. A file names every input, so it keeps its design if a later version changes a
//! default (decision M41-5); a reader takes a missing path at its default, and a path a later
//! version renamed or removed through [`PATH_MIGRATIONS`] (the file names the version that
//! wrote it). `sizing` is the sizing state (decision M41-4); a file without it is in the
//! forward mode.
//!
//! A share link carries the same JSON, compact, deflated and URL-safe base64 encoded: the
//! linkage tool's `?m=` scheme (`linkage-sim-rs/src/gui/state/file_io.rs`).
//!
//! Loading is all or nothing (decision M41-6): a file with an unknown path, a value of the
//! wrong type, a value [`InputSet::set`] refuses (a selector code outside its choices), a
//! newer version or a malformed sizing state changes nothing. The inputs and the sizing state
//! are both checked, so the refusal names every problem of either.

use std::fmt;
use std::io::{Read, Write};

use base64::Engine;
use base64::alphabet::URL_SAFE;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use flate2::Compression;
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use serde_json::{Map, Value as Json};

use crate::DesignInputs;
use crate::engine::meta::{FieldType, InputSet, Value, input_rows};
use crate::gui::format::non_finite_text;
use crate::gui::sizing::{SizingMode, SizingState, variable_from_key, variable_key};

/// The `format` of a design file.
pub const DESIGN_FORMAT: &str = "magcoupling-design";

/// The version this build writes and the newest it reads. Bump it when a path is renamed or
/// removed (with its [`PATH_MIGRATIONS`] entry) or a value's meaning changes, never for an
/// added input (a reader takes a missing path at its default).
pub const DESIGN_VERSION: u64 = 1;

/// An input path a later version renamed or removed: the last version that wrote the old
/// path, the old path, and the new path (`None`: the input was removed and its value is
/// dropped).
pub type PathMigration = (u64, &'static str, Option<&'static str>);

/// Every rename and removal of an input path since version 1, oldest first. A rename or a
/// removal adds its entry here and bumps [`DESIGN_VERSION`] in the same change: a reader
/// refuses an unknown path (decision M41-6), so without the entry every older file and share
/// link would stop opening.
pub const PATH_MIGRATIONS: &[PathMigration] = &[];

/// The public page share links point at when the host does not know its own address (the
/// native app). The web app uses its own address instead.
pub const PUBLIC_BASE_URL: &str = "https://linkage.colesorkness.com/magcoupling/";

/// The query parameter of a share link: the linkage tool's `?m=`.
pub const SHARE_PARAM: &str = "m";

/// The most bytes a share link may inflate to: a design is about 6 kB, so anything larger is
/// not one (and a deflate bomb stops here).
pub const MAX_DESIGN_BYTES: u64 = 1 << 20;

/// A design: what a file or a share link holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Design {
    pub inputs: DesignInputs,
    pub sizing: SizingState,
}

/// Why a design file or a share link was refused. Nothing was changed.
#[derive(Clone, Debug, PartialEq)]
pub enum LoadError {
    /// Not JSON at all (the parser's message).
    NotJson(String),
    /// JSON, but not a design file: no `"format": "magcoupling-design"`, or the top level is
    /// not an object, or it has keys a design file does not.
    NotADesign(String),
    /// Written by a newer version (or a version that is not a whole number).
    UnsupportedVersion(String),
    /// Values a design cannot hold: one message per problem, each naming its path or key in
    /// path order, the inputs' and the sizing state's (either list may be empty, not both).
    Refused {
        inputs: Vec<String>,
        sizing: Vec<String>,
    },
    /// A share link that does not decode (base64 or deflate).
    Link(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::NotJson(message) => write!(f, "not a JSON file: {message}"),
            LoadError::NotADesign(message) => write!(f, "not a design file: {message}"),
            LoadError::UnsupportedVersion(version) => write!(
                f,
                "design file version {version}: this calculator reads version {DESIGN_VERSION} and older"
            ),
            LoadError::Refused { inputs, sizing } => {
                let mut parts = Vec::new();
                if !inputs.is_empty() {
                    parts.push(format!("inputs refused: {}", inputs.join("; ")));
                }
                if !sizing.is_empty() {
                    parts.push(format!("sizing state refused: {}", sizing.join("; ")));
                }
                write!(f, "{}", parts.join("; "))
            }
            LoadError::Link(message) => write!(f, "share link refused: {message}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A number as JSON. JSON has no infinity or NaN (serde_json would write `null`, which reads
/// back as "not entered"), so a non-finite number is the string `"+inf"`, `"-inf"` or `"NaN"`,
/// the panel's display text (decision M41-15). The results export uses it too.
pub(crate) fn json_number(x: f64) -> Json {
    match non_finite_text(x) {
        Some(text) => Json::from(text),
        None => {
            Json::Number(serde_json::Number::from_f64(x).expect("a finite number is a JSON number"))
        }
    }
}

/// A field value as JSON (numbers by [`json_number`]).
pub(crate) fn json_value(value: &Value) -> Json {
    match value {
        Value::Num(x) => json_number(*x),
        Value::Int(i) => Json::from(*i),
        Value::Text(text) => Json::String(text.clone()),
        Value::None => Json::Null,
    }
}

/// The design as a JSON object (the results export embeds it).
pub(crate) fn design_json(design: &Design) -> Json {
    let mut inputs = Map::new();
    for row in input_rows(&design.inputs) {
        inputs.insert(row.path, json_value(&row.value));
    }
    let sizing = &design.sizing;
    let mut sizing_json = Map::new();
    sizing_json.insert("mode".to_owned(), Json::from(sizing.mode.key()));
    sizing_json.insert(
        "free_variable".to_owned(),
        Json::from(variable_key(sizing.variable)),
    );
    sizing_json.insert("target_torque_Nm".to_owned(), json_number(sizing.target_Nm));
    let mut top = Map::new();
    top.insert("format".to_owned(), Json::from(DESIGN_FORMAT));
    top.insert("version".to_owned(), Json::from(DESIGN_VERSION));
    top.insert("inputs".to_owned(), Json::Object(inputs));
    top.insert("sizing".to_owned(), Json::Object(sizing_json));
    Json::Object(top)
}

/// The design file text: every input and the sizing state, indented.
pub fn design_to_json(design: &Design) -> String {
    let mut text = serde_json::to_string_pretty(&design_json(design))
        .expect("a JSON value of maps, strings and numbers always serializes");
    text.push('\n');
    text
}

/// The input value a JSON value stands for, given the field's type; `None` when it does not
/// fit the type (a string for a number, a fraction for a count).
fn input_value(ty: FieldType, json: &Json) -> Option<Value> {
    match (ty, json) {
        (FieldType::F64, Json::Number(n)) => n.as_f64().map(Value::Num),
        (FieldType::I64, Json::Number(n)) => n.as_i64().map(Value::Int),
        (FieldType::OptF64, Json::Null) => Some(Value::None),
        (FieldType::OptF64, Json::Number(n)) => n.as_f64().map(Value::Num),
        (FieldType::Text, Json::String(text)) => Some(Value::Text(text.clone())),
        _ => None,
    }
}

/// What a field of type `ty` expects, for a refusal message.
fn expected(ty: FieldType) -> &'static str {
    match ty {
        FieldType::F64 => "a number",
        FieldType::I64 => "a whole number",
        FieldType::OptF64 => "a number or null",
        FieldType::Text => "a string",
        FieldType::NumOrText => "a number or a string",
    }
}

/// The path an input written as `path` by a file of `version` has now: every migration of a
/// version at or after `version` applied in order (a path renamed twice follows both); `None`
/// when a migration removed it.
fn migrated_path<'a>(path: &'a str, version: u64, migrations: &[PathMigration]) -> Option<&'a str> {
    let mut current = path;
    for &(written_by, old, new) in migrations {
        if version <= written_by && current == old {
            current = new?;
        }
    }
    Some(current)
}

/// The inputs a file's `inputs` object gives: the defaults with every listed path set, a path
/// an older `version` wrote taken through `migrations` first; or every problem.
fn inputs_from(
    json: &Json,
    version: u64,
    migrations: &[PathMigration],
) -> Result<DesignInputs, Vec<String>> {
    let Json::Object(map) = json else {
        return Err(vec!["\"inputs\" is not an object".to_owned()]);
    };
    let mut inputs = DesignInputs::default();
    let types: Vec<(String, FieldType)> = input_rows(&inputs)
        .into_iter()
        .map(|row| (row.path, row.meta.ty))
        .collect();
    let mut problems = Vec::new();
    let mut set_paths: Vec<&str> = Vec::new();
    for (written, json) in map {
        let Some(path) = migrated_path(written, version, migrations) else {
            continue; // removed by a later version: the value means nothing now
        };
        let Some(ty) = types.iter().find(|(p, _)| p == path).map(|(_, ty)| *ty) else {
            problems.push(format!("{written}: no such input"));
            continue;
        };
        if set_paths.contains(&path) {
            problems.push(format!("{written}: {path} is given twice"));
            continue;
        }
        set_paths.push(path);
        match input_value(ty, json) {
            Some(value) => {
                if let Err(error) = inputs.set(path, value) {
                    problems.push(error.to_string());
                }
            }
            None => problems.push(format!("{written}: expected {}, got {json}", expected(ty))),
        }
    }
    if problems.is_empty() {
        Ok(inputs)
    } else {
        Err(problems)
    }
}

/// The sizing state a file's `sizing` object gives; or every problem.
fn sizing_from(json: &Json) -> Result<SizingState, Vec<String>> {
    let Json::Object(map) = json else {
        return Err(vec!["\"sizing\" is not an object".to_owned()]);
    };
    let mut state = SizingState::default();
    let mut problems = Vec::new();
    for (key, value) in map {
        match (key.as_str(), value) {
            ("mode", Json::String(name)) => match SizingMode::from_key(name) {
                Some(mode) => state.mode = mode,
                None => problems.push(format!("unknown mode {name:?}")),
            },
            ("free_variable", Json::String(name)) => match variable_from_key(name) {
                Some(variable) => state.variable = variable,
                None => problems.push(format!("unknown free variable {name:?}")),
            },
            ("target_torque_Nm", Json::Number(n)) => match n.as_f64() {
                Some(target) if target.is_finite() && target > 0.0 => state.target_Nm = target,
                _ => problems.push(format!("target torque {n} is not a positive number")),
            },
            (key, value) => problems.push(format!("unexpected {key:?}: {value}")),
        }
    }
    if problems.is_empty() {
        Ok(state)
    } else {
        Err(problems)
    }
}

/// Reads a design file (the module docs give the format and the rules).
pub fn design_from_json(text: &str) -> Result<Design, LoadError> {
    let json: Json =
        serde_json::from_str(text).map_err(|error| LoadError::NotJson(error.to_string()))?;
    let Json::Object(top) = &json else {
        return Err(LoadError::NotADesign(
            "the top level is not an object".to_owned(),
        ));
    };
    if top.get("format") != Some(&Json::from(DESIGN_FORMAT)) {
        return Err(LoadError::NotADesign(format!(
            "no \"format\": \"{DESIGN_FORMAT}\""
        )));
    }
    let version = match top.get("version") {
        Some(json) => json
            .as_u64()
            .filter(|v| (1..=DESIGN_VERSION).contains(v))
            .ok_or_else(|| LoadError::UnsupportedVersion(json.to_string()))?,
        None => return Err(LoadError::UnsupportedVersion("missing".to_owned())),
    };
    if let Some(key) = top
        .keys()
        .find(|key| !matches!(key.as_str(), "format" | "version" | "inputs" | "sizing"))
    {
        return Err(LoadError::NotADesign(format!("unexpected key {key:?}")));
    }
    // Both are checked before either refuses, so the refusal names every problem.
    let inputs = match top.get("inputs") {
        Some(json) => inputs_from(json, version, PATH_MIGRATIONS),
        None => Ok(DesignInputs::default()),
    };
    let sizing = match top.get("sizing") {
        Some(json) => sizing_from(json),
        None => Ok(SizingState::default()),
    };
    match (inputs, sizing) {
        (Ok(inputs), Ok(sizing)) => Ok(Design { inputs, sizing }),
        (inputs, sizing) => Err(LoadError::Refused {
            inputs: inputs.err().unwrap_or_default(),
            sizing: sizing.err().unwrap_or_default(),
        }),
    }
}

/// URL-safe base64 that writes no padding and reads it either way (a link pasted with `=`
/// padding still loads).
const LINK_BASE64: GeneralPurpose = GeneralPurpose::new(
    &URL_SAFE,
    GeneralPurposeConfig::new()
        .with_encode_padding(false)
        .with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// The `?m=` value of a share link: the compact design JSON, deflated, URL-safe base64.
pub fn encode_share_payload(design: &Design) -> String {
    let json = design_json(design).to_string();
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(json.as_bytes())
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("finishing into a Vec cannot fail");
    LINK_BASE64.encode(compressed)
}

/// The design a `?m=` value holds (surrounding whitespace ignored).
pub fn decode_share_payload(payload: &str) -> Result<Design, LoadError> {
    let bytes = LINK_BASE64
        .decode(payload.trim())
        .map_err(|error| LoadError::Link(format!("not URL-safe base64 ({error})")))?;
    let mut text = String::new();
    DeflateDecoder::new(&bytes[..])
        .take(MAX_DESIGN_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| LoadError::Link(format!("not a compressed design ({error})")))?;
    if text.len() as u64 > MAX_DESIGN_BYTES {
        return Err(LoadError::Link(format!(
            "it inflates past {MAX_DESIGN_BYTES} bytes"
        )));
    }
    design_from_json(&text)
}

/// A share link: `base` (the page's address, e.g. [`PUBLIC_BASE_URL`]) with the `?m=` payload.
pub fn share_link(base: &str, design: &Design) -> String {
    format!("{base}?{SHARE_PARAM}={}", encode_share_payload(design))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sizing::FreeVariable;

    /// A design with a value of every input type changed, and a non-default sizing state.
    fn edited() -> Design {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41; // f64
        inputs.coupling.npole = 12; // i64 count
        inputs.coupling.backiron = 0; // selector
        inputs.coupling.magnets.part_inner = "K&J BX0X04-N52".to_owned(); // text
        inputs.coupling.magnets.axial_length_mm = Some(13.5); // optional, set
        inputs.metal.measured_drag_Nm = Some(0.0123);
        inputs.temperature.slip_loss.cap_integral_T2m4 = 5.27e-10; // tiny
        inputs.metal.life_events = 2.5e7; // large
        Design {
            inputs,
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::RingRadius,
                target_Nm: 3.25,
            },
        }
    }

    #[test]
    fn a_design_file_round_trips_bit_for_bit() {
        for design in [Design::default(), edited()] {
            let text = design_to_json(&design);
            assert_eq!(design_from_json(&text), Ok(design.clone()), "{text}");
        }
    }

    #[test]
    fn a_design_file_names_every_input_and_the_sizing_state() {
        let json: Json = serde_json::from_str(&design_to_json(&edited())).unwrap();
        let inputs = json["inputs"].as_object().unwrap();
        let rows = input_rows(&DesignInputs::default());
        assert_eq!(inputs.len(), rows.len());
        for row in &rows {
            assert!(inputs.contains_key(&row.path), "missing {}", row.path);
        }
        assert_eq!(json["format"], DESIGN_FORMAT);
        assert_eq!(json["version"], DESIGN_VERSION);
        assert_eq!(inputs["coupling.npole"], 12);
        assert_eq!(inputs["coupling.magnets.axial_length_mm"], 13.5);
        assert_eq!(inputs["coupling.magnets.grade_inner"], "");
        assert_eq!(json["sizing"]["mode"], "torque_to_magnets");
        assert_eq!(json["sizing"]["free_variable"], "ring_radius");
        assert_eq!(json["sizing"]["target_torque_Nm"], 3.25);
        // A blank optional input is null, not 0 or a missing key.
        let default: Json = serde_json::from_str(&design_to_json(&Design::default())).unwrap();
        assert_eq!(default["inputs"]["metal.measured_drag_Nm"], Json::Null);
    }

    #[test]
    fn a_share_link_round_trips_and_is_url_safe() {
        for design in [Design::default(), edited()] {
            let link = share_link(PUBLIC_BASE_URL, &design);
            let payload = link
                .strip_prefix("https://linkage.colesorkness.com/magcoupling/?m=")
                .expect("the link is the base, then ?m=");
            assert!(
                payload
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                "{payload}"
            );
            assert_eq!(decode_share_payload(payload), Ok(design.clone()));
            // Pasted with padding or surrounding whitespace, it still loads.
            let padded = format!("  {payload}{}\n", "=".repeat((4 - payload.len() % 4) % 4));
            assert_eq!(decode_share_payload(&padded), Ok(design));
        }
    }

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quotes this length (2,462 characters at the time of writing): the
        // whole design is in the link.
        let payload = encode_share_payload(&Design::default());
        assert!(payload.len() < 2_500, "{} characters", payload.len());
    }

    #[test]
    fn a_missing_path_or_sizing_state_takes_the_default() {
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"coupling.npole": 14}}"#;
        let design = design_from_json(text).unwrap();
        let mut want = DesignInputs::default();
        want.coupling.npole = 14;
        assert_eq!(design.inputs, want);
        assert_eq!(design.sizing, SizingState::default());
        let bare = r#"{"format": "magcoupling-design", "version": 1}"#;
        assert_eq!(design_from_json(bare), Ok(Design::default()));
    }

    #[test]
    fn a_value_outside_its_slider_range_loads_as_it_is() {
        // A hand-edited file or one from another version (Review Focus 1): set() checks type,
        // finiteness and choices, not the slider range, so the value is kept; the panel flags it.
        let text = r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 7.5, "coupling.npole": 64}}"#;
        let design = design_from_json(text).unwrap();
        assert_eq!(design.inputs.metal.face_gap_mm, 7.5);
        assert_eq!(design.inputs.coupling.npole, 64);
        assert_eq!(
            decode_share_payload(&encode_share_payload(&design)),
            Ok(design)
        );
    }

    #[test]
    fn a_float_input_accepts_a_whole_number_and_a_count_refuses_a_fraction() {
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"metal.face_gap_mm": 2, "coupling.npole": 12.0}}"#;
        let Err(LoadError::Refused { inputs, sizing }) = design_from_json(text) else {
            panic!("12.0 is not a whole number in JSON")
        };
        assert_eq!(
            inputs,
            vec!["coupling.npole: expected a whole number, got 12.0".to_owned()]
        );
        assert!(sizing.is_empty());
        let text =
            r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2}}"#;
        assert_eq!(
            design_from_json(text).unwrap().inputs.metal.face_gap_mm,
            2.0
        );
    }

    #[test]
    fn every_problem_in_the_inputs_is_reported_and_nothing_loads() {
        let text = r#"{"format": "magcoupling-design", "version": 1, "inputs": {
            "coupling.backiron": 7,
            "coupling.no_such": 1,
            "metal.face_gap_mm": "wide",
            "coupling.magnets.part_inner": null,
            "metal.measured_drag_Nm": "+inf",
            "metal.web_mm": 3.0
        }}"#;
        let Err(LoadError::Refused { inputs, sizing }) = design_from_json(text) else {
            panic!("refused")
        };
        assert!(sizing.is_empty());
        // serde_json's map is ordered by key.
        assert_eq!(
            inputs,
            vec![
                "coupling.backiron: 7 is not one of the choices".to_owned(),
                "coupling.magnets.part_inner: expected a string, got null".to_owned(),
                "coupling.no_such: no such input".to_owned(),
                "metal.face_gap_mm: expected a number, got \"wide\"".to_owned(),
                "metal.measured_drag_Nm: expected a number or null, got \"+inf\"".to_owned(),
            ]
        );
    }

    #[test]
    fn files_that_are_not_designs_are_refused() {
        let refuse = |text: &str| design_from_json(text).unwrap_err();
        assert!(matches!(refuse("not json"), LoadError::NotJson(_)));
        assert!(matches!(refuse("[1, 2]"), LoadError::NotADesign(_)));
        assert!(matches!(
            refuse(r#"{"format": "mechanism", "version": 1}"#),
            LoadError::NotADesign(_)
        ));
        assert!(matches!(
            refuse(r#"{"format": "magcoupling-design", "version": 1, "extra": 0}"#),
            LoadError::NotADesign(_)
        ));
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design", "version": 2}"#),
            LoadError::UnsupportedVersion("2".to_owned())
        );
        for version in ["0", "1.5", "\"1\"", "-1"] {
            let text = format!(r#"{{"format": "magcoupling-design", "version": {version}}}"#);
            assert!(
                matches!(refuse(&text), LoadError::UnsupportedVersion(_)),
                "{version}"
            );
        }
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design"}"#),
            LoadError::UnsupportedVersion("missing".to_owned())
        );
        assert_eq!(
            refuse(r#"{"format": "magcoupling-design", "version": 1, "inputs": []}"#),
            LoadError::Refused {
                inputs: vec!["\"inputs\" is not an object".to_owned()],
                sizing: Vec::new(),
            }
        );
    }

    #[test]
    fn a_malformed_sizing_state_is_refused() {
        let with = |sizing: &str| {
            design_from_json(&format!(
                r#"{{"format": "magcoupling-design", "version": 1, "sizing": {sizing}}}"#
            ))
        };
        for bad in [
            r#"{"mode": "sideways"}"#,
            r#"{"free_variable": "colour"}"#,
            r#"{"target_torque_Nm": 0}"#,
            r#"{"target_torque_Nm": -1.5}"#,
            r#"{"target_torque_Nm": "2.5"}"#,
            r#"{"target_Nm": 2.5}"#,
            r#"[]"#,
        ] {
            assert!(
                matches!(with(bad), Err(LoadError::Refused { inputs, sizing })
                    if inputs.is_empty() && sizing.len() == 1),
                "{bad}"
            );
        }
        // Every problem of the sizing state is named, in key order.
        assert_eq!(
            with(r#"{"mode": "x", "target_torque_Nm": 0}"#),
            Err(LoadError::Refused {
                inputs: Vec::new(),
                sizing: vec![
                    "unknown mode \"x\"".to_owned(),
                    "target torque 0 is not a positive number".to_owned(),
                ],
            })
        );
        let partial = with(r#"{"mode": "torque_to_magnets"}"#).unwrap().sizing;
        assert_eq!(
            partial,
            SizingState {
                mode: SizingMode::TorqueToMagnets,
                ..SizingState::default()
            }
        );
    }

    #[test]
    fn a_file_with_bad_inputs_and_a_bad_sizing_state_names_both() {
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"coupling.backiron": 7}, "sizing": {"mode": "sideways"}}"#;
        let error = design_from_json(text).unwrap_err();
        assert_eq!(
            error,
            LoadError::Refused {
                inputs: vec!["coupling.backiron: 7 is not one of the choices".to_owned()],
                sizing: vec!["unknown mode \"sideways\"".to_owned()],
            }
        );
        assert_eq!(
            error.to_string(),
            "inputs refused: coupling.backiron: 7 is not one of the choices; \
             sizing state refused: unknown mode \"sideways\""
        );
    }

    /// A made-up history of renames and a removal, for the migration tests (the real table,
    /// [`PATH_MIGRATIONS`], is empty while every path is version 1's).
    const RENAMES: [PathMigration; 3] = [
        (1, "metal.gap_mm", Some("metal.face_gap_v2_mm")),
        (1, "coupling.retired_flag", None),
        (2, "metal.face_gap_v2_mm", Some("metal.face_gap_mm")),
    ];

    #[test]
    fn an_older_file_s_renamed_and_removed_paths_are_migrated() {
        let json = |text: &str| -> Json { serde_json::from_str(text).unwrap() };
        // Version 1 wrote metal.gap_mm: renamed twice since; its removed input is dropped.
        let v1 = json(r#"{"metal.gap_mm": 2.0, "coupling.retired_flag": 1, "coupling.npole": 12}"#);
        let inputs = inputs_from(&v1, 1, &RENAMES).unwrap();
        assert_eq!(inputs.metal.face_gap_mm, 2.0);
        assert_eq!(inputs.coupling.npole, 12);
        // Version 2 wrote the middle name; its old name is no input of version 2.
        let v2 = json(r#"{"metal.face_gap_v2_mm": 1.5}"#);
        assert_eq!(
            inputs_from(&v2, 2, &RENAMES).unwrap().metal.face_gap_mm,
            1.5
        );
        assert_eq!(
            inputs_from(&json(r#"{"metal.gap_mm": 2.0}"#), 2, &RENAMES),
            Err(vec!["metal.gap_mm: no such input".to_owned()])
        );
        // A file of the version after the last rename reads the current paths only.
        assert_eq!(
            inputs_from(&v2, 3, &RENAMES),
            Err(vec!["metal.face_gap_v2_mm: no such input".to_owned()])
        );
        // An old and a new name of one input in one file: refused, not silently one of them.
        let both = json(r#"{"metal.face_gap_mm": 1.0, "metal.gap_mm": 2.0}"#);
        assert_eq!(
            inputs_from(&both, 1, &RENAMES),
            Err(vec![
                "metal.gap_mm: metal.face_gap_mm is given twice".to_owned()
            ])
        );
    }

    #[test]
    fn every_path_migration_leads_to_an_input_of_this_version() {
        let paths: Vec<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|row| row.path)
            .collect();
        let mut last_version = 1;
        for &(written_by, old, _) in PATH_MIGRATIONS {
            assert!(written_by >= last_version, "oldest first: {old}");
            last_version = written_by;
            assert!(
                written_by < DESIGN_VERSION,
                "{old}: bump DESIGN_VERSION with it"
            );
            assert!(!paths.iter().any(|p| p == old), "{old} is still an input");
            if let Some(now) = migrated_path(old, written_by, PATH_MIGRATIONS) {
                assert!(
                    paths.iter().any(|p| p == now),
                    "{old} leads to {now}, no input"
                );
            }
        }
    }

    #[test]
    fn broken_share_links_are_refused() {
        assert!(matches!(
            decode_share_payload("!!!"),
            Err(LoadError::Link(_))
        ));
        let not_deflate = LINK_BASE64.encode(b"plain text, not deflate");
        assert!(matches!(
            decode_share_payload(&not_deflate),
            Err(LoadError::Link(_))
        ));
        // A link to the linkage tool's mechanism decodes but is not a design.
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(br#"{"joints": []}"#).unwrap();
        let mechanism = LINK_BASE64.encode(encoder.finish().unwrap());
        assert!(matches!(
            decode_share_payload(&mechanism),
            Err(LoadError::NotADesign(_))
        ));
    }

    #[test]
    fn a_share_link_that_inflates_past_the_limit_is_refused() {
        let huge = vec![b' '; (MAX_DESIGN_BYTES + 10) as usize];
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(&huge).unwrap();
        let payload = LINK_BASE64.encode(encoder.finish().unwrap());
        assert!(payload.len() < 10_000, "deflate packs the blanks");
        let Err(LoadError::Link(message)) = decode_share_payload(&payload) else {
            panic!("refused")
        };
        assert!(message.contains("inflates past"), "{message}");
    }

    #[test]
    fn non_finite_numbers_are_strings_in_json() {
        assert_eq!(json_number(f64::INFINITY), Json::from("+inf"));
        assert_eq!(json_number(f64::NEG_INFINITY), Json::from("-inf"));
        assert_eq!(json_number(f64::NAN), Json::from("NaN"));
        assert_eq!(json_number(-0.5), serde_json::json!(-0.5));
        assert_eq!(json_value(&Value::None), Json::Null);
        assert_eq!(json_value(&Value::Int(-3)), Json::from(-3));
    }

    #[test]
    fn load_errors_read_as_sentences() {
        assert_eq!(
            LoadError::UnsupportedVersion("2".to_owned()).to_string(),
            "design file version 2: this calculator reads version 1 and older"
        );
        let refused = |inputs: &[&str], sizing: &[&str]| {
            LoadError::Refused {
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
                sizing: sizing.iter().map(|s| s.to_string()).collect(),
            }
            .to_string()
        };
        assert_eq!(
            refused(&["a: x", "b: y"], &[]),
            "inputs refused: a: x; b: y"
        );
        assert_eq!(refused(&[], &["m"]), "sizing state refused: m");
        assert_eq!(
            refused(&["a: x"], &["m", "n"]),
            "inputs refused: a: x; sizing state refused: m; n"
        );
    }
}
