//! A teaching note as the Explain panel draws it (spec Addendum A4; UX overhaul decision C-11).
//! A note ([`note_ui`]) has its diagram ([`crate::gui::diagrams`]), its "What to change"
//! entries, its "Watch out" lines and its sources under a "Where this comes from" fold that
//! starts closed. A `{result.path}` in a note's text is drawn as that result's number on the
//! design shown ([`live_value`]), or as "no prediction" where it has none. An entry's input is a
//! link to its row, or plain text with the reason where the input is not free on the design
//! shown ([`held_reason`]: Auto holds it or a part pick sets it; or the slider is at its end
//! that way); the links to the equations a note explains are [`explains_ui`]'s. The Equation
//! panel (`crate::gui::explorer`) shows a note's title alone, as its lesson link; this module
//! was part of it until BL-088.

use crate::engine::explain::note_text;
use crate::engine::explain::notes::{Change, Note};
use crate::engine::meta::{InputSet, ResultSet, Value};
use crate::engine::picks::set_by_pick;
use crate::gui::dashboard::{end_effect_out_of_range, greyed_by_end_effect};
use crate::gui::diagrams::diagram_ui;
use crate::gui::explorer::{crumb, term_label};
use crate::gui::guidance::result_text;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::readouts::registry;
use crate::gui::sizing::{SizingState, held_by_auto};
use crate::gui::typeset::glyph_safe;
use crate::gui::wrapping_line;
use crate::{DesignInputs, DesignResults};

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

/// The start of the links at the head of a note to the equations it explains.
pub const EXPLAINS: &str = "Explains";

/// What a note's change entries are drawn against: the design shown, its results, and why an
/// input is not free on it (`held`: Auto holds it or a part pick sets it, [`held_reason`];
/// `None` for a free input).
pub struct NoteView<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
    pub held: &'a dyn Fn(&str) -> Option<String>,
}

/// What a click in a teaching note, or in the lesson line under an equation, asks for
/// ([`Explorer::follow_note_link`](crate::gui::explorer::Explorer::follow_note_link)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteLink {
    /// Highlight this input's row, as a leaf term does.
    Input(&'static str),
    /// Show this reviewed note in the Explain panel (an equation's lesson, `lesson_ui`, or
    /// its closest note, [`note_fallback`](crate::gui::explorer::note_fallback)).
    Note(&'static str),
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
pub(crate) fn note_text_on(text: &str, view: &NoteView<'_>) -> String {
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
    wrapping_line(ui, |ui| {
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

/// The links at the head of a note in the Explain panel to the equations it explains (a
/// family's template is left out), after [`EXPLAINS`]; nothing for a note that explains none.
/// Returns the equation clicked, which the panel opens here.
pub fn explains_ui(ui: &mut egui::Ui, note: &Note) -> Option<&'static str> {
    let paths: Vec<&'static str> = note
        .equations
        .iter()
        .copied()
        .filter(|p| !p.contains('#') && registry().equation_for(p).is_some())
        .collect();
    let mut clicked = None;
    if !paths.is_empty() {
        wrapping_line(ui, |ui| {
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.strong(EXPLAINS);
            for path in paths {
                if ui
                    .link(crumb(path))
                    .on_hover_text(term_label(path))
                    .clicked()
                {
                    clicked = Some(path);
                }
            }
        });
    }
    clicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::explain::notes::{self, Review};
    use crate::gui::dashboard::result_info;
    use crate::gui::explorer::reviewed_note;
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::test_support::{design, nothing_held};

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

    /// Asserts that `note`'s entry on `input` draws its label as plain text: a click on it asks
    /// for no link, and " (<reason>)" is drawn once. The entry on `still_linked` stays a link.
    fn assert_entry_plain_with_reason(
        ctx: &egui::Context,
        note: &Note,
        view: &NoteView<'_>,
        input: &str,
        reason: &str,
        still_linked: &'static str,
    ) {
        use crate::gui::test_support::drawn_count;
        let (link, output) = click_in_note(ctx, note, view, label_for(input));
        assert_eq!(link, None, "{input}: an unlinked input is no link");
        assert_eq!(drawn_count(&output, &format!(" ({reason})")), 1, "{input}");
        let (link, _) = click_in_note(ctx, note, view, label_for(still_linked));
        assert_eq!(
            link,
            Some(NoteLink::Input(still_linked)),
            "{still_linked} stays a link"
        );
    }

    /// The slider of the input at `path`.
    fn slider(path: &str) -> crate::engine::meta::SliderRange {
        InputCatalogue::get()
            .entry(path)
            .unwrap()
            .meta
            .range
            .unwrap()
    }

    /// The panel's `held` with Auto off: only a pick holds an input of `inputs`.
    fn auto_off(inputs: &DesignInputs) -> impl Fn(&str) -> Option<String> + '_ {
        move |path| held_reason(&SizingState::default(), inputs, path)
    }

    /// Why `change` draws its input as plain text on `inputs`, with `held` naming held inputs.
    fn unlinked_with(
        inputs: &DesignInputs,
        change: &Change,
        held: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        let results = compute_all(inputs);
        let view = NoteView {
            inputs,
            results: &results,
            held,
        };
        unlinked_reason(change, &view)
    }

    /// Why `change` draws its input as plain text on `inputs` with Auto off ([`auto_off`]).
    fn unlinked_on(inputs: &DesignInputs, change: &Change) -> Option<String> {
        unlinked_with(inputs, change, &auto_off(inputs))
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
        let ctx = egui::Context::default();
        let (inputs, results) = design();
        let held_entry = &RENDER_NOTE.change[0];
        let held = |path: &str| (path == held_entry.input).then(|| HELD_BY_AUTO.to_owned());
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let free = RENDER_NOTE.change[1].input;
        assert_entry_plain_with_reason(
            &ctx,
            &RENDER_NOTE,
            &view,
            held_entry.input,
            HELD_BY_AUTO,
            free,
        );
    }

    #[test]
    fn an_entry_whose_input_cannot_move_its_way_names_why() {
        use crate::engine::explain::notes::Step;
        let entry = |input, step| Change {
            input,
            step,
            aim: None,
            text: "",
        };
        let reason =
            |inputs: &DesignInputs, change: &Change| unlinked_with(inputs, change, &nothing_held);
        let gap = "metal.face_gap_mm";
        let (lowest, highest) = (slider(gap).min, slider(gap).max);
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
        poles.coupling.npole = slider("coupling.npole").max as i64;
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
        assert_eq!(
            unlinked_with(&inputs, &entry(gap, Step::Lower), &held).as_deref(),
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
        let pick = format!(
            "set by the {} pick",
            label_for("coupling.magnets.part_outer")
        );
        let defaults = DesignInputs::default();
        assert_eq!(unlinked_on(&defaults, &width), Some(pick.clone()));
        assert_eq!(unlinked_on(&defaults, &thickness), Some(pick));
        let (_, custom) = probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the custom-dimensions probe design");
        assert_eq!(unlinked_on(&custom, &width), None);
        assert_eq!(unlinked_on(&custom, &thickness), None);
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
            unlinked_on(&inputs, &entry)
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
        let sleeve = "metal.sleeve_mm";
        let note = notes::note("a5.high_conductivity_sleeve_or_liner").unwrap();
        let entry = note
            .change
            .iter()
            .find(|c| c.input == sleeve)
            .expect("the sleeve note has an entry on the sleeve thickness");
        assert_eq!(entry.step, notes::Step::Lower);
        let mut inputs = DesignInputs::default();
        inputs.set(sleeve, Value::Num(slider(sleeve).min)).unwrap();
        let results = compute_all(&inputs);
        let held = auto_off(&inputs);
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &held,
        };
        let ctx = egui::Context::default();
        let liner = "metal.liner_mm";
        assert_entry_plain_with_reason(&ctx, note, &view, sleeve, AT_BOTTOM, liner);
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

    /// BL-064 (spec 6.1 and 7.11): the shaft-joint note's sentence 4 as the panel draws it: the
    /// temperature the inner hub's fit would need, beside the wheel shaft's live shrink, against
    /// the allowed temperature. At the defaults; at library N42 parts in both rings, where the
    /// inner ring's single-ring onset less the cure margin binds (owner R-15; the guide tests'
    /// `breaking(n42_parts)`); with mixed rings (inner N42M, outer N35; the guide tests'
    /// `breaking(mixed_rings)`), where the outer ring governs and the cap is still the inner
    /// ring's own onset less the margin (the owner's ring ruling); on an uncooled wheel shaft,
    /// whose shrink reads 0; and on a 40 °C one, its slider's top, where the shaft grows (the
    /// render the physics review decides on).
    #[test]
    fn the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink() {
        use crate::engine::explain::note_text::render;
        use crate::engine::temperature::RING_OUTER;
        use crate::gui::test_support::{mixed_rings, n42_parts, passing_design};
        let sentence = notes::note("shaft_joint")
            .expect("the shaft-joint note")
            .sentences[4];
        let drawn = |inputs: &DesignInputs| {
            let results = compute_all(inputs);
            render(sentence, |path: &str| live_value(&results, path))
        };
        let wheel_shaft_at = |celsius: f64| {
            let mut inputs = DesignInputs::default();
            inputs.shaft_joint.wheel_shaft_fit_C = celsius;
            inputs
        };
        let mut n42 = passing_design();
        n42_parts(&mut n42);
        // Mixed rings: the outer ring governs the magnets' limit, so its single-ring onset is not
        // the inner ring's own, which the bonded inner hub's cap reads.
        let mut mixed = passing_design();
        mixed_rings(&mut mixed);
        let demag = compute_all(&mixed).temperature.demag;
        assert_eq!(demag.demag_ring, RING_OUTER);
        assert_ne!(demag.onset_single_ring_C, demag.inner_onset_single_ring_C);
        for (inputs, words) in [
            (
                DesignInputs::default(),
                "would need 93.27 °C on the wheel shaft shrunk 18.99 µm; allowed: 100.0 °C,",
            ),
            (n42, "allowed: 59.83 °C,"),
            (mixed, "allowed: 81.35 °C,"),
            (
                wheel_shaft_at(20.0),
                "would need 247.6 °C on the wheel shaft shrunk 0 µm;",
            ),
            (
                wheel_shaft_at(40.0),
                "would need 266.3 °C on the wheel shaft shrunk -2.300 µm;",
            ),
        ] {
            let text = drawn(&inputs);
            assert!(text.contains(words), "{words:?} in {text:?}");
        }
    }

    #[test]
    fn every_note_draws_at_every_probe_design_with_glyphs_and_no_braces() {
        use crate::engine::explain::note_text::rendered_texts;
        use crate::engine::explain::probes::probe_designs;
        use crate::gui::test_support::{assert_glyphs, glyph_context};
        let ctx = glyph_context();
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
    fn a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links() {
        // Amendment A-25 T-6 (spec section 5, a narrow window): the pull-out note drawn alone in
        // a panel 400 points wide. Every change entry's input label is drawn inside the
        // panel, and a click on the last character of the inner magnet part's label, laid out in
        // a wrapped row, asks for that input's row.
        use crate::gui::test_support::{last_characters, primary_button, sized_frame, text_rects};
        let size = egui::vec2(400.0, 800.0);
        let note = reviewed_note("pullout_angle").expect("the pull-out note is reviewed");
        let (inputs, results) = design();
        let view = NoteView {
            inputs: &inputs,
            results: &results,
            held: &nothing_held,
        };
        let ctx = egui::Context::default();
        let frame = |events: Vec<egui::Event>| {
            let mut link = None;
            let output = sized_frame(&ctx, size, events, |ui| {
                link = note_ui(ui, note, &view);
            });
            (output, link)
        };
        frame(Vec::new());
        let (output, _) = frame(Vec::new());
        let panel = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        for change in note.change {
            let label = glyph_safe(label_for(change.input));
            let rects = text_rects(&output, &label);
            assert_eq!(rects.len(), 1, "{label:?} is drawn once");
            assert!(panel.contains_rect(rects[0]), "{label:?} at {:?}", rects[0]);
        }
        let part = "coupling.magnets.part_inner";
        let at = last_characters(&output, &glyph_safe(label_for(part)))[0];
        frame(vec![egui::Event::PointerMoved(at)]);
        frame(vec![primary_button(at, true)]);
        let (_, link) = frame(vec![primary_button(at, false)]);
        assert_eq!(link, Some(NoteLink::Input(part)));
    }
}
