//! Helpers for the headless egui tests of the panel: one frame with injected
//! input, input events, and inspection of what the frame painted. The same
//! pattern as `linkage-sim-rs/src/gui/test_support.rs` (a separate crate, so
//! the few helpers used here are repeated, not shared).

/// A design with f_end below 0 (the engine's short-magnet test, audit M9): 2 mm manual
/// blocks with c_end 0.5.
pub(crate) fn short_magnets() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.c_end = 0.5;
    inputs.coupling.magnets.part_inner.clear();
    inputs.coupling.magnets.part_outer.clear();
    inputs.coupling.magnets.manual_inner_length_mm = 2.0;
    inputs.coupling.magnets.manual_outer_length_mm = 2.0;
    inputs
}

/// The defaults with `npole` poles per ring, the ring size kept.
pub(crate) fn with_poles(npole: i64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.npole = npole;
    inputs
}

/// A design that passes all 8 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
/// to spare: the defaults with a 1.8 mm face gap (running clearance about 0.1 mm over its
/// target), a 2.0 mm cup wall (the flux needs about 1.85 mm at that gap) and a 1.6 N·m
/// requirement (hot-low torque about 2.0 N·m, a quarter over it: a hotter-rated grade, weaker
/// than N42SH, still meets it). The guidance tests break one check of it at a time, so a fix
/// that turns another check red shows.
pub(crate) fn passing_design() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.0;
    inputs.metal.required_min_Nm = 1.6;
    inputs
}

/// A hot day: 80 °C ambient and a 60 °C driving rise start the magnets at 140 °C, above
/// both the magnet and the adhesive limit, with the hottest-rated library part in use.
pub(crate) fn hot_day(inputs: &mut crate::DesignInputs) {
    inputs.temperature.duty.hot_ambient_C = 80.0;
    inputs.temperature.duty.driving_rise_C = 60.0;
}

/// No screw fits a 12 mm clamp boss.
pub(crate) fn small_boss(inputs: &mut crate::DesignInputs) {
    inputs.clamps.boss_od_mm = 12.0;
}

/// [`passing_design`] with a [`small_boss`]: it fails the clamp alone, so the clamp section's
/// pause decides whether the design passes.
pub(crate) fn only_the_clamp_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    small_boss(&mut inputs);
    inputs
}

/// The defaults at the clamp's limit: the clamp's safety factor raised until its capacity only
/// just covers the requirement.
pub(crate) fn clamp_at_its_limit() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    let base = crate::compute_all(&inputs);
    let crate::engine::meta::NumOrText::Num(capacity) = base.clamps.capacity_Nm else {
        panic!("a screw fits the example design's clamp")
    };
    inputs.clamps.safety_factor = capacity / base.clamps.max_torque_Nm * (1.0 - 1e-9);
    inputs
}

/// The screen of the headless frames [points]: a laptop window.
pub(crate) const SCREEN: egui::Vec2 = egui::vec2(1280.0, 1024.0);

/// One headless frame of `draw` inside a central panel on a screen of `size`, with `events`
/// as the frame's input. Returns what egui painted.
pub(crate) fn sized_frame(
    ctx: &egui::Context,
    size: egui::Vec2,
    events: Vec<egui::Event>,
    draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    sized_frame_at(ctx, size, None, events, draw)
}

/// [`sized_frame`] at the input time `time` [s] (`None`: egui adds its predicted frame time,
/// 1/60 s, to the last frame's).
pub(crate) fn sized_frame_at(
    ctx: &egui::Context,
    size: egui::Vec2,
    time: Option<f64>,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        time,
        ..Default::default()
    };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
}

/// Shows `ctx`'s tooltips at once: no delay, and the pointer need not rest.
pub(crate) fn tooltips_at_once(ctx: &egui::Context) {
    ctx.style_mut(|s| {
        s.interaction.tooltip_delay = 0.0;
        s.interaction.show_tooltips_only_when_still = false;
    });
}

/// Frames of `draw` on a screen of `size`, `ctx`'s tooltips at once ([`tooltips_at_once`]):
/// two to lay it out, one with the pointer moved to the point `target` finds in the second, and
/// two more, so what the pointer hovers shows its tooltip. Returns the second frame (the
/// pointer not there yet) and the last.
pub(crate) fn hover_frames(
    ctx: &egui::Context,
    size: egui::Vec2,
    target: impl FnOnce(&egui::FullOutput) -> egui::Pos2,
    mut draw: impl FnMut(&mut egui::Ui),
) -> (egui::FullOutput, egui::FullOutput) {
    tooltips_at_once(ctx);
    let mut frame = |time: f64, events: Vec<egui::Event>| {
        sized_frame_at(ctx, size, Some(time), events, &mut draw)
    };
    frame(0.0, Vec::new());
    let before = frame(0.1, Vec::new());
    frame(0.2, vec![egui::Event::PointerMoved(target(&before))]);
    frame(0.3, Vec::new());
    (before, frame(0.4, Vec::new()))
}

/// Hovers the first drawn `text` of `draw` on a screen of `size` ([`hover_frames`]), with the
/// spreadsheet details off and then on (spec S2-5: off hides every provenance line), and asserts
/// that the hover drew `hover(show_details)` and not `hover(!show_details)`, and that `path` is
/// a line of the hover text only with the details. `draw` draws the view with the readouts it is
/// handed. The hover text without the details can be the label alone, which the view may draw
/// too: the hover counts as one more drawn copy of its text.
pub(crate) fn assert_hover_names_path_only_with_details(
    size: egui::Vec2,
    path: &str,
    text: &str,
    hover: impl Fn(bool) -> String,
    mut draw: impl FnMut(&mut egui::Ui, &mut crate::gui::readouts::Readouts),
) {
    use crate::gui::readouts::Readouts;
    use crate::gui::typeset::TermColors;
    for show_details in [false, true] {
        let ctx = egui::Context::default();
        let (before, after) = hover_frames(
            &ctx,
            size,
            |output| text_center(output, text),
            |ui| draw(ui, &mut Readouts::new(TermColors::default(), show_details)),
        );
        let (want, other) = (hover(show_details), hover(!show_details));
        assert_eq!(
            drawn_count(&after, &want),
            drawn_count(&before, &want) + 1,
            "details {show_details}: the hover draws {want:?}"
        );
        assert_eq!(
            drawn_count(&after, &other),
            drawn_count(&before, &other),
            "details {show_details}: the hover draws {other:?}"
        );
        assert_eq!(
            want.lines().any(|line| line == path),
            show_details,
            "details {show_details}: {want:?}"
        );
    }
}

/// One headless frame of `draw` in a child ui whose max rect is `region`, on a screen of
/// `size`, as the centre region lays a view out above the Equation panel but without its clip:
/// what the view lays out past the region's foot shows in the rect the child used. Returns what
/// egui painted and that rect.
pub(crate) fn region_frame(
    ctx: &egui::Context,
    size: egui::Vec2,
    region: egui::Rect,
    mut draw: impl FnMut(&mut egui::Ui),
) -> (egui::FullOutput, egui::Rect) {
    let mut used = egui::Rect::NOTHING;
    let output = sized_frame(ctx, size, Vec::new(), |ui| {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(region));
        draw(&mut child);
        used = child.min_rect();
    });
    (output, used)
}

/// A key event with no modifiers: a press, or (`pressed` false) its release.
pub(crate) fn key_event(key: egui::Key, pressed: bool) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
}

/// A key tapped with no modifiers: pressed and released in one frame, as a user taps it. egui
/// keeps a pressed key down until its release and reads another press of it as a repeat.
pub(crate) fn key_tap(key: egui::Key) -> Vec<egui::Event> {
    vec![key_event(key, true), key_event(key, false)]
}

/// Ctrl+A (Cmd+A on a Mac) tapped: select all in the focused text field.
pub(crate) fn select_all() -> Vec<egui::Event> {
    [true, false]
        .map(|pressed| egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        })
        .to_vec()
}

/// A primary-button press (`pressed`) or release at `pos`.
pub(crate) fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

/// Every shape egui painted in a frame, the shapes nested in a `Shape::Vec`
/// flattened, in paint order: what the helpers below and the tests' own
/// filters walk.
pub(crate) fn flat_shapes(output: &egui::FullOutput) -> Vec<&egui::Shape> {
    fn walk<'a>(shape: &'a egui::Shape, shapes: &mut Vec<&'a egui::Shape>) {
        match shape {
            egui::Shape::Vec(nested) => nested.iter().for_each(|s| walk(s, shapes)),
            other => shapes.push(other),
        }
    }
    let mut shapes = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut shapes);
    }
    shapes
}

/// Every text egui drew in a frame (widgets and painter text), nested shapes
/// included, in paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) => Some(text.galley.text().to_owned()),
            _ => None,
        })
        .collect()
}

/// How many drawn texts of a frame equal `text`.
pub(crate) fn drawn_count(output: &egui::FullOutput, text: &str) -> usize {
    drawn_texts(output).iter().filter(|t| *t == text).count()
}

/// Screen rects of every drawn text equal to `needle`, in paint order.
pub(crate) fn text_rects(output: &egui::FullOutput, needle: &str) -> Vec<egui::Rect> {
    flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.galley.rect.translate(text.pos.to_vec2()))
            }
            _ => None,
        })
        .collect()
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    text_rects(output, needle).into_iter().next()
}

/// The centre of the first drawn text equal to `needle`: where a test points to hover it.
/// Panics when no such text is drawn.
pub(crate) fn text_center(output: &egui::FullOutput, needle: &str) -> egui::Pos2 {
    text_rect(output, needle)
        .unwrap_or_else(|| panic!("no {needle:?} drawn"))
        .center()
}

/// Asserts that every character of `text` but whitespace has a glyph in egui's default fonts
/// at proportional 14 (the UI's criterion): egui draws an empty box for one it lacks. `what`
/// names the text's source in the failure. The fonts load on `ctx`'s first frame.
pub(crate) fn assert_glyphs(ctx: &egui::Context, text: &str, what: &str) {
    let font = egui::FontId::proportional(14.0);
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        assert!(
            ctx.fonts(|f| f.has_glyph(&font, c)),
            "{what}: no glyph for {c:?} (U+{:04X}) in {text:?}",
            c as u32
        );
    }
}

/// The colour of the first drawn text equal to `needle`: its override, else its first section's
/// colour (a `RichText` colour or a painter text's), else the fallback colour.
pub(crate) fn text_color(output: &egui::FullOutput, needle: &str) -> Option<egui::Color32> {
    flat_shapes(output)
        .into_iter()
        .find_map(|shape| match shape {
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.override_text_color.unwrap_or_else(|| {
                    match text.galley.job.sections.first().map(|s| s.format.color) {
                        Some(color) if color != egui::Color32::PLACEHOLDER => color,
                        _ => text.fallback_color,
                    }
                }))
            }
            _ => None,
        })
}

/// Every sheet of an .xlsx file as calamine reads it: its name and its rows from A1 (calamine's
/// range starts at the first cell used; every sheet of the export uses A1), each row as wide as
/// the sheet's widest, `Data::Empty` where nothing was written.
pub(crate) fn read_xlsx(bytes: &[u8]) -> Vec<(String, Vec<Vec<calamine::Data>>)> {
    use calamine::Reader;
    let mut workbook: calamine::Xlsx<_> =
        calamine::open_workbook_from_rs(std::io::Cursor::new(bytes.to_vec()))
            .expect("an .xlsx file calamine reads");
    workbook
        .sheet_names()
        .into_iter()
        .map(|name| {
            let range = workbook.worksheet_range(&name).expect("the sheet reads");
            assert_eq!(
                range.start().unwrap_or((0, 0)),
                (0, 0),
                "{name} starts at A1"
            );
            let rows = range.rows().map(<[calamine::Data]>::to_vec).collect();
            (name, rows)
        })
        .collect()
}

/// The spreadsheet tests' snapshot of `inputs` and `results`: Magnets -> Torque, the workflow
/// order, a stand-in share link and export time.
pub(crate) fn snapshot<'a>(
    inputs: &'a crate::DesignInputs,
    results: &'a crate::DesignResults,
) -> crate::gui::spreadsheet::Snapshot<'a> {
    crate::gui::spreadsheet::Snapshot {
        inputs,
        results,
        sizing: crate::gui::sizing::SizingState::default(),
        sizing_status: None,
        input_order: crate::gui::inputs::InputOrder::Workflow,
        share_link: "https://example.test/magcoupling/?m=abc".to_owned(),
        exported_unix_s: 1_790_000_000,
    }
}

/// The text of the part `name` of an .xlsx file (a zip archive), e.g. `xl/styles.xml`.
pub(crate) fn xlsx_part(bytes: &[u8], name: &str) -> String {
    use std::io::Read;
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(bytes)).expect("an .xlsx file is a zip");
    let mut part = archive
        .by_name(name)
        .unwrap_or_else(|_| panic!("no part {name}"));
    let mut text = String::new();
    part.read_to_string(&mut text)
        .expect("an XML part is UTF-8");
    text
}
