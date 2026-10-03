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
