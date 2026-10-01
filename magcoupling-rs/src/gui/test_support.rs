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
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
        ..Default::default()
    };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
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

/// Every text egui drew in a frame (widgets and painter text), nested shapes
/// included, in paint order.
pub(crate) fn drawn_texts(output: &egui::FullOutput) -> Vec<String> {
    fn walk(shape: &egui::Shape, texts: &mut Vec<String>) {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| walk(s, texts)),
            egui::Shape::Text(text) => texts.push(text.galley.text().to_owned()),
            _ => {}
        }
    }
    let mut texts = Vec::new();
    for clipped in &output.shapes {
        walk(&clipped.shape, &mut texts);
    }
    texts
}

/// Screen rect of the first drawn text equal to `needle`, e.g. a button
/// label: lets a test click a widget whose id it cannot know.
pub(crate) fn text_rect(output: &egui::FullOutput, needle: &str) -> Option<egui::Rect> {
    fn walk(shape: &egui::Shape, needle: &str) -> Option<egui::Rect> {
        match shape {
            egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| walk(s, needle)),
            egui::Shape::Text(text) if text.galley.text() == needle => {
                Some(text.galley.rect.translate(text.pos.to_vec2()))
            }
            _ => None,
        }
    }
    output.shapes.iter().find_map(|c| walk(&c.shape, needle))
}
