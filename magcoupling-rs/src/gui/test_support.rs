//! Helpers for the headless egui tests of the panel: one frame with injected
//! input, input events, and inspection of what the frame painted. The same
//! pattern as `linkage-sim-rs/src/gui/test_support.rs` (a separate crate, so
//! the few helpers used here are repeated, not shared).

/// One headless frame of `draw` inside a central panel, with `events` as the
/// frame's input. Returns what egui painted.
pub(crate) fn central_panel_frame(
    ctx: &egui::Context,
    events: Vec<egui::Event>,
    mut draw: impl FnMut(&mut egui::Ui),
) -> egui::FullOutput {
    let input = egui::RawInput {
        events,
        ..Default::default()
    };
    ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| draw(ui));
    })
}

/// A key press with no modifiers.
pub(crate) fn key_press(key: egui::Key) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }
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
