//! The calculator's egui panel (feature `gui`), hostable by any egui app.
//!
//! [`MagcouplingPanel`] holds the design inputs and results and draws them:
//! the standalone app (feature `app`) shows it as a full page, the linkage app
//! in an `egui::Window` (M5). egui only, no eframe: a host brings its own
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

pub mod checks;
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
pub mod diagrams;
pub mod explain;
pub mod explorer;
mod format;
pub mod geometry;
pub mod geometry_view;
pub mod guidance;
pub mod history;
pub mod input_ui;
pub mod inputs;
pub mod look;
pub mod notes_view;
mod panel;
pub mod pickers;
pub mod plain_text;
pub mod plots;
pub mod readouts;
pub mod result_groups;
pub mod results_table;
pub mod session;
mod size_controls;
pub mod sizing;
pub mod spreadsheet;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tiles;
pub mod torque_card;
pub mod trace;
pub mod typeset;
pub mod xlsx;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{
    ASSUMPTIONS_FOLD, CentreView, DESIGN_MENU, DETAILED_INPUTS, EXPORT, LOAD_DESIGN,
    MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, RESET_TO_START, SHARE, SHOW_DETAILS,
    SIZING_LOG_PREFIX,
};
// The Size tab's gap text, public for the slice-3 screen dump (`size_controls` is private).
pub use size_controls::torque_gap_text;
// The Size column's rows, for the app's check of the smoke prompt's words; Auto's toggle and
// its line's button, for the app's width test.
#[cfg(all(test, feature = "app"))]
pub(crate) use size_controls::{AUTO, OPEN_SIZE_TAB, SIZE_ROWS};

/// A line of `ui` whose widgets wrap onto further lines, at the width `ui` has less one point:
/// `ui.horizontal_wrapped` with a point kept free. egui's text layout rounds the width it wraps
/// at to a whole point and lets a row run half a point past it, and a label in a wrapping line
/// takes each row as it was laid out, so a row that just fits stands a fraction of a point past
/// the line. In a side column that fraction widens the column, which egui then keeps that wide:
/// every wrapping line of the left column and of the Explain panel is drawn through this
/// (BL-099, BL-098).
pub(crate) fn wrapping_line<R>(
    ui: &mut egui::Ui,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<R> {
    let size = egui::vec2(ui.available_width() - 1.0, ui.spacing().interact_size.y);
    let wrapping = egui::Layout::left_to_right(egui::Align::Center).with_main_wrap(true);
    ui.allocate_ui_with_layout(size, wrapping, add)
}

/// Adds `label` to a wrapping line as one block in the rest of the line: egui lays a label out
/// in a wrapping line row by row, each row a Tab stop of its own, and only the first shows the
/// focus (SK-1). The block is one text row high (as epaint rounds it), so the line centres it as
/// it centred a label that fits. After a wrapped label, what follows starts a line of its own at
/// a widget's height: a wrapping line's next row takes the height of the one before. An input
/// row's label, a term's name in the Equation panel and the closest note's link there (an
/// `egui::Link`, a label drawn as a link) are drawn through this.
pub(crate) fn label_block(ui: &mut egui::Ui, label: impl egui::Widget) -> egui::Response {
    let row = ui.text_style_height(&egui::TextStyle::Body);
    let line = egui::vec2(
        ui.available_size_before_wrap().x,
        egui::emath::GuiRounding::round_to_pixels(row, ui.pixels_per_point()),
    );
    let response = ui
        .allocate_ui_with_layout(line, egui::Layout::top_down(egui::Align::Min), |ui| {
            ui.add(label)
        })
        .inner;
    if response.rect.height() > 1.5 * line.y {
        ui.end_row();
        ui.set_row_height(ui.spacing().interact_size.y);
    }
    response
}

/// Scrolls the scroll area whose content `ui` is, by the smallest amount that shows it whole,
/// to the widget the keyboard's focus has arrived on (BL-100: egui scrolls to no widget for
/// taking the focus). Called last in the area's content. Only in the first frame the area is
/// drawn with the focus there, so a wheel or a drag of the scroll bar afterwards is not fought.
/// Returns the widget's rect when it scrolls to it: egui hands both axes' scroll targets to the
/// innermost area alone, so an area nested in another passes the rect on to the outer one
/// ([`sideways_scroll_area`], BL-101).
pub(crate) fn follow_focus(ui: &egui::Ui) -> Option<egui::Rect> {
    let focused = ui.memory(|m| m.focused());
    // Where the focus was when this area was last drawn. egui's own `gained_focus` misses a
    // move made between two frames, as an Up or Down arrow key's is. Kept under the content's
    // unique id: egui gives a scroll area's content its parent's id, which sibling areas share
    // (the check list and the Explain panel under it, BL-101); the unique id holds while the
    // widgets before the area in its parent stay as they are.
    let seen = ui.data_mut(|d| {
        let key = ui.unique_id().with("follow_focus");
        let seen = d.get_temp_mut_or_default::<Option<egui::Id>>(key);
        std::mem::replace(seen, focused)
    });
    // A widget drawn in this area is clipped by this area's view, and no other is: one in
    // another panel, or in a list open over this one, would scroll the area to nothing. The
    // panels share one layer, so the clip check is the one that holds an area still while the
    // focus is elsewhere (a test wheels the left column and tabs past it); the layer check
    // adds little, as a list's widget passes the clip check only whole in the view, where no
    // scroll follows. A sideways area nested in an up-and-down one is clipped up and down by
    // the outer area's view (egui keeps the parent's clip on the axis an area does not scroll),
    // so its widgets are also those in its content: else it would take the outer area's,
    // read from the frame before, and scroll the outer area a second time (BL-101).
    if focused != seen
        && let Some(widget) = focused.and_then(|id| ui.ctx().read_response(id))
        && widget.layer_id == ui.layer_id()
        && widget.interact_rect == widget.rect.intersect(ui.clip_rect())
        && ui.min_rect().contains_rect(widget.rect)
    {
        ui.scroll_to_rect(widget.rect, None);
        return Some(widget.rect);
    }
    None
}

/// A sideways scroll area inside an up-and-down one (the Equation panel's formula, the Clamp
/// tab's screw-size table) that follows the keyboard's focus both ways (BL-101). It calls
/// [`follow_focus`] last in its content, so it scrolls sideways to the widget; egui gives it
/// both axes' targets, so the outer area's content asks again for the rect once it has ended,
/// and the outer area scrolls up or down to the widget. A widget wider than the view shows from
/// its start (egui's least scroll).
pub(crate) fn sideways_scroll_area<R>(
    ui: &mut egui::Ui,
    id_salt: &str,
    add: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    let (inner, followed) = egui::ScrollArea::horizontal()
        .id_salt(id_salt)
        .show(ui, |ui| {
            let inner = add(ui);
            (inner, follow_focus(ui))
        })
        .inner;
    if let Some(rect) = followed {
        ui.scroll_to_rect(rect, None);
    }
    inner
}
