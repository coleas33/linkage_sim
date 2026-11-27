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
// The Size column's rows, for the app's check of the smoke prompt's words.
#[cfg(all(test, feature = "app"))]
pub(crate) use size_controls::SIZE_ROWS;

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

/// Scrolls the scroll area whose content `ui` is, by the smallest amount that shows it whole,
/// to the widget the keyboard's focus has arrived on (BL-100: egui scrolls to no widget for
/// taking the focus). Called last in the area's content. Only in the first frame the area is
/// drawn with the focus there, so a wheel or a drag of the scroll bar afterwards is not fought.
///
/// Not for an area nested in another: egui hands a scroll to the innermost area alone.
pub(crate) fn follow_focus(ui: &egui::Ui) {
    let focused = ui.memory(|m| m.focused());
    // Where the focus was when this area was last drawn. egui's own `gained_focus` misses a
    // move made between two frames, as an Up or Down arrow key's is.
    let seen = ui.data_mut(|d| {
        let seen = d.get_temp_mut_or_default::<Option<egui::Id>>(ui.id().with("follow_focus"));
        std::mem::replace(seen, focused)
    });
    // A widget drawn in this area is clipped by this area's view, and no other is: one in
    // another panel, or in a list open over this one, would scroll the area to nothing.
    if focused != seen
        && let Some(widget) = focused.and_then(|id| ui.ctx().read_response(id))
        && widget.layer_id == ui.layer_id()
        && widget.interact_rect == widget.rect.intersect(ui.clip_rect())
    {
        ui.scroll_to_rect(widget.rect, None);
    }
}
