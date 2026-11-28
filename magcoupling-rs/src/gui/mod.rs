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

/// The ids of the Tab stops the page draws no focus on (BL-102): egui registers each scroll
/// area's drag surface and its two scroll bars, and each resizable panel's edge, as widgets
/// the keyboard can reach, and none of them shows it. Kept in the context's temporary data.
fn hidden_stops(ctx: &egui::Context) -> std::collections::HashSet<egui::Id> {
    ctx.data_mut(|d| d.get_temp::<std::collections::HashSet<egui::Id>>(egui::Id::NULL))
        .unwrap_or_default()
}

fn hide_stops(ctx: &egui::Context, ids: impl IntoIterator<Item = egui::Id>) {
    ctx.data_mut(|d| {
        d.get_temp_mut_or_default::<std::collections::HashSet<egui::Id>>(egui::Id::NULL)
            .extend(ids)
    });
}

/// An up-and-down scroll area salted `id_salt` (`.scroll([..])` changes its axes), its drag surface and scroll bars among the stops [`TabWalk`] passes
/// (egui's ids: `id.with("area")` for the surface, `id.with(0)` and `id.with(1)` for the bars;
/// the surface stays, as a touch screen drags the content by it). Every scroll area of the page
/// is made through this.
pub(crate) fn scroll_area(ui: &egui::Ui, id_salt: &str) -> egui::ScrollArea {
    let id = ui.make_persistent_id(egui::Id::new(id_salt));
    hide_stops(
        ui.ctx(),
        [id.with("area"), id.with(0_usize), id.with(1_usize)],
    );
    egui::ScrollArea::vertical().id_salt(id_salt)
}

/// The edge of the resizable panel `panel_id` (egui's `id.with("__resize")`), a stop
/// [`TabWalk`] passes.
pub(crate) fn hide_resize_edge(ctx: &egui::Context, panel_id: &str) {
    hide_stops(ctx, [egui::Id::new(panel_id).with("__resize")]);
}

/// The Tab key's walk, which passes the stops that show no focus (BL-102). egui hands the focus
/// to the next widget it registered, so a stop that shows nothing is reached and left by a press
/// that seems to do nothing. When a Tab (or Shift+Tab) press has put the focus on one, the focus
/// moves on at the end of the frame to the next (or the previous) widget egui would stop on, in
/// the order it registered them; past the last, the focus goes to none, as egui's does. Shift+Tab
/// puts the focus on its widget as the next frame begins, so the press before counts too. A click
/// or a drag on a surface arrives with no press and keeps the focus where egui puts it.
#[derive(Default)]
pub(crate) struct TabWalk {
    /// The focus at the end of the last frame.
    seen: Option<egui::Id>,
    /// Whether the last frame had a Tab press, and whether it held Shift.
    tabbed: Option<bool>,
}

impl TabWalk {
    /// Called at the end of the frame, once every widget is registered.
    pub(crate) fn pass_hidden_stops(&mut self, ctx: &egui::Context) {
        // The key event's own modifiers, as egui's focus reads them.
        let tab = ctx.input(|i| {
            i.events.iter().find_map(|event| match event {
                egui::Event::Key {
                    key: egui::Key::Tab,
                    pressed: true,
                    modifiers,
                    ..
                } => Some(modifiers.shift),
                _ => None,
            })
        });
        let walking = tab.or(self.tabbed);
        self.tabbed = tab;
        let mut focus = ctx.memory(|m| m.focused());
        if let (Some(id), Some(back)) = (focus, walking)
            && focus != self.seen
            && hidden_stops(ctx).contains(&id)
        {
            focus = next_stop(ctx, id, back);
            ctx.memory_mut(|m| match focus {
                Some(next) => m.request_focus(next),
                None => m.surrender_focus(id),
            });
        }
        self.seen = focus;
    }
}

/// The widget after `from` (before it, going `back`) that egui stops on and the page shows a
/// focus on, in registration order within `from`'s layer.
fn next_stop(ctx: &egui::Context, from: egui::Id, back: bool) -> Option<egui::Id> {
    let hidden = hidden_stops(ctx);
    let stops: Vec<egui::Id> = ctx.viewport(|v| {
        let widgets = &v.this_pass.widgets;
        let Some(layer) = widgets.get(from).map(|w| w.layer_id) else {
            return Vec::new();
        };
        widgets
            .get_layer(layer)
            .filter(|w| w.enabled && w.sense.is_focusable())
            .map(|w| w.id)
            .collect()
    });
    let at = stops.iter().position(|id| *id == from)?;
    let usable = |id: &&egui::Id| !hidden.contains(id);
    if back {
        stops[..at].iter().rev().find(usable).copied()
    } else {
        stops[at + 1..].iter().find(usable).copied()
    }
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
    // A stop that shows no focus (BL-102) is passed on within the frame: no scroll to it.
    let hidden = focused.is_some_and(|id| hidden_stops(ui.ctx()).contains(&id));
    if focused != seen
        && !hidden
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
    let (inner, followed) = scroll_area(ui, id_salt)
        .scroll([true, false])
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
