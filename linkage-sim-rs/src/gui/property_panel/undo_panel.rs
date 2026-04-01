//! Undo History section of the property panel.
//!
//! A collapsible section showing the undo/redo stack as a visual timeline
//! with numbered entries and undo/redo buttons.

use eframe::egui;
use crate::gui::state::AppState;
use super::pending_edits::PendingPropertyEdit;

/// Draw the "Undo History" collapsible section.
///
/// Shows the current position in the undo stack as a numbered timeline.
/// Undo/Redo buttons use the `PendingPropertyEdit` pattern to avoid
/// borrow conflicts (the property panel takes `&AppState` for reads,
/// mutations are deferred).
pub(super) fn draw_undo_section(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let undo_count = state.undo_history.undo_count();
    let redo_count = state.undo_history.redo_count();
    let total = undo_count + redo_count + 1; // +1 for current state

    let undo_color = state.nc(egui::Color32::from_rgb(160, 140, 200));
    egui::CollapsingHeader::new(
        egui::RichText::new("Undo History").color(undo_color),
    )
    .id_salt("undo_history")
    .default_open(false)
    .show(ui, |ui| {
        ui.label(format!(
            "{} undo / {} redo available",
            undo_count, redo_count,
        ));

        // Visual timeline -- show entries as a compact list.
        // Limit display to avoid very long lists for deep undo stacks.
        let max_display = 20;
        let skip = if total > max_display {
            total - max_display
        } else {
            0
        };

        if skip > 0 {
            ui.label(format!("  ... {} older entries", skip));
        }

        for i in skip..total {
            let is_current = i == undo_count;
            let label = if is_current {
                egui::RichText::new(format!("  [{:>2}] Current", i + 1)).strong()
            } else if i < undo_count {
                egui::RichText::new(format!("   {:>2}  (undo)", i + 1)).weak()
            } else {
                egui::RichText::new(format!("   {:>2}  (redo)", i + 1)).weak()
            };
            ui.label(label);
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(state.can_undo(), egui::Button::new("Undo"))
                .clicked()
            {
                *pending = Some(PendingPropertyEdit::Undo);
            }
            if ui
                .add_enabled(state.can_redo(), egui::Button::new("Redo"))
                .clicked()
            {
                *pending = Some(PendingPropertyEdit::Redo);
            }
        });
    });
}
