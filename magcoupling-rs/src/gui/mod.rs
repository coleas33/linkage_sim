//! The calculator's egui panel (feature `gui`), hostable by any egui app.
//!
//! [`MagcouplingPanel`] holds the design inputs and results and draws them:
//! the standalone app (feature `app`) shows it as a full page, the linkage app
//! in an `egui::Window` (M5). egui only, no eframe: a host brings its own
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

// Drawn by the panel from Task 5 of this stage's plan, which removes this allow.
#[allow(dead_code)]
pub mod checks;
pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
pub mod diagrams;
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
pub mod torque_card;
pub mod trace;
pub mod typeset;
pub mod xlsx;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{
    CentreView, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX,
    WELCOME,
};
// The Size tab's gap text, public for the slice-3 screen dump (`size_controls` is private).
pub use size_controls::torque_gap_text;
