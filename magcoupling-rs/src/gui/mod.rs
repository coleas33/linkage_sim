//! The calculator's egui panel (feature `gui`), hostable by any egui app.
//!
//! [`MagcouplingPanel`] holds the design inputs and results and draws them:
//! the standalone app (feature `app`) shows it as a full page, the linkage app
//! in an `egui::Window` (M5). egui only, no eframe: a host brings its own
//! window and event loop. The engine stays pure std; nothing here is compiled
//! without the feature.

pub mod clamp_drawing;
pub mod corrections;
pub mod dashboard;
mod format;
pub mod geometry;
pub mod geometry_view;
pub mod history;
pub mod input_ui;
pub mod inputs;
mod panel;
pub mod plots;
pub mod results_table;
pub mod session;
pub mod sizing;
#[cfg(test)]
pub(crate) mod test_support;

pub use format::{SIGNIFICANT_DIGITS, format_value, with_unit};
pub use inputs::KEY_DESIGN;
pub use panel::{CentreView, MagcouplingPanel, PanelRequest, SIZING_LOG_PREFIX};
