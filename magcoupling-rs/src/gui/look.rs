//! The panel's own part of the look (UX overhaul spec 6.3). The panel takes its colours from
//! the host's `Visuals`, and an egui theme has no field for a brand accent or for a passing
//! check: the panel owns those two, at the values of the spec's token table (6.1). The
//! standalone app's theme (`app::theme`) lists them in its table under the same names.

/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
/// check: red means a failing check and nothing else.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0xc8, 0x10, 0x2e);

/// A passing check, in a dark theme.
pub const OK: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xc3, 0x8a);
