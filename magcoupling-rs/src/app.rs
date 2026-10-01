//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a
//! full page, for the native binary `magcoupling-app` and the web binary
//! `magcoupling-web` (served at `/magcoupling/`).

use crate::gui::MagcouplingPanel;

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";

/// Id of the page's canvas element; `linkage-sim-rs/web/magcoupling/index.html`
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// The standalone app: one [`MagcouplingPanel`], full page.
#[derive(Default)]
pub struct MagcouplingApp {
    panel: MagcouplingPanel,
}

impl MagcouplingApp {
    /// Draws one frame: the panel in a scrolling central panel.
    pub fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.panel.ui(ui));
        });
    }
}

impl eframe::App for MagcouplingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ui(ctx);
    }
}

/// Opens the app in a native window; both binaries use it on the desktop.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_native() -> eframe::Result<()> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title(TITLE),
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
        options,
        Box::new(|_cc| Ok(Box::new(MagcouplingApp::default()))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::Value;
    use crate::gui::test_support::drawn_texts;
    use crate::{DesignInputs, compute_all, headline};

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::default();
        let texts = drawn_texts(&ctx.run(egui::RawInput::default(), |ctx| app.ui(ctx)));
        assert!(texts.iter().any(|t| t == "Magnetic coupling calculator"), "{texts:?}");
        // The clamp screw is the last headline row: the whole panel was laid out.
        let (key, screw) = headline(&compute_all(&DesignInputs::default())).pop().expect("15 rows");
        let Value::Text(screw) = screw else { panic!("{key}: {screw:?}") };
        assert!(texts.contains(&screw), "missing {screw:?} in {texts:?}");
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
        let page = include_str!("../../linkage-sim-rs/web/magcoupling/index.html");
        assert!(page.contains(&format!("id=\"{CANVAS_ID}\"")), "index.html has no canvas {CANVAS_ID}");
        assert!(page.contains(&format!("<title>{TITLE}</title>")), "index.html title");
    }
}
