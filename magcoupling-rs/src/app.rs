//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a full page,
//! for the native binary `magcoupling-app` and the web binary `magcoupling-web` (served at
//! `/magcoupling/`). It applies the linkage app's theme ([`theme`]) and does the panel's
//! platform work ([`files`]): saving files and picking a design file. The web entry also
//! opens a share link's design (`?m=`).

pub mod files;
pub mod theme;

use crate::gui::session::LoadError;
use crate::gui::{MagcouplingPanel, PanelRequest};

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";

/// Id of the page's canvas element; `linkage-sim-rs/web/magcoupling/index.html`
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// The log line of a share link opened at start-up (the web smoke test looks for it).
pub const SHARE_LINK_LOADED: &str = "magcoupling: loaded the design from the share link";

/// The log line of a design file loaded through the file picker (the web smoke test looks for
/// it: the panel's own "Design file loaded" is drawn on the canvas only).
pub const DESIGN_FILE_LOADED: &str = "magcoupling: loaded a design file";

/// The standalone app: one [`MagcouplingPanel`], full page.
#[derive(Default)]
pub struct MagcouplingApp {
    panel: MagcouplingPanel,
    picker: files::DesignPicker,
}

impl MagcouplingApp {
    /// The app at the default design, with the linkage app's theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        Self::default()
    }

    /// The panel.
    pub fn panel(&self) -> &MagcouplingPanel {
        &self.panel
    }

    /// Sets the address share links point at (the web page's own address).
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        self.panel.set_share_base(base);
    }

    /// Opens the design of a share link's `?m=` value as the session's start (no undo step:
    /// the first Undo keeps the shared design), and logs the outcome.
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        let outcome = self.panel.open_share_payload(payload);
        match &outcome {
            Ok(()) => log::info!(
                "{SHARE_LINK_LOADED} (sizing: {})",
                self.panel.sizing().mode.label()
            ),
            Err(error) => log::warn!("magcoupling: {error}"),
        }
        outcome
    }

    /// Draws one frame: the panel fills the window (its sides scroll on their own); then the
    /// panel's requests are done.
    pub fn ui(&mut self, ctx: &egui::Context) {
        if let Some(picked) = self.picker.take() {
            self.open_picked_file(picked);
        }
        egui::CentralPanel::default().show(ctx, |ui| self.panel.ui(ui));
        for request in self.panel.take_requests() {
            match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    if let Some(outcome) = files::save(&file_name, mime, &contents) {
                        self.panel.report(outcome);
                    }
                }
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
    }
}

impl MagcouplingApp {
    /// Hands a picked design file's text to the panel (a refusal shows there) and logs the
    /// outcome; a file that could not be read is reported.
    fn open_picked_file(&mut self, picked: Result<String, String>) {
        match picked {
            Ok(text) => match self.panel.load_design_file(&text) {
                Ok(()) => log::info!("{DESIGN_FILE_LOADED}"),
                Err(error) => log::warn!("magcoupling: {error}"),
            },
            Err(error) => self.panel.report(Err(error)),
        }
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
            .with_inner_size([1280.0, 800.0])
            .with_title(TITLE),
        ..Default::default()
    };
    eframe::run_native(
        TITLE,
        options,
        Box::new(|cc| Ok(Box::new(MagcouplingApp::new(&cc.egui_ctx)))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::Value;
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::drawn_texts;
    use crate::{DesignInputs, compute_all, headline};

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input(), |ctx| app.ui(ctx));
        let texts = drawn_texts(&ctx.run(input(), |ctx| app.ui(ctx)));
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        // The clamp screw is the last headline row: the dashboard was laid out.
        let (key, screw) = headline(&compute_all(&DesignInputs::default()))
            .pop()
            .expect("15 rows");
        let Value::Text(screw) = screw else {
            panic!("{key}: {screw:?}")
        };
        assert!(texts.contains(&screw), "missing {screw:?} in {texts:?}");
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
    }

    #[test]
    fn a_share_link_opens_its_design() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 2.0;
        app.open_share_payload(&encode_share_payload(&design))
            .unwrap();
        assert_eq!(app.panel().design(), design);
        // The session starts there: the first Ctrl+Z keeps the shared design.
        for pressed in [true, false] {
            let undo = egui::Event::Key {
                key: egui::Key::Z,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            };
            let input = egui::RawInput {
                events: vec![undo],
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| app.ui(ctx));
        }
        assert_eq!(app.panel().design(), design);
        assert!(app.open_share_payload("broken").is_err());
        assert_eq!(
            app.panel().design(),
            design,
            "a broken link changes nothing"
        );
    }

    #[test]
    fn a_picked_design_file_loads_and_a_refused_one_changes_nothing() {
        use crate::gui::session::design_to_json;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 2.0;
        app.open_picked_file(Ok(design_to_json(&design)));
        assert_eq!(app.panel().design(), design);
        app.open_picked_file(Ok("not a design".to_owned()));
        assert_eq!(app.panel().design(), design);
        app.open_picked_file(Err("the file is not UTF-8 text".to_owned()));
        assert_eq!(app.panel().design(), design);
    }

    #[test]
    fn the_smoke_test_share_link_opens_its_design() {
        use crate::engine::sizing::FreeVariable;
        use crate::gui::SIZING_LOG_PREFIX;
        use crate::gui::session::{decode_share_payload, design_from_json};
        use crate::gui::sizing::{SOLVED_PREFIX, SizingMode, SizingState};
        // .claude/workflows/gui-smoke.js opens /magcoupling/ with this link, loads the design
        // file through the picker and looks for the three log lines in the browser console.
        let script = include_str!("../../.claude/workflows/gui-smoke.js");
        let quoted = |name: &str| -> &str {
            let marker = format!("const {name} = '");
            let start = script.find(&marker).unwrap_or_else(|| panic!("no {name}")) + marker.len();
            let end = start + script[start..].find('\'').expect("its closing quote");
            &script[start..end]
        };
        let payload = quoted("MAGCOUPLING_SMOKE_PAYLOAD");
        let design = decode_share_payload(payload).expect("a valid share link");
        let mut want = Design::default();
        want.inputs.metal.face_gap_mm = 1.5;
        want.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
            target_Nm: 2.5,
        };
        assert_eq!(design, want);
        assert!(script.contains(SHARE_LINK_LOADED));
        let solved = format!("{SIZING_LOG_PREFIX}{}", SOLVED_PREFIX.trim_end());
        assert!(script.contains(&solved), "{solved}");
        let file =
            design_from_json(quoted("MAGCOUPLING_SMOKE_DESIGN_FILE")).expect("a design file");
        let mut want_file = Design::default();
        want_file.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(file, want_file);
        assert!(script.contains(DESIGN_FILE_LOADED));
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        app.open_share_payload(payload).unwrap();
        for time in [0.0, 0.5, 0.6] {
            let input = egui::RawInput {
                time: Some(time),
                ..Default::default()
            };
            let _ = ctx.run(input, |ctx| app.ui(ctx));
        }
        let shown = app.panel().shown_inputs();
        let length = shown.coupling.magnets.axial_length_mm.expect("sized");
        assert!((14.0..14.5).contains(&length), "{length}");
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
        let page = include_str!("../../linkage-sim-rs/web/magcoupling/index.html");
        assert!(
            page.contains(&format!("id=\"{CANVAS_ID}\"")),
            "index.html has no canvas {CANVAS_ID}"
        );
        assert!(
            page.contains(&format!("<title>{TITLE}</title>")),
            "index.html title"
        );
    }
}
