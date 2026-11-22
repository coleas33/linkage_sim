//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a full page,
//! for the native binary `magcoupling-app` and the web binary `magcoupling-web` (served at
//! `/tools/magcoupler/`). It applies the linkage app's theme ([`theme`]) and does the panel's
//! platform work ([`files`]): saving files and picking a design file. The web entry also
//! opens a share link's design (`?m=`).

pub mod files;
pub mod theme;

use crate::DesignInputs;
use crate::gui::session::{Design, LoadError, share_link};
use crate::gui::sizing::SizingState;
use crate::gui::{MagcouplingPanel, PanelRequest};

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";

/// Id of the page's canvas element; `linkage-sim-rs/web/tools/magcoupler/index.html`
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// The log line of a share link opened at start-up (the web smoke test looks for it).
pub const SHARE_LINK_LOADED: &str = "magcoupling: loaded the design from the share link";

/// The log line of a design file loaded through the file picker (the web smoke test looks for
/// it: the panel's own "Design file loaded" is drawn on the canvas only).
pub const DESIGN_FILE_LOADED: &str = "magcoupling: loaded a design file";

/// The design the app opens on when the page's address carries no share link (UX overhaul
/// spec 7.1; owner rulings O-8 and O-9). It keeps the example design's given facts (the
/// 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock
/// length on both rings, the sleeve, the liner, the allowances and every assumption) and is
/// sized inside them to pass every check. Seven inputs differ from the spreadsheet's, each on
/// its slider's step.
///
/// Required torque is 1.35 N·m: 0.35 N·m of hot wheel-bearing drag plus 1.00 N·m to accelerate
/// the wheel (the owner's figures; the tool takes the requirement as one number). The design
/// passes every check up to a Required torque of 1.44 N·m.
///
/// The keyway depth is 1.4 mm, the standard hub keyway of a 3 x 3 mm parallel key on a 10 mm
/// shaft (DIN 6885-1 and BS 4235-1: 1.4 mm, +0.1), where the spreadsheet has 1.7 mm (the
/// owner's ruling of 2026-11-22, backlog BL-080; register row A-17 in
/// `docs/analyses/2026-11-10-conservative-assumptions.md`). The hub wall past it is 2.55 mm,
/// so the Size tab's amber line for a wall under about 2.5 mm is not drawn; no check's verdict
/// or margin depends on which of the two depths is in. Its temperature check and the inner
/// hub's fitting temperature use the 3D fields stored for the example design, as every
/// design's do; they were not computed for this geometry (row A-18).
pub fn starting_design() -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = 8;
    inputs.coupling.inner_back_apothem_mm = 9.0;
    inputs.coupling.keyway_depth_mm = 1.4;
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.2;
    inputs.metal.cap_thread_dia_mm = 41.7;
    inputs.metal.required_min_Nm = 1.35;
    inputs
}

/// The page address that stands for `design` (UX overhaul spec 7.2): the clean address, `base`
/// alone, at the starting design with Auto off (what the page opens on without a link); for any
/// other design the share link Share copies. One encoding: [`share_link`].
pub fn address_of(base: &str, design: &Design) -> String {
    let starting = design.sizing == SizingState::default() && design.inputs == starting_design();
    if starting {
        base.to_owned()
    } else {
        share_link(base, design)
    }
}

/// Rewrites the page's address in place with `history.replaceState` (UX overhaul spec 7.2): no
/// history entry is added, so the browser's Back still leaves the tool. `false` when the
/// browser refused the write (Safari refuses more than about 100 calls in 10 seconds): the
/// address is then unchanged.
#[cfg(target_arch = "wasm32")]
fn replace_address(address: &str) -> bool {
    web_sys::window()
        .and_then(|window| window.history().ok())
        .is_some_and(|history| {
            history
                .replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(address))
                .is_ok()
        })
}

/// The native app has no address to write.
#[cfg(not(any(target_arch = "wasm32", test)))]
fn replace_address(_address: &str) -> bool {
    true
}

// The native tests' page: it records each write asked of it and can refuse one.
#[cfg(all(test, not(target_arch = "wasm32")))]
use tests::replace_address;

/// The standalone app: one [`MagcouplingPanel`], full page.
pub struct MagcouplingApp {
    panel: MagcouplingPanel,
    picker: files::DesignPicker,
    /// The design the page's address stands for (UX overhaul spec 7.2): the design the app
    /// opened on, then the last one whose address the page took. Compared with the panel's
    /// settled design every frame, so a frame that changes nothing encodes nothing.
    address_design: Design,
}

impl MagcouplingApp {
    /// The app at the starting design ([`starting_design`], the panel's baseline for the
    /// session), with the linkage app's theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        let panel = MagcouplingPanel::with_baseline(starting_design());
        let address_design = panel.design();
        Self {
            panel,
            picker: files::DesignPicker::default(),
            address_design,
        }
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
            Ok(()) => {
                // The address already carries this design: nothing is written until an edit.
                self.address_design = self.panel.settled_design().clone();
                log::info!(
                    "{SHARE_LINK_LOADED} (sizing: {})",
                    self.panel.sizing().mode.label()
                );
            }
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
        // The address follows the design once an edit has settled (UX overhaul spec 7.2), and
        // waits while a key is held: a held Ctrl+Z or Ctrl+Y settles a step at every key repeat,
        // and a write for each would pass the browser's limit. A refused write is asked again.
        // shortcut: asked again on every later frame with no back-off; fine while a refusal ends
        // within seconds (Safari's window), add a back-off if a browser can refuse for good.
        let key_held = ctx.input(|input| !input.keys_down.is_empty());
        let settled = self.panel.settled_design();
        if !key_held
            && *settled != self.address_design
            && replace_address(&address_of(self.panel.share_base(), settled))
        {
            self.address_design = settled.clone();
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
    use crate::compute_all;
    use crate::engine::meta::{InputSet, Value};
    use crate::gui::dashboard::{Level, warning_lines};
    use crate::gui::guidance::{counted_guides, design_status, level};
    use crate::gui::inputs::InputCatalogue;
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::{drawn_texts, texts_in_colors};

    /// The status line at the starting design and at the example design, with the shaft-clamp
    /// section paused as the app runs it (nine counted checks).
    const STARTING_STATUS: &str = "Starting design: passes all 9 checks";
    const EXAMPLE_STATUS: &str = "Example design: fails 4 of 9 checks";

    /// The seven inputs the starting design sizes (UX overhaul spec 7.1; the keyway depth by the
    /// owner's ruling of 2026-11-22, BL-080): the path, the spreadsheet's value and the starting
    /// design's.
    const STARTING_CHANGES: [(&str, f64, f64); 7] = [
        ("coupling.npole", 10.0, 8.0),
        ("coupling.inner_back_apothem_mm", 10.15, 9.0),
        ("coupling.keyway_depth_mm", 1.7, 1.4),
        ("metal.required_min_Nm", 2.5, 1.35),
        ("metal.face_gap_mm", 1.4, 1.8),
        ("metal.cup_wall_corner_mm", 1.8, 2.2),
        ("metal.cap_thread_dia_mm", 41.0, 41.7),
    ];

    thread_local! {
        /// The addresses this test asked the page to take, in order, refused ones included.
        static WRITES: std::cell::RefCell<Vec<String>> =
            const { std::cell::RefCell::new(Vec::new()) };
        /// Whether the page refuses a write, as Safari does past its rate limit.
        static REFUSES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// The page's address under test, in place of `history.replaceState`: it records the write
    /// asked for and takes it unless `REFUSES` is set.
    pub(super) fn replace_address(address: &str) -> bool {
        WRITES.with_borrow_mut(|writes| writes.push(address.to_owned()));
        !REFUSES.get()
    }

    /// The writes asked of the page since the last call, in order.
    fn take_writes() -> Vec<String> {
        WRITES.take()
    }

    /// The laptop screen the app's tests draw on.
    const LAPTOP: egui::Vec2 = egui::vec2(1280.0, 800.0);

    /// One frame of `app` on a screen of `size` with `events`.
    fn frame_on(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        size: egui::Vec2,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            events,
            ..Default::default()
        };
        ctx.run(input, |ctx| app.ui(ctx))
    }

    /// One frame of `app` on the laptop screen with `events`; the texts it drew.
    fn frame_with(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        events: Vec<egui::Event>,
    ) -> Vec<String> {
        drawn_texts(&frame_on(ctx, app, LAPTOP, events))
    }

    /// One frame of `app` with no input; the texts it drew.
    fn frame_texts(ctx: &egui::Context, app: &mut MagcouplingApp) -> Vec<String> {
        frame_with(ctx, app, Vec::new())
    }

    /// The address the page shows, as the app last wrote it (or opened on).
    fn address(app: &MagcouplingApp) -> String {
        address_of(app.panel.share_base(), &app.address_design)
    }

    /// The design with the face gap at `gap`, as a design file's text.
    fn gap_file(gap: f64) -> String {
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = gap;
        crate::gui::session::design_to_json(&design)
    }

    /// The starting design as a design (Auto off).
    fn starting() -> Design {
        Design {
            inputs: starting_design(),
            sizing: SizingState::default(),
        }
    }

    /// The starting design as slice A shipped it (main 2ca65ee; live from 2026-11-17 until
    /// BL-080): the spreadsheet's keyway depth and the example design's 3D fields.
    fn first_starting_design() -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.coupling.npole = 8;
        inputs.coupling.inner_back_apothem_mm = 9.0;
        inputs.metal.face_gap_mm = 1.8;
        inputs.metal.cup_wall_corner_mm = 2.2;
        inputs.metal.cap_thread_dia_mm = 41.7;
        inputs.metal.required_min_Nm = 1.35;
        inputs
    }

    /// The Size tab's amber line at the first starting design's hub wall.
    const FIRST_KEYWAY_LINE: &str = "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm.";

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
        match inputs.get(path) {
            Some(Value::Num(x)) => x,
            Some(Value::Int(i)) => i as f64,
            other => panic!("{path}: {other:?}"),
        }
    }

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
    }

    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
        // UX overhaul spec 8.1, test 1, pinned to compute_all. If a later physics fix turns
        // this red the cure is a new starting design, never a loosened test (spec 12.2).
        let inputs = starting_design();
        let results = compute_all(&inputs);
        // Nine counted checks while the shaft-clamp section is paused, ten with it shown.
        for (clamps_paused, total) in [(true, 9), (false, 10)] {
            let status = design_status(&inputs, &results, clamps_paused);
            assert_eq!(
                (status.bad, status.caution, status.total),
                (0, 0, total),
                "paused: {clamps_paused}"
            );
            for guide in counted_guides(clamps_paused) {
                assert_eq!(
                    level(guide, &inputs, &results),
                    Level::Good,
                    "{}",
                    guide.label
                );
            }
        }
        let warnings = warning_lines(&results);
        assert!(warnings.is_empty(), "{:?}", warnings[0].1);
        // No drawing note asks for a look (the wall rule's hint has no level).
        let notes = crate::gui::geometry::geometry(&inputs, &results, true).notes;
        assert!(notes.iter().all(|note| note.level.is_none()), "{notes:?}");
        // The numbers the design document and the README quote, at the precision they quote.
        let close = |got: f64, want: f64, tolerance: f64| (got - want).abs() < tolerance;
        let hot_low = results.metal.torque_hot_low_Nm;
        assert!(close(hot_low, 1.448, 0.0005), "hot-low {hot_low}");
        assert!(
            close(hot_low / inputs.metal.required_min_Nm - 1.0, 0.073, 0.0005),
            "the hot-low margin over 1.35 N·m"
        );
        let at_gearbox = results.metal.cold_high_input_Nm;
        assert!(
            close(at_gearbox, 0.510, 0.0005),
            "at the gearbox {at_gearbox}"
        );
        assert_eq!(inputs.coupling.gearbox_input_rating_Nm, 0.6);
        assert!(close(results.model.pullout_Nm, 1.704, 0.0005));
        assert!(close(results.metal.torque_cold_high_Nm, 2.423, 0.0005));
        // It passes up to a Required torque of 1.44 N·m, and not at 1.45.
        for (required, bad) in [(1.44, 0), (1.45, 1)] {
            let mut asked = starting_design();
            asked.metal.required_min_Nm = required;
            let status = design_status(&asked, &compute_all(&asked), true);
            assert_eq!(status.bad, bad, "Required torque {required}");
        }
    }

    #[test]
    fn the_starting_design_differs_from_the_example_on_slider_steps_and_in_no_assumption() {
        // UX overhaul spec 7.1, with the keyway depth of BL-080: seven inputs differ from the
        // spreadsheet's, each one a value a slider step reaches, so a user can walk from one
        // design to the other.
        let (start, example) = (starting_design(), DesignInputs::default());
        let mut differing = Vec::new();
        start.visit("", &mut |path, _, value| {
            if example.get(path) != Some(value) {
                differing.push(path.to_owned());
            }
        });
        differing.sort();
        let mut want = STARTING_CHANGES.map(|(path, _, _)| path.to_owned());
        want.sort();
        assert_eq!(differing, want);
        for (path, from, to) in STARTING_CHANGES {
            assert_eq!(number(&example, path), from, "{path}");
            assert_eq!(number(&start, path), to, "{path}");
            let range = InputCatalogue::get()
                .entry(path)
                .and_then(|entry| entry.meta.range)
                .unwrap_or_else(|| panic!("{path} has a slider"));
            let steps = (to - range.min) / range.step;
            assert!(
                (range.min..=range.max).contains(&to) && (steps - steps.round()).abs() < 1e-9,
                "{path}: {to} is not a step of {range:?}"
            );
        }
        // It changes no assumption, so the assumptions banner is quiet at both designs.
        for inputs in [&start, &example] {
            let modified = crate::engine::assumptions::modified(inputs);
            assert!(modified.is_empty(), "{:?}", modified[0].label);
        }
        // What the status line's hover says the design keeps (STARTING_DESIGN_HOVER): the
        // gearbox rating, the space and the magnet part; and it asks for less torque.
        for path in [
            "coupling.gearbox_input_rating_Nm",
            "metal.max_diameter_mm",
            "metal.max_overall_axial_mm",
            "metal.max_large_dia_axial_mm",
            "coupling.magnets.part_inner",
            "coupling.magnets.part_outer",
            "coupling.magnets.axial_length_mm",
        ] {
            assert!(start.get(path).is_some(), "{path} is an input");
            assert_eq!(start.get(path), example.get(path), "{path}");
        }
        assert!(start.metal.required_min_Nm < example.metal.required_min_Nm);
    }

    #[test]
    fn the_app_opens_the_starting_design() {
        // UX overhaul spec 8.1, test 3: the app hands the panel the starting design as its
        // baseline, so the first view names it and Reset all returns to it.
        use crate::gui::input_ui::CHANGED_DOT;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        assert_eq!(app.panel().inputs(), &starting_design());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        let count = |text: &str| texts.iter().filter(|t| *t == text).count();
        assert_eq!(count(STARTING_STATUS), 1, "{texts:?}");
        assert_eq!(count(CHANGED_DOT), 0, "no input is changed at the baseline");
        assert_eq!(
            count(crate::gui::OPEN_EXAMPLE),
            1,
            "the example design's button"
        );
        // The readouts show the requirement and its margin.
        assert_eq!(count("1.35 N·m"), 1, "{texts:?}");
        assert_eq!(count("margin +0.10 N·m (+7.3 %)"), 1, "{texts:?}");
        // The header takes a second row at this width; the torque card still opens.
        assert_eq!(count(crate::gui::torque_card::CAPTION), 1, "{texts:?}");
        // An edit, then Reset all: the starting design again, not the spreadsheet's.
        app.panel
            .load_design_file(&crate::gui::session::design_to_json(&Design::default()))
            .unwrap();
        app.panel.reset();
        assert_eq!(app.panel().inputs(), &starting_design());
    }

    #[test]
    fn the_starting_design_draws_no_caution_on_its_opening_tab() {
        // The owner's ruling of 2026-11-22 (BL-080; register row A-17). Until then the opening
        // tab drew one amber line, "Wall past keyway: 2.25 mm; existing guidance is about
        // 2.5 mm.", which no counted check judges. The starting design's keyway is now the
        // standard hub keyway of a 3 x 3 mm key on a 10 mm shaft (t2 = 1.4 mm, +0.1; DIN 6885-1
        // and BS 4235-1) in place of the spreadsheet's 1.7 mm, so the wall is 2.55 mm and
        // nothing on the opening tab is drawn in the caution or the failure colour: on the
        // laptop screen, and on a page tall enough to draw both side columns whole.
        let inputs = starting_design();
        assert_eq!(inputs.coupling.keyway_depth_mm, 1.4);
        let wall = compute_all(&inputs).model.hub_wall_past_key_mm;
        assert!((wall - 2.55).abs() < 1e-9, "{wall}");
        for size in [LAPTOP, egui::vec2(LAPTOP.x, 20_000.0)] {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            let output = frame_on(&ctx, &mut app, size, Vec::new());
            let texts = drawn_texts(&output);
            assert!(texts.iter().any(|t| t == STARTING_STATUS), "{texts:?}");
            for start in ["Wall past keyway", "Exploration only"] {
                assert!(!texts.iter().any(|t| t.starts_with(start)), "{texts:?}");
            }
            let visuals = ctx.style().visuals.clone();
            assert_eq!(
                texts_in_colors(&output, &[visuals.warn_fg_color, visuals.error_fg_color]),
                Vec::<String>::new(),
                "at {size:?}"
            );
        }
    }

    #[test]
    fn the_standard_keyway_depth_moves_no_check_of_the_starting_design() {
        // The owner's condition for the 1.4 mm keyway (2026-11-22): no verdict changes. With
        // the spreadsheet's 1.7 mm back in, every counted check has the same level and the same
        // terms with the same margins: the key bears on its 1.2 mm contact height under either
        // depth, and both walls stay closed. What the depth moves is the wall past it, and the
        // amber line with it.
        use crate::gui::guidance::check_terms;
        let start = starting_design();
        let mut deeper = start.clone();
        deeper.coupling.keyway_depth_mm = 1.7;
        let (results, results_deeper) = (compute_all(&start), compute_all(&deeper));
        for guide in counted_guides(false) {
            assert_eq!(
                level(guide, &start, &results),
                level(guide, &deeper, &results_deeper),
                "{}",
                guide.label
            );
            assert_eq!(
                check_terms(guide.check, &start, &results),
                check_terms(guide.check, &deeper, &results_deeper),
                "{}",
                guide.label
            );
        }
        assert!(start.shaft_joint.key_contact_mm < start.coupling.keyway_depth_mm);
        let (wall, wall_deeper) = (
            results.model.hub_wall_past_key_mm,
            results_deeper.model.hub_wall_past_key_mm,
        );
        assert!(wall >= 2.5 && wall_deeper < 2.5, "{wall}, {wall_deeper}");
    }

    #[test]
    fn a_link_and_a_file_saved_from_the_first_starting_design_open_it_as_it_was() {
        // BL-080 changed the starting design. A link and a design file name every input, so one
        // saved before the change opens the design it was saved from, with that design's
        // numbers. `tests/data/saved_designs/` holds the link Share copied and the file Save
        // wrote at the first starting design, both written by the code of main 2ca65ee.
        use crate::gui::session::{
            PUBLIC_BASE_URL, SHARE_PARAM, decode_share_payload, design_from_json,
        };
        let link = include_str!("../tests/data/saved_designs/starting_design_2ca65ee.link.txt");
        let file = include_str!("../tests/data/saved_designs/starting_design_2ca65ee.design.json");
        let payload = link
            .strip_prefix(&format!("{PUBLIC_BASE_URL}?{SHARE_PARAM}="))
            .expect("a share link");
        let first = Design {
            inputs: first_starting_design(),
            sizing: SizingState::default(),
        };
        assert_eq!(decode_share_payload(payload).unwrap(), first);
        assert_eq!(design_from_json(file).unwrap(), first);
        assert!(
            first != starting(),
            "it is not the starting design any more"
        );
        // Its numbers as they were: the wall past the 1.7 mm keyway, and the temperature margin
        // from the example design's 3D fields.
        let results = compute_all(&first.inputs);
        let close = |got: f64, want: f64| (got - want).abs() < 0.005;
        assert!(close(results.model.hub_wall_past_key_mm, 2.25));
        assert!(close(results.metal.torque_hot_low_Nm, 1.448));
        assert!(close(results.temperature.summary.margin_hot_day_C, 28.06));
        // In the app each opens that design, named "Your design" now, with its amber line; the
        // link stays the page's address.
        type Open = fn(&mut MagcouplingApp, &str);
        let opens: [(Open, &str); 2] = [
            (|app, link| app.open_share_payload(link).unwrap(), payload),
            (|app, file| app.open_picked_file(Ok(file.to_owned())), file),
        ];
        for (open, saved) in opens {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            open(&mut app, saved);
            assert_eq!(app.panel().design(), first);
            frame_texts(&ctx, &mut app);
            let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
            let texts = drawn_texts(&output);
            let count = |text: &str| texts.iter().filter(|t| *t == text).count();
            assert_eq!(count("Your design: passes all 9 checks"), 1, "{texts:?}");
            // The amber line is the one text in the caution colour (so the colour probe of
            // the_starting_design_draws_no_caution_on_its_opening_tab does see such a line).
            let visuals = ctx.style().visuals.clone();
            assert_eq!(
                texts_in_colors(&output, &[visuals.warn_fg_color, visuals.error_fg_color]),
                [FIRST_KEYWAY_LINE]
            );
            assert_eq!(app.address_design, first);
        }
    }

    #[test]
    fn the_example_design_opens_as_one_undo_step_from_the_starting_design() {
        // UX overhaul spec 8.1, test 4, on the real starting design (the panel's own tests hold
        // the mechanism on a test baseline).
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        app.panel.open_example_design();
        assert_eq!(app.panel().design(), Design::default());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == EXAMPLE_STATUS), "{texts:?}");
        // One undo step back: the starting design, named so.
        app.panel.undo();
        assert_eq!(app.panel().inputs(), &starting_design());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == STARTING_STATUS), "{texts:?}");
        // Redo, then Reset all: the baseline is still the starting design.
        app.panel.redo();
        assert_eq!(app.panel().design(), Design::default());
        app.panel.reset();
        assert_eq!(app.panel().inputs(), &starting_design());
    }

    #[test]
    fn a_file_that_names_one_input_reads_the_rest_from_the_spreadsheet_not_the_baseline() {
        // UX overhaul spec 7.1: a file or link made before this change opens to the same design
        // as before. An input a file leaves out takes the spreadsheet's value, whatever the
        // panel's baseline (the reader is not changed by this slice; this pins it).
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let file = r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}"#;
        app.open_picked_file(Ok(file.to_owned()));
        let mut want = Design::default();
        want.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(app.panel().design(), want);
    }

    #[test]
    fn the_spreadsheet_export_still_flags_against_the_workbook() {
        // UX overhaul spec 7.1 (C-22) and 8.1, test 4: in the export "changed" means "differs
        // from the workbook", whatever the panel's baseline: every input the starting design
        // changes at the starting design, none at the example design. The export's code is not
        // changed by this slice; this pins it.
        use crate::gui::spreadsheet::{CellValue, INPUT_COLUMNS, INPUTS_SHEET, YES, sheets};
        use crate::gui::test_support::snapshot;
        let column = |name: &str| {
            INPUT_COLUMNS
                .iter()
                .position(|c| *c == name)
                .unwrap_or_else(|| panic!("no column {name}"))
        };
        let (changed, path) = (column("Changed from default"), column("Path"));
        let flagged = |inputs: &DesignInputs| -> Vec<String> {
            let results = compute_all(inputs);
            let book = sheets(&snapshot(inputs, &results));
            let sheet = book
                .iter()
                .find(|sheet| sheet.name == INPUTS_SHEET)
                .expect("the Inputs sheet");
            let mut paths: Vec<String> = sheet
                .rows
                .iter()
                .filter(|row| {
                    row.len() == INPUT_COLUMNS.len()
                        && row[changed].value == CellValue::Text(YES.to_owned())
                })
                .map(|row| match &row[path].value {
                    CellValue::Text(path) => path.clone(),
                    other => panic!("a path cell holds {other:?}"),
                })
                .collect();
            paths.sort();
            paths
        };
        let mut want = STARTING_CHANGES.map(|(path, _, _)| path.to_owned());
        want.sort();
        assert_eq!(flagged(&starting_design()), want);
        assert_eq!(flagged(&DesignInputs::default()), Vec::<String>::new());
    }

    #[test]
    fn the_address_is_clean_at_the_starting_design_and_a_share_link_elsewhere() {
        // UX overhaul spec 7.2 and C-23: one encoding, the link Share copies.
        use crate::gui::session::{PUBLIC_BASE_URL, SHARE_PARAM, decode_share_payload};
        let base = PUBLIC_BASE_URL;
        assert_eq!(address_of(base, &starting()), base);
        // Share still copies the full link at the starting design: a pasted link names every
        // input, so it keeps its meaning if the starting design changes later.
        let ctx = egui::Context::default();
        let app = MagcouplingApp::new(&ctx);
        let prefix = format!("{base}?{SHARE_PARAM}=");
        let shared = app.panel().share_link();
        let payload = shared.strip_prefix(&prefix).expect("a full link");
        assert_eq!(decode_share_payload(payload).unwrap(), starting());
        // Any other design: its share link, which decodes to the design that made it.
        let mut auto_on = starting();
        auto_on.sizing.mode = crate::gui::sizing::SizingMode::TorqueToMagnets;
        let mut edited = starting();
        edited.inputs.metal.face_gap_mm = 1.81;
        for design in [Design::default(), auto_on, edited] {
            let address = address_of(base, &design);
            assert_eq!(address, share_link(base, &design));
            let payload = address.strip_prefix(&prefix).expect("a share link");
            assert_eq!(decode_share_payload(payload).unwrap(), design);
        }
    }

    #[test]
    fn a_link_stays_under_four_thousand_characters() {
        // UX overhaul spec 7.2: a link names every input (decision M41-5), about 2,700
        // characters whatever the design; the ceiling catches a format that grows past what
        // an address bar and a host take.
        let mut long_text = starting();
        long_text.inputs.coupling.magnets.part_inner = "X".repeat(200);
        long_text.inputs.coupling.magnets.part_outer = "Y".repeat(200);
        for design in [starting(), Design::default(), long_text] {
            let link = share_link(crate::gui::session::PUBLIC_BASE_URL, &design);
            assert!(link.len() <= 4_000, "{} characters", link.len());
            assert!(link.len() > 2_000, "{} characters: every input", link.len());
        }
    }

    #[test]
    fn the_address_follows_settled_edits_undo_and_redo() {
        // UX overhaul spec 7.2 and 8.1, test 5. The write itself (history.replaceState) is the
        // browser smoke run's; this holds which design the address stands for.
        use crate::gui::test_support::key_event;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let base = app.panel().share_base().to_owned();
        frame_texts(&ctx, &mut app);
        assert_eq!(
            address(&app),
            base,
            "the starting design: the clean address"
        );
        // An edit in progress (a key held down) writes nothing; its release does.
        app.panel.load_design_file(&gap_file(2.0)).unwrap();
        frame_with(&ctx, &mut app, vec![key_event(egui::Key::F12, true)]);
        assert_eq!(address(&app), base, "the edit has not settled");
        frame_with(&ctx, &mut app, vec![key_event(egui::Key::F12, false)]);
        let edited = app.panel().design();
        assert_eq!(address(&app), share_link(&base, &edited));
        assert_ne!(address(&app), base);
        // Idle frames keep it; undo returns the clean address; redo the link.
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, edited);
        app.panel.undo();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), base);
        app.panel.redo();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), share_link(&base, &edited));
        // The example design is a design like any other: its full link.
        app.panel.open_example_design();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), share_link(&base, &Design::default()));
        // Reset all: the starting design, the clean address again.
        app.panel.reset();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), base);
    }

    #[test]
    fn a_held_undo_or_redo_writes_the_address_once_at_its_release() {
        // Independent review, 2026-11-17: Ctrl+Z and Ctrl+Y act on every key repeat and each
        // step they reach is a settled design, so a write per step sent a write per repeat;
        // Safari refuses more than about 100 in 10 seconds.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let base = app.panel().share_base().to_owned();
        take_writes();
        // Five undo steps, each one settled and written on its own frame.
        let gaps = [1.5, 1.6, 1.7, 1.9, 2.0];
        for gap in gaps {
            app.panel.load_design_file(&gap_file(gap)).unwrap();
            frame_texts(&ctx, &mut app);
        }
        let last = app.panel().design();
        assert_eq!(
            take_writes().len(),
            gaps.len(),
            "one write per settled edit"
        );
        let key_event = |key, pressed, repeat| egui::Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat,
            modifiers: egui::Modifiers::COMMAND,
        };
        // Ctrl+Z held for 60 frames runs back to the starting design, Ctrl+Y forward again.
        for (key, reached) in [(egui::Key::Z, starting()), (egui::Key::Y, last)] {
            for frame in 0..60 {
                frame_with(&ctx, &mut app, vec![key_event(key, true, frame > 0)]);
            }
            assert_eq!(
                app.panel().design(),
                reached,
                "{key:?} ran through every step"
            );
            assert_eq!(
                take_writes(),
                Vec::<String>::new(),
                "{key:?}: no write while the key is held"
            );
            frame_with(&ctx, &mut app, vec![key_event(key, false, false)]);
            assert_eq!(
                take_writes(),
                [address_of(&base, &reached)],
                "{key:?}: one write at its release"
            );
        }
    }

    #[test]
    fn a_refused_address_write_is_tried_again_on_the_next_frame() {
        // Independent review, 2026-11-17: a write the browser refuses leaves the address as it
        // was, so the app must not count it as written.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let base = app.panel().share_base().to_owned();
        frame_texts(&ctx, &mut app);
        take_writes();
        REFUSES.set(true);
        app.panel.load_design_file(&gap_file(2.0)).unwrap();
        frame_texts(&ctx, &mut app);
        REFUSES.set(false);
        let edited = app.panel().design();
        let link = share_link(&base, &edited);
        assert_eq!(
            take_writes(),
            std::slice::from_ref(&link),
            "the write was asked for"
        );
        assert_eq!(
            app.address_design,
            starting(),
            "the address still stands for the last design it took"
        );
        // The next frame asks again; once the page takes the write, no frame asks a third time.
        frame_texts(&ctx, &mut app);
        assert_eq!(take_writes(), [link]);
        assert_eq!(app.address_design, edited);
        frame_texts(&ctx, &mut app);
        assert_eq!(take_writes(), Vec::<String>::new());
    }

    #[test]
    fn a_link_cut_at_two_thousand_characters_is_refused_and_the_header_says_why() {
        // A link is about 2,700 characters, and a host or a message app may cut a long address.
        // Whatever part of a base64 group the cut leaves (four cut lengths, one per remainder),
        // the link is refused: the design stays and the header shows the reader's reason.
        use crate::gui::session::{PUBLIC_BASE_URL, SHARE_PARAM};
        const NOT_BASE64: &str = "share link refused: not URL-safe base64 (";
        const NOT_COMPRESSED: &str = "share link refused: not a compressed design (";
        let link = share_link(PUBLIC_BASE_URL, &Design::default());
        let prefix = format!("{PUBLIC_BASE_URL}?{SHARE_PARAM}=");
        for length in [2_000, 1_999, 1_998, 1_997] {
            let payload = link[..length].strip_prefix(&prefix).expect("a share link");
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            let reason = app
                .open_share_payload(payload)
                .expect_err("a cut link is refused")
                .to_string();
            let wanted: &[&str] = match payload.len() % 4 {
                // Whole groups decode, and the compressed design they hold stops short.
                0 => &[NOT_COMPRESSED],
                // One symbol of a group is no base64.
                1 => &[NOT_BASE64],
                // Two or three symbols decode only when their spare bits are zero.
                _ => &[NOT_BASE64, NOT_COMPRESSED],
            };
            assert!(
                wanted.iter().any(|start| reason.starts_with(start)),
                "{length}: {reason}"
            );
            assert_eq!(
                app.panel().design(),
                starting(),
                "{length}: nothing changes"
            );
            frame_texts(&ctx, &mut app);
            let texts = frame_texts(&ctx, &mut app);
            assert!(
                texts.contains(&reason),
                "{length}: the header says why ({reason}): {texts:?}"
            );
            assert!(texts.iter().any(|t| t == STARTING_STATUS), "{length}");
            assert_eq!(
                app.address_design,
                starting(),
                "{length}: nothing is written"
            );
        }
    }

    #[test]
    fn a_link_opened_at_start_up_is_the_address_until_the_first_edit() {
        // UX overhaul spec 7.2: a link's address is not rewritten when it opens; a link with
        // the spreadsheet's values opens as the example design; a link that cannot be read
        // changes nothing, says why and leaves the address until the first edit.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        app.open_share_payload(&encode_share_payload(&Design::default()))
            .unwrap();
        assert_eq!(app.address_design, Design::default(), "no write is pending");
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == EXAMPLE_STATUS), "{texts:?}");
        assert_eq!(app.address_design, Design::default());
        // The baseline is still the starting design: Reset all returns to it.
        app.panel.reset();
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, starting());

        let mut app = MagcouplingApp::new(&ctx);
        assert!(app.open_share_payload("broken").is_err());
        assert_eq!(
            app.panel().design(),
            starting(),
            "a bad link changes nothing"
        );
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(
            texts.iter().any(|t| t.contains("not URL-safe base64")),
            "the header says why: {texts:?}"
        );
        assert_eq!(app.address_design, starting(), "nothing is written");
        // The first edit writes the address.
        app.panel.load_design_file(&gap_file(2.0)).unwrap();
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, app.panel().design());
        assert_ne!(app.address_design, starting());
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
        // .claude/workflows/gui-smoke.js opens /tools/magcoupler/ with this link, loads the design
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
        // The equation registry's log line, built when the panel starts.
        assert!(script.contains(crate::gui::readouts::REGISTRY_LOG_PREFIX));
        // The header's Show spreadsheet details toggle (spec S2-5), which the first view shows.
        assert!(
            script.contains(crate::gui::SHOW_DETAILS),
            "the smoke prompt names the header's toggle"
        );
        // Spec R14 (guidance slice 2): the steps name the welcome strip, a note's What to change
        // and its sources fold, the Geometry tab and the dashboard's last row as the page draws
        // them, so the prompt cannot go stale unnoticed.
        assert!(script.contains(crate::gui::WELCOME), "the welcome strip");
        assert!(
            script.contains(crate::gui::explorer::WHAT_TO_CHANGE),
            "a note's What to change"
        );
        assert!(
            script.contains(crate::gui::explorer::WHERE_FROM),
            "a note's sources fold"
        );
        assert!(
            script.contains(crate::gui::CentreView::Geometry.label()),
            "the Geometry tab"
        );
        let fit =
            crate::gui::guidance::guide(crate::gui::guidance::FIT_CHECK).expect("the fit guide");
        assert!(script.contains(fit.label), "the dashboard's last row");
        // Slice 3 (spec 8.6, test 26): the card step quotes the torque card whole as the page
        // draws it at the smoke design (Auto on the magnet length): the heading, the caption, the
        // Auto line and the held row; it names the other three rows by their starts. The G-3
        // review (plan T7) set the words before the prompt quoted them; this keeps it current.
        {
            use crate::gui::torque_card::{
                AUTO_HOLDS_LENGTH, CAPTION, CARD_HEADING, Control, held_line,
            };
            // The first view and the card step both look for the heading, so the script types it
            // once, as its own CARD_HEADING: a change of the heading reaches both steps.
            assert_eq!(
                quoted("CARD_HEADING"),
                CARD_HEADING,
                "the script's CARD_HEADING"
            );
            assert_eq!(
                script.matches(CARD_HEADING).count(),
                1,
                "the smoke prompt types the card's heading once, in CARD_HEADING"
            );
            for (what, text) in [
                ("the card's heading", CARD_HEADING.to_owned()),
                ("the card's caption", CAPTION.to_owned()),
                ("the card's Auto line", AUTO_HOLDS_LENGTH.to_owned()),
                ("the held row", held_line(Control::OverallLength)),
            ] {
                assert!(
                    script.contains(&text),
                    "the smoke prompt quotes {what} whole: {text}"
                );
            }
            for control in [
                Control::OutsideDiameter,
                Control::FaceGap,
                Control::MagnetsPerRing,
            ] {
                let start = format!("{}:", control.label());
                assert!(
                    script.contains(&start),
                    "the smoke prompt names the row starting {start}"
                );
            }
        }
        // The drag's timer keeps only the frames the app draws (performance review 4.1): eframe
        // asks for an animation frame at every display refresh, so the timer marks a frame by its
        // WebGL draw call. Read on the constant's own line, since the comment above it names
        // drawElements too; that line is the whole timer, as T10's extraction and the stub read
        // it. The card step installs this constant, not a timer of its own.
        let timer = script
            .lines()
            .find(|line| line.starts_with("const FRAME_TIMER = `"))
            .expect("the smoke prompt's frame timer");
        assert!(
            timer.ends_with('`'),
            "the smoke prompt's frame timer is one line"
        );
        assert!(
            timer.contains("drawElements") && timer.contains("if (window.__drew)"),
            "the smoke prompt's frame timer keeps only the frames the app draws"
        );
        assert!(
            script.contains("page.addInitScript(${FRAME_TIMER})"),
            "the card step installs the frame timer before it opens the link again"
        );
        // The drag step finds the Magnet face gap slider by the value its box shows, and judges
        // drag_moved against it: the link's gap at the slider's decimals (its step's, which egui
        // shows at a device pixel ratio of 1) with its unit, typed once as SMOKE_GAP_SHOWN.
        {
            use crate::gui::guidance::FACE_GAP;
            use crate::gui::inputs::{InputCatalogue, slider_range, step_decimals};
            let unit = InputCatalogue::get()
                .entry(FACE_GAP)
                .expect("the gap input")
                .meta
                .unit;
            let step = slider_range(FACE_GAP).expect("the gap slider's range").step;
            let shown = crate::gui::with_unit(
                format!(
                    "{:.*}",
                    step_decimals(step),
                    design.inputs.metal.face_gap_mm
                ),
                unit,
            );
            assert_eq!(
                quoted("SMOKE_GAP_SHOWN"),
                shown,
                "the script's SMOKE_GAP_SHOWN"
            );
            assert_eq!(
                script.matches(shown.as_str()).count(),
                1,
                "the smoke prompt types the gap's value once, in SMOKE_GAP_SHOWN"
            );
        }
        // The drag counts the sizing solve's console lines (sizing_lines, expected 0) by the
        // panel's own prefix, in its code and in the prompt's words.
        for (what, text) in [
            (
                "the drag's code",
                format!("m.text().includes('{SIZING_LOG_PREFIX}')"),
            ),
            (
                "the prompt's words",
                format!("the number of \"{SIZING_LOG_PREFIX}\" console lines"),
            ),
        ] {
            assert!(
                script.contains(&text),
                "the sizing lines are counted by SIZING_LOG_PREFIX in {what}: {text}"
            );
        }
        // Every step's console messages reach console_errors. The Playwright MCP's
        // browser_console_messages returns only the messages since the last navigation, so each
        // page is collected after its last step: the first page after the scroll and before the
        // link is opened again, the page opened again after the drag and before the close.
        let collect = "Collect the console messages at level \"info\"";
        let between = |from: &str, to: &str| -> &str {
            let start = script.find(from).unwrap_or_else(|| panic!("no {from}"));
            let end = script.find(to).unwrap_or_else(|| panic!("no {to}"));
            assert!(start < end, "the smoke prompt has {from} before {to}");
            &script[start..end]
        };
        assert!(
            between("dashboard_scrolls=true", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
        );
        assert!(
            between("drag_moved=true", "then close the browser").contains(collect),
            "the console messages of the page opened again are collected after the drag"
        );
        // UX overhaul slice A (spec 7.1, 7.2 and 8.1 test 5): the clean address opens the
        // starting design, whose status line the script types once as STARTING_STATUS; the
        // step names the example design's button; and after the drag the address step reloads
        // the page. Each of the two new pages has its console messages collected.
        {
            use crate::gui::guidance::status_line;
            let start = starting_design();
            let status = design_status(&start, &compute_all(&start), crate::engine::clamps::PAUSED);
            let (line, _) = status_line(&status, &start, &start);
            assert_eq!(
                quoted("STARTING_STATUS"),
                line,
                "the script's STARTING_STATUS"
            );
            assert_eq!(
                script.matches(line.as_str()).count(),
                1,
                "the smoke prompt types the starting design's status once, in STARTING_STATUS"
            );
            assert!(
                script.contains(crate::gui::OPEN_EXAMPLE),
                "the smoke prompt names the example design's button"
            );
            assert!(
                between("starting_design=true", "it carries a share link").contains(collect),
                "the clean page's console messages are collected before the link is opened"
            );
            assert!(
                script.contains("await page.reload()"),
                "the address step reloads the page"
            );
            assert!(
                between("address_follows=true", "then close the browser").contains(collect),
                "the reloaded page's console messages are collected before the close"
            );
        }
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
        let page = include_str!("../../linkage-sim-rs/web/tools/magcoupler/index.html");
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
