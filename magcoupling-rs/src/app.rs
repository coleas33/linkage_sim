//! The standalone app (feature `app`): [`MagcouplingApp`] shows the panel as a full page,
//! for the native binary `magcoupling-app` and the web binary `magcoupling-web` (served at
//! `/tools/magcoupler/`). It applies the calculator's theme ([`theme`]) and font ([`fonts`])
//! and does the panel's platform work ([`files`]): saving files and picking a design file.
//! The web entry also opens a share link's design (`?m=`).

pub mod files;
pub mod fonts;
pub mod theme;

use crate::DesignInputs;
use crate::engine::meta::{InputSet, Value};
use crate::gui::dashboard::{STARTING_3D_FIELDS, STORED_3D_INPUTS};
use crate::gui::session::{Design, LoadError, share_link};
use crate::gui::sizing::SizingState;
use crate::gui::{MagcouplingPanel, PanelRequest};

/// Window and page title.
pub const TITLE: &str = "Magnetic Coupling Calculator";

/// Id of the page's canvas element; `linkage-sim-rs/web/tools/magcoupler/index.html`
/// must use the same.
pub const CANVAS_ID: &str = "magcoupling_canvas";

/// Id of the page's loading line in `linkage-sim-rs/web/tools/magcoupler/index.html`, which
/// the app removes when it draws its first frame.
pub const LOADING_ID: &str = "magcoupling_loading";

/// The page's loading line, in the muted colour on the page's charcoal while the wasm file
/// loads (the stage 1 review, S-4): the one text of the page outside the canvas.
pub const LOADING_LINE: &str = "Loading the calculator";

/// The log line of a share link opened at start-up (the web smoke test looks for it).
pub const SHARE_LINK_LOADED: &str = "magcoupling: loaded the design from the share link";

/// The log line of a design file loaded through the file picker (the web smoke test looks for
/// it: the panel's own "Design file loaded" is drawn on the canvas only).
pub const DESIGN_FILE_LOADED: &str = "magcoupling: loaded a design file";

/// The design the app opens on when the page's address carries no share link (UX overhaul
/// spec 7.1; owner rulings O-8 and O-9). It keeps the example design's given facts (the
/// 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock
/// length on both rings, the sleeve, the liner, the allowances and every assumption) and is
/// sized inside them to pass every check. The seven inputs that size it and its fourteen 3D
/// fields differ from the spreadsheet's, each on its slider's step.
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
/// or margin depends on which of the two depths is in.
///
/// Its temperature check and the inner hub's fitting temperature read 3D fields computed for
/// this geometry ([`STARTING_3D_FIELDS`]: the reference routine `fields3d.run`, run once; the
/// owner's ruling of 2026-11-22, row A-18), where every other design the spreadsheet's values
/// start reads the example design's. The tool cannot compute them, so they are fixed numbers
/// like the example design's: they do not follow an edit, and a change to the geometry above
/// needs a new run (the test that pins them says how). The keyway is no part of the field
/// model: the routine reads the magnets, the bondlines, the steel surfaces behind the magnets,
/// the sleeve, the liner and the cap.
pub fn starting_design() -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = 8;
    inputs.coupling.inner_back_apothem_mm = 9.0;
    inputs.coupling.keyway_depth_mm = 1.4;
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.2;
    inputs.metal.cap_thread_dia_mm = 41.7;
    inputs.metal.required_min_Nm = 1.35;
    for (path, field) in STORED_3D_INPUTS.into_iter().zip(STARTING_3D_FIELDS) {
        inputs
            .set(path, Value::Num(field))
            .expect("every stored 3D field is a number input");
    }
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
    /// session), with the theme and the font applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        fonts::install(ctx);
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
        #[cfg(target_arch = "wasm32")]
        remove_loading_line();
    }
}

/// Removes the page's loading line ([`LOADING_ID`]) at the end of a frame: eframe paints the
/// frame in the same task, so the browser shows the app's first frame and the line's removal in
/// one paint. (Removed by the page's script when the module had loaded, the line left a blank
/// frame or two before the app's first.) After the first frame there is no line to find.
#[cfg(target_arch = "wasm32")]
fn remove_loading_line() {
    if let Some(line) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(LOADING_ID))
    {
        line.remove();
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
    use crate::gui::dashboard::{Level, warning_lines};
    use crate::gui::guidance::{counted_guides, design_status, level};
    use crate::gui::inputs::InputCatalogue;
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::{drawn_texts, side_rect, texts_in_colors};

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

    /// A run of the reference 3D routine (`fields3d.run`) for the starting design's geometry
    /// (BL-080): the output of `reference/magcoupling-py/tools/starting_design_fields.py` on
    /// 2026-11-22 (magpylib 5.2.3). `RECORDED_READS` is its `reads` list: everything the routine
    /// read, by this engine's paths. `RECORDED_FIELDS` is its `raw` list: the fourteen fields at
    /// full precision, in `STORED_3D_INPUTS`' order.
    ///
    /// To regenerate, after the starting design's geometry or the engine's reading of it has
    /// moved: run the script (its header says how), copy both lists here, and copy its
    /// `entered` list into `gui::dashboard::STARTING_3D_FIELDS`.
    const RECORDED_READS: [(&str, f64); 17] = [
        ("coupling.npole", 8.0),
        ("coupling.backiron", 1.0),
        ("coupling.inner_back_apothem_mm", 9.0),
        ("metal.bond_inner_mm", 0.05),
        ("metal.bond_outer_mm", 0.05),
        ("metal.cap_od_mm", 42.8),
        ("model.inner_length_mm", 12.7),
        ("model.inner_width_mm", 6.35),
        ("model.inner_thickness_mm", 3.17),
        ("model.outer_thickness_mm", 3.17),
        ("model.inner_br_T", 1.3),
        ("model.outer_face_apothem_mm", 13.97),
        ("model.outer_back_apothem_mm", 17.14),
        ("retainers.sleeve_id_mm", 25.204683460540704),
        ("retainers.sleeve_od_mm", 25.404683460540703),
        ("retainers.liner_id_mm", 27.490000000000002),
        ("retainers.liner_od_mm", 27.89),
    ];
    const RECORDED_FIELDS: [(&str, f64); 14] = [
        ("temperature.demag.h_rev_aligned_kA_m", 385.6470213208602),
        ("temperature.demag.h_rev_pullout_kA_m", 747.7472260964677),
        ("temperature.demag.h_rev_likepole_kA_m", 850.3845814018983),
        ("temperature.demag.h_rev_single_ring_kA_m", 585.433721206525),
        ("temperature.slip_loss.b_hub_T", 0.20209600565582997),
        ("temperature.slip_loss.b_cup_T", 0.20673376294569007),
        ("temperature.slip_loss.b_sleeve_T", 0.3349760675891163),
        ("temperature.slip_loss.b_liner_T", 0.3489279494170646),
        (
            "temperature.slip_loss.cap_integral_T2m4",
            4.043941097754211e-10,
        ),
        (
            "temperature.slip_loss.web_integral_T2m2",
            4.4050607298969813e-05,
        ),
        ("temperature.slip_loss.b_magnet_T", 0.17259565415202785),
        ("temperature.slip_loss.b_hub_free_T", 0.07416352937255892),
        ("temperature.slip_loss.b_cup_free_T", 0.0809945861055586),
        (
            "temperature.slip_loss.web_integral_free_T2m2",
            6.039253748156894e-06,
        ),
    ];

    /// A recorded field as the starting design stores it. The routine's eleven as
    /// `fields3d.apply_to_inputs` rounds them: whole kA/m; three decimals for a field, two for
    /// the field inside the blocks; three significant figures for the cap integral and four for
    /// the web's. The three free-space fields to four significant figures, as the engine's own
    /// defaults are held (correction E17).
    fn as_entered(path: &str, raw: f64) -> f64 {
        let name = path.rsplit('.').next().expect("a path");
        let significant = |figures: usize| format!("{raw:.*e}", figures - 1);
        let text = match name {
            _ if name.starts_with("h_rev") => format!("{raw:.0}"),
            "b_magnet_T" => format!("{raw:.2}"),
            "cap_integral_T2m4" => significant(3),
            _ if name.contains("free") || name == "web_integral_T2m2" => significant(4),
            _ => format!("{raw:.3}"),
        };
        text.parse().expect("a number")
    }

    /// Every input in which the starting design differs from the example design, sorted: the
    /// seven it sizes and its fourteen 3D fields.
    fn starting_differences() -> Vec<String> {
        let mut paths: Vec<String> = STARTING_CHANGES
            .iter()
            .map(|(path, _, _)| *path)
            .chain(crate::gui::dashboard::STORED_3D_INPUTS)
            .map(str::to_owned)
            .collect();
        paths.sort();
        paths
    }

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

    /// One frame of `app` on the laptop screen with `events`.
    fn frame_output(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            events,
            ..Default::default()
        };
        ctx.run(input, |ctx| app.ui(ctx))
    }

    /// [`frame_output`]; the texts it drew.
    fn frame_with(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        events: Vec<egui::Event>,
    ) -> Vec<String> {
        drawn_texts(&frame_output(ctx, app, events))
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
        assert_eq!(ctx.style().visuals, theme::visuals());
    }

    #[test]
    fn the_app_draws_its_text_in_inter() {
        // UX overhaul spec 6.2: the app installs its font; the panel alone does not.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let first = |ctx: &egui::Context| {
            ctx.fonts(|f| {
                f.lock().fonts.definitions().families[&egui::FontFamily::Proportional][0].clone()
            })
        };
        assert_eq!(first(&ctx), fonts::INTER_NAME);
        // A number keeps its width whatever its digits (the cut's tabular digits).
        let body = egui::TextStyle::Body.resolve(&ctx.style());
        let width = |digit| ctx.fonts(|f| f.glyph_width(&body, digit));
        assert_eq!(width('1'), width('8'));
        // A panel in a context no app set up keeps egui's fonts.
        let plain = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        let _ = plain.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        assert_ne!(first(&plain), fonts::INTER_NAME);
    }

    #[test]
    fn the_title_and_the_size_heading_are_semibold_and_the_top_bar_keeps_one_row() {
        // The stage 1 review (S-1): with one weight the title and the headings were only larger
        // than the text under them. In the app egui's Heading style is Inter SemiBold, which
        // draws the title and the left column's heading and no other text (the tab of the same
        // words stays Regular). A heavier face is wider, so at the smallest window, the laptop
        // screen and a desktop one the top bar's six words stay on the title's row and the
        // heading on one line inside the 320-point column (the column's width and the texts
        // cut on every tab are the BL-099 test's, in the same font).
        use crate::gui::test_support::flat_shapes;
        let semibold = fonts::semibold();
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            let output = frame_on(&ctx, &mut app, size, Vec::new());
            let mut heavy = Vec::new();
            for shape in flat_shapes(&output) {
                if let egui::Shape::Text(text) = shape {
                    let sections = &text.galley.job.sections;
                    let semibold_sections = sections
                        .iter()
                        .filter(|s| s.format.font_id.family == semibold)
                        .count();
                    if semibold_sections > 0 {
                        assert_eq!(semibold_sections, sections.len(), "{size:?}");
                        heavy.push((text.galley.text().to_owned(), text.visual_bounding_rect()));
                    }
                }
            }
            let names: Vec<&str> = heavy.iter().map(|(text, _)| text.as_str()).collect();
            let size_heading = crate::gui::CentreView::SizeAndTorque.label();
            assert_eq!(
                names,
                ["Magnetic coupling calculator", size_heading],
                "{size:?}"
            );
            let (title, heading) = (heavy[0].1, heavy[1].1);
            let column = side_rect(&ctx, "magcoupling_inputs");
            let style = ctx.style();
            let row = ctx.fonts(|f| f.row_height(&egui::TextStyle::Heading.resolve(&style)));
            assert!(
                column.contains_rect(heading) && heading.height() <= row,
                "{size:?}: the heading {heading:?} in {column:?}, rows of {row}"
            );
            // The top bar: every text over the left column (the title and the six words) on
            // the title's row, inside the window.
            let bar: Vec<(String, egui::Rect)> = flat_shapes(&output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::Text(text)
                        if text.visual_bounding_rect().bottom() <= column.top() =>
                    {
                        Some((text.galley.text().to_owned(), text.visual_bounding_rect()))
                    }
                    _ => None,
                })
                .collect();
            assert!(bar.len() >= 7, "{size:?}: {bar:?}");
            for (text, at) in &bar {
                assert!(
                    (at.center().y - title.center().y).abs() < 4.0 && at.right() <= size.x,
                    "{size:?}: {text:?} at {at:?}, the title at {title:?}"
                );
            }
        }
    }

    #[test]
    fn the_accent_marks_the_brand_and_the_active_tab_and_nothing_else() {
        // Decision C-5, in the app's own theme: crimson is never a text colour and never on a
        // value or a check. At the starting design and at a failing one the accent is on two
        // shapes, the brand mark and the line under the active tab.
        use crate::gui::look::ACCENT;
        use crate::gui::test_support::flat_shapes;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        for failing in [false, true] {
            if failing {
                app.panel.open_example_design();
                assert!(frame_texts(&ctx, &mut app).contains(&EXAMPLE_STATUS.to_owned()));
            }
            let output = frame_output(&ctx, &mut app, Vec::new());
            let mut marks = 0;
            for shape in flat_shapes(&output) {
                match shape {
                    egui::Shape::Text(text) => {
                        let mut colours = vec![text.fallback_color];
                        colours.extend(text.override_text_color);
                        colours.extend(text.galley.job.sections.iter().map(|s| s.format.color));
                        assert!(!colours.contains(&ACCENT), "{:?}", text.galley.text());
                    }
                    egui::Shape::Rect(rect) if rect.fill == ACCENT => marks += 1,
                    egui::Shape::LineSegment { stroke, .. } if stroke.color == ACCENT => marks += 1,
                    egui::Shape::Circle(circle) => assert_ne!(circle.fill, ACCENT),
                    _ => {}
                }
            }
            assert_eq!(marks, 2, "failing: {failing}");
        }
        // A failing check is the theme's red, which is not the accent.
        assert_ne!(theme::BAD, ACCENT);
    }

    #[test]
    fn every_control_shows_its_edge_on_the_panel_and_in_the_design_menu() {
        // The stage 1 review (E-4, R-6), in the app's own theme: a checkbox's box, a slider's
        // rail and a value box at rest were 1.1:1 to 1.2:1 on the panel. With both folds of the
        // left column open, every rect and circle the column draws (the sliders' rails and
        // handles, the value boxes, the pickers, the text boxes, the checkboxes and the order
        // toggle: no other shape is filled or outlined there at rest) shows an edge, by its
        // fill or its outline, at 3:1 or more against the panel. The Design menu draws a
        // checkbox's box with no outline: its fill is 3:1 or more against the menu.
        use crate::gui::test_support::{flat_shapes, text_center, text_rects};
        use crate::gui::{ASSUMPTIONS_FOLD, DESIGN_MENU, DETAILED_INPUTS, SHOW_DETAILS};
        use theme::contrast;
        // The stronger of a shape's fill and outline against `behind`.
        let edge = |fill: egui::Color32, stroke: egui::Stroke, behind: egui::Color32| {
            let outline = if stroke.width > 0.0 && stroke.color.a() > 0 {
                contrast(stroke.color, behind)
            } else {
                1.0
            };
            let fill = if fill.a() > 0 {
                contrast(fill, behind)
            } else {
                1.0
            };
            fill.max(outline)
        };
        // A window tall enough to draw every row of the column.
        let tall = egui::vec2(1280.0, 20_000.0);
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, tall, Vec::new());
        for fold in [DETAILED_INPUTS, ASSUMPTIONS_FOLD] {
            let output = frame_on(&ctx, &mut app, tall, Vec::new());
            click_at(&ctx, &mut app, tall, text_center(&output, fold), 6);
        }
        let output = frame_on(&ctx, &mut app, tall, Vec::new());
        let column = side_rect(&ctx, "magcoupling_inputs");
        let mut faint = Vec::new();
        let mut shapes = 0;
        for shape in flat_shapes(&output) {
            let (what, ratio) = match shape {
                egui::Shape::Rect(rect)
                    if column.contains_rect(rect.rect)
                        && rect.rect != column
                        && (rect.fill.a() > 0 || rect.stroke.color.a() > 0) =>
                {
                    (
                        format!("{:?}", rect.rect),
                        edge(rect.fill, rect.stroke, theme::PANEL),
                    )
                }
                egui::Shape::Circle(circle) if column.contains(circle.center) => (
                    format!("a circle at {:?}", circle.center),
                    edge(circle.fill, circle.stroke, theme::PANEL),
                ),
                _ => continue,
            };
            shapes += 1;
            if ratio < 3.0 {
                faint.push(format!("{what}: {ratio:.2}"));
            }
        }
        // Six size controls, twenty-one more sliders, their value boxes and the rest.
        assert!(shapes > 90, "{shapes} shapes");
        assert!(
            faint.is_empty(),
            "{} of {shapes} faint: {faint:#?}",
            faint.len()
        );
        // The Design menu, opened on the laptop screen, its fading in done.
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        click_at(&ctx, &mut app, LAPTOP, text_center(&output, DESIGN_MENU), 6);
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let label = text_rects(&output, SHOW_DETAILS);
        let label = label.first().expect("the menu shows the checkbox");
        let boxes: Vec<f64> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect)
                    if rect.rect.right() <= label.left()
                        && (rect.rect.center().y - label.center().y).abs() < 2.0
                        && (rect.rect.width() - rect.rect.height()).abs() < 0.5 =>
                {
                    Some(edge(rect.fill, rect.stroke, theme::PANEL2))
                }
                _ => None,
            })
            .collect();
        assert_eq!(boxes.len(), 1, "the checkbox's box: {boxes:?}");
        assert!(boxes[0] >= 3.0, "the menu's checkbox: {:.2}", boxes[0]);
    }

    #[test]
    fn an_open_picker_keeps_its_edge() {
        // egui draws an open picker's box in the open state's colours, not the resting ones.
        // Opened, Auto's picker keeps an outline at 3:1 or more against the panel, as at rest,
        // so it does not read as dimmed or disabled while its list shows.
        use crate::engine::sizing::FreeVariable;
        use crate::gui::sizing::auto_choice;
        use crate::gui::test_support::{flat_shapes, text_rects};
        use theme::contrast;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let output = frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let shown = text_rects(&output, auto_choice(FreeVariable::AxialLength));
        assert_eq!(shown.len(), 1, "Auto's picker shows its size: {shown:?}");
        let output = click_at(&ctx, &mut app, LAPTOP, shown[0].center(), 6);
        // Open: its list names the other sizes.
        let listed = text_rects(&output, auto_choice(FreeVariable::RingRadius));
        assert_eq!(listed.len(), 1, "the picker's list: {listed:?}");
        let outlines: Vec<f64> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect)
                    if rect.rect.contains_rect(shown[0])
                        && rect.rect.height() < 3.0 * shown[0].height() =>
                {
                    Some(if rect.stroke.width > 0.0 {
                        contrast(rect.stroke.color, theme::PANEL)
                    } else {
                        1.0
                    })
                }
                _ => None,
            })
            .collect();
        assert_eq!(outlines.len(), 1, "the picker's box: {outlines:?}");
        assert!(
            outlines[0] >= 3.0,
            "the open picker's outline: {:.2}",
            outlines[0]
        );
    }

    #[test]
    fn a_plot_legend_s_dots_show_at_three_to_one_on_their_discs_shown_or_hidden() {
        // egui_plot draws each legend entry, a toggle that hides or shows its line, as a disc
        // in the interact state's fill and outline with the line's dot on it while the line is
        // shown, so the dot alone shows the state. BL-108 (the review of part 2b-iii-b,
        // LOOK-1): on a disc in the controls' grey the dots were 1.6:1 to 2.6:1. In the app's
        // theme and font, on all five plots of the starting design and of the example design
        // (its sweeps have rows that do not fit): every dot is 3:1 or more against the disc it
        // is drawn on and every disc's ring 3:1 or more against the legend's background (the
        // field colour); an entry clicked hidden keeps its disc, fill and ring with no dot, and
        // a second click shows it again. The two designs draw all eight colours the plots give
        // a series: blue, amber, violet, the text, muted and the three levels' colours.
        use crate::gui::CentreView;
        use crate::gui::plots::PlotKind;
        use crate::gui::test_support::flat_shapes;
        use theme::contrast;
        // A frame's legend entries in the centre region: each a filled and outlined disc (a
        // plot's own points are one or the other) and, while its line is shown, the colour of
        // the smaller dot at its centre.
        let entries = |ctx: &egui::Context, output: &egui::FullOutput| {
            let (left, right) = (
                side_rect(ctx, "magcoupling_inputs").right(),
                side_rect(ctx, "magcoupling_dashboard").left(),
            );
            let circles: Vec<egui::epaint::CircleShape> = flat_shapes(output)
                .into_iter()
                .filter_map(|shape| match shape {
                    egui::Shape::Circle(c) if left < c.center.x && c.center.x < right => Some(*c),
                    _ => None,
                })
                .collect();
            circles
                .iter()
                .filter(|disc| disc.fill.a() > 0 && disc.stroke.width > 0.0)
                .map(|disc| {
                    let dot = circles
                        .iter()
                        .find(|dot| dot.center == disc.center && dot.radius < disc.radius);
                    (*disc, dot.map(|dot| dot.fill))
                })
                .collect::<Vec<_>>()
        };
        let mut colours = std::collections::HashSet::new();
        for example in [false, true] {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            if example {
                app.panel.open_example_design();
            }
            frame_on(&ctx, &mut app, LAPTOP, Vec::new());
            for kind in PlotKind::ALL {
                let what = format!("example design {example}, {kind:?}");
                let output = show_view(&ctx, &mut app, LAPTOP, CentreView::Plot(kind), 2);
                let shown = entries(&ctx, &output);
                assert!(!shown.is_empty(), "{what}: no legend entry");
                for (disc, dot) in &shown {
                    let ring = contrast(disc.stroke.color, theme::BG);
                    assert!(
                        ring >= 3.0,
                        "{what}: a disc's ring on the legend: {ring:.2}"
                    );
                    let dot = dot.unwrap_or_else(|| panic!("{what}: an entry with no dot"));
                    let ratio = contrast(dot, disc.fill);
                    assert!(
                        ratio >= 3.0,
                        "{what}: the dot {} on its disc {}: {ratio:.2}",
                        theme::hex(dot),
                        theme::hex(disc.fill)
                    );
                    colours.insert(dot);
                }
                // A click on the first entry hides its line: its disc stays, with no dot.
                let (first, first_dot) = shown[0];
                let hidden = entries(&ctx, &click_at(&ctx, &mut app, LAPTOP, first.center, 2));
                let hidden: Vec<_> = hidden
                    .iter()
                    .filter(|(d, _)| d.center == first.center)
                    .collect();
                assert_eq!(hidden.len(), 1, "{what}: the hidden entry's disc");
                assert_eq!(hidden[0].1, None, "{what}: the hidden entry's dot");
                assert_eq!(
                    (hidden[0].0.fill, hidden[0].0.stroke),
                    (first.fill, first.stroke),
                    "{what}: the hidden entry's disc"
                );
                // A second click shows it again.
                let again = entries(&ctx, &click_at(&ctx, &mut app, LAPTOP, first.center, 2));
                assert!(
                    again
                        .iter()
                        .any(|(d, dot)| d.center == first.center && *dot == first_dot),
                    "{what}: shown again"
                );
            }
        }
        assert_eq!(colours.len(), 8, "the dots' colours: {colours:?}");
    }

    #[test]
    fn a_tile_shows_its_edge_on_the_panel_as_a_framed_control_does() {
        // BL-109 (the review of part 2b-iii-b, LOOK-2): the four tiles over the drawing are
        // click targets that the panel frames itself, and they kept the hairline (1.2:1 on the
        // panel) where every control egui frames shows its edge at 3:1. In the app's theme, on
        // the first view at 1280 x 800, each tile's resting outline is the resting outline of
        // a framed control, 3:1 or more against the panel; a click on the Hot-low tile outlines
        // it in the strong text colour, another colour, at the same width, and no tile moves.
        use crate::gui::test_support::{flat_shapes, text_rects};
        use crate::gui::tiles::Tile;
        use theme::contrast;
        let inputs = starting_design();
        let results = compute_all(&inputs);
        // Each tile's frame: the rect in the tiles' fill around its number in the centre
        // region (the Hot-low tile's torque card repeats its number in the Explain panel).
        let frames = |ctx: &egui::Context, output: &egui::FullOutput| {
            let right = side_rect(ctx, "magcoupling_dashboard").left();
            Tile::ALL
                .iter()
                .map(|tile| {
                    let number: Vec<egui::Rect> =
                        text_rects(output, &tile.number(&inputs, &results).unwrap())
                            .into_iter()
                            .filter(|at| at.right() < right)
                            .collect();
                    assert_eq!(number.len(), 1, "{tile:?}'s number: {number:?}");
                    let frame: Vec<&egui::epaint::RectShape> = flat_shapes(output)
                        .into_iter()
                        .filter_map(|shape| match shape {
                            egui::Shape::Rect(r)
                                if r.fill == theme::PANEL2 && r.rect.contains_rect(number[0]) =>
                            {
                                Some(r)
                            }
                            _ => None,
                        })
                        .collect();
                    assert_eq!(frame.len(), 1, "{tile:?}'s frame: {frame:?}");
                    frame[0].clone()
                })
                .collect::<Vec<egui::epaint::RectShape>>()
        };
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_on(&ctx, &mut app, LAPTOP, Vec::new());
        let rest = theme::visuals().widgets.inactive.bg_stroke;
        let resting = frames(&ctx, &frame_on(&ctx, &mut app, LAPTOP, Vec::new()));
        for (tile, frame) in Tile::ALL.iter().zip(&resting) {
            let ratio = contrast(frame.stroke.color, theme::PANEL);
            assert!(
                ratio >= 3.0,
                "{tile:?}'s resting outline on the panel: {ratio:.2}"
            );
            assert_eq!(frame.stroke, rest, "{tile:?}");
        }
        let hot_low = Tile::ALL.iter().position(|&t| t == Tile::HotLow).unwrap();
        let output = click_at(&ctx, &mut app, LAPTOP, resting[hot_low].rect.center(), 2);
        let selected = frames(&ctx, &output);
        let mark = egui::Stroke::new(rest.width, theme::TEXT);
        assert_ne!(mark.color, rest.color);
        for ((tile, before), after) in Tile::ALL.iter().zip(&resting).zip(&selected) {
            let stroke = if *tile == Tile::HotLow { mark } else { rest };
            assert_eq!(after.stroke, stroke, "{tile:?}");
            assert_eq!(after.rect, before.rect, "{tile:?} moved");
        }
    }

    #[test]
    fn a_label_is_muted_and_its_value_is_in_the_text_colour() {
        // UX overhaul spec 6.1: the muted token is for labels and captions, the text token for
        // values. A reader scanning for a number finds it as the bright line of its row.
        use crate::gui::test_support::text_colors;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let all = |output: &egui::FullOutput, texts: &[&str], colour: egui::Color32| {
            for text in texts {
                let drawn = text_colors(output, text);
                assert!(!drawn.is_empty(), "no {text:?} drawn");
                assert!(drawn.iter().all(|&c| c == colour), "{text:?}: {drawn:?}");
            }
        };
        // The starting design: the right column's two headings and two of the Size tab's
        // readout labels. Then their values.
        let output = frame_output(&ctx, &mut app, Vec::new());
        let labels = [
            crate::gui::checks::HEADING,
            crate::gui::explain::HEADING,
            "Required torque",
            "Pull-out torque (nominal, at 50 °C)",
        ];
        all(&output, &labels, theme::MUTED);
        all(&output, &["1.704 N·m", "1.448 N·m"], theme::TEXT);
        // A check's name is what its row says: the text colour.
        all(&output, &["Running clearance"], theme::TEXT);
        // A tab that is not the active one is quiet too; the active tab's name (drawn as the
        // left side's heading as well) is the text colour.
        all(&output, &[crate::gui::plots::PLOTS], theme::MUTED);
        all(&output, &["Size and torque"], theme::TEXT);
        // The failing example design with its Running clearance check selected: the entry's
        // headings are captions, and under the last a row's label is muted and its value is
        // not.
        use crate::gui::explain::{NUMBERS, WHAT_IT_IS, WHAT_SETS_IT};
        use crate::gui::explorer::WHAT_TO_CHANGE;
        use crate::gui::test_support::{primary_button, text_center};
        app.panel.open_example_design();
        frame_texts(&ctx, &mut app);
        let laptop = WINDOWS[0];
        let output = frame_on(&ctx, &mut app, laptop, Vec::new());
        let row = text_center(&output, "Running clearance");
        frame_on(&ctx, &mut app, laptop, vec![egui::Event::PointerMoved(row)]);
        for pressed in [true, false] {
            frame_on(&ctx, &mut app, laptop, vec![primary_button(row, pressed)]);
        }
        frame_on(&ctx, &mut app, laptop, vec![egui::Event::PointerGone]);
        let output = frame_on(&ctx, &mut app, laptop, Vec::new());
        let headings = [WHAT_IT_IS, WHAT_SETS_IT, WHAT_TO_CHANGE, NUMBERS];
        all(&output, &headings, theme::MUTED);
        const CLEARANCE: &str = "metal.min_running_clearance_mm";
        let label = crate::gui::plain_text::label_for(CLEARANCE);
        all(&output, &[label], theme::MUTED);
        let results = compute_all(&DesignInputs::default());
        let lines = crate::gui::dashboard::shown_lines(&results, crate::engine::clamps::PAUSED);
        let line = lines.iter().find(|line| line.path == CLEARANCE);
        let value = line.expect("the clearance has a row").shown_value();
        all(&output, &[value.as_str()], theme::TEXT);
    }

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

    /// The two window sizes the look is checked at: the laptop screen and the smallest window
    /// supported (decision C-18).
    const WINDOWS: [egui::Vec2; 2] = [egui::vec2(1280.0, 800.0), egui::vec2(1024.0, 700.0)];

    /// The laptop screen the app's tests draw on.
    const LAPTOP: egui::Vec2 = WINDOWS[0];

    #[test]
    fn no_text_is_cut_in_any_tab_in_the_app_s_own_font() {
        // The whole-branch review (R-4): Inter is about 13 % wider than egui's built-in font,
        // which every panel layout test draws in, so a text that just fitted there can be cut
        // here and no other test sees it (R-2: the Pole sweep's legend lost its first letters).
        // The app as a browser draws it, in its theme and font, on every tab.
        use crate::gui::CentreView;
        use crate::gui::test_support::cut_texts;
        for size in WINDOWS {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            for view in CentreView::tabs(crate::engine::clamps::PAUSED) {
                // The pointer leaves after each click, so no tooltip is in the picture; then
                // the view settles.
                let output = show_view(&ctx, &mut app, size, view, 2);
                let cut = cut_texts(&output, screen);
                assert!(cut.is_empty(), "{size:?}, {}: cut: {cut:#?}", view.label());
            }
        }
    }

    #[test]
    fn the_tab_row_is_one_row_in_the_app_s_own_font() {
        // The review of part 2b-iii-a (T-1): the panel's test finds the tab row on one row in
        // egui's built-in font, and Inter is wider (R-4). From the smallest window up, in the
        // app's theme and font, every tab of the row sits on one row.
        use crate::gui::CentreView;
        use crate::gui::test_support::last_characters;
        for size in WINDOWS {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            let output = frame_on(&ctx, &mut app, size, Vec::new());
            let (left, right) = (
                side_rect(&ctx, "magcoupling_inputs").right(),
                side_rect(&ctx, "magcoupling_dashboard").left(),
            );
            // Each tab's row: the topmost of its words in the centre region (the left column's
            // heading bears the first tab's words).
            let rows: Vec<(&str, f32)> = CentreView::tab_row(crate::engine::clamps::PAUSED)
                .map(|view| {
                    let y = last_characters(&output, view.tab())
                        .into_iter()
                        .filter(|at| left < at.x && at.x < right)
                        .map(|at| at.y)
                        .min_by(f32::total_cmp)
                        .unwrap_or_else(|| panic!("{size:?}: no tab {:?}", view.tab()));
                    (view.tab(), y)
                })
                .collect();
            assert!(
                rows.iter().all(|(_, y)| (y - rows[0].1).abs() < 1.0),
                "{size:?}: {rows:?}"
            );
        }
    }

    /// Shows `view` as a person does on a screen of `size`: a click on its tab in the centre
    /// region (the left column's heading bears the first tab's words), then, for a plot, a
    /// click on its name in the Plots tab's picker (decision C-9), each click as [`click_at`]
    /// with `settle` frames. Returns the last frame.
    fn show_view(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        size: egui::Vec2,
        view: crate::gui::CentreView,
        settle: usize,
    ) -> egui::FullOutput {
        use crate::gui::test_support::last_characters;
        let mut output = frame_on(ctx, app, size, Vec::new());
        let mut clicks = vec![view.tab()];
        if matches!(view, crate::gui::CentreView::Plot(_)) {
            clicks.push(view.label());
        }
        for text in clicks {
            let (left, right) = (
                side_rect(ctx, "magcoupling_inputs").right(),
                side_rect(ctx, "magcoupling_dashboard").left(),
            );
            let at = last_characters(&output, text)
                .into_iter()
                .find(|at| left < at.x && at.x < right)
                .unwrap_or_else(|| panic!("{size:?}: no {text:?} in the centre region"));
            output = click_at(ctx, app, size, at, settle);
        }
        output
    }

    /// A click at `at` on a screen of `size`, the pointer gone after it, then `settle` frames
    /// (a fold takes six to open or shut). Returns the last frame.
    fn click_at(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        size: egui::Vec2,
        at: egui::Pos2,
        settle: usize,
    ) -> egui::FullOutput {
        use crate::gui::test_support::primary_button;
        for pressed in [true, false] {
            frame_on(ctx, app, size, vec![primary_button(at, pressed)]);
        }
        let mut output = frame_on(ctx, app, size, vec![egui::Event::PointerGone]);
        for _ in 0..settle {
            output = frame_on(ctx, app, size, Vec::new());
        }
        output
    }

    /// Asserts what holds a side column still (BL-099, BL-098), as the app is seen on a screen
    /// of `size` in the state `what`: on every tab of the centre region the side panel `id` is
    /// `width` points wide and no text of the page is cut sideways.
    fn assert_the_column_holds(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        size: egui::Vec2,
        (id, width): (&str, f32),
        what: &str,
    ) {
        use crate::gui::CentreView;
        use crate::gui::test_support::cut_texts;
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
        // The last frame may have been drawn on another screen: a wrapped row settles on the
        // second frame.
        for _ in 0..2 {
            frame_on(ctx, app, size, Vec::new());
        }
        // Last tab first: the page is left on its first view, the lightest to draw.
        let mut views: Vec<CentreView> = CentreView::tabs(crate::engine::clamps::PAUSED).collect();
        views.reverse();
        for view in views {
            let output = show_view(ctx, app, size, view, 2);
            let state = format!("{size:?}, {what}, on {}", view.label());
            assert_eq!(side_rect(ctx, id).width(), width, "{state}");
            let cut = cut_texts(&output, screen);
            assert!(cut.is_empty(), "{state}: cut: {cut:#?}");
        }
    }

    #[test]
    fn the_left_column_keeps_its_width_whatever_is_opened_in_the_app_s_own_font() {
        // BL-099: a row's label longer than its row ran past the column's edge, and egui's side
        // panel grows to what it holds and keeps the width: 330.1 points once the Detailed
        // inputs opened, 443.1 with the last group open, for the rest of the session, the
        // centre and its drawing that much narrower and text cut on two tabs. In the app's
        // theme and font, at the smallest window, the laptop screen and a desktop one, the
        // column is 320 points wide with the folds shut, with each fold opened and shut again,
        // with every input group and its Advanced heading opened and shut again, with a filter
        // that lists every input set and cleared, with an input traced (its group's heading
        // then bears a count), and with Auto on; and no text is cut, on any tab or on any row
        // of the column. The tabs are read where the column holds more than it did: a state
        // shut again draws what the shut state drew, so its width is the whole reading, and a
        // column never narrows by itself, so the last reading holds for each of them.
        use crate::engine::clamps::PAUSED;
        use crate::gui::explain::TRACE_LINK;
        use crate::gui::inputs::{ADVANCED_HEADING, FILTER_HINT, InputOrder};
        use crate::gui::test_support::{cut_texts, flat_shapes, key_tap, text_rects};
        use crate::gui::{ASSUMPTIONS_FOLD, AUTO, DETAILED_INPUTS, OPEN_SIZE_TAB};
        const COLUMN: (&str, f32) = ("magcoupling_inputs", 320.0);
        let groups = InputCatalogue::shown(PAUSED).groups_in(InputOrder::Workflow);
        let total = InputCatalogue::shown(PAUSED).all().count();
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
            // The same window with room for every row of the column: a heading out of view at
            // `size` is clicked here, and every row is read here for a cut and the column for
            // its width. The tabs, and the width again, are read at `size`.
            let tall = egui::vec2(size.x, 20_000.0);
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            // A frame of the tall window, settled: the last one may have been drawn at `size`.
            let settled = |app: &mut MagcouplingApp| {
                for _ in 0..2 {
                    frame_on(&ctx, app, tall, Vec::new());
                }
                frame_on(&ctx, app, tall, Vec::new())
            };
            // The left column in the tall window, as wide as it started and no text of the
            // page cut: how many texts it draws, and those equal to `text`, top to bottom.
            let column = |app: &mut MagcouplingApp, text: &str| -> (usize, Vec<egui::Rect>) {
                let output = settled(app);
                let cut = cut_texts(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, tall));
                assert!(cut.is_empty(), "{size:?}: cut in the tall window: {cut:#?}");
                let panel = side_rect(&ctx, COLUMN.0);
                assert_eq!(panel.width(), COLUMN.1, "{size:?}: at {text:?}");
                let right = panel.right();
                let rows = flat_shapes(&output)
                    .into_iter()
                    .filter(|shape| match shape {
                        egui::Shape::Text(text) => text.visual_bounding_rect().right() <= right,
                        _ => false,
                    })
                    .count();
                let found = text_rects(&output, text);
                (
                    rows,
                    found.into_iter().filter(|r| r.right() <= right).collect(),
                )
            };
            // A click on the first heading `text` of the left column under `below`: the column
            // draws more texts once it has opened, fewer once it has shut. Returns its foot.
            let click = |app: &mut MagcouplingApp, text: &str, below: f32, opens: bool| -> f32 {
                let (before, found) = column(app, text);
                let at = found.into_iter().find(|r| r.top() > below);
                let at = at.unwrap_or_else(|| panic!("{size:?}: no {text:?} in the left column"));
                click_at(&ctx, app, tall, at.center(), 6);
                let after = column(app, text).0;
                assert_ne!(after, before, "{size:?}: {text:?}");
                assert_eq!(after > before, opens, "{size:?}: {text:?} {before} {after}");
                at.bottom()
            };
            let holds = |app: &mut MagcouplingApp, what: &str| {
                assert_the_column_holds(&ctx, app, size, COLUMN, what);
            };
            holds(&mut app, "the folds shut");
            for fold in [DETAILED_INPUTS, ASSUMPTIONS_FOLD] {
                click(&mut app, fold, 0.0, true);
                holds(&mut app, &format!("{fold} open"));
                click(&mut app, fold, 0.0, false);
            }
            // Each input group and its Advanced heading, one group at a time: the groups under
            // the one opened are shut, so its Advanced heading is the first under its own.
            click(&mut app, DETAILED_INPUTS, 0.0, true);
            for (index, group) in groups.iter().enumerate() {
                let header = click(&mut app, group.label, 0.0, true);
                if group.has_advanced() {
                    click(&mut app, ADVANCED_HEADING, header, true);
                }
                // The last group's labels are the longest: the column was 443.1 points here.
                if index + 1 == groups.len() {
                    holds(
                        &mut app,
                        "the last input group and its Advanced heading open",
                    );
                }
                if group.has_advanced() {
                    click(&mut app, ADVANCED_HEADING, header, false);
                }
                click(&mut app, group.label, 0.0, false);
            }
            // A filter that lists every input (each path has a dot), then cleared.
            let filter = column(&mut app, FILTER_HINT).1[0];
            click_at(&ctx, &mut app, tall, filter.center(), 0);
            frame_on(
                &ctx,
                &mut app,
                tall,
                vec![egui::Event::Text(".".to_owned())],
            );
            let listed = format!("{total} of {total} inputs");
            assert_eq!(column(&mut app, &listed).1.len(), 1, "{size:?}: {listed}");
            holds(&mut app, "a filter that lists every input");
            // The tabs' clicks took the keyboard from the filter box: a click on its text gives
            // it back, the caret before or after the dot.
            let filter = column(&mut app, ".").1[0];
            click_at(&ctx, &mut app, tall, filter.center(), 0);
            for key in [egui::Key::Backspace, egui::Key::Delete] {
                frame_on(&ctx, &mut app, tall, key_tap(key));
            }
            assert_eq!(column(&mut app, FILTER_HINT).1.len(), 1, "{size:?}");
            // A trace counts the rows it frames in each group's heading, shut or open: the
            // Size column's Required torque traced (a click on its label, then on its entry's
            // link), and its group's heading is the longest of the column, 293 points in the
            // app's font with its count. The link again ends the trace.
            let traced = format!("{} (1 traced)", groups[0].label);
            for on in [true, false] {
                if on {
                    let label = column(&mut app, "Required torque").1[0];
                    click_at(&ctx, &mut app, tall, label.center(), 2);
                }
                let link = text_rects(&settled(&mut app), TRACE_LINK)[0];
                click_at(&ctx, &mut app, tall, link.center(), 2);
                let drawn = column(&mut app, &traced).1.len();
                assert_eq!(drawn, usize::from(on), "{size:?}: {traced}");
                if on {
                    holds(&mut app, "an input traced");
                }
            }
            // Auto on, the Detailed inputs still open: Auto's line (which size it sets and the
            // Open the Size tab button) heads the Key design group, 425 to 445 points wide were
            // that line not to wrap. Its toggle again turns it off.
            for on in [true, false] {
                let toggle = column(&mut app, AUTO).1[0];
                click_at(&ctx, &mut app, tall, toggle.center(), 2);
                let drawn = column(&mut app, OPEN_SIZE_TAB).1.len();
                assert_eq!(drawn, usize::from(on), "{size:?}: Auto's line");
                if on {
                    holds(&mut app, "Auto on, the Detailed inputs open");
                }
            }
            for _ in 0..2 {
                frame_on(&ctx, &mut app, size, Vec::new());
            }
            let width = side_rect(&ctx, COLUMN.0).width();
            assert_eq!(
                width, COLUMN.1,
                "{size:?}: everything shut and cleared again"
            );
        }
    }

    #[test]
    fn the_right_column_keeps_its_width_with_a_note_shown_in_the_app_s_own_font() {
        // BL-098 as a person meets it, in the app's theme and font at the smallest window and
        // the laptop screen: Learn on the top bar, then the title of the first note with a
        // diagram. The diagram was 300 points wide in a column that holds 284, so the column
        // grew to 316 and stayed there. With the note shown the right column is as wide as it
        // started and no text is cut, on every tab.
        use crate::gui::explain::{LEARN, shown_notes};
        use crate::gui::test_support::text_center;
        use crate::gui::typeset::glyph_safe;
        const COLUMN: (&str, f32) = ("magcoupling_dashboard", 300.0);
        let paused = crate::engine::clamps::PAUSED;
        let note = shown_notes(paused)
            .find(|note| note.diagram.is_some())
            .expect("a note with a diagram");
        let other = shown_notes(paused)
            .find(|other| other.id != note.id)
            .expect("another note");
        for size in WINDOWS {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            assert_the_column_holds(&ctx, &mut app, size, COLUMN, "the orientation");
            for text in [LEARN.to_owned(), glyph_safe(note.title)] {
                let output = frame_on(&ctx, &mut app, size, Vec::new());
                click_at(&ctx, &mut app, size, text_center(&output, &text), 2);
            }
            // The note has taken the list's place: its own title, and no other note's.
            let texts = drawn_texts(&frame_on(&ctx, &mut app, size, Vec::new()));
            assert!(texts.contains(&glyph_safe(note.title)), "{size:?}");
            assert!(!texts.contains(&glyph_safe(other.title)), "{size:?}");
            assert_the_column_holds(&ctx, &mut app, size, COLUMN, "a note with a diagram");
        }
    }

    #[test]
    fn every_input_row_takes_no_more_than_the_width_it_is_given_in_either_font() {
        // BL-099's rule, row by row: an input row is no wider than the width it is drawn in,
        // so no row can set the left column's width. In egui's default font and in the app's
        // theme and font (its 8-point button padding round the part and grade pickers), at the
        // three widths the 320-point column gives a row (the filtered list and the Assumptions
        // fold, a group, a group's Advanced heading), every input: as it is, changed (its reset
        // button), changed and outside its slider's range (the note too), and locked by Auto.
        use crate::gui::input_ui::input_row;
        use crate::gui::test_support::sized_frame;
        let inputs = starting_design();
        for themed in [false, true] {
            let ctx = egui::Context::default();
            if themed {
                theme::apply(&ctx);
                fonts::install(&ctx);
            }
            // (Changed, outside its range, locked.)
            let states = [
                (false, false, false),
                (true, false, false),
                (true, true, false),
                (false, false, true),
            ];
            let mut rows = 0;
            let mut wide = Vec::new();
            // A context loads its fonts in its first frame, and a wrapped line settles in its
            // second: the third frame is read.
            for frame in 0..3 {
                sized_frame(&ctx, LAPTOP, Vec::new(), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for entry in InputCatalogue::get().all() {
                            let value = inputs.get(&entry.path).unwrap_or(Value::None);
                            for (given, state) in [304.0_f32, 286.0, 268.0]
                                .into_iter()
                                .flat_map(|given| states.map(|state| (given, state)))
                            {
                                let (changed, outside, locked) = state;
                                let current = match (outside, entry.meta.range) {
                                    (false, _) => value.clone(),
                                    (true, Some(range)) => match value {
                                        Value::Int(_) => Value::Int(10 * range.max as i64 + 1),
                                        _ => Value::Num(10.0 * range.max + 1.0),
                                    },
                                    // No slider, no range to be outside of.
                                    (true, None) => continue,
                                };
                                let default = if changed {
                                    Value::Text("another value".to_owned())
                                } else {
                                    current.clone()
                                };
                                let used = ui
                                    .allocate_ui(egui::vec2(given, 0.0), |ui| {
                                        ui.push_id((given as i32, state), |ui| {
                                            input_row(
                                                ui, entry, &current, &default, None, locked, false,
                                            );
                                        });
                                        ui.min_rect().width()
                                    })
                                    .inner;
                                if frame == 2 {
                                    rows += 1;
                                    if used > given {
                                        wide.push(format!(
                                            "{} {state:?}: {used} in {given}",
                                            entry.path
                                        ));
                                    }
                                }
                            }
                        }
                    });
                });
            }
            assert!(rows > 3 * 3 * InputCatalogue::get().all().count(), "{rows}");
            assert!(wide.is_empty(), "themed {themed}: {wide:#?}");
        }
    }

    #[test]
    fn an_input_label_that_wraps_is_one_tab_stop_in_either_font() {
        // A label egui lays out in a wrapping line is allocated row by row, each row a Tab stop
        // of its own, and only the first shows the focus (egui 0.32 label.rs): since BL-099's
        // head line wraps, each wrapped label took one press that showed nothing (SK-1). The
        // input rows whose label wraps at the narrowest width the column gives a row, each
        // changed (its reset button): a Tab round over them has as many stops there as at a
        // width where nothing wraps, and the reset button sits close under the label.
        use crate::gui::input_ui::{RESET_LABEL, input_row};
        use crate::gui::inputs::InputEntry;
        use crate::gui::plain_text::label_for;
        use crate::gui::test_support::{key_tap, sized_frame, text_rect, text_rects, text_shapes};
        let inputs = starting_design();
        let all: Vec<&InputEntry> = InputCatalogue::get().all().collect();
        for themed in [false, true] {
            let context = || {
                let ctx = egui::Context::default();
                if themed {
                    theme::apply(&ctx);
                    fonts::install(&ctx);
                }
                ctx
            };
            // One frame of the rows at the width `given`, on a screen tall enough for them with
            // no scroll area (its drag surface is a stop of its own).
            let frame = |ctx: &egui::Context, given: f32, rows: &[&InputEntry], events| {
                let screen = egui::vec2(given + 40.0, 40_000.0);
                sized_frame(ctx, screen, events, |ui| {
                    for entry in rows {
                        let current = inputs.get(&entry.path).unwrap_or(Value::None);
                        let default = Value::Text("another value".to_owned());
                        ui.allocate_ui(egui::vec2(given, 0.0), |ui| {
                            input_row(ui, entry, &current, &default, None, false, false);
                        });
                    }
                })
            };
            // The rows whose label takes more than one line at `given`, each with its reset
            // button on the line under it at a widget's height (egui gives a wrapping line's
            // next row the height of the row before: here the two-line label's). A context
            // loads its fonts in its first frame, and a wrapped line settles in its second.
            let wrapped = |given: f32| {
                let ctx = context();
                frame(&ctx, given, &all, Vec::new());
                let output = frame(&ctx, given, &all, Vec::new());
                let resets = text_rects(&output, RESET_LABEL);
                let half_a_widget = ctx.style().spacing.interact_size.y / 2.0;
                let long: Vec<&InputEntry> = all
                    .iter()
                    .copied()
                    .filter(|entry| {
                        text_shapes(&output, label_for(&entry.path))
                            .any(|(_, text)| text.galley.rows.len() > 1)
                    })
                    .collect();
                for entry in &long {
                    let label = text_rect(&output, label_for(&entry.path)).expect("drawn");
                    let reset = resets
                        .iter()
                        .filter(|reset| reset.top() > label.top())
                        .min_by(|a, b| a.top().total_cmp(&b.top()))
                        .expect("its reset button");
                    let gap = reset.top() - label.bottom();
                    assert!(
                        (0.0..half_a_widget).contains(&gap),
                        "themed {themed}: {} {label:?} {reset:?}",
                        entry.path
                    );
                }
                long
            };
            // The stops of a Tab round from no focus.
            let stops = |given: f32, rows: &[&InputEntry]| {
                let ctx = context();
                frame(&ctx, given, rows, Vec::new());
                let mut seen = Vec::new();
                loop {
                    frame(&ctx, given, rows, key_tap(egui::Key::Tab));
                    // Past the last stop the focus wraps to the first, or is let go.
                    let focused = ctx.memory(|m| m.focused());
                    let Some(focused) = focused.filter(|id| !seen.contains(id)) else {
                        return seen.len();
                    };
                    seen.push(focused);
                    assert!(seen.len() < 1_000, "the Tab round never ends");
                }
            };
            let long = wrapped(268.0);
            assert!(!long.is_empty(), "themed {themed}: no label wraps");
            assert!(wrapped(4_000.0).is_empty(), "themed {themed}");
            let paths: Vec<&str> = long.iter().map(|entry| entry.path.as_str()).collect();
            assert_eq!(
                stops(268.0, &long),
                stops(4_000.0, &long),
                "themed {themed}: {paths:?}"
            );
        }
    }

    #[test]
    fn the_first_view_keeps_its_three_columns_its_tiles_and_the_fixed_3d_line_in_the_app_s_own_font()
     {
        // UX overhaul spec 8.2, test 15, and the stage 1 review's C-6: in the app's theme and
        // font, at the smallest window supported, the laptop screen and a desktop one. The four
        // tile numbers are drawn whole, between the Size column and the check list; the drawing
        // keeps its least height; the check list shows the line that says two checks read
        // stored 3D fields (register row A-18; spec 2.3: it stays visible), and the Explain
        // panel its orientation. The line is found by its constant, whatever its words.
        use crate::gui::dashboard::STORED_3D_LABEL;
        use crate::gui::explain::ORIENTATION;
        use crate::gui::geometry_view::MIN_DRAWING_HEIGHT;
        use crate::gui::test_support::{flat_shapes, shown_whole};
        use crate::gui::tiles::Tile;
        // The starting design, then the failing example design (its longer entries and
        // lines).
        for example in [false, true] {
            let inputs = if example {
                DesignInputs::default()
            } else {
                starting_design()
            };
            let results = compute_all(&inputs);
            for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
                let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
                let ctx = egui::Context::default();
                let mut app = MagcouplingApp::new(&ctx);
                if example {
                    app.panel.open_example_design();
                }
                for _ in 0..3 {
                    frame_on(&ctx, &mut app, size, Vec::new());
                }
                let output = frame_on(&ctx, &mut app, size, Vec::new());
                let whole = |text: &str| {
                    let rects = shown_whole(&output, text, screen);
                    assert_eq!(
                        rects.len(),
                        1,
                        "{size:?}: {text:?} drawn whole once: {rects:?}"
                    );
                    rects[0]
                };
                whole(STORED_3D_LABEL);
                whole(ORIENTATION);
                let size_column = whole("Required torque");
                // The folds of the detailed inputs and of the assumptions, the ways to the other
                // inputs, on the first view at every size (plan review E-4; UX overhaul spec 4.3
                // and 8.2, test 9: the Assumptions view is reachable from the opening tab).
                whole(crate::gui::DETAILED_INPUTS);
                whole(crate::gui::ASSUMPTIONS_FOLD);
                let checks = whole(crate::gui::checks::HEADING);
                let numbers: Vec<egui::Rect> = Tile::ALL
                    .iter()
                    .map(|tile| whole(&tile.number(&inputs, &results).unwrap()))
                    .collect();
                for number in &numbers {
                    assert!(
                        size_column.right() < number.left() && number.right() < checks.left(),
                        "{size:?}: a tile's number {number:?} between {size_column:?} and {checks:?}"
                    );
                }
                // The four tiles take one row where the centre holds four of the narrowest (the
                // laptop and the desktop windows: the smoke run's "four tiles side by side"),
                // else two rows of two (review TQ-1).
                let tops: std::collections::BTreeSet<i32> =
                    numbers.iter().map(|n| n.top().round() as i32).collect();
                let rows = if size.x >= WINDOWS[0].x { 1 } else { 2 };
                assert_eq!(tops.len(), rows, "{size:?}: {numbers:?}");
                // The drawing: the largest plate in the background colour.
                let drawing = flat_shapes(&output)
                    .into_iter()
                    .filter_map(|shape| match shape {
                        egui::Shape::Rect(r) if r.fill == theme::BG => Some(r.rect),
                        _ => None,
                    })
                    .max_by(|a, b| a.area().total_cmp(&b.area()))
                    .expect("the drawing's plate");
                assert!(
                    drawing.height() >= MIN_DRAWING_HEIGHT && drawing.bottom() <= size.y,
                    "{size:?}: {drawing:?}"
                );
                assert!(
                    numbers.iter().all(|n| n.bottom() < drawing.top()),
                    "{size:?}"
                );
                // Each note with a level, under the drawing (decision D-66), is drawn whole: the
                // example design's cap thread caution wraps to two rows at 1024 wide.
                let paused = crate::engine::clamps::PAUSED;
                let geometry = crate::gui::geometry::geometry(&inputs, &results, paused);
                for note in geometry.notes.iter().filter(|note| note.level.is_some()) {
                    whole(&note.text);
                }
            }
        }
    }

    #[test]
    fn the_first_view_draws_at_most_two_hundred_and_fifty_words_at_1280_by_800() {
        // UX overhaul spec 2.1 (goal G-1) and 8.2, test 8, measured as the inventory did: the
        // words the opening screen shows. 608 before this stage (the dashboard's rows, the
        // torque readouts and the welcome strip); 443 after its part 1, the callout list under
        // the drawing kept until its tags are selections (decision D-19); 427 after part 2a's
        // top bar; 272 after part 2b-i moved the Size column's lines into the Explain panel;
        // 200 after part 2b-ii-b moved the callout list and its notes to the Geometry tab.
        use crate::gui::test_support::words_shown;
        let size = WINDOWS[0];
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        for _ in 0..3 {
            frame_on(&ctx, &mut app, size, Vec::new());
        }
        let output = frame_on(&ctx, &mut app, size, Vec::new());
        let words = words_shown(&output, egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        assert!(words <= 250, "{words} words on the first view");
    }

    #[test]
    fn every_start_here_step_s_lesson_line_is_drawn_whole_in_the_app_s_own_font() {
        // Plan review UXP-1 (BL-085): the lesson line under the open equation ("Full lesson:"
        // and the note's title, a link) ran past the Equation panel's edge at 1024 x 700 on
        // the long titles (steps 3, 7 and 12). Each step of the walk, at the three window
        // sizes, in the app's theme and font: no text of the line is cut.
        use crate::engine::explain::notes::START_HERE;
        use crate::gui::explain::FULL_LESSON;
        use crate::gui::explorer::{NEXT, START_HERE_BUTTON, reviewed_note};
        use crate::gui::test_support::{cut_texts, primary_button, text_center};
        use crate::gui::typeset::glyph_safe;
        for size in [WINDOWS[1], WINDOWS[0], egui::vec2(1920.0, 1080.0)] {
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            let click = |app: &mut MagcouplingApp, text: &str| {
                let output = frame_on(&ctx, app, size, Vec::new());
                let at = text_center(&output, text);
                frame_on(&ctx, app, size, vec![egui::Event::PointerMoved(at)]);
                for pressed in [true, false] {
                    frame_on(&ctx, app, size, vec![primary_button(at, pressed)]);
                }
                frame_on(&ctx, app, size, vec![egui::Event::PointerGone]);
                frame_on(&ctx, app, size, Vec::new());
                frame_on(&ctx, app, size, Vec::new())
            };
            frame_on(&ctx, &mut app, size, Vec::new());
            let mut output = click(&mut app, START_HERE_BUTTON);
            for (step, &(id, _)) in START_HERE.iter().enumerate() {
                if step > 0 {
                    output = click(&mut app, NEXT);
                }
                let title = glyph_safe(reviewed_note(id).expect("a reviewed note").title);
                let line = [FULL_LESSON, title.as_str()];
                let cut: Vec<String> = cut_texts(&output, screen)
                    .into_iter()
                    .filter(|cut| {
                        line.iter()
                            .any(|text| cut.starts_with(&format!("{text:?} by")))
                    })
                    .collect();
                assert!(cut.is_empty(), "{size:?}, step {}: cut: {cut:#?}", step + 1);
            }
        }
    }

    #[test]
    fn the_widest_tooltip_stays_inside_the_window_in_the_app_s_own_font() {
        // The whole-branch review (R-1): the Verdict row's equation is the widest a tooltip
        // typesets, 1351 points at the tooltip's size in Inter, and ran off a 1280-point
        // window with the hover text over it.
        use crate::gui::dashboard::hover_text;
        use crate::gui::readouts::{OPEN_HINT, Readouts};
        use crate::gui::test_support::{cut_texts, hover_frames, text_center};
        const VERDICT: &str = "temperature.summary.verdict";
        for size in WINDOWS {
            let ctx = egui::Context::default();
            theme::apply(&ctx);
            fonts::install(&ctx);
            let (_, output) = hover_frames(
                &ctx,
                size,
                |output| text_center(output, "a value"),
                |ui| {
                    let rect = ui.label("a value").rect;
                    Readouts::default().show_over(ui, rect, VERDICT, |details| {
                        hover_text(VERDICT, details).unwrap_or_default()
                    });
                },
            );
            // The tooltip is up, with its equation.
            let texts = drawn_texts(&output);
            assert!(texts.contains(&OPEN_HINT.to_owned()), "{size:?}: {texts:?}");
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
            let cut = cut_texts(&output, screen);
            assert!(cut.is_empty(), "{size:?}: cut: {cut:#?}");
        }
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
        // The two checks that read its 3D fields (BL-080: its own since 2026-11-22). The
        // temperature check: the magnet limit and its margin over the hot-day start. The shaft
        // joints check: the bonded inner hub's fitting temperature, which the input caps, not
        // the single-ring field (its cap is the onset less the cure margin).
        let summary = &results.temperature.summary;
        assert!(close(summary.magnet_limit_C, 94.94, 0.005));
        assert!(close(summary.margin_hot_day_C, 29.94, 0.005));
        assert!(close(results.temperature.slip_loss.total_W, 2.252, 0.0005));
        let joint = &results.shaft_joint;
        assert!(close(joint.hub_fit_temp_C, 93.27, 0.005));
        assert_eq!(joint.hub_fit_allowed_C, inputs.shaft_joint.hub_fit_max_C);
        let field_cap = results.temperature.demag.inner_onset_single_ring_C
            - crate::engine::temperature::CURE_MARGIN_MIN_C;
        assert!(close(field_cap, 131.28, 0.005), "{field_cap}");
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
        // UX overhaul spec 7.1, with the keyway depth and the 3D fields of BL-080: the seven
        // inputs it sizes and its fourteen 3D fields differ from the spreadsheet's, each one a
        // value a slider step reaches, so a user can walk from one design to the other.
        let (start, example) = (starting_design(), DesignInputs::default());
        let mut differing = Vec::new();
        start.visit("", &mut |path, _, value| {
            if example.get(path) != Some(value) {
                differing.push(path.to_owned());
            }
        });
        differing.sort();
        assert_eq!(differing, starting_differences());
        let fields = RECORDED_FIELDS
            .map(|(path, raw)| (path, number(&example, path), as_entered(path, raw)));
        for (path, from, to) in STARTING_CHANGES.into_iter().chain(fields) {
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
        // The example design is an item of the Design menu, not on the first view.
        assert_eq!(count(crate::gui::OPEN_EXAMPLE), 0, "{texts:?}");
        assert_eq!(count(crate::gui::DESIGN_MENU), 1, "{texts:?}");
        // The readouts show the requirement and its margin.
        assert_eq!(count("1.35 N·m"), 1, "{texts:?}");
        // The gap breaks only between its torque and its percent.
        assert_eq!(count("margin +0.10 N·m"), 1, "{texts:?}");
        assert_eq!(count("(+7.3 %)"), 1, "{texts:?}");
        // The Hot-low tile says what it must reach, and the Explain panel is on its
        // orientation: the torque card is one click away, on that tile.
        assert_eq!(count(crate::gui::tiles::NEEDS), 1, "{texts:?}");
        assert_eq!(count("1.350 N·m"), 1, "{texts:?}");
        assert_eq!(count(crate::gui::explain::ORIENTATION), 1, "{texts:?}");
        assert_eq!(count(crate::gui::torque_card::CAPTION), 0, "{texts:?}");
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
        // Nor the cap's skirt caution (BL-082): its skirt, 0.55 mm, is over the minimum.
        use crate::gui::geometry::{MIN_CAP_SKIRT_MM, SKIRT_CAUTION_START, wall_cautions};
        let skirt = (inputs.metal.cap_od_mm - inputs.metal.cap_thread_dia_mm) / 2.0;
        assert!(
            (skirt - 0.55).abs() < 1e-9 && skirt > MIN_CAP_SKIRT_MM,
            "{skirt}"
        );
        assert_eq!(
            wall_cautions(&inputs, &compute_all(&inputs)),
            Vec::<String>::new()
        );
        for size in [LAPTOP, egui::vec2(LAPTOP.x, 20_000.0)] {
            let ctx = egui::Context::default();
            let mut app = MagcouplingApp::new(&ctx);
            frame_on(&ctx, &mut app, size, Vec::new());
            let output = frame_on(&ctx, &mut app, size, Vec::new());
            let texts = drawn_texts(&output);
            assert!(texts.iter().any(|t| t == STARTING_STATUS), "{texts:?}");
            for start in ["Wall past keyway", "Exploration only", SKIRT_CAUTION_START] {
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
        assert_eq!(flagged(&starting_design()), starting_differences());
        assert_eq!(flagged(&DesignInputs::default()), Vec::<String>::new());
    }

    #[test]
    fn the_starting_design_stores_the_3d_fields_of_its_own_geometry() {
        // The owner's ruling of 2026-11-22 (BL-080; register row A-18): "passes all 9 checks"
        // rests on 3D fields computed for the starting design, not on the example design's.
        // The tool cannot compute them (the 3D routine is not ported), so they are held to a
        // recorded run of the reference routine.
        use crate::engine::meta::ResultSet;
        use crate::engine::size_controls::{dimensions, resize_diameter, resize_length};
        use crate::gui::dashboard::{STARTING_3D_FIELDS, STORED_3D_INPUTS};
        let start = starting_design();
        let results = compute_all(&start);
        // The run is of this geometry: everything the routine read is what the engine has now.
        // The routine gives both rings the inner ring's length, width and Br.
        for (path, read) in RECORDED_READS {
            let now = match start.get(path).or_else(|| results.get(path)) {
                Some(Value::Num(x)) => x,
                Some(Value::Int(i)) => i as f64,
                other => panic!("{path}: {other:?}"),
            };
            assert!(
                (now - read).abs() < 1e-9,
                "{path}: {now}, the run read {read}"
            );
        }
        let model = &results.model;
        assert_eq!(
            (
                model.outer_length_mm,
                model.outer_width_mm,
                model.outer_br_T
            ),
            (
                model.inner_length_mm,
                model.inner_width_mm,
                model.inner_br_T
            )
        );
        // Every stored 3D field holds the run's value as entered, its own and not the example
        // design's, whose values this does not move.
        assert_eq!(RECORDED_FIELDS.map(|(path, _)| path), STORED_3D_INPUTS);
        let example = DesignInputs::default();
        for ((path, raw), stored) in RECORDED_FIELDS.into_iter().zip(STARTING_3D_FIELDS) {
            assert_eq!(stored, as_entered(path, raw), "{path}");
            assert_eq!(number(&start, path), stored, "{path}");
            assert_ne!(number(&example, path), stored, "{path}");
        }
        assert_eq!(
            number(&example, "temperature.demag.h_rev_likepole_kA_m"),
            863.0
        );
        // The fields never follow an edit: the Size tab's diameter and length sliders move
        // several inputs at once, and none of them is a 3D field.
        let size = dimensions(&start).expect("the starting design has a size");
        let edits = [
            resize_diameter(&start, size.diameter_mm + 1.0),
            resize_length(&start, size.overall_length_mm + 1.0),
        ];
        for edited in edits {
            let edited = edited.expect("a size inside its slider's range");
            assert!(edited != start, "the slider moved the design");
            for path in STORED_3D_INPUTS {
                assert_eq!(edited.get(path), start.get(path), "{path}");
            }
        }
        // The temperature Problem names the design whose fields it read, here the starting
        // design's, on a design edited away from it.
        let mut hot = start.clone();
        crate::gui::test_support::hot_day(&mut hot);
        let guide = crate::gui::guidance::guide(crate::gui::guidance::TEMPERATURE)
            .expect("the temperature guide");
        let problem = (guide.problem)(&hot, &compute_all(&hot)).expect("a hot day fails it");
        assert!(
            problem.ends_with(
                "(slip fields from the starting design; enter this design's own to confirm)."
            ),
            "{problem}"
        );
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
        // The Show spreadsheet details toggle (spec S2-5), in the Design menu (UX overhaul
        // decision C-12).
        assert!(
            script.contains(crate::gui::SHOW_DETAILS),
            "the smoke prompt names the details toggle"
        );
        // UX overhaul spec 4.2: the prompt names the top bar's buttons by the words the page
        // draws, finds the menu's items under Design, and no longer looks for the buttons that
        // left the bar.
        {
            use crate::gui::explain::LEARN;
            use crate::gui::{EXPORT, LOAD_DESIGN, RESET_TO_START, SHARE};
            for (what, text) in [
                ("the Design menu", crate::gui::DESIGN_MENU),
                ("the reset", RESET_TO_START),
                ("the file's load", LOAD_DESIGN),
                ("Share", SHARE),
                ("Export", EXPORT),
                ("Learn", LEARN),
            ] {
                assert!(
                    script.contains(&format!("\"{text}\"")),
                    "the smoke prompt names {what}: {text:?}"
                );
            }
            for gone in [
                "\"Reset all\"",
                "\"Copy share link\"",
                "\"Equation panel\" button",
                "after \"Reset all\"",
            ] {
                assert!(
                    !script.contains(gone),
                    "the smoke prompt still names {gone}"
                );
            }
            // The lesson link shows its note in the Explain panel (decision C-11), where the
            // step looks for it.
            let link = script
                .find("lesson_link_opened=true")
                .expect("the link's verdict");
            let verdict = &script[link..];
            let verdict = &verdict[..verdict.find('.').unwrap_or(verdict.len())];
            assert!(
                verdict.contains("Explain panel") && !verdict.contains("Equation panel"),
                "the note is judged in the Explain panel: {verdict}"
            );
        }
        // Spec R14 (guidance slice 2): the steps name a note's What to change and its sources
        // fold, the Geometry tab and the check list's last row as the page draws them, so the
        // prompt cannot go stale unnoticed.
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
        assert!(script.contains(fit.label), "the check list's last row");
        // UX overhaul slice B, stage 2 (spec 4.4, 4.5 and 5): the first view and the check's
        // entry are named by the words the page draws: the right column's two headings, the
        // orientation's opening words, the Hot-low tile's line, the entry's headings and its
        // lesson link; and the card step looks for the card under its heading in the entry.
        {
            use crate::gui::explain::{
                FULL_LESSON, NUMBERS, ORIENTATION, WHAT_A_CHANGE_DOES, WHAT_IT_IS,
            };
            let opening: String = ORIENTATION
                .split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ");
            for (what, text) in [
                ("the check list's heading", crate::gui::checks::HEADING),
                ("the Explain panel's heading", crate::gui::explain::HEADING),
                ("the orientation's opening words", opening.as_str()),
                ("the Hot-low tile's line", crate::gui::tiles::NEEDS),
                ("the entry's first heading", WHAT_IT_IS),
                ("the entry's numbers", NUMBERS),
                ("the entry's lesson link", FULL_LESSON),
                ("the tile's entry's card heading", WHAT_A_CHANGE_DOES),
            ] {
                assert!(
                    script.contains(text),
                    "the smoke prompt names {what}: {text}"
                );
            }
            let hot_low = crate::gui::guidance::guide("metal.hot_min_check").expect("its guide");
            assert!(script.contains(hot_low.label), "the check the step opens");
            // The link the step clicks opens a reviewed note with "What to change" lines, not
            // the plain "no note yet" text, which is no link (review TQ-3).
            let crate::gui::guidance::Learn::Note(id) = hot_low.learn else {
                panic!("the hot-low check's lesson is a note: {:?}", hot_low.learn);
            };
            let note = crate::gui::explorer::reviewed_note(id).expect("a reviewed note");
            assert!(!note.change.is_empty(), "{id}: its What to change lines");
            let entry = script
                .find("check_entry=true")
                .expect("the entry's verdict");
            let link = script
                .find("lesson_link_opened=true")
                .expect("the link's verdict");
            assert!(
                entry < link && script[entry..link].contains(FULL_LESSON),
                "the check's entry is judged before its lesson link is clicked"
            );
            // The dashboard is gone: the prompt does not send the agent looking for it.
            assert!(!script.to_lowercase().contains("dashboard"));
        }
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
            between("tag_entry=true was judged", "page.goto(").contains(collect),
            "the first page's console messages are collected before the link is opened again"
        );
        // UX overhaul slice B, stage 2, part 2b-ii (spec 5.2): after the tooltip a click on the
        // label it showed beside selects that input, and the step finds the input's entry in the
        // Explain panel by its trace link, typed once as TRACE_LINK.
        {
            use crate::gui::explain::TRACE_LINK;
            assert_eq!(quoted("TRACE_LINK"), TRACE_LINK, "the script's TRACE_LINK");
            assert_eq!(
                script.matches(TRACE_LINK).count(),
                1,
                "the smoke prompt types the trace link once, in TRACE_LINK"
            );
            assert!(
                between("tooltip_plain=true only if", "input_entry=true only if")
                    .contains("click that label"),
                "the input's entry is judged after the tooltip, from a click on its label"
            );
            assert!(
                between("input_entry=true only if", "page.goto(").contains("Explain panel"),
                "the input's entry is judged in the Explain panel"
            );
        }
        // UX overhaul slice B, stage 2, part 2b-ii-b (spec 4.4, 5.2): after the input's entry
        // a click on tag 1 of the drawing (the face gap's dimension) selects its result, and the
        // step finds the result's entry in the Explain panel by the name typed once as
        // TAG_LABEL, with its Equation link and no Equation panel opened.
        {
            use crate::gui::explain::EQUATION;
            let label = crate::gui::plain_text::label_for("model.face_gap_mm");
            assert_eq!(quoted("TAG_LABEL"), label, "the script's TAG_LABEL");
            assert_eq!(
                script.matches(label).count(),
                1,
                "the smoke prompt types the face gap's name once, in TAG_LABEL"
            );
            let step = between("tag_entry=true only if", "tag_entry=true was judged");
            assert!(
                between("input_entry=true only if", "tag_entry=true only if")
                    .contains("click the number \"1\""),
                "the tag's entry is judged after the input's, from a click on tag 1"
            );
            assert!(
                step.contains("Explain panel") && step.contains(&format!("\"{EQUATION}\"")),
                "the tag's entry is judged in the Explain panel by its Equation link: {step}"
            );
        }
        // UX overhaul slice B, stage 2, part 2b-ii-c (spec 4.3): on the Geometry tab the step
        // judges the left side as the Size column, by the words the page draws there (each
        // control's label and the two folds, with no "Design inputs" tab), before it opens the
        // Detailed inputs fold for the row the tooltip step hovers.
        {
            use crate::gui::{ASSUMPTIONS_FOLD, DETAILED_INPUTS};
            let tab = format!(
                "click the \"{}\" tab",
                crate::gui::CentreView::Geometry.label()
            );
            let step = between("same_column=true only if", "tooltip_plain=true only if");
            assert!(
                between(&tab, "tooltip_plain=true only if").contains("same_column=true only if"),
                "the left side is judged on the Geometry tab, before the tooltip"
            );
            for row in crate::gui::SIZE_ROWS {
                assert!(
                    step.contains(&format!("\"{}\"", row.label)),
                    "the step names the control {:?}",
                    row.label
                );
            }
            for fold in [DETAILED_INPUTS, ASSUMPTIONS_FOLD] {
                assert!(
                    step.contains(&format!("\"{fold}\"")),
                    "the step names the fold {fold:?}"
                );
            }
            assert!(
                step.contains("no tab named \"Design inputs\""),
                "the step looks for the tabs that swapped the side"
            );
            assert!(
                step.contains(&format!("click the fold \"{DETAILED_INPUTS}\"")),
                "the tooltip's row is in the Detailed inputs, opened first"
            );
        }
        // UX overhaul slice B, stage 2, part 2b-iii-a (spec 4.4, decision C-9): after the lesson
        // link and before the Geometry tab the step opens the Plots tab, judges its tab row and
        // its picker by the words the page draws (the tab row as the app draws it with the
        // shaft-clamp section paused, the five plots' names in their order), then picks Slip
        // heating and finds its plot by a name of its legend.
        {
            use crate::gui::CentreView;
            use crate::gui::plots::{HIGH_CASE, PLOTS, PlotKind};
            let open = format!("click \"{PLOTS}\" in the row of tabs");
            let geometry = format!("click the \"{}\" tab", CentreView::Geometry.label());
            assert!(
                between("lesson_link_opened=true only if", &geometry).contains(&open),
                "the Plots tab is opened after the lesson link, before the Geometry tab"
            );
            let step = between(&open, "same_column=true only if");
            // A list as the prompt writes it: "a", "b" and "c".
            let listed = |names: &[&str]| {
                let quoted: Vec<String> = names.iter().map(|name| format!("\"{name}\"")).collect();
                let (last, rest) = quoted.split_last().expect("a name");
                format!("{} and {last}", rest.join(", "))
            };
            let tabs: Vec<&str> = CentreView::tab_row(crate::engine::clamps::PAUSED)
                .map(CentreView::tab)
                .collect();
            let plots: Vec<&str> = PlotKind::ALL.iter().map(|kind| kind.label()).collect();
            for (what, list) in [
                ("the tab row", listed(&tabs)),
                ("the picker", listed(&plots)),
            ] {
                assert!(
                    step.contains(&list),
                    "the step names {what} in order: {list}"
                );
            }
            let pick = PlotKind::SlipHeating.label();
            for (what, text) in [
                ("the pick", format!("click \"{pick}\"")),
                ("the plot's legend", format!("\"{HIGH_CASE}\"")),
                ("the verdict", "plots_picker=true only if".to_owned()),
            ] {
                assert!(step.contains(&text), "the step names {what}: {text}");
            }
        }
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
        // UX overhaul slice B, stage 1 (spec 6): on the clean page the script reads the page's
        // background, typed once as PAGE_BACKGROUND and equal to the theme's panel colour as a
        // browser reports it, and judges the look in the screenshot: the brand mark before the
        // title and the line under the active tab.
        {
            let [r, g, b, _] = theme::PANEL.to_array();
            let background = format!("rgb({r}, {g}, {b})");
            assert_eq!(
                quoted("PAGE_BACKGROUND"),
                background,
                "the script's PAGE_BACKGROUND"
            );
            assert_eq!(
                script.matches(background.as_str()).count(),
                1,
                "the smoke prompt types the page's background once, in PAGE_BACKGROUND"
            );
            let clean_page = between("starting_design=true", "it carries a share link");
            for (what, text) in [
                ("judges the look", "look=true only if"),
                (
                    "reads the page's background",
                    "getComputedStyle(document.body).backgroundColor",
                ),
                ("names the title", "Magnetic coupling calculator"),
                (
                    "names the active tab",
                    crate::gui::CentreView::SizeAndTorque.label(),
                ),
            ] {
                assert!(
                    clean_page.contains(text),
                    "the clean page's step {what}: {text}"
                );
            }
        }
        // UX overhaul slice B, stage 2, part 2b-iii-b (the stage 1 review's S-4, S-1, E-4 and
        // R-6): on the clean page the look's step also reads that the page's loading line is
        // gone (its id typed once, as LOADING_ID, equal to the page's), and judges the heavier
        // title and heading and the grey edge of Auto's box and of the value boxes.
        {
            assert_eq!(quoted("LOADING_ID"), LOADING_ID, "the script's LOADING_ID");
            let look = between("Then the look:", "Collect the console");
            let gone = "document.getElementById('${LOADING_ID}') === null";
            for (what, text) in [
                ("reads that the loading line is gone", gone),
                (
                    "judges the heavier title and heading",
                    "heavier than the text under them",
                ),
                ("names Auto's box", crate::gui::AUTO),
                ("judges the controls' edges", "grey edge"),
            ] {
                assert!(look.contains(text), "the look's step {what}: {text}");
            }
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
    fn the_hub_page_does_not_name_the_paused_clamp_section() {
        // UX overhaul spec 9.2 and 8.3: no "clamp" on screen while the section is paused
        // (`engine::clamps::PAUSED`), and the hub's card for the calculator is a screen too.
        let hub = include_str!("../../linkage-sim-rs/web/tools/index.html");
        assert!(hub.contains("Magnetic Coupling Calculator"));
        assert!(!hub.to_lowercase().contains("clamp"));
    }

    #[test]
    fn the_page_wears_the_app_s_colours_and_a_loading_line_until_the_app_draws() {
        // The stage 1 review (S-4): while the wasm file loads, the page was charcoal with
        // nothing to say it was working, and the browser's own bars kept their colour. The page
        // names its language, gives the browser the panel colour and a dark scheme from its
        // head (before any style, so nothing flashes white), and shows one line in the muted
        // colour across the middle of the page, after the canvas. The app removes the line at
        // the end of its first frame (`remove_loading_line`, on wasm32 only), so the browser
        // shows the app and the line's removal in one paint; the page's script only starts the
        // app (a browser run checks both).
        let page = include_str!("../../linkage-sim-rs/web/tools/magcoupler/index.html");
        use theme::hex;
        assert!(
            page.contains("<html lang=\"en\">"),
            "index.html names its language"
        );
        let head = &page[..page.find("</head>").expect("a head")];
        for (what, meta) in [
            (
                "the browser's colour",
                format!(
                    "<meta name=\"theme-color\" content=\"{}\">",
                    hex(theme::PANEL)
                ),
            ),
            (
                "the dark scheme",
                "<meta name=\"color-scheme\" content=\"dark\">".to_owned(),
            ),
        ] {
            assert!(
                head.contains(&meta),
                "index.html's head gives {what}: {meta}"
            );
        }
        // The line, after the canvas, and its style: the muted colour, centred on the page.
        let line = format!("<p id=\"{LOADING_ID}\">{LOADING_LINE}</p>");
        let canvas = format!("<canvas id=\"{CANVAS_ID}\"></canvas>");
        let at = page.find(&line).expect("index.html has the loading line");
        assert!(page.find(&canvas).is_some_and(|canvas| canvas < at));
        let rule = format!("#{LOADING_ID} {{");
        let rule = &head[head.find(&rule).expect("the line's style")..];
        let rule = &rule[..rule.find('}').expect("its end")];
        for property in [
            format!("color: {};", hex(theme::MUTED)),
            "position: fixed;".to_owned(),
            "text-align: center;".to_owned(),
        ] {
            assert!(rule.contains(&property), "the line's style has {property}");
        }
        // The script starts the app and touches the line nowhere: the app removes it.
        let script = &page[page.find("<script").expect("a script")..];
        assert!(script.contains("init();") && !script.contains(LOADING_ID));
        // A new text on screen: plain words, no gearbox claim, every glyph in the page's font.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        use crate::gui::guidance::plain_text_violations;
        assert_eq!(plain_text_violations(LOADING_LINE), Vec::<&str>::new());
        assert!(!mentions_gearbox_protection(LOADING_LINE));
        assert!(LOADING_LINE.is_ascii());
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
