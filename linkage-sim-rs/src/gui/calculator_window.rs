//! Tools → Magnetic coupling: the magnetic coupling calculator (`magcoupling-rs`'s
//! `gui::MagcouplingPanel`) in an `egui::Window` of the linkage app. Spec:
//! `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, section M5.
//!
//! The calculator's state is its own: the panel is created the first time the window opens (its
//! equation registry is built then, not at the linkage app's start-up) and kept, open or closed,
//! until the app quits. Nothing here reads or writes the linkage model (decision M5-3).
//!
//! Place (decision M5-2): the window, frame included, keeps to the screen below the linkage app's
//! menu bar, so it never covers the Tools menu that toggles it
//! ([`CalculatorWindow::set_menu_bar_bottom`]).
//!
//! Keyboard (decision M5-5): the window has the keyboard from its opening or a press on it (or on
//! the band just outside its frame where egui lets the user grab its edge to resize it) until a
//! press on the linkage app: its panels, its canvas, its menu bar's buttons (File, Edit, View,
//! Image, Tools: a Background panel) or another window. A press on a popup (an open menu's items,
//! a drop-down list or a tooltip: egui's Foreground and Tooltip layers, and its Debug layer; the
//! linkage app's menus and the calculator's drop-downs alike) leaves the keyboard where it is.
//! While the window has the keyboard, the panel acts on
//! Ctrl+Z, Ctrl+Shift+Z and Ctrl+Y (`MagcouplingPanel::set_keyboard_shortcuts`) and, once the panel
//! has drawn, [`CalculatorWindow::show`] takes the frame's key events out but egui's own zoom keys
//! (`UI_ZOOM_KEYS`), so no linkage shortcut or canvas key (undo, save, new, delete, the arrow nudge,
//! F, Escape, Enter) runs on them while Ctrl+Plus, Ctrl+Minus and Ctrl+0 still zoom the whole UI
//! (natively; on the web eframe leaves those keys to the browser, which zooms the page).
//! `LinkageApp::update` therefore shows the window before anything else reads the keyboard (a test
//! in `gui/mod.rs` checks the order). Without the keyboard, or while the window is collapsed to its
//! title bar, the panel leaves the keys alone and the linkage app reads them as before.
//!
//! Known edges, documented rather than handled (backlog BL-039, BL-040): the keyboard follows
//! presses, not egui's keyboard focus, so after Tab moves the focus from one part into the other
//! the keys stay where they were until a click there; and the window, once raised, covers the
//! linkage app's "Recover Unsaved Work?" prompt if that shows at the same time (move or collapse
//! the window to answer it).
//!
//! Files (decision M5-6): the panel's requests are done here. Saving uses the linkage app's
//! `export::download` (a file dialog natively, a download on the web); "Load design" uses
//! [`DesignPicker`] (rfd's file dialog natively, rfd's file picker on the web).

use std::cell::RefCell;
use std::rc::Rc;

use eframe::egui;
use magcoupling::gui::{MagcouplingPanel, PanelRequest};

use super::export::download::{self, DownloadOutcome, FileFilter};

/// The window's title and the Tools menu item.
pub const TITLE: &str = "Magnetic coupling";

/// The URL query parameter that opens a tool when the web app starts (decision M5-4):
/// `?tool=magcoupling`. gui-smoke's linkage step opens the window with it.
pub const TOOL_PARAM: &str = "tool";

/// [`TOOL_PARAM`]'s value that opens this window.
pub const TOOL_MAGCOUPLING: &str = "magcoupling";

/// The path of the calculator's own page next to the linkage app (`web/magcoupling/`).
pub const MAGCOUPLING_PAGE_PATH: &str = "/magcoupling/";

/// Where the window first opens [points], below the menu bar and the two toolbars, and its
/// starting size (decision M5-2). egui keeps the window inside the screen.
const DEFAULT_POS: [f32; 2] = [80.0, 100.0];
const DEFAULT_SIZE: [f32; 2] = [1100.0, 700.0];

/// egui's own keyboard zoom of the whole UI (`egui::gui_zoom`, run at the end of every frame): the
/// window leaves these key events in place while it has the keyboard. In effect natively only:
/// eframe's web backend turns egui's keyboard zoom off and leaves these keys to the browser.
const UI_ZOOM_KEYS: [egui::KeyboardShortcut; 4] = [
    egui::gui_zoom::kb_shortcuts::ZOOM_IN,
    egui::gui_zoom::kb_shortcuts::ZOOM_IN_SECONDARY,
    egui::gui_zoom::kb_shortcuts::ZOOM_OUT,
    egui::gui_zoom::kb_shortcuts::ZOOM_RESET,
];

/// The address the calculator's share links point at on the web app served from `origin`: the
/// calculator's own page on the same server.
pub fn magcoupling_share_base(origin: &str) -> String {
    format!("{origin}{MAGCOUPLING_PAGE_PATH}")
}

/// The window's id (its area's and its layer's).
pub(crate) fn window_id() -> egui::Id {
    egui::Id::new("magcoupling_window")
}

/// The window's layer: a press there gives it the keyboard.
fn window_layer() -> egui::LayerId {
    egui::LayerId::new(egui::Order::Middle, window_id())
}

/// The rect of the window's contents when it first opens (`DEFAULT_POS`, `DEFAULT_SIZE`). The
/// window itself is a little larger: its frame, its title bar and the panel's own minimum width
/// reach about 72 points right of it and 48 below it.
#[cfg(test)]
pub(crate) fn default_rect() -> egui::Rect {
    egui::Rect::from_min_size(DEFAULT_POS.into(), DEFAULT_SIZE.into())
}

/// A point on the host right of and below the window as it first opens, clear of its frame and of
/// the band around it where egui resizes it.
#[cfg(test)]
pub(crate) fn beside_the_window() -> egui::Pos2 {
    default_rect().right_bottom() + egui::vec2(120.0, 50.0)
}

/// The calculator window: whether it is open, its panel, and whether it has the keyboard.
#[derive(Default)]
pub struct CalculatorWindow {
    open: bool,
    /// Created on the first opening, then kept, open or closed, until the app quits.
    panel: Option<MagcouplingPanel>,
    /// Whether the window has the keyboard (module docs); cleared when the window closes.
    keyboard: bool,
    /// Set by opening, until the window has been brought in front of the app's other windows and
    /// areas (the linkage app's welcome screen included).
    raise: bool,
    /// The address the panel's share links point at, applied when the panel is created.
    share_base: Option<String>,
    picker: DesignPicker,
    /// The bottom of the linkage app's menu bar [points], which the window keeps below
    /// ([`CalculatorWindow::set_menu_bar_bottom`]).
    menu_bar_bottom: f32,
}

impl CalculatorWindow {
    /// Whether the window is open.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Opens the window, which then has the keyboard (the first opening creates the panel), or
    /// closes it, which gives the keyboard back. The panel and its undo history are kept.
    pub fn set_open(&mut self, open: bool) {
        self.open = open;
        self.keyboard = open;
        self.raise = open;
        if open && self.panel.is_none() {
            let mut panel = MagcouplingPanel::new();
            if let Some(base) = &self.share_base {
                panel.set_share_base(base.clone());
            }
            self.panel = Some(panel);
        }
    }

    /// Whether the window has the keyboard (module docs).
    pub fn has_keyboard(&self) -> bool {
        self.open && self.keyboard
    }

    /// Sets the address the panel's share links point at (the web app: the calculator's own page
    /// on the same server, [`magcoupling_share_base`]).
    pub fn set_share_base(&mut self, base: impl Into<String>) {
        let base = base.into();
        if let Some(panel) = &mut self.panel {
            panel.set_share_base(base.clone());
        }
        self.share_base = Some(base);
    }

    /// Keeps the window below `bottom` [points], the bottom of the linkage app's menu bar, so it
    /// never covers the Tools menu that toggles it. `menu_bar::draw_menu_bar` hands it over every
    /// frame after the window has shown, so the window keeps below the last frame's menu bar.
    /// Before the menu bar's first frame it keeps to the whole screen: on a screen too short for
    /// it, a window opened at start-up shows cut off at the menu bar for one frame (its second;
    /// egui does not paint a window's first), then whole below it.
    pub fn set_menu_bar_bottom(&mut self, bottom: f32) {
        self.menu_bar_bottom = bottom;
    }

    /// Draws the window when it is open and does the panel's requests; while the window has the
    /// keyboard, takes this frame's key events out once the panel has drawn. Call it before
    /// anything else in the frame reads the keyboard (module docs).
    pub fn show(&mut self, ctx: &egui::Context) {
        if !self.open {
            return;
        }
        self.follow_presses(ctx);
        let Some(panel) = self.panel.as_mut() else {
            return; // set_open(true) always creates the panel
        };
        if let Some(picked) = self.picker.take() {
            open_picked_file(panel, picked);
        }
        panel.set_keyboard_shortcuts(self.keyboard);
        // Where the window may go: the screen below the linkage app's menu bar and the band where
        // egui grabs the window's edge, so it never covers the Tools menu that toggles it, nor
        // takes a press on it for a grab of its edge.
        let screen = ctx.screen_rect();
        let style = ctx.style();
        let top = screen
            .top()
            .max(self.menu_bar_bottom + style.interaction.resize_grab_radius_side);
        let area = egui::Rect::from_min_max(egui::pos2(screen.left(), top), screen.max);
        // egui 0.32 caps a window's contents at its constrain rect less its title bar, not less
        // its frame's margins (14 points), so on a screen smaller than the window the window is
        // larger than that rect and sticks out of it, where egui clips it and takes no press (an
        // area paints and interacts within its constrain rect). Cap the contents at the area
        // less all the window adds around them, so the whole window fits in the area.
        let max_contents = area.size() - window_chrome(ctx, &style);
        // A screen with no room for the window's frame and title bar (on the web a hidden canvas
        // reports 0 x 0; the native backend never sends an empty screen): skip the window this
        // frame. egui would otherwise keep the size it squeezed the window to, and show it back on
        // a real screen at its smallest, at the screen's corner.
        if max_contents.x <= 0.0 || max_contents.y <= 0.0 {
            return;
        }
        let mut open = true;
        let shown = egui::Window::new(TITLE)
            .id(window_id())
            .open(&mut open)
            .default_pos(DEFAULT_POS)
            .default_size(DEFAULT_SIZE)
            .constrain_to(area)
            .max_size(max_contents)
            .show(ctx, |ui| panel.ui(ui));
        // A collapsed window draws no panel (`inner` is `None`), so it takes no keys either.
        let drawn = shown.is_some_and(|response| response.inner.is_some());
        // egui brings a window in front when it first shows, but with it every area that first
        // shows in the same frame (the linkage app's welcome screen at start-up, drawn after the
        // window): bring it in front once more on the next frame.
        if self.raise && ctx.memory(|m| m.areas().visible_last_frame(&window_layer())) {
            ctx.move_to_top(window_layer());
            self.raise = false;
        }
        for request in panel.take_requests() {
            match request {
                PanelRequest::SaveFile {
                    file_name,
                    mime,
                    contents,
                } => {
                    let outcome = download::download_bytes(
                        &file_name,
                        mime,
                        &contents,
                        save_filter(&file_name),
                    );
                    if let Some(report) = report_of(outcome) {
                        panel.report(report);
                    }
                }
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
        if !open {
            self.set_open(false);
        }
        if self.has_keyboard() && drawn {
            ctx.input_mut(|i| {
                i.events.retain(|event| {
                    !matches!(event, egui::Event::Key { .. }) || is_ui_zoom_key(event)
                })
            });
        }
    }

    /// Moves the keyboard by this frame's last pointer press (module docs). The press is read
    /// from the events, so a press released in the same frame still counts.
    fn follow_presses(&mut self, ctx: &egui::Context) {
        let pressed_at = ctx.input(|i| {
            i.events.iter().rev().find_map(|event| match event {
                egui::Event::PointerButton {
                    pos, pressed: true, ..
                } => Some(*pos),
                _ => None,
            })
        });
        let Some(pos) = pressed_at else {
            return;
        };
        let layer = ctx.layer_id_at(pos);
        match layer {
            Some(layer) if layer == window_layer() => self.keyboard = true,
            Some(layer)
                if matches!(
                    layer.order,
                    egui::Order::Foreground | egui::Order::Tooltip | egui::Order::Debug
                ) => {}
            // The linkage app's panels and canvas (a Background layer) take it, unless the press
            // grabs the window's edge, which egui lets the user do from just outside its frame
            // (another window there takes it).
            _ if layer.is_none_or(|layer| layer.order == egui::Order::Background)
                && on_resize_band(ctx, pos) =>
            {
                self.keyboard = true
            }
            _ => self.keyboard = false,
        }
    }

    /// The panel, once the window has been opened.
    #[cfg(test)]
    pub(crate) fn panel(&self) -> Option<&MagcouplingPanel> {
        self.panel.as_ref()
    }

    /// The panel's design, once the window has been opened.
    #[cfg(test)]
    pub(crate) fn design(&self) -> magcoupling::gui::session::Design {
        self.panel().expect("the window was opened").design()
    }

    /// Loads `design` into the panel as a design file, once the window has been opened (the next
    /// frame makes it an undo step of the panel).
    #[cfg(test)]
    pub(crate) fn load_design(&mut self, design: &magcoupling::gui::session::Design) {
        let json = magcoupling::gui::session::design_to_json(design);
        let panel = self.panel.as_mut().expect("the window was opened");
        panel.load_design_file(&json).expect("a valid design");
    }
}

/// What the window adds around its contents [points]: its frame's margins, and its title bar
/// with the line under it, as egui 0.32's `Window::show` lays them out (the title in
/// `TextStyle::Heading`, at least the interact height, inside the frame's inner margins). The
/// small-screen test pins it: there the window fills the screen exactly.
fn window_chrome(ctx: &egui::Context, style: &egui::Style) -> egui::Vec2 {
    let frame = egui::Frame::window(style);
    let title = ctx.fonts(|fonts| {
        egui::RichText::new(TITLE)
            .heading()
            .font_height(fonts, style)
    });
    let title_bar = title.max(style.spacing.interact_size.y) + frame.inner_margin.sum().y;
    frame.total_margin().sum() + egui::vec2(0.0, title_bar + frame.stroke.width)
}

/// Whether `event` is one of egui's own zoom keys (`UI_ZOOM_KEYS`), matched as egui matches them.
fn is_ui_zoom_key(event: &egui::Event) -> bool {
    let egui::Event::Key { key, modifiers, .. } = event else {
        return false;
    };
    UI_ZOOM_KEYS
        .iter()
        .any(|zoom| zoom.logical_key == *key && modifiers.matches_logically(zoom.modifiers))
}

/// Whether `pos` is on the band around the window where egui lets the user grab its edge
/// (`resize_grab_radius_side` out from it) or a corner (`resize_grab_radius_corner` around it) to
/// resize it. The band lies outside the window's area, so `Context::layer_id_at` does not see it.
fn on_resize_band(ctx: &egui::Context, pos: egui::Pos2) -> bool {
    let Some(rect) = ctx.memory(|m| m.area_rect(window_id())) else {
        return false;
    };
    let style = ctx.style();
    let grab = &style.interaction;
    let corner = egui::Vec2::splat(2.0 * grab.resize_grab_radius_corner);
    rect.expand(grab.resize_grab_radius_side).contains(pos)
        || [
            rect.left_top(),
            rect.right_top(),
            rect.left_bottom(),
            rect.right_bottom(),
        ]
        .into_iter()
        .any(|at| egui::Rect::from_center_size(at, corner).contains(pos))
}

/// The native save dialog's filter for a file the panel saves: its design file and results
/// exports are JSON or CSV.
fn save_filter(file_name: &str) -> FileFilter {
    match file_name.rsplit_once('.').map(|(_, extension)| extension) {
        Some("json") => FileFilter {
            label: "JSON",
            extensions: &["json"],
        },
        Some("csv") => FileFilter {
            label: "CSV",
            extensions: &["csv"],
        },
        _ => FileFilter {
            label: "All files",
            extensions: &["*"],
        },
    }
}

/// What the panel shows after a save: the message, the error, or nothing when the user cancelled
/// the dialog.
fn report_of(outcome: DownloadOutcome) -> Option<Result<String, String>> {
    match outcome {
        DownloadOutcome::Saved(message) => Some(Ok(message)),
        DownloadOutcome::Cancelled => None,
        DownloadOutcome::Failed(error) => Some(Err(error)),
    }
}

/// Hands a picked design file's text to the panel (a refused file shows why there and changes
/// nothing); a file that could not be read is reported in the panel.
fn open_picked_file(panel: &mut MagcouplingPanel, picked: Result<String, String>) {
    match picked {
        Ok(text) => {
            if let Err(error) = panel.load_design_file(&text) {
                log::warn!("magcoupling: {error}");
            }
        }
        Err(error) => panel.report(Err(error)),
    }
}

/// A design file the user is picking ("Load design"). Natively rfd's dialog answers at once; on
/// the web rfd's file picker answers later, so the text lands in an inbox the window reads each
/// frame. The pattern of `magcoupling-rs/src/app/files.rs` (the standalone app's picker) with
/// the build cases of `export/download.rs`.
#[derive(Default)]
struct DesignPicker {
    inbox: Rc<RefCell<Option<Result<String, String>>>>,
}

impl DesignPicker {
    /// Shows the file dialog (JSON files); the text arrives in [`DesignPicker::take`].
    #[cfg(feature = "native")]
    fn pick(&mut self, _ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .add_filter("Design", &["json"])
            .pick_file();
        if let Some(path) = picked {
            *self.inbox.borrow_mut() = Some(
                std::fs::read_to_string(&path)
                    .map_err(|error| format!("Could not read {}: {error}", path.display())),
            );
        }
    }

    /// Shows the browser's file picker (JSON files); the text arrives in [`DesignPicker::take`]
    /// once the browser hands the file over.
    #[cfg(target_arch = "wasm32")]
    fn pick(&mut self, ctx: &egui::Context) {
        let inbox = Rc::clone(&self.inbox);
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let dialog = rfd::AsyncFileDialog::new().add_filter("Design", &["json"]);
            if let Some(file) = dialog.pick_file().await {
                let text = String::from_utf8(file.read().await)
                    .map_err(|_| "the file is not UTF-8 text".to_owned());
                *inbox.borrow_mut() = Some(text);
                ctx.request_repaint();
            }
        });
    }

    /// Without a file dialog (a desktop build without the `native` feature): says so.
    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn pick(&mut self, _ctx: &egui::Context) {
        *self.inbox.borrow_mut() = Some(Err(
            "Loading a design file needs the desktop build (feature native) or the web build"
                .to_owned(),
        ));
    }

    /// The picked file's text (or why it could not be read), once.
    fn take(&mut self) -> Option<Result<String, String>> {
        self.inbox.borrow_mut().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::test_support::{
        NATIVE_SCREEN, click_events, drag_events, drew_text, key_tap, magcoupling_gap_design,
        primary_button, screen_input, text_clip_rect, text_rect,
    };
    use magcoupling::gui::results_table::{CSV_FILE_NAME, JSON_FILE_NAME};
    use magcoupling::gui::session::{Design, PUBLIC_BASE_URL, design_to_json};

    /// The panel's heading (magcoupling-rs `gui::panel::HEADING`), a label: a click on it does
    /// nothing but give the window the keyboard.
    const HEADING: &str = "Magnetic coupling calculator";

    /// Where the host draws a Foreground area, standing in for an open menu or a drop-down list.
    const POPUP_AT: egui::Pos2 = egui::pos2(1250.0, 40.0);

    /// One host frame with `events`: the window first, as in `LinkageApp::update`, then a
    /// Foreground area and a central panel for the linkage app. Returns what egui painted and
    /// the keys the host still saw pressed after the window.
    fn frame(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        events: Vec<egui::Event>,
    ) -> (egui::FullOutput, Vec<egui::Key>) {
        let mut keys = Vec::new();
        let output = ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| {
            window.show(ctx);
            keys = ctx.input(|i| {
                i.events
                    .iter()
                    .filter_map(|event| match event {
                        egui::Event::Key {
                            key, pressed: true, ..
                        } => Some(*key),
                        _ => None,
                    })
                    .collect()
            });
            egui::Area::new(egui::Id::new("test_popup"))
                .order(egui::Order::Foreground)
                .fixed_pos(POPUP_AT)
                .show(ctx, |ui| ui.label("popup"));
            egui::CentralPanel::default().show(ctx, |ui| ui.label("linkage"));
        });
        (output, keys)
    }

    /// One frame of the window alone on a screen of `size` [points].
    fn window_frame(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        size: egui::Vec2,
    ) -> egui::FullOutput {
        ctx.run(screen_input(Vec::new(), size), |ctx| window.show(ctx))
    }

    /// One frame of the window and then an area on the layer `order` of `size` at `at` [points],
    /// drawn after the window as the linkage app draws its welcome screen and its other windows
    /// (`Order::Middle`), with `events`.
    fn frame_with_area(
        ctx: &egui::Context,
        window: &mut CalculatorWindow,
        order: egui::Order,
        at: egui::Pos2,
        size: egui::Vec2,
        events: Vec<egui::Event>,
    ) {
        let _ = ctx.run(screen_input(events, NATIVE_SCREEN), |ctx| {
            window.show(ctx);
            egui::Area::new(egui::Id::new("test_area"))
                .order(order)
                .fixed_pos(at)
                .show(ctx, |ui| {
                    ui.allocate_space(size);
                });
        });
    }

    /// Opens the window and runs the two frames a window takes to size itself and paint.
    fn opened(ctx: &egui::Context) -> (CalculatorWindow, egui::FullOutput) {
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        frame(ctx, &mut window, Vec::new());
        let (output, _) = frame(ctx, &mut window, Vec::new());
        (window, output)
    }

    /// A press at `at` (with the pointer moved there first), then its release.
    fn press(ctx: &egui::Context, window: &mut CalculatorWindow, at: egui::Pos2) {
        for events in click_events(at) {
            frame(ctx, window, events);
        }
    }

    /// Loads the design with the face gap at `gap_mm` and runs the frame that makes it an undo
    /// step of the panel.
    fn load_gap(ctx: &egui::Context, window: &mut CalculatorWindow, gap_mm: f64) {
        window.load_design(&magcoupling_gap_design(gap_mm));
        frame(ctx, window, Vec::new());
    }

    fn window_rect(ctx: &egui::Context) -> egui::Rect {
        ctx.memory(|m| m.area_rect(window_id()))
            .expect("the window is shown")
    }

    /// Asserts that the window fills `room` (within egui's rounding) and that egui painted all of
    /// it in `output` (its title's clip rect, its area's, holds the whole window).
    fn assert_fills(ctx: &egui::Context, output: &egui::FullOutput, room: egui::Rect, when: &str) {
        let rect = window_rect(ctx);
        assert!(
            (rect.min - room.min).length() < 0.5 && (rect.max - room.max).length() < 0.5,
            "{when}: the window {rect:?} does not fill {room:?}"
        );
        let clip = text_clip_rect(output, TITLE).expect("the title");
        assert!(
            clip.contains_rect(rect),
            "{when}: egui clips the window {rect:?} to {clip:?}"
        );
    }

    #[test]
    fn the_window_is_closed_until_opened_and_then_shows_the_panel() {
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        assert!(!window.is_open());
        assert!(
            window.panel().is_none(),
            "no panel before the first opening"
        );
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(!drew_text(&output, TITLE));
        assert!(!window.has_keyboard());

        let (window, output) = opened(&ctx);
        assert!(window.is_open());
        assert!(drew_text(&output, TITLE), "the window's title");
        assert!(drew_text(&output, HEADING), "the panel inside it");
        assert!(window.has_keyboard(), "an opened window has the keyboard");
    }

    #[test]
    fn an_opened_window_is_in_front_of_what_the_app_draws_after_it() {
        // At start-up (?tool=magcoupling) the linkage app's welcome screen, an area drawn after
        // the window, would otherwise cover it.
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        let inside = default_rect().center();
        for _ in 0..3 {
            let welcome_at = inside - egui::vec2(160.0, 120.0);
            frame_with_area(
                &ctx,
                &mut window,
                egui::Order::Middle,
                welcome_at,
                egui::vec2(320.0, 240.0),
                Vec::new(),
            );
        }
        assert_eq!(ctx.layer_id_at(inside), Some(window_layer()));
    }

    #[test]
    fn a_small_screen_keeps_the_whole_window_on_it() {
        // A browser window smaller than the window's starting size (1100 x 700 at 80, 100): the
        // window, frame included, fills the screen below the band where egui grabs its top edge
        // (exactly: this pins `window_chrome`), and egui paints all of it.
        let ctx = egui::Context::default();
        let mut window = CalculatorWindow::default();
        window.set_open(true);
        let size = egui::vec2(800.0, 600.0);
        let band = ctx.style().interaction.resize_grab_radius_side;
        let room = egui::Rect::from_min_max(egui::pos2(0.0, band), size.to_pos2());
        let mut output = window_frame(&ctx, &mut window, size);
        for _ in 0..2 {
            output = window_frame(&ctx, &mut window, size);
        }
        assert_fills(&ctx, &output, room, "opened");
        assert!(drew_text(&output, HEADING));
        // Dragged by its title bar past the top left corner.
        let title = text_rect(&output, TITLE).expect("the title").center();
        for events in drag_events(title, egui::pos2(-50.0, -50.0), 20) {
            let _ = ctx.run(screen_input(events, size), |ctx| window.show(ctx));
        }
        let output = window_frame(&ctx, &mut window, size);
        assert_fills(&ctx, &output, room, "dragged to the corner");
    }

    #[test]
    fn a_screen_without_room_skips_the_window_and_keeps_its_place_and_size() {
        // On the web a hidden canvas reports a 0 x 0 screen; the window comes back as it was.
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let before = window_rect(&ctx);
        let output = window_frame(&ctx, &mut window, egui::Vec2::ZERO);
        assert!(
            !drew_text(&output, HEADING),
            "nothing drawn on a 0 x 0 screen"
        );
        frame(&ctx, &mut window, Vec::new());
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert_eq!(window_rect(&ctx), before, "back in its place, at its size");
        assert!(drew_text(&output, HEADING));
    }

    #[test]
    fn closing_the_window_with_a_focused_field_gives_the_keys_back() {
        let ctx = egui::Context::default();
        let (mut window, output) = opened(&ctx);
        // The inner magnet part's text field (the first text drawn with the part's name).
        let field = text_rect(&output, "B842SH")
            .expect("the part field")
            .center();
        press(&ctx, &mut window, field);
        assert!(
            ctx.wants_keyboard_input(),
            "the field has the keyboard focus"
        );

        window.set_open(false);
        frame(&ctx, &mut window, Vec::new());
        frame(&ctx, &mut window, Vec::new());
        assert!(
            !ctx.wants_keyboard_input(),
            "the closed window's field lost its focus"
        );
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Delete, egui::Modifiers::NONE),
        );
        assert_eq!(keys, [egui::Key::Delete]);
    }

    #[test]
    fn closing_and_reopening_keeps_the_design_and_its_undo_history() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);

        window.set_open(false);
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(
            !drew_text(&output, HEADING),
            "a closed window draws nothing"
        );
        assert!(!window.has_keyboard(), "closing gives the keyboard back");

        window.set_open(true);
        frame(&ctx, &mut window, Vec::new());
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(drew_text(&output, HEADING));
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the design is kept"
        );
        // The undo history too: Ctrl+Z (the reopened window has the keyboard) undoes the load.
        frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(window.design(), Design::default());
    }

    #[test]
    fn the_keyboard_follows_presses_and_popups_leave_it_alone() {
        let ctx = egui::Context::default();
        let (mut window, output) = opened(&ctx);
        let heading = text_rect(&output, HEADING).expect("the heading").center();
        let popup = ctx
            .memory(|m| m.area_rect(egui::Id::new("test_popup")))
            .expect("the stand-in popup")
            .center();

        press(&ctx, &mut window, beside_the_window());
        assert!(
            !window.has_keyboard(),
            "a press on the linkage app takes it"
        );
        press(&ctx, &mut window, popup);
        assert!(
            !window.has_keyboard(),
            "an open menu or a drop-down leaves it with the linkage app"
        );
        press(&ctx, &mut window, heading);
        assert!(window.has_keyboard(), "a press on the window gives it");
        press(&ctx, &mut window, popup);
        assert!(
            window.has_keyboard(),
            "an open menu or a drop-down leaves it with the window"
        );
        // A press released in the same frame counts too.
        let outside = beside_the_window();
        frame(
            &ctx,
            &mut window,
            vec![
                primary_button(outside, true),
                primary_button(outside, false),
            ],
        );
        assert!(!window.has_keyboard());
    }

    #[test]
    fn a_press_on_a_tooltip_or_egui_s_debug_layer_leaves_the_keyboard_where_it_is() {
        // Popups beyond the menus and drop-downs (the Foreground layer, the previous test): egui's
        // Tooltip and Debug layers. A Middle layer (another window) is the positive control.
        for (order, keeps) in [
            (egui::Order::Tooltip, true),
            (egui::Order::Debug, true),
            (egui::Order::Middle, false),
        ] {
            let ctx = egui::Context::default();
            let (mut window, _) = opened(&ctx);
            let at = beside_the_window();
            for events in [Vec::new(), Vec::new()]
                .into_iter()
                .chain(click_events(at + egui::vec2(20.0, 20.0)))
            {
                frame_with_area(&ctx, &mut window, order, at, egui::vec2(40.0, 40.0), events);
            }
            assert_eq!(window.has_keyboard(), keeps, "a press on a {order:?} layer");
        }
    }

    #[test]
    fn a_press_on_the_band_where_egui_resizes_the_window_gives_it_the_keyboard() {
        // egui lets the user grab the window's edge from up to 5 points outside its frame and a
        // corner from up to 10 points around it (its Interaction style's grab radii).
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let right_of = |ctx: &egui::Context, points: f32| {
            let rect = window_rect(ctx);
            egui::pos2(rect.right() + points, rect.center().y)
        };

        press(&ctx, &mut window, beside_the_window());
        let before = window_rect(&ctx);
        press(&ctx, &mut window, right_of(&ctx, 3.0));
        assert!(window.has_keyboard(), "the right edge's band gives it");
        assert!(
            window_rect(&ctx).right() > before.right(),
            "egui took the press for a grab of the edge (the edge jumps to the pointer)"
        );
        press(&ctx, &mut window, beside_the_window());
        let corner = window_rect(&ctx).right_bottom() + egui::vec2(7.0, 7.0);
        press(&ctx, &mut window, corner);
        assert!(window.has_keyboard(), "the corner's band gives it");
        press(&ctx, &mut window, right_of(&ctx, 8.0));
        assert!(
            !window.has_keyboard(),
            "beyond the band, the linkage app takes it"
        );
    }

    #[test]
    fn another_window_over_the_band_takes_the_keyboard() {
        // egui gives a press there to the window on top: here an area just right of the
        // calculator's edge, as another linkage window could be.
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let rect = window_rect(&ctx);
        let neighbour_at = egui::pos2(rect.right() + 1.0, rect.center().y - 20.0);
        let press_at = egui::pos2(rect.right() + 3.0, rect.center().y);
        for events in [Vec::new(), Vec::new()]
            .into_iter()
            .chain(click_events(press_at))
        {
            frame_with_area(
                &ctx,
                &mut window,
                egui::Order::Middle,
                neighbour_at,
                egui::vec2(40.0, 40.0),
                events,
            );
        }
        assert!(!window.has_keyboard(), "the window on top took it");
    }

    #[test]
    fn with_the_keyboard_the_panel_undoes_and_the_host_sees_no_key() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(
            window.design(),
            Design::default(),
            "the panel undid the load"
        );
        assert!(keys.is_empty(), "the host saw {keys:?}");
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Y, egui::Modifiers::COMMAND),
        );
        assert_eq!(window.design(), magcoupling_gap_design(2.0), "and redid it");
        assert!(keys.is_empty(), "the host saw {keys:?}");
        // Minus alone is no zoom key: only Ctrl+Minus is (the next test).
        for key in [
            egui::Key::ArrowRight,
            egui::Key::Delete,
            egui::Key::F,
            egui::Key::Escape,
            egui::Key::Minus,
        ] {
            let (_, keys) = frame(&ctx, &mut window, key_tap(key, egui::Modifiers::NONE));
            assert!(keys.is_empty(), "{key:?}: the host saw {keys:?}");
        }
    }

    #[test]
    fn with_the_keyboard_egui_s_zoom_keys_still_zoom_the_ui() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        assert!(window.has_keyboard());
        for (key, zoom) in [
            (egui::Key::Plus, 1.1),
            (egui::Key::Equals, 1.2),
            (egui::Key::Minus, 1.1),
            (egui::Key::Num0, 1.0),
        ] {
            let (_, keys) = frame(&ctx, &mut window, key_tap(key, egui::Modifiers::COMMAND));
            assert_eq!(keys, [key], "Ctrl+{key:?} is left for egui");
            // egui zooms at the end of the frame and applies the new factor at the next one.
            frame(&ctx, &mut window, Vec::new());
            assert_eq!(ctx.zoom_factor(), zoom, "Ctrl+{key:?}");
        }
    }

    #[test]
    fn without_the_keyboard_the_panel_leaves_the_keys_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);
        press(&ctx, &mut window, beside_the_window());

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the panel did not undo"
        );
        assert_eq!(keys, [egui::Key::Z], "the host saw Ctrl+Z");
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::ArrowRight, egui::Modifiers::NONE),
        );
        assert_eq!(keys, [egui::Key::ArrowRight]);
    }

    #[test]
    fn a_collapsed_window_leaves_the_keys_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        load_gap(&ctx, &mut window, 2.0);
        // Collapsed with its title bar's triangle: only the title bar shows.
        let mut collapsing = egui::collapsing_header::CollapsingState::load_with_default_open(
            &ctx,
            window_id().with("collapsing"),
            true,
        );
        collapsing.set_open(false);
        collapsing.store(&ctx);
        // The collapse animates over a few frames (egui adds 1/60 s a frame).
        for _ in 0..30 {
            frame(&ctx, &mut window, Vec::new());
        }
        let (output, _) = frame(&ctx, &mut window, Vec::new());
        assert!(drew_text(&output, TITLE), "the title bar shows");
        assert!(!drew_text(&output, HEADING), "the panel does not");

        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(keys, [egui::Key::Z], "the host saw Ctrl+Z");
        assert_eq!(
            window.design(),
            magcoupling_gap_design(2.0),
            "the panel did not undo"
        );
    }

    #[test]
    fn a_closed_window_leaves_every_key_to_the_host() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        window.set_open(false);
        let (_, keys) = frame(
            &ctx,
            &mut window,
            key_tap(egui::Key::Z, egui::Modifiers::COMMAND),
        );
        assert_eq!(keys, [egui::Key::Z]);
    }

    #[test]
    fn share_links_point_at_the_base_set_before_or_after_the_first_opening() {
        let mut window = CalculatorWindow::default();
        window.set_share_base("http://localhost:8080/magcoupling/");
        window.set_open(true);
        let link = window.panel().expect("opened").share_link();
        assert!(
            link.starts_with("http://localhost:8080/magcoupling/?m="),
            "{link}"
        );
        window.set_share_base("http://127.0.0.1:9000/magcoupling/");
        let link = window.panel().expect("opened").share_link();
        assert!(
            link.starts_with("http://127.0.0.1:9000/magcoupling/?m="),
            "{link}"
        );
    }

    #[test]
    fn gui_smoke_opens_the_window_with_the_tool_parameter() {
        // .claude/workflows/gui-smoke.js's linkage step opens the linkage app with this query and
        // looks for the equation registry's log line, which the panel logs when it is created.
        let script = include_str!("../../../.claude/workflows/gui-smoke.js").replace("\r\n", "\n");
        let query = format!("const LINKAGE_TOOL_QUERY = '?{TOOL_PARAM}={TOOL_MAGCOUPLING}'");
        assert!(script.contains(&query), "gui-smoke.js has no {query}");
        // The linkage step's own check (the /magcoupling/ step names the prefix too).
        let check = format!(
            "magcoupling_window=true only if a console message contains \"{}\"",
            magcoupling::gui::readouts::REGISTRY_LOG_PREFIX
        );
        assert!(script.contains(&check), "gui-smoke.js has no {check}");
    }

    #[test]
    fn the_production_origin_gives_the_calculator_s_public_address() {
        assert_eq!(
            magcoupling_share_base("https://linkage.colesorkness.com"),
            PUBLIC_BASE_URL
        );
    }

    #[test]
    fn saved_files_get_the_filter_of_their_type() {
        for (name, label, extension) in [
            (JSON_FILE_NAME, "JSON", "json"),
            ("magcoupling-design.json", "JSON", "json"),
            (CSV_FILE_NAME, "CSV", "csv"),
            ("notes.txt", "All files", "*"),
            ("no_extension", "All files", "*"),
        ] {
            let filter = save_filter(name);
            assert_eq!(
                (filter.label, filter.extensions),
                (label, &[extension][..]),
                "{name}"
            );
        }
    }

    #[test]
    fn a_save_outcome_becomes_the_panel_s_report() {
        assert_eq!(
            report_of(DownloadOutcome::Saved("Saved: a.json".to_owned())),
            Some(Ok("Saved: a.json".to_owned()))
        );
        assert_eq!(report_of(DownloadOutcome::Cancelled), None);
        assert_eq!(
            report_of(DownloadOutcome::Failed("Write failed: denied".to_owned())),
            Some(Err("Write failed: denied".to_owned()))
        );
    }

    #[test]
    fn a_picked_design_file_loads_and_a_refused_one_changes_nothing() {
        let mut panel = MagcouplingPanel::new();
        let design = magcoupling_gap_design(2.0);
        open_picked_file(&mut panel, Ok(design_to_json(&design)));
        assert_eq!(panel.design(), design);
        open_picked_file(&mut panel, Ok("not a design".to_owned()));
        assert_eq!(panel.design(), design);
        open_picked_file(&mut panel, Err("the file is not UTF-8 text".to_owned()));
        assert_eq!(panel.design(), design);
    }

    #[test]
    fn a_picked_file_reaches_the_panel_on_the_next_frame() {
        let ctx = egui::Context::default();
        let (mut window, _) = opened(&ctx);
        let design = magcoupling_gap_design(2.0);
        *window.picker.inbox.borrow_mut() = Some(Ok(design_to_json(&design)));
        frame(&ctx, &mut window, Vec::new());
        assert_eq!(window.design(), design);
        assert!(window.picker.take().is_none(), "taken once");
    }
}
