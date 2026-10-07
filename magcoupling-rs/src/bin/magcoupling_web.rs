//! Magnetic coupling calculator: the WebAssembly entry (feature `app`).
//!
//! Built for wasm32-unknown-unknown and bound with wasm-bindgen by
//! `linkage-sim-rs/scripts/build_magcoupling_web.sh` into
//! `linkage-sim-rs/web/tools/magcoupler/`, served at `/tools/magcoupler/`. The JS glue
//! calls [`start`], which runs the app in the page's canvas through eframe's
//! WebRunner, as `linkage-sim-rs/src/bin/linkage_web.rs` does. On a desktop,
//! `cargo run --features app --bin magcoupling-web` opens the native window.

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    if let Err(error) = magcoupling::app::run_native() {
        eprintln!("magcoupling-web: {error}");
        std::process::exit(1);
    }
    // On wasm32, main does nothing: the JS glue calls start().
}

/// Starts the app in the canvas `magcoupling::app::CANVAS_ID`; called by the
/// wasm-bindgen glue when the module loads. A `?m=` share link in the page's
/// address opens its design; share links made here point at this page.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() {
    use magcoupling::app::{CANVAS_ID, MagcouplingApp};
    use magcoupling::gui::session::SHARE_PARAM;
    use wasm_bindgen::JsCast;

    // Route log macros to the browser console.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let window = web_sys::window().expect("no window");
    let location = window.location();
    let base = format!(
        "{}{}",
        location.origin().unwrap_or_default(),
        location.pathname().unwrap_or_default()
    );
    let payload = location
        .search()
        .ok()
        .and_then(|search| web_sys::UrlSearchParams::new_with_str(&search).ok())
        .and_then(|params| params.get(SHARE_PARAM))
        .filter(|payload| !payload.is_empty());

    let canvas = window
        .document()
        .expect("no document")
        .get_element_by_id(CANVAS_ID)
        .unwrap_or_else(|| panic!("no canvas element with id '{CANVAS_ID}'"))
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is not a canvas");

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(move |cc| {
                let mut app = MagcouplingApp::new(&cc.egui_ctx);
                app.set_share_base(base);
                if let Some(payload) = payload {
                    // A refused link is logged and shown in the panel; the default design stays.
                    let _ = app.open_share_payload(&payload);
                }
                Ok(Box::new(app))
            }),
        )
        .await
        .expect("Failed to start eframe");
}
