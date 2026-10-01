//! Magnetic coupling calculator: the WebAssembly entry (feature `app`).
//!
//! Built for wasm32-unknown-unknown and bound with wasm-bindgen by
//! `linkage-sim-rs/scripts/build_magcoupling_web.sh` into
//! `linkage-sim-rs/web/magcoupling/`, served at `/magcoupling/`. The JS glue
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
/// wasm-bindgen glue when the module loads.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() {
    use magcoupling::app::{CANVAS_ID, MagcouplingApp};
    use wasm_bindgen::JsCast;

    // Route log macros to the browser console.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let canvas = web_sys::window()
        .expect("no window")
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
            Box::new(|_cc| Ok(Box::new(MagcouplingApp::default()))),
        )
        .await
        .expect("Failed to start eframe");
}
