//! WebAssembly entry point for the linkage simulator.
//!
//! When compiled for wasm32, this binary serves as the browser entry point
//! using eframe's WebRunner. On native targets it falls back to the standard
//! native window (identical to `linkage_gui`).

fn main() {
    // Native fallback — the primary native binary is `linkage_gui`, but this
    // lets `cargo run --bin linkage-web` work on desktop for quick testing.
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let options = eframe::NativeOptions::default();
        eframe::run_native(
            "Linkage Simulator",
            options,
            Box::new(|cc| Ok(Box::new(linkage_sim_rs::gui::LinkageApp::new(cc)))),
        )
        .unwrap();
    }

    // WASM entry point is handled by start() below; main() is a no-op on wasm.
}

/// WASM entry point — called automatically by the JS glue code.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub async fn start() {
    // Redirect log macros to console.log.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(
            "linkage_canvas", // must match the <canvas id="..."> in index.html
            web_options,
            Box::new(|cc| Ok(Box::new(linkage_sim_rs::gui::LinkageApp::new(cc)))),
        )
        .await
        .expect("Failed to start eframe");
}
