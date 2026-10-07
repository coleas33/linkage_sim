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
    use wasm_bindgen::JsCast;

    // Redirect log macros to console.log.
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    // ── Check for ?m= URL parameter (shared mechanism) ───────────────
    let shared_mechanism_json = extract_url_mechanism_param();

    let canvas = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document")
        .get_element_by_id("linkage_canvas")
        .expect("no canvas element with id 'linkage_canvas'")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("element is not a canvas");

    let web_options = eframe::WebOptions::default();
    eframe::WebRunner::new()
        .start(
            canvas,
            web_options,
            Box::new(move |cc| {
                let mut app = linkage_sim_rs::gui::LinkageApp::new(cc);
                // If a ?m= parameter was found, load the shared mechanism.
                if let Some(json_str) = shared_mechanism_json {
                    app.load_shared_mechanism(&json_str);
                }
                Ok(Box::new(app))
            }),
        )
        .await
        .expect("Failed to start eframe");
}

/// Extract the `?m=` URL parameter (compressed+base64 mechanism JSON).
///
/// Returns `Some(json_string)` if a valid mechanism was found in the URL,
/// or `None` if no parameter exists or decoding fails.
#[cfg(target_arch = "wasm32")]
fn extract_url_mechanism_param() -> Option<String> {
    use linkage_sim_rs::gui::decode_mechanism_from_url;

    let encoded = url_param("m")?;

    match decode_mechanism_from_url(&encoded) {
        Ok(json) => {
            log::info!("Loaded shared mechanism from URL ({} bytes JSON)", json.len());
            Some(json)
        }
        Err(e) => {
            log::error!("Failed to decode shared mechanism from URL: {}", e);
            None
        }
    }
}

/// The value of the URL query parameter `name`, if the page's address has it and it is not empty.
#[cfg(target_arch = "wasm32")]
fn url_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    if search.is_empty() {
        return None;
    }
    // web_sys::UrlSearchParams takes the raw search string, leading '?' included.
    let params = web_sys::UrlSearchParams::new_with_str(&search).ok()?;
    params.get(name).filter(|value| !value.is_empty())
}
