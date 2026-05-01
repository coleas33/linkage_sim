//! Cross-platform "download a text file" helper.
//!
//! Native: prompts for a save path via `rfd::FileDialog` and writes the
//! contents with `std::fs::write`.
//!
//! Web: triggers a browser download by creating a `Blob`, wrapping it in
//! an `ObjectURL`, programmatically clicking a hidden anchor, then
//! revoking the URL. The browser handles the "save as" UI itself
//! (subject to the user's browser settings — most save to ~/Downloads
//! by default).
//!
//! The helper is intentionally synchronous on both platforms. Native
//! `rfd` opens a modal dialog; web's blob URL is set up and clicked
//! synchronously and the browser takes over from there.

#[derive(Debug, Clone, Copy)]
pub struct FileFilter {
    pub label: &'static str,
    pub extensions: &'static [&'static str],
}

/// Outcome of a download attempt — what to show in the status bar.
#[derive(Debug, Clone)]
pub enum DownloadOutcome {
    /// User saved (native) or browser triggered the download (web).
    Saved(String),
    /// User cancelled the native save dialog. Web cannot cancel — never returned there.
    Cancelled,
    /// Saving failed for a real reason (disk write error on native, JS error on web).
    Failed(String),
}

/// Trigger a download / save of the given text contents.
///
/// `default_filename` is the suggested filename (e.g. `"mechanism.svg"`).
/// `mime_type` is used by the web build to set the Blob MIME type
/// (e.g. `"image/svg+xml"`). The native build ignores it.
/// `filter` is used by the native save dialog to constrain visible
/// file types; the web build ignores it.
pub fn download_text(
    default_filename: &str,
    mime_type: &str,
    contents: &str,
    filter: FileFilter,
) -> DownloadOutcome {
    download_text_impl(default_filename, mime_type, contents, filter)
}

#[cfg(feature = "native")]
fn download_text_impl(
    default_filename: &str,
    _mime_type: &str,
    contents: &str,
    filter: FileFilter,
) -> DownloadOutcome {
    let Some(path) = rfd::FileDialog::new()
        .add_filter(filter.label, filter.extensions)
        .set_file_name(default_filename)
        .save_file()
    else {
        return DownloadOutcome::Cancelled;
    };
    match std::fs::write(&path, contents) {
        Ok(()) => DownloadOutcome::Saved(format!("Saved: {}", path.display())),
        Err(e) => DownloadOutcome::Failed(format!("Write failed: {}", e)),
    }
}

#[cfg(target_arch = "wasm32")]
fn download_text_impl(
    default_filename: &str,
    mime_type: &str,
    contents: &str,
    _filter: FileFilter,
) -> DownloadOutcome {
    use wasm_bindgen::JsCast;

    let Some(window) = web_sys::window() else {
        return DownloadOutcome::Failed("No window object".to_string());
    };
    let Some(document) = window.document() else {
        return DownloadOutcome::Failed("No document object".to_string());
    };

    // Build a Blob from the contents. js_sys::Array of Uint8Array is the
    // most portable input shape.
    let bytes = js_sys::Uint8Array::from(contents.as_bytes());
    let parts = js_sys::Array::new();
    parts.push(&bytes.into());

    let mut bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime_type);

    let blob = match web_sys::Blob::new_with_u8_array_sequence_and_options(
        &parts.into(),
        &bag,
    ) {
        Ok(b) => b,
        Err(e) => {
            return DownloadOutcome::Failed(format!("Blob failed: {:?}", e));
        }
    };

    let url = match web_sys::Url::create_object_url_with_blob(&blob) {
        Ok(u) => u,
        Err(e) => return DownloadOutcome::Failed(format!("ObjectURL failed: {:?}", e)),
    };

    // Build a transient anchor with `download` attribute and click it
    // programmatically. This is the standard "trigger save dialog"
    // pattern on the web.
    let anchor = match document
        .create_element("a")
        .and_then(|e| e.dyn_into::<web_sys::HtmlAnchorElement>().map_err(|_| {
            wasm_bindgen::JsValue::from_str("anchor cast failed")
        })) {
        Ok(a) => a,
        Err(_) => {
            let _ = web_sys::Url::revoke_object_url(&url);
            return DownloadOutcome::Failed("Failed to construct <a>".to_string());
        }
    };
    anchor.set_href(&url);
    anchor.set_download(default_filename);
    let style = anchor.style();
    let _ = style.set_property("display", "none");
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    } else {
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);

    DownloadOutcome::Saved(format!("Downloaded: {}", default_filename))
}
