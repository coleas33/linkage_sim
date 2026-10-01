//! The standalone app's file work, for the panel's requests (`gui::PanelRequest`): saving a
//! file (a design, a results export) and picking a design file to load. Natively a file dialog
//! (rfd); on the web a browser download, and rfd's file picker. The same pattern as
//! `linkage-sim-rs/src/gui/export/download.rs`.

/// Saves `contents`: natively to the file the user picks, on the web as a download. Returns
/// the message to show, or `None` when the user cancelled.
pub fn save(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    save_impl(file_name, mime, contents)
}

#[cfg(not(target_arch = "wasm32"))]
fn save_impl(file_name: &str, _mime: &str, contents: &str) -> Option<Result<String, String>> {
    let extension = file_name.rsplit_once('.').map_or("", |(_, ext)| ext);
    let path = rfd::FileDialog::new()
        .set_file_name(file_name)
        .add_filter(extension, &[extension])
        .save_file()?;
    Some(
        std::fs::write(&path, contents)
            .map(|()| format!("Saved {}", path.display()))
            .map_err(|error| format!("Could not save {}: {error}", path.display())),
    )
}

#[cfg(target_arch = "wasm32")]
fn save_impl(file_name: &str, mime: &str, contents: &str) -> Option<Result<String, String>> {
    Some(download(file_name, mime, contents.as_bytes()).map(|()| format!("Downloaded {file_name}")))
}

/// A browser download: a Blob behind a transient `<a download>` that is clicked.
#[cfg(target_arch = "wasm32")]
fn download(file_name: &str, mime: &str, contents: &[u8]) -> Result<(), String> {
    use wasm_bindgen::JsCast;

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(contents).into());
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts.into(), &bag)
        .map_err(|e| format!("Blob failed: {e:?}"))?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("object URL failed: {e:?}"))?;
    let anchor = document
        .create_element("a")
        .map_err(|e| format!("no <a>: {e:?}"))?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "not an <a>")?;
    anchor.set_href(&url);
    anchor.set_download(file_name);
    let _ = anchor.style().set_property("display", "none");
    if let Some(body) = document.body() {
        let _ = body.append_child(&anchor);
        anchor.click();
        let _ = body.remove_child(&anchor);
    } else {
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

/// A design file the user is picking: natively the dialog returns at once; on the web the
/// picker answers later, so the text arrives in an inbox the app reads each frame.
#[derive(Default)]
pub struct DesignPicker {
    #[cfg(target_arch = "wasm32")]
    inbox: std::rc::Rc<std::cell::RefCell<Option<Result<String, String>>>>,
    #[cfg(not(target_arch = "wasm32"))]
    inbox: Option<Result<String, String>>,
}

impl DesignPicker {
    /// Shows the file picker (JSON files); the text arrives in [`DesignPicker::take`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pick(&mut self, _ctx: &egui::Context) {
        let picked = rfd::FileDialog::new()
            .add_filter("Design", &["json"])
            .pick_file();
        self.inbox = picked.map(|path| {
            std::fs::read_to_string(&path)
                .map_err(|error| format!("Could not read {}: {error}", path.display()))
        });
    }

    /// Shows the file picker (JSON files); the text arrives in [`DesignPicker::take`] once the
    /// browser hands the file over.
    #[cfg(target_arch = "wasm32")]
    pub fn pick(&mut self, ctx: &egui::Context) {
        let inbox = self.inbox.clone();
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

    /// The picked file's text (or why it could not be read), once.
    pub fn take(&mut self) -> Option<Result<String, String>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inbox.take()
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.inbox.borrow_mut().take()
        }
    }
}
