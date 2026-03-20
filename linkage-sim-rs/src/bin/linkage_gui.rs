//! Linkage mechanism simulator — GUI application.

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Linkage Simulator \u{2014} Planar Mechanism Analysis"),
        ..Default::default()
    };

    eframe::run_native(
        "Linkage Simulator",
        options,
        Box::new(|cc| Ok(Box::new(linkage_sim_rs::gui::LinkageApp::new(cc)))),
    )
}
