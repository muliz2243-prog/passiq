// ============================================================
// PassIQ — main.rs
// Entry point. Creates the native eframe window and hands
// control to PassIQApp. Contains no business logic.
// ============================================================

mod app;
mod model;
mod paths;
mod report;

use app::PassIQApp;

fn main() -> eframe::Result {
    let viewport = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 720.0])
        .with_min_inner_size([900.0, 540.0]);

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "PassIQ — xP Analyzer",
        native_options,
        Box::new(|_cc| Ok(Box::new(PassIQApp::new()))),
    )
}
