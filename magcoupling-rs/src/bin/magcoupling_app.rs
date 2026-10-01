//! Magnetic coupling calculator: the native window (feature `app`).
//!
//! `cargo run --release --features app --bin magcoupling-app` from
//! `magcoupling-rs/`. The web build is `magcoupling-web`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    magcoupling::app::run_native()
}
