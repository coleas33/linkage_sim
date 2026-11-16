//! Export functionality: CSV sweep data, coupler traces, SVG/PNG/GIF images,
//! DXF geometry, HTML analysis reports, and firmware-friendly trajectory
//! command streams (JSON v1; G-code/Aerotech/Beckhoff/Galil planned).

mod animation;
mod csv;
pub mod download;
mod dxf;
mod escape;
pub mod firmware;
mod raster;
mod report;
pub mod schematic;
mod svg;

// Re-export all public items so callers can use `export::function_name` unchanged.
// String-returning generators are platform-independent and re-exported on
// both native and web. Path-writing wrappers (file I/O) and raster
// generation (resvg/gif crates) remain native-only.
pub use animation::{animation_export_available, generate_animation_html};
pub use csv::{generate_coupler_csv_string, generate_sweep_csv_string};
pub use dxf::generate_dxf_string;
#[cfg(feature = "raster")]
pub use raster::{generate_mechanism_gif_bytes, generate_mechanism_png_bytes};
#[cfg(feature = "raster")]
pub(crate) use raster::rasterize_svg_to_rgba;
pub use report::generate_html_report;
pub use svg::generate_svg_string;
