//! Export functionality: CSV sweep data, coupler traces, SVG/PNG/GIF images,
//! DXF geometry, HTML analysis reports, and firmware-friendly trajectory
//! command streams (JSON v1; G-code/Aerotech/Beckhoff/Galil planned).

mod csv;
pub mod download;
mod dxf;
pub mod firmware;
mod raster;
mod report;
pub mod schematic;
mod svg;

// Re-export all public items so callers can use `export::function_name` unchanged.
// String-returning generators are platform-independent and re-exported on
// both native and web. Path-writing wrappers (file I/O) and raster
// generation (resvg/gif crates) remain native-only.
pub use csv::{generate_coupler_csv_string, generate_sweep_csv_string};
#[cfg(feature = "native")]
pub use csv::{export_coupler_csv, export_sweep_csv};
pub use dxf::generate_dxf_string;
#[cfg(feature = "native")]
pub use dxf::export_mechanism_dxf;
#[cfg(feature = "raster")]
pub use raster::{generate_mechanism_gif_bytes, generate_mechanism_png_bytes};
#[cfg(all(feature = "native", feature = "raster"))]
pub use raster::{export_mechanism_gif, export_mechanism_png};
#[cfg(feature = "raster")]
pub(crate) use raster::rasterize_svg_to_rgba;
pub use report::generate_html_report;
pub use svg::generate_svg_string;
#[cfg(feature = "native")]
pub use svg::export_mechanism_svg;
