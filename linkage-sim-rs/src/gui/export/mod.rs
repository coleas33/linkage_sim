//! Export functionality: CSV sweep data, coupler traces, SVG/PNG/GIF images,
//! DXF geometry, and HTML analysis reports.

mod csv;
mod dxf;
mod raster;
mod report;
mod svg;

// Re-export all public items so callers can use `export::function_name` unchanged.
#[cfg(feature = "native")]
pub use csv::{export_coupler_csv, export_sweep_csv};
#[cfg(feature = "native")]
pub use dxf::{export_mechanism_dxf, generate_dxf_string};
#[cfg(feature = "native")]
pub use raster::{export_mechanism_gif, export_mechanism_png};
#[cfg(feature = "native")]
pub use report::generate_html_report;
#[cfg(feature = "native")]
pub use svg::{export_mechanism_svg, generate_svg_string};
