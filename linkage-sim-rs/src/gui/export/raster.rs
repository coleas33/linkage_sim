//! Raster export: PNG and animated GIF generation via resvg.

#[cfg(feature = "native")]
use crate::core::mechanism::Mechanism;
#[cfg(feature = "native")]
use crate::gui::sweep::SweepData;

#[cfg(feature = "native")]
use super::svg::generate_svg_string;

/// Rasterize an SVG string to RGBA pixel data at the given dimensions.
///
/// This is the shared rasterization core used by both PNG export and GIF
/// frame generation. Requires the `native` feature (depends on `resvg`).
#[cfg(feature = "native")]
pub(crate) fn rasterize_svg_to_rgba(
    svg_str: &str,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let opt = resvg::usvg::Options::default();
    let tree = resvg::usvg::Tree::from_str(svg_str, &opt)
        .map_err(|e| format!("Failed to parse SVG for rasterization: {}", e))?;

    let mut pixmap = resvg::tiny_skia::Pixmap::new(width, height)
        .ok_or_else(|| format!("Failed to create {}x{} pixmap", width, height))?;

    // Fill with the dark background color to match the SVG canvas background
    // and prevent white bleed-through on any transparent SVG elements.
    pixmap.fill(resvg::tiny_skia::Color::from_rgba8(30, 30, 35, 255));

    // Scale uniformly to fit, centering the shorter axis.
    let sx = width as f32 / tree.size().width();
    let sy = height as f32 / tree.size().height();
    let scale = sx.min(sy);
    let tx = (width as f32 - tree.size().width() * scale) / 2.0;
    let ty = (height as f32 - tree.size().height() * scale) / 2.0;

    let transform = resvg::tiny_skia::Transform::from_scale(scale, scale)
        .post_translate(tx, ty);

    resvg::render(&tree, transform, &mut pixmap.as_mut());

    Ok(pixmap.take())
}

/// Export the mechanism at its current pose as a PNG image.
///
/// Generates an SVG string, rasterizes it with resvg at the given dimensions,
/// and saves the result as a PNG file. Requires the `native` feature.
#[cfg(feature = "native")]
pub fn export_mechanism_png(
    path: &std::path::Path,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
    width: u32,
    height: u32,
) -> Result<(), String> {
    let svg_str = generate_svg_string(mechanism, q)?;
    let rgba = rasterize_svg_to_rgba(&svg_str, width, height)?;

    // Reconstruct a Pixmap from the raw RGBA data so we can use save_png.
    let pixmap = resvg::tiny_skia::Pixmap::from_vec(rgba, resvg::tiny_skia::IntSize::from_wh(width, height).unwrap())
        .ok_or_else(|| "Failed to reconstruct pixmap from RGBA data".to_string())?;

    pixmap.save_png(path)
        .map_err(|e| format!("Failed to save PNG: {}", e))
}

/// Export an animated GIF of the mechanism sweep.
///
/// Re-solves the mechanism position at each sampled sweep step, renders via
/// SVG + resvg, and encodes as a looping animated GIF. Requires the `native`
/// feature (depends on `resvg` and `gif`).
#[cfg(feature = "native")]
pub fn export_mechanism_gif(
    path: &std::path::Path,
    mech: &Mechanism,
    sweep: &SweepData,
    q_start: &nalgebra::DVector<f64>,
    omega: f64,
    theta_0: f64,
    width: u32,
    height: u32,
    frame_delay_cs: u16,
) -> Result<(), String> {
    use crate::solver::kinematics::solve_position;
    use gif::{Encoder, Frame, Repeat};

    let n_steps = sweep.angles_deg.len();
    if n_steps == 0 {
        return Err("No sweep data to export".to_string());
    }

    let file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create GIF file: {}", e))?;
    let mut encoder = Encoder::new(file, width as u16, height as u16, &[])
        .map_err(|e| format!("Failed to initialize GIF encoder: {}", e))?;
    encoder.set_repeat(Repeat::Infinite)
        .map_err(|e| format!("Failed to set GIF repeat: {}", e))?;

    // Target ~72 frames for a smooth animation; skip steps if sweep is denser.
    let step_skip = (n_steps / 72).max(1);
    let mut q_guess = q_start.clone();

    for i in (0..n_steps).step_by(step_skip) {
        let angle_rad = sweep.angles_deg[i].to_radians();
        let t = (angle_rad - theta_0) / omega;

        match solve_position(mech, &q_guess, t, 1e-10, 50) {
            Ok(result) if result.converged => {
                if let Ok(svg_str) = generate_svg_string(mech, &result.q) {
                    if let Ok(rgba) = rasterize_svg_to_rgba(&svg_str, width, height) {
                        let mut rgba_buf = rgba;
                        let mut frame = Frame::from_rgba_speed(
                            width as u16,
                            height as u16,
                            &mut rgba_buf,
                            10,
                        );
                        frame.delay = frame_delay_cs;
                        encoder.write_frame(&frame)
                            .map_err(|e| format!("Failed to write GIF frame: {}", e))?;
                    }
                }
                q_guess = result.q;
            }
            _ => {
                // Skip frames that fail to converge; the animation will still
                // be useful with the frames that do converge.
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    #[cfg(feature = "native")]
    fn export_png_produces_valid_file() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::CrankRocker);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let path = std::env::temp_dir().join("test_mechanism.png");
        export_mechanism_png(&path, &mech, &q, 1920, 1080)
            .expect("PNG export should succeed");

        // Verify the file exists and has reasonable size
        let metadata = std::fs::metadata(&path).expect("PNG file should exist");
        assert!(metadata.len() > 100, "PNG file should not be empty");

        // Verify PNG magic bytes
        let bytes = std::fs::read(&path).expect("should read PNG file");
        assert_eq!(
            &bytes[..4],
            &[0x89, 0x50, 0x4E, 0x47],
            "file should start with PNG magic bytes"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_png_empty_mechanism_returns_error() {
        use crate::core::mechanism::Mechanism;
        use nalgebra::DVector;

        let mut mech = Mechanism::new();
        mech.build().expect("build should succeed");
        let q = DVector::zeros(0);

        let path = std::env::temp_dir().join("test_mechanism_empty.png");
        let result = export_mechanism_png(&path, &mech, &q, 1920, 1080);
        assert!(result.is_err(), "empty mechanism should return an error");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_png_custom_dimensions() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        // Test with a smaller resolution
        let path = std::env::temp_dir().join("test_mechanism_small.png");
        export_mechanism_png(&path, &mech, &q, 640, 480)
            .expect("PNG export at 640x480 should succeed");

        let metadata = std::fs::metadata(&path).expect("PNG file should exist");
        assert!(metadata.len() > 100, "PNG file should not be empty");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_produces_valid_file() {
        use crate::gui::samples::SampleMechanism;
        use crate::gui::state::AppState;

        // Use AppState to get sweep data (which requires a full sample load).
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let mech = state.mechanism.as_ref().expect("mechanism should be loaded");
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some");

        let path = std::env::temp_dir().join("test_mechanism.gif");
        export_mechanism_gif(
            &path,
            mech,
            sweep,
            &state.q,
            state.driver_omega,
            state.driver_theta_0,
            400,
            300,
            5,
        )
        .expect("GIF export should succeed");

        // Verify the file exists and has reasonable size.
        let metadata = std::fs::metadata(&path).expect("GIF file should exist");
        assert!(metadata.len() > 100, "GIF file should not be empty");

        // Verify GIF magic bytes ("GIF89a" for animated GIFs).
        let bytes = std::fs::read(&path).expect("should read GIF file");
        assert_eq!(
            &bytes[..6],
            b"GIF89a",
            "file should start with GIF89a magic bytes"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_empty_sweep_returns_error() {
        use crate::gui::samples::{build_sample, SampleMechanism};

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let empty_sweep = SweepData {
            angles_deg: vec![],
            body_angles: HashMap::new(),
            coupler_traces: HashMap::new(),
            transmission_angles: None,
            driver_torques: None,
            kinetic_energy: vec![],
            potential_energy: vec![],
            total_energy: vec![],
            inverse_dynamics_torques: vec![],
            mechanical_advantage: vec![],
            joint_reaction_magnitudes: HashMap::new(),
            coupler_velocities: HashMap::new(),
            coupler_accelerations: HashMap::new(),
            toggle_angles: Vec::new(),
            active_range: None,
            sweep_mode: crate::gui::sweep::SweepMode::Angle,
        };

        let path = std::env::temp_dir().join("test_mechanism_empty.gif");
        let result = export_mechanism_gif(
            &path,
            &mech,
            &empty_sweep,
            &q0,
            2.0 * std::f64::consts::PI,
            0.0,
            400,
            300,
            5,
        );
        assert!(result.is_err(), "empty sweep should return an error");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_gif_crank_rocker_produces_valid_file() {
        use crate::gui::state::AppState;
        use crate::gui::samples::SampleMechanism;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let mech = state.mechanism.as_ref().expect("mechanism should be loaded");
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some");

        let path = std::env::temp_dir().join("test_crank_rocker.gif");
        export_mechanism_gif(
            &path,
            mech,
            sweep,
            &state.q,
            state.driver_omega,
            state.driver_theta_0,
            800,
            600,
            5,
        )
        .expect("CrankRocker GIF export should succeed");

        let metadata = std::fs::metadata(&path).expect("GIF file should exist");
        assert!(
            metadata.len() > 1000,
            "CrankRocker GIF should have multiple frames"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[cfg(feature = "native")]
    fn rasterize_svg_to_rgba_produces_correct_size() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = solve_position(&mech, &q0, 0.0, 1e-10, 50).expect("solver should succeed");
        let q = if result.converged { result.q } else { q0 };

        let svg = generate_svg_string(&mech, &q).expect("SVG generation should succeed");
        let rgba = rasterize_svg_to_rgba(&svg, 320, 240).expect("rasterization should succeed");

        // RGBA: 4 bytes per pixel
        assert_eq!(
            rgba.len(),
            320 * 240 * 4,
            "RGBA buffer should be width * height * 4 bytes"
        );
    }

    #[test]
    #[cfg(feature = "native")]
    fn export_chebyshev_lambda_pngs() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::solver::kinematics::solve_position;

        let (mech, q0) = build_sample(SampleMechanism::Chebyshev);

        let out_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("docs").join("chebyshev_lambda");
        std::fs::create_dir_all(&out_dir).expect("create output dir");

        let omega = 1.0;
        let theta_0 = 0.0;
        let angles = [0, 45, 90, 135, 180, 225, 270, 315];
        let mut q = q0.clone();

        for deg in 0..=315 {
            let t = ((deg as f64).to_radians() - theta_0) / omega;
            if let Ok(result) = solve_position(&mech, &q, t, 1e-10, 50) {
                if result.converged {
                    q = result.q.clone();
                    if angles.contains(&deg) {
                        let path = out_dir.join(format!("chebyshev_lambda_{:03}deg.png", deg));
                        export_mechanism_png(&path, &mech, &q, 1920, 1080)
                            .expect(&format!("export at {}° should succeed", deg));
                        eprintln!("Exported: {}", path.display());
                    }
                }
            }
        }
    }
}
