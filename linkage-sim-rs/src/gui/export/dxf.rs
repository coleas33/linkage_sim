//! DXF generation and file export for mechanism geometry.

use crate::core::mechanism::Mechanism;

/// Generate a DXF ASCII string for the mechanism at its current pose.
///
/// Exports: bodies as LINE entities, joints as CIRCLE entities (r=0.002m),
/// ground pivots as POINT entities. All coordinates in meters.
pub fn generate_dxf_string(
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<String, String> {
    use crate::core::constraint::Constraint;
    use crate::core::state::GROUND_ID;

    let state = mechanism.state();
    let bodies = mechanism.bodies();
    if bodies.len() <= 1 {
        return Err("No moving bodies to export".to_string());
    }

    let mut dxf = String::with_capacity(4096);

    // DXF header
    dxf.push_str("0\nSECTION\n2\nENTITIES\n");

    // Export bodies as LINE entities
    let mut body_ids: Vec<&String> = bodies.keys().collect();
    body_ids.sort();
    for body_id in &body_ids {
        let body = &bodies[*body_id];
        let mut pts: Vec<(&String, nalgebra::Vector2<f64>)> = body
            .attachment_points
            .iter()
            .map(|(name, local)| {
                let global = state.body_point_global(body_id, local, q);
                (name, global)
            })
            .collect();
        pts.sort_by_key(|(n, _)| n.as_str());

        // Lines between consecutive points
        for pair in pts.windows(2) {
            let (_, p1) = &pair[0];
            let (_, p2) = &pair[1];
            dxf.push_str(&format!(
                "0\nLINE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n11\n{:.6}\n21\n{:.6}\n31\n0.0\n",
                body_id, p1.x, p1.y, p2.x, p2.y
            ));
        }
        // Close polygon for 3+ points
        if pts.len() >= 3 {
            let (_, p_last) = pts.last().unwrap();
            let (_, p_first) = &pts[0];
            dxf.push_str(&format!(
                "0\nLINE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n11\n{:.6}\n21\n{:.6}\n31\n0.0\n",
                body_id, p_last.x, p_last.y, p_first.x, p_first.y
            ));
        }
    }

    // Export joints as CIRCLE entities (radius 0.002 m)
    let joint_radius = 0.002;
    for joint in mechanism.joints() {
        let pt = state.body_point_global(joint.body_i_id(), &joint.point_i_local(), q);
        let layer = if joint.is_revolute() {
            "JOINTS_REVOLUTE"
        } else if joint.is_prismatic() {
            "JOINTS_PRISMATIC"
        } else {
            "JOINTS"
        };
        dxf.push_str(&format!(
            "0\nCIRCLE\n8\n{}\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n40\n{:.6}\n",
            layer, pt.x, pt.y, joint_radius
        ));
    }

    // Export ground attachment points as POINT entities
    if let Some(ground) = bodies.get(GROUND_ID) {
        for (_, local) in &ground.attachment_points {
            let global = state.body_point_global(GROUND_ID, local, q);
            dxf.push_str(&format!(
                "0\nPOINT\n8\nGROUND\n10\n{:.6}\n20\n{:.6}\n30\n0.0\n",
                global.x, global.y
            ));
        }
    }

    // DXF footer
    dxf.push_str("0\nENDSEC\n0\nEOF\n");

    Ok(dxf)
}

/// Export mechanism geometry as a DXF file.
#[cfg(feature = "native")]
pub fn export_mechanism_dxf(
    path: &std::path::Path,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Result<(), String> {
    let dxf = generate_dxf_string(mechanism, q)?;
    std::fs::write(path, dxf).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
    #[test]
    fn generate_dxf_contains_line_and_circle_entities() {
        use crate::gui::samples::{build_sample, SampleMechanism};

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q = if result.converged { result.q } else { q0 };

        let dxf = generate_dxf_string(&mech, &q).expect("DXF generation should succeed");

        assert!(dxf.contains("SECTION"), "should have SECTION header");
        assert!(dxf.contains("ENTITIES"), "should have ENTITIES section");
        assert!(dxf.contains("LINE"), "should have LINE entities for bodies");
        assert!(dxf.contains("CIRCLE"), "should have CIRCLE entities for joints");
        assert!(dxf.contains("POINT"), "should have POINT entities for ground");
        assert!(dxf.contains("EOF"), "should have EOF marker");
    }
}
