//! HTML report generation for mechanism analysis summaries.

use crate::core::mechanism::Mechanism;
use crate::gui::sweep::SweepData;

use super::schematic::generate_schematic_svg;
use super::svg::generate_svg_string;

/// Generate an HTML report summarizing the current mechanism analysis.
///
/// Includes: mechanism diagram (embedded SVG), topology, dimensions, mass
/// properties, Grashof classification, torque/transmission/reaction envelopes,
/// force element summary, and energy data.
pub fn generate_html_report(
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
    sweep: &SweepData,
    grashof: Option<&crate::analysis::grashof::GrashofResult>,
    units: &super::super::state::DisplayUnits,
) -> Result<String, String> {
    use crate::analysis::envelopes::compute_envelope;
    use crate::core::state::GROUND_ID;

    // Prefer the labeled schematic (joint IDs, body labels, ground hatching,
    // constraint legend) for a report. Fall back to the regular SVG if the
    // schematic generator fails (empty mechanism etc).
    let svg = generate_schematic_svg(mechanism, q)
        .or_else(|_| generate_svg_string(mechanism, q))
        .unwrap_or_default();

    let mut html = String::with_capacity(16_000);
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n");
    html.push_str("<title>Linkage Mechanism Report</title>\n");
    html.push_str("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>\n");
    // KaTeX for LaTeX-quality math rendering. Auto-render scans the body
    // for \(...\) (inline), \[...\] (display), $..$ (inline), and $$..$$
    // (display) delimiters and converts them in-place.
    //
    // Loading strategy: katex.min.js (no defer, must load first) +
    // auto-render.min.js (defer, runs after document parse). The
    // renderMathInElement call lives in a DOMContentLoaded handler at the
    // end of <body> so it fires after the full body — including the
    // dynamically-rendered Plotly charts — is in the DOM. The handler
    // checks for the global before calling it, so a CDN failure produces
    // a console warning instead of a JS error that breaks the rest of
    // the page.
    html.push_str(
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' \
         integrity='sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV' \
         crossorigin='anonymous'>\n",
    );
    html.push_str(
        "<script src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' \
         integrity='sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8' \
         crossorigin='anonymous'></script>\n",
    );
    html.push_str(
        "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' \
         integrity='sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05' \
         crossorigin='anonymous'></script>\n",
    );
    html.push_str("<style>\n");
    html.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #1a1a2e; }\n");
    html.push_str("h1 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; }\n");
    html.push_str("h2 { color: #0f3460; margin-top: 28px; }\n");
    html.push_str("table { border-collapse: collapse; width: 100%; margin: 12px 0; }\n");
    html.push_str("th, td { border: 1px solid #ccc; padding: 6px 12px; text-align: left; }\n");
    html.push_str("th { background: #e8eaf6; font-weight: 600; }\n");
    html.push_str("tr:nth-child(even) { background: #f0f2f5; }\n");
    html.push_str(".diagram { text-align: center; margin: 16px 0; background: #1e1e23; border-radius: 8px; padding: 12px; }\n");
    html.push_str(".diagram svg { max-width: 100%; height: auto; }\n");
    html.push_str(".summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }\n");
    html.push_str(".card { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }\n");
    html.push_str(".card h3 { margin: 0 0 8px 0; font-size: 14px; color: #555; }\n");
    html.push_str(".card .value { font-size: 22px; font-weight: 700; color: #0f3460; }\n");
    html.push_str(".plotly-chart { margin: 16px 0; }\n");
    html.push_str(".footer { margin-top: 30px; padding-top: 10px; border-top: 1px solid #ccc; font-size: 12px; color: #888; }\n");
    // Math derivation section styling. KaTeX renders display equations with
    // its own .katex-display class; we just give h3 subsection headings a
    // distinct look to nest visually under the "Mathematical Derivation" h2.
    html.push_str("h3 { color: #0f3460; margin-top: 20px; font-size: 16px; }\n");
    // Make KaTeX display equations slightly more prominent: light tinted
    // background and a left accent matching other report blocks.
    html.push_str(".katex-display { background: #f0f2f5; border-left: 3px solid #0f3460; padding: 10px 14px; margin: 12px 0; border-radius: 4px; }\n");
    html.push_str("</style></head><body>\n");

    // -- Header ---------------------------------------------------------------
    html.push_str("<h1>Linkage Mechanism Report</h1>\n");
    html.push_str(&format!("<p>Generated: {}</p>\n", chrono_now()));

    // -- Topology summary cards -----------------------------------------------
    let n_bodies = mechanism.bodies().len().saturating_sub(1);
    let n_joints = mechanism.joints().len();
    let dof = mechanism.state().n_coords() as isize - mechanism.n_constraints() as isize;
    let total_mass: f64 = mechanism.bodies().values()
        .filter(|b| b.id != GROUND_ID)
        .map(|b| b.mass)
        .sum();

    html.push_str("<div class='summary'>\n");
    html.push_str(&format!("<div class='card'><h3>Bodies</h3><div class='value'>{}</div></div>\n", n_bodies));
    html.push_str(&format!("<div class='card'><h3>Joints</h3><div class='value'>{}</div></div>\n", n_joints));
    html.push_str(&format!("<div class='card'><h3>DOF</h3><div class='value'>{}</div></div>\n", dof));
    html.push_str(&format!("<div class='card'><h3>Total Mass</h3><div class='value'>{:.3} kg</div></div>\n", total_mass));
    html.push_str("</div>\n");

    // -- Grashof classification -----------------------------------------------
    if let Some(gr) = grashof {
        let label = match gr.classification {
            crate::analysis::grashof::GrashofType::CrankRocker => "Crank-Rocker (Grashof)",
            crate::analysis::grashof::GrashofType::DoubleCrank => "Double-Crank (Grashof)",
            crate::analysis::grashof::GrashofType::DoubleRocker => "Double-Rocker (Grashof)",
            crate::analysis::grashof::GrashofType::ChangePoint => "Change-Point",
            crate::analysis::grashof::GrashofType::NonGrashof => "Non-Grashof",
        };
        html.push_str(&format!("<p><strong>Classification:</strong> {}</p>\n", label));
        html.push_str(&format!(
            "<p>Link lengths: {:.3}, {:.3}, {:.3}, {:.3}{}</p>\n",
            units.length(gr.link_lengths[0]),
            units.length(gr.link_lengths[1]),
            units.length(gr.link_lengths[2]),
            units.length(gr.link_lengths[3]),
            units.length_suffix()
        ));
    }

    // -- Mechanism diagram ----------------------------------------------------
    html.push_str("<h2>Mechanism Diagram</h2>\n");
    html.push_str("<div class='diagram'>\n");
    html.push_str(&svg);
    html.push_str("\n</div>\n");

    // -- Dimensions table -----------------------------------------------------
    html.push_str("<h2>Dimensions</h2>\n");
    html.push_str("<table><tr><th>Body</th><th>Segment</th><th>Length</th></tr>\n");
    let mut body_ids: Vec<&String> = mechanism.bodies().keys().collect();
    body_ids.sort();
    for body_id in &body_ids {
        if *body_id == GROUND_ID { continue; }
        let body = &mechanism.bodies()[*body_id];
        let mut pts: Vec<(&String, &nalgebra::Vector2<f64>)> = body.attachment_points.iter().collect();
        pts.sort_by_key(|(n, _)| n.as_str());
        for pair in pts.windows(2) {
            let (na, pa) = pair[0];
            let (nb, pb) = pair[1];
            let dist = (pb - pa).norm();
            html.push_str(&format!(
                "<tr><td>{}</td><td>{} \u{2192} {}</td><td>{:.3}{}</td></tr>\n",
                body_id, na, nb, units.length(dist), units.length_suffix()
            ));
        }
    }
    html.push_str("</table>\n");

    // -- Mass properties table ------------------------------------------------
    html.push_str("<h2>Mass Properties</h2>\n");
    html.push_str("<table><tr><th>Body</th><th>Mass (kg)</th><th>Izz (kg*m^2)</th><th>CG local</th></tr>\n");
    for body_id in &body_ids {
        if *body_id == GROUND_ID { continue; }
        let body = &mechanism.bodies()[*body_id];
        html.push_str(&format!(
            "<tr><td>{}</td><td>{:.4}</td><td>{:.6}</td><td>({:.4}, {:.4})</td></tr>\n",
            body_id, body.mass, body.izz_cg, body.cg_local.x, body.cg_local.y
        ));
    }
    html.push_str("</table>\n");

    // -- Loop equations -------------------------------------------------------
    write_loop_equations_section(&mut html, mechanism, q);

    // -- Mathematical derivation ---------------------------------------------
    write_math_background_section(&mut html, mechanism, q);

    // -- Torque envelope ------------------------------------------------------
    if let Some(ref torques) = sweep.driver_torques {
        if let Some(env) = compute_envelope(torques) {
            html.push_str("<h2>Driver Torque Envelope</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!("<div class='card'><h3>Peak (abs)</h3><div class='value'>{:.3} N*m</div></div>\n",
                env.max_value.abs().max(env.min_value.abs())));
            html.push_str(&format!("<div class='card'><h3>RMS</h3><div class='value'>{:.3} N*m</div></div>\n", env.rms));
            html.push_str(&format!("<div class='card'><h3>Min</h3><div class='value'>{:.3} N*m</div></div>\n", env.min_value));
            html.push_str(&format!("<div class='card'><h3>Max</h3><div class='value'>{:.3} N*m</div></div>\n", env.max_value));
            html.push_str("</div>\n");
        }
        // Interactive torque plot
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let torques_json = float_vec_to_json(torques);
        add_plotly_line_chart(
            &mut html, "torque_plot",
            "Driver Torque vs Crank Angle",
            &angles_json, &torques_json,
            "Driver Torque", "Torque (N*m)", "#0f3460",
            None,
        );
    }

    // -- Actuator force envelope -------------------------------------------------
    if let Some(ref act_forces) = sweep.actuator_forces {
        if let Some(env) = compute_envelope(act_forces) {
            html.push_str("<h2>Actuator Force Envelope</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!("<div class='card'><h3>Peak (abs)</h3><div class='value'>{:.3} N</div></div>\n",
                env.max_value.abs().max(env.min_value.abs())));
            html.push_str(&format!("<div class='card'><h3>RMS</h3><div class='value'>{:.3} N</div></div>\n", env.rms));
            html.push_str(&format!("<div class='card'><h3>Min</h3><div class='value'>{:.3} N</div></div>\n", env.min_value));
            html.push_str(&format!("<div class='card'><h3>Max</h3><div class='value'>{:.3} N</div></div>\n", env.max_value));
            html.push_str("</div>\n");
        }
        // Interactive actuator force plot
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let forces_json = float_vec_to_json(act_forces);
        add_plotly_line_chart(
            &mut html, "actuator_force_plot",
            "Required Actuator Force vs Crank Angle",
            &angles_json, &forces_json,
            "Actuator Force", "Force (N)", "#c62828",
            None,
        );
    }

    // -- Actuator stroke envelope (length + speed + power) -------------------
    //
    // Length and speed come straight from the kinematic sweep (geometric
    // distance and its time derivative); power = F * dL/dt. Together with
    // the force envelope above, these are the four numbers a sizing
    // engineer needs from the report: peak force, peak speed, peak power,
    // and stroke range. Skip whichever traces aren't populated (they're
    // None on mechanisms with no LinearActuator force element).
    if let Some(ref act_lens) = sweep.actuator_lengths {
        if let Some(env) = compute_envelope(act_lens) {
            html.push_str("<h2>Actuator Stroke Length</h2>\n");
            html.push_str("<div class='summary'>\n");
            // Stroke range = max - min (the actuator's required travel).
            let stroke_range = env.max_value - env.min_value;
            html.push_str(&format!(
                "<div class='card'><h3>Min length</h3><div class='value'>{:.1} mm</div></div>\n",
                env.min_value * 1000.0
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Max length</h3><div class='value'>{:.1} mm</div></div>\n",
                env.max_value * 1000.0
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Stroke range</h3><div class='value'>{:.1} mm</div></div>\n",
                stroke_range * 1000.0
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Mean length</h3><div class='value'>{:.1} mm</div></div>\n",
                env.mean * 1000.0
            ));
            html.push_str("</div>\n");
        }
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        // Convert to mm for the plot — sizing engineers think in mm, not m.
        let lens_mm: Vec<f64> = act_lens.iter().map(|x| x * 1000.0).collect();
        let lens_json = float_vec_to_json(&lens_mm);
        add_plotly_line_chart(
            &mut html, "actuator_length_plot",
            "Actuator Length vs Crank Angle",
            &angles_json, &lens_json,
            "Actuator Length", "Length (mm)", "#e08020",
            None,
        );
    }
    if let Some(ref act_speeds) = sweep.actuator_speeds {
        if let Some(env) = compute_envelope(act_speeds) {
            html.push_str("<h2>Actuator Speed</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!(
                "<div class='card'><h3>Peak (abs)</h3><div class='value'>{:.3} m/s</div></div>\n",
                env.max_value.abs().max(env.min_value.abs())
            ));
            html.push_str(&format!(
                "<div class='card'><h3>RMS</h3><div class='value'>{:.3} m/s</div></div>\n",
                env.rms
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Min</h3><div class='value'>{:.3} m/s</div></div>\n",
                env.min_value
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Max</h3><div class='value'>{:.3} m/s</div></div>\n",
                env.max_value
            ));
            html.push_str("</div>\n");
        }
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let speeds_json = float_vec_to_json(act_speeds);
        add_plotly_line_chart(
            &mut html, "actuator_speed_plot",
            "Actuator Speed (dL/dt) vs Crank Angle",
            &angles_json, &speeds_json,
            "Actuator Speed", "Speed (m/s)", "#5555c0",
            None,
        );
    }
    if let Some(ref act_power) = sweep.actuator_power {
        if let Some(env) = compute_envelope(act_power) {
            html.push_str("<h2>Actuator Power</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!(
                "<div class='card'><h3>Peak (abs)</h3><div class='value'>{:.2} W</div></div>\n",
                env.max_value.abs().max(env.min_value.abs())
            ));
            html.push_str(&format!(
                "<div class='card'><h3>RMS</h3><div class='value'>{:.2} W</div></div>\n",
                env.rms
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Min</h3><div class='value'>{:.2} W</div></div>\n",
                env.min_value
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Max</h3><div class='value'>{:.2} W</div></div>\n",
                env.max_value
            ));
            html.push_str("</div>\n");
            html.push_str(&format!(
                "<p>Computed as <code>P = F·(dL/dt)</code> where F is the \
                 statics-derived actuator force. Inverse-dynamics-corrected \
                 power (which includes inertial loads) is shown separately if \
                 a non-zero motion profile was used.</p>\n",
            ));
        }
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let power_json = float_vec_to_json(act_power);
        add_plotly_line_chart(
            &mut html, "actuator_power_plot",
            "Required Actuator Power vs Crank Angle",
            &angles_json, &power_json,
            "Actuator Power", "Power (W)", "#208060",
            None,
        );
    }

    // -- Transmission angle range ---------------------------------------------
    if let Some(ref ta) = sweep.transmission_angles {
        if let Some(env) = compute_envelope(ta) {
            html.push_str("<h2>Transmission Angle</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!(
                "<div class='card'><h3>Min</h3><div class='value'>{:.1}\u{00b0}</div></div>\n",
                env.min_value
            ));
            html.push_str(&format!(
                "<div class='card'><h3>Max</h3><div class='value'>{:.1}\u{00b0}</div></div>\n",
                env.max_value
            ));
            html.push_str("</div>\n");
            if env.min_value < 40.0 {
                html.push_str("<p style='color: #c62828;'><strong>Warning:</strong> Minimum transmission angle is below 40 degrees — poor force transmission in this region.</p>\n");
            }
        }
        // Interactive transmission angle plot (with 40°/90° reference lines)
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let ta_json = float_vec_to_json(ta);
        let shapes = "[{type:'line',x0:0,x1:360,y0:40,y1:40,\
                       line:{color:'#c62828',dash:'dash',width:1}},\
                       {type:'line',x0:0,x1:360,y0:90,y1:90,\
                       line:{color:'#888',dash:'dot',width:1}}]";
        add_plotly_line_chart(
            &mut html, "transmission_plot",
            "Transmission Angle vs Crank Angle",
            &angles_json, &ta_json,
            "Transmission Angle", "Angle (deg)", "#1b7340",
            Some(shapes),
        );
    }

    // -- Joint reaction peaks -------------------------------------------------
    if !sweep.joint_reaction_magnitudes.is_empty() {
        html.push_str("<h2>Joint Reaction Peaks</h2>\n");
        html.push_str("<table><tr><th>Joint</th><th>Peak Force (N)</th><th>Mean Force (N)</th></tr>\n");
        let mut jids: Vec<&String> = sweep.joint_reaction_magnitudes.keys().collect();
        jids.sort();
        for jid in &jids {
            let vals = &sweep.joint_reaction_magnitudes[*jid];
            if let Some(env) = compute_envelope(vals) {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{:.3}</td><td>{:.3}</td></tr>\n",
                    jid, env.max_value, env.mean
                ));
            }
        }
        html.push_str("</table>\n");

        // Interactive joint reactions plot
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let colors = ["#0f3460", "#c62828", "#1b7340", "#e65100", "#6a1b9a", "#00695c", "#4527a0", "#ad1457"];
        html.push_str("<div id='reactions_plot' class='plotly-chart'></div>\n");
        html.push_str("<script>\n");
        html.push_str("Plotly.newPlot('reactions_plot', [\n");
        for (i, jid) in jids.iter().enumerate() {
            let vals = &sweep.joint_reaction_magnitudes[*jid];
            let vals_json = float_vec_to_json(vals);
            let color = colors[i % colors.len()];
            let comma = if i + 1 < jids.len() { "," } else { "" };
            html.push_str(&format!(
                "  {{x:{},y:{},type:'scatter',name:'{}',line:{{color:'{}'}}}}{}\n",
                angles_json, vals_json, jid, color, comma
            ));
        }
        html.push_str("], {title:'Joint Reactions vs Crank Angle',xaxis:{title:'Crank Angle (deg)'},yaxis:{title:'Reaction Force (N)'},margin:{t:40,b:50,l:60,r:20}}, {responsive:true});\n");
        html.push_str("</script>\n");
    }

    // -- Force element summary ------------------------------------------------
    let forces = mechanism.forces();
    if !forces.is_empty() {
        html.push_str("<h2>Force Elements</h2>\n");
        html.push_str("<table><tr><th>#</th><th>Type</th><th>Details</th></tr>\n");
        for (i, fe) in forces.iter().enumerate() {
            let (type_name, details) = force_element_summary(fe);
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>\n",
                i + 1, type_name, details
            ));
        }
        html.push_str("</table>\n");
    }

    // -- Energy summary -------------------------------------------------------
    if !sweep.kinetic_energy.is_empty() {
        if let (Some(ke_env), Some(pe_env)) = (
            compute_envelope(&sweep.kinetic_energy),
            compute_envelope(&sweep.potential_energy),
        ) {
            html.push_str("<h2>Energy Summary</h2>\n");
            html.push_str("<div class='summary'>\n");
            html.push_str(&format!("<div class='card'><h3>Peak KE</h3><div class='value'>{:.4} J</div></div>\n", ke_env.max_value));
            html.push_str(&format!("<div class='card'><h3>Peak PE</h3><div class='value'>{:.4} J</div></div>\n", pe_env.max_value));
            html.push_str("</div>\n");
        }

        // Interactive energy plot
        let angles_json = float_vec_to_json(&sweep.angles_deg);
        let ke_json = float_vec_to_json(&sweep.kinetic_energy);
        let pe_json = float_vec_to_json(&sweep.potential_energy);
        let te_json = float_vec_to_json(&sweep.total_energy);
        html.push_str("<div id='energy_plot' class='plotly-chart'></div>\n");
        html.push_str("<script>\n");
        html.push_str(&format!(
            "Plotly.newPlot('energy_plot', [\
             {{x:{angles},y:{ke},type:'scatter',name:'Kinetic Energy',line:{{color:'#c62828'}}}},\
             {{x:{angles},y:{pe},type:'scatter',name:'Potential Energy',line:{{color:'#1b7340'}}}},\
             {{x:{angles},y:{te},type:'scatter',name:'Total Energy',line:{{color:'#0f3460',dash:'dash'}}}}], \
             {{title:'Energy vs Crank Angle',xaxis:{{title:'Crank Angle (deg)'}},yaxis:{{title:'Energy (J)'}},\
             margin:{{t:40,b:50,l:60,r:20}}}}, {{responsive:true}});\n",
            angles = angles_json, ke = ke_json, pe = pe_json, te = te_json
        ));
        html.push_str("</script>\n");
    }

    // -- Coupler trace scatter plot -------------------------------------------
    if !sweep.coupler_traces.is_empty() {
        let colors = ["#0f3460", "#c62828", "#1b7340", "#e65100", "#6a1b9a", "#00695c"];
        let mut trace_names: Vec<&String> = sweep.coupler_traces.keys().collect();
        trace_names.sort();
        html.push_str("<h2>Coupler Traces</h2>\n");
        html.push_str("<div id='coupler_plot' class='plotly-chart'></div>\n");
        html.push_str("<script>\n");
        html.push_str("Plotly.newPlot('coupler_plot', [\n");
        for (i, name) in trace_names.iter().enumerate() {
            let pts = &sweep.coupler_traces[*name];
            let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
            let ys: Vec<f64> = pts.iter().map(|p| p[1]).collect();
            let xs_json = float_vec_to_json(&xs);
            let ys_json = float_vec_to_json(&ys);
            let color = colors[i % colors.len()];
            let comma = if i + 1 < trace_names.len() { "," } else { "" };
            html.push_str(&format!(
                "  {{x:{},y:{},mode:'lines',name:'{}',line:{{color:'{}'}}}}{}\n",
                xs_json, ys_json, name, color, comma
            ));
        }
        html.push_str("], {title:'Coupler Point Traces',xaxis:{title:'X (m)',scaleanchor:'y'},yaxis:{title:'Y (m)'},margin:{t:40,b:50,l:60,r:20}}, {responsive:true});\n");
        html.push_str("</script>\n");
    }

    // -- KaTeX auto-render ----------------------------------------------------
    //
    // Defer the auto-render call until the full body is parsed AND both
    // KaTeX scripts have loaded. The `defer` script load only guarantees
    // execution after document parse, not after paint, and the auto-render
    // contrib script is itself deferred — so we wait for both via a
    // load-event listener. Default delimiters ($..$, $$..$$, \(..\), \[..\])
    // cover both inline and display math. Wrapping in try/catch means a
    // CDN failure (or an invalid LaTeX expression in some future report)
    // doesn't break the rest of the page.
    html.push_str(
        "<script>\n\
         window.addEventListener('load', function() {\n\
           if (typeof renderMathInElement === 'function') {\n\
             try {\n\
               renderMathInElement(document.body, {\n\
                 delimiters: [\n\
                   {left: '$$', right: '$$', display: true},\n\
                   {left: '\\\\[', right: '\\\\]', display: true},\n\
                   {left: '\\\\(', right: '\\\\)', display: false},\n\
                   {left: '$', right: '$', display: false}\n\
                 ],\n\
                 throwOnError: false,\n\
                 strict: false\n\
               });\n\
             } catch (e) { console.warn('KaTeX render failed:', e); }\n\
           } else {\n\
             console.warn('KaTeX auto-render not loaded; equations will appear as raw LaTeX.');\n\
           }\n\
         });\n\
         </script>\n",
    );

    // -- Footer ---------------------------------------------------------------
    html.push_str("<div class='footer'>\n");
    html.push_str("<p>Generated by Linkage Mechanism Simulator (Rust/egui)</p>\n");
    html.push_str("</div>\n");
    html.push_str("</body></html>\n");

    Ok(html)
}

// ── Plotly helpers ──────────────────────────────────────────────────────────

/// Emit a single-series Plotly line chart vs. crank angle into `html`.
///
/// `x_data` and `y_data` must be pre-serialized JSON arrays (use
/// `float_vec_to_json`). `extra_shapes` is an optional JSON array of shape
/// objects for reference lines (e.g. the 40°/90° markers on the transmission
/// angle plot).
fn add_plotly_line_chart(
    html: &mut String,
    plot_id: &str,
    title: &str,
    x_data: &str,
    y_data: &str,
    series_name: &str,
    y_label: &str,
    color: &str,
    extra_shapes: Option<&str>,
) {
    html.push_str(&format!("<div id='{}' class='plotly-chart'></div>\n", plot_id));
    html.push_str("<script>\n");
    let shapes_clause = match extra_shapes {
        Some(s) => format!(",shapes:{}", s),
        None => String::new(),
    };
    html.push_str(&format!(
        "Plotly.newPlot('{}', [{{x:{},y:{},type:'scatter',name:'{}',line:{{color:'{}'}}}}], \
         {{title:'{}',xaxis:{{title:'Crank Angle (deg)'}},yaxis:{{title:'{}'}},\
         margin:{{t:40,b:50,l:60,r:20}}{}}}, {{responsive:true}});\n",
        plot_id, x_data, y_data, series_name, color, title, y_label, shapes_clause
    ));
    html.push_str("</script>\n");
}

/// Emit a multi-series Plotly line chart into `html`.
///
/// Each `(name, data_json, color, dash)` tuple becomes one scatter trace.
/// `dash` is an optional line style string (e.g. "dash", "dot"); pass None
/// for a solid line. X-axis and Y-axis labels are supplied as literal
/// strings so the caller can pick units (e.g. "X (m)" for coupler traces).
///
/// If `equal_aspect` is true, the Y axis uses scaleanchor='x' to force
/// 1:1 pixel ratio (used by coupler trace plots).
fn add_plotly_multi_chart(
    html: &mut String,
    plot_id: &str,
    title: &str,
    x_axis_label: &str,
    y_axis_label: &str,
    series: &[(&str, String, &str, Option<&str>)],
    equal_aspect: bool,
    x_data_for_all: Option<&str>,
) {
    html.push_str(&format!("<div id='{}' class='plotly-chart'></div>\n", plot_id));
    html.push_str("<script>\n");
    html.push_str(&format!("Plotly.newPlot('{}', [\n", plot_id));
    for (i, (name, y_json, color, dash)) in series.iter().enumerate() {
        let comma = if i + 1 < series.len() { "," } else { "" };
        let dash_clause = match dash {
            Some(d) => format!(",dash:'{}'", d),
            None => String::new(),
        };
        // For multi-series time plots (reactions, energy) all series share
        // one X vector; for XY plots (coupler) each series has its own X.
        let x_expr = x_data_for_all.unwrap_or("null /* per-series x provided */");
        html.push_str(&format!(
            "  {{x:{},y:{},type:'scatter',mode:'lines',name:'{}',line:{{color:'{}'{}}}}}{}\n",
            x_expr, y_json, name, color, dash_clause, comma
        ));
    }
    let y_scale = if equal_aspect { ",scaleanchor:'x'" } else { "" };
    html.push_str(&format!(
        "], {{title:'{}',xaxis:{{title:'{}'}},yaxis:{{title:'{}'{}}},\
         margin:{{t:40,b:50,l:60,r:20}}}}, {{responsive:true}});\n",
        title, x_axis_label, y_axis_label, y_scale
    ));
    html.push_str("</script>\n");
}

/// Get current timestamp as a formatted string. Native uses
/// `std::time::SystemTime`; wasm32 uses `js_sys::Date::now()` because
/// `SystemTime::now()` panics with "time not implemented on this platform"
/// on `wasm32-unknown-unknown`.
#[cfg(not(target_arch = "wasm32"))]
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format_unix_timestamp(now)
}

#[cfg(target_arch = "wasm32")]
fn chrono_now() -> String {
    let now = (js_sys::Date::now() / 1000.0) as u64;
    format_unix_timestamp(now)
}

/// Format a Unix timestamp (seconds since epoch) as an approximate UTC date string.
///
/// Uses simple arithmetic instead of a chrono dependency. The month/day are
/// approximate (assumes 365-day years and 30-day months) but sufficient for
/// display purposes in generated reports.
fn format_unix_timestamp(secs: u64) -> String {
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        years,
        month.min(12),
        day.min(31),
        h,
        m,
        s,
    )
}

/// Render the "Loop equations" section listing per-constraint Φ_J* with
/// symbolic forms, residuals at the report's pose `q` (t = 0), and Lagrange
/// multipliers from `solve_statics`. Multipliers are omitted when the solve
/// fails (the column simply doesn't appear in that case).
///
/// Sourced from the same `gui::eq_rendering` helpers used by the live panel
/// (E1) and canvas overlay (E2), so all three views stay in sync.
fn write_loop_equations_section(
    html: &mut String,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) {
    use crate::core::constraint::Constraint;
    use crate::gui::eq_rendering::{
        kind_at, lambda_string, n_constraints, residual_norm_string, symbolic_form,
    };
    use crate::solver::assembly::assemble_constraints;
    use crate::solver::statics::solve_statics;

    if !mechanism.is_built() {
        return;
    }

    // Reports always evaluate at t = 0 — the report captures the current
    // pose, not a time series. A future per-frame report could override this.
    let t_mech = 0.0;
    let phi = assemble_constraints(mechanism, q, t_mech);
    let lambdas = solve_statics(mechanism, q, t_mech)
        .ok()
        .map(|s| s.lambdas);

    html.push_str("<h2>Loop Equations</h2>\n");
    let dof = mechanism.state().n_coords() as isize - mechanism.n_constraints() as isize;
    html.push_str(&format!(
        "<p>Constraint count m = {}, coordinate count n = {}, DOF = {}.</p>\n",
        mechanism.n_constraints(),
        mechanism.state().n_coords(),
        dof,
    ));

    let has_lambdas = lambdas.is_some();
    html.push_str("<table>\n<tr><th>ID</th><th>Kind</th><th>Symbolic Φ</th><th>Residual</th>");
    if has_lambdas {
        html.push_str("<th>λ multiplier</th>");
    }
    html.push_str("</tr>\n");

    let n_constr = n_constraints(mechanism);
    for (i, c) in mechanism.all_constraints().enumerate() {
        if i >= n_constr {
            break;
        }
        let Some(kind) = kind_at(mechanism, i) else { continue };
        let driver_class = if kind.is_driver() {
            " style='background:#fff3cd;'"
        } else {
            ""
        };
        let id = html_escape(c.id());
        let kind_lbl = kind.short_label();
        let sym = html_escape(&symbolic_form(mechanism, i));
        let res = html_escape(&residual_norm_string(mechanism, i, &phi));
        html.push_str(&format!(
            "<tr{}><td>{}</td><td><code>{}</code></td><td><code>{}</code></td><td><code>{}</code></td>",
            driver_class, id, kind_lbl, sym, res,
        ));
        if has_lambdas {
            let lam = html_escape(&lambda_string(mechanism, i, lambdas.as_ref()));
            html.push_str(&format!("<td><code>{}</code></td>", lam));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table>\n");
    if !has_lambdas {
        html.push_str("<p><em>Lagrange multipliers omitted: statics solve failed at this pose.</em></p>\n");
    }
}

/// Minimal HTML escaping for the loop-equations table cells. Symbolic forms
/// contain `<` (in `≤`/`≥` they don't, but defensive anyway) and the
/// driver-meta strings can include arbitrary user expressions, so we escape
/// the four characters that would break table markup.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Geometry of a four-bar linkage extracted from a mechanism, in the
/// canonical Freudenstein orientation (crank → coupler → rocker → ground).
#[derive(Debug, Clone)]
struct FourbarLayout {
    crank_id: String,
    coupler_id: String,
    rocker_id: String,
    /// World position of the ground anchor on the crank side (J1).
    o2_world: nalgebra::Vector2<f64>,
    /// World position of the crank-coupler joint (J2).
    b_world: nalgebra::Vector2<f64>,
    /// World position of the coupler-rocker joint (J3).
    c_world: nalgebra::Vector2<f64>,
    /// World position of the ground anchor on the rocker side (J4).
    o4_world: nalgebra::Vector2<f64>,
    /// Crank length |OB|.
    a: f64,
    /// Coupler length |BC|.
    b: f64,
    /// Rocker length |CO4|.
    c: f64,
    /// Ground length |O2 O4|.
    d: f64,
}

/// Detect whether the mechanism is a planar 4-bar (one driver, four
/// revolute joints, three moving bodies + ground) and extract the
/// crank/coupler/rocker assignment plus link lengths. Returns `None`
/// for any topology that doesn't match Freudenstein's setup, e.g.
/// slider-cranks, multi-driver mechanisms, non-revolute joints.
fn detect_fourbar(
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
) -> Option<FourbarLayout> {
    use crate::core::constraint::{Constraint, JointConstraint};
    use crate::core::state::GROUND_ID;

    // 4 bodies total = ground + 3 moving (crank, coupler, rocker).
    if mechanism.bodies().len() != 4 {
        return None;
    }
    let n_revolute = mechanism
        .joints()
        .iter()
        .filter(|j| matches!(j, JointConstraint::Revolute(_)))
        .count();
    if n_revolute != 4 || mechanism.joints().len() != 4 {
        return None;
    }
    if mechanism.drivers().len() != 1 || !mechanism.linear_drivers().is_empty() {
        return None;
    }

    let drv = mechanism.drivers().first()?;
    // Crank is the non-ground side of the driver.
    let crank_id = if drv.body_i_id() == GROUND_ID {
        drv.body_j_id().to_string()
    } else if drv.body_j_id() == GROUND_ID {
        drv.body_i_id().to_string()
    } else {
        return None;
    };

    // Helper: among the revolute joints, find one matching a topology test
    // and return the joint reference.
    let find_revolute = |a: &str, b: &str| {
        mechanism.joints().iter().find_map(|j| {
            if let JointConstraint::Revolute(r) = j {
                let i = r.body_i_id();
                let jb = r.body_j_id();
                if (i == a && jb == b) || (i == b && jb == a) {
                    Some(r)
                } else {
                    None
                }
            } else {
                None
            }
        })
    };

    // Identify rocker = body connected to ground via a non-driver revolute joint.
    let rocker_id = mechanism.joints().iter().find_map(|j| {
        if let JointConstraint::Revolute(r) = j {
            let other = if r.body_i_id() == GROUND_ID {
                Some(r.body_j_id().to_string())
            } else if r.body_j_id() == GROUND_ID {
                Some(r.body_i_id().to_string())
            } else {
                None
            };
            other.filter(|id| id != &crank_id)
        } else {
            None
        }
    })?;

    // Coupler is the remaining body.
    let coupler_id = mechanism
        .body_order()
        .iter()
        .find(|b| **b != crank_id && **b != rocker_id && b.as_str() != GROUND_ID)
        .cloned()?;

    let j1 = find_revolute(GROUND_ID, &crank_id)?;
    let j2 = find_revolute(&crank_id, &coupler_id)?;
    let j3 = find_revolute(&coupler_id, &rocker_id)?;
    let j4 = find_revolute(&rocker_id, GROUND_ID)?;

    // Helper: extract anchor on the requested body, transform to world.
    let world_anchor =
        |joint: &crate::core::constraint::RevoluteJoint, body: &str| -> nalgebra::Vector2<f64> {
            let local = if joint.body_i_id() == body {
                *joint.point_i_local()
            } else {
                *joint.point_j_local()
            };
            mechanism.state().body_point_global(body, &local, q)
        };

    let o2_world = world_anchor(j1, GROUND_ID);
    let b_world = world_anchor(j2, &crank_id);
    let c_world = world_anchor(j3, &rocker_id);
    let o4_world = world_anchor(j4, GROUND_ID);

    let d = (o4_world - o2_world).norm();
    let a = (b_world - o2_world).norm();
    let b_len = (c_world - b_world).norm();
    let c_len = (c_world - o4_world).norm();

    Some(FourbarLayout {
        crank_id,
        coupler_id,
        rocker_id,
        o2_world,
        b_world,
        c_world,
        o4_world,
        a,
        b: b_len,
        c: c_len,
        d,
    })
}

/// Append a "Freudenstein's equation" subsection to the report, but only
/// when the mechanism is a planar 4-bar in Freudenstein's standard
/// configuration. Skipped silently otherwise (slider-cranks, multi-driver
/// mechanisms, etc.).
fn write_freudenstein_section(
    html: &mut String,
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
) {
    let Some(layout) = detect_fourbar(mechanism, q) else {
        // Non-4-bar mechanism — leave a brief note so the user knows
        // this section is *available*, just not applicable here.
        html.push_str("<h3>4b. Freudenstein's equation (4-bar only)</h3>\n");
        html.push_str(
            "<p><em>This section appears only when the mechanism is a planar \
             4-bar with four revolute joints and one revolute driver. \
             Freudenstein's equation is the closed-form analytical \
             relationship between input and output angles for that family. \
             For other topologies (slider-crank, multi-driver, prismatic \
             joints), the constraint cascade in section 4 is the only \
             general approach.</em></p>\n",
        );
        return;
    };

    let FourbarLayout {
        crank_id,
        coupler_id,
        rocker_id,
        o2_world,
        b_world,
        c_world,
        o4_world,
        a,
        b,
        c,
        d,
    } = &layout;

    // Geometric angles measured relative to the ground vector (O4 - O2).
    // Robust to mounting_angle and arbitrary world rotation.
    let ground_vec = o4_world - o2_world;
    let ground_len = ground_vec.norm();
    if ground_len < 1e-12 {
        return; // Degenerate; can't define Freudenstein.
    }
    let ground_unit = ground_vec / ground_len;
    let ground_perp = nalgebra::Vector2::new(-ground_unit.y, ground_unit.x);

    let crank_vec = b_world - o2_world;
    let rocker_vec = c_world - o4_world;

    let theta_2 = crank_vec.y.atan2(crank_vec.x); // world-frame angle of OB
    let theta_4 = rocker_vec.y.atan2(rocker_vec.x); // world-frame angle of O4 C
    // Also compute relative-to-ground angles (the form Freudenstein assumes
    // when the ground is along +x).
    let theta_2_rel = crank_vec.dot(&ground_perp).atan2(crank_vec.dot(&ground_unit));
    let theta_4_rel = rocker_vec
        .dot(&ground_perp)
        .atan2(rocker_vec.dot(&ground_unit));

    let k1 = d / a;
    let k2 = d / c;
    let k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c);

    let lhs = k1 * theta_4_rel.cos() - k2 * theta_2_rel.cos() + k3;
    let rhs = (theta_2_rel - theta_4_rel).cos();
    let resid = (lhs - rhs).abs();

    html.push_str("<h3>4b. Freudenstein's equation (closed-form 4-bar check)</h3>\n");
    html.push_str(
        "<p>For a planar 4-bar linkage, the loop-closure equation \
         (vector sum of all four links) reduces algebraically to a single \
         scalar equation relating the input crank angle \
         \\(\\theta_2\\) to the output rocker angle \\(\\theta_4\\). \
         This is <b>Freudenstein's equation</b> [Freudenstein 1954, 1955; \
         see Norton, <em>Design of Machinery</em> ch. 4]:</p>\n",
    );
    html.push_str(
        "\\[ K_1\\,\\cos\\theta_4 \\;-\\; K_2\\,\\cos\\theta_2 \\;+\\; K_3 \\;=\\; \\cos(\\theta_2 - \\theta_4) \\]\n",
    );
    html.push_str(
        "<p>where</p>\n",
    );
    html.push_str(
        "\\[ K_1 = \\frac{d}{a},\\qquad K_2 = \\frac{d}{c},\\qquad K_3 = \\frac{a^2 - b^2 + c^2 + d^2}{2\\,a\\,c} \\]\n",
    );
    html.push_str(
        "<p>with link lengths \\(a\\) (crank, input), \\(b\\) (coupler), \
         \\(c\\) (rocker, output), and \\(d\\) (ground / fixed link). \
         Angles \\(\\theta_2\\) and \\(\\theta_4\\) are measured from the \
         ground line at the respective ground pivots \\(O_2\\) and \\(O_4\\). \
         Derivation: square and add the real / imaginary parts of the loop \
         equation \\(a\\,e^{i\\theta_2} + b\\,e^{i\\theta_3} - c\\,e^{i\\theta_4} - d = 0\\) \
         to eliminate the coupler angle \\(\\theta_3\\).</p>\n",
    );

    html.push_str(&format!(
        "<p><b>Detected layout</b>: crank = <code>{}</code>, coupler = \
         <code>{}</code>, rocker = <code>{}</code>. Ground line from \
         \\(O_2 = ({:.4}, {:.4})\\) m to \\(O_4 = ({:.4}, {:.4})\\) m \
         (length \\(d = {:.4}\\) m).</p>\n",
        html_escape(crank_id),
        html_escape(coupler_id),
        html_escape(rocker_id),
        o2_world.x,
        o2_world.y,
        o4_world.x,
        o4_world.y,
        d,
    ));

    html.push_str("<p><b>Link lengths and Freudenstein constants for this mechanism</b>:</p>\n");
    html.push_str(&format!(
        "\\[ a = {:.4}\\,\\text{{m}},\\quad b = {:.4}\\,\\text{{m}},\\quad c = {:.4}\\,\\text{{m}},\\quad d = {:.4}\\,\\text{{m}} \\]\n",
        a, b, c, d,
    ));
    html.push_str(&format!(
        "\\[ K_1 = \\frac{{d}}{{a}} = \\frac{{{:.4}}}{{{:.4}}} = {:.6} \\]\n",
        d, a, k1,
    ));
    html.push_str(&format!(
        "\\[ K_2 = \\frac{{d}}{{c}} = \\frac{{{:.4}}}{{{:.4}}} = {:.6} \\]\n",
        d, c, k2,
    ));
    html.push_str(&format!(
        "\\[ K_3 = \\frac{{a^2 - b^2 + c^2 + d^2}}{{2\\,a\\,c}} = {:.6} \\]\n",
        k3,
    ));

    html.push_str("<p><b>Numerical verification at this pose</b>:</p>\n");
    html.push_str(&format!(
        "\\[ \\theta_2 = {:.6}\\,\\text{{rad}} = {:.3}^\\circ,\\quad \
         \\theta_4 = {:.6}\\,\\text{{rad}} = {:.3}^\\circ \\]\n",
        theta_2_rel,
        theta_2_rel.to_degrees(),
        theta_4_rel,
        theta_4_rel.to_degrees(),
    ));
    html.push_str(&format!(
        "\\[ \\text{{LHS}} = K_1\\cos\\theta_4 - K_2\\cos\\theta_2 + K_3 = {:.6} \\]\n",
        lhs,
    ));
    html.push_str(&format!(
        "\\[ \\text{{RHS}} = \\cos(\\theta_2 - \\theta_4) = {:.6} \\]\n",
        rhs,
    ));
    html.push_str(&format!(
        "\\[ |\\text{{LHS}} - \\text{{RHS}}| = {:.3e} \\quad \\text{{(should be at floating-point noise)}} \\]\n",
        resid,
    ));

    if resid > 1e-6 {
        html.push_str(&format!(
            "<p style='color: #c62828;'><strong>Warning:</strong> Freudenstein \
             residual is {:.3e}, larger than expected. This may indicate the \
             ground line is not aligned with the world +x axis (the report \
             measures angles relative to the ground line directly, so this \
             should be fine for tilted mounts) or that the position solve \
             didn't fully converge.</p>\n",
            resid,
        ));
    } else {
        html.push_str(
            "<p>The Freudenstein residual at floating-point noise confirms the \
             constraint cascade in section 4 produces a pose consistent with \
             the closed-form analytical equation. <em>This is the canonical \
             4-bar consistency check</em> \u{2014} if you want a single number \
             that proves the simulator's kinematics is mathematically correct \
             for this mechanism, this is it.</p>\n",
        );
    }

    html.push_str(
        "<p>Note: world-frame angles (relative to the world +x axis, ignoring \
         any mounting angle) are \
        ",
    );
    html.push_str(&format!(
        "\\(\\theta_2^{{\\,world}} = {:.3}^\\circ\\), \
         \\(\\theta_4^{{\\,world}} = {:.3}^\\circ\\). The values used in the \
         Freudenstein check above are measured relative to the actual ground \
         line, which makes the equation hold even when the mechanism is \
         tilted.</p>\n",
        theta_2.to_degrees(),
        theta_4.to_degrees(),
    ));

    // ── Continued: classical 4-bar engineering equations ─────────────────
    write_grashof_section(html, &layout);
    write_transmission_angle_section(html, mechanism, q, &layout);
    write_velocity_ratio_section(html, &layout, theta_2_rel, theta_4_rel);
    write_singular_configs_section(html, &layout);
}

/// Append "Grashof condition + classification" subsection. The Grashof
/// inequality \(s + l \leq p + q\) — where s, l are the shortest and
/// longest links and p, q the other two — predicts whether the input
/// link can rotate fully (Class I), the mechanism oscillates (Class II),
/// or sits at the change-point boundary (Class III).
fn write_grashof_section(html: &mut String, layout: &FourbarLayout) {
    let lengths = [layout.a, layout.b, layout.c, layout.d];
    let shortest = lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let longest = lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // p, q = the two non-shortest, non-longest lengths.
    let s_plus_l = shortest + longest;
    let total = lengths.iter().sum::<f64>();
    let p_plus_q = total - s_plus_l;

    let (class, name) = if (s_plus_l - p_plus_q).abs() < 1e-9 {
        ("III", "Change-point")
    } else if s_plus_l < p_plus_q {
        ("I", "Grashof (full rotation possible)")
    } else {
        ("II", "Non-Grashof (all links oscillate)")
    };

    // Determine motion type for Class I — depends on which link is shortest.
    let motion_type = if class == "I" {
        if shortest == layout.d {
            "double-crank (drag-link): both crank and rocker rotate fully"
        } else if shortest == layout.a {
            "crank-rocker: input crank rotates fully, output rocker oscillates"
        } else if shortest == layout.c {
            "rocker-crank: input rocker oscillates, output crank rotates fully"
        } else {
            "double-rocker (Class I): both inputs oscillate, but with full coupler rotation"
        }
    } else {
        "—"
    };

    html.push_str("<h3>4c. Grashof's condition (motion classification)</h3>\n");
    html.push_str(
        "<p>Grashof's theorem [Grashof 1883] classifies a 4-bar by comparing \
         the shortest \\(s\\) and longest \\(l\\) link lengths against the \
         other two \\(p\\) and \\(q\\). The inequality</p>\n",
    );
    html.push_str("\\[ s + l \\;\\leq\\; p + q \\]\n");
    html.push_str(
        "<p>determines whether at least one link can complete a full \
         revolution. Three classes:</p>\n",
    );
    html.push_str("<ul>\n<li><b>Class I (Grashof, \\(s + l < p + q\\))</b>: at least one link rotates fully. Sub-types depend on which link is shortest:\n<ul>\n<li>If shortest = ground (\\(d\\)): <em>double-crank</em> (both crank and rocker rotate)</li>\n<li>If shortest = side link adjacent to ground: <em>crank-rocker</em> (the short side rotates, the opposite side oscillates)</li>\n<li>If shortest = coupler: <em>double-rocker</em> (Class I but unusual — both side links oscillate while coupler rotates fully)</li>\n</ul></li>\n<li><b>Class II (non-Grashof, \\(s + l > p + q\\))</b>: all links oscillate; no full revolution possible.</li>\n<li><b>Class III (change-point, \\(s + l = p + q\\))</b>: boundary case; mechanism can pass through dead-center configurations and switch assembly modes.</li>\n</ul>\n");

    html.push_str(&format!(
        "<p><b>For this mechanism</b>: \
         \\(s = {:.4}\\,\\text{{m}}\\), \\(l = {:.4}\\,\\text{{m}}\\), \
         \\(p + q = {:.4}\\,\\text{{m}}\\), so \
         \\(s + l = {:.4}\\,\\text{{m}}\\). \
         Result: <b>Class {}</b> ({}). Motion type: {}.</p>\n",
        shortest,
        longest,
        p_plus_q,
        s_plus_l,
        class,
        name,
        motion_type,
    ));
}

/// Append "Transmission angle" subsection — the angle between the coupler
/// and the output rocker. Determines force-transmission quality. Below
/// ~40° or above ~140° the mechanism transmits force inefficiently and
/// can stall. Computed at the report's pose plus the analytical extreme
/// values that occur when the input crank is collinear with the ground.
fn write_transmission_angle_section(
    html: &mut String,
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
    layout: &FourbarLayout,
) {
    use crate::core::state::GROUND_ID;
    // Transmission angle μ = angle BCD where B is crank-coupler, C is
    // coupler-rocker, D is rocker-ground. Specifically μ is the angle
    // between vector BC (along coupler) and CD (along rocker), measured
    // as the interior angle of the 4-bar at vertex C.
    let bc = layout.c_world - layout.b_world; // along coupler
    let cd = layout.o4_world - layout.c_world; // along rocker
    let cos_mu = bc.dot(&cd) / (bc.norm() * cd.norm());
    let mu = cos_mu.clamp(-1.0, 1.0).acos();
    let mu_deg = mu.to_degrees();

    // Analytical extremes: when the crank is collinear with the ground,
    // the diagonal length BD is at its extremes (a + d or |d - a|).
    // Then by law of cosines on triangle BCD:
    //   cos(μ) = (b² + c² - BD²) / (2 b c)
    let a = layout.a;
    let b = layout.b;
    let c = layout.c;
    let d = layout.d;
    let bd_max = a + d;
    let bd_min = (d - a).abs();
    let cos_mu_at_max_bd = (b * b + c * c - bd_max * bd_max) / (2.0 * b * c);
    let cos_mu_at_min_bd = (b * b + c * c - bd_min * bd_min) / (2.0 * b * c);
    let mu_at_max_bd = cos_mu_at_max_bd.clamp(-1.0, 1.0).acos();
    let mu_at_min_bd = cos_mu_at_min_bd.clamp(-1.0, 1.0).acos();

    // The "minimum transmission angle" reachable is whichever of these is
    // closer to 0 or 180° — i.e. whichever has the more extreme cos.
    let mu_min = mu_at_max_bd.min(mu_at_min_bd);
    let mu_max = mu_at_max_bd.max(mu_at_min_bd);

    html.push_str("<h3>4d. Transmission angle</h3>\n");
    html.push_str(
        "<p>The <b>transmission angle</b> \\(\\mu\\) is the angle between \
         the coupler and the output rocker, measured at joint C. It \
         determines how much of the input force from the coupler actually \
         drives the rocker (vs. wasted as a radial force into the rocker \
         pivot).</p>\n",
    );
    html.push_str(
        "<p>By the law of cosines applied to triangle BCD (where B is the \
         crank-coupler joint, C the coupler-rocker joint, D the rocker-\
         ground joint), with the diagonal length \\(BD = |O_2 D - O_2 B|\\):</p>\n",
    );
    html.push_str(
        "\\[ \\cos\\mu = \\frac{b^2 + c^2 - BD^2}{2\\,b\\,c} \\]\n",
    );
    html.push_str(
        "<p>The diagonal \\(BD\\) ranges from \\(|d - a|\\) (crank aligned \
         with ground, on the same side) to \\(d + a\\) (crank aligned with \
         ground, opposite sides). These two extremes give the global \
         min/max transmission angles for the mechanism over a full crank \
         rotation:</p>\n",
    );
    html.push_str(&format!(
        "\\[ BD_\\min = |d - a| = {:.4}\\,\\text{{m}},\\quad BD_\\max = d + a = {:.4}\\,\\text{{m}} \\]\n",
        bd_min, bd_max,
    ));
    html.push_str(&format!(
        "\\[ \\mu_\\min = {:.3}^\\circ,\\quad \\mu_\\max = {:.3}^\\circ \\]\n",
        mu_min.to_degrees(),
        mu_max.to_degrees(),
    ));
    let _ = (mu_deg, q, mechanism, GROUND_ID); // silence unused
    html.push_str(&format!(
        "<p><b>At the report's pose</b>: \\(\\mu = {:.3}^\\circ\\).</p>\n",
        mu_deg,
    ));
    html.push_str(
        "<p><b>Engineering rule of thumb</b>: \\(\\mu\\) should stay within \
         \\([40^\\circ, 140^\\circ]\\) for efficient force transmission. \
         Values approaching 0° or 180° mean the coupler is nearly \
         collinear with the rocker, and the rocker takes near-zero torque \
         from the crank \u{2014} the mechanism approaches a dead-point \
         (toggle) where small input forces cannot drive the output.</p>\n",
    );
    if mu_min.to_degrees() < 40.0 || mu_max.to_degrees() > 140.0 {
        html.push_str(
            "<p style='color: #c62828;'><strong>Warning:</strong> the \
             transmission angle range exceeds the \\([40^\\circ, 140^\\circ]\\) \
             rule-of-thumb at some point in the cycle. Consider redesigning \
             link proportions for better force transmission.</p>\n",
        );
    }
}

/// Append "Velocity ratio / mechanical advantage" subsection. For a 4-bar,
/// the angular velocity ratio \(\dot\theta_4 / \dot\theta_2\) follows from
/// differentiating Freudenstein's equation, giving a closed-form expression
/// for the instantaneous gain at any pose.
fn write_velocity_ratio_section(
    html: &mut String,
    layout: &FourbarLayout,
    theta_2: f64,
    theta_4: f64,
) {
    // Differentiate Freudenstein:
    //   K_1 cos θ_4 - K_2 cos θ_2 + K_3 = cos(θ_2 - θ_4)
    // d/dt:
    //   -K_1 sin(θ_4) θ̇_4 + K_2 sin(θ_2) θ̇_2 = -sin(θ_2 - θ_4)·(θ̇_2 - θ̇_4)
    // Solve for θ̇_4 / θ̇_2:
    //   θ̇_4 / θ̇_2 = [ -K_2 sin(θ_2) + sin(θ_2 - θ_4) ] / [ -K_1 sin(θ_4) + sin(θ_2 - θ_4) ]
    let a = layout.a;
    let c = layout.c;
    let d = layout.d;
    let k1 = d / a;
    let k2 = d / c;
    let s_diff = (theta_2 - theta_4).sin();
    let num = -k2 * theta_2.sin() + s_diff;
    let den = -k1 * theta_4.sin() + s_diff;
    let ratio = if den.abs() > 1e-12 { num / den } else { f64::NAN };

    // Mechanical advantage = inverse of velocity ratio (output torque /
    // input torque, by virtual work).
    let ma = if ratio.abs() > 1e-12 { 1.0 / ratio } else { f64::NAN };

    html.push_str("<h3>4e. Velocity ratio &amp; mechanical advantage</h3>\n");
    html.push_str(
        "<p>The angular velocity ratio \\(\\dot\\theta_4 / \\dot\\theta_2\\) \
         (output rocker speed over input crank speed) follows from \
         differentiating Freudenstein's equation w.r.t. time:</p>\n",
    );
    html.push_str(
        "\\[ \\frac{\\dot\\theta_4}{\\dot\\theta_2} \\;=\\; \\frac{-K_2 \\sin\\theta_2 + \\sin(\\theta_2 - \\theta_4)}{-K_1 \\sin\\theta_4 + \\sin(\\theta_2 - \\theta_4)} \\]\n",
    );
    html.push_str(
        "<p>By the principle of virtual work (assuming a lossless mechanism), \
         the mechanical advantage \\(MA = T_{\\text{out}}/T_{\\text{in}}\\) \
         is the reciprocal of the velocity ratio:</p>\n",
    );
    html.push_str(
        "\\[ MA \\;=\\; \\frac{T_4}{T_2} \\;=\\; \\frac{\\dot\\theta_2}{\\dot\\theta_4} \\]\n",
    );
    html.push_str(&format!(
        "<p><b>At the report's pose</b>: \
         \\(\\dot\\theta_4 / \\dot\\theta_2 = {:.4}\\), so \
         \\(MA = {:.4}\\).</p>\n",
        ratio, ma,
    ));
    html.push_str(
        "<p>MA &gt;&gt; 1 means the mechanism amplifies torque (e.g. clamps, \
         toggle presses); MA &lt;&lt; 1 means it amplifies speed (e.g. \
         flying-shear cutters). MA → ∞ near dead points where velocity \
         ratio → 0 — these are the high-force regions.</p>\n",
    );
}

/// Append "Singular configurations" subsection — derives the dead-point
/// conditions analytically. Dead points occur when the transmission
/// angle is 0° or 180° (coupler and rocker collinear), making the
/// mechanism unable to propagate input motion to the output.
fn write_singular_configs_section(html: &mut String, layout: &FourbarLayout) {
    let a = layout.a;
    let b = layout.b;
    let c = layout.c;
    let d = layout.d;
    // Dead points: BD is at extreme, μ = 0 or π. Need b² + c² ± 2bc = BD².
    // BD = b + c (μ = 0) or BD = |b - c| (μ = π).
    // BD also = a + d or |d - a| (crank collinear with ground). So dead
    // points exist iff one of {b+c, |b-c|} equals one of {a+d, |d-a|}.
    let bd_at_0 = b + c;
    let bd_at_pi = (b - c).abs();
    let bd_max = a + d;
    let bd_min = (d - a).abs();
    let near = |x: f64, y: f64| (x - y).abs() < 1e-6;
    let dead_at_a_plus_d_zero = near(bd_max, bd_at_0);
    let dead_at_a_plus_d_pi = near(bd_max, bd_at_pi);
    let dead_at_d_minus_a_zero = near(bd_min, bd_at_0);
    let dead_at_d_minus_a_pi = near(bd_min, bd_at_pi);
    let any_dead = dead_at_a_plus_d_zero || dead_at_a_plus_d_pi
        || dead_at_d_minus_a_zero || dead_at_d_minus_a_pi;

    html.push_str("<h3>4f. Singular configurations (dead points)</h3>\n");
    html.push_str(
        "<p>A <b>dead point</b> (or toggle position) occurs when the coupler \
         is collinear with the rocker — equivalently, when \\(\\mu = 0\\) \
         or \\(\\mu = \\pi\\). At these poses, the mechanism's instantaneous \
         velocity ratio \\(\\dot\\theta_4 / \\dot\\theta_2 \\to 0\\), so no \
         torque applied to the input crank can move the output rocker. \
         Whether the mechanism <em>has</em> dead points is geometric:</p>\n",
    );
    html.push_str(
        "\\[ BD = b + c \\quad\\text{($\\mu = 0$)} \\quad \\text{or} \\quad BD = |b - c| \\quad\\text{($\\mu = \\pi$)} \\]\n",
    );
    html.push_str(
        "<p>Combined with the constraint that \\(BD\\) ranges over \
         \\([|d - a|, d + a]\\) over a full crank rotation, dead points \
         exist iff at least one of the four combinations \
         \\(\\{|d - a|, d + a\\} = \\{|b - c|, b + c\\}\\) is realisable.</p>\n",
    );
    html.push_str(&format!(
        "<p><b>For this mechanism</b>: \
         \\(b + c = {:.4}\\,\\text{{m}}\\), \\(|b - c| = {:.4}\\,\\text{{m}}\\), \
         \\(d + a = {:.4}\\,\\text{{m}}\\), \\(|d - a| = {:.4}\\,\\text{{m}}\\). \
         Dead-point condition: <b>{}</b>.</p>\n",
        bd_at_0, bd_at_pi, bd_max, bd_min,
        if any_dead {
            "DEAD POINTS PRESENT — the mechanism passes through at least one toggle position per cycle."
        } else {
            "no exact dead points (the geometry never quite lines up). Note transmission angle still gets close to 0° or 180° if the link lengths are near the threshold; see §4d."
        },
    ));
    html.push_str(
        "<p><b>Engineering significance</b>: dead points are bad for \
         actuators that drive through the input but useful for clamping \
         applications where infinite mechanical advantage is desired \
         momentarily. Class III (change-point) mechanisms always have dead \
         points by construction — the boundary \\(s + l = p + q\\) \
         literally <em>is</em> the BD-equality condition.</p>\n",
    );
}

/// Escape a string for inclusion inside a KaTeX `\text{...}` group.
///
/// KaTeX strict mode rejects bare LaTeX-special characters (notably `_`,
/// which it interprets as a math-mode subscript even inside `\text{}`).
/// User-supplied body names like `link_1` would silently fail to render
/// without this — the surrounding `\[ ... \]` would just appear as raw
/// source text in the report. The escapes here cover the cases that
/// actually appear in body / joint identifiers; we don't try to handle
/// arbitrary LaTeX (e.g. `$`, `\`) since those don't appear in valid IDs.
fn latex_text_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '_' => out.push_str("\\_"),
            '&' => out.push_str("\\&"),
            '%' => out.push_str("\\%"),
            '$' => out.push_str("\\$"),
            '#' => out.push_str("\\#"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            // `~` and `^` need both an escape and `{}` to render as plain
            // chars in text mode. Body IDs almost never use these, but
            // handle defensively.
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            // `\` gets passed through as `\textbackslash{}`. Body IDs
            // shouldn't contain it, but the escape keeps KaTeX happy.
            '\\' => out.push_str("\\textbackslash{}"),
            _ => out.push(c),
        }
    }
    out
}

/// Summarize a force element as (type_name, detail_string).
fn force_element_summary(fe: &crate::forces::elements::ForceElement) -> (&'static str, String) {
    use crate::forces::elements::ForceElement;
    match fe {
        ForceElement::Gravity(e) => ("Gravity", format!("g = ({:.2}, {:.2}) m/s^2", e.g_vector[0], e.g_vector[1])),
        ForceElement::LinearSpring(e) => ("Linear Spring", format!("k = {:.1} N/m, L0 = {:.4} m", e.stiffness, e.free_length)),
        ForceElement::TorsionSpring(e) => ("Torsion Spring", format!("k = {:.2} N*m/rad, free = {:.2} rad", e.stiffness, e.free_angle)),
        ForceElement::LinearDamper(e) => ("Linear Damper", format!("c = {:.2} N*s/m", e.damping)),
        ForceElement::RotaryDamper(e) => ("Rotary Damper", format!("c = {:.2} N*m*s/rad", e.damping)),
        ForceElement::ExternalForce(e) => ("External Force", format!("F = ({:.2}, {:.2}) N on {}", e.force[0], e.force[1], e.body_id)),
        ForceElement::ExternalTorque(e) => ("External Torque", format!("T = {:.2} N*m on {}", e.torque, e.body_id)),
        ForceElement::GasSpring(e) => ("Gas Spring", format!("F0 = {:.1} N, stroke = {:.4} m", e.initial_force, e.stroke)),
        ForceElement::BearingFriction(e) => ("Bearing Friction", format!("drag = {:.3}, viscous = {:.3}", e.constant_drag, e.viscous_coeff)),
        ForceElement::JointLimit(e) => ("Joint Limit", format!("range = [{:.1}, {:.1}] rad, k = {:.0} N*m/rad", e.angle_min, e.angle_max, e.stiffness)),
        ForceElement::Motor(e) => ("Motor", format!("T_stall = {:.2} N*m, w_nl = {:.2} rad/s", e.stall_torque, e.no_load_speed)),
        ForceElement::LinearActuator(e) => ("Linear Actuator", format!("F = {:.1} N, v_max = {:.3} m/s", e.force, e.speed_limit)),
        ForceElement::ForceZone(e) => ("Force Zone", format!("F = ({:.1}, {:.1}) N on {}", e.force[0], e.force[1], e.body_id)),
    }
}

/// Serialize a `Vec<f64>` to a JSON array string, replacing NaN/Infinity with null.
///
/// Plotly.js handles `null` gracefully (gaps in the trace) but not NaN.
fn float_vec_to_json(values: &[f64]) -> String {
    let mut buf = String::with_capacity(values.len() * 8 + 2);
    buf.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            buf.push(',');
        }
        if v.is_finite() {
            // Use enough precision to avoid visible stairstepping in plots.
            buf.push_str(&format!("{:.6}", v));
        } else {
            buf.push_str("null");
        }
    }
    buf.push(']');
    buf
}

/// Append a "Mathematical Derivation" section to the report — explains the
/// constraint-based kinematics + statics pipeline with KaTeX-rendered LaTeX
/// equations, parameterised by the actual mechanism (body count, n, m, DOF).
/// The math is the same for every planar linkage; only the dimensions change.
///
/// LaTeX delimiters: `\(...\)` inline, `\[...\]` display. The KaTeX
/// auto-render call in the document head walks the body and converts both.
///
/// Subsections (h3): Coordinates, Constraint Vector, Jacobian, Solve Cascade
/// (Position/Velocity/Acceleration/Statics + numeric outputs), Lagrange
/// Multipliers, Trajectory Mode.
fn write_math_background_section(
    html: &mut String,
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
) {
    use crate::core::state::GROUND_ID;
    use crate::solver::kinematics::{solve_acceleration, solve_velocity};

    if !mechanism.is_built() {
        return;
    }

    let n = mechanism.state().n_coords();
    let m = mechanism.n_constraints();
    let dof = n as isize - m as isize;
    let moving: Vec<&String> = mechanism
        .body_order()
        .iter()
        .filter(|b| b.as_str() != GROUND_ID)
        .collect();
    let n_moving_bodies = moving.len();
    let body_list: String = moving
        .iter()
        .map(|b| format!("<code>{}</code>", html_escape(b)))
        .collect::<Vec<_>>()
        .join(", ");

    html.push_str("<h2>Mathematical Derivation</h2>\n");
    html.push_str(
        "<p>Every number in the Loop Equations table comes from the same \
         four-level cascade: position solve \u{2192} velocity solve \u{2192} \
         acceleration solve \u{2192} statics. This section walks through it \
         with the same equations the simulator uses, plus the numeric solve \
         outputs at this pose.</p>\n",
    );

    // ── 1. Coordinates ────────────────────────────────────────────────────
    html.push_str("<h3>1. Coordinates \\(q\\)</h3>\n");
    html.push_str(&format!(
        "<p>Each non-ground body has 3 planar DOF: \\((x, y, \\theta)\\). This \
         mechanism has {} moving bodies ({}), so \\(q \\in \\mathbb{{R}}^{{{}}}\\):</p>\n",
        n_moving_bodies, body_list, n,
    ));
    // Build the q-vector display with each body's coords grouped.
    // Body names go inside \text{}; KaTeX strict mode rejects raw `_` and
    // other LaTeX-special characters there, so escape via latex_text_escape.
    // (Bodies named "link_1" etc. would otherwise leave the equation
    // un-rendered as raw source — bug reported by user 2026-05-01.)
    let body_q_strs: Vec<String> = moving
        .iter()
        .map(|b| {
            let safe = latex_text_escape(b);
            format!(
                "x_{{\\text{{{0}}}}}\\,y_{{\\text{{{0}}}}}\\,\\theta_{{\\text{{{0}}}}}",
                safe
            )
        })
        .collect();
    html.push_str(&format!(
        "\\[ q = \\begin{{bmatrix}} {} \\end{{bmatrix}}^T \\]\n",
        body_q_strs.join(" \\,\\big|\\, "),
    ));
    html.push_str(
        "<p>\\(r_i = (x_i, y_i)\\) is body \\(i\\)'s CG in the world frame; \
         \\(\\theta_i\\) is its orientation. Ground is locked at the origin \
         and contributes no coordinates. The \\(2\\times 2\\) rotation matrix is</p>\n",
    );
    html.push_str(
        "\\[ A(\\theta) = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix} \\]\n",
    );
    html.push_str(
        "<p>A body-local point \\(s\\) (e.g. a joint anchor in body coords) maps \
         to world via \\(r + A(\\theta)\\,s\\).</p>\n",
    );

    // ── 2. Constraint vector ──────────────────────────────────────────────
    html.push_str("<h3>2. Constraint vector \\(\\Phi(q,t)\\)</h3>\n");
    html.push_str(&format!(
        "<p>Stacks one row per constraint equation. This mechanism has \\(m = {}\\) \
         total constraint equations across the joints and drivers listed in the \
         Loop Equations table above.</p>\n",
        m,
    ));
    html.push_str("<p><b>Revolute joint</b> (2 eqs each, \\(\\Phi_{\\mathrm{rev}}\\)):</p>\n");
    html.push_str(
        "\\[ \\Phi_{\\mathrm{rev}}:\\ r_i + A(\\theta_i)\\,s_i^A - r_j - A(\\theta_j)\\,s_j^A = 0 \\]\n",
    );
    html.push_str(
        "<p>Says \"the two bodies share a pivot at this point.\" \\(s_i^A\\) is the \
         anchor's coords in body \\(i\\)'s local frame. Two scalar equations because \
         it's a 2D vector equality.</p>\n",
    );
    html.push_str("<p><b>Revolute driver</b> (1 eq, \\(\\Phi_{\\mathrm{rd}}\\)):</p>\n");
    html.push_str(
        "\\[ \\Phi_{\\mathrm{rd}}:\\ \\theta_j - \\theta_i - f(t) = 0,\\quad f(t) = \\theta_0 + \\omega\\,t \\]\n",
    );
    html.push_str("<p><b>Linear driver</b> (1 eq, \\(\\Phi_{\\mathrm{ld}}\\)):</p>\n");
    html.push_str(
        "\\[ \\Phi_{\\mathrm{ld}}:\\ \\| P_b - P_a \\| - L(t) = 0 \\]\n",
    );
    html.push_str(
        "<p>where \\(P_a = r_i + A(\\theta_i)\\,s_i^A\\) and \
         \\(P_b = r_j + A(\\theta_j)\\,s_j^A\\) are the world-frame positions of \
         the actuator's two attachment points, and \\(L(t)\\) is the prescribed \
         stroke length as a function of time. Constrains the distance between \
         two body points; geometrically this is a cylinder + piston whose total \
         length is driven kinematically.</p>\n",
    );
    html.push_str(
        "<p>Driver rows are the only rows that depend on \\(t\\). For \
         \\(\\Phi_{\\mathrm{ld}}\\), the velocity-level partial \
         \\(\\Phi_t = -\\dot L(t)\\) is the prescribed stroke rate; at the \
         acceleration level \\(\\Phi_{tt} = -\\ddot L(t)\\). The acceleration \
         RHS \\(\\gamma\\) for a linear driver also includes the centripetal \
         <code>v_perp²/L</code> term, where v_perp is the velocity component \
         perpendicular to the line of action.</p>\n",
    );
    html.push_str(
        "<p><b>LinearActuator force element</b> (not a constraint): if the \
         mechanism uses a LinearActuator force element instead of a LinearDriver \
         constraint, the actuator applies a constant force \\(F\\) along the \
         line of action between its two attachment points. The stroke length is \
         a <em>computed</em> output (not prescribed), and the actuator force \
         appears in \\(Q_{\\text{applied}}\\) for the statics solve rather than \
         in \\(\\Phi\\). Drawn in orange in the schematic above to distinguish \
         from constraint-style drivers (red).</p>\n",
    );
    html.push_str(
        "<p>At a feasible pose, \\(\\Phi = 0\\). The Residual column in the \
         Loop Equations table shows \\(\\|\\Phi_J\\|\\) per row \u{2014} should \
         all be \\(\\le 10^{-15}\\) (floating-point noise). A non-trivial \
         residual means the position solve didn't converge and downstream \
         forces / energies aren't trustworthy.</p>\n",
    );

    // ── 3. Jacobian ────────────────────────────────────────────────────────
    html.push_str("<h3>3. Constraint Jacobian \\(\\Phi_q\\)</h3>\n");
    html.push_str(&format!(
        "\\[ \\Phi_q = \\frac{{\\partial \\Phi}}{{\\partial q}} \\in \\mathbb{{R}}^{{{m} \\times {n}}} \\]\n",
        m = m, n = n,
    ));
    html.push_str(
        "<p>Block-sparse: each constraint row only has non-zero entries in the \
         columns belonging to the bodies it touches. With \
         \\(B(\\theta) = dA/d\\theta\\):</p>\n",
    );
    html.push_str(
        "<p>For \\(\\Phi_{\\mathrm{rev}}\\) between bodies \\(i\\) and \\(j\\):</p>\n",
    );
    html.push_str(
        "\\[ \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial r_i} = +I,\\quad \
         \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial \\theta_i} = +B(\\theta_i)\\,s_i^A,\\quad \
         \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial r_j} = -I,\\quad \
         \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial \\theta_j} = -B(\\theta_j)\\,s_j^A \\]\n",
    );
    html.push_str("<p>For \\(\\Phi_{\\mathrm{rd}}\\):</p>\n");
    html.push_str(
        "\\[ \\frac{\\partial \\Phi_{\\mathrm{rd}}}{\\partial \\theta_i} = -1,\\quad \
         \\frac{\\partial \\Phi_{\\mathrm{rd}}}{\\partial \\theta_j} = +1 \\]\n",
    );
    html.push_str(&format!(
        "<p><b>Determinacy check:</b> \\(n - m = {} - {} = {}\\). {}</p>\n",
        n,
        m,
        dof,
        if dof == 0 {
            "DOF = 0 means kinematically determinate \u{2014} the driver fully constrains the pose."
        } else if dof > 0 {
            "DOF &gt; 0 means under-constrained \u{2014} additional drivers needed to fix the pose."
        } else {
            "DOF &lt; 0 means over-constrained \u{2014} redundant constraints, may be inconsistent."
        },
    ));

    // ── 4. Solve cascade ───────────────────────────────────────────────────
    html.push_str("<h3>4. The four-level solve cascade</h3>\n");
    html.push_str("<p>Each level uses the same \\(\\Phi_q\\) and adds one time derivative.</p>\n");

    html.push_str("<p><b>Position</b> (find \\(q\\) such that \\(\\Phi=0\\)):</p>\n");
    html.push_str("\\[ \\Phi_q\\,\\Delta q = -\\Phi(q_k, t),\\qquad q_{k+1} = q_k + \\Delta q \\]\n");
    html.push_str(
        "<p>Iterate until \\(\\|\\Phi\\| < 10^{-10}\\). Source: \
         <code>src/solver/kinematics.rs</code>.</p>\n",
    );

    html.push_str(
        "<p><b>Velocity</b> (differentiate \\(\\Phi(q(t),t) = 0\\) once w.r.t. \\(t\\)):</p>\n",
    );
    html.push_str("\\[ \\Phi_q\\,\\dot q + \\Phi_t = 0 \\quad\\Longrightarrow\\quad \\Phi_q\\,\\dot q = -\\Phi_t \\]\n");
    html.push_str(
        "<p>A linear solve. The driver row's \\(-\\omega\\) injects \"the driven \
         body must rotate at \\(\\omega\\) rad/s\"; the joint rows propagate \
         that through the linkage.</p>\n",
    );

    html.push_str("<p><b>Acceleration</b> (differentiate again):</p>\n");
    html.push_str("\\[ \\Phi_q\\,\\ddot q = \\gamma(q, \\dot q, t) \\]\n");
    html.push_str(
        "<p>Where the RHS \\(\\gamma\\) is the assembled \"everything except \
         \\(\\Phi_q\\,\\ddot q\\)\" terms \u{2014} for revolute joints it's \
         \\(-B(\\theta)\\,s\\,\\dot\\theta^2\\) (centripetal), for the driver \
         \\(-\\ddot f(t)\\). Source: \
         <code>src/solver/assembly.rs::assemble_gamma</code>.</p>\n",
    );

    html.push_str("<p><b>Statics</b> (Lagrange multipliers from force balance):</p>\n");
    html.push_str(
        "\\[ M\\,\\ddot q = Q_{\\text{applied}} + \\Phi_q^T\\,\\lambda,\\qquad \\Phi_q\\,\\ddot q = \\gamma \\]\n",
    );
    html.push_str("<p>For pure statics (\\(\\ddot q = 0\\)):</p>\n");
    html.push_str("\\[ \\Phi_q^T\\,\\lambda = -Q_{\\text{applied}} \\]\n");
    html.push_str(
        "<p>Where \\(M\\) is the block-diagonal mass matrix, \
         \\(Q_{\\text{applied}}\\) are external forces (gravity, springs, motors), \
         and \\(\\lambda \\in \\mathbb{R}^m\\) are the Lagrange multipliers \
         \u{2014} one per constraint equation. Source: \
         <code>src/solver/statics.rs::solve_statics</code>.</p>\n",
    );

    // ── 4a. Numeric solve outputs at the report's pose ─────────────────────
    html.push_str("<h3>4a. Numeric solve outputs at this pose</h3>\n");
    html.push_str(
        "<p>The four-level cascade above produces concrete \\(q\\), \\(\\dot q\\), \
         and \\(\\ddot q\\) vectors. Below are the actual values at this report's \
         pose (\\(t = 0\\)), one row per moving body. Pose came from \
         <code>solve_position</code>; \\(\\dot q\\) from <code>solve_velocity</code> \
         (linear solve of \\(\\Phi_q\\,\\dot q = -\\Phi_t\\)); \\(\\ddot q\\) \
         from <code>solve_acceleration</code> (linear solve of \
         \\(\\Phi_q\\,\\ddot q = \\gamma\\)).</p>\n",
    );

    let q_dot = solve_velocity(mechanism, q, 0.0).ok();
    let q_ddot = q_dot
        .as_ref()
        .and_then(|qd| solve_acceleration(mechanism, q, qd, 0.0).ok());
    let mech_state = mechanism.state();

    // Position table
    html.push_str("<p><b>Position</b> \\(q\\):</p>\n");
    html.push_str("<table><tr><th>Body</th><th>\\(x\\) (m)</th><th>\\(y\\) (m)</th><th>\\(\\theta\\) (rad)</th></tr>\n");
    for body_id in &moving {
        let (x, y, theta) = mech_state.get_pose(body_id, q);
        html.push_str(&format!(
            "<tr><td><code>{}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td></tr>\n",
            html_escape(body_id), x, y, theta,
        ));
    }
    html.push_str("</table>\n");

    // Velocity table
    if let Some(ref qd) = q_dot {
        html.push_str("<p><b>Velocity</b> \\(\\dot q\\):</p>\n");
        html.push_str("<table><tr><th>Body</th><th>\\(\\dot x\\) (m/s)</th><th>\\(\\dot y\\) (m/s)</th><th>\\(\\dot\\theta\\) (rad/s)</th></tr>\n");
        for body_id in &moving {
            let (xd, yd, td) = mech_state.get_pose(body_id, qd);
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td></tr>\n",
                html_escape(body_id), xd, yd, td,
            ));
        }
        html.push_str("</table>\n");
    } else {
        html.push_str("<p><em>Velocity solve failed at this pose.</em></p>\n");
    }

    // Acceleration table
    if let Some(ref qdd) = q_ddot {
        html.push_str("<p><b>Acceleration</b> \\(\\ddot q\\):</p>\n");
        html.push_str("<table><tr><th>Body</th><th>\\(\\ddot x\\) (m/s\\(^2\\))</th><th>\\(\\ddot y\\) (m/s\\(^2\\))</th><th>\\(\\ddot\\theta\\) (rad/s\\(^2\\))</th></tr>\n");
        for body_id in &moving {
            let (xdd, ydd, tdd) = mech_state.get_pose(body_id, qdd);
            html.push_str(&format!(
                "<tr><td><code>{}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td><td><code>{:+.6}</code></td></tr>\n",
                html_escape(body_id), xdd, ydd, tdd,
            ));
        }
        html.push_str("</table>\n");
    } else {
        html.push_str(
            "<p><em>Acceleration solve failed at this pose (typically because the \
             velocity solve failed, or the mechanism is at a singular configuration \
             where \\(\\Phi_q\\) is rank-deficient).</em></p>\n",
        );
    }

    // ── 4b. Freudenstein's equation (closed-form 4-bar) ────────────────────
    write_freudenstein_section(html, mechanism, q);

    // ── 5. Lagrange multipliers ────────────────────────────────────────────
    html.push_str("<h3>5. Why \\(\\lambda\\) = reaction forces / driver torque</h3>\n");
    html.push_str(
        "<p>\\(\\Phi_q^T\\,\\lambda\\) is the generalized force the constraints \
         exert on \\(q\\). For a revolute joint between bodies \\(i\\) and \\(j\\):</p>\n",
    );
    html.push_str(
        "\\[ \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial r_i} = +I \\;\\Rightarrow\\; \\lambda \\text{ contributes } +\\lambda \\text{ to body } i \\text{'s translational EOM} \\]\n",
    );
    html.push_str(
        "\\[ \\frac{\\partial \\Phi_{\\mathrm{rev}}}{\\partial r_j} = -I \\;\\Rightarrow\\; \\lambda \\text{ contributes } -\\lambda \\text{ to body } j \\text{'s} \\]\n",
    );
    html.push_str(
        "<p>That's literally Newton's third law on the joint reaction. So \
         <b>\\(\\lambda\\) for a revolute constraint is the force vector \
         \\((F_x, F_y)\\) body \\(j\\) exerts on body \\(i\\) through the pin</b> \
         \u{2014} units of newtons. The components shown in the \\(\\lambda\\) \
         column above are exactly this force vector at the report's pose.</p>\n",
    );
    html.push_str(
        "<p>For the revolute driver row, \
         \\(\\partial \\Phi_{\\mathrm{rd}}/\\partial \\theta_j = +1\\), so</p>\n",
    );
    html.push_str(
        "\\[ +\\lambda\\,\\partial \\theta_j + (-\\lambda)\\,\\partial \\theta_i \\]\n",
    );
    html.push_str(
        "<p>appears in the \\(\\theta\\) equations of motion. That's a torque \
         pair: body \\(j\\) gets \\(+\\lambda\\) N\u{00B7}m, body \\(i\\) gets \
         \\(-\\lambda\\) N\u{00B7}m. So <b>\\(\\lambda\\) for the driver is the \
         torque the actuator must apply between the two bodies</b> to enforce \
         the prescribed \\(\\theta_j - \\theta_i = f(t)\\).</p>\n",
    );

    // ── 6. Trajectory mode ─────────────────────────────────────────────────
    html.push_str("<h3>6. Trajectory mode (inverse position control)</h3>\n");
    html.push_str(
        "<p>Forward sweep prescribes \\(f(t)\\) (the driver) and solves for \
         \\(q(t)\\). <b>Trajectory mode</b> prescribes a separate observable \
         \\(g(q) = h(t)\\) (e.g. WorldX of a coupler point), then a Newton outer \
         loop adjusts the driver input \\(u\\) until \\(g(q(u)) = h(t)\\):</p>\n",
    );
    html.push_str("\\[ r(u) = g(q(u)) - h(t) = 0 \\]\n");
    html.push_str(
        "\\[ u_{k+1} = u_k - \\frac{r(u_k)}{r'(u_k)},\\qquad r'(u) = \\nabla g(q)\\cdot\\frac{dq}{du} \\]\n",
    );
    html.push_str(
        "<p>The four-level cascade above runs at every outer-loop iteration, \
         with \\(u\\) swapped into the driver row in place of \\(f(t)\\). The \
         full velocity / acceleration inverses are derived in \
         <code>docs/superpowers/specs/2026-04-29-linkage-equations-reference.md</code> \
         \u{00A7}8 if you want \\(r'(u)\\) and \\(r''(u)\\) worked out per \
         ControlTarget variant.</p>\n",
    );

    // ── 7. Numerical validation ─────────────────────────────────────────────
    //
    // Shows the cascade actually working: residuals at each level should
    // be at floating-point noise. Lets a reader spot-check the solver
    // outputs against an external tool (Mathematica, MATLAB, NumPy, etc.)
    // by giving them concrete numbers to compare to.
    html.push_str("<h3>7. Numerical validation at this pose</h3>\n");
    html.push_str(
        "<p>The four-level cascade is mathematically self-consistent: at the \
         converged pose, every residual should be at floating-point noise \
         (\\(\\le 10^{-10}\\)). The numbers below let you spot-check the \
         solver against an external tool by computing the same residuals \
         from \\(q\\), \\(\\dot q\\), \\(\\ddot q\\) (section 4a) and \
         \\(\\Phi_q\\) (assembled per the partials in section 3).</p>\n",
    );

    use crate::solver::assembly::{assemble_constraints, assemble_jacobian, assemble_phi_t};
    let phi = assemble_constraints(mechanism, q, 0.0);
    let phi_q_mat = assemble_jacobian(mechanism, q, 0.0);
    let phi_t = assemble_phi_t(mechanism, q, 0.0);
    let phi_norm = phi.norm();
    html.push_str(
        "<p><b>Position residual</b> \\(\\|\\Phi(q, t)\\|\\):</p>\n",
    );
    html.push_str(&format!(
        "\\[ \\|\\Phi\\| = {:.3e} \\quad \\text{{(should be near zero)}} \\]\n",
        phi_norm,
    ));

    if let Some(ref qd) = q_dot {
        // Velocity check: Φ_q · q̇ + Φ_t = 0
        let vel_resid = (&phi_q_mat * qd) + &phi_t;
        html.push_str(
            "<p><b>Velocity residual</b> \\(\\|\\Phi_q\\,\\dot q + \\Phi_t\\|\\) \
             — verifies the velocity solve in section 4a is consistent with \
             \\(\\Phi_q\\) and \\(\\Phi_t\\):</p>\n",
        );
        html.push_str(&format!(
            "\\[ \\|\\Phi_q\\,\\dot q + \\Phi_t\\| = {:.3e} \\]\n",
            vel_resid.norm(),
        ));
    }

    if let (Some(ref qd), Some(ref qdd)) = (q_dot.as_ref(), q_ddot.as_ref()) {
        // Acceleration check: Φ_q · q̈ - γ = 0
        use crate::solver::assembly::assemble_gamma;
        let gamma = assemble_gamma(mechanism, q, qd, 0.0);
        let acc_resid = (&phi_q_mat * *qdd) - &gamma;
        html.push_str(
            "<p><b>Acceleration residual</b> \\(\\|\\Phi_q\\,\\ddot q - \\gamma\\|\\) \
             — verifies the acceleration solve is consistent with \\(\\gamma\\):</p>\n",
        );
        html.push_str(&format!(
            "\\[ \\|\\Phi_q\\,\\ddot q - \\gamma\\| = {:.3e} \\]\n",
            acc_resid.norm(),
        ));
    }

    // Full Φ_q matrix as a numerical table. Compact format: scientific
    // notation, ~4 decimal places. Column headers correspond to q's
    // layout (each body's x, y, θ).
    html.push_str(&format!(
        "<p><b>Constraint Jacobian \\(\\Phi_q\\)</b> at this pose ({} \u{00d7} {} matrix). \
         Rows are constraint rows in the order shown in the Loop Equations \
         table; columns are q components in body-order. Reproduce by \
         differentiating each \\(\\Phi\\) row w.r.t. each q component per \
         section 3.</p>\n",
        m, n,
    ));
    html.push_str("<table style='font-size: 11px; font-family: monospace;'>\n<tr><th></th>");
    // Column headers: x_body, y_body, θ_body for each moving body.
    for body_id in &moving {
        let safe = html_escape(body_id);
        html.push_str(&format!(
            "<th>x_{0}</th><th>y_{0}</th><th>θ_{0}</th>",
            safe
        ));
    }
    html.push_str("</tr>\n");
    for i in 0..m {
        html.push_str(&format!("<tr><td><b>row {}</b></td>", i + 1));
        for j in 0..n {
            let val = phi_q_mat[(i, j)];
            // Highlight non-zero entries; zero/near-zero entries get muted.
            let style = if val.abs() < 1e-12 {
                "color: #ccc;"
            } else {
                ""
            };
            html.push_str(&format!(
                "<td style='{}'>{:.3e}</td>",
                style, val,
            ));
        }
        html.push_str("</tr>\n");
    }
    html.push_str("</table>\n");

    // ── 8. Verification recipe ─────────────────────────────────────────────
    html.push_str("<h3>8. How to verify the math externally</h3>\n");
    html.push_str(
        "<p>To validate this report's solver outputs against an independent \
         tool (Mathematica, MATLAB, SymPy, Python+NumPy, etc.), follow these \
         steps:</p>\n",
    );
    html.push_str("<ol>\n");
    html.push_str(
        "<li><b>Read \\(q\\)</b> from section 4a's position table. Each row \
         is one body's \\((x, y, \\theta)\\) triple in SI units (m, rad).</li>\n",
    );
    html.push_str(
        "<li><b>Build \\(\\Phi(q, t=0)\\)</b> manually by writing out one row \
         per joint per the equations in section 2. For each revolute joint, \
         use the body anchor coordinates from the Loop Equations table's \
         constraint metadata. Driver rows use \\(f(0) = \\theta_0\\) (or \
         \\(L(0)\\) for a linear driver).</li>\n",
    );
    html.push_str(
        "<li><b>Verify \\(\\|\\Phi\\| \\approx 0\\)</b> — compare against the \
         numerical value in the \"Position residual\" line above.</li>\n",
    );
    html.push_str(
        "<li><b>Build \\(\\Phi_q\\)</b> from the partial-derivative formulas \
         in section 3. The result should match the matrix table above \
         entry-for-entry.</li>\n",
    );
    html.push_str(
        "<li><b>Solve \\(\\Phi_q\\,\\dot q = -\\Phi_t\\)</b> for \\(\\dot q\\). \
         The driver row of \\(\\Phi_t\\) is \\(-\\dot f(t) = -\\omega\\) \
         (revolute) or \\(-\\dot L(t)\\) (linear); all other rows are zero. \
         Compare \\(\\dot q\\) against section 4a's velocity table.</li>\n",
    );
    html.push_str(
        "<li><b>Solve \\(\\Phi_q\\,\\ddot q = \\gamma\\)</b> for \\(\\ddot q\\). \
         The \\(\\gamma\\) RHS is the centripetal term \
         \\(-B(\\theta)\\,s\\,\\dot\\theta^2\\) per joint plus \
         \\(-\\ddot f(t)\\) on the driver row; for constant-speed driver, \
         \\(\\ddot f = 0\\). Compare against section 4a's acceleration table.</li>\n",
    );
    html.push_str(
        "<li><b>Solve \\(\\Phi_q^T\\,\\lambda = -Q_{\\text{applied}}\\)</b> for \
         the Lagrange multipliers, where \\(Q_{\\text{applied}}\\) is the \
         stack of generalized forces (gravity contributes \\(m_i\\,g\\) into \
         each body's y-equation; force elements add their own contributions). \
         Compare against the \\(\\lambda\\) column in the Loop Equations \
         table.</li>\n",
    );
    html.push_str("</ol>\n");

    // ── 9. References ──────────────────────────────────────────────────────
    html.push_str("<h3>9. Further reading</h3>\n");
    html.push_str(
        "<p>The math used here is standard multibody-dynamics constraint \
         theory. Authoritative references:</p>\n",
    );
    html.push_str("<ul>\n");
    html.push_str(
        "<li>Haug, E.J., <em>Computer Aided Kinematics and Dynamics of \
         Mechanical Systems, Vol. I: Basic Methods</em>, Allyn &amp; Bacon, 1989. \
         Chapters 4\u{2013}6 cover the constraint Jacobian / velocity / \
         acceleration cascade exactly as implemented here.</li>\n",
    );
    html.push_str(
        "<li>Shabana, A.A., <em>Computational Dynamics</em>, 3rd ed., Wiley, \
         2010. Chapter 3 derives revolute / prismatic / driver constraints \
         in the same notation.</li>\n",
    );
    html.push_str(
        "<li>Nikravesh, P.E., <em>Computer-Aided Analysis of Mechanical \
         Systems</em>, Prentice-Hall, 1988. Standard reference for the \
         Lagrange multiplier interpretation.</li>\n",
    );
    html.push_str(
        "<li>Project's own equations reference: \
         <code>docs/superpowers/specs/2026-04-29-linkage-equations-reference.md</code> \
         \u{2014} this report's source-of-truth derivation, including the \
         inverse-kinematics extension for trajectory mode (\u{00A7}8).</li>\n",
    );
    html.push_str("</ul>\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_html_report_contains_expected_sections() {
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::gui::state::DisplayUnits;

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q = if result.converged { result.q } else { q0 };

        let (sweep, _) = crate::gui::sweep::compute_sweep_data(
            &mech, &q, 2.0 * std::f64::consts::PI, 0.0, 9.81, None,
        );

        let units = DisplayUnits::default();

        let html = generate_html_report(&mech, &q, &sweep, None, &units)
            .expect("report generation should succeed");

        assert!(html.contains("Linkage Mechanism Report"), "should have title");
        assert!(html.contains("Mechanism Diagram"), "should have diagram section");
        assert!(html.contains("<svg"), "should have embedded SVG");
        assert!(html.contains("Dimensions"), "should have dimensions table");
        assert!(html.contains("Mass Properties"), "should have mass properties table");
        assert!(html.contains("kg"), "should have mass units");
        // Loop equations section (E4)
        assert!(html.contains("Loop Equations"), "should have loop equations section");
        assert!(html.contains("Φ_rev"), "should render at least one revolute symbol");
        assert!(html.contains("Φ_rd"), "FourBar sample has a revolute driver");
        // Mathematical derivation section
        assert!(
            html.contains("Mathematical Derivation"),
            "should include the math walkthrough section"
        );
        // KaTeX is loaded via CDN; auto-render scans for \[..\] / \(..\) delimiters.
        assert!(
            html.contains("katex.min.css"),
            "should load KaTeX stylesheet for math rendering"
        );
        assert!(
            html.contains("renderMathInElement"),
            "should call KaTeX auto-render on body load"
        );
        // LaTeX delimiters and content (escaped \\ in source emits \ in HTML).
        assert!(
            html.contains("\\Phi_q") || html.contains("\\(\\Phi_q\\)"),
            "math section should reference the constraint Jacobian in LaTeX"
        );
        assert!(
            html.contains("\\Delta q = -\\Phi"),
            "math section should derive the Newton position solve in LaTeX"
        );
        assert!(
            html.contains("Lagrange multipliers"),
            "math section should explain λ as Lagrange multipliers"
        );
        assert!(
            html.contains("inverse position control"),
            "math section should mention trajectory mode (inverse)"
        );
        // Mechanism-specific parameterisation
        assert!(
            html.contains("3 moving bodies"),
            "math section should reference mechanism's actual body count"
        );
        // Numeric solve outputs (4a)
        assert!(
            html.contains("Numeric solve outputs at this pose"),
            "math section should include numeric q / q-dot / q-ddot tables"
        );
        assert!(
            html.contains("\\dot q") && html.contains("\\ddot q"),
            "math section should reference symbolic q-dot and q-ddot"
        );
        // Linear driver / actuator coverage (math section).
        assert!(
            html.contains("Linear driver"),
            "math section should include linear driver math"
        );
        assert!(
            html.contains("\\Phi_{\\mathrm{ld}}"),
            "math section should reference Φ_ld in LaTeX"
        );
        assert!(
            html.contains("LinearActuator force element"),
            "math section should distinguish LinearActuator force element from LinearDriver constraint"
        );
        // Worked-example detail (sections 7-9) for external math validation.
        assert!(
            html.contains("Numerical validation at this pose"),
            "should include numerical-validation section"
        );
        assert!(
            html.contains("Position residual") && html.contains("Velocity residual"),
            "should report residuals for each solve level"
        );
        assert!(
            html.contains("How to verify the math externally"),
            "should include the verification recipe"
        );
        assert!(
            html.contains("Further reading") && html.contains("Haug"),
            "should include the references list"
        );
        // The user-reported bug: body name with `_` (e.g. "link_1") would
        // leave \(... \text{link_1} ...\) un-rendered. Ensure the escape
        // landed; for the canonical 4-bar fixture (body names crank /
        // coupler / rocker — no underscores), the q-vector should still
        // contain `\text{crank}` etc., proving the math template emitted
        // correctly.
        assert!(
            html.contains("\\text{crank}"),
            "q-vector should use \\text{{}} for body names"
        );
        // Closed-form 4-bar engineering equations (sections 4b-4f).
        assert!(
            html.contains("Freudenstein"),
            "should include Freudenstein's equation"
        );
        assert!(
            html.contains("K_1\\,\\cos\\theta_4"),
            "Freudenstein equation should appear in LaTeX"
        );
        assert!(
            html.contains("Grashof"),
            "should include Grashof's condition"
        );
        assert!(
            html.contains("Class I") || html.contains("Class II") || html.contains("Class III"),
            "Grashof section should classify the 4-bar"
        );
        assert!(
            html.contains("Transmission angle"),
            "should include transmission angle math"
        );
        assert!(
            html.contains("\\cos\\mu"),
            "transmission angle section should derive cos μ"
        );
        assert!(
            html.contains("Velocity ratio") || html.contains("velocity ratio"),
            "should include velocity ratio / mechanical advantage"
        );
        assert!(
            html.contains("dead point") || html.contains("Dead point") || html.contains("Singular configurations"),
            "should include dead-point / singular configuration analysis"
        );
        // Plotly integration
        assert!(html.contains("plotly-2.35.2.min.js"), "should include plotly CDN");
        assert!(html.contains("Plotly.newPlot"), "should have at least one plotly chart");
        assert!(html.contains("energy_plot"), "should have energy plot div");
    }

    #[test]
    fn float_vec_to_json_handles_nan_and_infinity() {
        let vals = vec![1.0, f64::NAN, 2.5, f64::INFINITY, f64::NEG_INFINITY, 3.0];
        let json = float_vec_to_json(&vals);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        // NaN and infinities should become null
        assert!(json.contains("null"));
        // Finite values should be present
        assert!(json.contains("1.000000"));
        assert!(json.contains("2.500000"));
        assert!(json.contains("3.000000"));
        // Count nulls: 3 (NaN, +Inf, -Inf)
        assert_eq!(json.matches("null").count(), 3);
    }

    #[test]
    fn float_vec_to_json_empty() {
        let json = float_vec_to_json(&[]);
        assert_eq!(json, "[]");
    }

    #[test]
    fn float_vec_to_json_single_value() {
        let json = float_vec_to_json(&[42.0]);
        assert_eq!(json, "[42.000000]");
    }
}
