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
    sensor_config: &crate::gui::state::SensorConfig,
) -> Result<String, String> {
    use crate::analysis::envelopes::compute_envelope;
    use crate::core::state::GROUND_ID;

    // Prefer the labeled schematic (joint IDs, body labels, ground hatching,
    // constraint legend) for a report. Fall back to the regular SVG if the
    // schematic generator fails (empty mechanism etc).
    let svg = generate_schematic_svg(mechanism, q, sensor_config)
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
    write_math_background_section(&mut html, mechanism, q, sensor_config);

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

/// Get current timestamp formatted in US Eastern time (`EST` in winter,
/// `EDT` in summer). Native uses `std::time::SystemTime`; wasm32 uses
/// `js_sys::Date::now()` because `SystemTime::now()` panics with
/// "time not implemented on this platform" on `wasm32-unknown-unknown`.
#[cfg(not(target_arch = "wasm32"))]
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    format_unix_timestamp_eastern(now)
}

#[cfg(target_arch = "wasm32")]
fn chrono_now() -> String {
    let now = (js_sys::Date::now() / 1000.0) as i64;
    format_unix_timestamp_eastern(now)
}

/// Format a Unix timestamp as US-Eastern time (`EST` in winter, `EDT` in
/// summer). Uses a proper Gregorian calendar conversion (Howard Hinnant's
/// `days_from_civil` algorithm, valid for all dates after 1582) plus a
/// computed DST flag for US Eastern zone — accurate to the second.
///
/// The earlier implementation approximated months as 30 days and ignored
/// leap years; cumulative error reached ~14 days by 2026. Don't go back.
fn format_unix_timestamp_eastern(utc_secs: i64) -> String {
    // First pass: compute UTC (year, month, day, hour) so we can decide
    // whether DST is in effect, then re-shift to local time.
    let utc_days = utc_secs.div_euclid(86_400);
    let utc_secs_of_day = utc_secs.rem_euclid(86_400);
    let (utc_year, utc_month, utc_day) = days_to_ymd(utc_days);
    let utc_hour = (utc_secs_of_day / 3600) as u32;

    // EST = UTC-5, EDT = UTC-4. DST is in effect from 2 AM local time on
    // the 2nd Sunday of March through 2 AM local time on the 1st Sunday
    // of November. We approximate "2 AM local" using the UTC hour, which
    // is correct except in the ~2-hour transition window — acceptable
    // for a report timestamp.
    let dst = is_us_eastern_dst(utc_year, utc_month, utc_day, utc_hour);
    let offset_hours: i64 = if dst { -4 } else { -5 };
    let local_secs = utc_secs + offset_hours * 3600;

    let local_days = local_secs.div_euclid(86_400);
    let local_secs_of_day = local_secs.rem_euclid(86_400);
    let (year, month, day) = days_to_ymd(local_days);
    let h = (local_secs_of_day / 3600) as u32;
    let m = ((local_secs_of_day / 60) % 60) as u32;
    let s = (local_secs_of_day % 60) as u32;
    let zone = if dst { "EDT" } else { "EST" };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} {}",
        year, month, day, h, m, s, zone,
    )
}

/// Convert days since 1970-01-01 (Unix epoch) to (year, month, day).
///
/// Howard Hinnant's `civil_from_days`, hot-loop-safe, valid for all
/// proleptic Gregorian dates. See <https://howardhinnant.github.io/date_algorithms.html>.
/// Months are 1-indexed (1 = January … 12 = December); days are 1-indexed.
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Shift epoch so day 0 is 0000-03-01 (treats March as the start of
    // year, putting the leap day at the end — simplifies the algorithm).
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146_096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]; March-relative month index
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    let y_civil = if m <= 2 { y + 1 } else { y };
    (y_civil as i32, m, d)
}

/// Returns true iff the given UTC instant falls within the US-Eastern
/// daylight-saving period (2nd Sunday of March → 1st Sunday of November,
/// each transitioning at 2 AM local time = 7 AM UTC during the spring
/// transition / 6 AM UTC during the fall transition). Approximated as
/// "transitions at 7 AM UTC" since the report only displays time to
/// the second; the 1-hour fudge in the transition window is invisible.
fn is_us_eastern_dst(year: i32, month: u32, day: u32, hour: u32) -> bool {
    // Outside Mar–Nov: never DST. Inside Apr–Oct: always DST.
    if !(3..=11).contains(&month) {
        return false;
    }
    if (4..=10).contains(&month) {
        return true;
    }
    if month == 3 {
        // Spring forward: 2nd Sunday of March, 2 AM local (≈ 7 AM UTC).
        let second_sun = nth_weekday_of_month(year, 3, 7 /* Sunday */, 2);
        if day < second_sun {
            return false;
        }
        if day > second_sun {
            return true;
        }
        return hour >= 7;
    }
    // November: fall back on the 1st Sunday at 2 AM local (≈ 6 AM UTC).
    let first_sun = nth_weekday_of_month(year, 11, 7 /* Sunday */, 1);
    if day < first_sun {
        return true;
    }
    if day > first_sun {
        return false;
    }
    hour < 6
}

/// Return the day-of-month (1-indexed) of the `nth` `weekday` (1=Mon ..
/// 7=Sun) in `month` of `year`. Used for US DST transition computation
/// (2nd Sunday of March, 1st Sunday of November).
fn nth_weekday_of_month(year: i32, month: u32, weekday: u32, nth: u32) -> u32 {
    // Days since epoch for the 1st of `month`.
    let first_days = ymd_to_days(year, month, 1);
    // Day-of-week for the 1st: Hinnant's algorithm has Sunday=0..Saturday=6
    // for `((days + 4) mod 7)` since 1970-01-01 was a Thursday.
    let dow_sun_zero = ((first_days + 4).rem_euclid(7)) as u32; // 0=Sun..6=Sat
    let target_sun_zero = if weekday == 7 { 0 } else { weekday };
    let offset = (target_sun_zero + 7 - dow_sun_zero) % 7;
    1 + offset + 7 * (nth - 1)
}

/// Inverse of `days_to_ymd`: convert a (year, month, day) civil date to
/// days since 1970-01-01. Howard Hinnant's `days_from_civil`.
fn ymd_to_days(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64; // [0, 399]
    let m = m as u64;
    let d = d as u64;
    let mp = if m > 2 { m - 3 } else { m + 9 }; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146_096]
    era * 146_097 + doe as i64 - 719_468
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
    write_acceleration_analysis_section(html, &layout, theta_2_rel, theta_4_rel);
    write_forward_kinematics_section(html, &layout);
    write_actuator_position_section(html, mechanism, q, &layout);
    write_coordinate_frames_section(html, mechanism);
    write_embedded_control_recipe_section(html, &layout);
}

/// §4g — closed-form angular acceleration relation by differentiating
/// Freudenstein's equation twice. Lets a software engineer compute the
/// rocker (and coupler) angular acceleration from the crank's measured
/// kinematics without going near the constraint Jacobian.
fn write_acceleration_analysis_section(
    html: &mut String,
    layout: &FourbarLayout,
    theta_2: f64,
    theta_4: f64,
) {
    let a = layout.a;
    let c = layout.c;
    let d = layout.d;
    let k1 = d / a;
    let k2 = d / c;
    let s2 = theta_2.sin();
    let c2 = theta_2.cos();
    let s4 = theta_4.sin();
    let c4 = theta_4.cos();
    let sd = (theta_2 - theta_4).sin();
    let cd = (theta_2 - theta_4).cos();

    html.push_str("<h3>4g. Acceleration analysis (closed-form)</h3>\n");
    html.push_str(
        "<p>Differentiating Freudenstein's equation a second time w.r.t. \
         time yields the angular acceleration relation \\(\\ddot\\theta_4\\) \
         in terms of \\(\\theta_2\\), \\(\\theta_4\\), \\(\\dot\\theta_2\\), \
         \\(\\dot\\theta_4\\), and \\(\\ddot\\theta_2\\). Starting from the \
         first-time-derivative form:</p>\n",
    );
    html.push_str(
        "\\[ K_1 \\sin\\theta_4 \\,\\dot\\theta_4 - K_2 \\sin\\theta_2 \\,\\dot\\theta_2 = \\sin(\\theta_2 - \\theta_4) \\,(\\dot\\theta_2 - \\dot\\theta_4) \\]\n",
    );
    html.push_str(
        "<p>differentiating again and grouping by \\(\\ddot\\theta_4\\):</p>\n",
    );
    html.push_str(
        "\\[ \\boxed{ \\;\\ddot\\theta_4 \\;=\\; \\frac{ \
         (K_2 \\sin\\theta_2 + \\sin(\\theta_2 - \\theta_4))\\,\\ddot\\theta_2 \
         + K_2 \\cos\\theta_2\\,\\dot\\theta_2^2 \
         - K_1 \\cos\\theta_4\\,\\dot\\theta_4^2 \
         + \\cos(\\theta_2 - \\theta_4)\\,(\\dot\\theta_2 - \\dot\\theta_4)^2 \
         }{ K_1 \\sin\\theta_4 + \\sin(\\theta_2 - \\theta_4) } \\;} \\]\n",
    );
    html.push_str(
        "<p>The denominator vanishes at dead points (§4f) — geometrically \
         the same condition as <i>infinite mechanical advantage</i> in §4e. \
         Don't try to evaluate this near a toggle position; use the \
         constraint cascade in §4 instead, which handles the singularity \
         via SVD.</p>\n",
    );
    html.push_str(
        "<p>Coupler angular acceleration \\(\\ddot\\theta_3\\) follows from \
         differentiating the loop closure twice. Splitting into x and y \
         components:</p>\n",
    );
    html.push_str(
        "\\[ \\ddot\\theta_3 = \\frac{a\\,\\ddot\\theta_2 \\sin(\\theta_2 - \\theta_4) - c\\,\\ddot\\theta_4 \\sin(\\theta_3 - \\theta_4) + \\dots}{b\\,\\sin(\\theta_3 - \\theta_4)} \\]\n",
    );
    html.push_str(
        "<p>(velocity-squared terms omitted from the display; full form in \
         the source spec). For most control applications you only need \
         \\(\\ddot\\theta_4\\) — the rocker drives the load, and its \
         acceleration determines required actuator power.</p>\n",
    );

    // Numerical evaluation at the report's pose.
    let denom = k1 * s4 + sd;
    if denom.abs() > 1e-9 {
        // Pretend ω2 = 1 rad/s, α2 = 0 for a "unit input" demonstration.
        let omega2 = 1.0_f64;
        let alpha2 = 0.0_f64;
        // ω4 from velocity ratio.
        let v_num = -k2 * s2 + sd;
        let v_den = -k1 * s4 + sd;
        let omega4 = if v_den.abs() > 1e-9 {
            (v_num / v_den) * omega2
        } else {
            f64::NAN
        };
        let alpha4 = ((k2 * s2 + sd) * alpha2
            + k2 * c2 * omega2 * omega2
            - k1 * c4 * omega4 * omega4
            + cd * (omega2 - omega4).powi(2))
            / denom;
        html.push_str("<p><b>Worked example at this pose</b> with unit input rate:</p>\n");
        html.push_str(&format!(
            "\\[ \\dot\\theta_2 = 1\\,\\text{{rad/s}},\\quad \\ddot\\theta_2 = 0 \\;\\Rightarrow\\; \\dot\\theta_4 = {:.6}\\,\\text{{rad/s}},\\;\\; \\ddot\\theta_4 = {:.6}\\,\\text{{rad/s}}^2 \\]\n",
            omega4, alpha4,
        ));
        html.push_str(
            "<p>i.e. with the crank at a constant 1 rad/s, the rocker still \
             accelerates because the velocity ratio itself depends on pose. \
             The cross-term \\(\\cos(\\theta_2-\\theta_4)\\,(\\dot\\theta_2 - \\dot\\theta_4)^2\\) \
             is the centripetal contribution from the loop's instantaneous \
             curvature.</p>\n",
        );
    }
}

/// §4h — closed-form forward kinematics. Given θ₂, return θ₃ and θ₄ in
/// closed form (tangent half-angle for θ₄, atan2 for θ₃). This is THE
/// equation a software engineer needs when only the crank encoder
/// position is available — every other body angle follows by direct
/// computation, no iterative solve required.
fn write_forward_kinematics_section(html: &mut String, layout: &FourbarLayout) {
    let a = layout.a;
    let b = layout.b;
    let c = layout.c;
    let d = layout.d;
    let k1 = d / a;
    let k2 = d / c;
    let k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c);

    html.push_str("<h3>4h. Closed-form forward kinematics (\\(\\theta_2 \\to \\theta_3, \\theta_4\\))</h3>\n");
    html.push_str(
        "<p>Given the crank angle \\(\\theta_2\\) (e.g. from an encoder), \
         the coupler angle \\(\\theta_3\\) and rocker angle \\(\\theta_4\\) \
         can be computed in closed form without any iterative solver. \
         This is the equation a software engineer needs to embed in firmware.</p>\n",
    );
    html.push_str(
        "<p><b>Step 1.</b> Substitute Freudenstein into the form \\(A \\cos\\theta_4 + B \\sin\\theta_4 = C\\):</p>\n",
    );
    html.push_str(
        "\\[ A = K_1 - \\cos\\theta_2,\\qquad B = -\\sin\\theta_2,\\qquad C = K_2 \\cos\\theta_2 - K_3 \\]\n",
    );
    html.push_str(
        "<p><b>Step 2.</b> Tangent half-angle substitution \\(t = \\tan(\\theta_4/2)\\) reduces this to a quadratic:</p>\n",
    );
    html.push_str(
        "\\[ (A + C)\\,t^2 + 2B\\,t - (A - C) = 0 \\;\\Rightarrow\\; t = \\frac{-B \\pm \\sqrt{B^2 + A^2 - C^2}}{A + C} \\]\n",
    );
    html.push_str(
        "\\[ \\theta_4 = 2 \\arctan(t) \\]\n",
    );
    html.push_str(
        "<p>The \\(\\pm\\) gives the two assembly modes (\"open\" and \
         \"crossed\" / \"+\" and \"−\" branches). <b>Pick one branch at \
         calibration and stick with it</b> — switching mid-cycle requires \
         passing through a dead point. In firmware: pick the sign that \
         matches the rocker's known initial direction at startup.</p>\n",
    );
    html.push_str(
        "<p><b>Step 3.</b> Coupler angle from the loop's real and imaginary \
         components:</p>\n",
    );
    html.push_str(
        "\\[ \\theta_3 = \\operatorname{atan2}(c \\sin\\theta_4 - a \\sin\\theta_2,\\; d + c \\cos\\theta_4 - a \\cos\\theta_2) \\]\n",
    );
    html.push_str(&format!(
        "<p><b>For this mechanism</b>: \
         \\(K_1 = {:.4}\\), \\(K_2 = {:.4}\\), \\(K_3 = {:.4}\\). \
         All three constants are known at compile time once the link lengths \
         are fixed; firmware only needs the encoder reading \\(\\theta_2\\) \
         and four trig-function evaluations per loop iteration.</p>\n",
        k1, k2, k3,
    ));
}

/// §4i — linear-actuator length L as a function of crank angle θ₂. THE
/// control-loop relation when the linear actuator is the position sensor
/// (or actuator) and the crank is the input. Detects the actuator
/// endpoints from the LinearActuator force element or LinearDriver
/// constraint and computes the closed-form length using the §4h forward
/// kinematics.
fn write_actuator_position_section(
    html: &mut String,
    mechanism: &crate::core::mechanism::Mechanism,
    q: &nalgebra::DVector<f64>,
    _layout: &FourbarLayout,
) {
    use crate::core::constraint::Constraint;
    use crate::forces::elements::ForceElement;

    // Detect actuator endpoints (force-element or constraint).
    enum ActuatorKind {
        Force {
            body_a: String,
            point_a: nalgebra::Vector2<f64>,
            body_b: String,
            point_b: nalgebra::Vector2<f64>,
        },
        Driver {
            body_i: String,
            point_i: nalgebra::Vector2<f64>,
            body_j: String,
            point_j: nalgebra::Vector2<f64>,
        },
    }
    let actuator: Option<ActuatorKind> = mechanism
        .forces()
        .iter()
        .find_map(|fe| {
            if let ForceElement::LinearActuator(act) = fe {
                Some(ActuatorKind::Force {
                    body_a: act.body_a.clone(),
                    point_a: nalgebra::Vector2::new(act.point_a[0], act.point_a[1]),
                    body_b: act.body_b.clone(),
                    point_b: nalgebra::Vector2::new(act.point_b[0], act.point_b[1]),
                })
            } else {
                None
            }
        })
        .or_else(|| {
            mechanism.linear_drivers().first().map(|drv| {
                let pa = drv.point_a();
                let pb = drv.point_b();
                ActuatorKind::Driver {
                    body_i: drv.body_i_id().to_string(),
                    point_i: nalgebra::Vector2::new(pa[0], pa[1]),
                    body_j: drv.body_j_id().to_string(),
                    point_j: nalgebra::Vector2::new(pb[0], pb[1]),
                }
            })
        });

    html.push_str("<h3>4i. Linear actuator length as a function of crank angle</h3>\n");

    let Some(act) = actuator else {
        html.push_str(
            "<p><em>No linear actuator (LinearActuator force element or \
             LinearDriver constraint) detected in this mechanism. If you \
             intend to control a 4-bar via linear actuator position \
             feedback, add the actuator to the mechanism — the simulator \
             will then derive its length-vs-crank-angle equation here \
             automatically.</em></p>\n",
        );
        return;
    };

    let (body_a, point_a, body_b, point_b, label) = match act {
        ActuatorKind::Force {
            body_a,
            point_a,
            body_b,
            point_b,
        } => (body_a, point_a, body_b, point_b, "LinearActuator force element"),
        ActuatorKind::Driver {
            body_i,
            point_i,
            body_j,
            point_j,
        } => (body_i, point_i, body_j, point_j, "LinearDriver constraint"),
    };

    let pa_world = mechanism.state().body_point_global(&body_a, &point_a, q);
    let pb_world = mechanism.state().body_point_global(&body_b, &point_b, q);
    let l_now = (pb_world - pa_world).norm();

    html.push_str(&format!(
        "<p>Detected: <b>{}</b> between <code>{}</code> at local point \
         \\(({:.4}, {:.4})\\) and <code>{}</code> at local point \
         \\(({:.4}, {:.4})\\).</p>\n",
        label,
        html_escape(&body_a),
        point_a.x,
        point_a.y,
        html_escape(&body_b),
        point_b.x,
        point_b.y,
    ));
    html.push_str(
        "<p>The actuator length \\(L\\) is the world-frame distance \
         between its two attachment points:</p>\n",
    );
    html.push_str(
        "\\[ L(\\theta_2) = \\| P_b(\\theta_2) - P_a(\\theta_2) \\| \\]\n",
    );
    html.push_str(
        "<p>where each endpoint maps to world coordinates via</p>\n",
    );
    html.push_str(
        "\\[ P_x(\\theta_2) = r_x(\\theta_2) + A(\\theta_x(\\theta_2))\\,s_x \\]\n",
    );
    html.push_str(
        "<p>The body angles \\(\\theta_x(\\theta_2)\\) come from §4h, and \
         body CG positions \\(r_x\\) follow from rigid-body kinematics \
         (each body's position is determined by its angle plus one \
         attachment point's known world coords). End-to-end, given \
         \\(\\theta_2\\) the actuator length is computable in <b>O(constant) \
         operations</b>, no iteration required.</p>\n",
    );
    html.push_str(&format!(
        "<p><b>At the report's pose</b>: \\(P_a = ({:.4}, {:.4})\\) m, \
         \\(P_b = ({:.4}, {:.4})\\) m, so \
         \\(L = {:.4}\\) m = \\({:.2}\\) mm.</p>\n",
        pa_world.x, pa_world.y, pb_world.x, pb_world.y, l_now, l_now * 1000.0,
    ));
    html.push_str(
        "<p><b>For inverse control</b> (given a desired \\(L\\), find \
         \\(\\theta_2\\)): no closed form in general. Either pre-compute a \
         lookup table of \\(L(\\theta_2)\\) over the working range and \
         interpolate, or run a 1-D Newton iteration on \\(f(\\theta_2) = \
         L(\\theta_2) - L_{\\text{desired}}\\). The latter converges in \
         2-4 iterations in practice because \\(dL/d\\theta_2\\) is smooth \
         away from dead points.</p>\n",
    );
    html.push_str(
        "<p><b>Sensitivity</b> \\(dL/d\\theta_2\\) at this pose: \
         differentiate the inner product expression. Useful for control-\
         loop gain scheduling — when \\(dL/d\\theta_2\\) is small the \
         actuator has high mechanical advantage but also high position \
         resolution requirement.</p>\n",
    );
}

/// §4j — coordinate-frame conventions. Spelled out explicitly because
/// transcribing this math to firmware is the #1 place sign errors creep in.
fn write_coordinate_frames_section(
    html: &mut String,
    mechanism: &crate::core::mechanism::Mechanism,
) {
    html.push_str("<h3>4j. Coordinate frame conventions</h3>\n");
    html.push_str(
        "<p>The math above implicitly uses three frames. Embedded firmware \
         needs all three explicit so sign and offset errors don't creep in \
         during transcription.</p>\n",
    );
    html.push_str(
        "<table><tr><th>Frame</th><th>Origin</th><th>Axes</th><th>Use</th></tr>\n",
    );
    html.push_str(
        "<tr><td><b>World (inertial)</b></td><td>Arbitrary fixed point</td>\
         <td>+x rightward, +y upward, right-handed</td>\
         <td>Where gravity acts: \\(\\vec g = -g\\,\\hat y\\) when \
         mounting angle is zero.</td></tr>\n",
    );
    html.push_str(
        "<tr><td><b>Mounting</b></td><td>Same as world</td>\
         <td>Rotated by mounting angle \\(\\phi\\) from world</td>\
         <td>The mechanism is built in this frame; gravity in the mounting \
         frame is \\(\\vec g = -g(\\sin\\phi\\,\\hat x_m + \\cos\\phi\\,\\hat y_m)\\).</td></tr>\n",
    );
    html.push_str(
        "<tr><td><b>Body</b></td><td>Body's CG</td>\
         <td>Rotated by \\(\\theta_i\\) from mounting</td>\
         <td>Body-local attachment points are stored in this frame; \
         transform to mounting via \\(P_{\\mathrm{mount}} = r_i + A(\\theta_i)\\,s_{\\mathrm{body}}\\).</td></tr>\n",
    );
    html.push_str("</table>\n");

    // mounting_angle lives on AppState (UI/persistence concern), not on
    // the Mechanism core. The user configures it in the input panel; the
    // recipe below is generic w.r.t. the value.
    let _ = mechanism;
    html.push_str(
        "<p><b>If your mechanism is mounted at an angle</b> \\(\\phi\\) \
         relative to gravity (configured in the input panel as \"Mounting \
         angle\"): the loop equations and Freudenstein's equation are \
         unchanged because they're geometric. What <em>does</em> change is \
         the gravity vector \\(\\vec g\\) in the mounting frame: \
         \\(\\vec g_{\\mathrm{mount}} = -g(\\sin\\phi\\,\\hat x_m + \\cos\\phi\\,\\hat y_m)\\). \
         Only force / torque calculations (§5) depend on \\(\\phi\\) — the \
         kinematics in §4b–§4i are mounting-angle-invariant.</p>\n",
    );
    html.push_str(
        "<p><b>Sign convention</b> for \\(\\theta_i\\): counter-clockwise \
         positive (right-hand rule about +z). The encoder on a real crank \
         likely follows the opposite sign if mounted with the +z axis \
         pointing into the mechanism — verify your encoder's polarity \
         during calibration by manually rotating the crank to a known \
         angle and reading the sensor value.</p>\n",
    );
}

/// §4k — practical embedded-control recipe. Pseudocode and a checklist
/// of what you need on the firmware side.
fn write_embedded_control_recipe_section(html: &mut String, layout: &FourbarLayout) {
    let a = layout.a;
    let b = layout.b;
    let c = layout.c;
    let d = layout.d;
    let k1 = d / a;
    let k2 = d / c;
    let k3 = (a * a - b * b + c * c + d * d) / (2.0 * a * c);

    html.push_str("<h3>4k. Embedded control recipe</h3>\n");
    html.push_str(
        "<p>To control this mechanism in real time given the available \
         sensors (crank encoder, linear actuator position, link lengths, \
         mounting angle), here's the practical loop:</p>\n",
    );
    html.push_str("<ol>\n");
    html.push_str(
        "<li><b>Calibrate</b>: at startup, drive the crank to a known \
         physical reference (hard stop or limit switch). Read encoder \
         counts at that pose and store as \\(\\theta_2^{\\text{zero}}\\). \
         Repeat for the actuator if it has its own absolute reference, \
         otherwise estimate \\(L^{\\text{zero}}\\) by computing it from \
         the calibrated \\(\\theta_2\\) using §4i.</li>\n",
    );
    html.push_str(
        "<li><b>Per control cycle</b> (typically 1-10 kHz):</li>\n",
    );
    html.push_str("</ol>\n");
    html.push_str(&format!(
        "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
         font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
         overflow-x: auto;'>\
// Pre-computed at compile time from link lengths.
const float K1 = {:.6};   // d / a
const float K2 = {:.6};   // d / c
const float K3 = {:.6};   // (a^2 - b^2 + c^2 + d^2) / (2 a c)
const float A_CRANK = {:.4}f;     // crank length [m]
const float B_COUPLER = {:.4}f;   // coupler length [m]
const float C_ROCKER = {:.4}f;    // rocker length [m]
const float D_GROUND = {:.4}f;    // ground link length [m]
const int BRANCH = +1;            // assembly mode chosen at calibration

// Per-cycle: read sensors, compute predicted state, close the loop.
void control_cycle() {{
    // 1. Read sensors.
    float theta2 = read_encoder() - theta2_zero;     // crank angle [rad]
    float L_meas = read_actuator_pos() - L_zero;     // actuator length [m]

    // 2. Forward kinematics: theta4 from Freudenstein (§4h).
    float A_ = K1 - cosf(theta2);
    float B_ = -sinf(theta2);
    float C_ = K2 * cosf(theta2) - K3;
    float disc = B_*B_ + A_*A_ - C_*C_;
    if (disc &lt; 0) {{ /* unreachable pose, abort */ }}
    float t = (-B_ + BRANCH * sqrtf(disc)) / (A_ + C_);
    float theta4 = 2.f * atan2f(t, 1.f);
    float theta3 = atan2f(C_ROCKER * sinf(theta4) - A_CRANK * sinf(theta2),
                          D_GROUND + C_ROCKER * cosf(theta4)
                                   - A_CRANK * cosf(theta2));

    // 3. Predict actuator length from theta2 (§4i).
    float L_pred = compute_actuator_length(theta2, theta3, theta4);

    // 4. Cross-check sensor consistency.
    float L_err = L_meas - L_pred;
    if (fabsf(L_err) &gt; L_TOL) {{ /* sensor disagreement → fault */ }}

    // 5. Closed-loop output.
    float theta2_target = inverse_actuator_to_crank(L_target);  // §4i
    float u = pid_compute(theta2_target, theta2);
    write_motor_command(u);
}}
</pre>\n",
        k1, k2, k3, a, b, c, d,
    ));
    html.push_str(
        "<p>This loop uses the crank encoder as the primary sensor and the \
         actuator position as a redundant check. If the actuator IS the \
         input (you command its position directly and the crank rotates \
         passively), swap roles: invert §4i to get \\(\\theta_2\\) from \
         \\(L\\), then proceed.</p>\n",
    );
    html.push_str(
        "<p><b>What this loop assumes you've validated offline</b>:</p>\n",
    );
    html.push_str("<ul>\n");
    html.push_str(
        "<li>Mechanism stays in <b>one</b> assembly mode (fixed BRANCH sign). \
         If your application drives through a dead point, the BRANCH flips \
         and the formula above gives a wrong answer at the singular pose. \
         For safety: detect dead-point proximity (transmission angle in §4d) \
         and switch to a higher-order estimator near the singularity.</li>\n",
    );
    html.push_str(
        "<li>Link lengths are <b>accurately measured</b> (machinist's \
         tolerance, not nominal CAD). Errors of 1% in link length produce \
         degree-level errors in predicted angles. Calibrate by running \
         the mechanism through several known poses and least-squares \
         fitting the lengths.</li>\n",
    );
    html.push_str(
        "<li>Mounting angle is <b>characterized</b>. If you mount the \
         mechanism on a tilted surface, gravity rotates relative to the \
         linkage; the statics (and hence required motor torque) depends \
         on this. The mechanism's geometric kinematics (§4h, §4i) is \
         unaffected by mounting angle, but force / torque calculations \
         (§5) depend on it directly.</li>\n",
    );
    html.push_str(
        "<li><b>Backlash</b> in joints and actuator are below your control \
         resolution. The closed-form math assumes ideal pin joints; real \
         joints have play that shows up as hysteresis in \\(L(\\theta_2)\\) \
         when reversing direction.</li>\n",
    );
    html.push_str("</ul>\n");
}

/// Append "§4l — State estimation / sensor fusion" subsection.
///
/// Branches on the user's sensor configuration:
/// - 0 sensors → open-loop note (state predicted from input torque +
///   dynamics model only; no sensor fusion possible)
/// - 1 sensor → single-sensor estimator (low-pass + numerical
///   differentiation; no fusion)
/// - 2 sensors → full Extended Kalman Filter (EKF) setup with
///   measurement Jacobian H derived from §4i sensitivity.
///
/// In all cases, the section is parameterised to the actual sensor
/// noise σ values configured in the GUI so the emitted Q, R covariance
/// matrices are ready to copy into firmware.
fn write_state_estimation_section(
    html: &mut String,
    mechanism: &crate::core::mechanism::Mechanism,
    _q: &nalgebra::DVector<f64>,
    sensor_config: &crate::gui::state::SensorConfig,
) {
    use crate::core::constraint::Constraint;

    let n_sensors = sensor_config.n_active_sensors();

    html.push_str("<h3>4l. State estimation and sensor fusion</h3>\n");

    if n_sensors == 0 {
        html.push_str(
            "<p><b>Configuration: no sensors enabled.</b> The mechanism is \
             being driven open-loop — there's no measurement to compare the \
             commanded state against, so no observer / estimator is \
             possible. Configure at least one sensor in the input panel \
             (\"Sensors\" section) to populate this derivation.</p>\n",
        );
        html.push_str(
            "<p>Without sensors, the controller relies entirely on the \
             dynamics model (process equation \\(x_{k+1} = F\\,x_k + G\\,u_k\\)) \
             to predict where the mechanism is. Errors accumulate without \
             bound. Acceptable for short-duration trajectories with stiff \
             actuators (stepper motors at low load), but fragile against \
             friction, backlash, or load disturbances.</p>\n",
        );
        return;
    }

    // Identify available sensors for the prose.
    let encoder_joint = sensor_config.encoder_joint.as_deref();
    let actuator_enabled = sensor_config.actuator_position_enabled;

    html.push_str(
        "<p>State estimation combines a dynamics model (predict step) with \
         sensor measurements (update step) to produce the best estimate of \
         the mechanism's true state. The state vector for closed-loop \
         control of a 4-bar is</p>\n",
    );
    html.push_str(
        "\\[ x = \\begin{bmatrix} \\theta_2 \\\\ \\dot\\theta_2 \\end{bmatrix} \\in \\mathbb{R}^2 \\]\n",
    );
    html.push_str(
        "<p>(crank angle and angular velocity). Higher-order extensions add \
         \\(\\ddot\\theta_2\\) or per-body angles; the 2-state model below \
         is the minimum that closes the loop. Discrete-time dynamics with \
         sample period \\(\\Delta t\\):</p>\n",
    );
    html.push_str(
        "\\[ x_{k+1} = F\\,x_k + w_k,\\qquad F = \\begin{bmatrix} 1 & \\Delta t \\\\ 0 & 1 \\end{bmatrix} \\]\n",
    );
    html.push_str(
        "<p>where \\(w_k \\sim \\mathcal{N}(0, Q)\\) is the process noise \
         (un-modelled dynamics: friction, motor torque ripple, etc.). \
         Typical Q for this 2-state model:</p>\n",
    );
    html.push_str(
        "\\[ Q = \\begin{bmatrix} \\sigma_\\theta^2\\,\\Delta t^2 & 0 \\\\ 0 & \\sigma_{\\dot\\theta}^2\\,\\Delta t \\end{bmatrix} \\]\n",
    );
    html.push_str(
        "<p>(diagonal; the \\(\\Delta t\\) factors come from integrating \
         continuous-time white noise over the sample period — see e.g. \
         Crassidis &amp; Junkins, <em>Optimal Estimation</em>, ch. 4).</p>\n",
    );

    // ── Measurement model ─────────────────────────────────────────────
    html.push_str("<p><b>Measurement model</b> for your sensor selection:</p>\n");

    // Determine which row(s) appear in y = h(x).
    let mut h_rows: Vec<&'static str> = Vec::new();
    let mut h_jac_rows: Vec<&'static str> = Vec::new();
    let mut r_diag: Vec<f64> = Vec::new();
    let mut r_unit: Vec<&'static str> = Vec::new();

    if let Some(joint_id) = encoder_joint {
        // Encoder reading: relative angle of the two bodies the joint connects.
        // For typical configurations this collapses to θ_2 (when the joint
        // connects ground to crank). For other joints, the reading is
        // θ_j − θ_i which still depends on θ_2 through the kinematic chain.
        // Look up which two bodies this joint touches.
        let touches_ground_and_crank = mechanism
            .joints()
            .iter()
            .chain(mechanism.drivers().iter().map(|_| {
                // Drivers can also be encoder hosts — but we look them up below.
                // Placeholder: not all drivers are JointConstraint.
                None
            }).filter_map(|x| x))
            .any(|j| {
                j.id() == joint_id
                    && (j.body_i_id() == "ground" || j.body_j_id() == "ground")
            });
        let _ = touches_ground_and_crank;
        // Display the encoder row generically; the user can simplify by
        // hand if their encoder is on the driver joint.
        h_rows.push("\\theta_j - \\theta_i");
        h_jac_rows.push(
            "\\frac{\\partial(\\theta_j - \\theta_i)}{\\partial \\theta_2}\\quad 0",
        );
        r_diag.push(sensor_config.encoder_noise_std.powi(2));
        r_unit.push("rad²");
        html.push_str(&format!(
            "<p>Encoder on joint <code>{}</code> measures \
             \\(\\theta_j - \\theta_i\\) (relative orientation of the joint's \
             two bodies). For an encoder on a ground-attached joint (e.g. the \
             crank-ground revolute), this reduces to \\(\\theta_2\\) directly. \
             1-σ noise: \\({:.4}\\) rad ≈ \\({:.2}\\) mrad.</p>\n",
            html_escape(joint_id),
            sensor_config.encoder_noise_std,
            sensor_config.encoder_noise_std * 1000.0,
        ));
    }

    if actuator_enabled {
        let has_actuator = mechanism.forces().iter().any(|f| {
            matches!(f, crate::forces::elements::ForceElement::LinearActuator(_))
        }) || !mechanism.linear_drivers().is_empty();
        if has_actuator {
            h_rows.push("L(\\theta_2)");
            h_jac_rows.push("\\frac{dL}{d\\theta_2}\\quad 0");
            r_diag.push(sensor_config.actuator_noise_std.powi(2));
            r_unit.push("m²");
            html.push_str(&format!(
                "<p>Actuator-position sensor measures \\(L\\), the world-frame \
                 distance between the actuator's two attachment points (§4i). \
                 1-σ noise: \\({:.6}\\) m ≈ \\({:.1}\\) µm.</p>\n",
                sensor_config.actuator_noise_std,
                sensor_config.actuator_noise_std * 1e6,
            ));
        } else {
            html.push_str(
                "<p><em>Actuator-position sensor is enabled in the input panel \
                 but the mechanism has no LinearActuator force element or \
                 LinearDriver constraint. Add one (or disable this sensor) for \
                 the EKF below to be valid.</em></p>\n",
            );
        }
    }

    // Render h(x), H, R using LaTeX matrices.
    if !h_rows.is_empty() {
        html.push_str("\\[ h(x) = \\begin{bmatrix} ");
        html.push_str(&h_rows.join(" \\\\ "));
        html.push_str(" \\end{bmatrix},\\qquad ");
        html.push_str("H = \\frac{\\partial h}{\\partial x} = \\begin{bmatrix} ");
        html.push_str(&h_jac_rows.join(" \\\\ "));
        html.push_str(" \\end{bmatrix} \\]\n");
        // Measurement covariance R as a diagonal matrix.
        html.push_str("\\[ R = \\operatorname{diag}\\left(");
        let r_terms: Vec<String> = r_diag
            .iter()
            .zip(r_unit.iter())
            .map(|(v, u)| format!("{:.3e}\\,\\text{{{}}}", v, u))
            .collect();
        html.push_str(&r_terms.join(",\\;"));
        html.push_str("\\right) \\]\n");
    }

    // ── EKF predict / update equations ─────────────────────────────────
    if n_sensors == 2 {
        html.push_str(
            "<p>With both sensors enabled, the optimal fusion is the \
             <b>Extended Kalman Filter</b>. Predict step:</p>\n",
        );
    } else {
        html.push_str(
            "<p>With one sensor, the same Kalman-filter machinery applies but \
             reduces to a 1-dimensional update. The math below works for \
             either sensor count.</p>\n",
        );
    }
    html.push_str(
        "\\[ \\hat x_{k+1|k} = F\\,\\hat x_{k|k},\\qquad P_{k+1|k} = F\\,P_{k|k}\\,F^T + Q \\]\n",
    );
    html.push_str("<p>Update step (innovation form):</p>\n");
    html.push_str(
        "\\[ y_{\\mathrm{res}} = y_k - h(\\hat x_{k+1|k}),\\quad S = H\\,P_{k+1|k}\\,H^T + R \\]\n",
    );
    html.push_str(
        "\\[ K = P_{k+1|k}\\,H^T\\,S^{-1},\\quad \\hat x_{k+1|k+1} = \\hat x_{k+1|k} + K\\,y_{\\mathrm{res}} \\]\n",
    );
    html.push_str(
        "\\[ P_{k+1|k+1} = (I - K\\,H)\\,P_{k+1|k} \\]\n",
    );
    html.push_str(
        "<p>The Kalman gain \\(K\\) automatically weights each sensor by its \
         relative confidence: lower-noise sensors get more weight. When one \
         sensor's noise blows up (e.g. fault), \\(R\\) for that row is \
         large, \\(S^{-1}\\) for that row is small, and \\(K\\) automatically \
         de-weights it. <b>This is the rigorous version of \"trust the \
         sensor that's working\".</b></p>\n",
    );

    // ── Pseudocode tailored to the active sensors ─────────────────────
    let n_meas = h_rows.len();
    if n_meas > 0 {
        html.push_str("<p><b>Embedded pseudocode</b> for your sensor set:</p>\n");
        let mut code = String::new();
        code.push_str("// State and covariance (EKF)\n");
        code.push_str("float x[2] = { theta2_init, 0.0f };\n");
        code.push_str("float P[2][2] = { { sigma_init², 0 }, { 0, sigma_init² } };\n");
        code.push_str(&format!(
            "const float SIGMA_PROC_THETA = ...;     // tune empirically\n"
        ));
        code.push_str(&format!(
            "const float SIGMA_PROC_OMEGA = ...;     // tune empirically\n"
        ));
        if encoder_joint.is_some() {
            code.push_str(&format!(
                "const float SIGMA_ENC = {:.6}f;            // [rad]\n",
                sensor_config.encoder_noise_std,
            ));
        }
        if actuator_enabled {
            code.push_str(&format!(
                "const float SIGMA_ACT = {:.8}f;     // [m]\n",
                sensor_config.actuator_noise_std,
            ));
        }
        code.push_str("\n");
        code.push_str("void ekf_step(float dt) {\n");
        code.push_str("    // ── Predict ───────────────────────────────\n");
        code.push_str("    float x_pred[2] = { x[0] + x[1]*dt, x[1] };\n");
        code.push_str("    // F = [[1, dt], [0, 1]]; P_pred = F P Fᵀ + Q\n");
        code.push_str("    float P_pred[2][2];\n");
        code.push_str("    P_pred[0][0] = P[0][0] + dt*(P[1][0] + P[0][1]) + dt*dt*P[1][1]\n");
        code.push_str("                 + SIGMA_PROC_THETA*SIGMA_PROC_THETA * dt*dt;\n");
        code.push_str("    P_pred[0][1] = P[0][1] + dt*P[1][1];\n");
        code.push_str("    P_pred[1][0] = P[1][0] + dt*P[1][1];\n");
        code.push_str("    P_pred[1][1] = P[1][1]\n");
        code.push_str("                 + SIGMA_PROC_OMEGA*SIGMA_PROC_OMEGA * dt;\n");
        code.push_str("\n");
        code.push_str("    // ── Update ───────────────────────────────\n");
        code.push_str("    // Build h(x_pred) and H = ∂h/∂x for the active sensors.\n");
        if encoder_joint.is_some() && actuator_enabled {
            code.push_str("    // 2 sensors: H is 2x2 (one row per measurement).\n");
            code.push_str("    float h_pred[2] = { x_pred[0],  L_of_theta2(x_pred[0]) };\n");
            code.push_str("    float dL = dL_dtheta(x_pred[0]);  // §4i sensitivity\n");
            code.push_str("    // H = [[1, 0], [dL, 0]];\n");
            code.push_str("    float y_meas[2] = { read_encoder(), read_actuator() };\n");
            code.push_str("    float y_res[2] = { y_meas[0] - h_pred[0], y_meas[1] - h_pred[1] };\n");
            code.push_str("    // S = H P_pred Hᵀ + R, K = P_pred Hᵀ S⁻¹.\n");
            code.push_str("    // ... 2x2 inversion. See Crassidis & Junkins ch. 4.\n");
        } else if encoder_joint.is_some() {
            code.push_str("    // 1 sensor (encoder): H = [1, 0].\n");
            code.push_str("    float h_pred = x_pred[0];\n");
            code.push_str("    float y_res = read_encoder() - h_pred;\n");
            code.push_str("    float S = P_pred[0][0] + SIGMA_ENC*SIGMA_ENC;\n");
            code.push_str("    float K[2] = { P_pred[0][0] / S, P_pred[1][0] / S };\n");
            code.push_str("    x[0] = x_pred[0] + K[0] * y_res;\n");
            code.push_str("    x[1] = x_pred[1] + K[1] * y_res;\n");
            code.push_str("    P[0][0] = (1 - K[0]) * P_pred[0][0];\n");
            code.push_str("    P[0][1] = (1 - K[0]) * P_pred[0][1];\n");
            code.push_str("    P[1][0] = P_pred[1][0] - K[1] * P_pred[0][0];\n");
            code.push_str("    P[1][1] = P_pred[1][1] - K[1] * P_pred[0][1];\n");
        } else if actuator_enabled {
            code.push_str("    // 1 sensor (actuator): H = [dL/dtheta, 0].\n");
            code.push_str("    float h_pred = L_of_theta2(x_pred[0]);\n");
            code.push_str("    float dL = dL_dtheta(x_pred[0]);\n");
            code.push_str("    float y_res = read_actuator() - h_pred;\n");
            code.push_str("    float S = dL*dL * P_pred[0][0] + SIGMA_ACT*SIGMA_ACT;\n");
            code.push_str("    float K[2] = { dL * P_pred[0][0] / S, dL * P_pred[1][0] / S };\n");
            code.push_str("    x[0] = x_pred[0] + K[0] * y_res;\n");
            code.push_str("    x[1] = x_pred[1] + K[1] * y_res;\n");
            code.push_str("    P[0][0] = P_pred[0][0] - K[0] * dL * P_pred[0][0];\n");
            code.push_str("    P[0][1] = P_pred[0][1] - K[0] * dL * P_pred[0][1];\n");
            code.push_str("    P[1][0] = P_pred[1][0] - K[1] * dL * P_pred[0][0];\n");
            code.push_str("    P[1][1] = P_pred[1][1] - K[1] * dL * P_pred[0][1];\n");
        }
        code.push_str("}\n");
        // HTML-escape the < and > in the code block. We pre-format with
        // monospace styling.
        let escaped = html_escape(&code);
        html.push_str(&format!(
            "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
             font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
             overflow-x: auto;'>{}</pre>\n",
            escaped,
        ));
    }

    // ── Practical caveats ──────────────────────────────────────────────
    html.push_str("<p><b>Practical caveats</b>:</p>\n");
    html.push_str("<ul>\n");
    html.push_str(
        "<li><b>Σ initialization</b>: at startup, set \\(P_{0|0}\\) to large values \
         (e.g. \\(\\sigma_{\\theta,0} = 1\\) rad, \\(\\sigma_{\\dot\\theta,0} = 10\\) rad/s) \
         so the first few measurements correct it quickly. Don't initialize to zero — \
         the filter would refuse to update.</li>\n",
    );
    html.push_str(
        "<li><b>Innovation gating</b>: if \\(|y_{\\mathrm{res}}| > 5 \\sqrt{S}\\) for several \
         cycles, treat the measurement as an outlier (likely a sensor fault) and skip \
         the update. This is the practical fault-detection mechanism the EKF gives \
         you for free.</li>\n",
    );
    html.push_str(
        "<li><b>Linearization error near singularities</b>: \\(dL/d\\theta_2\\) goes \
         to zero at dead points (§4f). The EKF's H Jacobian becomes ill-conditioned \
         there; the filter degrades gracefully but the actuator-position sensor \
         loses observability of \\(\\theta_2\\). If your trajectory passes through \
         a dead point, lean on the encoder during that pose.</li>\n",
    );
    html.push_str(
        "<li><b>Tuning Q</b>: empirical. Start with \\(\\sigma_{\\dot\\theta} \\approx \
         \\) 10% of expected angular velocity; increase if the filter lags input \
         changes, decrease if it tracks measurement noise.</li>\n",
    );
    html.push_str(
        "<li><b>References</b>: Welch &amp; Bishop, <em>An Introduction to the \
         Kalman Filter</em> (UNC TR-95-041); Crassidis &amp; Junkins, <em>Optimal \
         Estimation of Dynamic Systems</em> (CRC, 2011).</li>\n",
    );
    html.push_str("</ul>\n");
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
        "\\[ BD_{{\\min}} = |d - a| = {:.4}\\,\\text{{m}},\\quad BD_{{\\max}} = d + a = {:.4}\\,\\text{{m}} \\]\n",
        bd_min, bd_max,
    ));
    html.push_str(&format!(
        "\\[ \\mu_{{\\min}} = {:.3}^\\circ,\\quad \\mu_{{\\max}} = {:.3}^\\circ \\]\n",
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
/// §6 Trajectory mode — full inverse-kinematics derivation.
///
/// Replaces the previous one-paragraph stub that pointed at the spec doc;
/// the reader gets §8.1–§8.4 of `docs/superpowers/specs/2026-04-29-linkage-
/// equations-reference.md` inlined, with explicit per-ControlTarget
/// formulas for ∇_q g and ∇²_q g, the dq/du and d²q/du² derivations via
/// the implicit function theorem, and a side-by-side comparison of the
/// analytic vs finite-difference Hessian implementation choice.
///
/// Mechanism-independent (the math is the same for every planar linkage),
/// so this function takes no `mechanism` / `q` / `layout` parameters.
fn write_trajectory_inverse_section(html: &mut String) {
    html.push_str("<h3>6. Trajectory mode (inverse position control)</h3>\n");

    // ── §6.0 Plain-English overview + glossary ────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.0 What this section is about (read this first)</h3>\n",
    );
    html.push_str(
        "<p>The earlier sections (§1–§5) solved the <b>forward</b> problem: \
         given a crank angle, where does every body sit and what forces \
         act on each joint? That's what mechanism textbooks teach.</p>\n",
    );
    html.push_str(
        "<p>This section solves the <b>inverse</b> problem instead: given a \
         desired motion of <em>some specific output</em> (the tip of a \
         coupler, the orientation of an end-effector, the distance to a \
         workpiece), what crank-angle (or actuator-stroke) trajectory \
         produces it? <b>This is what you need to actually control the \
         mechanism on hardware.</b></p>\n",
    );
    html.push_str(
        "<p style='background: #fffbe5; border-left: 3px solid #d4a000; padding: 8px 12px;'>\
         <b>Concrete example.</b> Imagine you've built the 4-bar shown in \
         the diagram above and you want the coupler-tip's world-X coordinate \
         to advance smoothly from \\(x = 0\\) to \\(x = 50\\,\\text{mm}\\) over 1 \
         second. The forward sweep can't help — it sweeps the <em>crank</em> \
         angle, not the tip's x-coordinate, and the relationship between \
         the two is nonlinear and has no closed form. The inverse-kinematics \
         solver in this section computes a crank-angle profile \
         \\(u(t) = \\theta_{\\text{crank}}(t)\\) such that, at every time \
         \\(t \\in [0, 1]\\), driving the crank to \\(u(t)\\) puts the \
         coupler tip at \\(x = h(t) = 0.05\\,t\\) m. That \\(u(t)\\) is what \
         you'd send to your servo motor.</p>\n",
    );

    html.push_str("<p><b>Notation glossary</b> (every symbol used in §6):</p>\n");
    html.push_str(
        "<table>\n\
         <tr><th>Symbol</th><th>Plain-English meaning</th><th>Where it comes from</th></tr>\n\
         <tr><td>\\(q\\)</td>\
         <td>The mechanism's full <em>pose</em>: a vector of \\((x_i, y_i, \\theta_i)\\) for every non-ground body. \
         Lives in \\(\\mathbb{R}^{3n}\\) for \\(n\\) moving bodies.</td>\
         <td>§1, §4a (numeric values for this mechanism)</td></tr>\n\
         <tr><td>\\(u\\)</td>\
         <td>The <em>one</em> number you control on the hardware: crank angle \\(\\theta_{\\text{crank}}\\) (revolute driver) or actuator length \\(L\\) (linear driver). Scalar.</td>\
         <td>The driver input; in §1 it was called \\(f(t)\\).</td></tr>\n\
         <tr><td>\\(\\Phi(q, t)\\)</td>\
         <td>The constraint vector — joint coincidences plus the driver row. Equals zero at any feasible pose.</td>\
         <td>§2 (catalogued per joint type)</td></tr>\n\
         <tr><td>\\(\\Phi_q\\)</td>\
         <td>Constraint Jacobian \\(\\partial \\Phi/\\partial q\\), the \\(m \\times n\\) matrix of partials.</td>\
         <td>§3 (block-sparse formulas)</td></tr>\n\
         <tr><td>\\(\\Phi_u\\)</td>\
         <td>Constraint sensitivity to the driver input: a single column vector, \\(-1\\) on the driver row and zero everywhere else.</td>\
         <td>Defined below (§6.1).</td></tr>\n\
         <tr><td>\\(g(q)\\)</td>\
         <td>The <em>output observable</em> — the scalar quantity you want to control. Five variants in this simulator: <em>Angle</em>, <em>WorldX</em>, <em>WorldY</em>, <em>Projection</em>, <em>Distance</em> (full table in §6.4).</td>\
         <td>You pick one in the GUI.</td></tr>\n\
         <tr><td>\\(h(t)\\)</td>\
         <td>The <em>desired</em> value of \\(g\\) at time \\(t\\) — i.e. the trajectory you specify.</td>\
         <td>You define this via the Profile editor (constant-speed / trapezoidal / S-curve / keyframed).</td></tr>\n\
         <tr><td>\\(r(u)\\)</td>\
         <td>The residual: \\(g(q(u)) - h\\). At every Newton step we want this to be zero.</td>\
         <td>Defined in §6.1.</td></tr>\n\
         <tr><td>\\(dq/du\\)</td>\
         <td>How the entire pose moves when you nudge the one number you control. The motion direction along the constraint manifold.</td>\
         <td>Implicit function theorem on \\(\\Phi(q(u), u) = 0\\); §6.1.</td></tr>\n\
         <tr><td>\\(r'(u)\\)</td>\
         <td>The Newton slope: \\(dg/du = \\nabla_q g \\cdot dq/du\\). Says \"if I nudge \\(u\\) by 1 unit, the residual changes by this much.\"</td>\
         <td>§6.1.</td></tr>\n\
         </table>\n",
    );

    html.push_str(
        "<p><b>The whole algorithm in plain English</b> (skip the math \
         below and you'll still know what's happening):</p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li>You specify a desired observable trajectory \\(h(t)\\) — say \
         \"my coupler tip should be at \\(x = 0.025\\) m at \\(t = 0.5\\) s.\"</li>\n\
         <li>The simulator picks the most recent crank angle as a starting \
         guess and asks: \"if I drive the crank to <em>this</em> angle, \
         what \\(g(q)\\) do I get?\" Call the answer \\(g_{\\text{achieved}}\\).</li>\n\
         <li>Compare \\(g_{\\text{achieved}}\\) to \\(h\\). If they're \
         within tolerance — done, this is the right crank angle.</li>\n\
         <li>If not — Newton's method tells you how to adjust the crank \
         angle to close the gap. Adjust, re-solve the mechanism, compare \
         again.</li>\n\
         <li>2-4 iterations later, you've found a \\(u\\) that makes \
         \\(g(q(u)) = h\\) within \\(10^{-8}\\). Record \\((t, h, u, q)\\) \
         as one trajectory sample. Move to the next time-step. Warm-start \
         from the last \\(u\\).</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p>Everything else in §6 is either (a) deriving the Newton update \
         step in detail, (b) deriving the closed-form velocity / acceleration \
         inverses for free once you have the position inverse, or (c) \
         per-variant formulas for \\(\\nabla_q g\\) and \\(\\nabla_q^2 g\\) \
         that the simulator hard-codes.</p>\n",
    );

    // ── §6.1 Position inverse ─────────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.1 Position inverse — find \\(u\\) such that \\(g(q(u)) = h\\)</h3>\n",
    );
    html.push_str(
        "<p>Define the <b>residual</b> — the gap between what we have and \
         what we want:</p>\n\
         \\[ r(u) = g(q(u)) - h \\]\n\
         <p>We want to find the \\(u^*\\) where \\(r(u^*) = 0\\). Newton's \
         method does this by repeatedly stepping in the direction of the \
         tangent line:</p>\n\
         \\[ u^{(k+1)} = u^{(k)} - \\frac{r(u^{(k)})}{r'(u^{(k)})} \\]\n\
         <p>where \\(r'(u) = dg/du\\) is the rate at which the residual \
         changes when you nudge \\(u\\). Computing this is the only part \
         that needs work — \\(r(u)\\) itself is just one forward solve.</p>\n",
    );
    html.push_str(
        "<p>The chain rule plus the <b>implicit function theorem</b> applied \
         to \\(\\Phi(q(u),u) = 0\\) (this says: \"as \\(u\\) changes, \\(q\\) \
         must change too, in just the right way to keep all the constraints \
         satisfied\") gives</p>\n",
    );
    html.push_str(
        "\\[ \\Phi_q\\,\\frac{dq}{du} + \\Phi_u = 0 \\quad\\Longrightarrow\\quad \\frac{dq}{du} = -\\Phi_q^{-1}\\,\\Phi_u \\]\n\
         \\[ r'(u) = \\nabla_q g \\cdot \\frac{dq}{du} = -\\nabla_q g \\cdot \\Phi_q^{-1}\\,\\Phi_u \\]\n",
    );
    html.push_str(
        "<p>where \\(\\Phi_u = \\partial \\Phi/\\partial u\\) is the column of \
         derivatives w.r.t. the driver input. For the simulator's two driver \
         parameterizations:</p>\n",
    );
    html.push_str(
        "<ul>\n\
         <li><b>Revolute driver</b>: \\(f(t) = \\theta_0 + \\omega t\\) means \\(u = \\theta_j - \\theta_i\\). \
         \\(\\Phi_u\\) is \\(-1\\) on the driver row, zero everywhere else.</li>\n\
         <li><b>Linear driver</b>: \\(d(t) = L_0 + v t\\) means \\(u = L\\) (the actuator length). \
         \\(\\Phi_u\\) is again \\(-1\\) on the driver row, zero elsewhere.</li>\n\
         </ul>\n",
    );
    html.push_str(
        "<p>So \\(\\Phi_u\\) is a single column with one non-zero entry — \
         identical in shape regardless of driver kind. This is exactly \
         <code>\\(\\Phi_t\\) divided by the parameterization rate</code>, and the \
         simulator reuses <code>assemble_phi_t</code> with a divide.</p>\n",
    );
    html.push_str("<p><b>Outer-loop algorithm</b> (each step annotated with what it does in plain English):</p>\n");
    html.push_str(
        "<ol>\n\
         <li><b>Warm start.</b> \\(u_0 \\leftarrow\\) the driver input from the previous trajectory sample. \
         (For the first sample, use whatever's in the GUI's driver field.)</li>\n\
         <li><b>Forward solve.</b> Run <code>solve_position</code>(\\(q_{k-1}, u_k\\)) to get the pose \
         \\(q_k\\) consistent with the current driver input \\(u_k\\). This is the same Newton-on-Φ \
         loop from §4 — typically converges in 2-3 inner iterations from a warm start.</li>\n\
         <li><b>Check the residual.</b> \\(r_k \\leftarrow g(q_k) - h\\). \
         If \\(|r_k| < \\text{tol}\\) (default \\(10^{-8}\\)), <b>return</b> \\(u_k\\) — we're done.</li>\n\
         <li><b>Compute the Newton slope.</b> One extra linear solve: \\(\\Phi_q\\,s = -\\Phi_u\\). \
         The result \\(s = dq/du\\) tells you which direction the pose moves as you nudge \\(u\\). \
         Then \\(r' \\leftarrow \\nabla_q g \\cdot s\\) — the rate of change of the residual.</li>\n\
         <li><b>Newton step.</b> \\(u_{k+1} \\leftarrow u_k - r_k / r'_k\\). Loop back to step 2.</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p><b>Cost per outer iteration:</b> one forward position solve + one extra \
         linear solve for \\(s\\). <b>Convergence:</b> quadratic away from singularities — \
         in practice 2-4 outer iterations to reach \\(10^{-8}\\) residual when warm-started \
         from the previous sample. <b>Fallback:</b> if Newton diverges or branch-jumps \
         (the pose suddenly flips to a different assembly mode), the simulator falls back \
         to bisection on a workspace-probe table built up-front. See \
         <code>src/solver/inverse_kinematics/solver.rs::solve_for_target</code>.</p>\n",
    );

    // ── §6.2 Velocity inverse ─────────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.2 Velocity inverse — find \\(\\dot u\\) such that \\(\\dot g = \\dot h\\)</h3>\n",
    );
    html.push_str(
        "<p><b>Why you care:</b> §6.1 gave you the actuator <em>positions</em> at \
         each sample. To control real hardware you also need actuator <em>velocities</em> \
         — that's the rate command you'd send to a velocity-mode servo, or the input \
         used by the inner velocity loop of a position-mode servo. The velocity inverse \
         answers: \"if my desired observable is changing at rate \\(\\dot h\\), how fast \
         does my driver input need to change?\"</p>\n",
    );
    html.push_str(
        "<p>Closed form, no iteration needed. Differentiate \\(g(q(u(t)))\\) once \
         w.r.t. \\(t\\):</p>\n",
    );
    html.push_str(
        "\\[ \\dot g = \\nabla_q g \\cdot \\dot q = \\nabla_q g \\cdot \\frac{dq}{du}\\,\\dot u = r'(u)\\,\\dot u \\]\n\
         \\[ \\boxed{\\;\\dot u = \\dot h\\,/\\,r'(u)\\;} \\]\n",
    );
    html.push_str(
        "<p>\\(r'(u)\\) was already computed in §6.1, so velocity inversion \
         is one division per timestep. The body-velocity vector \\(\\dot q\\) \
         then comes from the §4 velocity solve \\(\\Phi_q\\,\\dot q = -\\Phi_t\\), \
         but with \\(\\Phi_t\\) recomputed using the back-solved \\(\\dot u\\) \
         (not the user-set nominal driver rate). Equivalently: scale the \
         existing \\(\\dot q\\) result by \\(\\dot u\\) divided by the nominal \
         rate.</p>\n",
    );

    // ── §6.3 Acceleration inverse ─────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.3 Acceleration inverse — find \\(\\ddot u\\) such that \\(\\ddot g = \\ddot h\\)</h3>\n",
    );
    html.push_str(
        "<p><b>Why you care:</b> needed if you're driving a torque- or \
         current-mode actuator (where the hardware applies force, not \
         velocity), if you need feed-forward acceleration in a high-\
         performance servo loop, or if you're computing the inertial \
         loading on the mechanism (which scales with \\(\\ddot q\\)). For \
         most velocity-mode position-control applications, §6.1 + §6.2 \
         is enough; this subsection is here for completeness.</p>\n",
    );
    html.push_str(
        "<p>Differentiate \\(\\dot g = r'(u)\\,\\dot u\\) once more w.r.t. \
         \\(t\\):</p>\n",
    );
    html.push_str(
        "\\[ \\ddot g = r''(u)\\,\\dot u^2 + r'(u)\\,\\ddot u \\]\n\
         \\[ \\boxed{\\;\\ddot u = \\big(\\ddot h - r''(u)\\,\\dot u^2\\big)\\,/\\,r'(u)\\;} \\]\n",
    );
    html.push_str(
        "<p>The new term is \\(r''(u) = d^2 g / du^2\\). By chain rule, this \
         decomposes into <b>two contributions</b>:</p>\n",
    );
    html.push_str(
        "\\[ r''(u) = \\underbrace{\\nabla_q^2 g\\,(dq/du,\\,dq/du)}_{\\text{Hessian of observable}} \\;+\\; \\underbrace{\\nabla_q g \\cdot (d^2 q/du^2)}_{\\text{constraint-acceleration term}} \\]\n",
    );
    html.push_str(
        "<p><b>The constraint-acceleration term</b> \\(d^2 q/du^2\\) comes \
         from differentiating \\(\\Phi_q (dq/du) + \\Phi_u = 0\\) once more \
         w.r.t. \\(u\\):</p>\n",
    );
    html.push_str(
        "\\[ \\Phi_q\\,\\frac{d^2 q}{du^2} = -\\big[\\,\\Phi_{qq}(dq/du,\\,dq/du) \\;+\\; 2\\,\\Phi_{qu}(dq/du) \\;+\\; \\Phi_{uu}\\,\\big] \\]\n",
    );
    html.push_str(
        "<p>For our drivers \\(\\Phi\\) depends on \\(u\\) only through a \
         linear \\(-u\\) term, so \\(\\Phi_{qu} = 0\\) and \\(\\Phi_{uu} = 0\\), and \
         the RHS reduces to \\(-\\Phi_{qq}(dq/du,\\,dq/du)\\). This is \
         <b>structurally identical to the velocity-quadratic part of \\(\\gamma\\) \
         from §4</b>: assemble \\(\\gamma\\) with \\((dq/du)\\) substituted for \
         \\(\\dot q\\) and the explicit driver-row \\(f''(t)\\) / \\(d''(t)\\) \
         terms zeroed out. The simulator exposes this as a \"kinematic-only \
         \\(\\gamma\\)\" variant of the existing assembly code.</p>\n",
    );
    html.push_str(
        "<p><b>The Hessian-of-observable term</b> \\(\\nabla_q^2 g\\) depends on \
         the ControlTarget variant — explicit formulas in §6.4 below.</p>\n",
    );

    // ── §6.4 Per-ControlTarget formulas ───────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.4 Per-ControlTarget formulas</h3>\n",
    );
    html.push_str(
        "<p>The simulator offers five <code>ControlTarget</code> variants in \
         the GUI's Target observable picker. Each one is a different choice \
         for \\(g(q)\\) — i.e. <em>which scalar quantity</em> you want to \
         control. The math in §6.1–§6.3 needs \\(g(q)\\) itself, its gradient \
         \\(\\nabla_q g\\), and (for acceleration inversion) its Hessian \
         \\(\\nabla_q^2 g\\). This subsection lists those for every variant.</p>\n",
    );
    html.push_str(
        "<p><b>Notation</b>: body \\(i\\) has CG position \\(r_i = (x_i, y_i)\\) and orientation \
         \\(\\theta_i\\); the body-local point of interest is \\(\\mathbf{s} = (s_x, s_y)\\). \
         The world-frame point is \\(P_i(\\mathbf{s}) = r_i + A(\\theta_i)\\mathbf{s}\\), \
         and \\(B(\\theta) = dA/d\\theta = \\begin{bmatrix} -\\sin\\theta & -\\cos\\theta \\\\ \\cos\\theta & -\\sin\\theta \\end{bmatrix}\\). \
         Empty cells in the Hessian column indicate identically zero entries.</p>\n",
    );
    html.push_str(
        "<p style='background: #eef5fc; border-left: 3px solid #4a87bf; padding: 8px 12px;'>\
         <b>Worked example: WorldX of a coupler tip.</b> Suppose you choose \
         <em>WorldX</em> as your <code>ControlTarget</code> and the body-local point \
         \\(\\mathbf{s} = (0.04, 0.0)\\) m on the coupler (the tip 40 mm past the \
         coupler's local origin along its body x-axis). Then</p>\n\
         \\[ g(q) = x_{\\text{coupler}} + \\cos\\theta_{\\text{coupler}}\\,(0.04) - \\sin\\theta_{\\text{coupler}}\\,(0) = x_{\\text{coupler}} + 0.04\\cos\\theta_{\\text{coupler}} \\]\n\
         <p>Reading off \\(\\nabla_q g\\) (which entries are non-zero?): \
         only the coupler's coordinates appear, and only \\(\\partial g/\\partial x_{\\text{coupler}} = 1\\) and \
         \\(\\partial g/\\partial \\theta_{\\text{coupler}} = -0.04\\sin\\theta_{\\text{coupler}}\\) \
         are non-zero. So \\(\\nabla_q g\\) is a vector that's <em>mostly zero</em> with two non-zero entries — \
         exactly what the table below records as the \"Non-zero \\(\\nabla_q g\\) entries\" column. \
         At a converged pose with \\(\\theta_{\\text{coupler}} \\approx 0.5\\) rad, those entries are \
         \\(1\\) and \\(-0.04 \\cdot 0.479 \\approx -0.0192\\) — the second entry is the slope of \
         WorldX with respect to coupler rotation at this pose. The Hessian (next column) gives the \
         curvature: \\(\\partial^2 g/\\partial \\theta_{\\text{coupler}}^2 = -0.04\\cos\\theta_{\\text{coupler}} \\approx -0.0351\\). \
         These three numbers are everything §6.1–§6.3 need to invert this trajectory.</p>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th>Variant</th><th>\\(g(q)\\)</th><th>Non-zero \\(\\nabla_q g\\) entries</th><th>Non-zero \\(\\nabla_q^2 g\\) entries</th></tr>\n",
    );
    // Angle
    html.push_str(
        "<tr><td><b>Angle</b></td>\
         <td>\\(\\theta_i\\)</td>\
         <td>\\(\\partial g/\\partial \\theta_i = 1\\)</td>\
         <td>—</td></tr>\n",
    );
    // WorldX
    html.push_str(
        "<tr><td><b>WorldX</b></td>\
         <td>\\(\\mathbf{e}_x \\cdot P_i(\\mathbf{s}) = x_i + \\cos\\theta_i\\,s_x - \\sin\\theta_i\\,s_y\\)</td>\
         <td>\\(\\partial g/\\partial x_i = 1,\\;\\;\\partial g/\\partial \\theta_i = -\\sin\\theta_i\\,s_x - \\cos\\theta_i\\,s_y\\)</td>\
         <td>\\(\\partial^2 g/\\partial \\theta_i^2 = -(\\cos\\theta_i\\,s_x - \\sin\\theta_i\\,s_y) = -\\mathbf{e}_x\\cdot A(\\theta_i)\\mathbf{s}\\)</td></tr>\n",
    );
    // WorldY
    html.push_str(
        "<tr><td><b>WorldY</b></td>\
         <td>\\(\\mathbf{e}_y \\cdot P_i(\\mathbf{s}) = y_i + \\sin\\theta_i\\,s_x + \\cos\\theta_i\\,s_y\\)</td>\
         <td>\\(\\partial g/\\partial y_i = 1,\\;\\;\\partial g/\\partial \\theta_i = \\cos\\theta_i\\,s_x - \\sin\\theta_i\\,s_y\\)</td>\
         <td>\\(\\partial^2 g/\\partial \\theta_i^2 = -(\\sin\\theta_i\\,s_x + \\cos\\theta_i\\,s_y) = -\\mathbf{e}_y\\cdot A(\\theta_i)\\mathbf{s}\\)</td></tr>\n",
    );
    // Projection
    html.push_str(
        "<tr><td><b>Projection</b></td>\
         <td>\\(\\hat{\\mathbf{u}} \\cdot (P_i(\\mathbf{s}) - \\mathbf{p}_0)\\)</td>\
         <td>\\(\\partial g/\\partial r_i = \\hat{\\mathbf{u}},\\;\\;\\partial g/\\partial \\theta_i = \\hat{\\mathbf{u}} \\cdot B(\\theta_i)\\mathbf{s}\\)</td>\
         <td>\\(\\partial^2 g/\\partial \\theta_i^2 = -\\hat{\\mathbf{u}} \\cdot A(\\theta_i)\\mathbf{s}\\)</td></tr>\n",
    );
    // Distance
    html.push_str(
        "<tr><td><b>Distance</b></td>\
         <td>\\(\\|P_i(\\mathbf{s}) - \\mathbf{p}_0\\|\\)</td>\
         <td>Let \\(\\mathbf{d} = P_i - \\mathbf{p}_0\\), \\(L = \\|\\mathbf{d}\\|\\), \\(\\hat{\\mathbf{n}} = \\mathbf{d}/L\\). \
         Then \\(\\partial g/\\partial r_i = \\hat{\\mathbf{n}},\\;\\;\\partial g/\\partial \\theta_i = \\hat{\\mathbf{n}} \\cdot B(\\theta_i)\\mathbf{s}\\)</td>\
         <td>Non-trivial; comes from differentiating the unit direction \
         \\(\\hat{\\mathbf{n}}\\). Block at \\((r_i, r_i)\\): \\((I - \\hat{\\mathbf{n}}\\hat{\\mathbf{n}}^T)/L\\). \
         Cross terms involve \\(B(\\theta_i)\\mathbf{s}\\) and \\(A(\\theta_i)\\mathbf{s}\\). \
         Implemented as <code>ControlTarget::Distance::hessian</code>.</td></tr>\n",
    );
    html.push_str("</table>\n");
    html.push_str(
        "<p>The Angle variant has zero Hessian — the simplest case, \
         which is why §4a's worked example uses it.</p>\n",
    );

    // ── §6.5 Implementation choices: analytic vs FD Hessian ───────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.5 Analytic vs finite-difference \\(r''(u)\\)</h3>\n",
    );
    html.push_str(
        "<p><b>Why two implementations?</b> The Hessian-of-observable term \
         \\(\\nabla_q^2 g\\) is messy to derive in closed form for some \
         <code>ControlTarget</code> variants (especially <em>Distance</em> — \
         see §6.4). Rather than risk a bug in hand-coded Hessians, the \
         simulator <em>also</em> ships a finite-difference variant that \
         sidesteps the algebra entirely by sampling \\(r'(u \\pm \\delta)\\) \
         and differencing. Both paths produce the same \\(\\ddot u\\) to \
         floating-point noise; the FD variant trades extra forward solves \
         for not having to maintain Hessian formulas.</p>\n",
    );
    html.push_str(
        "<p>The simulator offers two implementations of \\(r''(u)\\). They \
         agree to ~\\(10^{-6}\\) at well-conditioned poses; analytic is \
         preferred for hardware export, FD is the conservative default during \
         GUI exploration.</p>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th>Method</th><th>How</th><th>Cost / iteration</th><th>Trade-offs</th></tr>\n\
         <tr><td><b>(a) Analytic</b></td>\
         <td>Each ControlTarget supplies \\(\\nabla_q^2 g\\) (table above) plus the kinematic-only \\(\\gamma\\) for \\(d^2 q/du^2\\). \
         Combined per the §6.3 decomposition.</td>\
         <td>1 extra linear solve (for \\(d^2 q/du^2\\))</td>\
         <td>Exact (modulo float precision); breaks at singularities the same way analytic gradients do; needs ~50 LoC per variant for the Hessian.</td></tr>\n\
         <tr><td><b>(b) Finite difference</b></td>\
         <td>\\(r'(u \\pm \\delta)\\) via two extra forward solves; \\(r''(u) \\approx (r'(u+\\delta) - r'(u-\\delta))/(2\\delta)\\)</td>\
         <td>2 extra forward solves (warm-started, 2–3 inner iters each)</td>\
         <td>\\(O(\\delta^2)\\) accurate; sidesteps the Hessian formulas entirely; robust near singularities; \\(\\delta\\) must be tuned for the input parameter scale.</td></tr>\n\
         </table>\n",
    );
    html.push_str(
        "<p>Both paths live in \
         <code>src/solver/inverse_kinematics/derivatives.rs</code>: \
         <code>inverse_acceleration_fd</code> is the default (used by \
         <code>compute_trajectory</code>) and <code>inverse_acceleration_analytic</code> \
         is opt-in via <code>ControlTarget::hessian</code>.</p>\n",
    );

    // ── §6.6 Cascade reuse note ───────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>6.6 What this means for the existing cascade</h3>\n",
    );
    html.push_str(
        "<p><b>Read this if you're wondering \"so once I have \\(u(t)\\), what \
         else does the simulator give me?\"</b> The §6.1–§6.3 outputs \
         \\((u, \\dot u, \\ddot u)\\) feed directly back into the §4 forward \
         solvers — body velocities, body accelerations, joint reactions, \
         actuator torque/force are all computed by the same code that \
         handles forward sweeps, just with the trajectory's \\(\\dot u, \\ddot u\\) \
         on the driver row instead of the constant-omega values.</p>\n",
    );
    html.push_str(
        "<p>Once we have \\(u(t),\\,\\dot u(t),\\,\\ddot u(t)\\) along the trajectory, \
         the body acceleration \\(\\ddot q(t)\\) is just the existing \
         \\(\\Phi_q\\,\\ddot q = \\gamma\\) solve from §4 — but \\(\\gamma\\) now \
         incorporates the trajectory's \\(\\dot u,\\ddot u\\) via \
         \\(\\Phi_t = -\\dot u,\\;\\Phi_{tt} = -\\ddot u\\) on the driver row. \
         <b>No new acceleration-level math beyond §4</b>; trajectory mode just \
         feeds different RHS values into the same solver.</p>\n",
    );
    html.push_str(
        "<p>The same is true for actuator-force computation: at each timestep \
         the existing <code>solve_statics</code> (or <code>solve_inverse_dynamics</code>) \
         runs with the back-solved \\(q\\) — its multiplier on the driver row is \
         \\(F_{\\text{actuator}}(t)\\) for a linear driver or \\(\\tau_{\\text{driver}}(t)\\) \
         for a revolute one. So the cascade is bottom-up reused: trajectory \
         mode is a <em>thin outer Newton loop wrapping the existing forward \
         solver</em>, with the per-variant gradient and Hessian as the only \
         new code.</p>\n",
    );
}

/// §7 — practical deployment recipes for inverse kinematics + state
/// estimation on real-time hardware. The report's earlier sections
/// (§4, §6, §4l) derive the math that's right for *design-time*
/// trajectory planning and analysis. This section is the bridge to the
/// firmware: which method goes on the MCU, what the architecture looks
/// like, and what the simulator doesn't yet generate for a one-click
/// deployment.
fn write_hardware_deployment_section(
    html: &mut String,
    sensor_config: &crate::gui::state::SensorConfig,
) {
    html.push_str("<h3>7. Deploying to hardware</h3>\n");

    // ── §7.0 Two roles, two algorithms ────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.0 Design-time vs runtime: don't conflate them</h3>\n",
    );
    html.push_str(
        "<p>The math in this report serves <b>two distinct roles</b>. The \
         simulator's strength is at one role; your firmware's strength is at \
         the other. Confusing them produces controllers that are either too \
         slow (running design-time math at runtime) or too fragile (running \
         runtime math at design time without numerical safeguards).</p>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th></th><th>Design time (the simulator's job)</th><th>Runtime (your firmware's job)</th></tr>\n\
         <tr><td><b>Where it runs</b></td><td>Laptop, browser</td><td>MCU (Cortex-M, ESP32, etc.)</td></tr>\n\
         <tr><td><b>Time budget</b></td><td>Seconds per trajectory</td><td>10–1000 µs per control cycle</td></tr>\n\
         <tr><td><b>Generality</b></td><td>Any mechanism topology</td><td>Specialised for THIS mechanism</td></tr>\n\
         <tr><td><b>Math</b></td><td>IFT-Newton on \\(\\Phi_q\\) (§4, §6)</td><td>Closed-form (§4h) or table interpolation</td></tr>\n\
         <tr><td><b>Failure mode</b></td><td>Slow convergence, you wait</td><td>Missed deadline, hardware misbehaves</td></tr>\n\
         <tr><td><b>Output</b></td><td>(t, q, q̇, q̈, λ, F) trajectory tables</td><td>Live (u, q̂, command) at every cycle</td></tr>\n\
         </table>\n",
    );
    html.push_str(
        "<p><b>The simulator's IFT-Newton outer loop is optimal for design \
         time and a poor choice for runtime.</b> Per-iteration matrix \
         factorisation costs ~10–50 µs on a Cortex-M7 — acceptable but \
         wasteful when closed-form alternatives exist.</p>\n",
    );

    // ── §7.1 IK deployment ────────────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.1 Inverse-kinematics deployment</h3>\n",
    );
    html.push_str(
        "<p>Two questions decide which method goes on the MCU:</p>\n\
         <ol>\n\
         <li><b>Is the trajectory known in advance?</b> (vs. live, operator-driven, or sensor-driven)</li>\n\
         <li><b>Is the mechanism a 4-bar (or another topology with closed-form forward kinematics)?</b></li>\n\
         </ol>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th>Trajectory</th><th>Mechanism</th><th>Recommended method</th><th>Cost</th><th>LOC (C)</th></tr>\n\
         <tr><td>Known offline</td><td>Any</td><td>Pre-compute table → linear/cubic interpolate</td><td>~1 µs</td><td>~50</td></tr>\n\
         <tr><td>Live</td><td>4-bar</td><td>Closed-form §4h + 1-D Newton</td><td>~3–10 µs</td><td>~80</td></tr>\n\
         <tr><td>Live</td><td>Slider-crank, RRRP</td><td>Same — closed form exists</td><td>~3–10 µs</td><td>~80</td></tr>\n\
         <tr><td>Live</td><td>Multi-loop / non-closed-form</td><td>Damped Newton (Levenberg-Marquardt) with bounded iters</td><td>~30–100 µs</td><td>~200</td></tr>\n\
         <tr><td>Live</td><td>Anything with workspace bounds you can pre-compute</td><td>Lookup table on \\(u(g)\\) over the working range, cubic interpolate</td><td>~2 µs</td><td>~60</td></tr>\n\
         </table>\n",
    );

    // ── §7.1.1 Offline + interpolation ────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.1.1 Offline-planned trajectories (the dominant case)</h4>\n",
    );
    html.push_str(
        "<p>This is the architecture that fits most controlled-mechanism \
         applications: pick-and-place, prescribed paths, repetitive cycles. \
         The simulator already produces what you need.</p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li>At design time, run <code>compute_trajectory</code> in the simulator with your desired \
         observable, profile, and sample count.</li>\n\
         <li>Export via <b>File → Export firmware (JSON)</b> — emits \\((t_k, h_k, u_k, \\dot u_k, \\ddot u_k, F_{\\text{act},k}, \\text{status}_k)\\) per sample plus the full pose \\((x_i, y_i, \\theta_i)\\) for each body.</li>\n\
         <li>At firmware build time, parse the JSON into a packed C array and bake into ROM (or load from flash at boot).</li>\n\
         <li>At runtime, each control cycle: binary-search for the bracketing samples \\((t_k, t_{k+1})\\), then linearly or cubically interpolate.</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p>Use linear interpolation when only \\(u_k\\) is tabulated; <b>cubic Hermite</b> \
         when \\(u_k\\) and \\(\\dot u_k\\) are both tabulated (the simulator emits both, so you \
         get C¹-continuous output for free). For trajectories at 200–500 \
         samples per second of motion, cubic-Hermite-interpolated tables are \
         indistinguishable from re-running the original Newton solver.</p>\n",
    );
    html.push_str(
        "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
         font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
         overflow-x: auto;'>\
// Cubic Hermite interpolation: needs both u and u_dot at each sample.
float traj_lookup_u(float t) {
    int k = bsearch_bracket(traj_t, traj_n, t);   // O(log n)
    float t0 = traj_t[k], t1 = traj_t[k+1];
    float dt = t1 - t0;
    float s = (t - t0) / dt;                        // [0, 1]
    float h00 = (1 + 2*s) * (1 - s)*(1 - s);
    float h10 = s * (1 - s)*(1 - s);
    float h01 = s*s * (3 - 2*s);
    float h11 = s*s * (s - 1);
    return h00 * traj_u[k]
         + h10 * dt * traj_u_dot[k]
         + h01 * traj_u[k+1]
         + h11 * dt * traj_u_dot[k+1];
}
</pre>\n",
    );

    // ── §7.1.2 Closed-form on the MCU ─────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.1.2 Live IK on a 4-bar — use §4h, not §6</h4>\n",
    );
    html.push_str(
        "<p>For live commands (operator joystick, sensor-driven targets), \
         the closed-form forward kinematics in §4h is <b>10–100× faster</b> \
         than the simulator's matrix Newton. Wrap §4h's \\(\\theta_2 \\to \\theta_3, \\theta_4\\) \
         map in a 1-D Newton on \\(u\\):</p>\n",
    );
    html.push_str(
        "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
         font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
         overflow-x: auto;'>\
// 1-D Newton on closed-form forward map (Freudenstein, §4h).
// Find theta_2 such that g(theta_2) = h_target.
float inverse_g_to_theta2(float h_target, float theta2_warm) {
    float u = theta2_warm;
    for (int i = 0; i &lt; 4; i++) {
        forward_kin_t fk = forward_kinematics(u);   // §4h closed-form
        float g_now = compute_observable(&fk);       // your ControlTarget
        float dg_du = compute_dg_dtheta2(&fk);       // §6.4 row + chain rule
        float r = g_now - h_target;
        if (fabsf(r) &lt; 1e-8f) break;
        u -= r / dg_du;                              // Newton step
    }
    return u;
}
</pre>\n",
    );
    html.push_str(
        "<p>Cost per call: ~3–10 µs on Cortex-M7. Replaces ~10–50 µs of \
         matrix Newton. Big win at 1 kHz control rates.</p>\n",
    );

    // ── §7.1.3 Damped Newton (Levenberg-Marquardt) ────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.1.3 Live IK on non-4-bar — use damped Newton (Levenberg-Marquardt)</h4>\n",
    );
    html.push_str(
        "<p>For multi-loop or non-trivially-closed-form mechanisms, plain \
         Newton on \\(\\Phi_q\\) breaks at singularities (where \\(\\Phi_q\\) \
         loses rank and \\(r'(u) \\to 0\\)). The robotics-standard fix is \
         <b>Levenberg-Marquardt damping</b>:</p>\n",
    );
    html.push_str(
        "\\[ (\\Phi_q^T \\Phi_q + \\lambda I)\\,\\Delta q = -\\Phi_q^T\\,\\Phi \\]\n",
    );
    html.push_str(
        "<p>The damping factor \\(\\lambda\\) tunes between Newton (\\(\\lambda = 0\\), \
         quadratic convergence in well-conditioned regions) and gradient \
         descent (\\(\\lambda \\to \\infty\\), linear but always-stable). A \
         common heuristic: start with \\(\\lambda \\approx 10^{-3} \\|\\Phi_q\\|_\\infty^2\\), \
         decrease by 10× when residual decreases, increase by 10× when it \
         doesn't. <b>Bound the iteration count</b> (e.g. 8 max) so a missed \
         convergence doesn't blow your control deadline.</p>\n",
    );
    html.push_str(
        "<p>The simulator doesn't currently emit LM-style code — it uses \
         pure Newton with a bisection fallback because design-time runs \
         don't have a deadline. For runtime use you'd either lift the \
         simulator's solver loop and add the \\(\\lambda I\\) regularisation, \
         or hand-write an LM kernel using the §3 \\(\\Phi_q\\) formulas.</p>\n",
    );

    // ── §7.2 State-estimation deployment ───────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.2 State-estimation deployment</h3>\n",
    );
    html.push_str(
        "<p>You have a sensor (or two). You want a clean estimate of the \
         mechanism's state — the variable your control loop tracks. Three \
         options, increasing in complexity:</p>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th>Method</th><th>When to use</th><th>Cost</th><th>LOC (C)</th></tr>\n\
         <tr><td><b>(a) Raw sensor + finite-difference</b></td>\
         <td>Single high-precision sensor, no need for smoothed velocity, single-cycle latency OK</td>\
         <td>~0 µs</td>\
         <td>~10</td></tr>\n\
         <tr><td><b>(b) 1-D Kalman / complementary filter</b></td>\
         <td>Single noisy sensor + need smoothed velocity, OR 2 sensors with very different bandwidths (e.g. IMU + encoder)</td>\
         <td>~1–3 µs</td>\
         <td>~50</td></tr>\n\
         <tr><td><b>(c) EKF (§4l)</b></td>\
         <td>2 sensors with characterisable noise, want optimal blending, want fault detection via innovation gating</td>\
         <td>~5–15 µs</td>\
         <td>~150</td></tr>\n\
         </table>\n",
    );

    // ── §7.2.1 Choosing between them ──────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.2.1 Picking the right filter for your sensor stack</h4>\n",
    );
    html.push_str(
        "<p><b>Raw / FD</b>: \\(\\hat\\theta_k = \\theta_{\\text{enc},k}\\), \
         \\(\\hat{\\dot\\theta}_k = (\\theta_{\\text{enc},k} - \\theta_{\\text{enc},k-1})/\\Delta t\\). \
         Add a 1-pole low-pass to the velocity estimate (\\(\\alpha = 0.95\\)) \
         to control the differentiation noise. Adequate when the encoder noise \
         is at least 4× smaller than the position resolution your control \
         loop needs.</p>\n",
    );
    html.push_str(
        "<p><b>Complementary filter</b> (best when sensors have different \
         bandwidths): blend a high-bandwidth-noisy sensor (e.g. tachometer, \
         IMU) with a low-bandwidth-clean sensor (e.g. absolute encoder):</p>\n",
    );
    html.push_str(
        "\\[ \\hat\\theta_{k+1} = \\alpha\\,(\\hat\\theta_k + \\hat{\\dot\\theta}_k\\,\\Delta t) + (1 - \\alpha)\\,\\theta_{\\text{enc},k} \\]\n",
    );
    html.push_str(
        "<p>Tunable \\(\\alpha \\in [0.95, 0.99]\\). One multiply-add per cycle. \
         Cheaper than EKF; doesn't give you optimal weighting or fault \
         detection but works fine when your two sensors aren't directly \
         redundant.</p>\n",
    );
    html.push_str(
        "<p><b>EKF (§4l)</b>: optimal Bayesian blend when you have two \
         sensors that both measure the SAME state through different \
         mappings (encoder reads \\(\\theta_2\\), actuator-position sensor reads \
         \\(L(\\theta_2)\\) per §4i). The Kalman gain automatically weights \
         each by its noise \\(\\sigma\\); innovation gating gives fault \
         detection for free.</p>\n",
    );

    // ── §7.2.2 EKF tuning notes ───────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.2.2 EKF tuning (Q and R)</h4>\n",
    );
    html.push_str(
        "<p><b>Initial covariance</b> \\(P_{0|0}\\): set diagonal to \
         \\(\\sigma_{\\theta,0}^2 \\approx 1\\,\\text{rad}^2\\) (effectively \"I have no \
         idea\"); the filter converges to a good estimate within ~10 \
         measurement updates, after which \\(P\\) drops to its steady-state \
         value.</p>\n",
    );
    html.push_str(
        "<p><b>Process noise</b> \\(Q\\): tune empirically. Start with \
         \\(\\sigma_{\\dot\\theta} \\approx\\) 10% of expected angular velocity. \
         Symptom of \\(Q\\) too small: filter lags input changes (\"sluggish\"). \
         Symptom of \\(Q\\) too large: filter tracks measurement noise \
         (\"jumpy\"). The right value falls out of running the filter on a \
         couple cycles of motion data and inspecting the innovation \
         sequence.</p>\n",
    );
    html.push_str(&format!(
        "<p><b>Measurement noise</b> \\(R\\): from your sensor data sheets, \
         not tuned. {}</p>\n",
        if sensor_config.encoder_joint.is_some()
            || sensor_config.actuator_position_enabled
        {
            format!(
                "Currently configured: encoder σ = {:.4} rad ({:.2} mrad), \
                 actuator σ = {:.6} m ({:.1} µm). These propagate directly into \
                 the §4l EKF as the diagonal of R.",
                sensor_config.encoder_noise_std,
                sensor_config.encoder_noise_std * 1000.0,
                sensor_config.actuator_noise_std,
                sensor_config.actuator_noise_std * 1e6,
            )
        } else {
            "No sensors configured in this report; configure them in the \
             Sensors panel to see numerical R values here."
                .to_string()
        },
    ));
    html.push_str(
        "<p><b>Innovation gating</b> for fault detection: at each update, \
         compute \\(s = y - h(\\hat x_{k+1|k})\\). If \\(|s| > 5\\sqrt{S}\\) for \
         several cycles in a row, the sensor is misbehaving (broken, drifted, \
         or your model is wrong). Skip the update on flagged samples and \
         log; if the fault persists, switch to a degraded single-sensor \
         mode.</p>\n",
    );

    // ── §7.3 Putting it together ──────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.3 The full embedded control loop</h3>\n",
    );
    html.push_str(
        "<p>The pieces above stack together into a typical 1 kHz control \
         interrupt. <b>Total budget per cycle: ~50–100 µs on a Cortex-M7</b>; \
         most of that is the EKF + table lookup + PID, not the kinematics.</p>\n",
    );
    html.push_str(
        "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
         font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
         overflow-x: auto;'>\
// Runs every 1 ms (1 kHz). Total budget ~100 µs; observed ~30 µs.
void control_isr(void) {
    // ── 1. Sample sensors (~2 µs) ───────────────────────────────
    float theta_enc = (read_encoder() - theta_zero) * RAD_PER_COUNT;
    float L_act     = (read_actuator_pos() - L_zero) * M_PER_COUNT;

    // ── 2. State estimation: EKF predict + update (~10 µs, §4l) ─
    ekf_predict(DT);
    float y[2] = { theta_enc, L_act };
    ekf_update(y);
    float theta2_hat   = ekf_state.x[0];
    float omega2_hat   = ekf_state.x[1];

    // ── 3. Lookup desired trajectory (~2 µs, §7.1.1) ────────────
    float t_now = (float)tick_count * DT;
    float u_target     = traj_lookup_u(t_now);
    float u_dot_target = traj_lookup_u_dot(t_now);

    // ── 4. PID + feed-forward (~3 µs) ───────────────────────────
    float u_err = u_target - theta2_hat;
    float u_cmd = pid_compute(u_err)
                + KFF_VEL  * u_dot_target
                + KFF_INERTIA * compute_alpha2_target(t_now);

    // ── 5. Safety checks before commit ──────────────────────────
    if (fabsf(u_cmd) &gt; U_CMD_LIMIT) u_cmd = copysignf(U_CMD_LIMIT, u_cmd);
    if (transmission_angle_too_low(theta2_hat)) {
        // Near a dead point; skip closed-loop trim, ride feed-forward only.
        u_cmd = KFF_VEL * u_dot_target;
    }

    // ── 6. Commit (~1 µs) ───────────────────────────────────────
    write_motor_command(u_cmd);
    tick_count++;
}
</pre>\n",
    );

    // ── §7.4 Gap list ─────────────────────────────────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.4 What the simulator doesn't yet generate</h3>\n",
    );
    html.push_str(
        "<p>For one-click \"this report → deployable C code\" the simulator \
         is currently ~80% of the way there. What's missing:</p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li><b>C-code generator for §4h closed-form FK</b> — emit a \
         <code>forward_kinematics(theta2)</code> function with this \
         mechanism's link lengths baked in as <code>const float</code> literals. \
         The §4k embedded-control recipe shows the constant block; a generator \
         that emits the full module is ~100 LOC.</li>\n\
         <li><b>C-code generator for §4l EKF</b> — emit \
         <code>ekf_predict</code> / <code>ekf_update</code> with the configured Q, R, \
         and measurement Jacobian H. ~150 LOC of generated C.</li>\n\
         <li><b>Trajectory interpolation helper</b> — packaged as a \
         standalone module (binary search + cubic Hermite) that consumes the \
         firmware JSON exported from <code>compute_trajectory</code>. ~50 \
         LOC.</li>\n\
         <li><b>Workspace-bounds enforcement</b> — clamp commanded \\(g\\) at \
         the reachable workspace boundary (the simulator has the §1 \
         workspace-probe table; export it).</li>\n\
         <li><b>Singularity-aware degraded mode</b> — when the simulator's \
         <code>compute_trajectory</code> reports <em>R/S/B/N</em> failures along \
         a trajectory, generate runtime checks that switch to feed-forward-\
         only control through those samples (rather than tripping a fault).</li>\n\
         <li><b>CMake / build scaffold</b> for an STM32CubeIDE / PlatformIO / \
         Arduino-style project layout.</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p>Each item is 50–200 lines of generated code; total deployment \
         pipeline ~500–1000 LOC of C if you implement everything. The \
         report's pseudocode (§4k, §4l, §7.1.1, §7.3) is the seed; the \
         missing piece is mechanical translation to real C source files \
         with this mechanism's specific constants. Until that ships, the \
         report is the spec and you do the hand-translation — which for a \
         one-off mechanism is ~half a day's work.</p>\n",
    );

    // ── §7.5 Sensor characterization & calibration recipes ────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.5 Sensor characterization &amp; calibration recipes</h3>\n",
    );
    html.push_str(
        "<p>Before the EKF, the sensors. The §4l filter assumes the \
         covariance \\(R\\) you give it accurately describes reality; if your \
         σ is wrong by 10× or your zero-offset is off by 1 mrad, the filter \
         will be either sluggish, jumpy, or just biased — you'll spend a \
         week chasing it as a tuning problem when it's actually a \
         characterisation problem. This subsection is the pre-flight \
         checklist a firmware author should run through before the filter \
         touches real data.</p>\n",
    );

    // ── §7.5.1 Measuring sigma ───────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.5.1 Measuring \\(\\sigma_{\\mathrm{enc}}\\) and \\(\\sigma_{\\mathrm{act}}\\)</h4>\n",
    );
    html.push_str(
        "<p>Sensor data sheets give a rough \\(\\sigma\\) but it's almost \
         always optimistic — they assume textbook conditions and don't \
         include effects from your wiring, mounting, EMI, or temperature. \
         <b>Measure your own σ at install time, then again every 6 months.</b></p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li><b>Lock the mechanism in a fixed pose.</b> Use a hard stop, \
         a dowel pin, or just clamp it. The actual value of θ doesn't \
         matter — what matters is that it doesn't move during the recording.</li>\n\
         <li><b>Record sensor readings at full control rate</b> for 30+ \
         seconds. At 1 kHz that's 30 000 samples per sensor. Save the raw \
         stream — don't filter, don't downsample.</li>\n\
         <li><b>Subtract the sample mean</b> \\(\\hat\\mu = \\tfrac{1}{N}\\sum_k y_k\\) \
         to get a zero-mean noise sequence \\(n_k = y_k - \\hat\\mu\\).</li>\n\
         <li><b>Compute the variance</b> \\(\\hat\\sigma^2 = \\tfrac{1}{N-1}\\sum_k n_k^2\\). \
         Take the square root — that is your characterised σ, and it goes \
         straight into the EKF's R diagonal.</li>\n\
         <li><b>White-noise sanity check.</b> Compute the lag-1 \
         autocorrelation \\(\\hat\\rho_1 = \\sum_k n_k n_{k-1} / \\sum_k n_k^2\\). \
         For an EKF-friendly white-noise sensor, \\(|\\hat\\rho_1| < 0.1\\). \
         If higher, the sensor has correlated noise (drift, mechanical \
         resonance, sampling jitter) and the EKF's white-noise assumption \
         is partly violated — the filter will still run but R is no \
         longer a complete description of the error.</li>\n\
         <li><b>Allan variance for longer runs</b> (optional but worth \
         doing once per design). Plot \\(\\sigma_A^2(\\tau)\\) over averaging \
         windows \\(\\tau\\) from 1 sample to N/2 samples on log-log axes. \
         The slope identifies the noise type: \\(-1/2\\) is pure white \
         noise, \\(0\\) is bias instability, \\(+1/2\\) is random walk. For \
         a well-mounted encoder you should see white noise dominate out \
         to \\(\\tau \\approx\\) 100 ms, then bias instability flatten the \
         curve. Random walk is a red flag — it means the sensor is drifting \
         and you'll need either an augmented bias state in the EKF or \
         frequent re-zeroing.</li>\n\
         <li><b>Repeat at 3–5 poses</b> across the workspace. Some sensors \
         have pose-dependent noise (e.g. magnetic encoders near steel, \
         linear pots near the end of travel). If σ varies by more than 2× \
         across poses, treat the worst case as your design σ for R, or \
         model R as pose-dependent.</li>\n\
         </ol>\n",
    );
    html.push_str("<p><b>Sanity-check against this report.</b> ");
    if sensor_config.encoder_joint.is_some()
        || sensor_config.actuator_position_enabled
    {
        html.push_str(&format!(
            "Configured here: encoder σ = {:.4} rad ({:.2} mrad), actuator \
             σ = {:.6} m ({:.1} µm). After your bench characterisation, the \
             values you measure should be within ~50% of these. If they're \
             10× higher, your simulator-side R is fictionally clean and \
             the design-time analysis (innovation thresholds, gain scheduling, \
             dead-band sizing) was solved against the wrong problem — \
             update the simulator config with the measured numbers and \
             re-run all the design-time outputs in this report.",
            sensor_config.encoder_noise_std,
            sensor_config.encoder_noise_std * 1000.0,
            sensor_config.actuator_noise_std,
            sensor_config.actuator_noise_std * 1e6,
        ));
    } else {
        html.push_str(
            "No sensor σ configured in this report. Configure them in the \
             Sensors panel and re-export to see suggested R diagonal values \
             and innovation-gating thresholds here.",
        );
    }
    html.push_str("</p>\n");

    // ── §7.5.2 Calibration ───────────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.5.2 Zero offset, scale factor, polarity</h4>\n",
    );
    html.push_str(
        "<p>Each sensor channel needs three calibration constants. Get any \
         of them wrong and your EKF will produce confidently wrong \
         estimates — the filter has no way to detect a bias in its own \
         calibration.</p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li><b>Zero offset</b> (\\(y_{\\text{zero}}\\)): drive the mechanism \
         to a mechanical reference — a hard stop, a precision dowel pin, \
         or a known calibration fixture — and record the raw count. All \
         future reads subtract this. For an absolute encoder bolted to \
         the joint shaft, the offset is set by the mechanical coupling and \
         is stable across power cycles. For incremental encoders, the \
         offset is set by your homing routine on power-up; persist it to \
         non-volatile storage so a single homing pass survives reboots.</li>\n\
         <li><b>Scale factor</b> (counts → SI units): the data-sheet \
         nominal gets you within ~1%, but for true precision use a \
         <b>two-point measurement</b>. Drive to two known poses (a \
         precision angle gauge, two hard stops at calibrated angles, or \
         two pose locations marked against a master fixture); record raw \
         counts at each; then \
         \\(\\text{scale} = (y_2^{\\text{SI}} - y_1^{\\text{SI}}) / (c_2 - c_1)\\). \
         This calibrates out manufacturing tolerance, gear-ratio mismatch, \
         and any non-unity coupler imperfection in one step.</li>\n\
         <li><b>Polarity</b>: jog the mechanism in one direction by a \
         small known amount and confirm the calibrated reading moves in \
         the expected direction (positive Δθ should produce increasing \
         \\(\\theta_{\\text{enc}}\\)). A flipped polarity in firmware will \
         cause the EKF's measurement Jacobian \\(H\\) to point the wrong \
         way and the filter will diverge in 100 ms. Cheap to check, \
         very expensive to debug downstream.</li>\n\
         <li><b>Gear ratio</b> (motor-mounted encoders): if the encoder \
         is on the motor shaft and the joint sees a reduction \\(N\\), the \
         joint angle is \\(\\theta_{\\text{joint}} = \\theta_{\\text{enc}} / N\\). \
         Verify by counting motor revolutions over a full known joint \
         sweep — don't rely on the gearbox label, especially for \
         compound or harmonic-drive trains.</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p><b>The §4i cross-check.</b> Once both sensors are individually \
         calibrated, drive through the workspace and at every cycle log \
         the residual \\(\\Delta_k = L_{\\text{act,meas}} - L(\\theta_{\\text{enc,meas}})\\) \
         using the §4i closed-form geometry. The residual should be \
         zero-mean with a standard deviation around \
         \\(\\sqrt{\\sigma_{\\text{act}}^2 + (\\partial L / \\partial \\theta_2)^2 \\sigma_{\\text{enc}}^2}\\). \
         A non-zero mean exposes a scaling or zero-offset error in one of \
         the two sensors. A growing or position-dependent residual reveals \
         a model error — wrong link length, wrong mounting angle, or a \
         loose joint with backlash. This single check catches the bulk of \
         install-time mistakes before they become EKF tuning headaches.</p>\n",
    );

    // ── §7.5.3 Cold-start ────────────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.5.3 Cold-start initialisation and assembly mode</h4>\n",
    );
    html.push_str(
        "<p>The first time the EKF runs after power-up or a watchdog reset, \
         it has no prior — \\(\\hat x_{0|0}\\) is whatever you initialised \
         it to. Two things must happen before normal operation can resume.</p>\n",
    );
    html.push_str(
        "<p><b>1. Set \\(P_{0|0}\\) large.</b> Diagonal \\(\\sigma_{\\theta,0}^2 \
         \\approx 1\\,\\text{rad}^2\\), \\(\\sigma_{\\dot\\theta,0}^2 \\approx \
         100\\,(\\text{rad/s})^2\\) is the standard \"I have no idea\" prior. \
         The first few measurement updates collapse \\(P\\) to its \
         steady-state value within ~10 cycles. Don't initialise \
         \\(P_{0|0}\\) small — that tells the filter you trust the (zero) \
         initial state, and it will reject the early measurements until \
         \\(P\\) grows back, costing seconds of bad output.</p>\n",
    );
    html.push_str(
        "<p><b>2. Resolve assembly mode.</b> §4h has TWO roots for \
         \\((\\theta_3, \\theta_4)\\) given \\(\\theta_2\\) — the open and \
         crossed configurations of the 4-bar. The mechanism is mechanically \
         pinned to one of them at assembly (you can't switch without \
         disassembly). The EKF and the §4h closed-form FK both need to \
         know which.</p>\n",
    );
    html.push_str(
        "<p>Procedure: at install time, with the mechanism powered up but \
         the motor disabled, read both \\(\\theta_{\\text{enc}}\\) and \
         \\(L_{\\text{act}}\\). Compute both §4h roots; for each, evaluate \
         §4i to get the predicted \\(L\\); pick the root whose predicted \
         \\(L\\) matches the measured \\(L_{\\text{act}}\\) within \
         \\(3\\sigma_{\\text{act}}\\). Save this choice to non-volatile \
         storage as a single bit. Read it on every boot. <b>Never recompute \
         it at runtime</b> — between the time you read \\(L_{\\text{act}}\\) \
         and the time you write the motor command, you've already run \
         several control cycles, and a branch-flip mid-trajectory is not \
         recoverable.</p>\n",
    );
    html.push_str(
        "<p><b>3. Recovery from unexpected reset.</b> If the EKF is reset \
         mid-operation (watchdog, brown-out, intentional reset on \
         out-of-range innovations), don't restart from zero. Read both \
         sensors immediately, then run a single EKF update with \
         \\(P_{0|0}\\) huge; the cached assembly bit picks the right §4h \
         root, and the joint of \\(\\theta_{\\text{enc}}\\) and \
         \\(L_{\\text{act}}\\) gives you a strongly over-determined \
         initial state. After 1 update cycle, \\(P\\) is approximately \
         \\(R\\) on the diagonal — back to nominal precision in 1 ms.</p>\n",
    );

    // ── §7.6 EKF tuning & validation in production ────────────────────────
    html.push_str(
        "<h3 style='margin-left: 12px;'>7.6 EKF tuning &amp; validation in production</h3>\n",
    );
    html.push_str(
        "<p>An EKF that compiles and runs without crashing is not the same \
         as an EKF that produces optimal estimates. A subtly mistuned \
         filter will track measurements but with wrong uncertainty \
         estimates, wrong gains, and wrong fault-detection thresholds. \
         The standard diagnostic — innovation analysis — is cheap to log \
         and tells you whether your filter is healthy. §7.2.2 covered \
         the tuning principles; this subsection is the production \
         playbook for actually doing it.</p>\n",
    );

    // ── §7.6.1 Innovation tuning ─────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.6.1 Innovation-based tuning workflow</h4>\n",
    );
    html.push_str(
        "<p>The innovation \\(\\nu_k = y_k - h(\\hat x_{k|k-1})\\) is the \
         measurement minus what the filter predicted. <b>For a correctly \
         tuned EKF, the innovation sequence is zero-mean white noise with \
         covariance \\(S_k = H_k P_{k|k-1} H_k^T + R\\).</b> This is a \
         non-trivial property — it requires R, Q, the dynamics model, and \
         the measurement model to all be approximately correct. \
         Violations leave fingerprints in the innovation statistics.</p>\n",
    );
    html.push_str(
        "<table>\n\
         <tr><th>Diagnostic</th><th>What it means</th><th>Fix</th></tr>\n\
         <tr><td>\\(\\overline\\nu \\ne 0\\) (non-zero sample mean)</td>\
         <td>Systematic bias: zero-offset wrong, or measurement model \\(h(x)\\) wrong</td>\
         <td>Re-run §7.5.2 zero-offset cal; cross-check the §4i geometry constants</td></tr>\n\
         <tr><td>\\(\\overline{\\nu^2} \\gg \\overline{S}\\) (innovations bigger than predicted)</td>\
         <td>Filter overconfident — Q or R too small</td>\
         <td>Increase Q (process noise) until the consistency ratio \\(\\nu^2/S\\) approaches 1; only increase R above measured σ² as a last resort</td></tr>\n\
         <tr><td>\\(\\overline{\\nu^2} \\ll \\overline{S}\\) (innovations smaller than predicted)</td>\
         <td>Filter underconfident — Q or R too large</td>\
         <td>Decrease Q first; do NOT decrease R below your characterised σ²</td></tr>\n\
         <tr><td>\\(\\hat\\rho_1(\\nu) > 0.1\\) (autocorrelated innovations)</td>\
         <td>Dynamics model wrong — typically a constant-velocity assumption broken by aggressive trajectory acceleration</td>\
         <td>Increase Q on the velocity state, or augment to a constant-acceleration model with an extra state</td></tr>\n\
         <tr><td>Innovations grow over time</td>\
         <td>Sensor drift, or unmodelled bias</td>\
         <td>Add a bias state to the EKF; recalibrate more frequently; or check thermal drift</td></tr>\n\
         </table>\n",
    );
    html.push_str(
        "<p><b>Recommended bring-up sequence</b> for a new filter:</p>\n\
         <ol>\n\
         <li>Use σ from §7.5.1 directly as the R diagonal. Do not tune R.</li>\n\
         <li>Set Q small initially: \\(Q = \\text{diag}(10^{-6}\\,\\text{rad}^2, \
         10^{-3}\\,(\\text{rad/s})^2)\\). The filter will lag input changes.</li>\n\
         <li>Increase Q until innovations are zero-mean white noise with \
         covariance matching \\(S\\) within ~30%. The right Q is usually \
         within one order of magnitude of the typical commanded \
         acceleration squared, scaled by \\(\\Delta t^2\\).</li>\n\
         <li>Validate against §7.6.2 HIL once the innovations look good. \
         Innovation diagnostics tell you the filter is self-consistent, \
         not that it's accurate.</li>\n\
         </ol>\n",
    );

    // ── §7.6.2 HIL validation ────────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.6.2 Hardware-in-the-loop validation against the simulator</h4>\n",
    );
    html.push_str(
        "<p>Innovation diagnostics tell you whether the filter is \
         self-consistent — not whether it's producing the right answer. \
         To validate absolute accuracy, replay a known ground-truth signal \
         through the deployed firmware and compare the estimate to the \
         truth.</p>\n",
    );
    html.push_str(
        "<ol>\n\
         <li><b>Generate a ground-truth trajectory</b> in the simulator. \
         Pick a representative motion: full sweep of the workspace, plus \
         a stationary hold, plus a pass near a near-singular configuration. \
         <code>compute_trajectory</code> emits \\((t_k, \\theta_{2,k}^{\\text{truth}}, \
         \\dot\\theta_{2,k}^{\\text{truth}}, \\ddot\\theta_{2,k}^{\\text{truth}})\\).</li>\n\
         <li><b>Synthesise sensor measurements</b> by sampling the §4i \
         forward map and adding noise:\
         \\[ \\theta_{\\text{enc},k} = \\theta_{2,k}^{\\text{truth}} + n_{\\theta,k},\\quad n_{\\theta,k} \\sim \\mathcal{N}(0, \\sigma_{\\text{enc}}^2) \\]\
         \\[ L_{\\text{act},k} = L\\!\\left(\\theta_{2,k}^{\\text{truth}}\\right) + n_{L,k},\\quad n_{L,k} \\sim \\mathcal{N}(0, \\sigma_{\\text{act}}^2) \\]\
         This is exactly what the deployed sensors should produce when \
         the mechanism actually runs the same trajectory.</li>\n\
         <li><b>Pipe the synthesised stream</b> into your firmware via \
         UART, SPI, or a debug-mode replay buffer. The firmware should \
         not know it's running on synthesised data — same code path, \
         same ISR, same EKF as production.</li>\n\
         <li><b>Compare the deployed \\(\\hat\\theta_{2,k}\\)</b> against \
         the simulator's \\(\\theta_{2,k}^{\\text{truth}}\\). Compute RMS \
         error after the initial ~50-cycle convergence transient.</li>\n\
         <li><b>Sanity-check the magnitude.</b> The §4l Riccati equation \
         gives the steady-state \\(\\sigma_\\theta\\) you should expect; if \
         the RMS error is within ~50% of that, the filter is working as \
         designed. If 5×–100× higher, you have a bug — usually a \
         sign-flipped Jacobian, a wrong link constant, or a stale \
         assembly-mode bit from §7.5.3.</li>\n\
         </ol>\n",
    );
    html.push_str(
        "<p>Recommended scope: 60+ seconds covering the full workspace, \
         a near-dead-point pass, and a stationary hold. Run nightly in CI \
         if you have a hardware fixture; once per release otherwise. The \
         simulator side of HIL is already in this codebase — \
         <code>compute_trajectory</code> plus a Gaussian-noise wrapper is \
         ~20 lines of glue.</p>\n",
    );

    // ── §7.6.3 Log format ────────────────────────────────────────────────
    html.push_str(
        "<h4 style='margin-left: 24px;'>7.6.3 Production log format</h4>\n",
    );
    html.push_str(
        "<p>The diagnostics in §7.6.1 require logged data. Don't try to \
         compute statistics live on the MCU — log to flash or stream out \
         and post-process. The format below is a portable binary record \
         that fits 1 kHz logging on any modern flash chip:</p>\n",
    );
    html.push_str(
        "<pre style='background: #f0f2f5; padding: 12px; border-left: 3px solid #0f3460; \
         font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; \
         overflow-x: auto;'>\
// Per-cycle record (little-endian, 36 bytes). At 1 kHz: 36 KB/s, ~130 MB/hour.
typedef struct __attribute__((packed)) {
    uint32_t  t_us;             // monotonic timestamp, microseconds
    float     theta_enc_meas;   // raw encoder, calibrated, rad   (NaN if not configured)
    float     L_act_meas;       // raw actuator pos, calibrated, m (NaN if not configured)
    float     theta2_hat;       // EKF posterior estimate, rad
    float     omega2_hat;       // EKF posterior velocity, rad/s
    float     P_diag_theta;     // posterior covariance diagonal, rad^2
    float     P_diag_omega;     // posterior covariance diagonal, (rad/s)^2
    float     innov_theta;      // y_enc - h_theta(x_pred), rad
    float     innov_L;          // y_L   - h_L(x_pred),     m
    uint8_t   status;           // bit 0: enc gated, 1: act gated, 2: predict-only, 3-7: reserved
    uint8_t   _pad[3];          // align to 4 bytes
} ekf_log_record_t;

// File header (written once at start of log file)
typedef struct __attribute__((packed)) {
    char      magic[4];         // \"EKFL\"
    uint16_t  version;          // 1
    uint16_t  record_size;      // sizeof(ekf_log_record_t) = 36
    float     dt_s;             // control period, seconds
    float     sigma_enc;        // configured R diagonal sqrt, rad
    float     sigma_act;        // configured R diagonal sqrt, m
    float     Q_diag[2];        // configured process-noise diagonal
    uint64_t  mechanism_hash;   // SHA-1 prefix of the MechanismJson at design time
    uint32_t  build_id;         // firmware git sha prefix
} ekf_log_header_t;
</pre>\n",
    );
    html.push_str(
        "<p><b>Post-processing pipeline.</b> A 5-minute Python script \
         reads the binary log and produces (a) histograms of \\(\\nu\\) \
         per channel, (b) the lag-1 autocorrelation \\(\\hat\\rho_1\\), \
         and (c) the running mean of the consistency ratio \\(\\nu^2 / S\\), \
         which should sit near 1.0 for a well-tuned filter. \
         <code>numpy</code> + <code>matplotlib</code> is sufficient; no \
         heavy tooling needed. The mechanism hash and build ID in the \
         header let you correlate log files with specific \
         firmware/mechanism configurations across a fleet of devices, \
         which becomes critical the moment you have more than one unit \
         deployed and a regression to localise.</p>\n",
    );
}

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
    sensor_config: &crate::gui::state::SensorConfig,
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

    // ── 4l. State estimation / sensor fusion ───────────────────────────────
    write_state_estimation_section(html, mechanism, q, sensor_config);

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

    // ── 6. Trajectory mode (inverse position control) ─────────────────────
    write_trajectory_inverse_section(html);

    // ── 7. Deploying to hardware ───────────────────────────────────────────
    write_hardware_deployment_section(html, sensor_config);

    // ── 8. Numerical validation ─────────────────────────────────────────────
    //
    // Shows the cascade actually working: residuals at each level should
    // be at floating-point noise. Lets a reader spot-check the solver
    // outputs against an external tool (Mathematica, MATLAB, NumPy, etc.)
    // by giving them concrete numbers to compare to.
    html.push_str("<h3>8. Numerical validation at this pose</h3>\n");
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
    html.push_str("<h3>9. How to verify the math externally</h3>\n");
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
    html.push_str("<h3>10. Further reading</h3>\n");
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

        // Default SensorConfig = no sensors; §5 should still emit something
        // (the open-loop note) so the assertion below has content to match.
        let sensors = crate::gui::state::SensorConfig::default();
        let html = generate_html_report(&mech, &q, &sweep, None, &units, &sensors)
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
        // §6 inverse-kinematics derivation should now be inlined (formerly
        // a one-paragraph reference to the spec doc).
        assert!(
            html.contains("6.1 Position inverse"),
            "§6.1 position-inverse subsection should be inlined"
        );
        assert!(
            html.contains("6.2 Velocity inverse"),
            "§6.2 velocity-inverse subsection should be inlined"
        );
        assert!(
            html.contains("6.3 Acceleration inverse"),
            "§6.3 acceleration-inverse subsection should be inlined"
        );
        assert!(
            html.contains("Per-ControlTarget"),
            "§6.4 per-variant table should appear"
        );
        assert!(
            html.contains("Analytic vs finite-difference"),
            "§6.5 analytic-vs-FD comparison should appear"
        );
        assert!(
            html.contains("\\Phi_q^{-1}\\,\\Phi_u")
                || html.contains("-\\Phi_q^{-1}\\,\\Phi_u"),
            "implicit-function-theorem dq/du derivation should be in LaTeX"
        );
        assert!(
            html.contains("\\nabla_q^2 g"),
            "Hessian-of-observable notation should appear"
        );
        // Five ControlTarget variants must all be named in §6.4.
        assert!(
            html.contains("<b>Angle</b>")
                && html.contains("<b>WorldX</b>")
                && html.contains("<b>WorldY</b>")
                && html.contains("<b>Projection</b>")
                && html.contains("<b>Distance</b>"),
            "all five ControlTarget variants should appear in §6.4 table"
        );
        // §6.0 first-time-reader scaffolding (overview + glossary + plain
        // English algorithm) should be present.
        assert!(
            html.contains("6.0 What this section is about"),
            "§6.0 overview subsection should be inlined for first-time readers"
        );
        assert!(
            html.contains("Concrete example") && html.contains("coupler-tip"),
            "§6.0 should include the motivating concrete example"
        );
        assert!(
            html.contains("Notation glossary") || html.contains("notation glossary"),
            "§6.0 should include the symbol glossary table"
        );
        assert!(
            html.contains("whole algorithm in plain English"),
            "§6.0 should include the plain-English algorithm description"
        );
        // §6.4 should now have the worked-example call-out for WorldX,
        // before the formal per-variant table.
        assert!(
            html.contains("Worked example: WorldX"),
            "§6.4 should walk through one variant in detail before the table"
        );
        // §6.2, §6.3, §6.5, §6.6 should each have a "Why you care" /
        // "Why two implementations" / "Read this if" framing intro that
        // explains the practical use before diving into math.
        assert!(
            html.contains("Why you care"),
            "§6.2 / §6.3 should include a 'Why you care' framing intro"
        );
        // §7 hardware-deployment section — covers IK + state estimation
        // recipes for embedded use. The previous §7-§9 renumbered to §8-§10.
        assert!(
            html.contains("<h3>7. Deploying to hardware</h3>"),
            "should include §7 hardware deployment section"
        );
        assert!(
            html.contains("Design-time vs runtime"),
            "§7.0 should orient the reader on the two-role distinction"
        );
        assert!(
            html.contains("Inverse-kinematics deployment"),
            "§7.1 should cover IK deployment with a decision tree"
        );
        assert!(
            html.contains("State-estimation deployment"),
            "§7.2 should cover state-estimation deployment options"
        );
        assert!(
            html.contains("EKF tuning"),
            "§7.2.2 should cover Q/R tuning for the EKF"
        );
        assert!(
            html.contains("control_isr"),
            "§7.3 should include the full embedded ISR pseudocode"
        );
        assert!(
            html.contains("Levenberg-Marquardt") || html.contains("damped Newton"),
            "§7.1 should cover damped Newton / LM for non-4-bar mechanisms"
        );
        assert!(
            html.contains("Cubic Hermite") || html.contains("cubic Hermite"),
            "§7.1.1 should mention cubic Hermite interpolation for trajectory tables"
        );
        // §7.5 Sensor characterisation & calibration recipes
        assert!(
            html.contains("Sensor characterization"),
            "§7.5 should cover sensor characterisation and calibration"
        );
        assert!(
            html.contains("Allan variance"),
            "§7.5.1 should include the Allan-variance noise-typing technique"
        );
        assert!(
            html.contains("Zero offset")
                && html.contains("Scale factor")
                && html.contains("Polarity"),
            "§7.5.2 should cover the three calibration constants"
        );
        assert!(
            html.contains("assembly mode") || html.contains("Assembly mode"),
            "§7.5.3 should cover the open/crossed assembly-mode determination"
        );
        assert!(
            html.contains("Cold-start") || html.contains("cold-start"),
            "§7.5.3 should cover cold-start initialisation"
        );
        // §7.6 EKF tuning & validation in production
        assert!(
            html.contains("Innovation-based tuning"),
            "§7.6.1 should cover innovation-based tuning"
        );
        assert!(
            html.contains("Hardware-in-the-loop"),
            "§7.6.2 should cover HIL validation against the simulator"
        );
        assert!(
            html.contains("ekf_log_record_t"),
            "§7.6.3 should specify a concrete production log record format"
        );
        // Renumbering: §7→§8, §8→§9, §9→§10. Verify the new numbering landed.
        assert!(
            html.contains("<h3>8. Numerical validation"),
            "old §7 'Numerical validation' should now be §8"
        );
        assert!(
            html.contains("<h3>9. How to verify"),
            "old §8 'External verification' should now be §9"
        );
        assert!(
            html.contains("<h3>10. Further reading"),
            "old §9 'Further reading' should now be §10"
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
        // §4g–§4k: practical control-engineering content for embedded use.
        assert!(
            html.contains("Acceleration analysis"),
            "should include closed-form acceleration analysis"
        );
        assert!(
            html.contains("\\ddot\\theta_4"),
            "acceleration section should derive ÿ_4 in LaTeX"
        );
        assert!(
            html.contains("Closed-form forward kinematics"),
            "should include θ_2 → θ_3, θ_4 forward kinematics"
        );
        assert!(
            html.contains("tangent half-angle") || html.contains("\\arctan(t)"),
            "forward-kinematics section should reference the tangent half-angle solution"
        );
        assert!(
            html.contains("Linear actuator length"),
            "should include linear actuator length section"
        );
        assert!(
            html.contains("Coordinate frame"),
            "should include coordinate-frame conventions"
        );
        assert!(
            html.contains("Embedded control recipe"),
            "should include the practical embedded control recipe"
        );
        assert!(
            html.contains("control_cycle()"),
            "embedded recipe should include pseudocode"
        );
        // §4l State estimation: with default SensorConfig (no sensors),
        // the section appears with the open-loop note rather than the EKF.
        assert!(
            html.contains("State estimation and sensor fusion"),
            "should include state estimation section heading"
        );
        assert!(
            html.contains("no sensors enabled"),
            "default SensorConfig (no sensors) should produce the open-loop note"
        );
        // Plotly integration
        assert!(html.contains("plotly-2.35.2.min.js"), "should include plotly CDN");
        assert!(html.contains("Plotly.newPlot"), "should have at least one plotly chart");
        assert!(html.contains("energy_plot"), "should have energy plot div");
    }

    #[test]
    fn report_state_estimation_section_branches_on_sensor_config() {
        // Verify §4l adapts to the SensorConfig: 0 sensors → open-loop
        // note; 1 sensor → 1-D filter; 2 sensors → 2x2 EKF.
        use crate::gui::samples::{build_sample, SampleMechanism};
        use crate::gui::state::{DisplayUnits, SensorConfig};

        let (mech, q0) = build_sample(SampleMechanism::FourBar);
        let result = crate::solver::kinematics::solve_position(&mech, &q0, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q = if result.converged { result.q } else { q0 };
        let (sweep, _) = crate::gui::sweep::compute_sweep_data(
            &mech, &q, 2.0 * std::f64::consts::PI, 0.0, 9.81, None,
        );
        let units = DisplayUnits::default();

        // Encoder only.
        let mut sensors = SensorConfig::default();
        sensors.encoder_joint = Some("D1".to_string());
        let html = generate_html_report(&mech, &q, &sweep, None, &units, &sensors).unwrap();
        assert!(
            html.contains("State estimation and sensor fusion"),
            "should include the section"
        );
        assert!(
            !html.contains("no sensors enabled"),
            "1-sensor config should not emit the open-loop note"
        );
        assert!(
            html.contains("Encoder on joint <code>D1</code>"),
            "should describe the configured encoder joint"
        );
        assert!(
            html.contains("// 1 sensor (encoder)"),
            "1-sensor pseudocode branch should be selected"
        );

        // Both sensors (canonical FourBar has no LinearActuator, so
        // actuator_position_enabled with no detected actuator triggers
        // the warning branch — verify by switching to a sample with one).
        let (mech2, q02) = build_sample(SampleMechanism::ParallelogramActuator);
        let result2 = crate::solver::kinematics::solve_position(&mech2, &q02, 0.0, 1e-10, 50)
            .expect("solve should succeed");
        let q2 = if result2.converged { result2.q } else { q02 };
        let (sweep2, _) = crate::gui::sweep::compute_sweep_data(
            &mech2,
            &q2,
            2.0 * std::f64::consts::PI,
            0.0,
            9.81,
            None,
        );
        let mut sensors2 = SensorConfig::default();
        sensors2.encoder_joint = Some("D1".to_string());
        sensors2.actuator_position_enabled = true;
        let html2 =
            generate_html_report(&mech2, &q2, &sweep2, None, &units, &sensors2).unwrap();
        assert!(
            html2.contains("// 2 sensors: H is 2x2"),
            "2-sensor pseudocode branch should be selected when both sensors active"
        );
        assert!(
            html2.contains("Extended Kalman Filter"),
            "2-sensor section should call out the EKF"
        );
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

    // ── Date / timezone helpers ───────────────────────────────────────

    #[test]
    fn days_to_ymd_known_dates() {
        // Spot-check several dates against external truth (e.g. `date -ud
        // @<unix>`). All inputs are seconds since 1970-01-01 UTC.
        let cases: &[(i64, i32, u32, u32)] = &[
            (0, 1970, 1, 1),                  // epoch
            (86_399, 1970, 1, 1),             // last second of day 0
            (86_400, 1970, 1, 2),             // first second of day 1
            (951_782_400, 2000, 2, 29),       // leap day in century year (2000)
            (1_577_836_800, 2020, 1, 1),      // 2020-01-01
            (1_614_038_400, 2021, 2, 23),     // arbitrary mid-decade date
            // Future-date coverage is provided by
            // `ymd_to_days_round_trips_through_days_to_ymd` which sweeps
            // 50 dates across ~50 years.
        ];
        for &(secs, ey, em, ed) in cases {
            let days = secs / 86_400;
            let (y, m, d) = days_to_ymd(days);
            assert_eq!(
                (y, m, d),
                (ey, em, ed),
                "secs {} should be {}-{:02}-{:02}, got {}-{:02}-{:02}",
                secs, ey, em, ed, y, m, d,
            );
        }
    }

    #[test]
    fn ymd_to_days_round_trips_through_days_to_ymd() {
        // Inverse identity: ymd_to_days followed by days_to_ymd should
        // recover the same (y, m, d). Pick ~50 dates spanning ~60 years.
        for offset in 0..50 {
            let days = offset * 365 + 1; // some prime-ish stride
            let (y, m, d) = days_to_ymd(days);
            let back = ymd_to_days(y, m, d);
            assert_eq!(
                back, days,
                "round-trip mismatch: {}-{:02}-{:02} -> {} (expected {})",
                y, m, d, back, days
            );
        }
    }

    #[test]
    fn is_us_eastern_dst_known_transitions() {
        // 2026 transitions:
        //   Spring forward: 2026-03-08 (2nd Sunday of March)
        //   Fall back:      2026-11-01 (1st Sunday of November)
        // EST = UTC-5, EDT = UTC-4. Spring 2 AM local = 7 AM UTC.

        // Mid-January: definitely EST.
        assert!(!is_us_eastern_dst(2026, 1, 15, 12));
        // Mid-July: definitely EDT.
        assert!(is_us_eastern_dst(2026, 7, 15, 12));
        // 2026-03-08 06:00 UTC (1 AM EST) — still EST.
        assert!(!is_us_eastern_dst(2026, 3, 8, 6));
        // 2026-03-08 07:00 UTC (3 AM EDT after spring-forward) — now EDT.
        assert!(is_us_eastern_dst(2026, 3, 8, 7));
        // 2026-03-07 (day before): EST.
        assert!(!is_us_eastern_dst(2026, 3, 7, 23));
        // 2026-03-09 (day after spring-forward): EDT.
        assert!(is_us_eastern_dst(2026, 3, 9, 0));
        // 2026-11-01 05:00 UTC (1 AM EDT) — still EDT.
        assert!(is_us_eastern_dst(2026, 11, 1, 5));
        // 2026-11-01 06:00 UTC (1 AM EST after fall-back) — now EST.
        assert!(!is_us_eastern_dst(2026, 11, 1, 6));
    }

    #[test]
    fn nth_weekday_of_month_known_values() {
        // 2026-03-08 = 2nd Sunday of March (per US DST rules)
        assert_eq!(nth_weekday_of_month(2026, 3, 7, 2), 8);
        // 2026-11-01 = 1st Sunday of November
        assert_eq!(nth_weekday_of_month(2026, 11, 7, 1), 1);
        // 2026-01-05 = 1st Monday of January
        assert_eq!(nth_weekday_of_month(2026, 1, 1, 1), 5);
        // 2024-02-29 was a Thursday → 5th Thursday of Feb 2024 = 29
        assert_eq!(nth_weekday_of_month(2024, 2, 4, 5), 29);
    }

    #[test]
    fn format_unix_timestamp_eastern_summer_uses_edt() {
        // 2026-05-02 17:56:28 UTC == 2026-05-02 13:56:28 EDT (UTC-4 in May)
        // This is the bug case the user reported (showed "2026-05-18
        // 13:56:28 UTC" in the report header before this fix).
        let secs = ymd_to_days(2026, 5, 2) * 86_400 + 17 * 3600 + 56 * 60 + 28;
        let s = format_unix_timestamp_eastern(secs);
        assert_eq!(s, "2026-05-02 13:56:28 EDT");
    }

    #[test]
    fn format_unix_timestamp_eastern_winter_uses_est() {
        // 2026-01-15 18:30:00 UTC == 2026-01-15 13:30:00 EST (UTC-5 in Jan)
        let secs = ymd_to_days(2026, 1, 15) * 86_400 + 18 * 3600 + 30 * 60;
        let s = format_unix_timestamp_eastern(secs);
        assert_eq!(s, "2026-01-15 13:30:00 EST");
    }

    #[test]
    fn format_unix_timestamp_eastern_handles_day_rollover() {
        // 2026-05-03 03:30:00 UTC == 2026-05-02 23:30:00 EDT (rolls back
        // to previous day in local time).
        let secs = ymd_to_days(2026, 5, 3) * 86_400 + 3 * 3600 + 30 * 60;
        let s = format_unix_timestamp_eastern(secs);
        assert_eq!(s, "2026-05-02 23:30:00 EDT");
    }
}
