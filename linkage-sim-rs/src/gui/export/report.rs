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
    // for \(...\) (inline) and \[...\] (display) delimiters and converts
    // them in-place. Loaded async via `defer`; the auto-render call fires
    // on the script's onload event.
    html.push_str(
        "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'>\n",
    );
    html.push_str(
        "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js'></script>\n",
    );
    html.push_str(
        "<script defer src='https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' \
         onload='renderMathInElement(document.body, { delimiters: [\
         {left: \"\\\\[\", right: \"\\\\]\", display: true}, \
         {left: \"\\\\(\", right: \"\\\\)\", display: false}, \
         ]});'></script>\n",
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
    let body_q_strs: Vec<String> = moving
        .iter()
        .map(|b| {
            let safe = html_escape(b);
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
    html.push_str(
        "<p>The driver row is the only row that depends on \\(t\\).</p>\n",
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
