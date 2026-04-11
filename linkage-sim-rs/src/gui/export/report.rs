//! HTML report generation for mechanism analysis summaries.

#[cfg(feature = "native")]
use crate::core::mechanism::Mechanism;
#[cfg(feature = "native")]
use crate::gui::sweep::SweepData;

#[cfg(feature = "native")]
use super::svg::generate_svg_string;

/// Generate an HTML report summarizing the current mechanism analysis.
///
/// Includes: mechanism diagram (embedded SVG), topology, dimensions, mass
/// properties, Grashof classification, torque/transmission/reaction envelopes,
/// force element summary, and energy data.
#[cfg(feature = "native")]
pub fn generate_html_report(
    mechanism: &Mechanism,
    q: &nalgebra::DVector<f64>,
    sweep: &SweepData,
    grashof: Option<&crate::analysis::grashof::GrashofResult>,
    units: &super::super::state::DisplayUnits,
) -> Result<String, String> {
    use crate::analysis::envelopes::compute_envelope;
    use crate::core::state::GROUND_ID;

    let svg = generate_svg_string(mechanism, q).unwrap_or_else(|_| String::new());

    let mut html = String::with_capacity(16_000);
    html.push_str("<!DOCTYPE html>\n<html><head><meta charset='utf-8'>\n");
    html.push_str("<title>Linkage Mechanism Report</title>\n");
    html.push_str("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>\n");
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

/// Get current timestamp as a formatted string.
#[cfg(feature = "native")]
fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
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

/// Summarize a force element as (type_name, detail_string).
#[cfg(feature = "native")]
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
#[cfg(feature = "native")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "native")]
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
