//! Force elements panel: listing, editing, and per-type detail editors.
//!
//! Also contains modulation helpers and point picker/fields UI.

use eframe::egui;
use meval;
use crate::forces::elements::*;
use super::pending_edits::PendingPropertyEdit;
use crate::gui::state::AppState;

/// Hover text of the LinearActuator force field. Sign convention of the
/// element (`forces::elements::evaluation`): positive = extension.
const ACTUATOR_FORCE_FIELD_TIP: &str = "Constant axial force applied by the actuator in Newtons. Positive = extension (pushes the attachment points apart), negative = retraction (pulls them together). Set to 0 to solve for the required actuator force from statics.";

/// Retain numeric drafts across frames and publish only a completed edit.
/// Uses the weight editor's release/focus-loss and Escape convention.
fn committed_force_number(
    ui: &mut egui::Ui,
    key: &str,
    value: &mut f64,
    hover: &str,
    configure: impl FnOnce(egui::DragValue<'_>) -> egui::DragValue<'_>,
) -> bool {
    let Some(committed) = super::weight_editor::committed_number(ui, key, hover, *value,
        |field| configure(field).clamp_existing_to_range(false)) else { return false };
    *value = committed;
    true
}

/// Draw the force elements section of the property panel.
///
/// Lists all non-gravity force elements with editable parameters in
/// collapsing headers, plus "Add ..." buttons for creating new elements.
/// Reads from the blueprint to avoid borrow conflicts with the mechanism.
pub(super) fn draw_force_elements_panel(
    ui: &mut egui::Ui,
    state: &AppState,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let Some(bp) = &state.blueprint else {
        ui.label("No blueprint loaded.");
        return;
    };

    // List existing force elements (skip Gravity -- toggled via View menu).
    let mut visible_count = 0u32;
    for (bp_idx, force) in bp.forces.iter().enumerate() {
        if matches!(force, ForceElement::Gravity(_)) {
            continue;
        }
        visible_count += 1;

        let header_label = format!("{} #{}", force.type_name(), visible_count);
        egui::CollapsingHeader::new(&header_label)
            .id_salt(format!("force_{}", bp_idx))
            .show(ui, |ui| {
                draw_force_element_details(ui, bp_idx, force, bp, pending);

                if ui.small_button("Remove").on_hover_text("Delete this force element from the mechanism").clicked() {
                    *pending = Some(PendingPropertyEdit::RemoveForce(bp_idx));
                }
            });
    }

    if visible_count == 0 {
        ui.label("No force elements.");
    }

    // Force elements are added via the toolbar ribbon (force_toolbar.rs).
}

/// Draw editable parameter fields for a single force element.
///
/// Completed numeric edits and immediate choices clone the current element
/// with the changed parameter and emit one `UpdateForce` pending edit.
fn draw_force_element_details(
    ui: &mut egui::Ui,
    index: usize,
    force: &ForceElement,
    blueprint: &crate::io::MechanismJson,
    pending: &mut Option<PendingPropertyEdit>,
) {
    match force {
        ForceElement::Gravity(_) => {} // skipped in caller

        ForceElement::LinearSpring(s) => {
            ui.label(format!("Body A: {}  Body B: {}", s.body_a, s.body_b));

            let mut stiffness = s.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if committed_force_number(ui, "stiffness", &mut stiffness, "Spring stiffness (N/m). Force = k \u{d7} (current_length \u{2212} free_length). Higher values produce a stiffer spring that resists extension/compression more strongly.",
                    |field| field.speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N/m"))
                {
                    let mut updated = s.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearSpring(updated),
                    });
                }
            });

            let mut free_len = s.free_length;
            ui.horizontal(|ui| {
                ui.label("L\u{2080}:");
                if committed_force_number(ui, "free_len", &mut free_len, "Free (unstretched) length of the spring in meters. The spring produces zero force when the distance between its attachment points equals this value. Shorter = spring is pre-loaded in tension; longer = pre-loaded in compression.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"))
                {
                    let mut updated = s.clone();
                    updated.free_length = free_len;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearSpring(updated),
                    });
                }
            });

            draw_point_picker(
                ui, "Pt A", &s.body_a, &s.point_a, &s.point_a_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = s.clone();
                    updated.point_a_name = name;
                    updated.point_a = coords;
                    ForceElement::LinearSpring(updated)
                },
                pending,
            );

            draw_point_picker(
                ui, "Pt B", &s.body_b, &s.point_b, &s.point_b_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = s.clone();
                    updated.point_b_name = name;
                    updated.point_b = coords;
                    ForceElement::LinearSpring(updated)
                },
                pending,
            );
        }

        ForceElement::LinearDamper(d) => {
            ui.label(format!("Body A: {}  Body B: {}", d.body_a, d.body_b));

            let mut damping = d.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if committed_force_number(ui, "damping", &mut damping, "Viscous damping coefficient (N\u{b7}s/m). Force = c \u{d7} extension_rate. Opposes relative velocity between the two attachment points. Higher values dissipate energy faster.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}s/m"))
                {
                    let mut updated = d.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearDamper(updated),
                    });
                }
            });

            draw_point_picker(
                ui, "Pt A", &d.body_a, &d.point_a, &d.point_a_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = d.clone();
                    updated.point_a_name = name;
                    updated.point_a = coords;
                    ForceElement::LinearDamper(updated)
                },
                pending,
            );

            draw_point_picker(
                ui, "Pt B", &d.body_b, &d.point_b, &d.point_b_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = d.clone();
                    updated.point_b_name = name;
                    updated.point_b = coords;
                    ForceElement::LinearDamper(updated)
                },
                pending,
            );
        }

        ForceElement::ExternalForce(f) => {
            ui.label(format!("Body: {}", f.body_id));

            let mut fx = f.force[0];
            let mut fy = f.force[1];
            ui.horizontal(|ui| {
                ui.label("Fx:");
                if committed_force_number(ui, "fx", &mut fx, "Horizontal component of the applied force in Newtons (world X-axis). Positive = rightward.",
                    |field| field.speed(0.1).suffix(" N"))
                {
                    let mut updated = f.clone();
                    updated.force[0] = fx;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalForce(updated),
                    });
                }
            });
            ui.horizontal(|ui| {
                ui.label("Fy:");
                if committed_force_number(ui, "fy", &mut fy, "Vertical component of the applied force in Newtons (world Y-axis). Positive = upward.",
                    |field| field.speed(0.1).suffix(" N"))
                {
                    let mut updated = f.clone();
                    updated.force[1] = fy;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalForce(updated),
                    });
                }
            });

            draw_point_picker(
                ui, "Local pt", &f.body_id, &f.local_point, &f.local_point_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = f.clone();
                    updated.local_point_name = name;
                    updated.local_point = coords;
                    ForceElement::ExternalForce(updated)
                },
                pending,
            );

            draw_modulation_fields(ui, index, &f.modulation, |m| {
                let mut updated = f.clone();
                updated.modulation = m;
                ForceElement::ExternalForce(updated)
            }, pending);
        }

        ForceElement::ExternalTorque(t) => {
            ui.label(format!("Body: {}", t.body_id));

            let mut torque = t.torque;
            ui.horizontal(|ui| {
                ui.label("\u{03c4}:");
                if committed_force_number(ui, "torque", &mut torque, "Pure torque applied to this body in N\u{b7}m. Positive = counter-clockwise. This torque acts directly on the body's rotational DOF, not through an attachment point.",
                    |field| field.speed(0.1)
                            .suffix(" N\u{00b7}m"))
                {
                    let mut updated = t.clone();
                    updated.torque = torque;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ExternalTorque(updated),
                    });
                }
            });

            draw_modulation_fields(ui, index, &t.modulation, |m| {
                let mut updated = t.clone();
                updated.modulation = m;
                ForceElement::ExternalTorque(updated)
            }, pending);
        }

        ForceElement::TorsionSpring(s) => {
            ui.label(format!("Body I: {}  Body J: {}", s.body_i, s.body_j));

            let mut stiffness = s.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if committed_force_number(ui, "stiffness", &mut stiffness, "Torsion spring stiffness (N\u{b7}m/rad). Torque = k \u{d7} (relative_angle \u{2212} free_angle). Acts on the relative angle between bodies I and J at their shared joint.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m/rad"))
                {
                    let mut updated = s.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::TorsionSpring(updated),
                    });
                }
            });

            let mut free_angle = s.free_angle;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}\u{2080}:");
                if committed_force_number(ui, "free_angle", &mut free_angle, "Free (unloaded) angle of the torsion spring in radians. The spring produces zero torque when the relative angle between bodies I and J equals this value.",
                    |field| field.speed(0.01)
                            .suffix(" rad"))
                {
                    let mut updated = s.clone();
                    updated.free_angle = free_angle;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::TorsionSpring(updated),
                    });
                }
            });
        }

        ForceElement::RotaryDamper(d) => {
            ui.label(format!("Body I: {}  Body J: {}", d.body_i, d.body_j));

            let mut damping = d.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if committed_force_number(ui, "damping", &mut damping, "Rotary viscous damping coefficient (N\u{b7}m\u{b7}s/rad). Torque = c \u{d7} relative_angular_velocity. Opposes relative rotation between bodies I and J at their shared joint.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"))
                {
                    let mut updated = d.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::RotaryDamper(updated),
                    });
                }
            });
        }

        ForceElement::GasSpring(gs) => {
            ui.label(format!("Body A: {}  Body B: {}", gs.body_a, gs.body_b));

            let mut initial_force = gs.initial_force;
            ui.horizontal(|ui| {
                ui.label("F\u{2080}:");
                if committed_force_number(ui, "initial_force", &mut initial_force, "Force at full extension (fully extended position) in Newtons. This is the minimum force the gas spring produces. Gas springs produce more force as they compress (shorter length = higher pressure = more force).",
                    |field| field.speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N"))
                {
                    let mut updated = gs.clone();
                    updated.initial_force = initial_force;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut extended_length = gs.extended_length;
            ui.horizontal(|ui| {
                ui.label("L_ext:");
                if committed_force_number(ui, "extended_length", &mut extended_length, "Fully extended (maximum) length of the gas spring in meters, measured between the two attachment points. At this length, the spring produces the initial force F\u{2080}.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"))
                {
                    let mut updated = gs.clone();
                    updated.extended_length = extended_length;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut stroke = gs.stroke;
            ui.horizontal(|ui| {
                ui.label("Stroke:");
                if committed_force_number(ui, "stroke", &mut stroke, "Available stroke (travel) of the gas spring in meters. The compressed length = extended_length \u{2212} stroke. The force increases as the spring compresses from extended to compressed position.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"))
                {
                    let mut updated = gs.clone();
                    updated.stroke = stroke;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut damping = gs.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if committed_force_number(ui, "damping", &mut damping, "Viscous damping coefficient of the gas spring (N\u{b7}s/m). Adds a velocity-dependent force that opposes extension/compression. Set to 0 for an ideal (frictionless) gas spring.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}s/m"))
                {
                    let mut updated = gs.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            let mut polytropic_exp = gs.polytropic_exp;
            ui.horizontal(|ui| {
                ui.label("n:");
                if committed_force_number(ui, "polytropic_exp", &mut polytropic_exp, "Polytropic exponent controlling the force-vs-compression curve. n=1.0 = isothermal (slow), n=1.4 = adiabatic (fast). Higher values make the force increase more steeply as the spring compresses. Typical gas springs use n \u{2248} 1.1\u{2013}1.3.",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX))
                {
                    let mut updated = gs.clone();
                    updated.polytropic_exp = polytropic_exp;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::GasSpring(updated),
                    });
                }
            });

            draw_point_picker(
                ui, "Pt A", &gs.body_a, &gs.point_a, &gs.point_a_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = gs.clone();
                    updated.point_a_name = name;
                    updated.point_a = coords;
                    ForceElement::GasSpring(updated)
                },
                pending,
            );

            draw_point_picker(
                ui, "Pt B", &gs.body_b, &gs.point_b, &gs.point_b_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = gs.clone();
                    updated.point_b_name = name;
                    updated.point_b = coords;
                    ForceElement::GasSpring(updated)
                },
                pending,
            );
        }

        ForceElement::BearingFriction(bf) => {
            ui.label(format!("Body I: {}  Body J: {}", bf.body_i, bf.body_j));

            let mut constant_drag = bf.constant_drag;
            ui.horizontal(|ui| {
                ui.label("Drag:");
                if committed_force_number(ui, "constant_drag", &mut constant_drag, "Constant friction torque (N\u{b7}m) that always opposes rotation at the joint, regardless of speed or load. Models seal drag or preload friction.",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m"))
                {
                    let mut updated = bf.clone();
                    updated.constant_drag = constant_drag;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut viscous_coeff = bf.viscous_coeff;
            ui.horizontal(|ui| {
                ui.label("Viscous:");
                if committed_force_number(ui, "viscous_coeff", &mut viscous_coeff, "Speed-dependent friction coefficient (N\u{b7}m\u{b7}s/rad). Friction torque contribution = viscous_coeff \u{d7} angular_velocity. Models lubricated bearing drag.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"))
                {
                    let mut updated = bf.clone();
                    updated.viscous_coeff = viscous_coeff;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut coulomb_coeff = bf.coulomb_coeff;
            ui.horizontal(|ui| {
                ui.label("\u{03bc}:");
                if committed_force_number(ui, "coulomb_coeff", &mut coulomb_coeff, "Coulomb friction coefficient (dimensionless). Friction torque = \u{b5} \u{d7} radial_load \u{d7} pin_radius. Models dry sliding contact friction at the joint pin.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX))
                {
                    let mut updated = bf.clone();
                    updated.coulomb_coeff = coulomb_coeff;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut pin_radius = bf.pin_radius;
            ui.horizontal(|ui| {
                ui.label("Pin r:");
                if committed_force_number(ui, "pin_radius", &mut pin_radius, "Joint pin radius in meters. Used with the Coulomb coefficient to compute friction torque: T_friction = \u{b5} \u{d7} F_radial \u{d7} pin_radius.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" m"))
                {
                    let mut updated = bf.clone();
                    updated.pin_radius = pin_radius;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut radial_load = bf.radial_load;
            ui.horizontal(|ui| {
                ui.label("Radial:");
                if committed_force_number(ui, "radial_load", &mut radial_load, "Estimated radial load on the joint pin in Newtons. Used to compute Coulomb friction. For more accurate results, use the actual joint reaction force from the statics solution.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N"))
                {
                    let mut updated = bf.clone();
                    updated.radial_load = radial_load;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });

            let mut v_threshold = bf.v_threshold;
            ui.horizontal(|ui| {
                ui.label("v_th:");
                if committed_force_number(ui, "v_threshold", &mut v_threshold, "Velocity threshold for the Coulomb friction smoothing function (rad/s). Below this speed, the friction torque transitions smoothly to zero to avoid numerical discontinuities. Typical values: 0.01\u{2013}0.1 rad/s.",
                    |field| field.speed(0.001)
                            .range(0.0..=f64::MAX)
                            .suffix(" rad/s"))
                {
                    let mut updated = bf.clone();
                    updated.v_threshold = v_threshold;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::BearingFriction(updated),
                    });
                }
            });
        }

        ForceElement::JointLimit(jl) => {
            ui.label(format!("Body I: {}  Body J: {}", jl.body_i, jl.body_j));

            let mut angle_min = jl.angle_min;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}_min:");
                if committed_force_number(ui, "angle_min", &mut angle_min, "Minimum allowed relative angle between bodies I and J in radians. When the joint angle drops below this limit, a stiff restoring torque pushes it back.",
                    |field| field.speed(0.01)
                            .suffix(" rad"))
                {
                    let mut updated = jl.clone();
                    updated.angle_min = angle_min;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut angle_max = jl.angle_max;
            ui.horizontal(|ui| {
                ui.label("\u{03b8}_max:");
                if committed_force_number(ui, "angle_max", &mut angle_max, "Maximum allowed relative angle between bodies I and J in radians. When the joint angle exceeds this limit, a stiff restoring torque pushes it back.",
                    |field| field.speed(0.01)
                            .suffix(" rad"))
                {
                    let mut updated = jl.clone();
                    updated.angle_max = angle_max;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut stiffness = jl.stiffness;
            ui.horizontal(|ui| {
                ui.label("k:");
                if committed_force_number(ui, "stiffness", &mut stiffness, "End-stop stiffness (N\u{b7}m/rad). Controls how stiffly the joint resists penetration past the angular limits. Higher values = harder stop.",
                    |field| field.speed(1.0)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m/rad"))
                {
                    let mut updated = jl.clone();
                    updated.stiffness = stiffness;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut damping = jl.damping;
            ui.horizontal(|ui| {
                ui.label("c:");
                if committed_force_number(ui, "damping", &mut damping, "End-stop damping (N\u{b7}m\u{b7}s/rad). Dissipates energy when the joint hits an angular limit, preventing oscillation. Set to a fraction of 2\u{d7}\u{221a}(k\u{d7}I) for critical damping.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m\u{00b7}s/rad"))
                {
                    let mut updated = jl.clone();
                    updated.damping = damping;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });

            let mut restitution = jl.restitution;
            ui.horizontal(|ui| {
                ui.label("e:");
                if committed_force_number(ui, "restitution", &mut restitution, "Coefficient of restitution at the angular end-stops (0\u{2013}1). 0 = perfectly inelastic (no bounce), 1 = perfectly elastic. Controls how much kinetic energy is retained after hitting a joint limit.",
                    |field| field.speed(0.01)
                            .range(0.0..=1.0))
                {
                    let mut updated = jl.clone();
                    updated.restitution = restitution;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::JointLimit(updated),
                    });
                }
            });
        }

        ForceElement::Motor(m) => {
            ui.label(format!("Body I: {}  Body J: {}", m.body_i, m.body_j));

            let mut stall_torque = m.stall_torque;
            ui.horizontal(|ui| {
                ui.label("\u{03c4}_s:");
                if committed_force_number(ui, "stall_torque", &mut stall_torque, "Stall torque (N\u{b7}m). Maximum torque the motor can produce at zero speed. The motor follows a linear torque-speed curve: torque = \u{3c4}_s \u{d7} (1 \u{2212} \u{3c9}/\u{3c9}\u{2080}).",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" N\u{00b7}m"))
                {
                    let mut updated = m.clone();
                    updated.stall_torque = stall_torque;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });

            let mut no_load_speed = m.no_load_speed;
            ui.horizontal(|ui| {
                ui.label("\u{03c9}\u{2080}:");
                if committed_force_number(ui, "no_load_speed", &mut no_load_speed, "No-load speed (rad/s). Maximum angular velocity of the motor when no external torque is applied. The motor produces zero torque at this speed.",
                    |field| field.speed(0.1)
                            .range(0.0..=f64::MAX)
                            .suffix(" rad/s"))
                {
                    let mut updated = m.clone();
                    updated.no_load_speed = no_load_speed;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });

            let mut direction = m.direction;
            ui.horizontal(|ui| {
                ui.label("Dir:");
                if committed_force_number(ui, "direction", &mut direction, "Motor direction multiplier. +1 = counter-clockwise, \u{2212}1 = clockwise. Scales the applied torque to set the intended rotation direction.",
                    |field| field.speed(0.1))
                {
                    let mut updated = m.clone();
                    updated.direction = direction;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::Motor(updated),
                    });
                }
            });
        }

        ForceElement::ForceZone(fz) => {
            // Target body dropdown: list bodies that have geometry set.
            let bodies_with_geom: Vec<String> = blueprint.bodies.iter()
                .filter(|(id, b)| b.geometry.is_some() && id.as_str() != "ground")
                .map(|(id, _)| id.clone())
                .collect();

            ui.horizontal(|ui| {
                ui.label("Target body:").on_hover_text("Body whose geometry elements will receive the zone force while the body's geometry overlaps the zone.");
                let current_label = if fz.body_id.is_empty() { "(none)" } else { &fz.body_id };
                egui::ComboBox::from_id_salt(format!("fz_body_{}", index))
                    .selected_text(current_label)
                    .show_ui(ui, |ui| {
                        for body_id in &bodies_with_geom {
                            if ui.selectable_label(fz.body_id == *body_id, body_id).clicked() {
                                let mut updated = fz.clone();
                                updated.body_id = body_id.clone();
                                *pending = Some(PendingPropertyEdit::UpdateForce {
                                    index,
                                    force: ForceElement::ForceZone(updated),
                                });
                            }
                        }
                        // Allow clearing the body selection.
                        if ui.selectable_label(fz.body_id.is_empty(), "(none)").clicked() {
                            let mut updated = fz.clone();
                            updated.body_id = String::new();
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::ForceZone(updated),
                            });
                        }
                    });
            });

            let mut fx = fz.force[0];
            ui.horizontal(|ui| {
                ui.label("Force X (N):");
                if committed_force_number(ui, "fx", &mut fx, "Horizontal force component (N) applied to geometry elements inside the zone. Positive = rightward.",
                    |field| field.speed(1.0).prefix("Fx: ")) {
                    let mut updated = fz.clone();
                    updated.force[0] = fx;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ForceZone(updated),
                    });
                }
            });

            let mut fy = fz.force[1];
            ui.horizontal(|ui| {
                ui.label("Force Y (N):");
                if committed_force_number(ui, "fy", &mut fy, "Vertical force component (N) applied to geometry elements inside the zone. Positive = upward.",
                    |field| field.speed(1.0).prefix("Fy: ")) {
                    let mut updated = fz.clone();
                    updated.force[1] = fy;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ForceZone(updated),
                    });
                }
            });

            // Zone min/max as editable DragValues (displayed in mm, stored in m).
            ui.label("Zone bounds (mm):");
            let mut min_x_mm = fz.zone_min[0] * 1e3;
            let mut min_y_mm = fz.zone_min[1] * 1e3;
            let mut max_x_mm = fz.zone_max[0] * 1e3;
            let mut max_y_mm = fz.zone_max[1] * 1e3;

            let mut zone_changed = false;
            ui.horizontal(|ui| {
                ui.label("Min X:");
                if committed_force_number(ui, "min_x_mm", &mut min_x_mm, "Minimum X coordinate of the force zone in mm. The zone applies its force while the body's geometry overlaps this bounding box.",
                    |field| field.speed(1.0).suffix(" mm")) {
                    zone_changed = true;
                }
                ui.label("Min Y:");
                if committed_force_number(ui, "min_y_mm", &mut min_y_mm, "Minimum Y coordinate of the force zone in mm.",
                    |field| field.speed(1.0).suffix(" mm")) {
                    zone_changed = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label("Max X:");
                if committed_force_number(ui, "max_x_mm", &mut max_x_mm, "Maximum X coordinate of the force zone in mm.",
                    |field| field.speed(1.0).suffix(" mm")) {
                    zone_changed = true;
                }
                ui.label("Max Y:");
                if committed_force_number(ui, "max_y_mm", &mut max_y_mm, "Maximum Y coordinate of the force zone in mm.",
                    |field| field.speed(1.0).suffix(" mm")) {
                    zone_changed = true;
                }
            });

            if zone_changed {
                let mut updated = fz.clone();
                updated.zone_min = [min_x_mm * 1e-3, min_y_mm * 1e-3];
                updated.zone_max = [max_x_mm * 1e-3, max_y_mm * 1e-3];
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(updated),
                });
            }

            // Application point: overlap centre, a locked body point, or the
            // shape's contact point (decision R-4).
            ui.separator();
            ui.label("Application point:");
            let mode = fz.app_mode();
            let mut picked = mode;
            ui.horizontal(|ui| {
                ui.radio_value(&mut picked, ZoneAppMode::Centroid, "Overlap centre")
                    .on_hover_text("The force acts at the centroid of the shape's overlap with the zone, found again at every pose.");
                ui.radio_value(&mut picked, ZoneAppMode::Locked, "Locked point")
                    .on_hover_text("The force acts at a fixed point on the body: set it below, or drag the F marker on the canvas.");
                ui.radio_value(&mut picked, ZoneAppMode::Contact, "Contact point")
                    .on_hover_text("The force acts where a surface pushing in the force's direction meets the shape: its lowest point for an upward force. On a wheel it stays straight below the hub as the wheel turns.");
            });
            if picked != mode {
                // A first lock seeds the shape's centre, not the body origin.
                let seed = blueprint
                    .bodies
                    .get(&fz.body_id)
                    .and_then(|b| b.geometry.as_ref())
                    .map_or([0.0, 0.0], |g| [g.offset.x, g.offset.y]);
                *pending = Some(PendingPropertyEdit::UpdateForce {
                    index,
                    force: ForceElement::ForceZone(fz.with_app_mode(picked, seed)),
                });
            }

            if let (ZoneAppMode::Locked, Some(lp)) = (mode, fz.body_local_app_point) {
                let mut lx_mm = lp[0] * 1e3;
                let mut ly_mm = lp[1] * 1e3;
                let mut ap_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Local X:");
                    if committed_force_number(ui, "lx_mm", &mut lx_mm, "Application point X in body-local coordinates (mm).",
                    |field| field.speed(1.0).suffix(" mm")) {
                        ap_changed = true;
                    }
                    ui.label("Local Y:");
                    if committed_force_number(ui, "ly_mm", &mut ly_mm, "Application point Y in body-local coordinates (mm).",
                    |field| field.speed(1.0).suffix(" mm")) {
                        ap_changed = true;
                    }
                });
                if ap_changed {
                    let mut updated = fz.clone();
                    updated.body_local_app_point = Some([lx_mm * 1e-3, ly_mm * 1e-3]);
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::ForceZone(updated),
                    });
                }
            }
        }

        ForceElement::LinearActuator(la) => {
            ui.label(format!("Body A: {}  Body B: {}", la.body_a, la.body_b));

            let mut force = la.force;
            ui.horizontal(|ui| {
                ui.label("F:");
                if committed_force_number(ui, "force", &mut force, ACTUATOR_FORCE_FIELD_TIP,
                    |field| field.speed(1.0)
                            .suffix(" N"))
                {
                    let mut updated = la.clone();
                    updated.force = force;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearActuator(updated),
                    });
                }
            });

            let mut speed_limit = la.speed_limit;
            ui.horizontal(|ui| {
                ui.label("v_max:");
                if committed_force_number(ui, "speed_limit", &mut speed_limit, "Maximum extension/retraction speed of the actuator in m/s. Set to 0 for unlimited speed.",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" m/s"))
                {
                    let mut updated = la.clone();
                    updated.speed_limit = speed_limit;
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: ForceElement::LinearActuator(updated),
                    });
                }
            });

            // ── Stroke Limits (collapsible) ──────────────────────────────
            let limits_label = if la.stroke_min == 0.0 && la.stroke_max == 0.0 {
                "Stroke Limits (disabled)"
            } else if la.stroke_min > 0.0 && la.stroke_max > 0.0 && la.stroke_min >= la.stroke_max {
                "Stroke Limits (min \u{2265} max, limits inactive)"
            } else {
                "Stroke Limits"
            };
            egui::CollapsingHeader::new(limits_label)
                .default_open(la.stroke_max > 0.0)
                .show(ui, |ui| {
                    let mut stroke_min = la.stroke_min;
                    ui.horizontal(|ui| {
                        let min_hint = if la.stroke_min == 0.0 && la.stroke_max > 0.0 {
                            "Min stroke (inactive):"
                        } else {
                            "Min stroke:"
                        };
                        ui.label(min_hint);
                        if committed_force_number(ui, "stroke_min", &mut stroke_min, "Minimum actuator length (fully retracted) in meters. Set to 0 to disable the lower stroke limit.",
                    |field| field.speed(0.001)
                                    .range(0.0..=f64::MAX)
                                    .suffix(" m"))
                        {
                            let mut updated = la.clone();
                            updated.stroke_min = stroke_min;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut stroke_max = la.stroke_max;
                    ui.horizontal(|ui| {
                        ui.label("Max stroke:");
                        if committed_force_number(ui, "stroke_max", &mut stroke_max, "Maximum actuator length (fully extended) in meters. Set to 0 to disable the upper stroke limit.",
                    |field| field.speed(0.001)
                                    .range(0.0..=f64::MAX)
                                    .suffix(" m"))
                        {
                            let mut updated = la.clone();
                            updated.stroke_max = stroke_max;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut stiffness = la.end_stop_stiffness;
                    ui.horizontal(|ui| {
                        ui.label("Stiffness (k):");
                        if committed_force_number(ui, "stiffness", &mut stiffness, "Stiffness of virtual end-stop springs at the stroke limits (N/m). Prevents the actuator from exceeding its stroke range.",
                    |field| field.speed(100.0)
                                    .range(0.0..=f64::MAX)
                                    .suffix(" N/m"))
                        {
                            let mut updated = la.clone();
                            updated.end_stop_stiffness = stiffness;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut damping = la.end_stop_damping;
                    ui.horizontal(|ui| {
                        ui.label("Damping (c):");
                        if committed_force_number(ui, "damping", &mut damping, "Damping of virtual end-stop contact (N\u{b7}s/m). Prevents oscillation at stroke limits.",
                    |field| field.speed(1.0)
                                    .range(0.0..=f64::MAX)
                                    .suffix(" N\u{00b7}s/m"))
                        {
                            let mut updated = la.clone();
                            updated.end_stop_damping = damping;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });

                    let mut restitution = la.end_stop_restitution;
                    ui.horizontal(|ui| {
                        ui.label("Restitution (e):");
                        if committed_force_number(ui, "restitution", &mut restitution, "Coefficient of restitution at stroke end-stops (0\u{2013}1). 0 = perfectly inelastic (no bounce), 1 = perfectly elastic.",
                    |field| field.speed(0.01)
                                    .range(0.0..=1.0))
                        {
                            let mut updated = la.clone();
                            updated.end_stop_restitution = restitution;
                            *pending = Some(PendingPropertyEdit::UpdateForce {
                                index,
                                force: ForceElement::LinearActuator(updated),
                            });
                        }
                    });
                });

            draw_point_picker(
                ui, "Pt A", &la.body_a, &la.point_a, &la.point_a_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = la.clone();
                    updated.point_a_name = name;
                    updated.point_a = coords;
                    ForceElement::LinearActuator(updated)
                },
                pending,
            );

            draw_point_picker(
                ui, "Pt B", &la.body_b, &la.point_b, &la.point_b_name,
                blueprint, index,
                |name, coords| {
                    let mut updated = la.clone();
                    updated.point_b_name = name;
                    updated.point_b = coords;
                    ForceElement::LinearActuator(updated)
                },
                pending,
            );

            // Convert this actuator into a LinearDriver constraint so
            // the mechanism can be analyzed in stroke-driven mode.
            // Removes any revolute drivers and the actuator force
            // element itself; the kinematic stroke is then commanded
            // by the constraint at f(t) = length_0 + velocity * t.
            ui.separator();
            if ui
                .button("Set as Linear Driver")
                .on_hover_text(
                    "Convert this actuator into a kinematic LinearDriver so the slider commands stroke directly. Replaces any revolute driver. The current stroke becomes length_0 so the pose doesn't jump.",
                )
                .clicked()
            {
                *pending = Some(PendingPropertyEdit::ConvertActuatorToLinearDriver { index });
            }
        }
    }
}

// ── Modulation helpers ────────────────────────────────────────────────────

/// Modulation type index for the ComboBox selector.
///
/// Maps TimeModulation variants to integer indices for the UI selector.
fn modulation_type_index(m: &TimeModulation) -> usize {
    match m {
        TimeModulation::Constant => 0,
        TimeModulation::Sinusoidal { .. } => 1,
        TimeModulation::Step { .. } => 2,
        TimeModulation::Ramp { .. } => 3,
        TimeModulation::Expression { .. } => 4,
    }
}

/// Human-readable label for a modulation type index.
fn modulation_type_label(idx: usize) -> &'static str {
    match idx {
        0 => "Constant",
        1 => "Sinusoidal",
        2 => "Step",
        3 => "Ramp",
        4 => "Expression",
        _ => "Unknown",
    }
}

/// Draw time modulation type selector and parameter fields.
///
/// When the modulation type or a parameter changes, the current element
/// is reconstructed via `make_element` and emitted as an UpdateForce edit.
fn draw_modulation_fields(
    ui: &mut egui::Ui,
    index: usize,
    modulation: &TimeModulation,
    make_element: impl Fn(TimeModulation) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    ui.separator();
    ui.label("Modulation:");

    let mut type_idx = modulation_type_index(modulation);
    let prev_idx = type_idx;

    egui::ComboBox::from_id_salt(format!("mod_type_{}", index))
        .selected_text(modulation_type_label(type_idx))
        .show_ui(ui, |ui| {
            for i in 0..5 {
                ui.selectable_value(&mut type_idx, i, modulation_type_label(i));
            }
        });

    // If the type changed, switch to new variant with default parameters.
    if type_idx != prev_idx {
        let new_mod = match type_idx {
            0 => TimeModulation::Constant,
            1 => TimeModulation::Sinusoidal { omega: 1.0, phase: 0.0 },
            2 => TimeModulation::Step { t_on: 0.0 },
            3 => TimeModulation::Ramp { t_start: 0.0, t_end: 1.0 },
            4 => TimeModulation::Expression { expr: "sin(2*pi*t)".into() },
            _ => TimeModulation::Constant,
        };
        *pending = Some(PendingPropertyEdit::UpdateForce {
            index,
            force: make_element(new_mod),
        });
        return;
    }

    // Draw parameter fields for the current modulation type.
    match modulation {
        TimeModulation::Constant => {} // no parameters

        TimeModulation::Sinusoidal { omega, phase } => {
            let mut w = *omega;
            ui.horizontal(|ui| {
                ui.label("\u{03c9}:");
                if committed_force_number(ui, "w", &mut w, "",
                    |field| field.speed(0.1).suffix(" rad/s"))
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Sinusoidal {
                            omega: w,
                            phase: *phase,
                        }),
                    });
                }
            });

            let mut p = *phase;
            ui.horizontal(|ui| {
                ui.label("Phase:");
                if committed_force_number(ui, "p", &mut p, "",
                    |field| field.speed(0.01).suffix(" rad"))
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Sinusoidal {
                            omega: *omega,
                            phase: p,
                        }),
                    });
                }
            });
        }

        TimeModulation::Step { t_on } => {
            let mut t = *t_on;
            ui.horizontal(|ui| {
                ui.label("t_on:");
                if committed_force_number(ui, "t", &mut t, "",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"))
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Step { t_on: t }),
                    });
                }
            });
        }

        TimeModulation::Ramp { t_start, t_end } => {
            let mut ts = *t_start;
            ui.horizontal(|ui| {
                ui.label("t_start:");
                if committed_force_number(ui, "ts", &mut ts, "",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"))
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Ramp {
                            t_start: ts,
                            t_end: *t_end,
                        }),
                    });
                }
            });

            let mut te = *t_end;
            ui.horizontal(|ui| {
                ui.label("t_end:");
                if committed_force_number(ui, "te", &mut te, "",
                    |field| field.speed(0.01)
                            .range(0.0..=f64::MAX)
                            .suffix(" s"))
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Ramp {
                            t_start: *t_start,
                            t_end: te,
                        }),
                    });
                }
            });
        }

        TimeModulation::Expression { expr } => {
            ui.horizontal(|ui| {
                let draft_id = ui.id().with(("force_modulation_expression", index));
                let mut text = ui.data(|data| data.get_temp::<String>(draft_id)).unwrap_or_else(|| expr.clone());
                ui.label("f(t):");
                let response = ui.text_edit_singleline(&mut text);
                if response.has_focus() {
                    ui.data_mut(|data| data.insert_temp(draft_id, text));
                    return;
                }
                ui.data_mut(|data| data.remove::<String>(draft_id));
                if response.lost_focus() && text != *expr && !ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(TimeModulation::Expression { expr: text }),
                    });
                }
            });

            // Validation hint: try parsing the expression
            if let Err(e) = expr.parse::<meval::Expr>() {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Parse error: {}", e),
                );
            } else if expr.parse::<meval::Expr>().ok().and_then(|e| e.bind("t").ok()).is_none() {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    "Error: cannot bind variable 't'",
                );
            }
        }
    }
}

// ── Point picker / fields ─────────────────────────────────────────────────

/// Draw a named-point picker dropdown for a force attachment point.
///
/// Populates the dropdown with the body's attachment points and mount points
/// from the blueprint (sorted by name). When a named point is selected both
/// the name and its coordinates are written back. If "custom coords..." is
/// selected the name is cleared and raw x/y DragValues are shown via
/// `draw_point_fields`.
#[allow(clippy::too_many_arguments)]
fn draw_point_picker(
    ui: &mut egui::Ui,
    label: &str,
    body_id: &str,
    current_point: &[f64; 2],
    current_name: &Option<String>,
    blueprint: &crate::io::MechanismJson,
    index: usize,
    make_element: impl Fn(Option<String>, [f64; 2]) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let body_json = blueprint.bodies.get(body_id);

    // Build sorted option list: attachment points first, then mount points.
    let mut options: Vec<(String, String, [f64; 2])> = Vec::new();
    if let Some(bj) = body_json {
        let mut att_names: Vec<&String> = bj.attachment_points.keys().collect();
        att_names.sort();
        for name in att_names {
            options.push((
                name.clone(),
                format!("{} (joint)", name),
                bj.attachment_points[name],
            ));
        }
        let mut mt_names: Vec<&String> = bj.mount_points.keys().collect();
        mt_names.sort();
        for name in mt_names {
            options.push((
                name.clone(),
                format!("{} (mount)", name),
                bj.mount_points[name],
            ));
        }
    }

    ui.horizontal(|ui| {
        ui.label(format!("{}:", label));
        let current_label = current_name
            .as_ref()
            .map(|n| n.as_str())
            .unwrap_or("custom");
        egui::ComboBox::from_id_salt(format!("{}-{}-{}", label, body_id, index))
            .selected_text(current_label)
            .show_ui(ui, |ui| {
                for (name, display, coords) in &options {
                    if ui
                        .selectable_label(current_name.as_ref() == Some(name), display)
                        .clicked()
                    {
                        *pending = Some(PendingPropertyEdit::UpdateForce {
                            index,
                            force: make_element(Some(name.clone()), *coords),
                        });
                    }
                }
                if ui
                    .selectable_label(current_name.is_none(), "custom coords...")
                    .clicked()
                {
                    *pending = Some(PendingPropertyEdit::UpdateForce {
                        index,
                        force: make_element(None, *current_point),
                    });
                }
            });
    });

    // Show raw x/y editors only when no named point is selected.
    if current_name.is_none() {
        draw_point_fields(ui, label, current_point, index, |pt| {
            make_element(None, pt)
        }, pending);
    }
}

/// Draw x/y DragValue fields for a 2D point, emitting an UpdateForce edit
/// when either component changes.
///
/// `make_element` takes the updated `[f64; 2]` and returns the full
/// `ForceElement` with that point replaced.
fn draw_point_fields(
    ui: &mut egui::Ui,
    label: &str,
    point: &[f64; 2],
    index: usize,
    make_element: impl Fn([f64; 2]) -> ForceElement,
    pending: &mut Option<PendingPropertyEdit>,
) {
    let mut x = point[0];
    let mut y = point[1];
    ui.horizontal(|ui| {
        ui.label(format!("{} x:", label));
        if committed_force_number(ui, "x", &mut x, "",
                    |field| field.speed(0.001).suffix(" m"))
        {
            *pending = Some(PendingPropertyEdit::UpdateForce {
                index,
                force: make_element([x, point[1]]),
            });
        }
    });
    ui.horizontal(|ui| {
        ui.label(format!("{} y:", label));
        if committed_force_number(ui, "y", &mut y, "",
                    |field| field.speed(0.001).suffix(" m"))
        {
            *pending = Some(PendingPropertyEdit::UpdateForce {
                index,
                force: make_element([point[0], y]),
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::samples::SampleMechanism;
    use crate::gui::test_support::{central_panel_frame, drew_text, key_tap, primary_button, text_rect, typed};

    fn bl045_editor_state() -> AppState {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 0.0;
        state.blueprint.as_mut().unwrap().forces = vec![ForceElement::ExternalForce(ExternalForceElement {
            body_id: "crank".into(), local_point: [0.0, 0.0], local_point_name: None,
            force: [10.0, 0.0], modulation: TimeModulation::Constant,
        })];
        state.rebuild();
        state.undo_history.clear();
        state.dirty = false;
        state
    }

    fn bl045_editor_frame(ctx: &egui::Context, state: &mut AppState, events: Vec<egui::Event>) -> (egui::FullOutput, bool) {
        let bp = state.blueprint.as_ref().unwrap().clone();
        let mut pending = None;
        let output = central_panel_frame(ctx, events, |ui| {
            draw_force_element_details(ui, 0, &bp.forces[0], &bp, &mut pending);
        });
        let committed = pending.is_some();
        super::super::pending_edits::apply_pending(state, pending);
        (output, committed)
    }

    fn bl045_fx(state: &AppState) -> f64 {
        let ForceElement::ExternalForce(force) = &state.blueprint.as_ref().unwrap().forces[0] else {
            panic!("external force fixture missing");
        };
        force.force[0]
    }

    fn bl045_fx_rect(ctx: &egui::Context, state: &mut AppState) -> egui::Rect {
        bl045_editor_frame(ctx, state, Vec::new());
        bl045_editor_frame(ctx, state, key_tap(egui::Key::Tab, egui::Modifiers::NONE));
        let id = ctx.memory(|m| m.focused()).expect("Tab focuses Fx");
        bl045_editor_frame(ctx, state, key_tap(egui::Key::Escape, egui::Modifiers::NONE));
        ctx.read_response(id).expect("Fx is drawn").rect
    }

    #[test]
    fn bl045_numeric_force_drag_commits_once_on_release() {
        let ctx = egui::Context::default();
        let mut state = bl045_editor_state();
        let start = bl045_fx_rect(&ctx, &mut state).center();
        let end = start + egui::vec2(40.0, 0.0);
        bl045_editor_frame(&ctx, &mut state, vec![egui::Event::PointerMoved(start)]);
        bl045_editor_frame(&ctx, &mut state, vec![primary_button(start, true)]);
        for at in [start + egui::vec2(20.0, 0.0), end] {
            assert!(!bl045_editor_frame(&ctx, &mut state, vec![egui::Event::PointerMoved(at)]).1,
                "drag previews must not commit/rebuild every frame");
            assert_eq!(bl045_fx(&state), 10.0);
            assert_eq!(state.undo_history.undo_count(), 0);
            assert!(!state.dirty);
        }
        assert!(bl045_editor_frame(&ctx, &mut state, vec![primary_button(end, false)]).1);
        assert!(bl045_fx(&state) > 10.0, "the retained draft commits on release");
        assert_eq!(state.undo_history.undo_count(), 1);
        assert!(state.dirty);
        assert!(!bl045_editor_frame(&ctx, &mut state, Vec::new()).1);
        state.undo();
        assert_eq!(bl045_fx(&state), 10.0);
    }

    #[test]
    fn bl045_typed_force_commits_on_enter_and_escape_discards_the_draft() {
        for finish in [egui::Key::Enter, egui::Key::Escape] {
            let ctx = egui::Context::default();
            let mut state = bl045_editor_state();
            bl045_editor_frame(&ctx, &mut state, Vec::new());
            bl045_editor_frame(&ctx, &mut state, key_tap(egui::Key::Tab, egui::Modifiers::NONE));
            assert!(!bl045_editor_frame(&ctx, &mut state, vec![typed("25")]).1, "typing is a preview");
            assert_eq!(bl045_fx(&state), 10.0);
            assert_eq!(state.undo_history.undo_count(), 0);
            let committed = bl045_editor_frame(&ctx, &mut state, key_tap(finish, egui::Modifiers::NONE)).1;
            assert_eq!(committed, finish == egui::Key::Enter);
            assert_eq!(bl045_fx(&state), if finish == egui::Key::Enter { 25.0 } else { 10.0 });
            assert_eq!(state.undo_history.undo_count(), usize::from(committed));
            assert_eq!(state.dirty, committed);
            assert!(!bl045_editor_frame(&ctx, &mut state, Vec::new()).1);
        }
    }

    #[test]
    fn bl045_keyboard_nudges_are_retained_until_focus_leaves_then_commit_once() {
        let ctx = egui::Context::default();
        let mut state = bl045_editor_state();
        bl045_editor_frame(&ctx, &mut state, Vec::new());
        bl045_editor_frame(&ctx, &mut state, key_tap(egui::Key::Tab, egui::Modifiers::NONE));
        for _ in 0..2 {
            assert!(!bl045_editor_frame(&ctx, &mut state, key_tap(egui::Key::ArrowUp, egui::Modifiers::NONE)).1);
            assert_eq!(bl045_fx(&state), 10.0);
        }
        assert!(bl045_editor_frame(&ctx, &mut state, key_tap(egui::Key::Tab, egui::Modifiers::NONE)).1);
        assert!((bl045_fx(&state) - 10.2).abs() < 1e-10);
        assert_eq!(state.undo_history.undo_count(), 1);
        assert!(state.dirty);
    }

    #[test]
    fn bl045_zone_mode_click_is_an_immediate_single_undoable_edit() {
        let ctx = egui::Context::default();
        let mut state = bl045_editor_state();
        state.blueprint.as_mut().unwrap().forces[0] = ForceElement::ForceZone(ForceZoneElement {
            body_id: "crank".into(), zone_min: [-1.0, -1.0], zone_max: [1.0, 1.0],
            force: [10.0, 0.0], label: None, body_local_app_point: None, at_contact_point: false,
        });
        state.rebuild();
        let output = bl045_editor_frame(&ctx, &mut state, Vec::new()).0;
        let at = text_rect(&output, "Contact point").expect("mode choice is drawn").center();
        bl045_editor_frame(&ctx, &mut state, vec![egui::Event::PointerMoved(at)]);
        bl045_editor_frame(&ctx, &mut state, vec![primary_button(at, true)]);
        assert!(bl045_editor_frame(&ctx, &mut state, vec![primary_button(at, false)]).1);
        let ForceElement::ForceZone(zone) = &state.blueprint.as_ref().unwrap().forces[0] else { panic!("zone missing"); };
        assert_eq!(zone.app_mode(), ZoneAppMode::Contact);
        assert_eq!(state.undo_history.undo_count(), 1);
        assert!(state.dirty);
        state.undo();
        let ForceElement::ForceZone(zone) = &state.blueprint.as_ref().unwrap().forces[0] else { panic!("zone missing"); };
        assert_eq!(zone.app_mode(), ZoneAppMode::Centroid);
    }

    #[test]
    fn bl045_modulation_expression_draft_survives_frames_and_enter_or_escape() {
        for finish in [egui::Key::Enter, egui::Key::Escape] {
            let ctx = egui::Context::default();
            let mut state = bl045_editor_state();
            let ForceElement::ExternalForce(force) = &mut state.blueprint.as_mut().unwrap().forces[0] else {
                panic!("external force fixture missing");
            };
            force.modulation = TimeModulation::Expression { expr: "sin(t)".into() };
            state.rebuild();
            let output = bl045_editor_frame(&ctx, &mut state, Vec::new()).0;
            let at = text_rect(&output, "sin(t)").expect("expression field is drawn").center();
            bl045_editor_frame(&ctx, &mut state, vec![egui::Event::PointerMoved(at)]);
            bl045_editor_frame(&ctx, &mut state, vec![primary_button(at, true)]);
            bl045_editor_frame(&ctx, &mut state, vec![primary_button(at, false)]);
            bl045_editor_frame(&ctx, &mut state, key_tap(egui::Key::A, egui::Modifiers::COMMAND));
            assert!(!bl045_editor_frame(&ctx, &mut state, vec![typed("2*t")]).1);
            let (output, committed) = bl045_editor_frame(&ctx, &mut state, Vec::new());
            assert!(!committed, "a draft is not a document edit");
            assert!(drew_text(&output, "2*t"), "the draft survives the next frame: {:?}", crate::gui::test_support::drawn_texts(&output));
            assert_eq!(state.undo_history.undo_count(), 0);
            assert!(!state.dirty);
            let committed = bl045_editor_frame(&ctx, &mut state, key_tap(finish, egui::Modifiers::NONE)).1;
            assert_eq!(committed, finish == egui::Key::Enter);
            let ForceElement::ExternalForce(force) = &state.blueprint.as_ref().unwrap().forces[0] else {
                panic!("external force fixture missing");
            };
            let TimeModulation::Expression { expr } = &force.modulation else { panic!("expression missing"); };
            assert_eq!(expr, if committed { "2*t" } else { "sin(t)" });
            assert_eq!(state.undo_history.undo_count(), usize::from(committed));
            assert_eq!(state.dirty, committed);
            assert!(!bl045_editor_frame(&ctx, &mut state, Vec::new()).1);
        }
    }

    /// The code's convention is positive = extension (the actuator pushes
    /// its attachment points apart); the field tip used to say the opposite.
    #[test]
    fn actuator_force_field_tip_states_the_extension_sign_convention() {
        assert!(ACTUATOR_FORCE_FIELD_TIP.contains("Positive = extension"), "{ACTUATOR_FORCE_FIELD_TIP}");
        assert!(!ACTUATOR_FORCE_FIELD_TIP.contains("Positive = tension"), "{ACTUATOR_FORCE_FIELD_TIP}");
        assert!(!ACTUATOR_FORCE_FIELD_TIP.contains("compression"), "{ACTUATOR_FORCE_FIELD_TIP}");
    }
}
