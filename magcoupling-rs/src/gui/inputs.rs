//! The input list of the panel's left side (spec M4 "Layout": "inputs, generated from
//! metadata, grouped as the package groups them ... A Key design group on top").
//!
//! [`InputCatalogue`] is built once from the engine's input metadata: the Key design group
//! ([`KEY_DESIGN`]), then every input arranged two ways ([`InputOrder`]). The workflow order
//! (the default, decision O-1) follows [`WORKFLOW`]: the groups in the order a designer works,
//! each with its rarely changed rows under a collapsed Advanced heading. The workbook order puts
//! every input in its package group (`coupling`, `metal`, `calibration`, `materials`,
//! `temperature`, `clamps`), each group split into sections by its nested input groups
//! (`coupling.magnets`, `temperature.demag`, ...). In either order every input is in exactly
//! one section; the Key design inputs are also in their group (decision M41-13), and both rows
//! edit the same value.

use std::sync::OnceLock;

use crate::DesignInputs;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value, input_rows};

/// The Key design group, in order (spec M4 "Layout": face gap, pole count, magnet part, axial
/// length, operating temperature, back iron, cup wall, conductance, measured drag). The axial
/// length is the A-2 override of both rings' length, blank by default (it moves the torque at
/// the default library part; the manual lengths do not); the sizing mode switch sits above
/// the group.
pub const KEY_DESIGN: [&str; 10] = [
    "metal.face_gap_mm",
    "coupling.npole",
    "coupling.magnets.part_inner",
    "coupling.magnets.part_outer",
    "coupling.magnets.axial_length_mm",
    "coupling.op_temp_C",
    "coupling.backiron",
    "metal.cup_wall_corner_mm",
    "temperature.thermal.conductance_W_K",
    "metal.measured_drag_Nm",
];

/// The heading of every input group and nested group, by path prefix, in the package's order.
pub const SECTION_LABELS: [(&str, &str); 18] = [
    ("coupling", "Coupling"),
    ("coupling.magnets", "Magnets"),
    ("metal", "Metal design"),
    ("calibration", "Calibration"),
    ("materials", "Materials"),
    ("materials.steel", "Back-iron steel"),
    ("materials.nickel", "Nickel plating"),
    ("materials.screws", "Screw classes"),
    ("materials.parts", "Part materials"),
    ("temperature", "Temperature design"),
    ("temperature.duty", "Duty"),
    ("temperature.demag", "Demagnetization"),
    ("temperature.adhesive", "Adhesive"),
    ("temperature.mismatch", "Thermal mismatch"),
    ("temperature.slip_loss", "Slip loss"),
    ("temperature.thermal", "Thermal network"),
    ("temperature.adhesive_life", "Adhesive life"),
    ("clamps", "Shaft clamps"),
];

/// How the left side orders the design inputs (decision O-1): by design workflow (the default)
/// or as the workbook's package groups. A per-session choice of view: no design file or share
/// link holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputOrder {
    #[default]
    Workflow,
    Workbook,
}

impl InputOrder {
    /// Both orders, in toggle order.
    pub const ALL: [InputOrder; 2] = [InputOrder::Workflow, InputOrder::Workbook];

    /// The toggle's text.
    pub const fn label(self) -> &'static str {
        match self {
            InputOrder::Workflow => "By workflow",
            InputOrder::Workbook => "Workbook groups",
        }
    }
}

/// The heading the advanced sections of a workflow group sit under, closed by default.
pub const ADVANCED_HEADING: &str = "Advanced";

/// One section of a workflow group ([`WORKFLOW`]).
#[derive(Clone, Copy, Debug)]
pub struct WorkflowSection {
    /// `<group>.<section>`; the group's id for its first section, drawn without a heading.
    pub id: &'static str,
    pub label: &'static str,
    /// Under the group's [`ADVANCED_HEADING`]: rows a designer rarely changes (fit and screw
    /// factors, physical and model constants, the optional adapter).
    pub advanced: bool,
    /// Its inputs, in the order a designer sets them.
    pub paths: &'static [&'static str],
}

/// One workflow group: an id, a heading and its sections, the advanced ones last.
#[derive(Clone, Copy, Debug)]
pub struct WorkflowGroup {
    pub id: &'static str,
    pub label: &'static str,
    pub sections: &'static [WorkflowSection],
}

/// The workflow groups, in the order a designer works (decision O-2): the requirements and
/// operating conditions first (the spec fixed before a magnet is chosen: the torque, the
/// temperatures, the space claim, the drive and the duty), then the magnets and their rings, the
/// gap, the housing around them, the shaft and its clamps, the materials (the adhesive and its
/// bondlines among them), the thermal and demagnetization inputs, then the calibration, the
/// results stored from a 3D run and the model's constants. Every input sits in exactly one
/// section (a test lists any that does not); the Key design inputs also sit on top, as in the
/// workbook order.
pub const WORKFLOW: [WorkflowGroup; 8] = [
    WorkflowGroup {
        id: "operating",
        label: "Requirements and operating conditions",
        sections: &[
            WorkflowSection {
                id: "operating",
                label: "Requirements and operating conditions",
                advanced: false,
                paths: &[
                    "metal.required_min_Nm",
                    "coupling.op_temp_C",
                    "metal.min_temp_C",
                    "metal.variation",
                ],
            },
            WorkflowSection {
                id: "operating.envelope",
                label: "Space claim",
                advanced: false,
                paths: &[
                    "metal.max_diameter_mm",
                    "metal.max_overall_axial_mm",
                    "metal.max_large_dia_axial_mm",
                ],
            },
            WorkflowSection {
                id: "operating.drive",
                label: "Drive and gearbox",
                advanced: false,
                paths: &[
                    "coupling.drive_torque_Nm",
                    "coupling.drive_safety_factor",
                    "coupling.gear_ratio",
                    "coupling.gear_efficiency",
                    "coupling.gearbox_input_rating_Nm",
                ],
            },
            WorkflowSection {
                id: "operating.duty",
                label: "Duty and life",
                advanced: false,
                paths: &[
                    "temperature.duty.wheel_rotor_rpm",
                    "temperature.duty.hot_ambient_C",
                    "temperature.duty.driving_rise_C",
                    "temperature.duty.life_hours",
                    "temperature.adhesive_life.service_years",
                    "temperature.adhesive_life.daily_swing_C",
                ],
            },
            WorkflowSection {
                id: "operating.slip",
                label: "Slip events",
                advanced: false,
                paths: &[
                    "metal.slip_rpm",
                    "metal.slip_event_s",
                    "metal.life_events",
                    "temperature.duty.fault_trip_s",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "magnets",
        label: "Magnets and rings",
        sections: &[
            WorkflowSection {
                id: "magnets",
                label: "Magnets and rings",
                advanced: false,
                paths: &[
                    "coupling.npole",
                    "coupling.magnets.part_inner",
                    "coupling.magnets.part_outer",
                    "coupling.magnets.axial_length_mm",
                    "coupling.faceted",
                    "coupling.inner_back_apothem_mm",
                ],
            },
            WorkflowSection {
                id: "magnets.grades",
                label: "Grades (manual dimensions)",
                advanced: false,
                paths: &[
                    "coupling.magnets.grade_inner",
                    "coupling.magnets.grade_outer",
                ],
            },
            WorkflowSection {
                id: "magnets.manual_inner",
                label: "Manual inner blocks",
                advanced: false,
                paths: &[
                    "coupling.magnets.manual_inner_length_mm",
                    "coupling.magnets.manual_inner_width_mm",
                    "coupling.magnets.manual_inner_thickness_mm",
                    "coupling.magnets.manual_inner_br_T",
                ],
            },
            WorkflowSection {
                id: "magnets.manual_outer",
                label: "Manual outer blocks",
                advanced: false,
                paths: &[
                    "coupling.magnets.manual_outer_length_mm",
                    "coupling.magnets.manual_outer_width_mm",
                    "coupling.magnets.manual_outer_thickness_mm",
                    "coupling.magnets.manual_outer_br_T",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "gap",
        label: "Gap and clearances",
        sections: &[
            WorkflowSection {
                id: "gap",
                label: "Gap and clearances",
                advanced: false,
                paths: &[
                    "metal.face_gap_mm",
                    "metal.sleeve_mm",
                    "metal.liner_mm",
                    "metal.residual_target_mm",
                ],
            },
            WorkflowSection {
                id: "gap.allowances",
                label: "Running-clearance allowances",
                advanced: false,
                paths: &[
                    "metal.shaft_displacement_mm",
                    "metal.runout_mm",
                    "metal.deflection_mm",
                    "metal.thermal_mm",
                    "metal.sleeve_form_mm",
                    "metal.magnet_position_mm",
                ],
            },
            WorkflowSection {
                id: "gap.bedding",
                label: "Bedding clearances",
                advanced: true,
                paths: &["metal.sleeve_bedding_mm", "metal.liner_bedding_mm"],
            },
        ],
    },
    WorkflowGroup {
        id: "housing",
        label: "Housing and retainers",
        sections: &[
            WorkflowSection {
                id: "housing",
                label: "Housing and retainers",
                advanced: false,
                paths: &[
                    "coupling.backiron",
                    "metal.cup_wall_corner_mm",
                    "metal.hub_length_mm",
                    "metal.cup_depth_mm",
                    "metal.web_mm",
                    "metal.boss_length_mm",
                    "metal.boss_od_mm",
                ],
            },
            WorkflowSection {
                id: "housing.cap",
                label: "Front cap",
                advanced: false,
                paths: &[
                    "metal.cap_axial_mm",
                    "metal.cap_od_mm",
                    "metal.cap_thread_dia_mm",
                    "metal.cap_thread_engagement_mm",
                ],
            },
            WorkflowSection {
                id: "housing.retainers",
                label: "Endplates and retainers",
                advanced: false,
                paths: &[
                    "metal.front_endplate_mm",
                    "metal.rear_endplate_mm",
                    "metal.retainer_span_mm",
                    "metal.rear_endplate_hole_mm",
                ],
            },
            WorkflowSection {
                id: "housing.adapter",
                label: "Optional adapter and its joint",
                advanced: true,
                paths: &[
                    "metal.adapter_flange_dia_mm",
                    "metal.adapter_flange_mm",
                    "metal.adapter_pilot_dia_mm",
                    "metal.adapter_pilot_mm",
                    "metal.adapter_boss_mm",
                    "metal.adapter_hardware_g",
                    "clamps.joint_screws",
                    "clamps.joint_bolt_circle_mm",
                    "clamps.joint_friction",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "shaft",
        label: "Shaft, key and clamps",
        sections: &[
            WorkflowSection {
                id: "shaft",
                label: "Shaft, key and clamps",
                advanced: false,
                paths: &[
                    "coupling.bore_mm",
                    "coupling.keyway_depth_mm",
                    "clamps.key_width_mm",
                    "clamps.key_contact_mm",
                ],
            },
            WorkflowSection {
                id: "shaft.clamp",
                label: "Clamp",
                advanced: false,
                paths: &[
                    "clamps.clamp_type",
                    "clamps.safety_factor",
                    "clamps.screw_class",
                    "clamps.alloy",
                    "clamps.boss_od_mm",
                    "clamps.clamp_length_mm",
                    "clamps.friction",
                    "clamps.preload_fraction",
                ],
            },
            WorkflowSection {
                id: "shaft.clamp_geometry",
                label: "Clamp geometry",
                advanced: false,
                paths: &[
                    "clamps.slit_mm",
                    "clamps.ligament_mm",
                    "clamps.wall_out_mm",
                    "clamps.grip_min_mm",
                    "clamps.axial_margin_mm",
                    "clamps.relief_mm",
                    "clamps.hinge_mm",
                ],
            },
            WorkflowSection {
                id: "shaft.screw_model",
                label: "Clamp and screw factors",
                advanced: true,
                paths: &[
                    "clamps.factor_one_piece",
                    "clamps.factor_two_piece",
                    "clamps.nut_factor",
                    "clamps.engagement_x_d",
                    "clamps.strip_sf",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "materials",
        label: "Materials",
        sections: &[
            WorkflowSection {
                id: "materials",
                label: "Materials",
                advanced: false,
                paths: &[
                    "materials.parts.back_iron",
                    "materials.parts.sleeve_liner",
                    "materials.parts.cap_housing",
                ],
            },
            WorkflowSection {
                id: "materials.steel",
                label: "Back-iron steel",
                advanced: false,
                paths: &[
                    "materials.steel.bsat_T",
                    "materials.steel.conductivity_S_m",
                    "materials.steel.mu_r_incremental",
                    "materials.steel.specific_heat_J_kgK",
                    "materials.steel.cte_per_C",
                    "materials.steel.modulus_GPa",
                    "materials.steel.density_g_cm3",
                ],
            },
            WorkflowSection {
                id: "materials.magnet",
                label: "Magnet properties",
                advanced: false,
                paths: &[
                    "temperature.mismatch.ndfeb_cte_per_C",
                    "temperature.mismatch.ndfeb_modulus_GPa",
                    "temperature.slip_loss.sigma_ndfeb_S_m",
                    "temperature.thermal.c_ndfeb",
                ],
            },
            WorkflowSection {
                id: "materials.adhesive",
                label: "Adhesive and bondlines",
                advanced: false,
                paths: &[
                    "temperature.adhesive.selected",
                    "metal.bond_inner_mm",
                    "metal.bond_outer_mm",
                    "temperature.mismatch.recommended_bondline_mm",
                    "temperature.mismatch.adhesive_shear_modulus_GPa",
                    "temperature.adhesive_life.hot_strength_retained",
                    "temperature.adhesive_life.fatigue_endurance",
                ],
            },
            WorkflowSection {
                id: "materials.other",
                label: "Other part properties",
                advanced: false,
                paths: &[
                    "temperature.slip_loss.sigma_316_S_m",
                    "temperature.thermal.c_316",
                    "temperature.thermal.c_aluminium",
                    "materials.nickel.thickness_mm",
                ],
            },
            WorkflowSection {
                id: "materials.mass",
                label: "Densities and mass allowance",
                advanced: false,
                paths: &[
                    "metal.steel_density_g_mm3",
                    "metal.sleeve_density_g_mm3",
                    "metal.al_density_g_mm3",
                    "metal.hardware_g",
                ],
            },
            WorkflowSection {
                id: "materials.screws",
                label: "Screw classes",
                advanced: true,
                paths: &[
                    "materials.screws.proof_12_9_MPa",
                    "materials.screws.proof_10_9_MPa",
                    "materials.screws.yield_A4_70_MPa",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "thermal",
        label: "Thermal and demagnetization",
        sections: &[
            WorkflowSection {
                id: "thermal",
                label: "Thermal and demagnetization",
                advanced: false,
                paths: &[
                    "temperature.thermal.conductance_W_K",
                    "metal.measured_drag_Nm",
                    "temperature.slip_loss.high_multiplier",
                ],
            },
            WorkflowSection {
                id: "thermal.demag",
                label: "Demagnetization",
                advanced: false,
                paths: &[
                    "temperature.demag.coercivity_source",
                    "temperature.demag.hcj20_kA_m",
                    "temperature.demag.beta_hcj_per_C",
                    "temperature.demag.knee_fraction",
                    "temperature.demag.design_margin_C",
                ],
            },
            WorkflowSection {
                id: "thermal.slip_model",
                label: "Slip-loss model",
                advanced: true,
                paths: &["temperature.slip_loss.end_factor"],
            },
        ],
    },
    WorkflowGroup {
        id: "calibration",
        label: "Calibration and model",
        sections: &[
            WorkflowSection {
                id: "calibration",
                label: "Calibration and model",
                advanced: false,
                paths: &[
                    "calibration.measured_torque_Nm",
                    "calibration.test_temp_C",
                    "calibration.spacing_mm",
                    "calibration.gap_definition",
                    "calibration.total_magnets",
                ],
            },
            WorkflowSection {
                id: "calibration.prototype",
                label: "Prototype magnets",
                advanced: false,
                paths: &[
                    "calibration.apothem_mm",
                    "calibration.magnet_length_mm",
                    "calibration.magnet_width_mm",
                    "calibration.magnet_thickness_mm",
                    "calibration.br_T",
                ],
            },
            WorkflowSection {
                id: "calibration.fea",
                label: "3D results: reference torques",
                advanced: false,
                paths: &["calibration.fea_torque1_Nm", "calibration.fea_torque2_Nm"],
            },
            WorkflowSection {
                id: "calibration.stored_demag",
                label: "3D results: stored reverse fields (refresh after a 3D run)",
                advanced: false,
                paths: &[
                    "temperature.demag.h_rev_aligned_kA_m",
                    "temperature.demag.h_rev_pullout_kA_m",
                    "temperature.demag.h_rev_likepole_kA_m",
                    "temperature.demag.h_rev_single_ring_kA_m",
                ],
            },
            WorkflowSection {
                id: "calibration.stored_slip",
                label: "3D results: stored slip-loss fields (refresh after a 3D run)",
                advanced: false,
                paths: &[
                    "temperature.slip_loss.b_hub_T",
                    "temperature.slip_loss.b_cup_T",
                    "temperature.slip_loss.b_sleeve_T",
                    "temperature.slip_loss.b_liner_T",
                    "temperature.slip_loss.cap_integral_T2m4",
                    "temperature.slip_loss.web_integral_T2m2",
                    "temperature.slip_loss.b_magnet_T",
                    "temperature.slip_loss.b_hub_free_T",
                    "temperature.slip_loss.b_cup_free_T",
                    "temperature.slip_loss.web_integral_free_T2m2",
                ],
            },
            WorkflowSection {
                id: "calibration.model",
                label: "Model constants",
                advanced: true,
                paths: &[
                    "coupling.c_end",
                    "coupling.max_harmonic",
                    "coupling.mu0",
                    "calibration.alpha_br_per_C",
                    "calibration.c_end",
                    "calibration.f_cal_original",
                    "calibration.mu0",
                ],
            },
        ],
    },
];

/// Where an optional input starts when the user enters a value (decision M41-12): the result
/// it overrides, unrounded. The axial length override enters at the inner ring's length in
/// use (both rings take it), so nothing moves. The measured drag enters at the model's
/// equivalent mean drag torque; entering any measured drag switches the thermal summary from
/// the not-measured high estimate to the measured branch, so the slip loss and the steady
/// temperatures move (the drag torque itself is the model's).
pub const OPTIONAL_SEEDS: [(&str, &str); 2] = [
    ("coupling.magnets.axial_length_mm", "model.inner_length_mm"),
    ("metal.measured_drag_Nm", "temperature.slip_loss.drag_Nm"),
];

/// One input: its path, metadata and default value.
#[derive(Clone, Debug, PartialEq)]
pub struct InputEntry {
    pub path: String,
    pub meta: &'static InputMeta,
    pub default: Value,
}

/// A run of inputs under one heading: in the workbook order a group's own fields or one of its
/// nested groups, in the workflow order a [`WorkflowSection`].
#[derive(Clone, Debug, PartialEq)]
pub struct InputSection {
    /// The workbook order's path prefix (`coupling`, `coupling.magnets`), or the workflow
    /// section's id (`magnets`, `magnets.grades`). A section whose id is its group's name is
    /// drawn without a heading.
    pub id: String,
    pub label: &'static str,
    /// Under the group's [`ADVANCED_HEADING`] (workflow order only).
    pub advanced: bool,
    pub entries: Vec<InputEntry>,
}

/// A group of the left side and its sections: a package group (`coupling`, ...) in schema
/// order, or a workflow group ([`WORKFLOW`]).
#[derive(Clone, Debug, PartialEq)]
pub struct InputGroup {
    pub name: String,
    pub label: &'static str,
    pub sections: Vec<InputSection>,
}

/// Every input, arranged for the left side of the panel.
#[derive(Clone, Debug, PartialEq)]
pub struct InputCatalogue {
    /// [`KEY_DESIGN`], in order.
    pub key_design: Vec<InputEntry>,
    /// Every input, by package group and section, in schema order (the workbook order).
    pub groups: Vec<InputGroup>,
    /// Every input, by workflow group and section ([`WORKFLOW`], the workflow order).
    pub workflow: Vec<InputGroup>,
}

/// The heading of a path prefix; `None` if [`SECTION_LABELS`] lacks it (a test checks none
/// does).
pub fn section_label(prefix: &str) -> Option<&'static str> {
    SECTION_LABELS
        .iter()
        .find(|(p, _)| *p == prefix)
        .map(|(_, label)| *label)
}

impl InputCatalogue {
    /// The catalogue of the engine's inputs, with the defaults of [`DesignInputs::default`].
    pub fn new() -> Self {
        let rows = input_rows(&DesignInputs::default());
        let entry = |row: &crate::engine::meta::InputRow| InputEntry {
            path: row.path.clone(),
            meta: row.meta,
            default: row.value.clone(),
        };
        let key_design = KEY_DESIGN
            .iter()
            .map(|&path| {
                let row = rows.iter().find(|row| row.path == path);
                entry(row.unwrap_or_else(|| panic!("KEY_DESIGN: no input {path}")))
            })
            .collect();
        let mut groups: Vec<InputGroup> = Vec::new();
        for row in &rows {
            let (prefix, _) = row
                .path
                .rsplit_once('.')
                .expect("every input sits in a group");
            let name = prefix.split('.').next().unwrap_or(prefix);
            if groups.last().is_none_or(|g| g.name != name) {
                groups.push(InputGroup {
                    name: name.to_owned(),
                    label: section_label(name).unwrap_or("Inputs"),
                    sections: Vec::new(),
                });
            }
            let group = groups.last_mut().expect("pushed above");
            if group.sections.last().is_none_or(|s| s.id != prefix) {
                group.sections.push(InputSection {
                    id: prefix.to_owned(),
                    label: section_label(prefix).unwrap_or("Inputs"),
                    advanced: false,
                    entries: Vec::new(),
                });
            }
            let section = group.sections.last_mut().expect("pushed above");
            section.entries.push(entry(row));
        }
        let workflow = WORKFLOW
            .iter()
            .map(|group| InputGroup {
                name: group.id.to_owned(),
                label: group.label,
                sections: group
                    .sections
                    .iter()
                    .map(|section| InputSection {
                        id: section.id.to_owned(),
                        label: section.label,
                        advanced: section.advanced,
                        entries: section
                            .paths
                            .iter()
                            .map(|&path| {
                                let row = rows.iter().find(|row| row.path == path);
                                entry(row.unwrap_or_else(|| panic!("WORKFLOW: no input {path}")))
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        Self {
            key_design,
            groups,
            workflow,
        }
    }

    /// The groups of `order`: the workflow groups or the package groups.
    pub fn groups_in(&self, order: InputOrder) -> &[InputGroup] {
        match order {
            InputOrder::Workflow => &self.workflow,
            InputOrder::Workbook => &self.groups,
        }
    }

    /// The group and the section of `order` that hold the input at `path`; `None` for a path
    /// that is no input.
    pub fn section_of(
        &self,
        order: InputOrder,
        path: &str,
    ) -> Option<(&InputGroup, &InputSection)> {
        self.groups_in(order).iter().find_map(|group| {
            group
                .sections
                .iter()
                .find(|section| section.entries.iter().any(|e| e.path == path))
                .map(|section| (group, section))
        })
    }

    /// The catalogue, built once: it depends only on the engine's metadata.
    pub fn get() -> &'static InputCatalogue {
        static CATALOGUE: OnceLock<InputCatalogue> = OnceLock::new();
        CATALOGUE.get_or_init(InputCatalogue::new)
    }

    /// Every entry of the package groups, in schema order.
    pub fn all(&self) -> impl Iterator<Item = &InputEntry> {
        self.groups
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
    }

    /// The entry of an input path.
    pub fn entry(&self, path: &str) -> Option<&InputEntry> {
        self.all().find(|entry| entry.path == path)
    }
}

impl Default for InputCatalogue {
    fn default() -> Self {
        Self::new()
    }
}

/// Decimal places of a slider step: the fewest that write the step exactly (0.01 → 2,
/// 0.005 → 3, 2 → 0, 1e-13 → 13). Slider values are rounded to them (decision M41-1), so a
/// value the slider sets is the decimal the user reads (1.41, not 1.4100000000000001), and
/// stepping back to a default lands on it exactly, for every default on its step grid (all
/// but the vacuum permeability's two, a test lists them).
pub fn step_decimals(step: f64) -> usize {
    (0..=15)
        .find(|&decimals| {
            let scaled = step * 10f64.powi(decimals as i32);
            scaled.round() >= 1.0 && (scaled - scaled.round()).abs() <= 1e-9 * scaled
        })
        .unwrap_or(15)
}

/// Whether a number lies outside its slider range (a value from a design file, a share link
/// or the struct: the slider keeps it until edited, decision M41-2, and the row flags it).
pub fn outside_range(range: Option<SliderRange>, value: &Value) -> bool {
    let x = match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        _ => return false,
    };
    range.is_some_and(|r| !(r.min..=r.max).contains(&x))
}

/// The result an optional input starts from ([`OPTIONAL_SEEDS`]).
pub fn optional_seed(path: &str) -> Option<&'static str> {
    OPTIONAL_SEEDS
        .iter()
        .find(|(input, _)| *input == path)
        .map(|(_, result)| *result)
}

/// A short note under a text input saying what the engine makes of the text: whether a part
/// name is a library part, whether a grade name is in the grade table.
pub fn text_hint(path: &str, text: &str) -> Option<&'static str> {
    match path {
        "coupling.magnets.part_inner" | "coupling.magnets.part_outer" => {
            Some(if library::lookup(text).is_some() {
                "Library part"
            } else {
                "Not a library part: the manual dimensions are used"
            })
        }
        "coupling.magnets.grade_inner" | "coupling.magnets.grade_outer" => {
            Some(if text.is_empty() {
                "Blank: the manual Br, no rating"
            } else if grades::grade(text).is_some() {
                "Grade table entry (used with manual dimensions)"
            } else {
                "Not in the grade table: the manual Br, no rating"
            })
        }
        _ => None,
    }
}

/// The hover text of an input: help, path, workbook cell, slider range, default, and whether
/// it is a model assumption.
pub fn input_tooltip(entry: &InputEntry) -> String {
    let meta = entry.meta;
    let mut lines = Vec::new();
    if !meta.help.is_empty() {
        lines.push(meta.help.to_owned());
    }
    lines.push(entry.path.clone());
    lines.push(meta.cell.map_or_else(
        || "Rust-only input (no workbook cell)".to_owned(),
        str::to_owned,
    ));
    if let Some(r) = meta.range {
        let unit = crate::gui::format::with_unit(String::new(), meta.unit);
        let scale = if r.log { ", logarithmic" } else { "" };
        lines.push(format!(
            "Slider {} to {}{unit}, step {}{scale}",
            r.min, r.max, r.step
        ));
    }
    let default = match (&entry.default, meta.ty) {
        (Value::None, FieldType::OptF64) => "blank".to_owned(),
        (Value::Text(text), _) if text.is_empty() => "blank".to_owned(),
        (Value::Int(code), _) if !meta.choices.is_empty() => {
            let choice = meta.choices.iter().find(|(c, _)| c == code);
            choice.map_or_else(|| code.to_string(), |(c, text)| format!("{c} = {text}"))
        }
        (value, _) => crate::gui::format::format_value(value),
    };
    lines.push(format!("Default: {default}"));
    if meta.assumption {
        lines.push("Model assumption (Addendum A3)".to_owned());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::{InputSet, ResultSet};

    #[test]
    fn every_input_is_in_exactly_one_section_in_schema_order() {
        let catalogue = InputCatalogue::new();
        let rows = input_rows(&DesignInputs::default());
        let listed: Vec<&str> = catalogue.all().map(|e| e.path.as_str()).collect();
        let schema: Vec<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        assert_eq!(listed, schema);
        let names: Vec<&str> = catalogue.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "coupling",
                "metal",
                "calibration",
                "materials",
                "temperature",
                "clamps"
            ]
        );
        for group in &catalogue.groups {
            for section in &group.sections {
                assert!(section.id.starts_with(&group.name), "{}", section.id);
                assert!(!section.entries.is_empty());
                assert!(!section.advanced, "no workbook section is advanced");
                for entry in &section.entries {
                    assert_eq!(entry.path.rsplit_once('.').unwrap().0, section.id);
                }
            }
        }
        assert_eq!(
            catalogue.groups_in(InputOrder::Workbook),
            &catalogue.groups[..]
        );
    }

    #[test]
    fn every_input_is_in_exactly_one_workflow_section() {
        // Fails when an input is added to the engine without a place in WORKFLOW, or a path
        // there names no input: the message lists each one.
        let schema: Vec<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|r| r.path)
            .collect();
        let listed: Vec<&str> = WORKFLOW
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.paths.iter().copied())
            .collect();
        let missing: Vec<&str> = schema
            .iter()
            .map(String::as_str)
            .filter(|path| !listed.contains(path))
            .collect();
        assert!(
            missing.is_empty(),
            "inputs without a workflow section: {missing:?}"
        );
        let unknown: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|path| !schema.iter().any(|s| s == path))
            .collect();
        assert!(
            unknown.is_empty(),
            "workflow paths that are no input: {unknown:?}"
        );
        let mut twice: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|path| listed.iter().filter(|p| *p == path).count() > 1)
            .collect();
        twice.dedup();
        assert!(
            twice.is_empty(),
            "inputs in two workflow sections: {twice:?}"
        );
        assert_eq!(listed.len(), schema.len());
        // The catalogue's workflow arrangement is WORKFLOW, entry for entry, and each entry is
        // the workbook arrangement's.
        let catalogue = InputCatalogue::new();
        let arranged: Vec<&str> = catalogue
            .groups_in(InputOrder::Workflow)
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(arranged, listed);
        for entry in catalogue
            .workflow
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
        {
            assert_eq!(Some(entry), catalogue.entry(&entry.path));
        }
    }

    #[test]
    fn the_workflow_groups_run_in_design_order_with_the_advanced_rows_last() {
        let labels: Vec<&str> = WORKFLOW.iter().map(|g| g.label).collect();
        assert_eq!(
            labels,
            [
                "Requirements and operating conditions",
                "Magnets and rings",
                "Gap and clearances",
                "Housing and retainers",
                "Shaft, key and clamps",
                "Materials",
                "Thermal and demagnetization",
                "Calibration and model",
            ]
        );
        let mut ids: Vec<&str> = Vec::new();
        for group in &WORKFLOW {
            // The first section is the group's own (drawn without a heading), never advanced.
            let first = &group.sections[0];
            assert_eq!(first.id, group.id);
            assert!(!first.advanced, "{}", group.id);
            let mut seen_advanced = false;
            for section in group.sections {
                assert!(!section.paths.is_empty(), "{}", section.id);
                if section.id != group.id {
                    assert!(
                        section.id.starts_with(&format!("{}.", group.id)),
                        "{}",
                        section.id
                    );
                }
                // The Advanced heading closes the group: no ordinary section after it.
                assert!(!seen_advanced || section.advanced, "{}", section.id);
                seen_advanced |= section.advanced;
                assert!(section.label.is_ascii(), "{}", section.label);
                ids.push(section.id);
            }
            assert!(group.label.is_ascii(), "{}", group.label);
        }
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "section ids are unique");
        // Every Key design input is in an ordinary section, so its group shows it.
        let catalogue = InputCatalogue::new();
        let section = |path: &str| {
            catalogue
                .section_of(InputOrder::Workflow, path)
                .unwrap_or_else(|| panic!("{path} is in no workflow section"))
                .1
        };
        for path in KEY_DESIGN {
            assert!(!section(path).advanced, "{path}");
        }
        // The rows a designer rarely changes are advanced (decision O-4): fit and screw
        // factors, physical and model constants, the optional adapter and its joint. The
        // clamp's two assumptions and the fields stored from a 3D run stay in view: they scale
        // the clamp capacity, and the stored fields must be refreshed after a geometry change.
        let advanced = |path: &str| section(path).advanced;
        for path in [
            "coupling.mu0",
            "calibration.mu0",
            "coupling.max_harmonic",
            "clamps.nut_factor",
            "clamps.joint_friction",
        ] {
            assert!(advanced(path), "{path}");
        }
        for path in [
            "clamps.friction",
            "clamps.preload_fraction",
            "temperature.demag.h_rev_pullout_kA_m",
            "temperature.slip_loss.b_hub_free_T",
        ] {
            assert!(!advanced(path), "{path}");
        }
        let rows = |advanced: bool| {
            WORKFLOW
                .iter()
                .flat_map(|g| g.sections.iter())
                .filter(|s| s.advanced == advanced)
                .map(|s| s.paths.len())
                .sum::<usize>()
        };
        assert_eq!(rows(true), 27, "decision O-4: 27 advanced rows");
        // Related inputs share a section (decision O-2): the adhesive and its bondlines, the
        // optional adapter and its joint; the 3D run's stored results sit in one group.
        for related in [
            &[
                "temperature.adhesive.selected",
                "metal.bond_inner_mm",
                "temperature.mismatch.recommended_bondline_mm",
                "temperature.mismatch.adhesive_shear_modulus_GPa",
                "temperature.adhesive_life.fatigue_endurance",
            ][..],
            &[
                "metal.adapter_flange_dia_mm",
                "clamps.joint_screws",
                "clamps.joint_friction",
            ],
        ] {
            for path in related {
                assert_eq!(section(path).id, section(related[0]).id, "{path}");
            }
        }
        for path in [
            "calibration.fea_torque1_Nm",
            "temperature.demag.h_rev_aligned_kA_m",
            "temperature.slip_loss.b_magnet_T",
        ] {
            let (group, _) = catalogue.section_of(InputOrder::Workflow, path).unwrap();
            assert_eq!(group.name, "calibration", "{path}");
        }
        assert_eq!(
            catalogue.section_of(InputOrder::Workflow, "no.such.input"),
            None
        );
        assert!(!advanced("metal.face_gap_mm"));
        assert_eq!(InputOrder::default(), InputOrder::Workflow);
        assert_eq!(ADVANCED_HEADING, "Advanced");
    }

    #[test]
    fn every_section_has_a_heading_and_every_heading_a_section() {
        let catalogue = InputCatalogue::new();
        let mut prefixes: Vec<&str> = catalogue
            .groups
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.sections.iter().map(|s| s.id.as_str()))
            })
            .collect();
        prefixes.dedup();
        for prefix in &prefixes {
            assert!(section_label(prefix).is_some(), "no heading for {prefix}");
        }
        for (prefix, _) in SECTION_LABELS {
            assert!(prefixes.contains(&prefix), "unused heading {prefix}");
        }
        assert_eq!(catalogue.groups[0].sections[1].label, "Magnets");
        assert_eq!(InputCatalogue::get(), &catalogue);
    }

    #[test]
    fn the_key_design_group_is_the_spec_list_with_the_axial_length_override() {
        let catalogue = InputCatalogue::new();
        let paths: Vec<&str> = catalogue
            .key_design
            .iter()
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(paths, KEY_DESIGN);
        let axial = catalogue.entry("coupling.magnets.axial_length_mm").unwrap();
        assert_eq!(axial.meta.ty, FieldType::OptF64);
        assert_eq!(axial.default, Value::None, "blank by default");
        // Every Key design entry is the same entry as in its group.
        for key in &catalogue.key_design {
            assert_eq!(Some(key), catalogue.entry(&key.path));
        }
    }

    #[test]
    fn the_inputs_cover_every_field_type_the_panel_draws() {
        let catalogue = InputCatalogue::new();
        let has = |pred: &dyn Fn(&InputEntry) -> bool| catalogue.all().any(pred);
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::F64 && e.meta.range.is_some_and(|r| r.log)
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && e.meta.choices.is_empty()
        ));
        assert!(has(
            &|e| e.meta.ty == FieldType::I64 && !e.meta.choices.is_empty()
        ));
        assert!(has(&|e| e.meta.ty == FieldType::OptF64));
        assert!(has(&|e| e.meta.ty == FieldType::Text));
        assert!(has(&|e| e.meta.rust_only));
        // No input is of a type the panel has no widget for.
        assert!(!has(&|e| e.meta.ty == FieldType::NumOrText));
        // Every number without choices has a slider range.
        assert!(!has(&|e| matches!(
            e.meta.ty,
            FieldType::F64 | FieldType::I64 | FieldType::OptF64
        ) && e.meta.choices.is_empty()
            && e.meta.range.is_none()));
    }

    #[test]
    fn step_decimals_write_each_step_exactly() {
        for (step, decimals) in [
            (2.0, 0),
            (1000.0, 0),
            (0.5, 1),
            (0.1, 1),
            (0.05, 2),
            (0.01, 2),
            (0.005, 3),
            (0.0001, 4),
            (1e-5, 5),
            (1e-7, 7),
            (1e-8, 8),
            (1e-11, 11),
            (1e-13, 13),
        ] {
            assert_eq!(step_decimals(step), decimals, "{step}");
        }
        for entry in InputCatalogue::new().all() {
            if let Some(r) = entry.meta.range {
                let d = step_decimals(r.step);
                assert!(d < 15, "{}: step {}", entry.path, r.step);
            }
        }
    }

    /// The inputs whose default is off its slider's step grid: the engine's step for the
    /// vacuum permeability (1e-11) is coarser than its default's last digit (1.256637e-6), so
    /// once nudged it never returns to the default (an open item of `04-memory.yaml`: the
    /// engine step should be 1e-12).
    const OFF_GRID_DEFAULTS: [&str; 2] = ["coupling.mu0", "calibration.mu0"];

    #[test]
    fn every_slider_default_is_on_its_step_grid_except_the_listed() {
        // Decision M41-1: a slider stores min + k * step rounded to the step's decimals, so
        // stepping back lands on a default exactly only if the default is such a value.
        let on_grid = |r: SliderRange, x: f64| {
            let k = (x - r.min) / r.step;
            let decimals = step_decimals(r.step);
            (k - k.round()).abs() < 1e-6 && format!("{x:.decimals$}").parse() == Ok(x)
        };
        let catalogue = InputCatalogue::new();
        let mut off_grid = Vec::new();
        for entry in catalogue.all() {
            let x = match entry.default {
                Value::Num(x) => x,
                Value::Int(i) => i as f64,
                _ => continue,
            };
            if let Some(r) = entry.meta.range
                && !on_grid(r, x)
            {
                off_grid.push(entry.path.as_str());
            }
        }
        assert_eq!(off_grid, OFF_GRID_DEFAULTS);
    }

    #[test]
    fn values_outside_the_slider_range_are_flagged() {
        let face_gap = InputCatalogue::new()
            .entry("metal.face_gap_mm")
            .unwrap()
            .meta
            .range;
        assert!(!outside_range(face_gap, &Value::Num(0.3)));
        assert!(!outside_range(face_gap, &Value::Num(5.0)));
        assert!(outside_range(face_gap, &Value::Num(5.0000001)));
        assert!(outside_range(face_gap, &Value::Num(0.29)));
        assert!(outside_range(face_gap, &Value::Int(7)));
        assert!(!outside_range(face_gap, &Value::None));
        assert!(!outside_range(None, &Value::Num(1e9)));
    }

    #[test]
    fn every_optional_input_starts_from_the_result_it_overrides() {
        let catalogue = InputCatalogue::new();
        let results = compute_all(&DesignInputs::default());
        let optional: Vec<&str> = catalogue
            .all()
            .filter(|e| e.meta.ty == FieldType::OptF64)
            .map(|e| e.path.as_str())
            .collect();
        let seeded: Vec<&str> = OPTIONAL_SEEDS.iter().map(|(input, _)| *input).collect();
        assert_eq!(optional, seeded);
        for (input, result) in OPTIONAL_SEEDS {
            assert_eq!(optional_seed(input), Some(result));
            let Some(Value::Num(seed)) = results.get(result) else {
                panic!("{result} is not a number")
            };
            let range = catalogue.entry(input).unwrap().meta.range.unwrap();
            assert!((range.min..=range.max).contains(&seed), "{input}: {seed}");
        }
    }

    #[test]
    fn entering_the_axial_length_at_its_seed_moves_nothing() {
        // Decision M41-12: the override starts at the inner ring's length (both default rings
        // are 12.7 mm B842SH), so the design is unchanged until the slider moves.
        let base = DesignInputs::default();
        let results = compute_all(&base);
        let Some(Value::Num(seed)) = results.get("model.inner_length_mm") else {
            panic!("a number")
        };
        let mut seeded = base.clone();
        seeded
            .set("coupling.magnets.axial_length_mm", Value::Num(seed))
            .unwrap();
        assert_eq!(compute_all(&seeded), results);
    }

    #[test]
    fn text_hints_say_what_the_engine_makes_of_the_text() {
        assert_eq!(
            text_hint("coupling.magnets.part_inner", "B842SH"),
            Some("Library part")
        );
        assert_eq!(
            text_hint("coupling.magnets.part_outer", "b842sh"),
            Some("Not a library part: the manual dimensions are used")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_inner", ""),
            Some("Blank: the manual Br, no rating")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "Y30"),
            Some("Grade table entry (used with manual dimensions)")
        );
        assert_eq!(
            text_hint("coupling.magnets.grade_outer", "N99"),
            Some("Not in the grade table: the manual Br, no rating")
        );
        assert_eq!(text_hint("metal.face_gap_mm", "1"), None);
        // Every text input has a hint.
        for entry in InputCatalogue::new().all() {
            if entry.meta.ty == FieldType::Text {
                assert!(text_hint(&entry.path, "").is_some(), "{}", entry.path);
            }
        }
    }

    #[test]
    fn tooltips_carry_help_path_cell_range_and_default() {
        let catalogue = InputCatalogue::new();
        assert_eq!(
            input_tooltip(catalogue.entry("metal.face_gap_mm").unwrap()),
            "Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119\n\
             Slider 0.3 to 5 mm, step 0.01\nDefault: 1.400"
        );
        let backiron = input_tooltip(catalogue.entry("coupling.backiron").unwrap());
        assert!(
            backiron.ends_with("Default: 1 = steel circuit"),
            "{backiron}"
        );
        let drag = input_tooltip(catalogue.entry("metal.measured_drag_Nm").unwrap());
        assert!(
            drag.contains("logarithmic") && drag.ends_with("Default: blank"),
            "{drag}"
        );
        let harmonic = input_tooltip(catalogue.entry("coupling.max_harmonic").unwrap());
        assert!(
            harmonic.contains("Rust-only input (no workbook cell)")
                && harmonic.ends_with("Model assumption (Addendum A3)"),
            "{harmonic}"
        );
        let npole = input_tooltip(catalogue.entry("coupling.npole").unwrap());
        assert!(npole.contains("Slider 4 to 40, step 2"), "{npole}");
    }
}
