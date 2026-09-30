//! The deviation registry: approved corrections to the workbook.
//!
//! The M1 math audit (`docs/analyses/2026-09-29-magcoupling-math-audit.md`)
//! found fourteen engine errors, E1 to E14, and the user approved every
//! correction on 2026-09-29 (E14 is documentation only). Everything else stays
//! workbook-exact. This registry is the one place that says where the port
//! departs from the workbook and why.
//!
//! # How a deviation is applied
//!
//! - **Formula corrections.** A module's `compute` takes `dev: Deviations` and
//!   branches at the corrected formula, keeping the workbook form beside it:
//!   `if dev.is_on(DeviationId::E10) { corrected } else { workbook }`.
//! - **Default corrections** (E1, E3, E5 change input defaults). The field's
//!   declared default is the corrected value; the entry records the workbook
//!   value in `workbook_input_defaults`, and `DesignInputs::defaults_with`
//!   (test-only) puts it back when the deviation is off.
//! - **Status.** An entry is `Planned` until its code lands, then `Applied`
//!   with `changes_at_defaults` listing every cell that changes at default
//!   inputs, with the workbook and the corrected value. A broad correction
//!   (more than 15 changed cells, decision D4) names a reviewed golden file in
//!   `changes_file` instead (E3: `tests/data/deviations/E3.json`, E4:
//!   `tests/data/deviations/E4.json`, E5: `tests/data/deviations/E5.json`).
//! - **Probes.** A correction that changes nothing at default inputs (E7 to
//!   E13) lists `probes`: input overrides on which it shows (the report's
//!   off-default example) and the cells it changes there, with the workbook
//!   and the corrected value.
//!
//! # The test-only switch
//!
//! Users always get [`Deviations::ALL`] (`compute_all`). The `workbook-parity`
//! feature, enabled only for tests, adds [`Deviations::NONE`] and
//! [`Deviations::only`], so workbook parity and the differential tests compare
//! against the workbook and the Python engine exactly, and each deviation can
//! be checked alone. The GUI never shows the switch.

use std::fmt;

#[cfg(feature = "workbook-parity")]
use super::meta::InputSet;
use super::meta::Value;

/// The M1 audit report every entry cites.
pub const REPORT: &str = "docs/analyses/2026-09-29-magcoupling-math-audit.md";

/// Identifier of an approved correction, numbered as in the audit report.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviationId {
    E1,
    E2,
    E3,
    E4,
    E5,
    E6,
    E7,
    E8,
    E9,
    E10,
    E11,
    E12,
    E13,
    E14,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 14] = [
        DeviationId::E1,
        DeviationId::E2,
        DeviationId::E3,
        DeviationId::E4,
        DeviationId::E5,
        DeviationId::E6,
        DeviationId::E7,
        DeviationId::E8,
        DeviationId::E9,
        DeviationId::E10,
        DeviationId::E11,
        DeviationId::E12,
        DeviationId::E13,
        DeviationId::E14,
    ];

    /// Position in [`DeviationId::ALL`] and in [`REGISTRY`].
    pub const fn index(self) -> usize {
        self as usize
    }

    const fn bit(self) -> u32 {
        1 << self.index()
    }
}

impl fmt::Display for DeviationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}", self.index() + 1)
    }
}

/// What a deviation changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviationClass {
    /// A formula, constant or default the engine computes with.
    Engine,
    /// Help text or README wording only; no number changes.
    Documentation,
}

/// Whether the correction is in the code yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviationStatus {
    /// Approved, not yet implemented: the engine is workbook-exact here.
    Planned,
    /// Implemented; `changes_at_defaults` (or the `changes_file`) is complete.
    Applied,
}

/// A constant value recorded in the registry (workbook or corrected).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    Num(f64),
    Int(i64),
    Text(&'static str),
    None,
}

impl Literal {
    /// The same value as a dynamic [`Value`].
    pub fn to_value(self) -> Value {
        match self {
            Literal::Num(x) => Value::Num(x),
            Literal::Int(i) => Value::Int(i),
            Literal::Text(s) => Value::Text(s.to_owned()),
            Literal::None => Value::None,
        }
    }
}

/// One cell that a deviation changes at default inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellChange {
    /// Workbook cell, e.g. `"Temperature design!C106"`.
    pub cell: &'static str,
    /// The workbook's value (equal to the snapshot, checked by a test).
    pub workbook: Literal,
    /// The value with the correction applied.
    pub corrected: Literal,
}

/// Inputs on which a correction that is neutral at defaults shows (the report's
/// off-default example), and the cells it changes there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    /// What the inputs represent (the report's example).
    pub label: &'static str,
    /// Input overrides by path, applied to the defaults.
    pub inputs: &'static [(&'static str, Literal)],
    /// Cells with their workbook value and corrected value for these inputs.
    pub expect: &'static [CellChange],
}

/// One approved correction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deviation {
    pub id: DeviationId,
    /// Short statement of the workbook error (the report's Group column).
    pub title: &'static str,
    pub class: DeviationClass,
    pub status: DeviationStatus,
    /// Every workbook cell the report names for this entry (inputs, corrected
    /// formulas and named downstream cells). Each exists in the snapshot.
    pub cells: &'static [&'static str],
    /// The correction, as approved (the report's Proposed correction column).
    pub corrected_formula: &'static str,
    /// Workbook defaults of inputs whose declared default this deviation
    /// corrects, by input path. Empty unless the deviation changes a default.
    pub workbook_input_defaults: &'static [(&'static str, Literal)],
    /// Help texts this correction rewords, as (input path, or table column path
    /// `group.table[*].field`, and the workbook's text). The metadata tests
    /// compare the recorded text with Python and the workbook, and require the
    /// port's help to differ.
    pub workbook_help: &'static [(&'static str, &'static str)],
    /// Every cell that changes at default inputs (complete once `Applied`).
    pub changes_at_defaults: &'static [CellChange],
    /// For a broad correction (decision D4: more than 15 cells change at
    /// defaults): the golden file, relative to `magcoupling-rs/`, that lists
    /// every changed cell as `{"cell": [workbook, corrected]}`. Rewritten by
    /// `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` and reviewed as a diff.
    /// `changes_at_defaults` stays empty for such entries.
    pub changes_file: Option<&'static str>,
    /// Off-default checks for corrections neutral at defaults (E7 to E13).
    pub probes: &'static [Probe],
}

impl Deviation {
    /// Where the evidence is: the audit report entry.
    pub fn evidence(&self) -> String {
        format!("{REPORT}, entry {}", self.id)
    }
}

/// The registry, in report order: `REGISTRY[id.index()].id == id`.
pub const REGISTRY: &[Deviation] = &[
    Deviation {
        id: DeviationId::E1,
        title: "Adhesive shear modulus does not match the selected adhesive",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Temperature design!C96",
            "Temperature design!C104",
            "Temperature design!C105",
            "Temperature design!C106",
            "Temperature design!C201",
            "Temperature design!C202",
        ],
        corrected_formula: "Temperature design!C96 = 0.107 GPa (Loctite AA 326: E = 0.300 GPa, nu = 0.4). \
            The per-adhesive shear modulus the report also suggests is deferred to Addendum A5 (decision D2).",
        workbook_input_defaults: &[(
            "temperature.mismatch.adhesive_shear_modulus_GPa",
            Literal::Num(0.55),
        )],
        workbook_help: &[("temperature.mismatch.adhesive_shear_modulus_GPa", "")],
        changes_at_defaults: &[
            CellChange {
                cell: "Temperature design!C96",
                workbook: Literal::Num(0.55),
                corrected: Literal::Num(0.107),
            },
            CellChange {
                cell: "Temperature design!C104",
                workbook: Literal::Num(46.12653767935644),
                corrected: Literal::Num(11.59827175128882),
            },
            CellChange {
                cell: "Temperature design!C105",
                workbook: Literal::Num(26.728261752019268),
                corrected: Literal::Num(6.027792167235714),
            },
            CellChange {
                cell: "Temperature design!C106",
                workbook: Literal::Text("Above the lap-shear strength at the block ends"),
                corrected: Literal::Text("Below the lap-shear strength"),
            },
            CellChange {
                cell: "Temperature design!C201",
                workbook: Literal::Num(11.365659614702167),
                corrected: Literal::Num(2.563198259452589),
            },
            CellChange {
                cell: "Temperature design!C202",
                workbook: Literal::Text("Above the fatigue endurance: qualify by thermal cycling"),
                corrected: Literal::Text("Below the fatigue endurance"),
            },
        ],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E2,
        title: "Clamp screw length leaves out the clamp slit",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Clamp screw sizes!C34",
            "Clamp screw sizes!D34",
            "Clamp screw sizes!E34",
            "Clamp screw sizes!F34",
            "Clamp screw sizes!G34",
            "Clamp screw sizes!C35",
            "Clamp screw sizes!D35",
            "Clamp screw sizes!E35",
            "Clamp screw sizes!F35",
            "Clamp screw sizes!G35",
            "Shaft clamps!C48",
        ],
        corrected_formula: "Length = CEILING(grip + slit + engagement x d, 2); \
            fits inside = length <= grip + slit + thread available; \
            report when no 2 mm step meets both rules; \
            the note is the Rust-only result clamps.length_note (decision D6).",
        workbook_input_defaults: &[],
        workbook_help: &[(
            "clamps.table[*].length_mm",
            "Grip plus required engagement, rounded up to an even length.",
        )],
        changes_at_defaults: &[
            CellChange {
                cell: "Clamp screw sizes!E34",
                workbook: Literal::Num(12.0),
                corrected: Literal::Num(14.0),
            },
            CellChange {
                cell: "Clamp screw sizes!E35",
                workbook: Literal::Int(1),
                corrected: Literal::Int(0),
            },
            CellChange {
                cell: "Shaft clamps!C48",
                workbook: Literal::Text("ISO 4762 M4 x 12, class 12.9"),
                corrected: Literal::Text("ISO 4762 M4 x 14, class 12.9"),
            },
        ],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E3,
        title: "Library remanence of the N42SH parts is below the vendor's published minimum",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C17",
            "Calculator!C21",
            "Calculator!C27",
            "Calculator!C31",
            "Calibration!C21",
        ],
        corrected_formula: "Br of library rows B842SH, BX042SH and BX082SH = 1.30 T \
            (vendor grade minimum; decision D1), with Calculator!C17, C27 and Calibration!C21 to match. \
            Rerunning fields3d with the new Br is M3 scope.",
        workbook_input_defaults: &[
            ("coupling.magnets.manual_inner_br_T", Literal::Num(1.29)),
            ("coupling.magnets.manual_outer_br_T", Literal::Num(1.29)),
            ("calibration.br_T", Literal::Num(1.29)),
        ],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: Some("tests/data/deviations/E3.json"),
        probes: &[],
    },
    Deviation {
        id: DeviationId::E4,
        title: "Pole sweep counts the inner bondline as hub wall",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Pole sweep!C6",
            "Pole sweep!C7",
            "Pole sweep!C8",
            "Pole sweep!C9",
            "Pole sweep!C10",
            "Pole sweep!C11",
            "Calculator!C53",
        ],
        corrected_formula: "a_i = MAX(w_i/(2 tan(pi/N)) + 0.05, bore/2 + keyway + 2.5 + inner bondline).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: Some("tests/data/deviations/E4.json"),
        probes: &[],
    },
    Deviation {
        id: DeviationId::E5,
        title: "Rear-web eddy loss uses the free-space field instead of the field at the steel surface",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Temperature design!C121",
            "Temperature design!C125",
            "Temperature design!C130",
        ],
        corrected_formula: "Multiply the web integral by 4 in fields3d.run; default \
            Temperature design!C121 = 4.14e-5 T^2 m^2 (3D, doubled at the steel surface); C125 unchanged. \
            M2 applies the default; the factor 4 inside fields3d.run is M3 scope.",
        workbook_input_defaults: &[(
            "temperature.slip_loss.web_integral_T2m2",
            Literal::Num(1.035e-5),
        )],
        workbook_help: &[("temperature.slip_loss.web_integral_T2m2", "3D.")],
        changes_at_defaults: &[],
        changes_file: Some("tests/data/deviations/E5.json"),
        probes: &[],
    },
    Deviation {
        id: DeviationId::E6,
        title: "Cup wall at the flats counts the outer bondline as steel",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &["Calculator!C63"],
        corrected_formula: "Calculator!C63 = C62/2 - (C60 + Metal design!C121).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[CellChange {
            cell: "Calculator!C63",
            workbook: Literal::Num(2.773232302834515),
            corrected: Literal::Num(2.7232323028345142),
        }],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E7,
        title: "Pull-out is taken at half a pole pitch, which is not always the maximum",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C76",
            "Calculator!C82",
            "Calculator!C88",
            "Calculator!C89",
            "Calculator!C90",
            "Calculator!C91",
            "Calculator!C92",
            "Calculator!C93",
            "Calibration!C40",
            "Calibration!C41",
            "Calibration!C42",
        ],
        corrected_formula: "Pull-out = max over electrical angle x of sum_n A_n sin(n x), \
            A_n = B_in,n B_on,n S_n / (2 mu0), in both circuits, the sweeps and the calibration; \
            optionally report the pull-out angle. Default outputs do not change.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "6 poles on the default hub, steel circuit (report: 0.861 -> 0.911 N m)",
                inputs: &[("coupling.npole", Literal::Int(6))],
                expect: &[
                    CellChange {
                        cell: "Calculator!C89",
                        workbook: Literal::Num(69917.71639509343),
                        corrected: Literal::Num(73953.7345241203),
                    },
                    CellChange {
                        cell: "Calculator!C93",
                        workbook: Literal::Num(0.8611577121363674),
                        corrected: Literal::Num(0.9108682621562417),
                    },
                    CellChange {
                        cell: "Gap sweep!X6",
                        workbook: Literal::Num(0.94859431918547),
                        corrected: Literal::Num(1.038687728764382),
                    },
                ],
            },
            Probe {
                label: "12-magnet no-iron prototype (calibration)",
                inputs: &[("calibration.total_magnets", Literal::Int(12))],
                expect: &[CellChange {
                    cell: "Calibration!C44",
                    workbook: Literal::Num(0.3518120782396669),
                    corrected: Literal::Num(0.4420296708028627),
                }],
            },
        ],
    },
    Deviation {
        id: DeviationId::E8,
        title: "Arc-magnet mode reuses flat-block corner geometry in three formulas",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C9",
            "Calculator!C57",
            "Metal design!C175",
            "Calculator!C111",
        ],
        corrected_formula: "Use the inner corner radius C55 everywhere: C9 = face gap - (C55 - face radius); \
            Metal design!C175 = 2 (C55 + bedding); C111 cavity = pi ap^2 for arcs.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "arc magnets (coupling.faceted = 0), the report's arc-mode rerun",
            inputs: &[("coupling.faceted", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Calculator!C9",
                    workbook: Literal::Num(1.0268256054339253),
                    corrected: Literal::Num(1.4),
                },
                CellChange {
                    cell: "Calculator!C93",
                    workbook: Literal::Num(2.9915853633580882),
                    corrected: Literal::Num(2.6472742027214955),
                },
                CellChange {
                    cell: "Calculator!C111",
                    workbook: Literal::Num(42.955452974403656),
                    corrected: Literal::Num(48.40907404648711),
                },
                CellChange {
                    cell: "Metal design!C11",
                    workbook: Literal::Text("Estimate covers hot min"),
                    corrected: Literal::Text("Below hot minimum"),
                },
                CellChange {
                    cell: "Metal design!C35",
                    workbook: Literal::Num(-0.4763487891321485),
                    corrected: Literal::Num(0.2700000000000007),
                },
                CellChange {
                    cell: "Metal design!C37",
                    workbook: Literal::Text("Below target"),
                    corrected: Literal::Text("Meets assumed target"),
                },
                CellChange {
                    cell: "Metal design!C175",
                    workbook: Literal::Num(27.43634878913215),
                    corrected: Literal::Num(26.69),
                },
                CellChange {
                    cell: "Shaft clamps!C48",
                    workbook: Literal::Text("None: enlarge the boss or the clamp length"),
                    corrected: Literal::Text("ISO 4762 M4 x 12, class 12.9"),
                },
            ],
        }],
    },
    Deviation {
        id: DeviationId::E9,
        title: "'No back iron' is applied to the torque but not to the cup's mass and wall check",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C6",
            "Calculator!C111",
            "Calculator!C113",
            "Materials!C22",
        ],
        corrected_formula: "Gate the cup and boss densities on C6 as the hub already is; \
            Materials!C22 returns 'No back iron' when C6 = 0; carry the masses into the thermal heat capacity.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Calculator!C111",
                    workbook: Literal::Num(60.753684123800234),
                    corrected: Literal::Num(20.896171609459955),
                },
                CellChange {
                    cell: "Calculator!C113",
                    workbook: Literal::Num(30.777554908688483),
                    corrected: Literal::Num(10.585910605536167),
                },
                CellChange {
                    cell: "Calculator!C114",
                    workbook: Literal::Num(156.86088643049283),
                    corrected: Literal::Num(96.81172961300027),
                },
                CellChange {
                    cell: "Temperature design!C141",
                    workbook: Literal::Num(74.18527010448489),
                    corrected: Literal::Num(45.78201892981087),
                },
                CellChange {
                    cell: "Materials!C22",
                    workbook: Literal::Text("Too thin: raise Metal design C122 to at least 2.0 mm"),
                    corrected: Literal::Text("No back iron"),
                },
            ],
        }],
    },
    Deviation {
        id: DeviationId::E10,
        title: "Gap flux density averages Br instead of summing each magnet's MMF",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C103",
            "Calculator!C104",
            "Calculator!C105",
            "Calculator!C106",
            "Materials!C20",
            "Materials!C21",
            "Materials!C22",
        ],
        corrected_formula: "B_gap = (Br_i t_i + Br_o t_o) / (t_i + t_o + g).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "N52 inner (3.17 mm) with a B861 outer (1.59 mm)",
            inputs: &[
                ("coupling.magnets.part_inner", Literal::Text("B842-N52")),
                ("coupling.magnets.part_outer", Literal::Text("B861")),
            ],
            expect: &[
                CellChange {
                    cell: "Calculator!C103",
                    workbook: Literal::Num(1.0242499999999999),
                    corrected: Literal::Num(1.0427944805194804),
                },
                CellChange {
                    cell: "Calculator!C104",
                    workbook: Literal::Num(1.9146646666666662),
                    corrected: Literal::Num(1.9493304822510817),
                },
                CellChange {
                    cell: "Materials!C20",
                    workbook: Literal::Num(1.9146646666666662),
                    corrected: Literal::Num(1.9493304822510817),
                },
            ],
        }],
    },
    Deviation {
        id: DeviationId::E11,
        title: "22 °C adhesive fatigue screen ignores the fatigue-endurance input",
        class: DeviationClass::Engine,
        status: DeviationStatus::Planned,
        cells: &["Temperature design!C91", "Temperature design!C195"],
        corrected_formula: "C91 margin = C195 x C78 / C86.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E12,
        title: "No guard for a hot-day start already above the temperature limit",
        class: DeviationClass::Engine,
        status: DeviationStatus::Planned,
        cells: &[
            "Temperature design!C19",
            "Temperature design!C23",
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C152",
            "Temperature design!C153",
        ],
        corrected_formula: "Times to limit = 0 when T0 >= T_lim, tested before the 'never' branch; \
            critical drag = max(0, T_lim - T0) G / omega.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E13,
        title: "A measured drag of exactly 0 stops the whole calculation",
        class: DeviationClass::Engine,
        status: DeviationStatus::Planned,
        cells: &["Temperature design!C156", "Temperature design!C157"],
        corrected_formula: "Rotations per °C = +inf when the heating power is 0; \
            optionally reject a negative drag; document that infinity can appear in results.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E14,
        title: "README and help text say only M3 fits a 22 mm boss",
        class: DeviationClass::Documentation,
        status: DeviationStatus::Planned,
        cells: &["Shaft clamps!C35"],
        corrected_formula: "Reword the README and the Shaft clamps!C35 help: \"At 22 mm M4 no longer fits. \
            Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator \
            recommends M2.5 x 3.\"",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
];

/// A set of deviations switched on. Users only ever get [`Deviations::ALL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deviations {
    mask: u32,
}

impl Deviations {
    /// Every approved correction: what every user-facing computation uses.
    pub const ALL: Self = Self {
        mask: (1 << DeviationId::ALL.len()) - 1,
    };

    /// Whether the correction `id` applies.
    pub const fn is_on(self, id: DeviationId) -> bool {
        self.mask & id.bit() != 0
    }
}

#[cfg(feature = "workbook-parity")]
impl Deviations {
    /// TEST-ONLY. No correction: results reproduce the workbook exactly.
    pub const NONE: Self = Self { mask: 0 };

    /// TEST-ONLY. Exactly one correction, to check it in isolation.
    pub const fn only(id: DeviationId) -> Self {
        Self { mask: id.bit() }
    }
}

/// TEST-ONLY. Puts back the workbook default of every input that a deviation
/// switched off in `dev` corrects (used by `DesignInputs::defaults_with`).
///
/// Panics if the registry names an input path or value the inputs refuse,
/// which a registry test catches.
#[cfg(feature = "workbook-parity")]
pub fn restore_workbook_defaults<T: InputSet>(
    inputs: &mut T,
    dev: Deviations,
    registry: &[Deviation],
) {
    for deviation in registry.iter().filter(|d| !dev.is_on(d.id)) {
        for &(path, value) in deviation.workbook_input_defaults {
            if let Err(e) = inputs.set(path, value.to_value()) {
                panic!("{} restores an invalid workbook default: {e}", deviation.id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_in_report_order_with_one_entry_per_id() {
        assert_eq!(REGISTRY.len(), DeviationId::ALL.len());
        for (i, (entry, id)) in REGISTRY.iter().zip(DeviationId::ALL).enumerate() {
            assert_eq!(entry.id, id);
            assert_eq!(id.index(), i);
            assert_eq!(id.to_string(), format!("E{}", i + 1));
        }
    }

    #[test]
    fn every_entry_is_complete() {
        for d in REGISTRY {
            assert!(
                !d.title.is_empty() && !d.corrected_formula.is_empty(),
                "{}",
                d.id
            );
            assert!(!d.cells.is_empty(), "{} names no cell", d.id);
            assert!(d.evidence().starts_with(REPORT), "{}", d.id);
            assert!(d.evidence().ends_with(&format!("entry {}", d.id)));
        }
    }

    #[test]
    fn planned_entries_change_nothing_yet() {
        for d in REGISTRY
            .iter()
            .filter(|d| d.status == DeviationStatus::Planned)
        {
            assert!(
                d.changes_at_defaults.is_empty(),
                "{} is Planned but lists changes",
                d.id
            );
            assert!(
                d.workbook_input_defaults.is_empty(),
                "{} is Planned but restores defaults",
                d.id
            );
            assert!(
                d.workbook_help.is_empty(),
                "{} is Planned but rewords help",
                d.id
            );
            assert!(
                d.changes_file.is_none(),
                "{} is Planned but names a golden file",
                d.id
            );
            assert!(d.probes.is_empty(), "{} is Planned but has probes", d.id);
        }
    }

    #[test]
    fn documentation_entries_change_no_number() {
        for d in REGISTRY
            .iter()
            .filter(|d| d.class == DeviationClass::Documentation)
        {
            assert!(
                d.changes_at_defaults.is_empty() && d.workbook_input_defaults.is_empty(),
                "{}",
                d.id
            );
        }
        assert_eq!(
            REGISTRY[DeviationId::E14.index()].class,
            DeviationClass::Documentation
        );
    }

    #[test]
    fn all_switches_every_deviation_on() {
        for id in DeviationId::ALL {
            assert!(Deviations::ALL.is_on(id), "{id}");
        }
    }

    #[test]
    fn none_and_only_select_exactly_what_they_say() {
        for id in DeviationId::ALL {
            assert!(!Deviations::NONE.is_on(id));
            for other in DeviationId::ALL {
                assert_eq!(
                    Deviations::only(id).is_on(other),
                    id == other,
                    "only({id}) at {other}"
                );
            }
        }
    }

    #[test]
    fn literal_converts_to_value() {
        assert_eq!(Literal::Num(1.5).to_value(), Value::Num(1.5));
        assert_eq!(Literal::Int(3).to_value(), Value::Int(3));
        assert_eq!(Literal::Text("n/a").to_value(), Value::Text("n/a".into()));
        assert_eq!(Literal::None.to_value(), Value::None);
    }

    mod restore {
        use super::super::*;
        use crate::engine::meta::{inputs, param};

        inputs! {
            pub struct Probe {
                fields {
                    br_T: f64 = 1.30 => param("T", "Br", "", "X!C1"),
                    code: i64 = 1 => param("-", "Code", "", "X!C2").choices(&[(0, "a"), (1, "b")]),
                }
            }
        }

        const FAKE: &[Deviation] = &[Deviation {
            id: DeviationId::E3,
            title: "fake",
            class: DeviationClass::Engine,
            status: DeviationStatus::Applied,
            cells: &["X!C1"],
            corrected_formula: "fake",
            workbook_input_defaults: &[("br_T", Literal::Num(1.29))],
            workbook_help: &[],
            changes_at_defaults: &[],
            changes_file: None,
            probes: &[],
        }];

        #[test]
        fn workbook_default_comes_back_only_when_the_deviation_is_off() {
            let mut off = Probe::default();
            restore_workbook_defaults(&mut off, Deviations::NONE, FAKE);
            assert_eq!(off.br_T, 1.29);

            let mut on = Probe::default();
            restore_workbook_defaults(&mut on, Deviations::ALL, FAKE);
            assert_eq!(on.br_T, 1.30);

            let mut other = Probe::default();
            restore_workbook_defaults(&mut other, Deviations::only(DeviationId::E1), FAKE);
            assert_eq!(other.br_T, 1.29, "E3 is off when only E1 is on");
        }

        #[test]
        #[should_panic(expected = "E3 restores an invalid workbook default")]
        fn an_invalid_registry_default_is_loud() {
            const BAD: &[Deviation] = &[Deviation {
                workbook_input_defaults: &[("code", Literal::Int(7))],
                ..FAKE[0]
            }];
            restore_workbook_defaults(&mut Probe::default(), Deviations::NONE, BAD);
        }
    }
}
