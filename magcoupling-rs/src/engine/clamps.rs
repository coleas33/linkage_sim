//! Shaft clamp sizing ('Shaft clamps' and 'Clamp screw sizes' sheets).
//!
//! Port of `reference/magcoupling-py/magcoupling/clamps.py`. Recommended
//! construction: a one-piece slotted clamp in 7075-T6 on a keyed shaft, one axial
//! slit through one wall plus a transverse relief cut, closed by tangential
//! ISO 4762 cap screws threaded straight into the far jaw.
//!
//! For each metric screw size (M2.5 to M6) the sheet checks geometry (screw
//! offset from the shaft axis, wall outside the hole, head seat, head-side jaw,
//! thread length in the far jaw), strength (preload from the screw proof load,
//! capped by aluminium thread stripping), capacity (clamp torque per screw) and
//! fit (screws needed against screws that fit along the clamp). The first size
//! that passes everything is recommended.
//!
//! This module holds the screw sizes ([`SCREW_SIZES`]), the inputs
//! ([`ClampInputs`], 25 cells), the 'Clamp screw sizes' table ([`ScrewRow`], 165
//! cells: 33 celled fields for each of the 5 sizes), the results
//! ([`ClampResults`], 33 result cells plus the Rust-only `length_note`),
//! [`MACHINING_STEPS`] and [`compute`].
//!
//! A screw class code outside 1 to 3 gives NaN numbers and the text `"#N/A"`
//! (decision D3, [`screw_class_name`] and `ScrewClasses::proof`): Python raises
//! `KeyError`.
//!
//! Deviations touching this sheet (see [`crate::engine::deviations::REGISTRY`]):
//! E2, applied (Clamp screw sizes rows 34 and 35, Shaft clamps!C48, and the
//! Rust-only `length_note`); E14, applied (the `boss_od_mm` help text only).

use std::f64::consts::PI;

use super::compat::{ceiling, floor_, fmt_fixed, fmt_num, py_min};
use super::deviations::{DeviationId, Deviations};
use super::materials::AluminiumAlloy;
use super::meta::{
    NumOrText, TableLayout, at_row, inputs, out, out_rust_only, param, results, rows, uncelled_col,
};

/// A metric screw size (Python `ScrewSize`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // Python's As_mm2
pub struct ScrewSize {
    pub name: &'static str,
    /// Nominal diameter.
    pub d_mm: f64,
    /// Coarse pitch.
    pub pitch_mm: f64,
    /// Tensile stress area.
    pub As_mm2: f64,
    /// Clearance hole, ISO 273 medium.
    pub hole_mm: f64,
    /// ISO 4762 head diameter (max).
    pub head_mm: f64,
    /// Head height.
    pub head_h_mm: f64,
    /// Hex key.
    pub hex_mm: f64,
}

/// The screw sizes of the sheet, M2.5 to M6.
pub const SCREW_SIZES: [ScrewSize; 5] = [
    ScrewSize {
        name: "M2.5",
        d_mm: 2.5,
        pitch_mm: 0.45,
        As_mm2: 3.39,
        hole_mm: 2.9,
        head_mm: 4.5,
        head_h_mm: 2.5,
        hex_mm: 2.0,
    },
    ScrewSize {
        name: "M3",
        d_mm: 3.0,
        pitch_mm: 0.5,
        As_mm2: 5.03,
        hole_mm: 3.4,
        head_mm: 5.5,
        head_h_mm: 3.0,
        hex_mm: 2.5,
    },
    ScrewSize {
        name: "M4",
        d_mm: 4.0,
        pitch_mm: 0.7,
        As_mm2: 8.78,
        hole_mm: 4.5,
        head_mm: 7.0,
        head_h_mm: 4.0,
        hex_mm: 3.0,
    },
    ScrewSize {
        name: "M5",
        d_mm: 5.0,
        pitch_mm: 0.8,
        As_mm2: 14.2,
        hole_mm: 5.5,
        head_mm: 8.5,
        head_h_mm: 5.0,
        hex_mm: 4.0,
    },
    ScrewSize {
        name: "M6",
        d_mm: 6.0,
        pitch_mm: 1.0,
        As_mm2: 20.1,
        hole_mm: 6.6,
        head_mm: 10.0,
        head_h_mm: 6.0,
        hex_mm: 5.0,
    },
];

/// 'Clamp screw sizes' columns for M2.5 to M6.
pub const TABLE_COLUMNS: [&str; 5] = ["C", "D", "E", "F", "G"];

// =========================================================================== inputs
inputs! {
    /// Shaft clamp inputs (Shaft clamps!C16:C66).
    pub struct ClampInputs {
        fields {
            safety_factor: f64 = 2.0 => param("-", "Safety factor, clamp alone",
                "Reversing torque and vibration; the key is extra.", "Shaft clamps!C16")
                .range(1.0, 5.0, 0.1),
            friction: f64 = 0.15 => param("-", "Friction coefficient, shaft to bore",
                "Degreased; 0.10 if the bore could be oily.", "Shaft clamps!C18")
                .range(0.05, 0.4, 0.005)
                .assumption(), // > 0: divides the screws needed
            clamp_type: i64 = 1 => param("-", "Clamp type",
                "1 = one-piece slotted, 2 = two-piece.", "Shaft clamps!C19")
                .choices(&[(1, "one-piece slotted"), (2, "two-piece")]),
            factor_one_piece: f64 = 0.8 => param("-", "Clamp factor, one-piece",
                "Share of screw force pressing the shaft.", "Shaft clamps!C20")
                .range(0.3, 1.0, 0.01), // > 0
            factor_two_piece: f64 = 1.0 => param("-", "Clamp factor, two-piece",
                "", "Shaft clamps!C21")
                .range(0.3, 1.2, 0.01), // > 0
            alloy: i64 = 1 => param("-", "Aluminium",
                "1 = 7075-T6, 2 = 6061-T6.", "Shaft clamps!C23")
                .choices(&[(1, "7075-T6"), (2, "6061-T6")]),
            screw_class: i64 = 1 => param("-", "Screw class",
                "1 = 12.9, 2 = 10.9, 3 = A4-70.", "Shaft clamps!C27")
                .choices(&[(1, "12.9"), (2, "10.9"), (3, "A4-70")]),
            preload_fraction: f64 = 0.75 => param("-", "Preload as a share of proof load",
                "", "Shaft clamps!C29")
                .range(0.3, 0.9, 0.01)
                .assumption(),
            nut_factor: f64 = 0.2 => param("-", "Nut factor",
                "Dry or with Loctite 243.", "Shaft clamps!C30")
                .range(0.1, 0.35, 0.01),
            engagement_x_d: f64 = 2.0 => param("-", "Thread engagement in aluminium (× screw diameter)",
                "", "Shaft clamps!C31")
                .range(0.5, 3.0, 0.05),
            strip_sf: f64 = 1.5 => param("-", "Safety factor on thread stripping",
                "", "Shaft clamps!C32")
                .range(1.0, 4.0, 0.1),
            boss_od_mm: f64 = 25.0 => param("mm", "Boss outside diameter",
                "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3.", "Shaft clamps!C35")
                .range(12.0, 60.0, 0.1),
            clamp_length_mm: f64 = 10.0 => param("mm", "Clamp length, free end to relief cut",
                "", "Shaft clamps!C36")
                .range(3.0, 40.0, 0.1), // > 0: divides the key pressure
            slit_mm: f64 = 0.8 => param("mm", "Slit width",
                "", "Shaft clamps!C37")
                .range(0.2, 3.0, 0.1),
            ligament_mm: f64 = 1.0 => param("mm", "Minimum ligament, bore to screw hole",
                "", "Shaft clamps!C38")
                .range(0.3, 5.0, 0.1),
            wall_out_mm: f64 = 1.0 => param("mm", "Minimum wall outside the screw hole",
                "", "Shaft clamps!C39")
                .range(0.3, 5.0, 0.1),
            grip_min_mm: f64 = 3.0 => param("mm", "Minimum head-side jaw (grip)",
                "", "Shaft clamps!C40")
                .range(0.5, 10.0, 0.1),
            axial_margin_mm: f64 = 1.0 => param("mm", "Axial margin, clamp end to counterbore edge",
                "", "Shaft clamps!C41")
                .range(0.0, 5.0, 0.1),
            relief_mm: f64 = 1.0 => param("mm", "Relief cut width",
                "", "Shaft clamps!C42")
                .range(0.3, 5.0, 0.1),
            hinge_mm: f64 = 8.0 => param("mm", "Hinge left under the relief cut",
                "Measured up from the far side of the boss.", "Shaft clamps!C43")
                .range(1.0, 30.0, 0.1),
            key_width_mm: f64 = 4.0 => param("mm", "Key width",
                "", "Shaft clamps!C58")
                .range(1.0, 10.0, 0.5),
            key_contact_mm: f64 = 1.5 => param("mm", "Key contact height in the hub",
                "", "Shaft clamps!C59")
                .range(0.5, 5.0, 0.1), // > 0: divides the key pressure
            joint_screws: i64 = 3 => param("-", "Adapter joint screws",
                "Pilot + 3 × M3 + dowel.", "Shaft clamps!C64")
                .range(1.0, 8.0, 1.0),
            joint_bolt_circle_mm: f64 = 26.0 => param("mm", "Bolt circle diameter",
                "", "Shaft clamps!C65")
                .range(10.0, 60.0, 0.5),
            joint_friction: f64 = 0.15 => param("-", "Friction coefficient, nickel plate on aluminium",
                "", "Shaft clamps!C66")
                .range(0.05, 0.4, 0.005),
        }
    }
}

// =========================================================================== table
rows! {
    /// One column of 'Clamp screw sizes' (rows 6-38), one per screw size.
    pub struct ScrewRow {
        size: String => uncelled_col("", "Screw size", ""),
        d_mm: f64 => at_row("mm", "Nominal diameter", "Standard data.", 6),
        pitch_mm: f64 => at_row("mm", "Thread pitch (coarse)", "", 7),
        As_mm2: f64 => at_row("mm²", "Tensile stress area", "", 8),
        hole_mm: f64 => at_row("mm", "Clearance hole (medium)", "ISO 273.", 9),
        head_mm: f64 => at_row("mm", "Head diameter", "ISO 4762 maximum.", 10),
        head_h_mm: f64 => at_row("mm", "Head height", "", 11),
        hex_mm: f64 => at_row("mm", "Hex key", "", 12),
        tap_drill_mm: f64 => at_row("mm", "Tap drill", "Nominal minus pitch.", 13),
        offset_mm: f64 => at_row("mm", "Screw offset from the shaft axis",
            "Half the bore, plus the ligament, plus half the clearance hole.", 14),
        wall_out_mm: f64 => at_row("mm", "Wall outside the screw hole", "", 15),
        head_fits: i64 => at_row("-", "Head seat fits inside the boss (1 = yes)", "", 16),
        grip_mm: f64 => at_row("mm", "Head-side jaw (grip)", "From the head seat to the slit.", 17),
        thread_avail_mm: f64 => at_row("mm", "Thread length available in the far jaw", "", 18),
        engagement_req_mm: f64 => at_row("mm", "Thread engagement required", "", 19),
        geometry_ok: i64 => at_row("-", "Geometry fits (1 = yes)", "Wall, head seat, grip and engagement all pass.", 20),
        preload_strength_N: f64 => at_row("N", "Preload from screw strength", "Share of proof load times stress area.", 21),
        preload_strip_N: f64 => at_row("N", "Preload limit from thread stripping", "Shear area about 0.6·π·d·engagement.", 22),
        preload_N: f64 => at_row("N", "Allowable preload", "The smaller of the two.", 23),
        head_pressure_MPa: f64 => at_row("MPa", "Pressure under the head", "", 24),
        head_check: String => at_row("", "Head pressure check", "", 25),
        torque_per_screw_Nm: f64 => at_row("N·m", "Clamp torque per screw",
            "Friction x screw force x shaft diameter x clamp factor.", 26),
        screws_needed: i64 => at_row("-", "Screws needed", "", 27),
        pitch_axial_mm: f64 => at_row("mm", "Axial screw spacing", "Head diameter plus 1 mm.", 28),
        screws_fit: i64 => at_row("-", "Screws that fit along the clamp", "", 29),
        works: i64 => at_row("-", "Size works (1 = yes)", "Geometry fits and enough screws fit.", 30),
        clamp_torque_Nm: f64 => at_row("N·m", "Clamp torque with the screws needed", "", 31),
        sf_coupling: f64 => at_row("-", "Safety factor on the coupling torque", "", 32),
        tightening_Nm: f64 => at_row("N·m", "Tightening torque", "Nut factor x preload x diameter.", 33),
        length_mm: f64 => at_row("mm", "Screw length", "Grip plus slit plus required engagement, rounded up to an even length (correction E2).", 34),
        length_ok: i64 => at_row("-", "Length stays inside the far jaw (1 = yes)", "", 35),
        cbore_dia_mm: f64 => at_row("mm", "Counterbore diameter", "Head plus 0.5 mm.", 36),
        cbore_depth_mm: f64 => at_row("mm", "Counterbore depth at the screw axis", "From the OD to the head seat.", 37),
        vent_port_ok: i64 => at_row("-", "Key passes the M6 vent port (1 = yes)", "4 mm key or smaller.", 38),
    }
}

// =========================================================================== results
results! {
    /// Shaft clamps sheet results (Shaft clamps!C14:C81) and the screw table.
    pub struct ClampResults {
        fields {
            shaft_mm: f64 => out("mm", "Shaft diameter", "", "Shaft clamps!C14"),
            max_torque_Nm: f64 => out("N·m", "Highest torque through the coupling",
                "Cold pull-out with +variation.", "Shaft clamps!C15"),
            required_Nm: f64 => out("N·m", "Torque the clamp must hold", "", "Shaft clamps!C17"),
            clamp_factor: f64 => out("-", "Clamp factor used", "", "Shaft clamps!C22"),
            al_shear_MPa: f64 => out("MPa", "Aluminium shear strength", "", "Shaft clamps!C24"),
            al_head_limit_MPa: f64 => out("MPa", "Limiting pressure under the screw head", "", "Shaft clamps!C25"),
            al_key_allow_MPa: f64 => out("MPa", "Key bearing allowable", "", "Shaft clamps!C26"),
            screw_proof_MPa: f64 => out("MPa", "Screw proof stress", "", "Shaft clamps!C28"),
            boss_radius_mm: f64 => out("mm", "Boss radius", "", "Shaft clamps!C44"),
            index: i64 => out("-", "Size index in the table", "0 = none fits.", "Shaft clamps!C47"),
            recommended: String => out("", "Recommended screw", "", "Shaft clamps!C48"),
            length_note: String => out_rust_only("", "Screw length note",
                "Set when no 2 mm length step of the recommended size both engages the required thread and stays inside the boss (correction E2); empty otherwise."),
            screws: NumOrText => out("-", "Screws per clamp", "", "Shaft clamps!C49"),
            tightening_Nm: NumOrText => out("N·m", "Tightening torque", "With Loctite 243.", "Shaft clamps!C50"),
            hex_mm: NumOrText => out("mm", "Hex key", "", "Shaft clamps!C51"),
            capacity_Nm: NumOrText => out("N·m", "Clamp torque capacity", "", "Shaft clamps!C52"),
            sf_coupling: NumOrText => out("-", "Safety factor on the coupling torque",
                "Clamp alone, before the key.", "Shaft clamps!C53"),
            head_check: String => out("", "Head pressure on the aluminium", "", "Shaft clamps!C54"),
            vent_port: String => out("", "Through the M6 vent port", "", "Shaft clamps!C55"),
            key_pressure_MPa: f64 => out("MPa", "Key bearing pressure on the hub at the clamp design torque",
                "", "Shaft clamps!C60"),
            key_sf: f64 => out("x", "Key alone: allowable over actual", "", "Shaft clamps!C61"),
            joint_preload_N: f64 => out("N", "Allowable preload per M3", "", "Shaft clamps!C67"),
            joint_torque_Nm: f64 => out("N·m", "Joint slip torque", "", "Shaft clamps!C68"),
            joint_sf: f64 => out("x", "Joint torque over the clamp design torque", "", "Shaft clamps!C69"),
            layout_offset_mm: NumOrText => out("mm", "Screw offset from the shaft axis", "", "Shaft clamps!C72"),
            layout_pitch_mm: NumOrText => out("mm", "Screw spacing", "", "Shaft clamps!C73"),
            layout_first_mm: NumOrText => out("mm", "First screw from the free end", "", "Shaft clamps!C74"),
            layout_cbore_dia_mm: NumOrText => out("mm", "Counterbore diameter (head side)", "", "Shaft clamps!C75"),
            layout_cbore_depth_mm: NumOrText => out("mm", "Counterbore depth at the screw axis, from the OD",
                "", "Shaft clamps!C76"),
            layout_grip_mm: NumOrText => out("mm", "Head-side jaw (grip)", "", "Shaft clamps!C77"),
            layout_tap_drill_mm: NumOrText => out("mm", "Tap drill, far jaw", "", "Shaft clamps!C78"),
            layout_thread_avail_mm: NumOrText => out("mm", "Thread length available in the far jaw",
                "", "Shaft clamps!C79"),
            layout_slit: String => out("", "Slit", "", "Shaft clamps!C80"),
            layout_relief: String => out("", "Relief cut", "", "Shaft clamps!C81"),
        }
        tables {
            table: ScrewRow => TableLayout::ColumnsAcross { sheet: "Clamp screw sizes", columns: &TABLE_COLUMNS },
        }
    }
}

/// The machining sequence (Python `MACHINING_STEPS`), for the drawing sheet.
pub const MACHINING_STEPS: [&str; 4] = [
    "1. Turn the OD and bore (H7) in one setup; cut the keyway 90° from where the slit will go.",
    concat!(
        "2. With the part still solid, drill the clearance hole and counterbore on the head-side jaw, then drill and tap the far jaw, ",
        "square to the slit plane at the screw offset. Drilling before slitting keeps the holes aligned."
    ),
    "3. Cut the relief slot, then the slit (slitting saw), leaving the hinge.",
    "4. Deburr, anodize, chase the thread, and assemble dry with Loctite 243 on the screw at the tightening torque.",
];

/// The recommendation when no size works (Shaft clamps!C48).
pub const NONE_FITS: &str = "None: enlarge the boss or the clamp length";

/// Screw class text for Shaft clamps!C48 (workbook CHOOSE order). Python raises
/// KeyError outside 1-3; here `"#N/A"` (decision D3).
pub fn screw_class_name(code: i64) -> &'static str {
    match code {
        1 => "12.9",
        2 => "10.9",
        3 => "A4-70",
        _ => "#N/A",
    }
}

/// Shaft clamps and Clamp screw sizes. `al_props`: the selected alloy.
#[allow(non_snake_case)] // Python names (R, T_req, Fb, Fs, F, Tper, T)
pub fn compute(
    ci: &ClampInputs,
    shaft_mm: f64,
    max_torque_Nm: f64,
    al_props: &AluminiumAlloy,
    screw_proof_MPa: f64,
    dev: Deviations,
) -> ClampResults {
    let R = ci.boss_od_mm / 2.0;
    let T_req = max_torque_Nm * ci.safety_factor;
    let k = if ci.clamp_type == 1 {
        ci.factor_one_piece
    } else {
        ci.factor_two_piece
    };
    let rows: Vec<ScrewRow> = SCREW_SIZES
        .iter()
        .map(|s| {
            let e = shaft_mm / 2.0 + ci.ligament_mm + s.hole_mm / 2.0;
            let wall = R - e - s.hole_mm / 2.0;
            let fits: i64 = if e + s.head_mm / 2.0 <= R { 1 } else { 0 };
            let grip = if fits == 1 {
                (R.powi(2) - (e + s.head_mm / 2.0).powi(2)).sqrt() - ci.slit_mm / 2.0
            } else {
                0.0
            };
            let avail = if e < R {
                (R.powi(2) - e.powi(2)).sqrt() - ci.slit_mm / 2.0
            } else {
                0.0
            };
            let ereq = ci.engagement_x_d * s.d_mm;
            let geo: i64 =
                if wall >= ci.wall_out_mm && fits == 1 && grip >= ci.grip_min_mm && avail >= ereq {
                    1
                } else {
                    0
                };
            let Fb = ci.preload_fraction * screw_proof_MPa * s.As_mm2;
            let Fs = 0.6 * PI * s.d_mm * ereq * al_props.shear_MPa / ci.strip_sf;
            let F = py_min(Fb, Fs);
            let p = F / (PI / 4.0 * (s.head_mm.powi(2) - s.hole_mm.powi(2)));
            let Tper = ci.friction * F * shaft_mm / 1000.0 * k;
            // Python int(ceiling(..)): the saturating cast never panics (NaN gives 0).
            let need = ceiling(T_req / Tper, 1.0) as i64;
            let pitch = s.head_mm + 1.0;
            let span = ci.clamp_length_mm - 2.0 * ci.axial_margin_mm - (s.head_mm + 0.5);
            let fit = if span >= 0.0 {
                (floor_(span / pitch, 1.0) as i64).saturating_add(1)
            } else {
                0
            };
            let works: i64 = if geo == 1 && need <= fit { 1 } else { 0 };
            // E2: the screw crosses the open slit before it reaches the far jaw.
            let e2 = dev.is_on(DeviationId::E2);
            let length = if fits == 1 {
                if e2 {
                    ceiling(grip + ci.slit_mm + ereq, 2.0)
                } else {
                    ceiling(grip + ereq, 2.0)
                }
            } else {
                0.0
            };
            let inside = if e2 {
                grip + ci.slit_mm + avail
            } else {
                grip + avail
            };
            ScrewRow {
                size: s.name.to_owned(),
                d_mm: s.d_mm,
                pitch_mm: s.pitch_mm,
                As_mm2: s.As_mm2,
                hole_mm: s.hole_mm,
                head_mm: s.head_mm,
                head_h_mm: s.head_h_mm,
                hex_mm: s.hex_mm,
                tap_drill_mm: s.d_mm - s.pitch_mm,
                offset_mm: e,
                wall_out_mm: wall,
                head_fits: fits,
                grip_mm: grip,
                thread_avail_mm: avail,
                engagement_req_mm: ereq,
                geometry_ok: geo,
                preload_strength_N: Fb,
                preload_strip_N: Fs,
                preload_N: F,
                head_pressure_MPa: p,
                head_check: if p <= al_props.head_pressure_limit_MPa {
                    "OK"
                } else {
                    "Use a hardened washer"
                }
                .to_owned(),
                torque_per_screw_Nm: Tper,
                screws_needed: need,
                pitch_axial_mm: pitch,
                screws_fit: fit,
                works,
                clamp_torque_Nm: need as f64 * Tper,
                sf_coupling: need as f64 * Tper / max_torque_Nm,
                tightening_Nm: ci.nut_factor * F * s.d_mm / 1000.0,
                length_mm: length,
                length_ok: if fits == 1 && length <= inside { 1 } else { 0 },
                cbore_dia_mm: s.head_mm + 0.5,
                cbore_depth_mm: if fits == 1 {
                    (R.powi(2) - e.powi(2)).sqrt()
                        - (R.powi(2) - (e + s.head_mm / 2.0).powi(2)).sqrt()
                } else {
                    0.0
                },
                vent_port_ok: if s.hex_mm <= 4.0 { 1 } else { 0 },
            }
        })
        .collect();

    let index = rows
        .iter()
        .position(|r| r.works == 1)
        .map_or(0, |i| i as i64 + 1);
    let chosen = rows.iter().find(|r| r.works == 1);
    let recommended = match chosen {
        Some(r) => format!(
            "ISO 4762 {} x {}, class {}",
            r.size,
            fmt_num(r.length_mm),
            screw_class_name(ci.screw_class)
        ),
        None => NONE_FITS.to_owned(),
    };
    let length_note = match chosen {
        Some(r) if dev.is_on(DeviationId::E2) && r.length_ok == 0 => format!(
            "No 2 mm length step of {} both engages {} mm of thread and stays inside the boss; {} x {} protrudes {} mm",
            r.size,
            fmt_num(r.engagement_req_mm),
            r.size,
            fmt_num(r.length_mm),
            fmt_fixed(
                r.length_mm - (r.grip_mm + ci.slit_mm + r.thread_avail_mm),
                2
            ),
        ),
        _ => String::new(),
    };
    let blank = NumOrText::Text("");
    let pick = |value: fn(&ScrewRow) -> f64| chosen.map_or(blank, |r| NumOrText::Num(value(r)));
    let key_p = 2.0 * T_req
        / ((shaft_mm / 1000.0) * (ci.key_contact_mm / 1000.0) * (ci.clamp_length_mm / 1000.0))
        / 1e6;
    let m3 = &rows[1]; // Python rows[1]: the M3 row
    let joint_T =
        ci.joint_friction * ci.joint_screws as f64 * m3.preload_N * ci.joint_bolt_circle_mm
            / 2000.0;
    ClampResults {
        shaft_mm,
        max_torque_Nm,
        required_Nm: T_req,
        clamp_factor: k,
        al_shear_MPa: al_props.shear_MPa,
        al_head_limit_MPa: al_props.head_pressure_limit_MPa,
        al_key_allow_MPa: al_props.key_bearing_allow_MPa,
        screw_proof_MPa,
        boss_radius_mm: R,
        index,
        recommended,
        length_note,
        screws: pick(|r| r.screws_needed as f64),
        tightening_Nm: pick(|r| r.tightening_Nm),
        hex_mm: pick(|r| r.hex_mm),
        capacity_Nm: pick(|r| r.clamp_torque_Nm),
        sf_coupling: pick(|r| r.sf_coupling),
        head_check: chosen.map_or(String::new(), |r| r.head_check.clone()),
        vent_port: chosen.map_or(String::new(), |r| {
            let text = if r.hex_mm <= 4.0 {
                "Yes: the key fits the 4 mm limit"
            } else {
                "No: key too large"
            };
            text.to_owned()
        }),
        key_pressure_MPa: key_p,
        key_sf: al_props.key_bearing_allow_MPa / key_p,
        joint_preload_N: m3.preload_N,
        joint_torque_Nm: joint_T,
        joint_sf: joint_T / T_req,
        layout_offset_mm: pick(|r| r.offset_mm),
        layout_pitch_mm: pick(|r| r.pitch_axial_mm),
        layout_first_mm: chosen.map_or(blank, |r| {
            NumOrText::Num(
                (ci.clamp_length_mm - (r.screws_needed as f64 - 1.0) * r.pitch_axial_mm) / 2.0,
            )
        }),
        layout_cbore_dia_mm: pick(|r| r.cbore_dia_mm),
        layout_cbore_depth_mm: pick(|r| r.cbore_depth_mm),
        layout_grip_mm: pick(|r| r.grip_mm),
        layout_tap_drill_mm: pick(|r| r.tap_drill_mm),
        layout_thread_avail_mm: pick(|r| r.thread_avail_mm),
        layout_slit: format!(
            "{} mm wide, bore to OD on one side, free end to the relief cut",
            fmt_fixed(ci.slit_mm, 1)
        ),
        layout_relief: format!(
            "{} mm wide at {} mm from the free end, {} mm deep from the slit side",
            fmt_fixed(ci.relief_mm, 1),
            fmt_fixed(ci.clamp_length_mm, 1),
            fmt_fixed(ci.boss_od_mm - ci.hinge_mm, 1)
        ),
        table: rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::materials::AL7075;

    /// A run with a 10 mm shaft and the default cold-high torque (3.7647 N·m).
    fn run(ci: &ClampInputs, alloy: &AluminiumAlloy) -> ClampResults {
        compute(ci, 10.0, 3.7647, alloy, 970.0, Deviations::NONE)
    }

    /// The M4 column of `run` (the size that works at the defaults).
    fn m4(ci: &ClampInputs) -> ScrewRow {
        let row = run(ci, &AL7075).table[2].clone();
        assert_eq!(row.size, "M4");
        row
    }

    #[test]
    fn size_checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8, on the M4 column (it works at the defaults). Each
        // right-hand side is an input its left-hand side does not read: one run supplies it.
        let m4 = m4(&ClampInputs::default());
        let ci = ClampInputs {
            wall_out_mm: m4.wall_out_mm, // wall >= wall_out
            grip_min_mm: m4.grip_mm,     // grip >= grip_min
            // avail >= ereq, ereq = engagement_x_d * d with d = 4 mm: dividing and multiplying by 4 is exact
            engagement_x_d: m4.thread_avail_mm / 4.0,
            // span >= 0: 9.5 - 2 * 1.0 - (7.0 + 0.5) = 0 exactly (axial margin 1 mm, M4 head 7 mm)
            clamp_length_mm: 9.5,
            ..ClampInputs::default()
        };
        // p <= head limit: the head pressure reads the engagement (stripping preload), so take it from this ci.
        let p = run(&ci, &AL7075).table[2].head_pressure_MPa;
        let alloy = AluminiumAlloy {
            head_pressure_limit_MPa: p,
            ..AL7075
        };
        let r = run(&ci, &alloy).table[2].clone();
        assert_eq!(
            (
                r.wall_out_mm,
                r.grip_mm,
                r.engagement_req_mm,
                r.head_pressure_MPa
            ),
            (ci.wall_out_mm, ci.grip_min_mm, r.thread_avail_mm, p)
        );
        assert_eq!(r.geometry_ok, 1);
        assert_eq!(r.screws_fit, 1, "a span of exactly 0 fits one screw");
        assert_eq!(r.head_check, "OK");
    }

    #[test]
    fn head_seat_fits_when_its_edge_is_exactly_on_the_boss_radius() {
        // M4: offset 10 / 2 + 1.0 + 4.5 / 2 = 8.25, head edge 8.25 + 7.0 / 2 = 11.75, so a 23.5 mm
        // boss (R = 11.75) puts the edge exactly on the wall. Every value is dyadic: exact.
        let ci = ClampInputs {
            boss_od_mm: 23.5,
            ..ClampInputs::default()
        };
        let r = m4(&ci);
        assert_eq!(r.offset_mm + r.head_mm / 2.0, ci.boss_od_mm / 2.0);
        assert_eq!(r.head_fits, 1);
        // Python takes the square-root branch at equality: sqrt(0) - slit / 2, not the 0 of `else`.
        assert_eq!(r.grip_mm, -ci.slit_mm / 2.0);
        // One ULP less boss: the head seat no longer fits.
        let below = ClampInputs {
            boss_od_mm: f64::from_bits(ci.boss_od_mm.to_bits() - 1),
            ..ClampInputs::default()
        };
        let r = m4(&below);
        assert!(r.offset_mm + r.head_mm / 2.0 > below.boss_od_mm / 2.0);
        assert_eq!((r.head_fits, r.grip_mm), (0, 0.0));
    }

    #[test]
    fn thread_is_available_only_while_the_offset_is_below_the_radius() {
        // M4 offset 8.25: a 16.5 mm boss (R = 8.25) puts the screw axis exactly on the wall.
        let ci = ClampInputs {
            boss_od_mm: 16.5,
            ..ClampInputs::default()
        };
        let r = m4(&ci);
        assert_eq!(r.offset_mm, ci.boss_od_mm / 2.0);
        // Python's `else` gives 0, where the square-root branch would give sqrt(0) - slit / 2 = -0.4.
        assert_eq!(r.thread_avail_mm, 0.0);
        // One ULP more boss: the square-root branch, which is negative this close to the wall.
        let above = ClampInputs {
            boss_od_mm: f64::from_bits(ci.boss_od_mm.to_bits() + 1),
            ..ClampInputs::default()
        };
        let r = m4(&above);
        assert!(r.offset_mm < above.boss_od_mm / 2.0);
        assert!(r.thread_avail_mm < 0.0);
    }

    #[test]
    fn a_size_works_when_the_screws_needed_equal_the_screws_that_fit() {
        // At the defaults M4 needs one screw and one fits.
        let r = m4(&ClampInputs::default());
        assert_eq!((r.geometry_ok, r.screws_needed, r.screws_fit), (1, 1, 1));
        assert_eq!(r.works, 1);
        // Twice the torque needs two screws: geometry still passes, one fits, so it no longer works.
        let ci = ClampInputs::default();
        let r = compute(&ci, 10.0, 2.0 * 3.7647, &AL7075, 970.0, Deviations::NONE).table[2].clone();
        assert_eq!((r.geometry_ok, r.screws_needed, r.screws_fit), (1, 2, 1));
        assert_eq!(r.works, 0);
    }

    #[test]
    fn screw_length_is_ok_when_it_equals_the_grip_plus_the_thread_available() {
        // Pythagorean values, all dyadic: shaft 10 mm and ligament 3.25 mm give offset 10.5 for M4,
        // a 35 mm boss (R = 17.5) gives sqrt(R^2 - 14^2) = 10.5 and sqrt(R^2 - 10.5^2) = 14, and a
        // 0.5 mm slit takes 0.25 from each: grip 10.25, thread available 13.75, sum 24 exactly.
        // Engagement 3 x 4 = 12 mm gives ceiling(10.25 + 12, 2) = 24: the length equals the sum.
        let ci = ClampInputs {
            boss_od_mm: 35.0,
            ligament_mm: 3.25,
            slit_mm: 0.5,
            engagement_x_d: 3.0,
            ..ClampInputs::default()
        };
        let r = m4(&ci);
        assert_eq!(r.head_fits, 1);
        assert_eq!(r.length_mm, r.grip_mm + r.thread_avail_mm);
        assert_eq!(r.length_mm, 24.0);
        assert_eq!(r.length_ok, 1);
        // 14 mm of engagement rounds the length up to 26: beyond the far jaw.
        let longer = ClampInputs {
            engagement_x_d: 3.5,
            ..ci
        };
        let r = m4(&longer);
        assert_eq!((r.length_mm, r.length_ok), (26.0, 0));
    }

    #[test]
    fn e2_screw_length_is_ok_when_it_equals_the_full_chord() {
        // Pythagorean values, all dyadic: shaft 6 mm and ligament 1 mm give offset 6.25 for M4, a
        // 32.5 mm boss (R = 16.25) gives sqrt(R^2 - 6.25^2) = 15 and sqrt(R^2 - 9.75^2) = 13, so the
        // chord from the head seat to the far OD is 28. A 0.5 mm slit takes 0.25 from each: grip
        // 12.75, thread available 14.75. Engagement 3.6875 x 4 = 14.75 makes grip + slit + engagement
        // exactly 28, so the corrected length equals the room the corrected check allows.
        let ci = ClampInputs {
            boss_od_mm: 32.5,
            slit_mm: 0.5,
            engagement_x_d: 3.6875,
            ..ClampInputs::default()
        };
        let m4_at = |ci: &ClampInputs, dev| {
            let row = compute(ci, 6.0, 3.7647, &AL7075, 970.0, dev).table[2].clone();
            assert_eq!(row.size, "M4");
            row
        };
        let e2 = Deviations::only(DeviationId::E2);
        let r = m4_at(&ci, e2);
        assert_eq!(
            (r.grip_mm, r.thread_avail_mm, r.engagement_req_mm),
            (12.75, 14.75, 14.75)
        );
        assert_eq!(r.length_mm, r.grip_mm + ci.slit_mm + r.thread_avail_mm);
        assert_eq!((r.length_mm, r.length_ok), (28.0, 1));
        // The workbook rounds 27.5 up to the same 28 mm but leaves the slit out of the room too.
        let r = m4_at(&ci, Deviations::NONE);
        assert_eq!((r.length_mm, r.length_ok), (28.0, 0));
        // A quarter millimetre more engagement rounds the corrected length up to 30: past the far OD.
        let longer = ClampInputs {
            engagement_x_d: 3.75,
            ..ci
        };
        let r = m4_at(&longer, e2);
        assert_eq!((r.length_mm, r.length_ok), (30.0, 0));
    }

    #[test]
    fn e2_length_note_is_set_only_when_the_recommended_screw_protrudes_with_the_correction_on() {
        let e2 = Deviations::only(DeviationId::E2);
        let with = |ci: &ClampInputs, dev| compute(ci, 10.0, 3.7647, &AL7075, 970.0, dev);
        // Defaults: the corrected M4 x 14 protrudes (the full text is pinned in tests/deviations.rs).
        let r = with(&ClampInputs::default(), e2);
        assert_eq!((r.index, r.table[2].length_ok), (3, 0));
        assert!(r.length_note.starts_with("No 2 mm length step of M4 "));
        // A 24.5 mm boss: the workbook check fails the recommended M4 x 12 (it leaves out the
        // slit), but with the correction off there is never a note.
        let ci = ClampInputs {
            boss_od_mm: 24.5,
            ..ClampInputs::default()
        };
        let r = with(&ci, Deviations::NONE);
        assert_eq!(
            (r.index, r.table[2].length_mm, r.table[2].length_ok),
            (3, 12.0, 0)
        );
        assert_eq!(r.length_note, "");
        // With the correction the same screw stays inside the boss: no note.
        let r = with(&ci, e2);
        assert_eq!(
            (r.index, r.table[2].length_mm, r.table[2].length_ok),
            (3, 12.0, 1)
        );
        assert_eq!(r.length_note, "");
        // No size fits: no recommendation, so no note.
        let small = ClampInputs {
            boss_od_mm: 12.0,
            ..ClampInputs::default()
        };
        let r = with(&small, e2);
        assert_eq!(
            (r.index, r.recommended.as_str(), r.length_note.as_str()),
            (0, NONE_FITS, "")
        );
    }
}
