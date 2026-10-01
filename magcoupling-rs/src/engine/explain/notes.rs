//! Addendum A4 teaching notes: static data keyed by note id, linked to the equations each
//! explains and to the A5 warning rules that cite them (`warnings::WarningRule::note_id`).
//!
//! The accuracy gate: a note is drafted from the M1 derivations it cites (`sources`) and
//! shows in the GUI only once a physics reviewer has checked it (`Review::Reviewed`, naming
//! the review record). [`note_for`] returns reviewed notes only; [`note_for_any_status`] is
//! for the review tooling. `tests/explain.rs` checks the links, the 2-6 sentence rule of a
//! reviewed note, and that each equation has at most one note.

/// A small diagram the M4 panel paints beside a note (egui painter, no image files).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagram {
    /// The magnetization's rectangular wave (blocks and gaps) and its first odd harmonics.
    SquareWaveHarmonics,
    /// Flux paths across the gap with and without back iron.
    FluxPathBackIron,
    /// Torque against electrical angle, the pull-out point marked.
    TorqueAngle,
    /// Field fringing at the magnet ends.
    EndFringing,
}

/// Where a note stands in the accuracy gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Review {
    /// Written, not yet checked: hidden from users.
    Draft,
    /// Checked by the physics reviewer: `record` names the review (report section or commit).
    Reviewed {
        reviewer: &'static str,
        date: &'static str,
        record: &'static str,
    },
}

/// One teaching note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Note {
    pub id: &'static str,
    pub title: &'static str,
    /// The equations it explains: result paths, or a family's target with `#` (all its members).
    pub equations: &'static [&'static str],
    /// 2 to 6 plain sentences at Physics 2 level (empty while a stub).
    pub sentences: &'static [&'static str],
    pub watch_out: Option<&'static str>,
    pub diagram: Option<Diagram>,
    /// The M1 derivations and report rows it is drafted from.
    pub sources: &'static [&'static str],
    pub review: Review,
}

const fn stub(id: &'static str, title: &'static str, equations: &'static [&'static str]) -> Note {
    Note {
        id,
        title,
        equations,
        sentences: &[],
        watch_out: None,
        diagram: None,
        sources: &[],
        review: Review::Draft,
    }
}

/// Every note. The torque chain's four are drafted (plan A-3 tracer); the rest are stubs
/// that fix the ids the "start here" order and the A5 warnings link to.
pub const NOTES: &[Note] = &[
    Note {
        id: "harmonics",
        title: "Harmonic decomposition",
        equations: &[
            "model.b_i#",
            "model.b_o#",
            "model.amp#_Pa",
            "model.tau_Pa",
            "calibration.amp#_Pa",
        ],
        sentences: &[
            "Each ring's magnetization alternates north and south around the circle, so along the gap it is a rectangular wave, blocks separated by gaps, not a smooth sine.",
            "Such a wave is a sum of sine waves at odd multiples of its basic frequency: harmonic n has n times as many wavelengths around the ring and an amplitude of 4/(nπ) times the remanence, scaled by sin(nπλ/2), where λ is the fraction of each pole the magnet fills; the gaps (λ < 1) are why that factor appears.",
            "Harmonic n of one ring pulls only on harmonic n of the other, so the total shear stress is a sum with one term per harmonic.",
            "Higher harmonics have shorter wavelengths, and their fields fade across the gap much faster, which is why the workbook keeps only 1, 3 and 5.",
        ],
        watch_out: Some(
            "A fill of exactly 0.4 makes the fifth harmonic vanish, because sin(5π·0.4/2) = sin(π) = 0: that is geometry, not an error.",
        ),
        diagram: Some(Diagram::SquareWaveHarmonics),
        sources: &[
            "M1 audit M3, M4, M7 (the planar harmonic model against exact 2D sections)",
            "Addendum A decision 29 (odd harmonics up to 11)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "back_iron_factor",
        title: "Back iron: the sinh factor against free space",
        equations: &["model.s#_iron", "model.s#_free", "model.k#"],
        sentences: &[
            "The geometry factor S_n says how much of harmonic n's field from one ring reaches through the other ring.",
            "With steel behind both rings, the field lines cross the gap and close through the steel instead of spreading into the air behind the magnets, and the factor takes the sinh form.",
            "The steel makes the field meet its surface at right angles, so the growing and decaying exponentials across the gap combine into sinh; in free space only the decaying one remains.",
            "Without back iron each ring's field leaks out behind it as well as across the gap, so the free-space factor, with its decay e^(−kg), is smaller.",
            "Both fall off with k·g, the gap measured against the harmonic's wavelength, so a small gap matters most for the high harmonics.",
        ],
        watch_out: None,
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "M1 audit M3 (planar factor against the cylindrical one)",
            "M1 audit M5 (steel at the magnet backs)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "pullout_angle",
        title: "Pull-out torque against rotation angle",
        equations: &[
            "model.tau#_Pa",
            "model.pullout_angle_rad",
            "model.iron_circuit_angle_rad",
            "model.free_circuit_angle_rad",
            "model.pullout_Nm",
            "calibration.pullout_angle_rad",
            "calibration.tau#_Pa",
        ],
        sentences: &[
            "Twist one ring against the other and the torque rises from zero, peaks and falls again: the peak is the pull-out torque, the most the coupling carries before it slips.",
            "Harmonic n contributes A_n sin(nφ), where φ is the electrical angle, 90° at half a pole pitch.",
            "With the fundamental alone the peak is at φ = 90°; a strong third harmonic can move it, so the calculator takes the angle where the summed curve is highest (correction E7).",
        ],
        watch_out: None,
        diagram: Some(Diagram::TorqueAngle),
        sources: &[
            "M1 audit E7 (T3-RC2)",
            "Addendum A decision 29 (one peak search for any harmonic set)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "end_effect",
        title: "End effect",
        equations: &["model.f_end", "calibration.f_end"],
        sentences: &[
            "The 2D model treats the rings as infinitely long, but real magnets end, and near each end the field fringes outward and carries less torque.",
            "The factor f_end = 1 − c_end·τ_p/L takes off a total of about c_end pole pitches, half at each end, from the active length L.",
            "It is empirical: within about 1 % of 3D for magnets longer than about 3 mm, but it turns negative for very short magnets, which the end-effect check flags.",
        ],
        watch_out: Some(
            "Below L = c_end·τ_p the factor is zero or negative and every torque computed from it is meaningless.",
        ),
        diagram: Some(Diagram::EndFringing),
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Draft,
    },
    stub(
        "br_temperature",
        "Remanence against temperature, and torque ∝ Br²",
        &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
        ],
    ),
    stub(
        "demagnetization",
        "Demagnetization: knee, permeance and the onset temperatures",
        &[],
    ),
    stub("slip_heating", "Eddy-current slip loss and skin depth", &[]),
    stub("thermal_time_constant", "The thermal time constant", &[]),
    stub("clamp_preload", "Clamp preload and friction", &[]),
    stub(
        "ferrite_cold_demag",
        "Ferrite: the demagnetization risk is at cold",
        &[],
    ),
    stub(
        "a5.non_ferromagnetic_back_iron",
        "Non-ferromagnetic back iron",
        &["materials.circuit_backiron"],
    ),
    stub(
        "a5.ferromagnetic_sleeve_or_liner",
        "Ferromagnetic sleeve or liner",
        &[],
    ),
    stub(
        "a5.high_conductivity_sleeve_or_liner",
        "High-conductivity sleeve or liner",
        &[],
    ),
    stub("a5.low_saturation", "Low saturation", &[]),
    stub(
        "a5.uncoated_low_alloy_steel",
        "Uncoated low-alloy steel",
        &[],
    ),
    stub(
        "a5.cte_mismatch_with_magnets",
        "Expansion mismatch with the magnets",
        &[],
    ),
];

/// The suggested reading order (spec A4 "Start here": torque chain → back iron →
/// temperature → demagnetization → slip heating → clamps): (note id, the equation it opens).
pub const START_HERE: &[(&str, &str)] = &[
    ("harmonics", "model.tau_Pa"),
    ("back_iron_factor", "model.s1_iron"),
    ("br_temperature", "model.br_inner_T_op"),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    ("clamp_preload", "clamps.capacity_Nm"),
];

/// Whether a note's equation entry (a path or a `#` template) covers `path`.
pub fn covers(entry: &str, path: &str) -> bool {
    match entry.split_once('#') {
        None => entry == path,
        Some((prefix, suffix)) => path
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(suffix))
            .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())),
    }
}

/// The note with this id.
pub fn note(id: &str) -> Option<&'static Note> {
    NOTES.iter().find(|n| n.id == id)
}

/// The note explaining `path`, whatever its review status (review tooling, tests).
pub fn note_for_any_status(path: &str) -> Option<&'static Note> {
    NOTES
        .iter()
        .find(|n| n.equations.iter().any(|e| covers(e, path)))
}

/// The note the equation panel shows for `path`: reviewed notes only (the accuracy gate).
pub fn note_for(path: &str) -> Option<&'static Note> {
    note_for_any_status(path).filter(|n| matches!(n.review, Review::Reviewed { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::warnings::WARNING_RULES;

    #[test]
    fn templates_cover_their_members_only() {
        assert!(covers("model.tau#_Pa", "model.tau11_Pa"));
        assert!(covers("model.tau#_Pa", "model.tau1_Pa"));
        assert!(!covers("model.tau#_Pa", "model.tau_Pa"));
        assert!(!covers("model.tau#_Pa", "model.taux_Pa"));
        assert!(covers("model.f_end", "model.f_end"));
    }

    #[test]
    fn ids_are_unique_and_every_link_resolves() {
        for (i, n) in NOTES.iter().enumerate() {
            assert!(NOTES[..i].iter().all(|m| m.id != n.id), "{} twice", n.id);
        }
        for rule in &WARNING_RULES {
            assert!(
                note(rule.note_id).is_some(),
                "warning {} links to missing note {}",
                rule.id,
                rule.note_id
            );
        }
        for (id, _) in START_HERE {
            assert!(note(id).is_some(), "start-here note {id}");
        }
    }

    #[test]
    fn a_draft_is_hidden_and_a_reviewed_note_is_complete() {
        assert!(note_for("model.f_end").is_none(), "drafts stay hidden");
        assert_eq!(
            note_for_any_status("model.f_end").map(|n| n.id),
            Some("end_effect")
        );
        for n in NOTES {
            if n.sentences.is_empty() {
                assert_eq!(
                    n.review,
                    Review::Draft,
                    "{}: a stub cannot be reviewed",
                    n.id
                );
                continue;
            }
            assert!(
                (2..=6).contains(&n.sentences.len()),
                "{}: 2 to 6 sentences",
                n.id
            );
            assert!(
                !n.sources.is_empty(),
                "{}: a drafted note cites its M1 sources",
                n.id
            );
        }
    }
}
