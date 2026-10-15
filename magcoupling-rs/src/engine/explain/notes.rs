//! Addendum A4 teaching notes: static data keyed by note id, linked to the equations each
//! explains and to the A5 warning rules that cite them (`warnings::WarningRule::note_id`).
//!
//! The accuracy gate: a note is drafted from the M1 derivations it cites (`sources`) and
//! shows in the GUI only once a physics reviewer has checked it (`Review::Reviewed`, naming
//! the review record). [`note_for`] returns reviewed notes only; [`note_for_any_status`] is
//! for the review tooling. The unit tests below check the warning and start-here links and
//! hold every note, reviewed or not, to 2-6 sentences and a cited source; `tests/explain.rs`
//! checks that each note names records and each equation has at most one note, and holds the
//! M4 release gate (`release_notes_are_reviewed`).
//!
//! Slice 2 (spec `docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md`, 3.1): a
//! note has three visible parts. What it is is `sentences`; What to change is `change`, each
//! entry an input path, a [`Step`], the result it moves and which way, and its effect and
//! trade-off; Watch out is `watch_out`. The `sources` sit under a fold that starts closed. The
//! slice's caps (3 to 5 sentences of at most 25 words, 1 to 5 change entries, 0 to 2
//! watch-outs of at most 30 words, a sentence 0 in plain words) are tests that stay ignored
//! until every note is rewritten.

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
    /// The intrinsic demagnetization curve with its knee, and the load lines of a few
    /// permeance coefficients.
    DemagKnee,
    /// First-order heating toward a steady temperature, the time constant marked.
    HeatingCurve,
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

/// Which way a "What to change" entry moves its input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// One slider step up.
    Raise,
    /// One slider step down.
    Lower,
    /// Another choice of a selector, or another part, grade or material pick.
    Choose,
    /// A number entered from a measurement or an outside calculation: no direction, no aim.
    Enter,
}

impl Step {
    /// The word an entry starts with on screen.
    pub const fn word(self) -> &'static str {
        match self {
            Step::Raise => "Raise",
            Step::Lower => "Lower",
            Step::Choose => "Choose",
            Step::Enter => "Enter",
        }
    }

    /// The way one slider step goes, as
    /// [`SliderRange::step_from`](crate::engine::meta::SliderRange::step_from) takes it: up for
    /// [`Step::Raise`], down for [`Step::Lower`], none for [`Step::Choose`] and [`Step::Enter`]
    /// (a choice has no slider; an entered number is not stepped).
    pub const fn up(self) -> Option<bool> {
        match self {
            Step::Raise => Some(true),
            Step::Lower => Some(false),
            Step::Choose | Step::Enter => None,
        }
    }
}

/// Which way an entry's aimed result moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sense {
    Up,
    Down,
}

impl Sense {
    /// Whether a result that went from `before` to `after` moved this way: strictly, so an
    /// unmoved result (or a NaN) never holds. The one rule the aim test and the reviewer's
    /// sheet both apply.
    pub fn holds(self, before: f64, after: f64) -> bool {
        match self {
            Sense::Up => after > before,
            Sense::Down => after < before,
        }
    }
}

/// One "What to change" entry. The GUI draws "`<Step word> <display label of input>: <text>`",
/// the label a link that highlights the input's row, or plain text with the reason where the
/// input is not free on the design shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// An input path of the catalogue; its label is never typed in a note.
    pub input: &'static str,
    /// [`Step::Choose`] for a selector or a pick; Raise or Lower for a number with a slider;
    /// [`Step::Enter`] for a number taken from a measurement or an outside calculation.
    pub step: Step,
    /// The result the entry says the move changes, and which way: `Some` for Raise and Lower
    /// (a test steps the input at the probe designs and checks it; a model assumption's step
    /// may have none, its effect depending on the design), `None` for Choose and Enter.
    pub aim: Option<(&'static str, Sense)>,
    /// The effect and its trade-off.
    pub text: &'static str,
}

/// One teaching note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Note {
    pub id: &'static str,
    pub title: &'static str,
    /// The equations it explains: result paths, or a family's target with `#` (all its members).
    pub equations: &'static [&'static str],
    /// What it is: plain sentences (slice 2: 3 to 5 of at most 25 words; sentence 0 answers
    /// the question of the row that links here, with no symbol and no live value).
    pub sentences: &'static [&'static str],
    /// What to change (slice 2: 1 to 5 entries).
    pub change: &'static [Change],
    /// Watch out (slice 2: 0 to 2 entries of at most 30 words).
    pub watch_out: &'static [&'static str],
    pub diagram: Option<Diagram>,
    /// The M1 derivations and report rows it is drafted from (drawn under a closed fold).
    pub sources: &'static [&'static str],
    pub review: Review,
}

/// Every note: 18, the spec's list (harmonics, back iron, pull-out against angle, end effect,
/// Br(T), demagnetization, slip loss, the thermal time constant, clamps and the six A5
/// warnings) plus ferrite's cold side (spec A6), the one-point calibration and slice 2's
/// torque_from_shear (how the coupling makes torque, Start here step 1).
pub const NOTES: &[Note] = &[
    Note {
        id: "torque_from_shear",
        title: "How the coupling makes torque",
        equations: &[
            "model.torque_2d_Nm",
            "model.area_lever_m3",
            "model.gap_radius_mm",
            "model.pole_pitch_mm",
            "model.active_length_mm",
        ],
        sentences: &[
            "Twist one ring against the other and the magnets across the gap pull each other back into line, so the rings pass torque.",
            "That sideways pull, spread over the cylinder at the gap's mid radius R_g, is a shear stress σ (at its peak, here: {model.tau_Pa}).",
            "σ times the cylinder's area 2π·R_g·L, times lever arm R_g, is the endless-ring torque T_2D = σ·2π·R_g²·L (magnet ends ignored), L the overlapping magnet length.",
            "σ grows with stronger magnet grades and magnetic steel behind the magnets; it falls as the gap widens or blocks fill less of each pole.",
            "The magnet count N keeps the gap radius R_g: blocks share one ring, each pole spanning τ_p = 2π·R_g/N; fewer flat blocks grow the cup.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "torque rises almost linearly; on the example design each millimetre beats a millimetre of radius. Costs: longer stack, higher clamp and gearbox loads.",
            },
            Change {
                input: "coupling.inner_back_apothem_mm",
                step: Step::Raise,
                aim: Some(("model.area_lever_m3", Sense::Up)),
                text: "more area and lever arm; at the same magnet count the blocks spread apart, so add magnets as it grows. The cup grows too.",
            },
            Change {
                input: "coupling.npole",
                step: Step::Raise,
                aim: Some(("model.torque_2d_Nm", Sense::Up)),
                text: "blocks fill more of each pole and torque usually rises, until they stop fitting; fewer usually lose torque and, with flat blocks, grow the cup.",
            },
        ],
        watch_out: &[
            "T_2D grows as R_g² only at fixed σ: a larger ring with the same magnet count fills each pole less, so the pull-out can even fall.",
            "The dashboard's pull-out torque is T_2D corrected for the magnet ends and by a calibration factor (here: {model.torque_2d_Nm} becomes {model.pullout_Nm}).",
        ],
        diagram: None,
        sources: &[
            "The torque model's equations: gap radius, pole pitch, overlapping length, area times lever arm, and the endless-ring torque",
            "Math audit of 2026-09-29: the planar harmonic model against exact 2D sections and real 3D rings",
            "Guidance audit of 2026-10-13: probes of 10 against 8 magnets, and of magnet length against ring radius",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#torque_from_shear",
        },
    },
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
            "Such a wave is a sum of sine waves at odd multiples of its basic frequency: harmonic n has n times as many wavelengths around the ring and an amplitude of 4/(nπ) times the remanence, scaled by sin(nπλ/2), where λ is the fraction of each pole the magnet fills (λ < 1 when there are gaps between blocks); that factor sets each harmonic's size and sign, so some amplitudes come out negative even with no gaps (sin(3π/2) = −1), and the fill decides which ones.",
            "Harmonic n of one ring pulls only on harmonic n of the other, so the total shear stress is a sum with one term per harmonic.",
            "Higher harmonics have shorter wavelengths and a lower amplitude ceiling, 4/(nπ) times the remanence (the fill can still make a higher one larger than a lower one), and above all their fields fade across the gap much faster, which is why the workbook keeps only 1, 3 and 5 (the calculator's default; it can sum the odd harmonics up to 11).",
        ],
        change: &[],
        watch_out: &[
            "A fill of exactly 0.4 makes the fifth harmonic vanish, because sin(5π·0.4/2) = sin(π) = 0: that is geometry, not an error.",
        ],
        diagram: Some(Diagram::SquareWaveHarmonics),
        sources: &[
            "M1 audit M3, M4, M7 (the planar harmonic model against exact 2D sections)",
            "Addendum A decision 29 (odd harmonics up to 11)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
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
        change: &[],
        watch_out: &[],
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "M1 audit M3 (planar factor against the cylindrical one)",
            "M1 audit M5 (steel at the magnet backs)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
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
        change: &[],
        watch_out: &[],
        diagram: Some(Diagram::TorqueAngle),
        sources: &[
            "M1 audit E7 (T3-RC2)",
            "Addendum A decision 29 (one peak search for any harmonic set)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "end_effect",
        title: "End effect",
        equations: &["model.f_end", "calibration.f_end"],
        sentences: &[
            "The 2D model treats the rings as infinitely long, but real magnets end, and near each end the field fringes outward and carries less torque.",
            "The factor f_end = 1 − c_end·τ_p/L takes off a total of about c_end pole pitches, half at each end, from the active length L.",
            "It is empirical: at the default design it is within about 1 % of 3D for magnets longer than about 3 mm (about a third of a pole pitch), it has no gap term (one M1 skeptic found it 11 % high at a 4.4 mm gap), and it turns negative for very short magnets, which the end-effect check flags.",
        ],
        change: &[],
        watch_out: &[
            "At or below L = c_end·τ_p (1.3 mm at the default pole pitch) the factor is zero or negative and every torque computed from it is meaningless; between that and about 3 mm the check still reads OK, yet the factor is far off (21 % below 3D at 2 mm, M1 audit M9).",
        ],
        diagram: Some(Diagram::EndFringing),
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "calibration",
        title: "The one-point calibration",
        equations: &[
            "model.f_cal",
            "calibration.f_cal_updated",
            "calibration.measured_over_model",
            "calibration.model_error",
        ],
        sentences: &[
            "The 2D harmonic model reads a few per cent low against real flat blocks (about 4 % without back iron, M1 audit M4; less with steel, M5), so the workbook scales it by a calibration factor: an assumed f_cal,0 (0.95, which adds conservatism rather than correcting that bias) or one measured on the bench.",
            "When a design has the measured prototype's circuit (no back iron), pole count and magnet part (B842SH on both rings), it uses the bench result: f_cal,1 = f_cal,0 · T_meas / T_model, which puts the model of the prototype on the 1.8 N·m measurement.",
            "Because T_model already contains f_cal,0, the assumed factor cancels: the bench correction is T_meas divided by the uncalibrated model, whatever f_cal,0 was.",
            "Every other design keeps the assumed 0.95, since one measurement cannot say how the model's error changes with the design; the rule does not compare gaps or radii, so a resized design with the prototype's circuit, pole count and magnet part still gets the bench factor.",
        ],
        change: &[],
        watch_out: &[
            "The bench value has two significant figures and an assumed 20 °C test temperature; it sits about 1 % above the 3D pull-out at the corrected remanence of 1.30 T (correction E3), and 2.5 % above it at the audit's 1.29 T (M1 audit P2).",
        ],
        diagram: None,
        sources: &[
            "M1 audit M4, M5 and M6 (the model's bias without and with back iron, and the calibration that absorbs it)",
            "M1 audit P2 (the bench value against physics)",
            "M1 audit E3 (the N42SH remanence corrected from 1.29 T to 1.30 T, Calibration!C21 included)",
            "Plan A-3 traceability: the f_cal,0 cancellation (tests/explain.rs CANCELLATIONS)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
    },
    Note {
        id: "br_temperature",
        title: "Remanence against temperature, and torque ∝ Br²",
        equations: &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
            "metal.torque_cold_Nm",
            "temperature.magnet_life.torque_hot_day_Nm",
            "temperature.magnet_life.torque_peak_Nm",
            "temperature.demag.torque_at_limit_Nm",
        ],
        sentences: &[
            "A magnet's remanence Br, the flux density it keeps with no applied field, falls reversibly as it warms: Br(ϑ) = Br(20 °C) · (1 + α (ϑ − 20 °C)), with α about −0.12 %/°C for sintered NdFeB.",
            "Each ring's field is proportional to its own Br, and the shear stress is a product of the two rings' fields, so the torque scales as Br,i · Br,o: the square of Br when both rings are one grade.",
            "So a 30 °C rise costs about 7 % of the torque: (1 − 0.0012 · 30)² ≈ 0.93.",
            "The loss is reversible: cool the magnet and the torque comes back, unless on the way it passed a demagnetization onset, the temperature where its reverse field reaches the knee, which is a separate, permanent loss; the calculator's demagnetization limit sits 10 °C below the lowest onset, the one while the coupling skips.",
        ],
        change: &[],
        watch_out: &[
            "Each ring keeps its own coefficient: a ferrite ring (α about −0.2 %/°C) loses torque faster with heat than an NdFeB one.",
        ],
        diagram: None,
        sources: &[
            "M1 audit, confirmed: the Br(T) and torque-temperature rows (Calculator C69, C70, C94; Metal design C8)",
            "Addendum A decision A2-7 (each ring with its own coefficient)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
    },
    Note {
        id: "demagnetization",
        title: "Demagnetization: knee, permeance and the onset temperatures",
        equations: &[
            "temperature.demag.h_ref_kA_m",
            "temperature.demag.t_ref_model_C",
            "temperature.demag.calibration_offset_C",
            "temperature.demag.onset_aligned_C",
            "temperature.demag.onset_pullout_C",
            "temperature.demag.onset_skipping_C",
            "temperature.demag.onset_single_ring_C",
            "temperature.demag.magnet_limit_C",
            "temperature.demag.inner_magnet_limit_C",
            "temperature.demag.outer_magnet_limit_C",
            "temperature.demag.demag_ring",
        ],
        sentences: &[
            "Inside a magnet the demagnetizing field H points against its magnetization (B still points along it); the stronger this reverse field, the closer the magnet is to losing magnetization for good.",
            "The loss starts at the knee of the intrinsic curve, taken as H_k = 0.9 · Hcj, and Hcj falls as NdFeB heats (β about −0.5 %/°C), so each reverse field has an onset temperature where the falling knee meets it; the reverse field itself shrinks with Br(T), which is why the onset needs both α and β.",
            "The reverse field depends on the magnet's surroundings, its permeance: a reference magnet on the load line B = −μ0 H (permeance coefficient 1) sees Br/(2 μ0), and like poles of the other ring facing it while the coupling skips push it highest, 863 kA/m here, so the skipping onset sets the limit.",
            "The onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature, and the design limit keeps a 10 °C margin below the skipping onset.",
            "With correction E20 each ring is checked with its own grade, and the ring with the lower limit governs.",
        ],
        change: &[],
        watch_out: &[
            "The default N42SH's temperature coefficients hold over 20–150 °C only: onsets above 150 °C are extrapolations, and so is the reference magnet's model knee near 160 °C that sets the shift, so a ±20 % change of the Hcj slope above 150 °C moves even the default limit by up to about 3 °C (M1 audit M13).",
        ],
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "M1 audit M13 (knee temperature outside the coefficient range)",
            "M1 audit rulings: the onset calibration (it uses 9.83 °C of the 10 °C margin)",
            "Addendum A decision 19 (E20: each ring's own grade)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
    },
    Note {
        id: "ferrite_cold_demag",
        title: "Ferrite: the demagnetization risk is at cold",
        equations: &[
            "temperature.demag.inner_cold_limit_C",
            "temperature.demag.outer_cold_limit_C",
            "temperature.demag.cold_ring",
            "temperature.demag.cold_limit_C",
            "temperature.demag.cold_check",
            "temperature.demag.cold_onset_aligned_C",
            "temperature.demag.cold_onset_pullout_C",
            "temperature.demag.cold_onset_skipping_C",
            "temperature.demag.cold_onset_single_ring_C",
        ],
        sentences: &[
            "In hard ferrite the coercivity rises as it warms (β about +0.35 %/°C), the opposite of NdFeB, so its knee falls as the magnet cools.",
            "Its remanence still rises as it cools, so the reverse field grows while the knee shrinks: below some temperature the reverse field passes the knee and the magnet loses magnetization.",
            "The calculator finds that cold onset, ϑ = 20 + (H_k − H)/(H α − H_k β), and checks the minimum magnet temperature against the skipping cold onset plus the 10 °C margin; heating moves a ferrite magnet's knee away from the reverse field, so its rating is the hot limit.",
        ],
        change: &[],
        watch_out: &[
            "A ferrite ring can pass every hot check and still fail at −40 °C: read the cold check too, and enter the ferrite design's own reverse fields, since the defaults (354 to 863 kA/m) come from the NdFeB default design and put a Y30 ring's cold limit at 336 °C.",
        ],
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "Spec A6 (ferrite's opposite-sign beta gets a teaching note)",
            "Addendum A decisions 3 (Y30 beta +0.35 %/°C) and 19 (E20's positive-beta branch)",
            "A-1 plan decision A13 (both rings; the cold side from the ring with the higher cold limit)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
    },
    Note {
        id: "slip_heating",
        title: "Eddy-current slip loss and skin depth",
        equations: &[
            "temperature.duty.field_omega_rad_s",
            "temperature.slip_loss.skin_depth_mm",
            "temperature.slip_loss.hub_W",
            "temperature.slip_loss.cup_W",
            "temperature.slip_loss.web_W",
            "temperature.slip_loss.cap_W",
            "temperature.slip_loss.magnets_W",
            "temperature.slip_loss.total_W",
            "temperature.slip_loss.drag_Nm",
        ],
        sentences: &[
            "When the coupling slips, each ring's alternating field sweeps past the other ring's parts at the field frequency f_e = p · n / 60, with p the pole pairs and n the slip speed in rev/min, and by Faraday's law it drives eddy currents in every conductor it crosses: the steel cup and hub, the sleeve, the liner, the cap and the magnets themselves.",
            "The currents turn power into heat, and that power is also the drag torque times the slip speed.",
            "In steel the currents crowd into a skin of depth δ = √(2/(ω_e μ0 μr σ)), with ω_e = 2πf_e the field's angular frequency, about 1.3 mm at the default slip, so the steel loss grows as speed to the power 1.5 rather than 2.",
            "Thin shells such as the sleeve and liner are thinner than their skin depth (about 34 mm in 316L), so their loss follows σ (ω_s r B)²/2 per unit volume times the 0.7 end factor, with ω_s the slip speed in rad/s, the square of speed.",
        ],
        change: &[],
        watch_out: &[
            "Aluminium parts (no back iron) are resistance-limited, not skin-limited, and see the weaker free-space field (correction E17); every loss here is an estimate with a factor-3 high case until a bench drag is measured.",
        ],
        diagram: None,
        sources: &[
            "M1 audit M11 (thin-skin steel losses against Stoll's half-space)",
            "M1 audit M12 (magnet eddy loss)",
            "M1 audit P1 (the shell end factor)",
            "Addendum A report section 5.5 (E17: the low-Reynolds form for aluminium parts)",
        ],
        review: Review::Reviewed {
            reviewer: "independent physics review 2026-10-01",
            date: "2026-10-01",
            record: "independent physics review 2026-10-01 of the A4 notes: commit \"docs(magcoupling-rs): apply the independent physics review to six A4 notes\" (the seven findings and their fixes)",
        },
    },
    Note {
        id: "thermal_time_constant",
        title: "The thermal time constant",
        equations: &[
            "temperature.thermal.heat_capacity_J_K",
            "temperature.thermal.time_constant_s",
            "temperature.thermal.t95_s",
            "temperature.thermal.steady_rise_est_C",
            "temperature.thermal.steady_rise_high_C",
            "temperature.thermal.time_to_limit_high",
            "temperature.thermal.time_to_limit_est",
            "temperature.thermal.temp_at_fault_C",
            "temperature.thermal.critical_drag_Nm",
        ],
        sentences: &[
            "The calculator treats the rotating coupling as one lump with heat capacity C = Σ m c, losing heat to its surroundings through one conductance G.",
            "Heated with power P, its temperature rises toward ϑ_start + P/G along ϑ(t) = ϑ_start + (P/G)(1 − e^(−t/τ)), with time constant τ = C/G: 63 % of the rise after τ and 95 % after 3τ.",
            "The steady rise P/G does not depend on C; C (with G) sets only how fast it is reached; one short slip event adds only P t / C.",
            "The time to the limit solves that curve for the limit temperature; it is 'never' when the steady temperature stays at or below the limit, and 0 when the start is already at or above it (correction E12).",
        ],
        change: &[],
        watch_out: &[
            "G is a placeholder (0.3 W/K, not measured): a 10 % higher G lowers the high-case steady temperature by about 2.5 °C (M1 audit placeholder table).",
        ],
        diagram: Some(Diagram::HeatingCurve),
        sources: &[
            "M1 audit placeholder inputs (conductance: −2.48 °C per +10 %)",
            "Addendum A decision 8 (E15: aluminium parts at their own specific heat)",
            "M1 audit E12 (a start above the limit)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "clamp_preload",
        title: "Clamp preload and friction",
        equations: &[
            "clamps.table[#].preload_strength_N",
            "clamps.table[#].preload_strip_N",
            "clamps.table[#].preload_N",
            "clamps.table[#].torque_per_screw_Nm",
            "clamps.table[#].screws_needed",
            "clamps.capacity_Nm",
        ],
        sentences: &[
            "A slotted clamp holds the shaft by friction: tightening each screw stretches it to a preload F, which squeezes the jaws onto the shaft.",
            "Friction resists slipping on both jaws at the shaft radius, so one screw holds about T = μ F d k, with d the shaft diameter and k the share of the screw force that reaches the shaft.",
            "The preload is 75 % of the screw's proof load, unless the aluminium thread's stripping strength divided by a safety factor of 1.5 is lower; the smaller of the two governs.",
            "The clamp must hold the cold-high torque times a safety factor of 2, which sets how many screws are needed.",
        ],
        change: &[],
        watch_out: &[
            "μ = 0.15 assumes a degreased bore; an oily bore (about 0.10) cuts the capacity by a third.",
        ],
        diagram: None,
        sources: &[
            "M1 audit M14 (the thread-stripping area)",
            "M1 audit E2 (the screw length and the slit)",
            "Addendum A3 assumptions: clamp friction and preload fraction",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.non_ferromagnetic_back_iron",
        title: "Non-ferromagnetic back iron",
        equations: &["materials.circuit_backiron"],
        sentences: &[
            "Steel behind the magnets gives each ring's flux an easy return path, so the flux crosses the gap and closes through the steel instead of spreading out behind the magnets.",
            "Stainless 304 and aluminium have a relative permeability near 1, like air, so choosing one switches the calculator to the free-space circuit: the geometry factor loses its sinh form and the torque drops by more than 40 % (2.69 to 1.54 N·m); the default design has the measured prototype's pole count and magnet part (B842SH on both rings), so the calculator also switches to the bench calibration (0.95 to 1.049) and shows 1.70 N·m.",
            "The flux that no longer closes through steel leaks out around the coupling, so a strong stray field extends outside it and pulls in steel chips and debris.",
        ],
        change: &[],
        watch_out: &[],
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "Spec A5 warning rules",
            "M1 audit M5 (steel at the magnet backs)",
            "Addendum A report section 5 (the no-back-iron circuit, E15 to E17)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.ferromagnetic_sleeve_or_liner",
        title: "Ferromagnetic sleeve or liner",
        equations: &[],
        sentences: &[
            "The sleeve and liner sit in the magnetic gap, the one place the flux must cross from ring to ring.",
            "A ferromagnetic sleeve or liner offers the flux an easy path along itself, from one pole to its neighbour on the same ring, and flux that takes that short cut never crosses the gap, so it carries no torque.",
            "Saturation caps the short cut at about B_sat · t per unit length, so the torque lost grows with the shell's thickness t and is a larger share for weaker magnets; since any ferromagnetic shell costs torque, the retainers are non-magnetic 316L, titanium, Inconel or PEEK.",
        ],
        change: &[],
        watch_out: &[],
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A report section 4 (the materials table: ferromagnetic or not)",
            "Plan A-3 Task 16 physics review, the saturation cap: a shell of thickness t diverts at most about B_sat · t per unit length (default sleeve 0.1 mm, liner 0.2 mm)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.high_conductivity_sleeve_or_liner",
        title: "High-conductivity sleeve or liner",
        equations: &[
            "temperature.slip_loss.sleeve_W",
            "temperature.slip_loss.liner_W",
        ],
        sentences: &[
            "The sleeve and liner sit in the strongest alternating field during slip, and as thin shells their eddy loss is proportional to their conductivity: σ t (ω_s r B)²/2 per unit area, with ω_s the slip speed, times an end factor of 0.7 for the shell's finite length.",
            "A material that conducts better than 316L (1.35 × 10⁶ S/m) therefore heats more for the same slip, raising the steady temperature and shortening the time to the limit.",
            "Titanium and Inconel conduct less than 316L and PEEK almost not at all, so none of the listed choices fires this warning; a typed conductivity can.",
        ],
        change: &[],
        watch_out: &[],
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit P1 (the shell loss and its end factor)",
            "Addendum A report section 4 (the conductivities)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.low_saturation",
        title: "Low saturation",
        equations: &["model.bsat_T", "model.backiron_needed_mm"],
        sentences: &[
            "The back-iron wall carries each pole's flux around to its neighbour, and the wall it needs is t = B_gap τ_p / (π B_des): the lower the flux density the steel is allowed, the thicker the wall.",
            "The design value B_des sits below saturation, 1.5 T for annealed 4140; a steel whose saturation is below 1.7 T (1018's design value, the highest the workbook names) is flagged, and so is a steel with no design value of its own, for which the wall check uses the design flux density input (1.5 T by default).",
            "Past saturation the steel's permeability collapses, flux leaks out of the circuit and the torque falls, so such a back iron needs thicker walls.",
        ],
        change: &[],
        watch_out: &[
            "The wall formula itself reads 12 to 16 % thin against exact 2D sections (M1 audit M1), so walls near the limit deserve margin.",
        ],
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A decision 20 (the design flux density feeds the wall check)",
            "M1 audit M1 and M2 (the back-iron requirement)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.uncoated_low_alloy_steel",
        title: "Uncoated low-alloy steel",
        equations: &[],
        sentences: &[
            "Plain and low-alloy steels such as 4140, 1018 and 12L14 rust in humid air; the stainless grades in the list (416, 17-4PH, 304) resist it.",
            "The workbook therefore plans a high-phosphorus electroless nickel plating, which is non-magnetic as plated and adds about 0.015 mm per surface.",
            "With no plating thickness entered, a plain or low-alloy steel back iron is flagged, because it needs a coating.",
        ],
        change: &[],
        watch_out: &[],
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Workbook Materials C26 (electroless nickel, 0.013–0.025 mm)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
    Note {
        id: "a5.cte_mismatch_with_magnets",
        title: "Expansion mismatch with the magnets",
        equations: &[],
        sentences: &[
            "Heating changes a part's length by α ΔT per unit length; NdFeB barely changes across its magnetization (about −0.8 × 10⁻⁶ /°C), while steel expands about 12 × 10⁻⁶ /°C and aluminium about 24 × 10⁻⁶ /°C.",
            "Where a magnet is glued to the hub, that difference is forced through the thin glue line as shear, largest at the block ends.",
            "The larger the mismatch and the temperature swing, the higher that shear, so a hub whose expansion differs from the magnets' by more than 15 × 10⁻⁶ /°C is flagged; the Volkersen screen puts a number on it.",
        ],
        change: &[],
        watch_out: &[],
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit E1 (the Volkersen shear-lag screen)",
            "Addendum A decision 16 (E18: the aluminium hub's expansion)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations: commit ee11bfd (what was checked, the ten fixes)",
        },
    },
];

/// The suggested reading order (slice-2 spec 3.3: how the coupling makes torque, the torque
/// chain with back iron, temperature, slipping and its heat, demagnetization, clamps): (note id,
/// the equation it opens). The slice's two later notes join at steps 8 and 9.
pub const START_HERE: &[(&str, &str)] = &[
    ("torque_from_shear", "model.torque_2d_Nm"),
    ("harmonics", "model.tau_Pa"),
    ("back_iron_factor", "model.s1_iron"),
    ("pullout_angle", "model.pullout_Nm"),
    ("end_effect", "model.f_end"),
    ("calibration", "model.f_cal"),
    ("br_temperature", "model.pullout_20C_Nm"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    (
        "thermal_time_constant",
        "temperature.thermal.time_constant_s",
    ),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("ferrite_cold_demag", "temperature.demag.cold_check"),
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

/// The notes as they stood before guidance slice 2, for the owner's sign-off sheet (task T20):
/// the format of `docs/analyses/2026-10-14-magcoupler-notes-before.txt` and its reader. A
/// header line, then per note a heading `## <id>` and one `<field>: <text>` line per title,
/// sentence, watch-out and source (fields `title`, `sentence`, `watch out`, `source`), in that
/// order, and a blank line. Tests only.
#[cfg(test)]
pub(crate) mod before {
    use super::NOTES;

    /// The committed snapshot, written once by `notes_before_snapshot` at the start of slice 2.
    pub(crate) const SNAPSHOT: &str =
        include_str!("../../../../docs/analyses/2026-10-14-magcoupler-notes-before.txt");

    /// The current notes in the snapshot's format.
    pub(crate) fn snapshot_text() -> String {
        let mut text = String::from(
            "# Teaching notes before guidance slice 2 (the text of commit b0166ad), written once by notes::tests::notes_before_snapshot\n",
        );
        for n in NOTES {
            text.push_str(&format!("\n## {}\n", n.id));
            let mut line = |field: &str, value: &str| {
                assert!(
                    !value.contains('\n'),
                    "{}: a line break in its {field}",
                    n.id
                );
                text.push_str(&format!("{field}: {value}\n"));
            };
            line("title", n.title);
            for sentence in n.sentences {
                line("sentence", sentence);
            }
            for watch_out in n.watch_out {
                line("watch out", watch_out);
            }
            for source in n.sources {
                line("source", source);
            }
        }
        text
    }

    /// A snapshot read back: each note's id and its `(field, text)` lines, in order.
    pub(crate) fn parse(text: &str) -> Vec<(&str, Vec<(&str, &str)>)> {
        let mut notes: Vec<(&str, Vec<(&str, &str)>)> = Vec::new();
        for line in text.lines() {
            if let Some(id) = line.strip_prefix("## ") {
                notes.push((id, Vec::new()));
            } else if let (Some(field_text), Some(note)) = (line.split_once(": "), notes.last_mut())
            {
                note.1.push(field_text);
            }
        }
        notes
    }
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
        for (id, path) in START_HERE {
            let n = note(id).unwrap_or_else(|| panic!("start-here note {id}"));
            assert!(
                n.equations.iter().any(|e| covers(e, path)),
                "start here {id}: the note does not explain {path}"
            );
        }
    }

    #[test]
    fn a_draft_is_hidden_and_every_note_is_complete() {
        assert_eq!(
            note_for_any_status("model.f_end").map(|n| n.id),
            Some("end_effect")
        );
        assert!(
            (15..=20).contains(&NOTES.len()),
            "spec A4: about 15 to 20 notes"
        );
        for n in NOTES {
            assert!(
                (2..=6).contains(&n.sentences.len()),
                "{}: 2 to 6 sentences",
                n.id
            );
            assert!(
                !n.sources.is_empty(),
                "{}: a note cites the M1 derivations it is drafted from",
                n.id
            );
            // The accuracy gate: the panel shows a note only once it is reviewed.
            let reviewed = matches!(n.review, Review::Reviewed { .. });
            for entry in n.equations {
                let path = entry.replace('#', "1");
                assert_eq!(
                    note_for_any_status(&path).map(|m| m.id),
                    Some(n.id),
                    "{path}"
                );
                assert_eq!(note_for(&path).is_some(), reviewed, "{}: {path}", n.id);
            }
        }
    }

    /// Slice-2 spec 3.3: Start here's 14 steps, in this order. A step whose note a later task of
    /// the slice adds is skipped until then, so every task keeps the order (T18 asserts all 14).
    const SLICE_2_ORDER: [&str; 14] = [
        "torque_from_shear",
        "harmonics",
        "back_iron_factor",
        "pullout_angle",
        "end_effect",
        "calibration",
        "br_temperature",
        "hot_low_cold_high",
        "when_slipping",
        "slip_heating",
        "thermal_time_constant",
        "demagnetization",
        "ferrite_cold_demag",
        "clamp_preload",
    ];

    #[test]
    fn start_here_follows_the_slice_2_order_from_how_the_coupling_makes_torque() {
        let ids: Vec<&str> = START_HERE.iter().map(|&(id, _)| id).collect();
        let want: Vec<&str> = SLICE_2_ORDER
            .iter()
            .copied()
            .filter(|id| ids.contains(id))
            .collect();
        assert_eq!(
            ids, want,
            "spec 3.3 order (a step outside the list fails too)"
        );
        assert_eq!(START_HERE[0], ("torque_from_shear", "model.torque_2d_Nm"));
        // The step-1 note owns the area-and-lever chain (spec 3.3 R3).
        for path in [
            "model.torque_2d_Nm",
            "model.area_lever_m3",
            "model.gap_radius_mm",
            "model.pole_pitch_mm",
            "model.active_length_mm",
        ] {
            assert_eq!(
                note_for_any_status(path).map(|n| n.id),
                Some("torque_from_shear"),
                "{path}"
            );
        }
    }

    /// The step-1 note keeps to the slice-2 caps and its sentence 0 is plain words, ahead of
    /// the all-notes tests that stay ignored until every note is rewritten.
    #[test]
    fn torque_from_shear_keeps_to_the_slice_2_caps() {
        let id = "torque_from_shear";
        let note = note(id).unwrap_or_else(|| panic!("no note {id}"));
        assert_eq!(cap_problems(note), Vec::<String>::new(), "{id}");
        assert_eq!(
            sentence_zero_problems(note.sentences[0]),
            Vec::<&str>::new(),
            "{id}"
        );
    }

    #[test]
    fn a_step_reads_as_its_word_and_goes_its_way() {
        let steps = [Step::Raise, Step::Lower, Step::Choose, Step::Enter];
        assert_eq!(steps.map(Step::word), ["Raise", "Lower", "Choose", "Enter"]);
        assert_eq!(steps.map(Step::up), [Some(true), Some(false), None, None]);
    }

    #[test]
    fn a_sense_holds_only_for_a_move_its_way() {
        assert!(Sense::Up.holds(1.0, 2.0));
        assert!(!Sense::Up.holds(2.0, 1.0));
        assert!(!Sense::Up.holds(1.0, 1.0));
        assert!(Sense::Down.holds(2.0, 1.0));
        assert!(!Sense::Down.holds(1.0, 2.0));
        assert!(!Sense::Down.holds(1.0, 1.0));
        for sense in [Sense::Up, Sense::Down] {
            assert!(!sense.holds(f64::NAN, 1.0), "{sense:?}");
            assert!(!sense.holds(1.0, f64::NAN), "{sense:?}");
        }
    }

    /// What is wrong with a change entry (slice-2 spec 3.1 and section 6 test 2): an input
    /// that is no input; Choose on a number, Enter on a choice, or Raise or Lower on a choice
    /// (a selector or a part, grade or material name); a number with no slider to step; an aim
    /// on a choice or an entered number, or none on a step (a model assumption's step may name
    /// none: its effect depends on the design); an aim that is no number result of the default
    /// design.
    fn change_problems(change: &Change) -> Vec<String> {
        use crate::engine::api::{DesignInputs, compute_all};
        use crate::engine::meta::{FieldType, input_rows, result_rows};
        let inputs = input_rows(&DesignInputs::default());
        let Some(input) = inputs.iter().find(|r| r.path == change.input) else {
            return vec![format!("{} is no input", change.input)];
        };
        let choice = !input.meta.choices.is_empty() || input.meta.ty == FieldType::Text;
        let mut problems = Vec::new();
        match (change.step, choice) {
            (Step::Choose, true) | (Step::Enter, false) => {
                if change.aim.is_some() {
                    problems.push(format!(
                        "{} {}: names no aim",
                        change.step.word(),
                        change.input
                    ));
                }
            }
            (Step::Raise | Step::Lower, false) => {
                if input.meta.range.is_none() {
                    problems.push(format!("{} has no slider to step", change.input));
                }
                match change.aim {
                    // A model assumption's effect can depend on the design (a prototype match, a
                    // picked grade): its note's own claims test pins it design by design.
                    None if input.meta.assumption => {}
                    None => problems.push(format!(
                        "{} {}: a step names the result it moves",
                        change.step.word(),
                        change.input
                    )),
                    Some((aim, _)) => {
                        let results = result_rows(&compute_all(&DesignInputs::default()));
                        match results.iter().find(|r| r.path == aim) {
                            None => problems.push(format!("aim {aim} is no result")),
                            Some(r)
                                if !matches!(
                                    r.meta.ty,
                                    FieldType::F64 | FieldType::I64 | FieldType::NumOrText
                                ) =>
                            {
                                problems.push(format!("aim {aim} is not a number"))
                            }
                            Some(_) => {}
                        }
                    }
                }
            }
            (step, choice) => problems.push(format!(
                "{} on {}, a {}",
                step.word(),
                change.input,
                if choice { "choice" } else { "number" }
            )),
        }
        problems
    }

    #[test]
    fn the_change_rules_name_each_kind_of_mistake() {
        let entry = |input, step, aim| Change {
            input,
            step,
            aim,
            text: "",
        };
        let up = Some(("model.pullout_Nm", Sense::Up));
        for good in [
            entry("metal.face_gap_mm", Step::Lower, up),
            entry("coupling.npole", Step::Raise, up),
            entry("coupling.magnets.axial_length_mm", Step::Raise, up),
            entry("materials.parts.back_iron", Step::Choose, None),
            entry("coupling.magnets.part_inner", Step::Choose, None),
            entry("coupling.backiron", Step::Choose, None),
            entry("metal.measured_drag_Nm", Step::Enter, None),
            entry("calibration.f_cal_original", Step::Lower, None),
        ] {
            assert_eq!(change_problems(&good), Vec::<String>::new(), "{good:?}");
        }
        for (bad, why) in [
            (entry("no.such.input", Step::Raise, up), "is no input"),
            (entry("model.pullout_Nm", Step::Raise, up), "is no input"),
            (
                entry("materials.parts.back_iron", Step::Raise, up),
                "Raise on materials.parts.back_iron, a choice",
            ),
            (
                entry("metal.face_gap_mm", Step::Choose, None),
                "Choose on metal.face_gap_mm, a number",
            ),
            (
                entry("materials.parts.back_iron", Step::Enter, None),
                "Enter on materials.parts.back_iron, a choice",
            ),
            (
                entry("materials.parts.back_iron", Step::Choose, up),
                "names no aim",
            ),
            (
                entry("metal.measured_drag_Nm", Step::Enter, up),
                "names no aim",
            ),
            (
                entry("metal.face_gap_mm", Step::Lower, None),
                "a step names the result it moves",
            ),
            (
                entry(
                    "metal.face_gap_mm",
                    Step::Lower,
                    Some(("coupling.npole", Sense::Up)),
                ),
                "aim coupling.npole is no result",
            ),
            (
                entry(
                    "metal.face_gap_mm",
                    Step::Lower,
                    Some(("materials.cup_wall_check", Sense::Up)),
                ),
                "aim materials.cup_wall_check is not a number",
            ),
        ] {
            let problems = change_problems(&bad);
            assert!(
                problems.iter().any(|p| p.contains(why)),
                "{bad:?}: {problems:?}"
            );
        }
    }

    #[test]
    fn every_change_entry_names_an_input_and_the_result_it_moves() {
        let failures: Vec<String> = NOTES
            .iter()
            .flat_map(|n| {
                n.change
                    .iter()
                    .flat_map(change_problems)
                    .map(move |p| format!("{}: {p}", n.id))
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// The one note that may have no change entry: its warning never fires on the library
    /// (slice-2 ruling R13).
    const NO_CHANGE_NOTE: &str = "a5.ferromagnetic_sleeve_or_liner";

    /// What breaks the slice-2 caps in one note (words are the runs between whitespace, as
    /// `plain::word_count` counts them): its sentences, change entries and watch-outs. The one
    /// cap rule: the ignored all-notes test and the per-note tests of the note tasks call it.
    fn cap_problems(n: &Note) -> Vec<String> {
        use super::super::plain::word_count;
        let mut problems = Vec::new();
        if !(3..=5).contains(&n.sentences.len()) {
            problems.push(format!(
                "{}: {} sentences, not 3 to 5",
                n.id,
                n.sentences.len()
            ));
        }
        for (i, s) in n.sentences.iter().enumerate() {
            if word_count(s) > 25 {
                problems.push(format!(
                    "{}: sentence {i} has {} words",
                    n.id,
                    word_count(s)
                ));
            }
        }
        let fewest = usize::from(n.id != NO_CHANGE_NOTE);
        if !(fewest..=5).contains(&n.change.len()) {
            problems.push(format!(
                "{}: {} change entries, not {fewest} to 5",
                n.id,
                n.change.len()
            ));
        }
        for c in n.change {
            if word_count(c.text) > 25 {
                problems.push(format!(
                    "{}: the {} entry has {} words",
                    n.id,
                    c.input,
                    word_count(c.text)
                ));
            }
        }
        if n.watch_out.len() > 2 {
            problems.push(format!("{}: {} watch-outs", n.id, n.watch_out.len()));
        }
        for (i, w) in n.watch_out.iter().enumerate() {
            if word_count(w) > 30 {
                problems.push(format!(
                    "{}: watch-out {i} has {} words",
                    n.id,
                    word_count(w)
                ));
            }
        }
        problems
    }

    #[test]
    fn the_cap_rule_names_each_kind_of_mistake() {
        // Test data lives for the test: leaked, to fill the notes' `&'static` fields.
        let words = |n: usize| -> &'static str { vec!["word"; n].join(" ").leak() };
        let leak = |v: Vec<&'static str>| -> &'static [&'static str] { v.leak() };
        let entry = |text| Change {
            input: "metal.face_gap_mm",
            step: Step::Lower,
            aim: None,
            text,
        };
        let changes = |n: usize| -> &'static [Change] { vec![entry("an entry"); n].leak() };
        let good = Note {
            id: "test.caps",
            title: "A note",
            equations: &[],
            sentences: &["One.", "Two.", "Three."],
            change: vec![entry(words(25))].leak(),
            watch_out: leak(vec![words(30), words(1)]),
            diagram: None,
            sources: &["A source."],
            review: Review::Draft,
        };
        assert_eq!(cap_problems(&good), Vec::<String>::new(), "the limits fit");
        for (bad, why) in [
            (
                Note {
                    sentences: &["One.", "Two."],
                    ..good
                },
                "2 sentences, not 3 to 5",
            ),
            (
                Note {
                    sentences: leak(vec!["One."; 6]),
                    ..good
                },
                "6 sentences, not 3 to 5",
            ),
            (
                Note {
                    sentences: leak(vec!["One.", words(26), "Three."]),
                    ..good
                },
                "sentence 1 has 26 words",
            ),
            (
                Note {
                    change: &[],
                    ..good
                },
                "0 change entries, not 1 to 5",
            ),
            (
                Note {
                    change: changes(6),
                    ..good
                },
                "6 change entries, not 1 to 5",
            ),
            (
                Note {
                    change: vec![entry(words(26))].leak(),
                    ..good
                },
                "the metal.face_gap_mm entry has 26 words",
            ),
            (
                Note {
                    watch_out: leak(vec!["A.", "B.", "C."]),
                    ..good
                },
                "3 watch-outs",
            ),
            (
                Note {
                    watch_out: leak(vec!["A.", words(31)]),
                    ..good
                },
                "watch-out 1 has 31 words",
            ),
        ] {
            let problems = cap_problems(&bad);
            assert!(
                problems.iter().any(|p| p.contains(why)),
                "{why}: {problems:?}"
            );
        }
        // The one note allowed no change entry.
        let none = Note {
            id: NO_CHANGE_NOTE,
            change: &[],
            ..good
        };
        assert_eq!(cap_problems(&none), Vec::<String>::new());
    }

    #[test]
    #[ignore = "un-ignored in T18 when every note is rewritten"]
    fn every_note_keeps_to_the_slice_2_caps() {
        let failures: Vec<String> = NOTES.iter().flat_map(cap_problems).collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// What makes a sentence 0 more than plain words (slice-2 spec 3.1, F10), each named once:
    /// a live value, a formula sign or a symbol's subscript, a Greek letter (a symbol), a glyph
    /// the panel redraws (ϑ, the superscript minus, ∝; sentence 0 is counted as drawn).
    fn sentence_zero_problems(sentence: &str) -> Vec<&'static str> {
        let has = |bad: fn(char) -> bool| sentence.chars().any(bad);
        let mut problems = Vec::new();
        if has(|c| matches!(c, '{' | '}')) {
            problems.push("a live value");
        }
        if has(|c| matches!(c, '=' | '_' | '^')) {
            problems.push("a formula sign or subscript");
        }
        if has(|c| ('\u{0370}'..='\u{03FF}').contains(&c)) {
            problems.push("a Greek letter");
        }
        if has(|c| matches!(c, '\u{03D1}' | '\u{207B}' | '\u{221D}')) {
            problems.push("a glyph the panel redraws");
        }
        problems
    }

    #[test]
    fn the_sentence_zero_rule_names_what_it_finds() {
        assert!(
            sentence_zero_problems("The coupling slips when the load passes its pull-out torque.")
                .is_empty()
        );
        assert!(
            sentence_zero_problems("At 20 °C the torque is about 7 % higher, 1.2 N·m.").is_empty()
        );
        assert_eq!(
            sentence_zero_problems("Here {metal.variation}."),
            ["a live value"]
        );
        assert_eq!(
            sentence_zero_problems("T = σ A r."),
            ["a formula sign or subscript", "a Greek letter"]
        );
        assert_eq!(
            sentence_zero_problems("S_n falls."),
            ["a formula sign or subscript"]
        );
        assert_eq!(
            sentence_zero_problems("torque ∝ Br²"),
            ["a glyph the panel redraws"]
        );
        assert_eq!(
            sentence_zero_problems("at ϑ"),
            ["a Greek letter", "a glyph the panel redraws"]
        );
    }

    #[test]
    #[ignore = "un-ignored in T18 when every note is rewritten"]
    fn every_sentence_zero_is_plain_words() {
        let failures: Vec<String> = NOTES
            .iter()
            .filter_map(|n| {
                let problems = sentence_zero_problems(n.sentences.first().copied().unwrap_or(""));
                (!problems.is_empty()).then(|| format!("{}: {problems:?}", n.id))
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// What reads as jargon in one note's text (slice-2 spec 3.4, A-13): in its title, a
    /// sentence, a change text, a watch-out or a source, a word the plain-words guard bans (a
    /// live value stands as the number 1) and an audit id (`plain::audit_ids`), which the guard
    /// does not know.
    fn plain_words_problems(n: &Note) -> Vec<String> {
        use super::super::note_text::render;
        use super::super::plain::{audit_ids, plain_text_violations};
        let texts = std::iter::once(n.title)
            .chain(n.sentences.iter().copied())
            .chain(n.change.iter().map(|c| c.text))
            .chain(n.watch_out.iter().copied())
            .chain(n.sources.iter().copied());
        let mut problems = Vec::new();
        for text in texts {
            let banned = plain_text_violations(&render(text, |_| Some("1".to_owned())));
            if !banned.is_empty() {
                problems.push(format!("{}: {banned:?} in {text:?}", n.id));
            }
            for id in audit_ids(text) {
                problems.push(format!("{}: an audit id {id} in {text:?}", n.id));
            }
        }
        problems
    }

    /// The notes rewritten in plain words so far (slice-2 spec 3.4, A-13): each note task adds
    /// its ids; the task that turns the all-notes tests on deletes this list and
    /// `the_rewritten_notes_read_in_plain_words`.
    const REWRITTEN: &[&str] = &["torque_from_shear"];

    #[test]
    fn the_plain_words_rule_names_each_kind_of_mistake() {
        let good = Note {
            id: "test.plain",
            title: "A plain title",
            equations: &[],
            sentences: &["One.", "Two with {model.pullout_Nm}.", "Three."],
            change: &[Change {
                input: "metal.face_gap_mm",
                step: Step::Lower,
                aim: None,
                text: "a change in plain words.",
            }],
            watch_out: &["A watch-out in plain words."],
            diagram: None,
            sources: &["Math audit of 2026-09-29: the planar harmonic model"],
            review: Review::Draft,
        };
        assert_eq!(plain_words_problems(&good), Vec::<String>::new());
        let leak = |v: Vec<&'static str>| -> &'static [&'static str] { v.leak() };
        for (bad, why) in [
            (
                Note {
                    title: "How the workbook does it",
                    ..good
                },
                "workbook",
            ),
            (
                Note {
                    sentences: leak(vec!["One.", "Two (correction E7).", "Three."]),
                    ..good
                },
                "correction E",
            ),
            (
                Note {
                    change: vec![Change {
                        text: "see Calculator!C5.",
                        ..good.change[0]
                    }]
                    .leak(),
                    ..good
                },
                "Calculator!",
            ),
            (
                Note {
                    watch_out: &["Per decision 29 it is so."],
                    ..good
                },
                "decision",
            ),
            (
                Note {
                    sources: &["M1 audit E7"],
                    ..good
                },
                "M1 audit",
            ),
            (
                Note {
                    sources: &["A source (TC-RC2)"],
                    ..good
                },
                "an audit id TC-",
            ),
            (
                Note {
                    sources: &["PA-3 says so"],
                    ..good
                },
                "an audit id PA-",
            ),
            (
                Note {
                    sources: &["CG-1 says so"],
                    ..good
                },
                "an audit id CG-",
            ),
            (
                Note {
                    watch_out: &["As PA-8 found, it is so."],
                    ..good
                },
                "an audit id PA-",
            ),
        ] {
            let problems = plain_words_problems(&bad);
            assert!(
                problems.iter().any(|p| p.contains(why)),
                "{why}: {problems:?}"
            );
        }
    }

    #[test]
    fn the_rewritten_notes_read_in_plain_words() {
        let failures: Vec<String> = REWRITTEN
            .iter()
            .flat_map(|id| {
                let note = note(id).unwrap_or_else(|| panic!("REWRITTEN names no note: {id}"));
                plain_words_problems(note)
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    #[test]
    fn the_before_snapshot_format_reads_back() {
        let text = before::snapshot_text();
        let notes = before::parse(&text);
        assert_eq!(notes.len(), NOTES.len());
        for ((id, lines), n) in notes.iter().zip(NOTES) {
            assert_eq!(*id, n.id);
            let want: Vec<(&str, &str)> = std::iter::once(("title", n.title))
                .chain(n.sentences.iter().map(|s| ("sentence", *s)))
                .chain(n.watch_out.iter().map(|w| ("watch out", *w)))
                .chain(n.sources.iter().map(|s| ("source", *s)))
                .collect();
            assert_eq!(*lines, want, "{id}");
        }
    }

    #[test]
    #[ignore = "writes docs/analyses/2026-10-14-magcoupler-notes-before.txt once, before the first note is rewritten (T4)"]
    fn notes_before_snapshot() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/analyses/2026-10-14-magcoupler-notes-before.txt");
        assert!(
            !path.exists(),
            "{} exists: it holds the notes as they stood before slice 2 and is never written again",
            path.display()
        );
        std::fs::write(&path, before::snapshot_text()).expect("the snapshot is written");
    }

    #[test]
    fn the_before_snapshot_holds_the_seventeen_notes_slice_2_started_from() {
        let notes = before::parse(before::SNAPSHOT);
        assert_eq!(notes.len(), 17);
        let ids: std::collections::BTreeSet<&str> = notes.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 17, "each note once");
        for (id, lines) in &notes {
            assert!(note(id).is_some(), "{id}: a note id is never removed");
            let count = |field: &str| lines.iter().filter(|(f, _)| *f == field).count();
            assert_eq!(count("title"), 1, "{id}");
            assert!((2..=6).contains(&count("sentence")), "{id}");
            assert!(count("watch out") <= 1, "{id}");
            assert!(count("source") >= 1, "{id}");
            let fields = ["title", "sentence", "watch out", "source"].map(count);
            assert_eq!(
                lines.len(),
                fields.iter().sum::<usize>(),
                "{id}: no other field"
            );
            assert!(lines.iter().all(|(_, text)| !text.is_empty()), "{id}");
        }
    }
}
