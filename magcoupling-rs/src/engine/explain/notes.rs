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

/// Every note: 17, the spec's list (harmonics, back iron, pull-out against angle, end effect,
/// Br(T), demagnetization, slip loss, the thermal time constant, clamps and the six A5
/// warnings) plus ferrite's cold side (spec A6) and the one-point calibration.
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
            "Such a wave is a sum of sine waves at odd multiples of its basic frequency: harmonic n has n times as many wavelengths around the ring and an amplitude of 4/(nπ) times the remanence, scaled by sin(nπλ/2), where λ is the fraction of each pole the magnet fills; that factor, which depends on the gaps between blocks (λ < 1), sets each harmonic's size and sign, so some amplitudes come out negative.",
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
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "M1 audit M3 (planar factor against the cylindrical one)",
            "M1 audit M5 (steel at the magnet backs)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: Some(Diagram::TorqueAngle),
        sources: &[
            "M1 audit E7 (T3-RC2)",
            "Addendum A decision 29 (one peak search for any harmonic set)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "At or below L = c_end·τ_p (1.3 mm at the default pole pitch) the factor is zero or negative and every torque computed from it is meaningless.",
        ),
        diagram: Some(Diagram::EndFringing),
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
            "The 2D harmonic model is a few per cent off real flat blocks, so the workbook scales it by a calibration factor.",
            "When a design has the measured prototype's circuit (no back iron), pole count and magnet part (B842SH on both rings), it uses the bench result: f_cal,1 = f_cal,0 · T_meas / T_model, which puts the model of the prototype on the 1.8 N·m measurement.",
            "Because T_model already contains f_cal,0, the assumed factor cancels: the bench correction is T_meas divided by the uncalibrated model, whatever f_cal,0 was.",
            "Every other design keeps the assumed 0.95, since one measurement cannot say how the model's error changes with the design; the rule does not compare gaps or radii, so a resized design with the prototype's circuit, pole count and magnet part still gets the bench factor.",
        ],
        watch_out: Some(
            "The bench value has two significant figures and an assumed 20 °C test temperature; it sits about 1 % above the 3D pull-out at the corrected remanence of 1.30 T (correction E3), and 2.5 % above it at the audit's 1.29 T (M1 audit P2).",
        ),
        diagram: None,
        sources: &[
            "M1 audit M4 and M6 (the model's bias and the calibration that absorbs it)",
            "M1 audit P2 (the bench value against physics)",
            "M1 audit E3 (the N42SH remanence corrected from 1.29 T to 1.30 T, Calibration!C21 included)",
            "Plan A-3 traceability: the f_cal,0 cancellation (tests/explain.rs CANCELLATIONS)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
            "The loss is reversible: cool the magnet and the torque comes back, unless it passed its demagnetization limit on the way, which is a separate, permanent loss.",
        ],
        watch_out: Some(
            "Each ring keeps its own coefficient: a ferrite ring (α about −0.2 %/°C) loses torque faster with heat than an NdFeB one.",
        ),
        diagram: None,
        sources: &[
            "M1 audit, confirmed: the Br(T) and torque-temperature rows (Calculator C69, C70, C94; Metal design C8)",
            "Addendum A decision A2-7 (each ring with its own coefficient)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "The default N42SH's temperature coefficients hold over 20–150 °C only: onsets above 150 °C are extrapolations, and so is the reference magnet's model knee near 160 °C that sets the shift, so a ±20 % change of the slope above 150 °C moves even the default limit by up to about 3 °C (M1 audit M13).",
        ),
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "M1 audit M13 (knee temperature outside the coefficient range)",
            "M1 audit rulings: the onset calibration (it uses 9.83 °C of the 10 °C margin)",
            "Addendum A decision 19 (E20: each ring's own grade)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "A ferrite ring can pass every hot check and still fail at −40 °C: read the cold check too.",
        ),
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "Spec A6 (ferrite's opposite-sign beta gets a teaching note)",
            "Addendum A decisions 3 (Y30 beta +0.35 %/°C) and 19 (E20's positive-beta branch)",
            "A-1 plan decision A13 (both rings; the cold side from the ring with the higher cold limit)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
            "Thin shells such as the sleeve and liner are thinner than their skin depth, so their loss follows σ (ω_s r B)²/2 per unit volume, with ω_s the slip speed in rad/s, the square of speed.",
        ],
        watch_out: Some(
            "Aluminium parts (no back iron) are resistance-limited, not skin-limited, and see the weaker free-space field (correction E17); every loss here is an estimate with a factor-3 high case until a bench drag is measured.",
        ),
        diagram: None,
        sources: &[
            "M1 audit M11 (thin-skin steel losses against Stoll's half-space)",
            "M1 audit M12 (magnet eddy loss)",
            "M1 audit P1 (the shell end factor)",
            "Addendum A report section 5.5 (E17: the low-Reynolds form for aluminium parts)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "G is a placeholder (0.3 W/K, not measured): a 10 % higher G lowers the high-case steady temperature by about 2.5 °C (M1 audit placeholder table).",
        ),
        diagram: Some(Diagram::HeatingCurve),
        sources: &[
            "M1 audit placeholder inputs (conductance: −2.48 °C per +10 %)",
            "Addendum A decision 8 (E15: aluminium parts at their own specific heat)",
            "M1 audit E12 (a start above the limit)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "μ = 0.15 assumes a degreased bore; an oily bore (about 0.10) cuts the capacity by a third.",
        ),
        diagram: None,
        sources: &[
            "M1 audit M14 (the thread-stripping area)",
            "M1 audit E2 (the screw length and the slit)",
            "Addendum A3 assumptions: clamp friction and preload fraction",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "Spec A5 warning rules",
            "M1 audit M5 (steel at the magnet backs)",
            "Addendum A report section 5 (the no-back-iron circuit, E15 to E17)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A report section 4 (the materials table: ferromagnetic or not)",
            "Plan A-3 Task 16 physics review, the saturation cap: a shell of thickness t diverts at most about B_sat · t per unit length (default sleeve 0.1 mm, liner 0.2 mm)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit P1 (the shell loss and its end factor)",
            "Addendum A report section 4 (the conductivities)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: Some(
            "The wall formula itself reads 12 to 16 % thin against exact 2D sections (M1 audit M1), so walls near the limit deserve margin.",
        ),
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A decision 20 (the design flux density feeds the wall check)",
            "M1 audit M1 and M2 (the back-iron requirement)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Workbook Materials C26 (electroless nickel, 0.013–0.025 mm)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
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
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit E1 (the Volkersen shear-lag screen)",
            "Addendum A decision 16 (E18: the aluminium hub's expansion)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-01",
            record: "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations",
        },
    },
];

/// The suggested reading order (spec A4 "Start here": torque chain → back iron →
/// temperature → demagnetization → slip heating → clamps): (note id, the equation it opens).
pub const START_HERE: &[(&str, &str)] = &[
    ("harmonics", "model.tau_Pa"),
    ("pullout_angle", "model.pullout_Nm"),
    ("end_effect", "model.f_end"),
    ("calibration", "model.f_cal"),
    ("back_iron_factor", "model.s1_iron"),
    ("br_temperature", "model.pullout_20C_Nm"),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("ferrite_cold_demag", "temperature.demag.cold_check"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    (
        "thermal_time_constant",
        "temperature.thermal.time_constant_s",
    ),
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
}
