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
//! watch-outs of at most 30 words, a sentence 0 in plain words) are tests over every note
//! [`shown`]: while the shaft-clamp section is paused, its note keeps its slice-1 text, hidden.

/// A small diagram the M4 panel paints beside a note (egui painter, no image files).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagram {
    /// The magnetization's rectangular wave (blocks and spaces) and its first odd harmonics.
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
    /// First-order heating toward a steady temperature, the time constant marked, and a limit
    /// below that level with the time the rise takes to reach it.
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

/// Every note: 20, the cap the completeness test holds (15 to 20): the spec's list
/// (harmonics, back iron, pull-out against angle, end effect, Br(T), demagnetization, slip
/// heating, the slip temperature, the shaft clamp and the six A5 warnings), ferrite's cold
/// side (spec A6), the one-point calibration, and slice 2's three: how the coupling makes
/// torque, the hot-low and cold-high band, and what happens when the coupling slips. The
/// clamp note stays here, but is not shown while the shaft-clamp section is paused
/// ([`shown`]).
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
            "The magnet count N keeps the gap radius R_g: blocks share one ring, each pole spanning τ_p = 2π·R_g/N; fewer flat blocks grow the cup.",
            "σ grows with stronger magnet grades and magnetic steel behind the magnets; it falls as the gap widens or blocks fill less of each pole.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "torque rises almost linearly; on the example design each millimetre beats a millimetre of radius. Costs: longer stack, higher shaft-joint and gearbox loads.",
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
            "The dashboard's pull-out torque, the most the rings are expected to pass before slipping, is T_2D corrected for the magnet ends and by a calibration factor (here: {model.torque_2d_Nm} becomes {model.pullout_Nm}).",
            "T_2D grows as R_g² only at fixed σ: a larger ring with the same magnet count fills each pole less, so the pull-out can even fall.",
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
        title: "Block fill and the magnet pattern (harmonics)",
        equations: &[
            "model.b_i#",
            "model.b_o#",
            "model.amp#_Pa",
            "model.tau_Pa",
            "calibration.amp#_Pa",
            "model.fill_inner",
            "model.fill_outer",
        ],
        sentences: &[
            "Around each ring, blocks alternate north and south with spaces between them, so the magnet pattern along the gap is squarish, not smooth.",
            "It equals a sine wave, the fundamental (one wave per pair of poles), plus weaker odd harmonics with 3, 5, 7 times the waves.",
            "The fundamental grows with sin(π·λ/2), where the fill λ is the block's width over the room each pole has on that ring.",
            "Each harmonic pulls only on its match on the other ring; higher pairs usually add little, being weaker and fading faster across the gap.",
            "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer; the fundamental alone peaks at {model.amp1_Pa}, the summed harmonics at σ = {model.tau_Pa}.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.part_outer",
                step: Step::Choose,
                aim: None,
                text: "a wider, equally thick outer block raises outer fill and torque if blocks fit; shaft-joint and gearbox loads rise; its grade may tolerate less heat.",
            },
            Change {
                input: "coupling.magnets.manual_outer_width_mm",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "with custom dimensions, a wider outer block fills more of its pole and raises torque while the blocks still fit; shaft-joint and gearbox loads rise.",
            },
            Change {
                input: "coupling.npole",
                step: Step::Raise,
                aim: Some(("model.fill_outer", Sense::Up)),
                text: "more blocks on the same ring fill more of each pole; they must still fit, and fewer magnets leave wider spaces between blocks.",
            },
            Change {
                input: "coupling.max_harmonic",
                step: Step::Choose,
                aim: None,
                text: "a model setting, not a design lever: keep at least 1, 3, 5; more for narrow, widely spaced blocks, where higher harmonics can move pull-out.",
            },
        ],
        watch_out: &[
            "At low fill, from few magnets, a large ring or narrow blocks, the third harmonic can change the pull-out by more than a tenth and move its angle.",
        ],
        diagram: Some(Diagram::SquareWaveHarmonics),
        sources: &[
            "Math audit of 2026-09-29: the planar harmonic model against exact 2D sections",
            "The odd-harmonic set, summed up to 11",
            "Physics read of 2026-10-14: the third harmonic at low fill (6 magnets, a 20 mm ring radius, narrow blocks)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#harmonics",
        },
    },
    Note {
        id: "back_iron_factor",
        title: "Back iron: why steel behind the magnets adds torque",
        equations: &["model.s#_iron", "model.s#_free", "model.k#"],
        sentences: &[
            "Magnetic steel behind each ring gives the flux an easy return path, so more of it crosses the gap instead of leaking out behind.",
            "The geometry factor S_n, which turns both rings' magnet strengths into pull across the gap, is therefore always larger with steel.",
            "On this design S_1 is {model.s1_iron} with magnetic steel and {model.s1_free} without; the gain is largest when the pole pitch is long against magnet thickness.",
            "Across the gap the fundamental fades about as e^(−π·g/τ_p), g being the gap, so what matters is the gap against the pole pitch.",
            "S_1 has a best pole pitch: too long for thin magnets and the field is weak; too short and it fades across the gap.",
        ],
        change: &[
            Change {
                input: "materials.parts.back_iron",
                step: Step::Choose,
                aim: None,
                text: "4140 or 1018 keep the steel circuit; 304 and aluminium switch to free space and lose much torque; 12L14, 416 and 17-4PH may saturate.",
            },
            Change {
                input: "coupling.backiron",
                step: Step::Choose,
                aim: None,
                text: "keep the steel circuit; with no back iron the torque model uses free space, whatever material is picked.",
            },
            Change {
                input: "metal.face_gap_mm",
                step: Step::Lower,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "less of the field fades across the gap, so torque rises; running clearance and room for the sleeve and liner shrink.",
            },
            Change {
                input: "coupling.magnets.manual_outer_thickness_mm",
                step: Step::Raise,
                aim: Some(("model.s1_iron", Sense::Up)),
                text: "a thicker outer block raises S_n with shrinking gains, but it lowers the outer fill, so past a point torque falls; the cup grows.",
            },
        ],
        watch_out: &[
            "The torque assumes the steel never saturates: where the cup wall check fails, the real coupling can fall below the torque shown.",
        ],
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "Math audit of 2026-09-29: the planar back-iron factor against the cylindrical one",
            "Math audit of 2026-09-29: steel at the magnet backs",
            "Guidance audit of 2026-10-13: S_1 with steel at 8, 10 and 12 magnets (its best pole pitch)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#back_iron_factor",
        },
    },
    Note {
        id: "pullout_angle",
        title: "Pull-out: the most torque before the rings slip",
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
            "Pull-out torque is the peak of torque against twist: the most the coupling is expected to pass before the rings slip, a nominal estimate.",
            "It is computed at the magnet temperature while running (here: {metal.op_temp_C}) as T_pull = T_2D·f_end·f_cal: the endless-ring torque times end and calibration factors.",
            "Angles in this model are electrical: φ = N/2 times the mechanical twist, so 90° is half a pole pitch, a twist of 180°/N.",
            "The fundamental alone peaks at 90°; at low fill a same-sign third harmonic above about a ninth of it shifts the peak below 90°.",
            "Loaded past it, the rings slip: torque then swings between plus and minus pull-out once per pole pair passed, averaging only a small drag.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "pull-out grows almost linearly with the overlap; the stack lengthens, and shaft-joint and gearbox loads rise.",
            },
            Change {
                input: "coupling.magnets.part_inner",
                step: Step::Choose,
                aim: None,
                text: "a same-size, stronger-grade block raises pull-out proportionally to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops.",
            },
            Change {
                input: "coupling.op_temp_C",
                step: Step::Lower,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "the pull-out is reported at this temperature; lower it only if the magnets really run cooler, or the estimate is optimistic.",
            },
        ],
        watch_out: &[
            "A slipping coupling may keep slipping after the load drops below pull-out: it locks again only once the slip speed falls, and meanwhile it heats.",
        ],
        diagram: Some(Diagram::TorqueAngle),
        sources: &[
            "Math audit of 2026-09-29: the pull-out is the highest point of the summed curve, not always half a pole pitch",
            "One peak search for any harmonic set",
            "Guidance audit of 2026-10-13: the same-sign third harmonic above a ninth; electrical against mechanical angle",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#pullout_angle",
        },
    },
    Note {
        id: "end_effect",
        title: "Magnet ends: the length lost to fringing",
        equations: &["model.f_end", "calibration.f_end"],
        sentences: &[
            "Near each magnet end some of the field fringes outward instead of crossing the gap, so the ends carry less torque than the middle.",
            "The model counts this as missing length: f_end = 1 − c_end·τ_p/L, with c_end an empirical coefficient, τ_p the pole pitch, L the magnet overlap.",
            "It takes c_end·τ_p, half at each end, off L; here f_end is {model.f_end}, τ_p {model.pole_pitch_mm} and L {model.active_length_mm}.",
            "So torque grows in step with L − c_end·τ_p: every extra millimetre of overlap adds the same torque, and the share lost shrinks.",
            "Steel-free 3D puts the example design's f_end within 3 % from 3.2 mm; with magnetic steel behind the magnets, the default, f_end likely reads high.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("model.f_end", Sense::Up)),
                text: "longer magnets lose a smaller share at the ends and add torque; the stack grows, and shaft-joint and gearbox loads rise with cold-high.",
            },
            Change {
                input: "coupling.c_end",
                step: Step::Raise,
                aim: Some(("model.f_end", Sense::Down)),
                text: "only with 3D or bench evidence; every torque falls, cold-high and the shaft-joint and gearbox loads included. The bench prototype keeps its own coefficient.",
            },
        ],
        watch_out: &[
            "Its end loss barely grows with the gap: in a 3D check at a 4.4 mm gap it read 11 % high, so treat wide-gap torque as optimistic.",
            "At short overlap f_end is least reliable and can read too high or too low; at or below c_end·τ_p there is no prediction and the end-effect check flags it.",
        ],
        diagram: Some(Diagram::EndFringing),
        sources: &[
            "Math audit of 2026-09-29: the end factor against free-space 3D rings, at wide gaps and for short magnets",
            "Physics review of 2026-10-15: the end factor against free-space 3D by magnet length (within 3 % from 3.2 mm), by magnets per ring and at a 20 mm ring radius",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#end_effect",
        },
    },
    Note {
        id: "calibration",
        title: "Calibration: what scales the torque estimate",
        equations: &[
            "model.f_cal",
            "calibration.f_cal_updated",
            "calibration.measured_over_model",
            "calibration.model_error",
            "calibration.model_torque_Nm",
        ],
        sentences: &[
            "The calculator multiplies its torque model by a calibration factor; for most designs that factor is an assumed allowance, not a measurement.",
            "The only bench test it holds is of a prototype without steel, so for a design backed by magnetic steel the factor is a judgement.",
            "A design gets the measured factor only with no magnetic steel behind the magnets, the prototype's magnets per ring and B842SH parts on both rings.",
            "The measured factor makes the prototype's modelled torque match its bench measurement; the rule ignores ring radius, gap and magnet length.",
            "Entering or leaving that match steps every torque between the two factors, by rule, not physics; the factor in use here is {model.f_cal}.",
        ],
        change: &[
            Change {
                input: "calibration.f_cal_original",
                step: Step::Lower,
                aim: None,
                text: "every torque outside a prototype match falls in proportion, cold-high included; a match ignores it, and no steel bench test backs any value.",
            },
            Change {
                input: "calibration.measured_torque_Nm",
                step: Step::Enter,
                aim: None,
                text: "the measured bench pull-out, never a value for your design; the measured factor rises with it, and only a prototype match uses it.",
            },
            Change {
                input: "calibration.test_temp_C",
                step: Step::Enter,
                aim: None,
                text: "the recorded bench magnet temperature; a warmer test weakens the modelled prototype, so the measured factor rises.",
            },
        ],
        watch_out: &[
            "The prototype inputs describe that one bench test: enter a real test's values, never edit them to match a new design, which gives a meaningless factor.",
        ],
        diagram: None,
        sources: &[
            "Math audit of 2026-09-29: the model's bias without and with back iron, and the calibration that absorbs it",
            "Math audit of 2026-09-29: the bench value against physics",
            "Math audit of 2026-09-29: the N42SH remanence corrected from 1.29 T to 1.30 T, the prototype's included",
            "Traceability check: the assumed factor cancels out of the bench correction",
            "Guidance audit of 2026-10-13: the step on leaving a match; the rule compares no size",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#calibration",
        },
    },
    Note {
        id: "br_temperature",
        title: "Heat weakens magnets: torque against temperature",
        equations: &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
            "metal.torque_cold_Nm",
            "temperature.magnet_life.torque_hot_day_Nm",
            "temperature.magnet_life.torque_peak_Nm",
            "temperature.demag.torque_at_limit_Nm",
            "model.alpha_br_per_C",
            "model.inner_alpha_br_per_C",
            "model.outer_alpha_br_per_C",
        ],
        sentences: &[
            "Magnets lose strength reversibly as they warm, so the pull-out torque depends on the magnet temperature.",
            "Remanence Br, a magnet's built-in flux density, falls by a fixed share of its 20 °C value per °C: the coefficient α.",
            "Torque is proportional to inner Br times outer Br, so each °C of warming changes it by a share of about α_i + α_o.",
            "Here the pull-out is {model.pullout_Nm} at {metal.op_temp_C} and {model.pullout_20C_Nm} at 20 °C: the dashboard's two pull-out rows.",
            "Each torque has its own temperature: hot-low the operating one, cold-high the minimum, the hot-day torque ambient plus the driving rise.",
        ],
        change: &[
            Change {
                input: "coupling.op_temp_C",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Down)),
                text: "set it to the magnet temperature while running; hotter lowers the pull-out and hot-low, not the 20 °C row.",
            },
            Change {
                input: "calibration.alpha_br_per_C",
                step: Step::Raise,
                aim: None,
                text: "a value nearer zero changes torque less with temperature, hot and cold. Take it from the datasheet; rings with a picked grade ignore it.",
            },
            Change {
                input: "coupling.magnets.grade_inner",
                step: Step::Choose,
                aim: None,
                text: "a grade whose coefficient is nearer zero loses a smaller share when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only.",
            },
            Change {
                input: "temperature.duty.hot_ambient_C",
                step: Step::Raise,
                aim: Some(("temperature.magnet_life.torque_hot_day_Nm", Sense::Down)),
                text: "a hotter day lowers the hot-day torque and the temperature margins; the hot-day start is this ambient plus the driving rise.",
            },
        ],
        watch_out: &[
            "Library magnets share one coefficient input; only a ring with custom dimensions and a picked grade uses its grade's own. In use here: inner {model.inner_alpha_br_per_C}, outer {model.outer_alpha_br_per_C}.",
            "The loss is reversible only inside the magnet's limits: too hot for NdFeB, or too cold for ferrite, and it loses strength for good.",
        ],
        diagram: None,
        sources: &[
            "Math audit of 2026-09-29, confirmed: the remanence and torque-against-temperature rows",
            "Design ruling: a ring with custom dimensions and a picked grade uses its grade's coefficient; library parts share one input",
            "Guidance audit of 2026-10-13: which temperature each torque uses; the two pull-out rows",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#br_temperature",
        },
    },
    Note {
        id: "hot_low_cold_high",
        title: "Hot-low and cold-high: the torque band",
        equations: &[
            "metal.torque_hot_low_Nm",
            "metal.torque_cold_high_Nm",
            "metal.hot_min_check",
            "metal.hot_margin",
            "metal.required_20C_Nm",
            "metal.required_20C_zero_scatter_Nm",
            "metal.cold_for_hot_min_Nm",
            "metal.cold_for_hot_min_input_Nm",
            "metal.torque_op_Nm",
            "metal.torque_20C_Nm",
            "metal.torque_cold_zero_var_Nm",
            "metal.cold_high_Nm",
            "metal.cold_high_input_Nm",
        ],
        sentences: &[
            "Hot-low is the weakest coupling expected at the operating temperature, and cold-high the strongest at the minimum temperature, allowing for production scatter.",
            "Hot-low, here {metal.torque_hot_low_Nm}, is the pull-out at the operating temperature less the variation allowance; it must reach the requirement.",
            "Cold-high, here {metal.torque_cold_high_Nm}, is the pull-out at the minimum temperature plus that allowance: the most torque expected before it slips.",
            "Whatever holds the hub on the shaft must carry cold-high with margin, and the gearbox output carries it too, so added torque loads both.",
            "On this design, hot-low passes once the 20 °C pull-out reaches {metal.required_20C_Nm}, the requirement taken back to 20 °C with the allowance.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("metal.torque_hot_low_Nm", Sense::Up)),
                text: "torque rises at every temperature, so hot-low rises; cold-high rises too, and with it the shaft-joint and gearbox loads; the stack grows.",
            },
            Change {
                input: "metal.variation",
                step: Step::Lower,
                aim: Some(("metal.torque_hot_low_Nm", Sense::Up)),
                text: "only with measured production scatter; the band narrows both ways, so hot-low rises and cold-high falls.",
            },
            Change {
                input: "coupling.op_temp_C",
                step: Step::Lower,
                aim: Some(("metal.torque_hot_low_Nm", Sense::Up)),
                text: "only if the magnets really run cooler; hot-low rises and cold-high stays. The hot-day torque is reported apart.",
            },
            Change {
                input: "metal.min_temp_C",
                step: Step::Lower,
                aim: Some(("metal.torque_cold_high_Nm", Sense::Up)),
                text: "a colder minimum raises cold-high and with it the shaft-joint and gearbox loads; enter the coldest the magnets really get.",
            },
        ],
        watch_out: &[
            "Hot-low uses the operating temperature you enter, not the hot-day start ({temperature.duty.hot_day_start_C} here); the hot-day torque is reported apart, without the allowance.",
            "Nothing checks cold-high against the gearbox: at its input it is {metal.cold_high_input_Nm} here; compare that with the gearbox's input rating yourself.",
        ],
        diagram: None,
        sources: &[
            "Model assumption: the production variation allowance (symmetric; an engineering allowance, not measured)",
            "Math audit of 2026-09-29, confirmed: the hot and cold torque rows and the hot-minimum check",
            "Design rulings of 2026-10-14 and 2026-10-15: the coupling sits on the gearbox output, and whatever holds the hub on the shaft carries the cold-high torque",
            "Guidance audit of 2026-10-13: which temperature each torque uses; the gearbox input",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#hot_low_cold_high",
        },
    },
    Note {
        id: "demagnetization",
        title: "Demagnetization: permanent loss from heat and slipping",
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
            "temperature.magnet_life.margin_onset_C",
            "temperature.magnet_life.margin_limit_C",
        ],
        sentences: &[
            "Too much heat combined with a strong opposing field can weaken a magnet for good, unlike the loss with temperature that cooling undoes.",
            "A field pushing against the magnet's direction, a reverse field, does no lasting harm until it passes the knee, where the magnet stops springing back.",
            "Heat lowers an NdFeB magnet's knee, so each reverse field has an onset temperature; above it, part of the magnetization is lost for good.",
            "Slipping passes like poles face to face, the worst reverse field; its onset, here {temperature.demag.onset_skipping_C}, sets an NdFeB ring's limit.",
            "Onsets are shifted to match the magnet's rated maximum temperature; each NdFeB ring's limit is its slipping onset less a margin; the lower limit governs.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.part_inner",
                step: Step::Choose,
                aim: None,
                text: "custom dimensions and an NdFeB grade rated hotter usually raise this ring's limit; those above N42SH are weaker, cutting torque, and the adhesive may govern.",
            },
            Change {
                input: "coupling.magnets.part_outer",
                step: Step::Choose,
                aim: None,
                text: "the same for the outer ring: the lower of the two limits governs, so a hotter grade on one ring alone may not help.",
            },
            Change {
                input: "temperature.demag.h_rev_likepole_kA_m",
                step: Step::Enter,
                aim: None,
                text: "enter it from a 3D field calculation of your design, which needs an outside solver; the stored value belongs to the example design.",
            },
        ],
        watch_out: &[
            "The reverse fields are stored from the example design's 3D model: they ignore magnet count, ring radius, gap, block size and grade, reading low for stronger grades.",
            "At their grade's own rating, onsets for N52, N50, N50M, N38UH, N35EH and 2:17 SmCo are shifted upward: limits may read optimistic; onsets past a grade's tested range are extrapolated.",
        ],
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "Math audit of 2026-09-29: the knee temperature outside the coefficient range",
            "Math audit of 2026-09-29: other calibration forms move the skipping onset by up to 9.83 °C",
            "Design ruling: each ring checked with its own grade",
            "Guidance audit of 2026-10-13: the stored reverse fields; the rating shift raises the onsets of grades rated above their model knee temperature",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#demagnetization",
        },
    },
    Note {
        id: "ferrite_cold_demag",
        title: "Ferrite magnets: demagnetization risk at the cold end",
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
            "Ferrite magnets risk permanent loss when cold, not hot: cooling lowers their knee while their own field, and so the reverse field, grows.",
            "Below a cold onset the reverse field passes the knee, and the ring loses part of its magnetization for good; slipping sets that onset.",
            "The cold limit is that onset plus a margin; the coldest the magnets get must stay at or above it, on both rings.",
            "Heat moves a ferrite knee away from the reverse field, so a ferrite ring's hot limit is simply its rated maximum temperature.",
            "With NdFeB on both rings the cold check does not apply and reads n/a.",
        ],
        change: &[
            Change {
                input: "coupling.magnets.part_inner",
                step: Step::Choose,
                aim: None,
                text: "custom dimensions with the Y30 grade model a ferrite ring, with far less torque than same-size library NdFeB parts, which trade cold risk for hot.",
            },
            Change {
                input: "coupling.magnets.part_outer",
                step: Step::Choose,
                aim: None,
                text: "the same for the outer ring; the cold check passes only if both rings pass.",
            },
            Change {
                input: "temperature.demag.h_rev_likepole_kA_m",
                step: Step::Enter,
                aim: None,
                text: "enter your ferrite design's own value from a 3D field calculation; the stored one is the NdFeB example's, and yours may still fail.",
            },
        ],
        watch_out: &[
            "The default reverse field for slipping, from the NdFeB example design, exceeds a Y30 knee at every temperature Y30 is rated for, so the check always fails with it.",
            "Y30's temperature coefficients are measured only from 20 to 120 °C; the cold check extrapolates them to the coldest temperature.",
        ],
        diagram: None,
        sources: &[
            "Design spec: ferrite's opposite-sign coercivity coefficient gets a teaching note",
            "Design rulings: Y30's coefficient (+0.35 %/°C) and the positive-coefficient branch of the demagnetization check",
            "Design ruling: both rings checked; the cold side from the ring with the higher cold limit",
            "Guidance audit of 2026-10-13: the cold limit above room temperature; the ferrite-only step; a typed positive coefficient is held within the slider's negative range; Y30's 20 to 120 °C coefficient range",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#ferrite_cold_demag",
        },
    },
    Note {
        id: "when_slipping",
        title: "When the coupling slips",
        equations: &["model.gearbox_input_ripple_Nm", "metal.slip_freq_Hz"],
        sentences: &[
            "Past pull-out the coupling slips: one ring turns against the other, and the torque between them, carried by the gearbox, swings back and forth.",
            "The swing runs from plus to minus the pull-out once per pole pair passed, so the load gets almost no torque on average.",
            "What remains is a drag from eddy currents, which the moving field drives in the metal parts, heating them: here an estimated {temperature.slip_loss.drag_Nm}.",
            "The rings re-lock only once the slip speed falls, so slow the drive; a load dropping just below pull-out is not enough.",
            "Each swing puts like poles face to face, the magnets' hardest case, and sends torque ripple to the gearbox: at its input, here {model.gearbox_input_ripple_Nm}.",
        ],
        change: &[
            Change {
                input: "temperature.duty.fault_trip_s",
                step: Step::Lower,
                aim: Some(("temperature.thermal.temp_at_fault_C", Sense::Down)),
                text: "a shorter unbroken slip heats the magnets less; detect slip and slow the drive within it, without tripping on brief slips.",
            },
            Change {
                input: "metal.slip_rpm",
                step: Step::Lower,
                aim: Some(("temperature.slip_loss.total_W", Sense::Down)),
                text: "slower slipping makes less heat and slower swings, here {metal.slip_freq_Hz}; the drive's behaviour after slip sets it, not the coupling.",
            },
        ],
        watch_out: &[
            "Nothing checks the gearbox against these swings: when cold they can reach the cold-high torque, on this design {metal.cold_high_input_Nm} at its input; check its rating.",
            "The drag, the heating and the magnets' limits use fields stored from the example design's 3D model; those fields do not follow your geometry or magnet grade.",
        ],
        diagram: None,
        sources: &[
            "The torque-angle model: pull-out as the peak of the summed harmonics, one cycle per pole pair",
            "The slip-loss and thermal model: the drag, the heating and the slip fault trip time",
            "The demagnetization model: the like-poles-facing reverse field while slipping sets the magnet limit",
            "The gearbox rows: the ripple at the gearbox input is the pull-out through the gear ratio and efficiency",
            "Guidance audit of 2026-10-13: what happens past pull-out; the rings re-lock only when the slip speed falls",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#when_slipping",
        },
    },
    Note {
        id: "slip_heating",
        title: "Slip heating: where the heat comes from",
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
            "temperature.duty.field_freq_Hz",
            "temperature.duty.pole_pairs",
            "temperature.duty.slip_rad_s",
        ],
        sentences: &[
            "When the coupling slips, each ring's field sweeps over the other ring's metal parts and magnets, driving eddy currents that heat them.",
            "The heat power is the drag torque times the slip speed: here an estimated {temperature.slip_loss.total_W}, a drag of {temperature.slip_loss.drag_Nm}.",
            "In thick magnetic steel the currents crowd into a thin surface skin, so its loss grows about as slip speed to the power 1.5.",
            "The thin sleeve, liner and cap, and the magnets, heat through their whole thickness, so their loss grows as the square of the speed.",
            "Without magnetic steel back iron the hub, cup and web are non-magnetic metal: they see a weaker field and heat through their whole thickness.",
        ],
        change: &[
            Change {
                input: "metal.measured_drag_Nm",
                step: Step::Enter,
                aim: None,
                text: "your bench value from a slipping test; it replaces the estimated loss and its high case in the temperature results.",
            },
            Change {
                input: "materials.parts.sleeve_liner",
                step: Step::Choose,
                aim: None,
                text: "titanium, Inconel or PEEK conduct less than 316L, so the sleeve and liner heat less; check their strength and temperature limits.",
            },
            Change {
                input: "metal.sleeve_mm",
                step: Step::Lower,
                aim: Some(("temperature.slip_loss.total_W", Sense::Down)),
                text: "a thinner sleeve heats less and widens the running clearance, but holds the inner magnets less securely; nothing here checks that.",
            },
            Change {
                input: "metal.liner_mm",
                step: Step::Lower,
                aim: Some(("temperature.slip_loss.total_W", Sense::Down)),
                text: "a thinner liner heats less and widens the running clearance, but holds the outer magnets less securely; nothing here checks that.",
            },
        ],
        watch_out: &[
            "The losses use fields stored from a 3D model of the example design; those fields do not follow your magnet count, ring radius, gap, magnet size or grade.",
            "Until a drag is measured, the high case, here {temperature.slip_loss.high_W}, is the estimate times an allowance for the model's errors; the temperature check uses it.",
        ],
        diagram: None,
        sources: &[
            "Math audit of 2026-09-29: thin-skin steel losses against Stoll's half-space solution",
            "Math audit of 2026-09-29: the magnet eddy loss",
            "Math audit of 2026-09-29: the shell end factor",
            "Design report: the weak-field, whole-thickness loss of non-magnetic body parts",
            "Guidance audit of 2026-10-13: the stored slip fields; the steel loss's speed exponent is about 1.54 to 1.64",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#slip_heating",
        },
    },
    Note {
        id: "thermal_time_constant",
        title: "Slip temperature: how hot, and how fast",
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
            "temperature.thermal.steady_est_C",
            "temperature.thermal.steady_high_C",
            "temperature.slip_life.rise_per_event_high_C",
            "temperature.magnet_life.peak_C",
        ],
        sentences: &[
            "Unbroken slip heats the coupling toward a steady rise set by loss and heat path; mass changes only how fast, which the time constant measures.",
            "One lump stands for the rotating parts: heat capacity C, conductance G to its surroundings, starting at the hot-day start, hot ambient plus driving rise.",
            "Slipping with loss P heats it toward a rise of P/G above that start, reaching 63 % after τ_th = C/G, here {temperature.thermal.time_constant_s}.",
            "One short slip adds only P·t/C, but many slips over the life add an average rise: the slip duty times P/G.",
            "The temperature check uses the peak: the hot-day start plus the hotter of a fault slip and one slip event, plus that average rise.",
        ],
        change: &[
            Change {
                input: "temperature.thermal.conductance_W_K",
                step: Step::Raise,
                aim: Some(("temperature.thermal.steady_high_C", Sense::Down)),
                text: "measure it first, as it is a placeholder; a better heat path through shafts and housing lowers the steady temperature and shortens τ_th.",
            },
            Change {
                input: "metal.measured_drag_Nm",
                step: Step::Enter,
                aim: None,
                text: "your bench value from a slipping test; the loss and its high case then follow the measurement.",
            },
            Change {
                input: "temperature.duty.fault_trip_s",
                step: Step::Lower,
                aim: Some(("temperature.magnet_life.peak_C", Sense::Down)),
                text: "a shorter unbroken slip can lower the peak, here {temperature.magnet_life.peak_C}, but not under a limit the hot-day start already reaches.",
            },
        ],
        watch_out: &[
            "Time to the limit reads 'never' when the high-case steady temperature, here {temperature.thermal.steady_high_C}, stays at or below the limit, here {temperature.summary.governing_limit_C}; that rests on a placeholder conductance.",
            "The single lump puts every part at one temperature; in a short fault the parts next to the gap run hotter than its average.",
        ],
        diagram: Some(Diagram::HeatingCurve),
        sources: &[
            "Math audit of 2026-09-29: the placeholder inputs and how far each moves the results, the conductance among them",
            "Design ruling: non-magnetic body parts at their own specific heat",
            "Math audit of 2026-09-29: a start above the limit",
            "Guidance audit of 2026-10-13: the panel's symbols τ_th and the hot-day start; the limit on the heating curve",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#thermal_time_constant",
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
        title: "No magnetic steel behind the magnets",
        equations: &["materials.circuit_backiron"],
        sentences: &[
            "Magnetic steel behind the rings carries flux from pole to pole; without it, flux spreads behind the magnets, less crosses the gap and torque drops.",
            "304 stainless and 6061 aluminium are non-magnetic, like air: picking either for the back iron, or switching the circuit off, selects the no-back-iron model.",
            "That model keeps much less torque: on this design the same layout gives {model.pullout_iron_Nm} with unsaturated steel and {model.pullout_noiron_Nm} without.",
            "Without steel the field also spreads outside the coupling and pulls in steel chips; the calculator does not compute this stray field.",
            "The hub, cup and boss then become non-magnetic metal, the cup-wall check no longer applies, and the expansion-mismatch caution fires.",
        ],
        change: &[
            Change {
                input: "materials.parts.back_iron",
                step: Step::Choose,
                aim: None,
                text: "4140 or 1018 restore steel torque but bring back the cup-wall check, plating, steel's mass and slip heat, and higher shaft-joint and gearbox loads.",
            },
            Change {
                input: "coupling.backiron",
                step: Step::Choose,
                aim: None,
                text: "the steel circuit, once a magnetic steel is picked, restores torque but adds mass, brings back the cup-wall check and raises shaft-joint and gearbox loads.",
            },
            Change {
                input: "coupling.magnets.axial_length_mm",
                step: Step::Raise,
                aim: Some(("model.pullout_Nm", Sense::Up)),
                text: "if you stay non-magnetic, longer magnets win back torque; the coupling grows longer and heavier, and shaft-joint and gearbox loads rise with the cold-high torque.",
            },
        ],
        watch_out: &[
            "Only no-back-iron designs with the bench prototype's magnets per ring and B842SH on both rings get its measured calibration; elsewhere the assumed factor applies, so the torque shown jumps.",
        ],
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "Material warning rules",
            "Math audit of 2026-09-29: steel at the magnet backs",
            "Design report: the no-back-iron circuit",
            "The one-point calibration rule (the bench factor only for the prototype's circuit, magnet count and magnet part)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.non_ferromagnetic_back_iron",
        },
    },
    Note {
        id: "a5.ferromagnetic_sleeve_or_liner",
        title: "Magnetic sleeve or liner",
        equations: &[],
        sentences: &[
            "The sleeve and liner sit in the gap that all torque-making flux must cross from ring to ring.",
            "A magnetic shell would let flux short-cut from pole to pole on one ring, so less would cross the gap and make torque.",
            "Every listed sleeve and liner material is non-magnetic, so this warning cannot fire; the torque model has no term for it.",
        ],
        change: &[],
        watch_out: &[],
        diagram: None,
        sources: &[
            "Material warning rules",
            "Design report: the materials table, magnetic or not",
            "An earlier physics review: a shell of thickness t diverts at most about B_sat · t of flux per unit length",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.ferromagnetic_sleeve_or_liner",
        },
    },
    Note {
        id: "a5.high_conductivity_sleeve_or_liner",
        title: "Sleeve or liner that conducts better than 316L",
        equations: &[
            "temperature.slip_loss.sleeve_W",
            "temperature.slip_loss.liner_W",
            "materials.sleeve_sigma_S_m",
        ],
        sentences: &[
            "When the coupling slips, the sleeve and liner cut through the other ring's field, and the eddy currents this drives heat them.",
            "Their heat rises in proportion to conductivity and thickness, and with the square of the slip speed and of the field they see.",
            "The caution fires above 316L's 1.35 million S/m; no listed material conducts more, but a higher value typed for 316L fires it.",
            "On this design the sleeve and liner make {temperature.slip_loss.sleeve_W} and {temperature.slip_loss.liner_W} of the {temperature.slip_loss.total_W} slip loss estimate.",
            "More heat raises the steady temperature while slipping and shortens the time to the limit.",
        ],
        change: &[
            Change {
                input: "materials.parts.sleeve_liner",
                step: Step::Choose,
                aim: None,
                text: "titanium, Inconel 625 or PEEK conduct less and heat less; titanium and Inconel cost more, PEEK is weaker, and strength is not checked here.",
            },
            Change {
                input: "metal.sleeve_mm",
                step: Step::Lower,
                aim: Some(("temperature.slip_loss.sleeve_W", Sense::Down)),
                text: "a thinner sleeve heats less in proportion and opens the running clearance; its strength as a retainer is not checked here.",
            },
            Change {
                input: "metal.liner_mm",
                step: Step::Lower,
                aim: Some(("temperature.slip_loss.liner_W", Sense::Down)),
                text: "a thinner liner heats less in proportion and opens the running clearance; its strength as a keeper is not checked here.",
            },
        ],
        watch_out: &[
            "The field at each shell is an entered value from the example design's 3D model; it does not follow your gap, magnets or pole count.",
        ],
        diagram: None,
        sources: &[
            "Material warning rules",
            "Math audit of 2026-09-29: the shell loss and its end factor",
            "Design report: the conductivities",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.high_conductivity_sleeve_or_liner",
        },
    },
    Note {
        id: "a5.low_saturation",
        title: "Back-iron wall and saturation",
        equations: &[
            "model.bsat_T",
            "model.gap_flux_density_T",
            "model.backiron_needed_mm",
            "materials.cup_wall_suggested_mm",
            "materials.cup_wall_check",
        ],
        sentences: &[
            "The steel cup behind the outer magnets carries each pole's flux round to its neighbours, so its wall must be thick enough not to saturate.",
            "The calculator's rule sizes the wall for half a pole's flux at the design flux density, a working level set below the steel's saturation.",
            "So the rule asks for more wall with stronger magnets or wider poles, from fewer magnets or a larger ring; here it gives {model.backiron_needed_mm}.",
            "The low-saturation caution fires for back irons with no design flux density of their own, or a known saturation below 1.7 T: 12L14, 416, 17-4PH.",
            "Here the rule uses {model.bsat_T}; for those steels that is the entered design flux density, which their own data may not support.",
        ],
        change: &[
            Change {
                input: "metal.cup_wall_corner_mm",
                step: Step::Raise,
                aim: Some(("model.cup_od_mm", Sense::Up)),
                text: "until the cup-wall check passes; the cup's outside diameter grows by twice the change, and the coupling gets heavier.",
            },
            Change {
                input: "materials.steel.bsat_T",
                step: Step::Lower,
                aim: Some(("model.backiron_needed_mm", Sense::Up)),
                text: "to the level your steel's data support; the wall needed rises as it falls, so the cup-wall check may then fail.",
            },
            Change {
                input: "materials.parts.back_iron",
                step: Step::Choose,
                aim: None,
                text: "4140 or 1018 clear the low-saturation caution; both rust, so keep a plating thickness above zero.",
            },
        ],
        watch_out: &[
            "With steel, the torque shown assumes it never saturates: a thin or saturating wall shows only in the cup-wall check, while the real torque would fall below the prediction.",
            "The rule is approximate: against exact 2D sections it asks for about 12 to 16 % too little wall at the example design, more with more poles.",
        ],
        diagram: None,
        sources: &[
            "Material warning rules",
            "Design ruling: the design flux density feeds the wall check",
            "Design ruling: the suggested wall is the rule's wall rounded up to 0.1 mm",
            "Math audit of 2026-09-29: the back-iron requirement, 12 to 16 % thin against exact 2D sections",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.low_saturation",
        },
    },
    Note {
        id: "a5.uncoated_low_alloy_steel",
        title: "Unplated carbon or low-alloy steel",
        equations: &[],
        sentences: &[
            "4140, 1018 and 12L14 rust in humid air, so a back iron made of them needs a coating.",
            "The calculator assumes a high-phosphorus electroless nickel plating, and this caution flags a plating thickness of zero.",
            "That thickness also sets the machining allowances: surfaces are cut under or over by it, twice on diameters, so plated parts fit.",
        ],
        change: &[
            Change {
                input: "materials.nickel.thickness_mm",
                step: Step::Raise,
                aim: Some(("materials.hub_flats_under_mm", Sense::Up)),
                text: "to the coating you will specify, typically 0.013 to 0.025 mm; the machining allowances grow with it.",
            },
            Change {
                input: "materials.parts.back_iron",
                step: Step::Choose,
                aim: None,
                text: "416 or 17-4PH need no plating here but trip the low-saturation caution; 304 needs none but loses steel torque and fires the expansion caution.",
            },
        ],
        watch_out: &[
            "This caution checks only that some thickness is entered; it does not judge whether the coating suits your environment.",
        ],
        diagram: None,
        sources: &[
            "Material warning rules",
            "The original spreadsheet's plating note (electroless nickel, 0.013–0.025 mm)",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.uncoated_low_alloy_steel",
        },
    },
    Note {
        id: "a5.cte_mismatch_with_magnets",
        title: "Hub expands much more than the magnets",
        equations: &[],
        sentences: &[
            "In the plane of the glue joint NdFeB barely expands with heat, while steel grows about 12 and aluminium about 24 µm/m per °C.",
            "The glue line between inner magnets and hub takes that difference as shear, highest at the block ends; the outer magnets are not screened.",
            "The caution fires above a 15 µm/m per °C difference: 304 stainless or 6061 aluminium hubs, or no back iron; no listed magnetic steel does.",
            "The caution ignores the temperature swing and the glue; the bond check counts both, here {temperature.mismatch.peak_shear_current_MPa} of peak end shear against {temperature.adhesive.lap_shear_MPa} lap-shear strength.",
        ],
        change: &[
            Change {
                input: "materials.parts.back_iron",
                step: Step::Choose,
                aim: None,
                text: "4140 or 1018 hubs cut the mismatch but bring back the cup-wall check, plating, steel's mass and slip heat, and higher shaft-joint and gearbox loads.",
            },
            Change {
                input: "coupling.backiron",
                step: Step::Choose,
                aim: None,
                text: "the steel circuit, with magnetic steel picked, bonds magnets to steel but adds mass, brings back the cup-wall check and raises shaft-joint and gearbox loads.",
            },
            Change {
                input: "metal.bond_inner_mm",
                step: Step::Raise,
                aim: Some(("temperature.mismatch.peak_shear_current_MPa", Sense::Down)),
                text: "a thicker glue line spreads the shear and lowers its peak; the hub is cut smaller to make room.",
            },
            Change {
                input: "temperature.mismatch.adhesive_shear_modulus_GPa",
                step: Step::Lower,
                aim: Some(("temperature.mismatch.peak_shear_current_MPa", Sense::Down)),
                text: "for a softer adhesive the peak shear falls; softer adhesives are often weaker or rated cooler.",
            },
        ],
        watch_out: &[
            "The screen uses one NdFeB expansion for every grade, so ferrite and SmCo rings, which expand more in that plane, get overstated shear and false cautions.",
            "The adhesive's shear modulus does not follow the adhesive picked: enter the new one's value when you change adhesive.",
        ],
        diagram: None,
        sources: &[
            "Material warning rules",
            "Math audit of 2026-09-29: the Volkersen shear-lag screen",
            "Design ruling: the aluminium hub's expansion",
        ],
        review: Review::Reviewed {
            reviewer: "physics reviewer (session model)",
            date: "2026-10-14",
            record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#a5.cte_mismatch_with_magnets",
        },
    },
];

/// The full suggested reading order (slice-2 spec 3.3, 14 steps: how torque is made, the
/// torque chain, temperature and the torque band, slipping and its heat, demagnetization,
/// the clamp): (note id, the equation it opens). The clamp step stays last: while the
/// shaft-clamp section is paused, [`START_HERE`] is the first 13 ([`start_here_for`]).
pub const START_HERE_ALL: &[(&str, &str)] = &[
    ("torque_from_shear", "model.torque_2d_Nm"),
    ("harmonics", "model.tau_Pa"),
    ("back_iron_factor", "model.s1_iron"),
    ("pullout_angle", "model.pullout_Nm"),
    ("end_effect", "model.f_end"),
    ("calibration", "model.f_cal"),
    ("br_temperature", "model.pullout_20C_Nm"),
    ("hot_low_cold_high", "metal.torque_hot_low_Nm"),
    ("when_slipping", "model.gearbox_input_ripple_Nm"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    (
        "thermal_time_constant",
        "temperature.thermal.time_constant_s",
    ),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("ferrite_cold_demag", "temperature.demag.cold_check"),
    ("clamp_preload", "clamps.capacity_Nm"),
];

/// The note the paused shaft-clamp section takes with it ([`shown_when`]).
pub const PAUSED_NOTE: &str = "clamp_preload";

/// The Start here order shown: [`START_HERE_ALL`], less its last step, the clamp note's, while
/// `clamps_paused`.
pub const fn start_here_for(clamps_paused: bool) -> &'static [(&'static str, &'static str)] {
    if clamps_paused {
        START_HERE_ALL.split_at(START_HERE_ALL.len() - 1).0
    } else {
        START_HERE_ALL
    }
}

/// The Start here order the panel walks: [`start_here_for`] the shaft-clamp section's pause
/// (`clamps::PAUSED`).
pub const START_HERE: &[(&str, &str)] = start_here_for(crate::engine::clamps::PAUSED);

/// Whether the note `id` is shown while the shaft-clamp section is paused (`clamps_paused`) or
/// not: every note but [`PAUSED_NOTE`] while paused, every note otherwise. Its review status is
/// the accuracy gate's ([`note_for`]), not this.
pub fn shown_when(id: &str, clamps_paused: bool) -> bool {
    !(clamps_paused && id == PAUSED_NOTE)
}

/// [`shown_when`] the shaft-clamp section's pause (`clamps::PAUSED`): what the review sheets and
/// the release gates ask. The panel reaches no note it hides: Start here walks [`START_HERE`],
/// and the clamp's dashboard row, the clamp note's only Why link, is hidden with the section.
pub fn shown(id: &str) -> bool {
    shown_when(id, crate::engine::clamps::PAUSED)
}

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

/// What reads as jargon in `note` (slice-2 spec 3.4, A-13), one line per hit: a word the
/// plain-words guard bans, or an audit id (`plain::audit_ids`), which the guard does not know.
/// It reads the title, the sentences, change texts and watch-outs drawn with `value` (as
/// `note_text::render` draws them) and the sources. The one plain-words rule for notes:
/// `tests/explain.rs` holds every note shown to it at the five probe designs.
pub fn plain_words_problems(note: &Note, value: impl Fn(&str) -> Option<String>) -> Vec<String> {
    use super::note_text::{render, rendered_texts};
    use super::plain::{audit_ids, plain_text_violations};
    let texts = std::iter::once(note.title.to_owned())
        .chain(
            rendered_texts(note)
                .into_iter()
                .map(|text| render(text, &value)),
        )
        .chain(note.sources.iter().map(|&source| source.to_owned()));
    let mut problems = Vec::new();
    for text in texts {
        let banned = plain_text_violations(&text);
        if !banned.is_empty() {
            problems.push(format!("{}: {banned:?} in {text:?}", note.id));
        }
        for id in audit_ids(&text) {
            problems.push(format!("{}: an audit id {id} in {text:?}", note.id));
        }
    }
    problems
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
        for (id, path) in START_HERE_ALL {
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

    #[test]
    fn the_torque_band_note_follows_the_temperature_note_in_start_here() {
        // Spec 3.3: step 8, after the temperature note (step 7), opening hot-low, in the full
        // order and in the order shown while the clamp section is paused (ruling 11).
        for paused in [false, true] {
            let steps = start_here_for(paused);
            let ids: Vec<&str> = steps.iter().map(|&(id, _)| id).collect();
            let at = ids
                .iter()
                .position(|&id| id == "hot_low_cold_high")
                .expect("the torque band is a Start here step");
            assert_eq!(at, 7, "step 8 (paused: {paused})");
            assert_eq!(ids[at - 1], "br_temperature", "paused: {paused}");
            assert_eq!(steps[at].1, "metal.torque_hot_low_Nm", "paused: {paused}");
        }
    }

    #[test]
    fn when_slipping_is_start_here_step_9_after_the_torque_band() {
        // Spec 3.3 (R4): 8 hot_low_cold_high, 9 when_slipping, 10 slip_heating, in the full
        // order and in the order shown while the clamp section is paused (ruling 11).
        for paused in [false, true] {
            let steps = start_here_for(paused);
            let ids: Vec<&str> = steps.iter().map(|&(id, _)| id).collect();
            let at = ids
                .iter()
                .position(|&id| id == "when_slipping")
                .unwrap_or_else(|| panic!("no start-here step: {ids:?}"));
            assert_eq!(at, 8, "step 9: {ids:?}");
            assert_eq!(ids[at - 1], "hot_low_cold_high", "{ids:?}");
            assert_eq!(ids[at + 1], "slip_heating", "{ids:?}");
            assert_eq!(steps[at].1, "model.gearbox_input_ripple_Nm");
        }
    }

    #[test]
    fn the_slip_and_thermal_notes_own_the_rows_spec_3_3_gives_them() {
        let owner = |path: &str| note_for_any_status(path).map(|n| n.id);
        for path in ["model.gearbox_input_ripple_Nm", "metal.slip_freq_Hz"] {
            assert_eq!(owner(path), Some("when_slipping"), "{path}");
        }
        for path in [
            "temperature.duty.field_freq_Hz",
            "temperature.duty.pole_pairs",
            "temperature.duty.slip_rad_s",
        ] {
            assert_eq!(owner(path), Some("slip_heating"), "{path}");
        }
        for path in [
            "temperature.magnet_life.peak_C",
            "temperature.thermal.steady_est_C",
            "temperature.thermal.steady_high_C",
            "temperature.slip_life.rise_per_event_high_C",
        ] {
            assert_eq!(owner(path), Some("thermal_time_constant"), "{path}");
        }
    }

    /// A-15's per-id form for T15's notes (A-28c item 9); the all-notes tests run the same two
    /// checks over every note shown.
    #[test]
    fn the_slip_notes_keep_the_caps() {
        for id in ["when_slipping", "slip_heating", "thermal_time_constant"] {
            assert_eq!(
                cap_problems(note(id).unwrap()),
                Vec::<String>::new(),
                "{id}"
            );
            assert_eq!(
                sentence_zero_problems(note(id).unwrap().sentences[0]),
                Vec::<&str>::new(),
                "{id}"
            );
        }
    }

    /// The text a note draws as prose: its sentences, its change entries' texts and its
    /// watch-outs (titles and sources aside).
    fn note_texts(n: &Note) -> Vec<&'static str> {
        n.sentences
            .iter()
            .copied()
            .chain(n.change.iter().map(|c| c.text))
            .chain(n.watch_out.iter().copied())
            .collect()
    }

    #[test]
    fn the_demagnetization_notes_keep_the_caps() {
        // The clamp note is not rewritten while its section is paused (ruling 11, A-28d).
        for id in ["demagnetization", "ferrite_cold_demag"] {
            assert_eq!(
                cap_problems(note(id).unwrap()),
                Vec::<String>::new(),
                "{id}"
            );
            assert_eq!(
                sentence_zero_problems(note(id).unwrap().sentences[0]),
                Vec::<&str>::new(),
                "{id}"
            );
        }
    }

    #[test]
    fn the_demagnetization_notes_own_the_rows_spec_3_3_gives_them() {
        let owner = |path: &str| note_for_any_status(path).map(|n| n.id);
        for path in [
            "temperature.magnet_life.margin_onset_C",
            "temperature.magnet_life.margin_limit_C",
        ] {
            assert_eq!(owner(path), Some("demagnetization"), "{path}");
        }
        // R7: the ferrite note drops the NdFeB knee picture, which shows the knee moving the
        // wrong way for ferrite; the demagnetization note keeps it.
        assert_eq!(note("ferrite_cold_demag").map(|n| n.diagram), Some(None));
        assert_eq!(
            note("demagnetization").map(|n| n.diagram),
            Some(Some(Diagram::DemagKnee))
        );
    }

    #[test]
    fn the_rewritten_notes_type_no_editable_default() {
        // Spec section 8 (slice 1, trap 6) and audit theme T6: a note never presents an
        // editable default (a stored reverse field, the knee fraction, the design margin) as a
        // fact about the design.
        let d = crate::DesignInputs::default();
        let demag = &d.temperature.demag;
        let mut defaults: Vec<String> = [
            demag.h_rev_aligned_kA_m,
            demag.h_rev_pullout_kA_m,
            demag.h_rev_likepole_kA_m,
            demag.h_rev_single_ring_kA_m,
        ]
        .iter()
        .map(|h| format!("{h}"))
        .collect();
        defaults.push(format!("{} °C", demag.design_margin_C));
        defaults.push(format!("{}", demag.knee_fraction));
        for id in ["demagnetization", "ferrite_cold_demag"] {
            for text in note_texts(note(id).unwrap()) {
                for value in &defaults {
                    assert!(
                        !text.contains(value.as_str()),
                        "{id}: {value:?} in {text:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_note_says_the_coupling_protects_the_gearbox() {
        // BL-059 and spec section 8: nothing checks the cold-high torque at the gearbox input
        // against a rating, so no note may claim the coupling protects the gearbox.
        for n in NOTES {
            for text in note_texts(n) {
                assert!(
                    !crate::engine::explain::plain::mentions_gearbox_protection(text),
                    "{}: {text}",
                    n.id
                );
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
        let ids: Vec<&str> = START_HERE_ALL.iter().map(|&(id, _)| id).collect();
        let want: Vec<&str> = SLICE_2_ORDER
            .iter()
            .copied()
            .filter(|id| ids.contains(id))
            .collect();
        assert_eq!(
            ids, want,
            "spec 3.3 order (a step outside the list fails too)"
        );
        assert_eq!(
            START_HERE_ALL[0],
            ("torque_from_shear", "model.torque_2d_Nm")
        );
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

    #[test]
    fn start_here_leaves_out_the_clamp_step_only_while_the_clamp_section_is_paused() {
        // Ruling 11: the clamp note's step is the whole order's last, and the only step the
        // pause takes out; the panel walks the order for `clamps::PAUSED`.
        assert_eq!(START_HERE_ALL.last().map(|&(id, _)| id), Some(PAUSED_NOTE));
        for paused in [false, true] {
            let want: Vec<(&str, &str)> = START_HERE_ALL
                .iter()
                .copied()
                .filter(|&(id, _)| shown_when(id, paused))
                .collect();
            assert_eq!(start_here_for(paused), want.as_slice(), "paused {paused}");
        }
        assert_eq!(start_here_for(false), START_HERE_ALL);
        assert_eq!(start_here_for(true).len(), START_HERE_ALL.len() - 1);
        assert_eq!(START_HERE, start_here_for(crate::engine::clamps::PAUSED));
    }

    #[test]
    fn only_the_clamp_note_is_hidden_and_only_while_the_clamp_section_is_paused() {
        for n in NOTES {
            assert!(shown_when(n.id, false), "{}", n.id);
            assert_eq!(shown_when(n.id, true), n.id != PAUSED_NOTE, "{}", n.id);
            assert_eq!(
                shown(n.id),
                shown_when(n.id, crate::engine::clamps::PAUSED),
                "{}",
                n.id
            );
        }
        // The note stays, unchanged, for the section's return.
        assert!(note(PAUSED_NOTE).is_some());
        assert_eq!(NOTES.iter().filter(|n| !shown_when(n.id, true)).count(), 1);
        // The hidden note is the one whose every equation the section hides; no other note's
        // are (a warning's note, with no equation, is not one).
        for n in NOTES {
            let all_hidden = !n.equations.is_empty()
                && n.equations
                    .iter()
                    .all(|e| crate::engine::clamps::hides(e, true));
            assert_eq!(all_hidden, n.id == PAUSED_NOTE, "{}", n.id);
        }
    }

    #[test]
    fn the_torque_note_glosses_each_pole_where_it_is_first_used() {
        // R8 (ruling 10 a): Start here step 1 says what a pole spans in the sentence that first
        // speaks of "each pole", not one sentence after it.
        let s = note("torque_from_shear").unwrap().sentences;
        let defines = s.iter().position(|t| t.contains("τ_p = 2π·R_g/N"));
        let first = s.iter().position(|t| t.contains("each pole"));
        assert!(defines.is_some(), "{s:#?}");
        assert_eq!(
            defines, first,
            "each pole is glossed where it is first used"
        );
    }

    #[test]
    fn the_torque_chain_notes_say_what_to_change_and_harmonics_owns_the_fill() {
        // Slice 2: Start here steps 2 to 4 say what to change, and the block fill they teach
        // belongs to the harmonics note (spec 3.3 R3).
        for id in ["harmonics", "back_iron_factor", "pullout_angle"] {
            let n = note(id).unwrap_or_else(|| panic!("{id}"));
            assert!(!n.change.is_empty(), "{id}: what to change");
        }
        for path in ["model.fill_inner", "model.fill_outer"] {
            assert_eq!(
                note_for_any_status(path).map(|n| n.id),
                Some("harmonics"),
                "{path}"
            );
        }
    }

    #[test]
    fn the_torque_note_glosses_pull_out_where_it_is_first_used() {
        // R8 (ruling 10 b): the note draws its sentences, then What to change, then Watch out.
        // "Pull-out" first appears in the watch-outs, so the first watch-out glosses it.
        let n = note("torque_from_shear").unwrap();
        let texts: Vec<&str> = n
            .sentences
            .iter()
            .copied()
            .chain(n.change.iter().map(|c| c.text))
            .collect();
        assert!(texts.iter().all(|t| !t.contains("pull-out")), "{texts:#?}");
        let w = n.watch_out;
        assert!(
            w[0].starts_with("The dashboard's pull-out torque, the most the rings"),
            "{w:?}"
        );
        // Ruling 13: the dashboard's number is a nominal estimate, not a bound.
        assert!(w[0].contains("expected to pass"), "{w:?}");
        assert_eq!(w.iter().position(|t| t.contains("pull-out")), Some(0));
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
    /// cap rule: the all-notes test and the per-note tests of the note tasks call it.
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
    fn every_note_keeps_to_the_slice_2_caps() {
        // Every note shown (ruling 11: while the shaft-clamp section is paused, its note keeps
        // its slice-1 text, hidden, until the section returns).
        let failures: Vec<String> = NOTES
            .iter()
            .filter(|n| shown(n.id))
            .flat_map(cap_problems)
            .collect();
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
    fn every_sentence_zero_is_plain_words() {
        let failures: Vec<String> = NOTES
            .iter()
            .filter(|n| shown(n.id))
            .filter_map(|n| {
                let problems = sentence_zero_problems(n.sentences.first().copied().unwrap_or(""));
                (!problems.is_empty()).then(|| format!("{}: {problems:?}", n.id))
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

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
        let one = |_: &str| Some("1".to_owned());
        assert_eq!(plain_words_problems(&good, one), Vec::<String>::new());
        // The rule reads a live value as drawn: the caller's value stands in the text.
        let leaked = plain_words_problems(&good, |_| Some("Calculator!C5".to_owned()));
        assert!(
            leaked.iter().any(|p| p.contains("Calculator!")),
            "{leaked:?}"
        );
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
            let problems = plain_words_problems(&bad, one);
            assert!(
                problems.iter().any(|p| p.contains(why)),
                "{why}: {problems:?}"
            );
        }
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

    #[test]
    fn the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them() {
        // Slice-2 spec 3.3: the back-iron wall note owns the cup-wall rule's rows beside the design
        // flux density; the sleeve note owns the conductivity its caution compares; the
        // open-circuit note keeps the circuit in effect (the panel's warning test clicks it).
        for (path, id) in [
            ("model.bsat_T", "a5.low_saturation"),
            ("model.gap_flux_density_T", "a5.low_saturation"),
            ("model.backiron_needed_mm", "a5.low_saturation"),
            ("materials.cup_wall_suggested_mm", "a5.low_saturation"),
            ("materials.cup_wall_check", "a5.low_saturation"),
            (
                "temperature.slip_loss.sleeve_W",
                "a5.high_conductivity_sleeve_or_liner",
            ),
            (
                "temperature.slip_loss.liner_W",
                "a5.high_conductivity_sleeve_or_liner",
            ),
            (
                "materials.sleeve_sigma_S_m",
                "a5.high_conductivity_sleeve_or_liner",
            ),
            (
                "materials.circuit_backiron",
                "a5.non_ferromagnetic_back_iron",
            ),
        ] {
            assert_eq!(note_for_any_status(path).map(|n| n.id), Some(id), "{path}");
        }
    }
}
