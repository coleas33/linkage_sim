# Verified: linkage-gui-export
## LGE-01: Per-joint reaction series are not padded, so they shift against angles_deg when early samples fail
- Verdict: CONFIRMED
- Checked: sweep/mod.rs:458 (reaction_data empty HashMap), 788-802 (lazy `entry().or_insert_with` push; Err arm pushes NaN only into existing vectors), 955-957 push_nan_row same, 1334-1353 trajectory same; git grep reaction_data in src finds no padding anywhere (9 hits, all listed). A leading failure creates the vector late, so it is shorter than angles_deg.
- Safe to apply without judgment: no, changes plotted/CSV output and needs a new test.

## LGE-02: A failed statics solve is stored as driver torque 0.0 in the forward sweep but NaN in trajectory mode
- Verdict: CONFIRMED
- Checked: sweep/mod.rs:626 `unwrap_or(0.0)`, 630 `push(0.0)` on Err; 1252 `unwrap_or(0.0)`, 1258 `push(f64::NAN)` on Err; comment at 743-744 says the 0 is fake. Trajectory also uses unwrap_or(0.0) for a missing torque, as the finder said.
- Safe to apply without judgment: no, changes plotted values, envelopes and actuator force.

## LGE-03: apply_motion_profile reads angles_deg as absolute 0-360 degrees and is applied to every sweep
- Verdict: PHYSICS-SUSPECT (described correctly)
- Checked: motion_profile.rs:56 doc "relative to the sweep start", 155 total_angle = 2*PI, 163 `data.angles_deg[i].to_radians()`; blueprint_ops.rs:1405-1433 builds sweep_range (custom range, mm->m for Linear) then calls apply_motion_profile unconditionally; sweep/mod.rs:49-57 confirms angles_deg holds metres in Stroke mode. theta_0 not read in apply_motion_profile.
- Safe to apply without judgment: no, physics-suspect.

## LGE-04: GIF export silently drops frames that fail and still returns Ok, possibly an empty GIF
- Verdict: CONFIRMED
- Checked: raster.rs:125-150 (nested `if let Ok` and `_ => {}` skip frames, then `Ok(buf)` at 152 with no frame count), 127 `/ omega` with no guard, 108/135 `width as u16`. Only the n_steps == 0 case returns Err (102-104).
- Safe to apply without judgment: no, adds error behaviour.

## LGE-05: Path-writing export wrappers are used only by their own tests (baseline "never used" settled)
- Verdict: CONFIRMED
- Checked: git grep over the whole worktree for each of the six names. Hits are only the definition, the `pub use` in export/mod.rs (22, 25, 29, 35), doc comments, and tests in csv.rs, svg.rs and raster.rs. export_mechanism_dxf has no test at all (definition and re-export only). No script, doc, web file or bin uses any of them.
- Safe to apply without judgment: no, removes tests and public API; owner decision.

## LGE-06: add_plotly_multi_chart is dead and its job is written three times inline in generate_html_report
- Verdict: CONFIRMED
- Checked: git grep add_plotly_multi_chart gives 1 hit (report.rs:567, no #[allow]). Inline Plotly.newPlot copies at report.rs:394-405 (reactions), 464-478 (coupler); energy also inline. Colour arrays at 391 (8) and 458 (6) differ in length. The helper's `null /* per-series x provided */` fallback is invalid for per-series x, as stated.
- Safe to apply without judgment: yes for deleting the helper only (never referenced on any target or feature; no test, script or doc names it). No for the refactor.
- Exact change: linkage-sim-rs/src/gui/export/report.rs lines 558-602: delete the doc comment starting `/// Emit a multi-series Plotly line chart into `html`.`, the whole `fn add_plotly_multi_chart(...) { ... }` (ends at the `}` on line 601) and the one blank line after it (602). New text: nothing (line 557 blank stays, then `/// Get current timestamp ...`).

## LGE-07: Body, joint, trace and sensor names go into HTML, SVG and JS unescaped in most exporters; escapers disagree
- Verdict: CONFIRMED
- Checked: report.rs:150 (body_id), 164, 383 (jid), 401 and 474 (`name:'{}'` raw), 418 (details); svg.rs:147 body_id, 221 joint.id() raw into SVG text; dxf.rs:48 body_id as layer name. html_escape (report.rs:827, 4 chars) vs xml_escape (schematic.rs:789, 5 chars incl. &apos;) are two copies. Not tested with a quote id; ids are user JSON.
- Safe to apply without judgment: no, output changes and a design choice.

## LGE-08: Parameters and variables silenced with `let _ =`, a dead computation, and a doc comment on the wrong function
- Verdict: CONFIRMED
- Checked: report.rs:1525 `let _ = mechanism;` (param at 1491), 1760-1772 `touches_ground_and_crank` over a chain with a "Placeholder" closure then `let _ = touches_ground_and_crank;`, 2136 `let _ = (mu_deg, q, mechanism, GROUND_ID); // silence unused` (mu_deg is in fact used at 2139; the `let _` is only for q, mechanism, GROUND_ID), call sites 1169 and 1175, misplaced doc at 2220-2223 above the `§6 Trajectory mode` doc at 2224, and `write_singular_configs_section` at 3404 with no doc.
- Safe to apply without judgment: no, signature changes at several sites and the unused-variable cleanup may leave other params unused.

## LGE-09: SweepData.active_range is always None; its writers and readers are dead
- Verdict: CONFIRMED
- Checked: git grep active_range: sweep/mod.rs:227 (field), 452, 831, 1356, 1400 assign None; no `Some(` assignment anywhere; readers plot_panel/coupler.rs:40, plot_panel/mod.rs:248, 577, 639; a test at sweep/mod.rs:1487 asserts is_none; docs/ai/05-update-tracker.md:2928 records it as always None.
- Safe to apply without judgment: no, serialised field, tests and docs use it.

## LGE-10: Two four-bar detectors with different rules
- Verdict: CONFIRMED
- Checked: sweep/fourbar.rs:17-109 (3 moving bodies, 4 revolutes, driver_body_pair, 2 attachment points per body, ground length from attachment points) vs report.rs:864-930 (4 bodies, 4 joints, exactly 1 driver and no linear drivers). One detail is off: `driver_body_pair()` (core/mechanism.rs:426) reads `drivers.first()`, the revolute drivers, so a linear-driver-only mechanism gives None in the sweep detector too; only a mechanism with both kinds would differ on that rule.
- Safe to apply without judgment: no, merges two behaviours.

## LGE-11: compute_trajectory takes a full SVD of the same Jacobian twice per sample
- Verdict: CONFIRMED
- Checked: sweep/mod.rs:1177-1193: `phi_q.clone().svd(true,true).solve(&neg_phi_t,1e-14)` then `phi_q.clone().svd(true,true).solve(&gamma,1e-14)`. phi_q is not used after line 1193 in the loop (grep phi_q in 1194-1350: no hits), so it can be moved. Same matrix both times, so the factorisation and the numbers are identical; error path is unchanged (first `?` returns before the second solve either way).
- Safe to apply without judgment: yes (no change of numbers or behaviour on any target or feature; no test, script or doc names this code).
- Exact change: linkage-sim-rs/src/gui/sweep/mod.rs lines 1181-1193.
  Old:
```
        let q_dot_k = phi_q
            .clone()
            .svd(true, true)
            .solve(&neg_phi_t, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;

        let mut gamma = assemble_gamma(mech, &q_k, &q_dot_k, t_mech_k);
        gamma[driver_row] = u_ddot_k;
        let q_ddot_k = phi_q
            .clone()
            .svd(true, true)
            .solve(&gamma, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;
```
  New:
```
        let phi_q_svd = phi_q.svd(true, true);
        let q_dot_k = phi_q_svd
            .solve(&neg_phi_t, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;

        let mut gamma = assemble_gamma(mech, &q_k, &q_dot_k, t_mech_k);
        gamma[driver_row] = u_ddot_k;
        let q_ddot_k = phi_q_svd
            .solve(&gamma, 1e-14)
            .map_err(|_| LinkageError::SvdSolveFailed)?;
```
  (Not compiled here; the first-pass reviewer should run the gate.)

## LGE-12: compute_sweep_data_with_weights and compute_trajectory repeat the same setup; SweepData is built long-hand three times
- Verdict: CONFIRMED
- Checked: `ma_bodies` blocks at sweep/mod.rs:505-511 and 1101-1110 are the same code; six `if actuator_info.is_some()` arms at 403-433 (actuator_lengths at 413); two full SweepData literals (389-454 and 1364-1403) plus test literals 1693, 1740. On drift: actuator_lengths is None in the trajectory literal (1381) and filled in the forward one (590, 971), as stated.
- Safe to apply without judgment: no, a refactor.

## LGE-13: write_sweep_csv codes each of eleven optional columns twice
- Verdict: CONFIRMED
- Checked: csv.rs:51-126 header flags and 130-219 row pushes: each optional series is tested once for the header and again for the row; the `{:.6}` / `.get(i).copied().unwrap_or(f64::NAN)` idiom is repeated per column (e.g. 133, 140, 143-152, 155-160). The count "eleven" was not recounted.
- Safe to apply without judgment: no, a refactor.

## LGE-14: Two samples ignore the requested driver joint, and the Custom 6-bar can return an unsolved start
- Verdict: CONFIRMED
- Checked: samples/fourbar.rs:578-579 and samples/special.rs:1033-1034 take `_driver_joint_id` and never use it (git grep `_driver_joint_id` in samples: exactly those 2 hits); special.rs:1083-1084 `log::warn!(... using zero q0); Ok((mech, q0))`; state/driver_ops.rs:26-51 takes the Ok arm, sets driver_joint_id and load cases to the requested joint and never falls through. The "28 of 30 validate" count was not recounted.
- Safe to apply without judgment: no, behaviour change.

## LGE-15: The four-bar skeleton is hand-copied into eleven sample builders although build_standard_fourbar exists
- Verdict: CONFIRMED
- Checked: samples/fourbar.rs:167-200 `build_standard_fourbar` (coupler point (x, 0.0), called at 243, 260, 276, 291, 525) vs the Hoeken builder 715-756, which is the same sequence with a coupler point at (0.0625, 0.0). The other builders' line numbers were not each read; "eleven" is not recounted.
- Safe to apply without judgment: no, a refactor.

## LGE-16: DXF overlay drops lines that cross the view, and import silently skips unsupported entities
- Verdict: CONFIRMED
- Checked: dxf_import.rs:287 `if canvas_rect.contains(p1) || canvas_rect.contains(p2)` (lines only; Circle and Arc at 291-310 have no such test), 206 `_ => {} // skip unsupported entities`, 245 `let _ = canvas_rect;` (canvas_rect is used at 287, so the comment is wrong), 306 `let _center`. LwPolyline (167-195) uses vertices only, so bulges are ignored. git grep for skipped/not imported/unsupported in this file: only line 206, so no message here; the panel code (360-1391) was not read, other files not searched for a message.
- Safe to apply without judgment: no, changes drawing behaviour; deleting 245 and 306 alone is a no-op but 306 binds a value, so left to the reviewer.

## LGE-17: svg.rs mechanism SVG and schematic.rs repeat the bounding-box and scaling block with different rules
- Verdict: CONFIRMED
- Checked: svg.rs:19-60 (attachment points only, 1e-10, `max = min + 1.0`, 15% margin) vs schematic.rs:70-141 (adds linear-driver points at 85-102, 1e-12, re-centres at 110-118, 20% pad); report.rs:28-30 `generate_schematic_svg(..).or_else(|_| generate_svg_string(..)).unwrap_or_default()`. Whether PNG/GIF clip a linear-driver mechanism was not run; the code shows svg.rs leaves the driver points out of the box.
- Safe to apply without judgment: no.

## LGE-18: index_at_driver doc says trajectory sweeps leave angles_deg empty, but compute_trajectory fills it with sample times
- Verdict: CONFIRMED
- Checked: sweep/mod.rs:243-245 doc vs 1211 `data.angles_deg.push(t_k)` and 993 doc "angles_deg holds sample times"; `is_stroke()` is false for Trajectory so `index_at_driver` compares seconds with `driver_value.to_degrees()`. Callers: state/mod.rs:878-882 (`current_sweep_index`, used at state/mod.rs:864, 898 and canvas/rendering/weights.rs:90); none of those lines guards trajectory mode. Whether a trajectory sweep reaches those readouts was not run.
- Safe to apply without judgment: no, behaviour change (a doc-only fix would be safe but is not the finder's action).

## LGE-19: FirmwareAdapter trait has one implementor and two methods nothing calls
- Verdict: CONFIRMED
- Checked: git grep over the worktree for file_extension(), display_name(), FirmwareAdapter, JsonAdapter: the two methods appear only at definitions and json.rs:279-280 (test); sole production caller menu_bar.rs:1007-1025 names JsonAdapter. docs/ai/04-memory.yaml:36 states the trait is a deliberate extension point ("implement on demand").
- Safe to apply without judgment: no, owner decision, and a doc says it is intentional.

## LGE-20: Two identical sample-count tests, and a sample whose enum name does not match what it builds
- Verdict: CONFIRMED
- Checked: samples/mod.rs:690-693 and 935-942 both assert `all().len() == 30`; variant PeaucellierLipkin (44) with label "Coupler Curve (figure-8)" (79) and builder `special::build_peaucellier_lipkin` (236). git grep PeaucellierLipkin finds only samples/mod.rs lines (44, 79, 115, 155, 188, 236, 849); no serialised-name check was done.
- Safe to apply without judgment: no, deleting even a duplicate test removes a test; the rename is an owner decision.

## Seen while checking
- sweep/mod.rs:805, report.rs and other files show mojibake in comments (e.g. "â€”") when read through PowerShell; this is the console decoding, not necessarily the file, so not reported as an error.
