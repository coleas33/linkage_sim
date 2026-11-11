### docs-living

Detail: `docs/analyses/2026-11-10-whole-project-audit/docs-living-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/docs-living.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| DOC-01 | README.md "File Structure" tree describes a Python layout that does not exist | CONFIRMED | For you: no, the replacement is either a new listing or a pointer to 03-structure.yaml, an owner choice. |
| DOC-02 | README.md and WASM_DEPLOYMENT.md say PNG/SVG/GIF/DXF export and HTML reports do not work in the browser; the code ships them | PARTLY. The documents contradict the code and each other; whether the browser export wo... | For you: no, the owner should confirm the web exports work before the text changes. |
| DOC-03 | docs/guides/SAMPLES.md lists 17 of the 30 samples, with names and groups the GUI does not use | CONFIRMED | For you: no, regenerating the guide writes new descriptive text and drops the old text. |
| DOC-04 | docs/FEATURES.md "Planned / Future" lists shipped features and lists one item twice | CONFIRMED | **Fixed** in `cc3d498`: `Required power curve` and `Actuator speed plot` are out of `Planned / Future` in docs/FEATURES.md (both shipped). Left for you: the force-margin and mobile-support lines. |
| DOC-05 | linkage-sim-rs/README.md is in the scope list but does not exist; the README points to a SYSTEM.md whose count is stale | CONFIRMED | **Fixed** in `cc3d498`: README.md no longer gives a count of Rust source files (it said 107; the tree has 137). |
| DOC-06 | SOLIDWORKS_IMPORT.md "Manual JSON" example cannot be loaded by the app | CONFIRMED | For you: no, the example should be rewritten from a real exported file. |
| DOC-07 | ARCHITECTURE.md and README describe a Python design (plugin registry, `asteval`, friction tiers, driver types) that was never built or differs from the code | CONFIRMED | For you: no, whether the plan text stays as design intent is an owner decision. |
| DOC-08 | VALIDATION.md "Versioned Benchmark Suite" and its run command do not match the repo | CONFIRMED | **Fixed** in `cc3d498`: the benchmark command in VALIDATION.md no longer passes `--benchmark-report`. Left for you: the format and planned sections. |
| DOC-09 | docs/ai/02-system.yaml has at least seven syntax errors, not one | CONFIRMED | **Fixed** in `d6f5880`: seven unquoted scalars in docs/ai/02-system.yaml quoted in place; the file loads and its words are unchanged. BL-015 set to fixed. |
| DOC-10 | Hard-coded counts in the living documents are stale | PARTLY. The stale counts hold; one sub-claim is wrong. | For you: no for the set (what 644 counts is unclear; SYSTEM.md is a dated generated map); the README.md:82 line alone is covered by DOC-05. |
| DOC-11 | 03-structure.yaml omits modules the update rule says it must list | CONFIRMED | **Fixed** in `cc3d498`: the sample checklist in docs/ai/03-structure.yaml names `label()`, the method that exists. Left for you: the modules the file does not list. |
| DOC-12 | EXTENSIBILITY.md promises a registry, schema-generated panels and generic serialization; the Rust code has none of them | CONFIRMED | For you: no, whether Rules 3-4 remain goals is an owner decision. |
| DOC-13 | Rules in CLAUDE.md section 5 and AGENTS.md point at agent files and skills that are not in the repository, and the one agent file that exists breaks the rule | CONFIRMED | For you: no, owner decision (track the agents or declare them local; the file is outside git). |
| DOC-14 | 04-memory.yaml `open_questions` is 60 percent closed items, and several open ones are stale | PARTLY. The size and counts hold; the two stale items are partly shown, the third is not. | For you: no, the resolved entries may hold detail the tracker lacks. |
| DOC-15 | AGENTS.md says CLAUDE.md "holds the project's working rules"; CLAUDE.md holds none of the repository's rules, and has stray characters | CONFIRMED | For you: no, CLAUDE.md is the owner's own prompt text. |
| DOC-16 | FEATURES.md claims "zero-warning compilation" for the WASM build; the baseline shows hundreds of warnings | NOT CONFIRMED | Dropped: the check could not confirm it. |
| DOC-17 | README.md Rust tree lists 8 of 15 integration test files and omits modules | CONFIRMED | For you: no, deleting versus regenerating the tree is a choice. |

### docs-magcoupling-history

Detail: `docs/analyses/2026-11-10-whole-project-audit/docs-magcoupling-history-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/docs-magcoupling-history.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| DMH-01 | README (and Cargo.toml comments) still say linkage-sim-rs depends on magcoupling-rs and that a gate checks lock agreement | CONFIRMED | For you: no. The Cargo.toml comments and README text are only prose, but README is read by include_str! tests (tests/explain.rs:1779-1860, gui/guidance.rs:6718), so the README edit needs a check that those pinned phrases are untouched (they are not on lines... |
| DMH-02 | README status line lists BL-072 as open although it is fixed and shipped | CONFIRMED | **Fixed** in `cc3d498`: the status line of magcoupling-rs/README.md no longer lists BL-072 as open. |
| DMH-03 | 17 plan and spec files are named by no other tracked file (1.76 MB), and the whole history set is 15.4 MB | PARTLY. The "named by no other tracked file" fact holds for all 17 (git grep -l -F of e... | For you: no. Owner decision on removing records; backlog and memory name the m5 plan. |
| DMH-04 | Update tracker is 313 kB in 110 entries; six entries hold 38 percent, and entries are split per task | CONFIRMED | For you: no. Trimming or moving history is an owner call. |
| DMH-05 | Update tracker, backlog and memory cite records under the gitignored .superpowers/ that no clone has | CONFIRMED (count slightly off: the tracker has 10 mentions, not 9; the tenth is at line... | For you: no. Needs a choice between a header note and committing the review files. |
| DMH-06 | set_keyboard_shortcuts exists for a host that no longer exists | CONFIRMED | For you: no. Removing a pub API and a test. |

### linkage-core-solver

Detail: `docs/analyses/2026-11-10-whole-project-audit/linkage-core-solver-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/linkage-core-solver.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| LCS-01 | Cargo.toml lists dependencies the code never uses | PARTLY. Holds for `open`, `wasm-bindgen-futures` and the dev-dependency `meval`. Not co... | For you: no. Removing dependencies rewrites the tracked linkage-sim-rs/Cargo.lock, a gate checks the lock files against each other, and the wasm32 build cannot be checked here. |
| LCS-02 | Constraint projection in forward dynamics only edits the recorded samples and understates drift | PHYSICS-SUSPECT (described correctly) | For you: no. Physics-suspect, and the intended behaviour is an owner decision. |
| LCS-03 | simulate_with_events repeats simulate and rk4_integrate, and nothing outside tests calls it | CONFIRMED | For you: no. Removing events is an owner decision and merging the loops is a refactor. |
| LCS-04 | Stale doc comments in reactions.rs describe code that no longer exists | CONFIRMED | **Fixed** in `12b744e`: two doc comments in solver/reactions.rs corrected (the tolerance check `is_valid` does not make; force zones are modelled). |
| LCS-05 | pass-1-only result is built twice, and the transposed Jacobian SVD solve is written three times | CONFIRMED | For you: no. Refactor. |
| LCS-06 | Reaction extraction reads a prismatic joint's two multipliers as an [Fx, Fy] force | PHYSICS-SUSPECT (described correctly) | For you: no. Physics-suspect. |
| LCS-07 | Cam profile "Spline" is documented as cubic but is piecewise linear, and its comments and guards disagree with the code | CONFIRMED (one count off: the redundant ground checks are four, at cam.rs:140-141, 162-... | **Fixed** in `12b744e`: the cam.rs comments say piecewise linear and linear scan, which is what the code does. Left for you: the guards, the `if let Ok`, and whether `Spline` should be cubic. |
| LCS-08 | solve_for_target's doc tells callers to cache the workspace probe, but there is no way to, and the app runs 64 forward solves per trajectory sample | CONFIRMED | For you: no. |
| LCS-09 | Inverse-kinematics helpers repeat the singularity guard four times and the finite-difference path repeats its own solves; the analytic path the notes call faster is unused | CONFIRMED | For you: no. |
| LCS-10 | Expression drivers re-parse their formula on every evaluation, and the DriverFn doc is stale | CONFIRMED | **Fixed** in `12b744e`: `DriverFn`'s doc no longer calls expression drivers a future step. Left for you: the formula is parsed again on every evaluation. |
| LCS-11 | Every solver call goes through String-keyed HashMap lookups and builds full-width matrices per joint | PARTLY. The structure holds; the statement that hashing and allocation dominate is unme... | For you: no. |
| LCS-12 | Dead or test-only public items, a stale allow, and config fields that nothing reads | CONFIRMED | **Fixed** in `6cbfac3`: the stale `#[allow(dead_code)]` in prismatic.rs is gone. Left for you: `State::set_angle` and `JointConstraint::is_cam_follower` (no caller, each one member of a public family), and `rtol`/`atol` and the local-frame reactions (tests... |
| LCS-13 | ControlTarget panics on bad data it can load from a file, repeats its normalisation three times, and hard-codes "ground" | CONFIRMED (the file-load path is plausible but not traced to a loader; the exact `Vecto... | For you: no. |
| LCS-14 | solve_for_target maps "no probe sample converged" to SvdSolveFailed, and its branch-jump threshold is a heuristic from the centre of gravity | PHYSICS-SUSPECT (described correctly) | For you: no. |
| LCS-15 | Small inconsistencies between sibling APIs in body.rs and mechanism.rs | CONFIRMED | For you: no. Behaviour change. |
| LCS-16 | Two tests assert nothing that can fail, and a fourth copy of the same 4-bar fixture sits in each module | PARTLY. `inverse_solve_status_constructs_all_variants` (severity.rs:64-79) has no asser... | For you: no. Deleting tests conflicts with the repo's "more tests than fewer" rule, and the fixture move is a refactor. |

### linkage-forces-analysis-io

Detail: `docs/analyses/2026-11-10-whole-project-audit/linkage-forces-analysis-io-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/linkage-forces-analysis-io.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| LFA-01 | Saving a cosine-stroke linear driver writes a different driver than it loads | CONFIRMED | For you: no. Owner decision (delete or complete the feature). |
| LFA-02 | Doc comments in io say drivers are skipped on save and not restored on load; the code does both | CONFIRMED (line numbers for schema.rs are a few lines off: the text is at schema.rs:56-... | **Fixed** in `12b744e`: the save and load docs in io/ say which drivers round-trip (constant-speed, expression, linear) and which are skipped (closures without DriverMeta). |
| LFA-03 | Dead error variant and a dead legacy joint variant in io | PARTLY. `UnknownJointType` is dead (confirmed). `JointJson::RevoluteDriver` is never co... | **Fixed** in `6cbfac3`: `SerializationError::UnknownJointType` removed (never built, never matched). Left for you: `JointJson::RevoluteDriver`, which the canvas and blueprint code still match and old files may hold. |
| LFA-04 | Four identical copies of the body-lookup block in `joint_to_json` | CONFIRMED | For you: no. Refactor. |
| LFA-05 | A force that names a point on an unknown body panics the loader, and the validator that would catch it is never called | CONFIRMED (the validator has nine test call sites, not six) | For you: no. |
| LFA-06 | The "four two-point forces" match is written out in eight places | CONFIRMED (the evaluator repeats were not opened) | For you: no. |
| LFA-07 | Failed compound expansion leaves orphan bodies in the mechanism | CONFIRMED (read only; no run) | For you: no. |
| LFA-08 | Every force evaluation allocates several full-size vectors per element | CONFIRMED (a spring allocates three vectors inside the evaluator: `total` plus two help... | For you: no. |
| LFA-09 | Actuator stroke limits need both ends set, which the field docs and one test do not say | CONFIRMED | For you: no. The same rule is duplicated in gui/canvas/rendering/force_render.rs:262, so a fix must change both. |
| LFA-10 | Schema fields nothing writes or reads | CONFIRMED | For you: no. Owner decision on old files, and tests set the field. |
| LFA-11 | Doc comments that contradict the code they sit on | CONFIRMED | **Fixed** in `12b744e`: comments in analysis/transmission.rs and forces/compound.rs corrected; the schema-version sentence of ARCHITECTURE.md corrected in `cc3d498`. Left for you: two dated comments in io/mod.rs (lines 6 and 10) and the test-module rename. |
| LFA-12 | Virtual-work check reports "mechanism not built" when the mechanism has no driver | CONFIRMED | For you: no. Adds an error variant. |
| LFA-13 | The same 4-bar test fixture is typed out again and again in this part | CONFIRMED (the other-module sites in the list were not all opened) | For you: no. Test refactor. |
| LFA-14 | `.map_err(/e/ SerializationError::Build(e.to_string()))` repeated nine times in from_json and four more lookups in to_json | CONFIRMED | For you: no. |
| LFA-15 | The time-modulation maths is written twice | CONFIRMED | For you: no. |
| LFA-16 | Gas spring force at full compression is 2e9 times the nominal force | PHYSICS-SUSPECT (described correctly) | For you: no. |
| LFA-17 | The motor element and the motor-sizing check treat reverse speed differently | PHYSICS-SUSPECT (described correctly) | For you: no. |
| LFA-18 | `check_grashof` panics on a NaN link length | PARTLY. The two `partial_cmp(..).unwrap()` calls exist (grashof.rs:74 and :87) and pani... | For you: no. It changes behaviour on NaN input. |

### linkage-gui-canvas-panels

Detail: `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-canvas-panels-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-canvas-panels.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| LGC-01 | Two dead functions with misleading names (driver_lambda_units, draw_actuator_stroke_annotation) | CONFIRMED | For you: two functions with no caller (`driver_lambda_units`, `draw_actuator_stroke_annotation`). The check gives the exact deletion; held because each reads as a feature that was started: delete or wire up is your call. |
| LGC-02 | Stale #[allow(dead_code)] attributes on live items | PARTLY. Holds: trajectory_panel/profile_input.rs:365 allow on `parse_keyframes_csv` (cf... | **Fixed** in `6cbfac3`: the stale `#[allow(dead_code)]` on `parse_keyframes_csv` is gone. Left for you: the `SegmentHit` allow. |
| LGC-03 | Coupler-trace dashes re-implement draw_dashed_line, and the dashed force-zone rectangle is written twice | CONFIRMED | For you: no (refactor, trace look changes) |
| LGC-04 | Arrow drawing duplicated; the "one arrowhead" helper is bypassed | PARTLY. Holds: draw_force_arrow (primitives.rs:390-418) and draw_external_force_arrow (... | For you: no |
| LGC-05 | Whole SweepData cloned every frame to shift two vectors | CONFIRMED | For you: no |
| LGC-06 | Fourteen copies of the plot trailer and the per-plot Plot builder | CONFIRMED | For you: no |
| LGC-07 | Two tabs both labelled "Actuator Force" in stroke mode | CONFIRMED | For you: no (owner wording) |
| LGC-08 | Reference lines span a hard-coded 0..360 x range | CONFIRMED | For you: no |
| LGC-09 | Doc comments attached to the wrong function, plus stale in-body comments | CONFIRMED | For you: no (deleting only the blank lines would attach the wrong docs to the wrong functions; each doc needs rewording) |
| LGC-10 | Force editor repeats one DragValue block about 50 times in a 1,100-line function | CONFIRMED | For you: no |
| LGC-11 | Length units: three different conventions inside one property panel | CONFIRMED | For you: no |
| LGC-12 | Link label edit has no undo step, no dirty flag, and forces a sweep recompute per keystroke | PARTLY. Holds: pending_edits.rs:239-251 UpdateLabel edits blueprint and mechanism and c... | For you: no |
| LGC-13 | Parametric panel: five LinearActuator fields have no current-value lookup, and a dead comment | PARTLY. Holds: blueprint_ops.rs:234-238 offers stroke_min, stroke_max, end_stop_stiffne... | For you: no (adds behaviour) |
| LGC-14 | Health and Diagnostics sections show the same facts twice with different thresholds, colours and wording | CONFIRMED | For you: no |
| LGC-15 | Actuator-sizing tutorial names menu items and a message that do not exist | CONFIRMED | **Fixed** in `e7767cf`: the actuator-sizing tutorial quotes the real diagnostic line and the real File menu items. |
| LGC-16 | Health panel prints body-frame angles (and metres as degrees) while the plots show display-frame values | CONFIRMED | For you: no |
| LGC-17 | Force toolbar "Joint Torques" pairs the selected link with an arbitrary neighbour | CONFIRMED | For you: no (owner decision) |
| LGC-18 | Motion ribbon runs up to 20 Newton position solves every frame | CONFIRMED | For you: no |
| LGC-19 | Link-bar drawing and per-element screen-point boilerplate copied in several places | CONFIRMED | For you: no |
| LGC-20 | Grid labels and hover tooltips ignore display units; Y grid labels lose sub-mm precision | CONFIRMED | For you: no |

### linkage-gui-export

Detail: `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-export-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-export.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| LGE-01 | Per-joint reaction series are not padded, so they shift against angles_deg when early samples fail | CONFIRMED | For you: no, changes plotted/CSV output and needs a new test. |
| LGE-02 | A failed statics solve is stored as driver torque 0.0 in the forward sweep but NaN in trajectory mode | CONFIRMED | For you: no, changes plotted values, envelopes and actuator force. |
| LGE-03 | apply_motion_profile reads angles_deg as absolute 0-360 degrees and is applied to every sweep | PHYSICS-SUSPECT (described correctly) | For you: no, physics-suspect. |
| LGE-04 | GIF export silently drops frames that fail and still returns Ok, possibly an empty GIF | CONFIRMED | For you: no, adds error behaviour. |
| LGE-05 | Path-writing export wrappers are used only by their own tests (baseline "never used" settled) | CONFIRMED | For you: no, removes tests and public API; owner decision. |
| LGE-06 | add_plotly_multi_chart is dead and its job is written three times inline in generate_html_report | CONFIRMED | For you: `add_plotly_multi_chart` has no caller, and its job is written inline three times in `generate_html_report`. Delete the helper or use it: one decision, yours. |
| LGE-07 | Body, joint, trace and sensor names go into HTML, SVG and JS unescaped in most exporters; escapers disagree | CONFIRMED | For you: no, output changes and a design choice. |
| LGE-08 | Parameters and variables silenced with `let _ =`, a dead computation, and a doc comment on the wrong function | CONFIRMED | For you: no, signature changes at several sites and the unused-variable cleanup may leave other params unused. |
| LGE-09 | SweepData.active_range is always None; its writers and readers are dead | CONFIRMED | For you: no, serialised field, tests and docs use it. |
| LGE-10 | Two four-bar detectors with different rules | CONFIRMED | For you: no, merges two behaviours. |
| LGE-11 | compute_trajectory takes a full SVD of the same Jacobian twice per sample | CONFIRMED | **Fixed** in `792f28c`: `compute_trajectory` factors the constraint Jacobian once per sample, not twice. Same numbers; the time saved was not measured. |
| LGE-12 | compute_sweep_data_with_weights and compute_trajectory repeat the same setup; SweepData is built long-hand three times | CONFIRMED | For you: no, a refactor. |
| LGE-13 | write_sweep_csv codes each of eleven optional columns twice | CONFIRMED | For you: no, a refactor. |
| LGE-14 | Two samples ignore the requested driver joint, and the Custom 6-bar can return an unsolved start | CONFIRMED | For you: no, behaviour change. |
| LGE-15 | The four-bar skeleton is hand-copied into eleven sample builders although build_standard_fourbar exists | CONFIRMED | For you: no, a refactor. |
| LGE-16 | DXF overlay drops lines that cross the view, and import silently skips unsupported entities | CONFIRMED | For you: no, changes drawing behaviour; deleting 245 and 306 alone is a no-op but 306 binds a value, so left to the reviewer. |
| LGE-17 | svg.rs mechanism SVG and schematic.rs repeat the bounding-box and scaling block with different rules | CONFIRMED | For you: no. |
| LGE-18 | index_at_driver doc says trajectory sweeps leave angles_deg empty, but compute_trajectory fills it with sample times | CONFIRMED | For you: no, behaviour change (a doc-only fix would be safe but is not the finder's action). |
| LGE-19 | FirmwareAdapter trait has one implementor and two methods nothing calls | CONFIRMED | For you: no, owner decision, and a doc says it is intentional. |
| LGE-20 | Two identical sample-count tests, and a sample whose enum name does not match what it builds | CONFIRMED | For you: no, deleting even a duplicate test removes a test; the rename is an owner decision. |

### linkage-gui-state

Detail: `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-state-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/linkage-gui-state.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| LGS-01 | Drag-and-drop file import only reads `file.bytes`, which native eframe never fills | PARTLY (the code reads as described; the native-egui behaviour is not provable from the... | For you: no (behaviour change, needs native check) |
| LGS-02 | Autosave recovery deletes the autosave file even when loading it failed | CONFIRMED | For you: no (changes recovery behaviour) |
| LGS-03 | Shortcut help lists Ctrl+S and Ctrl+Shift+S on the web build, where no handler exists | PARTLY. Holds: rows at mod.rs:1166-1167 are in an unconditional array; the only `Key::S... | For you: no (UI behaviour on web) |
| LGS-04 | The driver-angle to time mapping is written out five times in scope, and one copy lacks the zero-omega guard | CONFIRMED | For you: no (refactor across sites, unguarded sites would change at omega 0) |
| LGS-05 | Same block of code copied within state | CONFIRMED | For you: no (refactor) |
| LGS-06 | Doc comment of `solve_at_angle` is split | CONFIRMED | **Fixed** in `12b744e`: the first two lines of `solve_at_angle`'s doc, which sat on another function, are back on it. |
| LGS-07 | Menu "Parametric Study [WIP]" is a permanently disabled checkbox | CONFIRMED | For you: no (owner decision) |
| LGS-08 | Grid "Spacing" setting and `auto_grid_spacing` overwritten on first canvas frame | CONFIRMED | For you: no (owner decision) |
| LGS-09 | Eight-plus copies of the "generate text, download, report outcome" menu block | CONFIRMED | For you: no (refactor) |
| LGS-10 | Native open/save/recover failures are logged but never shown to the user | CONFIRMED | For you: no |
| LGS-11 | Native autosave is never cleaned up, sibling autosaves never offered, New brings the recovery prompt back | CONFIRMED (by code reading, not run) | For you: no |
| LGS-12 | A failed load leaves the app half-updated | CONFIRMED | For you: no |
| LGS-13 | Dead branches and copies inside `load_from_json_str` | CONFIRMED | For you: no (the else-branch removal needs a restructure that cannot be compile-checked here) |
| LGS-14 | Template and custom-sample tests write into the developer's real ~/.linkage-sim/ | CONFIRMED | For you: no |
| LGS-15 | Templates and custom samples are two full copies of the same persistence and CRUD code | CONFIRMED | For you: no |
| LGS-16 | Undo/redo restore skips refreshes that rebuild() and load do | CONFIRMED | For you: no |
| LGS-17 | Removing a body leaves linear drivers (and force point refs) that point at it | CONFIRMED (linear driver part); the force-point-ref sentence for remove_attachment_poin... | For you: no |
| LGS-18 | The expression-driver editor applies on every keystroke and again on focus loss | CONFIRMED | For you: no |
| LGS-19 | Dead methods and variables in state and the input panel | CONFIRMED | **Fixed** in `6cbfac3`: the dead local `current_theta_0` in gui/input_panel.rs is gone. Left for you: `update_prismatic_axis`, `move_attachment_point` and `sync_active_load_case`. |
| LGS-20 | Repeated blocks and per-frame work in `update()` | CONFIRMED (minor: the two demo blocks differ by one line, the advance copy at mod.rs:28... | For you: no |

### mag-engine-calc

Detail: `docs/analyses/2026-11-10-whole-project-audit/mag-engine-calc-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/mag-engine-calc.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MEC-01 | "quote the failing value one last digit past the limit" written three times | PARTLY. Holds for gearbox.rs and shaft_joint.rs: gearbox.rs:51-57 is `quoted_above(cold... | For you: no (moves a helper across modules and edits gearbox.rs; behaviour is identical but it needs a build to check imports, which I may not run). |
| MEC-02 | Cargo.toml comments name checks and a host that no longer exist | CONFIRMED | For you: no (the true wording depends on whether the owner wants the gate checks added; the finder marked it needs-judgment). |
| MEC-03 | slider ranges are re-read from a freshly built input table on every call; the GUI caches the same data | CONFIRMED (structure; cost per call not measured, as the finder said) | For you: no (changes the size-controls API and wants the gui-smoke frame timing). |
| MEC-04 | workbook tables carried for the parity test only, with a doc comment that says the GUI uses one | PARTLY. Holds: `VALIDATION_ITEMS` (metal_design.rs:181) is read only by tests/static_da... | For you: no (owner decision: show it, hide it behind cfg(test), or fix the comment). |

### mag-engine-explain

Detail: `docs/analyses/2026-11-10-whole-project-audit/mag-engine-explain-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/mag-engine-explain.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MEX-01 | Assumption helpers rebuild the defaults and the whole input schema on every call, and the GUI calls them every frame | CONFIRMED (structure; the cost per call is not measured, as the finder said) | For you: no, the fix changes signatures and caching (needs-judgment, as the finder said). |
| MEX-02 | Stale "placeholder" and "until then" comments in warnings.rs: the teaching notes exist | CONFIRMED | **Fixed** in `12b744e`: engine/warnings.rs no longer calls shipped code a placeholder. |
| MEX-03 | assumptions.rs says the traceability test "belongs to plan A-3" and does not exist; it does | CONFIRMED | **Fixed** in `12b744e`: engine/assumptions.rs no longer says it belongs to plan A-3. Left for you: the copies of the sentence in docs/ai/03-structure.yaml:49 and tests/assumptions.rs:6 and :268. |
| MEX-04 | `Registry::term_values` and `Registry::graph` are unused; `Registry::choices` and `RESULT_CHOICES` are used by tests only | PARTLY. `term_values` (registry.rs:584) and `graph` (:552) have no caller anywhere (sea... | For you: no (deleting public registry API that docs/ai/03-structure.yaml:64 lists by name, `graph` among them; owner's call). |
| MEX-05 | `scope::Status` and the `Pending` variant are scaffolding of a finished plan: no chain is, or can be, Pending | CONFIRMED | For you: no (the removal edits tests; the doc count fix alone is a taste call the finder marked as such). |
| MEX-06 | `notes_before_snapshot` is a write-once generator that can only fail | CONFIRMED that it cannot pass; the finder's second point is PARTLY wrong. notes.rs:2422... | For you: no (the owner's call; docs name it). |
| MEX-07 | The custom-eval escape hatch is dead code that a test insists stays dead | CONFIRMED | For you: no (removes a variant and edits a test; owner's choice to keep a reserved hook). |
| MEX-08 | `TableField.unit` and `TableField.label` are never read (37 rows each) | CONFIRMED, with a miscount: TABLE_FIELDS has 40 rows (tables.rs:37-76), so 80 strings,... | For you: no (data deletion across 40 rows; the finder offered wiring them instead). |
| MEX-09 | Markup functions `ceil`, `floor`, `cosh`, `tanh` are implemented, typeset and documented but no record uses them | CONFIRMED | For you: no (grammar surface; owner decision). |
| MEX-10 | The 23 shaft joint paths are typed in three places (SCOPE chain, note `equations`, and the records file) | CONFIRMED | For you: no (refactor; the finder rates it low value). |
| MEX-11 | api.rs says JSON export "is M4"; the design file export exists | CONFIRMED for the JSON sentence; the "996 at M2" half is a stale figure beside `rows.le... | **Fixed** in `12b744e`: engine/api.rs no longer says JSON export is M4; it shipped. |

### mag-gui-guidance

Detail: `docs/analyses/2026-11-10-whole-project-audit/mag-gui-guidance-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/mag-gui-guidance.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MGG-01 | CheckTerm.name and CheckTerm.margin are built in production but only tests read them | CONFIRMED | For you: no, the fields are `pub(crate)` and the tests build `CheckTerm` literals (4348); moving them is a design change. |
| MGG-02 | the fix search recomputes the same moved designs and an unchanged baseline | PARTLY (the code reads as described; the frame cost is not provable without a timing) | For you: no, memoisation changes the search code path. |
| MGG-03 | guide tables type a path or note id as a literal where the file already has a constant | PARTLY (holds for the six sites below; guidance.rs:199 and 1815 not changed here) | **Fixed** in `c452531`: six literals in gui/guidance.rs now name the file's own constants. Left for you: guidance.rs:199 and :1815, whose constants live elsewhere. |
| MGG-04 | the unit of a result is looked up with the same one-liner in seven places | CONFIRMED | For you: no, replacing the calls may leave `result_info` imports unused in some files (warning, possible clippy -D) and needs a compile to check. |
| MGG-05 | Value to f64 conversion is written out in several modules | CONFIRMED (more copies than listed) | For you: no, the differing arms need a choice per site. |
| MGG-06 | two public typeset methods exist only for tests | CONFIRMED | For you: two public typeset methods (`TermColors::is_empty`, `texts`) are called only by tests. Marking them `#[cfg(test)]` shrinks the library's public surface, so it is your call. |
| MGG-07 | three hand-run `#[ignore]` printers and a review-sheet builder that serves only them | CONFIRMED (the facts); the keep or delete choice is the owner's | For you: no, owner-decision and documents name them. |
| MGG-08 | a hand-kept result count in docs, repeated in four places, with a history of going stale | PARTLY (the copies are real and stale now; the claim that nothing checks the number is... | For you: no, the four copies (including docs/ai) should move together, and I could not run the test to confirm 419/1115 is current. |
| MGG-09 | the typesetter spells its stroke width and small-script size inline in several places | CONFIRMED | For you: no, a six-site refactor with new helper names; cannot be compile-checked here. |
| MGG-10 | two guidance tests share a 25-line skeleton | CONFIRMED | For you: no, a test refactor with a choice of helper shape. |

### mag-gui-panel

Detail: `docs/analyses/2026-11-10-whole-project-audit/mag-gui-panel-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/mag-gui-panel.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MGP-01 | The 14 stored-3D input paths are written twice, and the second copy is test-only | CONFIRMED | For you: no, moving the const under cfg(test) or adding a test is a design choice and docs/specs name the constant. |
| MGP-02 | Under Auto the design shown and the solve key are rebuilt several times a frame | PARTLY (the call sites hold; cost not measured) | For you: no, a caching change alters behaviour paths. |
| MGP-03 | Every drawn input row builds its tooltip strings each frame, hovered or not | CONFIRMED | For you: no, moving into on_hover_ui closures changes widget code and needs a check that tests reading hover text still pass. |
| MGP-04 | The same few lines are written five times in panel.rs (and twice in input_ui.rs) | CONFIRMED (panel.rs part read; input_ui.rs :230-263 not re-read) | For you: no, a refactor of production code. |
| MGP-05 | Comments and README still describe a sizing mode switch the UI no longer has | CONFIRMED for inputs.rs:27-28, panel.rs:133-134 and sizing.rs:48; PARTLY for panel.rs:1... | **Fixed** in `12b744e`: comments in gui/panel.rs, gui/sizing.rs and gui/inputs.rs no longer name a sizing mode switch the screen does not draw. Left for you: panel.rs:1342 and README line 33. |
| MGP-06 | A part or grade row says the same thing twice, in two different wordings | CONFIRMED | For you: no, UX wording is pinned by tests and the UX overhaul may rewrite the rows. |
| MGP-07 | `gui::torque_gap_text` is re-exported for a screen dump that is not in the repository | NOT CONFIRMED (nothing in the tree uses the re-export, but whether the external dump ex... | Dropped: the check could not confirm it. |
| MGP-08 | `key_widgets` is rebuilt every frame in production code but read only by tests | CONFIRMED | For you: no, cfg(test) on a struct field changes the struct per profile and the production code writing it needs gating too. |
| MGP-09 | Magic numbers and path strings typed in view code | CONFIRMED | For you: no, the 2.5 threshold move is a design decision. |
| MGP-10 | `minimum_diameter_mm` is cleared in three overlapping places | CONFIRMED | For you: no, open_design is public and a direct caller would lose the clear. |
| MGP-11 | An allowlist in a test is permanently empty | CONFIRMED | For you: no, finder marked owner-decision (a documented hook). |

### mag-gui-views

Detail: `docs/analyses/2026-11-10-whole-project-audit/mag-gui-views-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/mag-gui-views.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MGV-01 | The results table and the clamp tab draw the catalogue label; every other view draws the display label | CONFIRMED (the code reads as described; whether it is a defect is the owner's call) | For you: no, it changes what the user sees, and spec 3.5 and the planned UX overhaul decide which label the table shows. |
| MGV-02 | Stored-3D path lists typed twice, and the dashboard copies are used only by tests | CONFIRMED (both parts: used only by tests; the 14 paths are typed in both places) | For you: no. Moving or deleting the constants breaks the rustdoc links at dashboard.rs:18, 235 and engine/temperature.rs:55, the README line 351 and docs/ai text name them, and plain_text.rs:1531 (a different file's test) reads the dashboard constant, so th... |
| MGV-03 | Two `pub` functions in shipped code are called only from tests | CONFIRMED | For you: two public functions (`Cell::as_text`, `package_label`) are called only by tests. Moving them into the test modules shrinks the public surface, so it is your call. |
| MGV-04 | The Equation panel reads `clamps::PAUSED` while the panel passes `clamps_paused` everywhere else | CONFIRMED | For you: no. `explorer_ui` and its callers and tests need a new parameter, and the const `START_HERE_HOVER` has users; the signature choice is a judgment. |
| MGV-05 | plots.rs matches the sweep status by literal strings the engine owns | CONFIRMED | For you: plots.rs matches the sweep status by two literal strings the engine owns. Shared constants (the check's exact change) or a status enum: your choice, since the strings are also the Python oracle's. |
| MGV-06 | Every dashboard guide recomputes its level and notes each frame, also when its block is shut | PARTLY. Holds: `indexed_guide_ui` (dashboard.rs:856-872) builds `guide_notes(...)` ever... | For you: no. Passing the level only helps one row, and lazy notes change `GuideShown`/`guide_ui`, used by tests. |
| MGV-07 | The same two axis lines are written out in three diagram functions | PARTLY. Holds for two functions: torque_angle (diagrams.rs:294-301) and heating_curve (... | For you: no. A helper changes how the shapes vec is built in three functions where shape order and count may be pinned by tests, and the third function does not fit the "two axes" helper. |
| MGV-08 | The `app` feature lists dependencies its `gui` feature already enables | CONFIRMED | For you: harmless: the `app` feature repeats two dependencies its `gui` feature already enables. Some keep such lists explicit on purpose. |

### python

Detail: `docs/analyses/2026-11-10-whole-project-audit/python-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/python.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| PY-01 | Three Python force elements are used nowhere and tested nowhere | CONFIRMED | For you: no, owner decision (delete or test). |
| PY-02 | Dead public functions and one dead dataclass in src/linkage_sim | CONFIRMED | For you: no, public names of a library package. |
| PY-03 | A test with no assertion can never fail | PARTLY (the missing assertion holds; "can never fail" is slightly too strong) | For you: no, the intended expectation must be chosen. |
| PY-04 | The four golden mechanism builders are copied between export_golden.py and test_golden_fixtures.py | CONFIRMED | For you: no. |
| PY-05 | The ternary-plate constructor and the six-bar initial-guess solver are pasted 7 and 4 times | PARTLY | For you: no. |
| PY-06 | The same four-bar and slider-crank builders are rewritten in about 20 test files and 10 scripts | CONFIRMED | For you: no. |
| PY-07 | Two golden files are used only by the Python tests, and the Python loader turns a missing fixture into a skip | CONFIRMED | For you: no, owner decision. |
| PY-08 | Sweep-based finite-difference tests can pass without checking anything | PARTLY | For you: no. |
| PY-09 | scripts/export_gifs.py behaves differently from the other scripts and hides failures | CONFIRMED (line numbers off) | For you: no. |
| PY-10 | 19 MB of generated GIFs in output/gifs are used by nothing | CONFIRMED | For you: no, owner decision. |

### tests-linkage-infra

Detail: `docs/analyses/2026-11-10-whole-project-audit/tests-linkage-infra-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/tests-linkage-infra.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| INF-01 | Gate 7 re-runs every test gate 4 already ran | CONFIRMED (the overlap); whether to drop it is the owner's call | For you: no, the owner may want the engine-alone run kept as a purity proof. |
| INF-02 | Gate 2 (`cargo clippy --all-targets`, linkage-sim-rs) can only fail on a compile error | CONFIRMED (by reading; the title slightly overstates, since deny-by-default clippy lint... | For you: no, owner decision (comment, ratchet or leave). |
| INF-03 | Gate 11's empty-pin FAIL branch is unreachable | CONFIRMED | **Fixed** in `3c7f9bb`: gate 11 of gate.sh prints its `no pin found` message when the wasm-bindgen pin is missing; before, the script exited silently one line earlier. |
| INF-04 | deploy-web.yml compiles wasm-bindgen-cli from source on every run, so its cache entry buys nothing | PARTLY. The rebuild on every run holds; the suggested fix "drop `--force`" is not safe. | For you: no, a CI change that cannot be tested locally. |
| INF-05 | Docs say the deploy uses "stable" Rust, the workflow pins 1.89.0, and nothing else pins a toolchain | CONFIRMED | For you: no, owner decision (adding a toolchain file changes every local build). |
| INF-06 | audit-campaign.js and fix-campaign.js hard-code the main checkout's absolute path as ROOT | CONFIRMED (the hard-coded path); whether workflows should run elsewhere is the judgment | For you: no. |
| INF-07 | gui-smoke.js re-derives the calculator's `passed` from its checks but takes the linkage step's `passed` on the agent's word | CONFIRMED | For you: no, it changes the verdict logic of a gate (13). |
| INF-08 | rust-test-fixer agent hard-codes `model: opus`, against the repo's model rule | CONFIRMED | For you: no, owner decision, and the file is outside the worktree. |
| INF-09 | Post-edit cargo-check hook matches tests/*.rs but only checks the lib | CONFIRMED | For you: no, it changes the owner's local hook and its run time, and lies outside the worktree. |
| INF-10 | fbd-derive skill and fbd-math-reviewer agent cite `evaluation.rs:424-425` for the actuator sign convention; it now sits at 437-438 | CONFIRMED | For you: the two files (the fbd-derive skill and the fbd-math-reviewer agent) are git-ignored and exist only in your main checkout, so they were not touched (register row A-16). The check gives the exact replacement text. |
| INF-11 | tests/braindump_repro.rs is tracked and still the repro BL-017 depends on, yet its header says "TEMPORARY ... Not for commit" | CONFIRMED (one small slip: the "(untracked)" note is in BL-017's notes, backlog.yaml:38... | For you: no, a rename touches the test name, the backlog text and the BL-017 acceptance. |
| INF-12 | The same standard 4-bar is hand-built five times across two test files, with its initial-guess helper written twice | CONFIRMED (the count is low: six builders) | For you: no, a new shared module touching golden-fixture tests. |
| INF-13 | `first_actuator` is copied between two test files, and two helpers in one file have identical bodies | CONFIRMED | For you: no, it edits tests and needs the shared module of INF-12 for the cross-file part. |
| INF-14 | Assertions sit inside `if let Ok` / `if converged` / `return Ok(())` guards, so these tests pass when the solver fails every case | CONFIRMED for the structure; whether current cases reach the assertions was not run | For you: no, the fix changes test strictness and may expose failing cases. |
| INF-15 | `expression_driver_matches_constant_speed` (property test) tests the `meval` crate, not this repo; `meval` is also listed twice in Cargo.toml | CONFIRMED | **Fixed** in `e451bc7`: the duplicate `meval` dev-dependency is gone; Cargo.lock unchanged. Left for you: the property test that exercises the meval crate and no code of this repository. |
| INF-16 | Direct dependencies `wasm-bindgen-futures` and `uuid` of linkage-sim-rs are named nowhere in its code | PARTLY. `wasm-bindgen-futures` holds; `uuid` is not shown to be unneeded. | For you: no, a manifest and lockfile change that only a wasm32 check can prove. |
| INF-17 | .gitignore carries a stale heading and two commented-out rules | CONFIRMED | For you: the commented-out ignore rules also record that the golden fixtures and output GIFs are tracked on purpose; deleting them is a wording choice, not a defect. |

### tests-magcoupling

Detail: `docs/analyses/2026-11-10-whole-project-audit/tests-magcoupling-verified.md` (the check) and `docs/analyses/2026-11-10-whole-project-audit/tests-magcoupling.md` (the finder).

| id | finding | verdict | what happened |
|---|---|---|---|
| MT-01 | Test setup helpers copied between test files instead of living in tests/common | PARTLY. Holds: `fn design(edit)` is the same 5 lines at gearbox.rs:45-49 and shaft_join... | For you: no. It edits several test crates, the grade-pair helper has variants that differ on purpose, and the tests are what the gate runs. |
| MT-02 | Differential JSON (about 10 MB) is parsed again by almost every test that needs it | PARTLY. Holds: `load_cases` (common/mod.rs:324-365) reads and parses with no cache; Gre... | For you: no. A process-wide cache changes how tests share state and the return type of `load_cases` (callers take owned `Vec<Case>`), and the benefit is unmeasured. |
| MT-03 | tests/data/reference_values.json is a 147 kB byte-for-byte copy kept equal by a test | CONFIRMED | For you: no. It deletes a fixture and a test, and whether magcoupling-rs must build or publish alone is the owner's decision. |
| MT-04 | A no-op `let _ = &statics;` left in the traceability test | CONFIRMED | **Fixed** in `6cbfac3`: `let _ = &statics;` removed from magcoupling-rs/tests/explain.rs (the value is read a few lines later). |
| MT-05 | Test and generator doc comments no longer match what is tested | CONFIRMED | **Fixed** in `12b744e`: the header comment of magcoupling-rs/tests/static_data.rs no longer calls tested tables `later`. Left for you: gen_differential.py:35. |
| MT-06 | The ignore guard checks six files, but its name and doc say "only the review tools are ignored" | PARTLY. Holds: the guard (explain.rs:1712-1765) scans exactly six sources (notes.rs, pl... | For you: no. A directory walk or a renamed test is a design choice (and `include_str!` of 6 names is how the guard stays compile-time). |
| MT-07 | The vendored oracle's independent audit suite and its tools are run by nothing automatic | CONFIRMED (small miscount: audit/references has 13 modules, not 11; audit/tests has 17... | For you: no. Adding the suite to the gate is the owner's call and its current health is unknown. |
| MT-08 | The oracle's README tells readers to extend and edit the code that reference/README.md says never to edit | CONFIRMED | For you: no. The README is part of the vendored oracle folder, and what to keep as history is the owner's call. |
| MT-09 | Two example front ends in the oracle that nothing runs, tests or links to | CONFIRMED | For you: no. The files sit in the vendored oracle and the owner decides whether to keep them as history. |
