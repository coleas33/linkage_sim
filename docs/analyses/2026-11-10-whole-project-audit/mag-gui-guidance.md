# Audit findings: mag-gui-guidance
Scope read: guidance.rs production code (lines 1-3100) in full plus tests sampled (3100-3260, 3737-4040, 4820-5010, 5360-5530, 5800-5930, 9670-9942, test names for the rest); torque_card.rs production code (1-820) in full plus tests sampled (820-1070, 3500-3560, test list); typeset.rs production code (1-1030) in full, tests by name; plain_text.rs lines 1-130, 740-1430 and the HELP/HINTS rows checked by script for duplicate keys and texts; trace.rs lines 1-100.

## MGG-01: CheckTerm.name and CheckTerm.margin are built in production but only tests read them
- Kind: not needed
- Where: magcoupling-rs/src/gui/guidance.rs:732-737 (struct), 747-844 (check_terms, builds name String and margin for every term), 848-856 (joint_term); readers: torque_card.rs:803-804 uses only `.passes` and `.quoted`
- Evidence: `git grep check_terms` shows production callers only at torque_card.rs:800 and :802 (`quoted_changes`), which read `before.passes`, `after.passes` and `before.quoted`. `.name` and `.margin` are read only in guidance.rs tests (`terms()` at 4344, `margin()` at 4359, `failing_term_names` at 4368); the torque_card tests read only `.quoted` and `.passes`. The temperature terms get their name from `format!("{t:?}")` (the Rust Debug text, "HotDayMagnet(Inner)"), which no screen shows.
- Why it matters: `quoted_changes` runs once per number side per failing check (up to 8 sides), and for each term allocates the name Strings and works out a margin that nothing in the app reads; the margin formulas also exist a second time in the real fix code (TemperatureTerm::margin, JointTerms::margin, axis_reserves), so the two copies can drift (test check_terms_keeps_the_fix_tests_margins exists to police that). In the SHAFT_JOINT arm, `shaft_joints()` (line 806) builds links and joints and the arm then builds links and joints again (807-808).
- Suggested action: move name and margin (and the margin arithmetic) behind #[cfg(test)] or into the tests module, leaving prod `check_terms` to return passes and quoted; or reuse the Ok value of `shaft_joints` instead of rebuilding.
- Fix class: needs-judgment
- Confidence: likely (production readers searched with git grep across both crates; would be settled by deleting the two fields and seeing only test code fail to compile)

## MGG-02: the fix search recomputes the same moved designs and an unchanged baseline
- Kind: inefficient
- Where: magcoupling-rs/src/gui/guidance.rs:1139-1165 (`helps`), 1098-1107 (`all_fixes`), 468 / 189 / 331 / 358 (FACE_GAP Raise listed by four guides), 467 / 357 / 490-491 (MAGNET_LENGTH Lower listed by three guides)
- Evidence: `helps` runs `moves(...)` then `compute_all(design)` for every candidate move on every call, and every guide calls it for each of its own fixes, so one input step that two or three failing guides list (FACE_GAP Raise: clearance, gearbox, hub slip, cup slip; MAGNET_LENGTH Lower: gearbox, cup slip, space claim) runs the same `compute_all` once per guide. `let (to, from) = (margin(design, &after), margin(inputs, results))` (1159) recomputes the baseline margin for each candidate although it does not depend on it, and `counted` / `before` (1146-1150) are rebuilt on every call.
- Why it matters: `FixCache::get` recomputes all of this on every input change, that is on every frame of a slider drag, while any check fails (the default design fails several); gui-smoke times that drag against one 60 fps frame.
- Suggested action: memoise `(path, direction) -> (moved designs, their DesignResults)` inside one `all_fixes` pass, hoist the baseline margin and levels out of the candidate loop. Time `FixCache::get` on the default design before and after.
- Fix class: needs-judgment
- Confidence: unsure (the duplication is certain from the tables; whether it is visible in a frame needs a timing of `FixCache::get` on a failing design)

## MGG-03: guide tables type a path or note id as a literal where the file already has a constant
- Kind: inconsistent
- Where: magcoupling-rs/src/gui/guidance.rs:582 ("metal.hot_min_check" for HOT_MIN, line 148), 603 ("metal.clearance_check" for CLEARANCE, 149), 604 ("materials.cup_wall_check" for CUP_WALL, 150); 176 and 587 ("hot_low_cold_high" for TORQUE_BAND_NOTE, 637); 547 ("end_effect" for BANNER_NOTE, 641); 199 ("metal.cup_wall_corner_mm" for plain_text::CUP_WALL_INPUT, plain_text.rs:42); 1815 and 2421 ("temperature.demag.inner_onset_single_ring_C" twice, ONSET const only in the second)
- Evidence: in ROW_LEARN the other six checks use their constants (`(TEMPERATURE, ...)`, `(CLAMP, ...)`, `(SHAFT_JOINT, ...)`, `(GEARBOX, ...)`, `(SPACE_CLAIM, ...)`, `(END_EFFECT, ...)`) while the first three rows spell the path; GEARBOX_GUIDE uses `Learn::Note(TORQUE_BAND_NOTE)` (470) but HOT_MIN_GUIDE uses `Learn::Note("hot_low_cold_high")` (176), though the constant's own comment says a test keeps the two equal.
- Why it matters: a rule followed in most places and broken in a few; a renamed check or note needs several edits and a test to catch the miss.
- Suggested action: use the constants at those sites.
- Fix class: mechanical-safe
- Confidence: certain

## MGG-04: the unit of a result is looked up with the same one-liner in seven places
- Kind: inefficient (repetition)
- Where: magcoupling-rs/src/gui/guidance.rs:1843, 1982, 2007, 2424, 2557; magcoupling-rs/src/gui/torque_card.rs:812; magcoupling-rs/src/gui/explorer.rs:1133
- Evidence: each reads `result_info(path).map_or("", |info| info.meta.unit)`; the same file has `input_unit(path)` (1998) for inputs, but no counterpart for results, and `result_text`, `number_text` and `window_quotes` each re-spell it.
- Why it matters: the owner flags repetition; the unit rule (empty for an unknown path) lives in seven copies.
- Suggested action: add one `result_unit(path)` beside `result_info` (dashboard.rs) or `input_unit` and call it.
- Fix class: mechanical-safe
- Confidence: certain

## MGG-05: Value to f64 conversion is written out in several modules
- Kind: inefficient (repetition)
- Where: magcoupling-rs/src/gui/guidance.rs:1311-1317 (`number`), magcoupling-rs/src/gui/input_ui.rs:89-95 (`number`), magcoupling-rs/src/gui/explorer.rs:813, magcoupling-rs/src/gui/inputs.rs:964, magcoupling-rs/src/engine/compat.rs:188 (`as_f64`), magcoupling-rs/src/engine/explain/probes.rs:97
- Evidence: guidance.rs `Value::Num(x) => Some(x), Value::Int(i) => Some(i as f64), Value::Text(_) | Value::None => None` and input_ui.rs `Value::Num(x) => Some(*x), Value::Int(i) => Some(*i as f64), _ => None` are the same function (one takes the value by value, one by reference); no `Value::as_f64` method exists (`git grep "impl Value"` finds none).
- Why it matters: six copies of one conversion; the text and none arms differ between them (NaN, None, panic), which is the kind of drift the repetition invites.
- Suggested action: one `Value::as_f64() -> Option<f64>` in engine::meta and call it; keep the NaN or panic variants as thin wrappers.
- Fix class: needs-judgment
- Confidence: likely (the arms of compat.rs, probes.rs, explorer.rs and inputs.rs were seen by grep line only, not read in full)

## MGG-06: two public typeset methods exist only for tests
- Kind: not needed
- Where: magcoupling-rs/src/gui/typeset.rs:187-198 (`Laid::texts`), 129-131 (`TermColors::is_empty`)
- Evidence: `git grep` over magcoupling-rs finds `.texts()` only inside typeset.rs's `mod tests` (lines 1074, 1203, 1210, 1294 and more) and `TermColors::is_empty` only at typeset.rs:1231 (`assert!(TermColors::none().is_empty())`); explorer.rs, readouts.rs and panel.rs, the production users of typeset, call neither. They are `pub`, so rustc does not warn.
- Why it matters: public API surface and compiled code with no production caller.
- Suggested action: mark both `#[cfg(test)]`, or move into the tests module.
- Fix class: mechanical-safe
- Confidence: certain

## MGG-07: three hand-run `#[ignore]` printers and a review-sheet builder that serves only them
- Kind: not needed
- Where: magcoupling-rs/src/gui/guidance.rs:9678-9941 (`guidance_strings_for_review`, about 260 lines); magcoupling-rs/src/gui/plain_text.rs:1235-1252 (`overlay_strings_for_review`), 1348-1417 (`review_sheet`, `the_review_sheet_lists_every_input_of_its_groups_and_every_hint`, `help_for_review`)
- Evidence: each is `#[ignore = "prints ... for the physics review; run by hand"]` with `println!` and no assertion; guidance.rs:9678 cites "plan Task 11, step 1" (slice 1, shipped). `review_sheet` is called only by the ignored `help_for_review` and by the test that checks `review_sheet`'s own output, so that test can only fail if the review printer itself breaks.
- Why it matters: about 350 lines of test code that CI never runs and that reads strings which move with every help change (the guidance printer names constants of the size tab, plots, explorer and panel modules, so a rename there breaks the test build).
- Suggested action: if the planned UX overhaul will not re-run the G-3 text reviews, delete them; otherwise keep one printer and drop the self-test of `review_sheet`.
- Fix class: owner-decision
- Confidence: likely (what would settle it: whether the owner wants the printers for the next text review)

## MGG-08: a hand-kept result count in docs, repeated in four places, with a history of going stale
- Kind: inconsistent
- Where: magcoupling-rs/src/gui/trace.rs:7-8 ("416 of the 1112 results"); magcoupling-rs/README.md:328; docs/ai/04-memory.yaml:23; docs/ai/05-update-tracker.md:33
- Evidence: no test asserts 416 or 1112 (`git grep -n "416\b"` in tests finds only unrelated material names); docs/analyses/2026-10-14-magcoupler-help-review.md:3639-3642 (R-6, R-7) records that the same count was already stale twice ("392 of 1087", "393 of 1088") before BL-064 moved it to 416 of 1112.
- Why it matters: four copies of a number that changes whenever a result or an equation record is added, with nothing that checks it.
- Suggested action: either drop the number from the doc comment, or add a test that counts the explained results and compares with the number the docs name.
- Fix class: needs-judgment
- Confidence: unsure (I did not count the registry; a one-line test counting `result_rows` against `registry.upstream_inputs` would settle whether it is still right)

## MGG-09: the typesetter spells its stroke width and small-script size inline in several places
- Kind: inefficient (repetition)
- Where: magcoupling-rs/src/gui/typeset.rs:414, 472, 490, 526 (`(size / 14.0).max(1.0)` four times); 578 and 779 (`(size * 0.6).max(MIN_SIZE)` twice); 55 and 319-321 (`SCRIPT_SCALE` and `script_size` exist only for the 0.7 case)
- Evidence: `brackets`, `frac`, `sqrt` and `element` each compute `let stroke = (size / 14.0).max(1.0)` (or inline it in the Ink::Line); `sum` and the `Expr::Peak` arm each compute `let small = (size * 0.6).max(MIN_SIZE)`.
- Why it matters: one drawing rule (rule thickness) in four copies; changing it means finding all four.
- Suggested action: add `fn rule_width(size) -> f32` and a `SUM_LIMIT_SCALE` constant or `fn limit_size(size)` next to `script_size`.
- Fix class: mechanical-safe
- Confidence: certain

## MGG-10: two guidance tests share a 25-line skeleton
- Kind: inefficient (repetition)
- Where: magcoupling-rs/src/gui/guidance.rs:5408-5444 (`near_the_gearbox_limit_the_hot_minimum_guide_shows_no_fix`) and 5882-5919 (`near_the_joint_limit_the_hot_minimum_guide_shows_no_fix`)
- Evidence: both build a design that passes one check just under its limit, take `let [stepped] = &moves(&inputs, &results, longer.0, longer.1)[..] else { panic!("one step of longer magnets") }`, assert the other check turns Bad on `stepped`, loop `for paused in [false, true]` asserting `!helps(...)` and `assert_open_block_draws_no_fix(...)`, then build a `roomier` design and assert `fixes(&HOT_MIN_GUIDE, ...) == [longer]`; only the margin-making edit and the second check differ.
- Why it matters: the owner flags repetition; the same flow is also in `at_the_hot_minimum_s_limit_the_gearbox_guide...` (5447) in shortened form.
- Suggested action: one helper taking (design, other check, roomier edit).
- Fix class: needs-judgment
- Confidence: likely

## Leads settled
- Section 6 (long functions): `guidance_strings_for_review` (guidance.rs:9683, 259 lines): real but it is the ignored printer, see MGG-07; `check_terms_keeps_the_fix_tests_margins` (guidance.rs:4381, 255 lines), `the_shaft_joint_problem_quotes_the_live_numbers` (8449, 209 lines), `the_card_s_lines_read_as_drafted` (torque_card.rs:2175, 276 lines), `the_words_follow_the_order_of_the_tests` (torque_card.rs:1073, 271 lines): long test functions that each read as a table of expected strings, not defects, no finding.
- Section 6 (source files over 2,500 lines): guidance.rs (9,942 lines, about 3,100 of them production) and torque_card.rs (3,767 lines, about 820 production): the size is test code, not a defect; guidance.rs production is a candidate split (guides, problems, consequences) but that is a redesign idea, not reported.
- Section 4 (duplicated runs): no pair of the machine list falls inside my five files.
- Section 2 (items named once): none of the six items is in my part; a name-count of every fn, const, struct and enum of my five files across both crates found no private item with a single occurrence and no pub item without a user (the test-only pub methods are MGG-06).
- Section 5 (files nothing names): none in my part.

## Not read
- No file of my part is unread; partly read: guidance.rs tests 3260-3737, 4040-4820, 5010-5360, 5530-5800 and 5930-9670 (test names only); torque_card.rs tests 1070-3500 and 3560-3767 (test names only); plain_text.rs HELP row bodies 130-740 (checked by script for duplicate keys and texts, not read) and tests 1430-2366; typeset.rs tests 1030-1991 (names only); trace.rs tests 97-248.

