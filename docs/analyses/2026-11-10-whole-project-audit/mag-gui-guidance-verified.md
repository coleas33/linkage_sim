# Verified: mag-gui-guidance

## MGG-01: CheckTerm.name and CheckTerm.margin are built in production but only tests read them
- Verdict: CONFIRMED
- Checked: guidance.rs:728-844 (struct and check_terms, FIT_CHECK arm also builds names/margins), :806-808 (SHAFT_JOINT arm calls `shaft_joints(..)` then `shaft_joint_links(..)` again). Search of magcoupling-rs/src outside guidance.rs: only torque_card.rs:800, 802 call check_terms in production (they read passes/quoted); other torque_card hits (2493, 2553, 2607) are tests. `.name`/`.margin` read only in guidance tests (`terms()` 4344, `margin()` 4359, `failing_term_names` 4368).
- Safe to apply without judgment: no, the fields are `pub(crate)` and the tests build `CheckTerm` literals (4348); moving them is a design change.

## MGG-02: the fix search recomputes the same moved designs and an unchanged baseline
- Verdict: PARTLY (the code reads as described; the frame cost is not provable without a timing)
- Checked: guidance.rs:1139-1165 `helps`: `counted` and `before` are rebuilt on every call (1146-1150); `margin(inputs, results)` is evaluated inside the per-candidate closure (1159) though it does not depend on the candidate; each call runs `moves` then `compute_all(design)` per candidate. all_fixes (1098-1107) calls `fixes` per guide. The cross-guide overlap tables (FACE_GAP, MAGNET_LENGTH lists) were not re-read, and no timing was possible.
- Safe to apply without judgment: no, memoisation changes the search code path.

## MGG-03: guide tables type a path or note id as a literal where the file already has a constant
- Verdict: PARTLY (holds for the six sites below; guidance.rs:199 and 1815 not changed here)
- Checked: guidance.rs:148-150 define HOT_MIN, CLEARANCE, CUP_WALL; :637 TORQUE_BAND_NOTE = "hot_low_cold_high"; :641 BANNER_NOTE = "end_effect". Literals at :582, :603, :604 (ROW_LEARN rows; the later rows 613-618 use the constants), :587, :176, :547. Values are identical. plain_text::CUP_WALL_INPUT (plain_text.rs:42) is pub, but guidance.rs:199 would need an import; the "temperature.demag.inner_onset_single_ring_C" at 1815 is a literal beside a `const ONSET` that is local to another function (2421), so it cannot just be reused.
- Safe to apply without judgment: yes for the six sites (same string, compile-time constants, no test or script reads the source text).
- Exact change (magcoupling-rs/src/gui/guidance.rs):
  - line 582: old `("metal.hot_min_check", HOT_MIN_GUIDE.learn),` new `(HOT_MIN, HOT_MIN_GUIDE.learn),`
  - line 603: old `("metal.clearance_check", CLEARANCE_GUIDE.learn),` new `(CLEARANCE, CLEARANCE_GUIDE.learn),`
  - line 604: old `("materials.cup_wall_check", CUP_WALL_GUIDE.learn),` new `(CUP_WALL, CUP_WALL_GUIDE.learn),`
  - line 587: old `Learn::Note("hot_low_cold_high"),` new `Learn::Note(TORQUE_BAND_NOTE),`
  - line 176: old `learn: Learn::Note("hot_low_cold_high"),` new `learn: Learn::Note(TORQUE_BAND_NOTE),`
  - line 547: old `learn: Learn::Note("end_effect"),` new `learn: Learn::Note(BANNER_NOTE),`

## MGG-04: the unit of a result is looked up with the same one-liner in seven places
- Verdict: CONFIRMED
- Checked: search for `result_info(..).map_or("", |info| info.meta.unit)`: guidance.rs:1843, 1982, 2007, 2424, 2557; torque_card.rs:812; explorer.rs:1133 (7 hits). `input_unit` exists at guidance.rs:1998; `result_info` at dashboard.rs:489.
- Safe to apply without judgment: no, replacing the calls may leave `result_info` imports unused in some files (warning, possible clippy -D) and needs a compile to check.

## MGG-05: Value to f64 conversion is written out in several modules
- Verdict: CONFIRMED (more copies than listed)
- Checked: the Num/Int-to-f64 match appears at guidance.rs:1313-1315, input_ui.rs:91-93, engine/compat.rs:188-192, explorer.rs:812-814 (finite-only variant), inputs.rs:963-965 and 1469-1471, plain_text.rs:1912-1914, size_controls.rs:412-414, probes.rs:96-97, plus tests/schema.rs:84 and tests/common/mod.rs:396. Their other arms differ (None, return false, continue, unreachable, panic).
- Safe to apply without judgment: no, the differing arms need a choice per site.

## MGG-06: two public typeset methods exist only for tests
- Verdict: CONFIRMED
- Checked: `.texts()` has 17 hits in the whole worktree, all inside typeset.rs from line 1074 on (its tests); `TermColors::is_empty` (typeset.rs:129) is called only at :1231 in a test; no `.marks().is_empty` anywhere; magcoupling-rs/tests/ and the web files name neither. Items are `pub` in typeset.rs:129 and :187.
- Safe to apply without judgment: yes (the methods are removed only from non-test builds; no integration test, script or document uses them).
- Exact change (magcoupling-rs/src/gui/typeset.rs): add the attribute `#[cfg(test)]` on its own line directly above line 129 `    pub fn is_empty(&self) -> bool {` and above line 187 `    pub fn texts(&self) -> Vec<(String, Rect)> {` (keep the doc comments above the attribute).

## MGG-07: three hand-run `#[ignore]` printers and a review-sheet builder that serves only them
- Verdict: CONFIRMED (the facts); the keep or delete choice is the owner's
- Checked: #[ignore] at guidance.rs:9682 (guidance_strings_for_review), plain_text.rs:1239 (overlay_strings_for_review), :1405 (help_for_review); `review_sheet` at plain_text.rs:1351 called at 1391 (its own test) and 1414 (the ignored printer). README.md, docs/ai/05-update-tracker.md and several plans/analyses name these printers.
- Safe to apply without judgment: no, owner-decision and documents name them.

## MGG-08: a hand-kept result count in docs, repeated in four places, with a history of going stale
- Verdict: PARTLY (the copies are real and stale now; the claim that nothing checks the number is wrong)
- Checked: "416 of the 1112" at trace.rs:7 and README.md:328 (docs/ai/04-memory.yaml:23 and 05-update-tracker.md:33 not re-read; the former matched the search in a long line). But trace.rs:194 has `assert_eq!((explained, table_entries().len()), (419, 1115))` in `an_input_no_explained_result_reads_traces_none_and_says_so`, so a test does pin the count, and it reads 419 and 1115, not 416 and 1112: the doc copies are already out of date.
- Safe to apply without judgment: no, the four copies (including docs/ai) should move together, and I could not run the test to confirm 419/1115 is current.

## MGG-09: the typesetter spells its stroke width and small-script size inline in several places
- Verdict: CONFIRMED
- Checked: typeset.rs:414, 490, 526 (`let stroke = (size / 14.0).max(1.0);`) and :472 (`width: (size / 14.0).max(1.0)`); :578 and :779 (`let small = (size * 0.6).max(MIN_SIZE);`); SCRIPT_SCALE at :55 used by `script_size` (:319-321) only.
- Safe to apply without judgment: no, a six-site refactor with new helper names; cannot be compile-checked here.

## MGG-10: two guidance tests share a 25-line skeleton
- Verdict: CONFIRMED
- Checked: guidance.rs:5408-5444 and :5882-5919 read in full: same flow (design, `let [stepped] = &moves(..)[..] else { panic!("one step of longer magnets") }`, `for paused in [false, true]` with `!helps(..)` and `assert_open_block_draws_no_fix`, then a roomier design and `fixes(..) == [longer]`). The joint test adds an assert on the check text. The third test at 5447 was not read.
- Safe to apply without judgment: no, a test refactor with a choice of helper shape.

## Seen while checking
- None.
