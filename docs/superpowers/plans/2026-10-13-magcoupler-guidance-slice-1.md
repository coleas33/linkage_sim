# Magcoupler guidance, slice 1: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the magcoupler explain itself where the user looks:
- one torque number;
- the solver on the Size tab;
- "what moves what" lines;
- a Problem / Why / Fix block under each check;
- a Why link on every dashboard row;
- the Size tab as the first view.

**Architecture:** A pure GUI-side guidance layer (`gui/guidance.rs`) builds every hint from live `compute_all` results. Small engine helpers in `engine/size_controls.rs` and `engine/sizing.rs` provide breakdowns, fit search and range-bounded solves. The panel draws the hints. The torque model and the workbook-locked engine text are untouched.

**Tech Stack:** Rust; egui 0.32 / egui_plot 0.33; the headless egui harness in `magcoupling-rs/src/gui/test_support.rs`; `linkage-sim-rs/scripts/gate.sh`.

**Spec:** `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`. Read it with this plan; the plan argues from it.

**How this plan was made.** Ten drafters each wrote one task against a fixed interface contract. Several verified their red/green cycles in scratch copies of the crate. A consistency critic then checked every task against the code, the contract and the spec. **The "Amendments" section below settles every critic finding and is binding.** Where an amendment and a task's text disagree, the amendment wins. Each task section begins by pointing to its amendments.

## Global Constraints

- Audience: mechanically literate engineers who are not magnetics specialists. Plain words first, numbers from live results, the input to change and which way.
- Numbers in any text come from results or inputs at run time, never literals (spec G-3).
- No guidance or consequence text contains:
  - `C\d{2,3}` cell refs;
  - `E\d{1,2}` correction ids;
  - "workbook", except the exact phrase "Workbook example";
  - "Addendum", "Rust-only" or "Calculator!".

  `guidance::plain_text_violations` enforces this.
- Every Why sentence is at most 30 words and states one idea. A test enforces the limit.
- Glyphs: egui's default fonts have no U+2192 or U+2264, so write `->` and `<=`. Every new drawn string passes the existing glyph test (`assert_glyphs`).
- The workbook text lock: engine labels, help and verdict strings do not change (`tests/python_schema.rs`). Plain text lives in the GUI layer.
- Idle frames never edit inputs (`idle_frames_change_no_input`).
- Files and share links keep `DESIGN_VERSION`. GUI-only state (welcome dismissal, Minimum OD) is not saved.
- Engine modules stay free of egui types.
- `cargo clippy --all-targets --features app -- -D warnings` stays clean for magcoupling-rs. Run `cargo fmt`.
- Spec section 8: none of its physics traps may be written.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
  ```
- Commands:
  - engine tests: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test <file> <name>`
  - panel tests: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib <name>`
  - full gate: `bash linkage-sim-rs/scripts/gate.sh`

## Review Focus

The five failure modes most likely to bite a user that the drafted tests did not pin. Each has its test added in the owning task's amendments.

1. **The owner's own 8-magnet case with Minimum OD unset.** The fit line must not imply the coupling cannot get smaller when only the slider floor stops it. It must say how far it could go and that Minimum OD unlocks it (T5 and T6 amendments).
2. **Ring-size Auto.**
   - The locked Outside diameter slider must show the solved diameter: cap and thread follow the cup, as the slider itself does.
   - The outcome line must name Outside diameter, not an internal radius (T1 and T7 amendments).
3. **A Fix that does not help.**
   - Every Fix shown for a failing term must move that term toward passing.
   - No Fix may turn another passing check red on the same design (T4 amendment, test 5.2).
4. **Two fixes on the first screen that contradict each other.** Hot-min "lower the gap" against clearance "raise the gap" (T4 amendment).
5. **The dashboard overflowing the 1280×800 app window** so that rows the app test expects are no longer painted (T8 amendment).

---

## Amendments (binding; they override the task text)

### Contract changes (all tasks)

- **C-1 `CheckGuide::fixes`.** Its type becomes `pub fixes: Fixes`, with:
  - `pub enum Fixes { Static(&'static [(&'static str, Direction)]), PerTerm(fn(&DesignInputs, &DesignResults) -> Vec<(&'static str, Direction)>) }`
  - `pub fn fixes(guide: &CheckGuide, inputs: &DesignInputs, results: &DesignResults) -> Vec<(&'static str, Direction)>`

  Only the temperature guide uses `PerTerm`. T8 draws `guidance::fixes(..)`.
- **C-2 `Direction`.** It becomes `pub enum Direction { Raise, Lower, Change }`.
  - `fix_text` writes "Raise L", "Lower L" or "Change L".
  - `Change` is for choice and picker inputs (adhesive, magnet part or grade).
  - The rule "every fix path is an input with a slider" becomes "an input with a slider or choices, or a picker input". Test 2 is amended to match.
- **C-3 Label disambiguation.** `fix_text` uses the input's on-screen label from `InputCatalogue`. When two inputs share that label, it appends the group heading in parentheses, for example "Raise Boss outside diameter (Shaft clamps)". A test checks that every fix text is unique per path.
- **C-4 Size-tab aliases.** `guidance::SIZE_TAB_ALIASES: [(&str, &str); 1] = [("coupling.inner_back_apothem_mm", size_controls::DIAMETER)]`.
  - `fix_text` writes the alias label, giving "Raise Outside diameter" or "Lower Outside diameter".
  - In T8, a fix link on an aliased path selects `CentreView::SizeAndTorque` instead of calling `focus_input`.
- **C-5 `RING_SIZE_STAYS`.** The text becomes "The ring radius stays the same when you change the magnet count; the cup OD below can still change." The const name is unchanged. Spec test 10's expected text follows it.
- **C-6 `NO_PREDICTION`.**
  - The text becomes "no prediction (magnets too short for the end-effect model at this pole pitch)".
  - Add `pub const NO_TORQUE: &str = "no prediction (the torque is not finite for this design)";` in `gui/size_controls.rs`, for a non-finite torque while the end effect is in range.
- **C-7 One search-range helper.** T3's private `MagcouplingPanel::sizing_range()` is the only helper; T7 changes its body. T7 does not add `search_range()`.
- **C-8 The requirement refusal.**
  - `REQUIREMENT_NOT_POSITIVE: &str = "The requirement must be a positive torque."` lives in `gui/sizing.rs` and is added in T3.
  - T3 owns the `InvalidTarget` arm and its test, written as `for required in [-1.0, 0.0]`.
  - T7 drops its duplicate test and its edit of T3's test.
- **C-9 `with_ring_radius`.** Add `pub fn with_ring_radius(inputs: &DesignInputs, apothem_mm: f64) -> Option<DesignInputs>` to `engine/size_controls.rs`, in T1.
  - It applies a ring radius through the Outside diameter slider's own mapping, so the cap OD and cap thread follow the cup with their signed offsets.
  - It returns `None` where that mapping refuses.
  - T7 uses it to build the shown design of a ring-size solve.
- **C-10 Shared fixtures.** `gui/test_support.rs` gains:
  - `pub fn with_poles(npole: i64) -> DesignInputs`
  - `pub fn passing_design() -> DesignInputs`, a design where all 8 guided checks are Good; T4 builds it and documents it.

  T4, T5, T6 and T8 use them instead of their own copies. T1's engine test keeps a local `with_poles`, because integration tests cannot see `gui::test_support`.

### Spec deviations accepted by the controller (recorded in BL-055 and 02-system)

- **D-1 Keyway failures in neighbours.** A neighbour that fails only the keyway clause of `is_valid` still shows its torque. The keyway does not depend on the magnet count, and the Size tab and dashboard already flag it. A neighbour reads in words only when its blocks do not fit ("magnets do not fit") or the end effect is out of range or its torque is not finite ("no prediction"). The docs say "fails fit or the end-effect range".
- **D-2 Units in the mismatch notice.** The notice carries units: "This file's sizing target (X N·m) differed from its requirement (Y N·m); the requirement is used."
- **D-3 Requirement of zero or less.** When the requirement is ≤ 0, save leaves `target_torque_Nm` out. This keeps round trips exact, and load already accepts a file without it.

### T1 amendments

- Add `with_ring_radius` (C-9). Test it:
  - the result's apothem equals the request;
  - `dimensions(..).diameter_mm` equals the diameter the slider mapping gives for that apothem;
  - cap OD minus cup OD is preserved;
  - out-of-range requests return `None`.
- Literal engine numbers (`7.47074935571727`, housing 19.1, 18.8, 16.3) are either computed from the inputs, with housing = `cap_axial_mm + (cup_depth_mm - ring) + web_mm + boss_length_mm` using the code's field names, or carry a one-line derivation comment.

### T2 amendments

- Replace the literal `2.0695` with a computed value or give it a derivation comment.
- T3 replaces the whole `gui::sizing` test module, including T2's `a_refused_search_range_shows_why_and_keeps_the_inputs`. Keep its intent there as `an_empty_or_nan_search_range_is_refused_once`.

### T3 amendments

- C-7 and C-8.
- Step 1: replace the whole `#[cfg(test)] mod tests { ... }` block of `gui/sizing.rs`, not a line range.
- Add `decode_share_payload` to the panel test import: `use crate::gui::session::{decode_share_payload, design_from_json, encode_share_payload, encode_share_text, target_mismatch_notice};`.
- The expected count in step 10 is 13 `gui::sizing` tests.
- D-2 and D-3 are accepted. Record them in the task's decisions list.
- Label `idle_frames_in_torque_to_magnets_keep_the_requirement_as_loaded` as a guard test: it passes before the change by design.

### T4 amendments

- **C-1 to C-4 and C-10.**
- **Why sentences.** Use these, at most 30 words each. The G-3 physics review may still revise them:
  - Hot-low torque: "Magnets weaken as they warm, and production scatter makes some couplings weaker than nominal; hot-low is the pull-out torque after both, and it must reach the requirement."
  - Running clearance: "The sleeve and liner spin past each other across a small radial gap that must absorb runout, deflection, thermal growth and tolerances, or they rub."
  - Cup wall: "The steel ring behind the outer magnets carries each pole's flux to its neighbours; too thin a wall saturates, flux leaks, and torque falls below the prediction."
  - Temperature limits: "Too hot, and the magnets can lose strength for good or the adhesive can weaken; ferrite is the exception, at risk when too cold."
  - Shaft clamp: "The clamp holds by screw preload and friction, and must carry the most torque the coupling can pass, times a safety factor, with room for its screws."
  - Space claim: "The coupling must fit the envelope reserved for it: its diameter, its overall length and the length of the large-diameter bay."
  - End-effect model: "Field fringes at the magnet ends, so the model trims a length set by the pole pitch; magnets that short leave no usable torque prediction."
  - Magnets fit: "Each ring holds one block per pole; at a given ring size, more blocks leave each a narrower place, and a block wider than its place does not fit."
- Add test `every_why_is_one_short_sentence`: each `why` has at most 30 words, ends with a period, and has no further sentence-ending ". ".
- **Fixes:**
  - Hot-low: `[("coupling.magnets.axial_length_mm", Raise)]` only. The gap is removed: it contradicts the clearance fix on the default screen.
  - End effect: `[("coupling.magnets.axial_length_mm", Raise)]` only. `npole` is removed, because at the default ring 12 poles already fail fit.
  - Fit: `[("coupling.inner_back_apothem_mm", Raise), ("coupling.npole", Lower)]`. The apothem is shown as "Outside diameter" (C-4).
  - Space claim: keep the list. Its test moves the cap OD only down to `results.model.cup_od_mm`, not to the slider end.
  - Clamp: keep `clamps.boss_od_mm`. C-3 disambiguates its label. The G-3 review is told the clamp's boss is a separate input from the drawn `metal.boss_od_mm`.
  - Temperature: `Fixes::PerTerm(temperature_fixes)`. Per failing term:
    - slip-fault peak: `temperature.duty.fault_trip_s` Lower, plus `temperature.thermal.conductance_W_K` Raise;
    - hot-day start above a magnet limit: the magnet parts `coupling.magnets.part_inner` and `coupling.magnets.part_outer`, Change, plus conductance Raise;
    - hot-day start above the adhesive limit and the cure margin: the adhesive selection input, Change. Find its real path in `engine/temperature.rs` and its inputs; the critic names `temperature.adhesive.selected`, so verify it.
    - ferrite cold: the magnet parts, Change.

    The implementer verifies each path exists and that each fix moves its term toward passing (test 5.2 below).
- **Problem text:**
  - The ferrite cold term gets the clause "(from stored 3D fields that do not yet follow the magnet grade)".
  - The end-effect problem reads "...the model gives no torque until the magnets are longer than X mm and is rough just above that."
  - The arc fit problem reads "The N inner blocks together need P % of the circle at their mid-radius."
- **Test 5.1, numbers from results.**
  - Define a per-guide table of key result paths.
  - For each guide, take two failing designs with different results.
  - Assert that the two Problem texts differ, and that each contains `format_value(results.get(path))` for its key paths.
- **Test 5.2, fixes help.** For each guide and each fixture that fails it:
  - at least one fix in `fixes(..)` moves every failing term's margin toward passing, by one slider step or one choice change;
  - no fix turns another guided check from Good to Bad on that fixture.
- **`ROW_LEARN`** (spec 3.4: a Why link on every row).
  - Add `pub static ROW_LEARN: [(&'static str, Learn); N]` for every `dashboard::DASHBOARD` row path, and `pub fn row_learn(path: &str) -> Option<Learn>`.
  - A row that has a check uses its guide's `learn`. Value rows get the best reviewed note, assumption or `None(reason)`.
  - Test: every `DASHBOARD` row resolves.
- Every `Learn::Note` must pass `notes::note_for` (reviewed only).

### T5 amendments

- C-5, C-10, D-1.
- **Final merged imports in `guidance.rs`** (no duplicates with T4):
  - `use crate::engine::model::{blocks_fit, end_effect_in_range, pitch_share};`
  - `use crate::gui::format::{fixed, format_value, signed_percent, with_unit};`

  Keep only the ones used; clippy decides.
- **Review Focus 1, the fit line.**
  - Also compute `smallest_fitting_diameter(inputs, n, Some(abs_min))`, where `abs_min` is `dimensions(inputs)?.minimum_diameter_range.min`.
  - When the user-range result is `AtMinimum(d)` and the absolute result is `At(e)` with `e < d - FIT_TOLERANCE_MM`, the line reads "Fits down to {e} mm; lower Minimum OD to reach it."
  - Test this at 8 magnets with Minimum OD unset, against values computed in the test.
- `with_magnets_too_short_for_the_model_every_neighbour_reads_words` asserts the exact form per neighbour, computed in the test from `blocks_fit` and `end_effect_in_range`, not "either form".

### T6 amendments

- C-6, C-10.
- Build the gap text from T5's `fixed` and `signed_percent`, with no inline formatting. Document in one comment why the "short" form keeps a "-0 %".
- Step 8(a): add `slider` back to the `crate::gui::input_ui` import. T7 keeps it.
- Add `decode_share_payload` to the test import if T3's import does not already bring it.
- Push the Required slider's widget id into `design_widgets` before `run_sizing` reads `editing()`, so typing holds the debounce. If the frame order makes that impossible, record it as a known limitation in the task's notes and in the BL-055 notes.
- Owner regression (spec test 10) also asserts the T5 "lower Minimum OD to reach it" line at 8 magnets.

### T7 amendments

- C-7 and C-8: no `search_range`. Use T3's `sizing_range()`.
  - Its body: `FreeVariable::RingRadius => ring_radius_range(&self.inputs, self.minimum_diameter_mm).unwrap_or(variable.range())`.
  - Every other variable: `variable.range()`.
  - The rg check is "only inside `sizing_range`".
- Rename T7's test helper `beside` to `on_slider_row` (T6 owns `beside`), and update its call sites.
- `change_text` uses `crate::gui::format::fixed(v, 1)` for mm and whole numbers for the magnet count.
  - Ring size names Outside diameter, not the radius: "Outside diameter 42.8 -> 40.1 mm", computed with `dimensions(..).diameter_mm` of the inputs and of the shown design.
  - Test literals follow from computed values.
- **Review Focus 2.**
  - When the variable is `RingRadius`, the shown design is `with_ring_radius(&inputs, solved_apothem)`, falling back to the plain `apply` when that returns `None`, so the locked Outside diameter slider shows the solved OD.
  - Test: the locked slider's value equals `dimensions(shown).diameter_mm`, and cap OD minus cup OD is preserved.
- Delete T3's `the_sizing_target_is_the_requirement_s_row_and_one_undo_restores_it`. T6's `size_tab_required_slider_is_an_ordinary_design_edit` covers it; move its share-link and undo assertions there if T6's test lacks them.
- Import anchors after T3 and T6:
  - `use crate::engine::meta::{InputSet, SliderRange, Value};`
  - `use crate::{DesignInputs, DesignResults, compute_all};`
  - step 11.2 is already done by T3.
- Step 11.8 anchor: quote T6's final text of that block before appending.
- Add a test for critic item 2.7: with magnet-count Auto on, edit Outside diameter. Assert the inputs' own apothem is what the typed diameter maps to under the inputs' own count, or document the chosen behaviour in a comment and assert it.

### T8 amendments

- C-1, C-4, C-10; ROW_LEARN from T4. Every dashboard row gets a Why link.
- `the_fit_row_says_whether_the_blocks_fit_and_guides_a_misfit` uses `.expect("a misfit has a Problem")`, not `if let`.
- **Review Focus 5.** Fix `app::tests::the_app_shows_the_panel_as_a_full_page` (`magcoupling-rs/src/app.rs:135`): assert a top-of-dashboard signal (`design_status_text` of the default design) instead of the clamp row. Keep the 1280×800 screen.
- Re-measure the failing-test list after T7 has landed, then fix every one.
- Blocks for Caution rows are collapsed by default under a one-line header; Bad rows are open. This keeps the default dashboard short.

### T9 amendments

- T9 owns the `.claude/workflows/gui-smoke.js` edit (`geometry_view` becomes `first_view`). T10 does not touch it.
- Regenerate the broken-test list after T8 has landed. `Harness::on_screen`, or selecting `CentreView::Geometry` explicitly, fixes each.

### T10 amendments

- Build on T7's README text: steps 8.5 and 8.9 edit T7's wording. The check string is `"Opening the tab changes no mode"`.
- Delete step 9 (gui-smoke). Step 10 expects six files.
- Corrections:
  - 3.9: neighbours "fail fit or the end-effect range" (D-1);
  - the Sizing bullet: "Not reachable: best Y at X";
  - remove "The Torque -> Magnets target ... edit it".
- Add to 03-structure:
  - `engine::temperature::CURE_MARGIN_MIN_C` (if T4 adds it);
  - `ReadoutEvents::focus` and `Readouts::focus_input` (if T8 adds them);
  - `format::fixed` and `format::signed_percent` (T5);
  - `size_controls::with_ring_radius` (C-9).
- BL-055 notes list D-1 to D-3, the known limitations (Required slider debounce if applicable) and C-5's wording.
- The performance note: consequences recompute every drag frame, about 30 `compute_all` calls. This is accepted, and recorded in 02-system limitations.

---


---

### Task 1: Engine read-outs for the Size tab: diameter and length breakdowns, smallest fitting diameter, ring radius range

> **Amendments:** apply "T1 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Modify: `magcoupling-rs/src/engine/size_controls.rs`
  - module docs: insert after line 17
  - imports: line 24
  - new types: before `struct Geometry` (lines 37-46)
  - `geometry()`: line 81, and the struct literal at lines 154-167
  - new functions: after `dimensions_with_minimum_diameter` (ends line 183)
- Test: `magcoupling-rs/tests/size_controls.rs`
  - import block: lines 3-7
  - new helpers and tests: appended after line 406

**Interfaces:**
- Consumes. These are all existing and none of them change:
  - `geometry()` and `resize_diameter_using()`, both private in `size_controls.rs`. This is the diameter slider's own mapping, so there is no second formula.
  - `model::blocks_fit`, `api::compute_all`, `meta::SliderRange`
  - `results.metal.axial_stack_mm`, `results.model.outer_length_mm`, `results.model.cup_od_mm`
- Produces. These follow the contract exactly; only `FIT_TOLERANCE_MM` is new:
  - `pub struct DiameterBreakdown { pub cap_mm: f64, pub cup_mm: f64, pub boss_mm: f64 }`. Derives Clone, Copy, Debug, PartialEq. Its largest field equals `dimensions(..).diameter_mm` exactly, because `geometry()` now computes the outside diameter through the breakdown's private `outside_mm()`.
  - `pub fn diameter_breakdown(inputs: &DesignInputs) -> Option<DiameterBreakdown>`. Returns `None` exactly when `dimensions` does.
  - `pub struct LengthBreakdown { pub magnets_mm: f64, pub housing_mm: f64 }`. Derives Clone, Copy, Debug, PartialEq. `magnets_mm + housing_mm` equals `dimensions(..).overall_length_mm` to rounding (tests use 1e-9).
  - `pub fn length_breakdown(inputs: &DesignInputs) -> Option<LengthBreakdown>`
  - `pub enum DiameterFit { At(f64), AtMinimum(f64), Nowhere(f64) }`. Derives Clone, Copy, Debug, PartialEq.
  - `pub fn smallest_fitting_diameter(inputs: &DesignInputs, npole: i64, minimum_diameter_mm: Option<f64>) -> Option<DiameterFit>`
  - `pub fn ring_radius_range(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Option<SliderRange>`
  - Addition: `pub const FIT_TOLERANCE_MM: f64 = 1e-6;`, the bisection bracket. Tests use it.
- Decisions made in this task:
  1. **`magnets_mm` is the outer ring's length.** The cup cavity holds the outer ring, and `housing::axial_housing` makes the cavity follow it. `housing_mm` is the stack minus that length, not a second sum of cap, web and boss.
  2. **When `housing_mm` would be negative, `length_breakdown` returns `None`.** This happens with a blank override when the cavity is shorter than its ring by more than the cap, web and boss.
  3. **`smallest_fitting_diameter` searches the design with `npole` substituted.** It uses that design's own Size tab range.
  4. **`ring_radius_range` takes both ends from the diameter slider's mapping.** Cap, thread and boss floors therefore narrow the range just as they narrow the diameter slider.

**Verification already done.** Both TDD cycles below were run in a scratch copy of the crate, not in the repo. Each red step failed to compile with the E0432 error quoted. Each green step passed (26, then 32 tests). `cargo fmt --check` and `cargo clippy --all-targets -D warnings` (engine features) were clean. `sizing`, `robustness`, `python_schema` and `schema` stayed green. Measured fit edges, for reference:

| Case | Result |
|---|---|
| 10 magnets, flat, legacy range | `At(42.0043…)` |
| 8 magnets, legacy range | `AtMinimum(40.3105…)` |
| 8 magnets, Minimum OD 37 | `At(37.4208…)` |
| Arcs, 10 magnets, Minimum OD 20 | `At(39.5427…)` |
| `ring_radius_range(default, None)` | 9.0 to 27.8397 (the cap's 80 mm limit binds before the 30 mm radius maximum) |

---

#### Cycle A: the two breakdowns

- [ ] **Step 1: Write the failing tests.** In `magcoupling-rs/tests/size_controls.rs`, replace the import block (lines 3-7):

```rust
use magcoupling::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter, resize_diameter_with_minimum,
    resize_length,
};
use magcoupling::{DesignInputs, compute_all};
```

with:

```rust
use magcoupling::engine::size_controls::{
    DiameterBreakdown, diameter_breakdown, dimensions, dimensions_with_minimum_diameter,
    length_breakdown, resize_diameter, resize_diameter_with_minimum, resize_length,
};
use magcoupling::{DesignInputs, compute_all};
```

Then append at the end of the file:

```rust

/// The default design with `npole` magnets per ring.
fn with_poles(npole: i64) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = npole;
    inputs
}

#[test]
fn diameter_breakdown_names_cap_cup_and_boss_and_its_largest_is_the_outside_diameter() {
    let base = DesignInputs::default();
    let parts = diameter_breakdown(&base).unwrap();
    assert_eq!(
        parts,
        DiameterBreakdown {
            cap_mm: 42.8,
            cup_mm: compute_all(&base).model.cup_od_mm,
            boss_mm: 22.0,
        }
    );
    let largest = |b: DiameterBreakdown| b.cup_mm.max(b.cap_mm).max(b.boss_mm);
    assert_eq!(largest(parts), dimensions(&base).unwrap().diameter_mm);
    assert_eq!(largest(parts), parts.cap_mm, "the cap sets the default OD");

    let mut cup_sets = base.clone();
    cup_sets.metal.cap_od_mm = 35.0;
    cup_sets.metal.cap_thread_dia_mm = 34.0;
    let parts = diameter_breakdown(&cup_sets).unwrap();
    assert_eq!(largest(parts), parts.cup_mm);
    assert_eq!(largest(parts), dimensions(&cup_sets).unwrap().diameter_mm);

    let mut boss_sets = base.clone();
    boss_sets.metal.boss_od_mm = 50.0;
    let parts = diameter_breakdown(&boss_sets).unwrap();
    assert_eq!(largest(parts), parts.boss_mm);
    assert_eq!(largest(parts), dimensions(&boss_sets).unwrap().diameter_mm);
    close(parts.boss_mm, 50.0);
}

#[test]
fn eight_magnets_grow_the_cup_while_the_cap_keeps_the_outside_diameter() {
    // The owner's case (spec 1, T1): 10 -> 8 magnets at the default ring grows the cup, and
    // the outside diameter stays because the cap sets it.
    let ten = diameter_breakdown(&DesignInputs::default()).unwrap();
    let eight = diameter_breakdown(&with_poles(8)).unwrap();
    assert!(
        eight.cup_mm > ten.cup_mm,
        "{} vs {}",
        eight.cup_mm,
        ten.cup_mm
    );
    assert!(
        eight.cup_mm < eight.cap_mm,
        "{} vs {}",
        eight.cup_mm,
        eight.cap_mm
    );
    assert_eq!(eight.cap_mm, ten.cap_mm);
    close(
        dimensions(&with_poles(8)).unwrap().diameter_mm,
        eight.cap_mm,
    );
    close(eight.cap_mm, 42.8);
}

#[test]
fn length_breakdown_is_the_outer_ring_and_the_housing_around_it() {
    let base = DesignInputs::default();
    let parts = length_breakdown(&base).unwrap();
    close(parts.magnets_mm, 12.7);
    close(parts.housing_mm, 19.1);
    close(
        parts.magnets_mm + parts.housing_mm,
        dimensions(&base).unwrap().overall_length_mm,
    );

    // The length slider moves the magnets; the housing keeps its allowances.
    let shorter = resize_length(&base, 29.1).unwrap();
    let parts = length_breakdown(&shorter).unwrap();
    close(parts.magnets_mm, 10.0);
    close(parts.housing_mm, 19.1);
    close(parts.magnets_mm + parts.housing_mm, 29.1);

    // The cup cavity holds the outer ring: a 20 mm manual outer ring counts, not the 12.7 mm
    // inner part.
    let mut mixed = base.clone();
    mixed.coupling.magnets.part_outer.clear();
    mixed.coupling.magnets.manual_outer_length_mm = 20.0;
    mixed.metal.cup_depth_mm = 22.5;
    mixed.metal.retainer_span_mm = 22.0;
    let parts = length_breakdown(&mixed).unwrap();
    close(parts.magnets_mm, 20.0);
    close(parts.housing_mm, 18.8);
    close(
        parts.magnets_mm + parts.housing_mm,
        dimensions(&mixed).unwrap().overall_length_mm,
    );
    let parts = length_breakdown(&resize_length(&mixed, 33.8).unwrap()).unwrap();
    close(parts.magnets_mm, 15.0);
    close(parts.housing_mm, 18.8);

    // At the ring-length floor the cavity has no margin: the housing is the cap, web and boss.
    let mut floored = base.clone();
    floored.metal.hub_length_mm = 5.0;
    floored.metal.cup_depth_mm = 5.0;
    floored.metal.retainer_span_mm = 5.0;
    let parts = length_breakdown(&resize_length(&floored, 25.0).unwrap()).unwrap();
    close(parts.magnets_mm, 8.7);
    close(parts.housing_mm, 16.3);
}

#[test]
fn a_housing_shorter_than_its_outer_ring_has_no_length_breakdown() {
    // Blank override, a 40 mm manual outer ring in the default 15.5 mm cavity: the stack
    // (31.8 mm) is shorter than the ring, so no split describes it. The size controls stay.
    let mut short_housing = DesignInputs::default();
    short_housing.coupling.magnets.part_outer.clear();
    short_housing.coupling.magnets.manual_outer_length_mm = 40.0;
    close(dimensions(&short_housing).unwrap().overall_length_mm, 31.8);
    assert_eq!(length_breakdown(&short_housing), None);
    // With the override set the cavity follows the ring (decision A2-8): a split again.
    short_housing.coupling.magnets.axial_length_mm = Some(40.0);
    let parts = length_breakdown(&short_housing).unwrap();
    close(parts.magnets_mm, 40.0);
    close(parts.housing_mm, 16.3);
}

#[test]
fn breakdowns_are_none_exactly_where_the_size_controls_are() {
    let mut invalid = DesignInputs::default();
    invalid.metal.face_gap_mm = f64::NAN;
    assert_eq!(dimensions(&invalid), None);
    assert_eq!(diameter_breakdown(&invalid), None);
    assert_eq!(length_breakdown(&invalid), None);
    for count in [i64::MIN, -4, 0, 1, 2, 3, 9, 41, 42, i64::MAX] {
        let odd = with_poles(count);
        assert_eq!(dimensions(&odd), None, "count {count}");
        assert_eq!(diameter_breakdown(&odd), None, "count {count}");
        assert_eq!(length_breakdown(&odd), None, "count {count}");
    }
}
```

- [ ] **Step 2: Run the tests and confirm they fail.** `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls`. Expected: the test binary does not compile:

```
error[E0432]: unresolved imports `magcoupling::engine::size_controls::DiameterBreakdown`, `magcoupling::engine::size_controls::diameter_breakdown`, `magcoupling::engine::size_controls::length_breakdown`
```

- [ ] **Step 3: Implement the breakdowns.** Five exact edits to `magcoupling-rs/src/engine/size_controls.rs`.

(a) Module docs. Replace:

```rust
//! limits and the fixed boss. Changing that minimum only changes the available range.
```

with:

```rust
//! limits and the fixed boss. Changing that minimum only changes the available range.
//!
//! Read-outs for the Size tab come from the same geometry: what sets the outside diameter
//! ([`diameter_breakdown`]: the largest of the cup, the cap and the boss) and how the overall
//! length divides ([`length_breakdown`]: the outer ring the cup cavity holds, and the housing
//! around it).
```

(b) Types and `Geometry` fields. Replace:

```rust
/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
struct Geometry {
    controls: SizeControls,
```

with:

```rust
/// What sets the outside diameter [mm]: the cup (from the magnet layout), the threaded cap
/// and the shaft boss. The outside diameter is the largest of the three.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiameterBreakdown {
    pub cap_mm: f64,
    pub cup_mm: f64,
    pub boss_mm: f64,
}

impl DiameterBreakdown {
    /// The outside diameter: the largest of the three (one expression for every caller).
    fn outside_mm(&self) -> f64 {
        self.cup_mm.max(self.cap_mm).max(self.boss_mm)
    }
}

/// How the overall axial stack divides [mm]: the outer ring's magnets, which the cup cavity
/// holds, and the housing around them (the cap, the cavity's margin, the web and the boss).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LengthBreakdown {
    pub magnets_mm: f64,
    pub housing_mm: f64,
}

/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
struct Geometry {
    controls: SizeControls,
    diameter: DiameterBreakdown,
    /// The outer ring's axial length: the ring the cup cavity holds ([`axial_housing`]).
    magnets_length_mm: f64,
```

(c) In `geometry()`, replace this line. It keeps the same expression and order, so the result is bit for bit the same:

```rust
    let outside = cup.max(md.cap_od_mm).max(md.boss_od_mm);
```

with:

```rust
    let diameter = DiameterBreakdown {
        cap_mm: md.cap_od_mm,
        cup_mm: cup,
        boss_mm: md.boss_od_mm,
    };
    let outside = diameter.outside_mm();
```

(d) In the `Some(Geometry { .. })` literal at the end of `geometry()`, replace:

```rust
            length_range,
        },
        cup_od_mm: cup,
```

with:

```rust
            length_range,
        },
        diameter,
        magnets_length_mm: results.model.outer_length_mm,
        cup_od_mm: cup,
```

(e) After `dimensions_with_minimum_diameter`, replace:

```rust
    geometry(inputs, Some(minimum_od_mm)).map(|geometry| geometry.controls)
}
```

with:

```rust
    geometry(inputs, Some(minimum_od_mm)).map(|geometry| geometry.controls)
}

/// What sets the current outside diameter: its largest part is [`dimensions`]'s
/// `diameter_mm`. `None` exactly when [`dimensions`] is `None`.
pub fn diameter_breakdown(inputs: &DesignInputs) -> Option<DiameterBreakdown> {
    geometry(inputs, None).map(|geometry| geometry.diameter)
}

/// How the current overall length divides: the parts sum to [`dimensions`]'s
/// `overall_length_mm` (to rounding). `None` when [`dimensions`] is `None`, or when a blank
/// axial length override leaves the housing shorter than its outer ring, which no split into
/// magnets and housing describes.
pub fn length_breakdown(inputs: &DesignInputs) -> Option<LengthBreakdown> {
    let geometry = geometry(inputs, None)?;
    let magnets_mm = geometry.magnets_length_mm;
    let housing_mm = geometry.controls.overall_length_mm - magnets_mm;
    (magnets_mm.is_finite() && magnets_mm > 0.0 && housing_mm >= 0.0).then_some(LengthBreakdown {
        magnets_mm,
        housing_mm,
    })
}
```

- [ ] **Step 4: Run the tests and confirm they pass.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls`. Expected: `test result: ok. 26 passed; 0 failed` (the 21 existing tests plus 5 new).
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`. Expected: `Finished`, no warnings.

- [ ] **Step 5: Commit.**

```bash
git add magcoupling-rs/src/engine/size_controls.rs magcoupling-rs/tests/size_controls.rs
git commit -F - <<'EOF'
feat(magcoupler): engine breakdowns of the outside diameter and overall length

diameter_breakdown names the cup, cap and boss (the outside diameter is the
largest of the three); length_breakdown splits the axial stack into the outer
ring the cup cavity holds and the housing around it, and is None when a blank
override leaves the housing shorter than that ring. Both read the size
controls' own geometry, so they agree with dimensions() by construction.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```

---

#### Cycle B: the smallest fitting diameter and the ring radius range

- [ ] **Step 6: Write the failing tests.** In `magcoupling-rs/tests/size_controls.rs`, replace the import block written in Step 1 with:

```rust
use magcoupling::engine::model::blocks_fit;
use magcoupling::engine::size_controls::{
    DiameterBreakdown, DiameterFit, FIT_TOLERANCE_MM, diameter_breakdown, dimensions,
    dimensions_with_minimum_diameter, length_breakdown, resize_diameter,
    resize_diameter_with_minimum, resize_length, ring_radius_range, smallest_fitting_diameter,
};
use magcoupling::engine::sizing::FreeVariable;
use magcoupling::{DesignInputs, compute_all};
```

Then append at the end of the file:

```rust

/// `base` with true arcs instead of flat blocks.
fn as_arcs(mut base: DesignInputs) -> DesignInputs {
    base.coupling.faceted = 0;
    base
}

/// `base` with a manual outer ring of 10 mm wide blocks, so the outer ring sets the fit.
fn with_wide_outer_blocks(mut base: DesignInputs) -> DesignInputs {
    base.coupling.magnets.part_outer.clear();
    base.coupling.magnets.manual_outer_width_mm = 10.0;
    base
}

/// Whether both rings' blocks fit at an outside diameter, mapped to the inputs by the
/// diameter slider's own resize (with the Minimum OD setting when there is one).
fn fits_at(design: &DesignInputs, diameter_mm: f64, minimum_mm: Option<f64>) -> bool {
    let resized = match minimum_mm {
        Some(minimum) => resize_diameter_with_minimum(design, diameter_mm, minimum),
        None => resize_diameter(design, diameter_mm),
    }
    .expect("the diameter lies inside the Size tab's range");
    blocks_fit(&resized.coupling, &compute_all(&resized).model)
}

/// The diameter of an `At` fit, or a panic naming the other form.
fn fit_edge(fit: Option<DiameterFit>) -> f64 {
    match fit {
        Some(DiameterFit::At(diameter_mm)) => diameter_mm,
        other => panic!("expected a fit inside the range, got {other:?}"),
    }
}

#[test]
fn smallest_fitting_diameter_fits_at_the_result_and_not_a_thousandth_below() {
    // Flat blocks, 10 magnets, the legacy range: the inner flats bind inside the range.
    let base = DesignInputs::default();
    let edge = fit_edge(smallest_fitting_diameter(&base, 10, None));
    let range = dimensions(&base).unwrap().diameter_range;
    assert!(range.min < edge - 1e-3 && edge < range.max, "{edge}");
    assert!(fits_at(&base, edge, None));
    assert!(!fits_at(&base, edge - 1e-3, None));

    // The owner's 8 magnets with a lowered Minimum OD: between the existing 37 mm (too
    // narrow) and 38 mm (fits) cases.
    let edge = fit_edge(smallest_fitting_diameter(&base, 8, Some(37.0)));
    assert!(37.0 < edge && edge < 38.0, "{edge}");
    let eight = with_poles(8);
    assert!(fits_at(&eight, edge, Some(37.0)));
    assert!(!fits_at(&eight, edge - 1e-3, Some(37.0)));

    // Arcs, 10 magnets, a lowered Minimum OD: the inner arcs' pitch share binds.
    let arcs = as_arcs(base.clone());
    let edge = fit_edge(smallest_fitting_diameter(&arcs, 10, Some(20.0)));
    assert!(fits_at(&arcs, edge, Some(20.0)));
    assert!(!fits_at(&arcs, edge - 1e-3, Some(20.0)));

    // Wide outer blocks: the outer ring binds, for both shapes.
    for wide in [
        with_wide_outer_blocks(base.clone()),
        as_arcs(with_wide_outer_blocks(base.clone())),
    ] {
        let edge = fit_edge(smallest_fitting_diameter(&wide, 10, Some(20.0)));
        assert!(fits_at(&wide, edge, Some(20.0)));
        assert!(!fits_at(&wide, edge - 1e-3, Some(20.0)));
    }
}

#[test]
fn smallest_fitting_diameter_reports_both_range_ends() {
    let base = DesignInputs::default();
    // Four magnets fit at the smallest diameter, legacy or lowered.
    let four = with_poles(4);
    assert_eq!(
        smallest_fitting_diameter(&base, 4, None),
        Some(DiameterFit::AtMinimum(
            dimensions(&four).unwrap().diameter_range.min
        ))
    );
    assert_eq!(
        smallest_fitting_diameter(&base, 4, Some(20.0)),
        Some(DiameterFit::AtMinimum(
            dimensions_with_minimum_diameter(&four, 20.0)
                .unwrap()
                .diameter_range
                .min
        ))
    );
    // Arcs at 10 magnets fit from the legacy 9 mm radius floor up.
    let arcs = as_arcs(base.clone());
    let legacy = dimensions(&arcs).unwrap().diameter_range;
    assert_eq!(
        smallest_fitting_diameter(&arcs, 10, None),
        Some(DiameterFit::AtMinimum(legacy.min))
    );
    assert!(fits_at(&arcs, legacy.min, None));
    // A Minimum OD above the fit's edge is the answer as it is.
    assert_eq!(
        smallest_fitting_diameter(&base, 10, Some(50.0)),
        Some(DiameterFit::AtMinimum(50.0))
    );
    // A Minimum OD at the top of the range leaves one diameter: either form, never a search.
    assert_eq!(
        smallest_fitting_diameter(&base, 10, Some(200.0)),
        Some(DiameterFit::AtMinimum(80.0))
    );
    assert_eq!(
        smallest_fitting_diameter(&base, 40, Some(200.0)),
        Some(DiameterFit::Nowhere(80.0))
    );
    // Forty magnets fit nowhere up to the largest diameter, for both shapes.
    for design in [base.clone(), as_arcs(base.clone())] {
        let mut forty = design.clone();
        forty.coupling.npole = 40;
        let largest = dimensions(&forty).unwrap().diameter_range.max;
        assert_eq!(
            smallest_fitting_diameter(&design, 40, None),
            Some(DiameterFit::Nowhere(largest))
        );
        assert!(!fits_at(&forty, largest, None));
    }
}

#[test]
fn the_magnet_count_argument_replaces_the_designs_own() {
    let ten = DesignInputs::default();
    let eight = with_poles(8);
    assert_eq!(
        smallest_fitting_diameter(&ten, 8, Some(37.0)),
        smallest_fitting_diameter(&eight, 8, Some(37.0))
    );
    assert_eq!(
        smallest_fitting_diameter(&eight, 10, None),
        smallest_fitting_diameter(&ten, 10, None)
    );
    assert_ne!(
        smallest_fitting_diameter(&ten, 8, Some(37.0)),
        smallest_fitting_diameter(&ten, 10, Some(37.0))
    );
}

#[test]
fn blocks_fit_never_stops_as_the_diameter_grows_at_a_fixed_magnet_count() {
    // The basis of the bisection (module docs): over the widest diameter range (the lowest
    // Minimum OD) of every magnet count, flat blocks and arcs, the default blocks and a wide
    // outer ring, a design that fits keeps fitting as the diameter grows. Each sweep also
    // pins smallest_fitting_diameter to the samples on either side of its answer.
    const SAMPLES: usize = 48;
    let lowest = 0.1;
    let base = DesignInputs::default();
    for (blocks_name, blocks) in [
        ("default blocks", base.clone()),
        ("wide outer blocks", with_wide_outer_blocks(base.clone())),
    ] {
        for (shape_name, shape) in [("flat blocks", 1), ("arcs", 0)] {
            // How often each form was the answer: at the minimum, inside the range, nowhere.
            let mut forms = [0; 3];
            for npole in (4..=40).step_by(2) {
                let mut design = blocks.clone();
                design.coupling.faceted = shape;
                design.coupling.npole = npole;
                let case = format!("{blocks_name}, {shape_name}, {npole} magnets");
                let range = dimensions_with_minimum_diameter(&design, lowest)
                    .unwrap()
                    .diameter_range;
                let diameters: Vec<f64> = (0..=SAMPLES)
                    .map(|i| {
                        if i == SAMPLES {
                            range.max
                        } else {
                            range.min + (range.max - range.min) * i as f64 / SAMPLES as f64
                        }
                    })
                    .collect();
                let fits: Vec<bool> = diameters
                    .iter()
                    .map(|&diameter| fits_at(&design, diameter, Some(lowest)))
                    .collect();
                assert!(
                    fits.windows(2).all(|pair| pair[0] <= pair[1]),
                    "{case}: {fits:?}"
                );
                match smallest_fitting_diameter(&design, npole, Some(lowest)).unwrap() {
                    DiameterFit::AtMinimum(diameter) => {
                        forms[0] += 1;
                        assert_eq!(diameter, range.min, "{case}");
                        assert!(fits.iter().all(|&fit| fit), "{case}");
                    }
                    DiameterFit::At(edge) => {
                        forms[1] += 1;
                        for (&diameter, &fit) in diameters.iter().zip(&fits) {
                            if diameter >= edge {
                                assert!(fit, "{case}: {diameter} at or above {edge}");
                            } else if diameter < edge - FIT_TOLERANCE_MM {
                                assert!(!fit, "{case}: {diameter} below {edge}");
                            }
                        }
                    }
                    DiameterFit::Nowhere(diameter) => {
                        forms[2] += 1;
                        assert_eq!(diameter, range.max, "{case}");
                        assert!(fits.iter().all(|&fit| !fit), "{case}");
                    }
                }
            }
            // Not vacuous: every form occurs, so the sweeps cross fit edges.
            assert!(
                forms.iter().all(|&count| count > 0),
                "{blocks_name}, {shape_name}: {forms:?}"
            );
        }
    }
}

#[test]
fn ring_radius_range_spans_the_size_tabs_diameter_range() {
    let base = DesignInputs::default();
    let radius = FreeVariable::RingRadius.range();
    let diameters = dimensions(&base).unwrap().diameter_range;
    let ring = ring_radius_range(&base, None).unwrap();
    close(ring.min, 9.0);
    close(
        ring.min,
        resize_diameter(&base, diameters.min)
            .unwrap()
            .coupling
            .inner_back_apothem_mm,
    );
    assert_eq!(
        ring.max,
        resize_diameter(&base, diameters.max)
            .unwrap()
            .coupling
            .inner_back_apothem_mm
    );
    assert!(
        ring.max < radius.max,
        "the cap's 80 mm limit binds first: {}",
        ring.max
    );
    assert_eq!((ring.step, ring.log), (radius.step, radius.log));
    // A Minimum OD at the top of the range leaves one radius.
    let top = ring_radius_range(&base, Some(200.0)).unwrap();
    assert_eq!(top.min, top.max);
    assert_eq!(top.max, ring.max);

    // A lowered Minimum OD reaches below the radius slider's 9 mm floor (8 magnets at 37 mm).
    let ring = ring_radius_range(&with_poles(8), Some(37.0)).unwrap();
    close(ring.min, 7.47074935571727);
    assert!(ring.min < radius.min);

    // A boss larger than the legacy floor raises the smallest radius with the diameter.
    let mut boss = base.clone();
    boss.metal.boss_od_mm = 50.0;
    let ring = ring_radius_range(&boss, None).unwrap();
    close(
        ring.min,
        resize_diameter(&boss, 50.0)
            .unwrap()
            .coupling
            .inner_back_apothem_mm,
    );
    assert!(ring.min > radius.min, "{}", ring.min);
}

#[test]
fn fit_and_ring_range_are_none_where_the_size_controls_are() {
    let base = DesignInputs::default();
    let mut invalid = base.clone();
    invalid.metal.face_gap_mm = f64::NAN;
    assert_eq!(smallest_fitting_diameter(&invalid, 10, None), None);
    assert_eq!(ring_radius_range(&invalid, None), None);
    for count in [i64::MIN, -4, 0, 1, 2, 3, 9, 41, 42, i64::MAX] {
        assert_eq!(
            ring_radius_range(&with_poles(count), None),
            None,
            "count {count}"
        );
        assert_eq!(
            smallest_fitting_diameter(&base, count, None),
            None,
            "count {count}"
        );
        assert_eq!(
            smallest_fitting_diameter(&base, count, Some(37.0)),
            None,
            "count {count}"
        );
    }
    for minimum in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert_eq!(smallest_fitting_diameter(&base, 10, Some(minimum)), None);
        assert_eq!(ring_radius_range(&base, Some(minimum)), None);
    }
}
```

- [ ] **Step 7: Run the tests and confirm they fail.** `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls`. Expected: the test binary does not compile:

```
error[E0432]: unresolved imports `magcoupling::engine::size_controls::DiameterFit`, `magcoupling::engine::size_controls::FIT_TOLERANCE_MM`, `magcoupling::engine::size_controls::ring_radius_range`, `magcoupling::engine::size_controls::smallest_fitting_diameter`
```

- [ ] **Step 8: Implement the search and the range.** Four exact edits to `magcoupling-rs/src/engine/size_controls.rs`.

(a) Module docs. Replace:

```rust
//! length divides ([`length_breakdown`]: the outer ring the cup cavity holds, and the housing
//! around it).
```

with:

```rust
//! length divides ([`length_breakdown`]: the outer ring the cup cavity holds, and the housing
//! around it).
//!
//! [`smallest_fitting_diameter`] finds the smallest outside diameter at which a magnet count's
//! blocks fit, and [`ring_radius_range`] the ring radii the diameter range spans (the ring-size
//! search of inverse sizing). At a fixed magnet count the blocks only fit better as the ring
//! grows: a flat block's polygon flat widens with its apothem, and an arc's share of its pole
//! pitch falls. The fit is bisected on that basis; tests/size_controls.rs sweeps it for both
//! shapes.
```

(b) Imports. Replace:

```rust
use super::meta::{SliderRange, input_rows};
```

with:

```rust
use super::meta::{SliderRange, input_rows};
use super::model::blocks_fit;
```

(c) The fit type and tolerance, and `Copy` on `Geometry` so the closures can pass it by value. Replace:

```rust
/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
struct Geometry {
```

with:

```rust
/// Where a magnet count's blocks start to fit over the Size tab's diameter range [mm].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DiameterFit {
    /// They fit from this diameter up; [`FIT_TOLERANCE_MM`] below it they do not.
    At(f64),
    /// They fit at the range minimum (this value), so at every diameter in the range.
    AtMinimum(f64),
    /// They fit at no diameter in the range; the value is its maximum.
    Nowhere(f64),
}

/// The bisection of [`smallest_fitting_diameter`] stops when its bracket is this narrow [mm].
pub const FIT_TOLERANCE_MM: f64 = 1e-6;

/// The engine geometry and metadata used for a single edit. Every candidate starts from
/// the supplied design; moving between sizes never overwrites its axial housing allowances.
#[derive(Clone, Copy)]
struct Geometry {
```

(d) After `length_breakdown`, replace:

```rust
    (magnets_mm.is_finite() && magnets_mm > 0.0 && housing_mm >= 0.0).then_some(LengthBreakdown {
        magnets_mm,
        housing_mm,
    })
}
```

with:

```rust
    (magnets_mm.is_finite() && magnets_mm > 0.0 && housing_mm >= 0.0).then_some(LengthBreakdown {
        magnets_mm,
        housing_mm,
    })
}

/// The smallest outside diameter at which `npole` blocks per ring fit ([`blocks_fit`]), every
/// other input kept. It searches the Size tab's diameter range of the design with `npole`
/// magnets: [`dimensions`] when `minimum_diameter_mm` is `None`, else
/// [`dimensions_with_minimum_diameter`]. Each diameter maps to the inputs exactly as the
/// diameter slider maps it. The fit only improves as the ring grows (module docs), so a
/// bisection to [`FIT_TOLERANCE_MM`] finds its edge. `None` when that design has no diameter
/// range: invalid inputs, an odd or out-of-range count, or an invalid minimum.
pub fn smallest_fitting_diameter(
    inputs: &DesignInputs,
    npole: i64,
    minimum_diameter_mm: Option<f64>,
) -> Option<DiameterFit> {
    let mut design = inputs.clone();
    design.coupling.npole = npole;
    let geometry = geometry(&design, minimum_diameter_mm)?;
    let range = geometry.controls.diameter_range;
    // Every diameter tried lies inside the range, so the resize is never refused.
    let fits = |diameter_mm: f64| {
        resize_diameter_using(&design, diameter_mm, geometry)
            .is_ok_and(|resized| blocks_fit(&resized.coupling, &compute_all(&resized).model))
    };
    if fits(range.min) {
        return Some(DiameterFit::AtMinimum(range.min));
    }
    if !fits(range.max) {
        return Some(DiameterFit::Nowhere(range.max));
    }
    // `low` never fits and `high` always does.
    let (mut low, mut high) = (range.min, range.max);
    while high - low > FIT_TOLERANCE_MM {
        let middle = low + (high - low) / 2.0;
        if middle <= low || middle >= high {
            break; // adjacent doubles
        }
        if fits(middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some(DiameterFit::At(high))
}

/// The ring radii (`coupling.inner_back_apothem_mm`) at the ends of the Size tab's diameter
/// range, each mapped as the diameter slider maps it: [`dimensions`] when
/// `minimum_diameter_mm` is `None`, else [`dimensions_with_minimum_diameter`], so a lowered
/// minimum reaches below the radius slider's floor. The step and scale are the radius
/// input's. `None` when the design has no diameter range.
pub fn ring_radius_range(
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
) -> Option<SliderRange> {
    let geometry = geometry(inputs, minimum_diameter_mm)?;
    let diameters = geometry.controls.diameter_range;
    let radius_at = |diameter_mm: f64| {
        resize_diameter_using(inputs, diameter_mm, geometry)
            .ok()
            .map(|resized| resized.coupling.inner_back_apothem_mm)
    };
    let (min, max) = (radius_at(diameters.min)?, radius_at(diameters.max)?);
    (min.is_finite() && max.is_finite() && min <= max).then_some(SliderRange {
        min,
        max,
        ..geometry.radius_range
    })
}
```

- [ ] **Step 9: Run the tests and confirm they pass.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls`. Expected: `test result: ok. 32 passed; 0 failed`. The sweep takes about 1 s in a debug build.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls blocks_fit_never_stops_as_the_diameter_grows_at_a_fixed_magnet_count`. Expected: `1 passed`.

- [ ] **Step 10: Run the wider suite for the touched modules.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml`. Expected: every test binary `ok`. That includes `sizing`, `robustness`, `python_schema` and `schema`; no engine label, help or verdict string changed.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib size`. Expected: the panel's Size tab tests pass (`size_and_torque_has_direct_dimension_and_gap_controls`, `manual_size_*`, `loaded_smaller_designs_keep_their_size_and_expose_a_stable_minimum`); `gui/size_controls.rs` calls only unchanged APIs.
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`. Expected: `Finished`, no warnings.
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`. Expected: no output. The code above is already rustfmt-formatted.

- [ ] **Step 11: Commit.**

```bash
git add magcoupling-rs/src/engine/size_controls.rs magcoupling-rs/tests/size_controls.rs
git commit -F - <<'EOF'
feat(magcoupler): smallest fitting diameter and the ring radius range

smallest_fitting_diameter bisects blocks_fit to 1e-6 mm over the Size tab's
diameter range (with or without the Minimum OD) for any even magnet count,
reporting At, AtMinimum or Nowhere; ring_radius_range maps that diameter range
to the inner back apothem through the diameter slider's own mapping, for the
ring-size search. A sweep over every count, flat blocks and arcs, default and
wide outer blocks, pins the monotone fit the bisection relies on.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```

---

**Edge cases from spec section 5 and section 6, item 6, with their tests:**

| Edge case | Test |
|---|---|
| Fits at the result, not 1e-3 mm below; flat blocks, arcs, and the outer ring binding | `smallest_fitting_diameter_fits_at_the_result_and_not_a_thousandth_below` |
| Both range-end forms: `AtMinimum` (legacy, lowered, Minimum OD above the edge, single-point range) and `Nowhere` (40 magnets, both shapes, single-point range) | `smallest_fitting_diameter_reports_both_range_ends` |
| The fit is monotone in the diameter at a fixed magnet count (every count, both shapes, two block sets), and the bisection agrees with the sweep | `blocks_fit_never_stops_as_the_diameter_grows_at_a_fixed_magnet_count` |
| Ring size below the old 9 mm floor with a lowered Minimum OD (the input for Task 7's test 8) | `ring_radius_range_spans_the_size_tabs_diameter_range` (7.4707 mm) |
| Owner's case: at 8 magnets the cup grows and the cap still sets the outside diameter (the engine side of spec test 10) | `eight_magnets_grow_the_cup_while_the_cap_keeps_the_outside_diameter` |
| Invalid inputs, odd or out-of-range counts, invalid minimums | `breakdowns_are_none_exactly_where_the_size_controls_are`, `fit_and_ring_range_are_none_where_the_size_controls_are` |
| Housing shorter than its outer ring | `a_housing_shorter_than_its_outer_ring_has_no_length_breakdown` |

**Existing tests this change breaks:** none, so no edits are needed. All 21 existing tests in `tests/size_controls.rs` pass unchanged. The outside diameter is still computed as `cup.max(cap).max(boss)`, in the same order. The only other change to a private item is `#[derive(Clone, Copy)]` on `Geometry`.

**Docs:** Task 10 covers them under the contract. It should update:
- the `size_controls:` entry in `docs/ai/03-structure.yaml` (line 51), to name the four new functions and `FIT_TOLERANCE_MM`;
- the `tests:` entry in the same file (line 72), to say that `size_controls.rs` now has the breakdowns, the fit search and the monotone sweep.

**Notes for later tasks:**
1. **Task 5, display of `DiameterFit::At(d)`:** `d` is up to 1e-6 mm above the true edge. Printing it with `{:.1}` can round down to a diameter that does not fit (37.4208 shows as "37.4", and 37.4 does not fit). Round the shown value up to the displayed precision.
2. **Task 7, ring-size Auto:**
   - `FreeVariable::RingRadius.apply` moves only `inner_back_apothem_mm`. The Size tab's diameter slider (`resize_diameter_*`) also moves `cap_od_mm` and `cap_thread_dia_mm` by the cup's change.
   - So a ring-size Auto solution searched over `ring_radius_range` keeps the old cap. The outside diameter therefore cannot fall below the cap (42.8 mm at the defaults), even when the radius drops below 9 mm.
   - The owner should decide whether a solved radius is applied through the diameter mapping instead.
3. **Task 5, cost:** `smallest_fitting_diameter` costs one `geometry()` (`compute_all` plus `input_rows`) and about 27 `compute_all` calls. Only call it through `ConsequenceCache`.
4. **Task 5/6, length line wording:** `LengthBreakdown::magnets_mm` is the outer ring's length. With a blank override and different rings, it differs from the torque's active length (the shorter ring). When `length_breakdown` returns `None`, the length line should be omitted.

---

### Task 2: Inverse sizing over a caller's search range (`grid_in`, `solve_in`, `SizingError::InvalidRange`)

> **Amendments:** apply "T2 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Modify: `magcoupling-rs/src/engine/sizing.rs`. These parts change:
  - module docs at lines 10-13, 26-29 and 31-36;
  - `FreeVariable::grid` at lines 99-118;
  - the `SizingOutcome::Solved` doc at line 147;
  - `SizingError` at lines 155-162;
  - `solve` at lines 175-186;
  - the grid call at line 202.
- Modify: `magcoupling-rs/src/gui/sizing.rs`. The `SizingRunner::status` match is at lines 235-258, and the new arm goes after the `InvalidInputs` arm at lines 253-256. One test goes at the end of `mod tests`, after `a_refused_solve_shows_why_and_keeps_the_inputs`, which ends at line 480.
- Test: `magcoupling-rs/tests/sizing.rs`. The imports are at lines 10-15. The new `span` helper goes before `result_bits` at line 37. Nine tests go after `the_search_constants_are_the_documented_ones`, which ends at line 470, and before `the_default_design_is_inside_its_space_claim`.

**Interfaces:**
- Consumes:
  - `crate::engine::meta::SliderRange { min, max, step, log }`, which `engine/sizing.rs` already imports.
  - `FreeVariable::range()`, `FreeVariable::apply`, `is_valid`, and the private `Search` (unchanged).
- Produces (from the contract):
  - `impl FreeVariable { pub fn grid_in(self, range: SliderRange) -> Vec<f64> }`, where `grid() == grid_in(self.range())`.
  - `pub fn solve_in(inputs: &DesignInputs, variable: FreeVariable, target_Nm: f64, range: SliderRange) -> Result<SizingOutcome, SizingError>`.
  - `solve(inputs, variable, target) == solve_in(inputs, variable, target, variable.range())`. The results are identical and the existing tests do not change.
  - A new variant `SizingError::InvalidRange { min: f64, max: f64 }`, returned when `!(min.is_finite() && max.is_finite() && min <= max)`.
  - In `gui/sizing.rs`, the status text for that variant is `"Sizing refused: the search range is empty"`.

**Design choices in this task (each has a test):**
- **Refusal order:** first the target, then the range, then the inputs. `solve` always passes a valid slider range, so its behaviour does not change.
- **One check, two users:** a private `is_search_range` holds the range check. `solve_in` uses it to refuse a range, and `grid_in` uses it to return an empty grid for that range.
- **Magnets per ring:** the grid uses `min.ceil()..=max.floor()`. The old `min as i64` truncation would start a 4.5 range at 4, which is outside the range. For the slider range 4..40 the grid is unchanged.
- **Range of one value:** the grid is `dedup`'d, so `min == max` evaluates that value once instead of 65 times. This case happens when the Size tab's minimum OD is clamped to its maximum. For the slider ranges, whose cell ends are all distinct, nothing changes.
- **Ignored range fields:** `range.step` and `range.log` are not used, because the scan is linear in equal cells.
- **Do not commit between Step 3 and Step 7.** Adding the `SizingError` variant makes the `--features app` build fail with E0004 until Step 7 adds the status arm. The engine-only tests in Step 4 still compile and pass.

---

- [ ] **Step 1: Write the failing engine tests in `magcoupling-rs/tests/sizing.rs`**

Replace the imports at lines 10-15:
```rust
use magcoupling::engine::meta::{Value, result_rows};
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
    is_valid, solve,
};
```
with:
```rust
use magcoupling::engine::meta::{SliderRange, Value, result_rows};
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
    is_valid, solve, solve_in,
};
use std::f64::consts::PI;
```

Insert this helper directly before `/// Every result of a design as (path, bit pattern), so NaN equals NaN and nothing is rounded.`:
```rust
/// `range` with its ends replaced (its step and scale kept).
fn span(range: SliderRange, min: f64, max: f64) -> SliderRange {
    SliderRange { min, max, ..range }
}

```

Insert these tests directly before `#[test]\nfn the_default_design_is_inside_its_space_claim() {`:
```rust
#[test]
fn the_grid_is_the_grid_in_the_slider_range() {
    // Spec 4.3: solve is solve_in over the variable's slider range, so its grid is grid_in of
    // that range, value for value.
    for variable in FreeVariable::ALL {
        assert_eq!(
            variable.grid(),
            variable.grid_in(variable.range()),
            "{variable:?}"
        );
    }
}

#[test]
fn a_caller_range_is_scanned_inside_its_ends() {
    // A continuous variable: SCAN_CELLS + 1 ascending ends of equal cells from the range's
    // minimum to its maximum, below the ring radius slider's 9 mm floor too.
    let range = span(FreeVariable::RingRadius.range(), 6.0, 30.0);
    let grid = FreeVariable::RingRadius.grid_in(range);
    assert_eq!(grid.len(), SCAN_CELLS + 1);
    assert_eq!((grid[0], grid[SCAN_CELLS]), (6.0, 30.0));
    assert!(grid.windows(2).all(|w| w[0] < w[1]), "{grid:?}");
    assert!((grid[1] - grid[0] - 24.0 / SCAN_CELLS as f64).abs() < 1e-12);
    // Magnets per ring: the even whole numbers inside the range, never one below its minimum
    // (4.5 must not start the grid at 4).
    let poles = FreeVariable::MagnetsPerRing.range();
    assert_eq!(
        FreeVariable::MagnetsPerRing.grid_in(span(poles, 4.5, 13.0)),
        [6.0, 8.0, 10.0, 12.0]
    );
    assert_eq!(
        FreeVariable::MagnetsPerRing.grid_in(span(poles, 6.0, 6.0)),
        [6.0]
    );
    assert!(
        FreeVariable::MagnetsPerRing
            .grid_in(span(poles, 5.0, 5.9))
            .is_empty()
    );
}

#[test]
fn a_range_of_one_value_evaluates_that_value_once() {
    // The Size tab's range closes to one value when the minimum outside diameter is set to its
    // maximum: the grid holds that value once, and the solve meets the target there or reports
    // it as the best.
    let radius = span(FreeVariable::RingRadius.range(), 12.5, 12.5);
    assert_eq!(FreeVariable::RingRadius.grid_in(radius), [12.5]);
    let base = DesignInputs::default();
    let torque = hot_low(&FreeVariable::RingRadius.apply(&base, 12.5));
    let p = solved(solve_in(&base, FreeVariable::RingRadius, torque, radius));
    assert_eq!((p.value, p.torque_hot_low_Nm), (12.5, torque));
    match solve_in(&base, FreeVariable::RingRadius, 2.0 * torque, radius) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => {
            assert_eq!((best.value, best.torque_hot_low_Nm), (12.5, torque))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_range_with_no_valid_value_has_no_best_value() {
    // Spec 5, "no value valid": the 10 flats of the default design fit only from 9.77 mm of
    // ring radius, so a range of 6 to 9.5 mm holds no valid layout; 5 to 5.9 magnets per ring
    // holds no even count at all. Neither reports a best value.
    let base = DesignInputs::default();
    let radius = span(FreeVariable::RingRadius.range(), 6.0, 9.5);
    let poles = span(FreeVariable::MagnetsPerRing.range(), 5.0, 5.9);
    for (variable, range) in [
        (FreeVariable::RingRadius, radius),
        (FreeVariable::MagnetsPerRing, poles),
    ] {
        assert_eq!(
            solve_in(&base, variable, 1.0, range),
            Ok(SizingOutcome::NotReachable { best: None }),
            "{variable:?}"
        );
    }
}

#[test]
fn solve_is_solve_in_over_the_slider_range() {
    // Spec 4.3: solve becomes solve_in(.., variable.range()): identical outcomes, point for
    // point, for every free variable on two designs, at targets met, missed and refused. 2.0695
    // N m is the no-back-iron design's meeting interval inside one ring radius cell.
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    no_iron.metal.face_gap_mm = 0.5;
    for base in [DesignInputs::default(), no_iron] {
        for variable in FreeVariable::ALL {
            for target in [0.5, 2.0695, base.metal.required_min_Nm, 10.0, 0.0] {
                assert_eq!(
                    solve(&base, variable, target),
                    solve_in(&base, variable, target, variable.range()),
                    "{variable:?} {target}"
                );
            }
        }
    }
}

#[test]
fn a_range_that_is_not_finite_or_is_reversed_is_refused() {
    // The range is checked before the search: reversed, NaN or infinite ends are refused with
    // the range's own ends, and grid_in gives no values for them. (min == max is a range of
    // one value, not a refusal: a_range_of_one_value_evaluates_that_value_once.)
    let base = DesignInputs::default();
    let ends = [
        (30.0, 9.0),
        (f64::NAN, 30.0),
        (9.0, f64::NAN),
        (f64::NEG_INFINITY, 30.0),
        (9.0, f64::INFINITY),
    ];
    for variable in FreeVariable::ALL {
        for (min, max) in ends {
            let range = span(variable.range(), min, max);
            assert!(
                variable.grid_in(range).is_empty(),
                "{variable:?} {min} {max}"
            );
            match solve_in(&base, variable, 2.0, range) {
                Err(SizingError::InvalidRange { min: lo, max: hi }) => assert_eq!(
                    (lo.to_bits(), hi.to_bits()),
                    (min.to_bits(), max.to_bits()),
                    "{variable:?}"
                ),
                other => panic!("{variable:?} {min} {max}: {other:?}"),
            }
        }
    }
    // The order of the refusals: the target first, then the range, then the inputs.
    let reversed = span(FreeVariable::AxialLength.range(), 30.0, 9.0);
    assert_eq!(
        solve_in(&base, FreeVariable::AxialLength, -1.0, reversed),
        Err(SizingError::InvalidTarget(-1.0))
    );
    let mut bad = DesignInputs::default();
    bad.metal.face_gap_mm = f64::NAN;
    assert_eq!(
        solve_in(&bad, FreeVariable::AxialLength, 2.0, reversed),
        Err(SizingError::InvalidRange {
            min: 30.0,
            max: 9.0
        })
    );
}

#[test]
fn a_target_beyond_a_caller_range_is_not_reachable_inside_it() {
    // solve_in never leaves the caller's range: a target no value meets reports the best value
    // inside it, here each range's maximum, where the torque still rises (the length; 4 to 8
    // poles; the ring radius below its 14 mm peak).
    let base = DesignInputs::default();
    let cases = [
        (FreeVariable::AxialLength, 5.0, 20.0),
        (FreeVariable::MagnetsPerRing, 4.0, 8.0),
        (FreeVariable::RingRadius, 9.0, 12.0),
    ];
    for (variable, min, max) in cases {
        match solve_in(&base, variable, 10.0, span(variable.range(), min, max)) {
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => {
                assert_eq!(best.value, max, "{variable:?}");
                assert_eq!(best.inputs, variable.apply(&base, max), "{variable:?}");
            }
            other => panic!("{variable:?}: {other:?}"),
        }
    }
}

#[test]
fn a_wider_range_solves_the_ring_radius_below_the_slider_floor() {
    // Spec 4.3 and 6.8: the ring-size Auto passes the Size tab's range, which a lowered minimum
    // outside diameter takes below the slider's 9 mm. With 8 poles on the default hub the inner
    // flats fit from 7.67 mm (the block width over 2 tan(pi/8)) and the hot-low torque rises
    // from there to about 10.2 mm: the torque at 8.5 mm is first met at 8.5 mm over 6 to 30 mm,
    // and over the slider range at its floor, as it is.
    let mut base = DesignInputs::default();
    base.coupling.npole = 8;
    let slider = FreeVariable::RingRadius.range();
    assert_eq!(slider.min, 9.0, "the floor this test goes below");
    let wide = span(slider, 6.0, slider.max);
    let target = hot_low(&FreeVariable::RingRadius.apply(&base, 8.5));
    let p = solved(solve_in(&base, FreeVariable::RingRadius, target, wide));
    assert!((p.value - 8.5).abs() <= 1e-6, "{}", p.value);
    assert!(p.torque_hot_low_Nm >= target);
    assert_eq!(p.inputs, FreeVariable::RingRadius.apply(&base, p.value));
    assert!(is_valid(&p.inputs, &compute_all(&p.inputs)));
    assert_eq!(
        solved(solve(&base, FreeVariable::RingRadius, target)).value,
        slider.min
    );
    // is_valid still decides below the floor: a tiny target is met where the flats start to
    // fit, not at the range's 6 mm.
    let width = compute_all(&base).model.inner_width_mm;
    let fit_edge = width / (2.0 * (PI / 8.0).tan());
    assert!(fit_edge > wide.min && fit_edge < slider.min, "{fit_edge}");
    let edge = solved(solve_in(&base, FreeVariable::RingRadius, 0.1, wide));
    assert!(
        (edge.value - fit_edge).abs() <= 1e-6,
        "{} vs {fit_edge}",
        edge.value
    );
    let below = FreeVariable::RingRadius.apply(&base, edge.value - 1e-6);
    assert!(!blocks_fit(&below.coupling, &compute_all(&below).model));
}

#[test]
fn the_keyway_wall_still_counts_below_the_slider_floor() {
    // With 6 poles the flats fit below 6 mm, but the 10 mm bore and 1.7 mm keyway leave hub
    // wall past the key only above 6.75 mm (0.05 bond + 5 + 1.7): a tiny target over 6 to 30 mm
    // is met at that edge, below the slider's 9 mm floor.
    let mut base = DesignInputs::default();
    base.coupling.npole = 6;
    let wide = span(FreeVariable::RingRadius.range(), 6.0, 30.0);
    let edge =
        base.metal.bond_inner_mm + base.coupling.bore_mm / 2.0 + base.coupling.keyway_depth_mm;
    let p = solved(solve_in(&base, FreeVariable::RingRadius, 0.1, wide));
    assert!(compute_all(&p.inputs).model.hub_wall_past_key_mm > 0.0);
    assert!((p.value - edge).abs() <= 1e-6, "{} vs {edge}", p.value);
    let below = FreeVariable::RingRadius.apply(&base, p.value - 1e-6);
    assert!(!is_valid(&below, &compute_all(&below)));
}

```

- [ ] **Step 2: Run the engine tests and confirm they fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test sizing`

Expected: the build fails with these errors:
```
error[E0432]: unresolved import `magcoupling::engine::sizing::solve_in`
error[E0599]: no method named `grid_in` found for enum `FreeVariable` in the current scope
error[E0599]: no variant named `InvalidRange` found for enum `SizingError`
```

- [ ] **Step 3: Implement the engine changes in `magcoupling-rs/src/engine/sizing.rs`**

3a. Module docs, lines 12-13. Replace:
```rust
//! continuous variable is sampled in ascending order at the [`SCAN_CELLS`] + 1 ends of equal
//! cells of its slider range, and three refinements find what lies between two samples:
```
with:
```rust
//! continuous variable is sampled in ascending order at the [`SCAN_CELLS`] + 1 ends of equal
//! cells of its search range (its slider range, or the caller's range given to [`solve_in`]),
//! and three refinements find what lies between two samples:
```

3b. Module docs, lines 26-29. Replace:
```rust
//! Magnets per ring (poles per ring, one block per pole) steps through the even values of its
//! slider, with no refinement. Each value is one [`compute_all`] of the design with only the
//! free variable changed, so the answer is what the forward calculation shows, and every
//! value tried lies inside the slider range.
```
with:
```rust
//! Magnets per ring (poles per ring, one block per pole) steps through the even values inside
//! its search range, with no refinement. Each value is one [`compute_all`] of the design with
//! only the free variable changed, so the answer is what the forward calculation shows, and
//! every value tried lies inside the search range.
```

3c. Module docs, lines 31-36. Replace:
```rust
//! **Residual limits** (stated, not handled): a hump of the torque whose rise and fall both
//! lie inside one cell (0.328 mm of ring radius, 0.778 mm of axial length) leaves no trace on
//! the samples and is not seen, and neither is a validity change that reverts inside one
//! cell. Near a peak the torque is flat, so the peak search finds the peak torque to its
//! floating-point resolution only (about 1e-15 relative): a target within that of the true
//! peak may read "not reachable".
```
with:
```rust
//! **Residual limits** (stated, not handled): a hump of the torque whose rise and fall both
//! lie inside one cell (over the slider ranges, 0.328 mm of ring radius and 0.778 mm of axial
//! length; over a caller's range, its width over [`SCAN_CELLS`]) leaves no trace on the
//! samples and is not seen, and neither is a validity change that reverts inside one cell.
//! Near a peak the torque is flat, so the peak search finds the peak torque to its
//! floating-point resolution only (about 1e-15 relative): a target within that of the true
//! peak may read "not reachable".
```

3d. Replace `FreeVariable::grid` (lines 99-118, from `/// The values the coarse pass evaluates, ascending: ...` to the end of the closing `}` of `grid`) with:
```rust
    /// The values the coarse pass of [`solve`] evaluates: [`FreeVariable::grid_in`] the
    /// variable's slider range.
    pub fn grid(self) -> Vec<f64> {
        self.grid_in(self.range())
    }

    /// The values the coarse pass evaluates over `range`, ascending and each once: the
    /// [`SCAN_CELLS`] + 1 cell ends of a continuous variable, or every even whole number inside
    /// `range` for magnets per ring. Every value lies inside `range` (its step and scale are
    /// not used). A range of one value gives that value; a range [`solve_in`] refuses (an end
    /// not finite, or the minimum above the maximum) gives none.
    pub fn grid_in(self, range: SliderRange) -> Vec<f64> {
        if !is_search_range(range) {
            return Vec::new();
        }
        let r = range;
        let mut grid: Vec<f64> = match self {
            // ceil and floor: a minimum of 4.5 starts at 6, never at 4 below the range.
            FreeVariable::MagnetsPerRing => (r.min.ceil() as i64..=r.max.floor() as i64)
                .filter(|n| n % 2 == 0)
                .map(|n| n as f64)
                .collect(),
            FreeVariable::AxialLength | FreeVariable::RingRadius => (0..=SCAN_CELLS)
                .map(|i| {
                    if i == SCAN_CELLS {
                        r.max
                    } else {
                        r.min + (r.max - r.min) * i as f64 / SCAN_CELLS as f64
                    }
                })
                .collect(),
        };
        // A range of one value repeats it at every cell end: evaluate it once.
        grid.dedup();
        grid
    }
```

3e. Line 147, in `SizingOutcome::Solved`. Replace `/// The smallest value in the slider range that meets the target, within the module docs'` with `/// The smallest value in the search range that meets the target, within the module docs'`.

3f. In `SizingError` (lines 160-162), replace:
```rust
    /// The inputs fail [`DesignInputs::validate`] (every offending path, in schema order).
    InvalidInputs(Vec<SetError>),
}
```
with:
```rust
    /// The inputs fail [`DesignInputs::validate`] (every offending path, in schema order).
    InvalidInputs(Vec<SetError>),
    /// The search range of [`solve_in`] has an end that is not finite, or its minimum is above
    /// its maximum: the range's own ends.
    InvalidRange { min: f64, max: f64 },
}

/// Whether `range` can be searched: both ends finite and the minimum at most the maximum (a
/// range of one value counts).
fn is_search_range(range: SliderRange) -> bool {
    range.min.is_finite() && range.max.is_finite() && range.min <= range.max
}
```

3g. Replace the head of `solve` (lines 175-186):
```rust
/// Makes the hot-low torque of `inputs` meet `target_Nm` by adjusting `variable` alone (spec
/// A1; the module docs give the method, its limits and what counts).
#[allow(non_snake_case)] // unit suffix
pub fn solve(
    inputs: &DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
) -> Result<SizingOutcome, SizingError> {
    if !(target_Nm.is_finite() && target_Nm > 0.0) {
        return Err(SizingError::InvalidTarget(target_Nm));
    }
    inputs.validate().map_err(SizingError::InvalidInputs)?;
```
with:
```rust
/// Makes the hot-low torque of `inputs` meet `target_Nm` by adjusting `variable` alone (spec
/// A1; the module docs give the method, its limits and what counts), over the variable's
/// slider range: [`solve_in`] that range.
#[allow(non_snake_case)] // unit suffix
pub fn solve(
    inputs: &DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
) -> Result<SizingOutcome, SizingError> {
    solve_in(inputs, variable, target_Nm, variable.range())
}

/// [`solve`] over the caller's `range` of the free variable instead of its slider range (spec
/// guidance 4.3: the Size tab's ring-size Auto passes the radius range of its diameter range,
/// which a lowered minimum outside diameter takes below the slider's floor). Every value tried
/// lies inside `range`, and [`is_valid`] still decides what counts. Refuses, in this order, a
/// target that is not a positive finite torque, a range [`FreeVariable::grid_in`] cannot scan
/// (an end not finite, or the minimum above the maximum) and inputs that fail validation.
#[allow(non_snake_case)] // unit suffix
pub fn solve_in(
    inputs: &DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
    range: SliderRange,
) -> Result<SizingOutcome, SizingError> {
    if !(target_Nm.is_finite() && target_Nm > 0.0) {
        return Err(SizingError::InvalidTarget(target_Nm));
    }
    if !is_search_range(range) {
        return Err(SizingError::InvalidRange {
            min: range.min,
            max: range.max,
        });
    }
    inputs.validate().map_err(SizingError::InvalidInputs)?;
```
The rest of that body (the `evaluate` closure, `Search::new`, the `Ok(match ...)`) moves unchanged into `solve_in`, with one edit.

3h. In that body (old line 202), replace `    Ok(match search.over(&variable.grid()) {` with `    Ok(match search.over(&variable.grid_in(range)) {`.

`SliderRange` is already imported (`use super::meta::{SetError, SliderRange, input_rows};`), so no import changes.

- [ ] **Step 4: Run the engine tests and confirm they pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test sizing`

Expected: `test result: ok. 40 passed; 0 failed` (31 existing tests plus 9 new). Verified on a scratch copy of the crate, where it took about 0.2 s.

- [ ] **Step 5: Write the failing GUI status test in `magcoupling-rs/src/gui/sizing.rs`**

At the end of `mod tests`, after the closing `}` of `a_refused_solve_shows_why_and_keeps_the_inputs` and before the module's final `}`, add:
```rust

    #[test]
    fn a_refused_search_range_shows_why_and_keeps_the_inputs() {
        // The runner passes no range of its own yet (the Size tab's ring-size Auto will), so
        // the refusal is put in as the last outcome directly.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::RingRadius, 2.5);
        let runner = SizingRunner {
            last: Some((
                SolveKey::new(&inputs, &state),
                Err(SizingError::InvalidRange {
                    min: 30.0,
                    max: 9.0,
                }),
            )),
            ..SizingRunner::default()
        };
        assert_eq!(runner.value(state.variable), None);
        assert_eq!(runner.shown(&inputs, &state), inputs);
        let status = runner.status(&inputs, &state);
        assert_eq!(status, "Sizing refused: the search range is empty");
        assert!(status.is_ascii(), "every glyph is in egui's default fonts");
    }
```
`inverse` and `SolveKey::new` are the module's existing private helpers. The private fields `last` and `pending` can be reached from the child `tests` module.

- [ ] **Step 6: Run the GUI test and confirm it fails**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib a_refused_search_range`

Expected: the build fails with:
```
error[E0004]: non-exhaustive patterns: `Some(&Err(SizingError::InvalidRange { .. }))` not covered
   --> src\gui\sizing.rs:235:15
```

- [ ] **Step 7: Add the status arm in `SizingRunner::status`**

Replace:
```rust
            Some(Err(SizingError::InvalidInputs(errors))) => {
                let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
                format!("Sizing refused: invalid inputs ({})", paths.join(", "))
            }
            None => SOLVING.to_owned(),
```
with:
```rust
            Some(Err(SizingError::InvalidInputs(errors))) => {
                let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
                format!("Sizing refused: invalid inputs ({})", paths.join(", "))
            }
            Some(Err(SizingError::InvalidRange { .. })) => {
                "Sizing refused: the search range is empty".to_owned()
            }
            None => SOLVING.to_owned(),
```
The text is ASCII only, so it passes `test_support::assert_glyphs`. It contains no numbers and does not touch any engine label, help or verdict string, so `tests/python_schema.rs` is unaffected.

- [ ] **Step 8: Run the GUI sizing tests and confirm they pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::sizing`

Expected: `test result: ok. 10 passed; 0 failed`, which includes `a_refused_search_range_shows_why_and_keeps_the_inputs`.

- [ ] **Step 9: Run the wider suites for the touched modules**

Run each of these:
- `cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check`
  - Expected: no output and exit 0.
- `cargo test --manifest-path magcoupling-rs/Cargo.toml`
  - Expected: every test binary ends `ok`. This run covers:
    - `tests/sizing.rs` (40 tests);
    - `tests/robustness.rs`, including `sizing_never_panics_on_extreme_designs`, which calls `solve`;
    - `tests/size_controls.rs`;
    - the engine unit tests in `engine::sizing::tests`, which drive `Search` and are unchanged.
- `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`
  - Expected: `ok`, with 0 failed. This covers `gui::sizing`, the `gui::panel` sizing tests (`a_drag_in_progress_defers_the_solve` calls `engine::sizing::solve`), `gui::session` and `gui::spreadsheet`.
  - It gave 530 passed on the scratch copy at this branch's HEAD.
- `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
  - Expected: `Finished` with no warnings.

`cargo doc --no-deps --lib` adds no new rustdoc warnings. The only `sizing.rs` warning is the existing `[mm]` at line 59.

- [ ] **Step 10: Commit**

```bash
git add magcoupling-rs/src/engine/sizing.rs magcoupling-rs/src/gui/sizing.rs magcoupling-rs/tests/sizing.rs
git commit -F - <<'EOF'
feat(magcoupler): inverse sizing over a caller's search range (solve_in)

FreeVariable::grid_in and engine::sizing::solve_in search a caller's range of
the free variable; solve is solve_in over the slider range, with identical
outcomes. A range with an end that is not finite, or its minimum above its
maximum, is refused (SizingError::InvalidRange; the sizing status reads
"Sizing refused: the search range is empty"). The magnets grid stays inside
a fractional range and a one-value range is evaluated once. This prepares the
Size tab's ring-size Auto below the 9 mm slider floor (guidance spec 4.3).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```
The docs for `solve_in` and `grid_in` land in Task 10.

---

**Spec section 5 edge cases covered here (engine side):**
- **No value valid in the range** (ring size and magnets): `a_range_with_no_valid_value_has_no_best_value`.
- **Not reachable, best value inside the range:** `a_target_beyond_a_caller_range_is_not_reachable_inside_it`.
- **Ring size below the old 9 mm floor** (spec 6.8): `a_wider_range_solves_the_ring_radius_below_the_slider_floor`. It also checks that the fit rule still applies below 9 mm.
- **Keyway wall rule below 9 mm:** `the_keyway_wall_still_counts_below_the_slider_floor`.
- **Size-tab range closed to one value:** `a_range_of_one_value_evaluates_that_value_once`.

**Existing tests this change breaks:**
- **At runtime: none.** For the three slider ranges (4..40, 2..50.8 and 9..30), `grid()` returns the same values before and after the change: ceil and floor change nothing at whole-number ends, and the cell ends are all distinct, so `dedup` removes nothing. That keeps these tests unchanged:
  - `the_search_constants_are_the_documented_ones`;
  - `poles_stay_even`;
  - `a_target_beyond_the_range_is_not_reachable`;
  - `a_meeting_interval_inside_one_cell_is_found`;
  - `a_target_just_below_the_true_peak_is_solved`;
  - `a_target_met_only_where_the_keyway_starts_to_leave_wall_is_found`.

  Their outcomes are bit-identical, which `solve_is_solve_in_over_the_slider_range` pins.
- **At compile time: one break**, the exhaustive `match` in `SizingRunner::status` at `src/gui/sizing.rs:235`. Step 7 fixes it.

**Hand-off to Task 3:** Task 3 changes four helpers that the new test `gui::sizing::tests::a_refused_search_range_shows_why_and_keeps_the_inputs` uses:
- `inverse(variable, target_Nm)`;
- `SolveKey::new(&inputs, &state)`;
- `runner.shown(&inputs, &state)`;
- `runner.status(&inputs, &state)`.

`SizingState` loses `target_Nm`, and a range is added to `SolveKey` and to the runner methods. Once the runner takes a range, Task 3 should drive the refusal through a real solve instead of building `last` by hand. Replace the body with:
```rust
let inputs = DesignInputs::default();
let state = SizingState { mode: SizingMode::TorqueToMagnets, variable: FreeVariable::RingRadius };
let range = SliderRange { min: 30.0, max: 9.0, ..FreeVariable::RingRadius.range() };
let mut runner = SizingRunner::default();
runner.solve_now(&inputs, &state, range);
assert_eq!(runner.value(state.variable), None);
assert_eq!(runner.shown(&inputs, &state, range), inputs);
let status = runner.status(&inputs, &state, range);
assert_eq!(status, "Sizing refused: the search range is empty");
assert!(status.is_ascii(), "every glyph is in egui's default fonts");
```
This needs `use crate::engine::meta::SliderRange;` in the test module.

---

### Task 3: One torque number: the sizing target is `metal.required_min_Nm` (spec 4.4, decision G-2)

> **Amendments:** apply "T3 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Modify: `magcoupling-rs/src/gui/sizing.rs`
  - module docs, lines 1-14
  - imports, lines 16-19
  - `TARGET_RANGE_INPUT`, `SizingState`, `DEFAULT_TARGET_NM`, lines 81-107
  - `SolveKey`, lines 121-139
  - `SizingRunner` impl, lines 152-260
  - test module, lines 280-481
- Modify: `magcoupling-rs/src/gui/session.rs`
  - module docs, lines 15-29
  - `design_json`, lines 152-172
  - `sizing_from`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, lines 265-377
  - tests: `edited()` at lines 389-408, plus new tests
- Modify: `magcoupling-rs/src/gui/panel.rs`
  - module docs, lines 22-25
  - imports, lines 28, 36, 46-53
  - `TARGET_LABEL` doc, lines 95-96
  - field doc, line 219
  - `shown_inputs`, `sizing()`, lines 330-340
  - load functions, lines 353-421
  - `run_sizing`, `log_sizing`, `set_sizing_mode`, `sizing_ui`, lines 536-642
  - `export_spreadsheet`, lines 795-796
  - tests: import at line 1310; the glyph test at lines 3481-3570; six edited tests (3356, 3911, 3931, 3995, 4062); three new tests
- Modify: `magcoupling-rs/src/gui/spreadsheet.rs`
  - import, line 39
  - Summary rows, lines 314-326
  - test, lines 1176-1200
- Modify: `magcoupling-rs/src/app.rs`
  - test `the_smoke_test_share_link_opens_its_design`, lines 229-233
- Not modified: `magcoupling-rs/src/gui/history.rs`. Undo needs no change: the requirement lives in `Design.inputs`, which `History<Design>` already snapshots.
- Test: tests live inside the modules above (`gui::sizing::tests`, `gui::session::tests`, `gui::panel::tests`, `gui::spreadsheet::tests`, `app::tests`).

**Interfaces:**

Consumes, from Task 2 and existing code:
- Task 2: `crate::engine::sizing::solve_in(inputs: &DesignInputs, variable: FreeVariable, target_Nm: f64, range: SliderRange) -> Result<SizingOutcome, SizingError>`
- Task 2: `SizingError::InvalidRange { min: f64, max: f64 }`. Task 2 adds its `gui/sizing.rs` status arm with the text "Sizing refused: the search range is empty"; keep that arm.
- Existing: `FreeVariable::range()`, `crate::engine::meta::SliderRange` (Clone, Copy, Debug, PartialEq), `InputCatalogue`, `MagcouplingPanel::input_row_ui`, `History<Design>`.
- Test harness, all existing:
  - `Harness::new`, `Harness::size`, `Harness::frame`, `Harness::frame_after`, `Harness::solved` and the free function `count`, all in `gui/panel.rs` tests;
  - `test_support::{key_tap, flat_shapes}`.

Produces (contract names):
- `pub struct SizingState { pub mode: SizingMode, pub variable: FreeVariable }`. `target_Nm` and `DEFAULT_TARGET_NM` are removed.
- `pub const REQUIREMENT_INPUT: &str = "metal.required_min_Nm";` replaces `TARGET_RANGE_INPUT`.
- `SizingRunner`:
  - `update(&mut self, inputs, state, range: SliderRange, now, editing) -> Option<f64>`
  - `solve_now(&mut self, inputs, state, range)`
  - `is_current(&self, inputs, state, range) -> bool`
  - `shown(&self, inputs, state, range) -> DesignInputs`
  - `status(&self, inputs, state, range) -> String`
  - The range is part of `SolveKey`. The target is `inputs.metal.required_min_Nm`, which is already in the key.
- `gui/session.rs`: `pub fn target_mismatch_notice(target_Nm: f64, requirement_Nm: f64) -> String`.
- The panel shows that notice in its status line after `load_design_file`, `load_share_payload` and `open_share_payload`.
- The Torque -> Magnets target control is the requirement's own input row, edited through `InputSet::set`.
- Extra `gui/session.rs` API the contract does not name (Task 7 and Task 9 do not need it):
  - `pub struct LoadedDesign { pub design: Design, pub notice: Option<String> }`
  - `pub fn read_design_json(text) -> Result<LoadedDesign, LoadError>`
  - `pub fn read_share_payload(payload) -> Result<LoadedDesign, LoadError>`
  - `pub(crate) fn encode_share_text(text) -> String`
  - `design_from_json` and `decode_share_payload` keep their signatures as thin wrappers.
- Private `MagcouplingPanel::sizing_range(&self) -> SliderRange` returns `self.sizing.variable.range()`. Task 7 changes this one place for ring size.

---

- [ ] **Step 1: Replace the test module of `magcoupling-rs/src/gui/sizing.rs` (lines 280-481) with the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sizing::solve;

    #[test]
    fn every_mode_and_free_variable_round_trips_through_its_key() {
        for mode in SizingMode::ALL {
            assert_eq!(SizingMode::from_key(mode.key()), Some(mode));
        }
        for variable in FreeVariable::ALL {
            assert_eq!(variable_from_key(variable_key(variable)), Some(variable));
        }
        assert_eq!(SizingMode::from_key("Torque"), None);
        assert_eq!(variable_from_key(""), None);
    }

    #[test]
    fn the_default_state_is_the_forward_calculation_with_the_axial_length_free() {
        let state = SizingState::default();
        assert_eq!(state.mode, SizingMode::MagnetsToTorque);
        assert_eq!(state.variable, FreeVariable::AxialLength);
    }

    fn inverse(variable: FreeVariable) -> SizingState {
        SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable,
        }
    }

    /// The default design with the requirement, the sizing target (decision G-2), at
    /// `required_Nm`.
    #[allow(non_snake_case)] // unit suffix
    fn requiring(required_Nm: f64) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.metal.required_min_Nm = required_Nm;
        inputs
    }

    #[test]
    fn a_solve_waits_for_the_debounce_and_runs_once() {
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        assert_eq!(
            runner.update(&inputs, &state, range, 10.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.status(&inputs, &state, range), SOLVING);
        let wait = runner.update(&inputs, &state, range, 10.1, false).unwrap();
        assert!((wait - 0.15).abs() < 1e-12, "{wait}");
        assert_eq!(runner.solves, 0);
        assert_eq!(runner.update(&inputs, &state, range, 10.25, false), None);
        assert_eq!(runner.solves, 1);
        for frame in 0..10 {
            assert_eq!(
                runner.update(&inputs, &state, range, 10.3 + frame as f64, false),
                None
            );
        }
        assert_eq!(runner.solves, 1, "never per frame");
        assert!(runner.is_current(&inputs, &state, range));
    }

    #[test]
    fn an_edit_in_progress_or_a_new_change_restarts_the_wait() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        // A drag: still editing past the debounce, so nothing runs.
        runner.update(&inputs, &state, range, 1.0, true);
        assert_eq!(runner.solves, 0);
        // Released: the wait starts again from the release.
        runner.update(&inputs, &state, range, 1.1, false);
        runner.update(&inputs, &state, range, 1.2, false);
        assert_eq!(runner.solves, 0);
        inputs.metal.face_gap_mm = 1.5;
        runner.update(&inputs, &state, range, 1.4, false);
        runner.update(&inputs, &state, range, 1.5, false);
        assert_eq!(runner.solves, 0, "a new change restarts the wait");
        runner.update(&inputs, &state, range, 1.65, false);
        assert_eq!(runner.solves, 1);
    }

    #[test]
    fn solve_now_solves_a_pending_change_at_once() {
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        runner.update(&inputs, &state, range, 1.0, false);
        inputs.metal.face_gap_mm = 1.5;
        assert_eq!(
            runner.update(&inputs, &state, range, 1.1, false),
            Some(DEBOUNCE_S)
        );
        assert!(
            !runner.is_current(&inputs, &state, range),
            "waiting for the debounce"
        );
        runner.solve_now(&inputs, &state, range);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state, range));
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable at a 1.5 mm gap")
        };
        assert_eq!(
            point.inputs.metal.face_gap_mm, 1.5,
            "this design's solution"
        );
        // Nothing is left pending: the next frame does not solve again.
        assert_eq!(runner.update(&inputs, &state, range, 1.2, false), None);
        assert_eq!(runner.solves, 2);
    }

    #[test]
    fn the_free_variable_s_own_value_does_not_trigger_a_solve() {
        // The solve sets the free variable, so its value in the inputs is not part of the key.
        let mut inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(&inputs, &state, range, 0.0, false);
        runner.update(&inputs, &state, range, 1.0, false);
        inputs.coupling.magnets.axial_length_mm = Some(30.0);
        assert!(runner.is_current(&inputs, &state, range));
        assert_eq!(runner.update(&inputs, &state, range, 2.0, false), None);
        assert_eq!(runner.solves, 1);
    }

    /// The runner after a solve of `state` at `inputs`, over the free variable's slider range.
    fn solved(inputs: &DesignInputs, state: &SizingState) -> SizingRunner {
        let range = state.variable.range();
        let mut runner = SizingRunner::default();
        runner.update(inputs, state, range, 0.0, false);
        runner.update(inputs, state, range, 1.0, false);
        runner
    }

    #[test]
    fn a_solved_design_shows_the_solved_value() {
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = DesignInputs::default();
        let runner = solved(&inputs, &state);
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("2.5 N·m is reachable by the axial length")
        };
        assert_eq!(runner.value(state.variable), Some(point.value));
        assert_eq!(runner.shown(&inputs, &state, range), point.inputs);
        assert_eq!(
            runner.status(&inputs, &state, range),
            format!(
                "Solved at {} mm (hot-low torque 2.500 N·m)",
                format_value(&Value::Num(point.value))
            )
        );
        // Another free variable has no outcome yet.
        assert_eq!(runner.value(FreeVariable::RingRadius), None);
    }

    #[test]
    fn the_target_is_the_requirement() {
        // Decision G-2: the runner solves for metal.required_min_Nm, and an edit of it is a
        // change of the design: the last solve is stale until the next one runs.
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = requiring(3.0);
        let mut runner = solved(&inputs, &state);
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve(&inputs, state.variable, 3.0))
        );
        let Some(Ok(SizingOutcome::Solved(point))) = runner.outcome(state.variable) else {
            panic!("3.0 N·m is reachable by the axial length")
        };
        assert!((point.torque_hot_low_Nm - 3.0).abs() < 1e-6);
        let raised = requiring(3.5);
        assert!(!runner.is_current(&raised, &state, range));
        assert_eq!(runner.status(&raised, &state, range), SOLVING);
        assert_eq!(
            runner.update(&raised, &state, range, 2.0, false),
            Some(DEBOUNCE_S)
        );
        assert_eq!(runner.update(&raised, &state, range, 3.0, false), None);
        assert_eq!(runner.solves, 2);
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve(&raised, state.variable, 3.5))
        );
    }

    #[test]
    fn the_search_range_is_part_of_the_key() {
        // The panel passes the range to search (the ring size's follows the Size tab's
        // diameter range, Task 7): another range is another solve, over that range.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let full = state.variable.range();
        let mut runner = solved(&inputs, &state);
        let narrow = SliderRange {
            min: 20.0,
            max: 30.0,
            ..full
        };
        assert!(runner.is_current(&inputs, &state, full));
        assert!(!runner.is_current(&inputs, &state, narrow));
        assert_eq!(runner.status(&inputs, &state, narrow), SOLVING);
        assert_eq!(
            runner.update(&inputs, &state, narrow, 2.0, false),
            Some(DEBOUNCE_S)
        );
        runner.solve_now(&inputs, &state, narrow);
        assert_eq!(runner.solves, 2);
        assert!(runner.is_current(&inputs, &state, narrow));
        assert_eq!(
            runner.outcome(state.variable),
            Some(&solve_in(
                &inputs,
                state.variable,
                inputs.metal.required_min_Nm,
                narrow
            ))
        );
        // 2.5 N·m needs less than 20 mm: the narrow range's smallest value meets it.
        assert_eq!(runner.value(state.variable), Some(20.0));
    }

    #[test]
    fn a_value_outside_the_search_range_is_not_shown() {
        // A value from an earlier search outside the range asked for now (a raised minimum OD
        // moves the ring size's range past it) is not shown: the inputs show as they are until
        // the new search runs. Its bounds count as inside.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::AxialLength);
        let full = state.variable.range();
        let runner = solved(&inputs, &state);
        let value = runner.value(state.variable).expect("solved");
        let sized = state.variable.apply(&inputs, value);
        assert_eq!(runner.shown(&inputs, &state, full), sized);
        let above = SliderRange {
            min: value + 1.0,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, above), inputs);
        let below = SliderRange {
            max: value - 1.0,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, below), inputs);
        let exact = SliderRange {
            min: value,
            max: value,
            ..full
        };
        assert_eq!(runner.shown(&inputs, &state, exact), sized);
    }

    #[test]
    fn an_empty_or_nan_search_range_is_refused_once() {
        // The engine refuses an empty range (Task 2). A NaN bound still equals itself in the
        // key: one refused solve, not a wait restarted every frame.
        let inputs = DesignInputs::default();
        let state = inverse(FreeVariable::RingRadius);
        let full = state.variable.range();
        for range in [
            SliderRange {
                min: 12.0,
                max: 11.0,
                ..full
            },
            SliderRange {
                min: f64::NAN,
                ..full
            },
        ] {
            let mut runner = SizingRunner::default();
            runner.update(&inputs, &state, range, 0.0, false);
            assert_eq!(runner.update(&inputs, &state, range, 1.0, false), None);
            assert_eq!(runner.solves, 1);
            for frame in 0..5 {
                assert_eq!(
                    runner.update(&inputs, &state, range, 2.0 + frame as f64, false),
                    None
                );
            }
            assert_eq!(runner.solves, 1, "{range:?}: refused once, not re-awaited");
            assert_eq!(
                runner.status(&inputs, &state, range),
                "Sizing refused: the search range is empty"
            );
            assert_eq!(runner.shown(&inputs, &state, range), inputs);
        }
    }

    #[test]
    fn an_unreachable_target_shows_the_best_value() {
        let state = inverse(FreeVariable::AxialLength);
        let range = state.variable.range();
        let inputs = requiring(50.0);
        let runner = solved(&inputs, &state);
        assert_eq!(runner.value(state.variable), Some(50.8));
        assert_eq!(
            runner
                .shown(&inputs, &state, range)
                .coupling
                .magnets
                .axial_length_mm,
            Some(50.8)
        );
        assert_eq!(
            runner.status(&inputs, &state, range),
            "Not reachable (best 9.937 N·m at 50.80 mm)"
        );
        let poles = inverse(FreeVariable::MagnetsPerRing);
        let defaults = DesignInputs::default();
        assert_eq!(
            solved(&defaults, &poles).status(&defaults, &poles, poles.variable.range()),
            "Not reachable (best 2.285 N·m at 10)"
        );
    }

    #[test]
    fn a_refused_solve_shows_why_and_keeps_the_inputs() {
        // The requirement is the target: one that is not a positive torque (a file can hold
        // it; the slider cannot reach it) refuses the solve (spec 5).
        let state = inverse(FreeVariable::RingRadius);
        let range = state.variable.range();
        for (required, shown) in [(-1.0, "-1.000 N·m"), (0.0, "0 N·m")] {
            let inputs = requiring(required);
            let runner = solved(&inputs, &state);
            assert_eq!(runner.value(state.variable), None);
            assert_eq!(runner.shown(&inputs, &state, range), inputs);
            assert_eq!(
                runner.status(&inputs, &state, range),
                format!("Sizing refused: the requirement must be a positive torque (it is {shown})")
            );
        }
        let mut bad = DesignInputs::default();
        bad.coupling.backiron = 7;
        let runner = solved(&bad, &state);
        assert_eq!(
            runner.status(&bad, &state, range),
            "Sizing refused: invalid inputs (coupling.backiron)"
        );
    }
}
```

- [ ] **Step 2: Add the failing session tests to `magcoupling-rs/src/gui/session.rs`**

Replace `edited()` (lines 389-408) with:

```rust
    /// A design with a value of every input type changed, a non-default sizing state and a
    /// non-default requirement (the sizing target, decision G-2).
    fn edited() -> Design {
        let mut inputs = DesignInputs::default();
        inputs.metal.face_gap_mm = 1.41; // f64
        inputs.coupling.npole = 12; // i64 count
        inputs.coupling.backiron = 0; // selector
        inputs.coupling.magnets.part_inner = "K&J BX0X04-N52".to_owned(); // text
        inputs.coupling.magnets.axial_length_mm = Some(13.5); // optional, set
        inputs.metal.measured_drag_Nm = Some(0.0123);
        inputs.temperature.slip_loss.cap_integral_T2m4 = 5.27e-10; // tiny
        inputs.metal.life_events = 2.5e7; // large
        inputs.metal.required_min_Nm = 3.25; // the sizing target
        Design {
            inputs,
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::RingRadius,
            },
        }
    }
```

Append these tests at the end of the session test module, before its closing `}`:

```rust
    #[test]
    fn the_sizing_target_a_file_writes_is_its_requirement() {
        // Decision G-2: one torque number. The schema keeps "target_torque_Nm" (version 1
        // readers check it), equal to the requirement, and the version stays 1.
        assert_eq!(DESIGN_VERSION, 1, "decision G-2 keeps the format");
        let json: Json = serde_json::from_str(&design_to_json(&edited())).unwrap();
        assert_eq!(json["inputs"]["metal.required_min_Nm"], 3.25);
        assert_eq!(json["sizing"]["target_torque_Nm"], 3.25);
        // A requirement that is not a positive torque is no valid target: left out, so the
        // design still round-trips (file and link), with no notice.
        for requirement in [0.0, -1.5] {
            let mut design = edited();
            design.inputs.metal.required_min_Nm = requirement;
            let text = design_to_json(&design);
            let json: Json = serde_json::from_str(&text).unwrap();
            assert!(json["sizing"].get("target_torque_Nm").is_none(), "{text}");
            let want = LoadedDesign {
                design: design.clone(),
                notice: None,
            };
            assert_eq!(read_design_json(&text), Ok(want.clone()));
            assert_eq!(read_share_payload(&encode_share_payload(&design)), Ok(want));
        }
    }

    #[test]
    fn a_saved_design_reloads_exactly_with_no_notice() {
        for design in [Design::default(), edited()] {
            let want = LoadedDesign {
                design: design.clone(),
                notice: None,
            };
            assert_eq!(read_design_json(&design_to_json(&design)), Ok(want.clone()));
            assert_eq!(read_share_payload(&encode_share_payload(&design)), Ok(want));
        }
    }

    #[test]
    fn a_target_that_differs_from_the_requirement_loads_the_requirement_with_a_notice() {
        // A file written before decision G-2: its sizing target was its own number.
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"metal.required_min_Nm": 2.5},
            "sizing": {"mode": "torque_to_magnets", "target_torque_Nm": 3.25}}"#;
        let mut want = Design::default();
        want.sizing.mode = SizingMode::TorqueToMagnets;
        let loaded = read_design_json(text).unwrap();
        assert_eq!(loaded.design, want, "the requirement is used");
        assert_eq!(loaded.notice, Some(target_mismatch_notice(3.25, 2.5)));
        assert_eq!(
            target_mismatch_notice(3.25, 2.5),
            "This file's sizing target (3.25 N·m) differed from its requirement (2.5 N·m); \
             the requirement is used."
        );
        // The same text in a share link; the design-only readers give the design alone.
        let payload = encode_share_text(text);
        assert_eq!(read_share_payload(&payload), Ok(loaded));
        assert_eq!(design_from_json(text), Ok(want.clone()));
        assert_eq!(decode_share_payload(&payload), Ok(want));
        // A file without the requirement path has the default requirement, compared the same.
        let text = r#"{"format": "magcoupling-design", "version": 1, "sizing": {"target_torque_Nm": 3}}"#;
        assert_eq!(
            read_design_json(text).unwrap().notice,
            Some(target_mismatch_notice(3.0, 2.5))
        );
        // A difference below the display's four digits is named in full, not as two equal
        // numbers.
        let text = r#"{"format": "magcoupling-design", "version": 1, "sizing": {"target_torque_Nm": 2.50001}}"#;
        assert_eq!(
            read_design_json(text).unwrap().notice.as_deref(),
            Some(
                "This file's sizing target (2.50001 N·m) differed from its requirement \
                 (2.5 N·m); the requirement is used."
            )
        );
        // An equal target, or none: no notice.
        for sizing in [
            r#"{"target_torque_Nm": 2.5}"#,
            r#"{"mode": "torque_to_magnets"}"#,
            "{}",
        ] {
            let text =
                format!(r#"{{"format": "magcoupling-design", "version": 1, "sizing": {sizing}}}"#);
            assert_eq!(read_design_json(&text).unwrap().notice, None, "{sizing}");
        }
        // A malformed target is still refused, before any comparison (validated as before).
        let text = r#"{"format": "magcoupling-design", "version": 1, "sizing": {"target_torque_Nm": 0}}"#;
        assert!(matches!(
            read_design_json(text),
            Err(LoadError::Refused { .. })
        ));
    }
```

- [ ] **Step 3: Add the failing panel and spreadsheet tests**

In `magcoupling-rs/src/gui/panel.rs` tests, replace line 1310 `use crate::gui::session::encode_share_payload;` with:

```rust
    use crate::gui::session::{
        design_from_json, encode_share_payload, encode_share_text, target_mismatch_notice,
    };
```

In `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`, directly after `texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());` (line 3536), add:

```rust
        // The load's notice of a sizing target that differed from the requirement (G-2).
        texts.push(target_mismatch_notice(3.25, 2.5));
```

Add these tests after `a_share_link_carries_the_sizing_state` (ends at line 4078):

```rust
    #[test]
    fn the_sizing_target_is_the_requirement_s_row_and_one_undo_restores_it() {
        // Decision G-2: one torque number. In Torque -> Magnets the target is the requirement's
        // own input row (metal.required_min_Nm): an ordinary design edit that the solve meets,
        // that a share link carries and that one undo restores.
        let mut harness = Harness::new();
        let output = harness.size();
        let label = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .unwrap()
            .meta
            .label;
        assert_eq!(count(&output, label), 1, "the target row is the requirement's");
        assert_eq!(count(&output, TARGET_LABEL), 0, "no second target control");
        let steps = harness.panel.history.undo_len();
        // The target row is the first row the Key design group draws.
        let target = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(target));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(Vec::new());
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 2.51);
        assert_eq!(harness.panel.history.undo_len(), steps + 1, "one step");
        // The solve meets the new requirement after its debounce.
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!(harness.panel.runner.is_current(
            harness.panel.inputs(),
            harness.panel.sizing(),
            FreeVariable::AxialLength.range()
        ));
        let point = harness.solved();
        assert!((point.torque_hot_low_Nm - 2.51).abs() < 1e-6);
        assert_eq!(point.inputs.metal.required_min_Nm, 2.51);
        // A share link carries it: it is an input.
        let link = harness.panel.share_link();
        let payload = link.split("?m=").nth(1).unwrap();
        let mut other = MagcouplingPanel::new();
        other.load_share_payload(payload).unwrap();
        assert_eq!(other.inputs().metal.required_min_Nm, 2.51);
        // One undo restores it; the next solve meets it again.
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!((harness.solved().torque_hot_low_Nm - 2.5).abs() < 1e-6);
    }

    #[test]
    fn a_file_whose_sizing_target_differs_loads_its_requirement_and_says_so() {
        // A file written before decision G-2: its Torque -> Magnets target (3.25) was its own
        // number beside the requirement (2.5). The requirement is used, the solve meets it,
        // and the header says so (a wrapping line).
        let text = r#"{"format": "magcoupling-design", "version": 1,
            "inputs": {"metal.required_min_Nm": 2.5},
            "sizing": {"mode": "torque_to_magnets", "target_torque_Nm": 3.25}}"#;
        let notice = target_mismatch_notice(3.25, 2.5);
        let mut harness = Harness::new();
        harness.panel.load_design_file(text).unwrap();
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let status = format!("Design file loaded. {notice}");
        assert_eq!(count(&output, &status), 1);
        let wrap = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Text(t) if t.galley.text() == status => {
                    Some(t.galley.job.wrap.max_width)
                }
                _ => None,
            });
        assert!(wrap.is_some_and(f32::is_finite), "the line wraps: {wrap:?}");
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        assert!((harness.solved().torque_hot_low_Nm - 2.5).abs() < 1e-6);
        // The same text as a share link, opened at start-up or loaded: the same notice.
        let payload = encode_share_text(text);
        for open in [true, false] {
            let mut other = Harness::new();
            if open {
                other.panel.open_share_payload(&payload).unwrap();
            } else {
                other.panel.load_share_payload(&payload).unwrap();
            }
            other.frame(Vec::new());
            let output = other.frame(Vec::new());
            assert_eq!(
                count(&output, &format!("Design loaded from the share link. {notice}")),
                1,
                "open: {open}"
            );
            assert_eq!(other.panel.design(), harness.panel.design());
        }
    }

    #[test]
    fn idle_frames_in_torque_to_magnets_keep_the_requirement_as_loaded() {
        // The target row is the requirement's slider (SliderClamping::Edits, decision M41-2):
        // idle frames keep a file's value as it is, off the 0.01 grid and past the slider's
        // 10 N·m, and record no undo step.
        let mut design = Design::default();
        design.inputs.metal.required_min_Nm = 12.345;
        design.sizing.mode = SizingMode::TorqueToMagnets;
        let mut harness = Harness::new();
        harness.panel.open_design(design.clone());
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.history.undo_len(), 0);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.runner.solves, 1, "solved once, then idle");
    }
```

In `magcoupling-rs/src/gui/spreadsheet.rs`, make `torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row` (lines 1176-1200) fail before the change. Change its opening and its target assertion:

```rust
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.axial_length_mm = Some(14.2);
        inputs.metal.required_min_Nm = 3.25; // the sizing target (decision G-2)
        let results = compute_all(&inputs);
        let mut shot = snapshot(&inputs, &results);
        shot.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
        };
```

and

```rust
        assert_eq!(row(TARGET_LABEL)[1], Cell::number(3.25));
```

- [ ] **Step 4: Run the new tests and confirm they fail**

Command:
```
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::sizing
```
Expected: the build fails. The errors include:
- `error[E0061]: this method takes 4 arguments but 5 arguments were supplied` (`update`), with the same error for `status`, `is_current`, `solve_now` and `shown`;
- `error[E0063]: missing field `target_Nm` in initializer of `SizingState``;
- `error[E0425]: cannot find function `read_design_json` in this scope`;
- `error[E0432]: unresolved imports `crate::gui::session::encode_share_text`, `crate::gui::session::target_mismatch_notice``;
- `error[E0425]: cannot find value `REQUIREMENT_INPUT` in this scope`.

- [ ] **Step 5: Implement `magcoupling-rs/src/gui/sizing.rs`**

Replace the module docs (lines 1-14) with:

```rust
//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward
//! calculation, or Torque → Magnets, inverse sizing by [`crate::engine::sizing::solve_in`].
//!
//! The mode and the free variable are GUI state (A-2 decision A2-9): the engine takes them as
//! arguments. Design files and share links record them (decision M41-4). The target torque is
//! not GUI state: it is the requirement, the design input [`REQUIREMENT_INPUT`] (decision G-2:
//! one torque number, so the solve and the hot minimum check read the same value).
//!
//! [`SizingRunner`] runs the solve over the search range the panel gives: on a change, once
//! the design has been still for [`DEBOUNCE_S`] and no edit is in progress (no drag, no
//! typing), never per frame (a solve takes about 1 to 3 ms in a release build, longer on wasm,
//! against compute_all's 17 us). Until then the panel keeps showing the last solved value,
//! marked as solving. The design the panel shows in Torque → Magnets is the inputs with the
//! free variable at the solved value, or at the best value when the target is not reachable
//! (decision M41-8); the inputs keep their own value of the free variable until the user
//! leaves the mode (decision M41-7), when a change still waiting for its debounce is solved at
//! once ([`SizingRunner::solve_now`]).
```

Replace the imports (lines 16-19) with the following. Task 2 may have edited them; this is the final form.

```rust
use crate::DesignInputs;
use crate::engine::meta::{SliderRange, Value};
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve_in};
use crate::gui::format::{format_value, with_unit};
```

Replace lines 81-107, from `/// The input whose metadata gives the target torque` through the end of `impl Default for SizingState`, with:

```rust
/// The input inverse sizing meets: the hot minimum requirement. Decision G-2 (one torque
/// number): the sizing target is this design input, so the solve and the hot minimum check
/// read one value, and an edit of it is an ordinary design edit (undo, files, links).
pub const REQUIREMENT_INPUT: &str = "metal.required_min_Nm";

/// The sizing state: the mode and the free variable (the target is [`REQUIREMENT_INPUT`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizingState {
    pub mode: SizingMode,
    pub variable: FreeVariable,
}

impl Default for SizingState {
    fn default() -> Self {
        Self {
            mode: SizingMode::MagnetsToTorque,
            variable: FreeVariable::ALL[0],
        }
    }
}
```

Leave `DEBOUNCE_S`, `SOLVED_PREFIX`, `NOT_REACHABLE_PREFIX` and `SOLVING` unchanged. Replace `SolveKey` and its impl (lines 121-139) with:

```rust
/// What a solve depends on: the inputs with the free variable cleared (the solve sets it; the
/// target, [`REQUIREMENT_INPUT`], is among them), the free variable and the search range.
#[derive(Clone, Debug, PartialEq)]
struct SolveKey {
    inputs: DesignInputs,
    variable: FreeVariable,
    /// The search range's min, max, step and log flag as bit patterns: a NaN bound (the solve
    /// refuses it) still equals itself, so it is refused once instead of awaited every frame.
    range: [u64; 4],
}

impl SolveKey {
    fn new(inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> Self {
        Self {
            inputs: state.variable.apply(inputs, 0.0),
            variable: state.variable,
            range: [
                range.min.to_bits(),
                range.max.to_bits(),
                range.step.to_bits(),
                u64::from(range.log),
            ],
        }
    }
}
```

Leave the `SizingRunner` struct unchanged. Replace its whole `impl SizingRunner` block (lines 152-260) with the following. The `InvalidRange` arm is Task 2's: if Task 2 wrote it through a constant, keep Task 2's form.

```rust
impl SizingRunner {
    /// Called once per frame in Torque → Magnets mode with the search range, the frame's time
    /// and whether an edit is in progress. Solves when the design and the range have been
    /// unchanged for [`DEBOUNCE_S`] and no edit is in progress. Returns how long to wait before
    /// the next frame must run for a pending solve [s], or `None` when nothing is pending.
    pub fn update(
        &mut self,
        inputs: &DesignInputs,
        state: &SizingState,
        range: SliderRange,
        now: f64,
        editing: bool,
    ) -> Option<f64> {
        let key = SolveKey::new(inputs, state, range);
        if self.last.as_ref().is_some_and(|(last, _)| *last == key) {
            self.pending = None;
            return None;
        }
        let since = match &self.pending {
            Some((pending, since)) if *pending == key && !editing => *since,
            _ => {
                self.pending = Some((key, now));
                return Some(DEBOUNCE_S);
            }
        };
        let waited = now - since;
        if waited < DEBOUNCE_S {
            return Some(DEBOUNCE_S - waited);
        }
        self.solve_now(inputs, state, range);
        None
    }

    /// Solves this design over `range` now, whatever the debounce: the panel calls it once when
    /// the user leaves Torque → Magnets before a change was solved, so the value written into
    /// the inputs is this design's solution, not the last one's (decision M41-7). Never per
    /// frame. The target is the design's requirement ([`REQUIREMENT_INPUT`]).
    pub fn solve_now(&mut self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) {
        let outcome = solve_in(inputs, state.variable, inputs.metal.required_min_Nm, range);
        self.solves += 1;
        self.pending = None;
        self.last = Some((SolveKey::new(inputs, state, range), outcome));
    }

    /// Whether the last solve is for this design and this search range (nothing pending).
    pub fn is_current(&self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> bool {
        self.last
            .as_ref()
            .is_some_and(|(key, _)| *key == SolveKey::new(inputs, state, range))
    }

    /// The last outcome, if the last solve was for the same free variable (it may be stale:
    /// see [`SizingRunner::is_current`]).
    pub fn outcome(&self, variable: FreeVariable) -> Option<&Result<SizingOutcome, SizingError>> {
        match &self.last {
            Some((key, outcome)) if key.variable == variable => Some(outcome),
            _ => None,
        }
    }

    /// The free variable's value the panel shows: the solved value, or the best valid value
    /// when the target is out of reach; `None` before the first solve of this variable, when
    /// no value is valid, or when the solve refused.
    pub fn value(&self, variable: FreeVariable) -> Option<f64> {
        match self.outcome(variable)? {
            Ok(SizingOutcome::Solved(point)) => Some(point.value),
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => Some(best.value),
            _ => None,
        }
    }

    /// The design the panel shows in Torque → Magnets: `inputs` with the free variable at
    /// [`SizingRunner::value`] when that value lies in `range` (a value from an earlier search
    /// outside the range asked for now is not shown), else `inputs` as they are.
    pub fn shown(&self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> DesignInputs {
        match self
            .value(state.variable)
            .filter(|value| (range.min..=range.max).contains(value))
        {
            Some(value) => state.variable.apply(inputs, value),
            None => inputs.clone(),
        }
    }

    /// The status line under the sizing controls, for the search over `range`.
    pub fn status(&self, inputs: &DesignInputs, state: &SizingState, range: SliderRange) -> String {
        if !self.is_current(inputs, state, range) {
            return SOLVING.to_owned();
        }
        match self.outcome(state.variable) {
            Some(Ok(SizingOutcome::Solved(point))) => format!(
                "{SOLVED_PREFIX}{} (hot-low torque {})",
                variable_text(state.variable, point.value),
                torque_text(point.torque_hot_low_Nm)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) => format!(
                "{NOT_REACHABLE_PREFIX} (best {} at {})",
                torque_text(best.torque_hot_low_Nm),
                variable_text(state.variable, best.value)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: None })) => format!(
                "{NOT_REACHABLE_PREFIX}: no value of the free variable gives a valid design"
            ),
            Some(Err(SizingError::InvalidTarget(target))) => format!(
                "Sizing refused: the requirement must be a positive torque (it is {})",
                torque_text(*target)
            ),
            Some(Err(SizingError::InvalidRange { .. })) => {
                "Sizing refused: the search range is empty".to_owned()
            }
            Some(Err(SizingError::InvalidInputs(errors))) => {
                let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
                format!("Sizing refused: invalid inputs ({})", paths.join(", "))
            }
            None => SOLVING.to_owned(),
        }
    }
}
```

Leave `variable_text` and `torque_text` unchanged.

- [ ] **Step 6: Implement `magcoupling-rs/src/gui/session.rs`**

In the module docs, after the paragraph ending `a file without it is in the forward mode.` (line 21), insert:

```rust
//!
//! `sizing.target_torque_Nm` is the requirement, the input `metal.required_min_Nm` (decision
//! G-2: one torque number). A file writes it equal to the requirement. It leaves the key out
//! when the requirement is not a positive torque, which every reader refuses as a target, so
//! such a design still round-trips. A reader checks the key as before, then uses the
//! requirement. When the two differ (a file written before G-2), the load says so
//! ([`target_mismatch_notice`], in [`LoadedDesign::notice`]).
```

In `design_json`, replace `sizing_json.insert("target_torque_Nm".to_owned(), json_number(sizing.target_Nm));` (line 165) with:

```rust
    // Decision G-2: the target is the requirement. The key stays (version 1 readers check
    // it); a requirement that is not a positive torque is left out, since every reader refuses
    // it as a target, so its design still round-trips.
    let requirement = design.inputs.metal.required_min_Nm;
    if requirement.is_finite() && requirement > 0.0 {
        sizing_json.insert("target_torque_Nm".to_owned(), json_number(requirement));
    }
```

After `impl std::error::Error for LoadError {}` (line 128), insert:

```rust
/// A design read from a file or a share link, and what the reader resolved on the way.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedDesign {
    pub design: Design,
    /// [`target_mismatch_notice`] when the file's sizing target differed from its requirement
    /// (the requirement is used); `None` otherwise.
    pub notice: Option<String>,
}

/// What a load says when a file's sizing target (`sizing.target_torque_Nm`) differs from its
/// requirement (`metal.required_min_Nm`), decision G-2. The numbers are the file's in full
/// (shortest round-trip form), not rounded for display: two values that differ below the
/// panel's four significant digits would otherwise read the same.
#[allow(non_snake_case)] // unit suffix
pub fn target_mismatch_notice(target_Nm: f64, requirement_Nm: f64) -> String {
    format!(
        "This file's sizing target ({target_Nm} N·m) differed from its requirement \
         ({requirement_Nm} N·m); the requirement is used."
    )
}
```

Replace `sizing_from` and `design_from_json` (lines 265-339) with:

```rust
/// The sizing state a file's `sizing` object gives, and the target torque it names, if any; or
/// every problem.
fn sizing_from(json: &Json) -> Result<(SizingState, Option<f64>), Vec<String>> {
    let Json::Object(map) = json else {
        return Err(vec!["\"sizing\" is not an object".to_owned()]);
    };
    let mut state = SizingState::default();
    let mut target = None;
    let mut problems = Vec::new();
    for (key, value) in map {
        match (key.as_str(), value) {
            ("mode", Json::String(name)) => match SizingMode::from_key(name) {
                Some(mode) => state.mode = mode,
                None => problems.push(format!("unknown mode {name:?}")),
            },
            ("free_variable", Json::String(name)) => match variable_from_key(name) {
                Some(variable) => state.variable = variable,
                None => problems.push(format!("unknown free variable {name:?}")),
            },
            ("target_torque_Nm", Json::Number(n)) => match n.as_f64() {
                Some(torque) if torque.is_finite() && torque > 0.0 => target = Some(torque),
                _ => problems.push(format!("target torque {n} is not a positive number")),
            },
            (key, value) => problems.push(format!("unexpected {key:?}: {value}")),
        }
    }
    if problems.is_empty() {
        Ok((state, target))
    } else {
        Err(problems)
    }
}

/// Reads a design file (the module docs give the format and the rules): the design, and the
/// notice of a sizing target that differed from the requirement.
pub fn read_design_json(text: &str) -> Result<LoadedDesign, LoadError> {
    let json: Json =
        serde_json::from_str(text).map_err(|error| LoadError::NotJson(error.to_string()))?;
    let Json::Object(top) = &json else {
        return Err(LoadError::NotADesign(
            "the top level is not an object".to_owned(),
        ));
    };
    if top.get("format") != Some(&Json::from(DESIGN_FORMAT)) {
        return Err(LoadError::NotADesign(format!(
            "no \"format\": \"{DESIGN_FORMAT}\""
        )));
    }
    let version = match top.get("version") {
        Some(json) => json
            .as_u64()
            .filter(|v| (1..=DESIGN_VERSION).contains(v))
            .ok_or_else(|| LoadError::UnsupportedVersion(json.to_string()))?,
        None => return Err(LoadError::UnsupportedVersion("missing".to_owned())),
    };
    if let Some(key) = top
        .keys()
        .find(|key| !matches!(key.as_str(), "format" | "version" | "inputs" | "sizing"))
    {
        return Err(LoadError::NotADesign(format!("unexpected key {key:?}")));
    }
    // Both are checked before either refuses, so the refusal names every problem.
    let inputs = match top.get("inputs") {
        Some(json) => inputs_from(json, version, PATH_MIGRATIONS),
        None => Ok(DesignInputs::default()),
    };
    let sizing = match top.get("sizing") {
        Some(json) => sizing_from(json),
        None => Ok((SizingState::default(), None)),
    };
    match (inputs, sizing) {
        (Ok(inputs), Ok((sizing, target))) => {
            // Decision G-2: the requirement is the target; a file that says otherwise was
            // written before, and loads with the requirement and a notice.
            let requirement = inputs.metal.required_min_Nm;
            let notice = target
                .filter(|&target| target != requirement)
                .map(|target| target_mismatch_notice(target, requirement));
            Ok(LoadedDesign {
                design: Design { inputs, sizing },
                notice,
            })
        }
        (inputs, sizing) => Err(LoadError::Refused {
            inputs: inputs.err().unwrap_or_default(),
            sizing: sizing.err().unwrap_or_default(),
        }),
    }
}

/// Reads a design file: [`read_design_json`]'s design alone.
pub fn design_from_json(text: &str) -> Result<Design, LoadError> {
    read_design_json(text).map(|loaded| loaded.design)
}
```

Replace `encode_share_payload` and `decode_share_payload` (lines 350-377) with the following. `LINK_BASE64` and `share_link` are unchanged.

```rust
/// The `?m=` value of a share link: the compact design JSON, deflated, URL-safe base64.
pub fn encode_share_payload(design: &Design) -> String {
    encode_share_text(&design_json(design).to_string())
}

/// The `?m=` value that holds `text` (a design file's text): deflated, URL-safe base64.
/// [`read_share_payload`] reads it back as [`read_design_json`] reads the text.
pub(crate) fn encode_share_text(text: &str) -> String {
    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder
        .write_all(text.as_bytes())
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("finishing into a Vec cannot fail");
    LINK_BASE64.encode(compressed)
}

/// The design a `?m=` value holds (surrounding whitespace ignored), with
/// [`read_design_json`]'s notice.
pub fn read_share_payload(payload: &str) -> Result<LoadedDesign, LoadError> {
    let bytes = LINK_BASE64
        .decode(payload.trim())
        .map_err(|error| LoadError::Link(format!("not URL-safe base64 ({error})")))?;
    let mut text = String::new();
    DeflateDecoder::new(&bytes[..])
        .take(MAX_DESIGN_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| LoadError::Link(format!("not a compressed design ({error})")))?;
    if text.len() as u64 > MAX_DESIGN_BYTES {
        return Err(LoadError::Link(format!(
            "it inflates past {MAX_DESIGN_BYTES} bytes"
        )));
    }
    read_design_json(&text)
}

/// The design a `?m=` value holds: [`read_share_payload`]'s design alone.
pub fn decode_share_payload(payload: &str) -> Result<Design, LoadError> {
    read_share_payload(payload).map(|loaded| loaded.design)
}
```

- [ ] **Step 7: Implement `magcoupling-rs/src/gui/panel.rs`**

Module docs, lines 22-25: replace the Sizing paragraph with:

```rust
//! Sizing (spec Addendum A1): the mode switch tops the Key design group. In Torque → Magnets
//! the panel shows the design with the free variable at the solved value
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame), and the free variable's
//! row shows that value, locked. The target is the requirement (decision G-2): its own input
//! row under the free-variable picker.
```

Imports:
- Line 28: `use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};`
- Line 36: `use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row};`. `slider` is no longer used in this file.
- Lines 46-53:

```rust
use crate::gui::session::{
    Design, LoadError, LoadedDesign, PUBLIC_BASE_URL, design_to_json, read_design_json,
    read_share_payload, share_link,
};
use crate::gui::sizing::{
    REQUIREMENT_INPUT, SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState,
    variable_label,
};
```

Lines 95-96:

```rust
/// The label of the sizing target in the spreadsheet's Summary (Torque → Magnets): the
/// requirement's value (decision G-2; the panel shows the requirement's own row).
pub const TARGET_LABEL: &str = "Target hot-low torque";
```

Line 219: `/// The sizing mode and free variable (the target is the requirement input, decision G-2).`

Replace `shown_inputs` and the `sizing()` accessor (lines 328-340) with:

```rust
    /// The design the panel shows: the inputs, or in Torque → Magnets the inputs with the free
    /// variable at the solved value (the best value when the target is out of reach).
    pub fn shown_inputs(&self) -> DesignInputs {
        match self.sizing.mode {
            SizingMode::MagnetsToTorque => self.inputs.clone(),
            SizingMode::TorqueToMagnets => {
                self.runner
                    .shown(&self.inputs, &self.sizing, self.sizing_range())
            }
        }
    }

    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
    }

    /// The range inverse sizing searches: the free variable's slider range.
    fn sizing_range(&self) -> SliderRange {
        self.sizing.variable.range()
    }
```

Load functions: replace `load_design_file`, `load_share_payload` and `open_share_payload` with the following, keeping their doc comments:

```rust
    pub fn load_design_file(&mut self, text: &str) -> Result<(), LoadError> {
        self.load(read_design_json(text), "Design file loaded")
    }
```

```rust
    pub fn load_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load(
            read_share_payload(payload),
            "Design loaded from the share link",
        )
    }
```

```rust
    pub fn open_share_payload(&mut self, payload: &str) -> Result<(), LoadError> {
        self.load_with(
            read_share_payload(payload),
            "Design loaded from the share link",
            Self::open_design,
        )
    }
```

Then replace `load` and `load_with` (lines 395-421) with:

```rust
    /// Applies a loaded design as an edit that can be undone.
    fn load(
        &mut self,
        loaded: Result<LoadedDesign, LoadError>,
        done: &str,
    ) -> Result<(), LoadError> {
        self.load_with(loaded, done, Self::set_design)
    }

    /// Applies a loaded design with `apply` (an edit, or a new start) and says `done`, then the
    /// load's notice (a sizing target that differed from the requirement, decision G-2); on
    /// refusal changes nothing and shows why.
    fn load_with(
        &mut self,
        loaded: Result<LoadedDesign, LoadError>,
        done: &str,
        apply: fn(&mut Self, Design),
    ) -> Result<(), LoadError> {
        match loaded {
            Ok(LoadedDesign { design, notice }) => {
                apply(self, design);
                self.minimum_diameter_mm = None;
                self.status = Some(match notice {
                    Some(notice) => format!("{done}. {notice}"),
                    None => done.to_owned(),
                });
                Ok(())
            }
            Err(error) => {
                self.last_error = Some(error.to_string());
                self.status = None;
                Err(error)
            }
        }
    }
```

In `run_sizing` (lines 538-557), replace the `if let Some(wait) = ...` header with:

```rust
        let range = self.sizing_range();
        if let Some(wait) = self
            .runner
            .update(&self.inputs, &self.sizing, range, now, self.editing(ui))
        {
```

In `log_sizing` (lines 560-565):

```rust
        log::info!(
            "{SIZING_LOG_PREFIX}{}",
            self.runner
                .status(&self.inputs, &self.sizing, self.sizing_range())
        );
```

In `set_sizing_mode` (lines 572-578):

```rust
        if mode == SizingMode::MagnetsToTorque {
            let range = self.sizing_range();
            if !self.runner.is_current(&self.inputs, &self.sizing, range) {
                self.runner.solve_now(&self.inputs, &self.sizing, range);
                self.log_sizing();
            }
            self.inputs = self.runner.shown(&self.inputs, &self.sizing, range);
        }
```

In `sizing_ui`:
- Doc comment (lines 582-583): `/// The sizing controls at the top of the Key design group: the mode switch, then in` / `/// Torque → Magnets the free variable, the target (the requirement's own row) and the outcome.`
- Replace lines 606-622, from `let entry = InputCatalogue::get()` through `let status = self.runner.status(&self.inputs, &self.sizing);`, with:

```rust
        // The target is the requirement (decision G-2): its own input row, an ordinary design
        // edit (undo, files, links) like its row in the Metal design group.
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement is an input");
        let widget = self.input_row_ui(ui, entry, readouts);
        self.design_widgets.push(widget);
        let status = self
            .runner
            .status(&self.inputs, &self.sizing, self.sizing_range());
```

In `export_spreadsheet` (lines 795-796):

```rust
            sizing_status: (self.sizing.mode == SizingMode::TorqueToMagnets)
                .then(|| self.runner.status(&self.inputs, &self.sizing, self.sizing_range())),
```

- [ ] **Step 8: Implement `magcoupling-rs/src/gui/spreadsheet.rs` and fix `magcoupling-rs/src/app.rs`**

In `spreadsheet.rs`:
- Line 39: `use crate::gui::sizing::{REQUIREMENT_INPUT, SizingMode, SizingState, variable_label};`
- In `summary_sheet`, lines 315-326:

```rust
        let target_unit = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .map_or("", |entry| entry.meta.unit);
        rows.push(vec![
            Cell::text(FREE_VARIABLE_LABEL),
            Cell::text(variable_label(sizing.variable)),
        ]);
        // The target is the requirement (decision G-2).
        rows.push(vec![
            Cell::text(TARGET_LABEL),
            value_cell(&Value::Num(snapshot.inputs.metal.required_min_Nm)),
            Cell::text(target_unit),
        ]);
```

In `app.rs`, `the_smoke_test_share_link_opens_its_design` (lines 229-233):

```rust
        want.sizing = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::AxialLength,
        };
```

- [ ] **Step 9: Update the existing panel tests that set the old target (exact edits below; the full list is in the table at the end)**

- `a_loaded_design_file_replaces_the_design_and_can_be_undone`, line 3360: `loaded.sizing.target_Nm = 3.0;` becomes `loaded.inputs.metal.required_min_Nm = 3.0;`
- `leaving_torque_to_magnets_while_solving_writes_the_current_solution`, lines 3917-3928, become:

```rust
        harness.panel.inputs.metal.required_min_Nm = 3.0;
        assert_eq!(count(&harness.frame(Vec::new()), SOLVING), 1);
        harness.click_text(SizingMode::MagnetsToTorque.label());
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.runner.solves, 2, "solved on leaving");
        let mut required = DesignInputs::default();
        required.metal.required_min_Nm = 3.0;
        let Ok(SizingOutcome::Solved(point)) =
            crate::engine::sizing::solve(&required, FreeVariable::AxialLength, 3.0)
        else {
            panic!("3.0 N·m is reachable by the axial length")
        };
        assert_eq!(harness.panel.inputs(), &point.inputs);
        assert!((point.torque_hot_low_Nm - 3.0).abs() < 1e-6);
```

- `typing_in_the_results_search_does_not_defer_the_solve`:
  - line 3935: `harness.panel.inputs.metal.required_min_Nm = 3.0;`
  - line 3946: `harness.panel.inputs.metal.required_min_Nm = 3.5;`
- `an_unreachable_target_shows_the_best_value_and_its_overshoot`, line 3999: `harness.panel.inputs.metal.required_min_Nm = 50.0;`
- `a_share_link_carries_the_sizing_state`, line 4066: `harness.panel.inputs.metal.required_min_Nm = 3.0;`

- [ ] **Step 10: Format, then run the targeted tests and confirm they pass**

Commands, in order:
```
cargo fmt --manifest-path magcoupling-rs/Cargo.toml
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::sizing
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::session
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel::tests::the_sizing_target_is_the_requirement_s_row_and_one_undo_restores_it
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel::tests::a_file_whose_sizing_target_differs_loads_its_requirement_and_says_so
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel::tests::idle_frames_in_torque_to_magnets_keep_the_requirement_as_loaded
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::spreadsheet::tests::torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row
```
Expected: PASS, with every test in `gui::sizing` (15) and `gui::session` (all, 3 of them new) green.

- [ ] **Step 11: Run the wider suites for the touched modules and the gate's magcoupling checks**

Commands:
```
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::spreadsheet
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
cargo test --manifest-path magcoupling-rs/Cargo.toml
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
```
Expected: all pass.
- The `gui::panel` suite includes the glyph test, which now also checks the mismatch notice, plus the six edited tests.
- `app::tests::the_smoke_test_share_link_opens_its_design` still passes: the smoke link's target 2.5 equals its requirement 2.5, so no notice.
- The pre-zlib-rs link test `a_share_link_written_before_the_zlib_rs_backend_still_loads` passes unchanged.
- Clippy is clean: no unused `slider` or `solve` import, and `(range.min..=range.max).contains` instead of manual comparisons.

- [ ] **Step 12: Commit**

```
git add magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/gui/session.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/src/app.rs
git commit -m "$(cat <<'EOF'
feat(magcoupler): one torque number: the sizing target is the requirement

Decision G-2 (guidance slice 1, spec 4.4). SizingState loses target_Nm; the
Torque -> Magnets runner solves for metal.required_min_Nm over a search range
the panel passes (part of the solve key), and the target control is the
requirement's own input row (undo, files, links). Design files keep
"target_torque_Nm", equal to the requirement (DESIGN_VERSION 1; left out when
the requirement is not a positive torque, so such designs still round-trip).
A file whose target differs loads with the requirement and a notice in the
header.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
)"
```

---

**Existing tests that reference `target_Nm`, `DEFAULT_TARGET_NM` or `TARGET_RANGE_INPUT`, or that the new signatures break, and the exact fix:**

| Test (file:line) | Breaks because | Fix |
|---|---|---|
| `gui/sizing.rs:297` `the_default_state_is_the_forward_calculation_at_the_hot_minimum` | asserts `state.target_Nm` | replaced by `the_default_state_is_the_forward_calculation_with_the_axial_length_free` (Step 1) |
| `gui/sizing.rs:308` helper `inverse(variable, target_Nm)` | sets `target_Nm` | `inverse(variable)` plus a new `requiring(required_Nm)` helper (Step 1) |
| `gui/sizing.rs:317` `a_solve_waits_for_the_debounce_and_runs_once`, `:342` `an_edit_in_progress_...`, `:363` `solve_now_...`, `:391` `the_free_variable_s_own_value_...`, `:414` `a_solved_design_shows_the_solved_value` | runner methods take a range | `range` added to every call (Step 1) |
| `gui/sizing.rs:405` helper `solved(state)` | runner signature; target in state | `solved(inputs, state)` (Step 1) |
| `gui/sizing.rs:435` `an_unreachable_target_shows_the_best_value` | target 50.0 in state | `requiring(50.0)` (Step 1) |
| `gui/sizing.rs:460` `a_refused_solve_shows_why_and_keeps_the_inputs` | target -1.0 in state; old wording | requirement -1.0 and 0.0; text `"Sizing refused: the requirement must be a positive torque (it is -1.000 N·m)"` (Step 1) |
| `gui/session.rs:390` helper `edited()` | `target_Nm: 3.25` | `inputs.metal.required_min_Nm = 3.25` (Step 2); `a_design_file_names_every_input_and_the_sizing_state` (`json["sizing"]["target_torque_Nm"] == 3.25`) then passes unchanged |
| `gui/session.rs:628` `a_malformed_sizing_state_is_refused` (target 0, -1.5, "2.5") | nothing breaks | none: load still validates the target |
| `gui/spreadsheet.rs:1186` `torque_to_magnets_shows_the_sizing_and_notes_the_free_variable_s_row` | `target_Nm: 2.5` | requirement 3.25 in the inputs; asserts `Cell::number(3.25)` (Step 3) |
| `gui/panel.rs:3360` `a_loaded_design_file_replaces_the_design_and_can_be_undone` | `loaded.sizing.target_Nm` | `loaded.inputs.metal.required_min_Nm = 3.0` (Step 9) |
| `gui/panel.rs:3917` `leaving_torque_to_magnets_while_solving_writes_the_current_solution` | `sizing.target_Nm`; expected inputs | requirement 3.0, solved from inputs with requirement 3.0 (Step 9) |
| `gui/panel.rs:3935`, `:3946` `typing_in_the_results_search_does_not_defer_the_solve` | `sizing.target_Nm` | requirement 3.0 and 3.5 (Step 9) |
| `gui/panel.rs:3999` `an_unreachable_target_shows_the_best_value_and_its_overshoot` | `sizing.target_Nm` | requirement 50.0 (Step 9) |
| `gui/panel.rs:4066` `a_share_link_carries_the_sizing_state` | `sizing.target_Nm` | requirement 3.0 (Step 9) |
| `gui/panel.rs:3968` `the_json_export_in_torque_to_magnets_...` | `design_from_json` no longer imported at panel top level | the tests' session import now includes `design_from_json` (Step 3) |
| `app.rs:232` `the_smoke_test_share_link_opens_its_design` | `target_Nm: 2.5` | field removed (Step 8) |

`DEFAULT_TARGET_NM` appears only at its definition (`gui/sizing.rs:97`, removed). `TARGET_RANGE_INPUT` non-test uses: `gui/panel.rs:51,607` and `gui/spreadsheet.rs:39,316`, all renamed to `REQUIREMENT_INPUT`. Older plan documents in `docs/superpowers/plans/` mention the old names as history; they are not changed.

**Decisions inside this task (flag them in review):**
1. **`shown(.., range)`.** The contract gives it a range. A value from an earlier search outside `range` is not shown; the inputs show until the re-solve. In Task 3 every solved value lies inside its own range, so behaviour is unchanged. In Task 7, a raised minimum OD never shows a stale ring size below it. A value still inside the range keeps showing, so there is no flicker.
2. **Save writes `target_torque_Nm` only when the requirement is a positive finite torque.** This is a narrow deviation from "save still writes" so that designs with a requirement of 0 or less (reachable only through a file) still round-trip. Every reader refuses such a target.
3. **The notice quotes the file's numbers in full** (Rust `{}`), so "3.25 N·m" rather than "3.250 N·m". Differences below four significant digits never read as equal numbers.
4. **The Torque -> Magnets target control is the requirement's own input row** (`input_row_ui`, engine label "Required minimum service pull-out torque"). It is not a second labelled slider: one edit path, one name. `TARGET_LABEL` stays only as the spreadsheet Summary's row label.
5. **`SolveKey` stores the range as bit patterns,** so a NaN bound is refused once rather than restarting the debounce every frame.
6. **The forced switch to forward mode when the Size tab is selected (`panel.rs:686-690`) stays.** Task 7 removes it.
7. **Performance:** `FreeVariable::range()` rebuilds the input rows on each call. That is a handful of calls per frame, and only in Torque -> Magnets. This is acceptable now; Task 7 can cache it inside `sizing_range()` if profiling says so.
8. **Docs** are left to Task 10.

---

### Task 4: Check guides, design status and the plain-text guard (`gui/guidance.rs`)

> **Amendments:** apply "T4 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Create: `magcoupling-rs/src/gui/guidance.rs`. This holds the module docs, the implementation and 19 tests in its own `mod tests`.
- Modify: `magcoupling-rs/src/gui/mod.rs` at line 16. Add `pub mod guidance;` after `pub mod geometry_view;`.
- Modify: `magcoupling-rs/src/engine/temperature.rs` in two places:
  - after `pub const RING_OUTER` (about line 729), add `pub const CURE_MARGIN_MIN_C`;
  - in the summary verdict (about line 1370), `cure_margin >= 10.0` becomes `cure_margin >= CURE_MARGIN_MIN_C`.

  This is a refactor with no change in behaviour or strings. It keeps the threshold in one place: the temperature guide uses the same value the verdict applies, so the guide carries no second literal.
- Test: `magcoupling-rs/src/gui/guidance.rs` `mod tests`. Covers spec section 6 tests 1–4 and the glyph part of 11, plus a per-fix direction test and a test that the temperature terms reproduce the engine verdict.

**Interfaces:**
- **Consumes:**
  - `crate::gui::dashboard::{Level, check_level, result_info, DASHBOARD, CHECKS}`
  - `crate::gui::explorer::reviewed_note`
  - `crate::gui::inputs::InputCatalogue`
  - `crate::gui::format::{format_value, with_unit}`
  - `crate::engine::model::{blocks_fit, pitch_share}`
  - `crate::engine::clamps::SCREW_SIZES`
  - `crate::engine::assumptions::ASSUMPTIONS`
  - `crate::engine::meta::{InputSet, ResultSet, Value}`
  - test helpers `crate::gui::test_support::{assert_glyphs, short_magnets}`
- **Produces (contract, names unchanged):**
  - `pub enum Direction { Raise, Lower }`
  - `pub enum Learn { Note(&'static str), Assumption(&'static str), None(&'static str) }`
  - `pub struct CheckGuide { pub check, pub label, pub problem: fn(&DesignInputs, &DesignResults) -> Option<String>, pub why, pub fixes: &'static [(&'static str, Direction)], pub learn }`
  - `pub const FIT_CHECK: &str = "guidance.blocks_fit"`
  - `pub static CHECK_GUIDES: [CheckGuide; 8]`
  - `pub fn guide(check: &str) -> Option<&'static CheckGuide>`
  - `pub fn level(guide: &CheckGuide, inputs: &DesignInputs, results: &DesignResults) -> crate::gui::dashboard::Level`
  - `pub fn fix_text(path: &str, direction: Direction) -> String`
  - `pub struct DesignStatus { pub bad: usize, pub caution: usize, pub total: usize, pub is_default: bool }`
  - `pub fn design_status(inputs, results) -> DesignStatus`
  - `pub fn design_status_text(status: &DesignStatus) -> String`
  - `pub fn plain_text_violations(text: &str) -> Vec<&'static str>`
- **Produces (additions):**
  - `pub const WORKBOOK_EXAMPLE: &str = "Workbook example design"` and `pub const YOUR_DESIGN: &str = "Your design"`, so T8 and T9 can match the status line.
  - `pub const BANNED: [&str; 6]`: the patterns `plain_text_violations` reports.
  - `pub const crate::engine::temperature::CURE_MARGIN_MIN_C: f64 = 10.0`.

**Contract against spec:**
- The contract wins over spec 4.1. The `problem` arguments are `(&DesignInputs, &DesignResults)` and the field is `caution`, not `warn`.
- Spec test 2 says a fix input may have "a slider or choices". "Raise" and "Lower" need a number, so a fix must have a slider range, and no guide uses a choice input.

**Physics decisions (for the G-3 reviewer):**
- **Method.** Every fix was checked against the engine with a scratch crate (no repo edits). The `every_fix_moves_its_check_toward_passing` test now locks each direction in.
- **Hot-low torque.** Fixes: raise magnet length, lower face gap.
  - T ∝ L − c_end·τp, so magnet length is strictly increasing. Margin vs the requirement: −0.215 N·m at the defaults → +0.046 at 14 mm → +3.26 at 30 mm.
  - Face gap: 1.4 → 1.0 mm moves the margin −0.215 → +0.105.
  - Left out, ring radius: not monotone at 10 poles. 10.15 → 13 mm gives +0.086, 20 mm gives −0.342, 30 mm gives +0.007. This is trap 1 in spec section 8.
  - Left out, pole count: 12 and 14 raise the torque but do not fit at the default ring, and 16 falls again. This is trap 2.
  - Left out: the variation allowance (an assumption) and the operating temperature (a condition).
- **Running clearance.** Fixes: raise face gap, lower sleeve, lower liner.
  - The clearance gains the face gap one for one: 1.4 → 2.0 mm moves the margin −0.303 → +0.297.
  - Sleeve 0.1 → 0 mm adds 0.1; liner 0.2 → 0 mm adds 0.2.
  - Pole count has no effect. Ring radius is weak (+0.1 at 15 mm), so it is left out.
  - The movement allowances are marked UNCONFIRMED placeholders, so they are not offered as fixes.
- **Cup wall.** Fix: raise the wall only.
  - The engine rule is t = B_gap·τp/(π·B_des).
  - More poles also lower the need (1.60 mm at 12 poles) but do not fit at the default ring.
  - A wider gap lowers the need but costs torque.
  - Both are left out.
- **Temperature.** Fix: lower the slip fault trip time.
  - 60 → 0.1 s moves the magnet margin −10.9 → +23.2 °C on the slip-fault design.
  - Conductance and driving rise are assumptions; hot ambient, slip speed and minimum temperature are requirements.
  - Grade and adhesive are text or choice inputs, so they have no direction.
- **Shaft clamp.** Fixes: raise boss OD, raise clamp length.
  - "Some screw size works" can only improve as the boss grows: wall, head seat, grip and thread all grow with it.
  - It can only improve as the clamp grows: more screws fit along it.
- **Space claim.** Fixes: lower magnet length, lower cap OD, lower ring.
  - None of these ever makes any axis worse.
  - Which one helps depends on what sets the axis, and the Problem names the axis.
- **End effect.** Fixes: raise magnet length, raise pole count.
  - f_end = 1 − c_end·τp/L, with τp = 2πRg/N.
  - On the short design, 10 → 40 poles moves f_end −1.20 → +0.45.
- **Fit.** Fixes: raise ring, lower pole count.
  - Flat width is 2a·tan(π/N); an arc's share of the circle is w·N/(2π·r_mid). A bigger ring or fewer blocks widens each place.
- **Learn links:**

  | Guide | Link |
  |---|---|
  | Hot-low torque | Assumption `production_variation` (its rationale is the hot-low formula) |
  | Cup wall | Note `a5.low_saturation` (it derives the wall rule) |
  | Temperature limits | Note `demagnetization` (the magnet limit governs at the defaults) |
  | Shaft clamp | Note `clamp_preload` |
  | End-effect model | Note `end_effect` |
  | Running clearance, Space claim, Magnets fit | `Learn::None`, each with its reason |

- **Wording.**
  - None of the text says any of the spec section 8 traps.
  - The cup wall Why says the model treats the steel as unsaturated, so it promises no torque the model does not compute.
- **Note for T8.**
  - `FIT_CHECK` has no `result_info`, and `dashboard::dashboard_lines` panics with "DASHBOARD: no result" for such a path. So the fit row must be drawn from the guide, not added to `DASHBOARD`.
  - `level()` maps "no badge" to `Good`: a check that does not apply, or a check greyed while the end effect is out of range.

- [ ] **Step 1: Add the engine constant (refactor; tests green before and after)**

  Run first: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib engine::temperature`. Expected: `test result: ok. 25 passed`.

  In `magcoupling-rs/src/engine/temperature.rs`, replace

  ```rust
  /// See [`RING_INNER`].
  pub const RING_OUTER: &str = "outer";
  ```

  with

  ```rust
  /// See [`RING_INNER`].
  pub const RING_OUTER: &str = "outer";

  /// The smallest cure margin the temperature verdict accepts [°C] (Temperature design!C25):
  /// the adhesive's cure temperature at least this far below the single-ring demagnetization
  /// onset. The GUI's temperature guide quotes it (`gui::guidance`).
  pub const CURE_MARGIN_MIN_C: f64 = 10.0;
  ```

  and in `compute` replace

  ```rust
          && cure_margin >= 10.0
          && cold_ok; // E20: always true unless the coercivity falls on cooling
  ```

  with

  ```rust
          && cure_margin >= CURE_MARGIN_MIN_C
          && cold_ok; // E20: always true unless the coercivity falls on cooling
  ```

  Run again: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib engine::temperature`. Expected: `test result: ok. 25 passed`. `verdict_accepts_a_cure_margin_of_exactly_ten` still passes.

- [ ] **Step 2: Register the module and write the failing tests**

  In `magcoupling-rs/src/gui/mod.rs` replace

  ```rust
  pub mod geometry_view;
  pub mod history;
  ```

  with

  ```rust
  pub mod geometry_view;
  pub mod guidance;
  pub mod history;
  ```

  Create `magcoupling-rs/src/gui/guidance.rs` with the module docs, then the test module. Step 4 inserts the implementation between them.

  ```rust
  //! Check guides (guidance slice 1, spec 3.4 and 4.1): under a dashboard check that fails or
  //! asks for a look, what is wrong in this design's numbers (the Problem), why the check
  //! matters (the Why, one plain sentence) and which inputs to move which way (the Fix); where
  //! each row's "Why:" link leads ([`Learn`]); and the design status line that counts the
  //! failing checks ([`design_status`]).
  //!
  //! [`CHECK_GUIDES`] holds one guide per check of the dashboard's rows
  //! ([`crate::gui::dashboard::DASHBOARD`]) and one for the Magnets fit row ([`FIT_CHECK`]),
  //! which no result holds: the blocks fit is [`blocks_fit`] of the model's results. A guide's
  //! level is the dashboard's own badge ([`check_level`]); its Problem quotes the live results
  //! and inputs, each number as the readouts format it, and is `None` while the check passes.
  //! A fix is an input path and a direction; its text takes the input's label from the
  //! [`InputCatalogue`] ([`fix_text`]), so no guide types a label. Fixes are design changes:
  //! each has a slider, none is a model assumption, and a test moves each to its slider's end
  //! on a design that fails its check and sees the check's margin grow.
  //!
  //! The engine's verdict texts, the results table and the exports keep their wording (locked by
  //! `tests/python_schema.rs`): this is a GUI layer over them, in plain words
  //! ([`plain_text_violations`] lists what it must not say). Pure functions and data: no egui.

  #[cfg(test)]
  mod tests {
      use super::*;
      use crate::compute_all;
      use crate::engine::assumptions::ASSUMPTIONS;
      use crate::gui::dashboard::{CHECKS, DASHBOARD};
      use crate::gui::explorer::reviewed_note;
      use crate::gui::test_support::{assert_glyphs, short_magnets};
      use std::collections::BTreeSet;

      /// The defaults with `edit` applied.
      fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
          let mut inputs = DesignInputs::default();
          edit(&mut inputs);
          inputs
      }

      /// Passes all eight checks: the defaults with a wider gap (clearance), a thicker cup wall
      /// (back iron) and a 2 N·m requirement (hot-low torque 2.007 N·m).
      fn passing() -> DesignInputs {
          design(|i| {
              i.metal.face_gap_mm = 1.8;
              i.metal.cup_wall_corner_mm = 2.0;
              i.metal.required_min_Nm = 2.0;
          })
      }

      /// A hot day starting above the magnet limit: the temperature verdict asks for a look.
      fn hot_day() -> DesignInputs {
          design(|i| {
              i.temperature.duty.hot_ambient_C = 80.0;
              i.temperature.duty.driving_rise_C = 60.0;
          })
      }

      /// A long unbroken slip at a high slip speed: only the fault peak passes the magnet limit.
      fn slip_fault() -> DesignInputs {
          design(|i| {
              i.temperature.duty.fault_trip_s = 60.0;
              i.metal.slip_rpm = 6000.0;
          })
      }

      /// The fourth adhesive: its 60 °C limit is below the hot-day start.
      fn cool_adhesive() -> DesignInputs {
          design(|i| i.temperature.adhesive.selected = 4)
      }

      /// The second adhesive (cured at 120 °C) with a stored single-ring reverse field that puts
      /// the single-ring onset about 6 °C above the cure: only the cure margin fails.
      fn hot_cure() -> DesignInputs {
          design(|i| {
              i.temperature.adhesive.selected = 2;
              i.temperature.demag.h_rev_single_ring_kA_m = 700.0;
          })
      }

      /// Manual Y30 ferrite blocks: the cold demagnetization check fails (E20).
      fn ferrite() -> DesignInputs {
          design(|i| {
              i.coupling.magnets.part_inner.clear();
              i.coupling.magnets.part_outer.clear();
              i.coupling.magnets.grade_inner = "Y30".to_owned();
              i.coupling.magnets.grade_outer = "Y30".to_owned();
          })
      }

      /// No screw fits a 12 mm boss.
      fn small_boss() -> DesignInputs {
          design(|i| i.clamps.boss_od_mm = 12.0)
      }

      /// No screw fits a 3 mm clamp.
      fn short_clamp() -> DesignInputs {
          design(|i| i.clamps.clamp_length_mm = 3.0)
      }

      /// 50.8 mm magnets: past the overall length and the large-diameter bay.
      fn long_magnets() -> DesignInputs {
          design(|i| i.coupling.magnets.axial_length_mm = Some(50.8))
      }

      /// A 44 mm cap: past the diameter claim, the cap setting the diameter.
      fn large_cap() -> DesignInputs {
          design(|i| i.metal.cap_od_mm = 44.0)
      }

      /// A 12 mm ring: past the diameter claim, the cup setting the diameter.
      fn large_ring() -> DesignInputs {
          design(|i| i.coupling.inner_back_apothem_mm = 12.0)
      }

      /// A diameter claim that is not a number: the space claim asks for a look.
      fn unknown_claim() -> DesignInputs {
          design(|i| i.metal.max_diameter_mm = f64::NAN)
      }

      /// 40 flat blocks per ring at the default ring: they do not fit.
      fn crowded() -> DesignInputs {
          design(|i| i.coupling.npole = 40)
      }

      /// 40 arcs per ring at the default ring: they overlap.
      fn crowded_arcs() -> DesignInputs {
          design(|i| {
              i.coupling.npole = 40;
              i.coupling.faceted = 0;
          })
      }

      /// No back iron: the cup wall check does not apply.
      fn no_back_iron() -> DesignInputs {
          design(|i| i.coupling.backiron = 0)
      }

      /// Every design the tests use: the defaults, a passing design and, for every check, one
      /// that fails it (some fail several).
      fn designs() -> Vec<(&'static str, DesignInputs)> {
          vec![
              ("defaults", DesignInputs::default()),
              ("passing", passing()),
              ("hot_day", hot_day()),
              ("slip_fault", slip_fault()),
              ("cool_adhesive", cool_adhesive()),
              ("hot_cure", hot_cure()),
              ("ferrite", ferrite()),
              ("small_boss", small_boss()),
              ("short_clamp", short_clamp()),
              ("long_magnets", long_magnets()),
              ("large_cap", large_cap()),
              ("large_ring", large_ring()),
              ("unknown_claim", unknown_claim()),
              ("short_magnets", short_magnets()),
              ("crowded", crowded()),
              ("crowded_arcs", crowded_arcs()),
              ("no_back_iron", no_back_iron()),
          ]
      }

      /// The Problem of `check` for `inputs`.
      fn problem(check: &str, inputs: &DesignInputs) -> Option<String> {
          (guide(check).unwrap().problem)(inputs, &compute_all(inputs))
      }

      /// Every text the guides draw: labels, Whys, the reasons of the links without a note,
      /// fixes, every Problem of every test design and its status line.
      fn every_text() -> Vec<String> {
          let mut texts = Vec::new();
          for g in &CHECK_GUIDES {
              texts.push(g.label.to_owned());
              texts.push(g.why.to_owned());
              if let Learn::None(reason) = g.learn {
                  texts.push(reason.to_owned());
              }
              for &(path, direction) in g.fixes {
                  texts.push(fix_text(path, direction));
              }
          }
          for (_, inputs) in designs() {
              let results = compute_all(&inputs);
              for g in &CHECK_GUIDES {
                  texts.extend((g.problem)(&inputs, &results));
              }
              texts.push(design_status_text(&design_status(&inputs, &results)));
          }
          texts
      }

      #[test]
      fn the_guides_cover_every_dashboard_check_and_the_fit_row() {
          let dashboard: BTreeSet<&str> = DASHBOARD.iter().filter_map(|&(_, check)| check).collect();
          let mut guided: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.check).collect();
          assert_eq!(guided.len(), CHECK_GUIDES.len(), "one guide per check");
          assert!(guided.remove(FIT_CHECK), "the Magnets fit row has a guide");
          assert_eq!(guided, dashboard);
          for g in &CHECK_GUIDES {
              // Every guide names a real check: a design check of the results, or the fit row.
              assert!(
                  g.check == FIT_CHECK || CHECKS.contains(&g.check),
                  "{}",
                  g.check
              );
              assert_eq!(guide(g.check).map(|found| found.check), Some(g.check));
              assert!(!g.label.is_empty() && !g.why.is_empty(), "{}", g.check);
          }
          // The fit row's path is no result, and nothing but a check has a guide.
          assert!(result_info(FIT_CHECK).is_none());
          assert!(guide("model.pullout_Nm").is_none());
          assert!(guide("").is_none());
          let labels: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.label).collect();
          assert_eq!(labels.len(), CHECK_GUIDES.len(), "labels are distinct");
          assert_eq!(guide(FIT_CHECK).unwrap().label, "Magnets fit");
      }

      #[test]
      fn every_fix_is_a_design_input_with_a_slider() {
          let catalogue = InputCatalogue::get();
          for g in &CHECK_GUIDES {
              assert!(!g.fixes.is_empty(), "{}: a guide names a fix", g.check);
              let paths: BTreeSet<&str> = g.fixes.iter().map(|&(path, _)| path).collect();
              assert_eq!(paths.len(), g.fixes.len(), "{}: a fix once", g.check);
              for &(path, _) in g.fixes {
                  let entry = catalogue
                      .entry(path)
                      .unwrap_or_else(|| panic!("{}: {path} is no input", g.check));
                  assert!(
                      entry.meta.range.is_some(),
                      "{}: {path} has no slider to raise or lower",
                      g.check
                  );
                  assert!(
                      !entry.meta.assumption,
                      "{}: {path} is a model assumption, not a design change",
                      g.check
                  );
              }
          }
      }

      #[test]
      fn every_learn_link_resolves() {
          for g in &CHECK_GUIDES {
              match g.learn {
                  Learn::Note(id) => assert!(
                      reviewed_note(id).is_some(),
                      "{}: note {id} is missing or not reviewed",
                      g.check
                  ),
                  Learn::Assumption(id) => assert!(
                      ASSUMPTIONS.iter().any(|a| a.id == id),
                      "{}: no assumption {id}",
                      g.check
                  ),
                  Learn::None(reason) => {
                      assert!(!reason.trim().is_empty(), "{}: a reason", g.check)
                  }
              }
          }
      }

      #[test]
      fn fix_text_names_the_input_by_its_label() {
          assert_eq!(
              fix_text("metal.cup_wall_corner_mm", Direction::Raise),
              "Raise Minimum outer return ring wall"
          );
          assert_eq!(
              fix_text("coupling.npole", Direction::Lower),
              "Lower Number of poles per ring"
          );
          let catalogue = InputCatalogue::get();
          for g in &CHECK_GUIDES {
              for &(path, direction) in g.fixes {
                  let label = catalogue.entry(path).unwrap().meta.label;
                  assert_eq!(
                      fix_text(path, direction),
                      format!("{} {label}", direction.word())
                  );
              }
          }
          // A path that is no input reads as itself rather than failing the frame.
          assert_eq!(fix_text("no.such", Direction::Raise), "Raise no.such");
      }

      /// How far `check` is from passing, larger is better: the margin its verdict compares (the
      /// clamp: whether a screw size works; the space claim: minus the overshoots summed; the
      /// fit: the narrower ring's flat slack).
      fn margin(check: &str, inputs: &DesignInputs, results: &DesignResults) -> f64 {
          let (m, t, h) = (&results.model, &results.temperature, &results.housing);
          match check {
              HOT_MIN => results.metal.torque_hot_low_Nm - inputs.metal.required_min_Nm,
              CLEARANCE => results.metal.min_running_clearance_mm - inputs.metal.residual_target_mm,
              CUP_WALL => {
                  results.materials.cup_wall_corner_mm
                      - results.materials.backiron_thickness_needed_mm
              }
              TEMPERATURE => t
                  .summary
                  .margin_hot_day_C
                  .min(t.magnet_life.margin_limit_C)
                  .min(t.adhesive_life.margin_C),
              CLAMP => {
                  if results.clamps.index > 0 {
                      1.0
                  } else {
                      0.0
                  }
              }
              SPACE_CLAIM => -(h.diameter_overshoot_mm + h.length_overshoot_mm + h.bay_overshoot_mm),
              END_EFFECT => m.f_end,
              FIT_CHECK => (m.inner_flat_width_mm - m.inner_width_mm)
                  .min(m.outer_flat_width_mm - m.outer_width_mm),
              _ => panic!("no margin for {check}"),
          }
      }

      /// `inputs` with the input at `path` at its slider's end in `direction`.
      fn at_slider_end(inputs: &DesignInputs, path: &str, direction: Direction) -> DesignInputs {
          let entry = InputCatalogue::get().entry(path).unwrap();
          let range = entry.meta.range.unwrap();
          let end = match direction {
              Direction::Raise => range.max,
              Direction::Lower => range.min,
          };
          let value = match entry.default {
              Value::Int(_) => Value::Int(end as i64),
              _ => Value::Num(end),
          };
          let mut moved = inputs.clone();
          moved.set(path, value).unwrap();
          moved
      }

      #[test]
      fn every_fix_moves_its_check_toward_passing() {
          // (check, a design that fails it, a fix): the fix's input at its slider's end in the
          // fix's direction raises the check's margin. The plan's engine probes found each fix
          // monotone over its slider; the inputs left out are not (the ring radius against the
          // hot-low torque, the pole count against it) or are assumptions.
          let defaults = DesignInputs::default;
          let cases: Vec<(&str, DesignInputs, &str, Direction)> = vec![
              (
                  HOT_MIN,
                  defaults(),
                  "coupling.magnets.axial_length_mm",
                  Direction::Raise,
              ),
              (HOT_MIN, defaults(), "metal.face_gap_mm", Direction::Lower),
              (CLEARANCE, defaults(), "metal.face_gap_mm", Direction::Raise),
              (CLEARANCE, defaults(), "metal.sleeve_mm", Direction::Lower),
              (CLEARANCE, defaults(), "metal.liner_mm", Direction::Lower),
              (
                  CUP_WALL,
                  defaults(),
                  "metal.cup_wall_corner_mm",
                  Direction::Raise,
              ),
              (
                  TEMPERATURE,
                  slip_fault(),
                  "temperature.duty.fault_trip_s",
                  Direction::Lower,
              ),
              (CLAMP, small_boss(), "clamps.boss_od_mm", Direction::Raise),
              (
                  CLAMP,
                  short_clamp(),
                  "clamps.clamp_length_mm",
                  Direction::Raise,
              ),
              (
                  SPACE_CLAIM,
                  long_magnets(),
                  "coupling.magnets.axial_length_mm",
                  Direction::Lower,
              ),
              (
                  SPACE_CLAIM,
                  large_cap(),
                  "metal.cap_od_mm",
                  Direction::Lower,
              ),
              (
                  SPACE_CLAIM,
                  large_ring(),
                  "coupling.inner_back_apothem_mm",
                  Direction::Lower,
              ),
              (
                  END_EFFECT,
                  short_magnets(),
                  "coupling.magnets.axial_length_mm",
                  Direction::Raise,
              ),
              (
                  END_EFFECT,
                  short_magnets(),
                  "coupling.npole",
                  Direction::Raise,
              ),
              (
                  FIT_CHECK,
                  crowded(),
                  "coupling.inner_back_apothem_mm",
                  Direction::Raise,
              ),
              (FIT_CHECK, crowded(), "coupling.npole", Direction::Lower),
          ];
          // The table holds every fix of every guide, once.
          let key = |check: &'static str, path: &'static str, direction: Direction| {
              (check, path, direction == Direction::Raise)
          };
          let listed: BTreeSet<_> = cases.iter().map(|&(c, _, p, d)| key(c, p, d)).collect();
          let fixes: BTreeSet<_> = CHECK_GUIDES
              .iter()
              .flat_map(|g| g.fixes.iter().map(move |&(p, d)| key(g.check, p, d)))
              .collect();
          assert_eq!(listed, fixes);
          assert_eq!(listed.len(), cases.len());
          for (check, failing_design, path, direction) in cases {
              let before = compute_all(&failing_design);
              assert_ne!(
                  level_of(check, &failing_design, &before),
                  Level::Good,
                  "{check}: the case's design fails it"
              );
              let moved = at_slider_end(&failing_design, path, direction);
              let after = compute_all(&moved);
              let (from, to) = (
                  margin(check, &failing_design, &before),
                  margin(check, &moved, &after),
              );
              assert!(
                  to > from,
                  "{check}: {} moves the margin {from} -> {to}",
                  fix_text(path, direction)
              );
          }
      }

      #[test]
      fn plain_text_violations_names_each_banned_pattern() {
          assert_eq!(
              plain_text_violations("Too thin: raise Metal design C122 to at least 2.0 mm"),
              ["C\\d{2,3}"]
          );
          assert_eq!(plain_text_violations("E3 E7 E8"), ["E\\d{1,2}"]);
          assert_eq!(plain_text_violations("corrections E20."), ["E\\d{1,2}"]);
          assert_eq!(
              plain_text_violations("Calculator!C93"),
              ["C\\d{2,3}", "Calculator!"]
          );
          assert_eq!(
              plain_text_violations("Rust-only result (no workbook cell)"),
              ["workbook", "Rust-only"]
          );
          assert_eq!(plain_text_violations("Spec Addendum A5"), ["Addendum"]);
          assert_eq!(plain_text_violations("WORKBOOK"), ["workbook"]);
          // The status line's phrase excuses itself only, and only as written.
          assert_eq!(
              plain_text_violations("Workbook example design from the workbook"),
              ["workbook"]
          );
          assert_eq!(plain_text_violations("workbook example"), ["workbook"]);
          for clean in [
              "Workbook example design: fails 3 of 8 checks",
              "2.000e7 mm",
              "E123",
              "C1",
              "50.00 °C",
              "ISO 4762 M4 x 14, class 12.9",
              "Raise Minimum outer return ring wall",
              "",
          ] {
              assert!(plain_text_violations(clean).is_empty(), "{clean:?}");
          }
      }

      #[test]
      fn no_guide_text_carries_workbook_jargon() {
          for text in every_text() {
              assert_eq!(plain_text_violations(&text), Vec::<&str>::new(), "{text:?}");
          }
      }

      #[test]
      fn every_guide_text_has_glyphs_in_the_default_fonts() {
          let ctx = egui::Context::default();
          let _ = ctx.run(egui::RawInput::default(), |_| {});
          for text in every_text() {
              assert_glyphs(&ctx, &text, "guidance");
          }
      }

      #[test]
      fn each_problem_is_some_exactly_when_its_check_fails() {
          use Level::{Bad, Caution, Good};
          for (name, inputs) in designs() {
              let results = compute_all(&inputs);
              for g in &CHECK_GUIDES {
                  let badge = level(g, &inputs, &results);
                  assert_eq!(
                      (g.problem)(&inputs, &results).is_some(),
                      badge != Good,
                      "{name}: {}",
                      g.check
                  );
                  // The level is the dashboard's badge of the check (green where it gives none).
                  if g.check != FIT_CHECK {
                      assert_eq!(
                          check_level(&results, g.check).unwrap_or(Good),
                          badge,
                          "{name}: {}",
                          g.check
                      );
                  }
              }
          }
          // A passing design passes every guide; the defaults fail three, all red.
          let inputs = passing();
          let results = compute_all(&inputs);
          for g in &CHECK_GUIDES {
              assert_eq!(level(g, &inputs, &results), Good, "{}", g.check);
          }
          let inputs = DesignInputs::default();
          let results = compute_all(&inputs);
          let failing: Vec<(&str, Level)> = CHECK_GUIDES
              .iter()
              .map(|g| (g.check, level(g, &inputs, &results)))
              .filter(|&(_, badge)| badge != Good)
              .collect();
          assert_eq!(failing, [(HOT_MIN, Bad), (CLEARANCE, Bad), (CUP_WALL, Bad)]);
          // Each failing design fails the check it was built for, at the level the dashboard gives.
          for (inputs, check, want) in [
              (hot_day(), TEMPERATURE, Caution),
              (slip_fault(), TEMPERATURE, Caution),
              (cool_adhesive(), TEMPERATURE, Caution),
              (hot_cure(), TEMPERATURE, Caution),
              (ferrite(), TEMPERATURE, Caution),
              (small_boss(), CLAMP, Bad),
              (short_clamp(), CLAMP, Bad),
              (long_magnets(), SPACE_CLAIM, Bad),
              (large_cap(), SPACE_CLAIM, Bad),
              (large_ring(), SPACE_CLAIM, Bad),
              (unknown_claim(), SPACE_CLAIM, Caution),
              (short_magnets(), END_EFFECT, Bad),
              (crowded(), FIT_CHECK, Bad),
              (crowded_arcs(), FIT_CHECK, Bad),
          ] {
              let results = compute_all(&inputs);
              assert_eq!(level_of(check, &inputs, &results), want, "{check}");
          }
      }

      #[test]
      fn the_problems_quote_the_live_numbers() {
          let defaults = DesignInputs::default();
          assert_eq!(
              problem(HOT_MIN, &defaults).unwrap(),
              "Hot-low torque 2.285 N·m is below the required 2.500 N·m: the pull-out of 2.688 N·m at 50.00 °C, less the 15 % variation allowance."
          );
          assert_eq!(
              problem(CLEARANCE, &defaults).unwrap(),
              "Sleeve-to-liner gap 0.6768 mm less 0.7800 mm of movement allowances leaves -0.1032 mm, below the 0.2000 mm target."
          );
          assert_eq!(
              problem(CUP_WALL, &defaults).unwrap(),
              "Cup wall at the pocket corners is 1.800 mm; the flux it carries needs 1.919 mm."
          );
          // The numbers follow an edit: a 2.4 N·m requirement still fails, quoted as such.
          let edited = design(|i| i.metal.required_min_Nm = 2.4);
          assert!(
              problem(HOT_MIN, &edited)
                  .unwrap()
                  .contains("below the required 2.400 N·m"),
              "{:?}",
              problem(HOT_MIN, &edited)
          );
          // A 0.1 mm liner: the gap and the result after the allowances move by its 0.1 mm.
          let thinner = design(|i| i.metal.liner_mm = 0.1);
          let text = problem(CLEARANCE, &thinner).unwrap();
          assert!(text.starts_with("Sleeve-to-liner gap 0.7768 mm"), "{text}");
          assert!(text.contains("leaves -0.003174 mm"), "{text}");
      }

      #[test]
      fn the_temperature_problem_names_each_failing_term() {
          assert_eq!(
              problem(TEMPERATURE, &hot_day()).unwrap(),
              "Hot-day start 140.0 °C is not below the 93.06 °C limit. The slip-fault peak of 141.0 °C reaches the 93.06 °C magnet limit. The slip-fault peak of 141.0 °C reaches the 120.0 °C adhesive limit."
          );
          assert_eq!(
              problem(TEMPERATURE, &slip_fault()).unwrap(),
              "The slip-fault peak of 104.0 °C reaches the 93.06 °C magnet limit."
          );
          assert_eq!(
              problem(TEMPERATURE, &cool_adhesive()).unwrap(),
              "Hot-day start 65.00 °C is not below the 60.00 °C limit. The slip-fault peak of 65.96 °C reaches the 60.00 °C adhesive limit."
          );
          assert_eq!(
              problem(TEMPERATURE, &hot_cure()).unwrap(),
              "The adhesive cures at 120.0 °C, 5.955 °C below the 126.0 °C single-ring demagnetization onset; the margin must be at least 10.00 °C."
          );
          assert_eq!(
              problem(TEMPERATURE, &ferrite()).unwrap(),
              "The minimum magnet temperature -40.00 °C is below the 335.7 °C cold demagnetization limit."
          );
          assert_eq!(problem(TEMPERATURE, &DesignInputs::default()), None);
      }

      #[test]
      fn the_temperature_terms_reproduce_the_engine_verdict() {
          // The terms are read from the margins with the verdict's own comparisons (and its
          // cure-margin minimum): over designs on both sides of every term, they are empty
          // exactly when the verdict passes. The single-ring fields step the cure margin across
          // its minimum (666.36 kA/m puts it at 10 °C, as the engine's own test finds).
          let mut cases = designs();
          for field in [640.0, 650.0, 660.0, 666.3586306326292, 667.0, 680.0, 700.0] {
              cases.push((
                  "single-ring field",
                  design(|i| {
                      i.temperature.adhesive.selected = 2;
                      i.temperature.demag.h_rev_single_ring_kA_m = field;
                  }),
              ));
          }
          for (name, inputs) in cases {
              let results = compute_all(&inputs);
              let passes = check_level(&results, TEMPERATURE) == Some(Level::Good);
              assert_eq!(
                  temperature_terms(&inputs, &results).is_empty(),
                  passes,
                  "{name}: {}",
                  results.temperature.summary.verdict
              );
          }
      }

      #[test]
      fn the_clamp_problem_names_the_screw_range_and_the_torque() {
          let text = problem(CLAMP, &small_boss()).unwrap();
          assert_eq!(
              text,
              "No screw size from M2.5 to M6 fits the boss and the clamp length with enough screws to hold 7.647 N·m: the cold-high torque 3.823 N·m times a safety factor of 2.000."
          );
          assert_eq!(problem(CLAMP, &DesignInputs::default()), None);
      }

      #[test]
      fn the_space_claim_problem_names_each_axis_past_its_claim() {
          assert_eq!(
              problem(SPACE_CLAIM, &long_magnets()).unwrap(),
              "Overall length 69.90 mm is 34.90 mm over the 35.00 mm claim. Large-diameter stack 56.90 mm is 36.90 mm over the 20.00 mm bay."
          );
          assert_eq!(
              problem(SPACE_CLAIM, &large_cap()).unwrap(),
              "Rotating diameter 44.00 mm is 1.000 mm over the 43.00 mm claim."
          );
          assert_eq!(
              problem(SPACE_CLAIM, &unknown_claim()).unwrap(),
              "Rotating diameter cannot be checked: it or its claim is not a number."
          );
          assert_eq!(problem(SPACE_CLAIM, &DesignInputs::default()), None);
      }

      #[test]
      fn out_of_range_end_effect_gives_its_own_problem_and_greys_the_others() {
          // Spec 5: with the end-effect model out of range the hot-low torque has no prediction,
          // so the hot minimum and the clamp (computed from it) give no Problem; the end-effect
          // guide says what to change.
          let inputs = short_magnets();
          let results = compute_all(&inputs);
          assert_eq!(
              problem(END_EFFECT, &inputs).unwrap(),
              "Magnets 2.000 mm long with a pole pitch of 8.809 mm give an end-effect factor of -1.202; the model needs magnets longer than 4.405 mm."
          );
          for check in [HOT_MIN, CLAMP] {
              assert_eq!(level(guide(check).unwrap(), &inputs, &results), Level::Good);
              assert_eq!(problem(check, &inputs), None, "{check}");
          }
          assert_eq!(problem(END_EFFECT, &DesignInputs::default()), None);
      }

      #[test]
      fn the_fit_problem_names_the_rings_that_do_not_fit() {
          assert_eq!(
              problem(FIT_CHECK, &crowded()).unwrap(),
              "40 blocks per ring do not fit at this ring size. Inner blocks are 6.350 mm wide on 1.598 mm flats. Outer blocks are 6.350 mm wide on 2.317 mm flats."
          );
          assert_eq!(
              problem(FIT_CHECK, &crowded_arcs()).unwrap(),
              "40 blocks per ring do not fit at this ring size. Inner blocks need 344.5 % of the circle at their mid-radius. Outer blocks need 247.9 % of the circle at their mid-radius."
          );
          // 12 flat blocks: the inner flats are too narrow, the outer ones are not.
          assert_eq!(
              problem(FIT_CHECK, &design(|i| i.coupling.npole = 12)).unwrap(),
              "12 blocks per ring do not fit at this ring size. Inner blocks are 6.350 mm wide on 5.439 mm flats."
          );
          // The level follows the engine's fit rule, the one inverse sizing counts.
          for (_, inputs) in designs() {
              let results = compute_all(&inputs);
              let fits = blocks_fit(&inputs.coupling, &results.model);
              assert_eq!(level_of(FIT_CHECK, &inputs, &results) == Level::Good, fits);
          }
          assert_eq!(problem(FIT_CHECK, &DesignInputs::default()), None);
      }

      #[test]
      fn a_check_that_does_not_apply_gives_no_problem() {
          // Without back iron no magnetic rule sizes the cup wall (E9): no badge, no Problem.
          let inputs = no_back_iron();
          let results = compute_all(&inputs);
          assert_eq!(results.materials.cup_wall_check, "No back iron");
          assert_eq!(
              level(guide(CUP_WALL).unwrap(), &inputs, &results),
              Level::Good
          );
          assert_eq!(problem(CUP_WALL, &inputs), None);
      }

      #[test]
      fn the_design_status_counts_the_failing_checks() {
          let status = |inputs: &DesignInputs| design_status(inputs, &compute_all(inputs));
          let defaults = status(&DesignInputs::default());
          assert_eq!(
              defaults,
              DesignStatus {
                  bad: 3,
                  caution: 0,
                  total: 8,
                  is_default: true
              }
          );
          assert_eq!(
              design_status_text(&defaults),
              "Workbook example design: fails 3 of 8 checks"
          );
          // One fix: the cup wall at its suggested 2.0 mm.
          let fixed = status(&design(|i| i.metal.cup_wall_corner_mm = 2.0));
          assert_eq!(
              design_status_text(&fixed),
              "Your design: fails 2 of 8 checks"
          );
          assert_eq!(
              design_status_text(&status(&passing())),
              "Your design: passes all 8 checks"
          );
          // Cautions are counted apart from the failures.
          assert_eq!(
              design_status_text(&status(&hot_day())),
              "Your design: fails 3 of 8 checks, 1 caution"
          );
          let two_cautions = design(|i| {
              i.temperature.duty.hot_ambient_C = 80.0;
              i.temperature.duty.driving_rise_C = 60.0;
              i.metal.max_diameter_mm = f64::NAN;
          });
          assert_eq!(
              design_status_text(&status(&two_cautions)),
              "Your design: fails 3 of 8 checks, 2 cautions"
          );
          // Any edit leaves the example, even one that keeps the counts.
          let edited = status(&design(|i| i.metal.hardware_g = 7.0));
          assert_eq!((edited.bad, edited.is_default), (3, false));
      }

      #[test]
      fn the_status_text_reads_each_form() {
          let text = |bad, caution, is_default| {
              design_status_text(&DesignStatus {
                  bad,
                  caution,
                  total: 8,
                  is_default,
              })
          };
          assert_eq!(
              text(0, 0, true),
              "Workbook example design: passes all 8 checks"
          );
          assert_eq!(
              text(0, 1, false),
              "Your design: passes all 8 checks, 1 caution"
          );
          assert_eq!(text(1, 0, false), "Your design: fails 1 of 8 checks");
          assert_eq!(
              text(8, 3, true),
              "Workbook example design: fails 8 of 8 checks, 3 cautions"
          );
      }
  }
  ```

- [ ] **Step 3: Run the tests and see them fail**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance`

  Expected: the build fails because the names the module will define do not exist yet. For example:
  - `error[E0425]: cannot find value `CHECK_GUIDES` in this scope`
  - `error[E0425]` for `FIT_CHECK`, `HOT_MIN`, `guide`, `level`, `fix_text`, `plain_text_violations`, `temperature_terms`, `level_of`
  - `error[E0412]: cannot find type `DesignInputs``, `Direction`, `Learn`, `DesignStatus`

- [ ] **Step 4: Implement**

  Insert this between the module docs (after the `//!` block and its blank line) and `#[cfg(test)]`:

  ```rust
  use crate::engine::clamps::SCREW_SIZES;
  use crate::engine::meta::{InputSet, ResultSet, Value};
  use crate::engine::model::{blocks_fit, pitch_share};
  use crate::engine::temperature::CURE_MARGIN_MIN_C;
  use crate::gui::dashboard::{Level, check_level, result_info};
  use crate::gui::format::{format_value, with_unit};
  use crate::gui::inputs::InputCatalogue;
  use crate::{DesignInputs, DesignResults};

  /// Which way a fix moves its input.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum Direction {
      Raise,
      Lower,
  }

  impl Direction {
      /// The verb a fix's text starts with.
      const fn word(self) -> &'static str {
          match self {
              Direction::Raise => "Raise",
              Direction::Lower => "Lower",
          }
      }
  }

  /// Where a dashboard row's "Why:" link leads.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum Learn {
      /// A teaching note by id; every one named here has passed the accuracy gate
      /// ([`crate::gui::explorer::reviewed_note`], a test checks).
      Note(&'static str),
      /// An assumption of the A3 registry by id ([`crate::engine::assumptions::ASSUMPTIONS`]):
      /// its rationale.
      Assumption(&'static str),
      /// No note covers the check yet; the text says so.
      None(&'static str),
  }

  /// One check's guide.
  #[derive(Clone, Copy, Debug)]
  pub struct CheckGuide {
      /// The check's result path, a check of a [`crate::gui::dashboard::DASHBOARD`] row, or
      /// [`FIT_CHECK`].
      pub check: &'static str,
      /// The guide's plain name: the heading of its block, and the Magnets fit row's label.
      pub label: &'static str,
      /// What is wrong in this design, from its inputs and results, every number live; `None`
      /// while the check passes.
      pub problem: fn(&DesignInputs, &DesignResults) -> Option<String>,
      /// Why the check matters: one plain sentence.
      pub why: &'static str,
      /// The inputs to move and which way, by path: each a design input with a slider, never an
      /// assumption.
      pub fixes: &'static [(&'static str, Direction)],
      /// Where the row's "Why:" link leads.
      pub learn: Learn,
  }

  /// The Magnets fit row's check: no result holds it, so the GUI names it (a path no result
  /// has, which [`crate::gui::dashboard::result_info`] does not find).
  pub const FIT_CHECK: &str = "guidance.blocks_fit";

  // The dashboard checks the guides cover (`DASHBOARD`'s check column; a test compares).
  const HOT_MIN: &str = "metal.hot_min_check";
  const CLEARANCE: &str = "metal.clearance_check";
  const CUP_WALL: &str = "materials.cup_wall_check";
  const TEMPERATURE: &str = "temperature.summary.verdict";
  const CLAMP: &str = "clamps.recommended";
  const SPACE_CLAIM: &str = "housing.space_claim_check";
  const END_EFFECT: &str = "model.end_effect_check";

  /// The guides: the seven dashboard checks in the dashboard's order, then the Magnets fit row.
  /// Every Why and Problem template went through the slice's physics review (spec G-3), and none
  /// says what spec section 8 rules out.
  pub static CHECK_GUIDES: [CheckGuide; 8] = [
      CheckGuide {
          check: HOT_MIN,
          label: "Hot-low torque",
          problem: hot_min_problem,
          why: "Magnets lose strength as they warm and production scatter makes some couplings weaker than nominal, so the check takes the estimated pull-out at the operating temperature, less the variation allowance, and compares it with the requirement.",
          fixes: &[
              ("coupling.magnets.axial_length_mm", Direction::Raise),
              ("metal.face_gap_mm", Direction::Lower),
          ],
          learn: Learn::Assumption("production_variation"),
      },
      CheckGuide {
          check: CLEARANCE,
          label: "Running clearance",
          problem: clearance_problem,
          why: "The sleeve and the liner turn on opposite rings with a small radial gap between them, which must absorb shaft movement, runout, deflection, thermal growth and tolerances and still leave the target clearance, or they can rub; a wider magnet gap adds clearance but costs torque.",
          fixes: &[
              ("metal.face_gap_mm", Direction::Raise),
              ("metal.sleeve_mm", Direction::Lower),
              ("metal.liner_mm", Direction::Lower),
          ],
          learn: Learn::None("No note covers the running-clearance stack yet."),
      },
      CheckGuide {
          check: CUP_WALL,
          label: "Cup wall",
          problem: cup_wall_problem,
          why: "The steel ring behind the outer magnets carries each pole's flux around to its neighbours, and a wall thinner than that flux needs is driven past its design flux density toward saturation, where flux leaks out and the torque falls below what the model, which takes the steel as unsaturated, predicts.",
          fixes: &[("metal.cup_wall_corner_mm", Direction::Raise)],
          learn: Learn::Note("a5.low_saturation"),
      },
      CheckGuide {
          check: TEMPERATURE,
          label: "Temperature limits",
          problem: temperature_problem,
          why: "The magnets and their adhesive must stay inside their temperature limits from a hot-day start through a slip fault, during the adhesive cure and, for ferrite, at the coldest, since outside them the magnets can lose magnetization for good or the bond can weaken.",
          fixes: &[("temperature.duty.fault_trip_s", Direction::Lower)],
          learn: Learn::Note("demagnetization"),
      },
      CheckGuide {
          check: CLAMP,
          label: "Shaft clamp",
          problem: clamp_problem,
          why: "The shaft clamp holds by the friction its screws' preload creates, and it must hold the highest torque the coupling can pass (its cold pull-out plus the variation allowance) times a safety factor, so the boss and the clamp length must leave room for a screw size and enough screws.",
          fixes: &[
              ("clamps.boss_od_mm", Direction::Raise),
              ("clamps.clamp_length_mm", Direction::Raise),
          ],
          learn: Learn::Note("clamp_preload"),
      },
      CheckGuide {
          check: SPACE_CLAIM,
          label: "Space claim",
          problem: space_claim_problem,
          why: "The coupling has to fit the envelope reserved for it in the assembly: the diameter, the overall length and the length of the large-diameter bay.",
          fixes: &[
              ("coupling.magnets.axial_length_mm", Direction::Lower),
              ("metal.cap_od_mm", Direction::Lower),
              ("coupling.inner_back_apothem_mm", Direction::Lower),
          ],
          learn: Learn::None("No note covers the space claim yet."),
      },
      CheckGuide {
          check: END_EFFECT,
          label: "End-effect model",
          problem: end_effect_problem,
          why: "The model takes a length proportional to the pole pitch off the magnets for the field that fringes at their ends, so magnets no longer than that leave a zero or negative torque and every number computed from it is meaningless.",
          fixes: &[
              ("coupling.magnets.axial_length_mm", Direction::Raise),
              ("coupling.npole", Direction::Raise),
          ],
          learn: Learn::Note("end_effect"),
      },
      CheckGuide {
          check: FIT_CHECK,
          label: "Magnets fit",
          problem: fit_problem,
          why: "Each ring holds one block per pole, on a polygon of flats or, for arcs, around a circle, so at a given ring size more blocks leave each one a narrower place, and a block wider than its place does not fit.",
          fixes: &[
              ("coupling.inner_back_apothem_mm", Direction::Raise),
              ("coupling.npole", Direction::Lower),
          ],
          learn: Learn::None("No note covers block fit yet."),
      },
  ];

  /// The guide of `check` (a dashboard check or [`FIT_CHECK`]); `None` for any other path.
  pub fn guide(check: &str) -> Option<&'static CheckGuide> {
      CHECK_GUIDES.iter().find(|guide| guide.check == check)
  }

  /// The level of `guide`'s check in this design: the dashboard's badge of the check
  /// ([`check_level`]), green where it gives none (a check that does not apply, such as the cup
  /// wall without back iron, or one greyed while the end-effect model is out of range: the
  /// end-effect guide carries that); for [`FIT_CHECK`] red when the blocks do not fit.
  pub fn level(guide: &CheckGuide, inputs: &DesignInputs, results: &DesignResults) -> Level {
      level_of(guide.check, inputs, results)
  }

  /// [`level`] by check path.
  fn level_of(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Level {
      if check == FIT_CHECK {
          if blocks_fit(&inputs.coupling, &results.model) {
              Level::Good
          } else {
              Level::Bad
          }
      } else {
          check_level(results, check).unwrap_or(Level::Good)
      }
  }

  /// Whether `check` fails or asks for a look in this design: a Problem is built only then.
  fn failing(check: &str, inputs: &DesignInputs, results: &DesignResults) -> bool {
      level_of(check, inputs, results) != Level::Good
  }

  /// A fix's text: its direction, then the input's label as the inputs side shows it ("Raise
  /// Minimum outer return ring wall"); the path itself for a path that is no input (a test
  /// keeps every guide's fixes on real inputs).
  pub fn fix_text(path: &str, direction: Direction) -> String {
      let label = InputCatalogue::get()
          .entry(path)
          .map_or(path, |entry| entry.meta.label);
      format!("{} {label}", direction.word())
  }

  /// The design status line's start for the defaults (spec G-6: they stay the workbook's example).
  pub const WORKBOOK_EXAMPLE: &str = "Workbook example design";

  /// The design status line's start for any other design.
  pub const YOUR_DESIGN: &str = "Your design";

  /// How the design stands against the guides' checks.
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub struct DesignStatus {
      /// The checks that fail (red).
      pub bad: usize,
      /// The checks that ask for a look (amber), counted apart from the failures.
      pub caution: usize,
      /// Every guide's check: [`CHECK_GUIDES`]'s length.
      pub total: usize,
      /// Whether the inputs are the defaults (the workbook's example design).
      pub is_default: bool,
  }

  /// The design status of `inputs` and their `results`, counted live from the guides' levels.
  pub fn design_status(inputs: &DesignInputs, results: &DesignResults) -> DesignStatus {
      let count = |wanted: Level| {
          CHECK_GUIDES
              .iter()
              .filter(|guide| level(guide, inputs, results) == wanted)
              .count()
      };
      DesignStatus {
          bad: count(Level::Bad),
          caution: count(Level::Caution),
          total: CHECK_GUIDES.len(),
          is_default: *inputs == DesignInputs::default(),
      }
  }

  /// The design status line: "Workbook example design: fails 3 of 8 checks", "Your design: fails
  /// 2 of 8 checks", "Your design: passes all 8 checks", each followed by ", 1 caution" or ", 2
  /// cautions" when a check asks for a look.
  pub fn design_status_text(status: &DesignStatus) -> String {
      let design = if status.is_default {
          WORKBOOK_EXAMPLE
      } else {
          YOUR_DESIGN
      };
      let checks = if status.bad == 0 {
          format!("passes all {} checks", status.total)
      } else {
          format!("fails {} of {} checks", status.bad, status.total)
      };
      let cautions = match status.caution {
          0 => String::new(),
          1 => ", 1 caution".to_owned(),
          n => format!(", {n} cautions"),
      };
      format!("{design}: {checks}{cautions}")
  }

  /// What [`plain_text_violations`] looks for, in its order: a workbook cell reference
  /// (`C\d{2,3}`), a correction id (`E\d{1,2}` as a whole id), the word "workbook" in any case
  /// (but in [`WORKBOOK_EXAMPLE`]'s exact phrase), and three names of the engine's provenance.
  pub const BANNED: [&str; 6] = [
      "C\\d{2,3}",
      "E\\d{1,2}",
      "workbook",
      "Addendum",
      "Rust-only",
      "Calculator!",
  ];

  /// The phrase that may say "workbook": the design status line's.
  const ALLOWED_WORKBOOK: &str = "Workbook example";

  /// The [`BANNED`] patterns `text` contains, in [`BANNED`]'s order, each once. The guidance
  /// text is for engineers who are not reading the workbook: the tests of every task of the
  /// slice hold its strings to an empty list.
  pub fn plain_text_violations(text: &str) -> Vec<&'static str> {
      let chars: Vec<char> = text.chars().collect();
      let digits_after = |i: usize| {
          chars[i + 1..]
              .iter()
              .take_while(|c| c.is_ascii_digit())
              .count()
      };
      // C followed by two or more digits holds C\d{2,3}; E with one or two digits and then no
      // letter, digit or underscore is E\d{1,2}\b.
      let cell = chars
          .iter()
          .enumerate()
          .any(|(i, &c)| c == 'C' && digits_after(i) >= 2);
      let correction = chars.iter().enumerate().any(|(i, &c)| {
          let digits = digits_after(i);
          c == 'E'
              && (1..=2).contains(&digits)
              && chars
                  .get(i + 1 + digits)
                  .is_none_or(|&next| !next.is_alphanumeric() && next != '_')
      });
      let workbook =
          text.to_lowercase().matches("workbook").count() > text.matches(ALLOWED_WORKBOOK).count();
      let hits = [
          cell,
          correction,
          workbook,
          text.contains("Addendum"),
          text.contains("Rust-only"),
          text.contains("Calculator!"),
      ];
      BANNED
          .iter()
          .zip(hits)
          .filter_map(|(&pattern, hit)| hit.then_some(pattern))
          .collect()
  }

  // ------------------------------------------------------------------ the Problem builders

  /// The result at `path` as the readouts show it: four significant digits, then its unit.
  fn result_text(results: &DesignResults, path: &str) -> String {
      let unit = result_info(path).map_or("", |info| info.meta.unit);
      with_unit(
          format_value(&results.get(path).unwrap_or(Value::None)),
          unit,
      )
  }

  /// The input at `path` as a readout shows it: four significant digits, then its unit.
  fn input_text(inputs: &DesignInputs, path: &str) -> String {
      let unit = InputCatalogue::get()
          .entry(path)
          .map_or("", |entry| entry.meta.unit);
      with_unit(format_value(&inputs.get(path).unwrap_or(Value::None)), unit)
  }

  /// A number worked out here from results and inputs (no result holds it), in the unit of the
  /// result at `unit_of`.
  fn number_text(x: f64, unit_of: &str) -> String {
      let unit = result_info(unit_of).map_or("", |info| info.meta.unit);
      with_unit(format_value(&Value::Num(x)), unit)
  }

  /// A fraction as a percentage: four significant digits without trailing zeros (0.15 gives
  /// "15 %", 3.4449 gives "344.5 %").
  fn percent_text(fraction: f64) -> String {
      let text = format_value(&Value::Num(fraction * 100.0));
      let digits = if text.contains('.') && !text.contains('e') {
          text.trim_end_matches('0').trim_end_matches('.')
      } else {
          text.as_str()
      };
      format!("{digits} %")
  }

  /// `sentences` as one text, or `fallback` when there are none: a failing check always names
  /// at least one, and the fallback keeps its block from going blank if a verdict and its numbers
  /// ever disagree.
  fn joined_or(sentences: Vec<String>, fallback: &str) -> String {
      if sentences.is_empty() {
          fallback.to_owned()
      } else {
          sentences.join(" ")
      }
  }

  /// Whether a temperature margin fails the verdict's `margin > 0` (NaN fails it too).
  fn not_positive(margin: f64) -> bool {
      margin.is_nan() || margin <= 0.0
  }

  fn hot_min_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(HOT_MIN, inputs, results).then(|| {
          format!(
              "Hot-low torque {} is below the required {}: the pull-out of {} at {}, less the {} variation allowance.",
              result_text(results, "metal.torque_hot_low_Nm"),
              input_text(inputs, "metal.required_min_Nm"),
              result_text(results, "model.pullout_Nm"),
              input_text(inputs, "coupling.op_temp_C"),
              percent_text(inputs.metal.variation),
          )
      })
  }

  fn clearance_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(CLEARANCE, inputs, results).then(|| {
          format!(
              "Sleeve-to-liner gap {} less {} of movement allowances leaves {}, below the {} target.",
              result_text(results, "metal.sleeve_liner_clearance_mm"),
              result_text(results, "metal.adverse_movement_mm"),
              result_text(results, "metal.min_running_clearance_mm"),
              input_text(inputs, "metal.residual_target_mm"),
          )
      })
  }

  fn cup_wall_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(CUP_WALL, inputs, results).then(|| {
          format!(
              "Cup wall at the pocket corners is {}; the flux it carries needs {}.",
              result_text(results, "materials.cup_wall_corner_mm"),
              result_text(results, "materials.backiron_thickness_needed_mm"),
          )
      })
  }

  /// The temperature verdict's terms this design fails, a sentence each, in the verdict's
  /// order: the hot-day margin, the magnet and the adhesive margins at the slip-fault peak, the
  /// cure margin ([`CURE_MARGIN_MIN_C`]) and the cold check (E20, a ferrite ring).
  fn temperature_terms(inputs: &DesignInputs, results: &DesignResults) -> Vec<String> {
      let t = &results.temperature;
      let mut terms = Vec::new();
      if not_positive(t.summary.margin_hot_day_C) {
          terms.push(format!(
              "Hot-day start {} is not below the {} limit.",
              result_text(results, "temperature.summary.hot_day_start_C"),
              result_text(results, "temperature.summary.governing_limit_C"),
          ));
      }
      if not_positive(t.magnet_life.margin_limit_C) {
          terms.push(format!(
              "The slip-fault peak of {} reaches the {} magnet limit.",
              result_text(results, "temperature.magnet_life.peak_C"),
              result_text(results, "temperature.summary.magnet_limit_C"),
          ));
      }
      if not_positive(t.adhesive_life.margin_C) {
          terms.push(format!(
              "The slip-fault peak of {} reaches the {} adhesive limit.",
              result_text(results, "temperature.adhesive_life.peak_C"),
              result_text(results, "temperature.summary.adhesive_limit_C"),
          ));
      }
      let cure = t.summary.cure_margin_C;
      if cure.is_nan() || cure < CURE_MARGIN_MIN_C {
          terms.push(format!(
              "The adhesive cures at {}, {} below the {} single-ring demagnetization onset; the margin must be at least {}.",
              result_text(results, "temperature.adhesive.cure_C"),
              result_text(results, "temperature.summary.cure_margin_C"),
              result_text(results, "temperature.demag.onset_single_ring_C"),
              number_text(CURE_MARGIN_MIN_C, "temperature.summary.cure_margin_C"),
          ));
      }
      if check_level(results, "temperature.demag.cold_check") == Some(Level::Bad) {
          terms.push(format!(
              "The minimum magnet temperature {} is below the {} cold demagnetization limit.",
              input_text(inputs, "metal.min_temp_C"),
              result_text(results, "temperature.demag.cold_limit_C"),
          ));
      }
      terms
  }

  fn temperature_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(TEMPERATURE, inputs, results).then(|| {
          joined_or(
              temperature_terms(inputs, results),
              "A temperature limit is not met: see the temperature results.",
          )
      })
  }

  fn clamp_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(CLAMP, inputs, results).then(|| {
          format!(
              "No screw size from {} to {} fits the boss and the clamp length with enough screws to hold {}: the cold-high torque {} times a safety factor of {}.",
              SCREW_SIZES[0].name,
              SCREW_SIZES[SCREW_SIZES.len() - 1].name,
              result_text(results, "clamps.required_Nm"),
              result_text(results, "clamps.max_torque_Nm"),
              input_text(inputs, "clamps.safety_factor"),
          )
      })
  }

  fn space_claim_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(SPACE_CLAIM, inputs, results).then(|| {
          let h = &results.housing;
          // (axis, its overshoot, the overshoot's path, the dimension's, the claim's, the claim's name)
          let axes = [
              (
                  "Rotating diameter",
                  h.diameter_overshoot_mm,
                  "housing.diameter_overshoot_mm",
                  "metal.rotating_od_mm",
                  "metal.max_diameter_mm",
                  "claim",
              ),
              (
                  "Overall length",
                  h.length_overshoot_mm,
                  "housing.length_overshoot_mm",
                  "metal.axial_stack_mm",
                  "metal.max_overall_axial_mm",
                  "claim",
              ),
              (
                  "Large-diameter stack",
                  h.bay_overshoot_mm,
                  "housing.bay_overshoot_mm",
                  "metal.large_dia_stack_mm",
                  "metal.max_large_dia_axial_mm",
                  "bay",
              ),
          ];
          let sentences = axes
              .iter()
              .filter_map(|&(axis, over, over_path, dimension, claim, name)| {
                  if over > 0.0 {
                      Some(format!(
                          "{axis} {} is {} over the {} {name}.",
                          result_text(results, dimension),
                          result_text(results, over_path),
                          input_text(inputs, claim),
                      ))
                  } else if over.is_nan() {
                      Some(format!(
                          "{axis} cannot be checked: it or its {name} is not a number."
                      ))
                  } else {
                      None
                  }
              })
              .collect();
          joined_or(sentences, "The coupling is outside its space claim.")
      })
  }

  fn end_effect_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(END_EFFECT, inputs, results).then(|| {
          format!(
              "Magnets {} long with a pole pitch of {} give an end-effect factor of {}; the model needs magnets longer than {}.",
              result_text(results, "model.active_length_mm"),
              result_text(results, "model.pole_pitch_mm"),
              result_text(results, "model.f_end"),
              // f_end = 1 - c_end * pitch / L is positive exactly when L > c_end * pitch.
              number_text(
                  inputs.coupling.c_end * results.model.pole_pitch_mm,
                  "model.pole_pitch_mm"
              ),
          )
      })
  }

  fn fit_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
      failing(FIT_CHECK, inputs, results).then(|| {
          let m = &results.model;
          let n = inputs.coupling.npole;
          let mut sentences = vec![format!("{n} blocks per ring do not fit at this ring size.")];
          // The comparisons of `blocks_fit`: flat blocks against their flats, arcs by their
          // share of the pitch circle at the magnet mid-radius.
          if inputs.coupling.faceted == 1 {
              let rings = [
                  (
                      "Inner",
                      m.inner_flat_width_mm < m.inner_width_mm,
                      "model.inner_width_mm",
                      "model.inner_flat_width_mm",
                  ),
                  (
                      "Outer",
                      m.outer_flat_width_mm < m.outer_width_mm,
                      "model.outer_width_mm",
                      "model.outer_flat_width_mm",
                  ),
              ];
              for (ring, narrow, width, flat) in rings {
                  if narrow {
                      sentences.push(format!(
                          "{ring} blocks are {} wide on {} flats.",
                          result_text(results, width),
                          result_text(results, flat),
                      ));
                  }
              }
          } else {
              let shares = [
                  (
                      "Inner",
                      pitch_share(
                          m.inner_width_mm,
                          inputs.coupling.inner_back_apothem_mm,
                          m.inner_thickness_mm,
                          n as f64,
                      ),
                  ),
                  (
                      "Outer",
                      pitch_share(
                          m.outer_width_mm,
                          m.outer_face_apothem_mm,
                          m.outer_thickness_mm,
                          n as f64,
                      ),
                  ),
              ];
              for (ring, share) in shares {
                  if share > 1.0 {
                      sentences.push(format!(
                          "{ring} blocks need {} of the circle at their mid-radius.",
                          percent_text(share)
                      ));
                  }
              }
          }
          sentences.join(" ")
      })
  }
  ```

- [ ] **Step 5: Run the module's tests**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance`

  Expected: `test result: ok. 19 passed; 0 failed`.

- [ ] **Step 6: Run the wider suite and lints**

  Run in order:
  1. `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib engine::temperature`. Expected: `25 passed`.
  2. `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`. Expected: the lib tests read `548 passed; 0 failed`, and every integration suite reads `ok`. That includes `python_schema`, `sizing`, `size_controls` and `deviations`.
  3. `cargo test --manifest-path magcoupling-rs/Cargo.toml`. Expected: all `ok` (engine only, gate 4).
  4. `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`. Expected: `Finished` with no warnings.
  5. `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`. Expected: `Finished` with no warnings.
  6. `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`. Expected: no output.

  I already ran this whole step, plus Step 5, on a scratch copy of the crate with Steps 1, 2 and 4 applied, and all of it passed.

- [ ] **Step 7: Commit**

  ```bash
  git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/engine/temperature.rs
  git commit -F - <<'EOF'
  feat(magcoupler): add check guides, design status and plain-text guard

  gui/guidance.rs: one guide per dashboard check plus the Magnets fit row
  (a Problem from live results, one plain Why, Fix inputs with a direction,
  the row's Why: target), design_status and design_status_text, and
  plain_text_violations for every guidance string of the slice. The
  temperature verdict's cure-margin minimum becomes
  engine::temperature::CURE_MARGIN_MIN_C so the guide quotes the rule the
  verdict applies; no engine behaviour or text changes.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
  EOF
  ```

**Existing tests this task breaks:** none. The full `--features app` run above confirms it: 529 lib tests passed before and 548 after, with the 19 new ones, and every integration suite stayed green. The string lock in `tests/python_schema.rs` is unaffected: no engine label, help or verdict text changes. The docs (`docs/ai/03-structure.yaml` entry for `gui/guidance.rs`) are left to T10.

---

### Task 5: What-moves-what consequences (`Consequences`, `NeighbourOutcome`, `RING_SIZE_STAYS`, `ConsequenceCache`)

> **Amendments:** apply "T5 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


Spec 3.3 and 4.1, spec test 5, and the spec 5 edge cases that belong here: the magnet count at 4 or 40, an invalid or non-finite neighbour, and the end effect out of range. This task is pure code with no egui types. T6 draws the lines.

**Decisions made in this task.** Each one is flagged for the physics and whole-branch reviews.

1. **The keyway clause of `is_valid`.** The engine computes `model.hub_wall_past_key_mm` as `apothem - bondline - bore/2 - keyway depth` (`engine/model.rs:891,1044`). The magnet count does not appear in it. A neighbour therefore fails the keyway check exactly when the shown design does, and the Size tab already says "the keyway breaks through the hub". So a neighbour is classified with every other `is_valid` condition: `blocks_fit` gives `DoesNotFit`; the end effect out of range or a torque that is not finite gives `NoPrediction`. A keyway-only failure still shows the torque. Two tests pin this. One shows the keyway changes no neighbour's reading. The other shows Torque if and only if `is_valid` wherever the keyway leaves wall. To follow spec 3.3 literally instead, add `|| results.model.hub_wall_past_key_mm <= 0.0` to the `NoPrediction` condition in `neighbour_outcome` and delete the test `a_keyway_through_the_hub_changes_no_neighbour_s_reading`.
2. **Number formats.** The lines use the examples' fixed decimals, not `format_value`'s 4 significant digits:
   - lengths to 0.1 mm, the Size-tab sliders' precision;
   - torques to 0.01 N·m;
   - changes to whole signed percent.

   Units go through `with_unit`, and non-finite numbers use format.rs's `non_finite_text`. Two `pub(crate)` helpers go into `gui/format.rs`: `fixed(x, decimals)` and `signed_percent(fraction)`. T6's `torque_gap_text` should call them rather than repeat them.
3. **Rounding of the fit diameter.** A diameter named as fitting (`DiameterFit::At`) is rounded *up* to 0.1 mm, so the number the user reads fits. With the probe's default of 42.0043 mm, the line reads 42.1; 42.0 does not fit. The range ends (`AtMinimum`, `Nowhere`) round to nearest, as the sliders show them. One small inaccuracy remains: when the fit lies within 0.1 mm of a range maximum that is not a whole tenth, the rounded-up value can be up to 0.1 mm past that maximum.
4. **`change` when there is no baseline.** `change` is NaN when the shown design has no prediction (end effect out of range, or a pull-out that is not positive and finite). The line then leaves out the percent instead of comparing with a meaningless number. Because of the NaN, `Consequences` values that carry it never compare equal.

Engine facts, verified with a scratch probe against the current engine (not in the repo). The tests recompute all of these and assert none as literals:
- default 10 poles: pull-out 2.688;
- 8 poles: 1.968 (-27 %), cup grows 41.33 to 42.44;
- 12 poles do not fit at the default ring;
- 4 poles fit at the diameter range minimum;
- 40 poles fit nowhere up to the range maximum;
- `short_magnets()`: 8 gives no prediction, 12 does not fit.

**Files:**
- Modify: `magcoupling-rs/src/gui/format.rs`
  - add two helpers after `non_finite_text` (ends at line 56, before the `/// A number to [`SIGNIFICANT_DIGITS`] significant digits` doc of `format_number`);
  - add two tests at the end of `mod tests` (after `units_follow_the_value_except_dimensionless`, line 178).
- Modify: `magcoupling-rs/src/gui/guidance.rs` (created by T4; read it first):
  - imports at the top;
  - one paragraph appended to the `//!` module doc;
  - a new block after T4's last item and before `#[cfg(test)]`;
  - tests appended at the end of T4's `mod tests` (which has `use super::*;`).
- Test: in-module (`gui::guidance::tests`, `gui::format::tests`).

**Interfaces:**
- Consumes:
  - **T1** (`crate::engine::size_controls`): `DiameterBreakdown { cap_mm, cup_mm, boss_mm }`, `diameter_breakdown(&DesignInputs) -> Option<DiameterBreakdown>`, `LengthBreakdown { magnets_mm, housing_mm }`, `length_breakdown(&DesignInputs) -> Option<LengthBreakdown>`, `DiameterFit { At(f64), AtMinimum(f64), Nowhere(f64) }` (Nowhere carries the range max), `smallest_fitting_diameter(&DesignInputs, npole: i64, minimum_diameter_mm: Option<f64>) -> Option<DiameterFit>`.
  - **T4**: `gui/guidance.rs` with its `mod tests`; `plain_text_violations(&str) -> Vec<&'static str>` (tests).
  - **Engine**: `crate::compute_all`, `FreeVariable::MagnetsPerRing.range()` / `.apply()`, `FreeVariable::RingRadius.grid()` / `.apply()` (tests), `engine::model::{blocks_fit, end_effect_in_range}`, `engine::sizing::is_valid` (tests), `engine::size_controls::{dimensions, dimensions_with_minimum_diameter, resize_diameter}` (tests).
  - **GUI**: `gui::format::{with_unit, non_finite_text}`, `gui::test_support::{assert_glyphs, short_magnets}` (tests).
- Produces (contract):
  ```rust
  pub enum NeighbourOutcome { Torque { pullout_Nm: f64, change: f64 /*fraction vs current pull-out*/, cup_od_mm: f64 }, DoesNotFit, NoPrediction }
  pub struct Neighbour { pub npole: i64, pub outcome: NeighbourOutcome }
  pub struct Consequences { pub neighbours: Vec<Neighbour>, pub fit: Option<DiameterFit>, pub diameter: Option<DiameterBreakdown>, pub length: Option<LengthBreakdown> }
  impl Consequences { pub fn of(inputs: &DesignInputs, results: &DesignResults, minimum_diameter_mm: Option<f64>) -> Self; pub fn magnet_lines(&self, npole: i64) -> Vec<String>; pub fn diameter_line(&self) -> Option<String>; pub fn length_line(&self) -> Option<String> }
  pub const RING_SIZE_STAYS: &str = "Ring size stays the same when you change the magnet count.";
  #[derive(Default)] pub struct ConsequenceCache { .. }  impl ConsequenceCache { pub fn get(&mut self, inputs: &DesignInputs, results: &DesignResults, minimum_diameter_mm: Option<f64>) -> &Consequences }
  ```
- Produces (beyond the contract):
  - `pub(crate) fn fixed(x: f64, decimals: usize) -> String` and `pub(crate) fn signed_percent(fraction: f64) -> String` in `gui/format.rs`, for T6 to reuse;
  - `pub(crate) computed: usize` on `ConsequenceCache`, the same pattern as `SizingRunner::solves`;
  - private: `neighbours`, `neighbour_counts`, `prediction`, `neighbour_outcome`, `fit_line`, `up_to_tenth`, `Neighbour::line`.

---

- [ ] **Step 1: Write the failing format-helper tests.** Append to the end of `mod tests` in `magcoupling-rs/src/gui/format.rs`, after `units_follow_the_value_except_dimensionless` and before the module's closing `}`:

```rust
    #[test]
    fn fixed_decimals_follow_the_request_and_non_finite_numbers_keep_their_text() {
        assert_eq!(fixed(42.8, 1), "42.8");
        assert_eq!(fixed(42.44, 1), "42.4");
        assert_eq!(fixed(22.0, 1), "22.0");
        assert_eq!(fixed(1.96836, 2), "1.97");
        assert_eq!(fixed(f64::INFINITY, 1), "+inf");
        assert_eq!(fixed(f64::NAN, 2), "NaN");
    }

    #[test]
    fn a_change_reads_as_a_whole_signed_percentage_never_minus_zero() {
        assert_eq!(signed_percent(0.1121), "+11 %");
        assert_eq!(signed_percent(-0.2679), "-27 %");
        assert_eq!(signed_percent(1.1279), "+113 %");
        assert_eq!(signed_percent(0.0), "+0 %");
        // A small fall rounds to zero: no minus sign on a zero.
        assert_eq!(signed_percent(-0.004), "+0 %");
        assert_eq!(signed_percent(-0.0), "+0 %");
        assert_eq!(signed_percent(f64::NAN), "NaN %");
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }
```

- [ ] **Step 2: Run the format tests to see them fail.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::format`
  Expected: compile failure, `error[E0425]: cannot find function `fixed` in this scope` and `error[E0425]: cannot find function `signed_percent` in this scope`.

- [ ] **Step 3: Implement the format helpers.** In `magcoupling-rs/src/gui/format.rs`, insert right after the closing `}` of `non_finite_text` (line 56) and before `/// A number to [`SIGNIFICANT_DIGITS`] significant digits: fixed point when it`:

```rust
/// A number to `decimals` places: the Size tab's lines, whose precision follows their sliders
/// (0.1 mm) rather than [`SIGNIFICANT_DIGITS`]. A number that is not finite reads as its
/// [`non_finite_text`].
pub(crate) fn fixed(x: f64, decimals: usize) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None => format!("{x:.decimals$}"),
    }
}

/// A change, as a fraction, as a whole signed percentage with its unit: 0.1121 reads `+11 %`,
/// -0.2679 `-27 %`. A change that rounds to zero reads `+0 %`, never `-0 %`; one that is not
/// finite reads as its [`non_finite_text`] and the unit.
pub(crate) fn signed_percent(fraction: f64) -> String {
    if let Some(text) = non_finite_text(fraction) {
        return format!("{text} %");
    }
    let percent = (fraction * 100.0).round();
    // round() keeps the sign of a small fall (-0.4 rounds to -0.0, which prints "-0").
    let percent = if percent == 0.0 { 0.0 } else { percent };
    format!("{percent:+.0} %")
}
```

- [ ] **Step 4: Run the format tests to see them pass.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::format`
  Expected: PASS for all of `gui::format::tests` (9 existing and 2 new).

- [ ] **Step 5: Write the failing consequence tests.** Append these at the end of T4's `mod tests` in `magcoupling-rs/src/gui/guidance.rs`, before its closing `}`. They reach `DesignInputs`, `DesignResults`, `compute_all`, `FreeVariable`, `DiameterFit`, `smallest_fitting_diameter`, `diameter_breakdown`, `length_breakdown`, `blocks_fit`, `end_effect_in_range` and `plain_text_violations` through `use super::*`; Step 7 adds the imports. Everything else is written as a full path, as the panel tests do.

```rust
    // What moves what (spec 3.3, test 5): every number is recomputed here with compute_all.

    /// The consequences of `inputs` as the panel builds them: with the design's own results.
    fn consequences_of(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Consequences {
        Consequences::of(inputs, &compute_all(inputs), minimum_diameter_mm)
    }

    /// `inputs` with `npole` magnets per ring and nothing else changed.
    fn with_magnets(inputs: &DesignInputs, npole: i64) -> DesignInputs {
        FreeVariable::MagnetsPerRing.apply(inputs, npole as f64)
    }

    /// A design whose own end effect is just out of range while two more magnets bring it back
    /// in and fit: the first ring radius of the solver's grid where two more magnets fit, the
    /// magnets cut halfway between the two counts' end-effect limits (f_end = 1 - c_end * pole
    /// pitch / L is zero at L = c_end * pole pitch, and the pitch falls as the count rises).
    fn end_effect_edge() -> DesignInputs {
        let base = DesignInputs::default();
        let more = base.coupling.npole + 2;
        let radius = FreeVariable::RingRadius
            .grid()
            .into_iter()
            .find(|&radius| {
                let design = with_magnets(&FreeVariable::RingRadius.apply(&base, radius), more);
                blocks_fit(&design.coupling, &compute_all(&design).model)
            })
            .expect("two more magnets fit somewhere on the ring-radius slider");
        let mut design = FreeVariable::RingRadius.apply(&base, radius);
        // The end-effect coefficient's slider maximum: the length lands inside its slider.
        design.coupling.c_end = 0.5;
        let pitch = |npole: i64| compute_all(&with_magnets(&design, npole)).model.pole_pitch_mm;
        let length = design.coupling.c_end * (pitch(base.coupling.npole) + pitch(more)) / 2.0;
        design.coupling.magnets.axial_length_mm = Some(length);
        design
    }

    #[test]
    fn each_neighbour_is_the_design_with_only_the_magnet_count_changed() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
        assert_eq!(counts, [npole - 2, npole + 2]);
        for neighbour in &consequences.neighbours {
            let design = with_magnets(&inputs, neighbour.npole);
            let r = compute_all(&design);
            match neighbour.outcome {
                NeighbourOutcome::Torque {
                    pullout_Nm: pullout,
                    change,
                    cup_od_mm,
                } => {
                    assert!(crate::engine::sizing::is_valid(&design, &r), "{neighbour:?}");
                    assert_eq!(pullout, r.model.pullout_Nm);
                    assert_eq!(change, r.model.pullout_Nm / results.model.pullout_Nm - 1.0);
                    assert_eq!(cup_od_mm, r.model.cup_od_mm);
                }
                NeighbourOutcome::DoesNotFit => {
                    assert!(!blocks_fit(&design.coupling, &r.model), "{neighbour:?}")
                }
                NeighbourOutcome::NoPrediction => {
                    assert!(!end_effect_in_range(r.model.f_end), "{neighbour:?}")
                }
            }
        }
        // The owner's case (spec 1): two fewer magnets fit, lose torque and grow the cup; two
        // more do not fit at this ring size.
        match consequences.neighbours[0].outcome {
            NeighbourOutcome::Torque {
                change, cup_od_mm, ..
            } => {
                assert!(change < 0.0, "{change}");
                assert!(cup_od_mm > results.model.cup_od_mm, "{cup_od_mm}");
            }
            other => panic!("{} magnets fit at the default ring: {other:?}", npole - 2),
        }
        assert_eq!(
            consequences.neighbours[1].outcome,
            NeighbourOutcome::DoesNotFit
        );
    }

    #[test]
    fn the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let fewer = compute_all(&with_magnets(&inputs, npole - 2));
        let lines = Consequences::of(&inputs, &results, None).magnet_lines(npole);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert_eq!(
            lines[1],
            format!(
                "At {}: pull-out {:.2} N·m ({:+.0} %), cup OD {:.1} mm",
                npole - 2,
                fewer.model.pullout_Nm,
                (100.0 * (fewer.model.pullout_Nm / results.model.pullout_Nm - 1.0)).round(),
                fewer.model.cup_od_mm
            )
        );
        assert_eq!(lines[2], format!("At {}: magnets do not fit", npole + 2));
        assert_eq!(lines.len(), 4, "the fit line closes the list: {lines:?}");
    }

    #[test]
    fn at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown() {
        // Spec 5: at 4 or 40 magnets, only the neighbour inside the slider range.
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        for (npole, inside, outside) in [
            (fewest, fewest + 2, fewest - 2),
            (most, most - 2, most + 2),
        ] {
            let inputs = with_magnets(&DesignInputs::default(), npole);
            let consequences = consequences_of(&inputs, None);
            let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(counts, [inside], "{npole} magnets");
            let lines = consequences.magnet_lines(npole);
            assert!(
                lines.iter().any(|l| l.starts_with(&format!("At {inside}:"))),
                "{lines:?}"
            );
            assert!(
                !lines.iter().any(|l| l.starts_with(&format!("At {outside}:"))),
                "{lines:?}"
            );
        }
    }

    #[test]
    fn neighbours_are_even_counts_inside_the_slider_and_never_overflow() {
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        assert_eq!(neighbour_counts(fewest + 4), [fewest + 2, fewest + 6]);
        // A count from a design file may be odd or past the slider.
        assert!(neighbour_counts(fewest + 3).is_empty(), "odd: no even neighbour");
        assert_eq!(neighbour_counts(most + 2), [most]);
        assert_eq!(neighbour_counts(fewest - 2), [fewest]);
        assert!(neighbour_counts(most + 4).is_empty());
        assert!(neighbour_counts(i64::MAX).is_empty());
        assert!(neighbour_counts(i64::MIN).is_empty());
        // An odd count shows the ring-size line and no neighbour.
        let odd = with_magnets(&DesignInputs::default(), fewest + 5);
        let lines = consequences_of(&odd, None).magnet_lines(odd.coupling.npole);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(!lines.iter().any(|l| l.starts_with("At ")), "{lines:?}");
    }

    #[test]
    fn a_keyway_through_the_hub_changes_no_neighbour_s_reading() {
        // The hub wall past the keyway does not depend on the magnet count, so a neighbour
        // fails it exactly when the design shown does (the Size tab says so already).
        let mut broken = DesignInputs::default();
        broken.coupling.bore_mm = 16.0; // the bore and keyway sliders' maxima
        broken.coupling.keyway_depth_mm = 4.0;
        let mut intact = broken.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert!(compute_all(&broken).model.hub_wall_past_key_mm <= 0.0);
        assert!(compute_all(&intact).model.hub_wall_past_key_mm > 0.0);
        let read = |inputs: &DesignInputs| neighbours(inputs, &compute_all(inputs));
        assert_eq!(read(&broken), read(&intact));
        assert!(
            matches!(read(&broken)[0].outcome, NeighbourOutcome::Torque { .. }),
            "{:?}",
            read(&broken)
        );
    }

    #[test]
    fn with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it() {
        let range = FreeVariable::MagnetsPerRing.range();
        for design in [
            DesignInputs::default(),
            crate::gui::test_support::short_magnets(),
            end_effect_edge(),
        ] {
            for npole in (range.min as i64..=range.max as i64).step_by(2) {
                let inputs = with_magnets(&design, npole);
                for neighbour in neighbours(&inputs, &compute_all(&inputs)) {
                    let at = with_magnets(&design, neighbour.npole);
                    let r = compute_all(&at);
                    assert!(r.model.hub_wall_past_key_mm > 0.0);
                    assert_eq!(
                        matches!(neighbour.outcome, NeighbourOutcome::Torque { .. }),
                        crate::engine::sizing::is_valid(&at, &r),
                        "{npole} -> {neighbour:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented() {
        // Spec 5: words, not numbers. The design shown has no prediction either, so the
        // neighbour that has one shows its torque without a change against a meaningless one.
        let inputs = end_effect_edge();
        let results = compute_all(&inputs);
        assert!(
            !end_effect_in_range(results.model.f_end),
            "{}",
            results.model.f_end
        );
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        assert_eq!(
            consequences.neighbours[0],
            Neighbour {
                npole: npole - 2,
                outcome: NeighbourOutcome::NoPrediction
            }
        );
        let more = compute_all(&with_magnets(&inputs, npole + 2));
        match consequences.neighbours[1].outcome {
            NeighbourOutcome::Torque {
                pullout_Nm: pullout,
                change,
                cup_od_mm,
            } => {
                assert_eq!(pullout, more.model.pullout_Nm);
                assert!(change.is_nan(), "{change}");
                assert_eq!(cup_od_mm, more.model.cup_od_mm);
            }
            other => panic!("{other:?}"),
        }
        let lines = consequences.magnet_lines(npole);
        assert_eq!(lines[1], format!("At {}: no prediction", npole - 2));
        assert_eq!(
            lines[2],
            format!(
                "At {}: pull-out {:.2} N·m, cup OD {:.1} mm",
                npole + 2,
                more.model.pullout_Nm,
                more.model.cup_od_mm
            )
        );
    }

    #[test]
    fn with_magnets_too_short_for_the_model_every_neighbour_reads_words() {
        let inputs = crate::gui::test_support::short_magnets();
        let lines = consequences_of(&inputs, None).magnet_lines(inputs.coupling.npole);
        let neighbour_lines: Vec<&String> = lines.iter().filter(|l| l.starts_with("At ")).collect();
        assert_eq!(neighbour_lines.len(), 2, "{lines:?}");
        for line in neighbour_lines {
            assert!(
                line.ends_with(": no prediction") || line.ends_with(": magnets do not fit"),
                "{line}"
            );
        }
    }

    #[test]
    fn the_fit_line_names_a_diameter_that_fits_and_one_step_below_does_not() {
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        assert_eq!(
            consequences.fit,
            smallest_fitting_diameter(&inputs, npole, None)
        );
        let Some(DiameterFit::At(smallest)) = consequences.fit else {
            panic!(
                "the default magnets fit above the slider's minimum: {:?}",
                consequences.fit
            )
        };
        let line = consequences.magnet_lines(npole).pop().expect("a fit line");
        let prefix = format!("Smallest outside diameter that fits {npole} magnets: ");
        let shown: f64 = line
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(" mm"))
            .and_then(|number| number.parse().ok())
            .unwrap_or_else(|| panic!("{line}"));
        // Rounded up to the tenth (less the rounding's 1e-9 mm slack).
        assert!(
            shown >= smallest - 1e-9 && shown - smallest < 0.1,
            "{shown} for {smallest}"
        );
        let fits = |diameter: f64| {
            let design = crate::engine::size_controls::resize_diameter(&inputs, diameter)
                .expect("inside the diameter slider");
            blocks_fit(&design.coupling, &compute_all(&design).model)
        };
        assert!(fits(shown), "{line}");
        assert!(!fits(shown - 0.1), "{line}: one 0.1 mm step below fits too");
    }

    #[test]
    fn the_fit_line_s_edge_forms_name_the_ends_of_the_diameter_range() {
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        // The fewest magnets fit at the slider's smallest diameter, with the slider's own floor
        // and with the lowest minimum preference.
        let inputs = with_magnets(&DesignInputs::default(), fewest);
        let size = crate::engine::size_controls::dimensions(&inputs).expect("a size");
        let lowest = size.minimum_diameter_range.min;
        let lowered = crate::engine::size_controls::dimensions_with_minimum_diameter(&inputs, lowest)
            .expect("a size at the lowest minimum");
        for (minimum, floor) in [
            (None, size.diameter_range.min),
            (Some(lowest), lowered.diameter_range.min),
        ] {
            let consequences = consequences_of(&inputs, minimum);
            assert_eq!(
                consequences.fit,
                smallest_fitting_diameter(&inputs, fewest, minimum)
            );
            assert_eq!(consequences.fit, Some(DiameterFit::AtMinimum(floor)), "{minimum:?}");
            assert_eq!(
                consequences.magnet_lines(fewest).last(),
                Some(&format!("Fits down to {floor:.1} mm (the minimum)"))
            );
        }
        // The most magnets fit nowhere on the slider.
        let inputs = with_magnets(&DesignInputs::default(), most);
        let top = crate::engine::size_controls::dimensions(&inputs)
            .expect("a size")
            .diameter_range
            .max;
        let consequences = consequences_of(&inputs, None);
        assert_eq!(consequences.fit, Some(DiameterFit::Nowhere(top)));
        assert_eq!(
            consequences.magnet_lines(most).last(),
            Some(&format!("Does not fit at any diameter up to {top:.1} mm"))
        );
    }

    #[test]
    fn a_fitting_diameter_rounds_up_to_the_tenth_and_a_whole_tenth_stays() {
        assert_eq!(up_to_tenth(40.43), 40.5);
        assert_eq!(up_to_tenth(40.5), 40.5);
        assert_eq!(up_to_tenth(40.4000001), 40.5);
        // Floating-point noise a hair above a whole tenth stays at that tenth.
        assert_eq!(up_to_tenth(37.300000000000004), 37.3);
    }

    #[test]
    fn the_diameter_and_length_lines_read_the_engine_breakdowns() {
        // The default (the cap is the largest) and the fewest magnets (the cup is).
        let default = DesignInputs::default();
        let fewest = FreeVariable::MagnetsPerRing.range().min as i64;
        for inputs in [default.clone(), with_magnets(&default, fewest)] {
            let consequences = consequences_of(&inputs, None);
            let diameter = diameter_breakdown(&inputs).expect("a diameter breakdown");
            let length = length_breakdown(&inputs).expect("a length breakdown");
            assert_eq!(consequences.diameter, Some(diameter));
            assert_eq!(consequences.length, Some(length));
            assert_eq!(
                consequences.diameter_line(),
                Some(format!(
                    "Set by: cap {:.1}, cup {:.1}, boss {:.1} mm",
                    diameter.cap_mm, diameter.cup_mm, diameter.boss_mm
                ))
            );
            assert_eq!(
                consequences.length_line(),
                Some(format!(
                    "Magnets {:.1} mm + housing {:.1} mm",
                    length.magnets_mm, length.housing_mm
                ))
            );
        }
    }

    #[test]
    fn at_8_magnets_the_ring_size_stays_and_the_cap_sets_the_diameter() {
        // Spec test 10, the owner's case (10 -> 8 magnets): the ring radius is unchanged and
        // the cap is the largest of the three, which is why the diameter did not shrink.
        let default = DesignInputs::default();
        let fewer = default.coupling.npole - 2;
        let inputs = with_magnets(&default, fewer);
        assert_eq!(
            inputs.coupling.inner_back_apothem_mm,
            default.coupling.inner_back_apothem_mm
        );
        let consequences = consequences_of(&inputs, None);
        assert_eq!(consequences.magnet_lines(fewer)[0], RING_SIZE_STAYS);
        let d = consequences.diameter.expect("a diameter breakdown");
        assert!(d.cap_mm > d.cup_mm && d.cap_mm > d.boss_mm, "{d:?}");
        assert_eq!(
            d.cap_mm,
            crate::engine::size_controls::dimensions(&inputs)
                .expect("a size")
                .diameter_mm
        );
        assert!(
            consequences.diameter_line().expect("a line").starts_with("Set by: cap "),
            "{:?}",
            consequences.diameter_line()
        );
    }

    #[test]
    fn without_a_breakdown_or_a_fit_there_is_no_line_for_it() {
        let empty = Consequences {
            neighbours: Vec::new(),
            fit: None,
            diameter: None,
            length: None,
        };
        assert_eq!(empty.magnet_lines(10), [RING_SIZE_STAYS]);
        assert_eq!(empty.diameter_line(), None);
        assert_eq!(empty.length_line(), None);
    }

    #[test]
    fn every_consequence_line_has_its_glyphs_and_plain_words() {
        // Spec tests 3 and 11, over every line form: a torque, a torque without a change, both
        // word forms, the three fit forms and both breakdowns.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let range = FreeVariable::MagnetsPerRing.range();
        let default = DesignInputs::default();
        let designs = [
            default.clone(),
            with_magnets(&default, range.min as i64),
            with_magnets(&default, default.coupling.npole - 2),
            with_magnets(&default, range.max as i64),
            crate::gui::test_support::short_magnets(),
            end_effect_edge(),
        ];
        for inputs in &designs {
            let consequences = consequences_of(inputs, None);
            let mut lines = consequences.magnet_lines(inputs.coupling.npole);
            lines.extend(consequences.diameter_line());
            lines.extend(consequences.length_line());
            for line in &lines {
                crate::gui::test_support::assert_glyphs(&ctx, line, "a consequence line");
                assert_eq!(plain_text_violations(line), Vec::<&str>::new(), "{line}");
            }
        }
    }
```

- [ ] **Step 6: Run the consequence tests to see them fail.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance`
  Expected: compile failure in `gui::guidance::tests`, for example `error[E0433]: failed to resolve: use of undeclared type `Consequences``, `error[E0425]: cannot find value `RING_SIZE_STAYS` in this scope`, and `error[E0425]: cannot find function `neighbour_counts`` (also `neighbours`, `up_to_tenth`).

- [ ] **Step 7: Implement the consequences.** Make these three edits to `magcoupling-rs/src/gui/guidance.rs`.

  **(a)** Add to the file's `use` lines. T4 may already import some of these names, likely `compute_all`, `blocks_fit` (its FIT_CHECK) and `with_unit`. Keep exactly one import of each name; rustc rejects a duplicate with E0252.
  ```rust
  use crate::compute_all;
  use crate::engine::model::{blocks_fit, end_effect_in_range};
  use crate::engine::size_controls::{
      DiameterBreakdown, DiameterFit, LengthBreakdown, diameter_breakdown, length_breakdown,
      smallest_fitting_diameter,
  };
  use crate::engine::sizing::FreeVariable;
  use crate::gui::format::{fixed, signed_percent, with_unit};
  ```

  **(b)** Append this paragraph at the end of the `//!` module doc that T4 wrote, after a `//!` blank line:
  ```rust
  //! What moves what (spec 3.3): [`Consequences`] are the lines under the Size tab's controls:
  //! two magnets fewer and two more (each one `compute_all` of the design with only the count
  //! changed; words, not numbers, where the blocks do not fit or the model has no prediction),
  //! the smallest outside diameter at which the count fits, and what sets the outside diameter
  //! and the overall length ([`crate::engine::size_controls`]). Every number comes from the
  //! engine. [`ConsequenceCache`] keeps them until the inputs or the minimum diameter change.
  ```

  **(c)** Insert this block after T4's last item and before `#[cfg(test)]`:
  ```rust
  // What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision),
  // torques to 0.01 N·m and changes to whole percent.

  /// The first line under Magnets per ring, always shown: the ring radius
  /// (`coupling.inner_back_apothem_mm`) is its own input; the magnet count only sets how many
  /// blocks go around it.
  pub const RING_SIZE_STAYS: &str = "Ring size stays the same when you change the magnet count.";

  /// A neighbour line's words when its blocks do not fit.
  const NEIGHBOUR_DOES_NOT_FIT: &str = "magnets do not fit";

  /// A neighbour line's words when the model has no number there.
  const NEIGHBOUR_NO_PREDICTION: &str = "no prediction";

  /// What two magnets fewer or two more would do (spec 3.3).
  #[derive(Clone, Copy, Debug, PartialEq)]
  #[allow(non_snake_case)] // unit suffix, as the engine's names
  pub enum NeighbourOutcome {
      /// The blocks fit and the model predicts a torque.
      Torque {
          /// The neighbour's pull-out torque at the operating temperature (`model.pullout_Nm`).
          pullout_Nm: f64,
          /// Its change as a fraction of the pull-out of the design shown; NaN when that design
          /// has no prediction itself, so there is nothing to compare with.
          change: f64,
          /// The neighbour's cup outer diameter (`model.cup_od_mm`).
          cup_od_mm: f64,
      },
      /// The blocks do not fit at this ring size ([`blocks_fit`]).
      DoesNotFit,
      /// The blocks fit but the model has no number: the end effect is out of range
      /// ([`end_effect_in_range`]) or a torque is not finite.
      NoPrediction,
  }

  /// A magnet count next to the design's, and what it would do.
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub struct Neighbour {
      pub npole: i64,
      pub outcome: NeighbourOutcome,
  }

  impl Neighbour {
      /// "At N: ..." (spec 3.3): the torque and cup, or why there is no number.
      fn line(&self) -> String {
          let at = format!("At {}:", self.npole);
          match self.outcome {
              NeighbourOutcome::Torque {
                  pullout_Nm: pullout,
                  change,
                  cup_od_mm,
              } => {
                  let percent = if change.is_finite() {
                      format!(" ({})", signed_percent(change))
                  } else {
                      String::new()
                  };
                  format!(
                      "{at} pull-out {}{percent}, cup OD {}",
                      with_unit(fixed(pullout, 2), "N·m"),
                      with_unit(fixed(cup_od_mm, 1), "mm")
                  )
              }
              NeighbourOutcome::DoesNotFit => format!("{at} {NEIGHBOUR_DOES_NOT_FIT}"),
              NeighbourOutcome::NoPrediction => format!("{at} {NEIGHBOUR_NO_PREDICTION}"),
          }
      }
  }

  /// The "what moves what" readings of a design (spec 3.3), all from the engine.
  #[derive(Clone, Debug, PartialEq)]
  pub struct Consequences {
      /// Two magnets fewer and two more, ascending, where the count stays an even count inside
      /// the magnet slider.
      pub neighbours: Vec<Neighbour>,
      /// The smallest outside diameter at which the design's magnets fit, over the Size tab's
      /// diameter range.
      pub fit: Option<DiameterFit>,
      /// The three diameters the outside diameter is the largest of.
      pub diameter: Option<DiameterBreakdown>,
      /// The magnets and the housing that make up the overall length.
      pub length: Option<LengthBreakdown>,
  }

  impl Consequences {
      /// The consequences of `inputs`, whose results are `results` (`compute_all(inputs)`: each
      /// neighbour's change compares with this design's pull-out), with the Size tab's minimum
      /// diameter preference (`None`: the slider's own floor).
      pub fn of(
          inputs: &DesignInputs,
          results: &DesignResults,
          minimum_diameter_mm: Option<f64>,
      ) -> Self {
          Self {
              neighbours: neighbours(inputs, results),
              fit: smallest_fitting_diameter(inputs, inputs.coupling.npole, minimum_diameter_mm),
              diameter: diameter_breakdown(inputs),
              length: length_breakdown(inputs),
          }
      }

      /// The lines under Magnets per ring: [`RING_SIZE_STAYS`], one line per neighbour, then
      /// the smallest diameter that fits. `npole` is the count the consequences were built for
      /// (`inputs.coupling.npole`).
      pub fn magnet_lines(&self, npole: i64) -> Vec<String> {
          let mut lines = vec![RING_SIZE_STAYS.to_owned()];
          lines.extend(self.neighbours.iter().map(Neighbour::line));
          lines.extend(self.fit.map(|fit| fit_line(fit, npole)));
          lines
      }

      /// The line under Outside diameter: the three diameters it is the largest of.
      pub fn diameter_line(&self) -> Option<String> {
          self.diameter.map(|d| {
              format!(
                  "Set by: cap {}, cup {}, boss {}",
                  fixed(d.cap_mm, 1),
                  fixed(d.cup_mm, 1),
                  with_unit(fixed(d.boss_mm, 1), "mm")
              )
          })
      }

      /// The line under Overall length: the magnets and the housing around them.
      pub fn length_line(&self) -> Option<String> {
          self.length.map(|l| {
              format!(
                  "Magnets {} + housing {}",
                  with_unit(fixed(l.magnets_mm, 1), "mm"),
                  with_unit(fixed(l.housing_mm, 1), "mm")
              )
          })
      }
  }

  /// The neighbours of `inputs` (`results` are its own), each one `compute_all` of the design
  /// with only the magnet count changed.
  fn neighbours(inputs: &DesignInputs, results: &DesignResults) -> Vec<Neighbour> {
      let baseline = prediction(results);
      neighbour_counts(inputs.coupling.npole)
          .into_iter()
          .map(|npole| {
              let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
              let outcome = neighbour_outcome(&design, &compute_all(&design), baseline);
              Neighbour { npole, outcome }
          })
          .collect()
  }

  /// Two fewer and two more than `npole`, ascending, keeping the even counts inside the magnet
  /// slider (one block per pole; the slider steps by 2). A count from a design file may be odd
  /// or past the slider, and none overflows.
  fn neighbour_counts(npole: i64) -> Vec<i64> {
      let range = FreeVariable::MagnetsPerRing.range();
      [npole.checked_sub(2), npole.checked_add(2)]
          .into_iter()
          .flatten()
          .filter(|&n| n % 2 == 0 && (range.min..=range.max).contains(&(n as f64)))
          .collect()
  }

  /// The pull-out of the design shown when the model predicts one: the end effect in range and
  /// a positive, finite torque. Without it a neighbour's change has nothing to compare with.
  fn prediction(results: &DesignResults) -> Option<f64> {
      let pullout = results.model.pullout_Nm;
      (end_effect_in_range(results.model.f_end) && pullout.is_finite() && pullout > 0.0)
          .then_some(pullout)
  }

  /// What a neighbour design does (`results` are its own; `baseline` is [`prediction`] of the
  /// design shown). The conditions of [`crate::engine::sizing::is_valid`] but the keyway's: the
  /// hub wall past the keyway (apothem - bondline - bore / 2 - keyway depth) does not depend on
  /// the magnet count, so a neighbour fails it exactly when the design shown does, the Size tab
  /// already says so, and the torques still compare.
  fn neighbour_outcome(
      design: &DesignInputs,
      results: &DesignResults,
      baseline: Option<f64>,
  ) -> NeighbourOutcome {
      if !blocks_fit(&design.coupling, &results.model) {
          return NeighbourOutcome::DoesNotFit;
      }
      let pullout = results.model.pullout_Nm;
      if !end_effect_in_range(results.model.f_end)
          || !pullout.is_finite()
          || !results.metal.torque_hot_low_Nm.is_finite()
      {
          return NeighbourOutcome::NoPrediction;
      }
      NeighbourOutcome::Torque {
          pullout_Nm: pullout,
          change: baseline.map_or(f64::NAN, |base| pullout / base - 1.0),
          cup_od_mm: results.model.cup_od_mm,
      }
  }

  /// The fit line under Magnets per ring (spec 3.3). A diameter named as fitting is rounded up,
  /// so the number read fits; the range ends read as the diameter slider shows them.
  fn fit_line(fit: DiameterFit, npole: i64) -> String {
      match fit {
          DiameterFit::At(mm) => format!(
              "Smallest outside diameter that fits {npole} magnets: {}",
              with_unit(fixed(up_to_tenth(mm), 1), "mm")
          ),
          DiameterFit::AtMinimum(mm) => {
              format!("Fits down to {} (the minimum)", with_unit(fixed(mm, 1), "mm"))
          }
          DiameterFit::Nowhere(mm) => format!(
              "Does not fit at any diameter up to {}",
              with_unit(fixed(mm, 1), "mm")
          ),
      }
  }

  /// Slack of [`up_to_tenth`] [mm]: far below the fit search's 1e-6 mm, far above the
  /// floating-point noise on a diameter that is a whole tenth.
  const TENTH_SLACK_MM: f64 = 1e-9;

  /// `mm` rounded up to the lines' 0.1 mm, so a diameter they name as fitting fits: 42.0043
  /// reads 42.1, since 42.0 does not fit. A value within [`TENTH_SLACK_MM`] above a whole tenth
  /// stays at that tenth.
  fn up_to_tenth(mm: f64) -> f64 {
      ((mm - TENTH_SLACK_MM) * 10.0).ceil() / 10.0
  }
  ```

- [ ] **Step 8: Run the consequence tests to see them pass.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance`
  Expected: PASS for T4's tests and the 15 new ones (`each_neighbour_is_the_design_with_only_the_magnet_count_changed` through `every_consequence_line_has_its_glyphs_and_plain_words`).

- [ ] **Step 9: Write the failing cache test.** Append to the same `mod tests` in `magcoupling-rs/src/gui/guidance.rs`:

```rust
    #[test]
    fn the_cache_computes_once_per_design_and_minimum_and_follows_an_edit() {
        // Spec 4.1 and test 5: computed again only when the inputs or the minimum change; what
        // it returns is always Consequences::of the design asked about.
        let mut cache = ConsequenceCache::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let first = cache.get(&inputs, &results, None).clone();
        assert_eq!(first, Consequences::of(&inputs, &results, None));
        for _ in 0..5 {
            assert_eq!(cache.get(&inputs, &results, None), &first);
        }
        assert_eq!(cache.computed, 1, "a design asked about again is not computed again");
        // An edit: the neighbours' numbers follow it.
        let mut hot = inputs.clone();
        hot.coupling.op_temp_C += 30.0;
        let hot_results = compute_all(&hot);
        let edited = cache.get(&hot, &hot_results, None).clone();
        assert_eq!(cache.computed, 2);
        assert_eq!(edited, Consequences::of(&hot, &hot_results, None));
        assert_ne!(edited.neighbours, first.neighbours);
        let fewer = compute_all(&with_magnets(&hot, hot.coupling.npole - 2));
        assert!(
            matches!(
                edited.neighbours[0].outcome,
                NeighbourOutcome::Torque { pullout_Nm: pullout, .. } if pullout == fewer.model.pullout_Nm
            ),
            "{:?}",
            edited.neighbours
        );
        // A new minimum diameter is a new question too.
        let lowest = crate::engine::size_controls::dimensions(&hot)
            .expect("a size")
            .minimum_diameter_range
            .min;
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest)),
            &Consequences::of(&hot, &hot_results, Some(lowest))
        );
        assert_eq!(cache.computed, 3);
        cache.get(&hot, &hot_results, Some(lowest));
        assert_eq!(cache.computed, 3);
    }
```

- [ ] **Step 10: Run the cache test to see it fail.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance::tests::the_cache_computes_once`
  Expected: compile failure, `error[E0433]: failed to resolve: use of undeclared type `ConsequenceCache``.

- [ ] **Step 11: Implement the cache.** In `magcoupling-rs/src/gui/guidance.rs`, insert directly after `up_to_tenth` and before `#[cfg(test)]`:

```rust
/// [`Consequences`] kept for the design they were built for (spec 4.1): the panel asks every
/// frame, and the neighbours' `compute_all` runs and the fit search are not per-frame work. They
/// are computed again only when the inputs or the minimum diameter change.
#[derive(Default)]
pub struct ConsequenceCache {
    last: Option<CachedConsequences>,
    /// How many times consequences were computed (the tests check that a repeat computes
    /// nothing).
    pub(crate) computed: usize,
}

/// What a [`ConsequenceCache`] holds: the question and its answer.
struct CachedConsequences {
    inputs: DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: Consequences,
}

impl ConsequenceCache {
    /// [`Consequences::of`] `inputs` (whose results are `results`) with the minimum diameter
    /// preference, computed only when the inputs or the minimum differ from the last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
    ) -> &Consequences {
        let current = self.last.as_ref().is_some_and(|last| {
            last.inputs == *inputs && last.minimum_diameter_mm == minimum_diameter_mm
        });
        if !current {
            self.computed += 1;
            self.last = Some(CachedConsequences {
                inputs: inputs.clone(),
                minimum_diameter_mm,
                consequences: Consequences::of(inputs, results, minimum_diameter_mm),
            });
        }
        &self.last.as_ref().expect("filled above").consequences
    }
}
```

- [ ] **Step 12: Run the guidance tests to see them pass.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::guidance`
  Expected: PASS for all of `gui::guidance::tests`: T4's, the 15 from Step 5, and `the_cache_computes_once_per_design_and_minimum_and_follows_an_edit`.

- [ ] **Step 13: Run the wider suite for the touched modules, then format and lint.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::` (expected: PASS; the panel, sizing, session and size-control suites are unchanged)
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls` (expected: PASS; T1's functions that this task consumes)
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test sizing` (expected: PASS)
  - `rustfmt --edition 2024 magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/format.rs`, then `git diff --stat` (expected: changes only in those two files)
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` (expected: no warnings; `non_snake_case` is allowed on `NeighbourOutcome`, and bindings rename `pullout_Nm` to `pullout`)
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --features gui --lib -- -D warnings` (expected: no warnings; gate 9)

- [ ] **Step 14: Commit.**
  ```bash
  git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/format.rs
  git commit -F - <<'EOF'
  feat(magcoupler): what-moves-what consequences for the Size tab

  Add gui::guidance::Consequences: the magnet-count neighbours (each one
  compute_all with only coupling.npole changed; words, not numbers, where the
  blocks do not fit or the end-effect model has no prediction), the smallest
  fitting outside diameter (rounded up so the number read fits) and the
  diameter and length breakdowns, with their display lines and a cache keyed
  on the inputs and the minimum diameter. format.rs gains fixed-decimal and
  signed-percent text for the Size tab lines.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
  EOF
  ```

**Existing tests this change breaks:** none. The task only adds items: two `pub(crate)` functions in `gui/format.rs` and new items in `gui/guidance.rs`. No existing signature, string or behaviour changes, and no test scans gui source or enumerates format.rs's functions. The glyph test in `panel.rs` (`assert_glyphs` over the panel's texts, around line 3568) does not see these lines until T6 draws them. T6 should add `Consequences::magnet_lines`, `diameter_line` and `length_line` to that list, and call `crate::gui::format::{fixed, signed_percent}` in `torque_gap_text` rather than formatting its own percent.

---

### Task 6: Size tab torque readouts and what-moves-what lines

> **Amendments:** apply "T6 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Modify `magcoupling-rs/src/gui/size_controls.rs` (153 lines before this task):
  - module doc (lines 1-2);
  - imports (lines 4-10);
  - new constants and functions after `LINKED_HOUSING` (line 18);
  - `controls_ui` signature (lines 27-32) and its two loops (lines 72-105 and 116-145);
  - a new private `consequence_line` and a new `#[cfg(test)] mod tests` at the end of the file.
- Modify `magcoupling-rs/src/gui/panel.rs`. Line numbers are from before T3-T5, so find each edit by the quoted code.
  - imports (about lines 27-57);
  - field `consequences` after `minimum_diameter_mm` (about line 250);
  - `new()` (about line 297);
  - the `view_ui` SizeAndTorque arm (about lines 707-726);
  - new method `torque_readouts_ui` after `view_ui` (about line 784);
  - the `inputs_ui` SizeAndTorque arm (about lines 925-931);
  - new free fn `torque_readout` after `assumptions_banner` (about line 1274);
  - tests: an import block, helpers and 10 tests after `manual_size_adopts_the_inverse_solution_before_editing` (about line 3804); edit `manual_size_does_not_present_out_of_range_torque_as_a_prediction` (about lines 3784-3792).
- Tests go in the modules of both files.

**Interfaces:**
- Consumes:
  - T1 `engine::size_controls::{DiameterBreakdown, diameter_breakdown, DiameterFit}`, plus the existing `dimensions`.
  - T3 `gui::sizing::REQUIREMENT_INPUT`. The panel imports it after T3's rename. T3's `SizingRunner` range parameters are reached only through `shown_inputs()`, which T6 does not change.
  - T4 `gui::guidance::plain_text_violations`.
  - T5 `gui::guidance::{Consequences, ConsequenceCache, RING_SIZE_STAYS}`, `Consequences::{of, magnet_lines, diameter_line, length_line}` and its fields `neighbours`, `fit` and `diameter`.
  - Assumptions about T5, taken from spec 3.3; the tests check them:
    - `magnet_lines(npole)` returns every line for the Magnets control in display order, with `RING_SIZE_STAYS` first.
    - A neighbour line starts `"At N':"`.
    - `diameter_line()` starts `"Set by: cap "` and lists cap, cup, boss.
- Produces (from the contract), in `gui/size_controls.rs`:
  - `pub const REQUIRED: &str = "Required";`
  - `pub const HOT_LOW: &str = "Hot-low";`
  - `pub fn pullout_label(op_temp_C: f64) -> String`
  - `pub const NO_PREDICTION: &str = "no prediction (magnets too short for the end-effect model)";`
  - `pub fn torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool)`
- Private additions:
  - `pub(crate) fn predicts(results: &DesignResults, torque_Nm: f64) -> bool`;
  - `controls_ui` gains `consequences: &Consequences`;
  - panel field `consequences: ConsequenceCache`;
  - `MagcouplingPanel::torque_readouts_ui`;
  - free fn `torque_readout`.

Working directory for every command: `C:/Users/Cole/source/repos/linkage_simulation`.

- [ ] **Step 1: Write the failing unit tests for the readout texts.** Append to the end of `magcoupling-rs/src/gui/size_controls.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_all;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::test_support::{SCREEN, assert_glyphs, short_magnets, sized_frame};

    #[test]
    fn the_pull_out_label_names_the_live_operating_temperature() {
        assert_eq!(pullout_label(50.0), "Pull-out torque (nominal, at 50 °C)");
        assert_eq!(pullout_label(37.5), "Pull-out torque (nominal, at 37.5 °C)");
        assert_eq!(pullout_label(-40.0), "Pull-out torque (nominal, at -40 °C)");
        assert_eq!(pullout_label(-0.0), "Pull-out torque (nominal, at 0 °C)");
    }

    #[test]
    fn the_gap_reads_short_or_a_margin_against_the_requirement() {
        assert_eq!(
            torque_gap_text(2.21, 2.5),
            ("short 0.29 N·m (-12 %)".to_owned(), false)
        );
        assert_eq!(
            torque_gap_text(2.9, 2.5),
            ("margin +0.40 N·m (+16 %)".to_owned(), true)
        );
        // The engine's hot minimum check passes at equality (it fails `hot_low < required`).
        assert_eq!(
            torque_gap_text(2.5, 2.5),
            ("margin +0.00 N·m (+0 %)".to_owned(), true)
        );
        // A shortfall too small to show still reads short, with its sign.
        assert_eq!(
            torque_gap_text(2.499, 2.5),
            ("short 0.00 N·m (-0 %)".to_owned(), false)
        );
    }

    #[test]
    fn a_requirement_that_is_no_positive_torque_drops_the_percent() {
        // Spec 5 "Requirement <= 0": a design file can hold one (the slider stops at its
        // metadata minimum). The gap still reads, without a share of a non-positive torque.
        assert_eq!(torque_gap_text(2.0, 0.0), ("margin +2.00 N·m".to_owned(), true));
        assert_eq!(torque_gap_text(2.0, -1.0), ("margin +3.00 N·m".to_owned(), true));
        assert_eq!(torque_gap_text(-0.5, 0.0), ("short 0.50 N·m".to_owned(), false));
    }

    #[test]
    fn a_torque_is_shown_only_inside_the_end_effect_range_and_finite() {
        let results = compute_all(&DesignInputs::default());
        assert!(predicts(&results, results.metal.torque_hot_low_Nm));
        assert!(predicts(&results, results.model.pullout_Nm));
        // Spec 5: a torque that is not finite reads no prediction, banner or not.
        for torque in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!predicts(&results, torque), "{torque}");
        }
        // With the end-effect banner even a finite torque is no prediction.
        let short = compute_all(&short_magnets());
        assert!(crate::gui::dashboard::end_effect_banner(&short).is_some());
        assert!(short.metal.torque_hot_low_Nm.is_finite());
        assert!(!predicts(&short, short.metal.torque_hot_low_Nm));
        assert!(!predicts(&short, short.model.pullout_Nm));
    }

    #[test]
    fn the_readout_texts_have_glyphs_and_plain_words() {
        let ctx = egui::Context::default();
        // The fonts load on the context's first frame.
        sized_frame(&ctx, SCREEN, Vec::new(), |_| {});
        let texts = [
            REQUIRED.to_owned(),
            HOT_LOW.to_owned(),
            NO_PREDICTION.to_owned(),
            pullout_label(50.0),
            pullout_label(-37.5),
            torque_gap_text(2.21, 2.5).0,
            torque_gap_text(2.9, 2.5).0,
            torque_gap_text(2.0, 0.0).0,
        ];
        for text in &texts {
            assert_glyphs(&ctx, text, "a Size tab readout");
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }
}
```

- [ ] **Step 2: Run the unit tests and confirm they fail to compile.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::size_controls`
  Expected: compile errors `cannot find function 'pullout_label' in this scope`, the same for `torque_gap_text` and `predicts`, and `cannot find value 'REQUIRED'`, `'HOT_LOW'` and `'NO_PREDICTION'`.

- [ ] **Step 3: Add the readout texts to `magcoupling-rs/src/gui/size_controls.rs`.**

  (a) Replace the module doc (lines 1-2):
```rust
//! Direct controls for the complete envelope, alongside the existing magnet count and gap.
//! The engine adapter links housing dimensions; every torque still comes from compute_all.
//!
//! Under each control, its what-moves-what lines (spec 3.3: [`Consequences`], computed by the
//! guidance layer). The panel's centre view draws the torque readouts (spec 3.2) with the
//! texts here: [`REQUIRED`], [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and
//! [`NO_PREDICTION`] where the model makes none ([`predicts`]).
```
  (b) Replace `use crate::DesignInputs;` with `use crate::{DesignInputs, DesignResults};`. Move it to the end of the `use` block (rustfmt order), and add `use crate::gui::dashboard::end_effect_out_of_range;` after the `crate::engine::size_controls` import.

  (c) After the `pub const LINKED_HOUSING: &str = ...;` line, insert:
```rust

/// The label of the requirement's slider (`metal.required_min_Nm`).
pub const REQUIRED: &str = "Required";

/// The label of the hot-low torque (`metal.torque_hot_low_Nm`), its gap to the requirement
/// beside it ([`torque_gap_text`]).
pub const HOT_LOW: &str = "Hot-low";

/// What a torque readout shows, the gap left out, where the model makes no prediction
/// ([`predicts`]).
pub const NO_PREDICTION: &str = "no prediction (magnets too short for the end-effect model)";

/// The label of the nominal pull-out (`model.pullout_Nm`) at the operating temperature
/// `op_temp_C` [°C] (`metal.op_temp_C`), in its shortest decimal form: "50", or "37.5" at the
/// input's 0.5 °C step (-0 reads 0).
#[allow(non_snake_case)] // unit suffix
pub fn pullout_label(op_temp_C: f64) -> String {
    let op_temp_C = if op_temp_C == 0.0 { 0.0 } else { op_temp_C };
    format!("Pull-out torque (nominal, at {op_temp_C} °C)")
}

/// The hot-low torque's gap to the requirement, and whether it meets it: `hot_low_Nm >=
/// required_Nm`, as the engine's hot minimum check (which fails `hot_low < required`). Two
/// decimals of N·m and a whole percent of the requirement, each signed: "short 0.29 N·m
/// (-12 %)", "margin +0.40 N·m (+16 %)". The percent is left out when the requirement is not a
/// positive torque (the slider stops at its metadata minimum; a design file need not).
#[allow(non_snake_case)] // unit suffix
pub fn torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool) {
    let meets = hot_low_Nm >= required_Nm;
    let gap_Nm = (hot_low_Nm - required_Nm).abs();
    // A shortfall's word carries its sign; a margin's torque shows "+".
    let (word, torque, sign) = if meets {
        ("margin", format!("+{gap_Nm:.2}"), '+')
    } else {
        ("short", format!("{gap_Nm:.2}"), '-')
    };
    let text = if required_Nm > 0.0 {
        let percent = 100.0 * gap_Nm / required_Nm;
        format!("{word} {torque} N·m ({sign}{percent:.0} %)")
    } else {
        format!("{word} {torque} N·m")
    };
    (text, meets)
}

/// Whether the Size tab shows `torque_Nm` as a number: the end effect is inside the model's
/// range (no banner) and the torque is finite. Otherwise the readout reads [`NO_PREDICTION`]
/// (spec 3.2 and 5).
#[allow(non_snake_case)] // unit suffix
pub(crate) fn predicts(results: &DesignResults, torque_Nm: f64) -> bool {
    end_effect_out_of_range(results).is_none() && torque_Nm.is_finite()
}
```

- [ ] **Step 4: Run the unit tests.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::size_controls`
  Expected: 5 passed. Until step 7 uses `REQUIRED`, `HOT_LOW` and the other new items, rustc may print `dead_code` warnings for them. These are warnings, not errors.

- [ ] **Step 5: Write the failing panel tests.**

  (a) In `panel.rs` `mod tests`, after the existing `use crate::headline;` line, add:
```rust
    use crate::engine::size_controls::{DiameterFit, diameter_breakdown, dimensions};
    use crate::gui::guidance::{Consequences, RING_SIZE_STAYS, plain_text_violations};
    use crate::gui::size_controls::{
        DIAMETER, GAP, HOT_LOW, LENGTH, MAGNETS, NO_PREDICTION, REQUIRED, predicts, pullout_label,
        torque_gap_text,
    };
```
  (b) In `manual_size_does_not_present_out_of_range_torque_as_a_prediction`, replace `assert_eq!(count(&output, "Outside model range"), 2);` with:
```rust
        // Hot-low and the pull-out read no prediction, and the gap is not shown (spec 3.2, 5).
        assert_eq!(count(&output, NO_PREDICTION), 2);
        assert_eq!(beside(&output, HOT_LOW)[0].0, NO_PREDICTION);
        let (gap, _) = torque_gap_text(
            harness.panel.results().metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        assert_eq!(count(&output, &gap), 0, "{gap}");
```
  (c) After the closing `}` of `manual_size_adopts_the_inverse_solution_before_editing`, insert:
```rust
    /// `DesignInputs::default()` with `npole` magnets per ring.
    fn with_poles(npole: i64) -> DesignInputs {
        let mut inputs = DesignInputs::default();
        inputs.coupling.npole = npole;
        inputs
    }

    /// A harness on a screen of `screen` showing the Size tab for `inputs`, set as the panel's
    /// inputs: the tab clicked, then one more frame, so the inputs side draws its controls.
    fn size_tab(screen: egui::Vec2, inputs: DesignInputs) -> Harness {
        let mut harness = Harness::on_screen(screen);
        harness.panel.inputs = inputs;
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
        harness
    }

    /// The what-moves-what lines of the design the panel shows, as the Size tab computes them:
    /// that design, its own results, the panel's Minimum OD.
    fn shown_consequences(harness: &Harness) -> Consequences {
        let shown = harness.panel.shown_inputs();
        Consequences::of(&shown, &compute_all(&shown), harness.panel.minimum_diameter_mm)
    }

    /// The rows egui laid the first drawn text equal to `needle` out on: more than one once it
    /// wrapped.
    fn rows(output: &egui::FullOutput, needle: &str) -> usize {
        crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .find_map(|shape| match shape {
                egui::Shape::Text(text) if text.galley.text() == needle => {
                    Some(text.galley.rows.len())
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no {needle:?}"))
    }

    /// The texts drawn on the row of the label `label` and right of it, nearest first, with
    /// their rects: what a readout row shows beside its label (texts further right on the same
    /// row, the dashboard's, come after).
    fn beside(output: &egui::FullOutput, label: &str) -> Vec<(String, egui::Rect)> {
        let at = text_rect(output, label).unwrap_or_else(|| panic!("no {label:?}"));
        let mut row: Vec<(String, egui::Rect)> = crate::gui::test_support::flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Text(text) => {
                    let rect = text.galley.rect.translate(text.pos.to_vec2());
                    let on_row = (rect.center().y - at.center().y).abs() < at.height() / 2.0;
                    (on_row && rect.left() >= at.right())
                        .then(|| (text.galley.text().to_owned(), rect))
                }
                _ => None,
            })
            .collect();
        row.sort_by(|a, b| a.1.left().total_cmp(&b.1.left()));
        row
    }

    #[test]
    fn size_tab_reads_the_requirement_the_hot_low_gap_and_the_nominal_pull_out() {
        // Spec 3.2 at the default design, whose hot-low torque is below the requirement: the gap
        // reads short, in red, as the dashboard's hot minimum check fails.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let results = harness.panel.results().clone();
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, HOT_LOW, label.as_str()] {
            assert_eq!(count(&output, text), 1, "{text}");
        }
        assert_eq!(count(&output, "Expected torque"), 0, "replaced by the nominal pull-out");
        let shown = |path: &str| with_unit(format_value(&results.get(path).unwrap()), "N\u{b7}m");
        let required = harness.panel.inputs().metal.required_min_Nm;
        let (gap, meets) = torque_gap_text(results.metal.torque_hot_low_Nm, required);
        assert!(!meets && gap.starts_with("short "), "{gap}");
        let hot_low_row: Vec<String> = beside(&output, HOT_LOW)
            .into_iter()
            .take(2)
            .map(|(text, _)| text)
            .collect();
        assert_eq!(hot_low_row, [shown("metal.torque_hot_low_Nm"), gap.clone()]);
        assert_eq!(beside(&output, &label)[0].0, shown("model.pullout_Nm"));
        assert!(
            beside(&output, REQUIRED)[0].0.ends_with(" N\u{b7}m"),
            "the Required slider's value box"
        );
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(
            crate::gui::dashboard::check_level(&results, "metal.hot_min_check"),
            Some(Level::Bad),
            "the gap agrees with the hot minimum check"
        );
        // The label follows the live operating temperature.
        harness.panel.inputs.coupling.op_temp_C = 37.5;
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.results().metal.op_temp_C, 37.5);
        assert_eq!(count(&output, &pullout_label(37.5)), 1);
        assert_eq!(count(&output, &label), 0);
    }

    #[test]
    fn size_tab_required_slider_is_an_ordinary_design_edit() {
        // Spec 3.2 and G-2: the requirement's own slider is an edit like any other (one undo
        // step, carried by files and links), typed values clamped into its range (decision
        // M41-2); a value that is no number changes nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let before = harness.panel.design();
        let undo = harness.panel.history.undo_len();
        let type_in = |harness: &mut Harness, text: &str| {
            let output = harness.frame(Vec::new());
            let (_, value_box) = beside(&output, REQUIRED).remove(0);
            harness.click(value_box.center());
            harness.frame([select_all(), vec![egui::Event::Text(text.to_owned())]].concat());
            harness.frame(key_tap(egui::Key::Enter));
            harness.frame(Vec::new())
        };
        let output = type_in(&mut harness, "2");
        let mut lowered = before.inputs.clone();
        lowered.metal.required_min_Nm = 2.0;
        assert_eq!(harness.panel.inputs(), &lowered, "only the requirement changed");
        assert_eq!(harness.panel.results(), &compute_all(&lowered));
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), undo + 1);
        // The gap now reads a margin, in green, from the design's own hot-low torque.
        let hot_low = harness.panel.results().metal.torque_hot_low_Nm;
        let (gap, meets) = torque_gap_text(hot_low, 2.0);
        assert!(meets && gap.starts_with("margin +"), "{gap}");
        assert_eq!(
            crate::gui::test_support::text_color(&output, &gap),
            Some(Level::Good.color(&harness.ctx.style().visuals))
        );
        // Files and links carry it.
        let design = harness.panel.design();
        assert_eq!(decode_share_payload(&encode_share_payload(&design)).unwrap(), design);
        assert_eq!(design_from_json(&design_to_json(&design)).unwrap(), design);
        // Past the slider's range: clamped to its end.
        let range = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .unwrap()
            .meta
            .range
            .unwrap();
        type_in(&mut harness, &format!("{}", range.max * 5.0));
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, range.max);
        assert_eq!(harness.panel.history.undo_len(), undo + 2);
        // Not a number: nothing changes.
        type_in(&mut harness, "NaN");
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, range.max);
        assert_eq!(harness.panel.history.undo_len(), undo + 2);
        // One undo per edit.
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &lowered);
        harness.panel.undo();
        assert_eq!(harness.panel.design(), before);
    }

    #[test]
    fn size_tab_idle_frames_keep_an_off_step_or_out_of_range_requirement() {
        // Spec 5 and the invariant of `idle_frames_change_no_input`: the Required slider writes
        // only on an edit (decision M41-2), so a file's requirement off its step or past its
        // range is kept, and drawing the what-moves-what lines edits nothing.
        let range = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .unwrap()
            .meta
            .range
            .unwrap();
        for required in [2.345, range.max + 2.0] {
            let mut design = Design::default();
            design.inputs.metal.required_min_Nm = required;
            design.inputs.coupling.npole = 8;
            let mut harness = Harness::new();
            harness.panel.open_design(design.clone());
            harness.click_text(CentreView::SizeAndTorque.label());
            for _ in 0..10 {
                harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.design(), design, "{required}");
            assert_eq!(harness.panel.last_error, None, "{required}");
            assert_eq!(harness.panel.history.undo_len(), 0, "{required}: no edit reported");
        }
    }

    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
        // Spec 3.3 at the default design: the Set-by line under Outside diameter, the length
        // split under Overall length, and the magnet count's lines under Magnets per ring, the
        // ring-size line first, each under the one before.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        let consequences = shown_consequences(&harness);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let diameter = consequences.diameter_line().expect("the default design has a breakdown");
        let length = consequences.length_line().expect("and a length split");
        assert_eq!(count(&output, &diameter), 1);
        assert_eq!(count(&output, &length), 1);
        assert!(rect(DIAMETER).bottom() <= rect(diameter.as_str()).top());
        assert!(rect(diameter.as_str()).bottom() <= rect(LENGTH).top());
        assert!(rect(LENGTH).bottom() <= rect(length.as_str()).top());
        assert!(rect(length.as_str()).bottom() <= rect(GAP).top());
        let lines = consequences.magnet_lines(10);
        assert_eq!(lines.first().map(String::as_str), Some(RING_SIZE_STAYS));
        let mut above = rect(MAGNETS);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
            let at = rect(line.as_str());
            assert!(above.bottom() <= at.top() + 0.5, "{line:?} is not under the line above it");
            above = at;
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "drawing them edits nothing");
    }

    #[test]
    fn size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter() {
        // Spec 6 test 10, the owner's case of 2026-10-13: from 10 to 8 magnets the coupling did
        // not shrink and nothing said why. At 8 magnets the Size tab says the ring keeps its
        // size, and its Set-by line shows the cap as the largest part: the outside diameter is
        // the cap's, so fewer magnets cannot lower it.
        let mut harness = size_tab(SCREEN, with_poles(8));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, RING_SIZE_STAYS), 1);
        let shown = harness.panel.shown_inputs();
        let parts =
            diameter_breakdown(&shown).expect("the default ring at 8 magnets has a breakdown");
        assert!(parts.cap_mm > parts.cup_mm && parts.cap_mm > parts.boss_mm, "{parts:?}");
        assert_eq!(
            parts.cap_mm,
            dimensions(&shown).expect("size controls").diameter_mm,
            "the cap is the outside diameter"
        );
        let consequences = shown_consequences(&harness);
        assert_eq!(consequences.diameter, Some(parts));
        let set_by = consequences.diameter_line().expect("a Set-by line");
        assert!(set_by.starts_with("Set by: cap "), "{set_by}");
        assert_eq!(count(&output, &set_by), 1);
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        assert!(rect(DIAMETER).bottom() <= rect(set_by.as_str()).top(), "under Outside diameter");
        assert!(rect(set_by.as_str()).bottom() <= rect(LENGTH).top(), "above Overall length");
        assert!(rect(MAGNETS).bottom() <= rect(RING_SIZE_STAYS).top(), "under Magnets per ring");
    }

    #[test]
    fn size_tab_what_moves_what_lines_follow_each_edit() {
        // Spec 6 test 5: the lines are recomputed when the design changes (the cache keys on
        // it), from the magnet count's own slider and from any other input.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let at_10 = shown_consequences(&harness).magnet_lines(10);
        // The controls' widgets in drawing order: diameter, length, gap, magnets, Minimum OD.
        let magnets = harness.panel.design_widgets[3];
        harness.ctx.memory_mut(|m| m.request_focus(magnets));
        // Establish egui's arrow-key focus lock before sending the keyboard edit.
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowLeft));
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.inputs().coupling.npole, 8);
        let at_8 = shown_consequences(&harness).magnet_lines(8);
        assert_ne!(at_8, at_10);
        for line in &at_8 {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_10.iter().filter(|line| !at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // Another input: a wider face gap moves every neighbour's pull-out and cup.
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        let output = harness.frame(Vec::new());
        let wider = shown_consequences(&harness);
        let lines = wider.magnet_lines(8);
        assert_ne!(lines, at_8);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_8.iter().filter(|line| !lines.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        for line in wider.diameter_line().into_iter().chain(wider.length_line()) {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
    }

    #[test]
    fn size_tab_smallest_diameter_line_searches_with_the_minimum_od() {
        // Spec 3.3 and 3.5: the fit search covers the Size tab's diameter range, the Minimum OD
        // preference included. At 8 magnets the blocks fit at the slider's floor, so the line
        // takes its minimum form; a lower Minimum OD lets the search find the smallest fit.
        let mut harness = size_tab(SCREEN, with_poles(8));
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        let floor = shown_consequences(&harness);
        assert!(matches!(floor.fit, Some(DiameterFit::AtMinimum(_))), "{:?}", floor.fit);
        let undo = harness.panel.history.undo_len();
        harness.panel.minimum_diameter_mm = Some(34.0);
        let output = harness.frame(Vec::new());
        let lowered = shown_consequences(&harness);
        assert!(matches!(lowered.fit, Some(DiameterFit::At(_))), "{:?}", lowered.fit);
        let (old, new) = (floor.magnet_lines(8), lowered.magnet_lines(8));
        assert_ne!(old, new);
        for line in &new {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in old.iter().filter(|line| !new.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // A view preference: the design and its history stay as they were.
        assert_eq!(harness.panel.inputs(), &with_poles(8));
        assert_eq!(harness.panel.history.undo_len(), undo);
    }

    #[test]
    fn size_tab_magnet_slider_ends_show_only_the_neighbour_in_range() {
        // Spec 5: at 4 or 40 magnets per ring the line for 2 or 42 is left out.
        for (npole, inside, outside) in [(4, 6, 2), (40, 38, 42)] {
            let mut harness = size_tab(SCREEN, with_poles(npole));
            let output = harness.frame(Vec::new());
            let consequences = shown_consequences(&harness);
            let neighbours: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(neighbours, [inside], "{npole}");
            let lines = consequences.magnet_lines(npole);
            assert!(lines.contains(&RING_SIZE_STAYS.to_owned()), "{lines:?}");
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            let texts = drawn_texts(&output);
            assert!(
                !texts.iter().any(|t| t.starts_with(&format!("At {outside}:"))),
                "{npole}: {texts:?}"
            );
        }
    }

    #[test]
    fn size_tab_narrow_window_wraps_the_new_lines_inside_their_sides() {
        // Spec 5 "Narrow windows", at the M4-1 review's ~930 px window and the owner's 8 magnets:
        // the what-moves-what lines wrap inside the inputs side (its 320 points), one under the
        // next, and the torque readouts stay in the centre region, left of the dashboard.
        let mut harness = size_tab(egui::vec2(930.0, 1024.0), with_poles(8));
        let output = harness.frame(Vec::new());
        let side = |id: &str| {
            egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id))
                .unwrap_or_else(|| panic!("{id} is drawn"))
                .rect
        };
        let inputs_left = side("magcoupling_inputs").left();
        let inputs_right = inputs_left + INPUTS_WIDTH;
        let dashboard_left = side("magcoupling_dashboard").left();
        let rect = |text: &str| text_rect(&output, text).unwrap_or_else(|| panic!("no {text:?}"));
        let consequences = shown_consequences(&harness);
        let magnet_lines = consequences.magnet_lines(8);
        let mut above = rect(MAGNETS);
        for line in &magnet_lines {
            let at = rect(line.as_str());
            assert!(above.bottom() <= at.top() + 0.5, "{line:?} overlaps the line above it");
            above = at;
        }
        let mut lines = magnet_lines.clone();
        lines.extend(consequences.diameter_line());
        lines.extend(consequences.length_line());
        for line in &lines {
            let at = rect(line.as_str());
            assert!(
                at.left() >= inputs_left - 1.0 && at.right() <= inputs_right + 1.0,
                "{line:?} at {at:?} runs out of the inputs side ({inputs_left} to {inputs_right})"
            );
        }
        assert!(
            lines.iter().any(|line| rows(&output, line) > 1),
            "no line wrapped: the test no longer exercises wrapping"
        );
        let results = harness.panel.results().clone();
        let (gap, _) = torque_gap_text(
            results.metal.torque_hot_low_Nm,
            harness.panel.inputs().metal.required_min_Nm,
        );
        let label = pullout_label(results.metal.op_temp_C);
        for text in [REQUIRED, HOT_LOW, gap.as_str(), label.as_str()] {
            let at = rect(text);
            assert!(
                at.left() >= inputs_right - 1.0 && at.right() <= dashboard_left + 1.0,
                "{text:?} at {at:?} runs out of the centre ({inputs_right} to {dashboard_left})"
            );
        }
    }

    #[test]
    fn size_tab_new_texts_have_glyphs_and_plain_words() {
        // Spec 6 tests 3 and 11 for the Size tab, in each state its new texts take: the
        // default, a met requirement, both ends of the magnet slider, the owner's 8 magnets, no
        // prediction (short magnets) and a ring too small for its magnets. Every drawn text has
        // glyphs in the default fonts; the new ones name no workbook cell, correction or sheet.
        let mut met = DesignInputs::default();
        met.metal.required_min_Nm = 2.0;
        let too_small =
            crate::engine::size_controls::resize_diameter_with_minimum(&with_poles(8), 35.0, 34.0)
                .unwrap();
        let states = [
            (DesignInputs::default(), None),
            (met, None),
            (with_poles(4), None),
            (with_poles(8), None),
            (with_poles(40), None),
            (short_magnets(), None),
            (too_small, Some(34.0)),
        ];
        let mut harness = size_tab(egui::vec2(1280.0, 4000.0), DesignInputs::default());
        let mut drawn = Vec::new();
        let mut new_texts = vec![REQUIRED.to_owned(), HOT_LOW.to_owned()];
        for (inputs, minimum) in states {
            harness.panel.inputs = inputs;
            harness.panel.minimum_diameter_mm = minimum;
            // The second frame draws with any minimum the first one pinned.
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            let shown = harness.panel.shown_inputs();
            let results = compute_all(&shown);
            let consequences = shown_consequences(&harness);
            let mut texts = consequences.magnet_lines(shown.coupling.npole);
            if dimensions(&shown).is_some() {
                texts.extend(consequences.diameter_line());
                texts.extend(consequences.length_line());
            }
            texts.push(pullout_label(results.metal.op_temp_C));
            let hot_low = results.metal.torque_hot_low_Nm;
            texts.push(if predicts(&results, hot_low) {
                torque_gap_text(hot_low, shown.metal.required_min_Nm).0
            } else {
                NO_PREDICTION.to_owned()
            });
            let all = drawn_texts(&output);
            for text in &texts {
                assert!(all.contains(text), "{text:?} is not drawn");
            }
            new_texts.extend(texts);
            drawn.extend(all);
        }
        for text in &drawn {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the Size tab");
        }
        for text in &new_texts {
            let violations = plain_text_violations(text);
            assert!(violations.is_empty(), "{text:?}: {violations:?}");
        }
    }
```

- [ ] **Step 6: Run the panel tests and confirm the expected failures.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::panel::tests::size_tab_ gui::panel::tests::manual_size_does_not_present_out_of_range_torque_as_a_prediction`
  Expected: the tests compile and these fail:

  | Test | Expected failure |
  |---|---|
  | `manual_size_does_not_present...` | `assertion 'left == right' failed`, left 0, right 2 (no `NO_PREDICTION` drawn) |
  | `..._reads_the_requirement...` | `Required`: left 0, right 1 |
  | `..._required_slider_is_an_ordinary_design_edit` | panics `no "Required"` |
  | `..._draws_each_..._line_under_its_control` | count of the Set-by line: left 0, right 1 |
  | `..._owner_case_at_8_magnets...` | `RING_SIZE_STAYS`: left 0, right 1 |
  | `..._lines_follow_each_edit` | count of each line: left 0, right 1 |
  | `..._smallest_diameter_line_searches_with_the_minimum_od` | left 0, right 1 |
  | `..._magnet_slider_ends...` | left 0, right 1 |
  | `..._narrow_window...` | panics `no "<a magnet line>"` |
  | `..._new_texts_have_glyphs...` | `"<a magnet line>" is not drawn` |

  `size_tab_idle_frames_keep_an_off_step_or_out_of_range_requirement` passes before and after. It guards an invariant: it fails only if the new slider or the cache writes on idle frames.

- [ ] **Step 7: Draw the consequence lines in `controls_ui` (`magcoupling-rs/src/gui/size_controls.rs`).**

  (a) Add `use crate::gui::guidance::Consequences;` after the `crate::gui::dashboard` import.

  (b) Replace the doc and signature
```rust
/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
pub(crate) fn controls_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
) -> ControlsOutput {
```
  with
```rust
/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
/// `consequences` are `inputs`' what-moves-what lines (spec 3.3), each drawn under its control:
/// they show the design before this frame's edit, and the panel asks for the next frame.
pub(crate) fn controls_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: &Consequences,
) -> ControlsOutput {
```
  (c) In the diameter/length loop, change the header `for (label, path, mut value, range) in [` to `for (label, path, mut value, range, line) in [`. Add `consequences.diameter_line(),` as the fifth element of the DIAMETER tuple, after `size.diameter_range,`. Add `consequences.length_line(),` as the fifth element of the LENGTH tuple, after `size.length_range,`. Then replace
```rust
                    match resized {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
            }
            ui.weak(format!(
```
  with
```rust
                    match resized {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
                if let Some(line) = line {
                    consequence_line(ui, &line);
                }
            }
            ui.weak(format!(
```
  (d) Replace the gap/magnets loop header `for (label, path) in [(GAP, "metal.face_gap_mm"), (MAGNETS, "coupling.npole")] {` with
```rust
        for (label, path, lines) in [
            (GAP, "metal.face_gap_mm", Vec::new()),
            (
                MAGNETS,
                "coupling.npole",
                consequences.magnet_lines(inputs.coupling.npole),
            ),
        ] {
```
  and replace that loop's tail
```rust
                if let Err(error) = output.inputs.set(path, value) {
                    output.error = Some(format!("{error}"));
                }
            }
        }
        if let Some(widget) = minimum_widget {
```
  with
```rust
                if let Err(error) = output.inputs.set(path, value) {
                    output.error = Some(format!("{error}"));
                }
            }
            for line in &lines {
                consequence_line(ui, line);
            }
        }
        if let Some(widget) = minimum_widget {
```
  (e) After `controls_ui`'s closing `}` (before `#[cfg(test)]`), add:
```rust

/// One what-moves-what line under its control: weak text, wrapped to the side's width.
fn consequence_line(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(egui::RichText::new(text).weak()).wrap());
}
```

- [ ] **Step 8: Wire the cache and the torque readouts in `magcoupling-rs/src/gui/panel.rs`.**

  (a) Imports:
  - between `use crate::gui::geometry_view::geometry_ui;` and `use crate::gui::history::History;` add `use crate::gui::guidance::ConsequenceCache;`;
  - between the `use crate::gui::session::{...};` block and the `use crate::gui::sizing::{...};` block add `use crate::gui::size_controls;`;
  - check that T3 left `REQUIREMENT_INPUT` in the `crate::gui::sizing` import: `grep -n "REQUIREMENT_INPUT" magcoupling-rs/src/gui/panel.rs` must list the import line. If it does not, add `REQUIREMENT_INPUT` to that brace list.

  (b) Field. After
```rust
    /// The user-selected lower OD limit, a view preference that does not resize the design.
    minimum_diameter_mm: Option<f64>,
```
  add
```rust
    /// The Size tab's what-moves-what lines (spec 3.3) for the design shown and the minimum OD,
    /// recomputed only when either changes: view state, in no design file, link or undo step.
    consequences: ConsequenceCache,
```
  (c) In `new()`, after `minimum_diameter_mm: None,` add `consequences: ConsequenceCache::default(),`.

  (d) `inputs_ui` Size arm. Replace
```rust
            let shown = self.shown_inputs();
            let output = ui
                .add_enabled_ui(self.sizing.mode == SizingMode::MagnetsToTorque, |ui| {
                    crate::gui::size_controls::controls_ui(ui, &shown, self.minimum_diameter_mm)
                })
                .inner;
            self.design_widgets.extend(output.widgets);
```
  with
```rust
            let shown = self.shown_inputs();
            let minimum = self.minimum_diameter_mm;
            // The cache keys on the design alone, so it is handed that design's own results:
            // `self.results` can lag the inputs by a frame (the header's assumptions reset edits
            // them after the last compute), and a lagging pull-out would stick in the lines until
            // the next edit. One compute_all (about 17 us) per frame of this tab; the neighbour
            // runs and the fit search rerun only when the design or the minimum changes.
            let shown_results = compute_all(&shown);
            let consequences = self.consequences.get(&shown, &shown_results, minimum);
            let output = ui
                .add_enabled_ui(self.sizing.mode == SizingMode::MagnetsToTorque, |ui| {
                    crate::gui::size_controls::controls_ui(ui, &shown, minimum, consequences)
                })
                .inner;
            // The lines under the controls were drawn from the design before this frame's edit:
            // one more frame shows the edited design's.
            if output.inputs != shown || output.minimum_diameter_mm != minimum {
                ui.ctx().request_repaint();
            }
            self.design_widgets.extend(output.widgets);
```
  (e) `view_ui` Size arm. Replace
```rust
            CentreView::SizeAndTorque => {
                let outside_model = end_effect_banner(&self.results).is_some();
                ui.horizontal_wrapped(|ui| {
                    for (label, path) in [
                        ("Expected torque", "model.pullout_Nm"),
                        ("Hot-low torque", "metal.torque_hot_low_Nm"),
                    ] {
                        ui.label(label);
                        let value = self.results.get(path).unwrap_or(Value::None);
                        let text = if outside_model {
                            egui::RichText::new("Outside model range").weak()
                        } else {
                            egui::RichText::new(with_unit(format_value(&value), "N\u{b7}m"))
                        };
                        let response = ui.add(egui::Label::new(text).sense(egui::Sense::click()));
                        readouts.show(ui, response, path, || {
                            crate::gui::dashboard::hover_text(path).unwrap_or_default()
                        });
                    }
                });
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
```
  with
```rust
            CentreView::SizeAndTorque => {
                self.torque_readouts_ui(ui, readouts);
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
```
  (f) Between the closing `}` of `view_ui` and the doc comment of `export_spreadsheet`, add:
```rust

    /// The Size tab's torque readouts (spec 3.2): the requirement's slider, an ordinary design
    /// edit (undo, files, links); the hot-low torque with its gap to the requirement, red short
    /// or green margin; the nominal pull-out at the operating temperature. Where the model makes
    /// no prediction a torque reads [`size_controls::NO_PREDICTION`] and the gap is left out.
    fn torque_readouts_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts) {
        let entry = InputCatalogue::get()
            .entry(REQUIREMENT_INPUT)
            .expect("the requirement has input metadata");
        let range = entry.meta.range.expect("the requirement has a slider");
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::REQUIRED);
            let mut value = required;
            let response = ui
                .push_id(REQUIREMENT_INPUT, |ui| {
                    ui.add(slider(&mut value, entry.meta, range))
                })
                .inner;
            // Its value box is a text field of the design, as an input row's is.
            self.design_widgets.push(response.id);
            if response.changed() && value != required {
                self.last_error = self
                    .inputs
                    .set(REQUIREMENT_INPUT, Value::Num(value))
                    .err()
                    .map(|e| e.to_string());
                // The dashboard was drawn before this edit: the readouts below show it now, the
                // dashboard on the next frame.
                self.results = compute_all(&self.shown_inputs());
                ui.ctx().request_repaint();
            }
        });
        let required = self.inputs.metal.required_min_Nm;
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::HOT_LOW);
            if let Some(hot_low) =
                torque_readout(ui, &self.results, "metal.torque_hot_low_Nm", readouts)
            {
                let (gap, meets) = size_controls::torque_gap_text(hot_low, required);
                let level = if meets { Level::Good } else { Level::Bad };
                ui.colored_label(level.color(ui.visuals()), gap);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(size_controls::pullout_label(self.results.metal.op_temp_C));
            torque_readout(ui, &self.results, "model.pullout_Nm", readouts);
        });
    }
```
  (g) After the closing `}` of `fn assumptions_banner`, add:
```rust

/// One torque readout of the Size tab: the value of the result at `path` with its unit, a
/// readout (`readouts`: hover text and equation; a click opens it), or
/// [`size_controls::NO_PREDICTION`] where the model makes none ([`size_controls::predicts`]).
/// Returns the torque shown [N·m], if any.
fn torque_readout(
    ui: &mut egui::Ui,
    results: &DesignResults,
    path: &str,
    readouts: &mut Readouts,
) -> Option<f64> {
    let Some(Value::Num(torque)) = results.get(path) else {
        unreachable!("{path} is a torque result")
    };
    if !size_controls::predicts(results, torque) {
        ui.weak(size_controls::NO_PREDICTION);
        return None;
    }
    let text = with_unit(format_value(&Value::Num(torque)), "N\u{b7}m");
    let response = ui.add(egui::Label::new(text).sense(egui::Sense::click()));
    readouts.show(ui, response, path, || {
        crate::gui::dashboard::hover_text(path).unwrap_or_default()
    });
    Some(torque)
}
```

- [ ] **Step 9: Run the panel tests and the unit tests.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::panel::tests::size_tab_ gui::panel::tests::manual_size_does_not_present_out_of_range_torque_as_a_prediction`
    Expected: 11 passed.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::size_controls`
    Expected: 5 passed.

- [ ] **Step 10: Run the wider suites for the touched modules.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`
    Expected: all pass, including:
    - `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`, which now draws the new Size tab texts;
    - `idle_frames_change_no_input`;
    - `manual_size_has_an_adjustable_minimum_od`, `loaded_smaller_designs_keep_their_size_and_expose_a_stable_minimum` and `manual_size_edits_update_torque_and_are_one_undo_step`. Their `design_widgets[0]`/`[2]` indices are unchanged: the Required slider is pushed after the inputs side;
    - `inside_an_egui_window_no_centre_view_reaches_into_the_equation_panel` and `the_equation_panel_draws_in_a_tiny_window`;
    - the session, sizing and guidance suites.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test python_schema --test size_controls --test sizing`
    Expected: all pass. No engine label, help or verdict text changed.
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
    Expected: no warnings.
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`
    Expected: no diff. If it reports one, run `cargo fmt --manifest-path magcoupling-rs/Cargo.toml` and confirm with `git diff --stat` that only `size_controls.rs` and `panel.rs` changed.

- [ ] **Step 11: Commit.**
```bash
git add magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/panel.rs
git commit -F - <<'EOF'
feat(magcoupler): Size tab requirement, hot-low gap and what-moves-what lines

The Size tab's centre reads Required (a slider on metal.required_min_Nm,
an ordinary design edit: undo, files, links), Hot-low with its live gap to
the requirement (red "short", green "margin") and "Pull-out torque
(nominal, at T)" at the live operating temperature, replacing "Expected
torque". With the end-effect banner or a non-finite torque both read "no
prediction" and the gap is left out. Under Magnets per ring, Outside
diameter and Overall length the inputs side draws the guidance layer's
consequence lines from a ConsequenceCache on the panel (spec 3.2, 3.3),
including the owner's 8-magnet case: the ring keeps its size and the cap
sets the outside diameter.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```

**Spec section 5 edge cases owned by T6, and their tests:**

| Edge case | Test |
|---|---|
| Magnet count at 4 or 40 | `size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` |
| Neighbour invalid or not finite (words) | T5 owns the wording. T6 draws them at 40 magnets (does not fit) and with short magnets (no prediction) in `size_tab_new_texts_have_glyphs_and_plain_words` |
| Hot-low not finite, or end effect out of range | `a_torque_is_shown_only_inside_the_end_effect_range_and_finite` (unit), plus the edited `manual_size_does_not_present_out_of_range_torque_as_a_prediction` |
| Requirement <= 0 (gap text) | `a_requirement_that_is_no_positive_torque_drops_the_percent`. The solver's refusal wording belongs to T3 and T7 |
| Narrow windows and glyphs | `size_tab_narrow_window_wraps_the_new_lines_inside_their_sides`, `size_tab_new_texts_have_glyphs_and_plain_words`, `the_readout_texts_have_glyphs_and_plain_words` |
| Idle frames never edit | `size_tab_idle_frames_keep_an_off_step_or_out_of_range_requirement` |
| Files and links | `size_tab_required_slider_is_an_ordinary_design_edit` (share link and design file round trips) |

**Notes for the reviewers:**
- **Repeated code.** Until T7 removes it, T3's Key-design target slider in `sizing_ui` also edits `REQUIREMENT_INPUT`. It repeats the entry/range lookup and the set-and-error code of `torque_readouts_ui`. T7's contract replaces the Key-design sizing controls with a status line, which removes the copy. If T7 keeps that slider, T7 must move both into one private `requirement_slider_ui`.
- **Physics review (G-3).** New T6 wording: "Pull-out torque (nominal, at T °C)", "Hot-low", the short/margin gap, and `NO_PREDICTION`. `NO_PREDICTION` names one cause (magnets too short). The spec also uses it for a hot-low torque that is not finite while the end effect is in range. The reviewer should confirm whether validated inputs can reach that case.
- **Extra compute per frame.** The extra `compute_all` on the Size tab is deliberate (see the comment in step 8d). The cache keys on the inputs alone, so stale `self.results` would stay wrong in the lines until the next edit.

**Existing tests this change breaks, with the exact edit:**
- `manual_size_does_not_present_out_of_range_torque_as_a_prediction` (panel.rs, about line 3791): replace `assert_eq!(count(&output, "Outside model range"), 2);` with the block in step 5(b).
- No other test should break. Two to watch:
  - `inside_an_egui_window_no_centre_view_reaches_into_the_equation_panel` asserts that the window keeps its size on the Size tab. The new lines take the inputs side from about 370 to about 550 of the roughly 620 points available in the 1100 x 700 window, so it should stay green.
  - For T9: once the Size tab is the default view, `only_the_rows_drawn_this_frame_count_as_design_fields` will count the Size tab's widgets (5 controls plus the Required slider). That edit belongs in T9's list.

---

### Task 7: Auto solver on the Size tab (spec 3.5, G-4, G-7)

> **Amendments:** apply "T7 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


**Files:**
- Modify: `magcoupling-rs/src/gui/sizing.rs`. Module doc lines 1-2. Add `auto_choice` after `variable_label` (about 72-79) and `REQUIREMENT_NOT_POSITIVE` after `SOLVING` (about 118-119). Change the `SizingRunner::status` arms (about 230-259), split `variable_text` (about 262-272) and add `with_own_value`. Tests go in `mod tests` (about 280-481).
- Modify: `magcoupling-rs/src/gui/size_controls.rs`. Module doc and imports at 1-10. Add the Auto constants and functions after `LINKED_HOUSING` (line 18). Change the `controls_ui` signature (27-32) and its two slider calls (about 90-92 and 131-133). Add `size_slider` after `controls_ui`. Tests go in a new `#[cfg(test)] mod auto_tests`.
- Modify: `magcoupling-rs/src/gui/panel.rs`. Module doc 22-25, imports 27-57, `SIZED_NOTE` 104-105. Add `search_range` and `auto_outcome` after `shown_inputs` (328-335). Change `run_sizing`, `log_sizing` and `set_sizing_mode` (538-580), replace `sizing_ui` (582-642), change the tab row (683-692), `export_spreadsheet` (795-796) and the `inputs_ui` Size-tab block (918-938). Add `status_color` after `assumptions_banner` (1265-1274). Tests: the `Harness` impl (1361-1367) plus the edits listed at the end of this section.
- Modify: `magcoupling-rs/README.md`, lines 177-179 and 423-429.
- Test: the unit tests in `gui/sizing.rs` and `gui/size_controls.rs`, and the headless panel tests in `gui/panel.rs` (harness `Harness` with `frame`, `frame_after`, `click_text`, `click`, `solved`, the new `size_widget`, and `test_support::{drawn_texts, text_rects, text_color, key_tap, primary_button, short_magnets, assert_glyphs}`).

Line numbers are as of commit 022e254. T3 and T6 shift them; every edit below quotes its anchor text.

**Interfaces:**
- Consumes (contract):
  - T1: `engine::size_controls::ring_radius_range(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Option<SliderRange>`
  - T2: `engine::sizing::solve_in(inputs, variable, target_Nm, range) -> Result<SizingOutcome, SizingError>`, with `solve(..) == solve_in(.., variable.range())`
  - T3: `SizingState { pub mode: SizingMode, pub variable: FreeVariable }`, `REQUIREMENT_INPUT`, and the runner's target `inputs.metal.required_min_Nm`. `SizingRunner::{status(&self, inputs, state, range), solve_now(&mut self, inputs, state, range), shown(&self, inputs, state, range), is_current(&self, inputs, state, range)}`. The range is part of `SolveKey`, and T3's panel passes `self.sizing.variable.range()` at every call site.
  - T4: `gui::guidance::plain_text_violations(text: &str) -> Vec<&'static str>`
  - T6: `size_controls::controls_ui` with whatever parameters T6 added; T7 appends one more.
- Produces (contract, in `gui/size_controls.rs`):
  - `pub const AUTO: &str = "Auto: smallest";`
  - `pub const AUTO_SUFFIX: &str = "that meets the requirement";`
  - `pub const SET_BY_AUTO: &str = "Set by Auto; turn Auto off to edit";`
  - `pub fn variable_slider(variable: FreeVariable) -> &'static str` (AxialLength -> `LENGTH`, RingRadius -> `DIAMETER`, MagnetsPerRing -> `MAGNETS`)
  - `pub fn change_text(variable: FreeVariable, from: &DesignInputs, to: &DesignInputs) -> String`
  - `pub const AUTO_STATUS_START: &str = "Auto: ";`
  - Auto on == `SizingMode::TorqueToMagnets`. Ring size passes `ring_radius_range(inputs, minimum_diameter_mm)` as the range; the others pass `variable.range()`.
- Also produced (extra, nothing renamed):
  - `size_controls::{AUTO_HOVER, OPEN_SIZE_TAB, auto_status(variable) -> String, pub(crate) auto_ui(ui, SizingState) -> SizingState}`
  - `gui::sizing::{pub const fn auto_choice(variable) -> &'static str, pub const REQUIREMENT_NOT_POSITIVE: &str = "The requirement must be a positive torque.", pub(crate) fn with_own_value(variable, edited, own) -> DesignInputs, pub(crate) fn variable_number(variable, value) -> String, pub(crate) fn variable_unit(variable) -> &'static str}`
  - Private panel helpers `search_range`, `auto_outcome` and `status_color`.
  - Status wording: `"Not reachable: best X at V"` and `"No valid value of <choice> in range"`.
  - `SIZED_NOTE` becomes `"Set by Auto (Size tab)"`.

---

- [ ] **Step 1: Write the failing `gui/sizing.rs` tests and update three existing string assertions**

Append to `mod tests` in `magcoupling-rs/src/gui/sizing.rs`:

```rust
    #[test]
    fn each_auto_choice_names_the_size_it_sets() {
        assert_eq!(auto_choice(FreeVariable::AxialLength), "magnet length");
        assert_eq!(auto_choice(FreeVariable::MagnetsPerRing), "magnet count");
        assert_eq!(auto_choice(FreeVariable::RingRadius), "ring size");
    }

    #[test]
    fn a_size_tab_edit_keeps_the_inputs_own_value_of_the_free_variable() {
        let own = DesignInputs::default();
        for variable in FreeVariable::ALL {
            // A gap edit made on the design shown, whose free variable sits at its maximum.
            let mut edited = variable.apply(&own, variable.range().max);
            edited.metal.face_gap_mm = 1.5;
            let mut want = own.clone();
            want.metal.face_gap_mm = 1.5;
            assert_eq!(with_own_value(variable, edited, &own), want, "{variable:?}");
        }
        // A blank override comes back blank (above), a set one at its value.
        let set = FreeVariable::AxialLength.apply(&own, 20.0);
        let edited = FreeVariable::AxialLength.apply(&set, 30.0);
        assert_eq!(with_own_value(FreeVariable::AxialLength, edited, &set), set);
    }

    /// One solve of `variable` at `inputs` over its slider range, as the panel asks for it.
    fn solved_now(inputs: &DesignInputs, variable: FreeVariable) -> (SizingRunner, SizingState) {
        let state = SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable,
        };
        let mut runner = SizingRunner::default();
        runner.solve_now(inputs, &state, variable.range());
        (runner, state)
    }

    #[test]
    fn no_valid_value_names_the_auto_choice_and_keeps_the_inputs() {
        // 2 mm magnets: f_end <= 0 at every ring radius of the slider, so no layout counts.
        let inputs = crate::gui::test_support::short_magnets();
        let (runner, state) = solved_now(&inputs, FreeVariable::RingRadius);
        let range = FreeVariable::RingRadius.range();
        assert_eq!(
            runner.outcome(state.variable),
            Some(&Ok(SizingOutcome::NotReachable { best: None }))
        );
        assert_eq!(
            runner.status(&inputs, &state, range),
            "No valid value of ring size in range"
        );
        assert_eq!(runner.shown(&inputs, &state, range), inputs);
    }

    #[test]
    fn a_requirement_that_is_not_positive_is_refused_in_plain_words() {
        for requirement in [0.0, -1.0] {
            let mut inputs = DesignInputs::default();
            inputs.metal.required_min_Nm = requirement;
            let (runner, state) = solved_now(&inputs, FreeVariable::AxialLength);
            let range = FreeVariable::AxialLength.range();
            assert_eq!(
                runner.status(&inputs, &state, range),
                REQUIREMENT_NOT_POSITIVE,
                "{requirement}"
            );
            assert_eq!(runner.shown(&inputs, &state, range), inputs);
        }
    }
```

Change these existing assertions in the same file:
- In `an_unreachable_target_shows_the_best_value`, change the literal `"Not reachable (best 9.937 N·m at 50.80 mm)"` to `"Not reachable: best 9.937 N·m at 50.80 mm"`, and `"Not reachable (best 2.285 N·m at 10)"` to `"Not reachable: best 2.285 N·m at 10"`.
- In `a_refused_solve_shows_why_and_keeps_the_inputs`, the assertion on the refused requirement currently reads `"Sizing refused: the target -1.000 N·m is not a positive torque"`, or T3's rewording of it. Change its expected value to `REQUIREMENT_NOT_POSITIVE`.

In `magcoupling-rs/src/gui/panel.rs`, test `an_unreachable_target_shows_the_best_value_and_its_overshoot`, change `"Not reachable (best 9.937 N\u{b7}m at 50.80 mm)"` to `"Not reachable: best 9.937 N\u{b7}m at 50.80 mm"`.

- [ ] **Step 2: Run them and confirm the failure**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::sizing::tests`
Expected: the build fails with `error[E0425]: cannot find function 'auto_choice'`, `cannot find function 'with_own_value'` and `cannot find value 'REQUIREMENT_NOT_POSITIVE'`.

- [ ] **Step 3: Implement the `gui/sizing.rs` changes**

1. Module doc, line 1. Replace `//! The sizing mode of the Key design group (spec Addendum A1): Magnets → Torque, the forward` with `//! The sizing mode, the Size tab's Auto toggle (spec Addendum A1; guidance 3.5): Magnets → Torque, the forward`.

2. Imports. Change `use crate::engine::meta::Value;` to `use crate::engine::meta::{InputSet, Value};`. Keep a single import of `InputSet` if T3 already added one.

3. After `pub const fn variable_label(..) { .. }`, add:

```rust
/// The Size tab's name for a free variable, after "Auto: smallest" (spec guidance 3.5); each
/// sets the Size-tab slider [`crate::gui::size_controls::variable_slider`] names.
pub const fn auto_choice(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => "magnet length",
        FreeVariable::MagnetsPerRing => "magnet count",
        FreeVariable::RingRadius => "ring size",
    }
}
```

4. After `pub const SOLVING: &str = "Solving...";`, add:

```rust
/// The status line of a solve refused because the requirement (`metal.required_min_Nm`, the
/// target) is not a positive torque.
pub const REQUIREMENT_NOT_POSITIVE: &str = "The requirement must be a positive torque.";
```

5. In `SizingRunner::status`, replace the doc line `/// The status line under the sizing controls.` with `/// The status line of the solve: the Key design group's, and the Size tab's outcome line unless solved.` Then replace the three arms for `NotReachable { best: Some(best) }`, `NotReachable { best: None }` and `Err(SizingError::InvalidTarget(..))`, whatever T3 left in them, with:

```rust
            Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) => format!(
                "{NOT_REACHABLE_PREFIX}: best {} at {}",
                torque_text(best.torque_hot_low_Nm),
                variable_text(state.variable, best.value)
            ),
            Some(Ok(SizingOutcome::NotReachable { best: None })) => {
                format!("No valid value of {} in range", auto_choice(state.variable))
            }
            Some(Err(SizingError::InvalidTarget(_))) => REQUIREMENT_NOT_POSITIVE.to_owned(),
```

Leave the `Solved`, `InvalidInputs` and T2/T3 `InvalidRange` arms unchanged.

6. Replace `fn variable_text` (its doc comment through its closing brace) with:

```rust
/// A free variable's value as a number without its unit: magnets per ring as a whole number.
pub(crate) fn variable_number(variable: FreeVariable, value: f64) -> String {
    let shown = match variable {
        FreeVariable::MagnetsPerRing => Value::Int(value as i64),
        _ => Value::Num(value),
    };
    format_value(&shown)
}

/// The unit of a free variable's input (empty when it has none).
pub(crate) fn variable_unit(variable: FreeVariable) -> &'static str {
    crate::gui::inputs::InputCatalogue::get()
        .entry(variable.path())
        .map_or("", |entry| entry.meta.unit)
}

/// A free variable's value as the status line shows it, with its input's unit.
fn variable_text(variable: FreeVariable, value: f64) -> String {
    with_unit(variable_number(variable, value), variable_unit(variable))
}

/// `edited`, a Size-tab edit of the design shown, with the free variable back at its value in
/// `own`: while Auto is on, the inputs keep their own value of the variable it sets until Auto
/// turns off (decision M41-7).
pub(crate) fn with_own_value(
    variable: FreeVariable,
    edited: DesignInputs,
    own: &DesignInputs,
) -> DesignInputs {
    let path = variable.path();
    let value = own.get(path).expect("every free variable is an input");
    let mut design = edited;
    design
        .set(path, value)
        .expect("an input takes back a value it held");
    design
}
```

- [ ] **Step 4: Run the sizing tests**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::sizing::tests`
Expected: PASS, every test in `gui::sizing::tests`. Dead-code warnings for `with_own_value`, `variable_number` and `variable_unit` are expected until Steps 6 and 10 use them.

- [ ] **Step 5: Write the failing `gui/size_controls.rs` tests**

Append to `magcoupling-rs/src/gui/size_controls.rs`. It is a new module, named so it cannot collide with a `mod tests` T6 may have added:

```rust
#[cfg(test)]
mod auto_tests {
    use super::*;
    use crate::gui::guidance::plain_text_violations;
    use crate::gui::sizing::REQUIREMENT_NOT_POSITIVE;

    #[test]
    fn each_free_variable_locks_its_own_size_slider() {
        assert_eq!(variable_slider(FreeVariable::AxialLength), LENGTH);
        assert_eq!(variable_slider(FreeVariable::RingRadius), DIAMETER);
        assert_eq!(variable_slider(FreeVariable::MagnetsPerRing), MAGNETS);
        // Never the gap or the minimum: they stay the user's.
        for variable in FreeVariable::ALL {
            assert_ne!(variable_slider(variable), GAP);
            assert_ne!(variable_slider(variable), MINIMUM);
        }
    }

    #[test]
    fn change_text_names_the_variable_and_both_values() {
        let from = DesignInputs::default();
        // A blank override starts from the parts' own length (model.active_length_mm).
        assert_eq!(compute_all(&from).model.active_length_mm, 12.7);
        let to = FreeVariable::AxialLength.apply(&from, 13.5);
        assert_eq!(
            change_text(FreeVariable::AxialLength, &from, &to),
            "Magnet length 12.70 -> 13.50 mm"
        );
        let set = FreeVariable::AxialLength.apply(&from, 20.0);
        assert_eq!(
            change_text(FreeVariable::AxialLength, &set, &to),
            "Magnet length 20.00 -> 13.50 mm"
        );
        let to = FreeVariable::RingRadius.apply(&from, 8.5);
        assert_eq!(
            change_text(FreeVariable::RingRadius, &from, &to),
            "Ring radius 10.15 -> 8.500 mm"
        );
        let to = FreeVariable::MagnetsPerRing.apply(&from, 12.0);
        assert_eq!(
            change_text(FreeVariable::MagnetsPerRing, &from, &to),
            "Magnets per ring 10 -> 12"
        );
    }

    #[test]
    fn auto_status_names_the_choice_and_the_size_tab() {
        assert_eq!(
            auto_status(FreeVariable::AxialLength),
            "Auto: magnet length set by the solver (Size tab)"
        );
        for variable in FreeVariable::ALL {
            let status = auto_status(variable);
            assert!(status.starts_with(AUTO_STATUS_START), "{status}");
            assert!(status.contains(auto_choice(variable)), "{status}");
        }
    }

    #[test]
    fn every_auto_text_is_plain_words_with_ascii_arrows() {
        let from = DesignInputs::default();
        let mut texts: Vec<String> = [
            AUTO,
            AUTO_SUFFIX,
            AUTO_HOVER,
            SET_BY_AUTO,
            OPEN_SIZE_TAB,
            REQUIREMENT_NOT_POSITIVE,
        ]
        .map(str::to_owned)
        .to_vec();
        for variable in FreeVariable::ALL {
            texts.push(auto_choice(variable).to_owned());
            texts.push(auto_status(variable));
            let to = variable.apply(&from, variable.range().max);
            texts.push(change_text(variable, &from, &to));
        }
        for text in &texts {
            assert_eq!(plain_text_violations(text), Vec::<&str>::new(), "{text:?}");
            assert!(
                !text.contains('\u{2192}') && !text.contains('\u{2264}'),
                "{text:?}"
            );
        }
    }
}
```

- [ ] **Step 6: Run them and confirm the failure**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::size_controls::auto_tests`
Expected: the build fails with `error[E0425]: cannot find function 'variable_slider'`, `'change_text'` and `'auto_status'`, and with values `AUTO`, `AUTO_SUFFIX`, `AUTO_HOVER`, `SET_BY_AUTO`, `OPEN_SIZE_TAB` and `AUTO_STATUS_START` not found.

- [ ] **Step 7: Implement the `gui/size_controls.rs` changes**

1. Module doc. After line 2 (`//! The engine adapter links housing dimensions; every torque still comes from compute_all.`), add:

```rust
//! Auto (spec guidance 3.5) runs the inverse sizing from here: it locks the slider of the size it
//! sets ([`variable_slider`]) at the value shown and names the change ([`change_text`]).
```

2. Imports:
- Change `use crate::engine::meta::{InputSet, SliderRange, Value};` to `use crate::engine::meta::{InputMeta, InputSet, SliderRange, Value};`.
- Change `use crate::DesignInputs;` to `use crate::{DesignInputs, compute_all};`.
- Add:

```rust
use crate::engine::sizing::FreeVariable;
use crate::gui::format::with_unit;
use crate::gui::sizing::{
    SizingMode, SizingState, auto_choice, variable_label, variable_number, variable_unit,
};
```

3. After `pub const LINKED_HOUSING: &str = ...;`, add:

```rust
/// The Auto row (spec guidance 3.5): "Auto: smallest [size] that meets the requirement".
pub const AUTO: &str = "Auto: smallest";
pub const AUTO_SUFFIX: &str = "that meets the requirement";
/// The hover text of the Auto toggle.
pub const AUTO_HOVER: &str = "On: the solver sets the chosen size to the smallest value whose hot-low torque meets the requirement, skipping layouts whose magnets do not fit, whose keyway breaks through the hub or that the end-effect model cannot predict; its slider locks. Off: the design keeps the value shown.";
/// Under the slider Auto sets, and its hover text.
pub const SET_BY_AUTO: &str = "Set by Auto; turn Auto off to edit";
/// The start of the Key design group's line while Auto is on ([`auto_status`]).
pub const AUTO_STATUS_START: &str = "Auto: ";
/// The Key design group's button to the Size tab, beside [`auto_status`].
pub const OPEN_SIZE_TAB: &str = "Open the Size tab";

/// The Size-tab slider a free variable sets, by its label: Auto locks it.
pub fn variable_slider(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => LENGTH,
        FreeVariable::RingRadius => DIAMETER,
        FreeVariable::MagnetsPerRing => MAGNETS,
    }
}

/// The Auto outcome line of a solved design: the free variable's name, its value in `from` (the
/// inputs) and in `to` (the design shown), e.g. "Magnet length 12.70 -> 13.77 mm".
pub fn change_text(variable: FreeVariable, from: &DesignInputs, to: &DesignInputs) -> String {
    let name = match variable {
        FreeVariable::AxialLength => "Magnet length",
        other => variable_label(other),
    };
    let value = |design: &DesignInputs| variable_number(variable, own_value(variable, design));
    with_unit(
        format!("{name} {} -> {}", value(from), value(to)),
        variable_unit(variable),
    )
}

/// The free variable's value in `design` as the outcome line reads it: the axial length
/// override, or the magnets' active length when it is blank (`model.active_length_mm`).
fn own_value(variable: FreeVariable, design: &DesignInputs) -> f64 {
    match design.get(variable.path()) {
        Some(Value::Num(value)) => value,
        Some(Value::Int(value)) => value as f64,
        Some(Value::None) => compute_all(design).model.active_length_mm,
        other => unreachable!("a free variable is a number or a blank override: {other:?}"),
    }
}

/// The Key design group's line while Auto is on: which size the solver sets, and where.
pub fn auto_status(variable: FreeVariable) -> String {
    format!(
        "{AUTO_STATUS_START}{} set by the solver (Size tab)",
        auto_choice(variable)
    )
}

/// The Auto row: the toggle, the size it sets and the rest of the sentence. Returns the sizing
/// state asked for; the panel applies a mode change through its own switch, which adopts the
/// value shown when Auto turns off (decision M41-7).
pub(crate) fn auto_ui(ui: &mut egui::Ui, state: SizingState) -> SizingState {
    let mut asked = state;
    ui.horizontal_wrapped(|ui| {
        let mut on = state.mode == SizingMode::TorqueToMagnets;
        if ui.checkbox(&mut on, AUTO).on_hover_text(AUTO_HOVER).changed() {
            asked.mode = if on {
                SizingMode::TorqueToMagnets
            } else {
                SizingMode::MagnetsToTorque
            };
        }
        egui::ComboBox::from_id_salt("auto_free_variable")
            .selected_text(auto_choice(asked.variable))
            .show_ui(ui, |ui| {
                for variable in FreeVariable::ALL {
                    ui.selectable_value(&mut asked.variable, variable, auto_choice(variable));
                }
            });
        ui.label(AUTO_SUFFIX);
    });
    asked
}
```

4. `controls_ui`. Change the doc line `/// Draw before computing the panel's results, so the readouts reflect this frame's edits.` to:

```rust
/// Draw before computing the panel's results, so the readouts reflect this frame's edits.
/// `inputs` is the design shown. With `auto`, the slider of the size Auto sets
/// ([`variable_slider`]) is locked at its value there: disabled, saying why.
```

Then add `auto: Option<FreeVariable>,` as the last parameter, after any parameters T6 added. As the first statement of the body, before `let mut output = ControlsOutput {`, add:

```rust
    let is_locked = |label: &str| auto.is_some_and(|variable| variable_slider(variable) == label);
```

5. In the DIAMETER/LENGTH loop, replace

```rust
                let response = ui
                    .push_id(path, |ui| ui.add(slider(&mut value, entry.meta, range)))
                    .inner;
```

with

```rust
                let response = size_slider(ui, path, &mut value, entry.meta, range, is_locked(label));
```

Make the same replacement in the GAP/MAGNETS loop, where the same three lines follow `ui.label(label).on_hover_text(entry.meta.help);`. Leave every line T6 added after the slider where it is.

6. After the closing brace of `controls_ui`, add:

```rust
/// One size slider: a `locked` one (set by Auto) is disabled, so egui reports no change for it,
/// and it says why on hover and in a line under it.
fn size_slider(
    ui: &mut egui::Ui,
    path: &str,
    value: &mut f64,
    meta: &InputMeta,
    range: SliderRange,
    locked: bool,
) -> egui::Response {
    let response = ui
        .push_id(path, |ui| ui.add_enabled(!locked, slider(value, meta, range)))
        .inner
        .on_disabled_hover_text(SET_BY_AUTO);
    if locked {
        ui.weak(SET_BY_AUTO);
    }
    response
}
```

7. Keep `panel.rs` compiling. In `inputs_ui`, append `None` as the last argument of the `crate::gui::size_controls::controls_ui(...)` call, which is still inside the `ui.add_enabled_ui(..)` closure. Step 10 replaces it.

- [ ] **Step 8: Run the size-control tests and the existing Size-tab tests**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::size_controls::auto_tests`
Expected: PASS (4 tests).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib manual_size`
Expected: PASS. No size slider is locked while `auto` is `None`.

- [ ] **Step 9: Write the failing panel tests, the harness helpers and the existing-test edits**

In `magcoupling-rs/src/gui/panel.rs`, `mod tests`, replace `Harness::size` (its doc comment `/// Switches to Torque \u{2192} Magnets and lets the solve run.` through its closing brace) with:

```rust
        /// Turns Auto on (Torque -> Magnets) without changing the view shown, and lets the
        /// solve run. The Auto toggle's own clicks are tested on the Size tab.
        fn size(&mut self) -> egui::FullOutput {
            self.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            self.frame(Vec::new());
            self.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            self.frame(Vec::new());
            self.frame(Vec::new())
        }

        /// The Size tab's slider under `label` as drawn in the last frame (the Size tab must be
        /// shown): `size_controls::controls_ui` pushes its sliders first, in this order.
        fn size_widget(&self, label: &str) -> egui::Response {
            use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
            let index = [DIAMETER, LENGTH, GAP, MAGNETS]
                .iter()
                .position(|l| *l == label)
                .unwrap_or_else(|| panic!("no size slider {label}"));
            self.ctx
                .read_response(self.panel.design_widgets[index])
                .expect("the slider has a response")
        }
```

After `fn assert_locked_row_shows`, add:

```rust
    /// Whether `text` is drawn on the row of the slider whose rail is `rail`, at or right of its
    /// start: the slider's value box.
    fn beside(output: &egui::FullOutput, rail: egui::Rect, text: &str) -> bool {
        text_rects(output, text).iter().any(|r| {
            r.left() >= rail.left() && (rail.top()..=rail.bottom()).contains(&r.center().y)
        })
    }
```

Delete the test `manual_size_adopts_the_inverse_solution_before_editing`. `the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution` below replaces it. Then add these tests:

```rust
    #[test]
    fn auto_locks_the_matching_size_slider_and_says_why() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::size_controls::{
            AUTO, DIAMETER, GAP, LENGTH, MAGNETS, SET_BY_AUTO, change_text,
        };
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        tooltips_at_once(&harness);
        harness.click_text(CentreView::SizeAndTorque.label());
        // Off: every size slider is the user's.
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, SET_BY_AUTO), 0);
        for label in [DIAMETER, LENGTH, GAP, MAGNETS] {
            assert!(harness.size_widget(label).enabled(), "{label}");
        }
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let point = harness.solved();
        // Magnet length, the default choice, locks Overall length alone.
        assert!(!harness.size_widget(LENGTH).enabled(), "locked");
        for label in [DIAMETER, GAP, MAGNETS] {
            assert!(harness.size_widget(label).enabled(), "{label}");
        }
        assert_eq!(count(&output, SET_BY_AUTO), 1, "the line under the locked slider");
        // The locked slider shows the solved design's length, as the slider writes it.
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(&point.inputs)
            .expect("a sized design has size controls")
            .overall_length_mm;
        assert!(beside(&output, rail, &format!("{length:.1} mm")), "{length}");
        // The outcome line names the change, from the inputs' own length to the one shown.
        let line = change_text(FreeVariable::AxialLength, harness.panel.inputs(), &point.inputs);
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, &line),
            Some(Level::Good.color(&harness.ctx.style().visuals))
        );
        // Its hover says why it does not move.
        harness.frame_after(0.2, vec![egui::Event::PointerMoved(rail.center())]);
        assert_eq!(
            count(&harness.frame(Vec::new()), SET_BY_AUTO),
            2,
            "the line and the hover text"
        );
        // Neither arrow keys nor a drag move it.
        let before = (harness.panel.design(), harness.panel.shown_inputs());
        let id = harness.size_widget(LENGTH).id;
        harness.ctx.memory_mut(|m| m.request_focus(id));
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.frame(vec![primary_button(rail.center(), true)]);
        harness.frame(vec![egui::Event::PointerMoved(rail.right_center())]);
        harness.frame(vec![primary_button(rail.right_center(), false)]);
        harness.frame(Vec::new());
        assert_eq!((harness.panel.design(), harness.panel.shown_inputs()), before);
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "M41-7: the inputs keep their own (blank) length"
        );
        // Another size: its slider locks, Overall length is the user's again, nothing is adopted.
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::MagnetsPerRing));
        assert_eq!(harness.panel.sizing().variable, FreeVariable::MagnetsPerRing);
        harness.frame(Vec::new());
        assert!(!harness.size_widget(MAGNETS).enabled());
        assert!(harness.size_widget(LENGTH).enabled());
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn auto_solves_again_after_the_debounce_when_another_size_changes() {
        use crate::engine::sizing::solve_in;
        use crate::gui::size_controls::{AUTO, GAP, change_text};
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 1);
        let steps = harness.panel.history.undo_len();
        // A nudge of the gap on the Size tab edits the inputs, not the solved length.
        let gap = harness.size_widget(GAP).id;
        harness.ctx.memory_mut(|m| m.request_focus(gap));
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        let inputs = harness.panel.inputs().clone();
        assert!(inputs.metal.face_gap_mm > DesignInputs::default().metal.face_gap_mm);
        assert_eq!(
            inputs.coupling.magnets.axial_length_mm, None,
            "the inputs keep their own (blank) length"
        );
        assert_eq!(harness.panel.runner.solves, 1, "waiting for the debounce");
        assert_eq!(count(&output, SOLVING), 1);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        let range = FreeVariable::AxialLength.range();
        let required = inputs.metal.required_min_Nm;
        let Ok(SizingOutcome::Solved(want)) =
            solve_in(&inputs, FreeVariable::AxialLength, required, range)
        else {
            panic!("the requirement is reachable at the nudged gap")
        };
        assert_eq!(harness.panel.shown_inputs(), want.inputs);
        assert_eq!(harness.panel.results(), &compute_all(&want.inputs));
        let line = change_text(FreeVariable::AxialLength, &inputs, &want.inputs);
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        // The nudge is one undo step; undoing it keeps Auto on.
        assert_eq!(harness.panel.history.undo_len(), steps + 1);
        harness.panel.undo();
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
    }

    #[test]
    fn idle_frames_with_auto_on_change_nothing_and_solve_once() {
        use crate::gui::size_controls::AUTO;
        let mut harness = Harness::new();
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let design = harness.panel.design();
        let shown = harness.panel.shown_inputs();
        let steps = harness.panel.history.undo_len();
        for _ in 0..6 {
            harness.frame_after(0.1, Vec::new());
            assert_eq!(harness.panel.design(), design);
            assert_eq!(harness.panel.shown_inputs(), shown);
        }
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the inputs keep their own length"
        );
        assert_eq!(harness.panel.minimum_diameter_mm, None, "no view state moves either");
        assert_eq!(harness.panel.history.undo_len(), steps);
        assert_eq!(harness.panel.runner.solves, 1, "idle frames never solve");
    }

    #[test]
    fn auto_out_of_reach_shows_the_best_value_in_red() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::size_controls::{AUTO, LENGTH};
        use crate::gui::sizing::NOT_REACHABLE_PREFIX;
        let mut harness = Harness::new();
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let Some(Ok(SizingOutcome::NotReachable { best: Some(best) })) = harness
            .panel
            .runner
            .outcome(FreeVariable::AxialLength)
            .cloned()
        else {
            panic!("50 N·m is out of reach of the axial length")
        };
        let line = format!(
            "{NOT_REACHABLE_PREFIX}: best {} at {}",
            with_unit(format_value(&Value::Num(best.torque_hot_low_Nm)), "N\u{b7}m"),
            with_unit(format_value(&Value::Num(best.value)), "mm")
        );
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, &line),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        // The locked slider sits at the best value (decision M41-8); the inputs keep their own.
        assert_eq!(harness.panel.shown_inputs(), best.inputs);
        assert_eq!(harness.panel.inputs().coupling.magnets.axial_length_mm, None);
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(&best.inputs)
            .expect("the best design has size controls")
            .overall_length_mm;
        assert!(beside(&output, rail, &format!("{length:.1} mm")), "{length}");
    }

    #[test]
    fn auto_with_no_valid_ring_size_says_so_and_keeps_the_inputs_value() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::size_controls::{AUTO, DIAMETER};
        let mut harness = Harness::new();
        harness.panel.inputs = short_magnets();
        harness.panel.sizing.variable = FreeVariable::RingRadius;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        // 2 mm magnets: f_end <= 0 at every ring size, so no layout counts.
        let line = "No valid value of ring size in range";
        assert_eq!(count(&output, line), 1, "{:?}", drawn_texts(&output));
        assert_eq!(
            crate::gui::test_support::text_color(&output, line),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(harness.panel.shown_inputs(), short_magnets(), "nothing is sized");
        assert!(!harness.size_widget(DIAMETER).enabled());
        let rail = harness.size_widget(DIAMETER).rect;
        let diameter = dimensions(&short_magnets())
            .expect("2 mm magnets still have size controls")
            .diameter_mm;
        assert!(
            beside(&output, rail, &format!("{diameter:.1} mm")),
            "the slider keeps the inputs' value: {diameter}"
        );
    }

    #[test]
    fn auto_refuses_a_requirement_that_is_not_positive() {
        use crate::engine::size_controls::dimensions;
        use crate::gui::size_controls::{AUTO, LENGTH};
        use crate::gui::sizing::REQUIREMENT_NOT_POSITIVE;
        let mut harness = Harness::new();
        harness.panel.inputs.metal.required_min_Nm = 0.0;
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(&output, REQUIREMENT_NOT_POSITIVE),
            1,
            "{:?}",
            drawn_texts(&output)
        );
        assert_eq!(
            crate::gui::test_support::text_color(&output, REQUIREMENT_NOT_POSITIVE),
            Some(Level::Bad.color(&harness.ctx.style().visuals))
        );
        assert_eq!(&harness.panel.shown_inputs(), harness.panel.inputs());
        let rail = harness.size_widget(LENGTH).rect;
        let length = dimensions(harness.panel.inputs())
            .expect("the default design has size controls")
            .overall_length_mm;
        assert!(beside(&output, rail, &format!("{length:.1} mm")), "{length}");
    }

    #[test]
    fn auto_ring_size_searches_the_size_tab_range_below_the_old_floor() {
        use crate::engine::size_controls::ring_radius_range;
        use crate::engine::sizing::{is_valid, solve, solve_in};
        use crate::gui::size_controls::{AUTO, DIAMETER, LENGTH, change_text};
        use crate::gui::sizing::auto_choice;
        const RING: FreeVariable = FreeVariable::RingRadius;
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.npole = 8;
        harness.click_text(CentreView::SizeAndTorque.label());
        // The requirement: the hot-low torque of an 8.5 mm ring of 8 magnets, a valid layout.
        let at = RING.apply(harness.panel.inputs(), 8.5);
        let at_results = compute_all(&at);
        assert!(is_valid(&at, &at_results), "an 8.5 mm ring of 8 magnets fits");
        harness.panel.inputs.metal.required_min_Nm = at_results.metal.torque_hot_low_Nm;
        // Ring size, picked in the Auto row, then Auto on.
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(RING));
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(
            (harness.panel.sizing().mode, harness.panel.sizing().variable),
            (SizingMode::TorqueToMagnets, RING)
        );
        let inputs = harness.panel.inputs().clone();
        let required = inputs.metal.required_min_Nm;
        // At the default Minimum OD the search starts at the old 9 mm floor, where it meets.
        let floor = ring_radius_range(&inputs, None).expect("8 magnets define a size range");
        let at_floor = harness.solved();
        assert_eq!(
            Ok(SizingOutcome::Solved(at_floor.clone())),
            solve_in(&inputs, RING, required, floor)
        );
        assert!(at_floor.value >= 9.0, "{}", at_floor.value);
        assert_eq!(harness.panel.minimum_diameter_mm, None);
        // A lowered Minimum OD widens the search: one more solve, after the debounce.
        harness.panel.minimum_diameter_mm = Some(37.0);
        harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 1, "waiting for the debounce");
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.runner.solves, 2);
        let range =
            ring_radius_range(&inputs, Some(37.0)).expect("a bounded minimum defines a range");
        assert!(range.min < 8.5, "{range:?}");
        let point = harness.solved();
        assert_eq!(
            Ok(SizingOutcome::Solved(point.clone())),
            solve_in(&inputs, RING, required, range)
        );
        assert!(
            point.value < 9.0 && point.value >= range.min && point.value <= 8.5 + 1e-6,
            "{}",
            point.value
        );
        assert!(
            is_valid(&point.inputs, &compute_all(&point.inputs)),
            "fit and keyway wall still apply"
        );
        assert_eq!(harness.panel.inputs(), &inputs, "the inputs keep their own ring radius");
        assert_eq!(harness.panel.shown_inputs(), point.inputs);
        assert!(!harness.size_widget(DIAMETER).enabled(), "ring size locks Outside diameter");
        assert!(harness.size_widget(LENGTH).enabled());
        assert_eq!(count(&output, &change_text(RING, &inputs, &point.inputs)), 1);
        // Idle frames move neither the design nor the Minimum OD, and solve nothing.
        let (design, steps) = (harness.panel.design(), harness.panel.history.undo_len());
        for _ in 0..3 {
            harness.frame_after(0.1, Vec::new());
            assert_eq!(harness.panel.design(), design);
        }
        assert_eq!(harness.panel.minimum_diameter_mm, Some(37.0));
        assert_eq!(harness.panel.history.undo_len(), steps);
        assert_eq!(harness.panel.runner.solves, 2);
        // An odd magnet count defines no size range: ring size then searches its own slider.
        harness.click_text(CentreView::Geometry.label());
        harness.panel.inputs.coupling.npole = 7;
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let odd = harness.panel.inputs().clone();
        assert_eq!(ring_radius_range(&odd, Some(37.0)), None);
        assert_eq!(
            harness.panel.runner.outcome(RING),
            Some(&solve(&odd, RING, odd.metal.required_min_Nm))
        );
    }

    #[test]
    fn an_inverse_mode_design_file_opens_with_auto_on() {
        use crate::gui::size_controls::{LENGTH, SET_BY_AUTO, change_text};
        let design = Design {
            inputs: DesignInputs::default(),
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::AxialLength,
            },
        };
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness.panel.load_design_file(&design_to_json(&design)).unwrap();
        // Opening the Size tab switches no mode.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.design(), design);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(
            harness.panel.inputs(),
            &DesignInputs::default(),
            "the inputs keep their own length"
        );
        assert!(!harness.size_widget(LENGTH).enabled(), "locked");
        assert_eq!(count(&output, SET_BY_AUTO), 1);
        let shown = harness.panel.shown_inputs();
        let line = change_text(FreeVariable::AxialLength, harness.panel.inputs(), &shown);
        assert_eq!(count(&output, &line), 1, "{:?}", drawn_texts(&output));
        // A share link of the same design opens the same way, as the session's start.
        let mut linked = Harness::new();
        linked
            .panel
            .open_share_payload(&encode_share_payload(&design))
            .unwrap();
        linked.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(linked.panel.design(), design);
    }

    #[test]
    fn the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab() {
        use crate::gui::size_controls::{AUTO_STATUS_START, OPEN_SIZE_TAB, auto_status};
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        let output = harness.frame(Vec::new());
        // The solver's controls live on the Size tab: no mode switch, picker or target here.
        let gone = [
            SizingMode::MagnetsToTorque.label(),
            SizingMode::TorqueToMagnets.label(),
            FREE_VARIABLE_LABEL,
            TARGET_LABEL,
        ];
        for text in gone {
            assert_eq!(count(&output, text), 0, "{text}");
        }
        assert!(
            !drawn_texts(&output)
                .iter()
                .any(|t| t.starts_with(AUTO_STATUS_START)),
            "Auto is off"
        );
        // Auto on: one line naming the size it sets, and the button.
        let output = harness.size();
        assert_eq!(count(&output, &auto_status(FreeVariable::AxialLength)), 1);
        assert_eq!(count(&output, OPEN_SIZE_TAB), 1);
        for text in gone {
            assert_eq!(count(&output, text), 0, "{text}");
        }
        let design = harness.panel.design();
        harness.click_text(OPEN_SIZE_TAB);
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(harness.panel.design(), design, "the button edits nothing");
    }

    #[test]
    fn the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution() {
        use crate::gui::size_controls::{AUTO, LENGTH};
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness.size();
        let solved = harness.panel.shown_inputs();
        // Opening the Size tab switches nothing.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.shown_inputs(), solved);
        assert!(!harness.size_widget(LENGTH).enabled());
        // Auto off: the inputs adopt the length shown (decision M41-7); the slider is the user's.
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.inputs(), &solved);
        assert_eq!(harness.panel.results(), &compute_all(&solved));
        harness.frame(Vec::new());
        assert!(harness.size_widget(LENGTH).enabled());
    }

    #[test]
    fn the_auto_texts_have_glyphs_and_wrap_on_the_inputs_side() {
        use crate::gui::size_controls::{
            AUTO, AUTO_HOVER, AUTO_SUFFIX, LENGTH, OPEN_SIZE_TAB, SET_BY_AUTO, auto_status,
            change_text,
        };
        use crate::gui::sizing::{NOT_REACHABLE_PREFIX, REQUIREMENT_NOT_POSITIVE, auto_choice};
        // The M4-1 review's ~930 px window; the inputs side keeps its 320 points.
        let mut harness = Harness::on_screen(egui::vec2(930.0, 1024.0));
        tooltips_at_once(&harness);
        // The longest outcome line: a requirement out of reach.
        harness.panel.inputs.metal.required_min_Nm = 50.0;
        harness.frame(Vec::new());
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(AUTO);
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        let output = harness.frame(Vec::new());
        let texts = drawn_texts(&output);
        let outcome = texts
            .iter()
            .find(|t| t.starts_with(NOT_REACHABLE_PREFIX))
            .cloned()
            .expect("the outcome line");
        // The centre region starts at the Size tab's tab (the inputs side's heading of the same
        // text paints first, further left).
        let centre_left = text_rects(&output, CentreView::SizeAndTorque.label())
            .iter()
            .map(|r| r.left())
            .fold(f32::MIN, f32::max);
        let row = [
            AUTO,
            AUTO_SUFFIX,
            SET_BY_AUTO,
            auto_choice(FreeVariable::AxialLength),
            outcome.as_str(),
        ];
        for text in row {
            let rects = text_rects(&output, text);
            assert!(!rects.is_empty(), "{text:?} is drawn");
            for rect in rects {
                assert!(
                    rect.right() <= centre_left,
                    "{text:?} at {rect:?} runs past the inputs side ({centre_left})"
                );
            }
        }
        let mut all = texts;
        // The locked slider's hover text.
        let rail = harness.size_widget(LENGTH).rect;
        harness.frame_after(0.2, vec![egui::Event::PointerMoved(rail.center())]);
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Key design group's line, on another tab.
        harness.click_text(CentreView::Geometry.label());
        all.extend(drawn_texts(&harness.frame(Vec::new())));
        assert!(all.contains(&auto_status(FreeVariable::AxialLength)));
        // Every Auto text, each choice's.
        all.extend([AUTO_HOVER, OPEN_SIZE_TAB, REQUIREMENT_NOT_POSITIVE].map(str::to_owned));
        let from = DesignInputs::default();
        for variable in FreeVariable::ALL {
            all.push(auto_status(variable));
            let to = variable.apply(&from, variable.range().max);
            all.push(change_text(variable, &from, &to));
        }
        for text in &all {
            crate::gui::test_support::assert_glyphs(&harness.ctx, text, "the Auto texts");
        }
    }
```

Apply the edits to existing tests listed under "Existing tests this change breaks" (items 3 to 6).

- [ ] **Step 10: Run the panel tests and confirm the failures**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel::tests`
Expected:
- These fail with `no text "Auto: smallest"`, because the panel does not call `auto_ui` yet: `auto_locks_the_matching_size_slider_and_says_why`, `auto_solves_again_after_the_debounce_when_another_size_changes`, `idle_frames_with_auto_on_change_nothing_and_solve_once`, `auto_out_of_reach_shows_the_best_value_in_red`, `auto_with_no_valid_ring_size_says_so_and_keeps_the_inputs_value`, `auto_refuses_a_requirement_that_is_not_positive`, `auto_ring_size_searches_the_size_tab_range_below_the_old_floor`, `the_auto_texts_have_glyphs_and_wrap_on_the_inputs_side` and `leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step`.
- `the_free_variable_picker_switches_the_variable` fails with `no text "magnet length"`.
- `the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution` and `an_inverse_mode_design_file_opens_with_auto_on` fail on `assert_eq!` (`MagnetsToTorque` vs `TorqueToMagnets`): the tab click still forces forward mode.
- `the_key_design_group_shows_auto_s_status_and_a_button_to_the_size_tab` fails with `left: 1, right: 0` for `Magnets -> Torque`.
- Everything else passes.

- [ ] **Step 11: Implement the panel changes**

1. Module doc. Replace lines 22-25 (`//! Sizing (spec Addendum A1): the mode switch tops the Key design group. ...` through `//! row shows that value, locked.`) with:

```rust
//! Sizing (spec Addendum A1; guidance 3.5): the Size tab's Auto toggle is Torque → Magnets, the
//! solver's one home. In it the panel shows the design with the free variable at the solved value
//! ([`crate::gui::sizing::SizingRunner`], debounced, never per frame); the Size-tab slider and the
//! input row of the free variable show that value, locked, and the Key design group says so.
```

2. Imports:
- `use crate::engine::meta::{InputSet, ResultSet, Value};` becomes `use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};`.
- `use crate::engine::sizing::FreeVariable;` becomes `use crate::engine::sizing::{FreeVariable, SizingOutcome};`.
- Add `use crate::engine::size_controls::ring_radius_range;`.
- In `use crate::gui::sizing::{...}`, add `with_own_value` and remove `variable_label`.
- Then run `rg -n "REQUIREMENT_INPUT|\bslider\(" magcoupling-rs/src/gui/panel.rs`. Remove `REQUIREMENT_INPUT` from that import and `slider` from `use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row, slider};` if no use remains after edit 6 below. T7 removes their only uses, the Key design target slider; T6 may have added others.

3. `SIZED_NOTE`. Replace

```rust
/// The note under the free variable's row in Torque → Magnets.
pub const SIZED_NOTE: &str = "Set by Torque -> Magnets";
```

with

```rust
/// The note under the free variable's row while Auto (Torque → Magnets) sets it.
pub const SIZED_NOTE: &str = "Set by Auto (Size tab)";
```

4. After `pub fn shown_inputs(&self) -> DesignInputs { .. }`, add:

```rust
    /// The range Auto's solve searches (spec guidance 3.5): ring size over the ring radii of the
    /// Size tab's diameter range, its Minimum OD included
    /// ([`crate::engine::size_controls::ring_radius_range`]), or over its slider when the size
    /// controls define no range (an odd magnet count); every other free variable over its slider.
    /// Read from the inputs, never the design shown, so a solve cannot move its own range.
    fn search_range(&self) -> SliderRange {
        match self.sizing.variable {
            FreeVariable::RingRadius => {
                ring_radius_range(&self.inputs, self.minimum_diameter_mm)
                    .unwrap_or_else(|| FreeVariable::RingRadius.range())
            }
            variable => variable.range(),
        }
    }

    /// The Size tab's Auto outcome line and its colour: what the solve changed, from the
    /// inputs' own value to the value `shown` (good); else the runner's status line ("Not
    /// reachable: best ...", "No valid value of ... in range" or a refusal: bad; "Solving...":
    /// weak).
    fn auto_outcome(
        &self,
        shown: &DesignInputs,
        visuals: &egui::Visuals,
    ) -> (String, egui::Color32) {
        let range = self.search_range();
        let solved = self.runner.is_current(&self.inputs, &self.sizing, range)
            && matches!(
                self.runner.outcome(self.sizing.variable),
                Some(Ok(SizingOutcome::Solved(_)))
            );
        if solved {
            let line =
                crate::gui::size_controls::change_text(self.sizing.variable, &self.inputs, shown);
            (line, Level::Good.color(visuals))
        } else {
            let status = self.runner.status(&self.inputs, &self.sizing, range);
            let color = status_color(&status, visuals);
            (status, color)
        }
    }
```

5. In every non-test function of `panel.rs` (`shown_inputs`, `run_sizing`, `log_sizing`, `set_sizing_mode`, `export_spreadsheet`, and any T3 helper), replace each `self.sizing.variable.range()` T3 wrote with `self.search_range()`. Check: `rg -n "variable\.range\(\)" magcoupling-rs/src/gui/panel.rs` must then list only lines inside `mod tests` plus the two inside `search_range`. Also change the doc of `set_sizing_mode` from `/// Leaving inverse sizing adopts its result before a manual size edit.` to `/// Turning Auto off (leaving Torque → Magnets) adopts the value shown (decision M41-7), solving first a change still waiting for its debounce.`

6. Replace the whole `fn sizing_ui` with the following. This covers its doc comment (`/// The sizing controls at the top of the Key design group: ...`) through its closing brace, whatever T3 left in the body:

```rust
    /// Auto's line at the top of the Key design group (spec guidance 3.5: the solver's home is
    /// the Size tab): which size Auto sets, a button to the Size tab and the solve's status;
    /// then, when the Key design group does not list it, the free variable's row locked at the
    /// value shown (the ring radius). Nothing while Auto is off.
    fn sizing_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        if self.sizing.mode != SizingMode::TorqueToMagnets {
            return;
        }
        ui.horizontal_wrapped(|ui| {
            ui.label(crate::gui::size_controls::auto_status(self.sizing.variable));
            if ui.button(crate::gui::size_controls::OPEN_SIZE_TAB).clicked() {
                self.centre = CentreView::SizeAndTorque;
                ui.ctx().request_repaint();
            }
        });
        let status = self
            .runner
            .status(&self.inputs, &self.sizing, self.search_range());
        let color = status_color(&status, ui.visuals());
        ui.colored_label(color, status);
        // The free variable's row, locked at the value shown, when the Key design group does
        // not list it (the ring radius is in the Coupling group, closed by default).
        let path = self.sizing.variable.path();
        if !KEY_DESIGN.contains(&path) {
            let entry = InputCatalogue::get()
                .entry(path)
                .expect("every free variable is an input");
            let widget = self.input_row_ui(ui, entry, readouts);
            self.design_widgets.push(widget);
        }
        ui.separator();
    }
```

7. In `centre_ui`, remove the forced forward switch. Replace

```rust
                for view in CentreView::ALL {
                    let response = ui.selectable_value(&mut self.centre, view, view.label());
                    if response.changed() && view == CentreView::SizeAndTorque {
                        self.set_sizing_mode(SizingMode::MagnetsToTorque);
                        self.results = compute_all(&self.inputs);
                        ui.ctx().request_repaint();
                    }
                }
```

with

```rust
                for view in CentreView::ALL {
                    ui.selectable_value(&mut self.centre, view, view.label());
                }
```

8. `inputs_ui`, Size-tab block. Replace

```rust
            ui.heading(crate::gui::size_controls::HEADING);
            if self.sizing.mode != SizingMode::MagnetsToTorque
                && ui.button("Edit size manually").clicked()
            {
                self.set_sizing_mode(SizingMode::MagnetsToTorque);
            }
            let shown = self.shown_inputs();
```

with

```rust
            ui.heading(crate::gui::size_controls::HEADING);
            // Auto (spec guidance 3.5) is the solver's one home. Turning it off adopts the value
            // shown (decision M41-7) through the mode switch.
            let asked = crate::gui::size_controls::auto_ui(ui, self.sizing);
            self.sizing.variable = asked.variable;
            self.set_sizing_mode(asked.mode);
            let auto = (self.sizing.mode == SizingMode::TorqueToMagnets)
                .then_some(self.sizing.variable);
            let shown = self.shown_inputs();
            if auto.is_some() {
                let (line, color) = self.auto_outcome(&shown, ui.visuals());
                ui.add(egui::Label::new(egui::RichText::new(line).color(color)).wrap());
            }
```

Then unwrap the controls call. `let output = ui.add_enabled_ui(self.sizing.mode == SizingMode::MagnetsToTorque, |ui| { <call> }).inner;` becomes `let output = <call>;`, where `<call>` is the `crate::gui::size_controls::controls_ui(...)` call with all its arguments as they are, except the last argument `None` from Step 7, which becomes `auto`. Then replace

```rust
            if output.inputs != shown {
                self.inputs = output.inputs;
                self.last_error = output.error;
```

with

```rust
            if output.inputs != shown {
                // The controls edit the design shown: the inputs keep their own value of the size
                // Auto sets until Auto turns off (decision M41-7).
                self.inputs = match auto {
                    Some(variable) => with_own_value(variable, output.inputs, &self.inputs),
                    None => output.inputs,
                };
                self.last_error = output.error;
```

9. After `pub(crate) fn assumptions_banner(..) { .. }`, add:

```rust
/// The colour of a sizing status line ([`SizingRunner::status`]): good when solved, weak while
/// solving, bad otherwise (not reachable, no valid value, refused).
fn status_color(status: &str, visuals: &egui::Visuals) -> egui::Color32 {
    if status.starts_with(SOLVED_PREFIX) {
        Level::Good.color(visuals)
    } else if status == SOLVING {
        visuals.weak_text_color()
    } else {
        Level::Bad.color(visuals)
    }
}
```

- [ ] **Step 12: Run the panel tests**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel::tests`
Expected: PASS, all of `gui::panel::tests`, including the 11 new tests and the edited ones.

- [ ] **Step 13: Run the wider suite, clippy and fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: PASS. This covers lib tests (`gui::sizing`, `gui::size_controls`, `gui::session`, `gui::spreadsheet`, `gui::panel`, `app`) and the integration tests (`tests/sizing.rs`, `tests/size_controls.rs`, `tests/python_schema.rs`, ...), 0 failed.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
Expected: no warnings. An unused-import error naming `slider`, `variable_label` or `REQUIREMENT_INPUT` in `panel.rs` means Step 11.2 missed it: remove it and rerun.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml` and then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`
Expected: no diff.

- [ ] **Step 14: Update `magcoupling-rs/README.md`**

1. In "The panel", replace

```
exports remain available. Opening this tab switches to forward calculation, keeping any
inverse-sized design; direct edits use the existing undo/redo history and file/share format.
```

with

```
exports remain available. **Auto: smallest [magnet length | ring size | magnet count] that
meets the requirement** runs the inverse sizing on this tab: it locks the matching slider
(Overall length, Outside diameter, Magnets per ring) at the solved value, or at the best value
under a red "Not reachable: best Y at X", and solves again 0.25 s after any other edit. Ring
size searches the ring radii of this tab's diameter range, Minimum OD included, and moves the
ring radius alone (the cap and its thread stay). Turning Auto off keeps the value shown.
Opening the tab changes no mode; direct edits use the existing undo/redo history and
file/share format.
```

2. In the **Sizing** bullet, make these replacements:
- `the mode switch tops the Key design group. In\n  Torque -> Magnets the free variable` becomes `Auto on the Size tab is Torque -> Magnets (the Key design group shows\n  only a status line, the solve's status and a button to the tab). In it the free variable`.
- `"Not reachable (best Y at X)" (M41-8)` becomes `"Not reachable: best Y at X" (M41-8; "No valid value of <size> in range" when no value counts)`.
- `Leaving the\n  mode keeps the value` becomes `Turning Auto\n  off keeps the value`.

The test-table rows at lines 515 and 521, and `docs/ai/02-system.yaml` lines 103 and 439, are T10's work. T10 must not rewrite the two paragraphs above.

- [ ] **Step 15: Commit**

```bash
git add magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/README.md
git commit -F - <<'EOF'
feat(magcoupler): Auto solver on the Size tab

The Size tab's "Auto: smallest [size] that meets the requirement" toggle is
Torque -> Magnets: it locks the matching slider at the solved (or best)
value, says why under it and on hover, names the change ("Magnet length
12.70 -> 13.77 mm") or the miss ("Not reachable: best ...", "No valid value
of ring size in range"), and solves again after the debounce. Ring size
searches the Size tab's diameter range, Minimum OD included. Opening the
tab no longer forces forward mode; the Key design group shows a status line
and a button to the tab instead of the mode switch.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```

---

**Spec section 5 edge cases covered here:**

| Edge case | Test(s) |
|---|---|
| Not reachable (red line, slider at the best value) | `auto_out_of_reach_shows_the_best_value_in_red` |
| No value valid ("No valid value of ring size in range", slider keeps the inputs' value) | `auto_with_no_valid_ring_size_says_so_and_keeps_the_inputs_value`, `no_valid_value_names_the_auto_choice_and_keeps_the_inputs` |
| Locked slider dragged or nudged, hover text | `auto_locks_the_matching_size_slider_and_says_why` |
| Requirement <= 0 | `auto_refuses_a_requirement_that_is_not_positive`, `a_requirement_that_is_not_positive_is_refused_in_plain_words` |
| Inverse-mode file and link open with Auto on | `an_inverse_mode_design_file_opens_with_auto_on` |
| Narrow window and glyphs | `the_auto_texts_have_glyphs_and_wrap_on_the_inputs_side` |
| Idle frames | `idle_frames_with_auto_on_change_nothing_and_solve_once`, plus the idle loop in the ring test |

**Notes for the reviewer and the orchestrator:**

1. **Ring size moves the radius alone.** `FreeVariable::RingRadius::apply` changes only `coupling.inner_back_apothem_mm`; the cap OD and its thread do not follow as they do for a Size-tab diameter edit. I measured this on the default 8-pole design: every radius from 7.47 to 10.27 mm keeps the Outside diameter at the cap's 42.8 mm. So the locked Outside diameter does not shrink with a "smaller ring", and the cap overhang grows. T7 follows the contract and says this in the README. Changing it is an owner decision outside T7: either make ring-size Auto move the cap and thread with the cup (T1/T2 engine semantics), or rename the choice "ring radius".
2. **One status wording.** `SizingRunner::status` now holds the spec wording for not reachable, no valid value and requirement <= 0. The Key design line, the Size-tab outcome line, the spreadsheet's outcome cell and the web log all read it. The smoke log's `Solved at` prefix is unchanged. If T3 already reworded the `InvalidTarget` arm, Step 3 replaces that text.
3. **The Key design group keeps the solve's status line** under the Auto line. A user on another tab can still see why the design shows a different size, and the existing debounce and solved-design tests keep their Geometry-tab assertions. The mode switch, the free-variable picker and the target slider are gone. `TARGET_LABEL` and `FREE_VARIABLE_LABEL` stay because `gui/spreadsheet.rs` uses them.
4. **`SIZED_NOTE` changes text** so the locked input row no longer names the removed "Torque -> Magnets" switch. The spreadsheet's note column follows; its Summary row still writes the mode label.
5. **Cost of `search_range`.** In ring-size Auto it calls `ring_radius_range`, which validates, builds the input rows and runs one `compute_all`, about 6 times per frame. That is small next to the existing per-frame `compute_all`, so I added no cache.
6. **T9 handoff.** `Harness::size()` no longer changes the view. After T9 makes Size the default view, tests that read Key design rows after `size()` must click `CentreView::Geometry` first. Every T7 test sets its own tab and assumes no default.

**Existing tests this change breaks, with the exact edit:**

1. `gui/sizing.rs::an_unreachable_target_shows_the_best_value`: change `"Not reachable (best 9.937 N·m at 50.80 mm)"` to `"Not reachable: best 9.937 N·m at 50.80 mm"`, and `"Not reachable (best 2.285 N·m at 10)"` to `"Not reachable: best 2.285 N·m at 10"` (Step 1).
2. `gui/sizing.rs::a_refused_solve_shows_why_and_keeps_the_inputs`: the expected text of the refused requirement becomes `REQUIREMENT_NOT_POSITIVE` (Step 1).
3. `gui/panel.rs::an_unreachable_target_shows_the_best_value_and_its_overshoot`: change `"Not reachable (best 9.937 N\u{b7}m at 50.80 mm)"` to `"Not reachable: best 9.937 N\u{b7}m at 50.80 mm"` (Step 1).
4. `gui/panel.rs::Harness::size`: replaced as in Step 9. It now calls `set_sizing_mode(TorqueToMagnets)` plus a frame before the debounce frame, because the Key design mode switch is gone.
5. `gui/panel.rs::manual_size_adopts_the_inverse_solution_before_editing`: delete it. It asserts the removed forced switch. `the_size_tab_keeps_auto_on_and_auto_off_adopts_the_solution` replaces it.
6. `gui/panel.rs::the_solve_waits_for_the_debounce_and_never_runs_per_frame`: replace `harness.click_text(SizingMode::TorqueToMagnets.label());` with `harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);`.
7. `gui/panel.rs::leaving_torque_to_magnets_while_solving_writes_the_current_solution`: replace `harness.click_text(SizingMode::MagnetsToTorque.label());`, the line just above `assert_eq!(harness.panel.runner.solves, 2, "solved on leaving");`, with `harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);`.
8. `gui/panel.rs::leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step`: replace the body with the following. It keeps the T3 version's assertions and turns Auto off in the UI.

```rust
        use crate::gui::size_controls::AUTO;
        let mut harness = Harness::new();
        harness.size();
        let point = harness.solved();
        let steps = harness.panel.history.undo_len();
        // Auto off on the Size tab; opening the tab is no edit.
        harness.click_text(CentreView::SizeAndTorque.label());
        assert_eq!(harness.panel.history.undo_len(), steps);
        harness.click_text(AUTO);
        assert_eq!(harness.panel.sizing().mode, SizingMode::MagnetsToTorque);
        assert_eq!(harness.panel.inputs(), &point.inputs, "decision M41-7");
        assert_eq!(harness.panel.results(), &compute_all(&point.inputs));
        assert_eq!(harness.panel.history.undo_len(), steps + 1);
        harness.panel.undo();
        assert_eq!(harness.panel.sizing().mode, SizingMode::TorqueToMagnets);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
```

9. `gui/panel.rs::the_free_variable_picker_switches_the_variable`: replace its opening lines, from `let mut harness = Harness::new();` through `harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());`, with the block below. The rest of the test is unchanged.

```rust
        use crate::gui::sizing::auto_choice;
        let mut harness = Harness::new();
        harness.click_text(CentreView::Geometry.label());
        harness.size();
        // The picker is the Size tab's Auto row; the Key design group then shows the result.
        harness.click_text(CentreView::SizeAndTorque.label());
        harness.click_text(auto_choice(FreeVariable::AxialLength));
        harness.click_text(auto_choice(FreeVariable::RingRadius));
        assert_eq!(harness.panel.sizing().variable, FreeVariable::RingRadius);
        harness.click_text(CentreView::Geometry.label());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
```

10. Any T3 panel test that drives the Key design group's target slider (`TARGET_LABEL`): T7 removes that slider. Drive T6's Required slider on the Size tab (`size_controls::REQUIRED`) through T6's own test helper for it, and keep the assertions: `metal.required_min_Nm` moves and one undo restores it.
11. Any T6 panel test that turned Torque -> Magnets on and then opened the Size tab expecting forward mode: call `harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque)` before opening the tab. The tab no longer switches the mode.

---

### Task 8: Dashboard check guides (status line, Magnets fit row, Problem/Why/Fix blocks, Why links)

> **Amendments:** apply "T8 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


Spec sections 3.1 (status line) and 3.4. All code below was checked on a scratch copy of `022e254` with a stand-in for T4's `gui/guidance.rs` built to the interface contract. The four new panel tests, the eight new dashboard tests and the two event tests all pass, the full `--lib` suite passes (534 tests) once the 7 edits in Step 12 are made, and clippy with `-D warnings` and `cargo fmt --check` are clean.

**Gaps in the contract, and how this task fills them:**
1. **`MagcouplingPanel::focus_input` does not exist.** The way to highlight an input's row today is `Explorer::focus_input` (`explorer.rs:292`), which leaf terms use. Fix links and assumption links reach it through a new `ReadoutEvents::focus` event, handled in `Explorer::end_frame`. This is the same route a warning's Why link takes to `Explorer::open_note`. No panel wrapper is added, because it would be dead code.
2. **"Open the assumption" uses `Explorer::focus_input` on the assumption's first input path.**
   - Outside the Size and torque view, `panel.rs` `inputs_ui` (~951-960) switches to `InputsView::Assumptions` for an assumption path. The row is framed and scrolled to, with its rationale under it. This is the route `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view` already tests.
   - In the Size and torque view, "Detailed inputs" opens to the input's group.
   - The link's hover text is the rationale, so the rationale can be read in every view.
3. **The Magnets fit row is not added to `dashboard::DASHBOARD`.** That table only takes engine result paths:
   - `dashboard_lines` panics on a path with no `result_info`.
   - `the_dashboard_starts_with_the_headline_in_order` pins `paths[15..]`.
   - `out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter` zips the lines with `DASHBOARD`.

   `fit_ui` draws the fit row after the last table row (the end-effect flag).
4. **"Every row gets a Why: link" is read as every guided check row.** That is the 7 check rows (each check's own row, where the path equals the check) plus the fit row: 8 rows. Rows whose badge only borrows a check's verdict (for example `metal.torque_hot_low_Nm`) get no guide, so a block never appears twice.
5. **Layout of the block.** Under a red or amber row:
   - `Problem:` followed by T4's problem text, in the level's colour.
   - `Why:` followed by `guide.why`, with the learn link under the sentence in the same column.
   - `Fix:` followed by one link per fix.

   A green row has only `Why:` and the link. A Note or Assumption link is shown as a link; `Learn::None` shows weak "no note yet" with the reason as hover text. A `Learn::Note` whose note has not passed review shows "no note yet" too, because every note the GUI shows goes through `reviewed_note`.

**Files:**
- Modify `magcoupling-rs/src/gui/readouts.rs`: `ReadoutEvents` (HEAD 52-61); add `Readouts::focus_input` after `open_note` (HEAD 136-139); add a test at the end of `mod tests` (HEAD 253).
- Modify `magcoupling-rs/src/gui/explorer.rs`: `Explorer::end_frame` (HEAD 346-361); add a test at the end of `mod tests` (HEAD 1179).
- Modify `magcoupling-rs/src/gui/dashboard.rs`: module doc (end at HEAD 25), imports (HEAD 30-39), new constants after `WHY` (HEAD 77-78), `dashboard_ui` (HEAD 495-531), new private functions after it, test imports (HEAD 576-581), new tests at the end of `mod tests` (HEAD 1279).
- Modify `magcoupling-rs/src/gui/panel.rs`: the dashboard call in `MagcouplingPanel::ui` (HEAD 490-497), 7 existing tests (Step 12), 4 new tests at the end of `mod tests`. T1-T7 move these lines, so find each edit by the quoted context.
- Tests live in the `mod tests` of each of these four files.

**Interfaces:**
- **Consumes from T4** (`crate::gui::guidance`), exactly as in the contract:
  - `Direction`. It must be `Copy`, because `for &(path, direction) in guide.fixes` copies it; T4 should derive `Clone, Copy, Debug, PartialEq, Eq`.
  - `Learn::{Note, Assumption, None}`.
  - `CheckGuide { check, label, problem, why, fixes, learn }`, all fields pub, `problem: fn(&DesignInputs, &DesignResults) -> Option<String>`.
  - `FIT_CHECK`, `CHECK_GUIDES: [CheckGuide; 8]`, `guide(check)`, `level(guide, inputs, results) -> Level`, `fix_text(path, direction) -> String`.
  - `DesignStatus { bad, caution, total, is_default }`, `design_status`, `design_status_text`, `plain_text_violations`.
  - `level` must equal the dashboard badge, falling back to `Good` when the row has no badge (`check_level(results, check).unwrap_or(Level::Good)` for the 7). The test `a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it` enforces this; if it fails, fix T4's `level`, not this task.
- **Produces:**
  - `gui::readouts::ReadoutEvents::focus: Option<&'static str>` and `pub fn Readouts::focus_input(&mut self, path: &'static str)`.
  - `Explorer::end_frame` handles `focus` by calling `Explorer::focus_input`.
  - In `gui::dashboard`: `pub const PROBLEM = "Problem:"`, `FIX = "Fix:"`, `WHY_LINK = "Why:"`, `NO_NOTE_YET = "no note yet"`, `NOTE_NOT_REVIEWED`, `FITS = "Fit around the ring"`, `DOES_NOT_FIT = "Do not fit around the ring"`.
  - `pub fn dashboard_ui(ui, inputs: &DesignInputs, results: &DesignResults, readouts: &mut Readouts)`, with a new `inputs` argument (the design shown).
- **Physics review (G-3):** `FITS`, `DOES_NOT_FIT` and `NOTE_NOT_REVIEWED` are the only new sentences in this task. All other text comes from T4.

**Steps:**

- [ ] **Step 1: Write the failing event tests.** Append to `mod tests` in `magcoupling-rs/src/gui/readouts.rs`, before its closing `}`:

```rust

    #[test]
    fn a_check_guide_link_asks_for_its_input_s_row() {
        let mut readouts = Readouts::default();
        readouts.focus_input("metal.cup_wall_corner_mm");
        assert_eq!(readouts.events().focus, Some("metal.cup_wall_corner_mm"));
        assert_eq!(readouts.finish().focus, Some("metal.cup_wall_corner_mm"));
    }
```

  Append to `mod tests` in `magcoupling-rs/src/gui/explorer.rs`, before its closing `}`:

```rust

    #[test]
    fn a_check_guide_link_highlights_its_input_s_row_and_leaves_the_panel_as_it_was() {
        let ctx = egui::Context::default();
        let mut explorer = Explorer::default();
        explorer.open_path("model.pullout_Nm");
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                focus: Some("metal.cup_wall_corner_mm"),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(
            explorer.focus(),
            Some(&Focus {
                path: "metal.cup_wall_corner_mm".to_owned(),
                scroll: true,
            })
        );
        assert_eq!(
            explorer.trail(),
            ["model.pullout_Nm"],
            "the equation stays open"
        );
        assert!(explorer.open);
        // With the panel shut, it stays shut.
        let mut explorer = Explorer::default();
        explorer.end_frame(
            &ctx,
            ReadoutEvents {
                focus: Some("metal.variation"),
                ..ReadoutEvents::default()
            },
        );
        assert_eq!(
            explorer.focus().map(|f| f.path.as_str()),
            Some("metal.variation")
        );
        assert!(!explorer.open);
    }
```

- [ ] **Step 2: Run them and confirm they fail.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib a_check_guide_link`
  Expected: the lib test target does not compile, with 5 errors:
  - `error[E0599]: no method named focus_input found for struct readouts::Readouts`
  - `error[E0609]: no field focus on type &readouts::ReadoutEvents`
  - `error[E0609]: no field focus on type readouts::ReadoutEvents`
  - `error[E0560]: struct readouts::ReadoutEvents has no field named focus` (twice)

- [ ] **Step 3: Add the event.** In `readouts.rs`, replace

```rust
    /// The teaching note a warning's link asked for, by id.
    pub note: Option<&'static str>,
}
```

  with

```rust
    /// The teaching note a warning's or a check's Why link asked for, by id.
    pub note: Option<&'static str>,
    /// The input whose row a check guide's link asked to show (a fix, an assumption's Why
    /// link), by path.
    pub focus: Option<&'static str>,
}
```

  and insert directly after `pub fn open_note(&mut self, id: &'static str) { ... }`:

```rust

    /// Asks for the row of the input at `path` (a check guide's fix, an assumption's Why
    /// link).
    pub fn focus_input(&mut self, path: &'static str) {
        self.events.focus = Some(path);
    }
```

  In `explorer.rs`, replace the doc and the end of `end_frame`:

```rust
    /// Takes in what the user did with the readouts this frame: the value hovered marks its
    /// terms from the next frame (asked for at once); a value clicked opens here; a check
    /// guide's link highlights its input's row, as a leaf term does.
    pub fn end_frame(&mut self, ctx: &egui::Context, events: ReadoutEvents) {
```

  (only the doc's second line changes, and a third line is added), and after the existing block

```rust
        if let Some(id) = events.note {
            self.open_note(id);
            ctx.request_repaint();
        }
```

  insert

```rust
        if let Some(path) = events.focus {
            self.focus_input(path);
            ctx.request_repaint();
        }
```

- [ ] **Step 4: Run them again.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib a_check_guide_link`
  Expected: PASS, 2 passed (`gui::readouts::tests::a_check_guide_link_asks_for_its_input_s_row` and `gui::explorer::tests::a_check_guide_link_highlights_its_input_s_row_and_leaves_the_panel_as_it_was`).

- [ ] **Step 5: Write the failing dashboard tests.** In `dashboard.rs` `mod tests`, replace

```rust
    use crate::engine::meta::InputSet;
    use crate::gui::test_support::short_magnets;
```

  with

```rust
    use crate::engine::meta::InputSet;
    use crate::gui::guidance::{CHECK_GUIDES, Direction};
    use crate::gui::readouts::ReadoutEvents;
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, primary_button, short_magnets, sized_frame, text_color,
        text_rect,
    };
```

  Then append before the closing `}` of `mod tests`:

```rust

    /// A design that passes every guided check: the hot minimum low enough, the face gap and
    /// the cup wall raised (clearance, wall rule), and the space claim's diameter widened to
    /// take the larger cup.
    fn passing() -> DesignInputs {
        design(|i| {
            i.metal.required_min_Nm = 0.1;
            i.metal.face_gap_mm = 5.0;
            i.metal.cup_wall_corner_mm = 2.4;
            i.metal.max_diameter_mm = 60.0;
        })
    }

    /// The defaults with a hot day past the temperature limits: the verdict asks for a look.
    fn hot_day() -> DesignInputs {
        design(|i| {
            i.temperature.duty.hot_ambient_C = 80.0;
            i.temperature.duty.driving_rise_C = 60.0;
        })
    }

    /// The defaults at 40 poles: the blocks no longer fit around the ring.
    fn too_many_poles() -> DesignInputs {
        design(|i| i.coupling.npole = 40)
    }

    /// One frame of the dashboard of `inputs` at full width on a screen tall enough for every
    /// row: what it drew.
    fn dashboard_frame(ctx: &egui::Context, inputs: &DesignInputs) -> egui::FullOutput {
        let results = compute_all(inputs);
        sized_frame(ctx, egui::vec2(1280.0, 4000.0), Vec::new(), |ui| {
            dashboard_ui(ui, inputs, &results, &mut Readouts::default());
        })
    }

    #[test]
    fn every_guide_has_its_own_row_but_the_fit_row_which_is_drawn_apart() {
        // A guide draws under the row of its own check (the rows its verdict only badges have
        // none); the fit check is no result, so its row is drawn after the table's.
        for guide in &CHECK_GUIDES {
            let own_row = DASHBOARD.contains(&(guide.check, Some(guide.check)));
            assert_eq!(own_row, guide.check != FIT_CHECK, "{}", guide.check);
        }
        assert!(result_info(FIT_CHECK).is_none());
        let checks: BTreeSet<&str> = CHECK_GUIDES.iter().map(|g| g.check).collect();
        assert_eq!(checks.len(), CHECK_GUIDES.len(), "one guide per check");
    }

    #[test]
    fn a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it() {
        // A block shows exactly under a red or amber badge: a check greyed by f_end <= 0 (the
        // hot minimum, the screw) and a cup without back iron have no badge, and no block.
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing(),
            hot_day(),
            too_many_poles(),
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            design(|i| i.metal.max_diameter_mm = f64::NAN),
            design(|i| i.coupling.backiron = 0),
        ] {
            let results = compute_all(&inputs);
            for line in dashboard_lines(&results) {
                if let Some(guide) = guidance::guide(line.path) {
                    assert_eq!(
                        guidance::level(guide, &inputs, &results),
                        line.level.unwrap_or(Level::Good),
                        "{}",
                        line.path
                    );
                }
            }
        }
    }

    #[test]
    fn a_red_or_amber_check_shows_its_problem_why_and_fixes_and_every_guided_row_a_why_line() {
        let count = |texts: &[String], text: &str| texts.iter().filter(|t| *t == text).count();
        // Every fix text any guide can show.
        let fix_texts: BTreeSet<String> = CHECK_GUIDES
            .iter()
            .flat_map(|g| g.fixes.iter().map(|&(path, dir)| fix_text(path, dir)))
            .collect();
        let ctx = egui::Context::default();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing(),
            hot_day(),
            too_many_poles(),
        ] {
            let results = compute_all(&inputs);
            let texts = drawn_texts(&dashboard_frame(&ctx, &inputs));
            let flagged: Vec<&CheckGuide> = CHECK_GUIDES
                .iter()
                .filter(|g| red_or_amber(guidance::level(g, &inputs, &results)))
                .collect();
            let problems: Vec<String> = flagged
                .iter()
                .filter_map(|g| (g.problem)(&inputs, &results))
                .collect();
            assert_eq!(count(&texts, PROBLEM), problems.len());
            for problem in &problems {
                assert_eq!(count(&texts, problem), 1, "{problem:?}");
            }
            for guide in &CHECK_GUIDES {
                let shown = flagged.iter().any(|g| g.check == guide.check);
                assert_eq!(
                    count(&texts, guide.why),
                    usize::from(shown),
                    "{}",
                    guide.check
                );
            }
            // The fix links drawn are the red and amber guides' fixes, each once per guide.
            let mut want: Vec<String> = flagged
                .iter()
                .flat_map(|g| g.fixes.iter().map(|&(path, dir)| fix_text(path, dir)))
                .collect();
            let mut drawn: Vec<String> = texts
                .iter()
                .filter(|t| fix_texts.contains(*t))
                .cloned()
                .collect();
            want.sort();
            drawn.sort();
            assert_eq!(drawn, want);
            let with_fixes = flagged.iter().filter(|g| !g.fixes.is_empty()).count();
            assert_eq!(count(&texts, FIX), with_fixes);
            // Every guided row, any level, has its Why line.
            assert_eq!(count(&texts, WHY_LINK), CHECK_GUIDES.len());
        }
        // The workbook example's three red checks are the ones with blocks.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let red: BTreeSet<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| red_or_amber(guidance::level(g, &defaults, &results)))
            .map(|g| g.check)
            .collect();
        let want = [
            "metal.hot_min_check",
            "metal.clearance_check",
            "materials.cup_wall_check",
        ];
        assert_eq!(red, BTreeSet::from(want));
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        let cases = [
            // The workbook example: three red checks.
            (DesignInputs::default(), 3, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                2,
                0,
                Level::Bad,
            ),
            (passing(), 0, 0, Level::Good),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 3, 1, Level::Bad),
            // The passing design on the hot day: amber alone.
            (
                design(|i| {
                    i.temperature.duty.hot_ambient_C = 80.0;
                    i.temperature.duty.driving_rise_C = 60.0;
                    i.metal.required_min_Nm = 0.1;
                    i.metal.face_gap_mm = 5.0;
                    i.metal.cup_wall_corner_mm = 2.4;
                    i.metal.max_diameter_mm = 60.0;
                }),
                0,
                1,
                Level::Caution,
            ),
        ];
        for (inputs, bad, caution, colour) in cases {
            let results = compute_all(&inputs);
            let status = design_status(&inputs, &results);
            assert_eq!((status.bad, status.caution), (bad, caution));
            assert_eq!(status.total, CHECK_GUIDES.len());
            assert_eq!(status.is_default, inputs == DesignInputs::default());
            let output = dashboard_frame(&ctx, &inputs);
            let text = design_status_text(&status);
            assert_eq!(
                drawn_texts(&output).first(),
                Some(&text),
                "the status line tops the dashboard"
            );
            assert_eq!(text_color(&output, &text), Some(colour.color(&visuals)));
        }
    }

    #[test]
    fn the_fit_row_says_whether_the_blocks_fit_and_guides_a_misfit() {
        let ctx = egui::Context::default();
        let fit = guidance::guide(FIT_CHECK).unwrap();
        let count = |output: &egui::FullOutput, text: &str| {
            drawn_texts(output).iter().filter(|t| *t == text).count()
        };
        let output = dashboard_frame(&ctx, &DesignInputs::default());
        assert_eq!(count(&output, fit.label), 1);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (1, 0));
        assert_eq!(count(&output, fit.why), 0);
        // 40 poles: the blocks do not fit; a red badge, the problem and the guide's fixes.
        let inputs = too_many_poles();
        let results = compute_all(&inputs);
        assert!(!crate::engine::model::blocks_fit(
            &inputs.coupling,
            &results.model
        ));
        assert_eq!(guidance::level(fit, &inputs, &results), Level::Bad);
        let output = dashboard_frame(&ctx, &inputs);
        assert_eq!((count(&output, FITS), count(&output, DOES_NOT_FIT)), (0, 1));
        assert_eq!(count(&output, fit.why), 1);
        if let Some(problem) = (fit.problem)(&inputs, &results) {
            assert_eq!(count(&output, &problem), 1);
        }
        // The last badge drawn is the fit row's, red.
        let visuals = ctx.style().visuals.clone();
        let last_badge = crate::gui::test_support::flat_shapes(&output)
            .into_iter()
            .rev()
            .find_map(|shape| match shape {
                egui::Shape::Circle(c) if c.radius == 5.0 => Some(c.fill),
                _ => None,
            });
        assert_eq!(last_badge, Some(Level::Bad.color(&visuals)));
    }

    /// A guide of the cup wall check, the test's own, linking to `learn`.
    fn guide_with(learn: Learn) -> CheckGuide {
        CheckGuide {
            check: "materials.cup_wall_check",
            label: "Cup wall",
            problem: |_, results| Some(results.materials.cup_wall_check.clone()),
            why: "A test sentence.",
            fixes: &[("metal.cup_wall_corner_mm", Direction::Raise)],
            learn,
        }
    }

    /// Frames of `guide` at `level` on the defaults in `ctx`, a first one with no input, then one
    /// per entry of `events`: what the last drew and what the readouts heard in it.
    fn guide_frames(
        ctx: &egui::Context,
        guide: &CheckGuide,
        level: Level,
        events: Vec<Vec<egui::Event>>,
    ) -> (egui::FullOutput, ReadoutEvents) {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let mut last = None;
        for input in std::iter::once(Vec::new()).chain(events) {
            let mut readouts = Readouts::default();
            let output = sized_frame(ctx, egui::vec2(600.0, 400.0), input, |ui| {
                guide_ui(ui, guide, level, &inputs, &results, &mut readouts);
            });
            last = Some((output, readouts.finish()));
        }
        last.expect("one frame at least")
    }

    /// Where `guide` at `level` draws `text` on the defaults in `ctx`.
    fn guide_text_at(
        ctx: &egui::Context,
        guide: &CheckGuide,
        level: Level,
        text: &str,
    ) -> egui::Pos2 {
        let (output, _) = guide_frames(ctx, guide, level, Vec::new());
        text_rect(&output, text)
            .unwrap_or_else(|| panic!("no {text:?} in {:?}", drawn_texts(&output)))
            .center()
    }

    /// A click on `text` drawn by `guide` at `level`: what the readouts heard.
    fn click_in_guide(guide: &CheckGuide, level: Level, text: &str) -> ReadoutEvents {
        let ctx = egui::Context::default();
        let at = guide_text_at(&ctx, guide, level, text);
        let click = vec![
            vec![egui::Event::PointerMoved(at)],
            vec![primary_button(at, true)],
            vec![primary_button(at, false)],
        ];
        guide_frames(&ctx, guide, level, click).1
    }

    /// The texts drawn while `text` of `guide` at `level` is hovered, tooltips shown at once (no
    /// delay, the pointer need not rest): its hover text among them.
    fn hover_in_guide(guide: &CheckGuide, level: Level, text: &str) -> Vec<String> {
        let ctx = egui::Context::default();
        ctx.style_mut(|s| {
            s.interaction.tooltip_delay = 0.0;
            s.interaction.show_tooltips_only_when_still = false;
        });
        let at = guide_text_at(&ctx, guide, level, text);
        let hover = vec![vec![egui::Event::PointerMoved(at)], Vec::new()];
        drawn_texts(&guide_frames(&ctx, guide, level, hover).0)
    }

    #[test]
    fn a_why_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row() {
        // A reviewed note: its title (glyph-safe) asks for the note.
        let note = reviewed_note("end_effect").expect("a reviewed note");
        let events = click_in_guide(
            &guide_with(Learn::Note(note.id)),
            Level::Good,
            &glyph_safe(note.title),
        );
        assert_eq!((events.note, events.focus), (Some(note.id), None));
        // An assumption: its label asks for its first input's row; its hover text is the
        // rationale.
        let assumption = ASSUMPTIONS
            .iter()
            .find(|a| a.id == "production_variation")
            .unwrap();
        let guide = guide_with(Learn::Assumption(assumption.id));
        let events = click_in_guide(&guide, Level::Good, assumption.label);
        assert_eq!(
            (events.note, events.focus),
            (None, Some(assumption.paths[0]))
        );
        let texts = hover_in_guide(&guide, Level::Good, assumption.label);
        assert!(texts.iter().any(|t| t == assumption.rationale));
        // No note, or a note that has not passed review (no note has this id): "no note yet",
        // which asks for nothing; its hover text says why.
        for (learn, reason) in [
            (Learn::None("A test reason."), "A test reason."),
            (Learn::Note("no_such_note"), NOTE_NOT_REVIEWED),
        ] {
            let guide = guide_with(learn);
            assert_eq!(
                click_in_guide(&guide, Level::Good, NO_NOTE_YET),
                ReadoutEvents::default()
            );
            assert!(hover_in_guide(&guide, Level::Good, NO_NOTE_YET).contains(&reason.to_owned()));
        }
        // A red row's fix asks for its input's row.
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let events = click_in_guide(&guide_with(Learn::None("A test reason.")), Level::Bad, &fix);
        assert_eq!(
            (events.note, events.focus),
            (None, Some("metal.cup_wall_corner_mm"))
        );
    }

    #[test]
    fn the_block_shows_under_red_and_amber_in_their_colours_and_never_under_green() {
        let guide = guide_with(Learn::None("A test reason."));
        let problem = compute_all(&DesignInputs::default())
            .materials
            .cup_wall_check;
        let fix = fix_text("metal.cup_wall_corner_mm", Direction::Raise);
        let ctx = egui::Context::default();
        let visuals = ctx.style().visuals.clone();
        for level in [Level::Bad, Level::Caution] {
            let (output, _) = guide_frames(&ctx, &guide, level, Vec::new());
            let texts = drawn_texts(&output);
            for want in [
                PROBLEM,
                problem.as_str(),
                WHY_LINK,
                guide.why,
                FIX,
                fix.as_str(),
            ] {
                assert!(texts.iter().any(|t| t == want), "{level:?}: no {want:?}");
            }
            assert_eq!(text_color(&output, &problem), Some(level.color(&visuals)));
            // In order: the problem, the sentence (then its link), the fixes.
            let y = |text: &str| text_rect(&output, text).unwrap().top();
            assert!(y(&problem) < y(guide.why) && y(guide.why) < y(&fix));
        }
        let (output, _) = guide_frames(&ctx, &guide, Level::Good, Vec::new());
        assert_eq!(drawn_texts(&output), [WHY_LINK, NO_NOTE_YET]);
    }

    #[test]
    fn every_guide_line_has_glyphs_in_the_default_fonts_and_plain_words() {
        // Every guide at both levels (its sentence and fixes whatever the design) on designs
        // where each check fails (its problem line), the status lines, and the hover texts.
        let ctx = egui::Context::default();
        let mut plain: Vec<String> = Vec::new();
        for inputs in [
            DesignInputs::default(),
            short_magnets(),
            passing(),
            hot_day(),
            too_many_poles(),
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            design(|i| i.coupling.magnets.axial_length_mm = Some(50.8)),
        ] {
            let results = compute_all(&inputs);
            let output = sized_frame(&ctx, egui::vec2(600.0, 6000.0), Vec::new(), |ui| {
                let mut readouts = Readouts::default();
                status_ui(ui, &inputs, &results);
                fit_ui(ui, &inputs, &results, &mut readouts);
                for guide in &CHECK_GUIDES {
                    for level in [guidance::level(guide, &inputs, &results), Level::Bad] {
                        guide_ui(ui, guide, level, &inputs, &results, &mut readouts);
                    }
                }
            });
            plain.extend(drawn_texts(&output));
        }
        plain.extend([NOTE_NOT_REVIEWED, FITS, DOES_NOT_FIT].map(str::to_owned));
        // An assumption's rationale is the Assumptions view's own text, which may cite cells:
        // glyphs only.
        let mut rationales: Vec<&str> = Vec::new();
        for guide in &CHECK_GUIDES {
            match guide.learn {
                Learn::None(reason) => plain.push(reason.to_owned()),
                Learn::Assumption(id) => {
                    rationales.push(ASSUMPTIONS.iter().find(|a| a.id == id).unwrap().rationale)
                }
                Learn::Note(_) => {}
            }
        }
        for text in plain.iter().map(String::as_str).chain(rationales) {
            assert_glyphs(&ctx, text, "the check guides");
        }
        for text in &plain {
            assert!(
                guidance::plain_text_violations(text).is_empty(),
                "{text:?}: {:?}",
                guidance::plain_text_violations(text)
            );
        }
    }
```

- [ ] **Step 6: Write the failing panel tests.** Append to `mod tests` in `magcoupling-rs/src/gui/panel.rs`, before its closing `}`. The helpers used are all existing: `Harness::on_screen`, `Harness::new`, `harness.frame`, `harness.click_text`, `count`, `focus_rects`, `text_rect`, `text_rects`, `result_info`, `INPUTS_WIDTH`.

```rust

    #[test]
    fn a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing() {
        use crate::gui::guidance::{CHECK_GUIDES, fix_text, level};
        // Every fix of the workbook example's red checks, from the Geometry view (the input's
        // group or the Assumptions view opens) and from Size and torque (its Detailed inputs
        // open). Tall enough for every dashboard row.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        // The dashboard settles its width on the second frame, and the guides above a link wrap
        // to it: click where the link rests.
        harness.frame(Vec::new());
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let fixes: std::collections::BTreeSet<(&str, String)> = CHECK_GUIDES
            .iter()
            .filter(|g| level(g, &defaults, &results) == Level::Bad)
            .flat_map(|g| {
                g.fixes
                    .iter()
                    .map(|&(path, dir)| (path, fix_text(path, dir)))
            })
            .collect();
        assert!(!fixes.is_empty(), "the red checks have fixes");
        for view in [CentreView::Geometry, CentreView::SizeAndTorque] {
            for (path, text) in &fixes {
                // No frame left from the last fix.
                harness.panel.explorer.close();
                harness.panel.centre = view;
                harness.click_text(text);
                assert_eq!(
                    harness.panel.explorer.focus().map(|f| f.path.as_str()),
                    Some(*path),
                    "{view:?}"
                );
                assert!(
                    !harness.panel.explorer.open,
                    "the Equation panel stays shut"
                );
                let mut output = harness.frame(Vec::new());
                for _ in 0..10 {
                    output = harness.frame(Vec::new());
                }
                let label = InputCatalogue::get().entry(path).unwrap().meta.label;
                let rows: Vec<egui::Rect> = text_rects(&output, label)
                    .into_iter()
                    .filter(|r| r.left() < INPUTS_WIDTH)
                    .collect();
                let focus = focus_rects(&output);
                assert!(
                    rows.iter()
                        .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
                    "{view:?} {path}: {rows:?} {focus:?}"
                );
            }
        }
        // Following a fix edits nothing: no input, no undo step.
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_check_s_why_link_opens_its_note_in_the_equation_panel() {
        use crate::gui::guidance::{CHECK_GUIDES, Learn};
        let notes: Vec<&'static str> = CHECK_GUIDES
            .iter()
            .filter_map(|g| match g.learn {
                Learn::Note(id) => Some(id),
                _ => None,
            })
            .collect();
        assert!(!notes.is_empty(), "a guide links a reviewed note");
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        // Click where the links rest (the dashboard settles its width on the second frame).
        harness.frame(Vec::new());
        for id in notes {
            let note = crate::gui::explorer::reviewed_note(id).expect("a guide's note is reviewed");
            harness.panel.explorer.close();
            harness.click_text(&crate::gui::typeset::glyph_safe(note.title));
            assert!(harness.panel.explorer.open, "{id}");
            assert_eq!(harness.panel.explorer.note(), Some(id));
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(count(&output, note.sentences[0]), 1, "{id}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_status_line_counts_the_checks_at_default_and_after_a_fix() {
        use crate::gui::guidance::{design_status, design_status_text};
        let mut harness = Harness::new();
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        // The workbook example fails its hot minimum, its clearance and its cup wall.
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (3, 0, 8, true)
        );
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // A fix: the cup wall at 2.4 mm (its rule asks for 2.0 mm and up).
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        assert_eq!((status.bad, status.is_default), (2, false));
        assert_eq!(count(&output, &design_status_text(&status)), 1);
        // Reset all: the example again.
        harness.click_text(RESET_ALL);
        let output = harness.frame(Vec::new());
        let status = design_status(harness.panel.inputs(), harness.panel.results());
        assert_eq!((status.bad, status.is_default), (3, true));
        assert_eq!(count(&output, &design_status_text(&status)), 1);
    }

    #[test]
    fn a_narrow_window_wraps_the_check_guides_inside_the_dashboard() {
        // The M4-1 review's ~930 px window, tall enough for every row: every guide line of the
        // workbook example's red checks and the status line wrap inside the dashboard, never past
        // its right margin (a line past it would widen the dashboard into the centre region).
        use crate::gui::guidance::{
            CHECK_GUIDES, design_status, design_status_text, fix_text, level,
        };
        let width = 930.0;
        let mut harness = Harness::on_screen(egui::vec2(width, 3000.0));
        // The dashboard settles its width on the second frame.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let inputs = DesignInputs::default();
        let results = harness.panel.results().clone();
        let mut texts = vec![design_status_text(&design_status(&inputs, &results))];
        for guide in &CHECK_GUIDES {
            if level(guide, &inputs, &results) == Level::Bad {
                texts.extend((guide.problem)(&inputs, &results));
                texts.push(guide.why.to_owned());
                texts.extend(guide.fixes.iter().map(|&(path, dir)| fix_text(path, dir)));
            }
        }
        // Left of its first label's badge; right, the side panel's margin inside the frame's
        // (8 points each).
        let pullout = result_info("model.pullout_Nm").unwrap().meta.label;
        let dashboard = text_rect(&output, pullout).unwrap().left() - 30.0..=width - 16.0;
        for text in &texts {
            let rects = text_rects(&output, text);
            assert!(!rects.is_empty(), "{text:?} is drawn");
            for rect in rects {
                assert!(
                    dashboard.contains(&rect.left()) && dashboard.contains(&rect.right()),
                    "{text:?} at {rect:?}, the dashboard {dashboard:?}"
                );
            }
        }
    }
```

- [ ] **Step 7: Run them and confirm they fail.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::dashboard`
  Expected: the lib test target does not compile. Errors include:
  - `error[E0061]: this function takes 3 arguments but 4 arguments were supplied` (`dashboard_ui`)
  - `error[E0425]: cannot find function guide_ui / status_ui / fit_ui / red_or_amber / fix_text / design_status / design_status_text in this scope`
  - `error[E0425]: cannot find value PROBLEM / FIX / WHY_LINK / NO_NOTE_YET / NOTE_NOT_REVIEWED / FITS / DOES_NOT_FIT / FIT_CHECK / ASSUMPTIONS`
  - `error[E0412]: cannot find type CheckGuide / Learn`

- [ ] **Step 8: Implement the dashboard guides in `magcoupling-rs/src/gui/dashboard.rs`.**

  Append to the module doc, after `//! ([`DashboardLine::tooltip`]).`:

```rust
//!
//! Check guides (guidance spec 3.1 and 3.4, [`crate::gui::guidance`]): the design status line
//! tops the dashboard, and the Magnets fit row follows the end-effect flag (a GUI check of
//! `model::blocks_fit`, so it is drawn apart from [`DASHBOARD`], whose paths are results). Under
//! each guided check's own row, a red or amber level shows the problem from this design's
//! results, the guide's sentence and its fixes (each a link that shows its input's row), and
//! every guided row, at any level, has its Why line: a link to its reviewed note (opened in the
//! Equation panel), a link to its assumption (its row shown, the rationale in the hover text)
//! or "no note yet".
```

  Imports. Add `use crate::engine::assumptions::ASSUMPTIONS;` directly above `use crate::engine::explain::notes;`, and directly above `use crate::gui::readouts::Readouts;` add:

```rust
use crate::gui::guidance::{
    self, CheckGuide, FIT_CHECK, Learn, design_status, design_status_text, fix_text,
};
```

  Constants. Insert directly after `pub const WHY: &str = "Why";`:

```rust

/// The head of a failing check's problem line (guidance spec 3.4).
pub const PROBLEM: &str = "Problem:";

/// The head of a failing check's fix line: each fix a link that shows its input's row.
pub const FIX: &str = "Fix:";

/// The head of every guided check's Why line: the guide's sentence under a red or amber row,
/// then the link to read more.
pub const WHY_LINK: &str = "Why:";

/// What a Why line says when no reviewed note or assumption explains the check.
pub const NO_NOTE_YET: &str = "no note yet";

/// The hover text of a Why line whose note is not reviewed (sent back to Draft): it never shows.
pub const NOTE_NOT_REVIEWED: &str = "Its teaching note has not passed review.";

/// The Magnets fit row's value: the blocks fit around the ring, or not.
pub const FITS: &str = "Fit around the ring";
pub const DOES_NOT_FIT: &str = "Do not fit around the ring";
```

  Replace the doc comment and signature of `dashboard_ui` (HEAD 495-499) with:

```rust
/// Draws the dashboard of `results`, computed from `inputs`: the design status line, the
/// end-effect banner when f_end <= 0, then a row per line (badge and label, then the value and
/// the marker under them, so a narrow side still reads), a guided check's guide under its own
/// row, the stored-3D label after the last temperature row, and the Magnets fit row last. Each
/// result row is a readout (`readouts`): its hover text and equation while hovered, a click
/// opens it in the Equation panel.
pub fn dashboard_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    status_ui(ui, inputs, results);
    let lines = dashboard_lines(results);
```

  This replaces the old body's first statement, `let lines = dashboard_lines(results);`. The rest stays. In the row loop, replace

```rust
        readouts.show_over(ui, rect, line.path, || line.tooltip());
        if Some(index) == last_3d {
```

  with

```rust
        readouts.show_over(ui, rect, line.path, || line.tooltip());
        // A check's guide under its own row (the rows its verdict only badges have none).
        if let Some(guide) = guidance::guide(line.path) {
            let level = guidance::level(guide, inputs, results);
            guide_ui(ui, guide, level, inputs, results, readouts);
        }
        if Some(index) == last_3d {
```

  and replace the end of the function

```rust
        ui.add_space(4.0);
    }
}
```

  (the `}` that closes `dashboard_ui`, just before `/// The material warnings that fire`) with

```rust
        ui.add_space(4.0);
    }
    fit_ui(ui, inputs, results, readouts);
}

/// The design status line (guidance spec 3.1): whose design it is and how many guided checks
/// fail, in the colour of the worst level, over a separator.
fn status_ui(ui: &mut egui::Ui, inputs: &DesignInputs, results: &DesignResults) {
    let status = design_status(inputs, results);
    let level = if status.bad > 0 {
        Level::Bad
    } else if status.caution > 0 {
        Level::Caution
    } else {
        Level::Good
    };
    let text = egui::RichText::new(design_status_text(&status))
        .strong()
        .color(level.color(ui.visuals()));
    ui.add(egui::Label::new(text).wrap());
    ui.separator();
}

/// The Magnets fit row: its badge and label, whether the blocks fit (from the guide's level: red
/// when they do not), then its guide.
fn fit_ui(
    ui: &mut egui::Ui,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    let guide = guidance::guide(FIT_CHECK).expect("the Magnets fit row has a guide");
    let level = guidance::level(guide, inputs, results);
    ui.horizontal(|ui| {
        badge(ui, Some(level));
        ui.add(egui::Label::new(guide.label).wrap());
    });
    ui.horizontal(|ui| {
        ui.add_space(16.0);
        let value = if level == Level::Bad {
            DOES_NOT_FIT
        } else {
            FITS
        };
        ui.add(egui::Label::new(egui::RichText::new(value).strong()).wrap());
    });
    guide_ui(ui, guide, level, inputs, results, readouts);
}

/// Whether a check at `level` asks for action: red or amber, its guide shows the problem, the
/// sentence and the fixes.
fn red_or_amber(level: Level) -> bool {
    matches!(level, Level::Bad | Level::Caution)
}

/// A guided check's lines under its row, at `level`: when red or amber, the problem (from
/// `inputs` and `results`, in the level's colour), the Why line with the guide's sentence, and
/// the fixes; at any level, the Why line's link ([`learn_ui`]).
fn guide_ui(
    ui: &mut egui::Ui,
    guide: &CheckGuide,
    level: Level,
    inputs: &DesignInputs,
    results: &DesignResults,
    readouts: &mut Readouts,
) {
    let action = red_or_amber(level);
    if action && let Some(problem) = (guide.problem)(inputs, results) {
        guide_line(ui, PROBLEM, |ui| {
            let text = egui::RichText::new(problem).color(level.color(ui.visuals()));
            ui.add(egui::Label::new(text).wrap());
        });
    }
    guide_line(ui, WHY_LINK, |ui| {
        if action {
            ui.add(egui::Label::new(guide.why).wrap());
        }
        learn_ui(ui, &guide.learn, readouts);
    });
    if action && !guide.fixes.is_empty() {
        guide_line(ui, FIX, |ui| {
            for &(path, direction) in guide.fixes {
                if ui.link(fix_text(path, direction)).clicked() {
                    readouts.focus_input(path);
                }
            }
        });
    }
}

/// The link at the end of a Why line: a reviewed note's title (it asks for the note in the
/// Equation panel; a note not reviewed never shows), an assumption's label (it asks for its
/// input's row; the rationale is its hover text), or [`NO_NOTE_YET`] with the reason as its
/// hover text.
fn learn_ui(ui: &mut egui::Ui, learn: &Learn, readouts: &mut Readouts) {
    match *learn {
        Learn::Note(id) => match reviewed_note(id) {
            Some(note) => {
                if ui.link(glyph_safe(note.title)).clicked() {
                    readouts.open_note(note.id);
                }
            }
            None => {
                ui.weak(NO_NOTE_YET).on_hover_text(NOTE_NOT_REVIEWED);
            }
        },
        Learn::Assumption(id) => {
            let assumption = ASSUMPTIONS
                .iter()
                .find(|a| a.id == id)
                .expect("a guide's assumption is in the registry (guidance tests)");
            if ui
                .link(assumption.label)
                .on_hover_text(assumption.rationale)
                .clicked()
            {
                readouts.focus_input(assumption.paths[0]);
            }
        }
        Learn::None(reason) => {
            ui.weak(NO_NOTE_YET).on_hover_text(reason);
        }
    }
}

/// One line of a check's guide, 16 points in under its row: `head`, then `body` in a column
/// beside it that wraps within the width left.
fn guide_line(ui: &mut egui::Ui, head: &str, body: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal_top(|ui| {
        ui.add_space(16.0);
        ui.label(head);
        ui.vertical(body);
    });
}
```

- [ ] **Step 9: Pass the design shown to the dashboard.** In `magcoupling-rs/src/gui/panel.rs`, in `MagcouplingPanel::ui`, replace

```rust
                        .show(ui, |ui| dashboard_ui(ui, &self.results, &mut readouts));
```

  with

```rust
                        .show(ui, |ui| {
                            dashboard_ui(ui, &shown, &self.results, &mut readouts)
                        });
```

  `shown` is the `let shown = self.shown_inputs();` a few lines above. The results are computed from it, so in Torque -> Magnets (Auto) the status line reads "Your design".

- [ ] **Step 10: Run the new tests.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::dashboard`
    Expected: PASS, 23 passed (15 existing plus 8 new).
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing a_check_s_why_link_opens_its_note_in_the_equation_panel the_status_line_counts_the_checks_at_default_and_after_a_fix a_narrow_window_wraps_the_check_guides_inside_the_dashboard a_check_guide_link`
    Expected: PASS, 6 passed.

- [ ] **Step 11: Run the wider suite.**
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`
  Expected: exactly these 7 existing panel tests FAIL:
  - `every_headline_number_and_key_design_input_is_drawn_with_its_label`
  - `an_arrow_key_on_the_face_gap_slider_updates_the_headline`
  - `reset_all_restores_the_default_design_and_headline`
  - `the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers`
  - `a_design_past_the_space_claim_shows_its_overshoot`
  - `an_unreachable_target_shows_the_best_value_and_its_overshoot`
  - `hovering_a_value_marks_its_terms_where_their_values_are_shown`

  The cause: at the default design the dashboard grows from about 860 to about 1330 points (the status line, the three red blocks, eight Why lines and the fit row). In the 1280x1024 `Harness::new()` screen its lower rows (clamp screw, space claim, end effect) are scrolled out, and egui does not paint labels outside the clip.

  The hover test fails for a second, older reason. The dashboard widens on the second frame, a pre-existing effect of the cup wall row's "E9 E10" marker overflowing the 300-point default. The guide lines above the cup OD row now re-wrap at the new width and move the row up 28 points, so the pointer position read from the first frame lands on another row.

- [ ] **Step 12: Update those 7 tests.** Each is a one-line change in `panel.rs` `mod tests`. Find each by its name; T3 and T7 may have changed the bodies.
  - The first 6 above: replace their first line `let mut harness = Harness::new();` with
    `let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));`
    In `an_unreachable_target_shows_the_best_value_and_its_overshoot` only that line changes, whatever T3 and T7 made of the rest.
  - `hovering_a_value_marks_its_terms_where_their_values_are_shown`: keep `Harness::new()`. Replace

```rust
        tooltips_at_once(&harness);
        let output = harness.frame(Vec::new());
        let cup_od = harness.panel.results().model.cup_od_mm;
```

    with

```rust
        tooltips_at_once(&harness);
        // The dashboard settles its width on the second frame, and the check guides above the
        // cup OD wrap to it: measure where the row rests.
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let cup_od = harness.panel.results().model.cup_od_mm;
```

  - **Rule for any other test that fails the same way.** If a test added by T1-T7 reads a dashboard text below the first ten rows on the 1024-point screen, it gets the same first-line change. If it measures a dashboard position in its first frame and then clicks or hovers it, it gets the same settle frame.

- [ ] **Step 13: Run the wider suite again, then lint and format checks.**
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
    Expected: PASS, 0 failed, for the lib tests and every integration test (`tests/python_schema.rs` is untouched: no engine label, help or verdict text changed).
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
    Expected: no warnings.
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`
    Expected: no diff.

- [ ] **Step 14: Commit.**

```bash
git add magcoupling-rs/src/gui/readouts.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/panel.rs
git commit -F - <<'EOF'
feat(magcoupler): check guides on the dashboard

The dashboard opens with the design status line (whose design, how many
of the eight guided checks fail, cautions counted apart) and ends with a
Magnets fit row. Under a red or amber check: its problem from this
design's results, one plain sentence, and its fixes, each a link that
shows the input's row. Every guided row has a Why line linking its
reviewed note, its assumption, or "no note yet". Fix and assumption
links reach the input row through ReadoutEvents::focus, as leaf terms do.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
```

**Spec section 5 edge cases covered here:**

| Edge case | Test |
|---|---|
| Narrow windows: every new line wraps | `a_narrow_window_wraps_the_check_guides_inside_the_dashboard` |
| Glyphs (ASCII only, every guide text and hover text) | `every_guide_line_has_glyphs_in_the_default_fonts_and_plain_words` |
| Idle frames never edit inputs | `a_fix_link_shows_its_input_s_row_in_either_view_and_edits_nothing` (inputs unchanged and `undo_len() == 0` after many idle frames) |
| End effect out of range: a greyed check gets no block | `a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it` and `a_red_or_amber_check_shows_...` on `short_magnets()` |
| Note not reviewed, or no note | `a_why_link_asks_for_its_note_or_its_assumption_s_row_and_a_fix_for_its_input_s_row` |
| Fit fails (40 poles) | `the_fit_row_says_whether_the_blocks_fit_and_guides_a_misfit` |
| Caution counted apart from Bad | `the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour` |

**Existing tests this task breaks, and the edit to each:** the 7 tests in Step 11, with the edits in Step 12. No other existing test changes. `ReadoutEvents` gains a field, but every existing literal of it uses `..ReadoutEvents::default()` (`explorer.rs` tests), so they still compile.

**Notes for the implementer and reviewers:**
- At a 1024-point window the dashboard now needs its scroll bar to reach the space claim, end-effect and Magnets fit rows at the default design. This is the cost of spec 3.4 putting the blocks inline.
- The fix-link frame stays on the input row until the next value click, opened note or Equation panel close. This is the existing leaf-term behaviour (`Explorer::focus` is cleared only there).
- Docs are recorded in T10.

---

### Task 9: First view: open on Size and torque, with a welcome strip the user can hide

> **Amendments:** apply "T9 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


Spec 3.1, decision G-5 (reverses M42-2). A new panel opens on `CentreView::SizeAndTorque`. A one-line welcome strip sits above the centre tabs, and its Hide button removes it for the rest of the session. Hiding it is view state only: it is not saved in design files or share links and is not an undo step.

I checked this plan in a scratch copy of the crate, before T3-T8. Two changes would break tests:
- Changing only the default view breaks 64 existing panel tests. On the Size tab the inputs side shows the size controls, and the Key design group, the other groups, the filter and the Assumptions tab are folded under a closed "Detailed inputs" header.
- With the test harness kept on the Geometry tab, one test still breaks: `the_panel_works_inside_an_egui_window`. That test builds its own panel.

With all the changes below, 536 of 536 lib tests passed, and clippy `-D warnings` and `cargo fmt --check` were clean. A mutation check confirmed the new tests catch two faults: a strip that does not wrap, and a reset that clears the hidden flag.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`. Line numbers below are from before T3-T8, so find each edit by its text:
  - module doc (lines 5-6)
  - consts after `HEADING` (line 60)
  - `CentreView` docs (lines 172-176)
  - field after `centre` (line 243)
  - `new()` (line 294)
  - `centre_ui` (line 682)
  - new `welcome_ui` before `view_ui` (line 703)
  - test `Harness` (lines 1331-1345)
  - `the_geometry_view_is_the_default_and_follows_the_design_shown` (line 1462)
  - glyph test (line 3536)
  - `the_panel_works_inside_an_egui_window` (line 5322)
- Modify: `magcoupling-rs/src/gui/geometry_view.rs`, module doc lines 1-3.
- Modify: `.claude/workflows/gui-smoke.js`, lines 22, 27, 72-75, 82, 87 and 89. The web smoke currently expects the Geometry tab in its first screenshot.
- Test: `magcoupling-rs/src/gui/panel.rs`, in `mod tests`.

**Interfaces:**
- Consumes:
  - `CentreView::SizeAndTorque` and `crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS}` (both existing).
  - `Explorer::focus_input(&mut self, path: &str)` (existing).
  - From T4: `crate::gui::guidance::{design_status, design_status_text, plain_text_violations}`.
    - `pub fn design_status(inputs: &DesignInputs, results: &DesignResults) -> DesignStatus`
    - `pub fn design_status_text(status: &DesignStatus) -> String`
    - `pub fn plain_text_violations(text: &str) -> Vec<&'static str>`
  - From T8: the dashboard draws `design_status_text(..)` as one label.
- Produces:
  - `pub const WELCOME: &str = "Set the size -> read the torque -> open any red check to see why and what to change.";`
  - Private field `welcome_dismissed: bool` on `MagcouplingPanel`. It is not in `Design`, so it never reaches files, links or undo.
  - Extra for this task: `pub const HIDE_WELCOME: &str = "Hide";` and `pub const HIDE_WELCOME_HINT: &str`.
  - Test-only: `Harness::first_view()` and `Harness::of(panel, size)`.

---

- [ ] **Step 1: Keep the test harness on the Geometry tab (a refactor; behaviour unchanged).**

  In `magcoupling-rs/src/gui/panel.rs` `mod tests`, replace the `on_screen` function in `impl Harness`:

  ```rust
          /// A harness on a screen of `size`.
          pub(crate) fn on_screen(size: egui::Vec2) -> Self {
              let mut harness = Self {
                  ctx: egui::Context::default(),
                  panel: MagcouplingPanel::new(),
                  screen: size,
              };
              harness.frame(Vec::new());
              harness
          }
  ```

  with:

  ```rust
          /// A harness on a screen of `size`, on the Geometry tab: most tests drive the whole
          /// inputs side (the Key design group, the groups, the filter, the Assumptions view),
          /// which the Size tab a new panel opens on folds under "Detailed inputs". The first
          /// view's own tests use [`Harness::first_view`].
          pub(crate) fn on_screen(size: egui::Vec2) -> Self {
              let mut panel = MagcouplingPanel::new();
              panel.centre = CentreView::Geometry;
              Self::of(panel, size)
          }

          /// A harness on a new panel as it opens (decision G-5): the Size tab under the welcome
          /// strip, on the laptop screen.
          pub(crate) fn first_view() -> Self {
              Self::of(MagcouplingPanel::new(), SCREEN)
          }

          /// A harness drawing `panel` on a screen of `size`, one frame in.
          fn of(panel: MagcouplingPanel, size: egui::Vec2) -> Self {
              let mut harness = Self {
                  ctx: egui::Context::default(),
                  panel,
                  screen: size,
              };
              harness.frame(Vec::new());
              harness
          }
  ```

  Rename the geometry test. After this change its old first assertion would only re-check what the harness just set. Replace:

  ```rust
      #[test]
      fn the_geometry_view_is_the_default_and_follows_the_design_shown() {
          let mut harness = Harness::new();
          assert_eq!(harness.panel.centre, CentreView::Geometry);
          let output = harness.frame(Vec::new());
  ```

  with:

  ```rust
      #[test]
      fn the_geometry_view_follows_the_design_shown() {
          // The harness opens on the Geometry tab (a new panel opens on Size and torque:
          // a_new_panel_opens_on_size_and_torque_under_the_welcome_strip).
          let mut harness = Harness::new();
          let output = harness.frame(Vec::new());
  ```

  In `the_panel_works_inside_an_egui_window`, replace:

  ```rust
          let mut panel = MagcouplingPanel::new();
          let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
              let input = egui::RawInput {
                  events,
                  ..Default::default()
              };
              let output = ctx.run(input, |ctx| {
  ```

  with:

  ```rust
          let mut panel = MagcouplingPanel::new();
          // The Assumptions view is the inputs side of every view but Size and torque (which
          // folds the inputs under "Detailed inputs"; a new panel opens there).
          panel.centre = CentreView::Geometry;
          let frame = |panel: &mut MagcouplingPanel, events: Vec<egui::Event>| {
              let input = egui::RawInput {
                  events,
                  ..Default::default()
              };
              let output = ctx.run(input, |ctx| {
  ```

  A test that T3-T8 added and that builds `MagcouplingPanel::new()` directly, draws frames, and needs the Key design group or the Assumptions view needs the same `panel.centre = CentreView::Geometry;` line. Find candidates with `grep -n "MagcouplingPanel::new()" magcoupling-rs/src/gui/panel.rs`. Before T3-T8, line 5322 is the only one that needs it. The ones at lines 1579, 3093, 3416, 3443 and 3699 draw no frame. Lines 3213 and 4518 do not depend on the view.

- [ ] **Step 2: Run the panel suite. Expect PASS, since nothing has changed yet.**

  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::panel`

  Expected: `test result: ok.` with 0 failed.

- [ ] **Step 3: Write the failing tests.**

  Insert these immediately after `the_geometry_view_follows_the_design_shown`:

  ```rust
      #[test]
      fn a_new_panel_opens_on_size_and_torque_under_the_welcome_strip() {
          // Decision G-5 (spec 3.1, reversing M42-2): the first view is the Size tab with the
          // welcome strip above the centre tabs; showing them edits nothing, and Explain stays
          // off (M43-2).
          use crate::gui::guidance::{design_status, design_status_text};
          use crate::gui::size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
          let panel = MagcouplingPanel::new();
          assert_eq!(panel.centre, CentreView::SizeAndTorque);
          assert!(!panel.welcome_dismissed);
          assert!(!panel.explorer.explain);
          assert_eq!(
              MagcouplingPanel::default().centre,
              CentreView::SizeAndTorque
          );
          let mut harness = Harness::first_view();
          let output = harness.frame(Vec::new());
          assert_eq!(count(&output, WELCOME), 1);
          assert_eq!(count(&output, HIDE_WELCOME), 1);
          // The inputs side is the Size tab's: its controls, the Key design group folded away
          // under the detailed inputs.
          for label in [DIAMETER, LENGTH, GAP, MAGNETS] {
              assert_eq!(count(&output, label), 1, "{label}");
          }
          assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
          // The dashboard's design status line (spec 3.1) is on the first view, counted live.
          let status = design_status(harness.panel.inputs(), harness.panel.results());
          assert!(status.is_default);
          assert_eq!(count(&output, &design_status_text(&status)), 1);
          // The strip is in the centre region, above the tab row.
          let strip = text_rect(&output, WELCOME).expect("the strip");
          let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
          assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
          assert!(
              strip.left() >= INPUTS_WIDTH && strip.right() <= SCREEN.x - DASHBOARD_WIDTH,
              "{strip:?}"
          );
          assert_eq!(harness.panel.design(), Design::default());
          assert_eq!(harness.panel.history.undo_len(), 0);
      }

      #[test]
      fn hiding_the_welcome_strip_frees_its_row_and_edits_no_design() {
          let mut harness = Harness::first_view();
          let output = harness.frame(Vec::new());
          let tabs_under_strip = text_rect(&output, CentreView::Geometry.label()).unwrap();
          let file = design_to_json(&harness.panel.design());
          let link = harness.panel.share_link();
          harness.click_text(HIDE_WELCOME);
          assert!(harness.panel.welcome_dismissed);
          let output = harness.frame(Vec::new());
          assert_eq!(count(&output, WELCOME), 0);
          assert_eq!(count(&output, HIDE_WELCOME), 0);
          let tabs = text_rect(&output, CentreView::Geometry.label()).unwrap();
          assert!(
              tabs.top() < tabs_under_strip.top(),
              "the tab row moves up into the strip's room: {tabs:?} vs {tabs_under_strip:?}"
          );
          // A view choice of this session: the view, the design, its file and its link stay as
          // they were, and it is no undo step.
          assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
          assert_eq!(harness.panel.design(), Design::default());
          assert_eq!(design_to_json(&harness.panel.design()), file);
          assert_eq!(harness.panel.share_link(), link);
          assert_eq!(harness.panel.history.undo_len(), 0);
          assert!(!harness.panel.history.can_undo(&Design::default()));
          assert_eq!(harness.panel.last_error, None);
      }

      #[test]
      fn the_hidden_welcome_strip_stays_hidden_for_the_session() {
          // A loaded file, undo and redo, reset all, a refused file, a share link (as an edit and
          // as the session's start) and every view keep it hidden; only a new panel (the page
          // opened again) shows it.
          let hidden = |harness: &mut Harness, after: &str| {
              let output = harness.frame(Vec::new());
              assert!(harness.panel.welcome_dismissed, "{after}");
              assert_eq!(count(&output, WELCOME), 0, "{after}");
          };
          let mut harness = Harness::first_view();
          harness.click_text(HIDE_WELCOME);
          harness
              .panel
              .load_design_file(&design_to_json(&gap_design(2.0)))
              .unwrap();
          hidden(&mut harness, "a loaded design file");
          harness.click_text(UNDO);
          assert_eq!(harness.panel.design(), Design::default());
          hidden(&mut harness, "undo");
          harness.click_text(REDO);
          assert_eq!(harness.panel.design(), gap_design(2.0));
          hidden(&mut harness, "redo");
          harness.click_text(RESET_ALL);
          assert_eq!(harness.panel.design(), Design::default());
          hidden(&mut harness, "reset all");
          harness.panel.load_design_file("not a design").unwrap_err();
          hidden(&mut harness, "a refused design file");
          harness
              .panel
              .load_share_payload(&encode_share_payload(&gap_design(1.5)))
              .unwrap();
          hidden(&mut harness, "a share link");
          harness
              .panel
              .open_share_payload(&encode_share_payload(&gap_design(1.6)))
              .unwrap();
          hidden(&mut harness, "a share link opened as the session's start");
          for view in CentreView::ALL {
              harness.panel.centre = view;
              hidden(&mut harness, view.label());
          }
          assert!(!MagcouplingPanel::new().welcome_dismissed, "a new session");
      }

      #[test]
      fn a_share_link_opened_at_start_up_opens_on_the_first_view() {
          // The web app opens a link's design on a new panel (MagcouplingApp::open_share_payload):
          // the Size tab and the welcome strip, as for the default design.
          let mut panel = MagcouplingPanel::new();
          panel
              .open_share_payload(&encode_share_payload(&gap_design(2.0)))
              .unwrap();
          let mut harness = Harness::of(panel, SCREEN);
          let output = harness.frame(Vec::new());
          assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
          assert_eq!(count(&output, WELCOME), 1);
          assert_eq!(harness.panel.design(), gap_design(2.0));
      }

      #[test]
      fn idle_frames_on_the_first_view_change_no_input() {
          // The Size tab's controls (the Required slider among them), like every input row, write
          // only on an edit (decision M41-2): idle frames on the first view keep values off the
          // sliders' steps as they are.
          let mut design = Design::default();
          design.inputs.metal.face_gap_mm = 1.4123;
          design.inputs.coupling.inner_back_apothem_mm += 0.01234;
          design.inputs.coupling.magnets.axial_length_mm = Some(12.7123);
          design.inputs.metal.measured_drag_Nm = Some(0.012345);
          design.inputs.metal.required_min_Nm = 2.2345;
          let mut harness = Harness::first_view();
          harness.panel.open_design(design.clone());
          let mut output = harness.frame(Vec::new());
          for _ in 0..15 {
              output = harness.frame(Vec::new());
          }
          assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
          assert_eq!(count(&output, WELCOME), 1);
          assert_eq!(harness.panel.design(), design);
          assert_eq!(harness.panel.last_error, None);
          assert_eq!(harness.panel.history.undo_len(), 0);
      }

      #[test]
      fn a_focused_input_opens_the_detailed_inputs_on_the_first_view() {
          // On the Size tab the input groups sit under the closed "Detailed inputs": a leaf term
          // or a dashboard fix (spec 3.4) naming an input opens them and frames its row, for a
          // Key design input, an assumption and an input under an Advanced heading alike. A tall
          // screen draws every row.
          for path in [FACE_GAP, "coupling.c_end", "coupling.mu0"] {
              let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(1280.0, 3000.0));
              harness.panel.explorer.focus_input(path);
              let mut output = harness.frame(Vec::new());
              for _ in 0..10 {
                  output = harness.frame(Vec::new());
              }
              assert_eq!(harness.panel.centre, CentreView::SizeAndTorque, "{path}");
              let label = InputCatalogue::get().entry(path).unwrap().meta.label;
              let focus = focus_rects(&output);
              let rows: Vec<egui::Rect> = text_rects(&output, label)
                  .into_iter()
                  .filter(|r| r.left() < INPUTS_WIDTH)
                  .collect();
              assert!(
                  rows.iter()
                      .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
                  "{path}: {rows:?} {focus:?}"
              );
              assert_eq!(harness.panel.inputs(), &DesignInputs::default(), "{path}");
          }
      }

      #[test]
      fn the_welcome_strip_wraps_in_a_narrow_window() {
          // The M4-1 review's ~930 px window leaves the centre region about 300 points: the strip
          // wraps inside it, left of the dashboard, above the tab row.
          let width = 930.0;
          let mut harness = Harness::of(MagcouplingPanel::new(), egui::vec2(width, 1024.0));
          // The wrapped rows settle on the second frame (egui wraps from the last frame's widths).
          harness.frame(Vec::new());
          let output = harness.frame(Vec::new());
          let strip = text_rect(&output, WELCOME).expect("the strip");
          let hide = text_rect(&output, HIDE_WELCOME).expect("its button");
          let tabs = text_rect(&output, CentreView::Geometry.label()).expect("the tab row");
          assert!(
              strip.left() >= INPUTS_WIDTH && strip.right() <= width - DASHBOARD_WIDTH,
              "{strip:?}"
          );
          assert!(hide.right() <= width - DASHBOARD_WIDTH, "{hide:?}");
          assert!(
              strip.height() >= 2.0 * hide.height(),
              "wrapped: {strip:?} {hide:?}"
          );
          assert!(strip.bottom() <= tabs.top(), "{strip:?} above {tabs:?}");
      }

      #[test]
      fn the_welcome_strip_s_texts_are_plain_language() {
          // Spec section 6, test 3: no cell references, correction ids or workbook jargon.
          for text in [WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT] {
              assert_eq!(
                  crate::gui::guidance::plain_text_violations(text),
                  Vec::<&str>::new(),
                  "{text}"
              );
          }
      }
  ```

  In `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`, after the line `texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());`, add:

  ```rust
          texts.extend([WELCOME, HIDE_WELCOME, HIDE_WELCOME_HINT].map(str::to_owned));
  ```

- [ ] **Step 4: Run the new tests. Expect a compile failure.**

  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- welcome first_view`

  Expected errors:
  - `error[E0425]: cannot find value `WELCOME` in this scope`
  - the same for `HIDE_WELCOME` and `HIDE_WELCOME_HINT`
  - `error[E0609]: no field `welcome_dismissed` on type `gui::panel::MagcouplingPanel``

- [ ] **Step 5: Implement the change in `magcoupling-rs/src/gui/panel.rs`.**

  1. Module doc. Replace:

     ```rust
     //! the centre region between them ([`CentreView`]: the geometry view first, the results table
     //! last; decisions M42-1 and M42-2). Every result is recomputed with every approved correction
     ```

     with:

     ```rust
     //! the centre region between them ([`CentreView`]: one tab row, the results table last;
     //! decision M42-1), opening on Size and torque under a welcome strip the session can hide
     //! (decision G-5, which reverses M42-2). Every result is recomputed with every approved correction
     ```

  2. Constants. After `pub const HEADING: &str = "Magnetic coupling calculator";` and its blank line, insert:

     ```rust
     /// The welcome strip above the centre tabs (spec 3.1), until the session hides it.
     pub const WELCOME: &str =
         "Set the size -> read the torque -> open any red check to see why and what to change.";

     /// The button at the end of the welcome strip that hides it for the rest of the session.
     pub const HIDE_WELCOME: &str = "Hide";

     /// The hover text of [`HIDE_WELCOME`].
     pub const HIDE_WELCOME_HINT: &str =
         "Hide this line until the page is opened again; the design is not changed";

     ```

  3. `CentreView` docs. Replace:

     ```rust
         /// Direct envelope dimensions, magnet count and gap, with the live torque and geometry.
         SizeAndTorque,
         /// The end view and the side view to scale, with the dimension callouts (the default view:
         /// decision M42-2).
         Geometry,
     ```

     with the block below. If T6 or T7 reworded the `SizeAndTorque` line, keep their wording and append only the parenthesis.

     ```rust
         /// Direct envelope dimensions, magnet count and gap, with the live torque and geometry (the
         /// view a new panel opens on: decision G-5, which reverses M42-2).
         SizeAndTorque,
         /// The end view and the side view to scale, with the dimension callouts.
         Geometry,
     ```

  4. Field. Replace:

     ```rust
         /// The view the centre region shows.
         centre: CentreView,
     ```

     with:

     ```rust
         /// The view the centre region shows.
         centre: CentreView,
         /// Whether the session has hidden the welcome strip (spec 3.1): a view choice, never in a
         /// design file, a share link or the undo history, and kept by reset all and every load.
         welcome_dismissed: bool,
     ```

  5. `new()`. Replace `            centre: CentreView::Geometry,` with:

     ```rust
                 centre: CentreView::SizeAndTorque,
                 welcome_dismissed: false,
     ```

  6. `centre_ui`. Replace:

     ```rust
             confined(ui, "magcoupling_above_equation_panel", |ui| {
                 ui.horizontal_wrapped(|ui| {
     ```

     with:

     ```rust
             confined(ui, "magcoupling_above_equation_panel", |ui| {
                 if !self.welcome_dismissed {
                     self.welcome_ui(ui);
                 }
                 ui.horizontal_wrapped(|ui| {
     ```

  7. New method, inserted immediately before the doc line `/// The view of the centre region the tabs select, of `shown`, the design shown, its values`:

     ```rust
         /// The welcome strip (spec 3.1): what to do first, wrapping in a narrow region, and the
         /// button that hides it for the rest of the session.
         fn welcome_ui(&mut self, ui: &mut egui::Ui) {
             ui.horizontal_wrapped(|ui| {
                 ui.label(WELCOME);
                 if ui
                     .small_button(HIDE_WELCOME)
                     .on_hover_text(HIDE_WELCOME_HINT)
                     .clicked()
                 {
                     self.welcome_dismissed = true;
                 }
             });
         }

     ```

  8. In `magcoupling-rs/src/gui/geometry_view.rs`, replace:

     ```rust
     //! The geometry view (spec M4 "Layout": the centre region's default view, decision M42-2):
     //! paints [`crate::gui::geometry`] to scale with egui's painter, both views at one scale
     //! (decision M42-3), then lists the dimension callouts and the notes under them.
     ```

     with:

     ```rust
     //! The geometry view (spec M4 "Layout": the centre region's Geometry tab; Size and torque
     //! draws it too): paints [`crate::gui::geometry`] to scale with egui's painter, both views at
     //! one scale (decision M42-3), then lists the dimension callouts and the notes under them.
     ```

- [ ] **Step 6: Run the new tests and the glyph test. Expect PASS.**

  `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- welcome first_view every_text_the_panel_shows_has_glyphs_in_the_default_fonts`

  Expected: `test result: ok. 9 passed; 0 failed`. That is the 8 new tests plus the glyph test.

- [ ] **Step 7: Update the web smoke's first-view check in `.claude/workflows/gui-smoke.js`.**

  1. Line 22: replace `'design_file_loaded', 'geometry_view', 'explorer_ready'` with `'design_file_loaded', 'first_view', 'explorer_ready'`.
  2. Line 27: replace `    geometry_view: { type: 'boolean' },` with `    first_view: { type: 'boolean' },`.
  3. Lines 72-75. Replace:

     ```js
     // The magnetic coupling calculator: deterministic state through its share link, checked in
     // the console (the page is one canvas). The geometry view is the centre region's default view,
     // so the first screenshot shows it with no click. One click on the canvas, on the "Load design"
     // button found in a screenshot, checks the web file picker (rfd's HTML overlay) and the file load.
     ```

     with:

     ```js
     // The magnetic coupling calculator: deterministic state through its share link, checked in
     // the console (the page is one canvas). A new page opens on Size and torque under the welcome
     // strip (decision G-5), so the first screenshot shows both with no click. One click on the
     // canvas, on the "Load design" button found in a screenshot, checks the web file picker (rfd's
     // HTML overlay) and the file load.
     ```

  4. Line 82. Replace:

     `(inputs on the left, dashboard on the right, the geometry view in the middle) vs a blank page. geometry_view=true only if the middle region shows the geometry view: a row of tabs starting "Geometry", under it two drawings side by side (an end view of two rings of coloured magnet blocks inside a grey cup around a shaft, and a half side view with dashed space-claim lines), and a numbered list of dimension callouts under them ("1 Face gap ...", "4 Overall length: ...").`

     with:

     `(the size controls on the left, dashboard on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the middle region shows the first view with no click: at its top the line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, under it a row of tabs starting "Size and torque" (the selected tab), and under the tabs the torque readouts above a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft.`

  5. Line 87: replace `canvas present, geometry_view, share_link_loaded` with `canvas present, first_view, share_link_loaded`.
  6. Line 89: replace `geometry_view: false,` with `first_view: false,`.

  Check that nothing is left: `grep -n "geometry_view" .claude/workflows/gui-smoke.js` should print nothing.

- [ ] **Step 8: Run the wider suite.**

  1. `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check`. Expected: no output.
  2. `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`. Expected: `test result: ok.` with 0 failed. This includes the app tests; `app::tests::the_smoke_test_share_link_opens_its_design` reads `gui-smoke.js` and must stay green.
  3. `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`. Expected: `Finished`, no warnings.

- [ ] **Step 9: Commit.**

  ```bash
  git add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/geometry_view.rs .claude/workflows/gui-smoke.js
  git commit -F - <<'EOF'
  feat(magcoupler): open on Size and torque under a dismissible welcome strip

  A new panel opens on the Size tab (decision G-5, reversing M42-2) with
  "Set the size -> read the torque -> open any red check to see why and
  what to change." above the centre tabs; Hide removes it for the session
  (view state: not in files, links or undo; kept by reset all, loads, undo
  and redo; it wraps in a narrow window). The panel test harness opens on
  the Geometry tab explicitly so the tests of the whole inputs side keep
  their view; Harness::first_view tests the new start. The web smoke
  checks the first view instead of the Geometry tab.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
  EOF
  ```

**Spec section 5 edge cases covered here:**
- Narrow windows: `the_welcome_strip_wraps_in_a_narrow_window`, and the glyph test now includes the three strip strings, which are ASCII `->` only.
- Idle frames never edit inputs on the new first view: `idle_frames_on_the_first_view_change_no_input`.
- Not in files or links, and kept through the session: the hiding and stays-hidden tests.

**Existing tests that need an edit:**
1. `the_geometry_view_is_the_default_and_follows_the_design_shown`: renamed to `the_geometry_view_follows_the_design_shown`, and the `assert_eq!(harness.panel.centre, CentreView::Geometry);` line is dropped (Step 1). Its name would be wrong otherwise.
2. `the_panel_works_inside_an_egui_window`: fails at `assert_eq!(panel.inputs_view, InputsView::Assumptions)`, because the Size tab has no Assumptions view. Fix: add `panel.centre = CentreView::Geometry;` (Step 1).

**Existing tests that fail on the default flip alone but need no edit:** these 64 panel tests rely on the Key design group, the groups, the filter or the Assumptions tab being visible on a fresh harness. The Step 1 `Harness::on_screen` change fixes all of them:

`a_changed_input_shows_the_dot_and_its_reset_restores_the_default`, `a_drag_in_progress_defers_the_solve`, `a_drag_is_one_undo_step`, `a_held_arrow_key_is_one_undo_step`, `a_leaf_assumption_term_shows_its_row_in_the_assumptions_view`, `a_material_choice_sums_up_its_properties_and_fires_its_warnings`, `a_material_code_outside_the_choices_is_shown_and_kept`, `a_refused_design_file_changes_nothing_and_says_why`, `a_selector_switches_the_branch`, `a_share_link_carries_the_sizing_state`, `a_share_link_opened_at_start_up_is_not_an_undo_step`, `a_sum_s_harmonic_set_marks_its_row_and_its_h_finds_it`, `a_text_input_edits_the_part_name_and_says_what_it_resolves_to`, `a_trace_moves_no_input_row_so_a_second_click_on_the_label_ends_it`, `an_arrow_key_on_the_face_gap_slider_updates_the_headline`, `an_input_outside_its_slider_range_is_kept_until_edited_and_flagged`, `an_input_whose_results_have_no_equation_record_traces_none_and_says_so`, `an_unreachable_target_shows_the_best_value_and_its_overshoot`, `both_spreadsheet_buttons_queue_the_workbook_of_the_design_shown`, `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group`, `clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click`, `clicking_the_end_of_the_face_gap_rail_sets_its_maximum`, `copy_share_link_puts_the_design_s_link_on_the_clipboard`, `ctrl_z_in_a_text_field_is_left_to_the_field`, `each_arrow_nudge_is_one_undo_step`, `entering_the_measured_drag_starts_at_the_model_s_drag`, `every_group_opens_and_draws_a_row_for_each_of_its_inputs`, `every_headline_number_and_key_design_input_is_drawn_with_its_label`, `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`, `hovering_a_value_marks_its_terms_where_their_values_are_shown`, `idle_frames_change_no_input`, `leaving_torque_to_magnets_keeps_the_sized_value_as_one_undo_step`, `leaving_torque_to_magnets_while_solving_writes_the_current_solution`, `manual_size_adopts_the_inverse_solution_before_editing`, `only_the_rows_drawn_this_frame_count_as_design_fields`, `reset_all_can_be_undone`, `reset_all_restores_the_default_design_and_headline`, `save_design_queues_the_design_file_and_load_design_asks_for_one`, `size_and_torque_has_direct_dimension_and_gap_controls`, `stepping_back_to_the_default_lands_on_it_exactly`, `the_assumptions_banner_appears_on_change_and_clears_on_reset`, `the_assumptions_view_lists_each_assumption_with_its_rationale_and_source`, `the_axial_length_override_starts_blank_and_enters_at_the_ring_length`, `the_export_buttons_queue_the_files_for_the_host`, `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`, `the_free_variable_picker_switches_the_variable`, `the_free_variable_s_row_shows_the_solved_value_locked`, `the_geometry_view_is_the_default_and_follows_the_design_shown`, `the_grade_pickers_set_a_grade_by_its_id_or_none_for_either_ring`, `the_json_export_in_torque_to_magnets_holds_the_inputs_that_produced_the_results`, `the_keyboard_shortcuts_undo_and_redo`, `the_locked_free_variable_s_label_still_traces_it`, `the_order_toggle_switches_the_view_and_changes_no_input`, `the_outer_ring_s_part_picker_sets_the_outer_part_only`, `the_panel_works_inside_an_egui_window` (also needs edit 2 above), `the_part_picker_sets_a_library_part_or_custom_dimensions`, `the_pole_count_steps_by_two_and_stays_even`, `the_slider_range_ends_are_hard_stops_for_arrow_keys`, `the_solve_waits_for_the_debounce_and_never_runs_per_frame`, `the_spreadsheet_follows_the_input_order_shown_and_torque_to_magnets`, `torque_to_magnets_shows_the_solved_design`, `typing_a_part_name_is_one_undo_step_once_the_field_loses_focus`, `typing_in_the_results_search_does_not_defer_the_solve`, `undo_reverses_a_change_and_redo_brings_it_back`.

T3 and T7 may have renamed or removed some of these. Any test T3-T8 added through `Harness::new()` or `Harness::on_screen()` is covered by the same harness change. The strip itself broke no existing test, including the window and Equation-panel overlap tests.

**Behaviour to know:** on the Size tab, a leaf assumption term or a Fix link shows the input's row in its group under "Detailed inputs", not in the Assumptions view. `a_focused_input_opens_the_detailed_inputs_on_the_first_view` pins that behaviour.

**Handover to T10 (docs that still say Geometry is the default):**
- `docs/ai/03-structure.yaml` gui line: "CentreView (Geometry, the default; ...)" should name SizeAndTorque as the default, plus WELCOME / HIDE_WELCOME and `Harness::first_view`.
- `docs/ai/02-system.yaml:73`: "the geometry view to scale (the default)".
- `magcoupling-rs/README.md` lines 35, 195, 304 and 459. Line 459 also describes the smoke's "geometry view in the first screenshot".
- `docs/ai/04-memory.yaml`: add a G-5 entry. Line 77 is a historical RESOLVED record and stays as it is.

---

### Task 10: Documentation for guidance slice 1 (one torque number, Size-tab Auto, guidance layer, BL-055)

> **Amendments:** apply "T10 amendments" and the contract changes C-1..C-10 above (binding; they override this text).


Documentation only. No Rust file changes. The one non-doc edit is the `gui-smoke.js` prompt text (step 9), which the README describes. The controller may move it to T9 if T9 already covers the smoke prompt.

Run this task last, after T1 to T9 are merged on `magcoupling/guidance`. The docs state names and strings from those tasks, and the code is the authority. Step 1 checks them. If a check fails, copy the code's name or string into the snippets below. Do not change code.

**Files:**
- Modify `docs/ai/02-system.yaml`: lines 73, 80, 81, 92, 99, 100, 102, 103, a new block before line 104, after line 461, and the end of line 470.
- Modify `docs/ai/03-structure.yaml`: lines 35, 50, 51, 65 (six substrings), 70 and 72.
- Modify `docs/ai/04-memory.yaml`: after line 6, before and at the end of line 10, lines 67 and 77.
- Modify `docs/ai/05-update-tracker.md`: new entry above line 8 (`## 2026-10-12`).
- Modify `docs/ai/backlog.yaml`: new BL-055 above line 8 (`- id: BL-054`).
- Modify `magcoupling-rs/README.md`: status paragraph (lines 36 and 44), layout table (147, 150), The panel (170-188, 194-195), Dashboard and Geometry bullets (275-286, 304), Sizing bullet (423-430), smoke paragraph (457-460), tests table (509, 515, 521), regeneration paragraph (540).
- Modify `.claude/workflows/gui-smoke.js`: lines 73-74 and 82, prompt text only. The constants `MAGCOUPLING_SMOKE_PAYLOAD` and `MAGCOUPLING_SMOKE_DESIGN_FILE` stay untouched, because `app.rs::the_smoke_test_share_link_opens_its_design` reads them with `include_str!`.
- Test: an inline bash docs check (step 2) and a YAML parse of `03-structure.yaml`, `04-memory.yaml` and `backlog.yaml`. `02-system.yaml` is not strict YAML today (line 14 fails to parse), so it is checked by grep only.

**Interfaces:**
- Consumes (names and strings the docs cite, from the contract):
  - T1: `diameter_breakdown`, `length_breakdown`, `DiameterFit`, `smallest_fitting_diameter`, `ring_radius_range`.
  - T2: `solve_in`, `FreeVariable::grid_in`, `SizingError::InvalidRange`.
  - T3: `SizingState { mode, variable }`, `REQUIREMENT_INPUT`, `target_mismatch_notice`, `target_torque_Nm`.
  - T4: `CHECK_GUIDES`, `FIT_CHECK`, `design_status_text`, `plain_text_violations`.
  - T5: `Consequences`, `ConsequenceCache`, `RING_SIZE_STAYS`.
  - T6: `torque_gap_text`, `pullout_label`, `NO_PREDICTION`.
  - T7: `AUTO`, `SET_BY_AUTO`, `variable_slider`, `change_text`, `AUTO_STATUS_START`.
  - T8: `WHY_LINK`, `NO_NOTE_YET`.
  - T9: `WELCOME`, `welcome_dismissed`.
- Produces: the written record only. BL-055 is the backlog id, G-1 to G-8 are the decision ids, and 2026-10-13 is the tracker date.

---

- [ ] **Step 1: Check the facts the docs state against the merged code**

Run from the repo root:

```bash
cd C:/Users/Cole/source/repos/linkage_simulation
bad=0
has() { grep -qF -- "$2" "$1" || { echo "NOT FOUND in $1: $2"; bad=1; }; }
hasnt() { if grep -qF -- "$2" "$1"; then echo "STILL THERE in $1: $2"; bad=1; fi; }
E=magcoupling-rs/src/engine; G=magcoupling-rs/src/gui
has $E/sizing.rs 'pub fn solve_in'
has $E/sizing.rs 'pub fn grid_in'
has $E/sizing.rs 'InvalidRange'
for n in 'pub fn diameter_breakdown' 'pub fn length_breakdown' 'pub fn smallest_fitting_diameter' 'pub fn ring_radius_range' 'pub enum DiameterFit'; do has $E/size_controls.rs "$n"; done
has $G/sizing.rs 'pub const REQUIREMENT_INPUT'
has $G/sizing.rs 'pub const SOLVED_PREFIX'
has $G/sizing.rs 'Sizing refused: the search range is empty'
has $G/sizing.rs 'The requirement must be a positive torque.'
has $G/session.rs 'pub fn target_mismatch_notice'
has $G/session.rs 'target_torque_Nm'
for n in 'pub static CHECK_GUIDES' 'pub const FIT_CHECK' 'pub fn design_status_text' 'pub fn plain_text_violations' 'pub struct ConsequenceCache' 'pub const RING_SIZE_STAYS' 'Workbook example design'; do has $G/guidance.rs "$n"; done
for n in 'pub const REQUIRED' 'pub const HOT_LOW' 'pub fn pullout_label' 'pub const NO_PREDICTION' 'pub fn torque_gap_text' 'pub const AUTO' 'pub const SET_BY_AUTO' 'pub const AUTO_STATUS_START' 'pub fn variable_slider' 'pub fn change_text'; do has $G/size_controls.rs "$n"; done
has $G/size_controls.rs 'no prediction (magnets too short for the end-effect model)'
has $G/size_controls.rs 'Set by Auto; turn Auto off to edit'
has $G/panel.rs 'pub const WELCOME'
has $G/panel.rs 'Set the size -> read the torque'
has $G/panel.rs 'welcome_dismissed'
has $G/panel.rs 'centre: CentreView::SizeAndTorque'
hasnt $G/panel.rs 'view == CentreView::SizeAndTorque'
hasnt $G/sizing.rs 'TARGET_RANGE_INPUT'
hasnt $G/sizing.rs 'DEFAULT_TARGET_NM'
hasnt $G/dashboard.rs '"guidance.blocks_fit"'
grep -rqF 'WHY_LINK' $G || { echo "NOT FOUND in $G: WHY_LINK"; bad=1; }
grep -rqF 'NO_NOTE_YET' $G || { echo "NOT FOUND in $G: NO_NOTE_YET"; bad=1; }
echo "code facts: $([ $bad = 0 ] && echo OK || echo CHECK THE LINES ABOVE)"
```

Expected: `code facts: OK`.

- If a name is missing, adopt the code's name in every snippet below that cites it.
- If `dashboard.rs` still holds `"guidance.blocks_fit"` (the Magnets fit row was added to `DASHBOARD`), change "drawn beside `DASHBOARD`, in no engine table" in edits 3.3-c and 8.8 to "in `DASHBOARD`" and say so.
- If the requirement wording or the `Sizing refused` wording differs, copy the code's string into edit 3.9 and the README Sizing bullet.

- [ ] **Step 2: Write the failing docs check and run it**

Run from the repo root:

```bash
cd C:/Users/Cole/source/repos/linkage_simulation
fail=0
need() { grep -qF -- "$2" "$1" || { echo "MISSING in $1: $2"; fail=1; }; }
gone() { if grep -qF -- "$2" "$1"; then echo "STALE in $1: $2"; fail=1; fi; }
S=docs/ai/02-system.yaml; T=docs/ai/03-structure.yaml; M=docs/ai/04-memory.yaml
U=docs/ai/05-update-tracker.md; B=docs/ai/backlog.yaml; R=magcoupling-rs/README.md; W=.claude/workflows/gui-smoke.js
need $S '(the default until 2026-10-13)'
need $S 'Guidance slice 1 (2026-10-13, BL-055, decisions G-1 to G-8)'
need $S 'src/gui/guidance.rs, src/gui/size_controls.rs'
need $S 'sizing::solve_in over variable.range()'
need $S 'Opening this tab no longer changes the sizing mode'
need $S 'ring-size Auto searches this range'
need $S "or of a Size-tab slider, the Required slider's included"
need $S 'so the solve key is the inputs plus the search range'
need $S 'One torque number (decision G-2'
need $S 'The solver lives on the Size tab (decisions G-4, G-7)'
need $S 'The guidance layer (gui::guidance'
need $S 'The first view is Size and torque (decision G-5'
need $S 'A GUI control that mirrors an engine input must edit that input'
need $S 'guidance slice 1 built on branch magcoupling/guidance'
gone $S 'Opening this tab adopts any inverse solution and switches to forward calculation'
gone $S "(or the target's value box)"
need $T 'guidance slice 1 (BL-055) adds gui/guidance.rs'
need $T 'solve_in(inputs, variable, target, range)'
need $T 'ring_radius_range (that range as coupling.inner_back_apothem_mm)'
need $T 'CentreView (SizeAndTorque, the first view since 2026-10-13'
need $T 'the welcome strip (WELCOME, welcome_dismissed)'
need $T 'guidance.rs (pure, no egui'
need $T 'target_mismatch_notice'
need $T 'the Required slider (metal.required_min_Nm)'
need $T 'also solve_in, grid_in and InvalidRange'
gone $T 'CentreView (Geometry, the default;'
need $M 'Magcoupler guidance slice 1 (spec'
need $M 'NEXT (owner order, 2026-10-13, G-1)'
need $M 'the target is the requirement metal.required_min_Nm'
need $M 'the default until 2026-10-13: the first view is now Size and torque'
need $U 'Magcoupler guidance slice 1: one torque number, Size-tab solver, check guides (BL-055'
need $B '- id: BL-055'
need $R 'Guidance slice 1 complete'
need $R '(the Size tab'"'"'s Auto since slice 1)'
need $R '**Required** is a slider on'
need $R '**Check guides**'
need $R '`solve_in` (the search over a range the caller gives'
need $R '| `tests/size_controls.rs` |'
need $R 'Opening this tab changes nothing about the sizing mode'
gone $R 'Opening this tab switches to forward calculation'
gone $R '(the default view: no click)'
gone $R 'the geometry view is the default and follows the design shown'
gone $R 'the geometry view to scale (the default: both rings'
need $W 'a row of tabs starting "Size and torque"'
gone $W 'a row of tabs starting "Geometry"'
echo "docs check: $([ $fail = 0 ] && echo PASS || echo FAIL)"; [ $fail = 0 ]
```

Expected before the edits: `docs check: FAIL`, exit 1.
- Every `need` prints a `MISSING in ...` line.
- These `gone` lines print `STALE`: the 02-system forward-calculation sentence, the `(or the target's value box)` text, 03-structure `CentreView (Geometry, the default;`, the three README strings, and the gui-smoke `"Geometry"` tab text.

- [ ] **Step 3: Edit `docs/ai/02-system.yaml`**

Apply each edit with exact-string replacement (the file is LF). Use single quotes inside the YAML double-quoted strings, never double quotes.

**3.1** Line 73, geometry view as default:

```text
OLD: the geometry view to scale (the default), five plots
NEW: the geometry view to scale (the default until 2026-10-13), five plots
```

**3.2** After line 80, a new line in the same plain scalar at 6 spaces (avoid a colon followed by a space):

```text
OLD:       longer depends on the crate: its Tools menu links to the site.
NEW:       longer depends on the crate: its Tools menu links to the site.
       Guidance slice 1 (2026-10-13, BL-055, decisions G-1 to G-8) adds gui/guidance.rs (check guides, the design status line, what-moves-what consequence lines), the one torque number, the Size tab's Auto solver and the Size tab as the first view.
```

(The NEW block is the OLD line followed by the added line, indented 6 spaces.)

**3.3** Line 81, `key_files`:

```text
OLD: src/gui/dashboard.rs, src/gui/geometry.rs,
NEW: src/gui/dashboard.rs, src/gui/guidance.rs, src/gui/size_controls.rs, src/engine/size_controls.rs, src/gui/geometry.rs,
```

**3.4** Line 92, inverse sizing invariant. Replace the start of the sentence only:

```text
OLD: Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan
NEW: Inverse sizing (sizing::solve, which is sizing::solve_in over variable.range() with identical results) evaluates compute_all only inside the search range: the free variable's slider range or, for ring-size Auto, the Size tab's diameter range as an apothem range (engine::size_controls::ring_radius_range); a range with a non-finite end or min > max is SizingError::InvalidRange. A continuous variable's 64-cell scan (FreeVariable::grid_in(range))
```

**3.5** Line 99. This replaces the existing sentence that says opening the tab forces forward mode:

```text
OLD: Opening this tab adopts any inverse solution and switches to forward calculation; files/share links need no new schema.
NEW: Opening this tab no longer changes the sizing mode (until 2026-10-13 it adopted any inverse solution and switched to forward calculation): the tab is the first view and the solver's home (Auto, below). Files/share links need no new schema.
```

**3.6** Line 100, Minimum OD:

```text
OLD: Inverse sizing and engine input metadata retain their existing ranges.
NEW: Engine input metadata and the other inverse-sizing variables keep their ranges; ring-size Auto searches this range (the Minimum OD included).
```

**3.7** Line 102, edit in progress:

```text
OLD: a text field of an input row (or the target's value box) has focus
NEW: a text field of an input row (or of a Size-tab slider, the Required slider's included) has focus
```

**3.8** Line 103, runner. Replace the middle of the sentence:

```text
OLD: and no edit is in progress. In Torque -> Magnets the panel shows the inputs with the free variable at the solved (or best) value;
NEW: and no edit is in progress. The target is inputs.metal.required_min_Nm (SizingState holds only the mode and the free variable), so the solve key is the inputs plus the search range (state.variable.range(), or ring_radius_range for ring size) and a target edit re-solves after the debounce like any other edit. In Torque -> Magnets (the Size tab's Auto) the panel shows the inputs with the free variable at the solved (or best) value;
```

**3.9** Insert four invariants before line 104, which starts with `      - "The panel does no I/O`. Use the first 12 words of that line as `old_string` and put the four bullets before it:

```text
      - "One torque number (decision G-2, 2026-10-13; reverses A2-9 and M41-9 for the target only): the sizing target is metal.required_min_Nm (gui::sizing::REQUIREMENT_INPUT, formerly TARGET_RANGE_INPUT), the input that colours the checks, so SizingState holds only the mode and the free variable (no target_Nm, no DEFAULT_TARGET_NM). The Torque -> Magnets target and the Size tab's Required slider edit it through InputSet::set, an ordinary design edit (undo, files, links). A file or link still writes sizing.target_torque_Nm, equal to the requirement; DESIGN_VERSION and the schema are unchanged. Loading validates it as before; when it differs from the file's metal.required_min_Nm the requirement is used and the panel reports session::target_mismatch_notice. A requirement <= 0 is refused by the solver (The requirement must be a positive torque.), and a search range with a non-finite end or min > max by SizingError::InvalidRange."
      - "The solver lives on the Size tab (decisions G-4, G-7): an Auto toggle (SizingMode::TorqueToMagnets) with a free-variable choice that maps to a slider (gui::size_controls::variable_slider: magnet length -> Overall length, ring size -> Outside diameter, magnets per ring -> Magnets per ring; magnet length is the default). With Auto on, that slider is disabled and shows the solved (or best) value (SET_BY_AUTO), edits to any other input re-solve after DEBOUNCE_S, and the outcome line names the change (change_text) or the best value when the requirement is not reachable. Opening the Size tab no longer changes the sizing mode; the Key design group's mode switch is a status line (AUTO_STATUS_START) with a button to the Size tab, so the solver has one home. Ring-size Auto searches the Size tab's diameter range, the Minimum OD included, mapped to coupling.inner_back_apothem_mm by engine::size_controls::ring_radius_range (so below the old 9 mm radius floor when the minimum is lowered); sizing::is_valid still rejects every layout that fails fit, keyway wall or the end-effect range. Idle frames never edit an input."
      - "The guidance layer (gui::guidance: pure functions, no egui; spec docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md) is GUI-side and changes no engine result or string: the engine's labels, help and verdict texts keep the workbook wording (tests/python_schema.rs). CHECK_GUIDES holds one CheckGuide for each of the seven dashboard checks (metal.hot_min_check, metal.clearance_check, materials.cup_wall_check, temperature.summary.verdict, clamps.recommended, housing.space_claim_check, model.end_effect_check) and for the Magnets fit row, whose path FIT_CHECK (guidance.blocks_fit) is a GUI-only pseudo path drawn beside DASHBOARD, in no engine table. A guide gives a Problem built from the live results, a reviewed one-sentence Why, Fix links (an input path and a Direction; the on-screen label always comes from InputCatalogue, guidance never types one) and a Learn target (a reviewed note, an assumption, or a stated reason for none). level() reuses the dashboard's own level logic; design_status counts Bad and Caution rows over the eight guides and calls the design the Workbook example only while the inputs equal DesignInputs::default(). Every number in guidance text comes from results, never a literal, and no text matches plain_text_violations (cell references, correction ids, workbook, Addendum, Rust-only, Calculator!). Every new Why, Problem and consequence sentence passes a fresh-context session-model physics review before release (decision G-3)."
      - "The first view is Size and torque (decision G-5; reverses M42-2): MagcouplingPanel::new opens CentreView::SizeAndTorque under a one-line welcome strip (WELCOME) that a close button hides for the session (welcome_dismissed: GUI state, in no file, link or undo step). Defaults stay the workbook example (decision 28); the dashboard's status line counts its failing checks live. The Size tab shows Required (a slider on metal.required_min_Nm), Hot-low (metal.torque_hot_low_Nm) with its live gap to Required (torque_gap_text) and Pull-out torque (nominal, at the live coupling.op_temp_C); with the end-effect banner showing, Hot-low and the gap read no prediction instead of a number. Under the controls, Consequences::of says what moves what: the neighbour magnet counts N-2 and N+2 inside the slider range (compute_all of the design with only coupling.npole changed; a neighbour that fails sizing::is_valid reads in words, never a number), that the ring size stays when the magnet count changes (RING_SIZE_STAYS), the smallest outside diameter that fits N magnets (engine::size_controls::smallest_fitting_diameter), the cap, cup and boss diameters that set the outside diameter and the magnets plus housing lengths of the overall length (the engine::size_controls breakdowns, reusing the expressions dimensions uses). A ConsequenceCache recomputes only when the inputs or the Minimum OD change."
```

**3.10** Lessons. Insert one bullet after the `zip 8's deflate feature` bullet (anchor: its last sentence `...and check that they still decode."`). The bullet is a new line at 2-space indent:

```text
  - "A GUI control that mirrors an engine input must edit that input, not keep a copy: the Torque -> Magnets target was separate GUI state beside metal.required_min_Nm, so a lower target showed a green Solved beside a red Below hot minimum for the same design (2026-10-13 audit, fixed by decision G-2)."
```

**3.11** The end of line 470, the `magcoupling` status:

```text
OLD: decisions X-1 to X-12); next: M3 (live 3D fields)
NEW: decisions X-1 to X-12); guidance slice 1 built on branch magcoupling/guidance (BL-055, decisions G-1 to G-8: one torque number, the Size tab as the first view with an Auto solver, check guides and what-moves-what lines, gui/guidance.rs); next: guidance slice 2 (the note rewrite and three new notes), the torque sensitivity card, then M3 (live 3D fields)
```

- [ ] **Step 4: Edit `docs/ai/03-structure.yaml`**

**4.1** Line 35, `role` (append inside the plain scalar; no colon followed by a space):

```text
OLD: the teaching notes and diagrams, the material, part and grade pickers and the material warnings
NEW: the teaching notes and diagrams, the material, part and grade pickers and the material warnings; guidance slice 1 (BL-055) adds gui/guidance.rs (check guides, design status, what-moves-what lines), the one torque number, the Size tab's Auto solver and the Size tab as the first view
```

**4.2** Line 50, `sizing`:

```text
OLD: FreeVariable (axial length, magnets per ring, ring radius), SCAN_CELLS 64, VALUE_TOLERANCE_MM 1e-9, is_valid (decision A2-4), solve -> Solved / NotReachable {best}: the scan refined at the first crossing, each peak and each validity edge; SizingError; the search's
NEW: FreeVariable (axial length, magnets per ring, ring radius; range, grid and grid_in(range), the scan's cells over a caller's range), SCAN_CELLS 64, VALUE_TOLERANCE_MM 1e-9, is_valid (decision A2-4), solve_in(inputs, variable, target, range) and solve (the same over variable.range()) -> Solved / NotReachable {best}: the scan refined at the first crossing, each peak and each validity edge; SizingError (InvalidRange {min, max} for a range with a non-finite end or min > max); the search's
```

**4.3** Line 51, `size_controls`:

```text
OLD: same-size no-op and finite/range refusal)
NEW: same-size no-op and finite/range refusal; slice 1 adds diameter_breakdown (DiameterBreakdown: cap, cup and boss mm, the largest being dimensions' outside diameter), length_breakdown (LengthBreakdown: magnets mm + housing mm = dimensions' overall length), smallest_fitting_diameter -> DiameterFit {At, AtMinimum, Nowhere} (bisection of model::blocks_fit to 1e-6 mm over the Size tab's diameter range) and ring_radius_range (that range as coupling.inner_back_apothem_mm, for ring-size Auto), each reusing dimensions' own expressions)
```

**4.4** Line 65, `features.gui`. These five substrings are unique, and the sixth (4.4.f) is the `guidance.rs` insertion point:

- 4.4.a `panel.rs`' views:

  ```text
  OLD: CentreView (Geometry, the default; Plot(PlotKind) x5; Clamp; Results)
  NEW: CentreView (SizeAndTorque, the first view since 2026-10-13 (G-5; it was Geometry, M42-2); Geometry; Plot(PlotKind) x5; Clamp; Results)
  ```

- 4.4.b `panel.rs`' controls:

  ```text
  OLD: sizing controls and run_sizing)
  NEW: the welcome strip (WELCOME, welcome_dismissed), the Key design group's Auto status line (AUTO_STATUS_START) and run_sizing)
  ```

- 4.4.c `dashboard.rs`:

  ```text
  OLD: dashboard_lines, end_effect_banner)
  NEW: dashboard_lines, end_effect_banner; the guidance drawn on it (slice 1): the design status line, the Magnets fit row (FIT_CHECK, drawn beside DASHBOARD, in no engine table), under a Bad or Caution row Problem, Why and Fix with fix links to focus_input, and a Why: link on every row (WHY_LINK, NO_NOTE_YET))
  ```

- 4.4.d `session.rs`:

  ```text
  OLD: session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number)
  NEW: session.rs (Design, design_to_json/design_from_json, encode/decode_share_payload, share_link, LoadError, PATH_MIGRATIONS, json_number, target_mismatch_notice: a file whose target_torque_Nm differs from its metal.required_min_Nm loads with the requirement)
  ```

- 4.4.e `guidance.rs` entry and the new `sizing.rs` description (one replacement, since the two entries are adjacent):

  ```text
  OLD: , sizing.rs (SizingMode, SizingState, variable_key/label, SizingRunner (update, solve_now), DEBOUNCE_S 0.25)
  NEW: , guidance.rs (pure, no egui, slice 1: CheckGuide {check, label, problem, why, fixes, learn}, CHECK_GUIDES (the seven dashboard checks and FIT_CHECK), guide, level, fix_text, Direction, Learn {Note, Assumption, None}, DesignStatus, design_status, design_status_text, Consequences::of with magnet_lines, diameter_line and length_line (neighbours N-2 and N+2, the fit, the diameter and length breakdowns), Neighbour, NeighbourOutcome, ConsequenceCache, RING_SIZE_STAYS, plain_text_violations: the banned-pattern scan every guidance test runs), sizing.rs (SizingMode, SizingState {mode, variable} (the target is metal.required_min_Nm, REQUIREMENT_INPUT), variable_key/label, SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), DEBOUNCE_S 0.25)
  ```

**4.5** Line 70, `manual_size_gui`:

```text
OLD: keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history
NEW: keeps live engine torque and geometry in the centre and all dashboard checks, and joins edits to existing history; since slice 1 it is the first view and the solver's home: the Required slider (metal.required_min_Nm), Hot-low and its gap (torque_gap_text), pullout_label, the Auto toggle (AUTO, AUTO_SUFFIX, variable_slider, change_text, SET_BY_AUTO: the matching slider disabled) and the consequence lines under each control (Consequences, one ConsequenceCache held by the panel)
```

**4.6** Line 72, `tests`:

```text
OLD: sizing.rs, size_controls.rs (manual envelope mapping, retained allowances, bounds and no-ops), explain.rs
NEW: sizing.rs (also solve_in, grid_in and InvalidRange), size_controls.rs (manual envelope mapping, retained allowances, bounds and no-ops; the breakdowns, smallest_fitting_diameter and ring_radius_range), explain.rs
```

- [ ] **Step 5: Edit `docs/ai/04-memory.yaml`**

**5.1** After the first `active_issues` bullet (anchor: `  - "Agentic loop queue: see docs/ai/backlog.yaml (open items tracked there, not here)"`), insert:

```text
  - "Magcoupler guidance slice 1 (spec docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md, branch magcoupling/guidance, BL-055; decisions G-1 to G-8 in 05-update-tracker): built. Before release (G-8): a fresh-context session-model physics review of every Why, Problem and consequence sentence (G-3), the whole-branch review and linkage-sim-rs/scripts/gate.sh to GATE PASS; then merge to main and push (the push deploys) and check the deploy run."
```

**5.2** Insert before the `NEXT (user request 2026-10-03): teach the physics better than equations alone.` bullet:

```text
  - "NEXT (owner order, 2026-10-13, G-1): after guidance slice 1, slice 2 rewrites the teaching notes (the owner finds them too dense, silent on what to change and missing topics) and adds three new ones, with the 87 empty help strings and the on-screen jargon (four names for one radius); then a torque sensitivity card; then guided lessons, plot captions and drawing titles. Guides for the results-table checks off the dashboard come after slice 1 too."
```

**5.3** End of that 2026-10-03 bullet:

```text
OLD: First slice: the torque chain plus the card"
NEW: First slice: the torque chain plus the card. Since 2026-10-13 the order is set by decision G-1 (the entry above): note rewrite, sensitivity card, then lessons."
```

**5.4** M41-4 line, with a marker that the target part is superseded (the existing text then continues with `(gui::session \"sizing\")`):

```text
OLD: design files and share links record the sizing mode, free variable and target torque
NEW: design files and share links record the sizing mode, free variable and target torque (since 2026-10-13, decision G-2, the target is the requirement metal.required_min_Nm; the file still writes it as target_torque_Nm)
```

**5.5** M4-2 line:

```text
OLD: the centre region's CentreView holds the geometry view (the default), the five plot tabs
NEW: the centre region's CentreView holds the geometry view (the default until 2026-10-13: the first view is now Size and torque, decision G-5), the five plot tabs
```

- [ ] **Step 6: Add the tracker entry to `docs/ai/05-update-tracker.md`**

Replace `## 2026-10-12 — Adjustable manual minimum OD (BL-054)` with the block below, followed by a blank line and the same heading:

```markdown
## 2026-10-13 — Magcoupler guidance slice 1: one torque number, Size-tab solver, check guides (BL-055, branch magcoupling/guidance)
- Why: the owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to shrink; it did not, and nothing on screen said why. A read-only audit the same day (14 agents; 143 of 144 findings survived adversarial verification; the result is in the session scratchpad, not the repo) found eight themes. This slice takes four: the size model is invisible, two unlinked torque targets, red checks give no cause or remedy, and the teaching notes are unreachable (in part). Audience: mechanically literate engineers who are not magnetics specialists. Spec `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`.
- Owner decisions (2026-10-13): **G-1** order: this slice, the note rewrite with three new notes, a torque sensitivity card, then guided lessons. **G-2** one torque number: the sizing target is `metal.required_min_Nm` (reverses A2-9 and M41-9 for the target only; the mode and free variable stay GUI sizing state). **G-3** every new physics sentence gets one fresh-context session-model physics review per slice; numbers in it come from results, never literals; no Draft/Reviewed gate for hints. **G-4** the solver moves to the Size tab, magnet length stays the default free variable, the outcome names what changed. **G-5** the first view is Size and torque with a dismissible one-line welcome strip (reverses M42-2); Explain stays off by default (A4, M43-2). **G-6** defaults stay the workbook example (decision 28); the screen labels it and counts its failing checks live. **G-7** an Auto toggle locks the matching Size-tab slider and re-solves after other edits (existing `SizingRunner`, debounce and file format). **G-8** release: merge to main and deploy once `gate.sh` reaches GATE PASS and the physics and whole-branch reviews pass.
- One number: `SizingState` is the mode and the free variable; the target is `metal.required_min_Nm` (`REQUIREMENT_INPUT`), edited by the Required slider as an ordinary design edit (undo, files, links). Files and links still write `target_torque_Nm` (equal to the requirement) and keep `DESIGN_VERSION`; a file whose target differs from its requirement loads with the requirement and a notice (`target_mismatch_notice`).
- Size tab (now the first view, under a dismissible welcome strip): Required, Hot-low with its live gap and "Pull-out torque (nominal, at T)"; "no prediction" while the end-effect banner shows. Under the controls: the neighbour magnet counts' pull-out and cup OD (from `compute_all`, in words when a neighbour does not fit), "Ring size stays the same when you change the magnet count.", the smallest outside diameter that fits, the cap, cup and boss that set the outside diameter, and the magnets plus housing lengths.
- Auto solver: the Auto toggle replaces the Key design group's mode switch (now a status line with a button to the Size tab); the matching slider is locked, other edits re-solve after the debounce, the outcome names the change; ring size searches the Size tab's diameter range (Minimum OD included). Opening the Size tab no longer forces forward mode.
- Dashboard: a status line ("Workbook example design: fails N of 8 checks" at the defaults, "Your design: ..." otherwise), a Magnets fit row, Problem, Why and Fix with fix links under each red or amber row, and a Why: link on every row (reviewed note, assumption, or "no note yet"). `gui/guidance.rs` holds the guides, the status and the consequence lines as pure functions.
- Engine additions (torque model untouched): `engine::size_controls` diameter and length breakdowns, `smallest_fitting_diameter` and `ring_radius_range`; `sizing::solve_in` over a caller's range (`solve` is `solve_in` over the variable's range), `FreeVariable::grid_in`, `SizingError::InvalidRange`.
- Unchanged: the engine's labels, help and verdict strings (workbook text lock, `tests/python_schema.rs`), the results table and exports, input metadata, the deviation registry and the file schema.
- Tests, written first and red before each change: guide coverage and real check paths; every fix path an input with a slider or choices; reviewed notes, existing assumptions and reasons for none; no banned pattern in any guidance or consequence text; Problem lines at the default and at a passing design; consequence numbers equal `compute_all` of the neighbour and follow an edit; `smallest_fitting_diameter` (fits at the result, not 1e-3 mm below, both range-end forms, flats and arcs); the Required slider, one undo, the load notice and file and link round trips; Auto (locked slider, re-solve after the debounce, idle frames edit nothing, not reachable, ring size below the old 9 mm floor with a lowered Minimum OD); first view, welcome strip and status counts; the owner's 8-magnet case; glyphs for every new string.
```

- [ ] **Step 7: Add BL-055 to `docs/ai/backlog.yaml`**

Insert above `- id: BL-054` (followed by a blank line and the `- id: BL-054` line):

```yaml
- id: BL-055
  title: "Magcoupler gives no cause, remedy or consequence on screen: two unlinked torque targets, an invisible size model, red checks without guidance"
  dimension: gui
  risk: physics
  evidence: "Read-only audit 2026-10-13 (14 agents; 143 of 144 findings survived adversarial verification; themes T1, T2, T3 and part of T5) after the owner went from 10 to 8 magnets, lowered the torque target and the coupling did not shrink. At the default ring, 10 -> 8 magnets takes pull-out from 2.69 to 1.97 N m (-27 %), grows the cup OD from 41.3 to 42.4 mm and leaves the outside diameter at 42.8 mm (the cap sets it); the Torque -> Magnets target was separate GUI state (gui/sizing.rs:85-105), so a lower target showed a green 'Solved' beside a red 'Below hot minimum'; the default design opens with three Fails whose one remedy cites 'Metal design C122' (engine/materials.rs:244); 15 of 17 dashboard rows read 'No teaching note'. Owner decisions G-1 to G-8 (docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md)."
  acceptance: "One torque number: the sizing target is metal.required_min_Nm (the Required slider edits it, one undo restores it, files and links round-trip, a file whose target differs uses the requirement with a notice); the panel opens on Size and torque with a dismissible welcome strip; the Size tab shows Required, Hot-low with its live gap and Pull-out torque (nominal, at T), and under the controls what moves what (neighbour magnet counts from compute_all, the smallest fitting diameter, the cap, cup and boss that set the outside diameter, the magnets plus housing lengths); the solver is an Auto toggle on the Size tab (matching slider locked, re-solve after the debounce, ring size searching the Minimum OD range, opening the tab no longer forces forward mode); the dashboard has a design status line, a Magnets fit row, Problem, Why and Fix under red and amber rows and a Why: link on every row; no guidance text matches the banned patterns and every number comes from results; idle frames edit no input; engine verdict strings unchanged (tests/python_schema.rs); fresh-context physics review of every new physics sentence (G-3), whole-branch review and full GATE PASS"
  priority: 2
  status: fixed
  notes: "GUI-side layer plus engine helpers (engine::size_controls breakdowns, smallest_fitting_diameter and ring_radius_range; sizing::solve_in); torque model, deviations, input metadata, DESIGN_VERSION and the file schema unchanged. Reverses A2-9 and M41-9 (the target only) and M42-2. Out of scope, later slices: note rewrites and three new notes, 87 empty help strings, on-screen jargon (four names for one radius), the torque sensitivity card, guided lessons, plot captions, drawing titles, guides for results-table checks off the dashboard. Release (G-8): merge to main and push after the physics review, the whole-branch review and GATE PASS."
```

`risk: physics` is set because the slice writes physics sentences (G-3). That keeps any rework on the session model per CLAUDE.md section 5. Change it to `mechanical` only if the owner prefers; BL-053 and BL-054 were mechanical because they added no explanatory text.

- [ ] **Step 8: Edit `magcoupling-rs/README.md`**

**8.1** Status paragraph, line 36 and line 44:

```text
OLD: the geometry view to scale (the default: both rings, the housing in effect, the dimension
NEW: the geometry view to scale (the first view until 2026-10-13: both rings, the housing in effect, the dimension
```

```text
OLD: (decisions O-1 to O-8). Next: M3 (live 3D fields).
NEW: (decisions O-1 to O-8). Guidance slice 1 complete (spec `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`, BL-055, decisions G-1 to G-8): one torque number, the Size and torque tab as the first view with an Auto solver, check guides on the dashboard and what-moves-what lines (see [The panel](#the-panel)). Next: guidance slice 2 (the note rewrite), then M3 (live 3D fields).
```

**8.2** Layout table, line 147, the `sizing.rs` row (two replacements within the row):

```text
OLD: `solve` (a coarse scan of `SCAN_CELLS` cells, refined
NEW: `solve_in` (the search over a range the caller gives, its `FreeVariable::grid_in(range)` the scan's grid; `solve` is `solve_in` over the variable's own slider range, and the Size tab's ring-size Auto passes its diameter range as an apothem range: a coarse scan of `SCAN_CELLS` cells, refined
```

```text
OLD: `SizingOutcome`, `SizingError` |
NEW: `SizingOutcome`, `SizingError` (with `InvalidRange`: a range with a non-finite end or min > max) |
```

**8.3** New layout row for `size_controls.rs`, which has none today. Insert before `| \`src/engine/assumptions.rs\` |`:

```markdown
| `src/engine/size_controls.rs` | The Size and torque tab's engine side: `dimensions` (outside diameter, axial stack and metadata-derived ranges), `resize_diameter` and `resize_length` (with the explicit-minimum variants), no-ops and refusals; slice 1 adds `diameter_breakdown` (cap, cup and boss; the largest is the outside diameter), `length_breakdown` (magnets + housing = the overall length), `smallest_fitting_diameter` (`DiameterFit`: `At`, `AtMinimum`, `Nowhere`; bisection of `model::blocks_fit` to 1e-6 mm over the diameter range) and `ring_radius_range` (that range as `coupling.inner_back_apothem_mm`, for ring-size Auto). The torque model is untouched |
```

**8.4** `src/gui/` row, line 150 (three replacements within the row):

```text
OLD: the assumptions banner, the sizing controls)
NEW: the assumptions banner, the welcome strip, the sizing controls)
```

```text
OLD: `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`)
NEW: `session.rs` (design files and share links: `Design`, `design_to_json`, `design_from_json`, `encode_share_payload`, `decode_share_payload`, `LoadError`, `target_mismatch_notice`)
```

```text
OLD: `sizing.rs` (`SizingMode`, `SizingState`, `SizingRunner`: debounced inverse sizing)
NEW: `guidance.rs` (the guidance layer, pure functions: `CHECK_GUIDES`, `design_status`, `Consequences`, `ConsequenceCache`, `plain_text_violations`), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle and consequence lines), `sizing.rs` (`SizingMode`, `SizingState` (the mode and the free variable; the target is `metal.required_min_Nm`), `SizingRunner`: debounced inverse sizing over a search range)
```

**8.5** The panel section, lines 170-188. Replace the first line, the last two lines of the first paragraph, and the Minimum OD ending, then insert three paragraphs after the first paragraph.

```text
OLD: For manual size exploration, open **Size and torque**. Adjust **Outside diameter**,
NEW: **Size and torque** is the first view (decision G-5; it replaced the geometry view, M42-2).
A one-line strip over the tabs, "Set the size -> read the torque -> open any red check to see
why and what to change.", closes for the session (it is in no file, link or undo step). Adjust **Outside diameter**,
```

```text
OLD: exports remain available. Opening this tab switches to forward calculation, keeping any
inverse-sized design; direct edits use the existing undo/redo history and file/share format.
NEW: exports remain available. Direct edits use the existing undo/redo history and file/share
format. Opening this tab changes nothing about the sizing mode.

**Required** is a slider on `metal.required_min_Nm`, the requirement the dashboard's checks
compare against (decision G-2: there is one torque number; the Torque -> Magnets target was a
second one until 2026-10-13). Editing it is an ordinary design edit: undo, files and share
links carry it. **Hot-low** (`metal.torque_hot_low_Nm`) is shown with its gap to Required,
"short X N·m (-P %)" in red or "margin +X N·m (+P %)" in green, and **Pull-out torque
(nominal, at T)** is the nominal figure at the live operating temperature. While the
end-effect banner shows, Hot-low and the gap read "no prediction (magnets too short for the
end-effect model)" instead of a number.

Under the controls, short lines say what moves what, each read from the engine (the neighbour
magnet counts from `compute_all` of the design with only `coupling.npole` changed; the
diameters and lengths from `engine::size_controls`), never typed. For **Magnets per ring**:
"Ring size stays the same when you change the magnet count.", the pull-out and cup OD at N-2
and N+2 inside the slider range (a neighbour whose blocks do not fit, or whose end-effect model
is out of range, reads in words), and the smallest outside diameter that fits N magnets. For
**Outside diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside
diameter. For **Overall length**: "Magnets L mm + housing H mm".

**Auto: smallest [variable] that meets the requirement** is inverse sizing (see Sizing below),
moved to this tab (decisions G-4, G-7). Choose magnet length (the default), ring size or magnets
per ring; the matching slider (Overall length, Outside diameter, Magnets per ring) is locked at
the solved or best value, with "Set by Auto; turn Auto off to edit" on hover, other edits
re-solve after the debounce, and the outcome line names the change (for example "Magnet length
12.7 -> 9.8 mm") or the best value when the requirement is not reachable. Ring size searches the
range the diameter slider has, Minimum OD included.
```

```text
OLD: saved diameter. Reset all restores the original range. Inverse sizing retains its bounds.
NEW: saved diameter. Reset all restores the original range. Auto's ring-size search uses the
range the diameter slider shows; the other inverse-sizing variables keep their bounds.
```

**8.6** Centre-region sentence, lines 194-195 (two lines, as in the file):

```text
OLD: centre region between them (`CentreView`, one tab row: the geometry view, the five plots, the
clamp and the results table; decisions M42-1 and M42-2), under the end-effect banner when
NEW: centre region between them (`CentreView`, one tab row: Size and torque, the first view since
2026-10-13 (G-5), the geometry view, the five plots, the clamp and the results table;
decisions M42-1 and M42-2), under the end-effect banner when
```

**8.7** Geometry bullet:

```text
OLD: (`geometry.rs`, `geometry_view.rs`; the default view, M42-2)
NEW: (`geometry.rs`, `geometry_view.rs`; the Geometry tab, the first view until 2026-10-13, M42-2; the Size tab draws it too)
```

**8.8** Insert a new bullet before `- **Results table** (\`results_table.rs\`)`:

```markdown
- **Check guides** (`guidance.rs`, pure functions; decisions G-3, G-6): a status line tops the
  dashboard, "Workbook example design: fails N of 8 checks" while the inputs are the defaults,
  "Your design: fails N of 8 checks" or "passes all 8 checks" after an edit, with ", N cautions"
  added when amber rows exist (counted live from the guides). The eight guides are the
  dashboard's seven checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
  the recommended clamp, the space claim and the end-effect flag) and a **Magnets fit** row
  (`model::blocks_fit`, drawn beside `DASHBOARD`: it has no engine result of its own). Under a
  red or amber row a block gives the **Problem** (built from the live results), a one-sentence
  **Why** and a **Fix**: one or more inputs by their on-screen label and direction (for example
  "Raise Minimum outer return ring wall"), each a link to that input's row. Every row, whatever
  its level, has a **Why:** link: a reviewed teaching note, an assumption's rationale, or "no
  note yet". The guide is a GUI layer: the engine's verdict strings, the results table and the
  exports keep the workbook wording (`tests/python_schema.rs`). Every Why and Problem sentence
  passes a physics review (G-3), and the numbers in them always come from the results.
```

**8.9** Sizing bullet, lines 423-430. Replace the whole bullet:

```text
OLD:
- **Sizing** (`sizing.rs`, Addendum A1): the mode switch tops the Key design group. In
  Torque -> Magnets the free variable (axial length, magnets per ring, ring radius) and the target
  hot-low torque (2.5 N·m to start, M41-9) feed `sizing::solve`, which runs once the design has
  been still for 0.25 s and no edit is in progress, never per frame. The panel then shows the
  design with the free variable at the solved value, "Solved at X", or at the best value it
  found, "Not reachable (best Y at X)" (M41-8); the free variable's row shows that value, locked
  (under the status line when the Key design group does not list it: the ring radius). Leaving the
  mode keeps the value, solving first a change still waiting for its debounce (M41-7).
NEW:
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1): Auto is the
  Torque -> Magnets mode. The free variable (axial length, magnets per ring, ring radius) and
  the requirement `metal.required_min_Nm` (decision G-2: the one torque number; 2.5 N·m, the
  workbook's hot minimum, to start) feed `sizing::solve_in`, which runs once the design has
  been still for 0.25 s and no edit is in progress, never per frame. The panel then shows the
  design with the free variable at the solved value, "Solved at X", or at the best value it
  found, "Not reachable (best Y at X)" (M41-8); that variable's Size-tab slider shows the value,
  locked. The Key design group's mode switch is now a status line, "Auto: <variable> set by the
  solver (Size tab)", with a button to the Size tab. Leaving Auto keeps the value, solving first
  a change still waiting for its debounce (M41-7). A file or link still records
  `sizing.target_torque_Nm`, equal to the requirement; if a file's target differs from its
  `metal.required_min_Nm` the requirement is used and the status line says so.
```

**8.10** Smoke paragraph, lines 459-460 (two lines):

```text
OLD: `/tools/magcoupler/` through a pinned share link and checks the canvas, the geometry view in the first
screenshot (the default view: no click), the console lines
NEW: `/tools/magcoupler/` through a pinned share link and checks the canvas, the Size and torque tab with its
geometry drawing in the first screenshot (the first view: no click), the console lines
```

**8.11** Regeneration paragraph, line 540:

```text
OLD: Torque -> Magnets on the axial length at 2.5 N·m) when the design format changes
NEW: Auto on the axial length with the requirement at its default 2.5 N·m, which `target_torque_Nm` equals) when the design format changes
```

**8.12** Tests table:

- Row `tests/sizing.rs` (last words):

  ```text
  OLD: and a NaN override reads unknown. |
  NEW: and a NaN override reads unknown; slice 1: `solve_in` over a caller's range equals `solve` over the variable's own range (same results), `grid_in` is the scan's grid, and a range with a non-finite end or min > max is `SizingError::InvalidRange`. |
  ```

- New row before `| \`tests/explain.rs\` |`:

  ```markdown
  | `tests/size_controls.rs` | The Size tab's engine adapter: the envelope mapping to existing inputs, retained allowances, bounds, the Minimum OD variants, no-ops and refusals; slice 1: the diameter and length breakdowns equal what `dimensions` reports (the largest of cap, cup and boss is the outside diameter; magnets + housing is the overall length), `smallest_fitting_diameter` fits at its result and not 1e-3 mm below (flats and arcs, the predicate monotone in the radius) and reads `AtMinimum` and `Nowhere` at the range ends, and `ring_radius_range` is the diameter range as an apothem range. |
  ```

- Row `src/gui/panel.rs` (last words):

  ```text
  OLD: holds the solved length (noted) and the solve's outcome |
  NEW: holds the solved length (noted) and the solve's outcome; slice 1 (BL-055): the panel opens on Size and torque under the welcome strip, which closes for the session and leaves no undo step; the dashboard's status line counts the failing checks at the default design and after a fix; the Required slider edits `metal.required_min_Nm` (one undo restores it) and the Hot-low gap follows it; the consequence lines follow an input edit; Auto locks the matching slider, re-solves after the debounce, leaves idle frames alone and reports an unreachable requirement; opening the Size tab no longer changes the sizing mode; every new string has glyphs in egui's default fonts |
  ```

  In the same row, `the geometry view is the default and follows the design shown` becomes `the Geometry tab follows the design shown`.

- Row `src/gui/session.rs`, `history.rs`, `sizing.rs` (last words):

  ```text
  OLD: and their status lines |
  NEW: and their status lines; slice 1 (BL-055): the sizing state holds only the mode and the free variable, the target being `metal.required_min_Nm`; a file's `target_torque_Nm` that differs from its requirement loads with the requirement and `target_mismatch_notice`; file and link round trips stay exact; the runner keys on the inputs and the search range |
  ```

- New row before `| \`src/gui/format.rs\` (feature \`gui\`) |`:

  ```markdown
  | `src/gui/guidance.rs` (feature `gui`) | `CHECK_GUIDES` covers every dashboard check and the Magnets fit row, and each guide names a real check; every fix path is an input with a slider or choices; every `Learn::Note` is a reviewed note, every `Learn::Assumption` exists and every `Learn::None` states its reason; no guide or consequence text matches `plain_text_violations` (cell references, correction ids, "workbook" except "Workbook example", "Addendum", "Rust-only", "Calculator!"); each Problem line at the default design and none at a passing design; the consequence numbers equal `compute_all` of the neighbour design and follow an input edit; neighbours outside 4 to 40 magnets are not shown; the design status counts at the default and after a fix. |
  ```

- [ ] **Step 9: Keep the smoke prompt true (the only non-doc edit)**

Edit `.claude/workflows/gui-smoke.js`:

```text
OLD (lines 73-74): // the console (the page is one canvas). The geometry view is the centre region's default view,
// so the first screenshot shows it with no click. One click on the canvas, on the "Load design"
NEW: // the console (the page is one canvas). Size and torque is the centre region's first view (it
// draws the geometry view), so the first screenshot shows it with no click. One click on the canvas, on the "Load design"
```

```text
OLD (line 82, from "(inputs on the left" through the callout list): (inputs on the left, dashboard on the right, the geometry view in the middle) vs a blank page. geometry_view=true only if the middle region shows the geometry view: a row of tabs starting "Geometry", under it two drawings side by side (an end view of two rings of coloured magnet blocks inside a grey cup around a shaft, and a half side view with dashed space-claim lines), and a numbered list of dimension callouts under them ("1 Face gap ...", "4 Overall length: ...").
NEW: (the size controls on the left, the dashboard on the right, the Size and torque tab in the middle) vs a blank page. geometry_view=true only if the middle region shows the Size and torque tab with its geometry drawing: a row of tabs starting "Size and torque" then "Geometry", under it the torque readouts (Required, Hot-low, Pull-out torque), then two drawings side by side (an end view of two rings of coloured magnet blocks inside a grey cup around a shaft, and a half side view with dashed space-claim lines), and, if there is room, a numbered list of dimension callouts under them ("1 Face gap ...", "4 Overall length: ...").
```

Keep the schema key `geometry_view` (renaming it would ripple through the schema and the result handling).

- [ ] **Step 10: Run the docs check, the YAML parse and the smoke-link test**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation
# re-run the step 2 script: expected "docs check: PASS", exit 0, no MISSING/STALE lines
python -c "
import yaml
for f in ['docs/ai/03-structure.yaml','docs/ai/04-memory.yaml','docs/ai/backlog.yaml']:
    d = yaml.safe_load(open(f, encoding='utf-8')); print('OK', f, type(d).__name__, len(d))
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
assert b[0]['id'] == 'BL-055' and b[0]['status'] == 'fixed' and b[1]['id'] == 'BL-054'
print('backlog head OK')
"
git diff --check
git diff --stat
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design
```

Expected:
- `docs check: PASS`.
- Three `OK` lines (dict 8, dict 3, list 55) and `backlog head OK`.
- `git diff --check` prints nothing.
- `git diff --stat` lists exactly the seven files from the commit step.
- The smoke-link test reports `1 passed`. It reads `gui-smoke.js`, so this confirms the constants and log-line strings are intact. If it fails with a compile error on `target_Nm`, that is T3's edit, not this task's (see the list at the end).

Wider suite: no Rust file changed in this task, so no engine or panel suite is affected. The release gate (`bash linkage-sim-rs/scripts/gate.sh`) is the controller's step before G-8.

- [ ] **Step 11: Commit the documentation**

```bash
cd C:/Users/Cole/source/repos/linkage_simulation
git add docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml magcoupling-rs/README.md .claude/workflows/gui-smoke.js
git commit -m "$(cat <<'EOF'
docs(magcoupler): record guidance slice 1 (one torque number, Size-tab Auto solver, guidance layer, BL-055)

02-system: the target is metal.required_min_Nm (G-2, reverses A2-9/M41-9 for the target),
the Size tab is the first view (G-5, reverses M42-2) and the solver's home (no forced
forward mode), the guidance layer, solve_in and the size_controls helpers.
03-structure, 04-memory, the 2026-10-13 tracker entry, BL-055 and the README follow;
gui-smoke.js describes the Size tab as the first screenshot.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
)"
```

- [ ] **Step 12: After the physics review, the whole-branch review and GATE PASS (before the merge), record the result**

Run this only after both reviews pass and `bash linkage-sim-rs/scripts/gate.sh` prints GATE PASS.

Append to the end of the 2026-10-13 tracker entry (after its `Tests, written first` bullet):

```markdown
- Reviews and gate: the fresh-context physics review of every new Why, Problem and consequence sentence (G-3) and the whole-branch review passed; the full required gate ended **GATE PASS**.
```

In `docs/ai/backlog.yaml`, BL-055 `notes`:

```text
OLD: Release (G-8): merge to main and push after the physics review, the whole-branch review and GATE PASS."
NEW: The fresh-context physics review (G-3), the whole-branch review and the full gate (GATE PASS) passed; released per G-8."
```

Then:

```bash
git add docs/ai/05-update-tracker.md docs/ai/backlog.yaml
git commit -m "$(cat <<'EOF'
docs(magcoupler): record the guidance slice 1 reviews and gate result

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
)"
```

- [ ] **Step 13: After the merge to main, the push and a passing deploy run (G-8), close the open item**

Run this only once the deploy run has passed and `https://colesorkness.com/tools/magcoupler/` opens on Size and torque.

Append to the tracker entry:

```markdown
- Released (G-8): merged to main and pushed; the deploy run passed and colesorkness.com/tools/magcoupler/ opens on Size and torque.
```

In `docs/ai/04-memory.yaml`, delete the whole `  - "Magcoupler guidance slice 1 (spec ...` bullet from `active_issues`, since it is no longer open. Keep the `NEXT (owner order, 2026-10-13, G-1)` bullet in `open_questions`. Then:

```bash
python -c "import yaml; yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8')); print('OK')"
git add docs/ai/05-update-tracker.md docs/ai/04-memory.yaml
git commit -m "$(cat <<'EOF'
docs(magcoupler): record the guidance slice 1 release

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LXAocUbR8kmdqFDmf4jrQ4
EOF
)"
```

---

**Existing tests this task will break:** none. No Rust file changes. The one test that reads a file this task edits is `app.rs::the_smoke_test_share_link_opens_its_design`. It reads `gui-smoke.js` through `include_str!`, and step 9 leaves `MAGCOUPLING_SMOKE_PAYLOAD`, `MAGCOUPLING_SMOKE_DESIGN_FILE` and the three log-line strings unchanged, so it stays green.

**Heads-up for T3 (not caused by T10).** The same test builds `SizingState { mode: ..., variable: ..., target_Nm: 2.5 }` at `magcoupling-rs/src/app.rs` lines 219-223. T3 removes the field, so that literal stops compiling. The edit is to delete the `target_Nm: 2.5,` line. `assert_eq!(design, want)` still holds because the pinned link's `target_torque_Nm` of 2.5 equals the default requirement, so no mismatch notice applies.

**Facts these docs assert that belong to other tasks.** Step 1 checks all of them.
- The requirement wording "The requirement must be a positive torque." and the `Sizing refused: the search range is empty` wording (T2, T3).
- `SOLVED_PREFIX` and the `Solved at` log line survive (the smoke test needs them).
- The Magnets fit row is drawn beside `DASHBOARD` and not in it (T8).
- The Key design status line starts with `Auto: ` (T7).

---

### Task 11: Physics review, whole-branch review, gate and release (spec G-3, G-8, section 7)

**Files:**
- Test: `magcoupling-rs/src/gui/guidance.rs`. Add the test `guidance_strings_for_review` (`#[ignore]`) to the tests module.
- Modify: whatever the reviews require.

**Interfaces:**
- Consumes: everything from T1 to T10.
- Produces: a merged, deployed `main`.

- [ ] **Step 1: Generate every guidance string.** Add the ignored test `guidance_strings_for_review`.
  - It prints every string the user can see from this slice:
    - each `CHECK_GUIDES` `why`, and its `fixes(..)` rendered through `fix_text`;
    - the Problem output for every T4 fixture design and `passing_design()`;
    - the `ROW_LEARN` reasons;
    - `RING_SIZE_STAYS`, plus every `magnet_lines`, `diameter_line` and `length_line` form at 4, 8, 10 and 40 magnets and at Minimum OD unset and 37 mm;
    - the size-tab consts `NO_PREDICTION`, `NO_TORQUE`, `SET_BY_AUTO`, `AUTO`, `AUTO_SUFFIX` and `WELCOME`;
    - `change_text` for each variable;
    - `design_status_text` at default and passing;
    - `target_mismatch_notice(3.0, 2.5)`;
    - `REQUIREMENT_NOT_POSITIVE`.
  - Run it with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > guidance-strings.txt`, and keep the output outside the repo.
- [ ] **Step 2: Physics review (G-3).** A fresh-context reviewer on the session model reads the string dump, spec section 8 and the engine code behind each claim.
  - It reports wrong, misleading or trap statements, each with an exact rewrite.
  - Apply the rewrites, rerun the affected tests, and repeat until the reviewer passes the text.
- [ ] **Step 3: Whole-branch review.** A fresh-context reviewer reads `git diff main...HEAD` against the spec and this plan. It looks for correctness, missing edge cases, DRY and test strength. Fix the findings and re-review.
- [ ] **Step 4: Gate.** Run `bash linkage-sim-rs/scripts/gate.sh`. Expected: `GATE PASS`. Fix any failure and rerun.
- [ ] **Step 5: Record.** Add to BL-055's notes and the update tracker the review outcomes and the gate result.
- [ ] **Step 6: Release (G-8).**
  - `git checkout main && git merge --no-ff magcoupling/guidance`, with a merge message ending with the two trailer lines.
  - `git push origin main`.
  - Watch the "Deploy Web to Vercel" run with `gh run list --limit 1` and `gh run watch <id>`. Expected: success.

