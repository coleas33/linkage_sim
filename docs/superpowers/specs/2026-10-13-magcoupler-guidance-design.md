# Magcoupler guidance, slice 1: help where the user is looking

Date: 2026-10-13. Branch: `magcoupling/guidance`. Status: design approved in conversation; this
spec awaits the owner's review.

## 1. Why

The owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to
shrink. It did not, and nothing on screen said why. A read-only audit on 2026-10-13 (14 agents,
143 of 144 findings survived adversarial verification; the full result is in the session
scratchpad, `audit.json`) found eight themes. This slice addresses four of them:

- **T1, the size model is invisible.** The ring radius (`coupling.inner_back_apothem_mm`) is
  an independent input; the pole count only sets how many blocks go around it. At the default
  ring, 10 -> 8 magnets takes pull-out from 2.69 to 1.97 N m (-27 %), *grows* the cup OD from
  41.3 to 42.4 mm, and Outside diameter stays at 42.8 mm because the cap sets it. The only
  hover text is "Even; one block per pole."
- **T2, two unlinked torque targets.** The requirement (`metal.required_min_Nm`) colours the
  checks; the Torque -> Magnets target is separate GUI state (`gui/sizing.rs:85-105`). A lower
  target shows green "Solved" beside red "Below hot minimum".
- **T3, red checks give no cause or remedy.** The default design opens with three Fails; the
  one remedy cites "Metal design C122" (`engine/materials.rs:244`), not the on-screen input.
- **T5 (part), teaching is unreachable.** 15 of 17 dashboard rows read "No teaching note".

Audience (owner, 2026-10-13): mechanically literate engineers who are not magnetics
specialists. Owner's pain points: notes too dense, they do not say what to change, topics
missing. Order agreed: this slice (B), then the note rewrite and three new notes (A), then a
torque sensitivity card, then guided lessons.

## 2. Decisions (owner, 2026-10-13)

| Id | Decision |
|---|---|
| G-1 | Order: B (this slice) -> A (note rewrite + new notes) -> sensitivity card -> lessons. |
| G-2 | One torque number: the sizing target *is* `metal.required_min_Nm`. Reverses A2-9 and M41-9 for the target only (mode and free variable stay GUI sizing state). |
| G-3 | Every new physics sentence (Why, Problem templates, consequence wording) gets one fresh-context session-model physics review per slice. Numbers in it always come from results, never literals. No Draft/Reviewed gate for hints. |
| G-4 | The solver moves to the Size tab; magnet length stays the default free variable; the outcome names what changed. |
| G-5 | First view: Size and torque (reverses M42-2), with a dismissible one-line welcome strip. Explain stays off by default (A4, M43-2 unchanged). |
| G-6 | Defaults stay the workbook example (decision 28 unchanged); the screen labels it and counts its failing checks live. |
| G-7 | Solver UX: an Auto toggle that locks the matching Size-tab slider and re-solves after other edits (existing `SizingRunner`, debounce and file format). |
| G-8 | Release: merge to main and deploy once `gate.sh` reaches GATE PASS and the physics and whole-branch reviews pass. |

## 3. What the user sees

### 3.1 First view
- `MagcouplingPanel::new` opens `CentreView::SizeAndTorque`.
- A welcome strip above the centre tabs: "Set the size -> read the torque -> open any red
  check to see why and what to change." A close button hides it for the rest of the session
  (GUI state; not in files, links or undo).
- A design status line at the top of the dashboard:
  - inputs equal `DesignInputs::default()`: "Workbook example design: fails N of M checks";
  - otherwise "Your design: fails N of M checks" or "Your design: passes all M checks".
  - N and M are counted live from the check guides (3.4), Bad level only; Warn is counted
    separately ("2 cautions") when non-zero.

### 3.2 Torque readout (Size tab, centre)
- **Required**: a slider bound to `metal.required_min_Nm` (its own metadata range and step),
  an ordinary design edit (undo, files, links).
- **Hot-low**: `metal.torque_hot_low_Nm`, then the gap to Required, computed live:
  red "short X N m (-P %)" or green "margin +X N m (+P %)".
- **"Pull-out torque (nominal, at T)"** replaces "Expected torque", T being the live
  `coupling.op_temp_C`.
- With the end-effect banner showing, Hot-low and the gap read "no prediction (magnets too
  short for the end-effect model)" instead of a number.

### 3.3 "What moves what" lines (Size tab, under each control)
- **Magnets per ring** (N):
  - "Ring size stays the same when you change the magnet count." (always shown)
  - for each neighbour N-2 and N+2 inside the slider range: "At N': pull-out X N m (+-P %),
    cup OD Y mm" from `compute_all` of the design with only `coupling.npole` changed; a
    neighbour that fails `sizing::is_valid` reads "At N': magnets do not fit" or "no
    prediction" (end effect), never a number.
  - "Smallest outside diameter that fits N magnets: D mm" (3.5); edge forms: "fits down to D
    mm (the minimum)" and "does not fit at any diameter up to D mm".
- **Outside diameter**: "Set by: cap A, cup B, boss C mm" (the largest is the OD), from the
  engine breakdown (4.2).
- **Overall length**: "Magnets L mm + housing H mm".

### 3.4 Check guides (dashboard)
Every dashboard check (`dashboard::DASHBOARD` rows with a check: `metal.hot_min_check`,
`metal.clearance_check`, `materials.cup_wall_check`, `temperature.summary.verdict`,
`clamps.recommended`, `housing.space_claim_check`, `model.end_effect_check`) plus a new
**Magnets fit** row (`model::blocks_fit`; today only the Size tab says so and then points to
a dashboard check that does not exist).

- Under a Bad or Warn row, a three-line block:
  - **Problem**: built from live results ("Cup wall too thin: 1.9 mm, needs 2.4 mm").
  - **Why**: one plain sentence (reviewed text).
  - **Fix**: one or more inputs by on-screen label and direction ("Raise Minimum outer
    return ring wall"), each a link that calls `focus_input(path)`.
- Every row (any level) gets a **"Why:"** link: a reviewed note (`Explorer::open_note`), an
  assumption rationale, or an explicit "no note yet".
- The engine verdict strings, the results table and exports keep their workbook wording
  (locked by `tests/python_schema.rs`); the guide is a GUI-side layer.

### 3.5 Solver on the Size tab (G-4, G-7)
- "Auto: smallest [magnet length v] that meets the requirement" toggle. Choices map to the
  Size-tab sliders: magnet length -> Overall length, ring size -> Outside diameter, magnet
  count -> Magnets per ring.
- On: `SizingMode::TorqueToMagnets` with that free variable. The matching slider is disabled
  and shows the solved (or best) value with "set by Auto"; edits to other inputs re-solve after
  the existing debounce.
- Outcome line names the change: "Magnet length 12.7 -> 9.8 mm" (from the inputs' own value to
  the shown value), or red "Not reachable: best X N m at V".
- Ring size searches the Size tab's current diameter range (the Minimum OD preference
  included), mapped to `coupling.inner_back_apothem_mm` through `engine::size_controls`;
  `sizing::is_valid` still applies, so no layout that fails fit or keyway wall is accepted.
- Opening the Size tab no longer forces forward mode (`panel.rs:686-690` removed). The Key
  design group's mode switch is replaced by a status line ("Auto: magnet length set by the
  solver, Size tab") with a link to the Size tab, so the solver has one home.

## 4. How it is built

### 4.1 `gui/guidance.rs` (new; pure functions, no egui)
- `CheckGuide { check: &str, label: &str, problem: fn(&DesignResults, &DesignInputs) ->
  Option<String>, why: &str, fixes: &[(&str /*input path*/, Direction)], learn: Learn }`
  with `enum Learn { Note(&str), Assumption(&str), None(&str /*reason*/) }`.
- `CHECK_GUIDES: [CheckGuide; 8]`.
- Fix labels come from `InputCatalogue` at display time (one source); guidance text never
  types a label.
- `Consequences::of(inputs, results, minimum_diameter_mm)` builds 3.3. The neighbour
  `compute_all` runs (about 17 us each) are cached on the inputs (`PartialEq`) and recomputed
  only when the inputs change.
- `design_status(inputs, results) -> DesignStatus { bad, warn, total, is_default }`.

### 4.2 `engine/size_controls.rs` (two additions; torque model untouched)
- `diameter_breakdown(inputs) -> Option<DiameterBreakdown { cap_mm, cup_mm, boss_mm }>` and
  `length_breakdown(inputs) -> Option<LengthBreakdown { magnets_mm, housing_mm }>`, reusing
  the expressions `dimensions` already uses (no second formula).
- `smallest_fitting_diameter(inputs, npole, range) -> Fit { At(mm) | AtMinimum(mm) |
  Nowhere(max_mm) }`: bisection to 1e-6 mm of `blocks_fit` over the diameter range, a
  monotone predicate in the radius at fixed N (tested).

### 4.3 `engine/sizing.rs`
- `solve_in(inputs, variable, target, range)`: the existing search over a caller's range;
  `solve` becomes `solve_in(.., variable.range())`. Only the ring-size Auto passes the Size
  tab's range.

### 4.4 One torque number (G-2)
- `SizingState` loses `target_Nm`; the solver reads `inputs.metal.required_min_Nm`.
- `gui::session`: save still writes `"target_torque_Nm"` (equal to the requirement; schema and
  `DESIGN_VERSION` unchanged). Load validates it as before; when it differs from the file's
  `metal.required_min_Nm`, the requirement is used and the status shows "This file's sizing
  target (X) differed from its requirement (Y); the requirement is used."
- `SizingRunner` keys its cache on the inputs (which now include the target).

### 4.5 Panel changes (`gui/panel.rs`)
- Default view, welcome strip, design status line, Size-tab readouts and consequence lines,
  Auto toggle and locked slider, dashboard guide blocks and "Why:" links, Key design status
  line. The size-controls drawing stays in `gui/size_controls.rs`.

## 5. Edge cases
- Magnet count at 4 or 40: only the neighbour inside the range is shown.
- Neighbour invalid or not finite: words, not numbers (3.3).
- Hot-low not finite or end effect out of range: "no prediction" (3.2); the gap is not shown.
- Requirement <= 0: the existing refusal, worded "The requirement must be a positive torque."
- Auto not reachable: slider at the best value, red line; no value valid: "No valid value of
  ring size in range" and the slider keeps the inputs' value.
- Auto on and the user drags the locked slider: disabled; hover says "Set by Auto; turn Auto
  off to edit".
- Files: inverse-mode files open with Auto on (as today); a target/requirement mismatch is
  resolved as 4.4; round trips are exact.
- Narrow windows: all new lines wrap; glyphs pass `assert_glyphs` (ASCII `->`, `<=`; no lock
  emoji unless the default font has it).
- Idle frames never edit inputs (existing invariant `idle_frames_change_no_input`).

## 6. Tests (written first; each fails before its change)
1. `CHECK_GUIDES` covers every dashboard check and the fit row; every guide names a real check.
2. Every fix path is an input with a slider or choices; every `Learn::Note` is a reviewed note;
   every `Learn::Assumption` exists; every `Learn::None` has a reason.
3. No guidance or consequence text matches `C\d{2,3}`, `E\d{1,2}\b`, "workbook" (except the
   design status line's "Workbook example"), "Addendum", "Rust-only", "Calculator!".
4. Each Problem line at the default design and at a passing design (Bad -> None).
5. Consequence numbers equal `compute_all` of the neighbour design; they follow an input edit.
6. `smallest_fitting_diameter`: fits at the result, not 1e-3 mm below; both range-end forms;
   flat blocks and arcs.
7. One number: the target slider sets `metal.required_min_Nm`; one undo restores it; the load
   mismatch notice; save/load and share-link round trips.
8. Auto: slider locked; re-solve after debounce; idle frames do not edit; not reachable; ring
   size below the old 9 mm floor with a lowered Minimum OD.
9. First view: opens on Size; strip dismiss; status line counts at default and after a fix.
10. Regression for the owner's case: at 8 magnets the Size tab shows "Ring size stays the
    same..." and the Set-by breakdown with the cap as the largest.
11. Glyph test over every new string; the existing panel, session, sizing and size-control
    suites stay green.

## 7. Review and release
- Physics review (G-3): a fresh-context session-model reviewer reads every Why and Problem
  template and the consequence wording against the engine; the verification traps in 8 must
  not appear.
- Whole-branch review, then `linkage-sim-rs/scripts/gate.sh` to GATE PASS.
- Docs with the code: `docs/ai/02-system.yaml` (one-number target, Size-tab solver, default
  view, guidance layer), `03-structure.yaml` (`gui/guidance.rs`), `04-memory.yaml`,
  `05-update-tracker.md`, `backlog.yaml` (BL-055), `magcoupling-rs/README.md`.
- G-8: merge to main and push (deploys) after the gate and reviews pass; check the deploy run.

## 8. Physics traps (must not be written)
- Radius is the strongest lever at a fixed pole count (per mm, magnet length is stronger at
  the defaults).
- Fewer poles keep the same size (the cup grows) or always improve torque per mass.
- Pull-out torque is guaranteed; the rings re-lock as soon as the load falls below pull-out.
- The bench calibration applies beyond prototype-matching designs.
- 416 stainless is a safe back-iron fix (it trips the low-saturation caution).
- The stored 3D demagnetization fields follow geometry or grade.

## 9. Out of scope (later slices)
Note rewrites and the three new notes (slice 2, with the 87 empty help strings and on-screen
jargon such as four names for one radius); the torque sensitivity card (slice 3); guided
lessons, plot captions and drawing titles (later); guides for the results-table checks off the
dashboard.
