# BL-064 shaft joints: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dashboard checks both keyed shrink fits that hold the coupling's rotors on their shafts: the inner hub on the wheel shaft and the cup's boss on the gearbox output shaft. Each fit must hold the safety factor times cold-high at its loosest, stay elastic at its tightest, and be fittable within its allowed temperature, with the key checked alone as a backup. A guide and a reviewed teaching note explain the window of interference that works and the fitting order.

**Architecture:**
- **Engine:** a new Rust-only module, `magcoupling-rs/src/engine/shaft_joint.rs`, with its own input group (15) and result group (24). One pure per-joint function, called twice, covers the Lamé fit, the temperature terms, the NIST shaft shrink, the key and the fitting temperature. Two engine prerequisites come first: the inner ring's own single-ring onset (`temperature.demag.inner_onset_single_ring_C`) and one hub-material helper (`PartProperties::hub()`).
- **Explain layer:** 23 equation records, the assumption rows and a scope chain.
- **GUI:**
  - one counted dashboard row, before the gearbox row, greyed with the end-effect rows and labelled with the stored-3D rows;
  - its guide, with live numbers and per-cause fixes;
  - the inputs panel's new sections, the key drawn from the new width, and the exports.
- **Note:** `shaft_joint`, with a fresh-context physics review.
- **Unchanged:** no Python-locked row changes, the clamp section stays paused, and Auto is unchanged.

**Tech Stack:** Rust 1.89; egui 0.32; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`); `linkage-sim-rs/scripts/gate.sh`, where gate 12 runs the Python oracle from `reference/magcoupling-py/.venv`.

**Spec:** `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md` (owner-approved 2026-10-18, commit 0687034). The owner's decisions are in `.superpowers/sdd/2026-10-17-bl064-shaft-joint/owner-decisions.md`. The research, contract, drafts and critique are in the same gitignored folder.

**How this plan was made:**
1. A binding contract assigned the tasks.
2. Stage 1: the engine drafter built T1-T2 in a prototype, replayed byte for byte from the plan text. The docs and release drafter wrote T9-T10.
3. Stage 2: four drafters wrote T3-T8 on copies of that prototype.
4. A consistency and physics critic integrated T1-T7 into one prototype and ran it green (app lib 921). Its 18 amendments, and the controller's rulings below, are folded into the task text.

Provenance labels, used in the rulings and the Deviation index only (no step depends on them): the drafters are D-A (T1-T2, deviations DEV-1..5), D-B (T3, DV-B1..3), D-C (T4, DV-C1..8), D-D (T5-T6, D-1..9), D-E (T7-T8) and D-F (T9-T10). "Q<n>" after a drafter is its numbered open question; "D-D D-4" is D-D's deviation D-4. A-1 to A-18 are the critic's amendments, C-1 to C-9 the controller's rulings on them, PD-<n> the plan deviations.

Where a line number has moved because of an earlier task, trust the code and keep the intent.

## Global Constraints

- **Implementer model:** BL-064 is `risk: physics`, so every task and every review runs on the session model (CLAUDE.md section 5; precedence).
- **Commit trailers:** the two lines the executing session's attribution reminder gives, exactly as it gives them. At plan time they were:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```
  Every message file in this plan quotes these plan-time lines. If the executing session's reminder names another model or another session, write its two lines instead, in every message file. A trailer names the model and session that made the commit, never ones copied from this plan.
- **Commit messages:** written with the Write tool to `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T<n>.txt`, or `T<n>-fix-<k>.txt`, `plan.txt` (Before Task 1) and the names T7, T8 and T10 give. Then `git commit -F <file>`. Never pipe a heredoc.
- **Staging:** files by name only. Never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`. T8 Step 7 and T10 Step 3 name their files at run time, because the review decides what changes: list them from `git status --short`, one by one. T10's commits are Steps 3, 7 and 9. Its last step, Step 11, commits nothing: the branch delete and the auto-memory are outside the repo.
- **Build ledger:** the superpowers:subagent-driven-development skill's `progress.md`, in the workspace its `sdd-workspace` script derives from this plan's file name: `.superpowers/sdd/2026-10-18-bl064-shaft-joint/progress.md`. That is not the planning folder `.superpowers/sdd/2026-10-17-bl064-shaft-joint/` (`$W` in the steps), which holds the message files and the release tools. BL-059's plan and workspace shared one name; BL-064's do not. Each task review's deferred minor finding goes into the ledger as one line, `Task <N>: minor (deferred): <one-liner>`, the skill's form. A build under superpowers:executing-plans keeps the same file and lines by hand. T10 Step 8 reads them.
- **TDD:** red, then green. A task that adds tests only shows its red by planted faults, each undone afterwards.
- **Before each commit**, run all of these:
  - the touched suites;
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`;
  - the same without `--features app`;
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`.
- **Docs ship with the code,** in the task that makes them stale. T9 sweeps what is left and quotes texts read from the code, never drafts.
- **Never edit:**
  - Python-locked rows (`tests/python_schema.rs`), `reference/`, `tests/data/python_schema.json` or the differential files;
  - the committed note sign-off sheet.
- **`tests/data/input_schema.json`** is re-blessed (`MAGCOUPLING_BLESS=1 cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`) in these places, and nowhere else:
  - T2, for the 15 inputs;
  - T3, for the four assumption flags (exactly four lines);
  - T8 and T10's fix wave, whenever a `shaft_joint.*` input's label or help changes.

  Only `shaft_joint.*` paths may differ from `main`.
- **Gate 1** rewrites `docs/chebyshev_lambda/*.png`. Restore them with `git checkout -- docs/chebyshev_lambda`.
- **The clamp section stays paused** (`engine::clamps::PAUSED = true`, owner 6A):
  - no on-screen text says "clamp";
  - new paths contain neither "clamp" nor "gearbox";
  - the paused section's texts are not edited.
- **On-screen text** follows the spec's traps (section 6.2):
  - cold-high "can reach" the joints, never "strongest", "at most", "up to" or "most expected";
  - name the joint and its shaft;
  - nothing suggests an unsafe fitting temperature;
  - each temperature limit names its source;
  - "warm the part or cool the shaft", never "heat the hub";
  - plain words, glyph-safe, `plain_text_violations` empty.
- **Caps:**

  | Text | Limit |
  |---|---|
  | Help | at most 30 words, 1-2 sentences |
  | Why | one sentence, at most 30 words |
  | Note sentences | 3-5, each at most 25 words |
  | Watch-outs | at most 30 words |
- **Release:** STOP for the owner's review of the diff before any merge or push. A push to `main` deploys.

## Review Focus

These five inputs or conditions are the most likely to bite a user and have no line of their own in the spec. Each has its test in the task named.
1. **A typed value no slider reaches.**
   - A bore of 0 reads unknown, never a pass.
   - A key wider than the bore's circumference holds 0 N·m, with no window sentence.
   - A part diameter at or below zero reads "no wall" (unknown), never a pass (ruling C-3).
   - A safety factor at or below zero reads unknown, never a pass with negative key pressures (ruling C-4).

   T2, T5.
2. **Mixed magnet grades.** The bonded hub's cap reads the inner ring's own onset, whichever ring governs. A pin at inner N42H with outer `SmCo_2_17_30` asserts `demag_ring == "outer"` and the inner onset. T1, T2.
3. **A band set reversed, or a joint with no wall.** The other joint's red cause still shows and counts. The Problem lists the determinate joint first. T2, T5.
4. **The end-effect model out of range.** The row greys, uncounted. It also carries the stored-3D label. T5.
5. **A guide fix at its slider's end, or one step that would reverse the band.** It is held back, and the block shows the no-fix line. T5, T6.

## Controller rulings on the critique (binding; folded into the tasks)

- **C-1 (A-1 to A-4, must).** Adopted as written:
  - T7 does not redo T3's import;
  - T5's README Old text is taken after T4;
  - T3 edits the README chain list;
  - T2 edits the README "gearbox" filter clause.
- **C-2.** A-5, A-6, A-7, A-10, A-12, A-13, A-16, A-17 and A-18 are adopted as written.
- **C-3 (A-8).** A part diameter at or below zero, typed and reachable on no slider, reads "no wall" (unknown), never a judged wall. The test is `!(spec.outer_diameter_mm > d)` instead of `q >= 1.0`. The records and a test follow. This is plan deviation PD-1 from spec 3.1, which judged such a value. It handles a typed edge the spec did not name, and the owner confirms it in the plan review (Open owner questions, below).
- **C-4 (A-9, option A).** A safety factor at or below zero reads "Shaft joint check unknown: the safety factor is not above zero" (Caution, no fix, its Problem asks for at least 1). The G-3 review covers its text, and `tests/robustness.rs` pins 0 and -1. Plan deviation PD-2; the owner confirms it in the plan review (Open owner questions, below).
- **C-5 (A-11).** At a keyway depth of 0 or less, the drawing draws no key, because the check then has no key term. It is a local guard in `geometry.rs`, and the reviewed joint code is untouched. A NaN depth keeps its "not drawn" note. Plan deviation PD-3, **an owner decision in the plan review**: folded in as recommended. A decline is taken out before Task 1, as the Open owner questions below say.
- **C-6 (A-14, contract).** Task ownership follows the drafts:
  - the inputs-panel placement and `PACKAGE_LABELS` are T2's (DEV-1);
  - `CHAIN_LABELS` is T3's;
  - the XLSX Summary row and the eleven `every_fix_moves` cases are T5's (D-1, D-2);
  - T4 keeps the drawn key, the CSV/JSON/XLSX Inputs and Results sheets, the clamp-name guard, Auto, the `PAUSED` doc and `INERT_INPUTS`;
  - the re-bless rule is the one in the Global Constraints above.
- **C-7 (A-15).** This head is the plan head.
- **C-8. The drafters' questions:**
  - **Engine (D-A):**
    - its extra public surface is accepted (Q6): `Cause`, `JointTerms`, `joints`, `failing`, `verdict` and `api::shaft_joint_links`, so the guide reuses the engine's terms (DRY);
    - `tests/shaft_joint.rs` restates `passing_design()` as `passing()`, pinned by its own numbers (Q4);
    - the E18 hub-expansion rule in `temperature.rs:1069-1073` is recorded as a T9 observation (Q5).
  - **The check-name guard (D-D D-4).** Its name rule gains `name == "check"`, in T5's commit.
  - **Shared rounding helpers (D-D D-5).** `apart_from`, `rounded_up`, `rounded_down` and `shown_decimals` are adopted. The BL-059 tests that cover them pass unedited, and T10's reviewer checks it (A-17).
  - **The clamp-name guard (D-C Q3).** T5 adds `only_the_joint_fails()` and a hub maximum of 26 µm to its designs, and the guide's label and Why to its collected texts.
  - **The drafts' T8 item C5 (not ruling C-5), the paused clamp's key-width help, is declined.** Owner decision 6A keeps the paused section's texts unchanged, so T8 does not offer it.
  - **BL-072** stays its own item, tied to the owner's approved correction E24 and to BL-057 (D-F).
  - **The T7 physics review.** If it asks to change the engine model (R-1, R-2, R-3 or R-14), T7 stops and reports: that is the owner's call.
- **C-9. Backlog priorities** BL-069 3, BL-070 2, BL-071 2, BL-072 2, as drafted: the owner confirms or re-ranks them in the plan review.

## Plan deviations from the spec

- **PD-1 (C-3).** A non-positive part diameter reads "no wall".
- **PD-2 (C-4).** A safety factor at or below zero reads unknown.
- **PD-3 (C-5).** No key is drawn at a keyway depth of 0 or less.
- **The drafters' deviations** (DEV-1..5, DV-B1..3, DV-C1..8, D-1..9, D-E's and D-F's) are renumbered PD-4 to PD-43, each with its drafter label, where it lives and its ruling, in the section `## Deviation index` at the end of this plan. The per-task test counts are in `## Per-task test counts` after it, and where each amendment landed is in `## Amendment ledger`.

## Open owner questions (answered in the plan review, before Task 1)

1. **PD-1** (ruling C-3): a typed part diameter at or below zero reads "no wall" (unknown).
2. **PD-2** (ruling C-4): a typed safety factor at or below zero reads unknown.
3. **PD-3** (ruling C-5): no key is drawn at a keyway depth of 0 or less.
4. **The backlog priorities** (ruling C-9): BL-069 3, BL-070 2, BL-071 2, BL-072 2, as drafted.

The drafts' T8 item C5 is not open: it is declined (C-8, PD-31). PD-1 to PD-3 are folded into the tasks as recommended. A re-ranked priority is written as T9 Step 10 says: its `priority:` lines and Step 12's priority assertion.

**If the owner declines PD-1, PD-2 or PD-3,** the controller takes it out of this plan's text before the plan is committed (Before Task 1), never during a task. Every place the `## Amendment ledger` lists under its amendment goes back to the drafts' behaviour that `plan-critique.md` describes under the same amendment:
- **PD-1 (A-8):** `joint`'s no-wall test stays `if q >= 1.0 {`, and the records keep `[Q] >= 1`. The boss-at-−22 mm assertion and the A-8 doc and record lines go.
- **PD-2 (A-9):** the critique's option B. `UNKNOWN_SAFETY_FACTOR`, its Caution arm, its Problem sentence, its T8 strings rows and its doc lines go. The `for factor in [0.0, -1.0]` loop leaves `degenerate_shaft_joint_inputs_never_pass`, whose name it would then contradict, for a test of its own in `tests/robustness.rs` that pins the verdict the spec's code gives at 0 and −1 (the critique's probe read a pass). `robustness` then counts 14 from T2 on, and the Per-task test counts' T2 row says so.
- **PD-3 (A-11):** no guard in T4 Step 4, no keyway-0 check in T4 Step 2, and none of the A-11 words in T4's head, Files and Interfaces, README edit (f), `03-structure.yaml` edit (a), T4 Step 16's message or T9 Step 7's tracker.

Each Old/New pair a removal touches is then checked against the file it edits, and the PD line above is marked "declined by the owner". No other count moves: PD-1 and PD-3 add assertions to existing tests only.

## Before Task 1 (controller)

At session start the plan file is untracked, and every task's `git status --short` check expects only the owner's `?? docs/analyses/2026-05-28-press-4bar-analysis.md`. So the plan is committed first, as BL-059's was (819c040, its review recorded in 0ac7b30):

1. Fold the owner's answers to the Open owner questions into this plan's text.
2. Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/plan.txt` with the Write tool:
   ```text
   docs(magcoupler): BL-064 shaft joint implementation plan (T1-T10)

   <one paragraph: the ten tasks from the approved spec (0687034), the critique's 18
   amendments folded in, and the owner's answer to each open question: PD-1, PD-2 and
   PD-3 kept or declined, the BL-069 to BL-072 priorities>

   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
   ```
3. Commit it:
   ```bash
   git add docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md
   git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/plan.txt
   git status --short
   ```
   Expected: one commit of 1 file; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

During the build, progress and rulings go into the build ledger, not into this plan. A plan edit made anyway (a ticked checkbox included) is committed by name, with its own message file, before the next task's status check.

---

## Tasks

### Task 1: Engine prerequisites: the inner ring's own single-ring onset and the hub-material helper

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 4.1 (`temperature.rs`, `material_library.rs`), 4.5 (the onset record), 7 item 1 (the inner ring's onset result) and owner R-15 ring. Two engine prerequisites of the shaft joint check, each with its own tests, landing before the check itself:
- `temperature.demag.inner_onset_single_ring_C`: the inner ring's own single-ring onset (`inner.onsets[3]`), whichever ring governs, with its equation record in `records/demagnetization.rs` and its path in the `demagnetization` scope chain;
- `PartProperties::hub()`: the hub, cup and boss material in effect with the library yield of the metal in effect; the material warnings read it (a behaviour-preserving DRY refactor of `api.rs`), and Task 2 reads it for the joints.

Line numbers are at HEAD `0687034`; trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout (checked in the plan-time prototype).

**Files:**
- Modify: `magcoupling-rs/src/engine/temperature.rs` (the result, its value, two unit tests), `magcoupling-rs/src/engine/explain/records/demagnetization.rs` (the record), `magcoupling-rs/src/engine/explain/scope.rs` (the path and the module doc), `magcoupling-rs/src/engine/material_library.rs` (`HubMaterial`, `PartProperties::hub`, the `yield_MPa` doc), `magcoupling-rs/src/engine/api.rs` (the warnings read the helper)
- Modify (tests): `magcoupling-rs/tests/grades.rs`, `magcoupling-rs/tests/material_links.rs`, `magcoupling-rs/tests/explain.rs`, `magcoupling-rs/tests/schema.rs`, `magcoupling-rs/src/gui/trace.rs`
- Modify (docs): `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`
- Test: `magcoupling-rs/src/engine/temperature.rs` (`each_ring_shows_its_own_block` extended; new `the_inner_single_ring_onset_is_the_inner_block_s_whichever_ring_governs`); `magcoupling-rs/tests/grades.rs` (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round` extended; new `the_inner_ring_s_own_single_ring_onset_whichever_ring_governs`); `magcoupling-rs/tests/material_links.rs` (new `the_hub_material_follows_the_circuit_and_its_yield_the_metal_in_effect`, `the_expansion_warning_reads_the_hub_material_in_effect_unchanged`); `magcoupling-rs/tests/explain.rs` (a drift-guard edge point at mixed rings; the scope count); `magcoupling-rs/tests/schema.rs` (`engine_owned_help_is_plain_and_short`'s coverage list); `magcoupling-rs/src/gui/trace.rs` (the pinned totals move)

**Interfaces:**
- Consumes: `temperature::ring_demag`'s `RingDemag::onsets` (aligned, pull-out, skipping, single ring; the inner block built in `compute` at `temperature.rs:931`); `temperature::{RING_INNER, RING_OUTER}`; `material_library::{PartProperties, EngineProps, material, resolve}`; the 6061-T6 library record (`"6061_T6"`, yield 241 MPa).
- Produces:
  - `DemagResults::inner_onset_single_ring_C: f64`, path `temperature.demag.inner_onset_single_ring_C` (Rust-only, no cell, unit `°C`, label "Inner ring: single-ring onset"), right after `inner_magnet_limit_C`; its record (symbol `ϑ_{sr,i}`, `.corrected(&[E20])`); its path in the `demagnetization` scope chain right after `temperature.demag.onset_single_ring_C` (the scope 169 -> 170 paths).
  - `material_library::HubMaterial { props: EngineProps, yield_MPa: f64 }` and `PartProperties::hub(&self) -> HubMaterial`; in `api::compute` the local `let hub_material = parts.hub();` (Task 2 reads it for the joints' links).

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/engine/temperature.rs`, the test module: `each_ring_shows_its_own_block` gains the inner onset (spec 7.1: on these rings the NdFeB inner ring governs, so this alone would also hold for the governing block's onset); a new test on the swapped rings (built as `e20_checks_both_rings_each_side_from_the_weaker` builds them, `:2101`) is the pin that tells the inner ring's own onset from the governing one.

Old:
```rust
        assert!(mixed.inner_magnet_limit_C < mixed.outer_magnet_limit_C);
        assert_eq!(mixed.magnet_limit_C, mixed.inner_magnet_limit_C);
    }

```

New:
```rust
        assert!(mixed.inner_magnet_limit_C < mixed.outer_magnet_limit_C);
        assert_eq!(mixed.magnet_limit_C, mixed.inner_magnet_limit_C);
        // BL-064 (owner R-15 ring): the inner ring's own single-ring onset is the NdFeB-only
        // run's. The NdFeB ring governs here, so this alone would also hold for the governing
        // block's onset; the swapped rings below tell the two apart.
        assert_eq!(mixed.inner_onset_single_ring_C, ndfeb.onset_single_ring_C);
    }

    #[test]
    fn the_inner_single_ring_onset_is_the_inner_block_s_whichever_ring_governs() {
        // BL-064 (owner R-15 ring): the bonded inner hub's fitting cap reads the inner ring's
        // own single-ring onset, the fourth onset of the inner block. With the NdFeB ring
        // inside it governs, and the two onsets are one value bit for bit. Swapped (Y30 inside,
        // the NdFeB ring outside), the NdFeB ring governs: the governing onset is the NdFeB
        // ring's, while the inner ring's own is the ferrite's +inf (a positive beta never
        // reaches the knee on heating).
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let ndfeb = compute(&ti, &links(), e20).demag;
        let mixed = compute(&ti, &with_outer_of(links(), &ferrite_links()), e20).demag;
        assert_eq!(mixed.demag_ring, RING_INNER);
        assert_eq!(
            mixed.inner_onset_single_ring_C.to_bits(),
            mixed.onset_single_ring_C.to_bits()
        );
        let swapped = compute(&ti, &with_outer_of(ferrite_links(), &links()), e20).demag;
        assert_eq!(swapped.demag_ring, RING_OUTER);
        assert_eq!(swapped.onset_single_ring_C, ndfeb.onset_single_ring_C);
        assert!(swapped.onset_single_ring_C.is_finite());
        assert_eq!(swapped.inner_onset_single_ring_C, f64::INFINITY);
        // Without E20 the workbook checks the inner ring only: the inner ring governs either
        // way round, and the two onsets are one value.
        for k in [
            with_outer_of(links(), &ferrite_links()),
            with_outer_of(ferrite_links(), &links()),
        ] {
            let r = run(&ti, &k).demag;
            assert_eq!(r.demag_ring, RING_INNER);
            assert_eq!(
                r.inner_onset_single_ring_C.to_bits(),
                r.onset_single_ring_C.to_bits()
            );
        }
    }

```

(b) `magcoupling-rs/tests/grades.rs`: the mixed-rings test gains the inner onset either way round (the ferrite's +inf inside), and a new end-to-end test pins the inner ring's own onset through `compute_all`. The mixed-ring pins name the grade ids the test sets (`SmCo_2_17_30`, `SmCo_2_17_26`; spec 7.1, review 6 SF-2) and assert the ring that governs, so a mistyped grade id or part name (read as a gradeless magnet: "SmCo 2:17-30", the spec's prose name, gives the outer ring governing with 112.83 °C and the inner 153.28 °C) fails it. No geometry input moves the onset (spec 3.6 notes), so the defaults stand for `passing_design()` here; `tests/shaft_joint.rs` (Task 2) pins the same mixed rings at `passing_design()`.

Old:
```rust
        );
        assert_eq!(demag.alpha_br, ndfeb.0, "{label}");
        assert_eq!(
            (
```

New:
```rust
        );
        assert_eq!(demag.alpha_br, ndfeb.0, "{label}");
        // BL-064 (owner R-15 ring): the inner ring's own single-ring onset, whichever ring
        // governs: the NdFeB ring's inside, the ferrite's +inf inside (no knee on heating).
        let inner_onset = if y30_inner {
            f64::INFINITY
        } else {
            default_demag.onset_single_ring_C
        };
        assert_eq!(demag.inner_onset_single_ring_C, inner_onset, "{label}");
        assert_eq!(
            (
```

At the end of the file (the Old is its last lines), the new test:

Old:
```rust
            "{label}"
        );
    }
}

```

New:
```rust
            "{label}"
        );
    }
}

#[test]
fn the_inner_ring_s_own_single_ring_onset_whichever_ring_governs() {
    // BL-064 (owner R-15 ring): `temperature.demag.inner_onset_single_ring_C` is the inner
    // ring's own single-ring onset, whichever ring governs the demagnetization block (the ring
    // with the lower magnet limit, the inner on a tie). It reads the inner ring's block only:
    // the same as with the inner ring's part or grade on both rings, bit for bit. Where the
    // inner ring governs it is `onset_single_ring_C` bit for bit; where the outer governs the
    // two differ. Each pin asserts the ring that governs and both onsets, so a mistyped part
    // name or grade id (read as a magnet with no grade) fails it.
    use magcoupling::engine::explain::probes::probe_designs;
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    use magcoupling::{DesignInputs, compute_all};
    let library = |inner: &str, outer: &str| {
        let mut d = DesignInputs::default();
        d.coupling.magnets.part_inner = inner.to_owned();
        d.coupling.magnets.part_outer = outer.to_owned();
        d
    };
    let manual = |inner: &str, outer: &str| {
        let mut d = DesignInputs::default();
        let m = &mut d.coupling.magnets;
        m.part_inner.clear();
        m.part_outer.clear();
        m.grade_inner = inner.to_owned();
        m.grade_outer = outer.to_owned();
        d
    };
    // (design, the inner ring on both rings, the ring that governs, the governing onset, the
    // inner ring's own onset [°C], to 4 decimals).
    let cases = [
        (
            "the defaults (B842SH both)",
            DesignInputs::default(),
            library("B842SH", "B842SH"),
            RING_INNER,
            143.3565,
            143.3565,
        ),
        // The library's N42 block outside: its limit is lower, so it governs.
        (
            "inner B842SH, outer B842",
            library("B842SH", "B842"),
            library("B842SH", "B842SH"),
            RING_OUTER,
            69.8326,
            143.3565,
        ),
        (
            "inner B842, outer B842SH",
            library("B842", "B842SH"),
            library("B842", "B842"),
            RING_INNER,
            69.8326,
            69.8326,
        ),
        // SmCo 2:17-30 outside: its magnet limit (from the skipping field) is the lower, but its
        // single-ring onset the higher, so the governing onset is above the inner ring's own.
        (
            "manual N42H inside, SmCo_2_17_30 outside",
            manual("N42H", "SmCo_2_17_30"),
            manual("N42H", "N42H"),
            RING_OUTER,
            191.8772,
            112.8297,
        ),
        // SmCo 2:17-26 outside a manual N50M ring: the N50M ring governs.
        (
            "manual N50M inside, SmCo_2_17_26 outside",
            manual("N50M", "SmCo_2_17_26"),
            manual("N50M", "N50M"),
            RING_INNER,
            98.8029,
            98.8029,
        ),
    ];
    for (label, design, inner_on_both, ring, governing, inner) in cases {
        let demag = compute_all(&design).temperature.demag;
        assert_eq!(demag.demag_ring, ring, "{label}");
        assert!(
            (demag.onset_single_ring_C - governing).abs() < 5e-5,
            "{label}: {}",
            demag.onset_single_ring_C
        );
        assert!(
            (demag.inner_onset_single_ring_C - inner).abs() < 5e-5,
            "{label}: {}",
            demag.inner_onset_single_ring_C
        );
        let own = compute_all(&inner_on_both).temperature.demag;
        assert_eq!(
            own.demag_ring, RING_INNER,
            "{label}: a tie names the inner ring"
        );
        assert_eq!(
            demag.inner_onset_single_ring_C.to_bits(),
            own.onset_single_ring_C.to_bits(),
            "{label}: the inner ring's block alone"
        );
        assert_eq!(
            demag.inner_onset_single_ring_C.to_bits() == demag.onset_single_ring_C.to_bits(),
            ring == RING_INNER,
            "{label}"
        );
    }
    // The five probe designs: the inner ring governs at each, so the two are one value.
    for (name, design) in probe_designs() {
        let demag = compute_all(&design).temperature.demag;
        assert_eq!(demag.demag_ring, RING_INNER, "{name}");
        assert_eq!(
            demag.inner_onset_single_ring_C.to_bits(),
            demag.onset_single_ring_C.to_bits(),
            "{name}"
        );
    }
}

```

(c) `magcoupling-rs/tests/material_links.rs`: the helper's rule (spec 4.1, review SF-16) and the warnings' unchanged value, bit for bit.

Old:
```rust
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, SLEEVE_LINER_CHOICES, material,
};
use magcoupling::engine::materials::PartMaterialInputs;
```

New:
```rust
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, PartProperties, SLEEVE_LINER_CHOICES, material, resolve,
};
use magcoupling::engine::materials::PartMaterialInputs;
```

Old:
```rust
    assert!(failures.is_empty(), "{failures:#?}");
}

```

New:
```rust
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The parts' materials in effect for `inputs`, resolved as `api::compute` resolves them.
fn parts_of(inputs: &DesignInputs) -> PartProperties {
    resolve(
        &inputs.materials.parts,
        &inputs.materials.steel,
        inputs.coupling.backiron,
        &inputs.metal,
        &inputs.temperature.slip_loss,
        &inputs.temperature.thermal,
    )
}

/// The library yield strength of material `id` [MPa].
fn library_yield(id: &str) -> f64 {
    material(id)
        .and_then(|m| m.yield_MPa.value)
        .unwrap_or_else(|| panic!("{id} has a yield"))
}

#[test]
fn the_hub_material_follows_the_circuit_and_its_yield_the_metal_in_effect() {
    // BL-064 (spec section 4.1): the shaft joint check reads the hub, cup and boss material in
    // effect, `PartProperties::hub`: the steel values on the steel circuit, else the body's,
    // with the library yield of the metal in effect, not of the pick.
    use magcoupling::engine::temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA};
    // The default: the Materials inputs (12.3e-6 /°C, 205 GPa) and the library's 4140 yield.
    let default = parts_of(&DesignInputs::default()).hub();
    assert_eq!(
        (
            default.props.cte_per_C,
            default.props.modulus_GPa,
            default.yield_MPa
        ),
        (12.3e-6, 205.0, 415.0)
    );
    // Each pick: a ferromagnetic one keeps the steel circuit and gives the steel values; a
    // non-magnetic one (304, 6061) selects the free-space circuit and is the body. Either way
    // the yield is the pick's own.
    for (code, id) in BACK_IRON_CHOICES {
        let parts = parts_of(&with_back_iron(code));
        let hub = parts.hub();
        let ferromagnetic = material(id).expect(id).ferromagnetic;
        assert_eq!(parts.backiron, i64::from(ferromagnetic), "{id}");
        let props = if ferromagnetic {
            parts.steel
        } else {
            parts.body
        };
        assert_eq!(hub.props, props, "{id}");
        assert_eq!(hub.yield_MPa, library_yield(id), "{id}");
    }
    // No back iron (`coupling.backiron` 0): the body is the workbook's aluminium whatever a
    // ferromagnetic pick or a code outside the choices says, so the yield is 6061-T6's, not
    // the pick's; a non-magnetic pick is the body itself, with its own yield.
    for code in (1..=8).chain([99]) {
        let mut inputs = with_back_iron(code);
        inputs.coupling.backiron = 0;
        let parts = parts_of(&inputs);
        let hub = parts.hub();
        assert_eq!(hub.props, parts.body, "{code}");
        let id = match parts.back_iron.material {
            Some(m) if !m.ferromagnetic => m.id,
            _ => "6061_T6",
        };
        assert_eq!(hub.yield_MPa, library_yield(id), "{code}");
        if id == "6061_T6" {
            assert_eq!(
                (hub.props.cte_per_C, hub.props.modulus_GPa, hub.yield_MPa),
                (AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA, 241.0),
                "{code}"
            );
        }
    }
    // On the steel circuit a code outside the choices gives NaN values and a NaN yield (decision
    // D3: never another material), so the shaft joint check reads unknown.
    let unknown = parts_of(&with_back_iron(99)).hub();
    assert!(
        unknown.props.cte_per_C.is_nan()
            && unknown.props.modulus_GPa.is_nan()
            && unknown.yield_MPa.is_nan(),
        "{unknown:?}"
    );
}

#[test]
fn the_expansion_warning_reads_the_hub_material_in_effect_unchanged() {
    // BL-064: the material warnings read `PartProperties::hub` (one rule for the warnings and
    // the shaft joint check). The value is the one they read before, bit for bit (the steel's
    // expansion on the steel circuit, else the body's), and the expansion warning fires exactly
    // when it is further than the limit from the magnets'.
    use magcoupling::engine::warnings::CTE_MISMATCH_LIMIT_PER_C;
    let mut designs = Vec::new();
    for backiron in [1, 0] {
        for code in (1..=8).chain([99]) {
            let mut inputs = with_back_iron(code);
            inputs.coupling.backiron = backiron;
            designs.push((
                format!("back iron {code}, circuit input {backiron}"),
                inputs,
            ));
        }
    }
    let mut typed = DesignInputs::default();
    typed.materials.steel.cte_per_C = 16e-6;
    designs.push(("4140 expansion typed 16e-6".to_owned(), typed));
    let mut fired_count = 0;
    for (label, inputs) in designs {
        let parts = parts_of(&inputs);
        let before = if parts.backiron == 1 {
            parts.steel.cte_per_C
        } else {
            parts.body.cte_per_C
        };
        let hub_cte = parts.hub().props.cte_per_C;
        assert_eq!(hub_cte.to_bits(), before.to_bits(), "{label}");
        let mismatch = (hub_cte - inputs.temperature.mismatch.ndfeb_cte_per_C).abs()
            > CTE_MISMATCH_LIMIT_PER_C;
        let fires = fired(&inputs).contains(&"cte_mismatch_with_magnets");
        assert_eq!(fires, mismatch, "{label}");
        fired_count += usize::from(fires);
    }
    // Not vacuous: the aluminium and 304 bodies and the typed expansion fire it (9 with no
    // back iron, 304 and 6061 on the steel circuit input, the typed 4140).
    assert_eq!(fired_count, 12);
}

```

(d) `magcoupling-rs/tests/explain.rs`: a drift-guard edge point where the outer ring governs and the two onsets differ (spec 7.12's mixed-ring point for the inner onset's record; Task 3 need not add it), and the scope's union count (169 -> 170: the `demagnetization` chain gains the inner onset; Task 3 takes it to 193).

Old:
```rust
                default_num("temperature.thermal.steady_est_C"),
            )],
        ),
        // The clamp counts outside i64 (`saturation_artifact`), one point per kind, each
```

New:
```rust
                default_num("temperature.thermal.steady_est_C"),
            )],
        ),
        (
            // BL-064 (owner R-15 ring): the library's N42 block outside the default N42SH one.
            // The outer ring governs (its magnet limit is the lower), so the inner ring's own
            // single-ring onset (143.36 °C) differs from the governing one (69.83 °C): the
            // inner onset's record must read the inner block, not the governing one
            // (`tests/grades.rs` pins the ring and both onsets at this design).
            "inner B842SH, outer B842: the outer ring governs",
            vec![("coupling.magnets.part_outer", text("B842"))],
        ),
        // The clamp counts outside i64 (`saturation_artifact`), one point per kind, each
```

Old:
```rust
    assert_eq!(
        union.len(),
        169,
        "decision 31: the chains and the dashboard (report section 7), and plan A-3 decision G1's 10 new geometry callouts"
    );
}
```

New:
```rust
    assert_eq!(
        union.len(),
        170,
        "decision 31: the chains and the dashboard (report section 7), plan A-3 decision G1's 10 new geometry callouts, and BL-064's inner-ring single-ring onset"
    );
}
```

(e) `magcoupling-rs/tests/schema.rs`, `engine_owned_help_is_plain_and_short`: the not-vacuous list names the new Rust-only result, so its label and help are held to the plain-words guard and the 30-word, two-sentence cap.

Old:
```rust
        "housing.space_claim_check",
        "gearbox.cold_high_check",
        "metal.steel_cup_mass_g",
        "clamps.table[*].length_mm",
```

New:
```rust
        "housing.space_claim_check",
        "gearbox.cold_high_check",
        "temperature.demag.inner_onset_single_ring_C",
        "metal.steel_cup_mass_g",
        "clamps.table[*].length_mm",
```

- [ ] **Step 2: Run the new tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --no-fail-fast`

Expected: three targets do not compile (the field and the method do not exist yet); cargo stops before running any test:
```text
error[E0609]: no field `inner_onset_single_ring_C` on type `DemagResults`      (9 of these)
error[E0609]: no field `inner_onset_single_ring_C` on type `&DemagResults`
error[E0599]: no method named `hub` found for struct `PartProperties` in the current scope      (5 of these)
error: could not compile `magcoupling-rs` (lib test) due to 4 previous errors
error: could not compile `magcoupling-rs` (test "grades") due to 6 previous errors
error: could not compile `magcoupling-rs` (test "material_links") due to 5 previous errors
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain explained_chains_have_every_record_and_scope_paths_exist`

Expected:
```text
assertion `left == right` failed: decision 31: the chains and the dashboard (report section 7), plan A-3 decision G1's 10 new geometry callouts, and BL-064's inner-ring single-ring onset
  left: 169
 right: 170
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 57 filtered out
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema engine_owned_help_is_plain_and_short`

Expected:
```text
temperature.demag.inner_onset_single_ring_C is not engine-owned
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 10 filtered out
```

- [ ] **Step 3: The inner ring's own single-ring onset (`temperature.rs`)**

The result, right after `inner_magnet_limit_C` (label and help are drafts; the G-3 review of Task 8 sets the words), and its value, the inner block's fourth onset:

Old:
```rust
            inner_magnet_limit_C: f64 => out_rust_only("°C", "Inner ring: magnet design limit",
                "The inner ring's own limit, whichever ring governs: its onset while skipping minus the design margin, or with a positive beta its rating (no limit without one)."),
            inner_cold_limit_C: NumOrText => out_rust_only("°C", "Inner ring: cold magnet limit",
                "Positive beta only: the inner ring's cold onset while skipping plus the design margin; 'n/a' otherwise."),
```

New:
```rust
            inner_magnet_limit_C: f64 => out_rust_only("°C", "Inner ring: magnet design limit",
                "The inner ring's own limit, whichever ring governs: its onset while skipping minus the design margin, or with a positive beta its rating (no limit without one)."),
            inner_onset_single_ring_C: f64 => out_rust_only("°C", "Inner ring: single-ring onset",
                "The inner ring's own onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner hub's allowed fitting temperature stays below it."),
            inner_cold_limit_C: NumOrText => out_rust_only("°C", "Inner ring: cold magnet limit",
                "Positive beta only: the inner ring's cold onset while skipping plus the design margin; 'n/a' otherwise."),
```

Old:
```rust
        inner_beta_per_C: inner.beta,
        inner_magnet_limit_C: inner.mag_lim,
        inner_cold_limit_C: inner.cold_limit,
        outer_hcj20_kA_m: outer_block.hcj20,
```

New:
```rust
        inner_beta_per_C: inner.beta,
        inner_magnet_limit_C: inner.mag_lim,
        // BL-064 (owner R-15 ring): the inner ring's own single-ring onset, whichever ring
        // governs; the bonded inner hub's allowed fitting temperature reads it.
        inner_onset_single_ring_C: inner.onsets[3],
        inner_cold_limit_C: inner.cold_limit,
        outer_hcj20_kA_m: outer_block.hcj20,
```

The help names the bonded inner hub's allowed fitting temperature, which Task 2 adds; only the merge ships, and this commit's message says so.

- [ ] **Step 4: Its record and its scope path**

`magcoupling-rs/src/engine/explain/records/demagnetization.rs`, right after `temperature.demag.inner_magnet_limit_C`'s record (spec 4.5): that record with the single-ring field in place of the skipping field, no design margin and +inf (not the rating) for a positive beta. The `where` terms repeat `inner_magnet_limit_C`'s, as the inner and outer magnet-limit records already repeat each other (spec 4.5 records the trade: a per-ring calibration-offset result would let them share it; not done here). Its terms drill down to inputs, and the knee fraction and the Br coefficient move it at the defaults, so it needs no `CANCELLATIONS` entry (checked: the explain suite passes).

Old:
```rust
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - [Δϑ_{cal,i}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000,
```

New:
```rust
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - [Δϑ_{cal,i}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000,
                 [Δϑ_{cal,i}] = cases({model.inner_tmax_C} != "n/a"
                     => 20 + frac([H_k] - [H_{ref,i}], [H_k] · abs({temperature.demag.inner_beta_per_C}) - [H_{ref,i}] · abs({model.inner_alpha_br_per_C}))
                        - {model.inner_tmax_C};
                     else => 0)"#).corrected(&[E20]),
    // BL-064 (owner R-15 ring): the inner ring's own single-ring onset, whichever ring governs
    // (temperature::ring_demag's fourth onset of the inner block). The inner magnet limit's
    // record with the single-ring field in place of the skipping field, no design margin, and
    // +inf (not the rating) for a positive beta; its `where` terms repeat that record's.
    record("temperature.demag.inner_onset_single_ring_C", "ϑ_{sr,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m},
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_single_ring_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - [Δϑ_{cal,i}])
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000,
```

`magcoupling-rs/src/engine/explain/scope.rs`: the module doc's count and the path, right after the governing single-ring onset.

Old:
```rust
//! two parts, and the overshoot per axis past the space claim with the dimension it measures
//! (spec A1); 10 of the 11 are new (the running clearance is a dashboard path), so the
//! scope holds 169 paths.

/// Whether a chain's records are written.
```

New:
```rust
//! two parts, and the overshoot per axis past the space claim with the dimension it measures
//! (spec A1); 10 of the 11 are new (the running clearance is a dashboard path), so the
//! scope held 169 paths. BL-064 adds the inner ring's own single-ring onset to the
//! demagnetization chain (owner R-15 ring): 170 paths.

/// Whether a chain's records are written.
```

Old:
```rust
            "temperature.demag.onset_skipping_C",
            "temperature.demag.onset_single_ring_C",
            "temperature.demag.magnet_limit_C",
            "temperature.summary.onset_aligned_C",
```

New:
```rust
            "temperature.demag.onset_skipping_C",
            "temperature.demag.onset_single_ring_C",
            "temperature.demag.inner_onset_single_ring_C",
            "temperature.demag.magnet_limit_C",
            "temperature.summary.onset_aligned_C",
```

- [ ] **Step 5: The hub-material helper (`material_library.rs`) and the warnings that read it (`api.rs`)**

`magcoupling-rs/src/engine/material_library.rs`: the `yield_MPa` doc (spec 4.1: "not read by the engine" becomes read by the shaft joint check) and the helper, after `PartProperties`. The yield follows the material in effect, not the pick (review SF-16): on the steel circuit, or with a non-magnetic pick (304, 6061), the pick's own; otherwise (no back iron with a ferromagnetic pick or a code outside the choices) the body is the workbook's aluminium, so 6061-T6's 241 MPa. A code outside the choices on the steel circuit gives NaN, as `resolve` already gives NaN steel values.

Old:
```rust
    pub cte_1e6_per_K: Sourced,
    pub modulus_GPa: Sourced,
    /// Yield strength [MPa] (published minimum where one exists; not read by the engine).
    pub yield_MPa: Sourced,
    pub cp_J_kgK: Sourced,
```

New:
```rust
    pub cte_1e6_per_K: Sourced,
    pub modulus_GPa: Sourced,
    /// Yield strength [MPa] (published minimum where one exists; read by the shaft joint check
    /// through [`PartProperties::hub`]).
    pub yield_MPa: Sourced,
    pub cp_J_kgK: Sourced,
```

Old:
```rust
}

/// The workbook values of a part, as [`EngineProps`].
#[allow(non_snake_case)]
```

New:
```rust
}

/// The hub, cup and boss material in effect (BL-064, spec section 4.1): what the shaft joint
/// check and the material warnings read for the parts on the shafts.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct HubMaterial {
    /// The steel circuit's values ([`PartProperties::steel`]) with the circuit in effect
    /// steel, else the body's ([`PartProperties::body`]).
    pub props: EngineProps,
    /// The library yield strength [MPa] of the metal in effect: the back-iron pick's on the
    /// steel circuit or when the pick is non-magnetic, else the workbook's aluminium (6061-T6);
    /// NaN for a pick outside the choices on the steel circuit (its values are NaN too).
    pub yield_MPa: f64,
}

/// The library id of the workbook's aluminium hub, cup and boss (Metal design C42, Temperature
/// design C140, E18): the body with no back iron and no non-magnetic pick.
const WORKBOOK_ALUMINIUM: &str = "6061_T6";

impl PartProperties {
    /// The hub, cup and boss material in effect (BL-064): the steel values when the circuit
    /// in effect is steel (`backiron == 1`), else the body's; the yield follows the material in
    /// effect, not the pick: with no back iron (`coupling.backiron` 0) and a ferromagnetic pick
    /// or a code outside the choices, the body is the workbook's aluminium, so its yield is
    /// 6061-T6's.
    pub fn hub(&self) -> HubMaterial {
        let steel = self.backiron == 1;
        let pick = self.back_iron.material;
        let non_magnetic_pick = pick.is_some_and(|m| !m.ferromagnetic);
        let record = if steel || non_magnetic_pick {
            pick
        } else {
            material(WORKBOOK_ALUMINIUM)
        };
        HubMaterial {
            props: if steel { self.steel } else { self.body },
            yield_MPa: record.and_then(|m| m.yield_MPa.value).unwrap_or(f64::NAN),
        }
    }
}

/// The workbook values of a part, as [`EngineProps`].
#[allow(non_snake_case)]
```

`magcoupling-rs/src/engine/api.rs`: the warnings' hub expansion reads the helper (DRY: one rule, `api.rs:349-353` today). The local is named so Task 2's joint links read the same value.

Old:
```rust
    // Addendum A5: the material consequence warnings, on the materials in effect.
    let back_iron = parts.back_iron.material;
    let warn = warnings::compute(&warnings::WarningInputs {
        circuit_backiron: parts.backiron,
```

New:
```rust
    // Addendum A5: the material consequence warnings, on the materials in effect.
    let back_iron = parts.back_iron.material;
    // The hub, cup and boss material in effect (BL-064: one rule for the warnings and the shaft
    // joints).
    let hub_material = parts.hub();
    let warn = warnings::compute(&warnings::WarningInputs {
        circuit_backiron: parts.backiron,
```

Old:
```rust
        back_iron_needs_plating: back_iron.is_some_and(|m| m.needs_plating),
        plating_mm: mat_in.nickel.thickness_mm,
        hub_cte_per_C: if parts.backiron == 1 {
            parts.steel.cte_per_C
        } else {
            parts.body.cte_per_C
        },
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });
```

New:
```rust
        back_iron_needs_plating: back_iron.is_some_and(|m| m.needs_plating),
        plating_mm: mat_in.nickel.thickness_mm,
        hub_cte_per_C: hub_material.props.cte_per_C,
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });
```

`temperature.rs` keeps its own E18 choice of the hub's expansion for the bond-mismatch screen (`temperature.rs:1069-1073`: it toggles with the correction and reads the links); it is the same rule only with every correction on, so it is not folded into the helper (surgical; recorded as an observation for Task 9).

- [ ] **Step 6: Run the engine suite and see the new tests pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test grades --test material_links --test explain --test schema`

Expected: `grades` 12 passed (11 at HEAD, plus `the_inner_ring_s_own_single_ring_onset_whichever_ring_governs`); `material_links` 14 passed (12 + 2); `explain` 57 passed, 1 ignored (the drift guard holds the new record at every point, the mixed-ring one included); `schema` 11 passed.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib temperature`

Expected: `test result: ok.` with `each_ring_shows_its_own_block` and `the_inner_single_ring_onset_is_the_inner_block_s_whichever_ring_governs` among the passes.

- [ ] **Step 7: Run the app library tests and move the trace totals**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: `test result: FAILED. 886 passed; 1 failed; 6 ignored`:
```text
gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so
    panicked at src\gui\trace.rs:194:9:
  left: (393, 1088)
 right: (392, 1087)
```
One more result, and it has an equation record. In `magcoupling-rs/src/gui/trace.rs`:

Old:
```rust
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 392 of the 1087 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.
```

New:
```rust
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 393 of the 1088 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.
```

Old:
```rust
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (392, 1087));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

New:
```rust
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (393, 1088));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

Run it again. Expected: `test result: ok. 887 passed; 0 failed; 6 ignored`.

The results table, the spreadsheet and the result groups take the new row with no edit: it is in the `temperature` package, which has its heading, and in the `demagnetization` chain, which `CHAIN_LABELS` already heads.

- [ ] **Step 8: Docs: `magcoupling-rs/README.md`**

Seven edits. Each "Old" is a substring of one line; replace only it.

(a) The Layout table's `src/engine/material_library.rs` row (around :139). Old:
```text
each part's choices |
```
New:
```text
each part's choices; `PartProperties::hub` (`HubMaterial`, BL-064: the hub, cup and boss material in effect, the steel values on the steel circuit, else the body's, with the library yield of the metal in effect, 6061-T6's for the workbook aluminium body; the material warnings read its expansion) |
```

(b) The `src/engine/temperature.rs` row (around :141; spec 8, review 6 SF-3). Old:
```text
(130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3) |
```
New:
```text
(130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3; and the inner ring's own single-ring onset `inner_onset_single_ring_C`, whichever ring governs, for the BL-064 shaft-joint check, owner R-15 ring) |
```

(c) The trace paragraph (around :294). Old:
```text
the registry explains 392 of the 1087 results, so an input such as the
```
New:
```text
the registry explains 393 of the 1088 results, so an input such as the
```

(d) The Tests table's `src/gui/inputs.rs`, ... `results_table.rs` row (around :674). Old:
```text
392 of the 1087 results are explained, every input traceable
```
New:
```text
393 of the 1088 results are explained, every input traceable
```

(e) The Tests table's `tests/grades.rs` row (around :656). Old:
```text
the magnets' mass and the bond block's mass (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round`). |
```
New:
```text
the magnets' mass and the bond block's mass, and the inner ring's own single-ring onset, the ferrite's +inf inside (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round`); the inner ring's own single-ring onset (BL-064, owner R-15 ring) reads the inner ring's block alone, is the governing onset bit for bit where the inner ring governs (the defaults, the five probe designs, inner B842 with outer B842SH, manual N50M with SmCo_2_17_26 outside) and differs where the outer governs (inner B842SH with outer B842: 143.3565 against 69.8326 °C; manual N42H with SmCo_2_17_30 outside: 112.8297 against 191.8772 °C), each pin asserting the ring that governs (`the_inner_ring_s_own_single_ring_onset_whichever_ring_governs`). |
```

(f) The Tests table's `tests/material_links.rs` row (around :658). Old:
```text
every warning text reads in plain words (the guard, no audit or code names). |
```
New:
```text
every warning text reads in plain words (the guard, no audit or code names); the hub, cup and boss material in effect (`PartProperties::hub`, BL-064) is the steel values on the steel circuit, else the body's, with the yield of the metal in effect (the pick's, or 6061-T6's 241 MPa for the workbook aluminium body with the circuit switched off; NaN for a code outside the choices on the steel circuit), and the expansion warning reads it unchanged, bit for bit. |
```

(g) The Python-parity paragraph under "Regenerating test data" (around :704; spec 8: one Rust-only temperature result). Old:
```text
and no data file, and the parity, differential and Python schema tests skip `rust_only`
results, so gate 12 compares nothing new.
```
New:
```text
and no data file, and the parity, differential and Python schema tests skip `rust_only`
results, so gate 12 compares nothing new. A Rust-only result in a ported group (E20's
per-ring results, BL-064's `temperature.demag.inner_onset_single_ring_C`) is skipped the
same way.
```

- [ ] **Step 9: Docs: `docs/ai/03-structure.yaml`**

Three edits under `magcoupling_rs.engine_files`.

(a) The `material_library:` entry (around :46). Old:
```text
resolve -> PartProperties: the circuit in effect, the design flux density, the steel, body (no back iron), sleeve and cap values the engine reads)"
```
New:
```text
resolve -> PartProperties: the circuit in effect, the design flux density, the steel, body (no back iron), sleeve and cap values the engine reads; PartProperties::hub -> HubMaterial (BL-064: the hub, cup and boss material in effect, steel on the steel circuit else the body, and the library yield of the metal in effect; the warnings read its expansion))"
```

(b) The `temperature:` entry (around :57; spec 8, review 6 SF-3). Old:
```text
Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side)"
```
New:
```text
Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side, and inner_onset_single_ring_C (BL-064, owner R-15 ring: the inner ring's own single-ring onset, whichever ring governs))"
```

(c) The `explain:` entry (around :63; an unquoted scalar: no ": " may be added). Old:
```text
scope.rs SCOPE (decision 31 plus the G1 geometry callouts, 169 paths, all explained)
```
New:
```text
scope.rs SCOPE (decision 31 plus the G1 geometry callouts and BL-064's inner-ring single-ring onset, 170 paths, all explained)
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

- [ ] **Step 10: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`:

| binary | result |
|---|---|
| lib | 230 passed, 1 ignored |
| assumptions | 11 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 57 passed, 1 ignored |
| gearbox | 13 passed |
| grades | 12 passed |
| material_library | 4 passed |
| material_links | 14 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 12 passed |
| schema | 11 passed |
| size_controls | 38 passed |
| sizing | 44 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `887 passed; 0 failed; 6 ignored`; the two app binaries 0 tests; `explain` 58 passed, 1 ignored; every other integration binary as above.

The Python parity suites (`python_schema`, `parity`, `differential`, `deviations`) pass at HEAD's counts: the new result is `rust_only`.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`, then the same with `--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/api.rs
 M magcoupling-rs/src/engine/explain/records/demagnetization.rs
 M magcoupling-rs/src/engine/explain/scope.rs
 M magcoupling-rs/src/engine/material_library.rs
 M magcoupling-rs/src/engine/temperature.rs
 M magcoupling-rs/src/gui/trace.rs
 M magcoupling-rs/tests/explain.rs
 M magcoupling-rs/tests/grades.rs
 M magcoupling-rs/tests/material_links.rs
 M magcoupling-rs/tests/schema.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 11: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T1.txt` with the Write tool:
```text
feat(magcoupler): the inner ring's own single-ring onset and the hub-material helper (BL-064 T1)

Two engine prerequisites of the BL-064 shaft joint check.

- temperature.demag.inner_onset_single_ring_C (Rust-only): the inner ring's own
  single-ring onset, the inner block's fourth onset, whichever ring governs (owner
  R-15 ring). Equal to the governing onset bit for bit where the inner ring governs
  (the defaults and the five probe designs); with the outer ring governing they
  differ (inner B842SH, outer B842: 143.36 against 69.83 °C). Its record in
  records/demagnetization.rs and its path in the demagnetization scope chain (the
  scope 169 -> 170 paths; the trace totals 393 of 1,088).
- PartProperties::hub -> HubMaterial: the hub, cup and boss material in effect (the
  steel values on the steel circuit, else the body's) with the library yield of the
  metal in effect (6061-T6's 241 MPa for the workbook aluminium body). The material
  warnings read its expansion, unchanged bit for bit.

The new result's help names the bonded inner hub's allowed fitting temperature,
which the next commit adds; only the merge ships.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/explain/records/demagnetization.rs magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/material_library.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/gui/trace.rs magcoupling-rs/tests/grades.rs magcoupling-rs/tests/material_links.rs magcoupling-rs/tests/explain.rs magcoupling-rs/tests/schema.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T1.txt
git status --short
```

Expected: one commit of 12 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 1 moves** (each confirmed by running in the plan-time prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so` (and the module doc) | `(392, 1087)` -> `(393, 1088)` | one more result, with a record |
| `tests/explain.rs` `explained_chains_have_every_record_and_scope_paths_exist` | union 169 -> 170 | the `demagnetization` chain gains the inner onset |
| `temperature.rs` `each_ring_shows_its_own_block`; `tests/grades.rs` `mixed_rings_each_take_their_own_alpha_and_density_either_way_round` | extended | the inner onset asserted (the NdFeB ring's; the ferrite's +inf inside) |
| `tests/schema.rs` `engine_owned_help_is_plain_and_short` | list extended | the new Rust-only result is held to the help cap |
| `tests/explain.rs` `edge_points` | one point added | the inner onset's record at a design where it differs from the governing onset |

---

### Task 2: The `shaft_joint` engine module, its inputs-panel place, and every count the new inputs and results move

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 3, 4.1-4.4, 5.3 (the inputs panel's sections, plan deviation PD-4), 5.4 (the session), 7 items 1-3, 16 and 17 (the counts this task moves). The engine's keyed shrink-fit check for both joints, wired into `compute_all` after the gearbox check; its tests end to end and on the pure `joint` function; the inputs' place in the inputs panel; the share-link guard; the re-blessed `input_schema.json`.

Line numbers are at Task 1's commit; Task 1 touched only `api.rs` (the warnings, +3 lines around :339) and `trace.rs` among the files this task edits.

**What this task's commit does not show (it needs no part of Task 5):** the check is not on the dashboard yet: not in `DASHBOARD` or `CHECKS`, not classified by `verdict_level`, no guide. Its text shows in the results table under the package heading "Shaft joints" with no badge (`check_level` of a path outside `CHECKS` is `None`), so no badge can show where the end-effect or stored-3D rule would hide it. Evidence: with this task applied the app suite is green (lib 888 passed), and `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` (`dashboard.rs:1366-1407`) does not require the path in `CHECKS`: it matches names ending in `_check`, `_screen`, `verdict` or `reading`, and the spec's binding path `shaft_joint.check` ends in `check` (Task 5 widens the rule to a name equal to `check`, in its commit: plan deviation PD-23, ruling C-8).

**What this task takes from Task 4 (plan deviation PD-4).** Fifteen inputs without a place in the inputs panel fail eight GUI tests (`every_input_is_in_exactly_one_workflow_section`, `every_input_is_in_exactly_one_section_in_schema_order`, `every_section_has_a_heading_and_every_heading_a_section`, `the_shown_catalogue_leaves_out_the_paused_section_s_inputs`, `the_filter_matches_label_path_and_cell_by_section_in_the_order_shown`, panel's `idle_frames_change_no_input` and `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`, spreadsheet's `every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order`), so this task makes spec 5.3's placement: `SECTION_LABELS` 18 -> 19 with `("shaft_joint", "Shaft joints")`; the `shaft` section gains the two key rows after the clamp's; the new sections `shaft.fit` ("Shrink fit") and `shaft.fitting` ("Fitting temperatures"), before the paused clamp's. Task 4 no longer does these or their test moves.

**Files:**
- Create: `magcoupling-rs/src/engine/shaft_joint.rs`, `magcoupling-rs/tests/shaft_joint.rs`
- Modify: `magcoupling-rs/src/engine/mod.rs`, `magcoupling-rs/src/engine/api.rs`, `magcoupling-rs/src/gui/inputs.rs` (`SECTION_LABELS`, `WORKFLOW`), `magcoupling-rs/src/gui/result_groups.rs` (`PACKAGE_LABELS`), `magcoupling-rs/tests/data/input_schema.json` (re-blessed, never by hand)
- Modify (tests): `magcoupling-rs/tests/robustness.rs`, `magcoupling-rs/tests/schema.rs`, `magcoupling-rs/tests/gearbox.rs`, `magcoupling-rs/src/gui/session.rs`, `magcoupling-rs/src/gui/inputs.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/results_table.rs`, `magcoupling-rs/src/gui/trace.rs`
- Modify (docs): `magcoupling-rs/README.md`, `magcoupling-rs/src/lib.rs` (the crate doc), `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`
- Test: `magcoupling-rs/tests/shaft_joint.rs` (new, 30 tests); `magcoupling-rs/tests/robustness.rs` (`an_invalid_adhesive_code_selects_no_adhesive` extended; new `degenerate_shaft_joint_inputs_never_pass`); `magcoupling-rs/src/gui/session.rs` (new `a_design_file_and_a_share_link_from_before_the_shaft_joints_load_them_at_their_defaults`; the share-link guard); the moved tests of Steps 8 and 10

**Interfaces:**
- Consumes: `metal.torque_cold_high_Nm` (`MetalDesignResults`), `coupling.bore_mm`, `coupling.keyway_depth_mm`, `coupling.inner_back_apothem_mm`, `metal.bond_inner_mm`, `metal.boss_length_mm`, `metal.web_mm`, `metal.boss_od_mm`, `metal.min_temp_C`, `housing.hub_length_mm` (`axial.hub_length_mm` in `api::compute`), `temperature.adhesive.design_limit_C`, `temperature.demag.inner_onset_single_ring_C` (Task 1), `PartProperties::hub()` (Task 1, `hub_material` in `api::compute`), `temperature::CURE_MARGIN_MIN_C`, `compat::{fmt_fixed, py_max, py_min}`, `meta::{inputs, results, param_rust_only, out_rust_only, InputSet, Value}`.
- Produces (`magcoupling::engine::shaft_joint`; the names spec 4.1-4.2 binds are verbatim):
  - `ShaftJointInputs` (the 15 inputs of spec 4.1, in its order, Rust-only, no `.assumption()` yet: Task 3 adds the four flags with their `ASSUMPTIONS` rows, plan deviation PD-5) and `ShaftJointResults` (the 24 results, in spec order, Rust-only), as `DesignInputs::shaft_joint` (after `clamps`) and `DesignResults::shaft_joint` (after `gearbox`, before the sweeps).
  - Constants: `SHAFT_MODULUS_GPA`, `POISSON`, `SHAFT_CTE_PER_C`, `ROOM_C`, `YIELD_FACTOR`, `NIST_FE_2_25NI`, `NIST_FIT_TOP_K`, `INNER_HUB`, `CUP`; the verdicts `SHAFT_JOINTS_PASS`, `SHAFT_JOINT_UNKNOWN`, `UNKNOWN_NOT_A_NUMBER`, `UNKNOWN_NO_WALL`, `UNKNOWN_BAND_REVERSED`, `UNKNOWN_SAFETY_FACTOR` (ruling C-4), `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`, `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT`, `KEY_HAS_NO_BEARING_AREA`.
  - `hub_fit_allowed(input_C: f64, adhesive_C: f64, onset_C: f64) -> (f64, Option<HubFitLimit>)`, `enum HubFitLimit { Yours, Adhesive, MagnetOnset }` (spec 4.1; the guide of Task 5 calls it with the same three values to name the limit).
  - `enum Cause { Slips, BoreStress, CannotBeFitted, Key }` and `CAUSES` (the badge's order); `enum NoWall { BoreAsWideAsPart, KeywayThroughWall }` (the guide's two no-wall sentences).
  - `ShaftJointLinks` (the 15 engine values of spec 4.4, `values() -> [f64; 15]`); `JointSpec` (one joint's own values) and `JointCommon` (what both share); `JointTerms` (one joint's terms: `name`, `no_wall`, `band_reversed`, `keyed`, the per-µm values, the effective interferences, `holding_Nm`, `bore_stress_MPa`, `needed_um`, `elastic_max_um`, `shaft_shrink_um`, `fittable_max_um`, `fit_temp_C`, `fit_allowed_C`, `key_bearing_mm` (l), `key_contact_mm` (k), `key_pressure_MPa`, `required_Nm`, `stress_limit_MPa`, `slip_margin_um`, `key_margin_mm`; methods `determinate()`, `key_has_bearing_area()`, `compared()`, `fails(Cause)`, `margin(Cause)` (spec 5.2's margins, for Task 5's `terms()` and `helps()`), `text(Cause)`).
  - `joint(&JointSpec, &JointCommon) -> JointTerms`; `joint_specs(&ShaftJointInputs, &ShaftJointLinks) -> (JointCommon, [JointSpec; 2])`; `joints(..) -> [JointTerms; 2]` (inner hub first); `failing(&[JointTerms; 2]) -> Vec<(Cause, &JointTerms)>` (every failing term of the judged joints, in the badge's order: Task 5's Problem lists them); `verdict(..) -> String`; `compute(&ShaftJointInputs, &ShaftJointLinks) -> ShaftJointResults`; `shaft_shrink_um(shaft_C, bore_mm) -> f64`.
  - `api::shaft_joint_links(&DesignInputs, &DesignResults) -> ShaftJointLinks`: the 15 values rebuilt from a design's inputs and results (each is an input or a result), so the guide (Task 5) and the tests evaluate the joints as `compute_all` did; `the_joints_rebuilt_from_the_results_are_the_engine_s` pins it.
  - GUI: `SECTION_LABELS` entry `("shaft_joint", "Shaft joints")`; `WORKFLOW` sections `shaft.fit`, `shaft.fitting`; `PACKAGE_LABELS` entry `("shaft_joint", "Shaft joints")`.

- [ ] **Step 1: Write the failing engine tests**

Create `magcoupling-rs/tests/shaft_joint.rs`. It follows `tests/gearbox.rs`: a Rust-only engine result tested end to end through `compute_all`, and on the pure `joint` function for the edges. Its numbers are the spec's (sections 3.4-3.7, 4.2, 7.1-7.2, from the spec's scratch crate); the engine reproduces every one in the prototype. `passing()` restates `gui::test_support::passing_design()` (crate-private to the GUI, unreachable from an engine test): a four-line restatement that its own numbers pin (ruling C-8). Each R-15 test reads `CURE_MARGIN_MIN_C`, never a typed 10, and names R-15 or its ring ruling in a comment.

```rust
//! BL-064: the shaft joint check (`shaft_joint.check`, `src/engine/shaft_joint.rs`), end to end
//! through `compute_all` and on the pure `joint` function. Both rotors are held on their shafts
//! by a keyed shrink fit: the inner hub on the wheel shaft, the cup on the gearbox output shaft.
//! The fit alone must hold the safety factor times cold-high at its loosest, the part must stay
//! within yield ÷ 1.2 at its tightest, it must be fitted within its allowed temperature, and the
//! key alone carries the same torque as a backup (owner decisions 1A, 2A, 7A, R-15 and its ring
//! ruling). A Rust-only result group, tested here as the gearbox check is in `tests/gearbox.rs`.
//! The numbers are the spec's (sections 3.5 to 3.7), which its scratch crate computed.

use magcoupling::compute_all;
use magcoupling::engine::api::{DesignInputs, shaft_joint_links};
use magcoupling::engine::compat::py_min;
use magcoupling::engine::explain::probes::probe_designs;
use magcoupling::engine::meta::{InputSet, SliderRange, Value, input_rows, result_rows};
use magcoupling::engine::shaft_joint::{
    self, BORE_OVER_ITS_LIMIT, CANNOT_BE_FITTED, CAUSES, CUP, Cause, FIT_SLIPS, HubFitLimit,
    INNER_HUB, JointCommon, JointSpec, JointTerms, KEY_HAS_NO_BEARING_AREA, KEY_OVER_ITS_LIMIT,
    NoWall, ROOM_C, SHAFT_CTE_PER_C, SHAFT_JOINT_UNKNOWN, SHAFT_JOINTS_PASS, ShaftJointInputs,
    ShaftJointLinks, UNKNOWN_BAND_REVERSED, UNKNOWN_NO_WALL, UNKNOWN_NOT_A_NUMBER,
    UNKNOWN_SAFETY_FACTOR, YIELD_FACTOR, hub_fit_allowed, shaft_shrink_um,
};
use magcoupling::engine::temperature::CURE_MARGIN_MIN_C;

/// The default design with `edit` applied.
fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    edit(&mut inputs);
    inputs
}

/// `gui::test_support::passing_design()`, restated here (the GUI's test support is not reachable
/// from an engine test): the defaults with a 1.8 mm face gap, a 2.0 mm cup wall, a 1.6 N·m
/// requirement and a 1.0 N·m gearbox input rating. Its shaft joints pass (9.48 and 16.90
/// against 6.72 N·m).
fn passing() -> DesignInputs {
    design(|i| {
        i.metal.face_gap_mm = 1.8;
        i.metal.cup_wall_corner_mm = 2.0;
        i.metal.required_min_Nm = 1.6;
        i.coupling.gearbox_input_rating_Nm = 1.0;
    })
}

/// `base` with both rings on manual dimensions, grade id `inner` inside and `outer` outside.
fn manual(mut base: DesignInputs, inner: &str, outer: &str) -> DesignInputs {
    let m = &mut base.coupling.magnets;
    m.part_inner.clear();
    m.part_outer.clear();
    m.grade_inner = inner.to_owned();
    m.grade_outer = outer.to_owned();
    base
}

/// The shaft joint check's verdict.
fn check(inputs: &DesignInputs) -> String {
    compute_all(inputs).shaft_joint.check
}

/// The engine values the check read, rebuilt from the design (`api::shaft_joint_links`).
fn links(inputs: &DesignInputs) -> ShaftJointLinks {
    shaft_joint_links(inputs, &compute_all(inputs))
}

/// What the two joints are at `inputs`: what both share and each one's own values.
fn specs(inputs: &DesignInputs) -> (JointCommon, [JointSpec; 2]) {
    shaft_joint::joint_specs(&inputs.shaft_joint, &links(inputs))
}

/// Both joints' terms at `inputs`, inner hub first.
fn terms(inputs: &DesignInputs) -> [JointTerms; 2] {
    shaft_joint::joints(&inputs.shaft_joint, &links(inputs))
}

/// `x` within `tolerance` of `want`.
fn assert_near(x: f64, want: f64, tolerance: f64, what: &str) {
    assert!((x - want).abs() <= tolerance, "{what}: {x}, want {want}");
}

/// The badge's slip text.
fn slips(joint: &str, holding: &str, required: &str) -> String {
    format!("{FIT_SLIPS} ({joint}): holds {holding} N·m against {required} N·m")
}

/// The badge's stress text.
fn over_stress(joint: &str, stress: &str, limit: &str) -> String {
    format!("{BORE_OVER_ITS_LIMIT} ({joint}): {stress} MPa against {limit} MPa")
}

/// The badge's fitting text.
fn cannot_fit(joint: &str, needed: &str, allowed: &str) -> String {
    format!("{CANNOT_BE_FITTED} ({joint}): needs {needed} °C against {allowed} °C")
}

/// The badge's key text.
fn key_over(joint: &str, pressure: &str, limit: &str) -> String {
    format!("{KEY_OVER_ITS_LIMIT} ({joint}): {pressure} MPa against {limit} MPa")
}

/// The two numbers a failing badge quotes: (the failing value, the limit or need).
fn quoted(text: &str) -> (f64, f64) {
    let (_, rest) = text
        .split_once("): ")
        .unwrap_or_else(|| panic!("no numbers: {text}"));
    let rest = rest
        .strip_prefix("holds ")
        .or_else(|| rest.strip_prefix("needs "))
        .unwrap_or(rest);
    let (value, limit) = rest
        .split_once(" against ")
        .unwrap_or_else(|| panic!("{text}"));
    let number = |s: &str| -> f64 {
        s.split(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("{text}"))
    };
    (number(value), number(limit))
}

/// An input's slider.
fn slider(path: &str) -> SliderRange {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|row| row.path == path)
        .and_then(|row| row.meta.range)
        .unwrap_or_else(|| panic!("{path} has a slider"))
}

/// Every value of `path`'s slider, bottom to top.
fn slider_values(path: &str) -> Vec<f64> {
    let range = slider(path);
    let steps = ((range.max - range.min) / range.step).round() as u32;
    (0..=steps)
        .map(|k| range.min + f64::from(k) * range.step)
        .collect()
}

#[test]
fn the_check_is_a_rust_only_group_of_fifteen_inputs_and_twenty_four_results() {
    // Spec section 4.1, in its order: no workbook cell, no Python twin; the inputs after the
    // clamp group, the results after the gearbox group and before the sweeps.
    let inputs: Vec<_> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
        .collect();
    let input_paths: Vec<&str> = inputs.iter().map(|row| row.path.as_str()).collect();
    assert_eq!(
        input_paths,
        [
            "shaft_joint.hub_interference_min_um",
            "shaft_joint.hub_interference_max_um",
            "shaft_joint.cup_interference_min_um",
            "shaft_joint.cup_interference_max_um",
            "shaft_joint.safety_factor",
            "shaft_joint.friction",
            "shaft_joint.smoothing_um",
            "shaft_joint.rotor_over_shaft_C",
            "shaft_joint.key_width_mm",
            "shaft_joint.key_contact_mm",
            "shaft_joint.hub_fit_max_C",
            "shaft_joint.cup_fit_max_C",
            "shaft_joint.wheel_shaft_fit_C",
            "shaft_joint.output_shaft_fit_C",
            "shaft_joint.joining_clearance_um",
        ]
    );
    // Units, defaults and sliders (min, max, step), spec section 4.1.
    let want: [(&str, f64, [f64; 3]); 15] = [
        ("µm", 13.0, [0.0, 40.0, 0.5]),
        ("µm", 18.0, [0.0, 40.0, 0.5]),
        ("µm", 15.0, [0.0, 40.0, 0.5]),
        ("µm", 22.0, [0.0, 40.0, 0.5]),
        ("-", 2.0, [1.0, 4.0, 0.1]),
        ("-", 0.14, [0.05, 0.30, 0.01]),
        ("µm", 5.0, [0.0, 15.0, 0.1]),
        ("°C", 20.0, [0.0, 60.0, 1.0]),
        ("mm", 3.0, [1.0, 6.0, 0.5]),
        ("mm", 1.2, [0.5, 3.0, 0.1]),
        ("°C", 100.0, [20.0, 130.0, 1.0]),
        ("°C", 300.0, [20.0, 400.0, 5.0]),
        ("°C", -196.0, [-196.0, 40.0, 1.0]),
        ("°C", 20.0, [-20.0, 40.0, 1.0]),
        ("µm", 10.0, [0.0, 30.0, 0.5]),
    ];
    for (row, (unit, default, [min, max, step])) in inputs.iter().zip(want) {
        assert!(row.meta.rust_only, "{}", row.path);
        assert_eq!(row.meta.cell, None, "{}", row.path);
        assert_eq!(row.meta.unit, unit, "{}", row.path);
        assert_eq!(row.value, Value::Num(default), "{}", row.path);
        let range = row.meta.range.expect("a slider");
        assert_eq!(
            (range.min, range.max, range.step),
            (min, max, step),
            "{}",
            row.path
        );
        assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
    }
    let all_inputs = input_rows(&DesignInputs::default());
    let first = all_inputs
        .iter()
        .position(|row| row.path.starts_with("shaft_joint."))
        .expect("the group");
    assert!(all_inputs[first - 1].path.starts_with("clamps."));
    assert_eq!(first + 15, all_inputs.len(), "the last input group");

    let rows = result_rows(&compute_all(&DesignInputs::default()));
    let results: Vec<_> = rows
        .iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
        .collect();
    let units: Vec<(&str, &str)> = results
        .iter()
        .map(|row| (row.path.as_str(), row.meta.unit))
        .collect();
    assert_eq!(
        units,
        [
            ("shaft_joint.required_Nm", "N·m"),
            ("shaft_joint.part_modulus_GPa", "GPa"),
            ("shaft_joint.part_cte_per_C", "1/°C"),
            ("shaft_joint.part_yield_MPa", "MPa"),
            ("shaft_joint.stress_limit_MPa", "MPa"),
            ("shaft_joint.hub_holding_Nm", "N·m"),
            ("shaft_joint.cup_holding_Nm", "N·m"),
            ("shaft_joint.hub_bore_stress_MPa", "MPa"),
            ("shaft_joint.cup_bore_stress_MPa", "MPa"),
            ("shaft_joint.hub_key_pressure_MPa", "MPa"),
            ("shaft_joint.cup_key_pressure_MPa", "MPa"),
            ("shaft_joint.hub_needed_um", "µm"),
            ("shaft_joint.cup_needed_um", "µm"),
            ("shaft_joint.hub_elastic_max_um", "µm"),
            ("shaft_joint.cup_elastic_max_um", "µm"),
            ("shaft_joint.wheel_shaft_shrink_um", "µm"),
            ("shaft_joint.output_shaft_shrink_um", "µm"),
            ("shaft_joint.hub_fittable_max_um", "µm"),
            ("shaft_joint.cup_fittable_max_um", "µm"),
            ("shaft_joint.hub_fit_temp_C", "°C"),
            ("shaft_joint.cup_fit_temp_C", "°C"),
            ("shaft_joint.hub_fit_allowed_C", "°C"),
            ("shaft_joint.cup_fit_allowed_C", "°C"),
            ("shaft_joint.check", ""),
        ]
    );
    for row in &results {
        assert!(row.meta.rust_only, "{}", row.path);
        assert_eq!(row.meta.cell, None, "{}", row.path);
        assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
        let text = row.path == "shaft_joint.check";
        assert_eq!(matches!(row.value, Value::Text(_)), text, "{}", row.path);
    }
    let at = rows
        .iter()
        .position(|row| row.path == "shaft_joint.required_Nm")
        .expect("the group");
    assert!(
        rows[at - 1].path.starts_with("gearbox."),
        "{}",
        rows[at - 1].path
    );
    assert!(
        rows[at + 24].path.starts_with("gap_sweep["),
        "{}",
        rows[at + 24].path
    );
}

#[test]
fn the_verdict_texts_are_the_spec_s() {
    // Spec section 4.2: one pass, one unknown prefix with three reasons and the safety factor's
    // (ruling C-4), five failing prefixes.
    assert_eq!(
        SHAFT_JOINTS_PASS,
        "Both shaft joints hold and can be fitted"
    );
    assert_eq!(SHAFT_JOINT_UNKNOWN, "Shaft joint check unknown");
    for (text, reason) in [
        (UNKNOWN_NOT_A_NUMBER, "a value is not a number"),
        (UNKNOWN_NO_WALL, "no wall around the bore"),
        (
            UNKNOWN_BAND_REVERSED,
            "the minimum interference is above the maximum",
        ),
        (UNKNOWN_SAFETY_FACTOR, "the safety factor is not above zero"),
    ] {
        assert_eq!(text, format!("{SHAFT_JOINT_UNKNOWN}: {reason}"));
    }
    assert_eq!(FIT_SLIPS, "The fit slips");
    assert_eq!(BORE_OVER_ITS_LIMIT, "Bore stress above its limit");
    assert_eq!(
        CANNOT_BE_FITTED,
        "Cannot be fitted at the allowed temperature"
    );
    assert_eq!(KEY_OVER_ITS_LIMIT, "Key backup above its limit");
    assert_eq!(KEY_HAS_NO_BEARING_AREA, "The key has no bearing area");
    assert_eq!((INNER_HUB, CUP), ("inner hub", "cup"));
    // No text names the paused clamp, a path or a cell.
    for text in [
        SHAFT_JOINTS_PASS,
        UNKNOWN_NOT_A_NUMBER,
        UNKNOWN_NO_WALL,
        UNKNOWN_BAND_REVERSED,
        UNKNOWN_SAFETY_FACTOR,
        FIT_SLIPS,
        BORE_OVER_ITS_LIMIT,
        CANNOT_BE_FITTED,
        KEY_OVER_ITS_LIMIT,
        KEY_HAS_NO_BEARING_AREA,
    ] {
        assert!(!text.to_lowercase().contains("clamp"), "{text}");
        assert!(!text.contains('_') && !text.contains('!'), "{text}");
    }
}

#[test]
fn per_micrometre_each_joint_grips_and_stresses_as_the_spec_says() {
    // Spec section 3.4, the defaults: contact pressure, holding torque (friction 0.14, the 3 mm
    // key's arc taken off) and bore von Mises stress per µm of effective interference.
    let [hub, cup] = terms(&DesignInputs::default());
    assert_near(hub.pressure_per_um, 7.738, 5e-4, "inner hub pressure");
    assert_near(hub.holding_per_um, 2.0009, 5e-5, "inner hub holding");
    assert_near(hub.stress_per_um, 17.930, 5e-4, "inner hub stress");
    assert_near(cup.pressure_per_um, 8.132, 5e-4, "cup pressure");
    assert_near(cup.holding_per_um, 2.5073, 5e-5, "cup holding");
    assert_near(cup.stress_per_um, 17.879, 5e-4, "cup stress");
}

#[test]
fn the_defaults_pass_with_the_numbers_of_the_spec() {
    // Spec section 3.5: 4140 on a Ø10 bore, the hub 13-18 µm on a -196 °C wheel shaft, the cup
    // 15-22 µm on a 20 °C gearbox shaft.
    let r = compute_all(&DesignInputs::default());
    let s = &r.shaft_joint;
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
    assert_eq!(s.required_Nm, 2.0 * r.metal.torque_cold_high_Nm);
    assert_near(s.required_Nm, 7.6466, 5e-5, "required");
    assert_eq!(
        (s.part_modulus_GPa, s.part_cte_per_C, s.part_yield_MPa),
        (205.0, 12.3e-6, 415.0)
    );
    assert_eq!(s.stress_limit_MPa, 415.0 / YIELD_FACTOR);
    let pins: [(&str, f64, f64, f64); 18] = [
        ("hub holding", s.hub_holding_Nm, 9.4844, 5e-5),
        ("cup holding", s.cup_holding_Nm, 16.899, 5e-4),
        ("hub bore stress", s.hub_bore_stress_MPa, 241.70, 5e-3),
        ("cup bore stress", s.cup_bore_stress_MPa, 312.53, 5e-3),
        ("hub key", s.hub_key_pressure_MPa, 127.44, 5e-3),
        ("cup key", s.cup_key_pressure_MPa, 101.95, 5e-3),
        ("hub needed", s.hub_needed_um, 12.082, 5e-4),
        ("cup needed", s.cup_needed_um, 11.310, 5e-4),
        ("hub elastic", s.hub_elastic_max_um, 23.808, 5e-4),
        ("cup elastic", s.cup_elastic_max_um, 23.863, 5e-4),
        ("wheel shaft shrink", s.wheel_shaft_shrink_um, 18.988, 5e-4),
        ("gearbox shaft shrink", s.output_shaft_shrink_um, 0.0, 0.0),
        ("hub fittable", s.hub_fittable_max_um, 18.828, 5e-4),
        ("cup fittable", s.cup_fittable_max_um, 24.440, 5e-4),
        ("hub fit temperature", s.hub_fit_temp_C, 93.267, 5e-4),
        ("cup fit temperature", s.cup_fit_temp_C, 280.16, 5e-3),
        ("hub allowed", s.hub_fit_allowed_C, 100.0, 0.0),
        ("cup allowed", s.cup_fit_allowed_C, 300.0, 0.0),
    ];
    for (what, x, want, tolerance) in pins {
        assert_near(x, want, tolerance, what);
    }
    // The windows: 12.08-18.83 µm on the hub (the fitting edge binds, the elastic one is
    // 23.81), 11.31-23.86 µm on the cup (the stress edge binds, the fitting one is 24.44).
    assert!(s.hub_fittable_max_um < s.hub_elastic_max_um);
    assert!(s.cup_elastic_max_um < s.cup_fittable_max_um);
    // The bands sit inside them: 0.92 µm above the hub's needed minimum and 0.83 µm under its
    // fitting edge (one 0.5 µm step up on the maximum still fits, two do not).
    let i = &DesignInputs::default().shaft_joint;
    assert_near(
        i.hub_interference_min_um - s.hub_needed_um,
        0.918,
        5e-4,
        "hub",
    );
    assert_near(
        s.hub_fittable_max_um - i.hub_interference_max_um,
        0.828,
        5e-4,
        "hub",
    );
    for (maximum, passes) in [(18.5, true), (19.0, false)] {
        let r = compute_all(&design(|d| d.shaft_joint.hub_interference_max_um = maximum));
        assert_eq!(
            r.shaft_joint.check == SHAFT_JOINTS_PASS,
            passes,
            "{maximum}"
        );
    }
}

#[test]
fn the_bl067_corner_reads_cold_high_and_passes() {
    // Review MF-4: with the operating temperature (-40 °C) below the minimum (20 °C) and a
    // 0.1 variation, the check reads cold-high (3.1823 N·m), not the 3.657 N·m operating
    // pull-out with its allowance (3.3246 × 1.1). BL-067's fix of which pull-out the checks
    // compare moves this pin on purpose.
    let corner = design(|i| {
        i.coupling.op_temp_C = -40.0;
        i.metal.min_temp_C = 20.0;
        i.metal.variation = 0.1;
    });
    let r = compute_all(&corner);
    let s = &r.shaft_joint;
    assert_near(r.metal.torque_cold_high_Nm, 3.1823, 5e-5, "cold-high");
    assert_near(r.model.pullout_Nm * 1.1, 3.657, 5e-4, "operating pull-out");
    assert_near(s.required_Nm, 6.3647, 5e-5, "required");
    assert_near(s.hub_holding_Nm, 9.484, 5e-4, "hub holding");
    assert_near(s.cup_holding_Nm, 16.899, 5e-4, "cup holding");
    assert_near(s.hub_bore_stress_MPa, 233.09, 5e-3, "hub stress");
    assert_near(s.cup_bore_stress_MPa, 303.95, 5e-3, "cup stress");
    assert_near(s.hub_key_pressure_MPa, 106.08, 5e-3, "hub key");
    assert_near(s.cup_key_pressure_MPa, 84.86, 5e-3, "cup key");
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
}

#[test]
fn each_probe_design_reads_its_verdict() {
    // Spec section 3.6, in `probe_designs()` order: (name, the holding torques, the required
    // torque [N·m], the verdict). The allowed hub temperature is your 100 °C limit at each
    // (owner R-15: the inner ring governs at each, so the onset read is the governing one).
    let want: [(&str, f64, f64, f64, String); 5] = [
        (
            "default",
            9.4844,
            16.899,
            7.6466,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "8 magnets per ring",
            9.4844,
            16.899,
            5.5985,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "304 back iron",
            0.0,
            2.9274,
            4.8277,
            slips(INNER_HUB, "0.00", "4.83"),
        ),
        (
            "Y30 ferrite, custom dimensions",
            9.4844,
            16.899,
            0.6761,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
        (
            "20 mm ring radius",
            11.774,
            16.899,
            7.2222,
            SHAFT_JOINTS_PASS.to_owned(),
        ),
    ];
    let designs = probe_designs();
    assert_eq!(designs.len(), want.len());
    for ((name, inputs), (want_name, hub, cup, required, verdict)) in designs.into_iter().zip(want)
    {
        assert_eq!(name, want_name);
        let s = compute_all(&inputs).shaft_joint;
        assert_near(s.hub_holding_Nm, hub, 5e-4, name);
        assert_near(s.cup_holding_Nm, cup, 5e-4, name);
        assert_near(s.required_Nm, required, 5e-4, name);
        assert_eq!(s.hub_fit_allowed_C, 100.0, "{name}");
        assert_eq!(s.check, verdict, "{name}");
    }
}

#[test]
fn a_band_minimum_too_low_slips_at_its_own_joint() {
    let hub = check(&design(|i| i.shaft_joint.hub_interference_min_um = 10.0));
    assert_eq!(hub, slips(INNER_HUB, "3.48", "7.65"));
    let cup = check(&design(|i| i.shaft_joint.cup_interference_min_um = 10.0));
    assert_eq!(cup, slips(CUP, "4.36", "7.65"));
}

#[test]
fn a_band_maximum_too_high_is_over_the_stress_limit_and_cannot_be_fitted() {
    // The stress cause is the badge; the fitting cause follows it (spec section 4.3).
    let limit = "345.8";
    for (joint, maximum, stress) in [
        (INNER_HUB, 26.0, "385.1"),
        (INNER_HUB, 28.0, "421.0"),
        (CUP, 26.0, "384.0"),
        (CUP, 28.0, "419.8"),
    ] {
        let inputs = design(|i| {
            if joint == INNER_HUB {
                i.shaft_joint.hub_interference_max_um = maximum;
            } else {
                i.shaft_joint.cup_interference_max_um = maximum;
            }
        });
        assert_eq!(
            check(&inputs),
            over_stress(joint, stress, limit),
            "{joint} {maximum}"
        );
        let joints = terms(&inputs);
        let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
            .into_iter()
            .map(|(cause, j)| (cause, j.name))
            .collect();
        assert_eq!(
            failing,
            [(Cause::BoreStress, joint), (Cause::CannotBeFitted, joint)],
            "{joint} {maximum}"
        );
    }
}

#[test]
fn the_badge_takes_causes_in_order_then_the_inner_hub_before_the_cup() {
    // Spec section 4.3: within a cause the inner hub is named first; an earlier cause on the
    // cup outranks a later one on the inner hub.
    let both = design(|i| {
        i.shaft_joint.hub_interference_max_um = 28.0;
        i.shaft_joint.cup_interference_max_um = 28.0;
    });
    let joints = terms(&both);
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(
        failing,
        [
            (Cause::BoreStress, INNER_HUB),
            (Cause::BoreStress, CUP),
            (Cause::CannotBeFitted, INNER_HUB),
            (Cause::CannotBeFitted, CUP),
        ]
    );
    assert_eq!(check(&both), over_stress(INNER_HUB, "421.0", "345.8"));
    let cup_slips_hub_overstressed = design(|i| {
        i.shaft_joint.hub_interference_max_um = 26.0;
        i.shaft_joint.cup_interference_min_um = 10.0;
    });
    assert_eq!(
        check(&cup_slips_hub_overstressed),
        slips(CUP, "4.36", "7.65")
    );
    assert_eq!(
        CAUSES,
        [
            Cause::Slips,
            Cause::BoreStress,
            Cause::CannotBeFitted,
            Cause::Key
        ]
    );
}

#[test]
fn the_304_and_aluminium_windows_are_empty() {
    // Spec section 3.6: neither hub has a working interference at Ø10. 304: holding needs
    // 16.30 µm, the bore stays within its limit only up to 11.72. Aluminium (the 6061 pick, or
    // no back iron with any pick, one outside the choices too): 27.74 against 25.22.
    let r304 = compute_all(&design(|i| i.materials.parts.back_iron = 7)).shaft_joint;
    assert_near(r304.hub_needed_um, 16.3031, 5e-5, "304 needed");
    assert_near(r304.hub_elastic_max_um, 11.7235, 5e-5, "304 elastic");
    assert_near(r304.cup_needed_um, 15.7920, 5e-5, "304 cup needed");
    assert_near(r304.cup_elastic_max_um, 11.7439, 5e-5, "304 cup elastic");
    for aluminium in [
        design(|i| i.materials.parts.back_iron = 8),
        design(|i| i.coupling.backiron = 0),
        design(|i| {
            i.coupling.backiron = 0;
            i.materials.parts.back_iron = 2;
        }),
        design(|i| {
            i.coupling.backiron = 0;
            i.materials.parts.back_iron = 99;
        }),
    ] {
        let s = compute_all(&aluminium).shaft_joint;
        assert_eq!(
            (s.part_modulus_GPa, s.part_cte_per_C, s.part_yield_MPa),
            (68.9, 23.6e-6, 241.0)
        );
        assert_near(s.hub_needed_um, 27.74, 5e-3, "aluminium needed");
        assert_near(
            py_min(s.hub_elastic_max_um, s.hub_fittable_max_um),
            25.22,
            5e-3,
            "upper",
        );
        assert!(s.hub_needed_um > py_min(s.hub_elastic_max_um, s.hub_fittable_max_um));
        assert_eq!((s.hub_holding_Nm, s.cup_holding_Nm), (0.0, 0.0));
        assert_eq!(s.check, slips(INNER_HUB, "0.00", "4.83"));
    }
}

#[test]
fn the_end_that_loosens_the_fit_follows_the_part_s_expansion() {
    // Synthesis correction 1, spec section 3.3: a part expanding less than the shaft (416,
    // 10.08e-6 /°C) loosens cold, so its minimum is taken at the minimum magnet temperature;
    // one expanding more (304, 16.9e-6 /°C) loosens hot, at the adhesive's design limit with
    // the rotor 20 °C over its shaft. R-14 puts this hot-end form to the physics review: the
    // bond-limited form would take 416's minimum hot.
    for (code, cold) in [(4, true), (7, false)] {
        let inputs = design(|i| i.materials.parts.back_iron = code);
        let r = compute_all(&inputs);
        let (common, [hub, _]) = specs(&inputs);
        let alpha = r.shaft_joint.part_cte_per_C;
        let d = inputs.coupling.bore_mm;
        let mismatch = |t: f64| (SHAFT_CTE_PER_C - alpha) * (t - ROOM_C) * d * 1000.0;
        let cold_end = mismatch(common.cold_C);
        let hot_end = mismatch(common.hot_C) - alpha * common.rotor_over_shaft_C * d * 1000.0;
        assert_eq!(cold_end < hot_end, cold, "{code}");
        let loosest = if cold { cold_end } else { hot_end };
        let [terms_hub, _] = terms(&inputs);
        assert_eq!(
            terms_hub.effective_min_um,
            (hub.interference_min_um - common.smoothing_um + loosest).max(0.0),
            "{code}"
        );
    }
}

#[test]
fn an_uncooled_wheel_shaft_or_dry_ice_cannot_fit_the_inner_hub_nor_a_cooler_bare_cup() {
    for (shaft, needed) in [(20.0, "247.6"), (-78.5, "163.5")] {
        let inputs = design(|i| i.shaft_joint.wheel_shaft_fit_C = shaft);
        assert_eq!(
            check(&inputs),
            cannot_fit(INNER_HUB, needed, "100.0"),
            "{shaft}"
        );
    }
    let cup = design(|i| i.shaft_joint.cup_fit_max_C = 250.0);
    assert_eq!(check(&cup), cannot_fit(CUP, "280.2", "250.0"));
}

#[test]
fn the_allowed_hub_temperature_is_the_lowest_of_three() {
    // Owner R-15: the bonded inner hub's allowed fitting temperature is the lowest of your
    // limit, the selected adhesive's design limit and the inner ring's own single-ring onset
    // less the cure margin (CURE_MARGIN_MIN_C), with the limit that binds.
    let allowed = |inputs: &DesignInputs| {
        let r = compute_all(inputs);
        let (value, limit) = hub_fit_allowed(
            inputs.shaft_joint.hub_fit_max_C,
            r.temperature.adhesive.design_limit_C,
            r.temperature.demag.inner_onset_single_ring_C,
        );
        assert_eq!(value.to_bits(), r.shaft_joint.hub_fit_allowed_C.to_bits());
        (r, limit)
    };
    // The defaults: your 100 °C (the onset 143.3565 °C less 10 is 133.3565 °C).
    let (r, limit) = allowed(&DesignInputs::default());
    assert_eq!(limit, Some(HubFitLimit::Yours));
    assert_eq!(r.shaft_joint.hub_fit_allowed_C, 100.0);
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C,
        133.3565,
        5e-5,
        "the onset's cap",
    );
    // DP460 (code 4) caps it at its 60 °C design limit; a typed 150 °C (above the slider's
    // 130 °C top) is capped at AA 326's 120 °C.
    let (r, limit) = allowed(&design(|i| i.temperature.adhesive.selected = 4));
    assert_eq!(
        (r.shaft_joint.hub_fit_allowed_C, limit),
        (60.0, Some(HubFitLimit::Adhesive))
    );
    let (r, limit) = allowed(&design(|i| i.shaft_joint.hub_fit_max_C = 150.0));
    assert_eq!(
        (r.shaft_joint.hub_fit_allowed_C, limit),
        (120.0, Some(HubFitLimit::Adhesive))
    );
    // `breaking(n42_parts)`: the library's N42 blocks in both rings, onset 69.83 °C, so
    // 59.83 °C allowed; the badge read a pass before R-15.
    let mut n42_parts = passing();
    n42_parts.coupling.magnets.part_inner = "B842".to_owned();
    n42_parts.coupling.magnets.part_outer = "B842".to_owned();
    let (r, limit) = allowed(&n42_parts);
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        69.8326,
        5e-5,
        "N42 onset",
    );
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        59.8326,
        5e-5,
        "N42 allowed",
    );
    assert_near(r.shaft_joint.hub_fit_temp_C, 93.267, 5e-4, "N42 needed");
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "59.8"));
    // N50M on manual dimensions in both rings: its onset, 98.80 °C, is above the 93.27 °C
    // need; only the onset less 10 °C, 88.80 °C, is below it.
    let (r, limit) = allowed(&manual(passing(), "N50M", "N50M"));
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        98.8029,
        5e-5,
        "N50M onset",
    );
    assert!(r.temperature.demag.inner_onset_single_ring_C > r.shaft_joint.hub_fit_temp_C);
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        88.8029,
        5e-5,
        "N50M allowed",
    );
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "88.8"));
    // Ferrite's onset is +inf: the cap never binds (the Y30 probe and `breaking(ferrite)`).
    for ferrite in [
        probe_designs()
            .into_iter()
            .find(|(name, _)| *name == "Y30 ferrite, custom dimensions")
            .expect("the Y30 probe")
            .1,
        manual(passing(), "Y30", "Y30"),
    ] {
        let (r, limit) = allowed(&ferrite);
        assert_eq!(r.temperature.demag.inner_onset_single_ring_C, f64::INFINITY);
        assert_eq!(
            (r.shaft_joint.hub_fit_allowed_C, limit),
            (100.0, Some(HubFitLimit::Yours))
        );
    }
    // 2214 Hi-Temp (code 3, 177 °C) with a typed 150 °C: the onset less 10 °C binds.
    let (r, limit) = allowed(&design(|i| {
        i.temperature.adhesive.selected = 3;
        i.shaft_joint.hub_fit_max_C = 150.0;
    }));
    assert_eq!(limit, Some(HubFitLimit::MagnetOnset));
    assert_near(
        r.shaft_joint.hub_fit_allowed_C,
        133.3565,
        5e-5,
        "2214 allowed",
    );
    assert_eq!(r.shaft_joint.check, SHAFT_JOINTS_PASS);
}

#[test]
fn the_hub_fitting_slider_tops_out_below_the_inner_onset_less_the_cure_margin() {
    // Review SF-3: at the defaults the slider's 130 °C top is below the inner ring's own
    // single-ring onset less the cure margin (133.36 °C), so the slider alone never reaches it.
    let r = compute_all(&DesignInputs::default());
    let top = slider("shaft_joint.hub_fit_max_C").max;
    assert_eq!(top, 130.0);
    assert!(top < r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C);
}

#[test]
fn the_cap_reads_the_inner_ring_s_own_onset_with_mixed_rings() {
    // Owner R-15 ring: the bonded hub's fitting heat reaches only the inner magnets (the cup's
    // are bonded after the cup is fitted), so the cap reads the inner ring's own onset, whichever
    // ring governs. Each pin asserts the governing ring and both onsets, so a cap read from the
    // governing ring, or a mistyped part name or grade id, fails it.
    let demag_pins = |inputs: &DesignInputs, ring: &str, governing: f64, inner: f64| {
        let r = compute_all(inputs);
        let demag = &r.temperature.demag;
        assert_eq!(demag.demag_ring, ring);
        assert_near(
            demag.onset_single_ring_C,
            governing,
            5e-5,
            "governing onset",
        );
        assert_near(demag.inner_onset_single_ring_C, inner, 5e-5, "inner onset");
        r
    };
    // Inner B842SH with outer B842: the outer N42 governs at 69.83 °C; the hub passes, your
    // limit binding (window 11.62-18.83 µm).
    let mut inputs = passing();
    inputs.coupling.magnets.part_outer = "B842".to_owned();
    let r = demag_pins(&inputs, "outer", 69.8326, 143.3565);
    assert_eq!(r.shaft_joint.hub_fit_allowed_C, 100.0);
    assert_near(r.shaft_joint.hub_needed_um, 11.617, 5e-4, "needed");
    assert_near(r.shaft_joint.hub_fittable_max_um, 18.828, 5e-4, "fittable");
    assert_eq!(r.shaft_joint.check, SHAFT_JOINTS_PASS);
    // Inner B842 with outer B842SH: the inner B842's own 69.83 °C less 10.
    let mut inputs = passing();
    inputs.coupling.magnets.part_inner = "B842".to_owned();
    let r = demag_pins(&inputs, "inner", 69.8326, 69.8326);
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "59.8"));
    // 2214 Hi-Temp, the hub input at its 130 °C slider top and a 22 µm hub maximum, inner N42H
    // with outer SmCo_2_17_30 (grade ids) on manual dimensions: the outer SmCo governs with
    // 191.88 °C, the inner N42H's own onset is 112.83 °C. The same at `passing_design()` and at
    // the defaults.
    for base in [passing(), DesignInputs::default()] {
        let mut inputs = manual(base, "N42H", "SmCo_2_17_30");
        inputs.temperature.adhesive.selected = 3;
        inputs.shaft_joint.hub_fit_max_C = 130.0;
        inputs.shaft_joint.hub_interference_max_um = 22.0;
        let r = demag_pins(&inputs, "outer", 191.8772, 112.8297);
        assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "125.8", "102.8"));
    }
}

#[test]
fn a_long_magnet_makes_the_cup_slip_first() {
    // The hub grows with the magnets under the axial override, the boss does not: at 40 mm the
    // cup slips and its key follows; the cup starts to slip near a magnet length of 26.5 mm.
    let long = design(|i| i.coupling.magnets.axial_length_mm = Some(40.0));
    let verdict = check(&long);
    assert!(
        verdict.starts_with(&format!("{FIT_SLIPS} ({CUP})")),
        "{verdict}"
    );
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&terms(&long))
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(failing, [(Cause::Slips, CUP), (Cause::Key, CUP)]);
    for (length, slips_at_cup) in [(26.0, false), (27.0, true)] {
        let inputs = design(|i| i.coupling.magnets.axial_length_mm = Some(length));
        let [_, cup] = terms(&inputs);
        assert_eq!(cup.fails(Cause::Slips), slips_at_cup, "{length}");
    }
}

#[test]
fn exactly_at_each_limit_passes_and_the_next_float_fails() {
    // On `joint`, from the inner hub at the defaults: the need, the stress limit, the allowed
    // temperature and the key's limit each set exactly to the term compared with it.
    let (common, [hub, _]) = specs(&DesignInputs::default());
    let base = shaft_joint::joint(&hub, &common);
    let with = |common: JointCommon, spec: JointSpec| shaft_joint::joint(&spec, &common);
    // Slip: holding = required passes; a required torque one float above slips.
    let at = with(
        JointCommon {
            required_Nm: base.holding_Nm,
            ..common
        },
        hub,
    );
    assert!(!at.fails(Cause::Slips));
    let over = with(
        JointCommon {
            required_Nm: base.holding_Nm.next_up(),
            ..common
        },
        hub,
    );
    assert!(over.fails(Cause::Slips));
    // Stress: stress = limit passes; a limit one float below fails.
    let at = with(
        JointCommon {
            stress_limit_MPa: base.bore_stress_MPa,
            ..common
        },
        hub,
    );
    assert!(!at.fails(Cause::BoreStress));
    let under = with(
        JointCommon {
            stress_limit_MPa: base.bore_stress_MPa.next_down(),
            ..common
        },
        hub,
    );
    assert!(under.fails(Cause::BoreStress));
    // Fitting: needed = allowed passes; an allowed temperature one float below fails.
    let at = with(
        common,
        JointSpec {
            fit_allowed_C: base.fit_temp_C,
            ..hub
        },
    );
    assert!(!at.fails(Cause::CannotBeFitted));
    let under = with(
        common,
        JointSpec {
            fit_allowed_C: base.fit_temp_C.next_down(),
            ..hub
        },
    );
    assert!(under.fails(Cause::CannotBeFitted));
    // Key: with no interference (no bore stress) the limit set to the key pressure passes, the
    // key margin is 0 to rounding; a limit one float below fails.
    let unstressed = JointSpec {
        interference_max_um: 0.0,
        ..hub
    };
    let at = with(
        JointCommon {
            stress_limit_MPa: base.key_pressure_MPa,
            ..common
        },
        unstressed,
    );
    assert!(!at.fails(Cause::Key) && !at.fails(Cause::BoreStress));
    assert!(
        at.key_margin_mm.abs() <= 1e-12 * at.key_bearing_mm,
        "{}",
        at.key_margin_mm
    );
    let under = with(
        JointCommon {
            stress_limit_MPa: base.key_pressure_MPa.next_down(),
            ..common
        },
        unstressed,
    );
    assert!(under.fails(Cause::Key));
}

#[test]
fn a_failing_badge_never_quotes_two_equal_numbers() {
    // The BL-059 rounding (spec section 4.2): the limit half-up to its decimals, the failing
    // value pushed one last digit past it.
    let (common, [hub, _]) = specs(&DesignInputs::default());
    let base = shaft_joint::joint(&hub, &common);
    // A stress 0.04 MPa over the limit.
    let t = shaft_joint::joint(
        &hub,
        &JointCommon {
            stress_limit_MPa: base.bore_stress_MPa - 0.04,
            ..common
        },
    );
    assert_eq!(
        t.text(Cause::BoreStress),
        over_stress(INNER_HUB, "241.8", "241.7")
    );
    // A fitting temperature 0.04 °C over the allowed.
    let t = shaft_joint::joint(
        &JointSpec {
            fit_allowed_C: base.fit_temp_C - 0.04,
            ..hub
        },
        &common,
    );
    assert_eq!(
        t.text(Cause::CannotBeFitted),
        cannot_fit(INNER_HUB, "93.3", "93.2")
    );
    // A holding 0.004 N·m under the need.
    let t = shaft_joint::joint(
        &hub,
        &JointCommon {
            required_Nm: base.holding_Nm + 0.004,
            ..common
        },
    );
    assert_eq!(t.text(Cause::Slips), slips(INNER_HUB, "9.48", "9.49"));
    // A need under 0.005 N·m quotes as 0.00: the holding (0 here) is quoted 0.00, the need 0.01.
    let t = shaft_joint::joint(
        &JointSpec {
            interference_min_um: 0.0,
            ..hub
        },
        &JointCommon {
            required_Nm: 0.004,
            ..common
        },
    );
    assert!(t.fails(Cause::Slips));
    assert_eq!(t.text(Cause::Slips), slips(INNER_HUB, "0.00", "0.01"));
    // The key 0.04 MPa over its limit (no interference, so no bore stress).
    let t = shaft_joint::joint(
        &JointSpec {
            interference_max_um: 0.0,
            ..hub
        },
        &JointCommon {
            stress_limit_MPa: base.key_pressure_MPa - 0.04,
            ..common
        },
    );
    assert_eq!(t.text(Cause::Key), key_over(INNER_HUB, "127.5", "127.4"));
    // On the sliders: the hub's allowed temperature 93 °C against the 93.27 °C need.
    let slider_step = design(|i| i.shaft_joint.hub_fit_max_C = 93.0);
    assert_eq!(check(&slider_step), cannot_fit(INNER_HUB, "93.3", "93.0"));
    // Every limit around each value, across two last digits: the value quoted is above the
    // limit quoted (below the need quoted, for holding).
    for k in 0..=200 {
        let offset = f64::from(k) * 0.001;
        let stress = shaft_joint::joint(
            &hub,
            &JointCommon {
                stress_limit_MPa: base.bore_stress_MPa - offset - f64::EPSILON,
                ..common
            },
        );
        let (q, r) = quoted(&stress.text(Cause::BoreStress));
        assert!(q > r, "{offset}");
        let fit = shaft_joint::joint(
            &JointSpec {
                fit_allowed_C: base.fit_temp_C - offset - 1e-9,
                ..hub
            },
            &common,
        );
        let (q, r) = quoted(&fit.text(Cause::CannotBeFitted));
        assert!(q > r, "{offset}");
        let holding = shaft_joint::joint(
            &hub,
            &JointCommon {
                required_Nm: base.holding_Nm + offset * 0.1 + 1e-9,
                ..common
            },
        );
        let (q, r) = quoted(&holding.text(Cause::Slips));
        assert!(q < r, "{offset}");
    }
}

#[test]
fn a_value_that_is_not_a_number_reads_unknown() {
    // Spec section 4.4: NaN in any of the 15 inputs or any of the 15 engine values the check
    // reads, tested on the values themselves before any min or max. `set` refuses NaN; a struct
    // holds one.
    type PutInput = fn(&mut ShaftJointInputs);
    let inputs_nan: [PutInput; 15] = [
        |i| i.hub_interference_min_um = f64::NAN,
        |i| i.hub_interference_max_um = f64::NAN,
        |i| i.cup_interference_min_um = f64::NAN,
        |i| i.cup_interference_max_um = f64::NAN,
        |i| i.safety_factor = f64::NAN,
        |i| i.friction = f64::NAN,
        |i| i.smoothing_um = f64::NAN,
        |i| i.rotor_over_shaft_C = f64::NAN,
        |i| i.key_width_mm = f64::NAN,
        |i| i.key_contact_mm = f64::NAN,
        |i| i.hub_fit_max_C = f64::NAN,
        |i| i.cup_fit_max_C = f64::NAN,
        |i| i.wheel_shaft_fit_C = f64::NAN,
        |i| i.output_shaft_fit_C = f64::NAN,
        |i| i.joining_clearance_um = f64::NAN,
    ];
    for (k, put) in inputs_nan.into_iter().enumerate() {
        let mut inputs = DesignInputs::default();
        put(&mut inputs.shaft_joint);
        assert_eq!(check(&inputs), UNKNOWN_NOT_A_NUMBER, "input {k}");
    }
    type PutLink = fn(&mut ShaftJointLinks);
    let links_nan: [PutLink; 15] = [
        |l| l.cold_high_Nm = f64::NAN,
        |l| l.bore_mm = f64::NAN,
        |l| l.keyway_depth_mm = f64::NAN,
        |l| l.hub_length_mm = f64::NAN,
        |l| l.inner_back_apothem_mm = f64::NAN,
        |l| l.bond_inner_mm = f64::NAN,
        |l| l.boss_length_mm = f64::NAN,
        |l| l.web_mm = f64::NAN,
        |l| l.boss_od_mm = f64::NAN,
        |l| l.min_temp_C = f64::NAN,
        |l| l.adhesive_limit_C = f64::NAN,
        |l| l.inner_onset_single_ring_C = f64::NAN,
        |l| l.part_modulus_GPa = f64::NAN,
        |l| l.part_cte_per_C = f64::NAN,
        |l| l.part_yield_MPa = f64::NAN,
    ];
    let defaults = DesignInputs::default();
    let base = links(&defaults);
    assert_eq!(
        shaft_joint::compute(&defaults.shaft_joint, &base).check,
        SHAFT_JOINTS_PASS
    );
    for (k, put) in links_nan.into_iter().enumerate() {
        let mut links = base;
        put(&mut links);
        assert!(links.values()[k].is_nan(), "link {k} is the {k}th value");
        assert_eq!(
            shaft_joint::compute(&defaults.shaft_joint, &links).check,
            UNKNOWN_NOT_A_NUMBER,
            "link {k}"
        );
    }
    // Through the engine: no adhesive (code 0: a NaN design limit), a NaN single-ring reverse
    // field (a NaN inner onset), and a back-iron pick outside the choices on the steel circuit
    // (NaN part values).
    let no_adhesive = design(|i| i.temperature.adhesive.selected = 0);
    assert_eq!(check(&no_adhesive), UNKNOWN_NOT_A_NUMBER);
    let nan_field = compute_all(&design(|i| {
        i.temperature.demag.h_rev_single_ring_kA_m = f64::NAN;
    }));
    assert!(
        nan_field
            .temperature
            .demag
            .inner_onset_single_ring_C
            .is_nan()
    );
    assert!(nan_field.shaft_joint.hub_fit_allowed_C.is_nan());
    assert_eq!(nan_field.shaft_joint.check, UNKNOWN_NOT_A_NUMBER);
    let unknown_pick = compute_all(&design(|i| i.materials.parts.back_iron = 99)).shaft_joint;
    assert!(
        unknown_pick.part_modulus_GPa.is_nan()
            && unknown_pick.part_cte_per_C.is_nan()
            && unknown_pick.part_yield_MPa.is_nan()
    );
    assert_eq!(unknown_pick.check, UNKNOWN_NOT_A_NUMBER);
}

#[test]
fn a_nan_hub_limit_makes_the_allowed_temperature_nan_not_the_adhesive_s() {
    // Review 2, SF-3: `f64::min(NaN, 120)` is 120, so a NaN hub input would otherwise show an
    // allowed 120 °C and a fittable 21.29 µm while the verdict reads unknown.
    assert_eq!(f64::NAN.min(120.0), 120.0);
    let s = compute_all(&design(|i| i.shaft_joint.hub_fit_max_C = f64::NAN)).shaft_joint;
    assert!(s.hub_fit_allowed_C.is_nan() && s.hub_fittable_max_um.is_nan());
    assert_eq!(s.check, UNKNOWN_NOT_A_NUMBER);
}

#[test]
fn the_allowed_temperature_helper_drops_no_nan_and_names_the_limit_that_binds() {
    // Owner R-15, on `hub_fit_allowed`: py_min carries a NaN only in its first argument, so a
    // NaN anywhere is tested first.
    assert_eq!(py_min(100.0, f64::NAN), 100.0);
    for (input, adhesive, onset) in [
        (f64::NAN, 120.0, 143.0),
        (100.0, f64::NAN, 143.0),
        (100.0, 120.0, f64::NAN),
    ] {
        let (allowed, limit) = hub_fit_allowed(input, adhesive, onset);
        assert!(
            allowed.is_nan() && limit.is_none(),
            "{input} {adhesive} {onset}"
        );
    }
    let cap = |onset: f64| onset - CURE_MARGIN_MIN_C;
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 143.0),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 60.0, 143.0),
        (60.0, Some(HubFitLimit::Adhesive))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 69.0),
        (cap(69.0), Some(HubFitLimit::MagnetOnset))
    );
    // Ties name the earlier limit: an input equal to the adhesive's limit, and an onset exactly
    // the input plus the margin, name your limit; the onset tied with the adhesive names it.
    assert_eq!(
        hub_fit_allowed(120.0, 120.0, 143.0),
        (120.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 100.0 + CURE_MARGIN_MIN_C),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(130.0, 120.0, 120.0 + CURE_MARGIN_MIN_C),
        (120.0, Some(HubFitLimit::Adhesive))
    );
    // An onset of +inf (ferrite) never binds; one below 30 °C allows less than 20 °C.
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, f64::INFINITY),
        (100.0, Some(HubFitLimit::Yours))
    );
    assert_eq!(
        hub_fit_allowed(100.0, 120.0, 25.0),
        (cap(25.0), Some(HubFitLimit::MagnetOnset))
    );
    // Through the engine: the single-ring field at its slider's top (1500 kA/m) puts the inner
    // onset at -2.449 °C, so -12.45 °C allowed, quoted by the same never-equal rule.
    let top = slider("temperature.demag.h_rev_single_ring_kA_m").max;
    assert_eq!(top, 1500.0);
    let r = compute_all(&design(|i| {
        i.temperature.demag.h_rev_single_ring_kA_m = top
    }));
    assert_near(
        r.temperature.demag.inner_onset_single_ring_C,
        -2.449,
        5e-4,
        "onset",
    );
    assert_eq!(r.shaft_joint.check, cannot_fit(INNER_HUB, "93.3", "-12.4"));
}

#[test]
fn a_reversed_band_reads_unknown_unless_the_other_joint_fails() {
    // Review SF-8: a reversed band is not judged and reads unknown, naming the joint; its terms
    // are still computed. A judged joint's failure outranks it.
    let reversed = design(|i| i.shaft_joint.hub_interference_min_um = 20.0);
    assert_eq!(
        check(&reversed),
        format!("{UNKNOWN_BAND_REVERSED} ({INNER_HUB})")
    );
    let [hub, cup] = terms(&reversed);
    assert!(hub.band_reversed && !hub.determinate() && cup.determinate());
    assert_near(hub.holding_Nm, 23.49, 5e-3, "the terms are computed");
    assert!(shaft_joint::failing(&[hub, cup]).is_empty());
    let and_cup_slips = design(|i| {
        i.shaft_joint.hub_interference_min_um = 20.0;
        i.shaft_joint.cup_interference_min_um = 10.0;
    });
    assert_eq!(check(&and_cup_slips), slips(CUP, "4.36", "7.65"));
}

#[test]
fn a_joint_with_no_wall_reads_nan_and_is_not_judged() {
    // Review SF-8, SF-9: bore 16 mm in a 12 mm boss leaves the cup no wall (Q >= 1). The inner
    // hub is judged and slips; the cup's Lamé rows read NaN, its key, fittable and fitting rows
    // numbers.
    let no_cup_wall = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.metal.boss_od_mm = 12.0;
    });
    let s = compute_all(&no_cup_wall).shaft_joint;
    assert_eq!(s.check, slips(INNER_HUB, "4.57", "7.65"));
    assert!(
        s.cup_holding_Nm.is_nan()
            && s.cup_bore_stress_MPa.is_nan()
            && s.cup_needed_um.is_nan()
            && s.cup_elastic_max_um.is_nan()
    );
    assert!(
        s.cup_key_pressure_MPa.is_finite()
            && s.cup_fittable_max_um.is_finite()
            && s.cup_fit_temp_C.is_finite()
    );
    let [hub, cup] = terms(&no_cup_wall);
    assert_eq!(
        (hub.no_wall, cup.no_wall),
        (None, Some(NoWall::BoreAsWideAsPart))
    );
    // With the hub's minimum at 15 µm it holds: the cup's missing wall is the verdict.
    let mut holding = no_cup_wall.clone();
    holding.shaft_joint.hub_interference_min_um = 15.0;
    assert_eq!(check(&holding), format!("{UNKNOWN_NO_WALL} ({CUP})"));
    // Bore 16 mm with a 2.1 mm keyway: the hub's 2.10 mm wall is cut through; at 2.05 mm it is
    // not, and the hub slips.
    let cut = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.coupling.keyway_depth_mm = 2.1;
    });
    let [hub, cup] = terms(&cut);
    assert_eq!(
        (hub.no_wall, cup.no_wall),
        (Some(NoWall::KeywayThroughWall), None)
    );
    assert_eq!(check(&cut), format!("{UNKNOWN_NO_WALL} ({INNER_HUB})"));
    let uncut = design(|i| {
        i.coupling.bore_mm = 16.0;
        i.coupling.keyway_depth_mm = 2.05;
    });
    assert_eq!(check(&uncut), slips(INNER_HUB, "4.57", "7.65"));
    // A part diameter typed at or below zero has no wall either (ruling C-3, plan deviation
    // PD-1): keyless, with the boss typed at -22 mm, the inner hub passes and the cup is not
    // judged, where `q >= 1` would have judged it with the numbers of a +22 mm boss.
    let negative_boss = design(|i| {
        i.coupling.keyway_depth_mm = 0.0;
        i.metal.boss_od_mm = -22.0;
    });
    assert_eq!(check(&negative_boss), format!("{UNKNOWN_NO_WALL} ({CUP})"));
}

#[test]
fn the_key_alone_is_a_backup_with_its_own_edges() {
    // Keyway depth 0.3 mm (on its slider's 0.05 mm grid): the contact height is capped at
    // 0.3 mm. Both fits hold, both stresses and fitting temperatures pass, so the key is the
    // badge's first cause; the cup's key (407.8 MPa) follows.
    let shallow = design(|i| i.coupling.keyway_depth_mm = 0.3);
    assert_eq!(check(&shallow), key_over(INNER_HUB, "509.8", "345.8"));
    let joints = terms(&shallow);
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&joints)
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(failing, [(Cause::Key, INNER_HUB), (Cause::Key, CUP)]);
    assert_near(joints[1].key_pressure_MPa, 407.82, 5e-3, "cup key");
    // The contact height is capped at the keyway depth: 152.9 / 122.3 MPa at 1.0 mm.
    let s = compute_all(&design(|i| i.coupling.keyway_depth_mm = 1.0)).shaft_joint;
    assert_near(s.hub_key_pressure_MPa, 152.93, 5e-3, "hub key");
    assert_near(s.cup_key_pressure_MPa, 122.35, 5e-3, "cup key");
    // No keyway: no key term and no arc taken off (10.49 / 18.68 N·m); the key rows read 0 and
    // both key terms pass with margin +inf.
    let keyless = design(|i| i.coupling.keyway_depth_mm = 0.0);
    let s = compute_all(&keyless).shaft_joint;
    assert_near(s.hub_holding_Nm, 10.486, 5e-3, "keyless hub");
    assert_near(s.cup_holding_Nm, 18.683, 5e-3, "keyless cup");
    assert_eq!((s.hub_key_pressure_MPa, s.cup_key_pressure_MPa), (0.0, 0.0));
    for joint in terms(&keyless) {
        assert!(!joint.fails(Cause::Key));
        assert_eq!(joint.margin(Cause::Key), f64::INFINITY);
    }
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
    // A contact height typed -0.5 mm: no bearing area, +inf (never -305.9 MPa, a pass).
    let negative = design(|i| i.shaft_joint.key_contact_mm = -0.5);
    let s = compute_all(&negative).shaft_joint;
    assert_eq!(s.hub_key_pressure_MPa, f64::INFINITY);
    assert_eq!(
        check(&negative),
        format!("{KEY_HAS_NO_BEARING_AREA} ({INNER_HUB})")
    );
    // A contact height typed 0 (shared, so both keys) with the boss at 0: neither key has a
    // bearing area, and each key margin is -inf until the contact is above zero (review 2,
    // SF-5).
    let no_contact = design(|i| {
        i.shaft_joint.key_contact_mm = 0.0;
        i.metal.boss_length_mm = 0.0;
    });
    for joint in terms(&no_contact) {
        assert!(!joint.key_has_bearing_area(), "{}", joint.name);
        assert_eq!(
            joint.margin(Cause::Key),
            f64::NEG_INFINITY,
            "{}",
            joint.name
        );
    }
    // Boss length 0 and 0.5 mm (the cup's 2.5 + 0.5 = 3 mm, the key's width): no bearing
    // length, so no bearing area, never a negative pressure that passes.
    for boss in [0.0, 0.5] {
        let [_, cup] = terms(&design(|i| i.metal.boss_length_mm = boss));
        assert!(!cup.key_has_bearing_area(), "{boss}");
        assert_eq!(cup.key_pressure_MPa, f64::INFINITY, "{boss}");
        assert!(cup.fails(Cause::Key), "{boss}");
        assert_eq!(
            cup.text(Cause::Key),
            format!("{KEY_HAS_NO_BEARING_AREA} ({CUP})")
        );
    }
    // The inner hub at its slider's bottom length, 3.0 mm (L = b): holding 2.189 N·m, the key
    // +inf; both causes, the slip in the badge.
    let short_hub = design(|i| i.metal.hub_length_mm = 3.0);
    let s = compute_all(&short_hub).shaft_joint;
    assert_near(s.hub_holding_Nm, 2.189, 5e-4, "short hub");
    assert_eq!(s.hub_key_pressure_MPa, f64::INFINITY);
    assert_eq!(s.check, slips(INNER_HUB, "2.19", "7.65"));
    let failing: Vec<(Cause, &str)> = shaft_joint::failing(&terms(&short_hub))
        .into_iter()
        .map(|(cause, j)| (cause, j.name))
        .collect();
    assert_eq!(
        failing,
        [(Cause::Slips, INNER_HUB), (Cause::Key, INNER_HUB)]
    );
}

#[test]
fn each_margin_is_finite_where_its_term_can_fail_and_its_sign_is_the_verdict() {
    // Review MF-2: the key margin (bearing length less the length needed) is finite at boss
    // lengths 0 to 40 mm and its sign matches the key verdict, at contact heights 0.5, 0.8,
    // 1.2 and 1.7 mm. Review SF-1: the slip margin's sign matches the holding against the
    // required torque over the hub minimum 0 to 40 µm.
    let (common, [hub, cup]) = specs(&DesignInputs::default());
    let web = DesignInputs::default().metal.web_mm;
    for contact in [0.5, 0.8, 1.2, 1.7] {
        let common = JointCommon {
            key_contact_mm: contact,
            ..common
        };
        for boss in slider_values("metal.boss_length_mm") {
            let t = shaft_joint::joint(
                &JointSpec {
                    length_mm: boss + web,
                    ..cup
                },
                &common,
            );
            let margin = t.margin(Cause::Key);
            assert!(margin.is_finite(), "{contact} {boss}");
            assert_eq!(
                t.fails(Cause::Key),
                margin < 0.0,
                "{contact} {boss}: {margin}"
            );
        }
    }
    for minimum in slider_values("shaft_joint.hub_interference_min_um") {
        let t = shaft_joint::joint(
            &JointSpec {
                interference_min_um: minimum,
                ..hub
            },
            &common,
        );
        let margin = t.margin(Cause::Slips);
        assert!(margin.is_finite(), "{minimum}");
        assert_eq!(t.fails(Cause::Slips), margin < 0.0, "{minimum}: {margin}");
    }
    // The stress and fitting margins are the limit less the term.
    let t = shaft_joint::joint(&hub, &common);
    assert_eq!(
        t.margin(Cause::BoreStress),
        t.stress_limit_MPa - t.bore_stress_MPa
    );
    assert_eq!(
        t.margin(Cause::CannotBeFitted),
        t.fit_allowed_C - t.fit_temp_C
    );
}

#[test]
fn the_effective_interference_is_floored_at_zero() {
    // The 304 inner hub loses more than its minimum at the hot end: no grip, not a negative one.
    let [hub, _] = terms(&design(|i| i.materials.parts.back_iron = 7));
    assert_eq!(hub.effective_min_um, 0.0);
    assert_eq!(hub.holding_Nm, 0.0);
}

#[test]
fn a_shaft_above_room_temperature_grows_and_a_cooled_one_shrinks_as_nist_s_steel_does() {
    // Spec section 3.7, Ø10, diametral [µm]: the NIST Fe-2.25Ni fit below 265 K, the linear
    // 11.5e-6 /°C above.
    for (shaft, want) in [
        (40.0, -2.30),
        (-20.0, 4.55),
        (-40.0, 6.66),
        (-78.5, 10.35),
        (-196.0, 18.99),
    ] {
        assert_near(
            shaft_shrink_um(shaft, 10.0),
            want,
            5e-3,
            &format!("{shaft} °C"),
        );
    }
    assert_eq!(shaft_shrink_um(ROOM_C, 10.0), 0.0);
    // Continuous at 265 K and strictly falling as the shaft warms over the wheel shaft's slider.
    let edge = 265.0 - 273.15; // °C
    assert!((shaft_shrink_um(edge, 10.0) - shaft_shrink_um(edge - 1e-9, 10.0)).abs() < 1e-9);
    let values = slider_values("shaft_joint.wheel_shaft_fit_C");
    for pair in values.windows(2) {
        assert!(
            shaft_shrink_um(pair[1], 10.0) < shaft_shrink_um(pair[0], 10.0),
            "{pair:?}"
        );
    }
    // Through the engine: the gearbox shaft at 40 °C grows 2.30 µm; the cup still fits at
    // 298.9 °C against 300 °C.
    let s = compute_all(&design(|i| i.shaft_joint.output_shaft_fit_C = 40.0)).shaft_joint;
    assert_near(s.output_shaft_shrink_um, -2.30, 5e-3, "grows");
    assert_near(s.cup_fit_temp_C, 298.86, 5e-3, "cup fit temperature");
    assert_eq!(s.check, SHAFT_JOINTS_PASS);
}

#[test]
fn an_infinite_cold_high_slips() {
    // +inf fails where it is compared: the required torque +inf slips at the inner hub first.
    let defaults = DesignInputs::default();
    let infinite = ShaftJointLinks {
        cold_high_Nm: f64::INFINITY,
        ..links(&defaults)
    };
    assert_eq!(
        shaft_joint::compute(&defaults.shaft_joint, &infinite).check,
        slips(INNER_HUB, "9.48", "inf")
    );
}

#[test]
fn the_joints_rebuilt_from_the_results_are_the_engine_s() {
    // `api::shaft_joint_links` reads every value from the design's inputs and results; the check
    // recomputed from it is the engine's, at the probe designs and at the edge designs above.
    let mut designs: Vec<DesignInputs> = probe_designs().into_iter().map(|(_, d)| d).collect();
    designs.extend([
        passing(),
        design(|i| i.coupling.magnets.axial_length_mm = Some(40.0)),
        design(|i| {
            i.coupling.bore_mm = 16.0;
            i.metal.boss_od_mm = 12.0;
        }),
        design(|i| i.coupling.backiron = 0),
        design(|i| i.temperature.adhesive.selected = 0),
    ]);
    for inputs in designs {
        let r = compute_all(&inputs);
        let again = shaft_joint::compute(&inputs.shaft_joint, &shaft_joint_links(&inputs, &r));
        // NaN equal to NaN: compare the debug texts.
        assert_eq!(format!("{again:?}"), format!("{:?}", r.shaft_joint));
    }
}

#[test]
fn the_shaft_joint_inputs_move_no_result_outside_the_check() {
    // Spec section 7 item 3, in the spirit of the paused clamp's guard: each `shaft_joint.*`
    // input at both slider ends, at the five probe designs, changes only `shaft_joint.*` results
    // (NaN equal to NaN).
    let outside = |inputs: &DesignInputs| -> Vec<(String, String)> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .filter(|row| !row.path.starts_with("shaft_joint."))
            .map(|row| (row.path, format!("{:?}", row.value)))
            .collect()
    };
    let mut moved_inside = 0;
    for (name, design) in probe_designs() {
        let base = outside(&design);
        let base_check = compute_all(&design).shaft_joint;
        for row in input_rows(&design)
            .into_iter()
            .filter(|row| row.path.starts_with("shaft_joint."))
        {
            let range = row.meta.range.expect("a slider");
            for value in [range.min, range.max] {
                let mut edited = design.clone();
                edited
                    .set(&row.path, Value::Num(value))
                    .expect("a value the input accepts");
                assert_eq!(outside(&edited), base, "{name}: {} at {value}", row.path);
                moved_inside += usize::from(compute_all(&edited).shaft_joint != base_check);
            }
        }
    }
    // Not vacuous: the inputs move the check's own results.
    assert!(moved_inside > 100, "{moved_inside}");
}
```

`magcoupling-rs/tests/robustness.rs` (spec 7.2: no adhesive reads unknown; the degenerate typed inputs, review 2 SF-4):

Old:
```rust
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, NumOrText, SetErrorKind, Value, input_rows};

#[test]
```

New:
```rust
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, NumOrText, SetErrorKind, Value, input_rows};
use magcoupling::engine::shaft_joint::{FIT_SLIPS, UNKNOWN_NOT_A_NUMBER, UNKNOWN_SAFETY_FACTOR};

#[test]
```

Old:
```rust
            "{code}"
        );
    }
}

```

New:
```rust
            "{code}"
        );
        // BL-064: no adhesive is no design limit, so the bonded hub has no allowed fitting
        // temperature and the shaft joint check cannot be made.
        let s = compute_all_with(&inputs, Deviations::NONE).shaft_joint;
        assert!(s.hub_fit_allowed_C.is_nan(), "{code}");
        assert_eq!(s.check, UNKNOWN_NOT_A_NUMBER, "{code}");
    }
}

#[test]
fn degenerate_shaft_joint_inputs_never_pass() {
    // BL-064, review 2 SF-4: typed values no slider reaches. A bore of 0 gives an infinite
    // contact pressure over no contact, a NaN holding from finite inputs: unknown, never a
    // pass. A key wider than the bore's circumference (40 mm on Ø10) leaves no contact arc:
    // holding 0 and a needed minimum of +inf, never a negative holding or window end. A safety
    // factor of 0 or -1 reads unknown, never a pass with a key pressure below zero (ruling C-4,
    // plan deviation PD-2).
    let mut zero_bore = DesignInputs::default();
    zero_bore.coupling.bore_mm = 0.0;
    let s = compute_all(&zero_bore).shaft_joint;
    assert!(s.hub_holding_Nm.is_nan());
    assert_eq!(s.check, UNKNOWN_NOT_A_NUMBER);
    let mut wide_key = DesignInputs::default();
    wide_key.shaft_joint.key_width_mm = 40.0;
    let s = compute_all(&wide_key).shaft_joint;
    assert_eq!((s.hub_holding_Nm, s.cup_holding_Nm), (0.0, 0.0));
    assert_eq!(
        (s.hub_needed_um, s.cup_needed_um),
        (f64::INFINITY, f64::INFINITY)
    );
    assert_eq!(
        s.check,
        format!("{FIT_SLIPS} (inner hub): holds 0.00 N·m against 7.65 N·m")
    );
    for factor in [0.0, -1.0] {
        let mut inputs = DesignInputs::default();
        inputs.shaft_joint.safety_factor = factor;
        let s = compute_all(&inputs).shaft_joint;
        assert_eq!(s.check, UNKNOWN_SAFETY_FACTOR, "{factor}");
    }
}

```

`magcoupling-rs/tests/schema.rs`, `engine_owned_help_is_plain_and_short`: a Rust-only input and the check join the not-vacuous list, so the 15 labels and helps and the 24 are held to the plain-words guard and the help cap (spec 7.3).

Old:
```rust
        "gearbox.cold_high_check",
        "temperature.demag.inner_onset_single_ring_C",
        "metal.steel_cup_mass_g",
        "clamps.table[*].length_mm",
```

New:
```rust
        "gearbox.cold_high_check",
        "temperature.demag.inner_onset_single_ring_C",
        "shaft_joint.hub_interference_min_um",
        "shaft_joint.check",
        "metal.steel_cup_mass_g",
        "clamps.table[*].length_mm",
```

`magcoupling-rs/src/gui/session.rs` (spec 5.4, 7.16): a file and a link written before BL-064 load the new inputs at their defaults (no `DESIGN_VERSION` bump: a reader takes a missing path at its default, `session.rs:59-61`), and the share-link guard moves from 2,500 to 2,700 characters in the same commit as the inputs, quoting both lengths (2,643 measured: the 15 inputs at their defaults, as the spec's encoder found).

Old:
```rust

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quotes this length (2,462 characters at the time of writing): the
        // whole design is in the link.
        let payload = encode_share_payload(&Design::default());
        assert!(payload.len() < 2_500, "{} characters", payload.len());
    }

```

New:
```rust

    #[test]
    fn a_design_file_and_a_share_link_from_before_the_shaft_joints_load_them_at_their_defaults() {
        // BL-064 adds fifteen inputs with no version bump: a reader takes a missing path at its
        // default (decision M41-5). The link above was written before them; so is this file,
        // the link's own text with one input edited.
        let bytes = LINK_BASE64.decode(DEFAULT_PAYLOAD_49FBD3F).unwrap();
        let mut text = String::new();
        DeflateDecoder::new(&bytes[..])
            .read_to_string(&mut text)
            .unwrap();
        let json: Json = serde_json::from_str(&text).unwrap();
        let inputs = json["inputs"].as_object().unwrap();
        assert!(!inputs.keys().any(|path| path.starts_with("shaft_joint.")));
        assert_eq!(inputs["coupling.npole"], 10);
        let older = text.replace("\"coupling.npole\":10", "\"coupling.npole\":12");
        assert_ne!(older, text, "the file names the pole count");
        for (loaded, npole) in [
            (decode_share_payload(DEFAULT_PAYLOAD_49FBD3F).unwrap(), 10),
            (design_from_json(&older).unwrap(), 12),
        ] {
            assert_eq!(loaded.inputs.coupling.npole, npole);
            assert_eq!(
                loaded.inputs.shaft_joint,
                crate::engine::shaft_joint::ShaftJointInputs::default()
            );
        }
    }

    #[test]
    fn a_share_link_of_the_default_design_stays_short() {
        // Decision M41-5 quoted this length as 2,462 characters; BL-064's fifteen shaft joint
        // inputs take it to 2,643 (the whole design is in the link), so the guard moves from
        // 2,500 to 2,700.
        let payload = encode_share_payload(&Design::default());
        assert!(payload.len() < 2_700, "{} characters", payload.len());
    }

```

- [ ] **Step 2: Run them and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test shaft_joint --test robustness`

Expected: neither compiles:
```text
error[E0432]: unresolved imports `magcoupling::engine::shaft_joint`, `magcoupling::engine::shaft_joint`
error[E0432]: unresolved import `magcoupling::engine::api::shaft_joint_links`
error[E0609]: no field `shaft_joint` on type `DesignResults`      (41 of these; 33 more on `DesignInputs`, `&DesignInputs` and `&mut DesignInputs`)
error: could not compile `magcoupling-rs` (test "shaft_joint") due to 76 previous errors
error: could not compile `magcoupling-rs` (test "robustness") due to 7 previous errors
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema engine_owned_help_is_plain_and_short`

Expected: `shaft_joint.hub_interference_min_um is not engine-owned`, `test result: FAILED. 0 passed; 1 failed`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib session`

Expected: the library's tests do not compile:
```text
error[E0433]: failed to resolve: could not find `shaft_joint` in `engine`
error[E0609]: no field `shaft_joint` on type `api::DesignInputs`
error: could not compile `magcoupling-rs` (lib test) due to 2 previous errors
```

- [ ] **Step 3: Create the engine module**

Create `magcoupling-rs/src/engine/shaft_joint.rs` exactly as below. Notes on choices the spec leaves open:
- The unknown verdicts are built by a private `unknown!` macro over `concat!`, so the prefix `SHAFT_JOINT_UNKNOWN` is typed once and each reason is the spec's (`the_verdict_texts_are_the_spec_s` pins them).
- `JointTerms::fails` keeps the spec's NaN-failing form `!(holding >= required)` (section 4.3); clippy's `neg_cmp_op_on_partial_ord` rejects it under `-D warnings`, so that one method carries `#[allow(clippy::neg_cmp_op_on_partial_ord)]` with its reason (plan deviation PD-7).
- The NIST arithmetic is the spec scratch crate's, term for term, so its (C) numbers reproduce bit for bit (18.988 µm at -196 °C).
- `quoted_above`/`quoted_below` are the one helper pair of spec 4.2 for the four texts; where the need quotes as 0.00 N·m the holding is quoted 0.00 and the need 0.01.
- The 15 inputs carry no `.assumption()` here: `every_flagged_input_is_in_exactly_one_row` (`tests/assumptions.rs:60`) requires each flagged input in an `ASSUMPTIONS` row, which Task 3 writes with the flags (plan deviation PD-5).
- Rulings C-3 and C-4 (plan deviations PD-1 and PD-2): `joint`'s no-wall test is `!(spec.outer_diameter_mm > d)`, not `q >= 1.0`, so a part diameter typed at or below zero has no wall and is not judged (for positive diameters it is the same test); `verdict` returns `UNKNOWN_SAFETY_FACTOR` right after the NaN test, where `!(inputs.safety_factor > 0.0)`, with no fix in the guide. Both keep the NaN-failing form of `fails`, so each statement carries the same `allow` (clippy rejects the form under `-D warnings`; a statement-level `allow` was checked with clippy 0.1.89 and rustfmt).

```rust
//! BL-064: the shaft joints. Both rotors are held on their shafts by a keyed shrink fit (owner,
//! 2026-10-17: "Hub is shrink fit with key only"): the inner hub, its magnets bonded on its
//! flats, on the wheel shaft, and the cup, its magnets in its pocket, its rear web and integral
//! boss, on the gearbox output shaft. Both sit on the one bore (`coupling.bore_mm`). Cold-high
//! (`metal.torque_cold_high_Nm`: the pull-out at the minimum magnet temperature with the
//! production allowance) can reach each joint before the coupling slips, and while it slips
//! each joint carries a fully reversing swing.
//!
//! The rule (owner decision 2A): the fit alone must hold the safety factor times cold-high at
//! its loosest; the part must stay within its stress limit, yield ÷ [`YIELD_FACTOR`], at its
//! tightest; the key alone, a backup, is checked at the same torque against the same limit. The
//! fit's and the key's capacities are never added. Each part must also be fitted within its
//! allowed temperature (owner decision 7A) in the owner's order: the bare cup is shrunk onto the
//! gearbox shaft before its magnets are bonded; the inner hub is bonded first, then shrunk onto
//! the cooled wheel shaft, so its allowed temperature is also capped by the adhesive's design
//! limit and by the inner magnets' own single-ring onset less the cure margin (owner R-15 and
//! its ring ruling, [`hub_fit_allowed`]).
//!
//! Each joint is judged on four terms, in this order ([`CAUSES`]): the fit slips, the bore
//! stress is above its limit, it cannot be fitted at the allowed temperature, the key backup
//! (no bearing area, or above the limit). The badge shows the first failing term, naming the
//! first failing joint (inner hub, then cup). A joint with no wall around its bore or with its
//! band reversed is not judged; it reads unknown only when no judged joint fails. A value that
//! is not a number reads unknown before all of them, then a safety factor not above zero
//! ([`verdict`]). Equality passes; a failing badge never quotes two equal numbers (the BL-059
//! rounding).
//!
//! The temperature model: the interference moves with temperature by the expansion mismatch of
//! shaft and part. The loosest fit takes the end that loosens it more: the minimum magnet
//! temperature, or the selected adhesive's design limit with the rotor the rotor-over-shaft
//! allowance hotter than its shaft. The tightest fit takes the end that tightens it more. The
//! shafts' shrink at fitting comes from NIST's expansion fit for a nickel steel below 265 K.
//!
//! Not checked: fretting and fatigue of the joints under the reversing swing (BL-070), the
//! gearbox's own shaft, bearings and seal, the keyway corner's stress concentration, the bond
//! strain from fitting a bonded hub, and the key's shear.

use std::f64::consts::PI;

use super::compat::{fmt_fixed, py_max, py_min};
use super::meta::{InputSet, Value, inputs, out_rust_only, param_rust_only, results};
use super::temperature::CURE_MARGIN_MIN_C;

inputs! {
    /// The shaft joints' inputs (BL-064). Rust-only: no workbook cell.
    pub struct ShaftJointInputs {
        fields {
            hub_interference_min_um: f64 = 13.0 => param_rust_only("µm", "Inner hub interference, minimum",
                "The smallest measured diametral interference of the inner hub on the wheel shaft: shaft diameter less bore, both measured. Holding is checked at it.")
                .range(0.0, 40.0, 0.5),
            hub_interference_max_um: f64 = 18.0 => param_rust_only("µm", "Inner hub interference, maximum",
                "The largest measured diametral interference of the inner hub. Its bore stress and fitting temperature are checked at it.")
                .range(0.0, 40.0, 0.5),
            cup_interference_min_um: f64 = 15.0 => param_rust_only("µm", "Cup interference, minimum",
                "The smallest measured diametral interference of the cup's boss on the gearbox shaft. Holding is checked at it.")
                .range(0.0, 40.0, 0.5),
            cup_interference_max_um: f64 = 22.0 => param_rust_only("µm", "Cup interference, maximum",
                "The largest measured diametral interference of the cup. Its bore stress and fitting temperature are checked at it.")
                .range(0.0, 40.0, 0.5),
            safety_factor: f64 = 2.0 => param_rust_only("-", "Shaft joint safety factor",
                "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. The torque reverses many times in each slip.")
                .range(1.0, 4.0, 0.1),
            friction: f64 = 0.14 => param_rust_only("-", "Shrink-fit friction coefficient",
                "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it.")
                .range(0.05, 0.30, 0.01),
            smoothing_um: f64 = 5.0 => param_rust_only("µm", "Surface smoothing of the fit",
                "Interference lost as surface peaks flatten when fitted. It lowers both the grip and the bore stress; a ground shaft and honed bore lose less.")
                .range(0.0, 15.0, 0.1),
            rotor_over_shaft_C: f64 = 20.0 => param_rust_only("°C", "Rotor hotter than its shaft, running",
                "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls.")
                .range(0.0, 60.0, 1.0),
            key_width_mm: f64 = 3.0 => param_rust_only("mm", "Shaft key width",
                "Width of the parallel key in both joints (DIN 6885-1 gives 3 mm for a 10 mm shaft). Its slot takes this much of the fit's contact.")
                .range(1.0, 6.0, 0.5),
            key_contact_mm: f64 = 1.2 => param_rust_only("mm", "Shaft key contact height",
                "Height of the key face bearing on the inner hub and the cup, capped at the keyway depth (DIN 6885-1 gives 1.2 mm for 10 mm).")
                .range(0.5, 3.0, 0.1),
            hub_fit_max_C: f64 = 100.0 => param_rust_only("°C", "Hottest the bonded inner hub may be for fitting",
                "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and at the magnets' single-ring onset less 10 °C.")
                .range(20.0, 130.0, 1.0),
            cup_fit_max_C: f64 = 300.0 => param_rust_only("°C", "Hottest the bare cup may be for fitting",
                "The cup is fitted before its magnets are bonded. Stay below the steel's and its finish's treatment temperatures, and keep the heat off the gearbox seal.")
                .range(20.0, 400.0, 5.0),
            wheel_shaft_fit_C: f64 = -196.0 => param_rust_only("°C", "Wheel shaft temperature at fitting",
                "A cooled shaft shrinks, so the inner hub needs less heat. Dry ice is about -78 °C; liquid nitrogen is -196 °C.")
                .range(-196.0, 40.0, 1.0),
            output_shaft_fit_C: f64 = 20.0 => param_rust_only("°C", "Gearbox shaft temperature at fitting",
                "The gearbox output shaft cannot be immersed. Cold spray can take it to about -20 °C, the slider's bottom; never below the gearbox's rated range.")
                .range(-20.0, 40.0, 1.0),
            joining_clearance_um: f64 = 10.0 => param_rust_only("µm", "Joining clearance",
                "Clearance each part needs over its shaft to slide on before it grips; one value for both joints. Less needs a guided, single fast push to a stop.")
                .range(0.0, 30.0, 0.5),
        }
    }
}

results! {
    /// The shaft joint check (BL-064). Rust-only.
    pub struct ShaftJointResults {
        fields {
            required_Nm: f64 => out_rust_only("N·m", "Torque each shaft joint must hold",
                "Cold-high (the pull-out at the minimum magnet temperature, with the production allowance) times the safety factor. Each fit alone must hold it; the key alone is checked at it."),
            part_modulus_GPa: f64 => out_rust_only("GPa", "Inner hub and cup elastic modulus",
                "Of the inner hub and cup material in effect: the back-iron steel with a steel circuit, else the hub body's material. The fit's contact pressure rises with it."),
            part_cte_per_C: f64 => out_rust_only("1/°C", "Inner hub and cup expansion coefficient",
                "Of the inner hub and cup material in effect. Against the shafts' steel, a higher value loosens the fit hot and a lower one loosens it cold."),
            part_yield_MPa: f64 => out_rust_only("MPa", "Inner hub and cup yield strength",
                "The library's yield strength of the inner hub and cup material in effect; for 4140, the annealed value, conservative while the condition is open."),
            stress_limit_MPa: f64 => out_rust_only("MPa", "Bore stress limit",
                "The part's yield strength with a margin. At the tightest fit the bore stress must not exceed it, so the part stays elastic and keeps its grip."),
            hub_holding_Nm: f64 => out_rust_only("N·m", "Inner hub holding torque at its loosest",
                "Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."),
            cup_holding_Nm: f64 => out_rust_only("N·m", "Cup holding torque at its loosest",
                "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."),
            hub_bore_stress_MPa: f64 => out_rust_only("MPa", "Inner hub bore stress at its tightest",
                "Von Mises stress at the inner hub's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included."),
            cup_bore_stress_MPa: f64 => out_rust_only("MPa", "Cup bore stress at its tightest",
                "Von Mises stress at the cup's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included."),
            hub_key_pressure_MPa: f64 => out_rust_only("MPa", "Key pressure on the inner hub, key alone",
                "Pressure on the key's face in the inner hub were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area."),
            cup_key_pressure_MPa: f64 => out_rust_only("MPa", "Key pressure on the cup, key alone",
                "Pressure on the key's face in the cup were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area."),
            hub_needed_um: f64 => out_rust_only("µm", "Inner hub smallest measured interference that holds",
                "The lowest the band's minimum may be for the inner hub to hold the torque it must: the window's lower end, with smoothing and temperature included."),
            cup_needed_um: f64 => out_rust_only("µm", "Cup smallest measured interference that holds",
                "The lowest the band's minimum may be for the cup to hold the torque it must: the window's lower end, with smoothing and temperature included."),
            hub_elastic_max_um: f64 => out_rust_only("µm", "Inner hub largest measured interference within the stress limit",
                "The highest the band's maximum may be for the inner hub's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window."),
            cup_elastic_max_um: f64 => out_rust_only("µm", "Cup largest measured interference within the stress limit",
                "The highest the band's maximum may be for the cup's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window."),
            wheel_shaft_shrink_um: f64 => out_rust_only("µm", "Wheel shaft shrink at its fitting temperature",
                "How much the wheel shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows."),
            output_shaft_shrink_um: f64 => out_rust_only("µm", "Gearbox shaft shrink at its fitting temperature",
                "How much the gearbox shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows."),
            hub_fittable_max_um: f64 => out_rust_only("µm", "Inner hub largest measured interference that can be fitted",
                "The highest the band's maximum may be for the inner hub to be fitted: what its allowed temperature and the wheel shaft's shrink open, less the joining clearance."),
            cup_fittable_max_um: f64 => out_rust_only("µm", "Cup largest measured interference that can be fitted",
                "The highest the band's maximum may be for the cup to be fitted: what its allowed temperature and the gearbox shaft's shrink open, less the joining clearance."),
            hub_fit_temp_C: f64 => out_rust_only("°C", "Inner hub temperature the fit would need",
                "What the inner hub would need on its wheel shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further."),
            cup_fit_temp_C: f64 => out_rust_only("°C", "Cup temperature the fit would need",
                "What the bare cup would need on the gearbox shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further."),
            hub_fit_allowed_C: f64 => out_rust_only("°C", "Inner hub fitting temperature allowed",
                "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the magnets' single-ring onset less 10 °C; not a number if any is."),
            cup_fit_allowed_C: f64 => out_rust_only("°C", "Cup fitting temperature allowed",
                "Your limit for the bare cup, which is fitted before its magnets are bonded."),
            check: String => out_rust_only("", "Shaft joints: keyed shrink fits",
                "Each rotor's shrink fit must hold cold-high times the safety factor when loosest, stay elastic when tightest and be fitted within the allowed temperature; the key alone is a backup."),
        }
    }
}

/// The shafts' elastic modulus [GPa]: a generic alloy steel (the shafts' steels are not known).
pub const SHAFT_MODULUS_GPA: f64 = 205.0;

/// Poisson's ratio of the shafts and the parts.
pub const POISSON: f64 = 0.30;

/// The shafts' expansion coefficient [1/°C], from [`NIST_FIT_TOP_K`] up.
pub const SHAFT_CTE_PER_C: f64 = 11.5e-6;

/// Room temperature [°C]: the measured interference's temperature, and where a shaft's shrink
/// is zero.
pub const ROOM_C: f64 = 20.0;

/// The part's stress limit is its yield strength divided by this (DIN 7190 practice is 1.1 to
/// 1.2; the physics review confirms it). The key flank is held to the same limit.
pub const YIELD_FACTOR: f64 = 1.2;

/// NIST's fit of the linear expansion of nickel steel Fe-2.25Ni, the nearest ferritic steel
/// NIST lists: y = (L − L293) / L293 × 10⁵ = a0 + a1·T + a2·T² + a3·T³ + a4·T⁴, T in K, for
/// 4 to [`NIST_FIT_TOP_K`] (1.5 % fit error;
/// trc.nist.gov/cryogenics/materials/NickelSteel/NickelSteel_Fe2.25Ni)_rev.htm).
pub const NIST_FE_2_25NI: [f64; 5] = [-210.4, -0.05699, 5.072e-3, -1.381e-5, 1.897e-8];

/// The top of the NIST fit's range [K]. The shrink is the fit below it, anchored here, and
/// [`SHAFT_CTE_PER_C`] above it.
pub const NIST_FIT_TOP_K: f64 = 265.0;

/// 0 °C in kelvin.
const ZERO_C_IN_K: f64 = 273.15;

/// The inner hub's name in the check's texts.
pub const INNER_HUB: &str = "inner hub";

/// The cup's name in the check's texts.
pub const CUP: &str = "cup";

/// `shaft_joint.check` when both joints hold, stay within the stress limit, can be fitted and
/// their keys pass.
pub const SHAFT_JOINTS_PASS: &str = "Both shaft joints hold and can be fitted";

/// The unknown verdicts, each the prefix [`SHAFT_JOINT_UNKNOWN`] with its reason.
macro_rules! unknown {
    ($reason:literal) => {
        concat!("Shaft joint check unknown", $reason)
    };
}

/// The start of every unknown `shaft_joint.check`.
pub const SHAFT_JOINT_UNKNOWN: &str = unknown!("");

/// `shaft_joint.check` when a value it reads, or a term a judged joint compares, is not a
/// number.
pub const UNKNOWN_NOT_A_NUMBER: &str = unknown!(": a value is not a number");

/// The start of `shaft_joint.check` when a joint has no wall around its bore and no judged
/// joint fails; " ({joint})" follows.
pub const UNKNOWN_NO_WALL: &str = unknown!(": no wall around the bore");

/// The start of `shaft_joint.check` when a joint's minimum interference is above its maximum
/// and no judged joint fails; " ({joint})" follows.
pub const UNKNOWN_BAND_REVERSED: &str = unknown!(": the minimum interference is above the maximum");

/// `shaft_joint.check` when the safety factor is not above zero, a typed value no slider reaches
/// (ruling C-4): the torque each joint must hold would be zero or below, and a key pressure below
/// zero would pass.
pub const UNKNOWN_SAFETY_FACTOR: &str = unknown!(": the safety factor is not above zero");

/// The start of `shaft_joint.check` when a fit slips; " ({joint}): holds {holding} N·m against
/// {required} N·m" follows.
pub const FIT_SLIPS: &str = "The fit slips";

/// The start of `shaft_joint.check` when a bore stress is above its limit; " ({joint}): {stress}
/// MPa against {limit} MPa" follows.
pub const BORE_OVER_ITS_LIMIT: &str = "Bore stress above its limit";

/// The start of `shaft_joint.check` when a part cannot be fitted at its allowed temperature;
/// " ({joint}): needs {needed} °C against {allowed} °C" follows.
pub const CANNOT_BE_FITTED: &str = "Cannot be fitted at the allowed temperature";

/// The start of `shaft_joint.check` when the key alone would bear more than the limit;
/// " ({joint}): {pressure} MPa against {limit} MPa" follows.
pub const KEY_OVER_ITS_LIMIT: &str = "Key backup above its limit";

/// The start of `shaft_joint.check` when the key has no bearing length or no contact height;
/// " ({joint})" follows.
pub const KEY_HAS_NO_BEARING_AREA: &str = "The key has no bearing area";

/// The four terms each judged joint must pass, in the badge's order (spec section 4.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// The fit alone holds less than the required torque at its loosest.
    Slips,
    /// The bore stress at the tightest fit is above the stress limit.
    BoreStress,
    /// The temperature the fit would need is above the allowed one.
    CannotBeFitted,
    /// The key alone has no bearing area, or bears more than the stress limit.
    Key,
}

/// [`Cause`] in the badge's order.
pub const CAUSES: [Cause; 4] = [
    Cause::Slips,
    Cause::BoreStress,
    Cause::CannotBeFitted,
    Cause::Key,
];

/// Why a joint has no wall around its bore: its Lamé terms then mean nothing and it is not
/// judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoWall {
    /// The bore is as wide as the part (bore over outer diameter at least 1), or the part's
    /// diameter is typed at or below zero (ruling C-3).
    BoreAsWideAsPart,
    /// The keyway is at least as deep as the wall around the bore is thick.
    KeywayThroughWall,
}

/// Which of the three values caps the bonded inner hub's allowed fitting temperature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HubFitLimit {
    /// The owner's input, `shaft_joint.hub_fit_max_C`.
    Yours,
    /// The selected adhesive's design limit, `temperature.adhesive.design_limit_C`.
    Adhesive,
    /// The inner ring's own single-ring onset less [`CURE_MARGIN_MIN_C`],
    /// `temperature.demag.inner_onset_single_ring_C`.
    MagnetOnset,
}

/// The bonded inner hub's allowed fitting temperature [°C] (owner R-15): the lowest of the
/// owner's input, the selected adhesive's design limit and the inner ring's own single-ring
/// onset less [`CURE_MARGIN_MIN_C`], with the limit that binds (ties name the earlier: yours,
/// the adhesive's, the onset's). NaN and no limit when any of the three is NaN: `py_min` drops
/// a NaN in its second argument and `f64::min` in either, so no order makes a bare min safe.
/// An onset of +inf (ferrite) never binds.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn hub_fit_allowed(input_C: f64, adhesive_C: f64, onset_C: f64) -> (f64, Option<HubFitLimit>) {
    if input_C.is_nan() || adhesive_C.is_nan() || onset_C.is_nan() {
        return (f64::NAN, None);
    }
    // py_min returns one of its arguments bit for bit, so the equalities below are exact.
    let allowed = py_min(py_min(input_C, adhesive_C), onset_C - CURE_MARGIN_MIN_C);
    let limit = if allowed == input_C {
        HubFitLimit::Yours
    } else if allowed == adhesive_C {
        HubFitLimit::Adhesive
    } else {
        HubFitLimit::MagnetOnset
    };
    (allowed, Some(limit))
}

/// NIST's Fe-2.25Ni fit at `kelvin`: (L − L293) / L293 × 10⁵.
fn nist_y(kelvin: f64) -> f64 {
    let [a0, a1, a2, a3, a4] = NIST_FE_2_25NI;
    a0 + a1 * kelvin + a2 * kelvin.powi(2) + a3 * kelvin.powi(3) + a4 * kelvin.powi(4)
}

/// The diametral shrink [µm] of a steel shaft `bore_mm` across, taken from [`ROOM_C`] to
/// `shaft_C`: positive when colder, negative above 20 °C (it grows). [`SHAFT_CTE_PER_C`] from
/// [`NIST_FIT_TOP_K`] up; below it NIST's Fe-2.25Ni fit, anchored at [`NIST_FIT_TOP_K`], so the
/// shrink is continuous there (a constant coefficient overstates the shrink in liquid nitrogen
/// by about 31 %).
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn shaft_shrink_um(shaft_C: f64, bore_mm: f64) -> f64 {
    let kelvin = shaft_C + ZERO_C_IN_K;
    let strain = if kelvin >= NIST_FIT_TOP_K {
        SHAFT_CTE_PER_C * (ROOM_C - shaft_C)
    } else {
        SHAFT_CTE_PER_C * (ROOM_C - (NIST_FIT_TOP_K - ZERO_C_IN_K))
            + (nist_y(NIST_FIT_TOP_K) - nist_y(kelvin)) * 1e-5
    };
    strain * bore_mm * 1000.0
}

/// The engine values the check reads (spec section 4.4's fifteen), filled by `api::compute`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct ShaftJointLinks {
    /// Cold-high [N·m] (`metal.torque_cold_high_Nm`).
    pub cold_high_Nm: f64,
    /// Both joints' bore [mm] (`coupling.bore_mm`).
    pub bore_mm: f64,
    /// Both joints' keyway depth [mm] (`coupling.keyway_depth_mm`); 0 for no keyway.
    pub keyway_depth_mm: f64,
    /// The inner hub's length in effect [mm] (`housing.hub_length_mm`: the magnets' length
    /// under the axial override).
    pub hub_length_mm: f64,
    /// The inner magnets' back apothem, bondline included [mm] (`coupling.inner_back_apothem_mm`).
    pub inner_back_apothem_mm: f64,
    /// The inner bondline [mm] (`metal.bond_inner_mm`).
    pub bond_inner_mm: f64,
    /// The cup's boss length [mm] (`metal.boss_length_mm`).
    pub boss_length_mm: f64,
    /// The cup's rear web thickness [mm] (`metal.web_mm`): the bore runs through it too.
    pub web_mm: f64,
    /// The cup's boss outside diameter [mm] (`metal.boss_od_mm`).
    pub boss_od_mm: f64,
    /// The minimum magnet temperature [°C] (`metal.min_temp_C`): the cold end.
    pub min_temp_C: f64,
    /// The selected adhesive's design limit [°C] (`temperature.adhesive.design_limit_C`): the
    /// hot end, and a cap on the bonded inner hub's fitting temperature.
    pub adhesive_limit_C: f64,
    /// The inner ring's own single-ring onset [°C]
    /// (`temperature.demag.inner_onset_single_ring_C`): a cap on the bonded inner hub's fitting
    /// temperature, less [`CURE_MARGIN_MIN_C`].
    pub inner_onset_single_ring_C: f64,
    /// The inner hub and cup material's elastic modulus [GPa] (`PartProperties::hub`).
    pub part_modulus_GPa: f64,
    /// The inner hub and cup material's expansion coefficient [1/°C] (`PartProperties::hub`).
    pub part_cte_per_C: f64,
    /// The inner hub and cup material's library yield strength [MPa] (`PartProperties::hub`).
    pub part_yield_MPa: f64,
}

impl ShaftJointLinks {
    /// The fifteen values, for the unknown test.
    pub fn values(&self) -> [f64; 15] {
        [
            self.cold_high_Nm,
            self.bore_mm,
            self.keyway_depth_mm,
            self.hub_length_mm,
            self.inner_back_apothem_mm,
            self.bond_inner_mm,
            self.boss_length_mm,
            self.web_mm,
            self.boss_od_mm,
            self.min_temp_C,
            self.adhesive_limit_C,
            self.inner_onset_single_ring_C,
            self.part_modulus_GPa,
            self.part_cte_per_C,
            self.part_yield_MPa,
        ]
    }
}

/// One joint's own values; [`JointCommon`] holds what both joints share.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointSpec {
    /// [`INNER_HUB`] or [`CUP`].
    pub name: &'static str,
    /// The part's outer diameter around the bore [mm]: the inner hub's flats' incircle, the
    /// cup's boss (both conservative).
    pub outer_diameter_mm: f64,
    /// The engaged length [mm]: the inner hub's length, the cup's boss and web.
    pub length_mm: f64,
    /// The band's minimum measured interference [µm].
    pub interference_min_um: f64,
    /// The band's maximum measured interference [µm].
    pub interference_max_um: f64,
    /// The hottest the part may be at fitting [°C].
    pub fit_allowed_C: f64,
    /// Its shaft's temperature at fitting [°C].
    pub shaft_fit_C: f64,
}

/// What both joints share.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointCommon {
    /// The bore [mm].
    pub bore_mm: f64,
    /// The keyway depth [mm]; 0 for no keyway (no key term and no key arc).
    pub keyway_depth_mm: f64,
    /// The key's width [mm].
    pub key_width_mm: f64,
    /// The key face's contact height [mm], before the cap at the keyway depth.
    pub key_contact_mm: f64,
    /// The fit's friction coefficient.
    pub friction: f64,
    /// The interference lost to smoothing at fitting [µm].
    pub smoothing_um: f64,
    /// How much hotter the rotor runs than its shaft at the hot end [°C].
    pub rotor_over_shaft_C: f64,
    /// The clearance each part needs over its shaft to slide on [µm].
    pub joining_clearance_um: f64,
    /// The parts' elastic modulus [GPa].
    pub part_modulus_GPa: f64,
    /// The parts' expansion coefficient [1/°C].
    pub part_cte_per_C: f64,
    /// The cold end [°C]: the minimum magnet temperature.
    pub cold_C: f64,
    /// The hot end [°C]: the selected adhesive's design limit.
    pub hot_C: f64,
    /// The torque each fit, and each key alone, must hold [N·m].
    pub required_Nm: f64,
    /// The bore stress limit, and the key flank's [MPa].
    pub stress_limit_MPa: f64,
}

/// One joint's terms (spec sections 3.1 and 3.3), computed whether or not the joint is judged.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct JointTerms {
    /// [`INNER_HUB`] or [`CUP`].
    pub name: &'static str,
    /// Why the joint has no wall around its bore; `None` when it has one.
    pub no_wall: Option<NoWall>,
    /// The band's minimum is above its maximum.
    pub band_reversed: bool,
    /// Whether there is a keyway (depth above 0): the key term applies and its arc is taken off.
    pub keyed: bool,
    /// Contact pressure per µm of effective diametral interference [MPa/µm].
    pub pressure_per_um: f64,
    /// Holding torque per µm of effective interference [N·m/µm].
    pub holding_per_um: f64,
    /// Bore von Mises stress per µm of effective interference [MPa/µm].
    pub stress_per_um: f64,
    /// Effective interference at the loosest [µm]: the minimum less smoothing at the end that
    /// loosens the fit more, floored at 0.
    pub effective_min_um: f64,
    /// Effective interference at the tightest [µm]: the maximum less smoothing at the end that
    /// tightens the fit more, floored at 0.
    pub effective_max_um: f64,
    /// The fit's holding torque at its loosest [N·m]; NaN with no wall.
    pub holding_Nm: f64,
    /// The bore's von Mises stress at its tightest [MPa]; NaN with no wall.
    pub bore_stress_MPa: f64,
    /// The smallest measured minimum that holds the required torque [µm]; NaN with no wall.
    pub needed_um: f64,
    /// The largest measured maximum within the stress limit [µm]; NaN with no wall.
    pub elastic_max_um: f64,
    /// The shaft's diametral shrink at its fitting temperature [µm].
    pub shaft_shrink_um: f64,
    /// The largest measured maximum that can be fitted at the allowed temperature [µm].
    pub fittable_max_um: f64,
    /// The part's temperature the fit would need [°C].
    pub fit_temp_C: f64,
    /// The hottest the part may be at fitting [°C].
    pub fit_allowed_C: f64,
    /// The key's bearing length, the engaged length less the key width [mm].
    pub key_bearing_mm: f64,
    /// The key face's contact height, capped at the keyway depth [mm].
    pub key_contact_mm: f64,
    /// The key flank's pressure were the key to carry the required torque alone [MPa]: 0 with no
    /// keyway, +inf with no bearing area.
    pub key_pressure_MPa: f64,
    /// The torque each fit, and each key alone, must hold [N·m].
    pub required_Nm: f64,
    /// The bore stress limit, and the key flank's [MPa].
    pub stress_limit_MPa: f64,
    /// The slip term's margin [µm]: the measured minimum less the needed one.
    pub slip_margin_um: f64,
    /// The key term's margin [mm]: the bearing length less the length needed at the limit;
    /// −inf with no contact height, +inf with no keyway.
    pub key_margin_mm: f64,
}

impl JointTerms {
    /// Whether the joint is judged: it has a wall and its band is not reversed.
    pub fn determinate(&self) -> bool {
        self.no_wall.is_none() && !self.band_reversed
    }

    /// Whether the key has a bearing area: a positive bearing length and contact height.
    pub fn key_has_bearing_area(&self) -> bool {
        self.key_bearing_mm > 0.0 && self.key_contact_mm > 0.0
    }

    /// The terms the four comparisons read; a judged joint with any of them NaN reads unknown.
    pub fn compared(&self) -> [f64; 7] {
        [
            self.holding_Nm,
            self.required_Nm,
            self.bore_stress_MPa,
            self.stress_limit_MPa,
            self.fit_temp_C,
            self.fit_allowed_C,
            self.key_pressure_MPa,
        ]
    }

    /// Whether the joint fails `cause`'s comparison. Each is written so that a NaN fails it;
    /// equality passes. The key term does not apply without a keyway.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails
    pub fn fails(&self, cause: Cause) -> bool {
        match cause {
            Cause::Slips => !(self.holding_Nm >= self.required_Nm),
            Cause::BoreStress => !(self.bore_stress_MPa <= self.stress_limit_MPa),
            Cause::CannotBeFitted => !(self.fit_temp_C <= self.fit_allowed_C),
            Cause::Key => self.keyed && !(self.key_pressure_MPa <= self.stress_limit_MPa),
        }
    }

    /// `cause`'s margin, positive when it passes (spec section 5.2): the slip term in µm of the
    /// band's minimum, the stress term in MPa, the fitting term in °C, the key term in mm of
    /// bearing length.
    pub fn margin(&self, cause: Cause) -> f64 {
        match cause {
            Cause::Slips => self.slip_margin_um,
            Cause::BoreStress => self.stress_limit_MPa - self.bore_stress_MPa,
            Cause::CannotBeFitted => self.fit_allowed_C - self.fit_temp_C,
            Cause::Key => self.key_margin_mm,
        }
    }

    /// The badge's text for `cause` on this joint, its numbers never equal: the limit quoted
    /// half-up to its last decimal, the failing value pushed one last digit past it.
    pub fn text(&self, cause: Cause) -> String {
        let name = self.name;
        match cause {
            Cause::Slips => {
                let (holding, required) = quoted_below(self.holding_Nm, self.required_Nm, 2);
                format!("{FIT_SLIPS} ({name}): holds {holding} N·m against {required} N·m")
            }
            Cause::BoreStress => {
                let (stress, limit) = quoted_above(self.bore_stress_MPa, self.stress_limit_MPa, 1);
                format!("{BORE_OVER_ITS_LIMIT} ({name}): {stress} MPa against {limit} MPa")
            }
            Cause::CannotBeFitted => {
                let (needed, allowed) = quoted_above(self.fit_temp_C, self.fit_allowed_C, 1);
                format!("{CANNOT_BE_FITTED} ({name}): needs {needed} °C against {allowed} °C")
            }
            Cause::Key if !self.key_has_bearing_area() => {
                format!("{KEY_HAS_NO_BEARING_AREA} ({name})")
            }
            Cause::Key => {
                let (pressure, limit) =
                    quoted_above(self.key_pressure_MPa, self.stress_limit_MPa, 1);
                format!("{KEY_OVER_ITS_LIMIT} ({name}): {pressure} MPa against {limit} MPa")
            }
        }
    }
}

/// A value above its limit as the badge quotes them: (the value, the limit), the limit half-up
/// to `decimals`, the value at least one last digit above the limit as quoted.
fn quoted_above(value: f64, limit: f64, decimals: usize) -> (String, String) {
    let step = 10f64.powi(-(decimals as i32));
    let limit_text = fmt_fixed(limit, decimals);
    let quoted: f64 = limit_text.parse().unwrap_or(limit);
    (
        fmt_fixed(py_max(value, quoted + step), decimals),
        limit_text,
    )
}

/// A value below its need as the badge quotes them: (the value, the need), the need half-up to
/// `decimals`, the value at least one last digit below the need as quoted and not below 0.
/// Where the need itself quotes as 0, the value is quoted 0 and the need one last digit.
fn quoted_below(value: f64, need: f64, decimals: usize) -> (String, String) {
    let step = 10f64.powi(-(decimals as i32));
    let need_text = fmt_fixed(need, decimals);
    let quoted: f64 = need_text.parse().unwrap_or(need);
    if quoted <= 0.0 {
        return (fmt_fixed(0.0, decimals), fmt_fixed(step, decimals));
    }
    (
        fmt_fixed(py_max(0.0, py_min(value, quoted - step)), decimals),
        need_text,
    )
}

/// One joint's terms (spec sections 3.1 and 3.3): plane-stress Lamé thick cylinders on a solid
/// shaft, the key's arc taken off the contact, the temperature ends, the fitting temperature
/// and the key alone.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn joint(spec: &JointSpec, common: &JointCommon) -> JointTerms {
    let d = common.bore_mm;
    let keyed = common.keyway_depth_mm > 0.0;
    // Lamé: contact pressure and bore von Mises stress per µm of effective interference.
    let q = d / spec.outer_diameter_mm;
    let hoop = (1.0 + q * q) / (1.0 - q * q);
    let part_E_MPa = common.part_modulus_GPa * 1000.0;
    let shaft_E_MPa = SHAFT_MODULUS_GPA * 1000.0;
    let pressure_per_um = 1e-3
        / (d * ((1.0 / part_E_MPa) * (hoop + POISSON) + (1.0 / shaft_E_MPa) * (1.0 - POISSON)));
    // The key's slot takes its width off the contact arc (none without a keyway), floored at 0.
    let key_arc_mm = if keyed { common.key_width_mm } else { 0.0 };
    let contact_arc_mm = py_max(0.0, PI * d - key_arc_mm);
    let holding_per_um =
        common.friction * pressure_per_um * contact_arc_mm * spec.length_mm * d / 2.0 / 1000.0;
    let stress_per_um = (hoop * hoop + hoop + 1.0).sqrt() * pressure_per_um;
    // The interference's change at a uniform temperature; at the hot end the rotor also runs the
    // allowance warmer than its shaft.
    let alpha = common.part_cte_per_C;
    let mismatch_um = |t: f64| (SHAFT_CTE_PER_C - alpha) * (t - ROOM_C) * d * 1000.0;
    let rotor_warmer_um = alpha * common.rotor_over_shaft_C * d * 1000.0;
    let loosest_um = py_min(
        mismatch_um(common.cold_C),
        mismatch_um(common.hot_C) - rotor_warmer_um,
    );
    let tightest_um = py_max(mismatch_um(common.cold_C), mismatch_um(common.hot_C));
    let effective_min_um = py_max(
        0.0,
        spec.interference_min_um - common.smoothing_um + loosest_um,
    );
    let effective_max_um = py_max(
        0.0,
        spec.interference_max_um - common.smoothing_um + tightest_um,
    );
    // No wall: the bore is as wide as the part, or the keyway cuts through the wall. The Lamé
    // rows then mean nothing: NaN.
    let wall_mm = (spec.outer_diameter_mm - d) / 2.0;
    // the bore at least as wide as the part (Q >= 1), a part typed at or below zero included
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails it
    let no_wall = if !(spec.outer_diameter_mm > d) {
        Some(NoWall::BoreAsWideAsPart)
    } else if keyed && wall_mm <= common.keyway_depth_mm {
        Some(NoWall::KeywayThroughWall)
    } else {
        None
    };
    let lame = |x: f64| if no_wall.is_some() { f64::NAN } else { x };
    let holding_Nm = lame(effective_min_um * holding_per_um);
    let bore_stress_MPa = lame(effective_max_um * stress_per_um);
    let needed_um = lame(common.required_Nm / holding_per_um + common.smoothing_um - loosest_um);
    let elastic_max_um =
        lame(common.stress_limit_MPa / stress_per_um + common.smoothing_um - tightest_um);
    // Fitting: the part warmed, its shaft at its fitting temperature, the joining clearance
    // open. Measured interference: no smoothing is taken off.
    let shaft_shrink_um = shaft_shrink_um(spec.shaft_fit_C, d);
    let fittable_max_um = alpha * (spec.fit_allowed_C - ROOM_C) * d * 1000.0 + shaft_shrink_um
        - common.joining_clearance_um;
    let fit_temp_C = ROOM_C
        + (spec.interference_max_um + common.joining_clearance_um - shaft_shrink_um)
            / (alpha * d * 1000.0);
    // The key alone (form A: bearing length L − b), its contact capped at the keyway depth.
    let key_bearing_mm = spec.length_mm - common.key_width_mm;
    let key_contact_mm = py_min(common.key_contact_mm, common.keyway_depth_mm);
    let bearing = key_bearing_mm > 0.0 && key_contact_mm > 0.0;
    let key_pressure_MPa = if !keyed {
        0.0
    } else if bearing {
        2.0 * common.required_Nm * 1000.0 / (d * key_contact_mm * key_bearing_mm)
    } else {
        f64::INFINITY
    };
    let key_needed_mm = if key_contact_mm > 0.0 {
        2.0 * common.required_Nm * 1000.0 / (d * key_contact_mm * common.stress_limit_MPa)
    } else {
        f64::INFINITY
    };
    JointTerms {
        name: spec.name,
        no_wall,
        band_reversed: spec.interference_min_um > spec.interference_max_um,
        keyed,
        pressure_per_um,
        holding_per_um,
        stress_per_um,
        effective_min_um,
        effective_max_um,
        holding_Nm,
        bore_stress_MPa,
        needed_um,
        elastic_max_um,
        shaft_shrink_um,
        fittable_max_um,
        fit_temp_C,
        fit_allowed_C: spec.fit_allowed_C,
        key_bearing_mm,
        key_contact_mm,
        key_pressure_MPa,
        required_Nm: common.required_Nm,
        stress_limit_MPa: common.stress_limit_MPa,
        slip_margin_um: spec.interference_min_um - needed_um,
        key_margin_mm: if keyed {
            key_bearing_mm - key_needed_mm
        } else {
            f64::INFINITY
        },
    }
}

/// What the two joints are: what both share, and each joint's own values, inner hub first.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn joint_specs(
    inputs: &ShaftJointInputs,
    links: &ShaftJointLinks,
) -> (JointCommon, [JointSpec; 2]) {
    let common = JointCommon {
        bore_mm: links.bore_mm,
        keyway_depth_mm: links.keyway_depth_mm,
        key_width_mm: inputs.key_width_mm,
        key_contact_mm: inputs.key_contact_mm,
        friction: inputs.friction,
        smoothing_um: inputs.smoothing_um,
        rotor_over_shaft_C: inputs.rotor_over_shaft_C,
        joining_clearance_um: inputs.joining_clearance_um,
        part_modulus_GPa: links.part_modulus_GPa,
        part_cte_per_C: links.part_cte_per_C,
        cold_C: links.min_temp_C,
        hot_C: links.adhesive_limit_C,
        required_Nm: inputs.safety_factor * links.cold_high_Nm,
        stress_limit_MPa: links.part_yield_MPa / YIELD_FACTOR,
    };
    let (hub_allowed_C, _) = hub_fit_allowed(
        inputs.hub_fit_max_C,
        links.adhesive_limit_C,
        links.inner_onset_single_ring_C,
    );
    let hub = JointSpec {
        name: INNER_HUB,
        // The flats' incircle, under the bondline (conservative).
        outer_diameter_mm: 2.0 * (links.inner_back_apothem_mm - links.bond_inner_mm),
        length_mm: links.hub_length_mm,
        interference_min_um: inputs.hub_interference_min_um,
        interference_max_um: inputs.hub_interference_max_um,
        fit_allowed_C: hub_allowed_C,
        shaft_fit_C: inputs.wheel_shaft_fit_C,
    };
    let cup = JointSpec {
        name: CUP,
        // The boss (conservative for the web); the bore runs through the boss and the web.
        outer_diameter_mm: links.boss_od_mm,
        length_mm: links.boss_length_mm + links.web_mm,
        interference_min_um: inputs.cup_interference_min_um,
        interference_max_um: inputs.cup_interference_max_um,
        fit_allowed_C: inputs.cup_fit_max_C,
        shaft_fit_C: inputs.output_shaft_fit_C,
    };
    (common, [hub, cup])
}

/// Both joints' terms, inner hub first.
pub fn joints(inputs: &ShaftJointInputs, links: &ShaftJointLinks) -> [JointTerms; 2] {
    let (common, [hub, cup]) = joint_specs(inputs, links);
    [joint(&hub, &common), joint(&cup, &common)]
}

/// Every term the judged joints fail, in the badge's order: by cause ([`CAUSES`]), then inner
/// hub before cup.
pub fn failing(joints: &[JointTerms; 2]) -> Vec<(Cause, &JointTerms)> {
    CAUSES
        .iter()
        .flat_map(|&cause| {
            joints
                .iter()
                .filter(move |j| j.determinate() && j.fails(cause))
                .map(move |j| (cause, j))
        })
        .collect()
}

/// `shaft_joint.check` (spec section 4.3): unknown when any input or engine value it reads,
/// or any term a judged joint compares, is not a number; then unknown when the safety factor
/// is not above zero (ruling C-4); else the first failing term of a judged joint; else a joint
/// with no wall, then a reversed band, as unknown; else the pass.
pub fn verdict(
    inputs: &ShaftJointInputs,
    links: &ShaftJointLinks,
    joints: &[JointTerms; 2],
) -> String {
    let mut input_nan = false;
    inputs.visit("", &mut |_, _, value| {
        input_nan |= matches!(value, Value::Num(x) if x.is_nan());
    });
    let link_nan = links.values().iter().any(|x| x.is_nan());
    let term_nan = joints
        .iter()
        .filter(|j| j.determinate())
        .any(|j| j.compared().iter().any(|x| x.is_nan()));
    if input_nan || link_nan || term_nan {
        return UNKNOWN_NOT_A_NUMBER.to_owned();
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN fails it
    if !(inputs.safety_factor > 0.0) {
        return UNKNOWN_SAFETY_FACTOR.to_owned();
    }
    if let Some((cause, joint)) = failing(joints).first() {
        return joint.text(*cause);
    }
    if let Some(joint) = joints.iter().find(|j| j.no_wall.is_some()) {
        return format!("{UNKNOWN_NO_WALL} ({})", joint.name);
    }
    if let Some(joint) = joints.iter().find(|j| j.band_reversed) {
        return format!("{UNKNOWN_BAND_REVERSED} ({})", joint.name);
    }
    SHAFT_JOINTS_PASS.to_owned()
}

/// The shaft joint check of a design.
pub fn compute(inputs: &ShaftJointInputs, links: &ShaftJointLinks) -> ShaftJointResults {
    let [hub, cup] = joints(inputs, links);
    let check = verdict(inputs, links, &[hub, cup]);
    ShaftJointResults {
        required_Nm: hub.required_Nm,
        part_modulus_GPa: links.part_modulus_GPa,
        part_cte_per_C: links.part_cte_per_C,
        part_yield_MPa: links.part_yield_MPa,
        stress_limit_MPa: hub.stress_limit_MPa,
        hub_holding_Nm: hub.holding_Nm,
        cup_holding_Nm: cup.holding_Nm,
        hub_bore_stress_MPa: hub.bore_stress_MPa,
        cup_bore_stress_MPa: cup.bore_stress_MPa,
        hub_key_pressure_MPa: hub.key_pressure_MPa,
        cup_key_pressure_MPa: cup.key_pressure_MPa,
        hub_needed_um: hub.needed_um,
        cup_needed_um: cup.needed_um,
        hub_elastic_max_um: hub.elastic_max_um,
        cup_elastic_max_um: cup.elastic_max_um,
        wheel_shaft_shrink_um: hub.shaft_shrink_um,
        output_shaft_shrink_um: cup.shaft_shrink_um,
        hub_fittable_max_um: hub.fittable_max_um,
        cup_fittable_max_um: cup.fittable_max_um,
        hub_fit_temp_C: hub.fit_temp_C,
        cup_fit_temp_C: cup.fit_temp_C,
        hub_fit_allowed_C: hub.fit_allowed_C,
        cup_fit_allowed_C: cup.fit_allowed_C,
        check,
    }
}
```

- [ ] **Step 4: Register the module and compute the group**

`magcoupling-rs/src/engine/mod.rs` (alphabetical):

Old:
```rust
pub mod model;
pub mod picks;
pub mod size_controls;
pub mod sizing;
```

New:
```rust
pub mod model;
pub mod picks;
pub mod shaft_joint;
pub mod size_controls;
pub mod sizing;
```

`magcoupling-rs/src/engine/api.rs`: the import, the input group after `clamps` (the first top-level Rust-only input group), the result group after `gearbox`, `shaft_joint_links` before `impl DesignInputs`, the computation after `gearbox::compute` (`mdr`, `axial`, `temp` and `hub_material` all exist there), and the struct literal.

Old:
```rust
use super::metal_design::{self, MetalDesignInputs, MetalDesignResults, RetainerResults};
use super::model::{self, CouplingInputs, MassResults, ModelResults};
use super::sweeps::{self, SweepRow};
use super::temperature::{self, TemperatureInputs, TemperatureResults};
```

New:
```rust
use super::metal_design::{self, MetalDesignInputs, MetalDesignResults, RetainerResults};
use super::model::{self, CouplingInputs, MassResults, ModelResults};
use super::shaft_joint::{self, ShaftJointInputs, ShaftJointLinks, ShaftJointResults};
use super::sweeps::{self, SweepRow};
use super::temperature::{self, TemperatureInputs, TemperatureResults};
```

Old:
```rust
            temperature: TemperatureInputs,
            clamps: ClampInputs,
        }
    }
```

New:
```rust
            temperature: TemperatureInputs,
            clamps: ClampInputs,
            shaft_joint: ShaftJointInputs,
        }
    }
```

Old:
```rust
            housing: HousingResults,
            gearbox: GearboxResults,
        }
        tables {
```

New:
```rust
            housing: HousingResults,
            gearbox: GearboxResults,
            shaft_joint: ShaftJointResults,
        }
        tables {
```

Old:
```rust
        .map(|&(key, path)| (key, results.get(path).unwrap_or(Value::None)))
        .collect()
}

```

New:
```rust
        .map(|&(key, path)| (key, results.get(path).unwrap_or(Value::None)))
        .collect()
}

/// The engine values the shaft joint check read for `results` (BL-064), rebuilt from the design:
/// each is an input or a result, so the guide and the tests can evaluate the joints' terms
/// ([`shaft_joint::joints`]) as [`compute_all`] did.
pub fn shaft_joint_links(inputs: &DesignInputs, results: &DesignResults) -> ShaftJointLinks {
    ShaftJointLinks {
        cold_high_Nm: results.metal.torque_cold_high_Nm,
        bore_mm: inputs.coupling.bore_mm,
        keyway_depth_mm: inputs.coupling.keyway_depth_mm,
        hub_length_mm: results.housing.hub_length_mm,
        inner_back_apothem_mm: inputs.coupling.inner_back_apothem_mm,
        bond_inner_mm: inputs.metal.bond_inner_mm,
        boss_length_mm: inputs.metal.boss_length_mm,
        web_mm: inputs.metal.web_mm,
        boss_od_mm: inputs.metal.boss_od_mm,
        min_temp_C: inputs.metal.min_temp_C,
        adhesive_limit_C: results.temperature.adhesive.design_limit_C,
        inner_onset_single_ring_C: results.temperature.demag.inner_onset_single_ring_C,
        part_modulus_GPa: results.shaft_joint.part_modulus_GPa,
        part_cte_per_C: results.shaft_joint.part_cte_per_C,
        part_yield_MPa: results.shaft_joint.part_yield_MPa,
    }
}

```

Old:
```rust
    // BL-059: cold-high at the gearbox input against the gearbox input torque rating.
    let gearbox = gearbox::compute(mdr.cold_high_input_Nm, ci.gearbox_input_rating_Nm);

    let alloy = if inputs.clamps.alloy == 1 {
```

New:
```rust
    // BL-059: cold-high at the gearbox input against the gearbox input torque rating.
    let gearbox = gearbox::compute(mdr.cold_high_input_Nm, ci.gearbox_input_rating_Nm);
    // BL-064: the keyed shrink fits of the inner hub on the wheel shaft and of the cup on the
    // gearbox output shaft.
    let shaft_joint = shaft_joint::compute(
        &inputs.shaft_joint,
        &ShaftJointLinks {
            cold_high_Nm: mdr.torque_cold_high_Nm,
            bore_mm: ci.bore_mm,
            keyway_depth_mm: ci.keyway_depth_mm,
            hub_length_mm: axial.hub_length_mm,
            inner_back_apothem_mm: ci.inner_back_apothem_mm,
            bond_inner_mm: md.bond_inner_mm,
            boss_length_mm: md.boss_length_mm,
            web_mm: md.web_mm,
            boss_od_mm: md.boss_od_mm,
            min_temp_C: md.min_temp_C,
            adhesive_limit_C: temp.adhesive.design_limit_C,
            inner_onset_single_ring_C: temp.demag.inner_onset_single_ring_C,
            part_modulus_GPa: hub_material.props.modulus_GPa,
            part_cte_per_C: hub_material.props.cte_per_C,
            part_yield_MPa: hub_material.yield_MPa,
        },
    );

    let alloy = if inputs.clamps.alloy == 1 {
```

Old:
```rust
        housing,
        gearbox,
        gap_sweep: gap,
        pole_sweep: pole,
```

New:
```rust
        housing,
        gearbox,
        shaft_joint,
        gap_sweep: gap,
        pole_sweep: pole,
```

- [ ] **Step 5: Run the new engine tests and see them pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test shaft_joint --test robustness`

Expected: `shaft_joint` `test result: ok. 30 passed; 0 failed`; `robustness` `test result: ok. 13 passed; 0 failed` (12 at HEAD, plus `degenerate_shaft_joint_inputs_never_pass`).

The defaults reproduce spec 3.5 (prototype, full precision): required 7.646620740977276 N·m; holding 9.4844 / 16.8990 N·m; bore stress 241.701 / 312.532 MPa against 345.8333; key 127.444 / 101.955 MPa; needed 12.0815 / 11.3098 µm, elastic 23.8076 / 23.8626, fittable 18.8281 / 24.4400; fitting 93.2674 / 280.1626 °C against 100 / 300; wheel shaft shrink 18.9881 µm; "Both shaft joints hold and can be fitted". Every row of spec 3.6 the tests name reads as the spec gives it.

- [ ] **Step 6: Run the engine suite; move the gearbox test and re-bless the input schema**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --no-fail-fast`

Expected: two failures, both from the new groups:
```text
the_check_is_a_rust_only_text_result_with_no_cell_after_the_housing_group (tests/gearbox.rs)
    panicked at tests\gearbox.rs:351:5: shaft_joint.required_Nm
input_schema_json_is_current (tests/schema.rs)
    tests/data/input_schema.json is stale: run `MAGCOUPLING_BLESS=1 cargo test --test schema`, then regenerate the differential data
```

The gearbox group is now followed by the shaft joints', not the sweeps. In `magcoupling-rs/tests/gearbox.rs`:

Old:
```rust
    assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
    assert!(matches!(row.value, Value::Text(_)), "{:?}", row.value);
    // Schema order: the group follows the housing group, before the tables.
    assert!(
        rows[at - 1].path.starts_with("housing."),
        "{}",
        rows[at - 1].path
    );
    assert!(
        rows[at + 1].path.starts_with("gap_sweep["),
        "{}",
        rows[at + 1].path
```

New:
```rust
    assert!(!row.meta.label.is_empty() && !row.meta.help.is_empty());
    assert!(matches!(row.value, Value::Text(_)), "{:?}", row.value);
    // Schema order: the group follows the housing group, before the shaft joints (BL-064).
    assert!(
        rows[at - 1].path.starts_with("housing."),
        "{}",
        rows[at - 1].path
    );
    assert!(
        rows[at + 1].path.starts_with("shaft_joint."),
        "{}",
        rows[at + 1].path
```

Re-bless the input schema (the 15 Rust-only inputs; the differential data does not change: generator 2 leaves Rust-only inputs out):

Run: `MAGCOUPLING_BLESS=1 cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`, then `git diff --stat magcoupling-rs/tests/data/input_schema.json`

Expected: `test result: ok. 11 passed`; ` magcoupling-rs/tests/data/input_schema.json | 270 +++++` (insertions only: 15 entries of 18 lines, each `"rust_only": true`, `"cell": null`, `"assumption": false`).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok` (the counts of Step 12).

- [ ] **Step 7: Run the app library tests and see what the new groups move**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-fail-fast`

Expected: `test result: FAILED. 873 passed; 15 failed; 6 ignored`:
```text
gui::guidance::tests::the_fix_label_is_the_display_label_and_still_names_one_input
    left: [.., "clamps.alloy", "clamps.boss_od_mm", "clamps.key_width_mm"]   right: [.., "clamps.alloy", "clamps.boss_od_mm"]
gui::inputs::tests::every_input_is_in_exactly_one_workflow_section
    inputs without a workflow section: ["shaft_joint.hub_interference_min_um", ...]
gui::inputs::tests::every_input_is_in_exactly_one_section_in_schema_order
    left: [.., "clamps", "shaft_joint"]   right: [.., "clamps"]
gui::inputs::tests::every_section_has_a_heading_and_every_heading_a_section
    no heading for shaft_joint
gui::inputs::tests::the_filter_matches_label_path_and_cell_by_section_in_the_order_shown
    left: [.., "coupling.gearbox_input_rating_Nm", "shaft_joint.output_shaft_fit_C"]
gui::inputs::tests::the_shown_catalogue_leaves_out_the_paused_section_s_inputs
    left: [.., "temperature", "shaft_joint"]
gui::panel::tests::idle_frames_change_no_input                                   left: 0  right: 1
gui::panel::tests::the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design   left: 0  right: 1
gui::result_groups::tests::every_chain_path_names_results_and_every_package_has_a_heading   no heading for shaft_joint
gui::result_groups::tests::every_result_is_in_exactly_one_group                  left: 1088  right: 1112
gui::results_table::tests::the_lines_group_the_rows_with_only_the_headline_open_at_first   left: 1088  right: 1112
gui::results_table::tests::the_exports_carry_the_gearbox_check_with_no_cell
    shaft_joint.required_Nm,Torque each shaft joint must hold,7.646620740977276,N·m,
gui::spreadsheet::tests::every_result_is_on_one_row_by_physics_chain_with_its_value   left: 1088  right: 1112
gui::spreadsheet::tests::every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order   Workflow  left: 172  right: 187
gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so  left: (393, 1112)  right: (393, 1088)
```
Steps 8 and 9 place the inputs and head the results; Step 10 moves the tests that pin what moved.

- [ ] **Step 8: The inputs' place in the inputs panel (spec 5.3; plan deviation PD-4)**

`magcoupling-rs/src/gui/inputs.rs`: the package heading (`SECTION_LABELS` 18 -> 19, after `clamps`, package order), and the workflow sections. The `shaft` section shows bore, keyway depth, shaft key width and shaft key contact height while the clamp is paused (the clamp's key rows are hidden); the joint in force comes before the paused clamp, so the order shown is the same while paused and not, and no ordinary section follows the advanced `shaft.screw_model` (review SF-17). Section labels are ASCII and contain no "clamp"; no new advanced row (`rows(true) == 27` holds).

Old:
```rust

/// The heading of every input group and nested group, by path prefix, in the package's order.
pub const SECTION_LABELS: [(&str, &str); 18] = [
    ("coupling", "Coupling"),
    ("coupling.magnets", "Magnets"),
```

New:
```rust

/// The heading of every input group and nested group, by path prefix, in the package's order.
pub const SECTION_LABELS: [(&str, &str); 19] = [
    ("coupling", "Coupling"),
    ("coupling.magnets", "Magnets"),
```

Old:
```rust
    ("temperature.adhesive_life", "Adhesive life"),
    ("clamps", "Shaft clamps"),
];

```

New:
```rust
    ("temperature.adhesive_life", "Adhesive life"),
    ("clamps", "Shaft clamps"),
    ("shaft_joint", "Shaft joints"),
];

```

Old:
```rust
                    "clamps.key_width_mm",
                    "clamps.key_contact_mm",
                ],
            },
```

New:
```rust
                    "clamps.key_width_mm",
                    "clamps.key_contact_mm",
                    "shaft_joint.key_width_mm",
                    "shaft_joint.key_contact_mm",
                ],
            },
            // BL-064: the keyed shrink fits in force come before the paused clamp, so the order
            // shown is the same while the clamp is paused and not.
            WorkflowSection {
                id: "shaft.fit",
                label: "Shrink fit",
                advanced: false,
                paths: &[
                    "shaft_joint.hub_interference_min_um",
                    "shaft_joint.hub_interference_max_um",
                    "shaft_joint.cup_interference_min_um",
                    "shaft_joint.cup_interference_max_um",
                    "shaft_joint.safety_factor",
                    "shaft_joint.friction",
                    "shaft_joint.smoothing_um",
                    "shaft_joint.rotor_over_shaft_C",
                ],
            },
            WorkflowSection {
                id: "shaft.fitting",
                label: "Fitting temperatures",
                advanced: false,
                paths: &[
                    "shaft_joint.hub_fit_max_C",
                    "shaft_joint.cup_fit_max_C",
                    "shaft_joint.wheel_shaft_fit_C",
                    "shaft_joint.output_shaft_fit_C",
                    "shaft_joint.joining_clearance_um",
                ],
            },
```

- [ ] **Step 9: The results' heading (`PACKAGE_LABELS`)**

`magcoupling-rs/src/gui/result_groups.rs` (13 -> 14, schema order after `gearbox`). Task 3's `CHAIN_LABELS` entry then takes the 23 numeric results into the chain group once Task 3's scope chain exists; Task 5's `DASHBOARD` entry takes the check into the headline group; the package group then empties and is dropped, but the heading stays required (`every_chain_path_names_results_and_every_package_has_a_heading`, spec 5.4).

Old:
```rust
/// The heading of each package group of the other results, by the first segment of the path
/// (a table's index dropped: `gap_sweep[3].f_end` is in `gap_sweep`), in schema order.
pub const PACKAGE_LABELS: [(&str, &str); 13] = [
    ("calibration", "Calibration"),
    ("model", "Coupling model"),
```

New:
```rust
/// The heading of each package group of the other results, by the first segment of the path
/// (a table's index dropped: `gap_sweep[3].f_end` is in `gap_sweep`), in schema order.
pub const PACKAGE_LABELS: [(&str, &str); 14] = [
    ("calibration", "Calibration"),
    ("model", "Coupling model"),
```

Old:
```rust
    ("housing", "Housing"),
    ("gearbox", "Gearbox"),
    ("gap_sweep", "Gap sweep"),
    ("pole_sweep", "Pole sweep"),
```

New:
```rust
    ("housing", "Housing"),
    ("gearbox", "Gearbox"),
    ("shaft_joint", "Shaft joints"),
    ("gap_sweep", "Gap sweep"),
    ("pole_sweep", "Pole sweep"),
```

- [ ] **Step 10: Move the tests that pin what the new inputs and results moved**

(a) `magcoupling-rs/src/gui/inputs.rs` tests: the group names gain `shaft_joint`; the shown shaft group's sections are `shaft`, `shaft.fit`, `shaft.fitting`, and its first section's rows gain the two key rows; the "gearbox" filter matches 3 -> 4 (`shaft_joint.output_shaft_fit_C` last, labelled "Gearbox shaft temperature at fitting"), its workflow runs 1 -> 2 and workbook runs 2 (spec 7.17).

Old:
```rust
        }
        assert_eq!(shown.key_design, all.key_design);
        // The clamp's own group goes; the shaft group keeps the bore and the keyway depth.
        let names: Vec<&str> = shown.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
```

New:
```rust
        }
        assert_eq!(shown.key_design, all.key_design);
        // The clamp's own group goes; the shaft joints' group stays (BL-064). The shaft group
        // keeps the bore, the keyway depth and the shaft joints' key, then their shrink fit and
        // fitting temperatures.
        let names: Vec<&str> = shown.groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
```

Old:
```rust
                "calibration",
                "materials",
                "temperature"
            ]
        );
        let shaft = shown.workflow.iter().find(|g| g.name == "shaft").unwrap();
        assert_eq!(shaft.label, "Shaft and hub joint");
        let rows: Vec<&str> = shaft
            .sections
            .iter()
            .flat_map(|s| s.entries.iter())
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(rows, ["coupling.bore_mm", "coupling.keyway_depth_mm"]);
        // The screw classes only the clamp reads go with it.
        assert!(
```

New:
```rust
                "calibration",
                "materials",
                "temperature",
                "shaft_joint"
            ]
        );
        let shaft = shown.workflow.iter().find(|g| g.name == "shaft").unwrap();
        assert_eq!(shaft.label, "Shaft and hub joint");
        let sections: Vec<&str> = shaft.sections.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(sections, ["shaft", "shaft.fit", "shaft.fitting"]);
        let rows: Vec<&str> = shaft.sections[0]
            .entries
            .iter()
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(
            rows,
            [
                "coupling.bore_mm",
                "coupling.keyway_depth_mm",
                "shaft_joint.key_width_mm",
                "shaft_joint.key_contact_mm"
            ]
        );
        // The screw classes only the clamp reads go with it.
        assert!(
```

Old:
```rust
                "materials",
                "temperature",
                "clamps"
            ]
        );
```

New:
```rust
                "materials",
                "temperature",
                "clamps",
                "shaft_joint"
            ]
        );
```

Old:
```rust
                .collect()
        };
        let gearbox = [
            "coupling.gear_ratio",
            "coupling.gear_efficiency",
            "coupling.gearbox_input_rating_Nm",
        ];
        // By label (and path), ignoring case and the surrounding blanks, in either order.
```

New:
```rust
                .collect()
        };
        // BL-064: the gearbox shaft's fitting temperature is labelled with "Gearbox" too.
        let gearbox = [
            "coupling.gear_ratio",
            "coupling.gear_efficiency",
            "coupling.gearbox_input_rating_Nm",
            "shaft_joint.output_shaft_fit_C",
        ];
        // By label (and path), ignoring case and the surrounding blanks, in either order.
```

Old:
```rust
        // Each run is headed by its group, and its section unless it is the group's own.
        let workflow = filter_inputs(catalogue.groups_in(InputOrder::Workflow), "gearbox");
        assert_eq!(workflow.len(), 1);
        assert_eq!(
            workflow[0].heading(),
            "Requirements and operating conditions / Drive and gearbox"
        );
        let workbook = filter_inputs(catalogue.groups_in(InputOrder::Workbook), "gearbox");
        assert_eq!(workbook[0].heading(), "Coupling");
        // A blank filter matches every input; a filter nothing contains, none.
        for order in InputOrder::ALL {
```

New:
```rust
        // Each run is headed by its group, and its section unless it is the group's own.
        let workflow = filter_inputs(catalogue.groups_in(InputOrder::Workflow), "gearbox");
        assert_eq!(workflow.len(), 2);
        assert_eq!(
            workflow[0].heading(),
            "Requirements and operating conditions / Drive and gearbox"
        );
        assert_eq!(
            workflow[1].heading(),
            "Shaft and hub joint / Fitting temperatures"
        );
        let workbook = filter_inputs(catalogue.groups_in(InputOrder::Workbook), "gearbox");
        assert_eq!(workbook.len(), 2);
        assert_eq!(workbook[0].heading(), "Coupling");
        assert_eq!(workbook[1].heading(), "Shaft joints");
        // A blank filter matches every input; a filter nothing contains, none.
        for order in InputOrder::ALL {
```

(b) `magcoupling-rs/src/gui/panel.rs` tests: the idle-frames comment's count (172 -> 187, the count itself is computed); the filter box's "3 of N inputs" -> "4 of N" in both orders, with the new label and its heading; and `the_clamp_tab_and_the_clamp_inputs_show_only_while_the_section_shows` gains the shaft joints' rows in both clamp states (the key rows and each new section's first row and heading; a section's last row lies outside the harness's drawn area).

Old:
```rust
        );
        assert!(!harness.panel.history.can_undo(&design));
        // The workflow order, filtered by "." (in every path): all 172 rows, both vacuum
        // permeabilities (under Calibration and model's Advanced heading when unfiltered).
        harness.panel.input_order = InputOrder::Workflow;
```

New:
```rust
        );
        assert!(!harness.panel.history.can_undo(&design));
        // The workflow order, filtered by "." (in every path): all 187 rows, both vacuum
        // permeabilities (under Calibration and model's Advanced heading when unfiltered).
        harness.panel.input_order = InputOrder::Workflow;
```

Old:
```rust
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        assert_eq!(
            count(
```

New:
```rust
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
        assert_eq!(
            count(
```

Old:
```rust
            1
        );
        for label in [
            "Gearbox ratio",
            "Gearbox efficiency",
            "Gearbox input torque rating",
        ] {
            assert_eq!(count(&output, label), 1, "{label}");
```

New:
```rust
            1
        );
        // BL-064: the gearbox shaft's fitting temperature, under the shaft group.
        assert_eq!(
            count(&output, "Shaft and hub joint / Fitting temperatures"),
            1
        );
        for label in [
            "Gearbox ratio",
            "Gearbox efficiency",
            "Gearbox input torque rating",
            "Gearbox shaft temperature at fitting",
        ] {
            assert_eq!(count(&output, label), 1, "{label}");
```

Old:
```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Coupling"), 1);
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        // No match says so; a blank filter brings the groups back.
        harness.panel.input_filter = "no such input".to_owned();
```

New:
```rust
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Coupling"), 1);
        assert_eq!(count(&output, "Shaft joints"), 1);
        assert_eq!(count(&output, &format!("4 of {total} inputs")), 1);
        // No match says so; a blank filter brings the groups back.
        harness.panel.input_filter = "no such input".to_owned();
```

Old:
```rust
            assert_eq!(count(&output, label_for("coupling.bore_mm")), 1);
            assert_eq!(count(&output, label_for("coupling.keyway_depth_mm")), 1);
            assert_eq!(count(&output, "Clamp geometry"), shown);
            assert_eq!(count(&output, label_for("clamps.boss_od_mm")), shown);
```

New:
```rust
            assert_eq!(count(&output, label_for("coupling.bore_mm")), 1);
            assert_eq!(count(&output, label_for("coupling.keyway_depth_mm")), 1);
            // BL-064: the shaft joints' key and the first rows of their shrink fit and fitting
            // temperatures show in both states: the keyed shrink fits are the joints in force,
            // not the paused clamp.
            for path in [
                "shaft_joint.key_width_mm",
                "shaft_joint.key_contact_mm",
                "shaft_joint.hub_interference_min_um",
                "shaft_joint.hub_fit_max_C",
            ] {
                assert_eq!(count(&output, label_for(path)), 1, "{path}");
            }
            assert_eq!(count(&output, "Shrink fit"), 1);
            assert_eq!(count(&output, "Fitting temperatures"), 1);
            assert_eq!(count(&output, "Clamp geometry"), shown);
            assert_eq!(count(&output, label_for("clamps.boss_od_mm")), shown);
```

(c) `magcoupling-rs/src/gui/guidance.rs`, `the_fix_label_is_the_display_label_and_still_names_one_input` (plan deviation PD-6): C-3 heads a fix's label where another input's label equals it or ends with it; the new "Shaft key width" ends with the paused clamp's "Key width", so the clamp's own fix text gains its group heading (no on-screen fix names the clamp while it is paused). The binding label stays; the list grows to eight.

Old:
```rust
        assert_ne!(label_for(RING), size_controls::DIAMETER);
        // C-3 runs on the display labels: the inputs whose fix text adds a group heading are
        // still the seven whose label another input's equals or ends with (map-help 3.5).
        let headed: Vec<&str> = InputCatalogue::get()
            .all()
```

New:
```rust
        assert_ne!(label_for(RING), size_controls::DIAMETER);
        // C-3 runs on the display labels: the inputs whose fix text adds a group heading are
        // the eight whose label another input's equals or ends with (map-help 3.5): BL-064's
        // "Shaft key width" ends with the paused clamp's "Key width".
        let headed: Vec<&str> = InputCatalogue::get()
            .all()
```

Old:
```rust
                "clamps.alloy",
                "clamps.boss_od_mm",
            ]
        );
```

New:
```rust
                "clamps.alloy",
                "clamps.boss_od_mm",
                "clamps.key_width_mm",
            ]
        );
```

(d) `magcoupling-rs/src/gui/results_table.rs`, `the_exports_carry_the_gearbox_check_with_no_cell`: the gearbox check's CSV line is now followed by the shaft joints', not the sweeps.

Old:
```rust
        };
        // BL-059: the gearbox check is a Rust-only result, like the space claim. The CSV and
        // JSON exports carry its engine label and verdict, last of the result groups (after the
        // housing group, before the sweeps), with no unit and no spreadsheet cell; each of its
        // three verdicts exports as the engine writes it.
        const PATH: &str = "gearbox.cold_high_check";
        let label = result_info(PATH).expect("a result").meta.label;
```

New:
```rust
        };
        // BL-059: the gearbox check is a Rust-only result, like the space claim. The CSV and
        // JSON exports carry its engine label and verdict, after the housing group and before the
        // shaft joints (BL-064), with no unit and no spreadsheet cell; each of its three verdicts
        // exports as the engine writes it.
        const PATH: &str = "gearbox.cold_high_check";
        let label = result_info(PATH).expect("a result").meta.label;
```

Old:
```rust
            assert!(lines[row - 1].starts_with("housing."), "{}", lines[row - 1]);
            assert!(
                lines[row + 1].starts_with("gap_sweep["),
                "{}",
                lines[row + 1]
```

New:
```rust
            assert!(lines[row - 1].starts_with("housing."), "{}", lines[row - 1]);
            assert!(
                lines[row + 1].starts_with("shaft_joint."),
                "{}",
                lines[row + 1]
```

(e) `magcoupling-rs/src/gui/trace.rs`: 24 results, none explained yet (Task 3 writes their records).

Old:
```rust
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 393 of the 1088 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.
```

New:
```rust
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 393 of the 1112 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.
```

Old:
```rust
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (393, 1088));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

New:
```rust
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (393, 1112));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
```

The result-count tests of `result_groups.rs`, `results_table.rs` and `spreadsheet.rs` need no edit: they compare against the computed total, and pass once the package has its heading; `spreadsheet.rs`'s input rows pass once the inputs have workflow sections.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: `test result: ok. 888 passed; 0 failed; 6 ignored` (886 at HEAD, Task 1's temperature test and this task's session test).

No `tests/explain.rs` count moves in this task: the 15 inputs move only the shaft joint results, which have no equation record until Task 3 (`check_traceability` and `no_input_moves_a_result_the_graph_says_is_independent_of_it` pass unchanged).

- [ ] **Step 11: Docs**

`magcoupling-rs/README.md`, thirteen edits; each "Old" is a substring of one line or paragraph.

(a) The Layout table: after the `src/engine/gearbox.rs` row (around :147). Old:
```text
at the rating passes, +inf fails. Fatigue from repeated slips is not checked: the repeated slip swing as a fatigue load is BL-066 |
```
New:
```text
at the rating passes, +inf fails. Fatigue from repeated slips is not checked: the repeated slip swing as a fatigue load is BL-066 |
| `src/engine/shaft_joint.rs` | BL-064 shaft joints: `ShaftJointInputs` (15, the first top-level Rust-only input group) and `ShaftJointResults` (24, Rust-only: `shaft_joint.check` and 23 numbers). Each rotor's keyed shrink fit, the inner hub on the wheel shaft and the cup on the gearbox output shaft (both on `coupling.bore_mm`), must hold the safety factor times cold-high at its loosest, stay within yield ÷ `YIELD_FACTOR` (1.2) at its tightest and be fitted within its allowed temperature; the key alone is checked at the same torque as a backup, never added to the fit (owner decision 2A). The bonded inner hub's allowed temperature is `hub_fit_allowed`: the lowest of the input, the adhesive's design limit and the inner ring's own single-ring onset less `CURE_MARGIN_MIN_C`, NaN if any is NaN (owner R-15). Plane-stress Lamé on a solid shaft (`joint`, per joint; `joints`, `joint_specs`), the temperature ends (the minimum magnet temperature; the adhesive's design limit with the rotor the rotor-over-shaft allowance warmer), the shafts' shrink from NIST's Fe-2.25Ni fit below 265 K (`shaft_shrink_um`). The causes in the badge's order (`CAUSES`, then the inner hub before the cup, `failing`); a joint with no wall around its bore (a part diameter typed at or below zero included, ruling C-3) or a reversed band is not judged; a value that is not a number reads unknown first, then a safety factor not above zero (ruling C-4; `verdict`); the badge never quotes two equal numbers. `api::shaft_joint_links` rebuilds the 15 engine values it reads from a design's inputs and results |
```

(b) The Tests table: insert before the `tests/size_controls.rs` row (around :662). Old:
```text
| `tests/size_controls.rs` |
```
New:
```text
| `tests/shaft_joint.rs` | BL-064 shaft joints end to end and on `joint`: the 15 inputs and 24 results are Rust-only, in the spec's order, after the clamp inputs and the gearbox results; the verdict texts; per µm (inner hub 7.738 MPa, 2.0009 N·m, 17.930 MPa; cup 8.132, 2.5073, 17.879); the defaults pass (holding 9.4844 / 16.899 against 7.6466 N·m, bore stress 241.70 / 312.53 against 345.83 MPa, key 127.44 / 101.95 MPa, fitting 93.267 / 280.16 against 100 / 300 °C; windows 12.08-18.83 and 11.31-23.86 µm); the BL-067 corner reads cold-high (6.3647 N·m) and passes; the five probe designs (304 slips at the inner hub); a band minimum too low slips at its joint, a maximum too high is over the stress limit with the fitting cause after it; causes in order, then the inner hub before the cup; the 304 and aluminium windows are empty; 416 loosens cold, 304 hot; an uncooled or dry-ice wheel shaft cannot fit the inner hub; the allowed hub temperature is the lowest of three (owner R-15: N42 parts 59.83 °C, N50M 88.80 °C, ferrite never capped, 2214 Hi-Temp with 150 °C typed 133.36 °C) and reads the inner ring's own onset with mixed rings (inner B842SH, outer B842 passes; inner N42H, outer SmCo_2_17_30 needs 125.8 against 102.8 °C); a long magnet makes the cup slip first; equality passes and the next float fails; the badge never quotes two equal numbers; NaN in any of the 15 inputs or 15 engine values reads unknown; the allowed-temperature helper drops no NaN and names the limit that binds (ties the earlier); a reversed band and a missing wall are not judged (a boss typed at -22 mm has none); the key's edges (keyway 0.3, 1.0 and 0 mm, a contact typed -0.5 or 0, boss length 0 and 0.5, hub length 3 mm); each margin is finite with its sign the verdict; the shaft shrink (NIST, continuous at 265 K, falling over the slider); +inf cold-high slips; the joints rebuilt from the results are the engine's; the inputs move no result outside the check. |
| `tests/size_controls.rs` |
```

(c) The trace paragraph (around :294). Old:
```text
the registry explains 393 of the 1088 results, so an input such as the
```
New:
```text
the registry explains 393 of the 1112 results, so an input such as the
```

(d) The Tests table's `src/gui/inputs.rs`, ... row (around :675). Old:
```text
393 of the 1088 results are explained, every input traceable
```
New:
```text
393 of the 1112 results are explained, every input traceable
```

(e) The Python-parity paragraph (around :702-705). Old:
```text
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added. Nor does a Rust-only result group (`warnings`,
`housing`, `gearbox`: every field declared with `out_rust_only`): it has no `MODULES` entry
```
New:
```text
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added, nor when a Rust-only input group is (`shaft_joint`,
BL-064). Nor does a Rust-only result group (`warnings`, `housing`, `gearbox`, `shaft_joint`:
every field declared with `out_rust_only`): it has no `MODULES` entry
```

(f) The inputs panel paragraph (around :268; plan deviation PD-4). Old:
```text
  retainers, Shaft and hub joint, Materials (the adhesive and its bondlines in one section),
```
New:
```text
  retainers, Shaft and hub joint (the bore and the shaft key, then the shrink fit and the fitting
  temperatures, BL-064), Materials (the adhesive and its bondlines in one section),
```

(g) The XLSX Summary's share link (around :386). Old:
```text
the share link as text (about 2,500 characters, over Excel's
```
New:
```text
the share link as text (about 2,650 characters, over Excel's
```

(h) The Tests table's `src/gui/session.rs` row (around :677). Old:
```text
the default link stays under 2,500 characters, and the one 49fbd3f wrote (before the spreadsheet export's zlib-rs backend) still loads.
```
New:
```text
the default link stays under 2,700 characters (2,643 with BL-064's shaft joint inputs), and the one 49fbd3f wrote (before the spreadsheet export's zlib-rs backend) still loads; that link and a design file from before BL-064 load the shaft joint inputs at their defaults.
```

(i) The Tests table's `src/gui/inputs.rs`, ... row (around :674), the "gearbox" filter: this task adds the fourth input labelled "Gearbox" and moves both filter tests (ruling C-1). Old:
```text
(and "gearbox" still exactly the three gearbox inputs)
```
New:
```text
(and "gearbox" exactly the four inputs labelled with it: the drive's three and the gearbox shaft's fitting temperature, BL-064)
```

(j) The shaft-clamp pause bullet (around :459), the inputs side while paused (stale since this task's placement, plan deviation PD-4). Old:
```text
Shaft and hub joint group keeps the bore and the keyway depth
```
New:
```text
Shaft and hub joint group keeps the bore, the keyway depth and the shaft joints' key, shrink-fit and fitting-temperature inputs (BL-064)
```

(k) The `src/gui/inputs.rs`, ... `src/gui/results_table.rs` row (around :677), the pause (stale since this task's placement). Old:
```text
(the shaft group keeps the bore and the keyway depth)
```
New:
```text
(the shaft group keeps the bore, the keyway depth and the shaft joints' inputs, BL-064)
```

(l) The same row, the exports (the gearbox clause, stale since this task's test moved the gearbox check's neighbour to the shaft joints). Old:
```text
after the housing group and before the sweeps, with no unit and no cell;
```
New:
```text
after the housing group and before the shaft joints, with no unit and no cell; the CSV and JSON exports carry the shaft joints' 24 results (BL-064) together in schema order, after the gearbox check and before the sweeps, each with its engine label, its value at full precision, its unit and no cell (a value that is not a number and +inf as text), and the JSON design their 15 inputs, which read back as they were;
```

(m) The Tests table's `tests/robustness.rs` row (around :665). Old:
```text
and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic.
```
New:
```text
a debug-build time bound per frame (`compute_all` plus `headline`), and BL-064's shaft joint check on typed values no slider reaches (a bore of 0 reads unknown, never a pass; a key wider than the bore's circumference holds 0 N·m with a needed minimum of +inf; a safety factor of 0 or -1 reads unknown, never a pass: `degenerate_shaft_joint_inputs_never_pass`; no adhesive, code 0, reads unknown). The engine must never panic.
```

`magcoupling-rs/src/lib.rs`, the crate doc:

Old:
```rust
//! length override) are Rust-only and default to the ported behaviour. The
//! forward calculation also reports the A1 space claim ([`engine::housing`],
//! with the axial housing that follows the length override) and the BL-059
//! gearbox check ([`engine::gearbox`]: cold-high at the gearbox input against
//! its rating). Beyond it:
//! [`engine::sizing::solve`] (inverse sizing, Addendum A1) and
//! [`engine::assumptions`] (the A3 assumptions panel).
```

New:
```rust
//! length override) are Rust-only and default to the ported behaviour. The
//! forward calculation also reports the A1 space claim ([`engine::housing`],
//! with the axial housing that follows the length override), the BL-059
//! gearbox check ([`engine::gearbox`]: cold-high at the gearbox input against
//! its rating) and the BL-064 shaft joints ([`engine::shaft_joint`]: each
//! rotor's keyed shrink fit, with its own Rust-only inputs). Beyond it:
//! [`engine::sizing::solve`] (inverse sizing, Addendum A1) and
//! [`engine::assumptions`] (the A3 assumptions panel).
```

`docs/ai/02-system.yaml`, three edits (spec 8; Task 5's Step 6b edits the guidance, pause and stored-3D lines and, in its edit (8), appends this invariant's greying and stored-3D sentence).

(a) The Rust-only line (around :89). Old:
```text
      - Rust-only inputs and results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests; gen_differential.py never passes a Rust-only input to Python.
```
New:
```text
      - Rust-only inputs and results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests; gen_differential.py never passes a Rust-only input to Python. The shaft_joint input group (BL-064) is the first top-level Rust-only input group.
```

(b) The share-link invariant (around :475). Old:
```text
A magcoupling share link names every input and is about 2,500 characters, so the spreadsheet writes it as text."
```
New:
```text
A magcoupling share link names every input and is about 2,650 characters (2,643 at the defaults since BL-064), so the spreadsheet writes it as text."
```

(c) A new invariant, before the gearbox check's (around :384). Old:
```text
  - "The magcoupler's gearbox check (BL-059, gearbox.cold_high_check) compares one torque with one rating:
```
New:
```text
  - "The magcoupler's shaft joint check (BL-064, shaft_joint.check) reads cold-high (metal.torque_cold_high_Nm) and, for the bonded inner hub's allowed fitting temperature, the inner ring's own single-ring onset (temperature.demag.inner_onset_single_ring_C), never the governing ring's: that temperature is the lowest of the input, the adhesive's design limit and that onset less CURE_MARGIN_MIN_C, NaN if any is NaN (shaft_joint::hub_fit_allowed). Cold-high can reach each joint; it is not a bound on them (backlash, resonance and re-lock can exceed it on the gearbox side; BL-067). Not checked: fretting and fatigue of the joints (BL-070), the gearbox's own shaft, bearings and seal, the keyway corner's stress concentration and the key's shear."
  - "The magcoupler's gearbox check (BL-059, gearbox.cold_high_check) compares one torque with one rating:
```

`docs/ai/03-structure.yaml`, four edits.

(a) A module entry after `gearbox:` (around :53). Old:
```text
fatigue from repeated slips is not checked, the fatigue check is BL-066)"
```
New:
```text
fatigue from repeated slips is not checked, the fatigue check is BL-066)"
    shaft_joint: "shaft_joint.rs (BL-064 shaft joints: ShaftJointInputs = DesignInputs::shaft_joint, 15 Rust-only inputs, the first top-level Rust-only input group; ShaftJointResults = DesignResults::shaft_joint, 24 Rust-only results, computed in api.rs after gearbox::compute from ShaftJointLinks (15 engine values; api::shaft_joint_links rebuilds them from a design's inputs and results): each rotor's keyed shrink fit (inner hub on the wheel shaft, cup on the gearbox output shaft) holds the safety factor times cold-high at its loosest, stays within yield / YIELD_FACTOR at its tightest, can be fitted within its allowed temperature (hub_fit_allowed, owner R-15: the lowest of the input, the adhesive's design limit and the inner ring's own single-ring onset less CURE_MARGIN_MIN_C), and the key alone is checked as a backup (owner decision 2A); joint (Lame, per joint), joint_specs, joints, failing (CAUSES order, then inner hub before cup), verdict (unknown first on any NaN value or judged term, then on a safety factor not above zero; a joint with no wall, a part diameter typed at or below zero included, or a reversed band is not judged), shaft_shrink_um (NIST Fe-2.25Ni below 265 K); the verdict constants SHAFT_JOINTS_PASS, SHAFT_JOINT_UNKNOWN and its four reasons, FIT_SLIPS, BORE_OVER_ITS_LIMIT, CANNOT_BE_FITTED, KEY_OVER_ITS_LIMIT, KEY_HAS_NO_BEARING_AREA; the badge never quotes two equal numbers)"
```

(b) `addendum_only` (around :65). Old:
```text
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, gearbox.rs, explain/]
```
New:
```text
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, gearbox.rs, shaft_joint.rs, explain/]
```

(c) The `api:` entry (around :62; an unquoted scalar: no ": " added). Old:
```text
OPTIONAL_SEEDS (the result an optional input enters at, which gui::inputs re-exports); DesignInputs::validate;
```
New:
```text
OPTIONAL_SEEDS (the result an optional input enters at, which gui::inputs re-exports); shaft_joint_links (BL-064, the shaft joint check's engine values rebuilt from a design's inputs and results); DesignInputs::validate;
```

(d) The `tests:` line (around :75; spec 8, review SF-14; an unquoted scalar). Old:
```text
the band (cold-high over hot-low) the same at every magnet size, gap, count and back iron), size_controls.rs
```
New:
```text
the band (cold-high over hot-low) the same at every magnet size, gap, count and back iron), shaft_joint.rs (the BL-064 shaft-joint check end to end and on joint; the defaults and the probe designs, the edges, the never-equal badge, every unknown form, the cause order, the allowed hub temperature and the inner ring's own onset), size_controls.rs
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`. (`docs/ai/02-system.yaml` does not parse at HEAD either, at line 15 column 3, a pre-existing error BL-059 reported; these edits are inside quoted scalars and add no new error: the parser still stops at line 15.)

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count`

Expected: `test result: ok. 2 passed` (the docs-count tests read the README and `03-structure.yaml`; no count they hold moved).

- [ ] **Step 12: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`:

| binary | result |
|---|---|
| lib | 230 passed, 1 ignored |
| assumptions | 11 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 57 passed, 1 ignored |
| gearbox | 13 passed |
| grades | 12 passed |
| material_library | 4 passed |
| material_links | 14 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 13 passed |
| schema | 11 passed |
| shaft_joint | 30 passed |
| size_controls | 38 passed |
| sizing | 44 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `888 passed; 0 failed; 6 ignored`; the two app binaries 0 tests; `explain` 58 passed, 1 ignored; every other integration binary as above.

The Python parity suites (`python_schema`, `parity`, `differential`, `deviations`) pass at HEAD's counts (spec 7.3): every new input and result is `rust_only`.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`, then the same with `--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/api.rs
 M magcoupling-rs/src/engine/mod.rs
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/inputs.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/result_groups.rs
 M magcoupling-rs/src/gui/results_table.rs
 M magcoupling-rs/src/gui/session.rs
 M magcoupling-rs/src/gui/trace.rs
 M magcoupling-rs/src/lib.rs
 M magcoupling-rs/tests/data/input_schema.json
 M magcoupling-rs/tests/gearbox.rs
 M magcoupling-rs/tests/robustness.rs
 M magcoupling-rs/tests/schema.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
?? magcoupling-rs/src/engine/shaft_joint.rs
?? magcoupling-rs/tests/shaft_joint.rs
```

- [ ] **Step 13: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T2.txt` with the Write tool:
```text
feat(magcoupler): the shaft joint check, keyed shrink fits on both rotors (BL-064 T2)

The engine's check of the two rotor-to-shaft joints, each a keyed shrink fit on the
Ø10 bore: the inner hub on the wheel shaft and the cup on the gearbox output shaft.

- engine::shaft_joint (Rust-only): 15 inputs (the owner's bands 13-18 and 15-22 µm,
  safety factor 2, friction, smoothing, the rotor-over-shaft allowance, the DIN 6885-1
  3 x 3 key, the fitting temperatures) and 24 results. The fit alone must hold the
  safety factor times cold-high at its loosest, the part stays within yield / 1.2 at
  its tightest, it must be fitted within its allowed temperature (the bonded inner
  hub's is the lowest of the input, the adhesive's design limit and the inner ring's
  own single-ring onset less the cure margin, owner R-15), and the key alone is
  checked at the same torque as a backup (owner decision 2A). One verdict in a fixed
  cause order; a value that is not a number reads unknown first; the badge never
  quotes two equal numbers. The defaults pass: "Both shaft joints hold and can be
  fitted".
- api::shaft_joint_links rebuilds the engine values from a design's inputs and
  results for the guide and the tests.
- The inputs' place in the inputs panel (spec 5.3: the shaft section's key rows, the
  Shrink fit and Fitting temperatures sections) and the results' package heading;
  the default share link grows to 2,643 characters (guard 2,700); input_schema.json
  re-blessed. Moved: the trace totals (393 of 1,112), the gearbox group's neighbour,
  the "gearbox" filter (4 matches), the C-3 fix-label list (eight).

Not yet on the dashboard: the row, CHECKS, the verdict levels and the guide come
with Task 5; until then the check shows in the results table without a badge.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/engine/shaft_joint.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/session.rs magcoupling-rs/src/gui/trace.rs magcoupling-rs/src/lib.rs magcoupling-rs/tests/shaft_joint.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/tests/schema.rs magcoupling-rs/tests/gearbox.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T2.txt
git status --short
```

Expected: one commit of 19 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 2 moves** (each confirmed by running in the plan-time prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so` | `(393, 1088)` -> `(393, 1112)` | 24 results, no record yet |
| `tests/gearbox.rs` `the_check_is_a_rust_only_text_result_with_no_cell_after_the_housing_group` | next row `gap_sweep[` -> `shaft_joint.` | the shaft joints' group follows the gearbox's |
| `gui::results_table::tests::the_exports_carry_the_gearbox_check_with_no_cell` | next CSV line `gap_sweep[` -> `shaft_joint.` | the same, in the CSV export |
| `gui::session::tests::a_share_link_of_the_default_design_stays_short` | `< 2_500` -> `< 2_700` | 2,643 characters with the 15 inputs (measured) |
| `gui::guidance::tests::the_fix_label_is_the_display_label_and_still_names_one_input` | seven -> eight, `clamps.key_width_mm` added | "Shaft key width" ends with the paused clamp's "Key width" (plan deviation PD-6) |
| `gui::inputs::tests::the_shown_catalogue_leaves_out_the_paused_section_s_inputs` | groups + `shaft_joint`; the shaft group's sections `shaft`, `shaft.fit`, `shaft.fitting`; its first section's rows + the two key rows | the inputs' place (plan deviation PD-4) |
| `gui::inputs::tests::every_input_is_in_exactly_one_section_in_schema_order` | group names + `shaft_joint` | the new top-level input group |
| `gui::inputs::tests::the_filter_matches_label_path_and_cell_by_section_in_the_order_shown` | "gearbox" 3 -> 4 matches, workflow runs 1 -> 2, workbook runs 2 | "Gearbox shaft temperature at fitting" |
| `gui::panel::tests::the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design` | "3 of N inputs" -> "4 of N" (both orders); the new label and headings counted | the same |
| `gui::panel::tests::idle_frames_change_no_input` | comment "all 172 rows" -> "all 187 rows" | the count is computed; it failed only until the inputs had sections |
| `gui::panel::tests::the_clamp_tab_and_the_clamp_inputs_show_only_while_the_section_shows` | extended | the joint's rows show in both clamp states |
| `tests/schema.rs` `engine_owned_help_is_plain_and_short` | list extended | a Rust-only input and the check join |
| `tests/robustness.rs` `an_invalid_adhesive_code_selects_no_adhesive` | extended | no adhesive reads unknown on the shaft joint check |
| `result_groups`, `results_table`, `spreadsheet` result counts (`every_result_is_in_exactly_one_group`, `the_lines_group_the_rows_with_only_the_headline_open_at_first`, `every_result_is_on_one_row_by_physics_chain_with_its_value`) and `spreadsheet`'s `every_input_is_on_one_row_with_its_value_unit_and_cell_in_either_order` | no edit | they compare against computed totals: green once `PACKAGE_LABELS` heads the group and the inputs have sections |

No `tests/explain.rs` test moves in Task 2 (the inputs move only results without records until Task 3).

---

### Task 3: The explain layer: the shaft joint records, their scope chain and heading, and the four assumptions

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 4.5 (records, assumptions and scope), 5.4 (`CHAIN_LABELS`), 7 items 10 and 12, and the pinned counts of
item 17 that this task moves (the scope 170 -> 193, the trace totals, the assumption count). One equation record per
numeric `shaft_joint.*` result (23), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every
correction on, so the drift guard holds every number of the check to the engine; their scope chain `shaft_joint` and its
results-table heading; the four `ASSUMPTIONS` rows and the `.assumption()` flags on their inputs, with the widened
source rule. Task 1 already added the inner ring's onset record and its `demagnetization` chain path: this task does not
repeat them.

Line numbers are at Task 2's commit; trust the code and keep the intent. Commands run from the repository root
`C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout (checked in the plan-time prototype:
`cargo fmt --check` is silent).

**Plan deviations in this task** (PD-9 to PD-11 of the Deviation index):
- **PD-9 (`tests/data/input_schema.json` is re-blessed here too, not only in Task 2).** The four `.assumption()`
  flags change the blessed input schema: `input_schema_json_is_current` fails with "tests/data/input_schema.json is
  stale" until it is re-blessed (exactly four lines, `"assumption": false` -> `true`, checked in the prototype). The
  flags cannot move to Task 2: `every_flagged_input_is_in_exactly_one_row` needs their `ASSUMPTIONS` rows in the same
  commit, and `each_assumption_moves_what_depends_on_it_and_nothing_else` needs an explained result downstream of each
  row, which only this task's records give. So Step 8 re-blesses it with `MAGCOUPLING_BLESS=1`. Gate 12 is unaffected:
  `gen_differential.py` drops every `rust_only` input from the schema it reads (`outputs()`, "the Python engine has no
  Rust-only input"), so no generated file changes. The Global Constraints' re-bless rule names this task (ruling
  C-6; Task 2's plan deviation PD-5 is the same rule).
- **PD-10 (spec 4.5: "`symbols.rs` gives the 15 inputs symbols").** The records read three inputs no record read
  before, each of which the registry refuses without a symbol: `coupling.keyway_depth_mm` (the key arc, the key's
  contact cap and the no-wall rule) and `materials.steel.cte_per_C`, `materials.steel.modulus_GPa` (the default
  back-iron pick's values are the Materials inputs, spec 4.5). So `SYMBOLS` gains 18 entries, not 15.
- **PD-11 (spec 4.5: the allowed hub temperature "as `min(min(…), … - 10)` … with a NaN arm").** The markup's `min`
  is Python's fold (`eval.rs`, `Func::Min`, through `compat::py_min`): it keeps a NaN only in its first argument, so
  the spec's form would read your 100 °C limit where the inner onset is NaN (the edge point "the inner ring's
  single-ring onset 0/0" reaches one, the two limits at their defaults) while the engine reads NaN, and would drop a
  NaN adhesive limit the same way, as the spec feared. The markup has no NaN test (its only one would be `x != x`,
  which typesets as "ϑ ≠ ϑ"). The record is written instead as four arms: each of the three values when it is at most both
  others, in the order your limit, the adhesive's, the onset's (the order `hub_fit_allowed` names them, ties to the
  earlier), else `nan`. A NaN in any of the three fails every comparison, so it falls to `nan`, whichever its position.
  Same values bit for bit (`the_allowed_hub_temperature_record_is_the_helper_s_lowest_of_three` checks seven cases and
  a NaN in each position against the helper), and the arms name which limit binds, as the helper does.

**Files:**
- Create: `magcoupling-rs/src/engine/explain/records/shaft_joint.rs`
- Modify: `magcoupling-rs/src/engine/explain/records/mod.rs`, `magcoupling-rs/src/engine/explain/tables.rs` (three
  `back_iron` fields), `magcoupling-rs/src/engine/explain/symbols.rs`, `magcoupling-rs/src/engine/explain/scope.rs`,
  `magcoupling-rs/src/engine/assumptions.rs`, `magcoupling-rs/src/engine/shaft_joint.rs` (four `.assumption()`),
  `magcoupling-rs/src/gui/result_groups.rs` (`CHAIN_LABELS`, the module doc), `magcoupling-rs/src/gui/trace.rs` (the
  module doc's totals), `magcoupling-rs/tests/data/input_schema.json` (re-blessed, never by hand: plan deviation PD-9)
- Modify (docs): `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`, `docs/ai/02-system.yaml`
- Test: `magcoupling-rs/src/engine/explain/tables.rs` (new `the_back_iron_table_reads_each_pick_s_modulus_expansion_and_yield`);
  `magcoupling-rs/tests/explain.rs` (the drift guard gains the five probe designs, 12 shaft joint edge points and the
  "shaft joint inputs" augmentation; new `the_shaft_joint_chain_is_every_numeric_shaft_joint_result_in_schema_order`,
  `the_allowed_hub_temperature_record_is_the_helper_s_lowest_of_three`,
  `the_shaft_shrink_records_follow_the_engine_over_their_sliders`; two counts move);
  `magcoupling-rs/tests/assumptions.rs` (new `the_shaft_joint_rows_are_rust_only_and_cite_the_bl064_spec`; two tests
  move); `magcoupling-rs/src/gui/result_groups.rs` and `magcoupling-rs/src/gui/trace.rs` (one test each moves)

**Interfaces:**
- Consumes: Task 2's `shaft_joint.*` inputs and results and `magcoupling::engine::shaft_joint::{hub_fit_allowed,
  shaft_shrink_um, ROOM_C}`; Task 1's `temperature.demag.inner_onset_single_ring_C` and its record; the explained
  results `metal.torque_cold_high_Nm`, `materials.circuit_backiron`, `housing.hub_length_mm`,
  `temperature.adhesive.design_limit_C`; `temperature::CURE_MARGIN_MIN_C`;
  `material_library::{BACK_IRON_CHOICES, chosen}` and each library `Material`'s `engine` and `yield_MPa`;
  `explain::{Registry, Trace, probes}`; `compat::parity_close`.
- Produces:
  - `explain::records::shaft_joint::RECORDS` (23 records, symbols `T_{fit,req}`, `E_{part}`, `α_{part}`,
    `σ_{y,part}`, `σ_{lim}`, `T_{hold,hub|cup}`, `σ_{vM,hub|cup}`, `p_{key,hub|cup}`, `δ_{need,hub|cup}`,
    `δ_{el,hub|cup}`, `Δd_{wheel}`, `Δd_{gbx}`, `δ_{fit,hub|cup}`, `ϑ_{need,hub|cup}`, `ϑ_{allow,hub|cup}`); Task 7's
    note `shaft_joint` owns these 23 and the inner onset record (24 equations) and may quote these symbols.
  - `tables` fields `("back_iron", "modulus_GPa")` (`E_{BI}`, GPa), `("back_iron", "cte_per_C")` (`α_{BI}`, 1/°C),
    `("back_iron", "yield_MPa")` (`σ_{y,BI}`, MPa).
  - `SYMBOLS` entries for the 15 `shaft_joint.*` inputs (`δ_{min,hub}`, `δ_{max,hub}`, `δ_{min,cup}`, `δ_{max,cup}`,
    `S_{fit}`, `μ_{fit}`, `δ_{sm}`, `Δϑ_{rot}`, `b_{key}`, `h_{key}`, `ϑ_{lim,hub}`, `ϑ_{lim,cup}`, `ϑ_{wheel}`,
    `ϑ_{gbx}`, `s_{join}`), `coupling.keyway_depth_mm` (`t_{kw}`), `materials.steel.cte_per_C` (`α_{st,in}`),
    `materials.steel.modulus_GPa` (`E_{st,in}`).
  - `SCOPE` chain `shaft_joint` (`Status::Explained`, the 23 numeric paths in schema order) after `geometry`, before
    `dashboard`: the union is 193 paths. `CHAIN_LABELS` 8 -> 9 with `("shaft_joint", "Shaft joints")` between
    geometry and mass.
  - `ASSUMPTIONS` 14 -> 18: `fit_friction`, `fit_smoothing`, `rotor_over_shaft`, `joining_clearance`; `.assumption()`
    on `shaft_joint.friction`, `shaft_joint.smoothing_um`, `shaft_joint.rotor_over_shaft_C`,
    `shaft_joint.joining_clearance_um`.
  - For Task 5: `the_headline_comes_first_then_the_chains_then_the_other_results` asserts
    `section("shaft_joint.check") == ("shaft_joint", true)` (the package group under the other results, as long as
    the check is no dashboard row); Task 5's `DASHBOARD` row moves it to the headline, and Task 5 replaces exactly
    these three lines (the comment and the assertion, nothing above them):
    ```rust
            // The verdict is no chain path: until it is a dashboard row it sits in the package
            // group under the other results.
            assert_eq!(section("shaft_joint.check"), ("shaft_joint", true));
    ```
    with
    ```rust
            // The verdict is no chain path: it is a dashboard row, so the headline's (BL-064).
            assert_eq!(section("shaft_joint.check"), ("headline", false));
    ```

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/engine/explain/tables.rs`, the test module: a test that the three new back-iron fields read
what `PartProperties::hub` reads for every pick (around :239, before `corrected_library_fields_read_as_the_engine_uses_them`).
The default pick's three values are the spec's literal pin (section 3.2: 4140, 205 GPa, 12.3e-6 /°C, 415 MPa).

Old:
```rust
    #[test]
    fn corrected_library_fields_read_as_the_engine_uses_them() {
```

New:
```rust
    #[test]
    fn the_back_iron_table_reads_each_pick_s_modulus_expansion_and_yield() {
        // BL-064: the shaft joint records read the hub, cup and boss material in effect
        // (`PartProperties::hub`): a pick's engine modulus and expansion, and its library yield.
        let read = |code: i64, field: &str| match lookup("back_iron", &Value::Int(code), field) {
            Ok(Some(Value::Num(x))) => x,
            other => panic!("back_iron {code} {field}: {other:?}"),
        };
        for &(code, id) in &BACK_IRON_CHOICES {
            let m = chosen(&BACK_IRON_CHOICES, code).expect("a choice names a material");
            assert_eq!(read(code, "modulus_GPa"), m.engine.modulus_GPa, "{id}");
            assert_eq!(read(code, "cte_per_C"), m.engine.cte_per_C, "{id}");
            assert_eq!(
                read(code, "yield_MPa"),
                m.yield_MPa
                    .value
                    .expect("every back iron has a library yield"),
                "{id}"
            );
        }
        // The default pick's library values: 4140 annealed, 205 GPa, 12.3e-6 /°C, 415 MPa.
        assert_eq!(
            (
                read(1, "modulus_GPa"),
                read(1, "cte_per_C"),
                read(1, "yield_MPa")
            ),
            (205.0, 12.3e-6, 415.0)
        );
    }

    #[test]
    fn corrected_library_fields_read_as_the_engine_uses_them() {
```

(b) `magcoupling-rs/tests/explain.rs`. Ten edits.

(b1) The module doc (:4-13): the guard's points now include the probe designs and the shaft joint inputs.

Old:
```rust
//! text exact), corrections on (`compute_all`, what users see), at: the defaults; every
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override, the coercivity from the inputs with two ferrite betas,
//! the sleeve and cap materials, E17's free-space fields), without which the tau7-tau11
//! records would compare 0 with 0 and the ferrite and material branches would never run.
```

New:
```rust
//! text exact), corrections on (`compute_all`, what users see), at: the defaults; the five
//! probe designs; the edge points; every differential case (`tests/data/differential/*.json`,
//! 3,391 cases, starting from the corrected defaults); and each of those again under every
//! **augmentation**, the Rust-only inputs the Python generator never varies (the harmonic set,
//! the back-iron material, the grade mode, the axial override, the coercivity from the inputs
//! with two ferrite betas, the sleeve and cap materials, E17's free-space fields, the shaft
//! joint inputs), without which the tau7-tau11 records would compare 0 with 0 and the ferrite,
//! material and shaft joint branches would never run.
```

(b2) The imports (:46-48).

Old:
```rust
use magcoupling::engine::model::{ModelResults, blocks_fit};
use magcoupling::engine::size_controls::with_ring_radius;
use magcoupling::engine::temperature::{NO_COLD_ONSET, RING_INNER, RING_OUTER, demag_onset_C};
```

New:
```rust
use magcoupling::engine::model::{ModelResults, blocks_fit};
use magcoupling::engine::shaft_joint;
use magcoupling::engine::size_controls::with_ring_radius;
use magcoupling::engine::temperature::{
    CURE_MARGIN_MIN_C, NO_COLD_ONSET, RING_INNER, RING_OUTER, demag_onset_C,
};
```

(b3) A new augmentation, last in `augmentations()` (after "everything", around :157). The shaft joint inputs are
Rust-only, so no differential case moves them; without it every record that reads one would see it at one value only,
which the guard's anti-vacuity check refuses. Every value is on its slider and grid; both shafts land below 265 K, the
shrink's NIST arm.

Old:
```rust
            ("coupling.magnets.axial_length_mm", Value::Num(20.0)),
        ],
    ));
    v
}
```

New:
```rust
            ("coupling.magnets.axial_length_mm", Value::Num(20.0)),
        ],
    ));
    // BL-064: the shaft joint inputs (Rust-only) off their defaults, each on its slider: a
    // lower safety factor, a lower friction, a finer finish, a warmer rotor, a wider key with
    // less contact, the bonded hub's limit at 110 °C and the bare cup's at 250 °C, the wheel
    // shaft in dry ice and the gearbox shaft at its cold-spray floor (both below 265 K, the
    // shrink's NIST arm), and less joining clearance.
    v.push((
        "shaft joint inputs",
        vec![
            ("shaft_joint.hub_interference_min_um", Value::Num(11.5)),
            ("shaft_joint.hub_interference_max_um", Value::Num(20.0)),
            ("shaft_joint.cup_interference_min_um", Value::Num(12.5)),
            ("shaft_joint.cup_interference_max_um", Value::Num(25.0)),
            ("shaft_joint.safety_factor", Value::Num(1.5)),
            ("shaft_joint.friction", Value::Num(0.12)),
            ("shaft_joint.smoothing_um", Value::Num(3.0)),
            ("shaft_joint.rotor_over_shaft_C", Value::Num(30.0)),
            ("shaft_joint.key_width_mm", Value::Num(4.0)),
            ("shaft_joint.key_contact_mm", Value::Num(1.0)),
            ("shaft_joint.hub_fit_max_C", Value::Num(110.0)),
            ("shaft_joint.cup_fit_max_C", Value::Num(250.0)),
            ("shaft_joint.wheel_shaft_fit_C", Value::Num(-78.0)),
            ("shaft_joint.output_shaft_fit_C", Value::Num(-20.0)),
            ("shaft_joint.joining_clearance_um", Value::Num(8.0)),
        ],
    ));
    v
}
```

(b4) In `edge_points()`, beside `on_edge` (around :193): a closure that asserts each shaft joint point is where its
label says, as `on_edge` does for the temperature points (a point off its edge would pin nothing).

Old:
```rust
    let num = Value::Num;
    let onset = default_num("temperature.demag.onset_skipping_C");
```

New:
```rust
    // The shaft joint points (BL-064) check that they are where their label says too:
    // `joint_edge` applies the sets to the defaults and asserts `holds` of the results there.
    let joint_edge = |label: &'static str,
                      sets: Vec<(&'static str, Value)>,
                      holds: fn(&DesignResults) -> bool| {
        let results = compute_all(&defaults_with(label, &sets));
        assert!(
            holds(&results),
            "edge point {label} is not on its edge: {:?}",
            results.shaft_joint
        );
        (label, sets)
    };
    let num = Value::Num;
    let onset = default_num("temperature.demag.onset_skipping_C");
```

(b5) The 12 shaft joint edge points, after Task 1's mixed-ring point (around :463). Each takes a `cases` arm no
differential case reaches, or gives a value term its second value (the guard's two anti-vacuity rules): the no-wall arm
of each joint's four Lamé records, the keyless arms (the key pressures' 0 and the key arc's 0), each key's +inf arm,
the key contact capped by the keyway at two depths, the wheel shrink's linear arm, the allowed temperature's NaN arm
(from finite inputs: `set` refuses NaN), its adhesive arm at two adhesive limits and its onset arm (spec 3.6's
`breaking(n42_parts)` geometry-free case).

Old:
```rust
            "inner B842SH, outer B842: the outer ring governs",
            vec![("coupling.magnets.part_outer", text("B842"))],
        ),
```

New:
```rust
            "inner B842SH, outer B842: the outer ring governs",
            vec![("coupling.magnets.part_outer", text("B842"))],
        ),
        // BL-064, the shaft joint records: the branches no differential case reaches (the
        // shaft joint inputs are Rust-only; the "shaft joint inputs" augmentation moves them).
        joint_edge(
            // A 12 mm boss on a 16 mm bore: no wall around the cup's bore (bore over outer
            // diameter at least 1), so its four Lamé rows are NaN; the inner hub's are numbers.
            "shaft joints: bore 16 mm, boss 12 mm: no wall around the cup's bore",
            vec![
                ("coupling.bore_mm", num(16.0)),
                ("metal.boss_od_mm", num(12.0)),
            ],
            |r| {
                let [hub, cup] = lame_rows(r);
                cup.iter().all(|x| x.is_nan()) && hub.iter().all(|x| x.is_finite())
            },
        ),
        joint_edge(
            // A 2.1 mm keyway on a 16 mm bore cuts through the inner hub's 2.1 mm wall: its four
            // Lamé rows are NaN; the cup's are numbers.
            "shaft joints: bore 16 mm, keyway 2.1 mm: the keyway cuts through the inner hub's wall",
            vec![
                ("coupling.bore_mm", num(16.0)),
                ("coupling.keyway_depth_mm", num(2.1)),
            ],
            |r| {
                let [hub, cup] = lame_rows(r);
                hub.iter().all(|x| x.is_nan()) && cup.iter().all(|x| x.is_finite())
            },
        ),
        joint_edge(
            // No keyway: no key term (both key pressures read 0) and no key arc off the contact.
            "shaft joints: no keyway",
            vec![("coupling.keyway_depth_mm", num(0.0))],
            |r| {
                (
                    r.shaft_joint.hub_key_pressure_MPa,
                    r.shaft_joint.cup_key_pressure_MPa,
                ) == (0.0, 0.0)
            },
        ),
        joint_edge(
            // No boss: the cup's bore runs through its web alone, no longer than the key is
            // wide, so the cup's key has no bearing length (+inf).
            "shaft joints: boss length 0: the cup's key has no bearing length",
            vec![("metal.boss_length_mm", num(0.0))],
            |r| {
                r.shaft_joint.cup_key_pressure_MPa == f64::INFINITY
                    && r.shaft_joint.hub_key_pressure_MPa.is_finite()
            },
        ),
        joint_edge(
            // The hub length at its slider's bottom, 3 mm, the key's width: the inner hub's key
            // has no bearing length (+inf).
            "shaft joints: hub length 3 mm: the inner hub's key has no bearing length",
            vec![("metal.hub_length_mm", num(3.0))],
            |r| {
                r.shaft_joint.hub_key_pressure_MPa == f64::INFINITY
                    && r.shaft_joint.cup_key_pressure_MPa.is_finite()
            },
        ),
        // A keyway shallower than the key's contact height caps the contact at its depth. Two
        // depths, so the capped contact takes two values.
        joint_edge(
            "shaft joints: keyway 1 mm caps the key's contact height",
            vec![("coupling.keyway_depth_mm", num(1.0))],
            |r| key_contact_capped_at(r, 1.0),
        ),
        joint_edge(
            "shaft joints: keyway 0.3 mm caps the key's contact height",
            vec![("coupling.keyway_depth_mm", num(0.3))],
            |r| key_contact_capped_at(r, 0.3),
        ),
        joint_edge(
            // An uncooled wheel shaft: above 265 K the shrink is the linear expansion's, and at
            // room temperature it is 0.
            "shaft joints: wheel shaft at 20 °C",
            vec![("shaft_joint.wheel_shaft_fit_C", num(shaft_joint::ROOM_C))],
            |r| r.shaft_joint.wheel_shaft_shrink_um == 0.0,
        ),
        joint_edge(
            // No knee and no single-ring field (the coercivity from the inputs at 0 kA/m): the
            // inner ring's single-ring onset is 0/0, so the bonded inner hub's allowed fitting
            // temperature is not a number. `set` takes no NaN; finite inputs reach the record's
            // last arm this way.
            "shaft joints: the inner ring's single-ring onset 0/0",
            vec![
                ("temperature.demag.coercivity_source", Value::Int(0)),
                ("temperature.demag.hcj20_kA_m", num(0.0)),
                ("temperature.demag.h_rev_single_ring_kA_m", num(0.0)),
            ],
            |r| {
                r.temperature.demag.inner_onset_single_ring_C.is_nan()
                    && r.shaft_joint.hub_fit_allowed_C.is_nan()
            },
        ),
        // The selected adhesive's design limit caps the bonded inner hub's fitting temperature,
        // at two limits: DP460's 60 °C, and AA 326's 120 °C under the hub limit's slider top.
        joint_edge(
            "shaft joints: DP460 caps the inner hub's fitting temperature",
            vec![("temperature.adhesive.selected", Value::Int(4))],
            the_adhesive_caps_the_hub,
        ),
        joint_edge(
            "shaft joints: hub limit 130 °C: AA 326 caps the inner hub's fitting temperature",
            vec![("shaft_joint.hub_fit_max_C", num(130.0))],
            the_adhesive_caps_the_hub,
        ),
        joint_edge(
            // The library's N42 blocks in both rings: the inner ring's single-ring onset less the
            // cure margin (59.83 °C) caps the bonded inner hub's fitting temperature (owner R-15).
            "shaft joints: N42 blocks: the inner magnets' onset caps the inner hub's fitting temperature",
            vec![
                ("coupling.magnets.part_inner", text("B842")),
                ("coupling.magnets.part_outer", text("B842")),
            ],
            |r| {
                let cap = r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C;
                r.shaft_joint.hub_fit_allowed_C == cap
                    && cap < r.temperature.adhesive.design_limit_C
                    && cap < DesignInputs::default().shaft_joint.hub_fit_max_C
            },
        ),
```

(b6) The predicates' helpers, before `defaults_with` (around :515).

Old:
```rust
/// The defaults with `sets` applied (`label` names the point in a refusal).
fn defaults_with(label: &str, sets: &[(&str, Value)]) -> DesignInputs {
```

New:
```rust
/// The inner hub's and the cup's Lamé rows (holding, bore stress, needed minimum, elastic edge):
/// NaN for a joint with no wall around its bore (BL-064).
fn lame_rows(r: &DesignResults) -> [[f64; 4]; 2] {
    let s = &r.shaft_joint;
    [
        [
            s.hub_holding_Nm,
            s.hub_bore_stress_MPa,
            s.hub_needed_um,
            s.hub_elastic_max_um,
        ],
        [
            s.cup_holding_Nm,
            s.cup_bore_stress_MPa,
            s.cup_needed_um,
            s.cup_elastic_max_um,
        ],
    ]
}

/// Whether the key pressures of `r` are the defaults' with the key's contact height capped at
/// a keyway `depth_mm` below it (the pressure is inversely proportional to the contact).
fn key_contact_capped_at(r: &DesignResults, depth_mm: f64) -> bool {
    let defaults = DesignInputs::default();
    let contact_mm = defaults.shaft_joint.key_contact_mm;
    let d = compute_all(&defaults).shaft_joint;
    depth_mm < contact_mm
        && close(
            r.shaft_joint.hub_key_pressure_MPa,
            d.hub_key_pressure_MPa * contact_mm / depth_mm,
        )
        && close(
            r.shaft_joint.cup_key_pressure_MPa,
            d.cup_key_pressure_MPa * contact_mm / depth_mm,
        )
}

/// Whether the selected adhesive's design limit caps the bonded inner hub's allowed fitting
/// temperature in `r`, below the inner magnets' single-ring onset less the cure margin (BL-064,
/// owner R-15).
fn the_adhesive_caps_the_hub(r: &DesignResults) -> bool {
    let allowed = r.shaft_joint.hub_fit_allowed_C;
    allowed == r.temperature.adhesive.design_limit_C
        && allowed < r.temperature.demag.inner_onset_single_ring_C - CURE_MARGIN_MIN_C
}

/// The defaults with `sets` applied (`label` names the point in a refusal).
fn defaults_with(label: &str, sets: &[(&str, Value)]) -> DesignInputs {
```

(`close`, the file's 1e-12 relative helper, is already defined further down, around :2064.)

(b7) `guard_points()` (around :571) evaluates the five probe designs too (spec 7.12: the shaft joint records "at
every probe"; the spec's probe designs are no differential case).

Old:
```rust
/// The defaults, the edge points, then every differential case under every augmentation.
fn guard_points() -> Vec<Point> {
    let mut points = vec![Point {
        label: "defaults".into(),
        inputs: DesignInputs::default(),
    }];
    for (label, sets) in edge_points() {
```

New:
```rust
/// The defaults, the probe designs (BL-064, spec 7.12: the shaft joint records at every probe),
/// the edge points, then every differential case under every augmentation.
fn guard_points() -> Vec<Point> {
    let mut points = vec![Point {
        label: "defaults".into(),
        inputs: DesignInputs::default(),
    }];
    for (name, inputs) in probes::probe_designs() {
        points.push(Point {
            label: format!("probe design: {name}"),
            inputs,
        });
    }
    for (label, sets) in edge_points() {
```

(b8) `every_record_reproduces_the_engine_everywhere` (around :782-794): its point count gains the probe designs.

Old:
```rust
    // The defaults, the edge points, then every differential case under every augmentation
    // (no stale count: the data files are regenerated).
```
```rust
        1 + edge_points().len() + cases * augmentations().len(),
```

New:
```rust
    // The defaults, the probe designs, the edge points, then every differential case under
    // every augmentation (no stale count: the data files are regenerated).
```
```rust
        1 + probes::probe_designs().len() + edge_points().len() + cases * augmentations().len(),
```

(b9) `explained_chains_have_every_record_and_scope_paths_exist` (around :1532): the union moves 170 -> 193 (Task 1
moved it 169 -> 170); then three new tests and their helper, right after it.

Old:
```rust
    assert_eq!(
        union.len(),
        170,
        "decision 31: the chains and the dashboard (report section 7), plan A-3 decision G1's 10 new geometry callouts, and BL-064's inner-ring single-ring onset"
    );
}
```

New:
```rust
    assert_eq!(
        union.len(),
        193,
        "decision 31: the chains and the dashboard (report section 7), plan A-3 decision G1's 10 new geometry callouts, BL-064's inner-ring single-ring onset and its 23 shaft joint results"
    );
}

/// BL-064 (spec 4.5): the `shaft_joint` chain lists every numeric result of the shaft joint
/// check in schema order, each with its record; the verdict, a text, has none. The chain sits
/// after the geometry callouts and before the dashboard paths (the scope's "start here"
/// order, then the dashboard).
#[test]
fn the_shaft_joint_chain_is_every_numeric_shaft_joint_result_in_schema_order() {
    let r = Registry::build();
    let numeric: Vec<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .filter(|row| row.path.starts_with("shaft_joint.") && matches!(row.value, Value::Num(_)))
        .map(|row| row.path)
        .collect();
    assert_eq!(numeric.len(), 23, "{numeric:?}");
    let chain = SCOPE
        .iter()
        .find(|c| c.id == "shaft_joint")
        .expect("a shaft_joint chain");
    assert_eq!(chain.status, Status::Explained);
    let numeric: Vec<&str> = numeric.iter().map(String::as_str).collect();
    assert_eq!(chain.paths, numeric.as_slice());
    for path in chain.paths {
        assert!(r.equation_for(path).is_some(), "{path} has no record");
    }
    assert!(
        r.equation_for("shaft_joint.check").is_none(),
        "the verdict is a text: no record"
    );
    let at = |id: &str| SCOPE.iter().position(|c| c.id == id).expect(id);
    assert_eq!(at("shaft_joint"), at("geometry") + 1);
    assert_eq!(at("dashboard"), at("shaft_joint") + 1);
}

/// The record of `target` evaluated over `values` (path, value) alone, with its trace.
fn evaluate_over(r: &Registry, target: &str, values: &[(&str, f64)]) -> (Value, Trace) {
    let eq = r.equation_for(target).expect("a record");
    let src: BTreeMap<String, Value> = values
        .iter()
        .map(|&(path, x)| (path.to_owned(), Value::Num(x)))
        .collect();
    let mut trace = Trace::default();
    let value = r
        .evaluate(eq, &src, Some(&mut trace))
        .unwrap_or_else(|e| panic!("{target}: {e}"));
    (value, trace)
}

/// BL-064 (owner R-15 and its ring ruling): the bonded inner hub's allowed fitting temperature
/// reads your limit, the selected adhesive's design limit and the inner ring's own single-ring
/// onset (never the governing ring's), and its record is `shaft_joint::hub_fit_allowed`'s
/// value: the lowest of the three, the onset less the cure margin, ties to the earlier, an
/// onset of +inf never binding, and NaN when any of the three is NaN, in each position. The
/// drift guard reaches only a NaN onset (`set` takes no NaN); this reaches the other two.
#[test]
fn the_allowed_hub_temperature_record_is_the_helper_s_lowest_of_three() {
    let r = Registry::build();
    let target = "shaft_joint.hub_fit_allowed_C";
    let paths = [
        "shaft_joint.hub_fit_max_C",
        "temperature.adhesive.design_limit_C",
        "temperature.demag.inner_onset_single_ring_C",
    ];
    let terms: BTreeSet<&str> = r
        .equation_for(target)
        .expect("a record")
        .terms
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(terms, BTreeSet::from(paths));
    let record = |[input, adhesive, onset]: [f64; 3]| {
        let (value, _) = evaluate_over(
            &r,
            target,
            &[(paths[0], input), (paths[1], adhesive), (paths[2], onset)],
        );
        match value {
            Value::Num(x) => x,
            other => panic!("{other:?}"),
        }
    };
    let margin = CURE_MARGIN_MIN_C;
    for values in [
        [100.0, 120.0, 143.3565],       // yours (the defaults)
        [130.0, 120.0, 143.3565],       // the adhesive's
        [100.0, 120.0, 69.8326],        // the onset's (N42 blocks)
        [120.0, 120.0, 177.0],          // a tie: yours
        [100.0, 177.0, 100.0 + margin], // a tie with the onset: yours
        [150.0, 177.0, 143.3565],       // 2214 Hi-Temp, a typed 150 °C: the onset's
        [100.0, 120.0, f64::INFINITY],  // ferrite: the onset never binds
    ] {
        let (want, _) = shaft_joint::hub_fit_allowed(values[0], values[1], values[2]);
        assert_eq!(record(values).to_bits(), want.to_bits(), "{values:?}");
    }
    for nan_at in 0..3 {
        let mut values = [100.0, 120.0, 143.3565];
        values[nan_at] = f64::NAN;
        assert!(record(values).is_nan(), "a NaN {}", paths[nan_at]);
        assert_eq!(
            shaft_joint::hub_fit_allowed(values[0], values[1], values[2]).1,
            None
        );
    }
}

/// BL-064: each shaft's shrink record is the engine's `shaft_joint::shaft_shrink_um` at every
/// step of its shaft's slider, NIST's arm below 265 K and the linear one above both taken, not
/// only at the few temperatures the drift guard's points set.
#[test]
fn the_shaft_shrink_records_follow_the_engine_over_their_sliders() {
    let r = Registry::build();
    let bore = DesignInputs::default().coupling.bore_mm;
    for (target, input) in [
        (
            "shaft_joint.wheel_shaft_shrink_um",
            "shaft_joint.wheel_shaft_fit_C",
        ),
        (
            "shaft_joint.output_shaft_shrink_um",
            "shaft_joint.output_shaft_fit_C",
        ),
    ] {
        let range = slider(input);
        let steps = ((range.max - range.min) / range.step).round() as u32;
        let mut arms = BTreeSet::new();
        for k in 0..=steps {
            let celsius = range.min + f64::from(k) * range.step;
            let (got, trace) =
                evaluate_over(&r, target, &[(input, celsius), ("coupling.bore_mm", bore)]);
            let want = Value::Num(shaft_joint::shaft_shrink_um(celsius, bore));
            assert!(
                parity_close(&got, &want),
                "{target} at {celsius} °C: {got:?}, engine {want:?}"
            );
            arms.extend(trace.arms);
        }
        assert_eq!(arms.len(), 2, "{target}: both arms over the slider");
    }
}
```

(`slider`, the file's slider-range helper, is defined further down, around :2069.)

(b10) `each_assumption_moves_what_depends_on_it_and_nothing_else` (around :4470): 15 -> 19 assumption inputs.

Old:
```rust
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two),
    // every one of them with explained results downstream now that every chain is explained.
```
```rust
    assert_eq!(paths.len(), 15);
```

New:
```rust
    // Spec A3 testing: the nineteen assumption inputs (the eighteen rows, BL-064's four among
    // them; end effect is two), every one of them with explained results downstream now that
    // every chain is explained.
```
```rust
    assert_eq!(paths.len(), 19);
```

(c) `magcoupling-rs/tests/assumptions.rs`. Four edits.

(c1) Two constants, before `every_flagged_input_is_in_exactly_one_row` (around :60).

Old:
```rust
#[test]
fn every_flagged_input_is_in_exactly_one_row() {
```

New:
```rust
/// The words every BL-064 shaft joint assumption's source carries, citing the spec by name.
const BL064_SOURCE: &str = "BL-064 shaft joint spec";

/// What the source of an assumption on a Rust-only input (no workbook cell) cites in place of
/// a cell: the spec that added the input.
const RUST_ONLY_SOURCES: [&str; 2] = ["Addendum A3", BL064_SOURCE];

/// The BL-064 spec's path from the repository root, which the friction row's source names.
const BL064_SPEC: &str = "docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md";

#[test]
fn every_flagged_input_is_in_exactly_one_row() {
```

(c2) `the_rows_are_the_spec_v1_set` (around :78): the four new ids, after the spec's fourteen.

Old:
```rust
    // duration, clamp friction coefficient, preload fraction of proof load.
    let ids: Vec<&str> = ASSUMPTIONS.iter().map(|a| a.id).collect();
```
```rust
            "clamp_friction",
            "preload_fraction",
        ]
    );
```

New:
```rust
    // duration, clamp friction coefficient, preload fraction of proof load; then BL-064's four
    // (spec 4.5): the shrink-fit friction, the surface smoothing, the rotor hotter than its
    // shaft and the joining clearance.
    let ids: Vec<&str> = ASSUMPTIONS.iter().map(|a| a.id).collect();
```
```rust
            "clamp_friction",
            "preload_fraction",
            "fit_friction",
            "fit_smoothing",
            "rotor_over_shaft",
            "joining_clearance",
        ]
    );
```

(c3) `every_row_has_a_label_rationale_source_and_one_unit` (around :127): the widened source rule (spec 4.5,
review 2 MF-3).

Old:
```rust
            // The source cites the workbook cell of every workbook input it sets; a Rust-only
            // input (the harmonic set) has none and cites the spec.
            match m.cell {
                Some(cell) => assert!(a.source.contains(cell), "{}: source omits {cell}", a.id),
                None => assert!(a.source.contains("Addendum A3"), "{}", a.id),
            }
```

New:
```rust
            // The source cites the workbook cell of every workbook input it sets; a Rust-only
            // input has none and cites its spec instead: Addendum A3 (the harmonic set) or the
            // BL-064 shaft joint spec (the shaft joints' four; review 2, MF-3).
            match m.cell {
                Some(cell) => assert!(a.source.contains(cell), "{}: source omits {cell}", a.id),
                None => assert!(
                    RUST_ONLY_SOURCES.iter().any(|s| a.source.contains(s)),
                    "{}: a Rust-only input's source cites none of {RUST_ONLY_SOURCES:?}",
                    a.id
                ),
            }
```

(c4) A new test, before `the_workbook_defaults_are_the_shipped_defaults` (around :138). The G-3 review does not
cover sources (spec 6.3), so the words each source must keep are pinned here; the cited file must exist.

Old:
```rust
#[test]
fn the_workbook_defaults_are_the_shipped_defaults() {
```

New:
```rust
#[test]
fn the_shaft_joint_rows_are_rust_only_and_cite_the_bl064_spec() {
    // BL-064 (spec 4.5, review 2 MF-3): one row per shaft joint assumption, each on one
    // Rust-only input with no workbook cell, each source citing the spec by name (the G-3
    // review does not cover sources, so the words are pinned here); the friction row also names
    // the spec's file, which exists. None is hidden while the shaft-clamp section is paused.
    let row = |id: &str| {
        ASSUMPTIONS
            .iter()
            .find(|a| a.id == id)
            .unwrap_or_else(|| panic!("{id}: no assumption"))
    };
    for (id, path) in [
        ("fit_friction", "shaft_joint.friction"),
        ("fit_smoothing", "shaft_joint.smoothing_um"),
        ("rotor_over_shaft", "shaft_joint.rotor_over_shaft_C"),
        ("joining_clearance", "shaft_joint.joining_clearance_um"),
    ] {
        let a = row(id);
        assert_eq!(a.paths, [path], "{id}");
        let m = meta(path);
        assert!(m.rust_only && m.cell.is_none(), "{path}");
        assert!(a.source.contains(BL064_SOURCE), "{id}: {}", a.source);
        assert!(a.shown(true) && a.shown(false), "{id}");
    }
    assert!(
        row("fit_friction").source.contains(BL064_SPEC),
        "{}",
        row("fit_friction").source
    );
    let spec = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(BL064_SPEC);
    assert!(spec.is_file(), "{} is not a file", spec.display());
}

#[test]
fn the_workbook_defaults_are_the_shipped_defaults() {
```

(d) `magcoupling-rs/src/gui/result_groups.rs`, `the_headline_comes_first_then_the_chains_then_the_other_results`
(around :252-357): the chain group "shaft_joint" between geometry and mass (spec 5.4), Task 1's inner onset in the
demagnetization chain, and where the shaft joint rows sit. A chain and a package share the id `shaft_joint`, so the new
assertions read the group's `other` flag too.

Old:
```rust
        assert_eq!(
            ids[..9],
            [
                "headline",
                "torque",
                "temperature",
                "demagnetization",
                "slip_heating",
                "adhesive",
                "clamps",
                "geometry",
                "mass"
            ]
        );
        assert!(groups[..9].iter().all(|g| !g.other));
        assert!(groups[9..].iter().all(|g| g.other));
```

New:
```rust
        assert_eq!(
            ids[..10],
            [
                "headline",
                "torque",
                "temperature",
                "demagnetization",
                "slip_heating",
                "adhesive",
                "clamps",
                "geometry",
                "shaft_joint",
                "mass"
            ]
        );
        assert!(groups[..10].iter().all(|g| !g.other));
        assert!(groups[10..].iter().all(|g| g.other));
```

Old:
```rust
        assert_eq!(
            group_of("temperature.demag.inner_magnet_limit_C"),
            "demagnetization"
        );
```

New:
```rust
        assert_eq!(
            group_of("temperature.demag.inner_magnet_limit_C"),
            "demagnetization"
        );
        // BL-064 (owner R-15 ring): the inner ring's own single-ring onset is a path of the
        // demagnetization chain.
        assert_eq!(
            group_of("temperature.demag.inner_onset_single_ring_C"),
            "demagnetization"
        );
        // BL-064: the shaft joints' numeric results are the shaft_joint chain, between the
        // geometry callouts and the mass. A chain and a package share the id, so `section` reads
        // the group's `other` flag too.
        let section = |path: &str| {
            groups
                .iter()
                .find(|g| g.rows.iter().any(|&i| entries[i].path == path))
                .map(|g| (g.id, g.other))
                .unwrap_or_else(|| panic!("{path} in no group"))
        };
        for path in [
            "shaft_joint.required_Nm",
            "shaft_joint.hub_holding_Nm",
            "shaft_joint.cup_fit_allowed_C",
        ] {
            assert_eq!(section(path), ("shaft_joint", false), "{path}");
        }
        // The verdict is no chain path: until it is a dashboard row it sits in the package
        // group under the other results.
        assert_eq!(section("shaft_joint.check"), ("shaft_joint", true));
```

(e) `magcoupling-rs/src/gui/trace.rs`: 23 more results have an equation record (1,112 results, Task 2's total, stay).

Old (module doc, line 7):
```rust
//! the dependencies of the results it explains (the A-3 scope: 393 of the 1112 results), so a
```
New:
```rust
//! the dependencies of the results it explains (the A-3 scope: 416 of the 1112 results), so a
```

Old (`an_input_no_explained_result_reads_traces_none_and_says_so`, line 194):
```rust
        assert_eq!((explained, table_entries().len()), (393, 1112));
```
New:
```rust
        assert_eq!((explained, table_entries().len()), (416, 1112));
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib tables`

Expected:
```text
test engine::explain::tables::tests::the_back_iron_table_reads_each_pick_s_modulus_expansion_and_yield ... FAILED
back_iron 1 modulus_GPa: Err("table back_iron has no field modulus_GPa")
test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 227 filtered out
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --no-fail-fast --test assumptions --test explain`

Expected: `assumptions` gives `test result: FAILED. 10 passed; 2 failed`:
```text
the_shaft_joint_rows_are_rust_only_and_cite_the_bl064_spec: fit_friction: no assumption
the_rows_are_the_spec_v1_set: left: [... "clamp_friction", "preload_fraction"] right: [... "preload_fraction", "fit_friction", "fit_smoothing", "rotor_over_shaft", "joining_clearance"]
```
and `explain` gives `test result: FAILED. 55 passed; 5 failed; 1 ignored`:
```text
the_allowed_hub_temperature_record_is_the_helper_s_lowest_of_three: a record
the_shaft_joint_chain_is_every_numeric_shaft_joint_result_in_schema_order: a shaft_joint chain
each_assumption_moves_what_depends_on_it_and_nothing_else: left: 15 right: 19
the_shaft_shrink_records_follow_the_engine_over_their_sliders: a record
explained_chains_have_every_record_and_scope_paths_exist: left: 170 right: 193
```
`every_record_reproduces_the_engine_everywhere` already passes here: the probe designs, the 12 edge points (each
`joint_edge` predicate holds at Task 2's engine) and the new augmentation hold for every existing record; the shaft
joint records that need them come in Step 4.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- result_groups trace::`

Expected: `test result: FAILED. 9 passed; 2 failed`:
```text
gui::result_groups::tests::the_headline_comes_first_then_the_chains_then_the_other_results
  left: ["headline", "torque", "temperature", "demagnetization", "slip_heating", "adhesive", "clamps", "geometry", "mass", "calibration"]
 right: ["headline", "torque", "temperature", "demagnetization", "slip_heating", "adhesive", "clamps", "geometry", "shaft_joint", "mass"]
gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so
  left: (393, 1112)
 right: (416, 1112)
```

- [ ] **Step 3: The back-iron table's modulus, expansion and yield (`tables.rs`)**

`magcoupling-rs/src/engine/explain/tables.rs`: the module doc, three `TableField` rows after `design_flux_density_T`
(around :53), and their lookups (around :134). The yield reads NaN where the library has none, as
`PartProperties::hub` does (every back-iron choice has one; the test of Step 1 says so).

Old:
```rust
//! magnet library, the grade table, the part materials (back iron, sleeve and liner, cap and
//! housing), the aluminium alloys, the adhesives and the clamp screw sizes, each field as the
//! engine uses it with every approved correction on (the explorer describes what users see).
```

New:
```rust
//! magnet library, the grade table, the part materials (back iron, sleeve and liner, cap and
//! housing; the back iron's modulus, expansion and yield for the shaft joints), the aluminium
//! alloys, the adhesives and the clamp screw sizes, each field as the engine uses it with every
//! approved correction on (the explorer describes what users see).
```

Old:
```rust
    TableField { table: "back_iron", field: "design_flux_density_T", symbol: "B_{des,BI}", unit: "T", label: "Back-iron material design flux density (decision 20; none where the workbook gives none)" },
```

New:
```rust
    TableField { table: "back_iron", field: "design_flux_density_T", symbol: "B_{des,BI}", unit: "T", label: "Back-iron material design flux density (decision 20; none where the workbook gives none)" },
    TableField { table: "back_iron", field: "modulus_GPa", symbol: "E_{BI}", unit: "GPa", label: "Back-iron material elastic modulus (library)" },
    TableField { table: "back_iron", field: "cte_per_C", symbol: "α_{BI}", unit: "1/°C", label: "Back-iron material expansion coefficient (library)" },
    TableField { table: "back_iron", field: "yield_MPa", symbol: "σ_{y,BI}", unit: "MPa", label: "Back-iron material yield strength (library: the published minimum where one exists)" },
```

Old:
```rust
                "design_flux_density_T" => m.design_flux_density_T.map_or(Value::None, num),
```

New:
```rust
                "design_flux_density_T" => m.design_flux_density_T.map_or(Value::None, num),
                "modulus_GPa" => num(m.engine.modulus_GPa),
                "cte_per_C" => num(m.engine.cte_per_C),
                // As `PartProperties::hub` reads it: NaN where the library has no yield.
                "yield_MPa" => num(m.yield_MPa.value.unwrap_or(f64::NAN)),
```

The existing `every_field_reads_and_a_missing_key_is_none` covers the three new fields (key 1 reads, key 99 is `none`)
with no edit.

- [ ] **Step 4: Create the records file**

Create `magcoupling-rs/src/engine/explain/records/shaft_joint.rs` exactly as below. Notes on the transcription, each
checked in the prototype (the drift guard reproduces every record at every point, 1e-9 relative):
- **The no-wall arm.** The markup's conditions have no parentheses; `and` binds tighter than `or`, so
  `[D] <= d or t > 0 and w <= t` is the engine's `!(D > d) || (keyed && wall <= keyway)`. Ruling C-3 (plan deviation
  PD-1): the hub's arm reads `[D] <= {coupling.bore_mm}` and the cup's `{metal.boss_od_mm} <= {coupling.bore_mm}`, so
  a part diameter typed at or below zero has no wall; for positive diameters it is the old `[Q] >= 1`.
- **Floors and ends.** `max(0, …)` and `min(…)`, `max(…)` fold left as `py_max`/`py_min` do, in the engine's argument
  order, so a tie keeps the engine's winner and a floored term reads as a condition term, as the traceability test
  expects.
- **The key.** Its three arms are the engine's `keyed` (`keyway > 0`), then bearing length and contact both above 0,
  else +inf; no keyway gives 0.
- **The shrink.** It is the engine's own form, the NIST polynomial with its coefficients signed as `NIST_FE_2_25NI`
  (`-210.4 - 0.05699 · T` is the engine's `a0 + a1·T` bit for bit), anchored at 265 K; the polynomial's two locals are
  evaluated only on the arm that reads them.
- **The allowed hub temperature (plan deviation PD-11).** Four arms, each value at most both others, else `nan`.
- **The literals.** The engine's named constants are numbers here, each named in the module doc; the drift guard holds
  each one to the engine, so a change to a constant fails the guard until the record follows.

```rust
//! Shaft joints (BL-064): the keyed shrink fits that hold the inner hub on the wheel shaft and
//! the cup on the gearbox output shaft. One record per numeric result of the check (23); the
//! verdict `shaft_joint.check` is a text and has none.
//!
//! Each formula is transcribed from `shaft_joint::compute` (its `joint_specs`, `joint`,
//! `hub_fit_allowed` and `shaft_shrink_um`) and the part material from
//! `material_library::PartProperties::hub`, every approved correction on. Plane-stress Lamé
//! thick cylinders on a solid shaft: the contact pressure per µm of effective diametral
//! interference, the holding torque with the key's arc taken off the contact, the bore's von
//! Mises stress; the interference's change at the two temperature ends; the fitting
//! temperature with the shaft's shrink; the key alone. A joint with no wall around its bore
//! (the part's outer diameter at most the bore, a part typed at or below zero included, or a
//! keyway as deep as the wall) has NaN Lamé rows.
//!
//! The engine's named constants appear as numbers (the drift guard holds each to the engine):
//! `SHAFT_MODULUS_GPA` (205 GPa, written 205000 MPa), `POISSON` 0.3, `SHAFT_CTE_PER_C`
//! 11.5e-6 /°C, `ROOM_C` 20 °C, `YIELD_FACTOR` 1.2, `NIST_FE_2_25NI` (the five coefficients)
//! and `NIST_FIT_TOP_K` 265 K, with 0 °C at 273.15 K, and `temperature::CURE_MARGIN_MIN_C`,
//! 10 °C. The workbook's aluminium body (no back iron and no non-magnetic pick) is 6061-T6:
//! E18's 68.9 GPa and 23.6e-6 /°C (`temperature::AL_HUB_MODULUS_GPA`, `AL_HUB_CTE_PER_C`) and
//! the library's 241 MPa yield.
//!
//! The two joints' records repeat each other with their own outer diameter, length, band,
//! allowed temperature and shaft, as the inner and outer magnet-limit records do: the inner
//! hub's outer diameter is its flats' incircle under the bondline, 2 (a_i − b_i), its length
//! the hub's in effect; the cup's outer diameter is its boss, its length the boss and the web
//! the bore runs through.
//!
//! Locals: [D] the part's outer diameter, [Q] bore over outer diameter, [k_θ] the bore's hoop
//! factor (1 + Q²)/(1 − Q²), [p_1] the contact pressure per µm, [b] the key's arc, [T_1] the
//! holding torque per µm, [σ_1] the bore stress per µm, [Δ_c] and [Δ_h] the interference's
//! change at the cold and the hot end, [δ_{lo}] and [δ_{hi}] the loosest and tightest of them,
//! [l] and [k] the key's bearing length and contact height.

use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- What both joints share (shaft_joint::joint_specs, PartProperties::hub) ---
    record("shaft_joint.required_Nm", "T_{fit,req}", "{shaft_joint.safety_factor} * {metal.torque_cold_high_Nm}"),
    // The part material in effect: on the steel circuit the steel's values (the Materials
    // inputs for the default pick, else the pick's), else the body's (a non-magnetic pick's,
    // else the workbook's 6061-T6); the yield is the library's for the metal in effect.
    record("shaft_joint.part_modulus_GPa", "E_{part}",
        r#"cases({materials.circuit_backiron} = 1 and {materials.parts.back_iron} = 1 => {materials.steel.modulus_GPa};
               {materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "modulus_GPa");
               else => 68.9)"#),
    record("shaft_joint.part_cte_per_C", "α_{part}",
        r#"cases({materials.circuit_backiron} = 1 and {materials.parts.back_iron} = 1 => {materials.steel.cte_per_C};
               {materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "cte_per_C");
               else => 23.6e-6)"#),
    record("shaft_joint.part_yield_MPa", "σ_{y,part}",
        r#"cases({materials.circuit_backiron} = 1 or table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0
                   => table("back_iron", {materials.parts.back_iron}, "yield_MPa");
               else => 241)"#),
    // YIELD_FACTOR: the part stays elastic with a margin.
    record("shaft_joint.stress_limit_MPa", "σ_{lim}", "frac({shaft_joint.part_yield_MPa}, 1.2)"),

    // --- The inner hub on the wheel shaft (shaft_joint::joint) ---
    // The fit's holding torque at its loosest: friction × contact pressure × contact arc ×
    // length × radius per µm, times the effective minimum (the measured minimum less smoothing,
    // at the end that loosens it more, floored at 0).
    record("shaft_joint.hub_holding_Nm", "T_{hold,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [T_1] · max(0, {shaft_joint.hub_interference_min_um} - {shaft_joint.smoothing_um} + [δ_{lo}]))
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · {housing.hub_length_mm} · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    // The smallest measured minimum that holds the required torque.
    record("shaft_joint.hub_needed_um", "δ_{need,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.required_Nm}, [T_1]) + {shaft_joint.smoothing_um} - [δ_{lo}])
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · {housing.hub_length_mm} · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    // The bore's von Mises stress at the tightest fit: the stress per µm, √(k_θ² + k_θ + 1) p_1,
    // times the effective maximum (the measured maximum less smoothing, at the end that
    // tightens it more, floored at 0).
    record("shaft_joint.hub_bore_stress_MPa", "σ_{vM,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [σ_1] · max(0, {shaft_joint.hub_interference_max_um} - {shaft_joint.smoothing_um} + [δ_{hi}]))
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    // The largest measured maximum within the stress limit.
    record("shaft_joint.hub_elastic_max_um", "δ_{el,hub}",
        r#"cases([D] <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.stress_limit_MPa}, [σ_1]) + {shaft_joint.smoothing_um} - [δ_{hi}])
           where [D] = 2 · ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm}),
                 [Q] = frac({coupling.bore_mm}, [D]),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    // The key alone at the required torque (form A: bearing length L − b), its contact capped
    // at the keyway depth: 0 with no keyway, +inf with no bearing length or contact.
    record("shaft_joint.hub_key_pressure_MPa", "p_{key,hub}",
        r#"cases({coupling.keyway_depth_mm} > 0 and [l] > 0 and [k] > 0 => frac(2 · {shaft_joint.required_Nm} · 1000, {coupling.bore_mm} · [k] · [l]);
               {coupling.keyway_depth_mm} > 0 => inf;
               else => 0)
           where [l] = {housing.hub_length_mm} - {shaft_joint.key_width_mm},
                 [k] = min({shaft_joint.key_contact_mm}, {coupling.keyway_depth_mm})"#),
    // The wheel shaft's diametral shrink from 20 °C: the linear expansion from 265 K up; below
    // it NIST's Fe-2.25Ni fit, y = (L − L293)/L293 × 10⁵, anchored at 265 K.
    record("shaft_joint.wheel_shaft_shrink_um", "Δd_{wheel}",
        "cases([ϑ_K] >= 265 => 11.5e-6 · (20 - {shaft_joint.wheel_shaft_fit_C}) · {coupling.bore_mm} · 1000;
               else => (11.5e-6 · (20 - (265 - 273.15)) + ([y_{265}] - [y]) · 1e-5) · {coupling.bore_mm} · 1000)
           where [ϑ_K] = {shaft_joint.wheel_shaft_fit_C} + 273.15,
                 [y_{265}] = -210.4 - 0.05699 · 265 + 5.072e-3 · 265^2 - 1.381e-5 · 265^3 + 1.897e-8 · 265^4,
                 [y] = -210.4 - 0.05699 · [ϑ_K] + 5.072e-3 · [ϑ_K]^2 - 1.381e-5 · [ϑ_K]^3 + 1.897e-8 · [ϑ_K]^4"),
    // What the allowed temperature and the shaft's shrink open, less the joining clearance.
    record("shaft_joint.hub_fittable_max_um", "δ_{fit,hub}",
        "{shaft_joint.part_cte_per_C} · ({shaft_joint.hub_fit_allowed_C} - 20) · {coupling.bore_mm} · 1000
         + {shaft_joint.wheel_shaft_shrink_um} - {shaft_joint.joining_clearance_um}"),
    // The part's temperature the fit would need on its shaft as set (the measured maximum and
    // the joining clearance, less the shaft's shrink).
    record("shaft_joint.hub_fit_temp_C", "ϑ_{need,hub}",
        "20 + frac({shaft_joint.hub_interference_max_um} + {shaft_joint.joining_clearance_um} - {shaft_joint.wheel_shaft_shrink_um},
                   {shaft_joint.part_cte_per_C} · {coupling.bore_mm} · 1000)"),
    // Owner R-15: the lowest of your limit for the bonded inner hub, the selected adhesive's
    // design limit and the inner ring's own single-ring onset less the cure margin
    // (CURE_MARGIN_MIN_C, 10 °C), ties to the earlier, as `hub_fit_allowed` names them. Each arm
    // compares its value with both others, so a NaN in any of the three fails every arm: not a
    // number (the markup's min would carry a NaN only in its first argument).
    record("shaft_joint.hub_fit_allowed_C", "ϑ_{allow,hub}",
        "cases({shaft_joint.hub_fit_max_C} <= {temperature.adhesive.design_limit_C} and {shaft_joint.hub_fit_max_C} <= [ϑ_{cap}]
                   => {shaft_joint.hub_fit_max_C};
               {temperature.adhesive.design_limit_C} <= {shaft_joint.hub_fit_max_C} and {temperature.adhesive.design_limit_C} <= [ϑ_{cap}]
                   => {temperature.adhesive.design_limit_C};
               [ϑ_{cap}] <= {shaft_joint.hub_fit_max_C} and [ϑ_{cap}] <= {temperature.adhesive.design_limit_C} => [ϑ_{cap}];
               else => nan)
           where [ϑ_{cap}] = {temperature.demag.inner_onset_single_ring_C} - 10"),

    // --- The cup on the gearbox output shaft (shaft_joint::joint): its boss is the outer
    // diameter, its boss and web the length; otherwise as the inner hub's ---
    record("shaft_joint.cup_holding_Nm", "T_{hold,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [T_1] · max(0, {shaft_joint.cup_interference_min_um} - {shaft_joint.smoothing_um} + [δ_{lo}]))
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · ({metal.boss_length_mm} + {metal.web_mm}) · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    record("shaft_joint.cup_needed_um", "δ_{need,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.required_Nm}, [T_1]) + {shaft_joint.smoothing_um} - [δ_{lo}])
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [b] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0),
                 [T_1] = {shaft_joint.friction} · [p_1] · max(0, π · {coupling.bore_mm} - [b]) · ({metal.boss_length_mm} + {metal.web_mm}) · frac({coupling.bore_mm}, 2000),
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{lo}] = min([Δ_c], [Δ_h] - {shaft_joint.part_cte_per_C} · {shaft_joint.rotor_over_shaft_C} · {coupling.bore_mm} · 1000)"#),
    record("shaft_joint.cup_bore_stress_MPa", "σ_{vM,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => [σ_1] · max(0, {shaft_joint.cup_interference_max_um} - {shaft_joint.smoothing_um} + [δ_{hi}]))
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    record("shaft_joint.cup_elastic_max_um", "δ_{el,cup}",
        r#"cases({metal.boss_od_mm} <= {coupling.bore_mm} or {coupling.keyway_depth_mm} > 0 and frac({metal.boss_od_mm} - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
               else => frac({shaft_joint.stress_limit_MPa}, [σ_1]) + {shaft_joint.smoothing_um} - [δ_{hi}])
           where [Q] = frac({coupling.bore_mm}, {metal.boss_od_mm}),
                 [k_θ] = frac(1 + [Q]^2, 1 - [Q]^2),
                 [p_1] = frac(0.001, {coupling.bore_mm} · (frac([k_θ] + 0.3, 1000 · {shaft_joint.part_modulus_GPa}) + frac(1 - 0.3, 205000))),
                 [σ_1] = sqrt([k_θ]^2 + [k_θ] + 1) · [p_1],
                 [Δ_c] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({metal.min_temp_C} - 20) · {coupling.bore_mm} · 1000,
                 [Δ_h] = (11.5e-6 - {shaft_joint.part_cte_per_C}) · ({temperature.adhesive.design_limit_C} - 20) · {coupling.bore_mm} · 1000,
                 [δ_{hi}] = max([Δ_c], [Δ_h])"#),
    record("shaft_joint.cup_key_pressure_MPa", "p_{key,cup}",
        r#"cases({coupling.keyway_depth_mm} > 0 and [l] > 0 and [k] > 0 => frac(2 · {shaft_joint.required_Nm} · 1000, {coupling.bore_mm} · [k] · [l]);
               {coupling.keyway_depth_mm} > 0 => inf;
               else => 0)
           where [l] = {metal.boss_length_mm} + {metal.web_mm} - {shaft_joint.key_width_mm},
                 [k] = min({shaft_joint.key_contact_mm}, {coupling.keyway_depth_mm})"#),
    record("shaft_joint.output_shaft_shrink_um", "Δd_{gbx}",
        "cases([ϑ_K] >= 265 => 11.5e-6 · (20 - {shaft_joint.output_shaft_fit_C}) · {coupling.bore_mm} · 1000;
               else => (11.5e-6 · (20 - (265 - 273.15)) + ([y_{265}] - [y]) · 1e-5) · {coupling.bore_mm} · 1000)
           where [ϑ_K] = {shaft_joint.output_shaft_fit_C} + 273.15,
                 [y_{265}] = -210.4 - 0.05699 · 265 + 5.072e-3 · 265^2 - 1.381e-5 · 265^3 + 1.897e-8 · 265^4,
                 [y] = -210.4 - 0.05699 · [ϑ_K] + 5.072e-3 · [ϑ_K]^2 - 1.381e-5 · [ϑ_K]^3 + 1.897e-8 · [ϑ_K]^4"),
    record("shaft_joint.cup_fittable_max_um", "δ_{fit,cup}",
        "{shaft_joint.part_cte_per_C} · ({shaft_joint.cup_fit_allowed_C} - 20) · {coupling.bore_mm} · 1000
         + {shaft_joint.output_shaft_shrink_um} - {shaft_joint.joining_clearance_um}"),
    record("shaft_joint.cup_fit_temp_C", "ϑ_{need,cup}",
        "20 + frac({shaft_joint.cup_interference_max_um} + {shaft_joint.joining_clearance_um} - {shaft_joint.output_shaft_shrink_um},
                   {shaft_joint.part_cte_per_C} · {coupling.bore_mm} · 1000)"),
    // The bare cup is fitted before its magnets are bonded: your limit alone.
    record("shaft_joint.cup_fit_allowed_C", "ϑ_{allow,cup}", "{shaft_joint.cup_fit_max_C}"),
];
```

As the panel typesets it (the ignored `review_sheet` tool, `cargo test --manifest-path magcoupling-rs/Cargo.toml
--test explain review_sheet -- --ignored --nocapture`, prints every record), for example:
```text
ϑ_{allow,hub} = { ϑ_{lim,hub} if ϑ_{lim,hub} ≤ ϑ_{adh} and ϑ_{lim,hub} ≤ ϑ_{cap}; ϑ_{adh} if ϑ_{adh} ≤ ϑ_{lim,hub} and ϑ_{adh} ≤ ϑ_{cap}; ϑ_{cap} if ϑ_{cap} ≤ ϑ_{lim,hub} and ϑ_{cap} ≤ ϑ_{adh}; undefined otherwise }, where ϑ_{cap} = ϑ_{sr,i} − 10
p_{key,hub} = { (2 · T_{fit,req} · 1000)/(d_{bore} · k · l) if t_{kw} > 0 and l > 0 and k > 0; ∞ if t_{kw} > 0; 0 otherwise }, where l = L_{hub} − b_{key}, k = min(h_{key}, t_{kw})
σ_{y,part} = { σ_{y,BI}(material_{BI}) if circuit = 1 or ferro(material_{BI}) = 0; 241 otherwise }
```

- [ ] **Step 5: Register the records, give the inputs their symbols, add the scope chain**

`magcoupling-rs/src/engine/explain/records/mod.rs` (alphabetical module list; the batch last in `RECORDS`):

Old:
```rust
pub mod geometry;
pub mod slip_heating;
```
New:
```rust
pub mod geometry;
pub mod shaft_joint;
pub mod slip_heating;
```

Old:
```rust
    geometry::RECORDS,
];
```
New:
```rust
    geometry::RECORDS,
    shaft_joint::RECORDS,
];
```

`magcoupling-rs/src/engine/explain/symbols.rs`: the conventions in the module doc, then the entries (plan deviation PD-10: the 15
inputs and the three other inputs the records read first). The symbols are unique among all paths (the registry
refuses two paths with one symbol; checked).

Old (module doc, :10-14):
```rust
//! (`c_{end}`) or a radial clearance (`c_{nom}`, `c_{res}`), told apart by subscript and unit; δ
//! the skin depth (`δ`) or a radial movement (`δ_{shaft}`, `δ_{adv}`); r and R a radius
//! (`r_{mid}`, `R_g`), apart from the Calibration's measured-to-model ratio `r^{cal}`; Δ before a
//! quantity's letter a difference: a rise (`Δϑ_{ss,est}`), or a margin or reserve, the limit or
//! claim minus the value (`Δϑ_{mag}`; the space claim's `ΔD`, `ΔL`, `ΔL_{bay}`, a diameter
```

New:
```rust
//! (`c_{end}`) or a radial clearance (`c_{nom}`, `c_{res}`), told apart by subscript and unit; δ
//! the skin depth (`δ`), a radial movement (`δ_{shaft}`, `δ_{adv}`) or a shaft joint's diametral
//! interference in µm (`δ_{min,hub}`, `δ_{need,cup}`, `δ_{sm}`); r and R a radius
//! (`r_{mid}`, `R_g`), apart from the Calibration's measured-to-model ratio `r^{cal}`; Δ before a
//! quantity's letter a difference: a rise (`Δϑ_{ss,est}`, `Δϑ_{rot}`), a shaft's diametral
//! shrink (`Δd_{wheel}`), or a margin or reserve, the limit or
//! claim minus the value (`Δϑ_{mag}`; the space claim's `ΔD`, `ΔL`, `ΔL_{bay}`, a diameter
```

Old (module doc, :27-29):
```rust
//! Calculator quantity carries `MD` (`ϑ_{op,MD}`; `f_{e,MD}` is the field frequency p n / 60
//! the Temperature design calls `f_e`). Renaming any of these is M4's typesetting pass: a
//! symbol an A4 note quotes changes with the note, and a changed note goes back to Draft.
```

New:
```rust
//! Calculator quantity carries `MD` (`ϑ_{op,MD}`; `f_{e,MD}` is the field frequency p n / 60
//! the Temperature design calls `f_e`). The shaft joints (BL-064) name their two joints `hub` (the
//! inner hub) and `cup`, and their shafts `wheel` and `gbx` (the gearbox output); there E is an
//! elastic modulus (`E_{part}`; `E_{slip,MD}` is an energy), α an expansion coefficient
//! (`α_{part}`; elsewhere α is a Br temperature coefficient), σ a part's stress, yield or limit
//! (`σ_{vM,hub}`, `σ_{y,part}`, `σ_{lim}`), p a pressure (`p_{key,hub}`; bare `p` is the pole-pair
//! count), b and h the key's width and contact height (`b_{key}`, `h_{key}`; `b_i` is a bondline)
//! and s the joining clearance (`s_{join}`). Renaming any of these is M4's typesetting pass: a
//! symbol an A4 note quotes changes with the note, and a changed note goes back to Draft.
```

The letters the shaft joints reuse are named with what they mean elsewhere (`E_{slip,MD}` an energy, α a Br
temperature coefficient, bare `p` the pole-pair count), as the paragraph above does for C, t, D, m and A.

Old (:45):
```rust
    ("coupling.bore_mm", "d_{bore}"),
```
New:
```rust
    ("coupling.bore_mm", "d_{bore}"),
    ("coupling.keyway_depth_mm", "t_{kw}"),
```

Old (:126):
```rust
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
```
New:
```rust
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
    ("materials.steel.cte_per_C", "α_{st,in}"),
    ("materials.steel.modulus_GPa", "E_{st,in}"),
```

Old (:184-185):
```rust
    ("clamps.joint_friction", "μ_{joint}"),
];
```
New:
```rust
    ("clamps.joint_friction", "μ_{joint}"),
    // shaft_joint (BL-064)
    ("shaft_joint.hub_interference_min_um", "δ_{min,hub}"),
    ("shaft_joint.hub_interference_max_um", "δ_{max,hub}"),
    ("shaft_joint.cup_interference_min_um", "δ_{min,cup}"),
    ("shaft_joint.cup_interference_max_um", "δ_{max,cup}"),
    ("shaft_joint.safety_factor", "S_{fit}"),
    ("shaft_joint.friction", "μ_{fit}"),
    ("shaft_joint.smoothing_um", "δ_{sm}"),
    ("shaft_joint.rotor_over_shaft_C", "Δϑ_{rot}"),
    ("shaft_joint.key_width_mm", "b_{key}"),
    ("shaft_joint.key_contact_mm", "h_{key}"),
    ("shaft_joint.hub_fit_max_C", "ϑ_{lim,hub}"),
    ("shaft_joint.cup_fit_max_C", "ϑ_{lim,cup}"),
    ("shaft_joint.wheel_shaft_fit_C", "ϑ_{wheel}"),
    ("shaft_joint.output_shaft_fit_C", "ϑ_{gbx}"),
    ("shaft_joint.joining_clearance_um", "s_{join}"),
];
```

`magcoupling-rs/src/engine/explain/scope.rs`: the module doc's count (Task 1 wrote 170) and the chain, after
`geometry`, before `dashboard` (spec 4.5), its paths in schema order (the order of `ShaftJointResults`).

Old:
```rust
//! scope held 169 paths. BL-064 adds the inner ring's own single-ring onset to the
//! demagnetization chain (owner R-15 ring): 170 paths.
```
New:
```rust
//! scope held 169 paths. BL-064 adds the inner ring's own single-ring onset to the
//! demagnetization chain (owner R-15 ring) and a chain of the shaft joints, the 23 numeric
//! results of their check (spec 4.5), after the geometry callouts: 193 paths.
```

Old:
```rust
            "housing.bay_overshoot_mm",
        ],
    },
    Chain {
        id: "dashboard",
```
New:
```rust
            "housing.bay_overshoot_mm",
        ],
    },
    Chain {
        id: "shaft_joint",
        status: Status::Explained,
        paths: &[
            "shaft_joint.required_Nm",
            "shaft_joint.part_modulus_GPa",
            "shaft_joint.part_cte_per_C",
            "shaft_joint.part_yield_MPa",
            "shaft_joint.stress_limit_MPa",
            "shaft_joint.hub_holding_Nm",
            "shaft_joint.cup_holding_Nm",
            "shaft_joint.hub_bore_stress_MPa",
            "shaft_joint.cup_bore_stress_MPa",
            "shaft_joint.hub_key_pressure_MPa",
            "shaft_joint.cup_key_pressure_MPa",
            "shaft_joint.hub_needed_um",
            "shaft_joint.cup_needed_um",
            "shaft_joint.hub_elastic_max_um",
            "shaft_joint.cup_elastic_max_um",
            "shaft_joint.wheel_shaft_shrink_um",
            "shaft_joint.output_shaft_shrink_um",
            "shaft_joint.hub_fittable_max_um",
            "shaft_joint.cup_fittable_max_um",
            "shaft_joint.hub_fit_temp_C",
            "shaft_joint.cup_fit_temp_C",
            "shaft_joint.hub_fit_allowed_C",
            "shaft_joint.cup_fit_allowed_C",
        ],
    },
    Chain {
        id: "dashboard",
```

- [ ] **Step 6: The chain's results-table heading (`result_groups.rs`)**

`magcoupling-rs/src/gui/result_groups.rs`: `CHAIN_LABELS` 8 -> 9 (spec 5.4: between geometry and mass, so the scope's
chains keep the scope's order and the mass stays last), and the module doc's chain list.
`every_chain_path_names_results_and_every_package_has_a_heading` holds every scope chain but `dashboard` to a
`CHAIN_LABELS` entry in the scope's order, so the scope chain of Step 5 needs it. No `CHAIN_PREFIXES` entry: every
numeric shaft joint result is a chain path. `PACKAGE_LABELS` already has `("shaft_joint", "Shaft joints")` (Task 2).

Old:
```rust
//! [`DASHBOARD`]), then the physics chains: the A-3 chains of the explorer's scope ([`SCOPE`]:
//! torque, temperature, demagnetization, slip heating, clamps, geometry) and two of results the
```
New:
```rust
//! [`DASHBOARD`]), then the physics chains: the A-3 chains of the explorer's scope ([`SCOPE`]:
//! torque, temperature, demagnetization, slip heating, clamps, geometry, and BL-064's shaft
//! joints) and two of results the
```

Old:
```rust
pub const CHAIN_LABELS: [(&str, &str); 8] = [
```
```rust
    ("geometry", "Geometry"),
    ("mass", "Mass"),
];
```
New:
```rust
pub const CHAIN_LABELS: [(&str, &str); 9] = [
```
```rust
    ("geometry", "Geometry"),
    ("shaft_joint", "Shaft joints"),
    ("mass", "Mass"),
];
```

In this task's tree the results table shows the 23 numeric results under the chain group "Shaft joints" (between
Geometry and Mass) and the verdict under the other results' package group "Shaft joints"; Task 5's `DASHBOARD` row
moves the verdict to the headline, which empties and drops that package group.

- [ ] **Step 7: The four assumptions and their flags**

`magcoupling-rs/src/engine/assumptions.rs`: the array 14 -> 18 rows, the spec's four last, with the spec 4.5 drafts
(the G-3 review of Task 8 sets the labels' and rationales' words; it does not cover sources, so each source keeps "BL-064
shaft joint spec", which Step 1's test pins).

Old (:58-59):
```rust
/// The spec's v1 set, in the spec's order.
pub const ASSUMPTIONS: [Assumption; 14] = [
```
New:
```rust
/// The spec's v1 set, in the spec's order, then the shaft joints' four (BL-064, spec 4.5).
pub const ASSUMPTIONS: [Assumption; 18] = [
```

Old (the last row's end, around :155):
```rust
        rationale: "Screw preload as a share of the proof load (bolted-joint practice), capped by thread stripping in the aluminium.",
        source: "Workbook Shaft clamps!C29 (75 %); spec M1 (clamps).",
    },
];
```
New:
```rust
        rationale: "Screw preload as a share of the proof load (bolted-joint practice), capped by thread stripping in the aluminium.",
        source: "Workbook Shaft clamps!C29 (75 %); spec M1 (clamps).",
    },
    // BL-064: the shaft joints' keyed shrink fits. Rust-only inputs: each source cites the
    // spec by name in place of a workbook cell (`tests/assumptions.rs`).
    Assumption {
        id: "fit_friction",
        label: "Shrink-fit friction coefficient",
        paths: &["shaft_joint.friction"],
        rationale: "The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. 0.14 is for dry, degreased steel on steel; oil, a coating or a finish in the bore lowers it, and the holding torque falls in proportion.",
        source: "BL-064 shaft joint spec (`docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`); dry steel on steel: Gear Technology, Nov/Dec 2017 (0.145) and ANSI/AGMA 9003 (0.15).",
    },
    Assumption {
        id: "fit_smoothing",
        label: "Surface smoothing of the fit",
        paths: &["shaft_joint.smoothing_um"],
        rationale: "Interference lost as the surface peaks of bore and shaft flatten when the fit is made, taken off both ends of the measured band. 5 µm is conservative for holding with ground and honed surfaces; a finer finish loses less, which raises the grip and the bore stress together.",
        source: "BL-064 shaft joint spec; DIN 7190-1:2017 (smoothing 0.4·Rz per surface).",
    },
    Assumption {
        id: "rotor_over_shaft",
        label: "Rotor hotter than its shaft, running",
        paths: &["shaft_joint.rotor_over_shaft_C"],
        rationale: "Slip heating warms the rotors from the gap, so each rotor can run warmer than the shaft it sits on. Its bore then grows more than the shaft, and the grip at the hot end falls. 20 °C is a placeholder until a measurement or a thermal model gives the difference.",
        source: "BL-064 shaft joint spec (the hot end of the loosest fit, section 3.3).",
    },
    Assumption {
        id: "joining_clearance",
        label: "Joining clearance",
        paths: &["shaft_joint.joining_clearance_um"],
        rationale: "The gap each part needs over its shaft, after warming and cooling, to slide to its stop before the gap closes; one value for both joints. Too little and the part can seize half-way. 10 µm is an assumed value for a 10 mm bore; less needs a guided, single fast push to a stop.",
        source: "BL-064 shaft joint spec; DIN 7190-1:2017 §8.2 (the joining clearance enters the joining temperature).",
    },
];
```

`magcoupling-rs/src/engine/shaft_joint.rs` (Task 2's file): `.assumption()` on the four inputs (spec 4.1, "Notes on the
inputs"; Task 2's plan deviation PD-5 left it here), after each `.range(...)`, as `model.rs` writes `coupling.c_end`.

Old (around :62-71):
```rust
            friction: f64 = 0.14 => param_rust_only("-", "Shrink-fit friction coefficient",
                "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it.")
                .range(0.05, 0.30, 0.01),
            smoothing_um: f64 = 5.0 => param_rust_only("µm", "Surface smoothing of the fit",
                "Interference lost as surface peaks flatten when fitted. It lowers both the grip and the bore stress; a ground shaft and honed bore lose less.")
                .range(0.0, 15.0, 0.1),
            rotor_over_shaft_C: f64 = 20.0 => param_rust_only("°C", "Rotor hotter than its shaft, running",
                "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls.")
                .range(0.0, 60.0, 1.0),
```
New:
```rust
            friction: f64 = 0.14 => param_rust_only("-", "Shrink-fit friction coefficient",
                "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it.")
                .range(0.05, 0.30, 0.01)
                .assumption(),
            smoothing_um: f64 = 5.0 => param_rust_only("µm", "Surface smoothing of the fit",
                "Interference lost as surface peaks flatten when fitted. It lowers both the grip and the bore stress; a ground shaft and honed bore lose less.")
                .range(0.0, 15.0, 0.1)
                .assumption(),
            rotor_over_shaft_C: f64 = 20.0 => param_rust_only("°C", "Rotor hotter than its shaft, running",
                "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls.")
                .range(0.0, 60.0, 1.0)
                .assumption(),
```

Old (around :89-91):
```rust
            joining_clearance_um: f64 = 10.0 => param_rust_only("µm", "Joining clearance",
                "Clearance each part needs over its shaft to slide on before it grips; one value for both joints. Less needs a guided, single fast push to a stop.")
                .range(0.0, 30.0, 0.5),
```
New:
```rust
            joining_clearance_um: f64 = 10.0 => param_rust_only("µm", "Joining clearance",
                "Clearance each part needs over its shaft to slide on before it grips; one value for both joints. Less needs a guided, single fast push to a stop.")
                .range(0.0, 30.0, 0.5)
                .assumption(),
```

On screen the four inputs' hovers end with `ASSUMPTION_LINE` ("Model assumption: see Assumptions for why it has this
value"), the Assumptions view lists 18 rows (16 while the shaft-clamp section is paused: the BL-064 four are never
hidden), the XLSX Inputs sheet flags them, and a modified one raises the "Assumptions modified" banner and its reset
restores it. None of these needs an edit (the app suite passes unchanged at 889, Step 11).

- [ ] **Step 8: Re-bless the input schema (plan deviation PD-9)**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema input_schema_json_is_current`

Expected: `FAILED`, "tests/data/input_schema.json is stale: run `MAGCOUPLING_BLESS=1 cargo test --test schema`, then
regenerate the differential data".

Run (Git Bash): `MAGCOUPLING_BLESS=1 cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`

Expected: `test result: ok. 11 passed`.

Run: `git diff --stat magcoupling-rs/tests/data/input_schema.json` and `git diff magcoupling-rs/tests/data/input_schema.json`

Expected: `1 file changed, 4 insertions(+), 4 deletions(-)`: four `"assumption": false,` lines become
`"assumption": true,`, in the entries of `shaft_joint.friction`, `shaft_joint.smoothing_um`,
`shaft_joint.rotor_over_shaft_C` and `shaft_joint.joining_clearance_um`; nothing else. The differential data needs no
regeneration: `gen_differential.py` reads the schema without its `rust_only` inputs, so its outputs are unchanged
(gate 12 at Task 10 confirms it).

- [ ] **Step 9: Run the tests and see them pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib tables`

Expected: `test result: ok. 5 passed; 0 failed`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test assumptions --test explain --test schema`

Expected: `assumptions` `ok. 12 passed`; `explain` `ok. 60 passed; 0 failed; 1 ignored` (15 to 25 s in a debug
build: the guard now evaluates 5 probe designs, 12 more edge points and one more augmentation, 3,391 more points);
`schema` `ok. 11 passed`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- result_groups trace::`

Expected: `test result: ok. 11 passed; 0 failed`.

No `CANCELLATIONS` entry is needed (spec 7.12 asked to confirm it before adding any). With the four new assumptions,
`each_assumption_moves_what_depends_on_it_and_nothing_else` passes as it is: friction, smoothing, the rotor allowance
and the joining clearance each move every record on their active path at the defaults, and a floored effective
interference (where the friction's holding term is 0) is on the active path only where another point moves it. The
spec expected `calibration.alpha_br_per_C` -> `shaft_joint.hub_fit_allowed_C` and -> `shaft_joint.hub_fittable_max_um`
to need entries "unless another trace point has the onset binding on a library part". One does: in the prototype,
listing those two pairs and `temperature.demag.knee_fraction` -> `shaft_joint.hub_fit_allowed_C` in `CANCELLATIONS`
made the test fail with "CANCELLATIONS lists calibration.alpha_br_per_C → shaft_joint.hub_fit_allowed_C, but it moves:
remove the entry" (and the same for the other two), so each moves at some trace point and none may be listed. To
repeat the check, add the three entries, run
`cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain each_assumption_moves`, see the three "but it
moves" lines, and remove them again; never commit them.

- [ ] **Step 10: Docs**

`magcoupling-rs/README.md`. Nine edits; each "Old" is a substring of one line (the line numbers are at Task 2's
commit); replace only it.

(a) The Layout table's `src/engine/assumptions.rs` row (around :151). Old:
```text
`ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source)
```
New:
```text
`ASSUMPTIONS` (the spec's 14 rows and BL-064's four shaft joint rows, the shrink-fit friction, the surface smoothing, the rotor hotter than its shaft and the joining clearance, over the 19 inputs flagged `.assumption()`: label, input paths, rationale, source)
```

(b) The `src/engine/explain/` row (around :153), two substrings. Old:
```text
the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes)
```
New:
```text
the static tables a formula reads (`tables.rs`: magnets, grades, the part materials (the back iron's modulus, expansion and yield too, for the shaft joints), aluminium alloys, adhesives, screw sizes)
```
Old:
```text
the v1 scope (`scope.rs`, decision 31),
```
New:
```text
the v1 scope (`scope.rs`, decision 31, with BL-064's inner-ring single-ring onset and its `shaft_joint` chain: 193 paths),
```

(c) A new Layout row after the `src/engine/explain/` row (spec 8, review SF-14: a named row for the records file).
Insert after the line that ends `slice 2, spec 3.4) |`:
```markdown
| `src/engine/explain/records/shaft_joint.rs` | BL-064 shaft joint records: one per numeric `shaft_joint.*` result (23; the verdict, a text, has none), transcribed from `shaft_joint::compute` and `PartProperties::hub` with every correction on: the part material in effect as cases over the circuit and the back-iron pick (the Materials inputs for the default pick, a pick's library values, else the workbook's 6061-T6), each joint's Lamé rows (holding, bore stress, needed minimum, elastic edge) with a no-wall arm that gives NaN, the key alone (0 with no keyway, +inf with no bearing length or contact), each shaft's shrink as cases on 265 K with NIST's Fe-2.25Ni polynomial, the fitting temperature and the fittable edge, and the bonded inner hub's allowed fitting temperature as the lowest of your limit, the adhesive's design limit and the inner ring's own single-ring onset less 10 °C, NaN when any of the three is (owner R-15) |
```

(d) The `src/gui/` row's `result_groups.rs` (around :154). Old:
```text
`result_groups.rs` (`result_groups`: the headline, the eight chains, the other results by package;
```
New:
```text
`result_groups.rs` (`result_groups`: the headline, the nine chains, the other results by package;
```

(e) The trace paragraph (around :296). Old:
```text
the registry explains 393 of the 1112 results, so an input such as the
```
New:
```text
the registry explains 416 of the 1112 results, so an input such as the
```

(f) The Tests table's `tests/assumptions.rs` row (around :661). Old:
```text
every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets;
```
New:
```text
every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set, then BL-064's four; each row's source cites the workbook cell of every input it sets (a Rust-only input's source cites its spec instead: Addendum A3, or the BL-064 shaft joint spec); the shaft joints' four rows are on Rust-only inputs, cite the BL-064 shaft joint spec by name (the friction row also by its file, which exists) and show whether the shaft-clamp section is paused or not;
```

(g) The Tests table's `tests/explain.rs` row (around :666), four substrings. Old:
```text
reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults, at the edge points (
```
New:
```text
reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults, at the five probe designs, at the edge points (
```
Old:
```text
, and six points where a clamp count lies outside i64
```
New:
```text
, the shaft joints' branches, each asserted at its point (no wall around the cup's bore, a keyway through the inner hub's wall: those Lamé rows NaN; no keyway; no key bearing length in the cup or the inner hub; a keyway that caps the key's contact, at two depths; an uncooled wheel shaft; an inner single-ring onset of 0/0; the adhesive's limit, at two adhesives, and the inner magnets' onset capping the inner hub's fitting temperature), and six points where a clamp count lies outside i64
```
Old:
```text
the sleeve and cap materials, E17's free-space fields), corrections on;
```
New:
```text
the sleeve and cap materials, E17's free-space fields, the shaft joint inputs), corrections on;
```
Old:
```text
every path of an explained chain has a record and drills down to inputs.
```
New:
```text
every path of an explained chain has a record and drills down to inputs. The shaft joint chain (BL-064) is every numeric `shaft_joint.*` result in schema order, after the geometry callouts, each with its record and the verdict without one; the allowed hub temperature record is `shaft_joint::hub_fit_allowed`'s value (the lowest of three, ties to the earlier, +inf never binding) with a NaN in any of its three terms, and reads the inner ring's own onset; each shaft's shrink record follows `shaft_joint::shaft_shrink_um` at every step of its slider, both arms taken.
```
Old:
```text
each of the 15 assumption inputs reaches an explained result
```
New:
```text
each of the 19 assumption inputs reaches an explained result
```

(h) The Tests table's `src/gui/inputs.rs`, … `results_table.rs` row (around :677), two substrings. Old:
```text
the headline first, then the eight chains, then the other results,
```
New:
```text
the headline first, then the nine chains (the shaft joints between the geometry and the mass), then the other results,
```
Old:
```text
393 of the 1112 results are explained, every input traceable
```
New:
```text
416 of the 1112 results are explained, every input traceable
```

(i) The results table paragraph's chain list (around :359; ruling C-1). Old:
```text
demagnetization, slip heating, clamps and geometry, each with the results of its
```
New:
```text
demagnetization, slip heating, clamps, geometry and the shaft joints (BL-064), each with the results of its
```

`docs/ai/03-structure.yaml`. Four edits (the `explain:` entry is an unquoted scalar: none adds a ": ").

(a) The `assumptions:` entry (around :49). Old:
```text
ASSUMPTIONS, 14 rows over the 15 inputs flagged .assumption() with rationale and source;
```
New:
```text
ASSUMPTIONS, 18 rows over the 19 inputs flagged .assumption() with rationale and source (BL-064's four shaft joint rows cite the BL-064 shaft joint spec, no workbook cell);
```

(b) The `explain:` entry (around :64), two substrings. Old:
```text
records/ (torque, demagnetization, slip_heating, temperature, clamps, dashboard, geometry; #[rustfmt::skip] data)
```
New:
```text
records/ (torque, demagnetization, slip_heating, temperature, clamps, dashboard, geometry, shaft_joint; #[rustfmt::skip] data)
```
Old:
```text
scope.rs SCOPE (decision 31 plus the G1 geometry callouts and BL-064's inner-ring single-ring onset, 170 paths, all explained)
```
New:
```text
scope.rs SCOPE (decision 31 plus the G1 geometry callouts, BL-064's inner-ring single-ring onset and its shaft_joint chain of 23, 193 paths, all explained)
```

(c) The `gui:` entry's `result_groups.rs` (around :68). Old:
```text
CHAIN_LABELS (the scope's six chains, adhesive, mass)
```
New:
```text
CHAIN_LABELS (the scope's seven chains, the shaft joints among them, adhesive, mass)
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

`docs/ai/02-system.yaml`, the drift guard's invariant (around :97; its "(169)" was stale since Task 1 moved the scope to
170), two substrings. Old:
```text
reproduces the engine's result by the parity rule at the defaults and at every differential case under every augmentation, corrections on,
```
New:
```text
reproduces the engine's result by the parity rule at the defaults, the five probe designs, the edge points and every differential case under every augmentation, corrections on,
```
Old:
```text
every path of the explorer's scope (169) drills down to inputs.
```
New:
```text
every path of the explorer's scope (193) drills down to inputs.
```
(`02-system.yaml` has a YAML parse error at line 15 from before BL-059, reported then and not BL-064's; these two
edits are inside one quoted string and add no ": ", so they change nothing for a parser.)

- [ ] **Step 11: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`:

| binary | result |
|---|---|
| lib | 231 passed, 1 ignored |
| assumptions | 12 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 60 passed, 1 ignored |
| gearbox | 13 passed |
| grades | 12 passed |
| material_library | 4 passed |
| material_links | 14 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 13 passed |
| schema | 11 passed |
| shaft_joint | 30 passed |
| size_controls | 38 passed |
| sizing | 44 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

The Python parity suites (`python_schema`, `parity`, `differential`, `deviations`) pass at Task 2's counts: the
records, symbols, scope and assumptions are Rust-only, and the four flagged inputs are `rust_only`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `889 passed; 0 failed; 6 ignored`; the two app binaries 0 tests; `explain` `61 passed; 0 failed; 1
ignored`; every other integration binary as above.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`, then the same with
`--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/assumptions.rs
 M magcoupling-rs/src/engine/explain/records/mod.rs
 M magcoupling-rs/src/engine/explain/scope.rs
 M magcoupling-rs/src/engine/explain/symbols.rs
 M magcoupling-rs/src/engine/explain/tables.rs
 M magcoupling-rs/src/engine/shaft_joint.rs
 M magcoupling-rs/src/gui/result_groups.rs
 M magcoupling-rs/src/gui/trace.rs
 M magcoupling-rs/tests/assumptions.rs
 M magcoupling-rs/tests/data/input_schema.json
 M magcoupling-rs/tests/explain.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
?? magcoupling-rs/src/engine/explain/records/shaft_joint.rs
```

- [ ] **Step 12: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T3.txt` with the Write tool:
```text
feat(magcoupler): the shaft joint check's equation records, scope chain and assumptions (BL-064 T3)

Every number of the shaft joint check is explained and held to the engine:
records/shaft_joint.rs has one record per numeric shaft_joint result (23),
transcribed from shaft_joint::compute and PartProperties::hub: the part
material over the circuit and the back-iron pick (tables.rs reads the back
iron's modulus, expansion and yield), each joint's Lamé rows with a no-wall
arm (NaN), the key alone, each shaft's shrink on NIST's Fe-2.25Ni fit below
265 K, the fitting temperature and the fittable edge, and the bonded inner
hub's allowed fitting temperature, the lowest of your limit, the adhesive's
design limit and the inner ring's own onset less 10 °C, NaN when any is
(owner R-15; four arms, since the markup's min drops a NaN).

The scope gains the shaft_joint chain after the geometry callouts (193
paths) and the results table its heading (CHAIN_LABELS 9); the trace totals
are 416 of 1112. The drift guard also runs at the five probe designs, 12
shaft joint edge points (each asserted on its edge) and a shaft joint
inputs augmentation; the allowed-temperature record matches the helper for
a NaN in each position, and the shrink records match the engine at every
slider step.

Four assumptions: the shrink-fit friction, the surface smoothing, the rotor
hotter than its shaft and the joining clearance, flagged .assumption(); a
Rust-only assumption's source may cite the BL-064 shaft joint spec. No
CANCELLATIONS entry is needed (checked). input_schema.json is re-blessed
for the four flags only (gen_differential drops Rust-only inputs).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run (never `git add -A` or `git add .`; never stage `docs/analyses/2026-05-28-press-4bar-analysis.md` or
`docs/chebyshev_lambda/`):
```bash
git add magcoupling-rs/src/engine/explain/records/shaft_joint.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/tables.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/assumptions.rs magcoupling-rs/src/engine/shaft_joint.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/gui/trace.rs magcoupling-rs/tests/explain.rs magcoupling-rs/tests/assumptions.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T3.txt
git status --short
```

Expected: one commit of 15 files; `git status --short` then shows only
`?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

#### Pinned tests that move (each confirmed by running; old -> new, reason)

| test | old -> new | reason |
|---|---|---|
| `explain::explained_chains_have_every_record_and_scope_paths_exist` | union 170 -> 193; its message gains "and its 23 shaft joint results" | the `shaft_joint` chain's 23 paths (spec 7.17: 169 -> 193 over Tasks 1 and 3) |
| `explain::each_assumption_moves_what_depends_on_it_and_nothing_else` | `paths.len()` 15 -> 19; its comment "fifteen … fourteen rows" -> "nineteen … eighteen rows, BL-064's four among them" | the four new assumption inputs, each reaching an explained result (spec 7.10) |
| `explain::every_record_reproduces_the_engine_everywhere` | the point count `1 + edge_points + cases × augmentations` -> `1 + probe designs + edge_points + cases × augmentations`; augmentations 17 -> 18; edge points + 12 | spec 7.12 (the records at every probe and at a no-wall design); the shaft joint arms and value terms no differential case reaches |
| `assumptions::the_rows_are_the_spec_v1_set` | 14 ids -> 18 (+ `fit_friction`, `fit_smoothing`, `rotor_over_shaft`, `joining_clearance`) | the four rows (spec 4.5) |
| `assumptions::every_row_has_a_label_rationale_source_and_one_unit` | a Rust-only input's source must contain "Addendum A3" -> "Addendum A3" or "BL-064 shaft joint spec" (`RUST_ONLY_SOURCES`) | review 2 MF-3, spec 4.5 |
| `gui::result_groups::tests::the_headline_comes_first_then_the_chains_then_the_other_results` | `ids[..9]` -> `ids[..10]` with "shaft_joint" between "geometry" and "mass"; `groups[..9]`/`[9..]` -> `[..10]`/`[10..]`; + the inner onset in "demagnetization"; + the shaft joint rows' sections (the chain; the verdict in the package group until Task 5) | `CHAIN_LABELS` 8 -> 9 (spec 5.4) |
| `gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so` | `(393, 1112)` -> `(416, 1112)`; module doc likewise | the 23 records (spec 7.17: `(392, 1087)` -> `(416, 1112)` over Tasks 1-3) |
| `schema::input_schema_json_is_current` | the blessed file re-blessed (four `"assumption"` flags `false` -> `true`) | plan deviation PD-9 |

New: `tables::the_back_iron_table_reads_each_pick_s_modulus_expansion_and_yield`,
`explain::the_shaft_joint_chain_is_every_numeric_shaft_joint_result_in_schema_order`,
`explain::the_allowed_hub_temperature_record_is_the_helper_s_lowest_of_three`,
`explain::the_shaft_shrink_records_follow_the_engine_over_their_sliders`,
`assumptions::the_shaft_joint_rows_are_rust_only_and_cite_the_bl064_spec`. Counts: engine lib 230 -> 231, `explain`
57 -> 60 (58 -> 61 with `app`), `assumptions` 11 -> 12, app lib 888 -> 889.

Not moved, still true (each run in the prototype): `no_input_moves_a_result_the_graph_says_is_independent_of_it`
(soundness: with `each_assumption_moves_what_depends_on_it_and_nothing_else`, sensitivity, it is one of the two
callers of `check_traceability` that spec 7.12 names, both run over the new records; the graph holds every new edge; each `shaft_joint.*` input changes a result, so no `INERT_INPUTS` entry),
`records_never_panic_on_inputs_that_bypass_set`, `the_registry_is_consistent` (no custom eval; every new term has a
symbol), `every_explained_chain_drills_down_to_inputs` (the records' result terms all have records:
`metal.torque_cold_high_Nm`, `materials.circuit_backiron`, `housing.hub_length_mm`,
`temperature.adhesive.design_limit_C`, `temperature.demag.inner_onset_single_ring_C`),
`notes_link_to_records_and_each_equation_has_at_most_one` (no note owns the new records until Task 7),
`every_field_reads_and_a_missing_key_is_none`, `every_flagged_input_is_in_exactly_one_row`,
`changing_one_assumption_flags_exactly_its_row`, `reset_restores_every_assumption_and_keeps_the_design_inputs`,
`each_assumption_moves_a_result_at_the_default_design` (each new assumption at its far slider end moves a shaft joint
result), `every_label_and_rationale_reads_in_plain_words`,
`only_the_clamp_assumptions_are_hidden_and_only_while_the_clamp_section_is_paused`,
`every_chain_path_names_results_and_every_package_has_a_heading`, `every_result_is_in_exactly_one_group`, and every GUI
test of the Assumptions view, the inputs' hovers and the XLSX Inputs sheet.

---

### Task 4: The drawn key from the shaft joints' key width in both states, the exports of the joints, the clamp-name guard and Auto

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 5.3 (pinned at the export layer only: Task 2 places the inputs, its plan deviation PD-4), 5.4 (the drawing, the exports, Auto), 4.5's last bullet (the `PAUSED` doc), 7 items 13-15 and 17 (the key outline, the stale texts the key move leaves). The production change is one: the geometry view's key takes `shaft_joint.key_width_mm` and is drawn whether or not the paused shaft-clamp section shows, and no key is drawn at a keyway depth of 0 or less (ruling C-5, plan deviation PD-3). Everything else is tests, two shared test designs, the texts the key move makes stale, and docs.

**What this task does not do (plan deviations PD-12 to PD-14 of the Deviation index):**
- The inputs panel's place for the 15 inputs (`SECTION_LABELS`, the `shaft` section's key rows, `shaft.fit`, `shaft.fitting`) is Task 2's (its plan deviation PD-4: eight GUI tests fail without it), and so are the README phrases that placement and Task 2's export test leave stale (Task 2 Step 11 (j) to (l)). This task pins that place in the spreadsheet export, in both orders.
- The result-group headings are Task 2's (`PACKAGE_LABELS`) and Task 3's (`CHAIN_LABELS`, which `every_chain_path_names_results_and_every_package_has_a_heading` forces into the task that adds the scope chain). This task pins the chain heading's place in the Results sheet.
- The XLSX Summary's 19 rows with the joint badge and its note column, and "the check in the headline group", need Task 5's `DASHBOARD`, `CHECKS`, `verdict_level`, `END_EFFECT_ROWS` and `STORED_3D_ROWS`, so they are Task 5's, with the Summary tests. Task 5 Step 1e replaces the eighteen-row Summary test whole, from its `#[test]` to its closing brace, so this task's tests after it stay out of that Old text.

Line numbers are at Task 3's commit, approximate: trust the code and grep for the names.

**Files:**
- Modify: `magcoupling-rs/src/gui/geometry.rs` (the key from `shaft_joint.key_width_mm` in both states, none at a keyway depth of 0 or less; `end_view` loses its `clamps_paused` parameter; the comments; the tests)
- Modify (texts the key move leaves stale): `magcoupling-rs/src/engine/clamps.rs` (the `PAUSED` doc), `magcoupling-rs/tests/explain.rs` (`INERT_INPUTS`' reason for `clamps.key_width_mm`)
- Modify (tests): `magcoupling-rs/src/gui/test_support.rs` (two shared designs), `magcoupling-rs/src/gui/results_table.rs`, `magcoupling-rs/src/gui/spreadsheet.rs`, `magcoupling-rs/src/gui/panel.rs` (the clamp-name guard), `magcoupling-rs/tests/sizing.rs`
- Modify (docs): `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`
- Test: `gui::geometry::tests` (one moved, one renamed and moved, one new); `gui::results_table::tests::the_exports_carry_every_shaft_joint_input_and_result_with_no_cell` (new); `gui::spreadsheet::tests::the_inputs_and_results_sheets_carry_every_shaft_joint_row` (new); `gui::panel::tests::every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` (extended); `tests/sizing.rs` `auto_reads_no_shaft_joint_input_and_both_joints_hold_at_its_answer` (new) and `auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there` (its solve shared)

**Interfaces:**
- Consumes:
  - Task 2: `DesignInputs::shaft_joint` (`ShaftJointInputs`, the 15 inputs, `key_width_mm` default 3.0, slider 1.0 to 6.0 step 0.5), `DesignResults::shaft_joint` (`ShaftJointResults`, 24 results: `check` and 23 numbers, `cup_holding_Nm`, `cup_key_pressure_MPa`, `hub_holding_Nm`, `required_Nm`), `engine::shaft_joint::SHAFT_JOINTS_PASS`; `gui::inputs::WORKFLOW` group `shaft` with sections `shaft`, `shaft.fit`, `shaft.fitting`; `gui::inputs::section_label("shaft_joint")`.
  - Task 3: `engine::explain::scope::SCOPE` chain `shaft_joint` (the 23 numeric paths); `gui::result_groups::CHAIN_LABELS` entry `("shaft_joint", "Shaft joints")` between `geometry` and `mass`; the `.assumption()` flag on `shaft_joint.friction`, `shaft_joint.smoothing_um`, `shaft_joint.rotor_over_shaft_C`, `shaft_joint.joining_clearance_um`.
  - Existing: `gui::geometry::{geometry, Geometry, Piece, Part, Outline, Note, parts_not_drawn}`; `gui::results_table::{results_csv, results_json, exact_value, csv_field}`; `gui::session::{design_from_json, json_value}`; `gui::spreadsheet::{sheets, value_cell, level_cell, flag, RUST_ONLY}`; `engine::sizing::{solve, FreeVariable, SizingPoint}`.
- Produces:
  - `gui::geometry::geometry(inputs, results, clamps_paused)`: same signature; the key drawn in both states from `shaft_joint.key_width_mm`, and none at a keyway depth of 0 or less (ruling C-5; only the clamp-boss design check still follows `clamps_paused`). The private `end_view` no longer takes `clamps_paused`.
  - `gui::test_support::shaft_joints_off_default() -> DesignInputs` (every shaft joint input one slider step off its default) and `gui::test_support::cup_without_wall_or_key_bearing() -> DesignInputs` (bore 16 mm, boss Ø12 mm, boss length 0: the cup's four Lamé results NaN, its key pressure +inf), both `pub(crate)`, for any later task's tests (Task 5's no-wall Problem tests may use the second).
  - `tests/sizing.rs`: `fn auto_on_length_at_the_defaults() -> SizingPoint` (Auto on length at the defaults, 13.77 mm).

- [ ] **Step 1: Check the tree and what this task builds on**

Run:
```bash
git status --short
grep -c '("shaft_joint", "Shaft joints")' magcoupling-rs/src/gui/result_groups.rs
grep -c 'id: "shaft_joint"' magcoupling-rs/src/engine/explain/scope.rs
grep -c '\.assumption()' magcoupling-rs/src/engine/shaft_joint.rs
```
Expected: `?? docs/analyses/2026-05-28-press-4bar-analysis.md` alone; `2` (Task 3's `CHAIN_LABELS` entry and Task 2's `PACKAGE_LABELS` entry); `1` (Task 3's scope chain); `4` (Task 3's assumption flags). Anything else: Task 2 or Task 3 is not committed; stop.

- [ ] **Step 2: Write the failing drawing tests (`geometry.rs`)**

In `magcoupling-rs/src/gui/geometry.rs`, `mod tests`, in `the_end_view_draws_every_part_from_the_results`, replace:

```rust
        // From where its 4 mm sides cross the 10 mm bore to 1.7 mm past the bore.
        assert_eq!(key[1], [6.7, -2.0]);
        assert!((key[0][0] - 21.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn the_key_and_the_clamp_boss_check_are_drawn_only_while_the_clamp_section_shows() {
        // Ruling 11: the key (its width is a clamp input) and the check against the clamp
        // inputs' boss go with the paused section; the wall rule and the cap check stay.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        for (paused, keys, notes) in [(false, 1, 3), (true, 0, 2)] {
            let g = geometry(&defaults, &results, paused);
            let drawn = g.end.pieces.iter().chain(&g.side.pieces);
            assert_eq!(
                drawn.filter(|p| p.part == Part::Key).count(),
                keys,
                "paused {paused}"
            );
            assert_eq!(g.notes.len(), notes, "paused {paused}");
            assert_eq!(
                g.notes
                    .iter()
                    .any(|n| n.text.ends_with("in the clamp inputs")),
                !paused
            );
        }
    }
```

with:

```rust
        // From where its 3 mm sides (the shaft joints' DIN 6885-1 3 x 3 key, BL-064) cross the
        // 10 mm bore to 1.7 mm past the bore.
        assert_eq!(inputs.shaft_joint.key_width_mm, 3.0);
        assert_eq!(key[1], [6.7, -1.5]);
        assert!((key[0][0] - 22.75_f64.sqrt()).abs() < 1e-12);
    }

    /// The drawn keys of `g`, in both views.
    fn keys(g: &Geometry) -> Vec<&Piece> {
        g.end
            .pieces
            .iter()
            .chain(&g.side.pieces)
            .filter(|p| p.part == Part::Key)
            .collect()
    }

    #[test]
    fn the_key_is_drawn_in_both_states_and_the_clamp_boss_check_only_while_the_clamp_section_shows()
    {
        // Ruling 11: the check against the clamp inputs' boss goes with the paused section; the
        // wall rule and the cap check stay. BL-064: the key is the shaft joints' (its width is
        // `shaft_joint.key_width_mm`), the backup of the shrink fits that hold both rotors, so
        // it is drawn whether the paused section shows or not.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        for (paused, notes) in [(false, 3), (true, 2)] {
            let g = geometry(&defaults, &results, paused);
            assert_eq!(keys(&g).len(), 1, "paused {paused}");
            assert_eq!(g.notes.len(), notes, "paused {paused}");
            assert_eq!(
                g.notes
                    .iter()
                    .any(|n| n.text.ends_with("in the clamp inputs")),
                !paused
            );
        }
    }

    #[test]
    fn the_key_takes_the_shaft_joints_width_and_never_the_paused_clamp_s() {
        // BL-064 (spec 5.4): the key's sides sit at half the shaft joints' key width either side
        // of the x axis, from where they cross the bore to the keyway's depth past it, in both
        // states; the paused clamp's own key width moves nothing in the drawing; with no keyway
        // there is no key (ruling C-5).
        let defaults = DesignInputs::default();
        let bore = defaults.coupling.bore_mm / 2.0;
        let tip = bore + defaults.coupling.keyway_depth_mm;
        let slider = crate::gui::inputs::InputCatalogue::get()
            .entry("shaft_joint.key_width_mm")
            .and_then(|entry| entry.meta.range)
            .expect("the key width has a slider");
        for paused in [false, true] {
            // Each end of the key width's slider (1 and 6 mm on the 10 mm bore).
            for width in [slider.min, slider.max] {
                let inputs = design(|i| i.shaft_joint.key_width_mm = width);
                let g = geometry(&inputs, &compute_all(&inputs), paused);
                let drawn = keys(&g);
                assert_eq!(drawn.len(), 1, "{width} mm, paused {paused}");
                let w = width / 2.0;
                let seat = (bore * bore - w * w).sqrt();
                assert_eq!(
                    drawn[0].outline,
                    Outline::Polygon(vec![[seat, -w], [tip, -w], [tip, w], [seat, w]]),
                    "{width} mm, paused {paused}"
                );
            }
            // The paused clamp's key width, at each end of its slider: the same drawing.
            let clamp_slider = crate::gui::inputs::InputCatalogue::get()
                .entry("clamps.key_width_mm")
                .and_then(|entry| entry.meta.range)
                .expect("the clamp's key width has a slider");
            let at_defaults = geometry(&defaults, &compute_all(&defaults), paused);
            for width in [clamp_slider.min, clamp_slider.max] {
                let inputs = design(|i| i.clamps.key_width_mm = width);
                assert_eq!(
                    geometry(&inputs, &compute_all(&inputs), paused),
                    at_defaults,
                    "the clamp's key width {width} mm, paused {paused}"
                );
            }
            // A key typed wider than the bore: its sides start on the axis, still one key.
            let wide = design(|i| i.shaft_joint.key_width_mm = 12.0);
            let g = geometry(&wide, &compute_all(&wide), paused);
            let drawn = keys(&g);
            assert_eq!(drawn.len(), 1, "paused {paused}");
            assert_eq!(
                drawn[0].outline,
                Outline::Polygon(vec![[0.0, -6.0], [tip, -6.0], [tip, 6.0], [0.0, 6.0]])
            );
            // A key width that is not a number: no key, and the end view's note counts it.
            let nan = design(|i| i.shaft_joint.key_width_mm = f64::NAN);
            let g = geometry(&nan, &compute_all(&nan), paused);
            assert!(keys(&g).is_empty(), "paused {paused}");
            assert!(
                g.notes.contains(&Note {
                    text: parts_not_drawn(1, "end"),
                    level: Some(Level::Caution)
                }),
                "{:?}",
                g.notes
            );
            // No keyway (depth 0): the shaft joints have no key term, so no key is drawn (ruling
            // C-5, plan deviation PD-3).
            let keyless = design(|i| i.coupling.keyway_depth_mm = 0.0);
            let g = geometry(&keyless, &compute_all(&keyless), paused);
            assert!(keys(&g).is_empty(), "keyway 0, paused {paused}");
        }
    }
```

(`the_end_view_draws_every_part_from_the_results` binds `inputs` to `DesignInputs::default()` at its top; `design`, `Note`, `Level` and the private `parts_not_drawn` are in scope through `use super::*`.)

- [ ] **Step 3: Run the drawing tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::geometry::tests::the_end_view_draws_every_part_from_the_results gui::geometry::tests::the_key`

Expected: `test result: FAILED. 0 passed; 3 failed`, for these reasons (the key is still the clamp's 4 mm key, and not drawn while paused):
```text
gui::geometry::tests::the_end_view_draws_every_part_from_the_results
  left: [6.7, -2.0]
 right: [6.7, -1.5]
gui::geometry::tests::the_key_is_drawn_in_both_states_and_the_clamp_boss_check_only_while_the_clamp_section_shows
  assertion `left == right` failed: paused true
  left: 0
 right: 1
gui::geometry::tests::the_key_takes_the_shaft_joints_width_and_never_the_paused_clamp_s
  assertion `left == right` failed: 1 mm, paused false
  left: Polygon([[4.58257569495584, -2.0], [6.7, -2.0], [6.7, 2.0], [4.58257569495584, 2.0]])
 right: Polygon([[4.9749371855331, -0.5], [6.7, -0.5], [6.7, 0.5], [4.9749371855331, 0.5]])
```

- [ ] **Step 4: Draw the key from the shaft joints' width in both states (`geometry.rs`)**

In `magcoupling-rs/src/gui/geometry.rs`, `end_view`'s parameters, replace:

```rust
    diameter: Option<f64>,
    clamps_paused: bool,
    notes: &mut Vec<Note>,
) -> View {
```

with:

```rust
    diameter: Option<f64>,
    notes: &mut Vec<Note>,
) -> View {
```

In `end_view`'s body, replace:

```rust
    // The key in the hub's keyway, at 0°: from where its sides cross the bore to the keyway's
    // depth past the bore (Calculator C40), the key's width (Shaft clamps C58). The key is the
    // shaft-clamp section's: not drawn while it is paused.
    if !clamps_paused {
        let w = inputs.clamps.key_width_mm / 2.0;
        let seat = (bore * bore - w * w).max(0.0).sqrt();
        let tip = bore + c.keyway_depth_mm;
        pieces.push(Piece {
            part: Part::Key,
            outline: Outline::Polygon(vec![[seat, -w], [tip, -w], [tip, w], [seat, w]]),
        });
    }
```

with:

```rust
    // The key in the hub's keyway, at 0°: from where its sides cross the bore to the keyway's
    // depth past the bore (Calculator C40), the shaft joints' key width (BL-064). The key backs
    // up the shrink fits that hold both rotors, so it is drawn whether or not the paused
    // shaft-clamp section shows.
    // No keyway (depth 0 or less): no key, as the shaft joints then have no key term
    // (shaft_joint::joint). A depth that is not a number keeps its key and its "not drawn" note.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // a NaN depth is not "no keyway"
    if !(c.keyway_depth_mm <= 0.0) {
        let w = inputs.shaft_joint.key_width_mm / 2.0;
        let seat = (bore * bore - w * w).max(0.0).sqrt();
        let tip = bore + c.keyway_depth_mm;
        pieces.push(Piece {
            part: Part::Key,
            outline: Outline::Polygon(vec![[seat, -w], [tip, -w], [tip, w], [seat, w]]),
        });
    }
```

(Ruling C-5, plan deviation PD-3: the guard is local to the drawing, so the reviewed `shaft_joint::joint` is untouched. Clippy's `neg_cmp_op_on_partial_ord` rejects the NaN-keeping form under `-D warnings`, hence the statement's `allow`.)

Replace `geometry`'s doc and its call of `end_view`:

```rust
/// The geometry of the design shown: its inputs and the results computed from them; while
/// the shaft-clamp section is paused (`clamps_paused`) without its key and its boss check.
```

with:

```rust
/// The geometry of the design shown: its inputs and the results computed from them; while
/// the shaft-clamp section is paused (`clamps_paused`) without its boss check.
```

and

```rust
    let mut end = end_view(inputs, results, diameter, clamps_paused, &mut all_notes);
```

with:

```rust
    let mut end = end_view(inputs, results, diameter, &mut all_notes);
```

(`geometry` keeps `clamps_paused`: `notes` still reads it for the clamp-boss check. No caller changes: `geometry_view.rs`, `panel.rs` and the tests call `geometry`, not `end_view`.)

- [ ] **Step 5: Run the drawing tests and see them pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::geometry`

Expected: `test result: ok. 33 passed; 0 failed` (the `geometry` and `geometry_view` tests; the three of Step 2 among them). The NaN test `a_number_that_is_not_finite_draws_nothing_wrong_and_says_so` passes unchanged: it draws unpaused, where the key was drawn before too.

- [ ] **Step 6: Two shared test designs (`test_support.rs`)**

The export tests of Steps 7 and 8 read the same two designs (DRY). In `magcoupling-rs/src/gui/test_support.rs`, replace:

```rust
    inputs.clamps.safety_factor = capacity / base.clamps.max_torque_Nm * (1.0 - 1e-9);
    inputs
}
```

with:

```rust
    inputs.clamps.safety_factor = capacity / base.clamps.max_torque_Nm * (1.0 - 1e-9);
    inputs
}

/// The defaults with every shaft joint input (BL-064) one slider step off its default: up, or
/// down where up would leave the slider. Each input then reads back as itself, not its default.
pub(crate) fn shaft_joints_off_default() -> crate::DesignInputs {
    use crate::engine::meta::{InputSet, Value, input_rows};
    let mut inputs = crate::DesignInputs::default();
    for row in input_rows(&crate::DesignInputs::default())
        .into_iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
    {
        let (Value::Num(x), Some(range)) = (row.value, row.meta.range) else {
            panic!("{} is a number with a slider", row.path)
        };
        let step = if x + range.step <= range.max {
            range.step
        } else {
            -range.step
        };
        inputs
            .set(&row.path, Value::Num(x + step))
            .expect("a value on the slider");
    }
    inputs
}

/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
/// wall around its bore, so its holding, bore stress, needed and elastic results are not a
/// number, and its key no bearing length (the web alone, 2.5 mm, under the 3 mm key), so its
/// key pressure is +inf.
pub(crate) fn cup_without_wall_or_key_bearing() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.bore_mm = 16.0;
    inputs.metal.boss_od_mm = 12.0;
    inputs.metal.boss_length_mm = 0.0;
    inputs
}
```

- [ ] **Step 7: The CSV and JSON exports carry every shaft joint input and result (`results_table.rs`)**

In `magcoupling-rs/src/gui/results_table.rs`, `mod tests`, replace:

```rust
    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
```

with:

```rust
    #[test]
    fn the_exports_carry_every_shaft_joint_input_and_result_with_no_cell() {
        use crate::engine::meta::input_rows;
        use crate::gui::session::design_from_json;
        use crate::gui::test_support::{cup_without_wall_or_key_bearing, shaft_joints_off_default};
        // BL-064 (spec 7.15): the shaft joints' 15 inputs and 24 results are Rust-only. The CSV
        // carries each result once, together and in schema order, after the gearbox check and
        // before the sweeps, with its engine label, its value at full precision, its unit and no
        // cell; the JSON the same, and its design every input with its value. Three designs:
        // the defaults; every input one step off its default (the design reads back from the
        // JSON as it was); and a bore of 16 mm in a 12 mm boss with no boss length, where the
        // cup has no wall (its holding is not a number) and its key no bearing length (+inf):
        // both export as text, never as a number or a JSON null.
        for inputs in [
            DesignInputs::default(),
            shaft_joints_off_default(),
            cup_without_wall_or_key_bearing(),
        ] {
            let design = Design {
                inputs,
                ..Design::default()
            };
            let results = compute_all(&design.inputs);
            let rows: Vec<_> = result_rows(&results)
                .into_iter()
                .filter(|row| row.path.starts_with("shaft_joint."))
                .collect();
            assert_eq!(rows.len(), 24);
            assert!(rows.iter().all(|row| row.cell.is_none()));
            // The CSV: the 24 lines together, between the gearbox check and the sweeps.
            let csv = results_csv(&results);
            let lines: Vec<&str> = csv.split("\r\n").collect();
            let first = lines
                .iter()
                .position(|l| l.starts_with("shaft_joint."))
                .expect("the shaft joints' CSV lines");
            for (line, row) in lines[first..first + 24].iter().zip(&rows) {
                let value = exact_value(&row.value);
                let want = [row.path.as_str(), row.meta.label, &value, row.meta.unit, ""]
                    .map(csv_field)
                    .join(",");
                assert_eq!(*line, want);
            }
            assert!(
                lines[first - 1].starts_with("gearbox.cold_high_check,"),
                "{}",
                lines[first - 1]
            );
            assert!(
                lines[first + 24].starts_with("gap_sweep["),
                "{}",
                lines[first + 24]
            );
            // The JSON: the same rows in the same order, and the design's every input.
            let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
            let entries: Vec<&Json> = json["results"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["path"].as_str().unwrap().starts_with("shaft_joint."))
                .collect();
            assert_eq!(entries.len(), 24);
            for (entry, row) in entries.iter().zip(&rows) {
                assert_eq!(entry["path"], row.path.as_str());
                assert_eq!(entry["label"], row.meta.label);
                assert_eq!(entry["value"], json_value(&row.value), "{}", row.path);
                assert_eq!(entry["unit"], row.meta.unit);
                assert_eq!(entry["cell"], Json::Null);
            }
            let shaft_inputs: Vec<_> = input_rows(&design.inputs)
                .into_iter()
                .filter(|row| row.path.starts_with("shaft_joint."))
                .collect();
            assert_eq!(shaft_inputs.len(), 15);
            for row in &shaft_inputs {
                assert_eq!(
                    json["design"]["inputs"][row.path.as_str()],
                    json_value(&row.value),
                    "{}",
                    row.path
                );
            }
            let read = design_from_json(&json["design"].to_string()).expect("the design reads");
            assert_eq!(read.inputs.shaft_joint, design.inputs.shaft_joint);
        }
        // The edge design writes a value that is not a number and +inf as text.
        let results = compute_all(&cup_without_wall_or_key_bearing());
        assert!(results.shaft_joint.cup_holding_Nm.is_nan());
        assert_eq!(results.shaft_joint.cup_key_pressure_MPa, f64::INFINITY);
        let csv = results_csv(&results);
        let line = |path: &str| {
            csv.split("\r\n")
                .find(|l| l.starts_with(&format!("{path},")))
                .unwrap()
                .to_owned()
        };
        assert!(line("shaft_joint.cup_holding_Nm").contains(",NaN,"));
        assert!(line("shaft_joint.cup_key_pressure_MPa").contains(",+inf,"));
    }

    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
```

- [ ] **Step 8: The Inputs and Results sheets carry every shaft joint row (`spreadsheet.rs`)**

The headings are read from where the page takes them (`WORKFLOW`, `section_label`, `CHAIN_LABELS`), so Task 8's G-3 review may reword them without moving this test; the paths, the binding interfaces, are pinned. In `magcoupling-rs/src/gui/spreadsheet.rs`, `mod tests`, replace:

```rust
    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
```

with:

```rust
    /// The lines of `sheet` from its group heading `heading` up to the next group heading: each
    /// heading's text, each input or result row's path (column `path`).
    fn group_lines<'s>(sheet: &'s Sheet, path: usize, heading: &'s str) -> Vec<&'s str> {
        let start = sheet
            .rows
            .iter()
            .position(|row| row[0].style == CellStyle::Group && row[0].as_text() == heading)
            .unwrap_or_else(|| panic!("no group heading {heading}"));
        let mut lines = vec![heading];
        for row in &sheet.rows[start + 1..] {
            match row[0].style {
                CellStyle::Group => break,
                CellStyle::Section => lines.push(row[0].as_text()),
                _ => lines.push(row[path].as_text()),
            }
        }
        lines
    }

    #[test]
    fn the_inputs_and_results_sheets_carry_every_shaft_joint_row() {
        use crate::engine::explain::scope::SCOPE;
        use crate::engine::meta::{input_rows, result_rows};
        use crate::gui::inputs::{WORKFLOW, section_label};
        use crate::gui::result_groups::CHAIN_LABELS;
        use crate::gui::test_support::{cup_without_wall_or_key_bearing, shaft_joints_off_default};
        // BL-064 (spec 5.3, 5.4 and 7.15): each of the shaft joints' 15 inputs and 24 results
        // has one row, with its value and unit and no workbook cell. The Inputs sheet lays the
        // inputs out as the inputs side does: in the workflow order the key's two rows follow
        // the keyway depth (and the paused clamp's key rows, which the exports keep), then the
        // "Shrink fit" and "Fitting temperatures" sections; in the original-sheet order they
        // are the "Shaft joints" group, in schema order. The four assumptions are flagged. The
        // Results sheet lists the 23 numbers under the "Shaft joints" chain, between Geometry and
        // Mass; a value that is not a number and +inf are the CSV's text. The headings are read
        // from where the page takes them (the G-3 review may reword them); the paths are pinned.
        const FIT: [&str; 8] = [
            "shaft_joint.hub_interference_min_um",
            "shaft_joint.hub_interference_max_um",
            "shaft_joint.cup_interference_min_um",
            "shaft_joint.cup_interference_max_um",
            "shaft_joint.safety_factor",
            "shaft_joint.friction",
            "shaft_joint.smoothing_um",
            "shaft_joint.rotor_over_shaft_C",
        ];
        const FITTING: [&str; 5] = [
            "shaft_joint.hub_fit_max_C",
            "shaft_joint.cup_fit_max_C",
            "shaft_joint.wheel_shaft_fit_C",
            "shaft_joint.output_shaft_fit_C",
            "shaft_joint.joining_clearance_um",
        ];
        const ASSUMED: [&str; 4] = [
            "shaft_joint.friction",
            "shaft_joint.smoothing_um",
            "shaft_joint.rotor_over_shaft_C",
            "shaft_joint.joining_clearance_um",
        ];
        let schema: Vec<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|row| row.path)
            .filter(|path| path.starts_with("shaft_joint."))
            .collect();
        assert_eq!(schema.len(), 15);
        let shaft = WORKFLOW
            .iter()
            .find(|group| group.id == "shaft")
            .expect("the shaft group");
        let section = |id: &str| {
            shaft
                .sections
                .iter()
                .find(|section| section.id == id)
                .unwrap_or_else(|| panic!("no section {id}"))
                .label
        };
        let chain_label = |id: &str| {
            CHAIN_LABELS
                .iter()
                .find(|&&(chain, _)| chain == id)
                .unwrap_or_else(|| panic!("no chain {id}"))
                .1
        };
        let heading = section_label("shaft_joint").expect("the shaft joints' heading");
        let input_path = column(&INPUT_COLUMNS, "Path");
        for inputs in [DesignInputs::default(), shaft_joints_off_default()] {
            let results = compute_all(&inputs);
            let changed = inputs != DesignInputs::default();
            for order in InputOrder::ALL {
                let mut shot = snapshot(&inputs, &results);
                shot.input_order = order;
                let book = sheets(&shot);
                let sheet = sheet(&book, INPUTS_SHEET);
                for path in &schema {
                    let entry = InputCatalogue::get().entry(path).expect("an input");
                    assert_eq!(
                        row_of(sheet, input_path, path),
                        &vec![
                            Cell::text(entry.meta.label),
                            value_cell(&inputs.get(path).expect("an input")),
                            Cell::text(entry.meta.unit),
                            Cell::text(RUST_ONLY),
                            Cell::text(path.clone()),
                            flag(changed),
                            flag(ASSUMED.contains(&path.as_str())),
                            flag(false),
                            flag(false),
                            Cell::empty(),
                        ],
                        "{order:?}"
                    );
                }
                let want: Vec<&str> = match order {
                    InputOrder::Workflow => [
                        shaft.label,
                        "coupling.bore_mm",
                        "coupling.keyway_depth_mm",
                        "clamps.key_width_mm",
                        "clamps.key_contact_mm",
                        "shaft_joint.key_width_mm",
                        "shaft_joint.key_contact_mm",
                        section("shaft.fit"),
                    ]
                    .into_iter()
                    .chain(FIT)
                    .chain([section("shaft.fitting")])
                    .chain(FITTING)
                    .collect(),
                    InputOrder::Workbook => [heading]
                        .into_iter()
                        .chain(schema.iter().map(String::as_str))
                        .collect(),
                };
                // The workflow group goes on with the paused clamp's sections after these.
                let lines = group_lines(sheet, input_path, want[0]);
                assert_eq!(lines[..want.len().min(lines.len())], want[..], "{order:?}");
                if order == InputOrder::Workbook {
                    assert_eq!(
                        lines.len(),
                        want.len(),
                        "the group ends with the joints' inputs"
                    );
                }
            }
        }
        // The Results sheet: the chain's 23 numbers under its heading, then the check's row.
        let chain = SCOPE
            .iter()
            .find(|chain| chain.id == "shaft_joint")
            .expect("the shaft joints' chain");
        let mut numbers: Vec<&str> = chain.paths.to_vec();
        numbers.sort_unstable();
        let mut want: Vec<String> = result_rows(&compute_all(&DesignInputs::default()))
            .into_iter()
            .map(|row| row.path)
            .filter(|path| path.starts_with("shaft_joint.") && path != "shaft_joint.check")
            .collect();
        want.sort_unstable();
        assert_eq!(numbers, want, "the chain is the 23 numbers");
        let result_path = column(&RESULT_COLUMNS, "Path");
        let check = column(&RESULT_COLUMNS, "Check");
        for inputs in [DesignInputs::default(), cup_without_wall_or_key_bearing()] {
            let results = compute_all(&inputs);
            let book = sheets(&snapshot(&inputs, &results));
            let sheet = sheet(&book, RESULTS_SHEET);
            let group_headings: Vec<&str> = sheet
                .rows
                .iter()
                .filter(|row| row[0].style == CellStyle::Group)
                .map(|row| row[0].as_text())
                .collect();
            let at = group_headings
                .iter()
                .position(|&heading| heading == chain_label("shaft_joint"))
                .expect("the shaft joints' chain heading");
            assert_eq!(
                (group_headings[at - 1], group_headings[at + 1]),
                (chain_label("geometry"), chain_label("mass"))
            );
            assert_eq!(
                group_lines(sheet, result_path, chain_label("shaft_joint"))[1..],
                chain.paths[..]
            );
            for path in chain.paths.iter().chain(["shaft_joint.check"].iter()) {
                let info = result_info(path).expect("a result");
                let row = row_of(sheet, result_path, path);
                assert_eq!(row[0], Cell::text(info.meta.label), "{path}");
                assert_eq!(row[1], value_cell(&results.get(path).unwrap()), "{path}");
                assert_eq!(row[2], Cell::text(info.meta.unit), "{path}");
                let level = if CHECKS.contains(path) {
                    level_cell(check_level(&results, path))
                } else {
                    Cell::empty()
                };
                assert_eq!(row[check], level, "{path}");
                assert_eq!(row[4], Cell::text(RUST_ONLY), "{path}");
                assert_eq!(row[6], Cell::empty(), "{path}");
            }
        }
        let edge = cup_without_wall_or_key_bearing();
        let results = compute_all(&edge);
        let book = sheets(&snapshot(&edge, &results));
        let sheet = sheet(&book, RESULTS_SHEET);
        assert_eq!(
            row_of(sheet, result_path, "shaft_joint.cup_holding_Nm")[1],
            Cell::text("NaN")
        );
        assert_eq!(
            row_of(sheet, result_path, "shaft_joint.cup_key_pressure_MPa")[1],
            Cell::text("+inf")
        );
    }

    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
```

(The check's own row is matched to `CHECKS` and `check_level` generically, so this test holds before Task 5 lists the check and after: before, the row has no badge; after, it carries the badge's level. Verified on Tasks 3 to 5 together at plan time.)

- [ ] **Step 9: The clamp-name guard reaches the shaft joints (`panel.rs`)**

Spec 7.14: the guard passes with the new sections, inputs and results with no allow-list entry (`ALLOWED` stays empty). This makes that coverage explicit: while paused, the texts the guard collects reach the label and the help of each of the 15 inputs and 24 results, the two new section headings and the group's heading, so a later change that hid them with the clamp (or stopped collecting them) fails here. In `magcoupling-rs/src/gui/panel.rs`, `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, replace:

```rust
                let named: Vec<&String> = texts.iter().filter(|t| names_the_clamp(t)).collect();
                assert!(named.is_empty(), "{named:#?}");
            } else {
```

with:

```rust
                let named: Vec<&String> = texts.iter().filter(|t| names_the_clamp(t)).collect();
                assert!(named.is_empty(), "{named:#?}");
                // Not vacuous for the shaft joints (BL-064), which show while the clamp is
                // paused: the enumeration reaches the label and the help of each of their inputs
                // and results, their two input sections and their group's heading.
                use crate::engine::meta::{input_rows, result_rows};
                use crate::gui::inputs::section_label;
                use crate::gui::plain_text::help_for;
                use crate::gui::result_groups::CHAIN_LABELS;
                let reaches = |text: &str| texts.iter().any(|t| t.contains(text));
                let paths: Vec<String> = input_rows(&DesignInputs::default())
                    .into_iter()
                    .map(|row| row.path)
                    .chain(
                        result_rows(&compute_all(&DesignInputs::default()))
                            .into_iter()
                            .map(|row| row.path),
                    )
                    .filter(|path| path.starts_with("shaft_joint."))
                    .collect();
                assert_eq!(paths.len(), 15 + 24);
                for path in &paths {
                    for text in [label_for(path), help_for(path)] {
                        assert!(reaches(text), "{path}: {text:?} is not collected");
                    }
                }
                let shaft = WORKFLOW
                    .iter()
                    .find(|group| group.id == "shaft")
                    .expect("the shaft group");
                let headings: Vec<&str> = shaft
                    .sections
                    .iter()
                    .filter(|section| ["shaft.fit", "shaft.fitting"].contains(&section.id))
                    .map(|section| section.label)
                    .chain(section_label("shaft_joint"))
                    .chain(
                        CHAIN_LABELS
                            .iter()
                            .filter(|&&(chain, _)| chain == "shaft_joint")
                            .map(|&(_, label)| label),
                    )
                    .collect();
                assert_eq!(headings.len(), 4);
                for heading in headings {
                    assert!(reaches(heading), "{heading:?} is not collected");
                }
            } else {
```

(`WORKFLOW`, `label_for`, `compute_all` and `DesignInputs` are already in the test module's scope.)

- [ ] **Step 10: Auto reads no shaft joint input (`tests/sizing.rs`)**

Spec 5.4 and decision 21: Auto is unchanged. Its search reads only hot-low and the validity rule, so each shaft joint input at either slider end gives Auto the same answer, bit for bit. Both joints hold at that answer.

At the defaults Auto on length gives 13.7695 mm. There the inner hub holds 10.2647 N·m, the cup holds 16.8990 N·m and each must hold 8.3654 N·m. These are measured; spec 5.4's 10.27 / 16.90 / 8.37 are section 3.6's row at a rounded 13.77 mm override (plan deviation PD-15).

The BL-059 Auto test solves the same thing, so its solve and its 13.77 mm pin become one helper that both tests use (DRY).

In `magcoupling-rs/tests/sizing.rs`, replace:

```rust
use magcoupling::engine::meta::{SliderRange, Value, result_rows};
use magcoupling::engine::model::{
    END_EFFECT_OUT_OF_RANGE, RingFit, blocks_fit, pitch_share, ring_fits,
};
```

with:

```rust
use magcoupling::engine::meta::{InputSet, SliderRange, Value, input_rows, result_rows};
use magcoupling::engine::model::{
    END_EFFECT_OUT_OF_RANGE, RingFit, blocks_fit, pitch_share, ring_fits,
};
use magcoupling::engine::shaft_joint::SHAFT_JOINTS_PASS;
```

and replace:

```rust
#[test]
fn auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there() {
    // BL-059, owner decision 5A: Auto still sizes for the hot minimum check alone. Sized by
    // length to its own requirement (the Size tab's Auto on length searches the length slider,
    // as `solve` does), the default design's magnets reach 13.77 mm and hot-low meets the
    // requirement; cold-high at the gearbox input, about 0.88 N·m, is above the 0.6 N·m input
    // rating there. Cold-high over hot-low does not depend on the geometry, so no length passes
    // both, and the gearbox check is red at Auto's answer.
    let base = DesignInputs::default();
    let p = solved(solve(
        &base,
        FreeVariable::AxialLength,
        base.metal.required_min_Nm,
    ));
    assert!((p.value - 13.77).abs() < 0.005, "{}", p.value);
    let r = compute_all(&p.inputs);
```

with:

```rust
/// Auto on length at the defaults, sized to their own requirement (the Size tab's Auto on length
/// searches the length slider, as `solve` does): the magnets reach 13.77 mm.
fn auto_on_length_at_the_defaults() -> SizingPoint {
    let base = DesignInputs::default();
    let p = solved(solve(
        &base,
        FreeVariable::AxialLength,
        base.metal.required_min_Nm,
    ));
    assert!((p.value - 13.77).abs() < 0.005, "{}", p.value);
    p
}

#[test]
fn auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there() {
    // BL-059, owner decision 5A: Auto still sizes for the hot minimum check alone. Sized by
    // length to its own requirement, the default design's magnets reach 13.77 mm and hot-low
    // meets the requirement; cold-high at the gearbox input, about 0.88 N·m, is above the
    // 0.6 N·m input rating there. Cold-high over hot-low does not depend on the geometry, so no
    // length passes both, and the gearbox check is red at Auto's answer.
    let base = DesignInputs::default();
    let p = auto_on_length_at_the_defaults();
    let r = compute_all(&p.inputs);
```

and, after that test's closing brace (before `fn a_short_override_shrinks_the_axial_housing`), replace:

```rust
#[test]
fn a_short_override_shrinks_the_axial_housing() {
```

with:

```rust
#[test]
fn auto_reads_no_shaft_joint_input_and_both_joints_hold_at_its_answer() {
    // BL-064, decision 21: Auto is unchanged. It sizes for the hot minimum alone and reads no
    // shaft joint input: with each of the 15 at either end of its slider, Auto on length finds
    // the same length and the same hot-low torque, bit for bit. At its answer for the defaults
    // (13.7695 mm) both joints hold, the inner hub 10.2647 N·m and the cup 16.8990 N·m against
    // the 8.3654 N·m each must hold (the inner hub lengthens with the magnets, the cup's boss
    // does not), and the check passes. Not vacuous: there, each input moves the joints' results
    // exactly when it leaves its default.
    let base = DesignInputs::default();
    let p = auto_on_length_at_the_defaults();
    let joints = compute_all(&p.inputs).shaft_joint;
    assert_eq!(joints.check, SHAFT_JOINTS_PASS);
    for (got, want) in [
        (joints.hub_holding_Nm, 10.2647),
        (joints.cup_holding_Nm, 16.8990),
        (joints.required_Nm, 8.3654),
    ] {
        assert!((got - want).abs() < 5e-5, "{got} against {want}");
    }
    for row in input_rows(&base)
        .into_iter()
        .filter(|row| row.path.starts_with("shaft_joint."))
    {
        let range = row.meta.range.expect("a slider");
        for value in [range.min, range.max] {
            let mut edited = base.clone();
            edited
                .set(&row.path, Value::Num(value))
                .expect("a value on the slider");
            let q = solved(solve(
                &edited,
                FreeVariable::AxialLength,
                base.metal.required_min_Nm,
            ));
            assert_eq!(
                (q.value.to_bits(), q.torque_hot_low_Nm.to_bits()),
                (p.value.to_bits(), p.torque_hot_low_Nm.to_bits()),
                "{} at {value}",
                row.path
            );
            assert_eq!(
                compute_all(&q.inputs).shaft_joint != joints,
                Value::Num(value) != row.value,
                "{} at {value}",
                row.path
            );
        }
    }
}

#[test]
fn a_short_override_shrinks_the_axial_housing() {
```

- [ ] **Step 11: Run the five tests that pass on arrival**

They pin wiring Tasks 2 and 3 built (the engine's groups and order, the catalogue, the scope chain and its heading), so they pass when written; Step 12 proves each bites.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_exports_carry_every_shaft_joint the_inputs_and_results_sheets_carry every_text_the_panel_shows_has_glyphs`
Expected: `test result: ok. 3 passed; 0 failed`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test sizing auto_`
Expected: `test result: ok. 2 passed; 0 failed` (`auto_reads_no_shaft_joint_input_and_both_joints_hold_at_its_answer`, `auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there`).

- [ ] **Step 12: Prove each fails when its wiring breaks (six planted faults, each reverted)**

None of the six files is edited by this task before Step 13, so each must be clean again after its fault:

```bash
git diff --quiet HEAD -- magcoupling-rs/src/engine/api.rs magcoupling-rs/src/gui/session.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/engine/clamps.rs magcoupling-rs/src/engine/sizing.rs && echo clean
```

Expected: `clean`, before the first fault and after each undo. Use Step 11's two commands for every run.

1. `magcoupling-rs/src/engine/api.rs`, `DesignResults`: swap the two lines `            gearbox: GearboxResults,` and `            shaft_joint: ShaftJointResults,` (the joints' results before the gearbox's). Expected: `the_exports_carry_every_shaft_joint_input_and_result_with_no_cell` FAILS, printing the line before the first `shaft_joint.` line, `housing.retainer_span_mm,Retainer axial span in effect,14.5,mm,`; the other four pass. Undo it.
2. `magcoupling-rs/src/gui/session.rs`, `design_json`: replace
   ```rust
       for row in input_rows(&design.inputs) {
           inputs.insert(row.path, json_value(&row.value));
   ```
   with
   ```rust
       for row in input_rows(&design.inputs) {
           if row.path.starts_with("shaft_joint.") {
               continue;
           }
           inputs.insert(row.path, json_value(&row.value));
   ```
   Expected: `the_exports_carry_every_shaft_joint_input_and_result_with_no_cell` FAILS (`shaft_joint.hub_interference_min_um`, `left: Null`, `right: Number(13.0)`); the other four pass. Undo it.
3. `magcoupling-rs/src/gui/inputs.rs`, `WORKFLOW`: move the line `                    "shaft_joint.key_width_mm",` from the `shaft` section to the first line of the `shaft.fit` section's paths. Expected: `the_inputs_and_results_sheets_carry_every_shaft_joint_row` FAILS (`Workflow`: the key width listed under the Shrink fit heading); the other four pass. Undo it.
4. `magcoupling-rs/src/gui/result_groups.rs`: in `CHAIN_LABELS`, replace
   ```rust
   pub const CHAIN_LABELS: [(&str, &str); 9] = [
   ```
   with `pub const CHAIN_LABELS: [(&str, &str); 8] = [`, and
   ```rust
       ("geometry", "Geometry"),
       ("shaft_joint", "Shaft joints"),
       ("mass", "Mass"),
   ```
   with
   ```rust
       ("geometry", "Geometry"),
       ("mass", "Mass"),
   ```
   (the `PACKAGE_LABELS` line of the same text stays). Expected: `the_inputs_and_results_sheets_carry_every_shaft_joint_row` FAILS (`no chain shaft_joint`) and `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` FAILS (`left: 3`, `right: 4`: one heading fewer); the Auto tests pass. Undo it.
5. `magcoupling-rs/src/engine/clamps.rs`: replace `pub const SECTION: [&str; 2] = ["clamps.", "materials.screws."];` with `pub const SECTION: [&str; 3] = ["clamps.", "materials.screws.", "shaft_joint."];` (the pause hides the joints too). Expected: `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` FAILS (`shaft_joint.hub_interference_min_um: "Inner hub interference, minimum" is not collected`); the other four pass. Undo it.
6. `magcoupling-rs/src/engine/sizing.rs`, `is_valid`: replace
   ```rust
           && results.metal.torque_hot_low_Nm.is_finite()
   }
   ```
   with
   ```rust
           && results.metal.torque_hot_low_Nm.is_finite()
           && design.shaft_joint.safety_factor < 4.0
   }
   ```
   (Auto reads a shaft joint input). Expected: `auto_reads_no_shaft_joint_input_and_both_joints_hold_at_its_answer` FAILS (`expected a solution, got Ok(NotReachable { best: None })`, at the safety factor's slider top); `auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there` and the three lib tests pass. Undo it.

Run the `git diff --quiet` command above once more. Expected: `clean`.

- [ ] **Step 13: The texts the key move leaves stale (`clamps.rs`, `tests/explain.rs`)**

Spec 4.5's last bullet and 7.17. The `PAUSED` doc said the keyway drawing is hidden with the section; it now names the drawn key as the shaft joints' and records that un-pausing would show two joint methods (risk R-13; Task 9's README edit cites this doc).

In `magcoupling-rs/src/engine/clamps.rs`, replace:

```rust
/// other method"). While it is, the panel leaves out every input and result of [`SECTION`]:
/// the slotted clamp, the shaft key and the adapter-joint screws (their input groups, the Clamp
/// view, the dashboard's clamp rows and their check guide, the results table's clamp rows, the
/// clamp assumptions, the keyway in the geometry drawing and its boss design check, the
/// `clamp_preload` teaching note and its Start here step), and the clamp checks count toward
/// no verdict. The engine still computes the section, and design files, share links and the
/// exports keep every clamp input and result.
///
/// Set false to bring the section back: no other code edit is needed.
```

with:

```rust
/// other method"). While it is, the panel leaves out every input and result of [`SECTION`]:
/// the slotted clamp, its shaft key inputs and the adapter-joint screws (their input groups,
/// the Clamp view, the dashboard's clamp rows and their check guide, the results table's clamp
/// rows, the clamp assumptions, the geometry drawing's clamp-boss design check, the
/// `clamp_preload` teaching note and its Start here step), and the clamp checks count toward
/// no verdict. The engine still computes the section, and design files, share links and the
/// exports keep every clamp input and result. The geometry drawing's key is the shaft joints'
/// (`shaft_joint.key_width_mm`), drawn in both states.
///
/// BL-064 keeps the section paused (owner decision 6A, 2026-10-17): both rotors are held by
/// keyed shrink fits, which `engine::shaft_joint` checks. Set false to bring the section back:
/// no other code edit is needed, and the panel then shows the clamp beside the shrink-fit
/// check, two joint methods for one joint.
```

In `magcoupling-rs/tests/explain.rs`, `INERT_INPUTS`, replace:

```rust
        "only the drawing reads it (Python drawing.py); the key pressure divides by the contact height",
```

with:

```rust
        "the paused shaft-clamp section's key width: only its clamp drawing reads it (Python drawing.py), not the geometry view, whose key takes the shaft joints' own width (BL-064); the key pressure divides by the contact height",
```

(The entry stays: no result reads `clamps.key_width_mm`. The clamp tab's drawing, `gui/clamp_drawing.rs`, still does, so spec 7.17's draft "nothing reads it" is not used: plan deviation PD-16.)

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain no_input_moves_a_result_the_graph_says_is_independent_of_it`
Expected: `test result: ok. 1 passed` (the reason is text; `check_traceability`, over every input, still finds `clamps.key_width_mm` moves no result, as `INERT_INPUTS` says).

- [ ] **Step 14: Docs**

Each edit replaces the exact text in the first block (one occurrence in the file) with the second.

`magcoupling-rs/README.md`, seven edits (Task 2 Step 11 (j) to (l) makes the three phrases its placement and its export test leave stale: ruling C-2, A-6).

(a) The shaft-clamp pause bullet (around :461), what the switch hides. Old:
```text
the slotted clamp, the shaft key and the adapter-joint screws
```
New:
```text
the slotted clamp, its shaft key inputs and the adapter-joint screws
```

(b) The same bullet, the drawing. Old:
```text
no key in the geometry drawing and no clamp-boss design check
```
New:
```text
no clamp-boss design check in the geometry drawing (its key is the shaft joints', drawn in both states: BL-064)
```

(c) The Tests table's `tests/sizing.rs` row (around :662), at its end. Old:
```text
Auto on length at the defaults still sizes for the hot minimum alone (13.77 mm), and the gearbox check is red there.
```
New:
```text
Auto on length at the defaults still sizes for the hot minimum alone (13.77 mm), and the gearbox check is red there. BL-064 (decision 21): Auto reads no shaft joint input (each at either end of its slider gives the same length and hot-low torque, bit for bit), and both joints hold at its answer (the inner hub 10.26 N·m and the cup 16.90 N·m against 8.37 N·m).
```

(d) The `src/gui/panel.rs` row (around :673), the guard. Old:
```text
a clamp input row, the Shaft clamps group and the clamp friction assumption)
```
New:
```text
a clamp input row, the Shaft clamps group and the clamp friction assumption; while paused it reaches the label and help of every shaft-joint input and result, their Shrink fit and Fitting temperatures sections and the Shaft joints heading, BL-064)
```

(e) The `src/gui/geometry.rs` row (around :674), the end view. Old:
```text
the key from where it crosses the bore
```
New:
```text
the key, the shaft joints' 3 mm, from where it crosses the bore
```

(f) The same row, at its end. Old:
```text
the key and the clamp-boss design check only while the shaft-clamp section is not paused
```
New:
```text
the key in both states of the shaft-clamp pause, at the shaft joints' key width (BL-064: each end of its slider; one typed wider than the bore from the axis; one not a number not drawn and counted; none at a keyway depth of 0), never at the paused clamp's, and the clamp-boss design check only while the shaft-clamp section is not paused
```

(g) The `src/gui/spreadsheet.rs`, `src/gui/xlsx.rs` row (around :678). Old:
```text
and Rust-only for its cell, and the Results sheet's row carries the same level (BL-059);
```
New:
```text
and Rust-only for its cell, and the Results sheet's row carries the same level (BL-059); the shaft joints' 15 inputs and 24 results each on one row with no cell (BL-064): the inputs where the inputs side draws them in both orders, the four assumptions flagged, the 23 numbers under the Shaft joints chain between Geometry and Mass;
```

(The same row's "its 18 dashboard rows carry the gearbox check between the clamp's and the space claim's" is Task 5's, with its Summary tests: Task 5 Step 6a (25) to (27).)

`docs/ai/03-structure.yaml`, two edits in the `gui:` entry (an unquoted scalar: no ": " added).

(a) Old:
```text
geometry(inputs, results, clamps_paused: no key and no clamp-boss design check while paused)
```
New:
```text
geometry(inputs, results, clamps_paused: no clamp-boss design check while paused; the key, at shaft_joint.key_width_mm, is drawn in both states and not at a keyway depth of 0 or less, BL-064)
```

(b) Old:
```text
clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement)
```
New:
```text
clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), shaft_joints_off_default (BL-064: every shaft joint input one slider step off its default) and cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf)
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`
Expected: `ok`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count`
Expected: `test result: ok. 2 passed` (no count those tests read moved; without `--features app` the dashboard one is compiled out and 1 passes).

No `docs/ai/02-system.yaml` line moves: its pause invariant names `geometry_ui (geometry)` among the code that takes the pause, still true (the clamp-boss check).

- [ ] **Step 15: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every `test result: ok`; the lib Task 3's count + 3 passed, 6 ignored (the new `the_key_takes_the_shaft_joints_width_and_never_the_paused_clamp_s`, `the_exports_carry_every_shaft_joint_input_and_result_with_no_cell`, `the_inputs_and_results_sheets_carry_every_shaft_joint_row`; the renamed key test counts once); `sizing` Task 3's count + 1. Measured at plan time: lib 889 -> 892, sizing 44 -> 45, every other binary unchanged.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every `test result: ok`; the lib count unchanged (the drawing and export tests are behind the `gui` feature), `sizing` + 1. Measured: lib 231 unchanged, sizing 44 -> 45.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: each ends `Finished`, with no warning.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`
Expected: no output, exit 0.

Run: `git status --short`
Expected:
```text
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/clamps.rs
 M magcoupling-rs/src/gui/geometry.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/results_table.rs
 M magcoupling-rs/src/gui/spreadsheet.rs
 M magcoupling-rs/src/gui/test_support.rs
 M magcoupling-rs/tests/explain.rs
 M magcoupling-rs/tests/sizing.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 16: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T4.txt` with the Write tool:
```text
feat(magcoupler): the drawn key is the shaft joints', in both states; the exports carry the joints (BL-064 T4)

The geometry view's key now takes the shaft joints' key width (shaft_joint.key_width_mm,
the DIN 6885-1 3 x 3 key) and is drawn whether or not the paused shaft-clamp section
shows, and not at all without a keyway, where the joints have no key term; the paused
clamp's key width moves nothing in the drawing. The PAUSED doc and the inert-input reason
of the clamp's key width say so.

Tests: the key in both states at each end of its slider, typed wider than the bore, not a
number and with no keyway; the CSV, JSON and spreadsheet exports carry the 15 inputs and 24 results
with no cell (the inputs where the inputs side draws them, the four assumptions flagged,
the 23 numbers under the Shaft joints chain between Geometry and Mass, NaN and +inf as
text); the clamp-name guard reaches every shaft-joint label and help while paused; Auto
reads no shaft joint input and both joints hold at its answer. Moved: the key outline
(3 mm: [6.7, -1.5] and sqrt 22.75), the key test renamed (a key in both states), the
BL-059 Auto test's solve shared.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add docs/ai/03-structure.yaml magcoupling-rs/README.md magcoupling-rs/src/engine/clamps.rs magcoupling-rs/src/gui/geometry.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/tests/explain.rs magcoupling-rs/tests/sizing.rs
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T4.txt
git status --short
```
Expected: one commit of 10 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 4 moves** (each confirmed by running in the prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::geometry::tests::the_end_view_draws_every_part_from_the_results` | key outline `[6.7, -2.0]` and √21 -> `[6.7, -1.5]` and √22.75, with the 3 mm default asserted | the key is the shaft joints' 3 mm key, not the clamp's 4 mm one (spec 7.13) |
| `gui::geometry::tests::the_key_and_the_clamp_boss_check_are_drawn_only_while_the_clamp_section_shows` | renamed `the_key_is_drawn_in_both_states_and_the_clamp_boss_check_only_while_the_clamp_section_shows`; keys (unpaused, paused) (1, 0) -> (1, 1); the notes (3, 2) unchanged | the key is drawn in both states; the clamp-boss check still follows the pause (spec 5.4, review 2 SF-11) |
| `tests/sizing.rs` `auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there` | its solve and 13.77 mm pin move into `auto_on_length_at_the_defaults()`; its assertions unchanged | DRY: the new Auto test solves the same design |
| `gui::panel::tests::every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` | extended: while paused the collected texts reach every shaft-joint input's and result's label and help and the four headings | spec 7.14 made explicit, not vacuous |
| `tests/explain.rs` `INERT_INPUTS` (data of `check_traceability`) | the reason for `clamps.key_width_mm` reworded | the geometry view no longer reads it (spec 7.17; plan deviation PD-16) |

Moved by Task 5, not here: `gui::spreadsheet::tests::the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge`. Task 5 renames it `the_summary_lists_nineteen_dashboard_rows_with_the_gearbox_badge`, rebuilt over a shared `assert_summary_check_row`, and adds `the_summary_lists_the_shaft_joints_row_with_its_badge_and_the_stored_3d_note`.

---

### Task 5: The shaft joints on the dashboard: the row, `CHECKS`, verdict levels, greying, the stored-3D label and the guide

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 5.1 and 5.2 (the guide, with `learn: Learn::None("No note covers the shaft joints yet.")` until Task 7
flips it: the interim link, plan deviation PD-25), 7 items 4-8, 7.15's Summary row (plan deviation PD-13 hands it here)
and 7.17's counts that move with the row and the guide.

Line numbers are at Task 2's commit. Tasks 3 and 4 add lines to four of the files this task edits; none of their edits
touches an Old text below, so trust the code and keep the intent:
- Task 3 adds the `section` closure and its three lines to `result_groups.rs`'s
  `the_headline_comes_first_then_the_chains_then_the_other_results` (Step 1f flips exactly those three lines, as Task 3's
  Interfaces name them).
- Task 4 inserts tests into `spreadsheet.rs` after the Summary test this task replaces (Step 1e replaces that test
  whole, from its `#[test]` to its closing brace, so Task 4's tests after it are not in the Old text), extends
  `panel.rs`'s clamp-name guard at the end of its paused branch (Step 1d edits the design loop above it), and appends
  two designs to `test_support.rs` after `clamp_at_its_limit` (Step 1a inserts before it).

Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block below is
rustfmt's layout. Before you start: Task 4's commit is in and both suites are green.

**What this task does not do:** the note (Task 7 flips the guide's Why link to `Learn::Note("shaft_joint")`; the
assertion it flips is in Step 1c's `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`, whose guide variable is
`joint`), and the G-3 review of the guide's label, Why, Problem sentences and `STORED_3D_NOTE`'s new sentence (Task 8).
Task 6 adds the fix-coverage designs and the held-back fix tests.

**Files:**
- Modify: `magcoupling-rs/src/gui/dashboard.rs`: module doc `:5-17`; the `shaft_joint` import `:48`; `DASHBOARD`
  `:155-187`; `END_EFFECT_ROWS` `:188-203`; `STORED_3D_ROWS` `:205-211`; `STORED_3D_NOTE` `:218`; the
  `STORED_3D_INPUTS` doc `:220-223`; `CHECKS` doc and array `:257-295`; `verdict_level` `:306-388`; the `dashboard_ui`
  doc `:684`. Tests: imports `:1012`, and the tests at `:1022`, `:1112`, `:1360`, `:1380`, `:1497-1512`, `:1622-1630`,
  `:1660-1724`, a new test after `:1776`, `:2080`, `:2146`, `:2326-2341`.
- Modify: `magcoupling-rs/src/gui/guidance.rs`: module doc `:16-31`; imports `:43-52`; the `Fixes::PerTerm` doc
  `:114`; the check constants `:146`; after `CLAMP_GUIDE` `:216`; `CHECK_GUIDES` `:309-320`; `ROW_LEARN` `:363`;
  `cure_margin_texts` `:1379-1400`; the `design_status_text` doc `:1502-1505`; `quoted_limit` and `quoted_load`
  `:1575-1601`; after `clamp_problem` `:1723`; `gearbox_problem` `:1732-1744`. Tests: imports `:2338-2347`, after
  `designs()` `:2725`, `possible_fixes` `:2739`, `every_text` `:2771`, `:3028-3067` and two new tests after it,
  `terms` `:3255-3276`, `every_fix_moves_its_check_toward_passing_over_its_slider` `:3435`,
  `two_failing_checks_that_pull_one_input_apart_say_so` `:4039-4123`, `each_problem_is_some_exactly_when_its_check_fails`
  `:4298`, `the_problems_quote_numbers_from_the_results` `:4361-4414`, new tests after
  `a_check_that_does_not_apply_gives_no_problem` `:5871`, the status tests `:5883-6025` and a new one after them.
- Modify: `magcoupling-rs/src/gui/test_support.rs`: `passing_design`'s doc `:25-35`; a new design before `:73`.
- Modify (tests): `magcoupling-rs/src/gui/panel.rs` `:1622`, `:3138-3160`, the clamp-name guard's design loop
  `:4297-4300`, `:8839-8843`; `magcoupling-rs/src/gui/spreadsheet.rs` imports `:579` and the Summary test `:1194-1258`;
  `magcoupling-rs/src/gui/result_groups.rs` (Task 3's three lines in
  `the_headline_comes_first_then_the_chains_then_the_other_results`).
- Modify (docs): `magcoupling-rs/README.md`, `docs/ai/02-system.yaml` `:107-109`, `docs/ai/03-structure.yaml` `:68`.
- Test: the six `gui` files above (`--features app`); `tests/explain.rs`'s `the_docs_count_the_dashboard_rows_and_the_checks`
  holds the README and `03-structure.yaml` counts to the code (no edit). The engine-only suite runs unchanged.

**Interfaces:**
- Consumes (Task 2): `crate::engine::shaft_joint::{self, Cause, CAUSES, HubFitLimit, JointTerms, NoWall, INNER_HUB, CUP,
  SHAFT_JOINTS_PASS, SHAFT_JOINT_UNKNOWN, UNKNOWN_NOT_A_NUMBER, UNKNOWN_NO_WALL, UNKNOWN_BAND_REVERSED,
  UNKNOWN_SAFETY_FACTOR, FIT_SLIPS, BORE_OVER_ITS_LIMIT, CANNOT_BE_FITTED, KEY_OVER_ITS_LIMIT, KEY_HAS_NO_BEARING_AREA, joints, failing, hub_fit_allowed}`;
  `JointTerms`' fields (`name`, `no_wall`, `band_reversed`, `effective_max_um`, `holding_Nm`, `bore_stress_MPa`,
  `key_pressure_MPa`, `needed_um`, `elastic_max_um`, `fittable_max_um`, `fit_temp_C`, `fit_allowed_C`, `key_bearing_mm`,
  `key_contact_mm`, `required_Nm`, `stress_limit_MPa`) and methods (`determinate`, `fails`, `margin`);
  `crate::engine::api::shaft_joint_links`; the 24 `shaft_joint.*` results and 15 inputs. Task 1:
  `temperature.demag.inner_onset_single_ring_C`. The engine's `temperature::CURE_MARGIN_MIN_C`. Task 3: the `section`
  closure in `the_headline_comes_first_then_the_chains_then_the_other_results`.
- Produces:
  - `dashboard`: `DASHBOARD: [(&str, Option<&str>); 19]` with `("shaft_joint.check", Some("shaft_joint.check"))` at
    index 15 (`paths[15..]` = shaft joints, gearbox, space claim, end effect); `END_EFFECT_ROWS: [&str; 9]`;
    `STORED_3D_ROWS: [&str; 4]` (the joint row last); `CHECKS: [&str; 32]` (the joint check last); `verdict_level`
    Good for `SHAFT_JOINTS_PASS`, Caution for the `SHAFT_JOINT_UNKNOWN` prefix (every unknown form, the safety
    factor's of ruling C-4 included), Bad for the five failing prefixes;
    `STORED_3D_NOTE` with its new sentence.
  - `guidance` (private to the module and its tests): `const SHAFT_JOINT`, `const SHAFT_JOINT_GUIDE` (label "Shaft
    joints", the spec's Why, `Fixes::PerTerm(shaft_joint_fixes)`, `learn: Learn::None("No note covers the shaft joints
    yet.")`), `CHECK_GUIDES: [CheckGuide; 10]` with it at index 5, its `ROW_LEARN` entry
    `(SHAFT_JOINT, SHAFT_JOINT_GUIDE.learn)`; the input constants `HUB_MIN`, `HUB_MAX`, `CUP_MIN`, `CUP_MAX`,
    `KEY_CONTACT`, `WHEEL_SHAFT`, `GEARBOX_SHAFT`, `HUB_LENGTH`, `BOSS_LENGTH`; `struct JointSide` and
    `JOINT_SIDES: [JointSide; 2]` (inner hub, cup: the engine's order); `shaft_joints`, `joint_side`,
    `shaft_joint_fixes`, `reverses_the_band`; `shaft_joint_problem` and its builders `term_sentence`, `fitting_source`,
    `onset_texts`, `key_sentence`, `window_sentence`, `window_quotes`, `not_judged_sentences`, `apart_from`;
    `SHAFT_JOINT_NOT_A_NUMBER`, `SHAFT_JOINT_FALLBACK`; the shared rounding helpers `rounded_up`, `rounded_down`,
    `shown_decimals` (`quoted_limit`, `quoted_load` and `cure_margin_texts` read them: DRY, plan deviation PD-24).
  - Tests: `test_support::only_the_joint_fails()` (`passing_design()` with the inner hub's minimum at 8 µm); in
    `guidance`'s tests the `terms()` arm `SHAFT_JOINT` (eight terms named by `joint_term`: "inner hub slips", "inner hub
    bore stress", "inner hub fitting", "inner hub key", the same for "cup"), `shaft_joint_problem_designs()` (one design
    per Problem form) and `assert_in_order`, which Task 6 reads; `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`
    (Task 7 flips its learn assertion); in `spreadsheet`'s tests `assert_summary_check_row`.

- [ ] **Step 1: Write the failing tests (test support, the new tests, every pinned test that moves)**

**1a. `magcoupling-rs/src/gui/test_support.rs`.**

(1) `passing_design`'s doc: it passes all 10 guided checks, the shaft joints among them (spec 5.4).

At `:25`, replace

```rust
/// A design that passes all 9 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
```

with

```rust
/// A design that passes all 10 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
```

At `:31`, replace

```rust
/// workbook's gearbox reference, `model.gearbox_reference_Nm`, its only other reader). The
/// guidance tests break one check of it at a time, so a fix that turns another check red
/// shows.
```

with

```rust
/// workbook's gearbox reference, `model.gearbox_reference_Nm`, its only other reader). Its
/// shaft joints pass at the example's bands (BL-064: the inner hub holds about 9.48 N·m and
/// the cup 16.90 N·m against 6.72 N·m; the inner hub would need about 93.3 °C to fit, against
/// 100 °C allowed). The guidance tests break one check of it at a time, so a fix that turns
/// another check red shows.
```

(2) The design that fails the shaft joints alone (spec 5.4, `only_the_joint_fails()`), inserted before
`clamp_at_its_limit`'s doc:

At `:72`, replace

```rust

/// The defaults at the clamp's limit: the clamp's safety factor raised until its capacity only
```

with

```rust

/// [`passing_design`] with the inner hub's minimum interference at 8 µm: smoothing and the
/// hot end take all of it (its effective interference at the loosest is floored at 0), so the
/// inner hub slips, and nothing else fails (BL-064).
pub(crate) fn only_the_joint_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    inputs.shaft_joint.hub_interference_min_um = 8.0;
    inputs
}

/// The defaults at the clamp's limit: the clamp's safety factor raised until its capacity only
```

**1b. `magcoupling-rs/src/gui/dashboard.rs`, `mod tests`.**

(1) The imports:

At `:1012`, replace

```rust
        passing_design, primary_button, short_magnets, sized_frame, text_center, text_color,
        text_rect, tooltips_at_once, with_poles,
```

with

```rust
        only_the_joint_fails, passing_design, primary_button, short_magnets, sized_frame,
        text_center, text_color, text_rect, tooltips_at_once, with_poles,
```

(2) `the_dashboard_starts_with_the_headline_in_order`: the joint row is first past the headline (spec 5.1).

At `:1024`, replace

```rust
            [
                "gearbox.cold_high_check",
```

with

```rust
            [
                "shaft_joint.check",
                "gearbox.cold_high_check",
```

(3) `every_verdict_each_check_gives_is_classified` (spec 7.5): every text form the check writes, each reached as the
badge by a design from the defaults, so no form is classified from a constant alone (the key's text at a 0.3 mm keyway,
review 2 SF-9). Insert after the gearbox's five assertions:

At `:1112`, replace

```rust
        assert_eq!(gearbox(0.6, f64::NAN), Some(Caution));
```

with

```rust
        assert_eq!(gearbox(0.6, f64::NAN), Some(Caution));
        // BL-064 (spec 7.5): every text form the shaft joints check writes, each reached as the
        // badge by a design from the defaults, and its level: the pass; the five failing causes
        // (the key's two at a 0.3 mm keyway and at a contact height typed 0); the four unknown
        // forms (a value that is not a number, a safety factor typed 0 (ruling C-4), no wall
        // around a bore, a reversed band). So no form is classified from a constant alone.
        type Edit = fn(&mut DesignInputs);
        let joint_forms: [(Edit, &str, Level); 10] = [
            (|_| {}, shaft_joint::SHAFT_JOINTS_PASS, Good),
            (
                |i| i.shaft_joint.hub_interference_min_um = 8.0,
                shaft_joint::FIT_SLIPS,
                Bad,
            ),
            (
                |i| i.shaft_joint.hub_interference_max_um = 26.0,
                shaft_joint::BORE_OVER_ITS_LIMIT,
                Bad,
            ),
            (
                |i| i.shaft_joint.wheel_shaft_fit_C = 20.0,
                shaft_joint::CANNOT_BE_FITTED,
                Bad,
            ),
            (
                |i| i.coupling.keyway_depth_mm = 0.3,
                shaft_joint::KEY_OVER_ITS_LIMIT,
                Bad,
            ),
            (
                |i| i.shaft_joint.key_contact_mm = 0.0,
                shaft_joint::KEY_HAS_NO_BEARING_AREA,
                Bad,
            ),
            (
                |i| i.temperature.adhesive.selected = 0,
                shaft_joint::UNKNOWN_NOT_A_NUMBER,
                Caution,
            ),
            (
                |i| i.shaft_joint.safety_factor = 0.0,
                shaft_joint::UNKNOWN_SAFETY_FACTOR,
                Caution,
            ),
            (
                |i| {
                    i.coupling.bore_mm = 16.0;
                    i.metal.boss_od_mm = 12.0;
                    i.shaft_joint.hub_interference_min_um = 15.0;
                },
                shaft_joint::UNKNOWN_NO_WALL,
                Caution,
            ),
            (
                |i| i.shaft_joint.hub_interference_min_um = 20.0,
                shaft_joint::UNKNOWN_BAND_REVERSED,
                Caution,
            ),
        ];
        for (edit, start, want) in joint_forms {
            let results = compute_all(&design(edit));
            let text = &results.shaft_joint.check;
            assert!(text.starts_with(start), "{text:?} starts with {start:?}");
            assert_eq!(
                verdict_level("shaft_joint.check", text),
                Some(want),
                "{text:?}"
            );
            assert_eq!(
                check_level(&results, "shaft_joint.check"),
                Some(want),
                "{text:?}"
            );
        }
```

(4) `every_verdict_of_every_other_check_is_classified`: no arm reads another check's text.

At `:1360`, replace

```rust
        );
        assert_eq!(verdict_level("warnings.no_such_rule", "text"), None);
```

with

```rust
        );
        assert_eq!(
            verdict_level("gearbox.cold_high_check", shaft_joint::SHAFT_JOINTS_PASS),
            None
        );
        assert_eq!(
            verdict_level(
                "shaft_joint.check",
                crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING
            ),
            None
        );
        assert_eq!(verdict_level("shaft_joint.check", "OK"), None);
        assert_eq!(verdict_level("warnings.no_such_rule", "text"), None);
```

(5) `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`: a result named `check` is a check too
(the binding path `shaft_joint.check` ends in `check` but not in `_check`, so the guard missed it; plan deviation
PD-23, ruling C-8).

At `:1380`, replace

```rust
        // A new check (a text named *_check, *_screen, *verdict or *reading, or a warning)
        // fails here until it is listed and classified. The clamp table's and the sweeps' rows
        // rate candidates, not the design.
        for row in &rows {
            let name = row.path.rsplit('.').next().unwrap();
            let named = name.ends_with("_check")
```

with

```rust
        // A new check (a text named check, *_check, *_screen, *verdict or *reading, or a
        // warning) fails here until it is listed and classified. The clamp table's and the
        // sweeps' rows rate candidates, not the design.
        for row in &rows {
            let name = row.path.rsplit('.').next().unwrap();
            let named = name == "check"
                || name.ends_with("_check")
```

(6) `the_end_effect_rows_are_the_dashboard_rows_that_move_with_the_end_effect_coefficient` (spec 5.1, 7.6): two rows past
the headline now. Then the stored-3D membership test is renamed (spec 7.6: the joint row is no headline row) and says
why the joint row is in it:

At `:1497`, replace

```rust
        // The headline's rows, and one row past it: the gearbox check, computed from cold-high.
```

with

```rust
        // The headline's rows, and two rows past it: the shaft joints and the gearbox checks,
        // both computed from cold-high.
```

At `:1502`, replace

```rust
        assert_eq!(past_the_headline, ["gearbox.cold_high_check"]);
    }

    #[test]
    fn the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs() {
```

with

```rust
        assert_eq!(
            past_the_headline,
            ["shaft_joint.check", "gearbox.cold_high_check"]
        );
    }

    #[test]
    fn the_stored_3d_rows_are_the_dashboard_rows_that_move_with_the_stored_3d_inputs() {
        // Owner R-15: the shaft joints row is one of them, through the inner ring's single-ring
        // onset that caps the bonded inner hub's fitting temperature (it moves with the
        // single-ring reverse field at its slider's top, back iron 1).
```

(7) `the_default_dashboard_has_badges_markers_and_the_3d_rows` (spec 7.6): the joint row is green with no marker; four
stored-3D lines, each hover carrying the label; the last stored-3D line is the joint row in both clamp states. Then a
new test: the label is drawn once, below the joint row's label and above the gearbox row's, in both states (spec 7.6's
"a new assertion there, or in a dashboard test").

At `:1622`, replace

```rust
        assert_eq!(line("model.end_effect_check").level, Some(Level::Good));
```

with

```rust
        assert_eq!(line("model.end_effect_check").level, Some(Level::Good));
        // BL-064: the shaft joints pass at the example design, Rust-only.
        let joint = line("shaft_joint.check");
        assert_eq!(joint.value, crate::engine::shaft_joint::SHAFT_JOINTS_PASS);
        assert_eq!(joint.level, Some(Level::Good));
        assert_eq!(joint.marker, "", "Rust-only: no cell");
```

At `:1628`, replace

```rust
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 3);
```

with

```rust
        assert_eq!(lines.iter().filter(|l| l.stored_3d).count(), 4);
        // The stored-3D label is drawn after the last stored-3D row shown: the shaft joints row,
        // whether the clamp section shows or not (owner R-15).
        for paused in [true, false] {
            let shown = shown_lines(&compute_all(&DesignInputs::default()), paused);
            let last = shown.iter().rev().find(|l| l.stored_3d).map(|l| l.path);
            assert_eq!(last, Some("shaft_joint.check"), "paused {paused}");
        }
    }

    #[test]
    fn the_stored_3d_label_is_drawn_once_under_the_shaft_joints_row() {
        // Owner R-15: one label, after the last stored-3D row, now the shaft joints row: below
        // its label and above the gearbox row's, whether the clamp section shows or not.
        let ctx = egui::Context::default();
        for paused in [true, false] {
            let output = dashboard_frame_with(&ctx, &DesignInputs::default(), paused);
            let texts = drawn_texts(&output);
            let count = texts.iter().filter(|t| *t == STORED_3D_LABEL).count();
            assert_eq!(count, 1, "paused {paused}");
            let y = |text: &str| {
                text_rect(&output, text)
                    .unwrap_or_else(|| panic!("{text:?} drawn"))
                    .center()
                    .y
            };
            let label = y(STORED_3D_LABEL);
            assert!(label > y(label_for("shaft_joint.check")), "paused {paused}");
            assert!(
                label < y(label_for("gearbox.cold_high_check")),
                "paused {paused}"
            );
        }
```

(8) `out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter` (spec 7.6): four checks
among the greyed rows; in range the joint is green and not greyed.

At `:1663`, replace

```rust
        // so the three checks among the greyed rows give no badge and are not listed as failing.
```

with

```rust
        // so the four checks among the greyed rows give no badge and are not listed as failing.
```

At `:1675`, replace

```rust
                "clamps.recommended",
                "gearbox.cold_high_check"
```

with

```rust
                "clamps.recommended",
                "shaft_joint.check",
                "gearbox.cold_high_check"
```

At `:1681`, replace

```rust
            // The verdict has a level of its own (red, green and green here): the rule drops it.
```

with

```rust
            // The verdict has a level of its own (red, then green three times here): the rule
            // drops it.
```

At `:1716`, replace

```rust
        // gearbox check red.
```

with

```rust
        // shaft joints green, the gearbox check red.
```

At `:1723`, replace

```rust
        );
        assert!(!greyed_by_end_effect(&defaults, "gearbox.cold_high_check"));
```

with

```rust
        );
        assert!(!greyed_by_end_effect(&defaults, "shaft_joint.check"));
        assert_eq!(
            check_level(&defaults, "shaft_joint.check"),
            Some(Level::Good)
        );
        assert!(!greyed_by_end_effect(&defaults, "gearbox.cold_high_check"));
```

(9) New test, after `out_of_range_end_effect_greys_the_gearbox_check_whatever_its_text_reads` (spec 5.1, 7.6; review 2
SF-13): at f_end < 0 the row greys whatever its text reads, a bore stress above its limit included, keeps its stored-3D
note beside the greyed one, and the status line counts it neither way.

At `:1776`, replace

```rust
    }

    #[test]
    fn a_design_past_the_space_claim_shows_the_overshoot_in_red() {
```

with

```rust
    }

    #[test]
    fn out_of_range_end_effect_greys_the_shaft_joint_check_whatever_its_text_reads() {
        // BL-064 (spec 5.1, 7.6). The check reads cold-high, which comes from the pull-out: on
        // short magnets it is below zero, so the engine's text reads a pass from a meaningless
        // number. The whole row greys, so a bore stress above its limit, which does not read
        // cold-high, is hidden there too (review 2, SF-13: one row cannot grey in part; the
        // end-effect check fails there under its banner). Either way the row is greyed without
        // a badge, keeps its stored-3D note beside the greyed one, has no level in the
        // dashboard, the results table or its failing filter, its guide shows no Problem, and
        // the status line counts it neither way.
        use crate::engine::shaft_joint::{BORE_OVER_ITS_LIMIT, SHAFT_JOINTS_PASS};
        use crate::gui::results_table::{entry_index, entry_level, table_entries};
        let joint = "shaft_joint.check";
        let guide = guidance::guide(joint).expect("the shaft joints check's guide");
        let entry = &table_entries()[entry_index(joint).expect("the shaft joints check's row")];
        let mut overstressed = short_magnets();
        overstressed.shaft_joint.hub_interference_max_um = 26.0;
        let mut statuses = Vec::new();
        for (inputs, text) in [
            (short_magnets(), SHAFT_JOINTS_PASS),
            (overstressed, BORE_OVER_ITS_LIMIT),
        ] {
            let results = compute_all(&inputs);
            assert!(end_effect_out_of_range(&results).is_some());
            assert!(results.metal.torque_cold_high_Nm < 0.0);
            let verdict = &results.shaft_joint.check;
            assert!(verdict.starts_with(text), "{verdict}");
            assert!(verdict_level(joint, verdict).is_some(), "{verdict}");
            assert!(greyed_by_end_effect(&results, joint));
            assert_eq!(check_level(&results, joint), None);
            let line = dashboard_lines(&results)
                .into_iter()
                .find(|l| l.path == joint)
                .expect("the shaft joints row");
            assert!(line.greyed && line.stored_3d);
            assert_eq!(line.level, None);
            let hover = line.tooltip(false);
            assert!(hover.contains(END_EFFECT_OUT_OF_RANGE), "{hover}");
            assert!(hover.contains(STORED_3D_LABEL), "{hover}");
            assert_eq!(entry_level(&results, entry), None);
            assert!(
                failing_checks(&results, false)
                    .iter()
                    .all(|&(path, _)| path != joint)
            );
            assert_eq!(guidance::level(guide, &inputs, &results), Level::Good);
            assert_eq!((guide.problem)(&inputs, &results), None);
            statuses.push(design_status(&inputs, &results, false));
        }
        assert_eq!(statuses[0], statuses[1]);
    }

    #[test]
    fn a_design_past_the_space_claim_shows_the_overshoot_in_red() {
```

(10) `while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard` (spec 7.7):

At `:2080`, replace

```rust
            "Your design: passes all 8 checks"
```

with

```rust
            "Your design: passes all 9 checks"
```

(11) `a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it`: two joint designs join its list (red, amber).

At `:2146`, replace

```rust
            design(|i| i.coupling.backiron = 0),
        ] {
```

with

```rust
            design(|i| i.coupling.backiron = 0),
            only_the_joint_fails(),
            design(|i| i.shaft_joint.hub_interference_min_um = 20.0),
        ] {
```

(12) `the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour` (spec 7.7): the joint alone is one red;
a reversed band is one amber.

At `:2326`, replace

```rust
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
```

with

```rust
        passing_hot_day.temperature.duty = hot_day().temperature.duty;
        let mut passing_reversed = passing_design();
        passing_reversed.shaft_joint.hub_interference_min_um = 20.0;
```

At `:2339`, replace

```rust
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
```

with

```rust
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's minimum at 8 µm: the shaft joints alone.
            (only_the_joint_fails(), 1, 0, Level::Bad),
            // The passing design with the inner hub's band reversed: the shaft joints ask for a
            // look.
            (passing_reversed, 0, 1, Level::Caution),
```

**1c. `magcoupling-rs/src/gui/guidance.rs`, `mod tests`.**

(1) The imports:

At `:2338`, replace

```rust
    use crate::engine::model::ring_fits;
```

with

```rust
    use crate::engine::model::ring_fits;
    use crate::engine::shaft_joint::CAUSES;
```

At `:2346`, replace

```rust
        only_the_gearbox_fails, passing_design, short_magnets, sized_frame, small_boss, with_poles,
```

with

```rust
        only_the_gearbox_fails, only_the_joint_fails, passing_design, short_magnets, sized_frame,
        small_boss, with_poles,
```

(2) One design per sentence form of the joint's Problem (spec 5.2), after `designs()`. Few are in `designs()`; their
Problems join `every_text()`, so the plain-words, glyph, ring-size and protect guards run over every form (spec 7.8).
The inner hub's Q ≥ 1 form is reached only off the sliders (bore 21 mm, spec 5.2: "only a typed value reaches it").

At `:2725`, replace

```rust

    /// The Problem of `check` for `inputs`.
```

with

```rust

    /// One design per sentence form of the shaft joints' Problem (spec 5.2), each the passing
    /// design with joint inputs moved: (its name, the design, words its Problem holds, the
    /// form's own). Few of them are in [`designs`]; their Problems join [`every_text`].
    fn shaft_joint_problem_designs() -> Vec<(&'static str, DesignInputs, &'static str)> {
        vec![
            (
                "inner hub slips",
                only_the_joint_fails(),
                "At its loosest the inner hub holds",
            ),
            (
                "cup slips",
                breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0),
                "At its loosest the cup holds",
            ),
            (
                "inner hub overstressed",
                breaking(|i| i.shaft_joint.hub_interference_max_um = 26.0),
                "the inner hub's bore stress is",
            ),
            (
                "cup overstressed",
                breaking(|i| i.shaft_joint.cup_interference_max_um = 26.0),
                "the cup's bore stress is",
            ),
            (
                "uncooled wheel shaft",
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
                "your limit for the bonded inner hub",
            ),
            (
                "the adhesive's limit",
                breaking(cool_adhesive),
                "the selected adhesive's design limit",
            ),
            (
                "the magnets' onset",
                breaking(n42_parts),
                "the magnets' single-ring onset",
            ),
            (
                "cup cannot be fitted",
                breaking(|i| i.shaft_joint.cup_fit_max_C = 250.0),
                "your limit for the bare cup",
            ),
            (
                "key above its limit",
                breaking(|i| i.coupling.keyway_depth_mm = 0.3),
                "Alone, the key would bear",
            ),
            (
                "cup key without length",
                breaking(|i| i.metal.boss_length_mm = 0.0),
                "its boss and web",
            ),
            (
                "inner hub key without length",
                breaking(|i| i.metal.hub_length_mm = 3.0),
                "is no longer than the key is wide",
            ),
            (
                "key without contact",
                breaking(|i| i.shaft_joint.key_contact_mm = 0.0),
                "is not above zero",
            ),
            (
                "no window under the stress limit",
                breaking(|i| i.materials.parts.back_iron = 7),
                "the bore stays within its limit only up to",
            ),
            (
                "no window that can be fitted",
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = -78.0),
                "it can be fitted only up to",
            ),
            (
                "no joining clearance",
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
                "do not even open its",
            ),
            (
                "not a number",
                breaking(|i| i.temperature.adhesive.selected = 0),
                "cannot be made",
            ),
            (
                "safety factor not above zero",
                breaking(|i| i.shaft_joint.safety_factor = 0.0),
                "The safety factor,",
            ),
            (
                "cup without a wall",
                breaking(|i| {
                    i.coupling.bore_mm = 16.0;
                    i.metal.boss_od_mm = 12.0;
                }),
                "is as wide as its boss",
            ),
            (
                "inner hub keyway through its wall",
                breaking(|i| {
                    i.coupling.bore_mm = 16.0;
                    i.coupling.keyway_depth_mm = 2.1;
                }),
                "cuts through the wall",
            ),
            (
                "inner hub without a wall",
                breaking(|i| i.coupling.bore_mm = 21.0),
                "reaches the flats",
            ),
            (
                "reversed band",
                breaking(|i| i.shaft_joint.hub_interference_min_um = 20.0),
                "is above its maximum",
            ),
        ]
    }

    /// The Problem of `check` for `inputs`.
```

(3) `possible_fixes`: the joint's fixes are every term's of both joints.

At `:2739`, replace

```rust
            (Fixes::PerTerm(_), SPACE_CLAIM) => SPACE_CLAIM_AXES.to_vec(),
```

with

```rust
            (Fixes::PerTerm(_), SPACE_CLAIM) => SPACE_CLAIM_AXES.to_vec(),
            (Fixes::PerTerm(_), SHAFT_JOINT) => JOINT_SIDES
                .iter()
                .flat_map(|side| CAUSES.map(|cause| side.fixes(cause)))
                .collect(),
```

(4) `every_text`: the joint's two Problem constants and every Problem form.

At `:2771`, replace

```rust
        ] {
            texts.push(fallback.to_owned());
        }
```

with

```rust
            SHAFT_JOINT_NOT_A_NUMBER,
            SHAFT_JOINT_FALLBACK,
        ] {
            texts.push(fallback.to_owned());
        }
        // Every sentence form of the shaft joints' Problem, most of which no design of
        // `designs()` reaches.
        for (_, inputs, _) in shaft_joint_problem_designs() {
            texts.extend(shaft_joint_problem(&inputs, &compute_all(&inputs)));
        }
```

(5) `the_gearbox_check_is_a_dashboard_row_with_its_guide` moves (its row 15 -> 16, its guide after the joint's), and two
new tests follow it: the joint's row, check and guide (spec 7.4, with the interim link, plan deviation PD-25), and the
guide's two joints held to the engine's, bit for bit.

At `:3028`, replace

```rust
    /// one dashboard row of its own, after the headline's rows and before the space claim, a
    /// design check of the results, and a guide, between the clamp's and the space claim's,
    /// whose Why link and its row's lead to the torque-band note, which owns cold-high at the
    /// gearbox input. The ripple row stays a value with no check.
```

with

```rust
    /// one dashboard row of its own, after the headline's rows and the shaft joints row and
    /// before the space claim, a design check of the results, and a guide, between the shaft
    /// joints' and the space claim's, whose Why link and its row's lead to the torque-band note,
    /// which owns cold-high at the gearbox input. The ripple row stays a value with no check.
```

At `:3046`, replace

```rust
                (15, (GEARBOX, Some(GEARBOX))),
```

with

```rust
                (16, (GEARBOX, Some(GEARBOX))),
```

At `:3064`, replace

```rust
        assert_eq!(at(GEARBOX), at(CLAMP) + 1);
        assert_eq!(at(SPACE_CLAIM), at(GEARBOX) + 1);
    }

```

with

```rust
        assert_eq!(at(GEARBOX), at(SHAFT_JOINT) + 1);
        assert_eq!(at(SPACE_CLAIM), at(GEARBOX) + 1);
    }

    /// The shaft joints check (BL-064): one dashboard row of its own at index 15, after the
    /// headline's rows (the hidden clamp row last among them) and before the gearbox row; a
    /// design check of the results, listed last (its group follows the gearbox's in the
    /// schema); a guide between the clamp's and the gearbox's, labelled apart from the clamp's,
    /// counted while the clamp section is paused, whose Why link is its row's.
    #[test]
    fn the_shaft_joint_check_is_a_dashboard_row_with_its_guide() {
        let rows: Vec<(usize, (&str, Option<&str>))> = DASHBOARD
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, (row, check))| {
                row.starts_with("shaft_joint.")
                    || check.is_some_and(|c| c.starts_with("shaft_joint."))
            })
            .collect();
        assert_eq!(rows, [(15, (SHAFT_JOINT, Some(SHAFT_JOINT)))]);
        assert_eq!(DASHBOARD[14].0, CLAMP);
        assert_eq!(DASHBOARD[16].0, GEARBOX);
        assert_eq!(CHECKS.last(), Some(&SHAFT_JOINT));
        assert_eq!(
            CHECKS
                .iter()
                .filter(|c| c.starts_with("shaft_joint."))
                .count(),
            1
        );
        let joint = guide(SHAFT_JOINT).expect("the shaft joints check has a guide");
        assert_eq!(joint.label, "Shaft joints");
        assert_eq!(row_learn(SHAFT_JOINT), Some(joint.learn));
        // Until its reviewed note links it (plan Task 7): no note yet, with the reason.
        assert_eq!(
            joint.learn,
            Learn::None("No note covers the shaft joints yet.")
        );
        assert!(counts(joint, true) && counts(joint, false));
        let at = |check: &str| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        assert_eq!(at(SHAFT_JOINT), at(CLAMP) + 1);
        assert_eq!(at(GEARBOX), at(SHAFT_JOINT) + 1);
    }

    /// The guide's two joints are the engine's, in its order, and each path it reads is the
    /// result or input of that joint: at the defaults and at a design where every number
    /// differs between the joints, each result equals the engine's term bit for bit.
    #[test]
    fn the_shaft_joint_sides_are_the_engine_s_joints() {
        for inputs in [DesignInputs::default(), only_the_joint_fails()] {
            let results = compute_all(&inputs);
            let joints = shaft_joints(&inputs, &results).expect("a judged check");
            for (side, joint) in JOINT_SIDES.iter().zip(&joints) {
                assert_eq!(side.name, joint.name);
                assert_eq!(joint_side(joint.name).map(|(_, s)| s.name), Some(side.name));
                let result = |path: &str| match results.get(path) {
                    Some(Value::Num(x)) => x.to_bits(),
                    other => panic!("{path}: {other:?}"),
                };
                for (path, term) in [
                    (side.holding, joint.holding_Nm),
                    (side.stress, joint.bore_stress_MPa),
                    (side.key, joint.key_pressure_MPa),
                    (side.needed, joint.needed_um),
                    (side.elastic, joint.elastic_max_um),
                    (side.fittable, joint.fittable_max_um),
                    (side.fit_temp, joint.fit_temp_C),
                    (side.fit_allowed, joint.fit_allowed_C),
                ] {
                    assert_eq!(result(path), term.to_bits(), "{}: {path}", side.name);
                }
                for path in [side.min, side.max, side.shaft_fit] {
                    assert!(InputCatalogue::get().entry(path).is_some(), "{path}");
                }
            }
        }
        assert_eq!(joint_side("no such joint").map(|(i, _)| i), None);
    }

```

(6) `terms()`: the engine's four terms of each joint with spec 5.2's margins (an indeterminate joint's terms pass; with
no keyway both key terms pass with +inf), and the term names:

At `:3255`, replace

```rust
            )],
            END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0)],
```

with

```rust
            )],
            // The engine's four terms of each joint, with the guide's margins (spec 5.2). A joint
            // that is not judged (no wall, a reversed band, or the check cannot be made) passes
            // them; with no keyway the key terms pass with a margin of +inf.
            SHAFT_JOINT => {
                // Not judged while the check cannot be made (`shaft_joints` is `None`).
                let judged = shaft_joints(inputs, results).is_some();
                let links = shaft_joint_links(inputs, results);
                shaft_joint::joints(&inputs.shaft_joint, &links)
                    .iter()
                    .flat_map(|joint| {
                        CAUSES.map(|cause| {
                            let passes = !(judged && joint.determinate() && joint.fails(cause));
                            term(&joint_term(joint.name, cause), joint.margin(cause), passes)
                        })
                    })
                    .collect()
            }
            END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0)],
```

At `:3273`, replace

```rust

    /// The margin of the term `name` of `check`.
```

with

```rust

    /// The name of the shaft joint `joint`'s term `cause` in [`terms`]: "inner hub slips", "cup
    /// bore stress", "inner hub fitting", "cup key".
    fn joint_term(joint: &str, cause: Cause) -> String {
        let term = match cause {
            Cause::Slips => "slips",
            Cause::BoreStress => "bore stress",
            Cause::CannotBeFitted => "fitting",
            Cause::Key => "key",
        };
        format!("{joint} {term}")
    }

    /// The margin of the term `name` of `check`.
```

(7) `every_fix_moves_its_check_toward_passing_over_its_slider`: its table must list every slider fix of every guide,
so the joint's eleven slider fixes land with the guide (plan deviation PD-20; BL-059's T2 did the same). Insert after the
two `GEARBOX` cases:

At `:3435`, replace

```rust
                "gearbox input torque",
            ),
            (
                SPACE_CLAIM,
```

with

```rust
                "gearbox input torque",
            ),
            // BL-064. The hub's minimum goes past the band's maximum (to the slider's 40 µm):
            // the margin keeps rising, though the band then reads reversed.
            (
                SHAFT_JOINT,
                only_the_joint_fails(),
                HUB_MIN,
                Direction::Raise,
                "inner hub slips",
            ),
            (
                SHAFT_JOINT,
                only_the_joint_fails(),
                HUB_LENGTH,
                Direction::Raise,
                "inner hub slips",
            ),
            (
                SHAFT_JOINT,
                only_the_joint_fails(),
                FACE_GAP,
                Direction::Raise,
                "inner hub slips",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0),
                CUP_MIN,
                Direction::Raise,
                "cup slips",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0),
                BOSS_LENGTH,
                Direction::Raise,
                "cup slips",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0),
                MAGNET_LENGTH,
                Direction::Lower,
                "cup slips",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.hub_interference_max_um = 26.0),
                HUB_MAX,
                Direction::Lower,
                "inner hub bore stress",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.cup_interference_max_um = 26.0),
                CUP_MAX,
                Direction::Lower,
                "cup bore stress",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
                WHEEL_SHAFT,
                Direction::Lower,
                "inner hub fitting",
            ),
            (
                SHAFT_JOINT,
                breaking(|i| i.shaft_joint.cup_fit_max_C = 250.0),
                GEARBOX_SHAFT,
                Direction::Lower,
                "cup fitting",
            ),
            // 50.8 mm magnets: the cup's key alone bears about 390 MPa at a 1.2 mm contact,
            // under the 1.7 mm keyway depth that caps it.
            (
                SHAFT_JOINT,
                breaking(long_magnets),
                KEY_CONTACT,
                Direction::Raise,
                "cup key",
            ),
            (
                SPACE_CLAIM,
```

(8) `two_failing_checks_that_pull_one_input_apart_say_so` (spec R-6): without back iron the cup's Lower magnet length
and the hot minimum's Raise magnet length pull one input apart; one note in each block.

At `:4039`, replace

```rust
        // over (magnet length), and the blocks against a diameter claim (Outside diameter). Each
        // fix helps its own check, so both stay, and each of the two blocks says, in the same
        // words, that the input pulls two ways and another input has to settle one of them
        // (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 4] = [
            ("defaults", &[MAGNET_LENGTH]),
```

with

```rust
        // over (magnet length), the hot-low torque against the cup's slip without back iron
        // (magnet length: the cup does not lengthen with the magnets, BL-064), and the blocks
        // against a diameter claim (Outside diameter). Each fix helps its own check, so both
        // stay, and each of the two blocks says, in the same words, that the input pulls two ways
        // and another input has to settle one of them (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 5] = [
            ("defaults", &[MAGNET_LENGTH]),
            ("no_back_iron", &[MAGNET_LENGTH]),
```

At `:4121`, replace

```rust
        assert_eq!(trade_offs(&all, at(GEARBOX)), [gearbox.as_str()]);
        let crowded = with(with_poles(12), |i| i.metal.max_diameter_mm = 42.0);
```

with

```rust
        assert_eq!(trade_offs(&all, at(GEARBOX)), [gearbox.as_str()]);
        // No back iron: the cup slips (BL-064) beside the failing hot-low torque, the same note
        // in both blocks.
        let aluminium = breaking(no_back_iron);
        let all = all_fixes(&aluminium, &compute_all(&aluminium), false);
        let joint = note(MAGNET_LENGTH, HOT_MIN_GUIDE.label, SHAFT_JOINT_GUIDE.label);
        assert_eq!(trade_offs(&all, at(HOT_MIN)), [joint.as_str()]);
        assert_eq!(trade_offs(&all, at(SHAFT_JOINT)), [joint.as_str()]);
        let crowded = with(with_poles(12), |i| i.metal.max_diameter_mm = 42.0);
```

(9) `each_problem_is_some_exactly_when_its_check_fails` (spec 7.8; it runs every guide over `designs()`): its level
table gains six joint designs.

At `:4298`, replace

```rust
            (only_the_gearbox_fails(), GEARBOX, Bad),
```

with

```rust
            (only_the_gearbox_fails(), GEARBOX, Bad),
            (only_the_joint_fails(), SHAFT_JOINT, Bad),
            (breaking(no_back_iron), SHAFT_JOINT, Bad),
            (breaking(cool_adhesive), SHAFT_JOINT, Bad),
            (breaking(n42_parts), SHAFT_JOINT, Bad),
            (
                breaking(|i| i.shaft_joint.hub_interference_min_um = 20.0),
                SHAFT_JOINT,
                Caution,
            ),
            (
                breaking(|i| i.temperature.adhesive.selected = 0),
                SHAFT_JOINT,
                Caution,
            ),
```

(10) `the_problems_quote_numbers_from_the_results` (spec 7.8): a `SHAFT_JOINT` case, and the window's ends recomputed
from the results at two designs whose windows differ.

At `:4361`, replace

```rust
                &["metal.cold_high_input_Nm"],
            ),
            (
```

with

```rust
                &["metal.cold_high_input_Nm"],
            ),
            (
                SHAFT_JOINT,
                design(|i| i.shaft_joint.hub_interference_min_um = 10.0),
                design(|i| i.shaft_joint.hub_interference_min_um = 9.0),
                &[
                    "shaft_joint.hub_holding_Nm",
                    "shaft_joint.required_Nm",
                    "metal.torque_cold_high_Nm",
                ],
            ),
            (
```

At `:4412`, replace

```rust
        }
        // The gearbox Problem's sentence 2 (owner decision 4A) quotes numbers worked out from
```

with

```rust
        }
        // The shaft joints' window is worked out from the results: its lower end the smallest
        // minimum that holds, rounded up, its upper end the smaller of the two upper edges,
        // rounded down (spec 5.2), at two designs whose windows differ.
        let edits: [fn(&mut DesignInputs); 2] = [
            |i| i.shaft_joint.hub_interference_min_um = 10.0,
            |i| {
                i.shaft_joint.hub_interference_min_um = 10.0;
                i.shaft_joint.wheel_shaft_fit_C = -150.0;
            },
        ];
        for edit in edits {
            let inputs = design(edit);
            let results = compute_all(&inputs);
            let text = problem(SHAFT_JOINT, &inputs).expect("the inner hub slips");
            let s = &results.shaft_joint;
            let upper = s.hub_elastic_max_um.min(s.hub_fittable_max_um);
            for quoted in [
                number_text(rounded_up(s.hub_needed_um), "shaft_joint.hub_needed_um"),
                number_text(rounded_down(upper), "shaft_joint.hub_needed_um"),
            ] {
                assert!(text.contains(&quoted), "{quoted} in {text}");
            }
        }
        // The gearbox Problem's sentence 2 (owner decision 4A) quotes numbers worked out from
```

(11) New tests after `a_check_that_does_not_apply_gives_no_problem` (spec 5.2, 4.3, 7.8):
`every_shaft_joint_problem_form_is_reached`,
`the_shaft_joint_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made`, the helper `assert_in_order`,
`the_shaft_joint_problem_quotes_the_live_numbers` (the spec's literal pins: hub minimum 10, the 304 probe, an uncooled
wheel shaft, DP460, library N42 parts, N50M on manual dimensions, 2214 Hi-Temp at a typed 150 °C, inner N42M with outer
N35; the margin built from `CURE_MARGIN_MIN_C`; with a window, no empty-window sentence),
`a_shaft_joint_window_typed_at_its_quoted_ends_passes`,
`a_window_narrower_than_the_last_shown_digit_quotes_its_ends_with_more_decimals`,
`a_failing_shaft_joint_number_never_reads_equal_to_its_bound` and `the_shaft_joint_texts_never_order_the_torques`
(review 2 MF-1; the note's half is Task 7's).

At `:5871`, replace

```rust

    // ------------------------------------------------------------------ the design status
```

with

```rust

    #[test]
    fn every_shaft_joint_problem_form_is_reached() {
        // Spec 5.2 and 4.3: a design for each sentence form; each fails the check (red, or amber
        // for the forms of a joint that is not judged and of a value that is not a number), and
        // its Problem holds the form's words, never the fallback.
        for (name, inputs, words) in shaft_joint_problem_designs() {
            let results = compute_all(&inputs);
            assert_ne!(
                level_of(SHAFT_JOINT, &inputs, &results),
                Level::Good,
                "{name}"
            );
            let text = shaft_joint_problem(&inputs, &results)
                .unwrap_or_else(|| panic!("{name}: no Problem"));
            assert!(text.contains(words), "{name}: {words:?} in {text}");
            assert_ne!(text, SHAFT_JOINT_FALLBACK, "{name}");
        }
    }

    #[test]
    fn the_shaft_joint_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made() {
        // Red with a joint that fails; amber while a joint is not judged, or while the check
        // cannot be made (then that sentence alone, and no fix); green, with no Problem, while it
        // passes and while the end-effect model is out of range (the row is greyed, whatever its
        // text: a bore stress above its limit reads red there).
        use Level::{Bad, Caution, Good};
        let joint = guide(SHAFT_JOINT).unwrap();
        let not_a_number = breaking(|i| i.temperature.adhesive.selected = 0);
        let no_factor = breaking(|i| i.shaft_joint.safety_factor = 0.0);
        let overstressed_short = with(short_magnets(), |i| {
            i.shaft_joint.hub_interference_max_um = 26.0;
        });
        for (name, inputs, want) in [
            ("defaults", DesignInputs::default(), Good),
            ("passing", passing_design(), Good),
            ("joint only", only_the_joint_fails(), Bad),
            ("not a number", not_a_number.clone(), Caution),
            ("safety factor 0", no_factor.clone(), Caution),
            (
                "reversed band",
                breaking(|i| i.shaft_joint.hub_interference_min_um = 20.0),
                Caution,
            ),
            ("short magnets", short_magnets(), Good),
            (
                "short magnets overstressed",
                overstressed_short.clone(),
                Good,
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(level(joint, &inputs, &results), want, "{name}");
            assert_eq!(
                (joint.problem)(&inputs, &results).is_some(),
                want != Good,
                "{name}"
            );
        }
        let results = compute_all(&overstressed_short);
        assert!(
            results
                .shaft_joint
                .check
                .starts_with(shaft_joint::BORE_OVER_ITS_LIMIT),
            "{}",
            results.shaft_joint.check
        );
        let results = compute_all(&not_a_number);
        assert_eq!(results.shaft_joint.check, shaft_joint::UNKNOWN_NOT_A_NUMBER);
        assert_eq!(
            shaft_joint_problem(&not_a_number, &results).as_deref(),
            Some(SHAFT_JOINT_NOT_A_NUMBER)
        );
        assert_eq!(fixes(joint, &not_a_number, &results, false), []);
        // A safety factor typed at or below zero (ruling C-4): its sentence alone, and no fix.
        let results = compute_all(&no_factor);
        assert_eq!(results.shaft_joint.check, shaft_joint::UNKNOWN_SAFETY_FACTOR);
        assert_eq!(
            shaft_joint_problem(&no_factor, &results),
            Some(format!(
                "The safety factor, {}, is not above zero: enter at least 1.",
                input_text(&no_factor, "shaft_joint.safety_factor")
            ))
        );
        assert_eq!(fixes(joint, &no_factor, &results, false), []);
    }

    /// Asserts that `text` holds each of `parts`, in their order.
    fn assert_in_order(text: &str, parts: &[&str]) {
        let mut rest = text;
        for part in parts {
            let at = rest
                .find(part)
                .unwrap_or_else(|| panic!("{part:?}, in order, in {text}"));
            rest = &rest[at + part.len()..];
        }
    }

    #[test]
    fn the_shaft_joint_problem_quotes_the_live_numbers() {
        // Spec 5.2 and 7.8: each number as the readouts show it (four significant digits); the
        // window's ends quoted inward (review SF-11: the results 12.0815 and 18.8281 µm read
        // 12.09 and 18.82 µm), the edge that binds named; the source of the inner hub's allowed
        // fitting temperature named with its live number, the onset's margin written from
        // CURE_MARGIN_MIN_C with the allowed temperature's decimals (owner R-15).
        let margin = |decimals: usize| with_unit(fixed(CURE_MARGIN_MIN_C, decimals), "°C");
        let text = |inputs: DesignInputs| problem(SHAFT_JOINT, &inputs).expect("it fails");
        let hub_min_10 = text(design(|i| i.shaft_joint.hub_interference_min_um = 10.0));
        assert_in_order(
            &hub_min_10,
            &[
                "At its loosest the inner hub holds 3.482 N·m",
                "7.647 N·m",
                "cold-high, 3.823 N·m",
                "safety factor of 2.000",
                "On the inner hub, a measured interference from 12.09 µm to 18.82 µm",
                "fit at the allowed temperature",
                "(its band is 10.00 µm to 18.00 µm)",
            ],
        );
        // With a window there is no empty-window sentence (spec 7.8).
        assert!(
            !hub_min_10.contains("No measured interference works"),
            "{hub_min_10}"
        );
        // 304 back iron: no window on either joint, each edge below the need named.
        assert_in_order(
            &text(design(|i| i.materials.parts.back_iron = 7)),
            &[
                "No measured interference works on the inner hub here: holding needs at least 16.31 µm, but the bore stays within its limit only up to 11.72 µm.",
                "No measured interference works on the cup here: holding needs at least 15.80 µm, but the bore stays within its limit only up to 11.74 µm.",
            ],
        );
        // An uncooled wheel shaft (review MF-6, SF-12, review 2 SF-6b).
        assert_in_order(
            &text(design(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0)),
            &[
                "Fitting the inner hub would need it at 247.6 °C with the wheel shaft at 20.00 °C, above the 100.0 °C allowed: your limit for the bonded inner hub, 100.0 °C.",
                "holding needs at least 12.09 µm, but the allowed temperatures do not even open its 10.00 µm joining clearance.",
            ],
        );
        // DP460: the adhesive's design limit binds.
        assert_in_order(
            &text(breaking(cool_adhesive)),
            &[
                "would need it at 93.27 °C with the wheel shaft at -196.0 °C, above the 60.00 °C allowed: the selected adhesive's design limit, 60.00 °C.",
                "from 11.14 µm to 13.90 µm would hold and fit at the allowed temperature (its band is 13.00 µm to 18.00 µm)",
            ],
        );
        // Library N42 parts (owner R-15): the inner ring's onset less the margin binds.
        assert_in_order(
            &text(breaking(n42_parts)),
            &[
                &format!(
                    "above the 59.83 °C allowed: the magnets' single-ring onset, 69.83 °C, less {}.",
                    margin(2)
                ),
                "from 11.62 µm to 13.88 µm",
            ],
        );
        // N50M on manual dimensions: its onset is above the need, the onset less the margin
        // below it.
        assert_in_order(
            &text(breaking(|i| graded(i, "N50M"))),
            &[
                &format!(
                    "above the 88.80 °C allowed: the magnets' single-ring onset, 98.80 °C, less {}.",
                    margin(2)
                ),
                "from 12.21 µm to 17.45 µm",
            ],
        );
        // 2214 Hi-Temp (177 °C) and the hub's limit typed at 150 °C: the onset less the margin,
        // 133.4 °C, binds; the margin takes the allowed temperature's one decimal (review 4,
        // SF-5).
        assert_in_order(
            &text(breaking(|i| {
                i.temperature.adhesive.selected = 3;
                i.shaft_joint.hub_fit_max_C = 150.0;
                i.shaft_joint.hub_interference_max_um = 26.0;
            })),
            &[&format!(
                "above the 133.4 °C allowed: the magnets' single-ring onset, 143.4 °C, less {}.",
                margin(1)
            )],
        );
        // Mixed rings (owner R-15 ring ruling): the inner N42M's own onset, never the outer
        // N35's 59.93 °C.
        let mixed = text(breaking(|i| {
            graded(i, "N42M");
            i.coupling.magnets.grade_outer = "N35".to_owned();
        }));
        assert_in_order(
            &mixed,
            &[&format!(
                "above the 81.35 °C allowed: the magnets' single-ring onset, 91.35 °C, less {}.",
                margin(2)
            )],
        );
        assert!(!mixed.contains("59.93"), "{mixed}");
        // Spec 7.2 (review 2, SF-4): a key typed wider than the bore's circumference (40 mm on Ø10)
        // leaves no contact arc: each slip sentence says the joint holds 0, and no window sentence
        // follows (its smallest minimum is +inf).
        let wide = text(design(|i| i.shaft_joint.key_width_mm = 40.0));
        assert!(wide.starts_with("At its loosest the inner hub holds 0 N·m"), "{wide}");
        assert!(!wide.contains("measured interference"), "{wide}");
        // Spec 7.1: at a 26 µm hub maximum the badge is the bore stress; the fitting cause follows it
        // in the Problem, then the window.
        assert_in_order(
            &text(design(|i| i.shaft_joint.hub_interference_max_um = 26.0)),
            &[
                "At its tightest the inner hub's bore stress is 385.1 MPa",
                "Fitting the inner hub would need it at 158.3 °C",
                "(its band is 13.00 µm to 26.00 µm)",
            ],
        );
    }

    #[test]
    fn a_shaft_joint_window_typed_at_its_quoted_ends_passes() {
        // Review SF-11: the window is quoted inward, so a band typed at its quoted ends holds,
        // stays within the stress limit and fits. At the defaults: 12.09 to 18.82 µm on the
        // inner hub, 11.31 to 23.86 µm on the cup.
        let defaults = DesignInputs::default();
        let s = compute_all(&defaults).shaft_joint;
        let ends = |needed: f64, elastic: f64, fittable: f64| {
            let (low, high) =
                window_quotes(needed, elastic.min(fittable), "shaft_joint.hub_needed_um");
            let value = |text: String| -> f64 { text.split(' ').next().unwrap().parse().unwrap() };
            (value(low), value(high))
        };
        let hub = ends(s.hub_needed_um, s.hub_elastic_max_um, s.hub_fittable_max_um);
        let cup = ends(s.cup_needed_um, s.cup_elastic_max_um, s.cup_fittable_max_um);
        assert_eq!((hub, cup), ((12.09, 18.82), (11.31, 23.86)));
        let typed = design(|i| {
            i.shaft_joint.hub_interference_min_um = hub.0;
            i.shaft_joint.hub_interference_max_um = hub.1;
            i.shaft_joint.cup_interference_min_um = cup.0;
            i.shaft_joint.cup_interference_max_um = cup.1;
        });
        assert_eq!(
            compute_all(&typed).shaft_joint.check,
            shaft_joint::SHAFT_JOINTS_PASS
        );
    }

    #[test]
    fn a_window_narrower_than_the_last_shown_digit_quotes_its_ends_with_more_decimals() {
        // BL-059 NB-1, spec 5.2: where the ends rounded inward at the readouts' digits would read
        // equal or crossed, both take more decimals, as few as read them apart in order.
        let unit = "shaft_joint.hub_needed_um";
        let quotes = |needed: f64, upper: f64| window_quotes(needed, upper, unit);
        assert_eq!(
            quotes(12.0815, 18.8281),
            ("12.09 µm".into(), "18.82 µm".into())
        );
        assert_eq!(
            quotes(12.0815, 12.0850),
            ("12.082 µm".into(), "12.085 µm".into())
        );
        assert_eq!(
            quotes(12.081512, 12.081588),
            ("12.08152 µm".into(), "12.08158 µm".into())
        );
        // Below 10 µm the readouts show three decimals: from four.
        assert_eq!(
            quotes(9.62012, 9.62088),
            ("9.6202 µm".into(), "9.6208 µm".into())
        );
    }

    #[test]
    fn a_failing_shaft_joint_number_never_reads_equal_to_its_bound() {
        // BL-059 MF-1, spec 5.2: a holding read equal to the need is quoted one last digit under
        // it, a stress, a fitting temperature or a key pressure read equal to its limit one last
        // digit over it. The results are set by hand: no design on the sliders lands there.
        let inputs = only_the_joint_fails();
        let mut results = compute_all(&inputs);
        let s = &mut results.shaft_joint;
        s.hub_holding_Nm = s.required_Nm - 1e-6;
        s.hub_bore_stress_MPa = s.stress_limit_MPa + 1e-6;
        s.hub_fit_temp_C = s.hub_fit_allowed_C + 1e-6;
        s.hub_key_pressure_MPa = s.stress_limit_MPa + 1e-6;
        let s = &results.shaft_joint;
        let read = |path: &str| result_text(&results, path);
        for (path, bound, above, bound_path) in [
            (
                "shaft_joint.hub_holding_Nm",
                s.required_Nm,
                false,
                "shaft_joint.required_Nm",
            ),
            (
                "shaft_joint.hub_bore_stress_MPa",
                s.stress_limit_MPa,
                true,
                "shaft_joint.stress_limit_MPa",
            ),
            (
                "shaft_joint.hub_fit_temp_C",
                s.hub_fit_allowed_C,
                true,
                "shaft_joint.hub_fit_allowed_C",
            ),
            (
                "shaft_joint.hub_key_pressure_MPa",
                s.stress_limit_MPa,
                true,
                "shaft_joint.stress_limit_MPa",
            ),
        ] {
            assert_eq!(
                read(path),
                read(bound_path),
                "{path}: the readouts read equal"
            );
            let quoted = apart_from(&results, path, bound, above);
            assert_ne!(quoted, read(bound_path), "{path}");
            let number = |text: &str| -> f64 { text.split(' ').next().unwrap().parse().unwrap() };
            let (q, b) = (number(&quoted), number(&read(bound_path)));
            assert!(
                if above { q > b } else { q < b },
                "{path}: {quoted} against {}",
                read(bound_path)
            );
        }
        // Apart already: as the readout shows it.
        assert_eq!(
            apart_from(
                &compute_all(&inputs),
                "shaft_joint.hub_holding_Nm",
                6.72,
                false
            ),
            result_text(&compute_all(&inputs), "shaft_joint.hub_holding_Nm")
        );
    }

    /// The shaft joints' texts are static where they can be, so they hold at every reachable
    /// input (review 2, MF-1; BL-067): cold-high "can reach" a joint, it is no bound, and with
    /// the operating temperature below the minimum (each on its slider) the operating pull-out
    /// with its allowance is above it. No text the check shows may call any torque the
    /// strongest or the most expected.
    #[test]
    fn the_shaft_joint_texts_never_order_the_torques() {
        let mut corner = DesignInputs::default();
        for (path, x) in [
            ("coupling.op_temp_C", -40.0),
            ("metal.min_temp_C", 20.0),
            ("metal.variation", 0.1),
        ] {
            let range = slider_range(path);
            assert!(
                range.min <= x && x <= range.max,
                "{path}: {x} on its slider"
            );
            corner = with_number(&corner, path, x).unwrap_or_else(|| panic!("{path}: {x}"));
        }
        let results = compute_all(&corner);
        let allowance = 1.0 + corner.metal.variation;
        assert!(
            results.model.pullout_Nm * allowance > results.metal.torque_cold_high_Nm,
            "the operating pull-out with its allowance, {}, above cold-high, {}",
            results.model.pullout_Nm * allowance,
            results.metal.torque_cold_high_Nm
        );
        let mut texts: Vec<String> = Vec::new();
        for row in crate::engine::meta::input_rows(&corner) {
            if row.path.starts_with("shaft_joint.") {
                texts.extend([row.meta.label.to_owned(), row.meta.help.to_owned()]);
            }
        }
        for row in crate::engine::meta::result_rows(&results) {
            if row.path.starts_with("shaft_joint.") {
                texts.extend([row.meta.label.to_owned(), row.meta.help.to_owned()]);
            }
        }
        texts.extend(
            [
                shaft_joint::SHAFT_JOINTS_PASS,
                shaft_joint::UNKNOWN_NOT_A_NUMBER,
                shaft_joint::UNKNOWN_NO_WALL,
                shaft_joint::UNKNOWN_BAND_REVERSED,
                shaft_joint::UNKNOWN_SAFETY_FACTOR,
                shaft_joint::FIT_SLIPS,
                shaft_joint::BORE_OVER_ITS_LIMIT,
                shaft_joint::CANNOT_BE_FITTED,
                shaft_joint::KEY_OVER_ITS_LIMIT,
                shaft_joint::KEY_HAS_NO_BEARING_AREA,
                SHAFT_JOINT_GUIDE.label,
                SHAFT_JOINT_GUIDE.why,
                SHAFT_JOINT_NOT_A_NUMBER,
                SHAFT_JOINT_FALLBACK,
            ]
            .map(str::to_owned),
        );
        let corner_failing = with(corner, |i| i.shaft_joint.hub_interference_min_um = 8.0);
        let designs = designs()
            .into_iter()
            .map(|(_, inputs)| inputs)
            .chain(
                shaft_joint_problem_designs()
                    .into_iter()
                    .map(|(_, inputs, _)| inputs),
            )
            .chain([corner_failing]);
        for inputs in designs {
            texts.extend(problem(SHAFT_JOINT, &inputs));
        }
        assert!(texts.len() > 50, "{}", texts.len());
        for text in texts {
            let lower = text.to_lowercase();
            for word in ["strongest", "most expected", "most torque expected"] {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }
        }
    }

    // ------------------------------------------------------------------ the design status
```

(12) The status tests (spec 7.7, 7.17): `the_design_status_counts_the_failing_checks` (of 9 -> of 10),
`a_design_failing_only_the_clamp_passes_while_the_section_is_paused` (8 -> 9 paused, 9 -> 10 shown),
`the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not` (the same), and a new
`the_shaft_joint_check_counts_in_the_status_whether_the_clamp_section_shows_or_not`.

At `:5883`, replace

```rust
                total: 9,
```

with

```rust
                total: 10,
```

At `:5888`, replace

```rust
            design_status_text(&defaults),
            "Example design: fails 4 of 9 checks"
```

with

```rust
            design_status_text(&defaults),
            "Example design: fails 4 of 10 checks"
```

At `:5895`, replace

```rust
            "Your design: fails 3 of 9 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 9 checks"
```

with

```rust
            "Your design: fails 3 of 10 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 10 checks"
```

At `:5904`, replace

```rust
            "Your design: fails 4 of 9 checks, 1 caution"
```

with

```rust
            "Your design: fails 4 of 10 checks, 1 caution"
```

At `:5912`, replace

```rust
            "Your design: fails 4 of 9 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall and the gearbox check.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 9 checks"
```

with

```rust
            "Your design: fails 4 of 10 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall, the shaft joints and the gearbox check.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 10 checks"
```

At `:5966`, replace

```rust
        assert_eq!((shown.bad, shown.caution, shown.total), (1, 0, 9));
        assert_eq!(
            design_status_text(&shown),
            "Your design: fails 1 of 9 checks"
        );
        let paused = design_status(&inputs, &results, true);
        assert_eq!((paused.bad, paused.caution, paused.total), (0, 0, 8));
        assert_eq!(
            design_status_text(&paused),
            "Your design: passes all 8 checks"
```

with

```rust
        assert_eq!((shown.bad, shown.caution, shown.total), (1, 0, 10));
        assert_eq!(
            design_status_text(&shown),
            "Your design: fails 1 of 10 checks"
        );
        let paused = design_status(&inputs, &results, true);
        assert_eq!((paused.bad, paused.caution, paused.total), (0, 0, 9));
        assert_eq!(
            design_status_text(&paused),
            "Your design: passes all 9 checks"
```

At `:5981`, replace

```rust
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 4 of 8 checks"
```

with

```rust
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 4 of 9 checks"
```

At `:5998`, replace

```rust
            "Example design: fails 4 of 8 checks"
        );
        assert_eq!(
            status(&defaults, false),
            "Example design: fails 4 of 9 checks"
        );
        let passing = passing_design();
        assert_eq!(status(&passing, true), "Your design: passes all 8 checks");
        assert_eq!(status(&passing, false), "Your design: passes all 9 checks");
```

with

```rust
            "Example design: fails 4 of 9 checks"
        );
        assert_eq!(
            status(&defaults, false),
            "Example design: fails 4 of 10 checks"
        );
        let passing = passing_design();
        assert_eq!(status(&passing, true), "Your design: passes all 9 checks");
        assert_eq!(status(&passing, false), "Your design: passes all 10 checks");
```

At `:6017`, replace

```rust
            "Your design: fails 1 of 8 checks"
        );
        assert_eq!(
            status(&gearbox_only, false),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&breaking(unknown_rating), true),
            "Your design: passes all 8 checks, 1 caution"
```

with

```rust
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&gearbox_only, false),
            "Your design: fails 1 of 10 checks"
        );
        assert_eq!(
            status(&breaking(unknown_rating), true),
            "Your design: passes all 9 checks, 1 caution"
        );
    }

    #[test]
    fn the_shaft_joint_check_counts_in_the_status_whether_the_clamp_section_shows_or_not() {
        // BL-064 (spec 7.7): counted, and outside the clamp section, so the pause takes only the
        // clamp out of the count. The example design passes it; with the passing design's inner
        // hub minimum at 8 µm it fails alone, red; a reversed band asks for a look; with the
        // clamp also failing, the clamp still moves no count while paused.
        let status = |inputs: &DesignInputs, paused: bool| {
            design_status_text(&design_status(inputs, &compute_all(inputs), paused))
        };
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(level_of(SHAFT_JOINT, &defaults, &results), Level::Good);
        let joint_only = only_the_joint_fails();
        let results = compute_all(&joint_only);
        let failing: Vec<(&str, Level)> = CHECK_GUIDES
            .iter()
            .map(|g| (g.check, level(g, &joint_only, &results)))
            .filter(|&(_, badge)| badge != Level::Good)
            .collect();
        assert_eq!(failing, [(SHAFT_JOINT, Level::Bad)]);
        assert_eq!(
            status(&joint_only, true),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&joint_only, false),
            "Your design: fails 1 of 10 checks"
        );
        let reversed = breaking(|i| i.shaft_joint.hub_interference_min_um = 20.0);
        assert_eq!(
            status(&reversed, true),
            "Your design: passes all 9 checks, 1 caution"
        );
        let with_the_clamp = with(only_the_joint_fails(), small_boss);
        assert_eq!(
            status(&with_the_clamp, true),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&with_the_clamp, false),
            "Your design: fails 2 of 10 checks"
```

**1d. `magcoupling-rs/src/gui/panel.rs`, `mod tests`.**

(1) The imports:

At `:1622`, replace

```rust
        only_the_clamp_fails, primary_button, select_all, short_magnets, sized_frame,
        sized_frame_at, text_center, text_rect, text_rects, tooltips_at_once, with_poles,
```

with

```rust
        only_the_clamp_fails, only_the_joint_fails, primary_button, select_all, short_magnets,
        sized_frame, sized_frame_at, text_center, text_rect, text_rects, tooltips_at_once,
        with_poles,
```

(2) `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too`: nine checks among the dashboard's
lines, the joint's greyed too.

At `:3138`, replace

```rust
        // invalid pull-out and the gearbox check of a cold-high below zero (both green by their
        // texts) carry none, and the failing filter leaves the hot minimum out.
```

with

```rust
        // invalid pull-out, and the shaft joints and gearbox checks of a cold-high below zero
        // (all three green by their texts) carry none, and the failing filter leaves the hot
        // minimum out.
```

At `:3158`, replace

```rust
        assert_eq!(checks.len(), 8);
```

with

```rust
        assert_eq!(checks.len(), 9);
```

(3) `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` (spec 7.8's clamp guard over the Problem;
ruling C-8): while paused the guard also collects the dashboard drawn on three designs that fail the joints
(`only_the_joint_fails()`, a 26 µm hub maximum, an uncooled wheel shaft), and holds each one's Problem to be drawn
(computed from the panel's own inputs and results: the panel solves Auto on its design). The joint's label and Why are
collected already (`counted_guides(true)`; the joint counts while paused); an assertion after the loop makes that
coverage explicit. Insert after the five-design loop:

At `:4297`, replace

```rust
                }
                // The teaching notes shown, their change entries as drawn, their Start here
```

with

```rust
                }
                // The shaft joints' guide (BL-064) counts while the section is paused: its
                // Problem drawn, open under its red row, where the inner hub slips, where it is
                // over its stress limit and cannot be fitted either, and on an uncooled wheel
                // shaft.
                let joint = crate::gui::guidance::guide("shaft_joint.check").expect("its guide");
                let mut overstressed = DesignInputs::default();
                overstressed.shaft_joint.hub_interference_max_um = 26.0;
                let mut uncooled = DesignInputs::default();
                uncooled.shaft_joint.wheel_shaft_fit_C = 20.0;
                for inputs in [only_the_joint_fails(), overstressed, uncooled] {
                    harness.panel.inputs = inputs;
                    harness.frame(Vec::new());
                    let drawn = drawn_texts(&harness.frame(Vec::new()));
                    let panel = &harness.panel;
                    let problem =
                        (joint.problem)(panel.inputs(), panel.results()).expect("the joints fail");
                    assert!(drawn.contains(&problem), "{problem}");
                    texts.extend(drawn);
                }
                // Not vacuous for the guide (ruling C-8): its label and Why are collected.
                for text in [joint.label, joint.why] {
                    assert!(texts.iter().any(|t| t.contains(text)), "{text:?} is not collected");
                }
                // The teaching notes shown, their change entries as drawn, their Start here
```

(4) `the_status_line_counts_the_checks_at_default_and_after_a_fix`:

At `:8839`, replace

```rust
        // gearbox check.
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (4, 0, 9, true)
```

with

```rust
        // gearbox check, of the ten guided checks (the shaft joints pass).
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (4, 0, 10, true)
```

**1e. `magcoupling-rs/src/gui/spreadsheet.rs`, `mod tests`** (spec 7.15; plan deviation PD-13 hands the Summary row here:
this task's row breaks the eighteen-row test).

(1) The imports:

At `:579`, replace

```rust
    use crate::gui::test_support::{short_magnets, snapshot};
```

with

```rust
    use crate::gui::test_support::{only_the_joint_fails, passing_design, short_magnets, snapshot};
```

(2) Replace the whole of `the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge`, from its `#[test]` to its
closing brace (the Old text ends there: Task 4's tests follow it), with a shared row helper, the renamed gearbox test
(19 rows, its neighbour the joint's row) and the joint's Summary test (the stored-3D label in its note column, the greyed
note winning; owner R-15):

Old (the whole test):

```rust
    #[test]
    fn the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge() {
        // BL-059: the dashboard gains the gearbox check, so the Summary lists 18 rows, the
        // gearbox's between the clamp's and the space claim's: its engine label and verdict, no
        // unit, its badge's level, Rust-only for the cell, its path and no correction. It is red
        // at the defaults, green with a 1.0 N·m rating, and greyed with no level while the
        // end-effect model is out of range; the Results sheet's row carries the same level.
        const PATH: &str = "gearbox.cold_high_check";
        let label = result_info(PATH).expect("a result").meta.label;
        let mut within = DesignInputs::default();
        within.coupling.gearbox_input_rating_Nm = 1.0;
        let path_column = column(&DASHBOARD_COLUMNS, "Path");
        for (inputs, level, note) in [
            (DesignInputs::default(), Some(Level::Bad), ""),
            (within, Some(Level::Good), ""),
            (short_magnets(), None, GREYED_NOTE),
        ] {
            let results = compute_all(&inputs);
            let book = sheets(&snapshot(&inputs, &results));
            let rows = &sheet(&book, SUMMARY_SHEET).rows;
            let header = rows
                .iter()
                .position(|r| r.first().map(Cell::as_text) == Some(DASHBOARD_COLUMNS[0]))
                .unwrap();
            let dashboard = &rows[header + 1..header + 1 + 18];
            assert_eq!(
                rows[header + 1 + 18],
                Vec::<Cell>::new(),
                "18 rows, then a blank"
            );
            let paths: Vec<&str> = dashboard
                .iter()
                .map(|row| row[path_column].as_text())
                .collect();
            let at = paths
                .iter()
                .position(|&path| path == PATH)
                .expect("the gearbox row");
            assert_eq!(
                (paths[at - 1], paths[at + 1]),
                ("clamps.recommended", "housing.space_claim_check")
            );
            assert_eq!(
                dashboard[at],
                [
                    Cell::text(label),
                    Cell::text(results.gearbox.cold_high_check.clone()),
                    Cell::empty(),
                    level_cell(level),
                    Cell::text(RUST_ONLY),
                    Cell::text(PATH),
                    Cell::empty(),
                    Cell::text(note),
                ]
            );
            let results_sheet = sheet(&book, RESULTS_SHEET);
            let row = row_of(results_sheet, column(&RESULT_COLUMNS, "Path"), PATH);
            assert_eq!(row[column(&RESULT_COLUMNS, "Check")], level_cell(level));
            assert_eq!(
                row[column(&RESULT_COLUMNS, "Workbook cell")],
                Cell::text(RUST_ONLY)
            );
        }
    }
```

New:

```rust
    /// Asserts the Summary of `inputs` lists one row per dashboard row, then a blank, and that
    /// the row of the Rust-only check at `path` sits between `neighbours`: its engine label and
    /// text, no unit, its badge's `level`, Rust-only for the cell, its path, no correction and
    /// `note`; and that the Results sheet's row carries the same level.
    fn assert_summary_check_row(
        inputs: &DesignInputs,
        path: &str,
        neighbours: (&str, &str),
        level: Option<Level>,
        note: &str,
    ) {
        let results = compute_all(inputs);
        let book = sheets(&snapshot(inputs, &results));
        let rows = &sheet(&book, SUMMARY_SHEET).rows;
        let header = rows
            .iter()
            .position(|r| r.first().map(Cell::as_text) == Some(DASHBOARD_COLUMNS[0]))
            .unwrap();
        let dashboard = &rows[header + 1..header + 1 + DASHBOARD.len()];
        assert_eq!(
            rows[header + 1 + DASHBOARD.len()],
            Vec::<Cell>::new(),
            "one row per dashboard row, then a blank"
        );
        let path_column = column(&DASHBOARD_COLUMNS, "Path");
        let paths: Vec<&str> = dashboard
            .iter()
            .map(|row| row[path_column].as_text())
            .collect();
        let at = paths
            .iter()
            .position(|&p| p == path)
            .unwrap_or_else(|| panic!("the row of {path}"));
        assert_eq!((paths[at - 1], paths[at + 1]), neighbours, "{path}");
        let Some(Value::Text(text)) = results.get(path) else {
            panic!("{path} is text")
        };
        assert_eq!(
            dashboard[at],
            [
                Cell::text(result_info(path).expect("a result").meta.label),
                Cell::text(text),
                Cell::empty(),
                level_cell(level),
                Cell::text(RUST_ONLY),
                Cell::text(path),
                Cell::empty(),
                Cell::text(note),
            ],
            "{path}"
        );
        let results_sheet = sheet(&book, RESULTS_SHEET);
        let row = row_of(results_sheet, column(&RESULT_COLUMNS, "Path"), path);
        assert_eq!(row[column(&RESULT_COLUMNS, "Check")], level_cell(level));
        assert_eq!(
            row[column(&RESULT_COLUMNS, "Workbook cell")],
            Cell::text(RUST_ONLY)
        );
    }

    #[test]
    fn the_summary_lists_nineteen_dashboard_rows_with_the_gearbox_badge() {
        // BL-059: the gearbox's row, between the shaft joints' (BL-064) and the space claim's.
        // It is red at the defaults, green with a 1.0 N·m rating, and greyed with no level while
        // the end-effect model is out of range.
        assert_eq!(DASHBOARD.len(), 19);
        const PATH: &str = "gearbox.cold_high_check";
        let neighbours = ("shaft_joint.check", "housing.space_claim_check");
        let mut within = DesignInputs::default();
        within.coupling.gearbox_input_rating_Nm = 1.0;
        for (inputs, level, note) in [
            (DesignInputs::default(), Some(Level::Bad), ""),
            (within, Some(Level::Good), ""),
            (short_magnets(), None, GREYED_NOTE),
        ] {
            assert_summary_check_row(&inputs, PATH, neighbours, level, note);
        }
    }

    #[test]
    fn the_summary_lists_the_shaft_joints_row_with_its_badge_and_the_stored_3d_note() {
        // BL-064 (spec 7.15; owner R-15): the shaft joints' row, between the clamp's and the
        // gearbox's. It reads a stored 3D field, so its note column carries the stored-3D label,
        // green at the defaults and at the passing design, red where the inner hub slips; while
        // the end-effect model is out of range it is greyed with no level, and the greyed note
        // wins.
        const PATH: &str = "shaft_joint.check";
        let neighbours = ("clamps.recommended", "gearbox.cold_high_check");
        for (inputs, level, note) in [
            (DesignInputs::default(), Some(Level::Good), STORED_3D_LABEL),
            (passing_design(), Some(Level::Good), STORED_3D_LABEL),
            (only_the_joint_fails(), Some(Level::Bad), STORED_3D_LABEL),
            (short_magnets(), None, GREYED_NOTE),
        ] {
            assert_summary_check_row(&inputs, PATH, neighbours, level, note);
        }
    }
```

**1f. `magcoupling-rs/src/gui/result_groups.rs`**, `the_headline_comes_first_then_the_chains_then_the_other_results`:
Task 3's three lines (its Interfaces name them), the comment and the assertion, nothing above them. Old:

```rust
        // The verdict is no chain path: until it is a dashboard row it sits in the package
        // group under the other results.
        assert_eq!(section("shaft_joint.check"), ("shaft_joint", true));
```

New:

```rust
        // The verdict is no chain path: it is a dashboard row, so the headline's (BL-064).
        assert_eq!(section("shaft_joint.check"), ("headline", false));
```

- [ ] **Step 2: Run the tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: FAIL to compile, `could not compile magcoupling-rs (lib test) due to 100 previous errors` (96 on the
plan-time prototype; ruling C-4's tests add four unresolved names), all in the test modules of `dashboard.rs` and
`guidance.rs`:
- `error[E0425]: cannot find value \`SHAFT_JOINT\` in this scope` (34);
- `error[E0433]: failed to resolve: use of unresolved module or unlinked crate \`shaft_joint\`` (26);
- `use of undeclared type \`Cause\`` (4) and `cannot find type \`Cause\`` (1);
- three each of `SHAFT_JOINT_NOT_A_NUMBER`, `SHAFT_JOINT_GUIDE` and `SHAFT_JOINT_FALLBACK`, four of
  `shaft_joint_problem`; two each of `JOINT_SIDES`, `window_quotes`, `joint_side`, `apart_from` and `shaft_joints`;
  one each of the nine input constants, `shaft_joint_links`, `rounded_up` and `rounded_down`.

(Beyond the compile errors, measured on the prototype: with this task's code in place but its registrations taken out
(no `DASHBOARD` row, no `CHECKS`, `END_EFFECT_ROWS` or `STORED_3D_ROWS` entry, no `CHECK_GUIDES` or `ROW_LEARN` entry;
every helper and `verdict_level` arm kept), 27 lib tests fail by assertion:
- `dashboard` (9): `the_dashboard_starts_with_the_headline_in_order`,
  `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`,
  `the_end_effect_rows_are_the_dashboard_rows_that_move_with_the_end_effect_coefficient`,
  `the_default_dashboard_has_badges_markers_and_the_3d_rows`, `the_stored_3d_label_is_drawn_once_under_the_shaft_joints_row`,
  `out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter`,
  `out_of_range_end_effect_greys_the_shaft_joint_check_whatever_its_text_reads`,
  `while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard`,
  `the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour`;
- `guidance` (12): `the_gearbox_check_is_a_dashboard_row_with_its_guide`,
  `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`, `every_fix_moves_its_check_toward_passing_over_its_slider`,
  `two_failing_checks_that_pull_one_input_apart_say_so`, `the_problems_quote_numbers_from_the_results`,
  `the_shaft_joint_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made`,
  `the_shaft_joint_problem_quotes_the_live_numbers`, `the_shaft_joint_texts_never_order_the_torques`,
  `the_design_status_counts_the_failing_checks`, `a_design_failing_only_the_clamp_passes_while_the_section_is_paused`,
  `the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not`,
  `the_shaft_joint_check_counts_in_the_status_whether_the_clamp_section_shows_or_not`;
- `panel` (3): `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too`,
  `the_status_line_counts_the_checks_at_default_and_after_a_fix`,
  `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`;
- `result_groups` (1): `the_headline_comes_first_then_the_chains_then_the_other_results`;
- `spreadsheet` (2): both Summary tests.)

- [ ] **Step 3: The row, the check, its levels, its greying and the stored-3D label (`dashboard.rs`)**

(1) The module doc (spec 5.1; the stored-3D bullet names the joint row, owner R-15):

At `:5`, replace

```rust
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the gearbox
//! check (BL-059), the A1 space claim badge and the audit M9 end-effect flag. A row's badge
//! comes from a check verdict ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
```

with

```rust
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the shaft
//! joints check (BL-064), the gearbox check (BL-059), the A1 space claim badge and the audit M9
//! end-effect flag. A row's badge comes from a check verdict ([`verdict_level`]); its marker
//! from the deviation registry ([`crate::gui::corrections`]). Two notes from the user's
//! decisions of 2026-09-30:
```

At `:14`, replace

```rust
//! - **Stored 3D values.** The temperature rows that read the stored 3D fields
//!   ([`STORED_3D_ROWS`]) carry the label [`STORED_3D_LABEL`]: the fields
//!   ([`STORED_3D_INPUTS`]) are 3D results for the example design and stay fixed when the
//!   design changes; no live 3D model runs.
```

with

```rust
//! - **Stored 3D values.** The temperature rows and the shaft joints row that read the stored
//!   3D fields ([`STORED_3D_ROWS`]; the shaft joints row through the inner ring's single-ring
//!   onset, which caps the bonded inner hub's fitting temperature, owner R-15) carry the label
//!   [`STORED_3D_LABEL`]: the fields ([`STORED_3D_INPUTS`]) are 3D results for the example
//!   design and stay fixed when the design changes; no live 3D model runs.
```

(2) The import:

At `:48`, replace

```rust
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
```

with

```rust
use crate::engine::model::END_EFFECT_OUT_OF_RANGE;
use crate::engine::shaft_joint;
```

(3) `DASHBOARD` 18 -> 19, the row at index 15, after the hidden clamp row and before the gearbox row (spec 5.1). The
path stays a literal here, in `CHECKS` and in `verdict_level`, as every other check's does (BL-059's R-8).

At `:155`, replace

```rust
pub const DASHBOARD: [(&str, Option<&str>); 18] = [
```

with

```rust
pub const DASHBOARD: [(&str, Option<&str>); 19] = [
```

At `:179`, replace

```rust
    ("clamps.recommended", Some("clamps.recommended")),
```

with

```rust
    ("clamps.recommended", Some("clamps.recommended")),
    ("shaft_joint.check", Some("shaft_joint.check")),
```

(4) `END_EFFECT_ROWS` 8 -> 9 (spec 5.1; review 2 SF-13: the whole row greys):

At `:190`, replace

```rust
/// headline's and the gearbox check, computed from cold-high. Per-row greying of the results
/// table waits for plan A-3's dependency graph; the table shows the banner instead (decision
/// M41-11), and its three checks among these rows (the hot minimum, the recommended screw, the
/// gearbox check) give no badge there either ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 8] = [
```

with

```rust
/// headline's and the shaft joints and gearbox checks, both computed from cold-high. The whole
/// shaft joints row greys, its stress and fitting terms too, which do not read cold-high: one
/// row cannot grey in part, and the end-effect check fails there under its banner (BL-064).
/// Per-row greying of the results table waits for plan A-3's dependency graph; the table shows
/// the banner instead (decision M41-11), and its four checks among these rows (the hot minimum,
/// the recommended screw, the shaft joints, the gearbox check) give no badge there either
/// ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 9] = [
```

At `:201`, replace

```rust
    "clamps.recommended",
    "gearbox.cold_high_check",
```

with

```rust
    "clamps.recommended",
    "shaft_joint.check",
    "gearbox.cold_high_check",
```

(5) `STORED_3D_ROWS` 3 -> 4 (owner R-15) and `STORED_3D_NOTE`'s new sentence (spec 5.1's draft; Task 8 sets the
words), and the `STORED_3D_INPUTS` doc (review 4 SF-4):

At `:206`, replace

```rust
/// demagnetization check and the slip-loss fields): the headline rows that move with them (a
/// test probes each at its slider ends).
pub const STORED_3D_ROWS: [&str; 3] = [
    "temperature.summary.governing_limit_C",
    "temperature.summary.margin_hot_day_C",
    TEMPERATURE_VERDICT,
```

with

```rust
/// demagnetization check and the slip-loss fields): the dashboard rows that move with them (a
/// test probes each at its slider ends). The three temperature rows, and the shaft joints row:
/// the bonded inner hub's allowed fitting temperature is capped at the inner ring's single-ring
/// onset less the cure margin (owner R-15), and the onset follows the stored single-ring
/// reverse field.
pub const STORED_3D_ROWS: [&str; 4] = [
    "temperature.summary.governing_limit_C",
    "temperature.summary.margin_hot_day_C",
    TEMPERATURE_VERDICT,
    "shaft_joint.check",
```

At `:218`, replace

```rust
pub const STORED_3D_NOTE: &str = "The temperature checks use the demagnetization reverse fields and the slip-heating fields, computed in 3D for the example design. Those fields stay fixed when you change the geometry, magnet count or grade; for another design, enter its own 3D fields to check the limits.";
```

with

```rust
pub const STORED_3D_NOTE: &str = "The temperature checks use the demagnetization reverse fields and the slip-heating fields, computed in 3D for the example design. The bonded inner hub's fitting limit on the shaft joints row uses one of the reverse fields too. Those fields stay fixed when you change the geometry, magnet count or grade; for another design, enter its own 3D fields to check the limits.";
```

At `:223`, replace

```rust
/// ([`STORED_3D_SENTENCE`]); [`STORED_3D_ROWS`] are the headline rows that move with them.
```

with

```rust
/// ([`STORED_3D_SENTENCE`]); [`STORED_3D_ROWS`] are the dashboard rows that move with them:
/// the temperature rows and the shaft joints row.
```

(6) `CHECKS` 31 -> 32, the joint check last (schema order: its group follows `gearbox`):

At `:259`, replace

```rust
/// screens, the material warnings, the space claim and the gearbox check. Each gives a badge
/// level ([`verdict_level`]): the results table shows it, and its failing filter lists the
/// checks that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps'
/// status texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 31] = [
```

with

```rust
/// screens, the material warnings, the space claim, the gearbox check and the shaft joints
/// check. Each gives a badge level ([`verdict_level`]): the results table shows it, and its
/// failing filter lists the checks that fail ([`failing_checks`]). The clamp table's per-size
/// checks and the sweeps' status texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 32] = [
```

At `:293`, replace

```rust
    "housing.space_claim_check",
    "gearbox.cold_high_check",
];
```

with

```rust
    "housing.space_claim_check",
    "gearbox.cold_high_check",
    "shaft_joint.check",
];
```

(7) `verdict_level` (spec 4.2, 5.1): the five failing prefixes as one list, read by one arm, then the pass and the
unknown prefix (every unknown form starts with it):

At `:306`, replace

```rust
    ];
    match (path, text) {
```

with

```rust
    ];
    // BL-064: each failing shaft joint text starts with its cause's constant.
    const SHAFT_JOINT_FAILS: [&str; 5] = [
        shaft_joint::FIT_SLIPS,
        shaft_joint::BORE_OVER_ITS_LIMIT,
        shaft_joint::CANNOT_BE_FITTED,
        shaft_joint::KEY_OVER_ITS_LIMIT,
        shaft_joint::KEY_HAS_NO_BEARING_AREA,
    ];
    match (path, text) {
```

At `:383`, replace

```rust
        }
        ("model.end_effect_check", "OK") => Some(Good),
```

with

```rust
        }
        ("shaft_joint.check", shaft_joint::SHAFT_JOINTS_PASS) => Some(Good),
        // Every unknown form: not a number, a safety factor not above zero (ruling C-4), no
        // wall around a bore, a reversed band.
        ("shaft_joint.check", t) if t.starts_with(shaft_joint::SHAFT_JOINT_UNKNOWN) => {
            Some(Caution)
        }
        ("shaft_joint.check", t) if SHAFT_JOINT_FAILS.iter().any(|&f| t.starts_with(f)) => {
            Some(Bad)
        }
        ("model.end_effect_check", "OK") => Some(Good),
```

(8) The `dashboard_ui` doc (spec 5.1: the label follows the last stored-3D row, now the joint row; no code change, it
already uses `rposition`):

At `:684`, replace

```rust
/// block over its Why line, the stored-3D label after the last temperature row, and the
```

with

```rust
/// block over its Why line, the stored-3D label after the last stored-3D row, and the
```

- [ ] **Step 4: The guide (`guidance.rs`)**

(1) The module doc:

At `:16`, replace

```rust
//! hottest-rated library part). The temperature and space-claim guides pick their fixes by the
//! terms of their verdicts that fail ([`Fixes::PerTerm`]: each temperature limit, ring by
//! ring; each axis of the claim); every other guide has one list. A fix is shown only where
//! its move helps: one slider step, or one other choice of its input alone (those two picks
//! count as one), that turns no passing guided check red; a fix picked per term must also
//! raise that term's margin, while each fix of a static list raises its check from a failing
//! design to its slider's end (a test moves each there). No move leaves a ring's magnet
//! without a rating, since the engine does not calibrate such a magnet's limit. These checks
//! compute the design each move reaches, so the panel asks through a [`FixCache`]. A failing
//! check left without a fix says so ([`NO_FIX`]), and two failing checks that pull one input
//! both ways each say so ([`trade_offs`]), as does a fix that sets back another failing check
//! ([`side_effects`]). A fix's text takes the input's label from the
//! [`InputCatalogue`], or the Size tab's name for it ([`fix_text`]), so no guide types a
//! label. The tests hold the fixes to helping: on each test design that fails a check, every
//! failing term of the check has a shown fix that moves it toward passing, and no shown fix
//! turns another check red.
```

with

```rust
//! hottest-rated library part). The temperature, space-claim and shaft joints guides pick their
//! fixes by the terms of their verdicts that fail ([`Fixes::PerTerm`]: each temperature limit,
//! ring by ring; each axis of the claim; each term of each shaft joint); every other guide has
//! one list. A fix is shown only where its move helps: one slider step, or one other choice of
//! its input alone (those two picks count as one), that turns no passing guided check red; a
//! fix picked per term must also raise that term's margin, while each fix of a static list
//! raises its check from a failing design to its slider's end (a test moves each there). No
//! move leaves a ring's magnet without a rating, since the engine does not calibrate such a
//! magnet's limit. These checks compute the design each move reaches, so the panel asks through
//! a [`FixCache`]. A failing check left without a fix says so ([`NO_FIX`]), and two failing
//! checks that pull one input both ways each say so ([`trade_offs`]), as does a fix that sets
//! back another failing check ([`side_effects`]). A fix's text takes the input's label from the
//! [`InputCatalogue`], or the Size tab's name for it ([`fix_text`]), so no guide types a label.
//! The tests hold the fixes to helping: on each test design that fails a check, every failing
//! term of the check has a shown fix that moves it toward passing, and no shown fix turns
//! another check red.
```

(2) The imports:

At `:43`, replace

```rust

use crate::engine::clamps::SCREW_SIZES;
```

with

```rust

use crate::engine::api::shaft_joint_links;
use crate::engine::clamps::SCREW_SIZES;
```

At `:51`, replace

```rust
};
use crate::engine::size_controls::{
```

with

```rust
};
use crate::engine::shaft_joint::{self, Cause, HubFitLimit, JointTerms, NoWall};
use crate::engine::size_controls::{
```

(3) The `Fixes::PerTerm` doc:

At `:114`, replace

```rust
    /// the term it serves (the temperature and space-claim guides), with the shaft-clamp
    /// section's pause ([`helps`]).
```

with

```rust
    /// the term it serves (the temperature, space-claim and shaft joints guides), with the
    /// shaft-clamp section's pause ([`helps`]).
```

(4) The check constant:

At `:146`, replace

```rust
const CLAMP: &str = "clamps.recommended";
```

with

```rust
const CLAMP: &str = "clamps.recommended";
const SHAFT_JOINT: &str = "shaft_joint.check";
```

(5) After `CLAMP_GUIDE` (spec 5.2): the guide, the inputs it names, the two joints as the guide reads and names them,
and the fixes per term with the three held back by rule. `shaft_joints` rebuilds the engine's joints from the design's
inputs and results (`shaft_joint_links`, Task 2), so the guide judges what `compute_all` judged; `None` while the check
cannot be made. The margin each fix is checked against is the engine's `JointTerms::margin` (spec 5.2's margins).

At `:216`, replace

```rust

const GEARBOX_GUIDE: CheckGuide = CheckGuide {
```

with

```rust

const SHAFT_JOINT_GUIDE: CheckGuide = CheckGuide {
    check: SHAFT_JOINT,
    label: "Shaft joints",
    problem: shaft_joint_problem,
    why: "A shrink fit's grip holds each rotor on its shaft; it must carry cold-high with margin when loosest, stay elastic when tightest, and be fitted within the allowed temperatures.",
    fixes: Fixes::PerTerm(shaft_joint_fixes),
    learn: Learn::None("No note covers the shaft joints yet."),
};

// The shaft joints' inputs the guide names (BL-064).
const HUB_MIN: &str = "shaft_joint.hub_interference_min_um";
const HUB_MAX: &str = "shaft_joint.hub_interference_max_um";
const CUP_MIN: &str = "shaft_joint.cup_interference_min_um";
const CUP_MAX: &str = "shaft_joint.cup_interference_max_um";
const KEY_CONTACT: &str = "shaft_joint.key_contact_mm";
const WHEEL_SHAFT: &str = "shaft_joint.wheel_shaft_fit_C";
const GEARBOX_SHAFT: &str = "shaft_joint.output_shaft_fit_C";
const HUB_LENGTH: &str = "metal.hub_length_mm";
const BOSS_LENGTH: &str = "metal.boss_length_mm";

/// One shaft joint as its guide reads and names it (BL-064), in the engine's order
/// ([`shaft_joint::joints`]: the inner hub, then the cup; a test holds the two to it).
struct JointSide {
    /// The joint's name in the engine's texts ([`shaft_joint::INNER_HUB`], [`shaft_joint::CUP`]).
    name: &'static str,
    /// The shaft it is fitted on.
    shaft: &'static str,
    /// Its band: the minimum and the maximum measured interference (inputs).
    min: &'static str,
    max: &'static str,
    /// Its shaft's temperature at fitting (an input).
    shaft_fit: &'static str,
    /// Its results: the holding torque, the bore stress, the key pressure, the window's edges
    /// (the smallest minimum that holds, the largest maximum within the stress limit and the
    /// largest that can be fitted), the fitting temperature needed and the one allowed.
    holding: &'static str,
    stress: &'static str,
    key: &'static str,
    needed: &'static str,
    elastic: &'static str,
    fittable: &'static str,
    fit_temp: &'static str,
    fit_allowed: &'static str,
    /// The fixes of each of its terms ([`JointSide::fixes`]; spec 5.2).
    slip_fixes: &'static [(&'static str, Direction)],
    stress_fixes: &'static [(&'static str, Direction)],
    fitting_fixes: &'static [(&'static str, Direction)],
    key_fixes: &'static [(&'static str, Direction)],
}

impl JointSide {
    /// The fixes of the joint's `cause` term.
    const fn fixes(&self, cause: Cause) -> &'static [(&'static str, Direction)] {
        match cause {
            Cause::Slips => self.slip_fixes,
            Cause::BoreStress => self.stress_fixes,
            Cause::CannotBeFitted => self.fitting_fixes,
            Cause::Key => self.key_fixes,
        }
    }
}

/// The two shaft joints. A larger face gap lowers cold-high, and with it the torque each fit
/// must hold; shorter magnets do too, but only the cup is offered them, since the inner hub
/// lengthens with its magnets. Not offered (spec 5.2): the assumptions (friction, smoothing,
/// the rotor-over-shaft allowance, the joining clearance), the safety factor, the allowed
/// fitting temperatures, the back iron, the keyway depth and the magnets' part or grade.
const JOINT_SIDES: [JointSide; 2] = [
    JointSide {
        name: shaft_joint::INNER_HUB,
        shaft: "wheel shaft",
        min: HUB_MIN,
        max: HUB_MAX,
        shaft_fit: WHEEL_SHAFT,
        holding: "shaft_joint.hub_holding_Nm",
        stress: "shaft_joint.hub_bore_stress_MPa",
        key: "shaft_joint.hub_key_pressure_MPa",
        needed: "shaft_joint.hub_needed_um",
        elastic: "shaft_joint.hub_elastic_max_um",
        fittable: "shaft_joint.hub_fittable_max_um",
        fit_temp: "shaft_joint.hub_fit_temp_C",
        fit_allowed: "shaft_joint.hub_fit_allowed_C",
        slip_fixes: &[
            (HUB_MIN, Direction::Raise),
            (HUB_LENGTH, Direction::Raise),
            (FACE_GAP, Direction::Raise),
        ],
        stress_fixes: &[(HUB_MAX, Direction::Lower)],
        fitting_fixes: &[(HUB_MAX, Direction::Lower), (WHEEL_SHAFT, Direction::Lower)],
        key_fixes: &[
            (KEY_CONTACT, Direction::Raise),
            (HUB_LENGTH, Direction::Raise),
        ],
    },
    JointSide {
        name: shaft_joint::CUP,
        shaft: "gearbox shaft",
        min: CUP_MIN,
        max: CUP_MAX,
        shaft_fit: GEARBOX_SHAFT,
        holding: "shaft_joint.cup_holding_Nm",
        stress: "shaft_joint.cup_bore_stress_MPa",
        key: "shaft_joint.cup_key_pressure_MPa",
        needed: "shaft_joint.cup_needed_um",
        elastic: "shaft_joint.cup_elastic_max_um",
        fittable: "shaft_joint.cup_fittable_max_um",
        fit_temp: "shaft_joint.cup_fit_temp_C",
        fit_allowed: "shaft_joint.cup_fit_allowed_C",
        slip_fixes: &[
            (CUP_MIN, Direction::Raise),
            (BOSS_LENGTH, Direction::Raise),
            (MAGNET_LENGTH, Direction::Lower),
            (FACE_GAP, Direction::Raise),
        ],
        stress_fixes: &[(CUP_MAX, Direction::Lower)],
        fitting_fixes: &[
            (CUP_MAX, Direction::Lower),
            (GEARBOX_SHAFT, Direction::Lower),
        ],
        key_fixes: &[
            (KEY_CONTACT, Direction::Raise),
            (BOSS_LENGTH, Direction::Raise),
        ],
    },
];

/// The shaft joints of a design as the engine judges them, inner hub first, rebuilt from its
/// inputs and results ([`shaft_joint_links`]); `None` while the check cannot be made (a value
/// it reads is not a number, or the safety factor is not above zero: ruling C-4), where no term
/// is judged.
fn shaft_joints(inputs: &DesignInputs, results: &DesignResults) -> Option<[JointTerms; 2]> {
    let check = results.shaft_joint.check.as_str();
    (check != shaft_joint::UNKNOWN_NOT_A_NUMBER && check != shaft_joint::UNKNOWN_SAFETY_FACTOR)
        .then(|| shaft_joint::joints(&inputs.shaft_joint, &shaft_joint_links(inputs, results)))
}

/// The index in [`JOINT_SIDES`] of the joint named `name`, and its side.
fn joint_side(name: &str) -> Option<(usize, &'static JointSide)> {
    JOINT_SIDES
        .iter()
        .enumerate()
        .find(|(_, side)| side.name == name)
}

/// The shaft joints' fixes ([`Fixes::PerTerm`]): for each term a judged joint fails, in the
/// badge's order ([`shaft_joint::failing`]), each of that joint's fixes for the term whose move
/// helps ([`helps`] against the term's margin, [`JointTerms::margin`]), each fix once; none
/// while the check cannot be made. Three are held back by rule (spec 5.2): raising a band's
/// minimum or lowering its maximum where one step puts the minimum above the maximum (the row
/// would read amber, and the margin's rise would count as help), and raising the key's contact
/// height for a joint whose key has no bearing length while its contact height is above zero
/// (no contact height gives that key any bearing area).
fn shaft_joint_fixes(
    inputs: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
) -> Vec<(&'static str, Direction)> {
    let Some(joints) = shaft_joints(inputs, results) else {
        return Vec::new();
    };
    let mut shown: Vec<(&'static str, Direction)> = Vec::new();
    for (cause, joint) in shaft_joint::failing(&joints) {
        let Some((index, side)) = joint_side(joint.name) else {
            continue;
        };
        let margin = |design: &DesignInputs, after: &DesignResults| {
            shaft_joint::joints(&design.shaft_joint, &shaft_joint_links(design, after))[index]
                .margin(cause)
        };
        for &fix in side.fixes(cause) {
            let contact_without_length = fix == (KEY_CONTACT, Direction::Raise)
                && joint.key_bearing_mm <= 0.0
                && joint.key_contact_mm > 0.0;
            if !shown.contains(&fix)
                && !contact_without_length
                && !reverses_the_band(inputs, results, side, fix)
                && helps(inputs, results, fix, Some(&margin), clamps_paused)
            {
                shown.push(fix);
            }
        }
    }
    shown
}

/// Whether one move of `fix`, an input of the band of `side`, puts the band's minimum above its
/// maximum; `false` for any other input.
fn reverses_the_band(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    (path, direction): (&str, Direction),
) -> bool {
    if path != side.min && path != side.max {
        return false;
    }
    moves(inputs, results, path, direction)
        .iter()
        .any(|design| {
            let band = |path: &str| design.get(path).and_then(number).unwrap_or(f64::NAN);
            band(side.min) > band(side.max)
        })
}

const GEARBOX_GUIDE: CheckGuide = CheckGuide {
```

(6) `CHECK_GUIDES` 9 -> 10, the joint's guide between the clamp's and the gearbox's (spec 5.2), and its `ROW_LEARN` entry
(it reads the guide's `learn`, so Task 7's flip of the guide changes the row's link too):

At `:309`, replace

```rust
/// The guides: the eight dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 9] = [
```

with

```rust
/// The guides: the nine dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 10] = [
```

At `:317`, replace

```rust
    CLAMP_GUIDE,
```

with

```rust
    CLAMP_GUIDE,
    SHAFT_JOINT_GUIDE,
```

At `:363`, replace

```rust
    (CLAMP, CLAMP_GUIDE.learn),
```

with

```rust
    (CLAMP, CLAMP_GUIDE.learn),
    (SHAFT_JOINT, SHAFT_JOINT_GUIDE.learn),
```

(7) `cure_margin_texts` reads the shared `shown_decimals` (DRY with the onset's quoting below; plan deviation PD-24; its
tests are unchanged):

At `:1379`, replace

```rust
    const MARGIN: &str = "temperature.summary.cure_margin_C";
    let onset = format_value(&Value::Num(results.temperature.demag.onset_single_ring_C));
```

with

```rust
    const MARGIN: &str = "temperature.summary.cure_margin_C";
```

At `:1388`, replace

```rust
    match onset.split_once('.') {
        Some((_, decimals)) if !onset.contains('e') && margin.is_finite() => {
            let shown = fixed(margin, decimals.len());
```

with

```rust
    match shown_decimals(results.temperature.demag.onset_single_ring_C) {
        Some(decimals) if margin.is_finite() => {
            let shown = fixed(margin, decimals);
```

At `:1399`, replace

```rust
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals.len()), unit),
```

with

```rust
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals), unit),
```

(8) The `design_status_text` doc (spec 7.17):

At `:1502`, replace

```rust
/// The design status line: "Example design: fails 4 of 9 checks", "Your design: fails
/// 3 of 9 checks", "Your design: passes all 9 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 8 while the shaft-clamp section is paused.
```

with

```rust
/// The design status line: "Example design: fails 4 of 10 checks", "Your design: fails
/// 3 of 10 checks", "Your design: passes all 10 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 9 while the shaft-clamp section is paused.
```

(9) `quoted_limit` and `quoted_load` read the shared `rounded_down` and `rounded_up`, which the window's ends use too,
and `shown_decimals` follows them (plan deviation PD-24):

At `:1575`, replace

```rust
    let step = last_digit_step(x);
    let down = (x / step + LAST_DIGIT_NOISE).floor() * step;
```

with

```rust
    let down = rounded_down(x);
```

At `:1593`, replace

```rust
    let step = last_digit_step(x);
    let up = (x / step - LAST_DIGIT_NOISE).ceil() * step;
```

with

```rust
    let up = rounded_up(x);
```

At `:1601`, replace

```rust

/// The most decimals [`band_quotes`] gives a number near one: f64 holds no more of its digits.
```

with

```rust

/// `x` rounded down at its last shown digit, float noise aside ([`LAST_DIGIT_NOISE`]); a
/// number that is not finite or not positive as it is.
fn rounded_down(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    let step = last_digit_step(x);
    (x / step + LAST_DIGIT_NOISE).floor() * step
}

/// `x` rounded up at its last shown digit, float noise aside ([`LAST_DIGIT_NOISE`]); a number
/// that is not finite or not positive as it is.
fn rounded_up(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return x;
    }
    let step = last_digit_step(x);
    (x / step - LAST_DIGIT_NOISE).ceil() * step
}

/// How many decimals the readouts show for `x` ([`format_value`]'s text): `None` where that
/// text has no fixed decimals (an exponent, a whole number, or not a number).
fn shown_decimals(x: f64) -> Option<usize> {
    let text = format_value(&Value::Num(x));
    match text.split_once('.') {
        Some((_, decimals)) if !text.contains('e') => Some(decimals.len()),
        _ => None,
    }
}

/// The most decimals [`band_quotes`] gives a number near one: f64 holds no more of its digits.
```

(10) After `clamp_problem` (spec 5.2, 4.3): the Problem. While the check cannot be made, its sentence alone (a value
that is not a number; or, ruling C-4, a safety factor not above zero: "The safety factor, {its value}, is not above zero:
enter at least 1.", with no fix); else a
sentence per failing term of a judged joint in the badge's order (`shaft_joint::failing`), then each judged joint's
window (or why there is none) where it fails a slip, stress or fitting term, then why a joint is not judged. A failing
number never reads equal to its bound (`apart_from`, BL-059 MF-1); the window's ends are quoted inward with more decimals
where they would read equal or crossed (`window_quotes`, review SF-11, BL-059 NB-1); the inner hub's fitting source is
named by the engine's own `hub_fit_allowed` (owner R-15, spec 4.1: the number and its name from one place), the onset and
the margin with the allowed temperature's decimals (`onset_texts`, review 4 SF-5). The `None` arm of `fitting_source` is
not reached (the check reads unknown while any of its three values is not a number); it reads in plain words all the same.

At `:1723`, replace

```rust

/// The gearbox check's Problem when it cannot be made (it asks for a look): a value it
```

with

```rust

/// The shaft joints' Problem when the check cannot be made: a value it reads is not a number.
const SHAFT_JOINT_NOT_A_NUMBER: &str =
    "The shaft joint check cannot be made: a value it reads is not a number.";

/// The shaft joints' Problem when no sentence reads as failing although the check does
/// ([`joined_or`]).
const SHAFT_JOINT_FALLBACK: &str = "A shaft joint does not pass: see the shaft joint results.";

/// The shaft joints' Problem (spec 5.2): while the check cannot be made (a value that is not a
/// number, or a safety factor not above zero: ruling C-4), its sentence alone; else a
/// sentence per term a judged joint fails, in the badge's order ([`shaft_joint::failing`]),
/// with its live numbers; then, for each judged joint failing a slip, stress or fitting term,
/// the window of measured interference that would work, or why there is none
/// ([`window_sentence`]); then each joint that is not judged, and why
/// ([`not_judged_sentences`]).
fn shaft_joint_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    if !failing(SHAFT_JOINT, inputs, results) {
        return None;
    }
    // A safety factor typed at or below zero (ruling C-4): its sentence alone, with no fix.
    if results.shaft_joint.check == shaft_joint::UNKNOWN_SAFETY_FACTOR {
        return Some(format!(
            "The safety factor, {}, is not above zero: enter at least 1.",
            input_text(inputs, "shaft_joint.safety_factor")
        ));
    }
    let Some(joints) = shaft_joints(inputs, results) else {
        return Some(SHAFT_JOINT_NOT_A_NUMBER.to_owned());
    };
    let mut sentences: Vec<String> = shaft_joint::failing(&joints)
        .into_iter()
        .filter_map(|(cause, joint)| {
            let (_, side) = joint_side(joint.name)?;
            Some(term_sentence(inputs, results, side, joint, cause))
        })
        .collect();
    for (side, joint) in JOINT_SIDES.iter().zip(&joints) {
        let windowed = joint.determinate()
            && [Cause::Slips, Cause::BoreStress, Cause::CannotBeFitted]
                .into_iter()
                .any(|cause| joint.fails(cause));
        if windowed {
            sentences.extend(window_sentence(inputs, results, side));
        }
    }
    for (side, joint) in JOINT_SIDES.iter().zip(&joints) {
        sentences.extend(not_judged_sentences(inputs, side, joint));
    }
    Some(joined_or(sentences, SHAFT_JOINT_FALLBACK))
}

/// The sentence of a term `cause` the joint of `side` fails, its numbers live, a failing value
/// never read equal to its bound ([`apart_from`]).
fn term_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    joint: &JointTerms,
    cause: Cause,
) -> String {
    let name = side.name;
    match cause {
        Cause::Slips => format!(
            "At its loosest the {name} holds {}, less than the {} it must hold: cold-high, {}, times the safety factor of {}.",
            apart_from(results, side.holding, joint.required_Nm, false),
            result_text(results, "shaft_joint.required_Nm"),
            result_text(results, "metal.torque_cold_high_Nm"),
            input_text(inputs, "shaft_joint.safety_factor"),
        ),
        Cause::BoreStress => format!(
            "At its tightest the {name}'s bore stress is {}, above the {} limit ({} yield with a margin).",
            apart_from(results, side.stress, joint.stress_limit_MPa, true),
            result_text(results, "shaft_joint.stress_limit_MPa"),
            result_text(results, "shaft_joint.part_yield_MPa"),
        ),
        Cause::CannotBeFitted => format!(
            "Fitting the {name} would need it at {} with the {} at {}, above the {} allowed: {}.",
            apart_from(results, side.fit_temp, joint.fit_allowed_C, true),
            side.shaft,
            input_text(inputs, side.shaft_fit),
            result_text(results, side.fit_allowed),
            fitting_source(inputs, results, side),
        ),
        Cause::Key => key_sentence(inputs, results, side, joint),
    }
}

/// What sets the allowed fitting temperature of the joint of `side`, with its live number (trap
/// 9): the cup's is your limit for the bare cup; the bonded inner hub's is the lowest of three
/// (owner R-15), named by the engine's own rule ([`shaft_joint::hub_fit_allowed`]), so the
/// number and its name come from one place.
fn fitting_source(inputs: &DesignInputs, results: &DesignResults, side: &JointSide) -> String {
    if side.name != shaft_joint::INNER_HUB {
        return "your limit for the bare cup".to_owned();
    }
    let (_, limit) = shaft_joint::hub_fit_allowed(
        inputs.shaft_joint.hub_fit_max_C,
        results.temperature.adhesive.design_limit_C,
        results.temperature.demag.inner_onset_single_ring_C,
    );
    match limit {
        Some(HubFitLimit::Yours) => format!(
            "your limit for the bonded inner hub, {}",
            input_text(inputs, "shaft_joint.hub_fit_max_C")
        ),
        Some(HubFitLimit::Adhesive) => format!(
            "the selected adhesive's design limit, {}",
            result_text(results, "temperature.adhesive.design_limit_C")
        ),
        Some(HubFitLimit::MagnetOnset) => {
            let (onset, margin) = onset_texts(results);
            format!("the magnets' single-ring onset, {onset}, less {margin}")
        }
        // Not reached: the check cannot be made while one of the three is not a number.
        None => "a limit that is not a number".to_owned(),
    }
}

/// The inner ring's single-ring onset and the cure margin as the bonded inner hub's fitting
/// source quotes them (owner R-15), each with as many decimals as the allowed temperature
/// shows, so the three numbers subtract ("above the 59.83 °C allowed: the magnets' single-ring
/// onset, 69.83 °C, less 10.00 °C"), as [`cure_margin_texts`] quotes the cure margin. Both read
/// as any result does where the allowed temperature's text has no fixed decimals.
fn onset_texts(results: &DesignResults) -> (String, String) {
    const ONSET: &str = "temperature.demag.inner_onset_single_ring_C";
    match shown_decimals(results.shaft_joint.hub_fit_allowed_C) {
        Some(decimals) => {
            let unit = result_info(ONSET).map_or("", |info| info.meta.unit);
            let onset = results.temperature.demag.inner_onset_single_ring_C;
            (
                with_unit(fixed(onset, decimals), unit),
                with_unit(fixed(CURE_MARGIN_MIN_C, decimals), unit),
            )
        }
        None => (
            result_text(results, ONSET),
            number_text(CURE_MARGIN_MIN_C, ONSET),
        ),
    }
}

/// The key term's sentence of the joint of `side` (spec 5.2): no contact height (it alone, where
/// the key has no bearing length either: no length gives it bearing area without contact); no
/// bearing length; else the key's pressure, alone, above the limit.
fn key_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
    joint: &JointTerms,
) -> String {
    let name = side.name;
    let key_width = input_text(inputs, "shaft_joint.key_width_mm");
    if joint.key_contact_mm <= 0.0 {
        format!(
            "The key has no bearing area in the {name}: its contact height, {}, is not above zero.",
            input_text(inputs, KEY_CONTACT)
        )
    } else if joint.key_bearing_mm <= 0.0 && name == shaft_joint::INNER_HUB {
        format!(
            "The key has no bearing area in the inner hub: the hub, {} long, is no longer than the key is wide, {key_width}.",
            result_text(results, "housing.hub_length_mm")
        )
    } else if joint.key_bearing_mm <= 0.0 {
        format!(
            "The key has no bearing area in the cup: its boss and web, {} and {} long, together are no longer than the key is wide, {key_width}.",
            input_text(inputs, BOSS_LENGTH),
            input_text(inputs, "metal.web_mm")
        )
    } else {
        format!(
            "Alone, the key would bear {} on the {name}, above the {} limit.",
            apart_from(results, side.key, joint.stress_limit_MPa, true),
            result_text(results, "shaft_joint.stress_limit_MPa")
        )
    }
}

/// The window of measured interference that would work on the joint of `side` (spec 5.2):
/// from the smallest minimum that holds to the smaller of the largest maximum within the stress
/// limit and the largest that can be fitted, its clause naming the edge that binds, beside the
/// joint's band; or, where the smallest minimum is above that, why no interference works, each
/// edge below it named. The ends are quoted inward ([`window_quotes`], [`rounded_up`],
/// [`rounded_down`]), so a band typed at them passes. `None` where the smallest minimum is not
/// finite (a key wider than the bore's circumference leaves no contact): the slip sentence's
/// holding of 0 says it.
fn window_sentence(
    inputs: &DesignInputs,
    results: &DesignResults,
    side: &JointSide,
) -> Option<String> {
    let value = |path: &str| number(results.get(path).unwrap_or(Value::None)).unwrap_or(f64::NAN);
    let (needed, elastic, fittable) = (
        value(side.needed),
        value(side.elastic),
        value(side.fittable),
    );
    if !needed.is_finite() {
        return None;
    }
    let name = side.name;
    let (upper, clause) = if elastic <= fittable {
        (elastic, "stay within the stress limit")
    } else {
        (fittable, "fit at the allowed temperature")
    };
    if needed <= upper {
        let (low, high) = window_quotes(needed, upper, side.needed);
        return Some(format!(
            "On the {name}, a measured interference from {low} to {high} would hold and {clause} (its band is {} to {}).",
            input_text(inputs, side.min),
            input_text(inputs, side.max)
        ));
    }
    let mut edges = Vec::new();
    if elastic < needed {
        edges.push(format!(
            "the bore stays within its limit only up to {}",
            number_text(rounded_down(elastic), side.elastic)
        ));
    }
    if fittable < needed && fittable <= 0.0 {
        edges.push(format!(
            "the allowed temperatures do not even open its {} joining clearance",
            input_text(inputs, "shaft_joint.joining_clearance_um")
        ));
    } else if fittable < needed {
        edges.push(format!(
            "it can be fitted only up to {}",
            number_text(rounded_down(fittable), side.fittable)
        ));
    }
    Some(format!(
        "No measured interference works on the {name} here: holding needs at least {}, but {}.",
        number_text(rounded_up(needed), side.needed),
        listed(&edges, "and")
    ))
}

/// A window's ends as the shaft joints' Problem quotes them, in the unit of the result at
/// `unit_of` (review SF-11, as BL-059 quotes its band): `needed` rounded up and `upper` rounded
/// down at each one's last shown digit, so a band typed at the quoted ends passes; where those
/// read equal or crossed (a window narrower than the readouts' last digit), both with more
/// decimals, as few as read them apart in order (BL-059 NB-1).
fn window_quotes(needed: f64, upper: f64, unit_of: &str) -> (String, String) {
    let (low, high) = (rounded_up(needed), rounded_down(upper));
    if quoted_value(low) < quoted_value(high) {
        return (number_text(low, unit_of), number_text(high, unit_of));
    }
    let at = |decimals: usize| {
        let step = 10f64.powi(-(decimals as i32));
        (
            (needed / step - LAST_DIGIT_NOISE).ceil() * step,
            (upper / step + LAST_DIGIT_NOISE).floor() * step,
        )
    };
    let first = [low, high]
        .into_iter()
        .filter_map(shown_decimals)
        .max()
        .map_or(SIGNIFICANT_DIGITS as usize, |decimals| decimals + 1);
    let (decimals, (low, high)) = (first..=MOST_DECIMALS)
        .map(|decimals| (decimals, at(decimals)))
        .find(|&(_, (low, high))| low < high)
        .unwrap_or_else(|| (MOST_DECIMALS, at(MOST_DECIMALS)));
    let unit = result_info(unit_of).map_or("", |info| info.meta.unit);
    (
        with_unit(fixed(low, decimals), unit),
        with_unit(fixed(high, decimals), unit),
    )
}

/// Why the joint of `side` is not judged, if it is not (spec 4.3): no wall around its bore (the
/// bore as wide as the part, or the keyway through the wall), its band reversed, or both.
fn not_judged_sentences(
    inputs: &DesignInputs,
    side: &JointSide,
    joint: &JointTerms,
) -> Vec<String> {
    let name = side.name;
    let bore = input_text(inputs, "coupling.bore_mm");
    let mut sentences = Vec::new();
    match joint.no_wall {
        Some(NoWall::BoreAsWideAsPart) if name == shaft_joint::INNER_HUB => sentences.push(format!(
            "The inner hub's bore, {bore}, reaches the flats its magnets are bonded on, so it leaves no wall to grip the wheel shaft."
        )),
        Some(NoWall::BoreAsWideAsPart) => sentences.push(format!(
            "The cup's bore, {bore}, is as wide as its boss, {}, so it leaves no wall to grip the gearbox shaft.",
            input_text(inputs, "metal.boss_od_mm")
        )),
        Some(NoWall::KeywayThroughWall) => sentences.push(format!(
            "The {name}'s keyway, {} deep, cuts through the wall around its {bore} bore, so the fit has no closed ring to grip the {}.",
            input_text(inputs, "coupling.keyway_depth_mm"),
            side.shaft
        )),
        None => {}
    }
    if joint.band_reversed {
        sentences.push(format!(
            "The {name}'s minimum interference, {}, is above its maximum, {}: enter the band's smaller value as the minimum.",
            input_text(inputs, side.min),
            input_text(inputs, side.max)
        ));
    }
    sentences
}

/// A failing value at `path` as a Problem quotes it beside its `bound` (BL-059 MF-1): as its
/// readout shows it, or, where it would read the same as the bound, one last digit past it: up
/// above a limit (`above`, [`quoted_load`]), down under a need ([`quoted_limit`]).
fn apart_from(results: &DesignResults, path: &str, bound: f64, above: bool) -> String {
    let x = number(results.get(path).unwrap_or(Value::None)).unwrap_or(f64::NAN);
    let reads_apart = if above {
        quoted_value(x) > quoted_value(bound)
    } else {
        quoted_value(x) < quoted_value(bound)
    };
    match (reads_apart, above) {
        (true, _) => result_text(results, path),
        (false, true) => number_text(quoted_load(x, bound), path),
        (false, false) => number_text(quoted_limit(x, bound), path),
    }
}

/// The gearbox check's Problem when it cannot be made (it asks for a look): a value it
```

(11) `gearbox_problem` reads `apart_from` (DRY: the same never-equal rule; plan deviation PD-24;
`sentence_1_never_reads_two_equal_numbers` and its neighbours pass unchanged):

At `:1732`, replace

```rust
        Level::Bad => {
            let torque = results.metal.cold_high_input_Nm;
```

with

```rust
        Level::Bad => {
```

At `:1738`, replace

```rust
            let torque_text = if quoted_value(torque) > quoted_value(rating) {
                result_text(results, "metal.cold_high_input_Nm")
            } else {
                number_text(quoted_load(torque, rating), "metal.cold_high_input_Nm")
            };
```

with

```rust
            let torque_text = apart_from(results, "metal.cold_high_input_Nm", rating, true);
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: `test result: ok. 905 passed; 0 failed; 6 ignored` (Task 4's 892 + 13 new tests; the stored-3D membership
test and the Summary test are renamed; measured on the integrated tree of Tasks 1 to 7).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count`

Expected: FAIL, `the_docs_count_the_dashboard_rows_and_the_checks` (README.md should say "dashboard's 19 rows"): the
README still counts 18. Step 6 fixes it.

- [ ] **Step 6: Docs**

Each edit below is a substring of one line (or a few lines) of the file at Task 4's commit; each Old text occurs once.

**6a. `magcoupling-rs/README.md`** (the dashboard paragraph, the status counts, the guides, the pause bullet's count,
the results table's row count, and the test table rows of `panel.rs`, `dashboard.rs`, `spreadsheet.rs` and
`guidance.rs`):

(1) Around `:302`. Old:

```text
  record: the trace then stays. Each group heading counts the rows the trace frames.
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the gearbox check
```

New:

```text
  record: the trace then stays. Each group heading counts the rows the trace frames.
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the shaft joints
  check (the keyed shrink fits that hold both rotors on their shafts, BL-064), the gearbox check
```

(2) Around `:313`. Old:

```text
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the temperature rows that read the stored 3D
  fields carry "Fixed 3D fields from the example design" (the fields are 3D results for the
```

New:

```text
  (audit M9) the rows computed from the pull-out (`END_EFFECT_ROWS`) are greyed without badges
  under the banner "End-effect model out of range"; the rows that read the stored 3D fields
  (`STORED_3D_ROWS`: the temperature rows, and the shaft joints row through the inner ring's
  single-ring onset, which caps the bonded inner hub's fitting temperature, owner R-15) carry
  "Fixed 3D fields from the example design", drawn once after the last of them (the fields are 3D results for the
```

(3) Around `:324`. Old:

```text
fails N of 9 checks" while the
```

New:

```text
fails N of 10 checks" while the
```

(4) Around `:326`. Old:

```text
fails N of 9 checks" or "passes
```

New:

```text
fails N of 10 checks" or "passes
```

(5) Around `:326`. Old:

```text
or "passes all 9 checks" after an
```

New:

```text
or "passes all 10 checks" after an
```

(6) Around `:327`. Old:

```text
line reads "of 8 checks": the example
```

New:

```text
line reads "of 9 checks": the example
```

(7) Around `:327`. Old:

```text
"fails 4 of 8 checks"). The nine guides are the
```

New:

```text
"fails 4 of 9 checks"). The ten guides are the
```

(8) Around `:328`. Old:

```text
  dashboard's eight checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
```

New:

```text
  dashboard's nine checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
```

(9) Around `:329`. Old:

```text
  the recommended clamp, the gearbox check, the space claim and the end-effect flag) and a **Magnets fit** row
```

New:

```text
  the recommended clamp, the shaft joints, the gearbox check, the space claim and the end-effect flag) and a **Magnets fit** row
```

(10) Around `:343`. Old:

```text
"Lower Outside diameter" is left out of every guide while the Size tab's Minimum OD already stops
the slider (`guidance::within_minimum_diameter`). While the four reverse-field inputs hold their
```

New:

```text
"Lower Outside diameter" is left out of every guide while the Size tab's Minimum OD already stops
the slider (`guidance::within_minimum_diameter`). The shaft joints' guide (BL-064) names each
failing joint and term with its live numbers, then the window of measured interference that would
work (its ends quoted inward, so a band typed at them passes) or why none does, and which limit
sets the bonded inner hub's allowed fitting temperature (your limit, the adhesive's design limit,
or the inner ring's single-ring onset less the cure margin); its fixes are picked per joint and
term (the band, the hub or boss length, the face gap, the cup's magnet length, a shaft's fitting
temperature, the key's contact height), holding back a band step that would reverse the band and
a contact height where the key has no bearing length. While the four reverse-field inputs hold their
```

(11) Around `:370`. Old:

```text
  dashboard's 18 rows, open; then the closed chain groups: the A-3 chains torque, temperature,
```

New:

```text
  dashboard's 19 rows, open; then the closed chain groups: the A-3 chains torque, temperature,
```

(12) Around `:471`. Old:

```text
status line counts 8 guides, a hidden
```

New:

```text
status line counts 9 guides, a hidden
```

(13) Around `:683`. Old:

```text
name and Why, a Why link's
```

New:

```text
name and Why, the shaft joints' Problem as drawn on three designs that fail them (BL-064: the inner hub slipping, over its stress limit, on an uncooled wheel shaft), a Why link's
```

(14) Around `:687`. Old:

```text
`HEADLINE`, then the gearbox check, the
```

New:

```text
`HEADLINE`, then the shaft joints check, the gearbox check, the
```

(15) Around `:687`. Old:

```text
is not a number); `END_EFFECT_ROWS` are exactly
```

New:

```text
is not a number; every text form of the shaft joints check reached as the badge by a design: the pass, its five failing causes, its four unknown forms); `END_EFFECT_ROWS` are exactly
```

(16) Around `:687`. Old:

```text
headline's and the gearbox check, `STORED_3D_ROWS` exactly those
```

New:

```text
headline's and the shaft joints and gearbox checks, `STORED_3D_ROWS` exactly those
```

(17) Around `:687`. Old:

```text
one of the 31 checks is a
```

New:

```text
one of the 32 checks is a
```

(18) Around `:687`. Old:

```text
named as a check is listed; every
```

New:

```text
named as a check (a text named check, or ending in _check, _screen, verdict or reading) is listed; every
```

(19) Around `:687`. Old:

```text
end-effect range the three checks among the
```

New:

```text
end-effect range the four checks among the
```

(20) Around `:687`. Old:

```text
minimum, the recommended screw, the gearbox check)
```

New:

```text
minimum, the recommended screw, the shaft joints, the gearbox check)
```

(21) Around `:687`. Old:

```text
and the gearbox check greys whatever its text reads (a cold-high below
```

New:

```text
and the gearbox and shaft joints checks grey whatever their texts read (a cold-high below
```

(22) Around `:687`. Old:

```text
below it reads above): no badge on
```

New:

```text
below it reads above; the shaft joints read a pass, a bore stress above its limit red, and their row keeps its stored-3D note): no badge on
```

(23) Around `:687`. Old:

```text
plain words, and every stored 3D
```

New:

```text
plain words, and the label is drawn once, under the shaft joints row, the last of the four stored-3D rows, whether the clamp section shows or not (owner R-15); every stored 3D
```

(24) Around `:687`. Old:

```text
clamp ("passes all 8 checks"); the failing
```

New:

```text
clamp ("passes all 9 checks"); the failing
```

(25) Around `:688`. Old:

```text
corrections applied); its 18 dashboard rows carry
```

New:

```text
corrections applied); its 19 dashboard rows carry
```

(26) Around `:688`. Old:

```text
check between the clamp's and the space
```

New:

```text
check between the shaft joints' and the space
```

(27) Around `:688`. Old:

```text
the same level (BL-059); the shaft joints' 15 inputs
```

New:

```text
the same level (BL-059); the shaft joints' row sits between the clamp's and the gearbox's with its level (green at the defaults and at the passing design, red where the inner hub slips) and the stored-3D label in its note column, the greyed note winning out of the end-effect range (BL-064, owner R-15); the shaft joints' 15 inputs
```

(Task 4's edit (g) put the shaft joints' Inputs and Results clause between "the same level (BL-059);" and "the
Assumptions sheet;", so the Old text anchors on its start: ruling C-1, A-2.)

(28) Around `:690`. Old:

```text
clamp passes all 8 counted checks while
```

New:

```text
clamp passes all 9 counted checks while
```

(29) Around `:690`. Old:

```text
fails 1 of 9 when not; a
```

New:

```text
fails 1 of 10 when not; a
```

(30) Around `:690`. Old:

```text
right after the headline's, in `CHECKS`, with
```

New:

```text
right after the shaft joints', in `CHECKS`, with
```

(31) Around `:690`. Old:

```text
guide between the clamp's and the space
```

New:

```text
guide between the shaft joints' and the space
```

(32) Around `:690`. Old:

```text
fails 4 of 8 checks while paused
```

New:

```text
fails 4 of 9 checks while paused
```

(33) Around `:690`. Old:

```text
and 4 of 9 with it; the
```

New:

```text
and 4 of 10 with it; the
```

(34) Around `:690`. Old:

```text
N·m, passes all 8, and with the
```

New:

```text
N·m, passes all 9, and with the
```

(35) Around `:690`. Old:

```text
end (200 points). |
```

New:

```text
end (200 points). BL-064 (T5): the shaft joints check is one dashboard row, after the headline's and before the gearbox's, last in `CHECKS`, with a guide between the clamp's and the gearbox's, labelled "Shaft joints", counted while the clamp section is paused, whose Why link is its row's (`the_shaft_joint_check_is_a_dashboard_row_with_its_guide`); the guide's two joints are the engine's, each result it reads the engine's term bit for bit; its Problem shows exactly when the check fails or cannot be made (not greyed: a bore stress above its limit on short magnets shows none), reaches every sentence form (each cause on each joint, the key's three, the window and the empty window's three edges, the three sources of the inner hub's allowed fitting temperature, not a number alone, a safety factor not above zero alone, no wall twice, a reversed band), quotes the live numbers with the window quoted inward (a band typed at the quoted ends passes; a window narrower than the readouts' last digit takes more decimals), names the inner ring's own onset with mixed rings, never reads a failing number equal to its bound, and no shaft joint text calls a torque the strongest or the most expected, at the BL-067 corner and every test design; it counts in the status line whether the clamp section shows or not (the passing design with the inner hub's minimum at 8 µm fails it alone, 1 of 9 while paused and 1 of 10 with it); without back iron the cup's shorter-magnet fix and the hot minimum pull the magnet length apart, one note in each block. |
```

**6b. `docs/ai/02-system.yaml`** (`:107` the guidance layer, `:108` the pause line, `:109` the stored-3D line, and
Task 2's shaft joint invariant, around `:384`):

(1) Around `:107`. Old:

```text
each of the eight dashboard checks (metal.hot_min_check,
```

New:

```text
each of the nine dashboard checks (metal.hot_min_check,
```

(2) Around `:107`. Old:

```text
metal.clearance_check, materials.cup_wall_check, temperature.summary.verdict, clamps.recommended, gearbox.cold_high_check, housing.space_claim_check, model.end_effect_check)
```

New:

```text
metal.clearance_check, materials.cup_wall_check, temperature.summary.verdict, clamps.recommended, shaft_joint.check, gearbox.cold_high_check, housing.space_claim_check, model.end_effect_check)
```

(3) Around `:107`. Old:

```text
PerTerm for the temperature and space-claim guides, which pick
```

New:

```text
PerTerm for the temperature, space-claim and shaft joints guides, which pick
```

(4) Around `:107`. Old:

```text
verdict terms that fail. A fix is
```

New:

```text
verdict terms that fail (the shaft joints' per joint and term over JOINT_SIDES, holding back a band step that would reverse the band and a key contact height where the key has no bearing length; BL-064). A fix is
```

(5) Around `:107`. Old:

```text
rows over the nine guides and calls
```

New:

```text
rows over the ten guides and calls
```

(6) Around `:108`. Old:

```text
path while paused (clamps::SECTION). Code that branches
```

New:

```text
path while paused (clamps::SECTION; the shaft joints check, shaft_joint.check, BL-064, is outside it and counts while paused). Code that branches
```

(7) Around `:109`. Old:

```text
compares with them. The strings passed
```

New:

```text
compares with them. It also says the bonded inner hub's fitting limit on the shaft joints row uses one of the reverse fields (owner R-15: the row is in STORED_3D_ROWS and carries the label, drawn once after the last stored-3D row, now the shaft joints row). The strings passed
```

(8) Task 2's shaft joint invariant (around `:384`; ruling C-2, A-5: one invariant, with its greying and stored-3D
clauses). Old:

```text
NaN if any is NaN (shaft_joint::hub_fit_allowed).
```

New:

```text
NaN if any is NaN (shaft_joint::hub_fit_allowed). Its dashboard row greys, uncounted, with the end-effect rows (END_EFFECT_ROWS) while f_end <= 0, since it reads cold-high, and is a stored-3D row (STORED_3D_ROWS), since that onset follows the stored single-ring reverse field.
```

**6c. `docs/ai/03-structure.yaml`** (`:68` the gui entry: `CHECKS`, `STORED_3D_ROWS`, `CHECK_GUIDES`, `design_status`,
the test designs):

(1) Around `:68`. Old:

```text
by severity), CHECKS (31 design checks), verdict_level,
```

New:

```text
by severity), CHECKS (32 design checks), verdict_level,
```

(2) Around `:68`. Old:

```text
failing filter's included), STORED_3D_ROWS, STORED_3D_INPUTS (the 14
```

New:

```text
failing filter's included), STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15), STORED_3D_INPUTS (the 14
```

(3) Around `:68`. Old:

```text
learn}, CHECK_GUIDES (the eight dashboard checks and FIT_CHECK), guide, counts (every
```

New:

```text
learn}, CHECK_GUIDES (the nine dashboard checks and FIT_CHECK; the shaft joints' guide, BL-064, picks its fixes per joint and term over JOINT_SIDES), guide, counts (every
```

(4) Around `:68`. Old:

```text
guides that count: 8 while the clamp
```

New:

```text
guides that count: 9 while the clamp
```

(5) Around `:68`. Old:

```text
the gearbox check alone) and clamp_at_its_limit (the
```

New:

```text
the gearbox check alone), only_the_joint_fails (passing_design with the inner hub's minimum interference at 8 µm, failing the shaft joints alone, BL-064) and clamp_at_its_limit (the
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count`

Expected: `test result: ok. 2 passed` (`the_docs_count_the_dashboard_rows_and_the_checks` and
`the_docs_count_the_notes_and_the_steps`).

Run: `python -c "import yaml,sys; [yaml.safe_load(open(p, encoding='utf-8')) for p in sys.argv[1:]]" docs/ai/03-structure.yaml`

Expected: no output. (`02-system.yaml` does not parse before this task either: "could not find expected ':'" at its
line 15, BL-059's R-10, reported to the owner then. This task's edits there are inside existing quoted strings at
`:107-109` and add no new error: the same message, the same line, after it.)

Task 9's stale-text sweep then finds none of the texts it lists under Task 5.

- [ ] **Step 7: Both suites, clippy both ways, fmt**

Run:
```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
cargo test --manifest-path magcoupling-rs/Cargo.toml
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml
git status --short
```

Expected: every `test result: ok.`; the lib with `app` 905 (Task 4's 892 + 13); the lib without `app` unchanged at 231
(this task's code is under the `gui` feature); every integration binary at Task 4's counts (`explain` 61 with `app`, 60
without; `sizing` 45; `shaft_joint` 30). Both clippy runs end `Finished` with no warning; fmt changes
nothing. `git status --short` lists exactly the nine files of Steps 1-6 and the owner's untracked
`docs/analyses/2026-05-28-press-4bar-analysis.md`.

- [ ] **Step 8: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T5.txt` with the Write tool:

```text
feat(magcoupler): the shaft joints on the dashboard, with their guide (BL-064 T5)

The keyed shrink fits that hold both rotors get one counted dashboard row,
after the hidden clamp row and before the gearbox row (19 rows). The check
joins CHECKS (32) with its verdict levels: green when both joints hold and
can be fitted, amber for each unknown form, red for each failing cause.
It reads cold-high, so it greys with the end-effect rows (9); it reads the
inner ring's single-ring onset, so it is a stored-3D row (4) and the drawn
label moves under it (owner R-15).

Its guide (10 guides) names each failing joint and term with live numbers,
the window of measured interference that would work, quoted inward, or why
none does, and which limit sets the bonded inner hub's allowed fitting
temperature. Its fixes are picked per joint and term. The Why link reads
"No note covers the shaft joints yet." until Task 7's note.

The status line reads "Example design: fails 4 of 9 checks" while the clamp
section is paused, "4 of 10" with it shown. The XLSX Summary lists 19 rows.
Shared rounding and never-equal helpers now serve the gearbox Problem and
the cure margin too.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T5.txt
git status --short
```

Expected: one commit of 9 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests this task moves** (each confirmed by running in the prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::dashboard::tests::the_dashboard_starts_with_the_headline_in_order` | `paths[15..]` [gearbox, space claim, end effect] -> [shaft joints, gearbox, space claim, end effect] | the row at index 15 (spec 5.1) |
| `gui::dashboard::tests::every_verdict_each_check_gives_is_classified` | + the ten text forms of the joint check, each reached as the badge | spec 7.5; ruling C-4 |
| `gui::dashboard::tests::every_verdict_of_every_other_check_is_classified` | + three cross-check assertions | no arm reads another check's text |
| `gui::dashboard::tests::every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` | its name rule + `name == "check"` | `shaft_joint.check` escaped the rule (plan deviation PD-23) |
| `gui::dashboard::tests::the_end_effect_rows_are_the_dashboard_rows_that_move_with_the_end_effect_coefficient` | past the headline [gearbox] -> [shaft joints, gearbox] | `END_EFFECT_ROWS` 8 -> 9 |
| `gui::dashboard::tests::the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs` | renamed `..._the_dashboard_rows_...`; body unchanged but a comment | the joint row is no headline row (spec 7.6; owner R-15) |
| `gui::dashboard::tests::the_default_dashboard_has_badges_markers_and_the_3d_rows` | stored-3D lines 3 -> 4; + the joint row green, no marker; + the last stored-3D line is the joint row, paused or not | `STORED_3D_ROWS` 3 -> 4 |
| `gui::dashboard::tests::out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter` | greyed list + `shaft_joint.check`; "three" -> "four"; + the joint green and not greyed in range | `END_EFFECT_ROWS` |
| `gui::dashboard::tests::while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard` | "passes all 8 checks" -> "passes all 9 checks" | one more counted guide |
| `gui::dashboard::tests::a_guide_s_level_is_its_row_s_badge_in_range_and_out_of_it` | + `only_the_joint_fails()` and a reversed hub band | the joint's badge is its guide's level |
| `gui::dashboard::tests::the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour` | + (joint only: 1 red) and (reversed band: 1 amber) | spec 7.7 |
| `gui::guidance::tests::the_gearbox_check_is_a_dashboard_row_with_its_guide` | row 15 -> 16; `at(GEARBOX) == at(CLAMP) + 1` -> `at(SHAFT_JOINT) + 1`; doc | the joint row and guide come first (spec 7.4) |
| `gui::guidance::tests::every_fix_moves_its_check_toward_passing_over_its_slider` | + 11 `SHAFT_JOINT` cases | its table lists every slider fix of every guide (plan deviation PD-20) |
| `gui::guidance::tests::two_failing_checks_that_pull_one_input_apart_say_so` | `pulled_apart` 4 -> 5 (+ `no_back_iron`); + the hot minimum against the joint, one note in each block | without back iron the cup's shorter-magnet fix opposes the hot minimum's (spec R-6) |
| `gui::guidance::tests::each_problem_is_some_exactly_when_its_check_fails` | level table + six joint designs | spec 7.8 |
| `gui::guidance::tests::the_problems_quote_numbers_from_the_results` | + a `SHAFT_JOINT` case; + the window recomputed from the results | spec 7.8 |
| `gui::guidance::tests::the_design_status_counts_the_failing_checks` | total 9 -> 10; every "of 9" -> "of 10" | one more counted guide (spec 7.7) |
| `gui::guidance::tests::a_design_failing_only_the_clamp_passes_while_the_section_is_paused` | (1, 0, 9) -> (1, 0, 10), "1 of 9" -> "1 of 10", (0, 0, 8) -> (0, 0, 9), "all 8" -> "all 9", "4 of 8" -> "4 of 9" | the same |
| `gui::guidance::tests::the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not` | every "of 8" / "all 8" -> 9, every "of 9" / "all 9" -> 10 | the same |
| `gui::guidance::tests` helpers `every_text`, `possible_fixes`, `terms` | + the joint's constants and Problem forms; + its per-term fixes; + its eight terms | the guards and the fix tests cover the new guide |
| `gui::panel::tests::out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too` | `checks.len()` 8 -> 9; comment | one more check among the greyed lines |
| `gui::panel::tests::the_status_line_counts_the_checks_at_default_and_after_a_fix` | (4, 0, 9, true) -> (4, 0, 10, true) | one more counted guide |
| `gui::panel::tests::every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` | + three designs failing the joints, each Problem asserted drawn; the guide's label and Why asserted collected | spec 7.8's clamp guard over the Problem (ruling C-8) |
| `gui::spreadsheet::tests::the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge` | renamed `the_summary_lists_nineteen_dashboard_rows_with_the_gearbox_badge`, over `assert_summary_check_row`; 18 -> `DASHBOARD.len()` (asserted 19); the gearbox's neighbour `clamps.recommended` -> `shaft_joint.check` | the joint's row (spec 7.15; plan deviation PD-13) |
| `gui::result_groups::tests::the_headline_comes_first_then_the_chains_then_the_other_results` | Task 3's `("shaft_joint", true)` -> `("headline", false)` | the check is a dashboard row (Task 3's Interfaces) |
| `tests/explain.rs` `the_docs_count_the_dashboard_rows_and_the_checks` | no edit: red after Step 4, green after Step 6 | the README's "dashboard's 19 rows" |

---

### Task 6: Fix coverage: the designs that fail the joints, the held-back fixes, and the edges most likely to bite

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec 5.2 (the fixes: each margin monotone over its fix's slider, the three held back by rule, "Where the onset binds")
and 7.9 (`designs()`, the re-probed designs, the held-back fix tests, near the joint limit). Task 5's code is in: this
task adds no production code. Its tests pass on Task 5's code when written; Step 3 shows each one bites by breaking
Task 5's code on purpose, one break at a time, and undoing it.

Line numbers are at Task 5's commit (in the prototype, Task 2 plus Task 5: Tasks 3 and 4 do not touch `guidance.rs`).

**Files:**
- Modify (tests): `magcoupling-rs/src/gui/guidance.rs`: `designs()` (after `("gearbox_only", only_the_gearbox_fails()),`);
  new tests before `two_failing_checks_that_pull_one_input_apart_say_so`.
- Modify (docs): `magcoupling-rs/README.md` (the `guidance.rs` test row).
- Test: `guidance`'s tests (`--features app`).

**Interfaces:**
- Consumes (Task 5, in `guidance`'s tests): `SHAFT_JOINT`, `SHAFT_JOINT_GUIDE`, the input constants `HUB_MIN`, `HUB_MAX`,
  `CUP_MIN`, `CUP_MAX`, `KEY_CONTACT`, `WHEEL_SHAFT`, `GEARBOX_SHAFT`, `HUB_LENGTH`, `BOSS_LENGTH` and the shared
  `MAGNET_LENGTH`, `FACE_GAP`; `shaft_joint_problem`, `possible_fixes`, `terms` (the term names "inner hub slips", ...,
  "cup key"), `margin`, `assert_in_order`, `only_the_joint_fails()`; the existing `helps`, `moves`, `fixes`, `level_of`,
  `slider_values`, `with_number`, `assert_open_block_draws_no_fix`, `breaking`, `graded`, `cool_adhesive`, `n42_parts`,
  `long_magnets`, `no_back_iron`, `PART_PATHS`, `GRADE_PATHS`. From the engine: `shaft_joint::{joints, JointTerms,
  CUP, INNER_HUB, CANNOT_BE_FITTED, FIT_SLIPS, UNKNOWN_BAND_REVERSED, SHAFT_JOINTS_PASS}`, `api::shaft_joint_links`,
  `temperature::CURE_MARGIN_MIN_C`.
- Produces: five `designs()` entries (`joint_slips`, `joint_overstressed`, `joint_cannot_be_fitted`,
  `cup_key_no_bearing_area`, `inner_hub_key_no_bearing_area`), so every test that runs over `designs()` covers the joint
  failing alone (`each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red`,
  `each_problem_is_some_exactly_when_its_check_fails`, `every_text`'s guards); twelve tests; the helper
  `near_the_joint_limit()`.

- [ ] **Step 1: Write the tests**

(1) `designs()` (spec 7.9): the joint failing alone, five ways.

At `:3259`, replace

```rust
            ("gearbox_only", only_the_gearbox_fails()),
```

with

```rust
            ("gearbox_only", only_the_gearbox_fails()),
            // BL-064 (spec 7.9): the shaft joints, each design failing them alone. The inner hub
            // slips; its bore stress and fitting temperature are over (a 26 µm maximum); it
            // cannot be fitted on an uncooled wheel shaft; the cup's boss and web, and then the
            // inner hub, are no longer than the key is wide (each also slips).
            ("joint_slips", only_the_joint_fails()),
            (
                "joint_overstressed",
                breaking(|i| i.shaft_joint.hub_interference_max_um = 26.0),
            ),
            (
                "joint_cannot_be_fitted",
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
            ),
            (
                "cup_key_no_bearing_area",
                breaking(|i| i.metal.boss_length_mm = 0.0),
            ),
            (
                "inner_hub_key_no_bearing_area",
                breaking(|i| i.metal.hub_length_mm = 3.0),
            ),
```

(2) Twelve tests, before `two_failing_checks_that_pull_one_input_apart_say_so`:
- `the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other`: spec 7.9's re-probed `designs()` pinned,
  design by design and term by term (only library N42 parts move with owner R-15; the rest as the spec lists);
- `the_shaft_joint_fixes_follow_the_failing_terms`: the fixes per joint and term in the badge's order;
- `raising_a_minimum_is_not_offered_where_one_step_reverses_the_band` (review SF-10) and its mirror
  `lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band` (review 2 SF-2: the open block draws the no-fix
  line, paused or not; the window sentence names the band);
- `raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length` (review MF-2, SF-2, review 2 SF-5:
  both joints; a contact typed 0 or -0.5 mm);
- `with_no_keyway_both_key_terms_pass_with_an_infinite_margin` (review 2 SF-5c);
- `each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider` (spec 5.2: every slider value of every
  joint fix, flat only where the term reads nothing that moves; it asserts which reading is flat);
- `near_the_joint_limit()` and `near_the_joint_limit_the_hot_minimum_guide_shows_no_fix` (spec 7.9, R-7);
- `where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade` (owner R-15 and its ring ruling: library N42
  parts and N50M on manual dimensions; the joint asserted, not the status, since N50M fails the cup wall too);
- `with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down` (review MF-5);
- `the_default_inner_hub_band_is_two_steps_from_failing_on_fitting` (spec R-9);
- `a_joint_that_is_not_judged_keeps_out_of_the_other_joint_s_badge_and_fixes` (spec 4.3, review SF-8).

At `:4885`, replace

```rust
    }

    #[test]
    fn two_failing_checks_that_pull_one_input_apart_say_so() {
```

with

```rust
    }

    #[test]
    fn the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other() {
        // Spec 7.9, the test designs re-probed: without back iron both aluminium parts slip;
        // 50.8 mm magnets make the cup slip and its key alone bear too much (the cup does not
        // lengthen with the magnets); DP460's design limit, and the inner ring's single-ring onset
        // of library N42 blocks (owner R-15), keep the inner hub from being fitted; and the five
        // designs that break the joint alone. Every other test design passes every term, the
        // short magnets' greyed row too (a cold-high below zero asks nothing of a fit).
        let want: Vec<(&str, Vec<String>)> = [
            ("n42_parts", &["inner hub fitting"][..]),
            ("cool_adhesive", &["inner hub fitting"]),
            ("long_magnets", &["cup slips", "cup key"]),
            ("no_back_iron", &["inner hub slips", "cup slips"]),
            ("joint_slips", &["inner hub slips"]),
            (
                "joint_overstressed",
                &["inner hub bore stress", "inner hub fitting"],
            ),
            ("joint_cannot_be_fitted", &["inner hub fitting"]),
            ("cup_key_no_bearing_area", &["cup slips", "cup key"]),
            (
                "inner_hub_key_no_bearing_area",
                &["inner hub slips", "inner hub key"],
            ),
        ]
        .into_iter()
        .map(|(name, terms)| (name, terms.iter().map(|t| (*t).to_owned()).collect()))
        .collect();
        let failing: Vec<(&str, Vec<String>)> = designs()
            .into_iter()
            .filter_map(|(name, inputs)| {
                let results = compute_all(&inputs);
                let failing: Vec<String> = terms(SHAFT_JOINT, &inputs, &results)
                    .into_iter()
                    .filter(|t| !t.passes)
                    .map(|t| t.name)
                    .collect();
                (!failing.is_empty()).then_some((name, failing))
            })
            .collect();
        assert_eq!(failing, want);
    }

    #[test]
    fn the_shaft_joint_fixes_follow_the_failing_terms() {
        // Spec 5.2: each failing term's fixes, joint by joint, each shown only where one step
        // moves that term toward passing and turns no passing check red, each once, in the
        // badge's order (the slips first, the inner hub's before the cup's).
        use Direction::{Lower, Raise};
        let joint = guide(SHAFT_JOINT).unwrap();
        let shown = |inputs: DesignInputs| fixes(joint, &inputs, &compute_all(&inputs), false);
        let hub_slip = [(HUB_MIN, Raise), (HUB_LENGTH, Raise), (FACE_GAP, Raise)];
        let cup_slip = [
            (CUP_MIN, Raise),
            (BOSS_LENGTH, Raise),
            (MAGNET_LENGTH, Lower),
            (FACE_GAP, Raise),
        ];
        assert_eq!(shown(only_the_joint_fails()), hub_slip);
        assert_eq!(
            shown(breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0)),
            cup_slip
        );
        // The inner hub's bore stress and fitting temperature: its maximum down; its wheel shaft
        // is at the slider's bottom already.
        assert_eq!(
            shown(breaking(|i| i.shaft_joint.hub_interference_max_um = 26.0)),
            [(HUB_MAX, Lower)]
        );
        // An uncooled wheel shaft: the maximum down, or the shaft colder.
        assert_eq!(
            shown(breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0)),
            [(HUB_MAX, Lower), (WHEEL_SHAFT, Lower)]
        );
        // The bare cup allowed 250 °C: its maximum down, or the gearbox shaft colder (to the
        // slider's -20 °C).
        assert_eq!(
            shown(breaking(|i| i.shaft_joint.cup_fit_max_C = 250.0)),
            [(CUP_MAX, Lower), (GEARBOX_SHAFT, Lower)]
        );
        // No back iron: both aluminium parts slip; the face gap once, under the inner hub.
        let mut both = hub_slip.to_vec();
        both.extend([
            (CUP_MIN, Raise),
            (BOSS_LENGTH, Raise),
            (MAGNET_LENGTH, Lower),
        ]);
        assert_eq!(shown(breaking(no_back_iron)), both);
        // 50.8 mm magnets: the cup slips and its key alone bears too much.
        let mut long = cup_slip.to_vec();
        long.push((KEY_CONTACT, Raise));
        assert_eq!(shown(breaking(long_magnets)), long);
        assert_eq!(shown(passing_design()), []);
    }

    #[test]
    fn raising_a_minimum_is_not_offered_where_one_step_reverses_the_band() {
        // Review SF-10. The passing design with the inner hub's band at 10-10 µm slips. One step
        // up on the minimum (10.5 µm) raises the slip margin, so `helps` alone would show it, but
        // it puts the minimum above the maximum, and the row would read amber, not green. The
        // hub length and the face gap are shown; the window sentence names the band to enter.
        // With a step of room in the band, the minimum is offered.
        let joint = guide(SHAFT_JOINT).unwrap();
        let inputs = breaking(|i| {
            i.shaft_joint.hub_interference_min_um = 10.0;
            i.shaft_joint.hub_interference_max_um = 10.0;
        });
        let results = compute_all(&inputs);
        let raise_min = (HUB_MIN, Direction::Raise);
        let slip =
            |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, "inner hub slips", d, r);
        assert!(helps(&inputs, &results, raise_min, Some(&slip), false));
        let [stepped] = &moves(&inputs, &results, raise_min.0, raise_min.1)[..] else {
            panic!("one step of the minimum")
        };
        let after = compute_all(stepped);
        assert!(
            after
                .shaft_joint
                .check
                .starts_with(shaft_joint::UNKNOWN_BAND_REVERSED)
        );
        assert_eq!(level_of(SHAFT_JOINT, stepped, &after), Level::Caution);
        assert_eq!(
            fixes(joint, &inputs, &results, false),
            [(HUB_LENGTH, Direction::Raise), (FACE_GAP, Direction::Raise)]
        );
        let text = shaft_joint_problem(&inputs, &results).unwrap();
        let band = format!(
            "(its band is {} to {})",
            input_text(&inputs, HUB_MIN),
            input_text(&inputs, HUB_MAX)
        );
        assert!(text.contains(&band), "{band} in {text}");
        let roomy = with(inputs, |i| i.shaft_joint.hub_interference_max_um = 10.5);
        let shown = fixes(joint, &roomy, &compute_all(&roomy), false);
        assert!(shown.contains(&raise_min), "{shown:?}");
    }

    #[test]
    fn lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band() {
        // Review 2, SF-2, the mirror. The passing design with DP460 and the inner hub's band at
        // 18-18 µm fails on fitting alone (93.3 against 60.0 °C). One step down on the maximum
        // raises the fitting margin but reverses the band, and the wheel shaft is at its slider's
        // bottom: no fix, and the open block draws the no-fix line with the clamp section shown
        // and paused; the window sentence names the band to enter (11.14 to 13.90 µm). With a
        // step of room in the band, the maximum is offered.
        let joint = guide(SHAFT_JOINT).unwrap();
        let inputs = breaking(|i| {
            cool_adhesive(i);
            i.shaft_joint.hub_interference_min_um = 18.0;
        });
        let results = compute_all(&inputs);
        let red: Vec<&str> = CHECK_GUIDES
            .iter()
            .filter(|g| level(g, &inputs, &results) == Level::Bad)
            .map(|g| g.check)
            .collect();
        assert_eq!(red, [SHAFT_JOINT]);
        let lower_max = (HUB_MAX, Direction::Lower);
        let fitting =
            |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, "inner hub fitting", d, r);
        assert!(helps(&inputs, &results, lower_max, Some(&fitting), false));
        assert!(moves(&inputs, &results, WHEEL_SHAFT, Direction::Lower).is_empty());
        let problem = shaft_joint_problem(&inputs, &results).unwrap();
        assert_in_order(
            &problem,
            &[
                "from 11.14 µm to 13.90 µm",
                "(its band is 18.00 µm to 18.00 µm)",
            ],
        );
        let index = CHECK_GUIDES
            .iter()
            .position(|g| g.check == SHAFT_JOINT)
            .unwrap();
        for paused in [false, true] {
            assert_open_block_draws_no_fix(&inputs, &results, index, &problem, paused);
        }
        let roomy = with(inputs, |i| i.shaft_joint.hub_interference_min_um = 17.5);
        assert_eq!(
            fixes(joint, &roomy, &compute_all(&roomy), false),
            [lower_max]
        );
    }

    #[test]
    fn raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length() {
        // Spec 5.2 (review MF-2, SF-2): a key no longer than its joint's engaged length less its
        // width has no bearing length, and no contact height gives it bearing area, though the
        // key margin rises with the height. Only the length fix is shown: the boss for the cup,
        // the hub for the inner hub. With the shared contact height typed 0, the contact is what
        // each key lacks: its fix is shown, and each key sentence is the contact one, never the
        // length one beside it (review 2, SF-5); a contact typed below zero reads as typed.
        let joint = guide(SHAFT_JOINT).unwrap();
        let contact = (KEY_CONTACT, Direction::Raise);
        for (name, inputs, term, length) in [
            (
                "cup",
                breaking(|i| i.metal.boss_length_mm = 0.0),
                "cup key",
                (BOSS_LENGTH, Direction::Raise),
            ),
            (
                "inner hub",
                breaking(|i| i.metal.hub_length_mm = 3.0),
                "inner hub key",
                (HUB_LENGTH, Direction::Raise),
            ),
        ] {
            let results = compute_all(&inputs);
            let key = |d: &DesignInputs, r: &DesignResults| margin(SHAFT_JOINT, term, d, r);
            assert!(key(&inputs, &results).is_finite(), "{name}");
            assert!(
                helps(&inputs, &results, contact, Some(&key), false),
                "{name}"
            );
            let shown = fixes(joint, &inputs, &results, false);
            assert!(!shown.contains(&contact), "{name}: {shown:?}");
            assert!(shown.contains(&length), "{name}: {shown:?}");
        }
        let no_contact = breaking(|i| {
            i.metal.boss_length_mm = 0.0;
            i.shaft_joint.key_contact_mm = 0.0;
        });
        let results = compute_all(&no_contact);
        assert!(fixes(joint, &no_contact, &results, false).contains(&contact));
        let text = shaft_joint_problem(&no_contact, &results).unwrap();
        for name in [shaft_joint::INNER_HUB, shaft_joint::CUP] {
            let sentence = format!(
                "The key has no bearing area in the {name}: its contact height, {}, is not above zero.",
                input_text(&no_contact, KEY_CONTACT)
            );
            assert!(text.contains(&sentence), "{sentence} in {text}");
        }
        assert!(!text.contains("its boss and web"), "{text}");
        let negative = breaking(|i| i.shaft_joint.key_contact_mm = -0.5);
        let text = problem(SHAFT_JOINT, &negative).unwrap();
        let typed = format!(
            "its contact height, {}, is not above zero",
            input_text(&negative, KEY_CONTACT)
        );
        assert!(text.contains(&typed), "{typed} in {text}");
    }

    #[test]
    fn with_no_keyway_both_key_terms_pass_with_an_infinite_margin() {
        // Review 2, SF-5c: without a keyway the key term does not apply; its margin is +inf,
        // never the -inf a contact capped at a zero depth would give, so no fix chases it.
        let keyless = breaking(|i| i.coupling.keyway_depth_mm = 0.0);
        let results = compute_all(&keyless);
        let terms = terms(SHAFT_JOINT, &keyless, &results);
        assert_eq!(terms.len(), 8);
        for name in ["inner hub key", "cup key"] {
            let t = terms.iter().find(|t| t.name == name).unwrap();
            assert!(t.passes && t.margin == f64::INFINITY, "{name}: {t:?}");
        }
        assert_eq!(results.shaft_joint.check, shaft_joint::SHAFT_JOINTS_PASS);
    }

    #[test]
    fn each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider() {
        // Spec 5.2: each margin is monotone over its fix's whole slider, every value at the
        // slider's own step from bottom to top (the hub's minimum past the band's maximum
        // included), rising the way the fix moves. It is flat only where nothing the term reads
        // moves: the bore stress against the band's maximum where smoothing takes all of it (the
        // effective interference is floored at zero), the key against the contact height past
        // the keyway depth (the contact is capped there), and the inner hub's slip against the
        // hub length where the axial override sets the hub's length (BL-064).
        type Reading = fn(&JointTerms) -> f64;
        let effective_max: Reading = |j| j.effective_max_um;
        let contact: Reading = |j| j.key_contact_mm;
        let length: Reading = |j| j.key_bearing_mm;
        let cup_min_10 = || breaking(|i| i.shaft_joint.cup_interference_min_um = 10.0);
        let cases: Vec<(DesignInputs, &str, Direction, &str, Option<Reading>)> = vec![
            (
                only_the_joint_fails(),
                HUB_MIN,
                Direction::Raise,
                "inner hub slips",
                None,
            ),
            (
                only_the_joint_fails(),
                HUB_LENGTH,
                Direction::Raise,
                "inner hub slips",
                None,
            ),
            (
                only_the_joint_fails(),
                FACE_GAP,
                Direction::Raise,
                "inner hub slips",
                None,
            ),
            (cup_min_10(), CUP_MIN, Direction::Raise, "cup slips", None),
            (
                cup_min_10(),
                BOSS_LENGTH,
                Direction::Raise,
                "cup slips",
                None,
            ),
            (
                cup_min_10(),
                MAGNET_LENGTH,
                Direction::Lower,
                "cup slips",
                None,
            ),
            (
                breaking(|i| i.shaft_joint.hub_interference_max_um = 26.0),
                HUB_MAX,
                Direction::Lower,
                "inner hub bore stress",
                Some(effective_max),
            ),
            (
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
                HUB_MAX,
                Direction::Lower,
                "inner hub fitting",
                None,
            ),
            (
                breaking(|i| i.shaft_joint.cup_interference_max_um = 26.0),
                CUP_MAX,
                Direction::Lower,
                "cup bore stress",
                Some(effective_max),
            ),
            (
                breaking(|i| i.shaft_joint.cup_fit_max_C = 250.0),
                CUP_MAX,
                Direction::Lower,
                "cup fitting",
                None,
            ),
            (
                breaking(|i| i.shaft_joint.wheel_shaft_fit_C = 20.0),
                WHEEL_SHAFT,
                Direction::Lower,
                "inner hub fitting",
                None,
            ),
            (
                breaking(|i| i.shaft_joint.cup_fit_max_C = 250.0),
                GEARBOX_SHAFT,
                Direction::Lower,
                "cup fitting",
                None,
            ),
            (
                breaking(long_magnets),
                KEY_CONTACT,
                Direction::Raise,
                "cup key",
                Some(contact),
            ),
            (
                breaking(|i| i.metal.boss_length_mm = 0.0),
                BOSS_LENGTH,
                Direction::Raise,
                "cup key",
                None,
            ),
            (
                breaking(|i| i.metal.hub_length_mm = 3.0),
                HUB_LENGTH,
                Direction::Raise,
                "inner hub key",
                None,
            ),
            (
                breaking(|i| i.coupling.magnets.axial_length_mm = Some(40.0)),
                HUB_LENGTH,
                Direction::Raise,
                "inner hub slips",
                Some(length),
            ),
        ];
        // Every slider fix of the guide is swept.
        let swept: BTreeSet<(&str, bool)> = cases
            .iter()
            .map(|&(_, path, direction, _, _)| (path, direction == Direction::Raise))
            .collect();
        let listed: BTreeSet<(&str, bool)> = possible_fixes(guide(SHAFT_JOINT).unwrap())
            .into_iter()
            .map(|(path, direction)| (path, direction == Direction::Raise))
            .collect();
        assert_eq!(swept, listed);
        let mut flats = 0;
        for (base, path, direction, term, flat_reading) in cases {
            let index = usize::from(term.starts_with(shaft_joint::CUP));
            let mut values = slider_values(path);
            if direction == Direction::Lower {
                values.reverse();
            }
            let at = |x: f64| {
                let design = with_number(&base, path, x).unwrap();
                let results = compute_all(&design);
                let joints =
                    shaft_joint::joints(&design.shaft_joint, &shaft_joint_links(&design, &results));
                (margin(SHAFT_JOINT, term, &design, &results), joints[index])
            };
            let (mut last, mut last_joint) = at(values[0]);
            for &x in &values[1..] {
                let (now, joint) = at(x);
                let what = format!("{term} against {path} at {x}: {last} -> {now}");
                assert!(now >= last, "{what}");
                if now == last {
                    let reading = flat_reading.unwrap_or_else(|| panic!("flat: {what}"));
                    assert_eq!(reading(&joint), reading(&last_joint), "{what}");
                    flats += 1;
                }
                (last, last_joint) = (now, joint);
            }
        }
        // The flats exist (the floor, the cap and the override each reach theirs).
        assert!(flats > 0);
    }

    /// The defaults with the cup's minimum interference 0.001 µm above the smallest that holds:
    /// one step of longer magnets raises cold-high, and with it the torque the cup must hold,
    /// past what it holds (spec 7.9, near the joint limit; the cup does not lengthen with the
    /// magnets).
    fn near_the_joint_limit() -> DesignInputs {
        let needed = compute_all(&DesignInputs::default())
            .shaft_joint
            .cup_needed_um;
        design(|i| i.shaft_joint.cup_interference_min_um = needed + 0.001)
    }

    #[test]
    fn near_the_joint_limit_the_hot_minimum_guide_shows_no_fix() {
        // Spec 7.9 and R-7, a test of its own as BL-059's: the cup holds just above what it
        // must and the hot minimum fails. One step of longer magnets, the hot minimum's only fix,
        // makes the cup slip (its badge reads two numbers apart), so `helps` holds the fix back
        // and the hot minimum's open block draws the no-fix line, with the shaft-clamp section
        // shown and paused. The joint stays green. With half a micrometre more on the cup's
        // minimum, the fix shows again.
        let inputs = near_the_joint_limit();
        let results = compute_all(&inputs);
        assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Good);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Bad);
        let longer = (MAGNET_LENGTH, Direction::Raise);
        let [stepped] = &moves(&inputs, &results, longer.0, longer.1)[..] else {
            panic!("one step of longer magnets")
        };
        let after = compute_all(stepped);
        assert_eq!(level_of(SHAFT_JOINT, stepped, &after), Level::Bad);
        assert_eq!(
            after.shaft_joint.check,
            format!(
                "{} ({}): holds 7.64 N·m against 7.65 N·m",
                shaft_joint::FIT_SLIPS,
                shaft_joint::CUP
            )
        );
        let hot_min = CHECK_GUIDES
            .iter()
            .position(|g| g.check == HOT_MIN)
            .unwrap();
        let problem = hot_min_problem(&inputs, &results).unwrap();
        for paused in [false, true] {
            assert!(
                !helps(&inputs, &results, longer, None, paused),
                "paused {paused}"
            );
            assert_open_block_draws_no_fix(&inputs, &results, hot_min, &problem, paused);
        }
        let roomier = with(inputs, |i| i.shaft_joint.cup_interference_min_um += 0.5);
        let results = compute_all(&roomier);
        assert_eq!(fixes(&HOT_MIN_GUIDE, &roomier, &results, false), [longer]);
    }

    #[test]
    fn where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade() {
        // Owner R-15 and its ring ruling (spec 5.2, "Where the onset binds"; 7.9). Library N42
        // blocks in both rings cap the bonded inner hub's allowed fitting temperature at the
        // inner ring's own single-ring onset, 69.83 °C, less the cure margin: 59.83 °C against
        // the 93.27 °C its fit needs. The joint fails on that term alone; the temperature check
        // stays amber. One step down on the hub's maximum raises the fitting margin and turns
        // nothing red, so it is shown; the wheel shaft is at its slider's bottom. The magnets'
        // part or grade is the temperature guide's fix, never the joint's. N50M on manual
        // dimensions too: its onset, 98.80 °C, is above the need; only the onset less the margin,
        // 88.80 °C, is below it (the cup wall fails there too: the joint alone is asserted).
        let joint = guide(SHAFT_JOINT).unwrap();
        for (name, inputs, want_onset) in [
            ("n42_parts", breaking(n42_parts), 69.8326),
            ("N50M", breaking(|i| graded(i, "N50M")), 98.8029),
        ] {
            let results = compute_all(&inputs);
            let s = &results.shaft_joint;
            let onset = results.temperature.demag.inner_onset_single_ring_C;
            assert!((onset - want_onset).abs() < 1e-4, "{name}: {onset}");
            assert_eq!(s.hub_fit_allowed_C, onset - CURE_MARGIN_MIN_C, "{name}");
            assert!((s.hub_fit_temp_C - 93.2674).abs() < 1e-4, "{name}");
            assert_eq!(
                s.check,
                format!(
                    "{} ({}): needs 93.3 °C against {} °C",
                    shaft_joint::CANNOT_BE_FITTED,
                    shaft_joint::INNER_HUB,
                    fixed(onset - CURE_MARGIN_MIN_C, 1)
                ),
                "{name}"
            );
            let failing: Vec<String> = terms(SHAFT_JOINT, &inputs, &results)
                .into_iter()
                .filter(|t| !t.passes)
                .map(|t| t.name)
                .collect();
            assert_eq!(failing, ["inner hub fitting"], "{name}");
            assert_eq!(
                level_of(TEMPERATURE, &inputs, &results),
                Level::Caution,
                "{name}"
            );
            let shown = fixes(joint, &inputs, &results, false);
            assert_eq!(shown, [(HUB_MAX, Direction::Lower)], "{name}");
            let fitting =
                |d: &DesignInputs| margin(SHAFT_JOINT, "inner hub fitting", d, &compute_all(d));
            let [stepped] = &moves(&inputs, &results, HUB_MAX, Direction::Lower)[..] else {
                panic!("{name}: one step down")
            };
            assert!(fitting(stepped) > fitting(&inputs), "{name}");
            assert!(moves(&inputs, &results, WHEEL_SHAFT, Direction::Lower).is_empty());
        }
        for path in PART_PATHS.into_iter().chain(GRADE_PATHS) {
            assert!(
                possible_fixes(joint).iter().all(|&(p, _)| p != path),
                "{path}"
            );
        }
    }

    #[test]
    fn with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down() {
        // Review MF-5 (spec 7.9): DP460's 60 °C design limit caps the bonded inner hub's allowed
        // fitting temperature below the 93.27 °C its fit needs, the only failing term; one step
        // down on its maximum takes the need to 89.20 °C, and that fix is shown; the wheel shaft
        // is at its slider's bottom.
        let inputs = breaking(cool_adhesive);
        let results = compute_all(&inputs);
        let failing: Vec<String> = terms(SHAFT_JOINT, &inputs, &results)
            .into_iter()
            .filter(|t| !t.passes)
            .map(|t| t.name)
            .collect();
        assert_eq!(failing, ["inner hub fitting"]);
        assert_eq!(
            fixes(guide(SHAFT_JOINT).unwrap(), &inputs, &results, false),
            [(HUB_MAX, Direction::Lower)]
        );
        let [stepped] = &moves(&inputs, &results, HUB_MAX, Direction::Lower)[..] else {
            panic!("one step down")
        };
        let need = |r: &DesignResults| format_value(&Value::Num(r.shaft_joint.hub_fit_temp_C));
        assert_eq!(
            (need(&results), need(&compute_all(stepped))),
            ("93.27".to_owned(), "89.20".to_owned())
        );
    }

    #[test]
    fn the_default_inner_hub_band_is_two_steps_from_failing_on_fitting() {
        // Review focus (spec R-9): the example's inner hub band sits 0.83 µm under its fitting
        // edge. One step up on its maximum (18.5 µm) still passes; two (19 µm) fail on fitting,
        // 101.4 against 100.0 °C, and the guide offers the band back down: not the wheel shaft,
        // already at its slider's bottom, nor the allowed temperature (a limit, not a lever).
        let one = design(|i| i.shaft_joint.hub_interference_max_um = 18.5);
        assert_eq!(
            compute_all(&one).shaft_joint.check,
            shaft_joint::SHAFT_JOINTS_PASS
        );
        let two = design(|i| i.shaft_joint.hub_interference_max_um = 19.0);
        let results = compute_all(&two);
        assert_eq!(
            results.shaft_joint.check,
            format!(
                "{} ({}): needs 101.4 °C against 100.0 °C",
                shaft_joint::CANNOT_BE_FITTED,
                shaft_joint::INNER_HUB
            )
        );
        assert_eq!(
            fixes(guide(SHAFT_JOINT).unwrap(), &two, &results, false),
            [(HUB_MAX, Direction::Lower)]
        );
        let text = problem(SHAFT_JOINT, &two).unwrap();
        assert!(
            text.contains("your limit for the bonded inner hub"),
            "{text}"
        );
    }

    #[test]
    fn a_joint_that_is_not_judged_keeps_out_of_the_other_joint_s_badge_and_fixes() {
        // Spec 4.3 (review SF-8): beside a joint that fails, a joint that is not judged turns
        // nothing red and offers no fix; the badge and the fixes are the failing joint's, and its
        // Problem gives the failing joint's causes and window first, then why the other is not
        // judged. A reversed inner hub band with the cup slipping; the cup without a wall (bore
        // 16 mm, boss 12 mm) with the inner hub slipping. Without the failing joint (the hub's
        // minimum at 15 µm holds on the 16 mm bore) each reads amber with no fix.
        use Direction::{Lower, Raise};
        fn reversed(i: &mut DesignInputs) {
            i.shaft_joint.hub_interference_min_um = 20.0;
        }
        fn no_wall(i: &mut DesignInputs) {
            i.coupling.bore_mm = 16.0;
            i.metal.boss_od_mm = 12.0;
        }
        fn no_wall_alone(i: &mut DesignInputs) {
            no_wall(i);
            i.shaft_joint.hub_interference_min_um = 15.0;
        }
        let joint = guide(SHAFT_JOINT).unwrap();
        let cases = [
            (
                breaking(|i| {
                    reversed(i);
                    i.shaft_joint.cup_interference_min_um = 10.0;
                }),
                shaft_joint::CUP,
                vec![
                    (CUP_MIN, Raise),
                    (BOSS_LENGTH, Raise),
                    (MAGNET_LENGTH, Lower),
                    (FACE_GAP, Raise),
                ],
                [
                    "At its loosest the cup holds",
                    "On the cup, a measured interference",
                    "The inner hub's minimum interference",
                ],
            ),
            (
                breaking(no_wall),
                shaft_joint::INNER_HUB,
                vec![(HUB_MIN, Raise), (HUB_LENGTH, Raise), (FACE_GAP, Raise)],
                [
                    "At its loosest the inner hub holds",
                    "On the inner hub, a measured interference",
                    "The cup's bore",
                ],
            ),
        ];
        for (inputs, failing, want, order) in cases {
            let results = compute_all(&inputs);
            let badge = &results.shaft_joint.check;
            assert!(
                badge.starts_with(&format!("{} ({failing})", shaft_joint::FIT_SLIPS)),
                "{badge}"
            );
            assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Bad);
            assert_eq!(fixes(joint, &inputs, &results, false), want, "{badge}");
            assert_in_order(&shaft_joint_problem(&inputs, &results).unwrap(), &order);
        }
        for edit in [reversed as fn(&mut DesignInputs), no_wall_alone] {
            let inputs = breaking(edit);
            let results = compute_all(&inputs);
            assert_eq!(level_of(SHAFT_JOINT, &inputs, &results), Level::Caution);
            assert_eq!(fixes(joint, &inputs, &results, false), []);
        }
    }

    #[test]
    fn two_failing_checks_that_pull_one_input_apart_say_so() {
```

- [ ] **Step 2: Run them; they pass on Task 5's code**

Run:
```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_shaft_joint_fixes_follow_the_failing_terms raising_a_minimum_is_not_offered lowering_a_maximum_is_not_offered raising_the_key_contact_is_not_offered with_no_keyway_both_key_terms each_shaft_joint_fix_moves near_the_joint_limit_the_hot_minimum where_the_magnets_onset_binds with_a_cool_adhesive_the_inner_hub the_default_inner_hub_band a_joint_that_is_not_judged the_shaft_joints_fail_on_these each_failing_term_has_a_fix every_fix_moves_its_check_toward_passing
```

Expected: `test result: ok. 14 passed; 0 failed` (the twelve new tests, and the two that run over every guide's fixes,
now over the five new designs too).

- [ ] **Step 3: See each test bite (planted faults, each undone)**

First stage Step 1's file: `git add magcoupling-rs/src/gui/guidance.rs`. Then make each break below in that file, run
`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`, see exactly the tests named fail, and undo
the break with `git checkout -- magcoupling-rs/src/gui/guidance.rs` (it restores the staged text). Each result below was
measured on the prototype with that command:

1. In `shaft_joint_fixes`, delete the line `                && !reverses_the_band(inputs, results, side, fix)`.
   Expected: FAILED, 2 failed: `raising_a_minimum_is_not_offered_where_one_step_reverses_the_band` and
   `lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band`.
2. In `shaft_joint_fixes`, delete the line `                && !contact_without_length`.
   Expected: FAILED, 1 failed: `raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length`.
3. In `JOINT_SIDES`, the cup's `slip_fixes`, delete `            (MAGNET_LENGTH, Direction::Lower),`.
   Expected: FAILED, 5 failed: `the_shaft_joint_fixes_follow_the_failing_terms`,
   `each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider`,
   `a_joint_that_is_not_judged_keeps_out_of_the_other_joint_s_badge_and_fixes`,
   `every_fix_moves_its_check_toward_passing_over_its_slider` and `two_failing_checks_that_pull_one_input_apart_say_so`.
4. In `JOINT_SIDES`, the inner hub's `fitting_fixes`, `&[(HUB_MAX, Direction::Lower), (WHEEL_SHAFT, Direction::Lower)]`
   -> `&[(WHEEL_SHAFT, Direction::Lower)]`.
   Expected: FAILED, 6 failed: `each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red`,
   `lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band`,
   `the_default_inner_hub_band_is_two_steps_from_failing_on_fitting`, `the_shaft_joint_fixes_follow_the_failing_terms`,
   `where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade` and
   `with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down`.
5. In `designs()`, delete the line `            ("joint_slips", only_the_joint_fails()),`.
   Expected: FAILED, 1 failed: `the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other` (the other tests
   that run over `designs()` still pass with one design fewer).

(Task 5's own window rule bites the same way: in `window_quotes`, `if quoted_value(low) < quoted_value(high) {` ->
`if true {` fails exactly `a_window_narrower_than_the_last_shown_digit_quotes_its_ends_with_more_decimals`.)

After the last undo, `git diff --stat` is empty and `git diff --cached --stat` lists the one test file.

- [ ] **Step 4: Docs**

`magcoupling-rs/README.md`, the Tests table's `src/gui/guidance.rs` row: Task 6's sentence goes at the row's end,
after Task 5's ("BL-064 (T5): ... one note in each block."):

(1) Around `:690`. Old:

```text
in each block. |
```

New:

```text
in each block. BL-064 (T6): the test designs include the shaft joints failing alone (the inner hub slipping; its bore stress and fitting temperature over; an uncooled wheel shaft; the cup's, then the inner hub's, key without bearing length), so every shown fix moves its failing term and turns no passing check red there too; the joints fail on exactly the test designs and terms spec 7.9 names (without back iron both slip; 50.8 mm magnets the cup's slip and key; DP460 and library N42 parts the inner hub's fitting; and those five) and pass on every other; the fixes follow the failing terms joint by joint; each fix's margin rises the way the fix moves over its whole slider, flat only where the term reads nothing that moves (the effective interference floored at zero, the contact height capped at the keyway depth, the hub's length set by the axial override); a band step that would reverse the band and a contact height where the key has no bearing length are held back (the open block draws the no-fix line where nothing else helps), and with no keyway both key terms pass with an infinite margin; near the joint limit (the cup's minimum 0.001 µm over what holds) the hot minimum's longer-magnet fix is held back; where the inner magnets' onset or the adhesive's limit caps the bonded hub, the joint offers its band, never the magnets' part or grade; the example's inner hub band is two steps of its maximum from failing on fitting; a joint that is not judged keeps out of the other joint's badge and fixes. |
```

- [ ] **Step 5: Both suites, clippy both ways, fmt**

Run Task 5 Step 7's commands.

Expected: every `test result: ok.`; the lib with `app` 917 (Task 5's 905 + 12); the lib without `app` unchanged at 231;
every integration binary unchanged (`explain` 61 with `app`, 60 without). Both clippy runs end `Finished` with no warning; fmt changes nothing.

- [ ] **Step 6: Commit**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T6.txt` with the Write tool:

```text
test(magcoupler): fix coverage of the shaft joints' guide (BL-064 T6)

The test designs gain the shaft joints failing alone: the inner hub
slipping, over its stress limit, unfittable on an uncooled wheel shaft, and
each joint's key without bearing length. Every test that runs over the
designs now covers them, so each shown fix moves its failing term and turns
no passing check red there too.

New tests pin which test designs fail the joints and on which terms, the
fixes per joint and term, each fix's margin over its whole slider, the three
fixes held back by rule (a band step that would reverse the band, either
way, and the key's contact height without bearing length), no keyway,
near the joint limit, the onset and adhesive caps (the band is offered,
never the magnets' grade), the example's tight inner hub band and a joint
that is not judged beside one that fails. Each was seen to fail with the
rule it guards broken. No production code changes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/README.md
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T6.txt
git status --short
```

Expected: one commit of 2 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests this task moves:** none edited. Five tests that run over `designs()` run over five more designs and pass
unchanged: `each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red`,
`each_problem_is_some_exactly_when_its_check_fails`, and the guards over `every_text()`
(`no_guide_text_carries_workbook_jargon`, `every_guide_text_has_glyphs_in_the_default_fonts`,
`no_guide_text_says_the_coupling_protects_the_gearbox`).

#### Review focus (Tasks 5 and 6): the edges most likely to bite a user

Each is reachable on the sliders unless it says otherwise, none is a main line of the spec, and each has a test:
1. **The example's inner hub band is 0.83 µm under its fitting edge** (spec R-9). Two 0.5 µm steps on its maximum fail on
   fitting, 101.4 against 100.0 °C, and the guide offers the band back down, never the wheel shaft (already at its
   slider's bottom) or the allowed temperature. `the_default_inner_hub_band_is_two_steps_from_failing_on_fitting` (T6).
2. **A band one step wide.** Raising its minimum, or lowering its maximum, one step reverses the band and turns the row
   amber, though the margin rises; both are held back, the open block draws the no-fix line where nothing else helps,
   and the window sentence names the band to enter. `raising_a_minimum_is_not_offered_where_one_step_reverses_the_band`,
   `lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band` (T6).
3. **An uncooled wheel shaft** (the user forgets the liquid nitrogen): 247.6 °C needed, the source named, and the empty
   window says the allowed temperatures do not even open the joining clearance.
   `the_shaft_joint_problem_quotes_the_live_numbers` (T5), `the_shaft_joint_fixes_follow_the_failing_terms` (T6).
4. **A lower magnet grade or part** (owner R-15): the inner ring's onset caps the bonded hub; the joint offers its band,
   never the magnets' grade, and the Problem quotes the inner ring's own onset with mixed rings.
   `where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade` (T6),
   `the_shaft_joint_problem_quotes_the_live_numbers` (T5).
5. **A short hub or no boss length:** the key has no bearing length, and raising its contact height cannot help; only the
   length is offered. `raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length` (T6).
6. **Near the joint limit:** longer magnets, the hot minimum's only fix, would make the cup slip, so it is held back.
   `near_the_joint_limit_the_hot_minimum_guide_shows_no_fix` (T6).
7. **Short magnets (f_end ≤ 0):** the whole row greys, a bore stress above its limit included, under the end-effect
   banner. `out_of_range_end_effect_greys_the_shaft_joint_check_whatever_its_text_reads` (T5).
8. **A band typed at the quoted window's ends passes** (the window is quoted inward).
   `a_shaft_joint_window_typed_at_its_quoted_ends_passes` (T5).
9. **A joint that is not judged** (a reversed band, no wall) beside one that fails keeps out of the badge and the fixes.
   `a_joint_that_is_not_judged_keeps_out_of_the_other_joint_s_badge_and_fixes` (T6).

---

### Task 7: The teaching note `shaft_joint`: its Start here step, its links and its physics review

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 6.1, 6.3 (the note's review) and 7 item 11; owner decision 8A (a new reviewed note, the note cap raised from
20 to 21) and owner R-15 with its ring ruling (note sentence 4 and the first source). The note owns the shaft joint
check's 23 numeric results and the inner ring's own single-ring onset (24 equations). Its Start here step goes just before
the clamp note's, so it is the last step shown while the clamp section is paused. The Start here hover gains its topic.
The sign-off sheet generator marks it as added after the slice; the sheet is not regenerated. The shaft-joint row's Why
link (its guide's and its `ROW_LEARN` entry's) flips from Task 5's `Learn::None` to `Learn::Note("shaft_joint")` when the
note passes its fresh-context physics review. The review's record goes under a new `## shaft_joint` section of
`docs/analyses/2026-10-14-magcoupler-notes-review.md`.

**How the note moves through the gate.** The note goes in as `Review::Draft` (Step 3). Between Step 3 and Step 10 the
release gates are red by design, and no commit is made: Start here and the reviewed-note tests require every note shown
to be reviewed. The Why link flips only with the PASS (Step 10): `every_learn_link_resolves` and
`release_notes_are_reviewed` forbid a link to an unreviewed note. The note becomes `Reviewed` in the same commit that
appends its section to the report, as every slice-2 note did.

**What this task leaves alone** (checked at plan time on a prototype of Tasks 1 to 7, where every test of Tasks 5 and
6 passes with the flipped link):
- `start_here_for` is unchanged. It drops the order's last step while paused, and that step is still the clamp's.
  `PAUSED_NOTE`, `shown_when` and `shown` are unchanged too.
- The before snapshot is unchanged. `the_before_snapshot_format_reads_back` round-trips the current `NOTES`, all 21.
  `the_before_snapshot_holds_the_seventeen_notes_slice_2_started_from` reads only the frozen file.
- The committed sign-off sheet `docs/analyses/2026-10-14-magcoupler-notes-signoff.md` is not regenerated (spec 6.1,
  decision 23). Never run `notes_signoff_sheet`, which overwrites it.
- The reviewed `demagnetization` note is unchanged. It keeps the governing onset and the per-ring magnet limits. The
  inner ring's own onset goes to this note, because a record has at most one note.
- The torque-band note's sentence 3 and its source, and the 14 "shaft-joint and gearbox loads rise" change entries, stay
  true and stay as they are (spec 6.1).
- `docs/ai/02-system.yaml` :486 records slice 2's status ("Start here in 14 steps (13 while paused)"). It is history,
  and Task 9 owns that line.
- Nothing in this task changes an engine value, a Python-locked row or `tests/data/input_schema.json`.

**Files:**
- Modify `magcoupling-rs/src/engine/explain/notes.rs`:
  - the `NOTES` doc;
  - the new `Note` literal, just before `clamp_preload`'s;
  - the `START_HERE_ALL` doc and a new entry;
  - in its tests: the cap in `a_draft_is_hidden_and_every_note_is_complete`, `SLICE_2_ORDER` and its doc, and one new
    test.
- Modify `magcoupling-rs/src/gui/explorer.rs`:
  - the `start_here_topics!` macro and the `start_here_text` doc;
  - in its tests: the hover test, the bar test, one new render test, and the sign-off sheet (`AFTER_THE_SLICE`,
    `ADDED_AFTER_THE_SLICE`, `CLAMPS_PAUSED_LINE`, `signoff_sheet` and its test).
- Modify `magcoupling-rs/src/gui/guidance.rs`:
  - `SHAFT_JOINT_GUIDE.learn`;
  - in its tests: one new test, and Task 5's `the_shaft_joint_check_is_a_dashboard_row_with_its_guide` (the interim link's
    assertion).
- Modify `magcoupling-rs/src/gui/panel.rs` (tests): one new test.
- Modify `magcoupling-rs/tests/explain.rs`:
  - the renamed Start here test;
  - a new section with two tests and two helpers.
- Modify `docs/analyses/2026-10-14-magcoupler-notes-review.md`: append `## shaft_joint` at the end of the file.
- Docs:
  - `magcoupling-rs/README.md`: ~:29, ~:504, ~:506-507, ~:524-526, the test-table rows ~:663, ~:670, ~:677 and ~:678,
    and ~:747;
  - `docs/ai/03-structure.yaml`: :63 and :74.
- Gitignored outputs:
  - `magcoupling-rs/target/notes-review-BL064.txt` and `magcoupling-rs/target/g3/BL064-note.diff`;
  - `.superpowers/sdd/2026-10-17-bl064-shaft-joint/t7-review-r<N>.md`, each review round's answer.

**Interfaces:**
- Consumes:
  - Task 1: `temperature.demag.inner_onset_single_ring_C` and its record (`records/demagnetization.rs`).
  - Task 2:
    - `DesignResults::shaft_joint` (23 numeric results and `check`);
    - `magcoupling::engine::api::shaft_joint_links(&DesignInputs, &DesignResults) -> ShaftJointLinks`;
    - from `magcoupling::engine::shaft_joint`: `joints(&ShaftJointInputs, &ShaftJointLinks) -> [JointTerms; 2]` (with
      `JointTerms::{name, stress_per_um}`) and
      `hub_fit_allowed(f64, f64, f64) -> (f64, Option<HubFitLimit>)`;
    - the helps of `shaft_joint.hub_fit_max_C` and `shaft_joint.hub_fit_allowed_C`.
  - `magcoupling::engine::temperature::CURE_MARGIN_MIN_C`.
  - Task 3: the 23 records of `records/shaft_joint.rs`. `notes_link_to_records_and_each_equation_has_at_most_one` and
    `every_start_here_step_opens_an_equation_with_its_reviewed_note` need them.
  - Task 5:
    - `const SHAFT_JOINT: &str = "shaft_joint.check"` and `SHAFT_JOINT_GUIDE` in `gui/guidance.rs`, its
      `learn: Learn::None("No note covers the shaft joints yet.")` (the interim link, plan deviation PD-25);
    - its `ROW_LEARN` entry `(SHAFT_JOINT, SHAFT_JOINT_GUIDE.learn)`;
    - the dashboard row at index 15;
    - the test `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`, whose guide variable is `joint` and which
      asserts the interim link (Step 10c).
  - From the codebase: `probes::probe_designs`, `note_text::{render, rendered_texts, segments}`,
    `explorer::{live_value, reviewed_note, start_here_hover, start_here_text}`, `test_support::passing_design`, and the
    review tool `notes_for_review`.
- Produces:
  - the note `shaft_joint` ("Shaft joints: a keyed shrink fit"), reviewed, with 24 equations;
  - `START_HERE_ALL`: 15 steps, 14 while paused, with the new step `("shaft_joint", "shaft_joint.hub_holding_Nm")`;
  - `START_HERE_HOVER`, which ends "..., demagnetization, shaft joints" while paused;
  - `SHAFT_JOINT_GUIDE.learn == Learn::Note("shaft_joint")`, and the same through `ROW_LEARN`;
  - in the explorer tests: `AFTER_THE_SLICE`, `ADDED_AFTER_THE_SLICE` and the reworded `CLAMPS_PAUSED_LINE`;
  - in `tests/explain.rs`: the helpers `cure_margin_words` and `note_texts_with_sources`;
  - the review record `## shaft_joint`, with the reviewer's rulings on the spec's risks R-1 to R-4 and R-14 (Task 10
    reads them from that section; Task 8's reviewer reads them for any text they change);
  - the final, reviewed note texts (Task 8's reviewer reads them for consistency, and Task 9 and Task 10 quote them).

**Pinned tests this task moves:**

| Test | Old | New | Reason |
|---|---|---|---|
| `engine::explain::notes::tests::a_draft_is_hidden_and_every_note_is_complete` | `(15..=20)`, "spec A4: about 15 to 20 notes" | `(15..=21)`, the message names the owner's raise | The owner raised the cap for the shaft joints (8A). |
| `engine::explain::notes::tests::start_here_follows_the_slice_2_order_from_how_the_coupling_makes_torque` (its `SLICE_2_ORDER`) | 14 ids | 15, with `"shaft_joint"` just before `"clamp_preload"` | A step outside the list fails that test. |
| `tests/explain.rs`: `start_here_is_fourteen_steps_and_thirteen_while_paused`, renamed `start_here_is_fifteen_steps_and_fourteen_while_paused` | `FULL: [&str; 14]`, `FULL[..13]` | `FULL: [&str; 15]` with `"shaft_joint"` before `"clamp_preload"`, `FULL[..14]` | The new step. |
| `gui::explorer::tests::the_start_here_bar_names_every_step_it_shows` | 14 steps, 13 while paused; two bars pinned | 15 and 14; a third bar pinned: "Start here 14 of {steps}: Shaft joints: a keyed shrink fit" | The new step is the 14th in both orders. |
| `gui::explorer::tests::the_start_here_hover_names_the_clamp_step_only_while_the_section_shows` | (none) | adds: the paused hover ends "demagnetization, shaft joints" | The new topic. |
| `gui::explorer::tests::the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice` | `NOTES.len() - NEW_NOTES.len()` notes in the snapshot; the notes without a before text are `NEW_NOTES` | `- AFTER_THE_SLICE.len()`; those notes are `NEW_NOTES` and `AFTER_THE_SLICE` together; `ADDED_AFTER_THE_SLICE` shows once in both switch states; `CLAMPS_PAUSED_LINE` no longer says the clamp returns with BL-064 | The note has no text before slice 2 and is not slice 2's (spec 6.1, review SF-14). |
| Task 5's `gui::guidance::tests::the_shaft_joint_check_is_a_dashboard_row_with_its_guide` | the interim link, `Learn::None(..)` | `Learn::Note("shaft_joint")` | The flip (plan deviation PD-25). |
| `tests/explain.rs::the_docs_count_the_notes_and_the_steps` (code unchanged) | README and `03-structure.yaml`: 20 notes, 19 shown, 13 steps | 21 notes, 20 shown, 14 steps | It holds the docs to the code. |

New tests (red at Step 2):
- `engine::explain::notes::tests::the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step`
- `tests/explain.rs::the_shaft_joint_note_s_claims_hold_at_every_probe_design`
- `tests/explain.rs::the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps`
- `gui::explorer::tests::the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink`
- `gui::guidance::tests::the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note`
- `gui::panel::tests::the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`

The note's half of spec 7.8 (no text orders the torques) is in `the_shaft_joint_note_s_claims_hold_at_every_probe_design`.
Task 5's `the_shaft_joint_texts_never_order_the_torques` cannot see the note, which does not exist until this task.

- [ ] **Step 1: Write the failing tests and move the pinned ones**

  1a. `magcoupling-rs/src/engine/explain/notes.rs`, the tests module: `SLICE_2_ORDER` (~:1604). Old:
  ```rust
      /// Slice-2 spec 3.3: Start here's 14 steps, in this order. A step whose note a later task of
      /// the slice adds is skipped until then, so every task keeps the order (T18 asserts all 14).
      const SLICE_2_ORDER: [&str; 14] = [
  ```
  New:
  ```rust
      /// Slice-2 spec 3.3: Start here's steps, in this order, with BL-064's shaft-joint step just
      /// before the clamp's (15). A step whose note a later task adds is skipped until then, so
      /// every task keeps the order (`tests/explain.rs` asserts all 15).
      const SLICE_2_ORDER: [&str; 15] = [
  ```
  and in the same array, old:
  ```rust
          "ferrite_cold_demag",
          "clamp_preload",
      ];
  ```
  new:
  ```rust
          "ferrite_cold_demag",
          "shaft_joint",
          "clamp_preload",
      ];
  ```

  1b. The same tests module: insert the new test just before `    #[test]\n    fn the_rewritten_notes_type_no_editable_default() {`
  (~:1557):
  ```rust
      /// BL-064 (spec 6.1, owner decision 8A): the shaft-joint note keeps the slice-2 caps and a
      /// plain sentence 0; it owns every numeric result of the shaft joint check, in their schema
      /// order, and the inner ring's own single-ring onset the bonded inner hub's cap reads (owner
      /// R-15 ring; the demagnetization note keeps the governing onset); and it is Start here's step
      /// just before the clamp note's, opening the inner hub's holding torque, so it is the last
      /// step shown while the shaft-clamp section is paused.
      #[test]
      fn the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step() {
          use crate::engine::meta::{FieldType, result_rows};
          let n = note("shaft_joint").expect("the shaft-joint note");
          assert_eq!(cap_problems(n), Vec::<String>::new());
          assert_eq!(sentence_zero_problems(n.sentences[0]), Vec::<&str>::new());
          let results = crate::compute_all(&crate::DesignInputs::default());
          let mut rows: Vec<String> = result_rows(&results)
              .into_iter()
              .filter(|r| r.path.starts_with("shaft_joint.") && r.meta.ty == FieldType::F64)
              .map(|r| r.path)
              .collect();
          assert_eq!(rows.len(), 23, "the check's numeric results: {rows:?}");
          rows.push("temperature.demag.inner_onset_single_ring_C".to_owned());
          assert_eq!(n.equations, rows.as_slice());
          for path in &rows {
              assert_eq!(
                  note_for_any_status(path).map(|m| m.id),
                  Some("shaft_joint"),
                  "{path}: one note owns it"
              );
          }
          for paused in [false, true] {
              let steps = start_here_for(paused);
              let ids: Vec<&str> = steps.iter().map(|&(id, _)| id).collect();
              let at = ids
                  .iter()
                  .position(|&id| id == "shaft_joint")
                  .unwrap_or_else(|| panic!("no shaft-joint step: {ids:?}"));
              assert_eq!(steps[at].1, "shaft_joint.hub_holding_Nm");
              assert_eq!(ids[at - 1], "ferrite_cold_demag", "{ids:?}");
              assert_eq!(
                  ids.get(at + 1).copied(),
                  (!paused).then_some(PAUSED_NOTE),
                  "the clamp's step follows it, and only while the section shows: {ids:?}"
              );
          }
      }

  ```
  The equality with the engine's numeric `shaft_joint.*` results is the guard against the note's list drifting from the
  check (the note types the 24 paths, as every note does). `note_for_any_status` returns the first note in `NOTES` that
  covers a path. The demagnetization note comes before this one, so a second owner of the onset would fail here.

  1c. `magcoupling-rs/tests/explain.rs`, the imports: no edit. Task 3's Step 1(b2) already imports `CURE_MARGIN_MIN_C`
  and `magcoupling::engine::shaft_joint`.

  1d. The same file: the Start here test (~:1215-1243). Old:
  ```rust
  /// Spec 3.3 (R4), section 6 test 5 and ruling 11: the full Start here order is fourteen
  /// steps, the torque note first and the clamp note last; while the shaft-clamp section is
  /// paused `START_HERE` shows the first thirteen. Each note is a step once; with the six
  /// warning notes that is every note shown, in both switch states. (That each step's note
  /// explains its equation is `notes.rs`'s `ids_are_unique_and_every_link_resolves`.)
  #[test]
  fn start_here_is_fourteen_steps_and_thirteen_while_paused() {
      const FULL: [&str; 14] = [
  ```
  New:
  ```rust
  /// Spec 3.3 (R4), section 6 test 5, ruling 11 and BL-064 (spec 6.1): the full Start here
  /// order is fifteen steps, the torque note first, the shaft-joint note just before the clamp
  /// note, the clamp note last; while the shaft-clamp section is paused `START_HERE` shows the
  /// first fourteen. Each note is a step once; with the six warning notes that is every note
  /// shown, in both switch states. (That each step's note explains its equation is `notes.rs`'s
  /// `ids_are_unique_and_every_link_resolves`.)
  #[test]
  fn start_here_is_fifteen_steps_and_fourteen_while_paused() {
      const FULL: [&str; 15] = [
  ```
  In the same test, old:
  ```rust
          "ferrite_cold_demag",
          "clamp_preload",
      ];
      let ids = |steps: &[(&'static str, &'static str)]| -> Vec<&'static str> {
  ```
  new:
  ```rust
          "ferrite_cold_demag",
          "shaft_joint",
          "clamp_preload",
      ];
      let ids = |steps: &[(&'static str, &'static str)]| -> Vec<&'static str> {
  ```
  and old:
  ```rust
      assert_eq!(ids(notes::start_here_for(true)), FULL[..13].to_vec());
  ```
  new:
  ```rust
      assert_eq!(ids(notes::start_here_for(true)), FULL[..14].to_vec());
  ```

  1e. The same file: insert a new section just before the three lines
  `// ---...---` / `// A3 traceability` / `// ---...---` (after the closing `}` of `the_ferrite_note_s_entries_hold`,
  ~:4047):
  ```rust
  // ---------------------------------------------------------------------------------------
  // BL-064: the shaft-joint note
  // ---------------------------------------------------------------------------------------

  /// The cure margin as the shaft-joint texts type it, "10 °C": the engine's
  /// `temperature::CURE_MARGIN_MIN_C`, which the bonded inner hub's fitting cap reads (owner
  /// R-15). Built from the constant, so a change to the margin fails every text that types it.
  fn cure_margin_words() -> String {
      format!("{CURE_MARGIN_MIN_C} °C")
  }

  /// BL-064 (spec 6.1, 7.8 and 7.11): what the shaft-joint note says holds at the five probe
  /// designs. Sentence 1 quotes each joint's holding torque at its loosest against the torque each
  /// must hold, the safety factor times cold-high. Sentence 3 quotes the cup's bore stress against
  /// the limit in a "here" clause (review 5 SF-4, review 6 SF-1): per µm of effective
  /// interference the bore stress is E·δ/d times √3/2, about 0.87 (a thick wall), to 1.0 (a thin
  /// one), so it follows interference over diameter and hardly the wall. Sentence 4 quotes the
  /// temperature the inner hub's fit would need beside the wheel shaft's live shrink, never the
  /// shaft's input temperature (review 4 MF-2, review 5 MF-2), against the allowed temperature,
  /// the lowest of the three limits it names with the cure margin typed as the engine keeps it
  /// (owner R-15; review MF-6). The watch-outs carry the owner's "Cup temp" decision (review
  /// MF-1) and name the cooled wheel shaft and the bonded inner hub (review 2 SF-6a); nothing in
  /// the engine checks a seal, a bearing or fretting; the default wheel shaft is colder than the
  /// minimum magnet temperature; a ferrite ring's coercivity falls as it cools. No text of the
  /// note orders the torques (review 2 MF-1, spec 7.8).
  #[test]
  fn the_shaft_joint_note_s_claims_hold_at_every_probe_design() {
      use magcoupling::engine::api::shaft_joint_links;
      use magcoupling::engine::shaft_joint::hub_fit_allowed;
      let note = notes::note("shaft_joint").expect("the shaft-joint note");
      let s = note.sentences;
      assert_eq!(
          values_in(s[1]),
          [
              "shaft_joint.hub_holding_Nm",
              "shaft_joint.cup_holding_Nm",
              "shaft_joint.required_Nm"
          ]
      );
      assert_eq!(
          values_in(s[3]),
          [
              "shaft_joint.cup_bore_stress_MPa",
              "shaft_joint.stress_limit_MPa"
          ]
      );
      assert!(
          s[3].contains("here") && s[3].contains("cup's"),
          "sentence 3 names the joint whose stress it quotes, here: {:?}",
          s[3]
      );
      assert_eq!(
          values_in(s[4]),
          [
              "shaft_joint.hub_fit_temp_C",
              "shaft_joint.wheel_shaft_shrink_um",
              "shaft_joint.hub_fit_allowed_C"
          ]
      );
      let margin = cure_margin_words();
      for words in [
          "Here",
          "wheel shaft",
          "your limit",
          "adhesive's",
          "single-ring onset",
          &format!("onset less {margin}"),
      ] {
          assert!(
              s[4].contains(words),
              "sentence 4 says {words:?}: {:?}",
              s[4]
          );
      }
      let [cup_side, wheel_side] = note.watch_out else {
          panic!("two watch-outs: {:?}", note.watch_out);
      };
      for words in ["bare cup", "quickly", "gearbox seal"] {
          assert!(cup_side.contains(words), "{words:?} in {cup_side:?}");
      }
      for words in [
          "liquid nitrogen",
          "wheel shaft",
          "inner hub",
          "minimum magnet temperature",
      ] {
          assert!(wheel_side.contains(words), "{words:?} in {wheel_side:?}");
      }
      assert!(
          note.sources
              .iter()
              .any(|source| source.contains(&format!("held {margin} below"))),
          "a source names the margin: {:?}",
          note.sources
      );
      for (name, inputs, results) in &probe_points() {
          let r = &results.shaft_joint;
          assert_eq!(
              r.required_Nm,
              inputs.shaft_joint.safety_factor * results.metal.torque_cold_high_Nm,
              "{name}: each must hold the safety factor times cold-high"
          );
          // E [GPa] · δ [µm] / d [mm] in MPa per µm: GPa · 1000 × µm · 0.001.
          let thin_wall = r.part_modulus_GPa / inputs.coupling.bore_mm;
          for joint in shaft_joint::joints(&inputs.shaft_joint, &shaft_joint_links(inputs, results)) {
              let share = joint.stress_per_um / thin_wall;
              assert!(
                  (0.86..=1.0).contains(&share),
                  "{name}, {}: the bore stress is {share} of E·δ/d",
                  joint.name
              );
          }
          let (allowed, _) = hub_fit_allowed(
              inputs.shaft_joint.hub_fit_max_C,
              results.temperature.adhesive.design_limit_C,
              results.temperature.demag.inner_onset_single_ring_C,
          );
          assert_eq!(
              r.hub_fit_allowed_C.to_bits(),
              allowed.to_bits(),
              "{name}: the allowed temperature is the lowest of the three"
          );
          assert!(
              inputs.shaft_joint.wheel_shaft_fit_C < inputs.metal.min_temp_C,
              "{name}: the wheel shaft is fitted colder than the minimum magnet temperature"
          );
      }
      // "nothing here checks the gearbox's shaft, bearings or seal, or fretting": no result
      // names them (a check of one changes this watch-out).
      let unchecked: Vec<String> = result_rows(&compute_all(&DesignInputs::default()))
          .into_iter()
          .map(|row| row.path)
          .filter(|path| {
              let path = path.to_lowercase();
              ["seal", "bearing", "fretting"]
                  .iter()
                  .any(|word| path.contains(word))
          })
          .collect();
      assert!(unchecked.is_empty(), "{unchecked:?}");
      // "ferrite magnets lose coercivity cold": the ferrite probe's rings have a positive Hcj
      // coefficient (the ferrite note's own claim, `the_ferrite_note_s_claims_hold`).
      let ferrite = compute_all(&probe_inputs("Y30 ferrite, custom dimensions"));
      assert!(ferrite.temperature.demag.inner_beta_per_C > 0.0);
      // No order of the torques: cold-high can reach the joints; it is no bound on them.
      let texts = note_texts_with_sources(note);
      for banned in ["strongest", "most expected", "most torque expected"] {
          for text in &texts {
              assert!(
                  !text.to_lowercase().contains(banned),
                  "{banned:?} in {text:?}"
              );
          }
      }
  }

  /// Every text of a note a reader sees: its title, sentences, change entries, watch-outs and
  /// sources.
  fn note_texts_with_sources(note: &notes::Note) -> Vec<&'static str> {
      std::iter::once(note.title)
          .chain(note_text::rendered_texts(note))
          .chain(note.sources.iter().copied())
          .collect()
  }

  /// Review 4, SF-3 (spec 7.11): the four shaft-joint texts that type the cure margin type the
  /// engine's: the helps of the bonded inner hub's fitting limit (`shaft_joint.hub_fit_max_C`)
  /// and of the allowed temperature (`shaft_joint.hub_fit_allowed_C`), the note's sentence 4 and
  /// the note's first source. A change to `CURE_MARGIN_MIN_C` fails all four.
  #[test]
  fn the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps() {
      let margin = cure_margin_words();
      let defaults = DesignInputs::default();
      let input_help = input_rows(&defaults)
          .into_iter()
          .find(|row| row.path == "shaft_joint.hub_fit_max_C")
          .map(|row| row.meta.help)
          .expect("the bonded inner hub's fitting limit");
      let result_help = result_rows(&compute_all(&defaults))
          .into_iter()
          .find(|row| row.path == "shaft_joint.hub_fit_allowed_C")
          .map(|row| row.meta.help)
          .expect("the allowed fitting temperature");
      let note = notes::note("shaft_joint").expect("the shaft-joint note");
      for (what, text, words) in [
          (
              "the input's help",
              input_help,
              format!("onset less {margin}"),
          ),
          (
              "the result's help",
              result_help,
              format!("onset less {margin}"),
          ),
          (
              "sentence 4",
              note.sentences[4],
              format!("onset less {margin}"),
          ),
          (
              "the first source",
              note.sources[0],
              format!("held {margin} below"),
          ),
      ] {
          assert!(text.contains(&words), "{what} says {words:?}: {text:?}");
      }
  }

  ```
  Notes on the code:
  - `values_in`, `probe_points` and `probe_inputs` exist in this file already (BL-059 and slice 2).
  - The share of E·δ/d measured 0.8666 to 0.8884 at the probes in the prototype. The lowest is the 20 mm ring's inner
    hub, Ø39.9, a thick wall near √3/2 = 0.8660. The highest is the 304 parts, E 193 GPa on a 205 GPa shaft.
  - `CURE_MARGIN_MIN_C` is 10.0, and `format!("{}", 10.0_f64)` prints "10".

  1f. `magcoupling-rs/src/gui/explorer.rs`, the tests module. The hover test (~:1460). Old:
  ```rust
      #[test]
      fn the_start_here_hover_names_the_clamp_step_only_while_the_section_shows() {
          assert_eq!(
              start_here_hover(false),
              format!("{}, clamps", start_here_hover(true))
          );
  ```
  New:
  ```rust
      #[test]
      fn the_start_here_hover_names_the_clamp_step_only_while_the_section_shows() {
          assert_eq!(
              start_here_hover(false),
              format!("{}, clamps", start_here_hover(true))
          );
          // BL-064: the shaft-joint step is the last shown while the section is paused, so its
          // topic ends the hover there, after demagnetization's.
          assert!(
              start_here_hover(true).ends_with("demagnetization, shaft joints"),
              "{}",
              start_here_hover(true)
          );
  ```

  1g. The bar test (~:1515). Old:
  ```rust
          // Spec 3.3 and ruling 11: the bar walks START_HERE, the full order of fourteen steps
          // or, while the shaft-clamp section is paused, its first thirteen (the clamp step, the
          // last, hidden); the torque note first; each bar names its step of the steps shown and
          // its note, and only the ferrite step says it is for ferrite rings only (TC-32).
          assert_eq!(notes::start_here_for(false).len(), 14);
          assert_eq!(notes::start_here_for(true).len(), 13);
          assert_eq!(START_HERE[0].0, "torque_from_shear");
          // Two bars pinned as drawn: the first step, and the ferrite step, the 13th in both
          // orders. The count is the steps shown, so the pins hold in both switch states.
  ```
  New:
  ```rust
          // Spec 3.3, ruling 11 and BL-064: the bar walks START_HERE, the full order of fifteen
          // steps or, while the shaft-clamp section is paused, its first fourteen (the clamp step,
          // the last, hidden); the torque note first; each bar names its step of the steps shown
          // and its note, and only the ferrite step says it is for ferrite rings only (TC-32).
          assert_eq!(notes::start_here_for(false).len(), 15);
          assert_eq!(notes::start_here_for(true).len(), 14);
          assert_eq!(START_HERE[0].0, "torque_from_shear");
          // Three bars pinned as drawn: the first step, the ferrite step, the 13th in both orders,
          // and the shaft-joint step, the 14th in both (BL-064). The count is the steps shown, so
          // the pins hold in both switch states.
  ```
  and in the same test, old:
  ```rust
                  "Start here 13 of {steps} (ferrite rings only): {}",
                  ferrite.title
              )
          );
  ```
  new:
  ```rust
                  "Start here 13 of {steps} (ferrite rings only): {}",
                  ferrite.title
              )
          );
          let joints = reviewed_note("shaft_joint").expect("the shaft-joint note is reviewed");
          assert_eq!(
              start_here_text(13),
              format!("Start here 14 of {steps}: {}", joints.title)
          );
  ```

  1h. Insert the render test just before `    #[test]\n    fn every_note_draws_at_every_probe_design_with_glyphs_and_no_braces() {`
  (~:2334). These are the spec's deliberate literal renders (spec 6.1 and 7.11):
  ```rust
      /// BL-064 (spec 6.1 and 7.11): the shaft-joint note's sentence 4 as the panel draws it: the
      /// temperature the inner hub's fit would need, beside the wheel shaft's live shrink, against
      /// the allowed temperature. At the defaults; at library N42 parts in both rings, where the
      /// inner ring's single-ring onset less the cure margin binds (owner R-15; the guide tests'
      /// `breaking(n42_parts)`); on an uncooled wheel shaft, whose shrink reads 0; and on a 40 °C
      /// one, its slider's top, where the shaft grows (the render the physics review decides on).
      #[test]
      fn the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink() {
          use crate::engine::explain::note_text::render;
          use crate::gui::test_support::passing_design;
          let sentence = notes::note("shaft_joint")
              .expect("the shaft-joint note")
              .sentences[4];
          let drawn = |inputs: &DesignInputs| {
              let results = compute_all(inputs);
              render(sentence, |path: &str| live_value(&results, path))
          };
          let wheel_shaft_at = |celsius: f64| {
              let mut inputs = DesignInputs::default();
              inputs.shaft_joint.wheel_shaft_fit_C = celsius;
              inputs
          };
          let mut n42_parts = passing_design();
          n42_parts.coupling.magnets.part_inner = "B842".to_owned();
          n42_parts.coupling.magnets.part_outer = "B842".to_owned();
          for (inputs, words) in [
              (
                  DesignInputs::default(),
                  "would need 93.27 °C on the wheel shaft shrunk 18.99 µm; allowed: 100.0 °C,",
              ),
              (n42_parts, "allowed: 59.83 °C,"),
              (
                  wheel_shaft_at(20.0),
                  "would need 247.6 °C on the wheel shaft shrunk 0 µm;",
              ),
              (
                  wheel_shaft_at(40.0),
                  "would need 266.3 °C on the wheel shaft shrunk -2.300 µm;",
              ),
          ] {
              let text = drawn(&inputs);
              assert!(text.contains(words), "{words:?} in {text:?}");
          }
      }

  ```

  1i. The sign-off sheet, in the same tests module (~:2757-2932). Old:
  ```rust
      /// What the sheet says in place of a new note's old text.
      const NEW_IN_SLICE: &str = "(new in slice 2)";

      /// What the sheet says while the shaft-clamp section is paused (ruling 11).
      const CLAMPS_PAUSED_LINE: &str = "The shaft-clamp section is paused (owner, 2026-10-15): its note, clamp_preload, keeps its earlier text, is not shown on screen and is not part of this sign-off; it returns, rewritten and reviewed, with that section (BL-064).";
  ```
  New:
  ```rust
      /// What the sheet says in place of a new note's old text.
      const NEW_IN_SLICE: &str = "(new in slice 2)";

      /// The notes added after slice 2 (BL-064's shaft joints): no text of theirs before the slice
      /// either, and not the slice's own. The sheet is not regenerated for them (BL-064 spec 6.1):
      /// the owner's review of the diff is the new note's sign-off.
      const AFTER_THE_SLICE: [&str; 1] = ["shaft_joint"];

      /// What the sheet says in place of the old text of a note added after the slice.
      const ADDED_AFTER_THE_SLICE: &str =
          "(added after slice 2; its own physics review is its record)";

      /// What the sheet says while the shaft-clamp section is paused (ruling 11).
      const CLAMPS_PAUSED_LINE: &str = "The shaft-clamp section is paused (owner, 2026-10-15): its note, clamp_preload, keeps its earlier text, is not shown on screen and is not part of this sign-off; the shafts' keyed shrink fits are checked instead (owner, 2026-10-17).";
  ```
  In `signoff_sheet`, old:
  ```rust
                  None => sheet.push_str(&format!("{NEW_IN_SLICE}\n")),
              }
  ```
  new:
  ```rust
                  None if AFTER_THE_SLICE.contains(&note.id) => {
                      sheet.push_str(&format!("{ADDED_AFTER_THE_SLICE}\n"));
                  }
                  None => sheet.push_str(&format!("{NEW_IN_SLICE}\n")),
              }
  ```
  In `the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`, old:
  ```rust
          // T4's snapshot holds the 17 notes there were, each once and with text; the three
          // notes slice 2 added have none.
          let before = parse(SNAPSHOT);
          let ids: Vec<&str> = before.iter().map(|(id, _)| *id).collect();
          assert_eq!(ids.len(), notes::NOTES.len() - NEW_NOTES.len(), "{ids:?}");
  ```
  new:
  ```rust
          // T4's snapshot holds the 17 notes there were, each once and with text; the three
          // notes slice 2 added and the one added after it (BL-064) have none.
          let before = parse(SNAPSHOT);
          let ids: Vec<&str> = before.iter().map(|(id, _)| *id).collect();
          assert_eq!(
              ids.len(),
              notes::NOTES.len() - NEW_NOTES.len() - AFTER_THE_SLICE.len(),
              "{ids:?}"
          );
  ```
  old:
  ```rust
          let mut new_notes = NEW_NOTES.to_vec();
          new_notes.sort_unstable();
          assert_eq!(without, new_notes);
  ```
  new:
  ```rust
          let mut new_notes = [NEW_NOTES.as_slice(), AFTER_THE_SLICE.as_slice()].concat();
          new_notes.sort_unstable();
          assert_eq!(without, new_notes);
  ```
  old:
  ```rust
              assert_eq!(
                  sheet.matches(NEW_IN_SLICE).count(),
                  NEW_NOTES.len(),
                  "paused: {paused}"
              );
              assert_eq!(sheet.contains(CLAMPS_PAUSED_LINE), paused);
  ```
  new:
  ```rust
              assert_eq!(
                  sheet.matches(NEW_IN_SLICE).count(),
                  NEW_NOTES.len(),
                  "paused: {paused}"
              );
              // The note added after the slice is shown in both switch states, so its line shows
              // once in both.
              assert_eq!(
                  sheet.matches(ADDED_AFTER_THE_SLICE).count(),
                  AFTER_THE_SLICE.len(),
                  "paused: {paused}"
              );
              assert_eq!(sheet.contains(CLAMPS_PAUSED_LINE), paused);
  ```
  `CLAMPS_PAUSED_LINE` is spec 6.1's draft, verbatim. The test's last block holds every note's placeholders to a value at
  the default design (`live_value(..).is_some()`). With the new note in `NOTES`, that is the explorer's ship test of spec
  7.11.

  1j. `magcoupling-rs/src/gui/guidance.rs`, the tests module. Insert just before the divider line
  `    // ------------------------------------------------------------------ the fix texts`, so it follows Task 5's
  shaft-joint tests (`the_shaft_joint_check_is_a_dashboard_row_with_its_guide`, then
  `the_shaft_joint_sides_are_the_engine_s_joints`):
  ```rust
      /// BL-064 (spec 6.1 and 7.4): the shaft-joint row's Why link, its guide's and its
      /// `ROW_LEARN` entry's, opens the shaft-joint note, which has passed its physics review and
      /// explains the joint's holding torque, the number Start here opens it on.
      #[test]
      fn the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note() {
          let shaft_joint = guide(SHAFT_JOINT).expect("the shaft joint check has a guide");
          assert_eq!(shaft_joint.learn, Learn::Note("shaft_joint"));
          assert_eq!(row_learn(SHAFT_JOINT), Some(shaft_joint.learn));
          let note = reviewed_note("shaft_joint").expect("the shaft-joint note is reviewed");
          assert!(
              note.equations.contains(&"shaft_joint.hub_holding_Nm"),
              "{:?}",
              note.equations
          );
      }

  ```

  1k. `magcoupling-rs/src/gui/panel.rs`, the tests module. Insert just before
  `    #[test]\n    fn the_failing_filter_lists_the_gearbox_check_with_its_red_badge() {` (after
  `the_gearbox_row_s_own_why_link_opens_the_torque_band_note`). Keep the local `const`: no task adds a module-level
  test constant for `"shaft_joint.check"` to `panel.rs` (BL-059's `GEARBOX_CHECK` has no BL-064 twin there):
  ```rust
      #[test]
      fn the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note() {
          // BL-064: the shaft-joint row's Why link leads to the shaft-joint note, which no other
          // row links, so its title is drawn once on the dashboard, under the row's label and
          // above the next row's. The click edits nothing.
          use crate::gui::dashboard::DASHBOARD;
          use crate::gui::typeset::glyph_safe;
          const SHAFT_JOINT_CHECK: &str = "shaft_joint.check";
          let mut harness = Harness::on_screen(TALL);
          let output = harness.frame(Vec::new());
          let note = crate::gui::explorer::reviewed_note("shaft_joint")
              .expect("the shaft-joint note is reviewed");
          let on_dashboard = |x: f32| x > TALL.x - DASHBOARD_WIDTH;
          let label_top = |path: &str| {
              text_rects(&output, label_for(path))
                  .into_iter()
                  .find(|r| on_dashboard(r.center().x))
                  .unwrap_or_else(|| panic!("{path} on the dashboard"))
                  .top()
          };
          let index = DASHBOARD
              .iter()
              .position(|&(path, _)| path == SHAFT_JOINT_CHECK)
              .expect("the shaft-joint row");
          let (row, next) = (
              label_top(SHAFT_JOINT_CHECK),
              label_top(DASHBOARD[index + 1].0),
          );
          let links: Vec<egui::Pos2> = last_characters(&output, &glyph_safe(note.title))
              .into_iter()
              .filter(|at| on_dashboard(at.x))
              .collect();
          assert_eq!(links.len(), 1, "one link to the note: {links:?}");
          assert!(
              links[0].y > row && links[0].y < next,
              "under the shaft-joint row: {links:?}, row {row}, next {next}"
          );
          harness.click(links[0]);
          assert!(harness.panel.explorer.open);
          assert_eq!(harness.panel.explorer.note(), Some("shaft_joint"));
          assert_eq!(harness.panel.inputs(), &DesignInputs::default());
      }

  ```
  The test runs at the defaults, where the joint passes: the row's Why line is drawn whether or not its check fails.

- [ ] **Step 2: Run the tests; they fail**

  Run:
  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_shaft_joint_note the_shaft_joint_row_s_why_link the_start_here_bar_names_every_step_it_shows the_start_here_hover_names_the_clamp_step_only_while_the_section_shows the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice
  cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- start_here_is_fifteen_steps_and_fourteen_while_paused the_shaft_joint_note_s_claims_hold_at_every_probe_design the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps
  ```
  Expected: the filters select these 7 and 3 tests and no other, and every one FAILED. The lib run fails 7 tests:
  ```
  engine::explain::notes::tests::the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step: the shaft-joint note
  gui::explorer::tests::the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink: the shaft-joint note
  gui::explorer::tests::the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice: assertion `left == right` failed: ["harmonics", "back_iron_factor", ...] (17 against 16)
  gui::explorer::tests::the_start_here_bar_names_every_step_it_shows: assertion `left == right` failed (14 against 15)
  gui::explorer::tests::the_start_here_hover_names_the_clamp_step_only_while_the_section_shows: A suggested order: how the coupling makes torque, the magnet pattern, back iron, pull-out, magnet ends, calibration, temperature, slipping and its heat, demagnetization
  gui::guidance::tests::the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note: assertion `left == right` failed (Learn::None against Learn::Note)
  gui::panel::tests::the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note: the shaft-joint note is reviewed
  ```
  The explain run fails 3 tests:
  ```
  start_here_is_fifteen_steps_and_fourteen_while_paused: assertion `left == right` failed
  the_shaft_joint_note_s_claims_hold_at_every_probe_design: the shaft-joint note
  the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps: the shaft-joint note
  ```

- [ ] **Step 3: Add the note as a draft, its Start here step and its hover topic**

  3a. `magcoupling-rs/src/engine/explain/notes.rs`, the `NOTES` doc (~:150). Old:
  ```rust
  /// Every note: 20, the cap the completeness test holds (15 to 20): the spec's list
  /// (harmonics, back iron, pull-out against angle, end effect, Br(T), demagnetization, slip
  /// heating, the slip temperature, the shaft clamp and the six A5 warnings), ferrite's cold
  /// side (spec A6), the one-point calibration, and slice 2's three: how the coupling makes
  /// torque, the hot-low and cold-high band, and what happens when the coupling slips. The
  /// clamp note stays here, but is not shown while the shaft-clamp section is paused
  /// ([`shown`]).
  ```
  New:
  ```rust
  /// Every note: 21, the cap the completeness test holds (15 to 21; the owner raised it from 20
  /// for the shaft joints, BL-064): the spec's list (harmonics, back iron, pull-out against
  /// angle, end effect, Br(T), demagnetization, slip heating, the slip temperature, the shaft
  /// clamp and the six A5 warnings), ferrite's cold side (spec A6), the one-point calibration,
  /// slice 2's three (how the coupling makes torque, the hot-low and cold-high band, and what
  /// happens when the coupling slips), and the keyed shrink fits that hold the rotors on their
  /// shafts (BL-064). The clamp note stays here, but is not shown while the shaft-clamp section
  /// is paused ([`shown`]).
  ```

  3b. Insert the note just before `    Note {\n        id: "clamp_preload",` (~:898). The texts are spec 6.1's drafts. One
  change entry's text, for Lower `shaft_joint.hub_interference_max_um`, is this plan's draft: the spec gives that entry's
  aim but no text. The physics reviewer sets every word (Step 8).
  ```rust
      Note {
          id: "shaft_joint",
          title: "Shaft joints: a keyed shrink fit",
          // The shaft joint check's 23 numeric results, in their schema order, and the inner
          // ring's own single-ring onset that caps the bonded inner hub's fitting temperature
          // (owner R-15 and its ring ruling): the reviewed demagnetization note keeps the
          // governing onset and the per-ring magnet limits.
          equations: &[
              "shaft_joint.required_Nm",
              "shaft_joint.part_modulus_GPa",
              "shaft_joint.part_cte_per_C",
              "shaft_joint.part_yield_MPa",
              "shaft_joint.stress_limit_MPa",
              "shaft_joint.hub_holding_Nm",
              "shaft_joint.cup_holding_Nm",
              "shaft_joint.hub_bore_stress_MPa",
              "shaft_joint.cup_bore_stress_MPa",
              "shaft_joint.hub_key_pressure_MPa",
              "shaft_joint.cup_key_pressure_MPa",
              "shaft_joint.hub_needed_um",
              "shaft_joint.cup_needed_um",
              "shaft_joint.hub_elastic_max_um",
              "shaft_joint.cup_elastic_max_um",
              "shaft_joint.wheel_shaft_shrink_um",
              "shaft_joint.output_shaft_shrink_um",
              "shaft_joint.hub_fittable_max_um",
              "shaft_joint.cup_fittable_max_um",
              "shaft_joint.hub_fit_temp_C",
              "shaft_joint.cup_fit_temp_C",
              "shaft_joint.hub_fit_allowed_C",
              "shaft_joint.cup_fit_allowed_C",
              "temperature.demag.inner_onset_single_ring_C",
          ],
          sentences: &[
              "A shrink fit holds each rotor on its shaft: the bore is made smaller than the shaft, and friction from the grip carries the torque.",
              "At its loosest the inner hub holds {shaft_joint.hub_holding_Nm} here and the cup {shaft_joint.cup_holding_Nm}; each must hold {shaft_joint.required_Nm}.",
              "Measured interference is not what grips: smoothing takes micrometres off at fitting, and a rotor running warmer than its shaft loses more.",
              "Tighter is not safer here: the cup's bore stress, {shaft_joint.cup_bore_stress_MPa} against {shaft_joint.stress_limit_MPa}, follows interference over diameter, hardly the wall.",
              "Here the inner hub would need {shaft_joint.hub_fit_temp_C} on the wheel shaft shrunk {shaft_joint.wheel_shaft_shrink_um}; allowed: {shaft_joint.hub_fit_allowed_C}, lowest of your limit, adhesive's, single-ring onset less 10 °C.",
          ],
          change: &[
              Change {
                  input: "shaft_joint.cup_interference_min_um",
                  step: Step::Raise,
                  aim: Some(("shaft_joint.cup_holding_Nm", Sense::Up)),
                  text: "the cup grips harder at its loosest; its maximum must still stay within the stress limit and fit.",
              },
              Change {
                  input: "shaft_joint.hub_interference_max_um",
                  step: Step::Lower,
                  aim: Some(("shaft_joint.hub_bore_stress_MPa", Sense::Down)),
                  text: "the inner hub's bore stress and the temperature its fit would need both fall; its minimum must stay at or below it.",
              },
              Change {
                  input: "shaft_joint.wheel_shaft_fit_C",
                  step: Step::Raise,
                  aim: Some(("shaft_joint.hub_fit_temp_C", Sense::Up)),
                  text: "a warmer wheel shaft shrinks less, so the inner hub needs more heat.",
              },
              Change {
                  input: "metal.boss_length_mm",
                  step: Step::Raise,
                  aim: Some(("shaft_joint.cup_holding_Nm", Sense::Up)),
                  text: "a longer boss grips over more length; the overall length grows.",
              },
              Change {
                  input: "shaft_joint.smoothing_um",
                  step: Step::Lower,
                  aim: Some(("shaft_joint.cup_holding_Nm", Sense::Up)),
                  text: "only with a measured finish: the grip rises, and so does the bore stress.",
              },
          ],
          watch_out: &[
              "Fit the bare cup quickly and keep its heat off the gearbox seal; nothing here checks the gearbox's shaft, bearings or seal, or fretting under the reversing swing.",
              "A wheel shaft cooled in liquid nitrogen can chill the bonded inner hub below the minimum magnet temperature, widening the bond's cold swing; ferrite magnets lose coercivity cold.",
          ],
          diagram: None,
          sources: &[
              "The owner's decisions of 2026-10-17: a keyed shrink fit on both rotors, the interference bands, the fitting order and temperatures, and the bonded inner hub held 10 °C below its magnets' single-ring onset, the margin the adhesive cure keeps",
              "Lamé thick-cylinder fit pressure and bore stress (Shigley; Roark)",
              "DIN 7190-1:2017 interference fits: the 0.4·Rz smoothing rule that makes 5 µm conservative for holding; the joining clearance",
              "DIN 6885-1 parallel keys: 3 × 3 for shafts over 8 up to 10 mm",
              "NIST cryogenic material properties: nickel steel Fe-2.25Ni, linear expansion",
              "Friction of dry steel on steel: Gear Technology, Nov/Dec 2017 (0.145); ANSI/AGMA 9003 (0.15)",
          ],
          review: Review::Draft,
      },
  ```
  Notes on the literal:
  - Raw word counts, as `plain::word_count` counts them:
    - sentences: 25, 17, 22, 19, 25;
    - change entries: 18, 22, 13, 11, 14;
    - watch-outs: 28, 28.
  - The wheel shaft's entry is Raise, not Lower. The default is −196 °C, its slider's bottom, so a Lower entry would step
    at no probe design, and `every_change_entry_moves_its_aim_the_way_it_says` fails an aim checked at no design.
  - The "10 °C" in sentence 4 and in the first source is the cure margin. Step 1's
    `the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps` builds it from `CURE_MARGIN_MIN_C`.
  - "decisions of" does not match the plain-words ban `decision \d`.

  3c. Run the cap test; it fails, as it should with a 21st note:
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib a_draft_is_hidden_and_every_note_is_complete`
  Expected: `test engine::explain::notes::tests::a_draft_is_hidden_and_every_note_is_complete ... FAILED`, with the
  message `spec A4: about 15 to 20 notes`.

  3d. Raise the cap (owner decision 8A). In that test (~:1406), old:
  ```rust
          assert!(
              (15..=20).contains(&NOTES.len()),
              "spec A4: about 15 to 20 notes"
          );
  ```
  new:
  ```rust
          assert!(
              (15..=21).contains(&NOTES.len()),
              "spec A4: about 15 to 20 notes; the owner raised the cap to 21 for the shaft joints (BL-064)"
          );
  ```

  3e. The Start here order (~:1196-1218). Old:
  ```rust
  /// The full suggested reading order (slice-2 spec 3.3, 14 steps: how torque is made, the
  /// torque chain, temperature and the torque band, slipping and its heat, demagnetization,
  /// the clamp): (note id, the equation it opens). The clamp step stays last: while the
  /// shaft-clamp section is paused, [`START_HERE`] is the first 13 ([`start_here_for`]).
  ```
  New:
  ```rust
  /// The full suggested reading order (slice-2 spec 3.3 and BL-064, 15 steps: how torque is
  /// made, the torque chain, temperature and the torque band, slipping and its heat,
  /// demagnetization, the shaft joints, the clamp): (note id, the equation it opens). The clamp
  /// step stays last: while the shaft-clamp section is paused, [`START_HERE`] is the first 14
  /// ([`start_here_for`]).
  ```
  and old:
  ```rust
      ("ferrite_cold_demag", "temperature.demag.cold_check"),
      ("clamp_preload", "clamps.capacity_Nm"),
  ];
  ```
  new:
  ```rust
      ("ferrite_cold_demag", "temperature.demag.cold_check"),
      ("shaft_joint", "shaft_joint.hub_holding_Nm"),
      ("clamp_preload", "clamps.capacity_Nm"),
  ];
  ```

  3f. `magcoupling-rs/src/gui/explorer.rs`, the hover's topics (~:116). Old:
  ```rust
          "A suggested order: how the coupling makes torque, the magnet pattern, back iron, pull-out, magnet ends, calibration, temperature, slipping and its heat, demagnetization"
  ```
  New:
  ```rust
          "A suggested order: how the coupling makes torque, the magnet pattern, back iron, pull-out, magnet ends, calibration, temperature, slipping and its heat, demagnetization, shaft joints"
  ```
  and the `start_here_text` doc (~:198). Old:
  ```rust
  /// The start of the start-here bar: "Start here 13 of 13 (ferrite rings only): <title>" on the
  /// ferrite step while the shaft-clamp section is paused (13 of 14 with it), the step and its
  /// title alone on every other one.
  ```
  New:
  ```rust
  /// The start of the start-here bar: "Start here 13 of 14 (ferrite rings only): <title>" on the
  /// ferrite step while the shaft-clamp section is paused (13 of 15 with it), the step and its
  /// title alone on every other one.
  ```

- [ ] **Step 4: Run; the note's own tests pass, and the review gates are red**

  4a. Run Step 2's two commands.
  - The explain run: `test result: ok. 3 passed`.
  - The lib run: 4 of the 7 pass. These 3 still fail, and they pass at Step 10:
    - `the_start_here_bar_names_every_step_it_shows`: "the shaft-joint note is reviewed";
    - `the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note`: the link is still `Learn::None`;
    - `the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`: "the shaft-joint note is reviewed".

  4b. Run both suites:
  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --no-fail-fast
  cargo test --manifest-path magcoupling-rs/Cargo.toml --no-fail-fast
  ```
  Expected: exactly these ten tests fail with `--features app`, and the four not marked "gui" fail without it. Every
  other test passes, these among them:
  - the caps and plain sentence 0, `every_note_keeps_to_the_slice_2_caps` and `every_sentence_zero_is_plain_words`;
  - `every_note_reads_in_plain_words_at_every_probe_design`;
  - `every_note_placeholder_names_a_number_result_and_renders_at_every_probe_design` (every placeholder names a number
    result, sits in a "here" clause and renders at the five probes);
  - `every_change_entry_moves_its_aim_the_way_it_says` (each of the five aims moves its way one step from every probe
    design);
  - `every_change_entry_names_an_input_and_the_result_it_moves`;
  - the glyph tests;
  - `notes_link_to_records_and_each_equation_has_at_most_one` (with Task 3's 23 records and Task 1's inner-onset record,
    every equation of the note has a record and one note);
  - `the_before_snapshot_format_reads_back`;
  - `only_the_clamp_note_is_hidden_and_only_while_the_clamp_section_is_paused`;
  - `the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`.

  The ten failures:
  ```
  release_notes_are_reviewed: not yet reviewed: ["shaft_joint"]
  every_note_names_its_committed_review_section: shaft_joint: not reviewed
  every_note_s_review_date_is_recorded_in_its_section: shaft_joint: not reviewed
  the_docs_count_the_notes_and_the_steps: README.md should say "the 21 A4 teaching notes"
  gui::explorer::tests::every_start_here_step_opens_an_equation_with_its_reviewed_note (gui): shaft_joint.hub_holding_Nm
  gui::explorer::tests::the_start_here_bar_names_every_step_it_shows (gui): the shaft-joint note is reviewed
  gui::guidance::tests::the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note (gui): assertion `left == right` failed
  gui::panel::tests::every_start_here_step_shows_with_glyphs_in_the_default_fonts (gui): shaft_joint: "Start here 14 of 14: "
  gui::panel::tests::every_teaching_note_shows_with_glyphs_in_the_default_fonts (gui): shaft_joint
  gui::panel::tests::the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note (gui): the shaft-joint note is reviewed
  ```
  All ten pass at Step 11 except `the_docs_count_the_notes_and_the_steps`, which passes at Step 6. Make no commit in
  this state.

- [ ] **Step 5: Docs that ship with this change**

  Each old text below occurs exactly once (check each with
  `grep -c -F -- "<old text>" magcoupling-rs/README.md`). This held in the prototype, where every pair of this step was
  applied by a script that asserted each count. If a count is not 1 on the branch, an earlier task's docs step has added
  the same words: anchor on the longer clause around them, and keep the words this step adds. The README and
  `03-structure.yaml` have CRLF line ends: keep them. Leave the words "is hidden with its section until BL-064)" (README
  ~:505) alone: Task 9's Step 5 replaces exactly those words.

  **(a) `magcoupling-rs/README.md` ~:29** (one line). Old:
  ```text
  the 20 A4 teaching notes behind a physics-review gate (20 notes, 19 shown while the clamp section is paused; guidance slice 2 rewrote 16 of the 17 it started from and added 3, each reviewed note by note).
  ```
  New:
  ```text
  the 21 A4 teaching notes behind a physics-review gate (21 notes, 20 shown while the clamp section is paused; guidance slice 2 rewrote 16 of the 17 it started from and added 3, and BL-064 added the shaft joints' note, each reviewed note by note).
  ```

  **(b) ~:504** (one line of the Teaching notes bullet). Old:
  ```text
    slice 2 spec): 20 notes, 19 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`:
  ```
  New:
  ```text
    slice 2 spec): 21 notes, 20 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`:
  ```

  **(c) ~:506-507** (two lines become three). Old:
  ```text
    rewrote 16 of them and added 3 (how the coupling makes torque, the hot-low and cold-high band, what happens when
    the coupling slips); each note shown has three parts: **What it is** (3 to 5 sentences of at most 25 words; the
  ```
  New:
  ```text
    rewrote 16 of them and added 3 (how the coupling makes torque, the hot-low and cold-high band, what happens when
    the coupling slips), and BL-064 added one, the shaft joints' keyed shrink fits (`shaft_joint`); each note shown
    has three parts: **What it is** (3 to 5 sentences of at most 25 words; the
  ```

  **(d) ~:524-526** (from the middle of :524). Old:
  ```text
  "Start here" walks `notes::START_HERE`, its 13 steps while the
    clamp section is paused (the full order, `notes::START_HERE_ALL`, has 14, the clamp note last;
    `notes::start_here_for(clamps::PAUSED)`), from how the coupling makes torque to the cold side of ferrite magnets,
  ```
  New:
  ```text
  "Start here" walks `notes::START_HERE`, its 14 steps while the
    clamp section is paused (the full order, `notes::START_HERE_ALL`, has 15, the shaft-joint note just before the
    clamp note, the clamp note last; `notes::start_here_for(clamps::PAUSED)`), from how the coupling makes torque to
    the shaft joints' keyed shrink fits,
  ```

  **(e) ~:663, the `tests/explain.rs` row.** Old:
  ```text
  `start_here_is_fourteen_steps_and_thirteen_while_paused` pins the full order (`START_HERE_ALL`: 14 steps, the torque note first, the clamp note last), the 13 shown while paused (`start_here_for`)
  ```
  New:
  ```text
  BL-064: `the_shaft_joint_note_s_claims_hold_at_every_probe_design` holds the shaft-joint note at the five probe designs (each joint's holding at its loosest against the safety factor times cold-high; the cup's bore stress per µm of effective interference 0.86 to 1.0 times E·δ/d; the allowed fitting temperature the lowest of the three limits sentence 4 names, quoted beside the wheel shaft's live shrink, never its input temperature; the watch-outs' bare cup, gearbox seal, liquid nitrogen, wheel shaft and inner hub; no result checks a seal, a bearing or fretting; no text orders the torques) and `the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps` holds the margin both helps of the bonded inner hub's fitting limit, sentence 4 and the first source type to `temperature::CURE_MARGIN_MIN_C`; `start_here_is_fifteen_steps_and_fourteen_while_paused` pins the full order (`START_HERE_ALL`: 15 steps, the torque note first, the shaft-joint note just before the clamp note, the clamp note last), the 14 shown while paused (`start_here_for`)
  ```

  **(f) ~:678, the explorer and notes row.** Two replacements. Old:
  ```text
  (T4's snapshot, `notes::before`; the three new notes say so)
  ```
  New:
  ```text
  (T4's snapshot, `notes::before`; the three new notes say so, and the shaft-joint note, added after the slice by BL-064, says its own physics review is its record, in both switch states; the sheet is not regenerated for it)
  ```
  Old:
  ```text
  (`the_start_here_bar_names_every_step_it_shows`: 14 steps in the full order, 13 while the clamp section is paused)
  ```
  New:
  ```text
  (`the_start_here_bar_names_every_step_it_shows`: 15 steps in the full order, 14 while the clamp section is paused, the shaft-joint step the 14th in both; the hover ends with its topic while paused; `the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink` pins its sentence 4 as drawn at the defaults, at library N42 parts, and on a wheel shaft at 20 and 40 °C; `the_shaft_joint_note_keeps_the_caps_owns_the_joint_rows_and_precedes_the_clamp_step`: its 24 equations are the check's 23 numeric results and the inner ring's own single-ring onset)
  ```

  **(g) ~:677, the `src/gui/guidance.rs` row.** Old:
  ```text
  every `Learn::Note` is a reviewed note,
  ```
  New:
  ```text
  every `Learn::Note` is a reviewed note (BL-064: the shaft-joint row's, its guide's and its own, is the reviewed shaft-joint note, `the_shaft_joint_row_s_why_link_opens_the_reviewed_shaft_joint_note`),
  ```

  **(h) ~:670, the `src/gui/panel.rs` row.** Old:
  ```text
  opens that note and changes no input;
  ```
  New:
  ```text
  opens that note and changes no input; BL-064: the shaft-joint row's own Why link, the one copy of the shaft-joint note's title on the dashboard, under the row's label, opens that note and changes no input (`the_shaft_joint_row_s_why_link_opens_the_shaft_joint_note`);
  ```

  **(i) ~:747** (the A4 paragraph, one line). Old:
  ```text
  There are 20, the cap: the spec's list
  ```
  New:
  ```text
  There are 21, the cap (the owner raised it from 20 for the shaft joints, BL-064): the spec's list
  ```
  and old:
  ```text
  what happens when the coupling slips); 20 notes, 19 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`: the shaft-clamp note keeps its earlier text, hidden with its section). All 20 are signed off: the notes slice 2 wrote or rewrote by its per-note physics review (`docs/analyses/2026-10-14-magcoupler-notes-review.md`), the rest by plan A-3 Task 16 and the independent review of 2026-10-01. `START_HERE` opens 13 of them in order while the clamp section is paused (how torque is made, the torque chain, temperature and the torque band, slipping and its heat, demagnetization); the full order, `START_HERE_ALL`, ends with the clamp as step 14.
  ```
  new:
  ```text
  what happens when the coupling slips), and BL-064's shaft joints; 21 notes, 20 shown while the clamp section is paused (`engine::clamps::PAUSED`, `notes::shown`: the shaft-clamp note keeps its earlier text, hidden with its section). All 21 are signed off: the notes slice 2 wrote or rewrote, and the shaft-joint note, by their per-note physics reviews (`docs/analyses/2026-10-14-magcoupler-notes-review.md`), the rest by plan A-3 Task 16 and the independent review of 2026-10-01. `START_HERE` opens 14 of them in order while the clamp section is paused (how torque is made, the torque chain, temperature and the torque band, slipping and its heat, demagnetization, the shaft joints); the full order, `START_HERE_ALL`, ends with the clamp as step 15.
  ```

  **(j) `docs/ai/03-structure.yaml` :63.** Old:
  ```text
  notes.rs NOTES (20 notes, 19 shown while the clamp section is paused; the cap: the 17 A4 notes plus slice 2's torque_from_shear, hot_low_cold_high and when_slipping; START_HERE_ALL, 14 steps in the slice-2 order,
  ```
  New:
  ```text
  notes.rs NOTES (21 notes, 20 shown while the clamp section is paused; the cap: the 17 A4 notes plus slice 2's torque_from_shear, hot_low_cold_high and when_slipping, and BL-064's shaft_joint (the owner raised the cap to 21); START_HERE_ALL, 15 steps in the slice-2 order with shaft_joint (opening shaft_joint.hub_holding_Nm) just before the clamp step,
  ```
  and old:
  ```text
  the order the panel walks, 13 steps while paused;
  ```
  new:
  ```text
  the order the panel walks, 14 steps while paused;
  ```

  **(k) `docs/ai/03-structure.yaml` :74.** Old:
  ```text
  start_here_is_fourteen_steps_and_thirteen_while_paused,
  ```
  New:
  ```text
  start_here_is_fifteen_steps_and_fourteen_while_paused, the_shaft_joint_note_s_claims_hold_at_every_probe_design, the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps,
  ```

  None of the new texts matches Task 9's stale-text patterns ("the 20 A4 teaching notes", "20 notes, 19 shown", "its 13
  steps", "has 14, the clamp note last", "14 steps in the full order, 13", "clamp as step 14", "opens 13 of them",
  "There are 20, the cap", "All 20 are signed off"). The code lines Task 9's table gives this task are `notes.rs` :1196
  and :1604, and `explorer.rs` :2764. Steps 1 and 3 rewrote them, and they no longer match its code patterns
  ("reading order (slice-2 spec 3.3, 14 steps", "Start here's 14 steps").

- [ ] **Step 6: Run the docs test; it passes**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count_the_notes_and_the_steps`
  Expected: `test result: ok. 1 passed`. It reads "the 21 A4 teaching notes", "its 14 steps", "NOTES (21 notes" and, while
  the clamp section is paused, "21 notes, 20 shown while the clamp section is paused" in both files.

- [ ] **Step 7: Write the review dump and the diff for the reviewer**

  ```bash
  NOTES_FOR_REVIEW=shaft_joint NOTES_REVIEW_OUT=target/notes-review-BL064.txt cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture
  mkdir -p magcoupling-rs/target/g3
  git diff -- magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/tests/explain.rs > magcoupling-rs/target/g3/BL064-note.diff
  grep -c "<<<" magcoupling-rs/target/notes-review-BL064.txt
  ```
  Expected:
  - the first command prints `1 notes at 5 designs: target/notes-review-BL064.txt` and `test result: ok. 1 passed`;
  - the `grep -c` prints `0`: no text is refused by the parser or flagged by the guard;
  - in the dump, `==== shaft_joint (Draft)`. Then at each design, in the order default, 8 magnets, 304 back iron, Y30
    ferrite, 20 mm ring (prototype). The dump numbers the sentences from 1 (`i + 1`); the spec, the tests and the brief
    count from 0, so the dump's line `2.` is sentence 1:
    - the dump's `2.` (sentence 1) reads "At its loosest the inner hub holds 9.484 N·m here and the cup 16.90 N·m; each
      must hold 7.647 N·m." The holdings at the other designs are 9.484 / 16.90, 0 / 2.927, 9.484 / 16.90 and
      11.77 / 16.90 N·m. The required torques are 5.598, 4.828, 0.6761 and 7.222 N·m.
    - the dump's `4.` (sentence 3) reads "Tighter is not safer here: the cup's bore stress, 312.5 MPa against
      345.8 MPa, ...", with 346.3 MPa against 170.8 MPa at the 304 design.
    - the dump's `5.` (sentence 4) reads "Here the inner hub would need 93.27 °C on the wheel shaft shrunk 18.99 µm;
      allowed: 100.0 °C, ...", with 73.32 °C at the 304 design.
    - Every change entry's line ends "moves as it says". At the default design, one step moves:
      - the cup's holding from 16.899 to 18.153 N·m (cup minimum);
      - the inner hub's bore stress from 241.70 to 232.74 MPa (hub maximum);
      - the needed temperature from 93.267 to 93.687 °C (wheel shaft);
      - the cup's holding from 16.899 to 17.008 N·m (boss length);
      - the cup's holding from 16.899 to 17.150 N·m (smoothing).

- [ ] **Step 8 (controller): the physics review, one fresh-context reviewer, at most three rounds**

  Dispatch it with the Agent tool: subagent `general-purpose`, `model` omitted (session model: physics, CLAUDE.md section
  5), fresh context. Save each answer verbatim as `.superpowers/sdd/2026-10-17-bl064-shaft-joint/t7-review-r<N>.md`. The
  brief (round 1, verbatim):

  ```text
  You are a fresh-context physics reviewer (session model) for BL-064 in the magnetic coupling calculator (crate
  magcoupling-rs; repo C:/Users/Cole/source/repos/linkage_simulation, branch magcoupling/bl064-shaft-joint). Read-only: do
  not edit the repository and run no git command that changes state. Probe the engine only from a scratch crate outside
  the repository, in your scratchpad, with its own CARGO_TARGET_DIR there. Its Cargo.toml:
      [package]
      name = "bl064review"
      version = "0.1.0"
      edition = "2024"
      publish = false
      [dependencies]
      magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
      [workspace]
  The library is `magcoupling`: magcoupling::engine::api::{DesignInputs, compute_all, shaft_joint_links};
  magcoupling::engine::shaft_joint::{joints, hub_fit_allowed, shaft_shrink_um} and its constants; the five probe designs
  are magcoupling::engine::explain::probes::probe_designs(), and one slider step of an input is probes::step_input. A
  guide's Problem is (g.problem)(&inputs, &results) for g in magcoupling::gui::guidance::CHECK_GUIDES (the shaft joints'
  guide has the label "Shaft joints"). The first build takes several minutes. Use no Markdown heading in your answer (no
  line that starts with "#"): it is pasted verbatim into a report whose "## " lines are section boundaries.

  What BL-064 is. Both rotors are held on their shafts by a keyed shrink fit (owner, 2026-10-17: "Hub is shrink fit with
  key only"). The inner hub, its magnets bonded on its flats, sits on the wheel shaft. The cup, with its magnets in its
  pocket, its rear web and its integral boss, sits on the gearbox output shaft. Both are on the one Ø10 bore. One counted
  dashboard check, shaft_joint.check (engine/shaft_joint.rs), asks four things of each joint, in this order:
  - the fit alone holds the safety factor (2.0) times cold-high at its loosest;
  - the bore stays within yield / 1.2 at its tightest;
  - the part can be fitted within its allowed temperature, with its shaft at its fitting temperature;
  - the key alone, a backup, carries the same torque within the same limit.
  Fit and key capacities are never added. The owner's decisions are in the spec's section 2; read them. They include:
  - one band per joint (inner hub 13-18 µm, cup 15-22 µm, placeholders);
  - the fitting order: the bare cup is shrunk onto the gearbox shaft first, then its magnets are bonded; the inner hub
    is bonded first, then shrunk onto the cooled wheel shaft;
  - the allowed temperatures. The bonded inner hub's is 100 °C, capped at the selected adhesive's design limit. Owner
    R-15 and its ring ruling also cap it at the inner ring's own single-ring onset less 10 °C, CURE_MARGIN_MIN_C. The
    bare cup's is 300 °C;
  - the shaft temperatures: the wheel shaft at -196 °C (liquid nitrogen), the gearbox shaft at 20 °C;
  - 4140 with the bore masked from its finish; the yield 415 MPa (annealed, conservative).

  The note under review: shaft_joint, "Shaft joints: a keyed shrink fit" (engine/explain/notes.rs). It is new: added
  after guidance slice 2, for which the owner raised the note cap from 20 to 21. It is Start here step 14 of 15, just
  before the paused clamp note's, so it is the last step shown while the clamp section is paused. The shaft-joint row's
  Why link will open it. It owns 24 equations: the check's 23 numeric results and the inner ring's own single-ring onset
  (temperature.demag.inner_onset_single_ring_C), so its Explain shows under each of them. Its texts are the spec's
  section 6.1 drafts. One change entry's text, "Lower Inner hub interference, maximum", is the plan's draft: the spec
  gave its aim only. The note at the five probe designs is in magcoupling-rs/target/notes-review-BL064.txt (each change
  entry with its aim's value before and after one slider step). The dump numbers the sentences 1 to 5; this brief, the
  spec and the tests count them 0 to 4, so the dump's "2." is sentence 1. Use 0 to 4 in your answer. The diff is
  magcoupling-rs/target/g3/BL064-note.diff.

  Read:
  - the spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md, sections 1, 2, 3, 4.1 to 4.4, 5.2 (the
    guide's Problem, for consistency), 6.1, 6.2, 10 and 11;
  - the engine behind every claim:
    - engine/shaft_joint.rs (joint, joint_specs, verdict, compute, hub_fit_allowed, shaft_shrink_um, the constants);
    - engine/api.rs (grep "shaft_joint::compute" and "fn shaft_joint_links");
    - engine/temperature.rs (grep "inner_onset_single_ring_C" and "CURE_MARGIN_MIN_C");
    - engine/material_library.rs (grep "fn hub");
    - engine/explain/records/shaft_joint.rs (the 23 records) and records/demagnetization.rs (the inner onset's record);
  - gui/guidance.rs (SHAFT_JOINT_GUIDE, shaft_joint_problem) and gui/dashboard.rs (DASHBOARD, END_EFFECT_ROWS,
    STORED_3D_ROWS, STORED_3D_NOTE);
  - the slice-2 note rules in docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md, sections 3.1 to 3.4 and
    8;
  - for the form of an earlier review, docs/analyses/2026-10-14-magcoupler-notes-review.md, section "## when_slipping".

  Traps (the spec's section 6.2; each one written in the note is a must-fix):
   1. Never add the fit's and the key's torques. The key carries nothing until the fit slips.
   2. Cold-high can reach the joints; it is not a bound on them (backlash, resonance crossings and re-lock ring-down can
      exceed it on the gearbox side), and with the operating temperature below the minimum it is not the strongest
      torque expected either (BL-067). Write "can reach"; never "at most", "the strongest" or "the most expected".
   3. No ratio or efficiency. Both joints are on the output side; never quote metal.cold_high_input_Nm.
   4. 166.7 Hz is the swing frequency at the defaults: 16.7 full cycles, about 33 reversals, per 0.1 s slip event
      (metal.slip_freq_Hz 166.67 Hz x metal.slip_event_s 0.1 s; one cycle is two reversals).
   5. Measured is not effective. Name which one a number is. Assembly uses the measured maximum (no smoothing taken off);
      holding uses the effective minimum.
   6. "Tighter is safer" is false at Ø10. Never print a fit class as a fix ("H7/s6"); say "set the interference by
      measuring the shaft".
   7. Temperature cuts both ways. A part above the shaft's expansion (304, aluminium) loses grip hot; below it (416,
      17-4PH H900), cold. Never "temperature does not affect the fit".
   8. Speed does not loosen this fit (0.0017 µm at 2000 rpm); do not cite it.
   9. Name the limit's source with every fitting temperature: "your limit for the bonded inner hub", "the adhesive's
      design limit", "the magnets' single-ring onset less 10 °C" (owner R-15; the hub's own magnets' onset, the inner
      ring's, by its ring ruling), "your limit for the bare cup". "Rated to 120 °C" is not "safe at 120 °C": at 0.05 mm
      bondline the engine's own mismatch screen passes the lap shear near 113 °C. Never write "safe". A needed
      temperature is what the fit would take, never a temperature to use: quote it against the allowed one.
  10. Do not quote the grade's 150 °C as the magnets' limit for fitting: the rotor alone sees its single-ring field
      (onset 143.4 °C); below the onset the flux lost hot returns on cooling ("reversible", never "no loss").
  11. The gearbox: never suggest cooling its shaft below its rated range, or with dry ice or liquid nitrogen; never
      suggest pressing the cup on against its bearings; name the catalog only as the source of a default (the GAM model
      is unconfirmed); never "protect".
  12. Cryogenic temperatures: helps may name the bath temperatures as facts, never as advice; no text makes -196 °C
      sound routine; a cryogenic shaft can chill the bonded inner hub below the minimum magnet temperature.
  13. No induction heating of a magnetized rotor; if a method is named, "oven" or "hot plate".
  14. "Shrink fit" is not "heat the hub": write "warm the part or cool the shaft"; never "heat until it slides on"
      (seizing half-way is the failure; a stuck rotor cannot be tapped home).
  15. The fitting order is the owner's: the bare cup first, then its magnets bonded; the inner hub bonded, then fitted on
      a cooled shaft. Never "fit the bare hub first".
  16. The torque is not why fitting is hard; do not write "lower the safety factor to make it fit" (SF 1.5 saves about
      1 µm). The finish and the band width decide.
  17. The key row: a Ø10 shaft takes DIN 6885-1 3 x 3; name the row; the 1.7 mm keyway default matches no row.
  18. "Holds without slipping up to ...", never "permanent" or "no backlash": micro-slip at the part's ends happens below
      the slip torque.
  19. Friction and smoothing are assumptions: capacity scales with friction; smoothing lowers grip and stress together.
  20. Nothing checks the gearbox's shaft, bearings or seal, the keyway corner's stress concentration, the bond strain
      from fitting a bonded hub, how the liner and band are held when a rotor is heated, or fretting. Say so where it
      matters; claim none of them is fine.
  21. No "clamp" anywhere, not "clamping pressure": write "contact pressure" or "grip".
  22. Plain words and glyphs: no cell ids, no "BL-064", no "Rust-only"; "->", "Ø", "-20 °C" (the house minus sign); no
      U+2192 or U+2300.

  Rules the words must keep (tests hold them):
  - the caps:
    - 3 to 5 sentences of at most 25 words each;
    - sentence 0 in plain words, with no live value, no symbol and no formula sign;
    - 1 to 5 change entries of at most 25 words each;
    - at most 2 watch-outs of at most 30 words each;
    - words are the runs between whitespace, and a placeholder counts as one word.
  - Live values:
    - a live value appears only as a {result.path} placeholder that names a NUMBER RESULT. Never an input: a placeholder
      that names an input fails the test and reads "no prediction" on screen;
    - it sits inside a clause with the whole word "here" or "on this design";
    - no unit is typed after it;
    - none appears in the title, sentence 0 or a source.
  - Each Raise or Lower change entry names the result it moves and which way. That must hold one slider step from each
    of the five probe designs where the input can step; it does today, and the dump shows each.
  - Plain words: no cell, correction or audit id (TC-, PA-, CG-), no "decision <n>", no "BL-064", no "clamp", no
    "Rust-only"; no "gearbox" with any form of "protect".
  - The tests pin these words. If you change one, say so and say what the test should pin instead:
    - sentence 1's placeholders: shaft_joint.hub_holding_Nm, shaft_joint.cup_holding_Nm and shaft_joint.required_Nm;
    - sentence 3's placeholders, shaft_joint.cup_bore_stress_MPa and shaft_joint.stress_limit_MPa, and its words "here"
      and "cup's";
    - sentence 4's placeholders: shaft_joint.hub_fit_temp_C, shaft_joint.wheel_shaft_shrink_um and
      shaft_joint.hub_fit_allowed_C, never the input shaft_joint.wheel_shaft_fit_C;
    - sentence 4's words "Here", "wheel shaft", "your limit", "adhesive's", "single-ring onset" and "onset less 10 °C".
      The 10 °C is the engine's CURE_MARGIN_MIN_C, which the test formats;
    - exactly two watch-outs: the first with "bare cup", "quickly" and "gearbox seal" (the owner's "Cup temp"
      decision); the second with "liquid nitrogen", "wheel shaft", "inner hub" and "minimum magnet temperature";
    - the first source, with "held 10 °C below";
    - no text with "strongest", "most expected" or "most torque expected".

  Probe at least:
  (1) The five probe designs. Expected:
      - sentence 1: the inner hub 9.484, 9.484, 0, 9.484 and 11.77 N·m; the cup 16.90, 16.90, 2.927, 16.90 and
        16.90 N·m; each must hold 7.647, 5.598, 4.828, 0.6761 and 7.222 N·m;
      - sentence 3: the cup's bore stress 312.5 MPa against 345.8 MPa (346.3 against 170.8 MPa at the 304 probe);
      - sentence 4: 93.27 °C (73.32 °C at the 304 probe) on the wheel shaft shrunk 18.99 µm, allowed 100.0 °C.
  (2) Sentence 4 off the probes, as the panel draws it:
      - the tests' passing design (the defaults with face gap 1.8 mm, cup wall 2.0 mm, requirement 1.6 N·m and
        gearbox input rating 1.0 N·m) with part_inner and part_outer set to "B842": "allowed: 59.83 °C" (the inner
        ring's onset, 69.83 °C, less 10 °C binds);
      - the wheel shaft at 20 °C: "would need 247.6 °C on the wheel shaft shrunk 0 µm"; at -78.5 °C: 163.5 °C;
      - the wheel shaft at 40 °C, its slider's top: "would need 266.3 °C on the wheel shaft shrunk -2.300 µm". The
        spec's section 6.1 leaves this render to you. "Shrunk -2.300 µm" is true (the shrink's help says it is negative
        above 20 °C, where the shaft grows) but reads oddly, and "with a wheel shaft shrink of {...}" costs one word
        over the cap. Decide, and give the words;
      - 2214 Hi-Temp with the hub input typed 150 °C (allowed 133.4 °C, the onset binds);
      - adhesive DP460 (60 °C, the adhesive binds);
      - N50M on manual dimensions in both rings at the passing design (88.80 °C);
      - inner N42M with outer N35 on manual dimensions (81.35 °C: the inner ring's own onset, 91.35 °C, never the
        outer's 59.93 °C);
      - a ferrite inner ring (onset +inf: the cap drops out).
      Is "lowest of your limit, adhesive's, single-ring onset less 10 °C" true and clear at each?
  (3) Each change entry: its direction and its trade-off words at the five probes and at the slider's ends (the dump
      gives one step from each probe).
      - Cup minimum: does "its maximum must still stay within the stress limit and fit" say the right thing? (The
        minimum cannot pass the maximum.)
      - Hub maximum, the plan's draft: do the bore stress and the needed fitting temperature both fall?
      - Wheel shaft: the entry says "needs more heat"; judge it against trap 14.
      - Boss length: "the overall length grows" (metal.axial_stack_mm).
      - Smoothing, an assumption: "only with a measured finish: the grip rises, and so does the bore stress".
  (4) Sentence 3's physics. Per µm of effective diametral interference, for equal moduli, the bore's von Mises stress is
      between sqrt(3)/2 (a thick wall) and 1.0 (a thin wall) times E·δ/d. At the probes it is 0.8666 (the 20 mm ring's
      hub, Ø39.9) to 0.8884 (304, E 193 GPa) times E·δ/d. Does "follows interference over diameter, hardly the wall"
      hold for every back iron and bore on the sliders (6061: E 68.9 GPa on a 205 GPa shaft)?
  (5) The watch-outs.
      - R-4: the bare cup at up to 300 °C (its slider reaches 400 °C) on the gearbox output shaft, toward its seal and
        grease (the GAM catalog's range ends at 90 °C).
      - A heat-cured adhesive then cures the cup's magnets on the fitted cup, which heats the gearbox output again
        (EA 9514 at 120 °C for 60 min; 2214 Hi-Temp at 121 °C; the spec's section 10). Should the note or the
        adhesive's help name it?
      - The liquid-nitrogen chill of the bonded inner hub (traps 12 and 20). A hub settling near -55 °C gives the bond a
        77 K cold swing, against the 62 K the engine's mismatch screen covers (a rough figure).
      - Ferrite at the cold end: the Y30 probe's rings have a positive Hcj coefficient.
  (6) The sources: DIN 6885-1 (3 x 3 for shafts over 8 up to 10 mm); DIN 7190-1:2017 (the 0.4·Rz smoothing rule; the
      joining clearance); NIST's Fe-2.25Ni linear-expansion fit (the coefficients are engine/shaft_joint.rs's
      NIST_FE_2_25NI); Gear Technology Nov/Dec 2017 (0.145); ANSI/AGMA 9003 (0.15). Say which you verified and how, and
      mark anything from memory.
  (7) The spec's section 11 risks, each for a ruling:
      - R-1: the same smoothing is taken off the maximum. At smoothing 0 the cup's 15-22 µm band reads 401.9 MPa
        against 345.8 MPa. Keep it, or take a smaller or no smoothing on the stress side? If kept, say where the
        warning lives: the smoothing input's help, the fit_smoothing rationale, or a watch-out.
      - R-2: the yield factor 1.2; a safety factor of 2 for a reversing torque; friction 0.14; the 5 µm smoothing; the
        0.001·d joining clearance; the shafts' E, ν and expansion; the NIST Fe-2.25Ni curve standing in for the shafts'
        steel; the key flank held to yield / 1.2; the UltraOx process temperature and 4140's tempering temperature,
        against the 300 °C cup default and its 400 °C slider top.
      - R-3: a constant expansion coefficient for the cup warmed to 300 °C (conservative).
      - R-4: the seal side, as above.
      - R-14: which body sits at the bond limit at the hot end. This build puts the shaft at the adhesive's limit and
        the rotor the allowance above it; the other form is bond-limited. The table in the spec's section 11, R-14,
        gives both for five materials.
      For each: CONFIRM, or CHANGE with the exact change, its numbers at the defaults and the five probes, and the texts
      it would change. A CHANGE to the engine's model is the owner's decision; this review does not apply it.

  Return, for the record. It is pasted verbatim under "## shaft_joint" in
  docs/analyses/2026-10-14-magcoupler-notes-review.md, so no line of it may start with "#".
  1. "Reviewer: physics reviewer (session model), fresh context, <date>; round <n>."
  2. Read: what you read, with file and line.
  3. Numbers reproduced from compute_all: at the five probe designs, every placeholder of the note and every aim before
     and after its step; then every extra probe, with its numbers.
  4. Per field (the title; sentences 0 to 4; change entries 0 to 4; watch-outs 0 and 1; sources 0 to 5): PASS, or
     MUST-FIX with the exact replacement text (within the rules above), why (the engine fact or probe that makes the
     draft false or misleading) and the test assertion that would pin the claim. SHOULD-FIX offered, in the same form.
  5. Rulings on R-1, R-2 (each item), R-3, R-4 and R-14: CONFIRM or CHANGE, as asked in (7), each labelled with the
     risk's name.
  6. Traps checked, 1 to 22, one line each.
  7. Form: word counts, raw and rendered at the default design; the placeholder and "here" rule; sentence 0.
  8. Verdict for the note: PASS or MUST_FIX. Also say whether any CHANGE in item 5 must go to the owner.
  Your words win over the drafts within the rules above: write the text you would ship.
  ```

  Rounds 2 and 3 use the same brief, with three additions, and each round is a new fresh context:
  - the round's changes, each field old -> new;
  - "Re-review only the changed fields and the fields whose meaning depends on them; the earlier rounds' rulings on
    R-1 to R-4 and R-14 stand unless a change touches them";
  - the earlier answers' paths.

- [ ] **Step 9: Apply the must-fix items; at most three rounds**

  For each must-fix item on the note, in order:
  1. Where the item names a claim a test can check, write that assertion first:
     - in `the_shaft_joint_note_s_claims_hold_at_every_probe_design`;
     - or, for a drawn render, in `the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink`.
     Run it, see it fail on the draft, then apply the text.
  2. Copy the reviewer's replacement text verbatim into the note's literal in `notes.rs`.
  3. If the item changes a pinned word or placeholder, change the pin in the same edit:
     - the placeholder lists and the word lists in `the_shaft_joint_note_s_claims_hold_at_every_probe_design`;
     - `the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps`, keeping the margin built from `CURE_MARGIN_MIN_C`;
     - the render pins of the explorer test;
     - the README clause of Step 5(e) that names the words.
  4. Re-run Step 4b's suites: the same ten tests and no others are red, less the docs test. Re-run Step 7's dump:
     `grep -c "<<<"` prints 0.

  Then dispatch the next round (Step 8). Stop when a round returns PASS for the note.

  **Rulings on R-1 to R-4 and R-14 (item 5 of the answer).**
  - A CONFIRM is recorded as it stands.
  - A CHANGE that asks only for words in this note is a must-fix on the note, applied as above.
  - A CHANGE that asks for words in another text is handed to Task 8's reviewer as a must-fix input. Examples are R-1's
    warning in the smoothing input's help or in the `fit_smoothing` rationale, and R-4's line in the adhesive's help.
    Task 8's brief carries it.
  - A CHANGE to the engine's model is the owner's decision. Examples are a smaller or no smoothing on the stress side,
    the bond-limited hot end, another yield factor or another expansion at 300 °C. It moves Task 2's pinned numbers and
    the spec's tables. **Stop** before Step 10 and report it to the controller with the ruling verbatim: it is the
    owner's call (ruling C-8). Continue only on the controller's word, and record the decision with its date in Step 11.

  If round 3 still returns a must-fix on the note, **stop**: report it to the controller with the open items. It is an
  owner decision; do not run a fourth round. Should-fix items stay offered and are not applied unless the controller says
  so, as in slice 2 and BL-059. The reviewer's words win over the drafts within the caps.

- [ ] **Step 10: The note passes: it becomes Reviewed, and the Why link flips**

  Take the date on the day the round that returned PASS ran: `PASS_DATE="$(date +%F)"; echo "$PASS_DATE"`. Below and
  in Step 11, `<PASS_DATE>` is the date it printed. A shell variable does not outlive its Bash call, so type the date in.

  10a. `notes.rs`, the note's literal. Old:
  ```rust
          review: Review::Draft,
      },
      Note {
          id: "clamp_preload",
  ```
  New:
  ```rust
          review: Review::Reviewed {
              reviewer: "physics reviewer (session model)",
              date: "<PASS_DATE>",
              record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#shaft_joint",
          },
      },
      Note {
          id: "clamp_preload",
  ```

  10b. `magcoupling-rs/src/gui/guidance.rs`, `SHAFT_JOINT_GUIDE` (Task 5). Old:
  ```rust
      learn: Learn::None("No note covers the shaft joints yet."),
  ```
  New:
  ```rust
      learn: Learn::Note("shaft_joint"),
  ```
  `ROW_LEARN`'s entry, `(SHAFT_JOINT, SHAFT_JOINT_GUIDE.learn)` (Task 5), follows on its own.

  10c. Task 5's `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`, the assertion of the interim link (Task 5
  Step 1c(5) wrote it). Old:
  ```rust
          // Until its reviewed note links it (plan Task 7): no note yet, with the reason.
          assert_eq!(
              joint.learn,
              Learn::None("No note covers the shaft joints yet.")
          );
  ```
  New:
  ```rust
          // Its reviewed note, the shaft-joint note, is its Why link (plan Task 7).
          assert_eq!(joint.learn, Learn::Note("shaft_joint"));
  ```
  Its `assert_eq!(row_learn(SHAFT_JOINT), Some(joint.learn));` stays. No other test reads the interim reason. Check:
  `git grep -n -F "No note covers the shaft joints yet" -- magcoupling-rs` prints nothing (exit 1).

  10d. Run Step 2's lib command. Expected: `test result: ok. 7 passed`.
  Then run:
  `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- release_notes_are_reviewed every_note_names_its_committed_review_section every_note_s_review_date_is_recorded_in_its_section`
  Expected (prototype):
  ```
  test release_notes_are_reviewed ... ok
  test every_note_names_its_committed_review_section ... FAILED
  test every_note_s_review_date_is_recorded_in_its_section ... FAILED
  assertion `left == right` failed: shaft_joint: its section of docs/analyses/2026-10-14-magcoupler-notes-review.md
  shaft_joint: its review date <PASS_DATE> is not recorded under ## shaft_joint
  test result: FAILED. 1 passed; 2 failed
  ```

- [ ] **Step 11: Append the record to `docs/analyses/2026-10-14-magcoupler-notes-review.md`; the gates pass**

  Append only: history is not rewritten. Add the block below after the file's last line, which is the end of
  `## a5.cte_mismatch_with_magnets`, and keep the file's CRLF line ends. Angle-bracket slots are filled from the
  reviewer's answers, from Step 9 and from the controller's decisions, and from nothing else; the rest is verbatim. Leave
  out "Round 2" and "Round 3" where they did not run.

  **No pasted line may start with "#".** The brief asks for none. If an answer has one anyway, indent that line by two
  spaces in the paste and say so in the bullet after "The reviewer's report". Otherwise a stray `## ` line would split
  this section, or open another note's section.

  ```markdown
  ## shaft_joint

  BL-064 (branch `magcoupling/bl064-shaft-joint`; spec
  `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`, sections 6.1, 6.3 and 11). New
  note "Shaft joints: a keyed shrink fit", added after guidance slice 2 (the owner raised the note
  cap from 20 to 21, decision 8A). It is Start here step 14 of 15, just before the clamp note's, so
  it is the last step shown while the shaft-clamp section is paused. It owns the shaft joint check's
  23 numeric results and the inner ring's own single-ring onset
  (`temperature.demag.inner_onset_single_ring_C`), 24 equations. The shaft-joint row's Why link
  leads here. Reviewer: physics reviewer (session model), fresh context. Date: <PASS_DATE>.

  Rounds: <n>. Verdict: PASS.

  - Fields reviewed: the title, 5 sentences, 5 change entries, 2 watch-outs and 6 sources, at the
    five probe designs (`magcoupling-rs/target/notes-review-BL064.txt`, git-ignored), with the diff
    `magcoupling-rs/target/g3/BL064-note.diff`. Each round's full answer is
    `.superpowers/sdd/2026-10-17-bl064-shaft-joint/t7-review-r<N>.md`.
  - Must-fix items applied (old -> new, and why): <"none", or one bullet per item with its round>.
  - Should-fix offered, not applied (open for the controller): <"none", or one bullet per item>.
  - Sentence 4 on a 40 °C wheel shaft ("shrunk -2.300 µm", left to this review by spec 6.1): <the
    reviewer's decision and the words shipped>.
  - Rulings on the spec's section 11 risks, as returned: R-1 <CONFIRM, or CHANGE: ...>; R-2 <each
    item>; R-3 <...>; R-4 <...>; R-14 <...>. <For each CHANGE: what was done with it (applied to
    this note; handed to Task 8's G-3 review; or the owner's decision, with its date).>
  - Tests: `the_shaft_joint_note_s_claims_hold_at_every_probe_design` (`tests/explain.rs`) holds the
    note's claims at the five probe designs: each joint's holding against the safety factor times
    cold-high; the cup's bore stress per µm 0.86 to 1.0 times E·δ/d; the allowed temperature the
    lowest of the three limits sentence 4 names; the watch-outs' words; no result checks a seal, a
    bearing or fretting; no text orders the torques. `the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps`
    builds the "10 °C" of sentence 4, the first source and the two helps from `CURE_MARGIN_MIN_C`.
    `the_shaft_joint_note_draws_the_fit_temperature_beside_the_wheel_shaft_s_shrink` (`gui/explorer.rs`)
    pins sentence 4 as drawn at the defaults, at library N42 parts and on a 20 and a 40 °C wheel
    shaft. `every_change_entry_moves_its_aim_the_way_it_says` holds the five aims at the five probe
    designs. Each failed on the missing note and passes on the shipped text.
  - Sign-off: no new sign-off sheet (spec 6.1, decision 23). The owner's review of the BL-064 diff is
    this note's sign-off; the sheet generator marks it "(added after slice 2; its own physics review
    is its record)", and `docs/analyses/2026-10-14-magcoupler-notes-signoff.md` stays the slice-2
    record.

  The reviewer's report for the record, round by round, as returned: <"verbatim", or "verbatim, with
  the lines that began with '#' indented by two spaces">.

  Round 1:

  <round 1's answer, verbatim>

  Round 2:

  <round 2's answer, verbatim>
  ```

  Check the section, then run the gates:
  ```bash
  grep -c "^## shaft_joint" docs/analyses/2026-10-14-magcoupler-notes-review.md
  tr -d '\r' < docs/analyses/2026-10-14-magcoupler-notes-review.md | awk '/^## /{on=($0=="## shaft_joint")} on' | grep -n "^#"
  tr -d '\r' < docs/analyses/2026-10-14-magcoupler-notes-review.md | awk '/^## /{on=($0=="## shaft_joint")} on' | grep -c -E 'R-1[^0-9]|R-2|R-3|R-4|R-14'
  cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- release_notes_are_reviewed every_note_names_its_committed_review_section every_note_s_review_date_is_recorded_in_its_section
  ```
  Expected:
  - `1`;
  - only `1:## shaft_joint`, plus, after a re-review, its `### Re-review (<date>)` or
    `### Re-review after the whole-branch review (<date>)` lines; any other `#` line is a pasted heading: indent it,
    as above;
  - a count of at least 1 (Task 10 reads the rulings from this section);
  - `test result: ok. 3 passed`.

- [ ] **Step 12: Suites, clippy both ways, fmt**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
  ```
  Expected:
  - every `test result:` line reads `ok`. This task adds, over Task 6's counts:
    - the lib: 4 tests with `--features app`, 1 without;
    - `tests/explain.rs`: 2 in both runs. The renamed Start here test keeps its count.
  - both clippy runs end `Finished` with no warning. The explorer test's closure argument is `celsius`, not a
    unit-suffixed name, so `non_snake_case` does not fire;
  - `--check` prints nothing.

  Measured on the integrated tree of Tasks 1 to 7: the lib 921 passed with `--features app` (Task 6's 917 + 4) and 232
  without (231 + 1); `tests/explain.rs` 63 with `--features app` and 62 without (61 and 60, + 2), every note and review
  gate green.

- [ ] **Step 13: Commit**

  Run `git status --short`. Expected: ` M` on exactly these eight files, and
  `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged):
  - `magcoupling-rs/src/engine/explain/notes.rs`
  - `magcoupling-rs/src/gui/explorer.rs`
  - `magcoupling-rs/src/gui/guidance.rs`
  - `magcoupling-rs/src/gui/panel.rs`
  - `magcoupling-rs/tests/explain.rs`
  - `magcoupling-rs/README.md`
  - `docs/ai/03-structure.yaml`
  - `docs/analyses/2026-10-14-magcoupler-notes-review.md`

  Write the message with the Write tool to `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T7.txt`:

  ```text
  feat(magcoupler): BL-064 shaft-joint note, its Start here step and its physics review (PASS)

  A new reviewed teaching note, shaft_joint ("Shaft joints: a keyed shrink fit"),
  owns the shaft joint check's 23 numeric results and the inner ring's own
  single-ring onset; the owner raised the note cap from 20 to 21. Start here gains
  its step just before the clamp's (15 steps, 14 while the clamp section is
  paused) and its hover topic. The shaft-joint row's Why link, its guide's and its
  row's, now opens the note. The sign-off sheet generator marks it as added after
  slice 2 (no sheet regeneration) and no longer says the clamp returns with
  BL-064. tests/explain.rs holds the note's claims at the five probe designs and
  the cure margin the four texts type to the engine's constant. The note passed a
  fresh-context physics review (<n> rounds), recorded under "## shaft_joint" in
  docs/analyses/2026-10-14-magcoupler-notes-review.md with the reviewer's rulings
  on the spec's risks R-1 to R-4 and R-14.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md docs/ai/03-structure.yaml docs/analyses/2026-10-14-magcoupler-notes-review.md
  git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T7.txt
  ```

  A second commit is used only if a review fix lands after this commit, for example when the controller asks for
  another round after an owner decision. Its subject is `fix(magcoupler): BL-064 shaft-joint note re-review round <n>: …`
  and its message file `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T7-fix.txt`. The same date and record rules
  apply: a new round's PASS date goes into the note and into a `### Re-review (<date>)` heading under `## shaft_joint`,
  which `every_note_s_review_date_is_recorded_in_its_section` accepts.

---

### Task 8: G-3 review of every new on-screen text, and its record

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec sections 6.2 (the traps) and 6.3 (the review's scope). One fresh-context session-model review (slice-1 decision
G-3) covers every new on-screen string of BL-064 outside the note. The audience is engineers who are not magnet
specialists. Nothing may be false at the defaults, at the five probe designs or at any reachable input; the designs of
spec section 3.6 are the reviewer's starting set. Must-fix items are applied where each string lives, once. Tests
follow through the constants, and `tests/data/input_schema.json` is re-blessed when an input's label or help changes.
The record goes into the help-review report under a new `## BL-064 shaft joints` section.

**Files:**
- Modify, only as the review asks:
  - `magcoupling-rs/src/engine/shaft_joint.rs`, `magcoupling-rs/src/engine/temperature.rs`,
    `magcoupling-rs/src/engine/assumptions.rs`;
  - `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/inputs.rs`,
    `magcoupling-rs/src/gui/result_groups.rs`, `magcoupling-rs/src/gui/plain_text.rs`;
  - `magcoupling-rs/tests/data/input_schema.json`, re-blessed, never by hand, when an input's label or help changes;
  - the tests that pin a reworded string by literal, found by grep in Step 3;
  - the doc lines that quote a reworded string.
- Modify: `docs/analyses/2026-10-14-magcoupler-help-review.md`. Append `## BL-064 shaft joints` at the end of the file,
  after `## BL-059 gearbox check`.
- Gitignored outputs:
  - `magcoupling-rs/target/g3/BL064-inputs.txt`, `BL064-guidance.txt`, `BL064-overlay.txt`, `BL064-strings.diff` and
    `BL064-note.txt`;
  - `.superpowers/sdd/2026-10-17-bl064-shaft-joint/t8-g3-r<N>.md`.

**Interfaces:**
- Consumes:
  - Task 1: the inner onset's label and help (`temperature.rs`, `DemagResults`);
  - Task 2:
    - the 15 input labels and helps, the 24 result labels and helps and the 11 verdict constants
      (`UNKNOWN_SAFETY_FACTOR` among them, ruling C-4);
    - the badge tails in `JointTerms::text`;
    - the section labels "Shrink fit" and "Fitting temperatures" and the `SECTION_LABELS` and `PACKAGE_LABELS` entries
      (`("shaft_joint", "Shaft joints")`);
  - Task 3: the four `ASSUMPTIONS` rows (`fit_friction`, `fit_smoothing`, `rotor_over_shaft`, `joining_clearance`) and
    the `CHAIN_LABELS` entry `("shaft_joint", "Shaft joints")`;
  - Task 4: no new on-screen string;
  - Task 5: `SHAFT_JOINT_GUIDE` (label, Why), `shaft_joint_problem` and its sentence templates (the safety factor's of
    ruling C-4 among them), `STORED_3D_NOTE`'s new
    sentence;
  - Task 7: the reviewed note, for consistency, and any ruling of its physics review that asks for a change in a text
    this review covers (Task 7, Step 9);
  - the review tools `help_for_review`, `guidance_strings_for_review`, `overlay_strings_for_review` and
    `notes_for_review`.
- Produces: the reviewed strings (Task 9 quotes them, and Task 10's STOP summary lists them), and the G-3 record.

**The strings under review.** The approximate anchors are from the plan-time prototype; the final lines are what Step 1's
grep prints. "Drawn" is where the string shows with the clamp section paused (the default).

| # | Strings | Where they live (file, anchor) | Drawn |
|---|---|---|---|
| 1 | The 15 input labels and helps (spec 4.1's drafts, as Task 2 shipped them) | `engine/shaft_joint.rs`, the `inputs!` block of `pub struct ShaftJointInputs`: one `param_rust_only(unit, label, help)` per field | The inputs panel's "Shaft and hub joint" group: the two key rows in its first section, then the sections "Shrink fit" and "Fitting temperatures". Each row's hover. The fix texts ("Raise <label>"). The note's change entries (the label as a link). The exports. |
| 2 | The 24 result labels and helps | the same file, the `results!` block of `pub struct ShaftJointResults`: one `out_rust_only(unit, label, help)` per field | The results table's chain group "Shaft joints" (the 23 numbers). The dashboard row: the check's label is the row's label, and its help is the row's hover. The equation panel's heading. The exports. |
| 3 | The eleven verdict constants: `SHAFT_JOINTS_PASS`; `SHAFT_JOINT_UNKNOWN` (the `unknown!` macro) with `UNKNOWN_NOT_A_NUMBER`, `UNKNOWN_NO_WALL`, `UNKNOWN_BAND_REVERSED` and `UNKNOWN_SAFETY_FACTOR` (ruling C-4); `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`, `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT`, `KEY_HAS_NO_BEARING_AREA` | the same file, the `pub const` block after `INNER_HUB` and `CUP` | The dashboard row's value with its badge, the results table, the failing filter, the XLSX Summary |
| 4 | The badge tails: " ({joint}): holds {q} N·m against {r} N·m"; " ({joint}): {q} MPa against {r} MPa" (stress and key); " ({joint}): needs {q} °C against {r} °C"; " ({joint})"; the joint names `INNER_HUB` "inner hub" and `CUP` "cup" | the same file, `impl JointTerms`, `pub fn text(&self, cause: Cause)` (and `verdict` for the unknown forms) | as row 3 |
| 5 | The inner ring's single-ring onset: label "Inner ring: single-ring onset" and its help | `engine/temperature.rs`, `DemagResults`, the row `inner_onset_single_ring_C: f64 => out_rust_only(` | The results table (Demagnetization chain). The equation panel, whose Explain now shows the shaft-joint note. |
| 6 | The guide's label "Shaft joints" and its Why | `gui/guidance.rs`, `const SHAFT_JOINT_GUIDE` (`label`, `why`) | The heading of the guide block; the trade-off lines that name the guide; the Why under the row |
| 7 | The Problem sentences: every form of spec 5.2, and the safety factor's (ruling C-4: "The safety factor, {its value}, is not above zero: enter at least 1.", with no fix) | `gui/guidance.rs`: `fn shaft_joint_problem` (the safety factor's sentence is in it) and, under the names Task 5 gives them, the constants `SHAFT_JOINT_NOT_A_NUMBER` and `SHAFT_JOINT_FALLBACK` and the sentence builders `term_sentence` (slip, stress, fitting), `fitting_source` and `onset_texts` (the allowed temperature's source), `key_sentence`, `window_sentence` and `not_judged_sentences` | The red or amber block under the row |
| 8 | The four assumption labels and rationales (the sources are not reviewed, spec 6.3) | `engine/assumptions.rs`, `ASSUMPTIONS`, the rows `id: "fit_friction"`, `id: "fit_smoothing"`, `id: "rotor_over_shaft"`, `id: "joining_clearance"` | The Assumptions view; Explain's "Assumption behind this value:" line; the term tags |
| 9 | The section labels "Shrink fit" and "Fitting temperatures" | `gui/inputs.rs`, `WORKFLOW`, the sections `id: "shaft.fit"` and `id: "shaft.fitting"` | The inputs panel's section headings |
| 10 | "Shaft joints" in `SECTION_LABELS` | `gui/inputs.rs`, `SECTION_LABELS`, `("shaft_joint", "Shaft joints")` | The input catalogue's heading in workbook order (`inputs.rs` test ~:1729 `workbook[1].heading()`) |
| 11 | "Shaft joints" in `PACKAGE_LABELS` | `gui/result_groups.rs`, `PACKAGE_LABELS` | Never drawn: the package group stays empty and is dropped (spec 5.4). Reviewed for consistency with rows 10 and 12. |
| 12 | "Shaft joints" in `CHAIN_LABELS` | `gui/result_groups.rs`, `CHAIN_LABELS` | The results table's chain group heading, between Geometry and Mass |
| 13 | `STORED_3D_NOTE`'s new sentence (owner R-15, spec 5.1) | `gui/dashboard.rs`, `pub const STORED_3D_NOTE` | The hover of the four stored-3D rows, the shaft-joint row's among them |

Read in context only. These are judged for how the new words read in them, and changed only where a new word makes them
wrong, or as offered:
- C1, spec 5.3's should-fix candidates. Each is now incomplete, not false: the overlay helps of `coupling.bore_mm`,
  `coupling.keyway_depth_mm`, `metal.boss_length_mm` and `temperature.demag.h_rev_single_ring_kA_m` (`gui/plain_text.rs`,
  `HELP`). Their Python-locked labels and engine helps stay; any change is an overlay `HELP` row.
- C2: the fix texts of the shaft joints' block ("Raise Inner hub interference, minimum", ...), built by `fix_text` from
  row 1's labels.
- C3: the status line "Example design: fails 4 of 9 checks"; the shaft-joint row's hover with "Fixed 3D fields from the
  example design"; and Explain's line under the row, "Closest teaching note: Shaft joints: a keyed shrink fit".
- C4: the note `shaft_joint` as Task 7 shipped it, for consistency only (its physics review set its words).

- [ ] **Step 1: Locate the strings and write the review sheets**

  ```bash
  git grep -n -e 'pub struct ShaftJointInputs' -e 'pub struct ShaftJointResults' -e 'pub const INNER_HUB' -e 'pub const CUP' -e 'pub const SHAFT_JOINTS_PASS' -e 'macro_rules! unknown' -e 'pub const UNKNOWN_SAFETY_FACTOR' -e 'pub const FIT_SLIPS' -e 'pub const BORE_OVER_ITS_LIMIT' -e 'pub const CANNOT_BE_FITTED' -e 'pub const KEY_OVER_ITS_LIMIT' -e 'pub const KEY_HAS_NO_BEARING_AREA' -e 'pub fn text(&self, cause: Cause)' -- magcoupling-rs/src/engine/shaft_joint.rs
  git grep -n -e 'inner_onset_single_ring_C: f64 => out_rust_only' -- magcoupling-rs/src/engine/temperature.rs
  git grep -n -e 'id: "fit_friction"' -e 'id: "fit_smoothing"' -e 'id: "rotor_over_shaft"' -e 'id: "joining_clearance"' -- magcoupling-rs/src/engine/assumptions.rs
  git grep -n -e 'const SHAFT_JOINT_GUIDE' -e 'fn shaft_joint_problem' -e 'const SHAFT_JOINT_NOT_A_NUMBER' -e 'const SHAFT_JOINT_FALLBACK' -e '^fn term_sentence' -e '^fn fitting_source' -e '^fn onset_texts' -e '^fn key_sentence' -e '^fn window_sentence' -e '^fn not_judged_sentences' -- magcoupling-rs/src/gui/guidance.rs
  git grep -n -e 'pub const STORED_3D_NOTE' -- magcoupling-rs/src/gui/dashboard.rs
  git grep -n -e 'id: "shaft.fit"' -e 'id: "shaft.fitting"' -e '("shaft_joint", "Shaft joints")' -- magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/result_groups.rs
  git grep -n -e '"coupling.bore_mm",' -e '"coupling.keyway_depth_mm",' -e '"metal.boss_length_mm",' -e '"temperature.demag.h_rev_single_ring_kA_m",' -- magcoupling-rs/src/gui/plain_text.rs
  mkdir -p magcoupling-rs/target/g3
  HELP_REVIEW_GROUPS=shaft HELP_REVIEW_OUT=target/g3/BL064-inputs.txt cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib help_for_review -- --ignored > /dev/null
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/g3/BL064-guidance.txt
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/g3/BL064-overlay.txt
  NOTES_FOR_REVIEW=shaft_joint NOTES_REVIEW_OUT=target/g3/BL064-note.txt cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture > /dev/null
  git diff -U0 main -- magcoupling-rs/src/engine/shaft_joint.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/assumptions.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/gui/plain_text.rs > magcoupling-rs/target/g3/BL064-strings.diff
  grep -c "(shaft_joint\." magcoupling-rs/target/g3/BL064-inputs.txt
  grep -n "Shaft joints" magcoupling-rs/target/g3/BL064-guidance.txt | head -20
  grep -n -A2 -e "\] coupling.bore_mm" -e "\] coupling.keyway_depth_mm" -e "\] metal.boss_length_mm" -e "\] temperature.demag.h_rev_single_ring_kA_m" magcoupling-rs/target/g3/BL064-overlay.txt
  ```
  Expected:
  - Each `git grep` prints its lines. Copy them into the brief below.
  - Each `cargo test` ends `test result: ok. 1 passed`.
  - The input sheet's count is `15`. Its headings run, in the prototype:
    - the bore, the keyway depth, the paused clamp's two key rows (`clamps.key_width_mm` and `clamps.key_contact_mm`,
      listed because the sheet prints every path of the `shaft` section, but hidden on screen and not reviewed), then
      the shaft joints' two key rows, "Shaft key width" and "Shaft key contact height";
    - the eight "Shrink fit" rows and the five "Fitting temperatures" rows;
    - then the paused clamp's rows. These are not shown on screen and not reviewed.
  - The guidance sheet's lines tagged `[why/Shaft joints]`, `[fix/Shaft joints/<design>]`, `[note/Shaft joints/<design>]`
    and `[problem/Shaft joints/<design>]` cover Task 6's `designs()`. On the plan-time prototype of Tasks 5 to 7
    there were 26 fix lines, 9 Problem lines and 2 note lines, and
    `[explain fallback/shaft_joint.check] Closest teaching note: Shaft joints: a keyed shrink fit`.
  - The overlay sheet prints the four context helps of C1, each "now" and "was". The "was" of the bore, the keyway
    depth and the boss length is empty: their engine help is "".

- [ ] **Step 2 (controller): the G-3 review, one fresh-context reviewer, at most three rounds**

  Dispatch it with the Agent tool: subagent `general-purpose`, `model` omitted (session model: physics, CLAUDE.md section
  5), fresh context. Save each answer verbatim as `.superpowers/sdd/2026-10-17-bl064-shaft-joint/t8-g3-r<N>.md`. The brief
  (round 1, verbatim). Paste Step 1's grep output where it says so. Paste Task 7's record of any ruling that asks for a
  change in a text this review covers (Task 7, Step 9), or write "none".

  ```text
  You are a fresh-context physics reviewer (session model) doing the G-3 review of BL-064's new on-screen texts in the
  magnetic coupling calculator (crate magcoupling-rs; repo C:/Users/Cole/source/repos/linkage_simulation, branch
  magcoupling/bl064-shaft-joint). Read-only: do not edit the repository and run no git command that changes state. Probe
  the engine only from a scratch crate outside the repository, in your scratchpad, with its own CARGO_TARGET_DIR there.
  Its Cargo.toml:
      [package]
      name = "bl064g3"
      version = "0.1.0"
      edition = "2024"
      publish = false
      [dependencies]
      magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
      [workspace]
  The library is `magcoupling`: magcoupling::engine::api::{DesignInputs, compute_all}. A guide's Problem is
  (g.problem)(&inputs, &results) for g in magcoupling::gui::guidance::CHECK_GUIDES (the shaft joints' guide has the label
  "Shaft joints"). The five probe designs are magcoupling::engine::explain::probes::probe_designs(). The test designs in
  gui::test_support are crate-private: build them from the inputs given below. The first build takes several minutes. Use
  no Markdown heading in your answer (no line that starts with "#"): it is pasted verbatim into a report whose "## " lines
  are section boundaries.

  Audience: engineers who are not magnet specialists. Plain words. Every string must be true at the default design, at
  the five probe designs and at every reachable input: every value inside its slider, and the typed values the spec names
  (a key width of 40 mm, a key contact height of -0.5 mm, a bore of 0, a bonded-hub input of 150 °C, an end-effect
  coefficient of 1.5, a safety factor of 0 or -1). The sliders:
  - interference minimum and maximum, each joint: 0-40 µm, step 0.5;
  - safety factor 1.0-4.0; friction 0.05-0.30; smoothing 0-15 µm; rotor over shaft 0-60 °C;
  - key width 1-6 mm; key contact height 0.5-3.0 mm;
  - the bonded inner hub's limit 20-130 °C; the bare cup's 20-400 °C;
  - the wheel shaft -196 to 40 °C; the gearbox shaft -20 to 40 °C; the joining clearance 0-30 µm;
  - the bore 4-16 mm; the keyway depth 0-4 mm; the boss length 0-40 mm; the hub length 3-40 mm;
  - the single-ring reverse field up to 1500 kA/m.

  What BL-064 is: the spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md (read sections 1 to 6 and 11;
  section 2 holds the owner's decisions). Both rotors are held on their shafts by a keyed shrink fit: the bonded inner hub
  on the wheel shaft, the cup on the gearbox output shaft. One counted check, shaft_joint.check, asks four things of each
  joint, in this order:
  - the fit alone holds the safety factor times cold-high at its loosest;
  - the bore stays within yield / 1.2 at its tightest;
  - the part can be fitted within its allowed temperature;
  - the key alone, a backup, carries the same torque within the same limit.
  The bonded inner hub's allowed temperature is the lowest of three: the input, the selected adhesive's design limit, and
  the inner ring's own single-ring onset less 10 °C (owner R-15 and its ring ruling).

  The strings under review, with file and line (from git grep):
  <paste Step 1's git grep outputs>
   1 the 15 input labels and helps;
   2 the 24 result labels and helps (the check's label is the dashboard row's label, its help the row's hover);
   3 the eleven verdict constants (the safety factor's unknown form among them);
   4 the badge tails and the joint names;
   5 the inner ring's single-ring onset's label and help;
   6 the guide's label and Why;
   7 every Problem sentence form (spec 5.2), the safety factor's among them ("The safety factor, <its value>, is
     not above zero: enter at least 1.", shown with no fix);
   8 the four assumption labels and rationales (not the sources);
   9 the section labels "Shrink fit" and "Fitting temperatures";
  10 the input catalogue heading "Shaft joints";
  11 the results package heading "Shaft joints" (never drawn: its group is empty; consistency only);
  12 the results chain heading "Shaft joints";
  13 the new sentence of the stored-3D hover note.

  Read in context only. Judge just how the new words read there; change them only where a new word makes them wrong, or
  as offered below:
  - C1, offered by the spec (section 5.3) as should-fix candidates, each now incomplete, not false: the overlay helps
    of
    - the bore ("the shaft in the hub bore": it is now both shafts and both fits);
    - the keyway depth (its Python-locked label is "Keyway depth in the hub", but it now also sets the cup's keyway, caps
      the key's contact height and enters the no-wall rule of both joints);
    - the boss length (it now sets the cup's fit length);
    - the single-ring reverse field ("higher lowers the cure margin": a higher field now also lowers the bonded inner
      hub's allowed fitting temperature).
    Their Python-locked labels and engine helps stay; any change is an overlay help (gui/plain_text.rs, HELP).
  - C2: the fix texts of the shaft joints' block.
  - C3: the status line "Example design: fails 4 of 9 checks"; the joint row's hover with "Fixed 3D fields from the
    example design"; Explain's "Closest teaching note: Shaft joints: a keyed shrink fit".
  - C4: for consistency only, not for review: the note shaft_joint as Task 7 shipped it (its physics review set its
    words; engine/explain/notes.rs).
  Rulings of the note's physics review that ask for a change in a text you review (Task 7, Step 9):
  <paste them, or "none">.

  Sheets in magcoupling-rs/target/g3/:
  - BL064-inputs.txt: the shaft-and-hub-joint group's inputs as drawn, with each help. The clamp's rows in it are the
    paused section's, not shown on screen and not reviewed;
  - BL064-guidance.txt: every guide string at every test design; the shaft joints' lines carry "Shaft joints";
  - BL064-overlay.txt: each overlay row, now and was;
  - BL064-strings.diff: the source diff against main;
  - BL064-note.txt: the note at the five probe designs, for consistency.

  Read the code behind every string:
  - engine/shaft_joint.rs: joint, joint_specs, verdict, compute, hub_fit_allowed and JointTerms::text. Equality passes;
    +inf fails where compared; NaN reads unknown; a failing badge never quotes two equal numbers;
  - engine/api.rs: grep "shaft_joint::compute" and "fn shaft_joint_links";
  - engine/temperature.rs: grep "inner_onset_single_ring_C";
  - engine/assumptions.rs: the four rows;
  - gui/guidance.rs: SHAFT_JOINT_GUIDE, shaft_joint_problem and shaft_joint_fixes;
  - gui/dashboard.rs: DASHBOARD, END_EFFECT_ROWS, STORED_3D_ROWS, STORED_3D_NOTE, verdict_level;
  - gui/inputs.rs: WORKFLOW's "shaft" group and SECTION_LABELS;
  - gui/result_groups.rs: CHAIN_LABELS and PACKAGE_LABELS.

  Traps (the spec's section 6.2; each one written in a string is a must-fix):
   1. Never add the fit's and the key's torques. The key carries nothing until the fit slips.
   2. Cold-high can reach the joints; it is not a bound on them (backlash, resonance crossings and re-lock ring-down can
      exceed it on the gearbox side), and with the operating temperature below the minimum it is not the strongest
      torque expected either (BL-067). Write "can reach"; never "at most", "the strongest" or "the most expected".
   3. No ratio or efficiency. Both joints are on the output side; never quote metal.cold_high_input_Nm.
   4. 166.7 Hz is the swing frequency at the defaults: 16.7 full cycles, about 33 reversals, per 0.1 s slip event
      (metal.slip_freq_Hz 166.67 Hz x metal.slip_event_s 0.1 s; one cycle is two reversals).
   5. Measured is not effective. Name which one a number is. Assembly uses the measured maximum (no smoothing taken off);
      holding uses the effective minimum.
   6. "Tighter is safer" is false at Ø10. Never print a fit class as a fix ("H7/s6"); say "set the interference by
      measuring the shaft".
   7. Temperature cuts both ways. A part above the shaft's expansion (304, aluminium) loses grip hot; below it (416,
      17-4PH H900), cold. Never "temperature does not affect the fit".
   8. Speed does not loosen this fit (0.0017 µm at 2000 rpm); do not cite it.
   9. Name the limit's source with every fitting temperature: "your limit for the bonded inner hub", "the adhesive's
      design limit", "the magnets' single-ring onset less 10 °C" (owner R-15; the hub's own magnets' onset, the inner
      ring's, by its ring ruling), "your limit for the bare cup". "Rated to 120 °C" is not "safe at 120 °C": at 0.05 mm
      bondline the engine's own mismatch screen passes the lap shear near 113 °C. Never write "safe". A needed
      temperature is what the fit would take, never a temperature to use: quote it against the allowed one. Trap 9
      applies to the Problem, the note, the labels and the helps; the badge ("Cannot be fitted at the allowed
      temperature (inner hub): needs 93.3 °C against 93.0 °C") is exempt from naming the source, since it says "allowed"
      and the Problem under it names the source.
  10. Do not quote the grade's 150 °C as the magnets' limit for fitting: the rotor alone sees its single-ring field
      (onset 143.4 °C); below the onset the flux lost hot returns on cooling ("reversible", never "no loss").
  11. The gearbox: never suggest cooling its shaft below its rated range, or with dry ice or liquid nitrogen; never
      suggest pressing the cup on against its bearings; name the catalog only as the source of a default (the GAM model
      is unconfirmed); never "protect".
  12. Cryogenic temperatures: helps may name the bath temperatures as facts, never as advice; no text makes -196 °C
      sound routine; a cryogenic shaft can chill the bonded inner hub below the minimum magnet temperature.
  13. No induction heating of a magnetized rotor; if a method is named, "oven" or "hot plate".
  14. "Shrink fit" is not "heat the hub": write "warm the part or cool the shaft"; never "heat until it slides on"
      (seizing half-way is the failure; a stuck rotor cannot be tapped home).
  15. The fitting order is the owner's: the bare cup first, then its magnets bonded; the inner hub bonded, then fitted on
      a cooled shaft. Never "fit the bare hub first".
  16. The torque is not why fitting is hard; do not write "lower the safety factor to make it fit" (SF 1.5 saves about
      1 µm). The finish and the band width decide.
  17. The key row: a Ø10 shaft takes DIN 6885-1 3 x 3; name the row; the 1.7 mm keyway default matches no row.
  18. "Holds without slipping up to ...", never "permanent" or "no backlash": micro-slip at the part's ends happens below
      the slip torque.
  19. Friction and smoothing are assumptions: capacity scales with friction; smoothing lowers grip and stress together.
  20. Nothing checks the gearbox's shaft, bearings or seal, the keyway corner's stress concentration, the bond strain
      from fitting a bonded hub, how the liner and band are held when a rotor is heated, or fretting. Say so where it
      matters; claim none of them is fine.
  21. No "clamp" anywhere, not "clamping pressure": write "contact pressure" or "grip" (the panel's guard is a
      case-insensitive substring test over texts, paths and cells under details).
  22. Plain words and glyphs: no cell ids, no "BL-064", no "Rust-only"; "->", "Ø", "-20 °C" (the house minus sign); no
      U+2192 or U+2300.

  Caps (tests hold them):
  - Engine labels and helps:
    - an engine label is plain words: no cell id, "workbook", "Rust-only", "decision <n>", correction or audit id;
    - an engine help is plain words, at most 30 words, in one or two sentences, and never "safe" (spec 4.1).
  - The guide:
    - the Why is one sentence of at most 30 words ending in a full stop, with no ". " inside;
    - each Problem sentence stays one sentence;
    - the guide's label is distinct from the other guides' labels, and never "Shaft clamp".
  - Every text: no "clamp" in any case; no "gearbox" with any form of "protect"; glyph-safe. A section label is also
    ASCII.
  - STORED_3D_NOTE: plain words, "example design", no "live", "stay fixed". Its one sentence that names the slip-heating
    fields says no "limit".
  - An overlay help: at most 30 words in one or two sentences, no braces, no "guarantee". A direction it claims is
    pinned on a named design (gui/plain_text.rs, the direction table): say on which design your proposed claim shows.
    The bonded inner hub's allowed temperature moves with the single-ring field only where the onset binds, for
    instance with library B842 parts in both rings, never at the defaults.
  - The badge never quotes two equal numbers (the engine's rounding). Keep the tails' form "{q} ... against {r} ...".
  - The words "onset less 10 °C" in the helps of the bonded inner hub's limit and of the allowed temperature are the
    engine's cure margin; a test builds them from CURE_MARGIN_MIN_C. Keep "onset less 10 °C", or say what the test
    should pin instead.
  - A label you change renames an input or result on screen, in the fix texts and in the note's change entries. Say if
    a new label would collide with another. For instance "Shaft key width" ends with the paused clamp's "Key width", and
    "Gearbox shaft temperature at fitting" is one of the inputs a search for "gearbox" finds.

  Probe at least, from compute_all, and gui::guidance for the Problems:
  (1) The defaults: "Both shaft joints hold and can be fitted", and no Problem. The results:
      - required 7.647 N·m; E 205 GPa, expansion 1.23e-5 /°C, yield 415 MPa, limit 345.8 MPa;
      - holding 9.484 / 16.90 N·m; bore stress 241.7 / 312.5 MPa; key 127.4 / 102.0 MPa;
      - needed 12.08 / 11.31 µm; elastic 23.81 / 23.86 µm; shrink 18.99 / 0 µm; fittable 18.83 / 24.44 µm;
      - fitting 93.27 / 280.2 °C against 100 / 300 °C.
      Is each label and help true there?
  (2) The five probe designs. At the 304 probe: "The fit slips (inner hub): holds 0.00 N·m against 4.83 N·m", with the
      empty-window sentences.
  (3) The spec's section 3.6 designs as the starting set, at least:
      - hub minimum 10 ("holds 3.48 N·m against 7.65 N·m"); cup minimum 10;
      - hub maximum 26 and 28 (385.1 and 421.0 MPa against 345.8 MPa); cup maximum 26 and 28;
      - friction 0.10; smoothing 0 (the cup reads 401.9 MPa);
      - the wheel shaft at 20 °C (needs 247.6 °C) and at -78.5 °C; the gearbox shaft at -20 and 40 °C; the bare cup
        allowed 250 °C;
      - adhesive DP460 (allowed 60 °C); adhesive code 0 (a NaN limit: unknown);
      - 2214 Hi-Temp with the hub input typed 150 °C and the hub maximum 26 µm ("the magnets' single-ring onset,
        143.4 °C, less 10.0 °C");
      - the passing design with N50M on manual dimensions in both rings (88.80 °C); inner N42M with outer N35 on manual
        dimensions (81.35 °C); the passing design with library B842 parts (59.83 °C);
      - keyway depth 0.3 mm (badge "Key backup above its limit (inner hub): 509.8 MPa against 345.8 MPa"); keyway depth
        0 (no key);
      - boss length 0; hub length 3.0 mm;
      - bore 16 mm with a 12 mm boss (no wall, cup); bore 16 mm with keyway 2.1 mm (no wall, inner hub);
      - the hub band 20 over 18 (reversed), and the same with the cup minimum at 10;
      - typed: bore 0, key width 40 mm, contact height -0.5 mm;
      - typed: a safety factor of 0 and of -1 ("Shaft joint check unknown: the safety factor is not above zero", amber;
        its Problem "The safety factor, <its value>, is not above zero: enter at least 1." and no fix);
      - a back-iron pick outside the choices on the steel circuit (not a number); coupling.backiron 0 (aluminium);
      - an axial override of 40 mm (the cup slips);
      - the row greyed: c_end 1.5, typed above its 0-0.5 slider (the spec's section 3.6 row; cold-high reads
        -0.1726 N·m), and the tests' short magnets, inside the sliders (c_end 0.5, both rings on manual dimensions with
        a blank part and 2 mm long magnets);
      - the BL-067 corner (operating temperature -40 °C, minimum 20 °C, variation 0.1).
      The passing design is the defaults with face gap 1.8 mm, cup wall 2.0 mm, requirement 1.6 N·m and gearbox input
      rating 1.0 N·m.
  (4) Each input's label and help at its slider's ends. For instance the gearbox shaft's "Cold spray can take it to
      about -20 °C, the slider's bottom; never below the gearbox's rated range", the wheel shaft's bath temperatures
      (trap 12), and the bonded hub's "the magnets' single-ring onset" (the inner ring's own). Judge
      `shaft_joint.wheel_shaft_fit_C`'s help ('... so the inner hub needs less heat') against trap 14.
  (5) Each result's help at a no-wall design: its holding, bore stress, needed and elastic rows read NaN. Also the key
      pressure's "0 with no keyway; +inf with no bearing area".
  (6) The inner onset's label and help with mixed rings: the passing design with inner B842SH and outer B842 (inner
      onset 143.36 °C; the governing onset 69.83 °C).
  (7) STORED_3D_NOTE at the joint row with the single-ring field at its slider's top, 1500 kA/m ("needs 93.3 °C against
      -12.4 °C").
  (8) The four assumption rationales at their slider ends: friction (holding in proportion), smoothing 0 and 15,
      rotor over shaft 0 and 60, joining clearance 0 and 30.
  (9) No order of the torques: at the BL-067 corner, cold-high 3.1823 N·m is below the operating pull-out with its
      allowance, 3.657 N·m. No string says "strongest", "most expected" or "at most" of cold-high.
  (10) The headings as drawn: the inputs panel's "Shaft and hub joint" group, with its sections "Shaft and hub joint",
      "Shrink fit" and "Fitting temperatures"; the results table's chain group "Shaft joints" between Geometry and Mass.

  Return, for the record. It is pasted verbatim into docs/analyses/2026-10-14-magcoupler-help-review.md, under
  "## BL-064 shaft joints", so no line of it may start with "#".
  1. "Reviewer: physics reviewer (session model), fresh context, <date>; round <n>."
  2. Read: what you read; the probes you ran, with their numbers.
  3. A table, one row per string (its number and field, as listed above): PASS or MUST-FIX, with the reason.
  4. Each MUST-FIX: the string's place (constant, macro row or field), Old, New (the exact text, within the caps), Why
     (the engine fact and the probe that make the old text false or misleading) and, where a claim can be pinned, the
     test assertion.
  5. SHOULD-FIX offered, in the same form. Include the four overlay helps of C1, each PASS as it is, or a proposed
     text.
  6. Traps checked, 1 to 22, one line each. Typed numbers: every number typed in a string, and whether it is allowed.
     The bath temperatures, the DIN 6885-1 row and the cure margin are; no other is.
  7. The context lines: fine, or why not.
  8. Verdict: PASS or MUST_FIX.
  Your words win over the drafts within the caps: write the text you would ship.
  ```

  Rounds 2 and 3 use the same brief with three changes, and each round is a new fresh context:
  - add the round's changes, each string old -> new;
  - add "Re-review only the changed strings and every string whose meaning depends on them; round 1's findings on the
    unchanged strings stand";
  - give the earlier answers' paths.

- [ ] **Step 3: Apply the must-fix items: constants first, tests follow through the constants**

  Edit each must-fix string only where it lives, once (the table's "Where they live" column):
  - **An input's label or help** (row 1): its `param_rust_only(...)` row in `engine/shaft_joint.rs`.
    - Then re-bless the input schema in the same change: `tests/data/input_schema.json` carries every input's label and
      help (Task 2's plan deviation PD-5). Run
      `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`. Expected before the bless:
      `test input_schema_json_is_current ... FAILED`, with "tests/data/input_schema.json is stale: run
      `MAGCOUPLING_BLESS=1 cargo test --test schema`, then regenerate the differential data". Then run
      `MAGCOUPLING_BLESS=1 cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`. Expected:
      `test result: ok. 11 passed`, and `git diff --stat` shows `tests/data/input_schema.json` with one line per changed
      field.
    - The differential data needs no regeneration: the inputs are Rust-only and the Python oracle never sees them.
    - A changed `hub_fit_max_C` help keeps "onset less 10 °C", or Task 7's
      `the_shaft_joint_texts_type_the_cure_margin_the_engine_keeps` changes in the same edit, with the margin still
      built from `CURE_MARGIN_MIN_C`.
    - A renamed label: run `git grep -n -F "<old label>" -- magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai`.
      The known pins:
      - the "gearbox" input filter: `inputs.rs`, Task 2's moved filter test, and `panel.rs`'s
        `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`, if "Gearbox shaft temperature at fitting"
        loses or keeps "gearbox";
      - Task 2's plan deviation PD-6 list in `the_fix_label_is_the_display_label_and_still_names_one_input`, if "Shaft key width" no
        longer ends with "Key width";
      - Task 6's fix-text pins.
      The note's change entries do not type labels.
  - **A result's label or help** (row 2) or the inner onset's (row 5): its `out_rust_only(...)` row (`shaft_joint.rs`,
    `temperature.rs`). `tests/schema.rs`'s `engine_owned_help_is_plain_and_short` and
    `every_result_label_and_help_passes_the_plain_words_guard` hold the caps. A changed `hub_fit_allowed_C` help keeps
    "onset less 10 °C", or Task 7's margin test changes with it, as above.
  - **A verdict constant** (row 3): the constant. `verdict_level`, the classification tests and Task 2's tests read the
    constants.
  - **A badge tail or a joint name** (row 4): the `format!` in `JointTerms::text`, or `INNER_HUB`/`CUP`. Then grep the
    tests that pin whole badges: `git grep -n -e "N·m against" -e "MPa against" -e "°C against" -- magcoupling-rs/tests magcoupling-rs/src/gui`.
    Change Task 2's, Task 5's and Task 6's literal badges in the same edit.
  - **The guide's label or Why** (row 6): the field of `SHAFT_JOINT_GUIDE`. Tests read them through `CHECK_GUIDES`:
    `every_why_is_one_short_sentence`, the distinct-label test, and the plain-words, glyph, clamp and protect bans over
    `every_text()`.
  - **A Problem sentence** (row 7): its `format!` template in the builder that writes it (`term_sentence`,
    `fitting_source`, `key_sentence`, `window_sentence`, `not_judged_sentences`) or its constant
    (`SHAFT_JOINT_NOT_A_NUMBER`, `SHAFT_JOINT_FALLBACK`). Task 5's Problem tests recompute the numbers with the same
    formatters, but they pin the words around them, for example in `shaft_joint_problem_designs()` and the SHAFT_JOINT
    case of `the_problems_quote_numbers_from_the_results`. Find each with
    `git grep -n -F "<old words>" -- magcoupling-rs/src/gui/guidance.rs` and change it in the same edit.
  - **An assumption label or rationale** (row 8): its `ASSUMPTIONS` row. If a test ties the row's label to its input's
    label (grep `tests/assumptions.rs`), change both, and re-bless as above.
  - **A section or group heading** (rows 9-12): the `WORKFLOW` section label, the `SECTION_LABELS`, `PACKAGE_LABELS` or
    `CHAIN_LABELS` entry. The tests that pin the headings follow: `inputs.rs` ~:1724 ("Shaft and hub joint / Fitting
    temperatures") and ~:1729 (`workbook[1].heading()`), and Task 4's result-group tests.
  - **`STORED_3D_NOTE`** (row 13): the constant. Its tests hold the plain-words rules
    (`the_stored_3d_label_and_note_say_the_fields_are_fixed_in_plain_words`,
    `the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits`).
  - **An overlay help of C1**, if the reviewer makes it a must-fix or the controller adopts the should-fix: its `HELP` row
    in `gui/plain_text.rs`, at most 30 words. `every_overlay_text_is_plain_words_with_glyphs_and_within_its_caps` holds
    it, and batch B's direction pins stay true (`metal.boss_length_mm` still raises `metal.axial_stack_mm`). A new
    claim of direction in an overlay help gets a row in that test's direction table (`plain_text.rs`, the
    `(input, design, result, Moves)` rows, designs built by `claim_design`). For instance, if the single-ring field's
    help gains "a higher field lowers the bonded inner hub's allowed fitting temperature", the pin cannot be on
    "example", where your 100 °C limit binds and the allowed temperature stays. Add a claim design where the inner
    ring's onset binds, `"N42 parts" => { set("coupling.magnets.part_inner", Value::Text("B842".to_owned())); set("coupling.magnets.part_outer", Value::Text("B842".to_owned())); }`
    (allowed 59.83 °C, the onset less the cure margin), and the row
    `("temperature.demag.h_rev_single_ring_kA_m", "N42 parts", "shaft_joint.hub_fit_allowed_C", Down)`. See it fail on
    "example" first (it reads Stays there), then pass on "N42 parts".

  Then:
  - run `git grep -n -F "<old text>" -- magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai` for each
    changed string. Replace every test literal that has a constant with the constant. Update every remaining literal and
    every doc line to the new text;
  - where a must-fix names a claim a test can check, write that assertion first, see it fail on the old text, then apply
    the text;
  - run `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app` and
    `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`; all `ok`;
  - rewrite the Step 1 sheets, then dispatch the next round (Step 2).

  At most three rounds. If round 3 still returns a must-fix, **stop** and report to the controller with the open items: it
  is an owner decision. Should-fix items, C1's among them, stay offered and are not applied unless the controller says so.

- [ ] **Step 4: Write the record in `docs/analyses/2026-10-14-magcoupler-help-review.md`**

  Append the block below at the end of the file, after the last line of `## BL-059 gearbox check`, keeping the file's
  CRLF line ends. It has the shape of the BL-059 section. Angle-bracket slots are filled from the reviewer's answers and
  from Step 3; everything else is verbatim. Leave out "Round 2" and "Round 3" if they did not run. If an answer has a line
  starting with "#" despite the brief, indent it by two spaces and say so where the reports begin.

  ```markdown
  ## BL-064 shaft joints

  - Scope: BL-064's new on-screen strings (spec
    `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`, sections 6.2 and 6.3), reviewed
    under the same G-3 rule:
    - the engine (`engine/shaft_joint.rs`): the 15 input labels and helps, the 24 result labels and
      helps (the check's label is the dashboard row's label), the eleven verdict texts
      (`SHAFT_JOINTS_PASS`, the `SHAFT_JOINT_UNKNOWN` forms with the safety factor's, `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`,
      `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT`, `KEY_HAS_NO_BEARING_AREA`) with their tails
      (`JointTerms::text`) and the joint names;
    - the inner ring's single-ring onset (`engine/temperature.rs`): its label and help;
    - the four assumption rows (`engine/assumptions.rs`: `fit_friction`, `fit_smoothing`,
      `rotor_over_shaft`, `joining_clearance`), their labels and rationales;
    - the guide (`gui/guidance.rs`): its label and Why (`SHAFT_JOINT_GUIDE`) and every Problem
      sentence (`shaft_joint_problem`);
    - the headings: "Shrink fit" and "Fitting temperatures" (`gui/inputs.rs`, `WORKFLOW`), "Shaft
      joints" in `SECTION_LABELS`, `CHAIN_LABELS` and `PACKAGE_LABELS` (the last never drawn);
    - the stored-3D hover note's new sentence (`gui/dashboard.rs`, `STORED_3D_NOTE`);
    - read in context: the four overlay helps the spec's section 5.3 offered as should-fix
      candidates (the bore, the keyway depth, the boss length, the single-ring reverse field),
      the fix texts, the status line ("Example design: fails 4 of 9 checks"), the stored-3D label on
      the joint row and Explain's closest-note line; the note `shaft_joint` for consistency only
      (its own physics review is recorded under `## shaft_joint` of
      `docs/analyses/2026-10-14-magcoupler-notes-review.md`).
  - Read: the sheets `magcoupling-rs/target/g3/BL064-inputs.txt` (`help_for_review`, group
    `shaft`), `BL064-guidance.txt` (`guidance_strings_for_review`), `BL064-overlay.txt`
    (`overlay_strings_for_review`), `BL064-note.txt` (`notes_for_review`) and `BL064-strings.diff`
    (`git diff -U0 main` of the source files); the spec, sections 1 to 6 and 11; the engine and
    GUI code behind every string. Each round probed the engine from a scratch crate outside the
    repository: <the probes each round ran, with their numbers, as returned>.
  - Rulings of the note's physics review handed to this review: <"none", or each with what this
    review did with it>.
  - Round 1: <n> must-fix items, text applied verbatim (constants first, tests through the
    constants):
    - `<the constant, macro row or field>`.
      - Old: "<text>"
      - New: "<text>"
      - Why: <the engine fact and the probe, as returned>.

    Should-fix offered, not applied (open for the controller): <"none", or SF-n with its text and
    why>.

    Recorded, no change asked: <the reviewer's observations, as returned>.
  - Round 2: <PASS, no must-fix | n must-fix items, as above>. It re-reviewed the changed strings
    and every string whose meaning depends on them; round 1's findings on the unchanged strings
    stand.
  - What happened to the should-fix items: <applied (where) or left open, each>.
  - Tests that followed the changes: <each test changed, and why | "none: every test reads the
    strings through their constants">. `tests/data/input_schema.json`: <"re-blessed for the changed
    input labels and helps" | "unchanged">.
  - Traps: none of the spec's 22 is written in any string. <Or the trap a round found, and its
    fix.>
  - Verdict: PASS (round <n>).

  ### Report for the record, round 1 (as returned)

  <round 1's answer, verbatim>

  ### Report for the record, round 2 (as returned)

  <round 2's answer, verbatim>
  ```

- [ ] **Step 5: Docs that quote a changed string**

  For every string the review changed, run
  `git grep -n -F "<old text>" -- magcoupling-rs/README.md docs/ai`. It must print nothing once the README and `docs/ai`
  lines quote the new text. Tasks 2 to 5 and 7 wrote those lines; this task updates only the words. If nothing changed,
  there is nothing to do here. Task 9 quotes the final texts in `05-update-tracker.md`.

- [ ] **Step 6: Suites, clippy both ways, fmt**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
  ```
  Expected:
  - every `test result:` line reads `ok`. The counts are Task 7's, plus any assertion test a must-fix added;
  - `input_schema_json_is_current` passes, after the re-bless if an input's text changed;
  - both clippy runs are clean;
  - `--check` prints nothing.

- [ ] **Step 7: Commit (two commits if the review changed strings)**

  Run `git status --short`. Expected: only the files this task touched, and
  `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged).

  **If any round changed strings,** commit the text fixes first. Stage each file by name: the sources, the re-blessed
  `magcoupling-rs/tests/data/input_schema.json` if an input changed, the tests, and the doc lines from Steps 3 and 5.
  Write the message with the Write tool to `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T8-fix.txt`:

  ```text
  fix(magcoupler): BL-064 G-3 review: <strings changed>

  <one paragraph: each string old -> new in a few words, and the probe that made the
  old text false or misleading, from the record; "input_schema.json re-blessed" if an
  input's label or help changed>

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add <each changed source, test, schema and doc file, by name>
  git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T8-fix.txt
  ```

  Then commit the record, with the message file `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T8-record.txt`:

  ```text
  docs(magcoupler): BL-064 G-3 review of the shaft joints' on-screen texts (PASS at round <n>)

  One fresh-context session-model review of the shaft joint check's input and result
  labels and helps, its verdict texts and their tails, the inner ring's single-ring
  onset's label and help, the guide's label, Why and Problems, the four assumption
  rows, the section and group headings and the stored-3D note's new sentence, probed
  at the defaults, the five probe designs and the reachable edges of the spec's
  section 3.6. Recorded in docs/analyses/2026-10-14-magcoupler-help-review.md,
  "## BL-064 shaft joints".

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add docs/analyses/2026-10-14-magcoupler-help-review.md
  git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T8-record.txt
  ```

  **If no round changed strings,** make only the record commit.

---

### Task 9: Docs consistency pass, the docs/ai records and the backlog (BL-064 notes and status, BL-060, BL-066, BL-067, new BL-069 to BL-072)

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Spec section 8. This task writes the docs no code task makes stale (the status and "next" lines, `04-memory`, the
tracker, the backlog) and sweeps every count BL-064 moves. It also builds a small read-only tool, the screen dump,
which prints what the calculator shows at the example design; this task reads the shipped texts and numbers from it
(the BL-059 ruling B-1 practice: the docs quote what ships, never a draft) and Task 10 reuses it for the owner's
summary.

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash, on branch
`magcoupling/bl064-shaft-joint`. Edits are made with the Edit tool (exact old text -> exact new text; the files are
CRLF, which the Edit tool matches). `YYYY-MM-DD` anywhere below means the date `date +%F` prints on the day the step
runs. `W` below is `.superpowers/sdd/2026-10-17-bl064-shaft-joint` (gitignored).

Rules for this task:
- Docs only (the tool lives in the gitignored workspace and builds into the gitignored `magcoupling-rs/target/`).
- `docs/ai/05-update-tracker.md` history is never rewritten: this task adds one new top entry.
- A stale line that T1-T8 own (each task's docs step names its lines; the owner column of Step 4's table says
  which) and that is still stale here means that task's docs step was not applied: apply that task's docs step text
  from this plan verbatim (no new wording here), and name it in the commit body. The lines no other task owns get
  their exact text below.
- Every on-screen text or number this task quotes is checked against what the branch ships (Step 3); where they
  differ, the shipped text is written.
- Each Edit below gives its old text. If an old text is not found, read the line: if an earlier task already wrote
  this step's new text, skip the edit; otherwise STOP and report the line to the controller.

**Files:**
- Create (gitignored): `$W/screen-dump/src/main.rs`, `$W/screen-dump/after/Cargo.toml` (and the `Cargo.lock` cargo
  writes beside it), `$W/check_shipped.py`, `$W/screen-T9.tsv`, `$W/notes-T9.txt`, `$W/msg/T9.txt`; the tool's build
  goes to `magcoupling-rs/target/screen-dump/` (gitignored).
- Modify: `magcoupling-rs/README.md` (the status paragraph's "Next:" sentence, ~:46; two phrases of the
  shaft-clamp pause bullet, ~:459; one phrase of the Teaching notes bullet, ~:505)
- Modify: `docs/ai/04-memory.yaml` (active issues :7 and :8, a new BL-064 active issue; open questions: NEXT :12, a
  new owner-facts question, the pause item :15, the ordering item's trace totals :20)
- Modify: `docs/ai/05-update-tracker.md` (a new top entry)
- Modify: `docs/ai/backlog.yaml` (BL-064 acceptance, status and notes; BL-060 notes; BL-066 acceptance; BL-067
  acceptance and notes; new BL-069, BL-070, BL-071 and BL-072 after BL-068)
- Modify: `docs/ai/02-system.yaml` (the pause line's tag, ~:108; `current_statuses.magcoupling`, ~:486; one
  `known_limitations` entry; the shaft joint invariant is Task 2's, with Task 5's greying and stored-3D sentence)
- Test: no new test. Step 13 runs `tests/explain.rs`'s docs-count tests (`the_docs_count_the_notes_and_the_steps`
  and, with the `gui` feature, `the_docs_count_the_dashboard_rows_and_the_checks`: they read the README and
  `03-structure.yaml`), both suites, both clippy runs and `cargo fmt --check`.

**Interfaces:**
- Consumes: T1-T8 committed: `temperature.demag.inner_onset_single_ring_C`; `engine::shaft_joint`
  (`ShaftJointInputs` (15), `ShaftJointResults` (24), `SHAFT_JOINTS_PASS`, `hub_fit_allowed`); `shaft_joint.check`
  at `DASHBOARD` index 15 (19 rows), `CHECKS` (32), `END_EFFECT_ROWS` (9), `STORED_3D_ROWS` (4), `SHAFT_JOINT_GUIDE`
  in `CHECK_GUIDES` (10), `only_the_joint_fails()`; the sections `shaft.fit` and `shaft.fitting`; the note
  `shaft_joint`; the four `ASSUMPTIONS` rows; the review records (`## shaft_joint` in
  `docs/analyses/2026-10-14-magcoupler-notes-review.md`, `## BL-064 shaft joints` in
  `docs/analyses/2026-10-14-magcoupler-help-review.md`); T1-T8's README and docs/ai lines; the public API the screen
  dump reads (all present at 0687034: `input_rows`, `result_rows`, `InputSet::set`, `ResultSet::get`, `shown_lines`,
  `shown_text`, `check_level`, `design_status`, `design_status_text`, `counted_guides`, `label_for`, `help_for`,
  `WORKFLOW`, `SECTION_LABELS`, `CHAIN_LABELS`, `PACKAGE_LABELS`, `ASSUMPTIONS`, `NOTES`, `START_HERE`,
  `START_HERE_ALL`, `START_HERE_HOVER`, `SCOPE`, `table_entries`, `registry`, `encode_share_payload`,
  `probe_designs`, `PAUSED`, `hides`).
- Produces (Task 10 edits these exact strings after the owner's OK): the tracker heading
  `## 2026-10-17 — Magcoupler shaft joints: the keyed shrink fits that hold the rotors on their shafts (BL-064, branch magcoupling/bl064-shaft-joint)`;
  BL-064 `  status: in_progress` and its notes' last sentence
  `Release (spec section 9) is T10: whole-branch review, GATE PASS with gate 12 running, web build and gui-smoke, then the owner's review of the diff; status stays in_progress until the owner's OK and the push.`;
  the 04-memory active issue that starts `  - "Magcoupler shaft joints (BL-064, branch magcoupling/bl064-shaft-joint;`
  and its phrase
  `built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, gui-smoke) and then the owner's review of the diff come before the merge and the push (risk: physics).`;
  the NEXT phrase `then BL-064, the shaft joints`; the 02-system phrase `built on branch magcoupling/bl064-shaft-joint`;
  the screen dump `$W/screen-dump/` (Task 10 adds its `before/` side) and `$W/check_shipped.py`.

- [ ] **Step 1: Preconditions**

Run:
```bash
git branch --show-current
git status --short
git log --oneline 0687034..HEAD
```
Expected: `magcoupling/bl064-shaft-joint`; `git status` prints only
`?? docs/analyses/2026-05-28-press-4bar-analysis.md`; the log holds the commit of every commit step of T1-T8, each
with the subject that step names (T7 and T8 may add a review-fix commit each), after the plan's commit from Before
Task 1 (and the commit of any later plan edit). Anything else: STOP and finish the owning task first.

- [ ] **Step 2: Build the screen dump and read what the branch ships**

1. Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-dump/src/main.rs` with the Write tool, exactly:

```rust
//! BL-064 release tool (not part of the crate): prints what the calculator shows at the
//! example design (the defaults) as one record per line, `kind<TAB>key<TAB>text`, so two
//! checkouts can be compared line by line. Built against one checkout's `magcoupling-rs`
//! (feature `gui`) by `before/Cargo.toml` or `after/Cargo.toml`. Every text comes from the
//! code's own tables and functions; nothing here is typed from a draft.

use std::collections::BTreeSet;

use magcoupling::engine::assumptions::ASSUMPTIONS;
use magcoupling::engine::clamps::{PAUSED, hides};
use magcoupling::engine::explain::notes::{NOTES, START_HERE, START_HERE_ALL, note, shown};
use magcoupling::engine::explain::probes::probe_designs;
use magcoupling::engine::explain::scope::SCOPE;
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};
use magcoupling::gui::dashboard::{
    CHECKS, DASHBOARD, END_EFFECT_ROWS, STORED_3D_LABEL, STORED_3D_NOTE, STORED_3D_ROWS,
    check_level, shown_lines, shown_text,
};
use magcoupling::gui::explorer::START_HERE_HOVER;
use magcoupling::gui::guidance::{CHECK_GUIDES, counted_guides, design_status, design_status_text};
use magcoupling::gui::inputs::{SECTION_LABELS, WORKFLOW};
use magcoupling::gui::plain_text::{help_for, label_for};
use magcoupling::gui::readouts::registry;
use magcoupling::gui::result_groups::{CHAIN_LABELS, PACKAGE_LABELS};
use magcoupling::gui::results_table::table_entries;
use magcoupling::gui::session::{Design, encode_share_payload};
use magcoupling::{DesignInputs, compute_all};

/// One record: no tab or line break inside a field.
fn out(kind: &str, key: &str, text: &str) {
    let clean = |s: &str| s.replace(['\t', '\r'], " ").replace('\n', " / ");
    println!("{}\t{}\t{}", clean(kind), clean(key), clean(text));
}

fn main() {
    let inputs = DesignInputs::default();
    let results = compute_all(&inputs);

    // The status line, while the shaft-clamp section is paused (as shipped) and not.
    for (key, paused) in [("paused", true), ("unpaused", false)] {
        out(
            "status",
            key,
            &design_status_text(&design_status(&inputs, &results, paused)),
        );
    }

    // The dashboard rows as drawn (the section's pause as shipped), then each row's hover.
    let lines = shown_lines(&results, PAUSED);
    for (i, line) in lines.iter().enumerate() {
        let mut text = format!("{}: {}", label_for(line.path), line.shown_value());
        if let Some(level) = line.level {
            text.push_str(&format!(" [{level:?}]"));
        }
        if line.greyed {
            text.push_str(" (greyed)");
        }
        if line.stored_3d {
            text.push_str(&format!(" ({STORED_3D_LABEL})"));
        }
        out("row", line.path, &format!("{:02}. {text}", i + 1));
    }
    for line in &lines {
        out("row hover", line.path, &line.tooltip(false));
    }
    // Each check guide's label and Why (the Problem and fixes are in the guidance dump).
    for guide in CHECK_GUIDES.iter() {
        out("guide label", guide.check, guide.label);
        out("guide why", guide.check, guide.why);
    }
    out("stored 3D", "label", STORED_3D_LABEL);
    out("stored 3D", "note", STORED_3D_NOTE);
    let last_stored = lines.iter().rev().find(|line| line.stored_3d);
    out(
        "stored 3D",
        "label drawn after",
        &last_stored.map_or(String::from("(none)"), |line| {
            label_for(line.path).to_owned()
        }),
    );

    // Every input: its on-screen label and help, unit, default as shown, slider, and whether
    // the paused section hides it.
    for row in input_rows(&inputs) {
        let range = row.meta.range.map_or(String::new(), |r| {
            format!(" slider {} to {} step {}", r.min, r.max, r.step)
        });
        let hidden = if hides(&row.path, PAUSED) {
            " (hidden while the clamp section is paused)"
        } else {
            ""
        };
        out(
            "input",
            &row.path,
            &format!(
                "{} [{}] default {}{range}{hidden}{}: {}",
                label_for(&row.path),
                row.meta.unit,
                shown_text(&row.path, &row.value),
                if row.meta.assumption {
                    " (assumption)"
                } else {
                    ""
                },
                help_for(&row.path)
            ),
        );
    }

    // Every result at the defaults: label, unit, value as the table shows it, help.
    for row in result_rows(&results) {
        out(
            "result",
            &row.path,
            &format!(
                "{} [{}] = {}: {}",
                label_for(&row.path),
                row.meta.unit,
                shown_text(&row.path, &row.value),
                help_for(&row.path)
            ),
        );
    }

    // The assumptions view.
    for a in ASSUMPTIONS.iter() {
        out(
            "assumption",
            a.id,
            &format!(
                "{}: {} | inputs {} | source {}",
                a.label,
                a.rationale,
                a.paths.join(", "),
                a.source
            ),
        );
    }

    // The workflow input groups as shown while paused: each section's label and its rows.
    for group in WORKFLOW.iter() {
        for section in group.sections.iter() {
            let shown_paths: Vec<&str> = section
                .paths
                .iter()
                .copied()
                .filter(|p| !hides(p, PAUSED))
                .collect();
            if shown_paths.is_empty() {
                continue;
            }
            let labels: Vec<&str> = shown_paths.iter().map(|p| label_for(p)).collect();
            out(
                "input section",
                section.id,
                &format!(
                    "{} > {}{}: {}",
                    group.label,
                    section.label,
                    if section.advanced { " (advanced)" } else { "" },
                    labels.join("; ")
                ),
            );
        }
    }

    // Teaching notes and the Start here walk.
    for n in NOTES.iter() {
        let state = if shown(n.id) { "shown" } else { "hidden" };
        out("note", n.id, &format!("{} ({state})", n.title));
    }
    out("start here", "hover", START_HERE_HOVER);
    for (i, (id, _)) in START_HERE.iter().enumerate() {
        let title = note(id).map_or("(no note)", |n| n.title);
        out("start here", &format!("step {:02}", i + 1), title);
    }

    // The share link of the default design.
    out(
        "share link",
        "default design",
        &format!(
            "{} characters",
            encode_share_payload(&Design::default()).len()
        ),
    );

    // The shaft-joint row's verdict and the status line at the five probe designs and at the
    // spec's probes of each cause (BL-064 spec sections 3.6 and 4.2). An edit this checkout
    // has no input for, or a value `set` refuses, is reported, never skipped silently.
    let mut designs: Vec<(String, Result<DesignInputs, String>)> = probe_designs()
        .into_iter()
        .map(|(name, inputs)| (format!("probe: {name}"), Ok(inputs)))
        .collect();
    let edits: [(&str, &[(&str, Value)]); 12] = [
        (
            "inner hub minimum 10 µm",
            &[("shaft_joint.hub_interference_min_um", Value::Num(10.0))],
        ),
        (
            "cup minimum 10 µm",
            &[("shaft_joint.cup_interference_min_um", Value::Num(10.0))],
        ),
        (
            "inner hub maximum 26 µm",
            &[("shaft_joint.hub_interference_max_um", Value::Num(26.0))],
        ),
        (
            "cup maximum 26 µm",
            &[("shaft_joint.cup_interference_max_um", Value::Num(26.0))],
        ),
        (
            "wheel shaft at 20 °C when fitting",
            &[("shaft_joint.wheel_shaft_fit_C", Value::Num(20.0))],
        ),
        (
            "bare cup allowed 250 °C",
            &[("shaft_joint.cup_fit_max_C", Value::Num(250.0))],
        ),
        (
            "keyway depth 0.3 mm",
            &[("coupling.keyway_depth_mm", Value::Num(0.3))],
        ),
        (
            "shaft boss length 0 mm",
            &[("metal.boss_length_mm", Value::Num(0.0))],
        ),
        (
            "inner hub band 20 to 18 µm",
            &[("shaft_joint.hub_interference_min_um", Value::Num(20.0))],
        ),
        (
            "bore 16 mm in a 12 mm boss",
            &[
                ("coupling.bore_mm", Value::Num(16.0)),
                ("metal.boss_od_mm", Value::Num(12.0)),
            ],
        ),
        (
            "library N42 parts in both rings",
            &[
                (
                    "coupling.magnets.part_inner",
                    Value::Text(String::from("B842")),
                ),
                (
                    "coupling.magnets.part_outer",
                    Value::Text(String::from("B842")),
                ),
            ],
        ),
        (
            "the DP460 adhesive (60 °C)",
            &[("temperature.adhesive.selected", Value::Int(4))],
        ),
    ];
    for (name, list) in edits {
        let mut design = DesignInputs::default();
        let applied = list.iter().try_for_each(|(path, value)| {
            design
                .set(path, value.clone())
                .map_err(|e| format!("{path}: {e:?}"))
        });
        designs.push((name.to_owned(), applied.map(|()| design)));
    }
    for (name, design) in designs {
        let text = match design {
            Err(why) => format!("(not on this checkout: {why})"),
            Ok(design) => {
                let results = compute_all(&design);
                let verdict = results.get("shaft_joint.check").map_or_else(
                    || String::from("(no shaft-joint check)"),
                    |value| shown_text("shaft_joint.check", &value),
                );
                let level = check_level(&results, "shaft_joint.check")
                    .map_or(String::new(), |level| format!(" [{level:?}]"));
                let status = design_status_text(&design_status(&design, &results, PAUSED));
                format!("{verdict}{level} | {status}")
            }
        };
        out("verdict", &name, &text);
    }

    // The counts the docs and the pinned tests quote.
    let explained = table_entries()
        .iter()
        .filter(|e| registry().equation_for(&e.path).is_some())
        .count();
    let scope: BTreeSet<&str> = SCOPE.iter().flat_map(|c| c.paths.iter().copied()).collect();
    for (key, n) in [
        ("inputs", input_rows(&inputs).len()),
        ("results", result_rows(&results).len()),
        ("results explained", explained),
        ("scope paths", scope.len()),
        ("scope chains", SCOPE.len()),
        ("dashboard rows", DASHBOARD.len()),
        ("dashboard rows shown while paused", lines.len()),
        ("checks", CHECKS.len()),
        ("guides", CHECK_GUIDES.len()),
        ("guides counted while paused", counted_guides(true).count()),
        ("end-effect rows", END_EFFECT_ROWS.len()),
        ("stored-3D rows", STORED_3D_ROWS.len()),
        ("notes", NOTES.len()),
        ("notes shown", NOTES.iter().filter(|n| shown(n.id)).count()),
        ("start here steps shown", START_HERE.len()),
        ("start here steps in all", START_HERE_ALL.len()),
        ("assumptions", ASSUMPTIONS.len()),
        ("input sections", SECTION_LABELS.len()),
        ("result chain groups", CHAIN_LABELS.len()),
        ("result package groups", PACKAGE_LABELS.len()),
    ] {
        out("count", key, &n.to_string());
    }
}
```

2. Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-dump/after/Cargo.toml`, exactly (Task 10 adds a
   `before/` twin that points at `main`'s checkout):

```toml
[package]
name = "screen-dump-after"
version = "0.1.0"
edition = "2024"
publish = false

# Its own workspace: a BL-064 release tool, not part of any crate of the repository.
[workspace]

[[bin]]
name = "screen-dump-after"
path = "../src/main.rs"

[dependencies]
magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
```

3. Run (the first build takes several minutes):
```bash
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
cp magcoupling-rs/Cargo.lock $W/screen-dump/after/Cargo.lock
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path $W/screen-dump/after/Cargo.toml > $W/screen-T9.tsv
awk -F'\t' '{print $1}' $W/screen-T9.tsv | sort | uniq -c
NOTES_FOR_REVIEW=shaft_joint NOTES_REVIEW_OUT="$PWD/$W/notes-T9.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture 2>&1 | grep -E 'notes at|^test result'
awk '/^-- at the /{d=($0=="-- at the default design"); on=0; next} d && /^Watch out:/{on=1; next} /^[A-Z]/{on=0} d && on' $W/notes-T9.txt
git status --short
```
Expected: the record counts `18 assumption`, `20 count`, `10 guide label`, `10 guide why`, `187 input`,
`34 input section`, `21 note`, `1112 result`, `18 row`, `18 row hover`, `1 share link`, `15 start here`, `2 status`,
`3 stored 3D`, `17 verdict` (measured at 0687034 for the tool's dry run: 14, 20, 9, 9, 172, 32, 20, 1087, 17, 17, 1,
14, 2, 3, 17; the 34 sections are the 32 shown today plus T2's `shaft.fit` and `shaft.fitting`); the notes run prints `1 notes at 5 designs: ...` and
`test result: ok. 1 passed`; the awk prints the note's watch-outs at the default design (two lines starting
`  - `), which Step 10 reads; `git status --short` prints only the owner's untracked file.

- [ ] **Step 3: Check every text and number this task quotes against what ships (ruling B-1)**

Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/check_shipped.py` with the Write tool, exactly:

```python
"""BL-064 T9 Step 3 (ruling B-1 of BL-059, kept for BL-064): every on-screen text and number
Task 9's docs quote, checked against what the branch ships (the screen-dump tool's records).
Prints one line per quote: OK, or MISMATCH with the shipped text; exits 1 on any MISMATCH.

Usage: python -I check_shipped.py SCREEN_TSV
"""

import sys

# (kind, key, how, the text Task 9 quotes, where Task 9 quotes it)
# how: "has" = the record contains the text; "starts" = it starts with it; "is" = equal.
QUOTES = [
    ("status", "paused", "is", "Example design: fails 4 of 9 checks", "tracker, BL-064 notes, 04-memory items 2 and 3"),
    ("status", "unpaused", "is", "Example design: fails 4 of 10 checks", "tracker, BL-064 notes"),
    ("row", "shaft_joint.check", "has", "Both shaft joints hold and can be fitted [Good]", "tracker, BL-064 notes"),
    ("guide label", "shaft_joint.check", "is", "Shaft joints", "tracker (the guide's label)"),
    ("note", "shaft_joint", "starts", "Shaft joints: a keyed shrink fit (shown)", "tracker (the note's title)"),
    ("input section", "shaft.fit", "has", "> Shrink fit", "tracker (the section's label)"),
    ("input section", "shaft.fitting", "has", "> Fitting temperatures", "tracker (the section's label)"),
    ("input", "shaft_joint.output_shaft_fit_C", "starts", "Gearbox shaft temperature at fitting", "BL-066 acceptance (the filter's fourth gearbox input)"),
    ("result", "shaft_joint.required_Nm", "has", " = 7.647: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.hub_holding_Nm", "has", " = 9.484: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.cup_holding_Nm", "has", " = 16.90: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.hub_fit_temp_C", "has", " = 93.27: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.hub_fit_allowed_C", "has", " = 100.0: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.cup_fit_temp_C", "has", " = 280.2: ", "tracker, BL-064 notes"),
    ("result", "shaft_joint.cup_fit_allowed_C", "has", " = 300.0: ", "tracker, BL-064 notes"),
    ("verdict", "library N42 parts in both rings", "has", "needs 93.3 °C against 59.8 °C", "BL-064 notes (R-15)"),
    ("share link", "default design", "is", "2643 characters", "tracker"),
    ("count", "inputs", "is", "187", "tracker"),
    ("count", "results", "is", "1112", "tracker, 04-memory ordering item"),
    ("count", "results explained", "is", "416", "tracker, 04-memory ordering item"),
    ("count", "scope paths", "is", "193", "tracker"),
    ("count", "dashboard rows", "is", "19", "tracker"),
    ("count", "checks", "is", "32", "tracker"),
    ("count", "guides", "is", "10", "tracker"),
    ("count", "end-effect rows", "is", "9", "tracker"),
    ("count", "stored-3D rows", "is", "4", "tracker"),
    ("count", "notes", "is", "21", "tracker"),
    ("count", "notes shown", "is", "20", "tracker"),
    ("count", "start here steps in all", "is", "15", "tracker"),
    ("count", "start here steps shown", "is", "14", "tracker"),
    ("count", "assumptions", "is", "18", "tracker"),
    ("count", "input sections", "is", "19", "tracker"),
    ("count", "result chain groups", "is", "9", "tracker"),
]


def main(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            kind, key, text = line.rstrip("\r\n").split("\t", 2)
            records[(kind, key)] = text
    bad = 0
    for kind, key, how, want, where in QUOTES:
        got = records.get((kind, key))
        ok = got is not None and (
            (how == "is" and got == want)
            or (how == "starts" and got.startswith(want))
            or (how == "has" and want in got)
        )
        bad += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} [{kind}] {key}: wants {how} {want!r}"
              + ("" if ok else f"; ships {got!r}; quoted in: {where}"))
    # The stored-3D label is drawn after the joint row (owner R-15): its last stored-3D row.
    row = records.get(("row", "shaft_joint.check"), "")
    after = records.get(("stored 3D", "label drawn after"), "")
    drawn = bool(after) and row.split(". ", 1)[-1].startswith(after + ": ")
    bad += not drawn
    print(f"{'OK      ' if drawn else 'MISMATCH'} the stored-3D label is drawn after the joint row ({after!r})")
    print(f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
```

Run:
```bash
python -I .superpowers/sdd/2026-10-17-bl064-shaft-joint/check_shipped.py .superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-T9.tsv
```
Expected: every line starts `OK`, the stored-3D line included, and the last line is `0 mismatch(es)` (exit 0).

On a `MISMATCH` the shipped text wins (the G-3 review, T8, or the physics review, T7, may have reworded a draft, and
a physics ruling of spec section 11 can move a number):
1. In Steps 5-11, wherever this plan's text quotes the expected text (the line's `quoted in:` names where), write the
   shipped one instead, the sentence otherwise unchanged.
2. Change that entry's expected text in `QUOTES` to the shipped one and run the check again, until `0 mismatch(es)`.
3. Add one line per substitution to the commit body (Step 14), for example
   `Quotes the shipped guide label "<shipped>" in place of the draft "Shaft joints".`
A record that ships `None` (the record is missing) is not a wording change: STOP and report it; the owning task's
code is not in.

- [ ] **Step 4: The stale-text sweep, before the edits (red)**

Run:
```bash
STALE='dashboard.s 18 rows|every one of the 31 checks|CHECKS \(31 design checks\)|status line counts 8 guides|8 while the clamp section is paused|(of|all) 8 checks|N of 9 checks|nine guides are|eight (dashboard )?checks|the 20 A4 teaching notes|20 notes, 19 shown|There are 20, the cap|All 20 are signed off|opens 13 of them|clamp as step 14|its 13 steps|has 14, the clamp note last|14 steps in the full order, 13|392 (of|explained)|(^|[^0-9.,])1,?087([^0-9]|$)|169 paths|scope .169.|eight chains|six chains|slip heating, clamps and geometry|the temperature rows that read the stored 3D|exactly the three gearbox|keeps the bore and the keyway depth|no key in the geometry drawing|until BL-064|BL-064 adds that joint check|[Nn]ext: BL-064|BL-064 \(a shaft-to-hub joint check|BL-064: the shaft-to-hub joint|T12b; BL-064\)|brings the section back with no other code edit\.'
grep -n -o -E ".{0,30}($STALE).{0,30}" magcoupling-rs/README.md docs/ai/01-meta.yaml docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml | cut -c1-170
echo ---
CSTALE='eight dashboard checks|8 while the shaft-clamp section is paused|reading order \(slice-2 spec 3\.3, 14 steps|Start here.s 14 steps|scope holds 169 paths|392 of the 1087|2,462 characters at the time of writing|Summary lists 18 rows|not read by the engine|slip heating, clamps, geometry\)|stored-3D label after the last temperature row|[Tt]he temperature rows that read the stored 3D fields|the headline rows that move with|section.s: not drawn while it is paused|keyway in the geometry drawing|only the drawing reads it \(Python drawing\.py\)|with that section \(BL-064\)|the_key_and_the_clamp_boss_check_are_drawn_only_while_the_clamp_section_shows|the_stored_3d_rows_are_the_headline_rows'
grep -rn -o -E ".{0,30}($CSTALE).{0,30}" magcoupling-rs/src magcoupling-rs/tests --include=*.rs | cut -c1-170
```
What the patterns catch (measured at 0687034: 49 matches on 31 README and docs/ai lines from the first command, 20 code
lines from the second, every one a text BL-064 makes stale), each with the task whose docs or code step owns it:

| Text at 0687034 | Where | Owner |
|---|---|---|
| "the 20 A4 teaching notes", "20 notes, 19 shown", "There are 20, the cap", "All 20 are signed off", "opens 13 of them", "clamp as step 14", "its 13 steps", "has 14, the clamp note last", "14 steps in the full order, 13" | README ~:29, ~:504, ~:524-525, ~:678, ~:747; 03-structure :63 | T7 |
| "eight chains", "six chains", "slip heating, clamps and geometry", "169 paths", "scope (169)" | README ~:153, ~:359, ~:674; 03-structure :63, :67; 02-system :97 | T3 |
| "392 of the 1087" | README ~:294, ~:674 | T1, T2 and T3 (the totals move in each; T3's are the last) |
| "the temperature rows that read the stored 3D", "N of 9 checks", "of 8 checks", "nine guides are", "dashboard's eight checks", "dashboard's 18 rows", "status line counts 8 guides", "every one of the 31 checks", "passes all 8 checks", "4 of 8 checks"; "eight dashboard checks"; "CHECKS (31 design checks)", "8 while the clamp section is paused" | README ~:311, ~:319-323, ~:358, ~:459, ~:674, ~:677; 02-system :107; 03-structure :67 | T5 |
| "exactly the three gearbox inputs" | README ~:674 | T2 (Step 11i) |
| "keeps the bore and the keyway depth" | README ~:459, ~:674 | T2 (Step 11j, 11k) |
| "no key in the geometry drawing" | README ~:459 | T4 |
| "Next: BL-064", "T12b; BL-064)", "brings the section back with no other code edit.", "until BL-064" | README ~:46, ~:459, ~:505; 02-system :108, :486 | T9 (Steps 5 and 11) |
| BL-064's and BL-059's lines, "392 explained results of 1087" | 04-memory :7, :8, :12, :15, :20 | T9 (Step 6) |
| code: `material_library.rs:91` ("not read by the engine") | | T1 |
| code: `session.rs:579` (the share-link comment) | | T2 |
| code: `scope.rs:12`, `result_groups.rs:3`, `trace.rs:7` | | T3 (T1 and T2 move `trace.rs` too) |
| code: `geometry.rs:449-450` and `:981`, `clamps.rs:43` (the `PAUSED` doc), `tests/explain.rs:4341` (`INERT_INPUTS`' reason) | | T4 |
| code: `dashboard.rs:14`, `:206`, `:223`, `:684`, `:1506`; `guidance.rs:309`, `:1505`; `spreadsheet.rs:1196` (the eighteen-row Summary test, which Task 5 replaces) | | T5 |
| code: `notes.rs:1196`, `:1604`; `explorer.rs:2764` (`CLAMPS_PAUSED_LINE`) | | T7 |

Patterns are chosen so the new texts do not match them ("of 9 checks" is the new paused count, so only the unpaused
form "N of 9 checks" is a pattern). Expected after T1-T8 (the text is what matters, the line numbers move with the
rows T1-T3 add): exactly these first-command lines, the ones this task owns, and nothing after `---`:
```
magcoupling-rs/README.md:46:see [The panel](#the-panel)). Next: BL-064 (the shaft-to-hub joint), the
magcoupling-rs/README.md:459:0-15; slice-2 ruling 11, task T12b; BL-064): "Get rid of the clamping wit
magcoupling-rs/README.md:459:it. Setting `PAUSED` to false brings the section back with no other code edit.
magcoupling-rs/README.md:505:nd is hidden with its section until BL-064). Slice 2
docs/ai/02-system.yaml:108:026-10-15, slice-2 ruling 11, T12b; BL-064): engine::clamps::PAUSED is th
docs/ai/02-system.yaml:486:i/guidance.rs GEARBOX_GUIDE); next: BL-064, then the torque sensitivity
docs/ai/04-memory.yaml:7:7, BL-058 and BL-060..BL-065 (BL-064: the shaft-to-hub joint, the shaft-clamp section paus
docs/ai/04-memory.yaml:8:eads 'Example design: fails 4 of 8 checks' while the shaft-clamp sectio
docs/ai/04-memory.yaml:12:x check, shipped 2026-10-17): BL-064 (a shaft-to-hub joint check for the chosen method; the cl
docs/ai/04-memory.yaml:15:shrink fit or another method: BL-064 adds that joint check, then un-pauses (PAUSED false
docs/ai/04-memory.yaml:20:es). Tracing reaches only the 392 explained results of 1087 (one more sin
---
```
Any other line: it belongs to the task the table names. Apply that task's docs (or code comment) step from this
plan as the Rules above say, and note the item for the commit body (Step 14). A line that matches but is correct as
it stands after BL-064 (a false positive; none is expected): STOP and report it to the controller with the line,
rather than editing the pattern list.

- [ ] **Step 5: README (four edits in three places)**

1. The status paragraph (~:46). Replace
```text
Next: BL-064 (the shaft-to-hub joint), then the torque sensitivity card (slice 3), then M3 (live 3D fields); BL-065, BL-066 and BL-067 are open in `docs/ai/backlog.yaml`.
```
with
```text
The shaft-joint check complete (spec `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`, BL-064, owner decisions 1A-8A, R-15 and its ring ruling): one counted dashboard check of the two keyed shrink fits that hold the rotors on their shafts (the inner hub on the wheel shaft, the cup on the gearbox output shaft), with its guide and the teaching note `shaft_joint`; the shaft-clamp section stays paused (see [The panel](#the-panel)). Next: the torque sensitivity card (slice 3), then M3 (live 3D fields); BL-065 to BL-072 are open in `docs/ai/backlog.yaml`.
```
2. The shaft-clamp pause bullet (~:459), its tag. Replace
```text
(owner, 2026-10-15; slice-2 ruling 11, task T12b; BL-064)
```
with
```text
(owner, 2026-10-15; slice-2 ruling 11, task T12b; kept paused by BL-064, owner decision 6A of 2026-10-17: the rotors' keyed shrink fits are checked instead)
```
3. The same bullet's last sentence. Replace
```text
Setting `PAUSED` to false brings the section back with no other code edit.
```
with
```text
Setting `PAUSED` to false brings the section back with no other code edit; the panel would then show the clamp beside the shaft-joint check, two joint methods for one joint (BL-064 spec risk R-13, recorded in the `PAUSED` doc).
```
4. The Teaching notes bullet (~:505). Replace
```text
is hidden with its section until BL-064)
```
with
```text
is hidden with its section, which BL-064 keeps paused)
```
(The counts on the same lines, "20 notes, 19 shown" and the status counts, are T7's and T5's; Step 4 checked them.)

- [ ] **Step 6: `docs/ai/04-memory.yaml`**

1. Active issue :7 (slice 2). Replace
```text
Open from it: BL-057, BL-058 and BL-060..BL-065 (BL-064: the shaft-to-hub joint, the shaft-clamp section paused; BL-065: the end factor with magnetic steel behind the magnets); BL-059 is the gearbox check (next item)."
```
with
```text
Open from it: BL-057, BL-058, BL-060..BL-063 and BL-065 (BL-065: the end factor with magnetic steel behind the magnets); BL-059 is the gearbox check and BL-064 the shaft joints (the next two items)."
```
2. Active issue :8 (BL-059). Replace
```text
the example design reads 'Example design: fails 4 of 8 checks' while the shaft-clamp section is paused. Open from it: BL-066
```
with
```text
the example design now reads 'Example design: fails 4 of 9 checks' while the shaft-clamp section is paused (BL-064's shaft-joint check, which passes there, is the ninth). Open from it: BL-066
```
3. A new active issue after the BL-059 one. Replace
```text
  - "Payload weights Track 2: before merge
```
with
```text
  - "Magcoupler shaft joints (BL-064, branch magcoupling/bl064-shaft-joint; spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md, plan docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md; owner decisions 1A-8A, R-15 and its ring ruling): built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, gui-smoke) and then the owner's review of the diff come before the merge and the push (risk: physics). One counted dashboard check, shaft_joint.check, of the two keyed shrink fits that hold the rotors on their shafts (the inner hub on the wheel shaft, the cup on the gearbox output shaft): each fit alone holds the safety factor times cold-high at its loosest, stays within yield / 1.2 at its tightest and can be fitted within its allowed temperature, and the key alone is a backup, never added; the bonded hub's allowed fitting temperature is also capped at the inner ring's own single-ring onset less 10 °C (R-15), so the row is a stored-3D row. The example design reads 'Example design: fails 4 of 9 checks' while the shaft-clamp section is paused, and the section stays paused (owner decision 6A). Open from it: the owner's facts (open_questions), BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the joints), BL-071 (separate shaft diameters), BL-072 (the temperature verdict's cure margin reads the governing ring's onset)."
  - "Payload weights Track 2: before merge
```
4. NEXT (:12). Replace the whole item
```text
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 shipped 2026-10-16; BL-059, the gearbox check, shipped 2026-10-17): BL-064 (a shaft-to-hub joint check for the chosen method; the clamp section stays paused until it lands), then slice 3, a torque sensitivity card; then guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Other engine gaps split from slice 2 and BL-059: BL-057 (E21), BL-058 (E22), BL-060 (the clamp boss link, paused with the clamp section), BL-061..BL-063 (verify first), BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it), BL-066 (the repeated slip swing against the gearbox's nominal rating; needs the owner's datasheet value), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out)."
```
with
```text
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 shipped 2026-10-16; BL-059, the gearbox check, shipped 2026-10-17; then BL-064, the shaft joints): slice 3, a torque sensitivity card; then guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Other engine gaps split from slice 2, BL-059 and BL-064: BL-057 (E21), BL-058 (E22), BL-060 (the clamp boss link, paused with the clamp section, which BL-064 keeps paused), BL-061..BL-063 (verify first), BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it), BL-066 (the repeated slip swing against the gearbox's nominal rating; needs the owner's datasheet value), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out; the gearbox and shaft-joint checks both read cold-high), BL-068 (the hot-minimum Problem can read two equal numbers), BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the shaft joints), BL-071 (separate shaft diameters; needs the owner's sizes), BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24, to land with E21)."
```
5. A new open question after the BL-059 datasheet question. Replace
```text
  - "Decided (slice 2, T8 help batch B;
```
with
```text
  - "Open (BL-064, owner facts, 2026-10-17; none blocks the build, all in BL-064's notes): the gearbox output shaft's real diameter, tolerance and key; the wheel shaft's diameter, tolerance and material (both Ø10 for now: BL-071); the real interference bands (inner hub 13-18 µm and cup 15-22 µm are placeholders); the 4140 condition (annealed, 415 MPa, the check's conservative yield, or pre-hardened) and the UltraOx process and its temperature (the bare cup's 300 °C default and 400 °C slider top must stay below it and below 4140's tempering temperature; BL-069); the key length and the real keyway depth (the Python-locked 1.7 mm default, coupling.keyway_depth_mm, matches no DIN 6885-1 row: 3 × 3 has a hub depth t2 of 1.4 mm; the key contact input, 1.2 mm, is separate); the bondline thickness (at 0.05 mm the engine's thermal-mismatch screen reaches the AA 326 lap shear near 113 °C, at 0.10 mm near 198 °C; the bonded hub's 100 °C default rests on the 0.05 mm figure)."
  - "Decided (slice 2, T8 help batch B;
```
6. The pause item (:15), two phrases. Replace
```text
hides the slotted clamp, the shaft key and the adapter-joint screws on screen;
```
with
```text
hides the slotted clamp, its shaft key inputs and the adapter-joint screws on screen (since BL-064 the drawn key is the shaft joints' own, shaft_joint.key_width_mm);
```
and replace
```text
The likely replacement is a permanent shrink fit or another method: BL-064 adds that joint check, then un-pauses (PAUSED false, no other edit) or replaces the section. BL-060 is paused with it."
```
with
```text
The replacement is a keyed shrink fit on both rotors (owner, 2026-10-17): BL-064 checks it and keeps the section paused and unchanged (owner decision 6A); un-pausing (PAUSED false, no other edit) would show the clamp beside the shrink-fit check, two joint methods for one joint (BL-064 spec risk R-13). BL-060 is paused with it."
```
7. The ordering item (:20, now :22 after the two inserts). Replace
```text
Tracing reaches only the 392 explained results of 1087 (one more since BL-059's gearbox check).
```
with
```text
Tracing reaches only the 416 explained results of 1112 (BL-064 added 24 explained results, the shaft joints' 23 numbers and the inner ring's single-ring onset, and one text check).
```

- [ ] **Step 7: `docs/ai/05-update-tracker.md`: the new top entry**

Replace
```text
## 2026-10-16 — Magcoupler gearbox check: cold-high at the gearbox input against its rating (BL-059, branch magcoupling/bl059-gearbox-check)
```
with
```markdown
## 2026-10-17 — Magcoupler shaft joints: the keyed shrink fits that hold the rotors on their shafts (BL-064, branch magcoupling/bl064-shaft-joint)
- Why: nothing checked how the two rotors are held on their shafts once the shaft-clamp section was paused (2026-10-15). The owner's ruling of 2026-10-17: "Hub is shrink fit with key only" (no clamp screws, no adapter joint). The drive line is motor -> GAM 5:1 gearbox -> coupling -> wheel, and both rotor-to-shaft joints carry the coupling's torque: the inner hub (its magnets bonded on its ten flats) on the wheel shaft, and the cup (its integral boss) on the gearbox output shaft. Cold-high (`metal.torque_cold_high_Nm`, 3.823 N·m at the defaults) can reach each joint before the coupling slips; it is not a bound on them (backlash, resonance crossings and re-lock ring-down can exceed it on the gearbox side, and with the operating temperature below the minimum the operating pull-out is above it, BL-067). Spec `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md` (approved by the owner 2026-10-18 at 0687034, after six review rounds); plan `docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md` (T1-T10); research (gitignored) `.superpowers/sdd/2026-10-17-bl064-shaft-joint/`.
- Owner decisions (2026-10-17): **1A** both joints in one counted dashboard row, one pure joint function evaluated twice, the verdict naming the failing joint; **2A** the fit alone must hold the safety factor times cold-high at its loosest and the part must stay within yield ÷ 1.2 (`YIELD_FACTOR`) at its tightest; the key alone is a backup checked at the same torque; fit and key capacities are never added; **3A** minimum and maximum measured diametral interference as inputs, one band per joint (placeholders until the real fits are set: inner hub 13-18 µm, cup 15-22 µm); **4A** DIN 6885-1 3 × 3 key inputs, width 3.0 mm and contact height 1.2 mm, shared by both joints (the paused clamp's key inputs stay with the paused section); **5A** a safety factor input, 2.0, on slip and on the key backup; **6A** the shaft-clamp section stays paused and unchanged; **7A** "cannot be fitted" is a counted cause per joint: the part's needed temperature, 20 °C + (maximum interference + joining clearance − the shaft's shrink at its fitting temperature) ÷ (α_part · d), at or below its allowed temperature. Process: the bare cup is shrunk onto the gearbox shaft first and its magnets bonded after; the inner hub is bonded first and shrunk onto the cooled wheel shaft (wheel shaft −196 °C, liquid nitrogen; gearbox shaft 20 °C; the shrink below 20 °C from NIST's Fe-2.25Ni expansion data, not a constant coefficient); allowed fitting temperatures 100 °C for the bonded hub, never above the adhesive's design limit, and 300 °C for the bare cup. **8A** a new reviewed note `shaft_joint`, the note cap 20 -> 21. Material 4140 with UltraOx, the bore masked (the library's 4140 yield, 415 MPa, conservative while the condition is open); both shafts Ø10 (`coupling.bore_mm`). **R-15** (after spec review 3): the bonded hub's allowed fitting temperature is also capped at the magnets' single-ring onset less `CURE_MARGIN_MIN_C` (10 °C), the temperature verdict's cure margin, so lower grades read "cannot be fitted" and the row is a stored-3D row; its **ring ruling** (after spec review 5): the onset is the inner ring's own, a new Rust-only result `temperature.demag.inner_onset_single_ring_C`, never the governing ring's. Release: the owner's review of the diff before the merge (risk: physics).
- Engine (T1, T2): `temperature.demag.inner_onset_single_ring_C` (the inner ring's own single-ring onset beside `inner_magnet_limit_C`, equal to the governing onset wherever the inner ring governs) with its record in `records/demagnetization.rs` and its path in the `demagnetization` scope chain; `PartProperties::hub()` in `material_library.rs` (the part material in effect and its yield; the warnings' hub expansion in `api.rs` reads it too; `temperature.rs` keeps its own E18 choice of the hub's expansion for the bond-mismatch screen, `temperature.rs:1069-1073`, toggled with the correction: the same rule with every correction on, so it is not folded into the helper, an observation with no change). `engine::shaft_joint` (Rust-only, no cell, like `gearbox`): `ShaftJointInputs` (15, the first top-level Rust-only input group) and `ShaftJointResults` (24), computed in `api.rs` after `gearbox::compute`: plane-stress Lamé fits, the holding with the key's arc removed (floored at zero), the bore's von Mises stress, the key flank with the contact height capped at the keyway depth over the bearing length L − b (+∞ and a fail without bearing area; no key term without a keyway); the loosest and tightest ends (the minimum magnet temperature, and the adhesive's design limit with the rotor over its shaft); `hub_fit_allowed` (NaN when any of its three values is NaN; ties name the earlier limit); the cause order (not a number; a safety factor not above zero, plan deviation PD-2; the fit slips; the bore stress; cannot be fitted; the key; then, per joint, no wall around the bore, a part diameter typed at or below zero included, plan deviation PD-1, or a reversed band), only a determinate joint judged; badge numbers never equal; the verdict constants `SHAFT_JOINTS_PASS` ("Both shaft joints hold and can be fitted"), `SHAFT_JOINT_UNKNOWN`, `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`, `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT` and `KEY_HAS_NO_BEARING_AREA`. `tests/shaft_joint.rs` end to end; `input_schema.json` re-blessed; the default share link grows from 2,462 to 2,643 characters (guard below 2,700). No Python-ported input, result, label, help, default, range or cell changed, so gate 12's parity is unaffected. 187 inputs; 1,112 results.
- Explain layer (T3): `records/shaft_joint.rs` (23 records; the drift guard holds each to the engine at every probe design and at a design with no wall around the bore); the `shaft_joint` scope chain after `geometry`: the scope holds 193 paths and the results table 416 explained results of 1,112; `CHAIN_LABELS` 9 (the shaft joints between Geometry and Mass); four `ASSUMPTIONS` rows (`fit_friction`, `fit_smoothing`, `rotor_over_shaft`, `joining_clearance`; 18 rows), whose Rust-only sources the widened rule of `tests/assumptions.rs` accepts ("BL-064 shaft joint spec").
- Panel, drawing and exports (T2, T4, T5): the Shaft and hub joint group gains the two key rows and the sections "Shrink fit" and "Fitting temperatures", before the paused clamp's (`SECTION_LABELS` 19; T2); the key is drawn in the bore in both states from `shaft_joint.key_width_mm`, and none at a keyway depth of 0 or less (T4; plan deviation PD-3); CSV, JSON and the XLSX Inputs and Results sheets carry every new input and result (T4), and the XLSX Summary lists 19 dashboard lines (T5).
- Dashboard, guide and fixes (T5, T6): `DASHBOARD` 19 (the row at index 15, after the hidden clamp row and before the gearbox row; Magnets fit is still drawn last), `CHECKS` 32, `END_EFFECT_ROWS` 9 (it reads cold-high, so it greys, uncounted, while the end-effect model is out of range), `STORED_3D_ROWS` 4 (R-15: its hover carries the "Fixed 3D fields from the example design" label, the drawn label now follows it, and `STORED_3D_NOTE` names the bonded hub's fitting limit). `SHAFT_JOINT_GUIDE` (label "Shaft joints") joins `CHECK_GUIDES` (10) between the clamp's and the gearbox's: its Problem names the joint and the cause with live numbers, the window of measured interference that would work, quoted inward (or says in numbers that there is none), and the source of the hub's allowed temperature (your limit, the adhesive's design limit, or the magnets' single-ring onset less 10 °C); `Fixes::PerTerm` per cause and joint, with three fixes held back by rule (a band step that would reverse the band, either way; the key's contact height where the key has no bearing length). The example design reads "Example design: fails 4 of 9 checks" while the shaft-clamp section is paused ("Example design: fails 4 of 10 checks" with it shown); `passing_design()` passes all 9; `only_the_joint_fails()` (the hub minimum at 8 µm) fails 1 of 9. At the defaults the inner hub holds 9.484 N·m against the 7.647 N·m it must hold and needs 93.27 °C against 100.0 °C allowed; the cup holds 16.90 N·m and needs 280.2 °C against 300.0 °C. Auto unchanged.
- Note and reviews (T7, T8): the new note `shaft_joint` ("Shaft joints: a keyed shrink fit"; its Start here step just before `clamp_preload`: 15 steps, 14 while paused; 21 notes, 20 shown), physics-reviewed (`docs/analyses/2026-10-14-magcoupler-notes-review.md#shaft_joint`); every new on-screen text G-3 reviewed (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## BL-064 shaft joints`). `CLAMPS_PAUSED_LINE` no longer says the clamp returns with BL-064; no new sign-off sheet (the owner reviews the diff).
- Docs and backlog (T9): the README, docs/ai 02-05; BL-064 (the acceptance amended to the owner's ruling; the owner decisions, the open facts and the default verdict in its notes), BL-060 (paused with the clamp section, which BL-064 keeps paused), BL-066 (its filter tests now move from four gearbox inputs to five), BL-067 (the shrink-fit check reads cold-high too; one fix serves both checks); filed BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the joints under the reversing slip swing), BL-071 (separate shaft diameters) and BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24 of 2026-10-01, to land with E21, BL-057). Not covered by the check (spec section 10): fretting and fatigue (BL-070), the keyway corner's stress concentration, the gearbox's own shaft, bearings, seal and heat soak, the bond strain from fitting a bonded hub, the rotor chill from a cryogenic shaft, frost on a cold shaft, and the key's shear.

## 2026-10-16 — Magcoupler gearbox check: cold-high at the gearbox input against its rating (BL-059, branch magcoupling/bl059-gearbox-check)
```

- [ ] **Step 8: `docs/ai/backlog.yaml`: BL-064**

Replace BL-064's acceptance, priority, status and notes lines (title, evidence and the rest stay: like every other
item, the title states the problem the item fixed). Old text (unique):
```text
  acceptance: "owner picks the joint method; red test first; a dashboard check of the joint's holding torque against cold-high times a safety factor, with its slice-1 guide (Problem, Why, Fix) and a reviewed note; the optional adapter's bolted joint (its slip torque clamps.joint_torque_Nm and its margin clamps.joint_sf, which today divides by the clamp's design torque, cold-high times the clamp's safety factor; hidden with the section) is checked again against cold-high times a safety factor or given its own check; the keyway depth (coupling.keyway_depth_mm, 1.7 mm by default, which still thins the hub wall and enters Auto's validity rule) is set for the chosen joint (0 mm for a keyless shrink fit); then set engine::clamps::PAUSED to false (the section returns with no other edit) or replace the clamp section; physics review; GATE PASS"
  priority: 2
  status: open
  notes: "Opened by slice 2 T12b (ruling 11 4A). While paused the clamp math, Python parity, design files, share links and the exports keep every clamp input and result; the screen shows none and the clamp counts toward no verdict. BL-060 (the clamp's boss against the drawn boss) is paused with it. Un-pausing (engine::clamps::PAUSED = false) needs no other code edit, but turns the slice-2 note gates on for clamp_preload (caps, plain words, its review record) and the docs-count test: the note's rewrite and per-note physics review, dropped from slice-2 T16 (A-28), and the docs come first."
```
New text:
```text
  acceptance: "owner picks the joint method (done 2026-10-17: a keyed shrink fit on both rotors, 'Hub is shrink fit with key only', no clamp screws and no adapter joint; this acceptance amended to that ruling, spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md section 2); red test first; one counted dashboard check of both joints, the inner hub on the wheel shaft and the cup on the gearbox output shaft: each fit alone holds the safety factor times cold-high at its loosest, stays within its stress limit at its tightest and can be fitted within its allowed temperature, and the key alone is checked as a backup at the same torque, never added to the fit; its slice-1 guide (Problem, Why, Fix) and a new reviewed note (shaft_joint); the shaft-clamp section stays paused and unchanged (engine::clamps::PAUSED stays true, owner decision 6A) and the optional adapter's bolted joint is not checked, the owner's design having none; the keyway depth (coupling.keyway_depth_mm, 1.7 mm by default) stays the input it is, its real value recorded as an open fact; physics and G-3 reviews; the owner's review of the diff; GATE PASS with gate 12 running"
  priority: 2
  status: in_progress
  notes: "Opened by slice 2 T12b (ruling 11 4A). While paused the clamp math, Python parity, design files, share links and the exports keep every clamp input and result; the screen shows none and the clamp counts toward no verdict. BL-060 (the clamp's boss against the drawn boss) is paused with it. Un-pausing (engine::clamps::PAUSED = false) needs no other code edit, but turns the slice-2 note gates on for clamp_preload (caps, plain words, its review record) and the docs-count test: the note's rewrite and per-note physics review, dropped from slice-2 T16 (A-28), and the docs come first; since BL-064 it would also show the clamp beside the shrink-fit check, two joint methods for one joint (spec risk R-13; the PAUSED doc says so). Owner ruling 2026-10-17: 'Hub is shrink fit with key only'. The acceptance was amended to it: it first asked for the chosen method's check, the optional adapter's bolted joint checked again, the keyway depth set for the chosen joint (0 mm for a keyless fit) and the clamp section un-paused or replaced. Owner decisions (2026-10-17; spec section 2, approved 2026-10-18 at 0687034; .superpowers/sdd/2026-10-17-bl064-shaft-joint/owner-decisions.md, gitignored): 1A both joints in one counted row whose verdict names the failing joint; 2A the fit alone holds the safety factor times cold-high at its loosest, the part stays within yield / 1.2 at its tightest, the key alone is a backup at the same torque, never added; 3A minimum and maximum measured diametral interference per joint (placeholders until the real fits are set: inner hub 13-18 µm, cup 15-22 µm); 4A DIN 6885-1 3 x 3 key inputs (width 3.0 mm, contact height 1.2 mm, shared); 5A safety factor 2.0; 6A the shaft-clamp section stays paused and unchanged; 7A 'cannot be fitted' is a counted cause per joint; the fitting order: the bare cup is shrunk onto the gearbox shaft first and its magnets bonded after, the inner hub is bonded first and shrunk onto the cooled wheel shaft (wheel shaft -196 °C, liquid nitrogen; gearbox shaft 20 °C, never immersed, its slider's bottom -20 °C for cold spray); allowed fitting temperatures 100 °C for the bonded hub, never above the adhesive's design limit, and 300 °C for the bare cup; 8A a new reviewed note shaft_joint, the note cap 20 -> 21; material 4140 with UltraOx, the bore masked (the fit uses the bare bore; the library's 4140 yield, 415 MPa, conservative while the condition is open); both shafts Ø10 through coupling.bore_mm. R-15 (owner, after spec review 3): the bonded hub's allowed fitting temperature is the lowest of the input, the adhesive's design limit and the magnets' single-ring onset less CURE_MARGIN_MIN_C (10 °C), so lower grades read 'cannot be fitted' (library N42 parts in both rings at the defaults: needs 93.3 °C against 59.8 °C) and the row is a stored-3D row; its ring ruling (owner, after spec review 5): the onset is the inner ring's own, the new Rust-only result temperature.demag.inner_onset_single_ring_C, never the governing ring's, since the hub's fitting heat reaches only the inner magnets (the cure margin's side of that gap is BL-072). Open facts (none blocks the build; also in docs/ai/04-memory.yaml): the gearbox output shaft's real diameter, tolerance and key; the wheel shaft's diameter, tolerance and material; the real interference bands; the 4140 condition and the UltraOx process and its temperature; the key length and the real keyway depth (the 1.7 mm default matches no DIN 6885-1 row: 3 x 3 has a hub depth t2 of 1.4 mm); the bondline thickness (0.05 or 0.10 mm moves the bond screen's temperature from about 113 °C to about 198 °C). At the defaults the check reads 'Both shaft joints hold and can be fitted' (the inner hub holds 9.484 N m against the 7.647 N m it must hold and needs 93.27 °C against 100.0 °C allowed; the cup holds 16.90 N m and needs 280.2 °C against 300.0 °C), and the example design reads 'Example design: fails 4 of 9 checks' while the shaft-clamp section is paused ('Example design: fails 4 of 10 checks' with it shown). Built on magcoupling/bl064-shaft-joint by docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md (T1-T10), spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md: the inner ring's own single-ring onset and the hub-material helper (T1); engine::shaft_joint, Rust-only, 15 inputs and 24 results (T2); its 23 equation records, the shaft_joint scope chain and four assumptions (T3); the inputs panel's sections (T2), the key drawn in both states and the exports (T4); the dashboard row, CHECKS, the guide and its per-term fixes (T5, T6); the note shaft_joint, physics-reviewed (docs/analyses/2026-10-14-magcoupler-notes-review.md#shaft_joint, T7); the G-3 review of every new text (docs/analyses/2026-10-14-magcoupler-help-review.md, BL-064 shaft joints, T8); the docs and backlog (T9). Split out: BL-069 (the UltraOx finish against the nickel plating model), BL-070 (fretting and fatigue of the joints under the reversing slip swing), BL-071 (separate shaft diameters), BL-072 (the temperature verdict's cure margin reads the governing ring's onset). AGENTS.md's fbd-math-reviewer pass is linkage-specific (its agent reviews 4-bar FBDs); BL-064's physics passes are the T7 note review, the T8 G-3 review and the T10 whole-branch review on the session model, as BL-059's were. Release (spec section 9) is T10: whole-branch review, GATE PASS with gate 12 running, web build and gui-smoke, then the owner's review of the diff; status stays in_progress until the owner's OK and the push."
```

- [ ] **Step 9: `docs/ai/backlog.yaml`: BL-060, BL-066 and BL-067**

1. BL-060's notes. Replace
```text
was dropped when the clamp section was paused (ruling 11, A-28); it comes back with BL-064."
```
with
```text
was dropped when the clamp section was paused (ruling 11, A-28); it would come back only with the clamp section, which BL-064 keeps paused (owner decision 6A, 2026-10-17: the rotors' keyed shrink fits are checked instead)."
```
2. BL-066's acceptance (spec section 8: the joint adds a "gearbox"-labelled input). First confirm the count from the
   code:
```bash
sed -n '/let gearbox = \[/,/\];/p' magcoupling-rs/src/gui/inputs.rs | grep -c '^ *"'
```
   Expected: `4` (T2 moved the filter test; Step 3 checked the label `Gearbox shaft temperature at fitting`).
   Then replace
```text
the 'gearbox' filter tests that find exactly three gearbox inputs, one test in gui/inputs.rs and one in gui/panel.rs (the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design, with two '3 of N inputs' assertions), move to four)
```
with
```text
the 'gearbox' filter tests that find exactly four gearbox inputs since BL-064 added the gearbox shaft's fitting temperature (shaft_joint.output_shaft_fit_C), one test in gui/inputs.rs and one in gui/panel.rs (the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design, with two '4 of N inputs' assertions), move to five)
```
3. BL-067's acceptance, option (b). Replace
```text
(b) the gearbox check compares the larger of cold-high and the operating pull-out with the allowance, at the gearbox input; then a red test at op -40 C, min 20 C, variation 0.1;
```
with
```text
(b) the gearbox check and the shaft-joint check (BL-064's shaft_joint.check, whose required torque is the safety factor times cold-high) compare the larger of cold-high and the operating pull-out with the allowance, the gearbox check at the gearbox input; one choice serves both checks; then a red test at op -40 C, min 20 C, variation 0.1 for each check (tests/shaft_joint.rs pins the joint's corner on purpose and names this item);
```
4. BL-067's notes. Replace
```text
If BL-064 shows the clamp section again before this item lands, the clamp Problem needs the same fix (the acceptance)."
```
with
```text
BL-064 keeps the clamp section paused (owner decision 6A, 2026-10-17), so the clamp Problem needs the same fix only if the section is shown again before this item lands (the acceptance). BL-064 (2026-10-17) adds a second check that reads cold-high: shaft_joint.required_Nm = shaft_joint.safety_factor × metal.torque_cold_high_Nm (magcoupling-rs/src/engine/shaft_joint.rs), so in this corner (op -40 C, min 20 C, variation 0.1) both joints are checked at 2 × 3.1823 = 6.3647 N m while the operating pull-out with its allowance is 3.657 N m (3.3246 N m × 1.1); tests/shaft_joint.rs pins that corner and names this item, so this item's fix moves the pin on purpose. BL-064's texts claim no order of the torques (its spec review 2, MF-1): the required torque's help defines cold-high without ranking it, and the_shaft_joint_texts_never_order_the_torques keeps 'strongest', 'most expected' and 'most torque expected' out of every shaft-joint text at this corner. One fix of which pull-out the checks compare serves both."
```

- [ ] **Step 10: `docs/ai/backlog.yaml`: new BL-069, BL-070, BL-071 and BL-072**

The priorities below (BL-069 3; BL-070, BL-071 and BL-072 2) are the drafted ones; the owner confirms or re-ranks them
in the plan review (ruling C-9). Write the owner's, and set Step 12's priority assertion to match.

BL-070's evidence ends with one of two sentences, chosen from Step 2's watch-out lines: if a watch-out at the
default design contains "fretting" (case-insensitive; `grep -c -i fretting` on the printed lines), use sentence (F);
otherwise use sentence (N).
- (F): `The shaft_joint note's watch-out says nothing here checks fretting under the reversing swing.`
- (N): `The shaft_joint note does not name fretting; spec section 10 lists it as not checked.`

Insert the four items between BL-068 and `- id: BL-055`. Old text (unique):
```text
any change to on-screen text needs its own G-3 round."
- id: BL-055
```
New text (with (F) or (N) in place of `<SENTENCE F OR N>`, the one edit this step makes to the text below):
```text
any change to on-screen text needs its own G-3 round."
- id: BL-069
  title: "The plating model assumes electroless nickel on the steel back iron, while the owner finishes the 4140 inner hub and cup with UltraOx and masks the bore"
  dimension: physics
  risk: physics
  evidence: "Owner (BL-064 decisions Material and Nickel, 2026-10-17; docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md section 2): the inner hub and the cup are 4140 with an UltraOx finish, the bore masked from it. The engine's plating model is electroless nickel: materials.nickel.thickness_mm (0.015 mm per surface, 'High-phosphorus EN (10–12 % P): non-magnetic as plated', Materials!C26, magcoupling-rs/src/engine/materials.rs:55) sets the machining allowances materials.hub_flats_under_mm, cup_pockets_over_mm, bores_over_dia_mm (2 × 0.015 = 0.030 mm on the bore's diameter, Materials!C29, materials.rs:263) and ods_under_dia_mm; the 4140 library record carries needs_plating: true (magcoupling-rs/src/engine/material_library.rs:144) and the uncoated low-alloy steel warning reads 'Uncoated low-alloy steel: it corrodes, so it needs plating; the example design uses electroless nickel.' (magcoupling-rs/src/engine/warnings.rs:85). BL-064's shaft-joint check reads no plating input: with the bore masked the fit uses the bare bore. The bore allowance the Materials sheet reports still assumes a nickel layer in the bore."
  acceptance: "owner gives the UltraOx process (its layer thickness per surface, whether it builds or converts the surface, its process temperature, which surfaces are masked); then decide with the owner how the calculator models the finish (a finish choice beside electroless nickel, or a thickness per surface with the bore's own); red test first: with UltraOx and a masked bore the bore allowance is zero and the warning names the finish in use; Python parity and the workbook deviation recorded for any ported cell that changes (Materials!C26, C29); the a5.uncoated_low_alloy_steel note and the plating thickness help updated in the same commit; G-3 and physics reviews; GATE PASS with gate 12 running"
  priority: 3
  status: open
  notes: "Split from BL-064 (spec section 8; owner decisions Material and Nickel, 2026-10-17). The UltraOx process temperature also bounds the bare cup's fitting temperature (BL-064's 300 °C default and its 400 °C slider top must stay below it): an open fact in BL-064's notes."
- id: BL-070
  title: "Nothing checks fretting or fatigue of the two shaft joints under the reversing slip swing (about 6.7e8 torque reversals per life at the defaults)"
  dimension: physics
  risk: physics
  evidence: "BL-064 spec sections 1 and 10 (docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md): while the coupling slips each rotor-to-shaft joint carries a fully reversing swing of about cold-high's amplitude, more on the gearbox side with backlash, at metal.slip_freq_Hz (166.7 Hz at the defaults): 16.7 full cycles, about 33 reversals, per 0.1 s slip event (metal.slip_event_s); metal.magnetic_cycles (slip frequency × event time × life events, magcoupling-rs/src/engine/metal_design.rs:479) is 3.33e8 cycles per life at the defaults, about 6.7e8 reversals (spec review 2, MF-2). BL-064's shaft_joint.check (magcoupling-rs/src/engine/shaft_joint.rs) is a static check: each fit alone holds the safety factor (2.0) times cold-high at its loosest, and the key alone is a backup at the same torque. A shrink fit micro-slips at the part's ends below its slip torque, which wears the contact (fretting) and starts fatigue cracks at the hub ends and the keyway corner (whose stress concentration, Kt about 2-3 from memory, is not modelled either); a key flank under a reversing load frets and fatigues too. <SENTENCE F OR N>"
  acceptance: "verify first (sourced: a fretting-fatigue criterion for keyed shrink fits under fully reversed torsion, such as DIN 743 with DIN 7190-1:2017's guidance on reversing loads, skeptic-checked; the slip duty to check, slip events per hour and per life, from the owner, shared with BL-066); then red test first: a counted check or caution of the joints' reversing load against the chosen criterion at the defaults and the five probe designs, with its slice-1 guide; the shaft_joint note's watch-out updated in the same commit; G-3 and physics reviews; the owner's review of the diff; GATE PASS with gate 12 running"
  priority: 2
  status: open
  notes: "Split from BL-064 (spec sections 8 and 10; spec review 2, MF-2), beside BL-066 (the same slip swing against the gearbox's nominal rating: one slip-duty input can serve both). BL-064's safety factor help says the torque reverses many times in each slip; its texts never call the fit permanent or free of backlash (its trap 18)."
- id: BL-071
  title: "One bore input serves both shaft joints: the gearbox output shaft and the wheel shaft are both taken as Ø10 until their real diameters are known"
  dimension: physics
  risk: physics
  evidence: "Owner (BL-064 decision Shafts, 2026-10-17): 'Both shafts Ø10 for now'; the real gearbox output and wheel shaft sizes come later. coupling.bore_mm (Calculator!C39, 10 mm) is the one bore of both joints in BL-064's shaft_joint.check (magcoupling-rs/src/engine/shaft_joint.rs: each joint's contact pressure, holding, bore stress, key pressure, shrink and fitting temperature read it), and the masses read it too: the web's (magcoupling-rs/src/engine/model.rs:1256), the hub's (:1263) and the boss's (:1271), as does the hub wall (:931). The key inputs (shaft_joint.key_width_mm, shaft_joint.key_contact_mm: DIN 6885-1 3 × 3 for shafts over 8 up to 10 mm) and the joining clearance (shaft_joint.joining_clearance_um, 10 µm, an assumption resting on a Ø10 bore) are one value each for both joints."
  acceptance: "owner gives each shaft's diameter and tolerance (and the wheel shaft's material); then red test first: a second bore input for the cup's joint on the gearbox output shaft (Rust-only; input_schema.json re-blessed; design files and share links without it load at its default), each joint's terms reading its own bore, the boss and web masses the cup's bore, and each joint's key row and joining clearance following its own bore; the share-link length guard re-measured; Python parity unchanged (coupling.bore_mm keeps its cell); the shaft_joint note and every help that says both shafts are one size updated in the same commit; G-3 and physics reviews; GATE PASS with gate 12 running"
  priority: 2
  status: open
  notes: "Split from BL-064 (spec section 8; owner decision Shafts, 2026-10-17). Until the owner gives the shaft sizes the check uses Ø10 for both; BL-064's notes list the shafts' diameters and tolerances among the open facts."
- id: BL-072
  title: "The temperature verdict's cure margin reads the governing ring's single-ring onset, not each ring's own: a cure can pass while it reaches one ring's onset"
  dimension: physics
  risk: physics
  evidence: "magcoupling-rs/src/engine/temperature.rs at fb79095: the summary's cure margin is on_cu − sel.cure_C (:1382), checked against CURE_MARGIN_MIN_C (10 °C, :746) at :1387, where on_cu is the governing ring's single-ring onset (hot.onsets, :978; temperature.demag.onset_single_ring_C, :992), the ring with the lower magnet limit, the inner on a tie (:969-972). Each ring's own cure heats that ring's bond on its own carrier (BL-064 owner decision Process: the bare cup's magnets are bonded on the fitted cup, the inner hub's before fitting). Where the two rings differ the onset read can be above a ring's own, so the margin passes a cure that reaches that ring's onset: over the 289 grade pairs on manual dimensions at passing_design(), the read is above the inner ring's own in 10 and above the lower of the two rings' own in 20, by up to 79.05 °C (inner N42H with outer SmCo 2:17-30, grade id SmCo_2_17_30: with 2214 Hi-Temp, cured at 121 °C, the margin passes at 70.88 °C while against the inner ring's own onset it is −8.17 °C) (BL-064 spec section 11, R-15; its scratch crate, out.txt 13.7). The fix was approved by the owner on 2026-10-01 as correction E24 (docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md, section 2.8 'The fix' and decision 2A: C24 = MIN(C59_inner, C59_outer) − cure, gated on E20, landed in the same change as E21), and has not landed: E21 is BL-057, open."
  acceptance: "red test first: at inner N42H with outer SmCo_2_17_30 on manual dimensions at passing_design() with 2214 Hi-Temp, the cure margin reads against the lower of the two rings' own single-ring onsets (BL-064's temperature.demag.inner_onset_single_ring_C is the inner half; an outer twin is added) and the temperature verdict fails there (−8.17 °C below the 10 °C margin); every design where the inner ring governs, the defaults included, unchanged bit for bit; the owner-approved correction E24 registered with its audit row and probes, landed with or after E21 (BL-057) as decision 2A of 2026-10-01 says; the demagnetization note's cure-margin sentences re-read in the same commit; physics review; GATE PASS with gate 12 running"
  priority: 2
  status: open
  notes: "Split from BL-064 (spec section 8; the owner's ring ruling of 2026-10-17 keeps it: BL-064's fitting cap reads the inner ring's own onset already, so this item is the cure margin's side alone). It is the owner-approved E24 of the Addendum A-4 verification (decision 2A, 2026-10-01, recorded in docs/ai/04-memory.yaml's A-4 item), which was to land with E21 (BL-057); BL-064's spec did not cite it."
- id: BL-055
```

- [ ] **Step 11: `docs/ai/02-system.yaml`: the pause tag, the status and a known limitation**

(The shaft joint invariant is Task 2's, with Task 5's greying and stored-3D sentence: ruling C-2, A-5. This step adds no
second one.)

1. The pause line (~:108). Replace
```text
(owner 2026-10-15, slice-2 ruling 11, T12b; BL-064)
```
with
```text
(owner 2026-10-15, slice-2 ruling 11, T12b; kept paused by BL-064, owner decision 6A of 2026-10-17)
```
2. `current_statuses.magcoupling` (~:486). Replace
```text
; next: BL-064, then the torque sensitivity card (slice 3), then M3 (live 3D fields)"
```
with
```text
; the shaft-joint check (BL-064, owner decisions 1A-8A, R-15 and its ring ruling) built on branch magcoupling/bl064-shaft-joint: engine::shaft_joint (Rust-only) and one counted dashboard check of the two keyed shrink fits that hold the rotors on their shafts, with its guide (gui/guidance.rs SHAFT_JOINT_GUIDE) and the reviewed note shaft_joint; the shaft-clamp section stays paused; next: the torque sensitivity card (slice 3), then M3 (live 3D fields)"
```
3. A `known_limitations` entry after the gearbox check's. Replace
```text
is open (owner decision 2A, 04-memory)."
```
with
```text
is open (owner decision 2A, 04-memory)."
  - "The magcoupler's shaft-joint check (BL-064, shaft_joint.check) is static: each fit at its loosest against the safety factor times cold-high, its bore stress at its tightest, its fitting temperature, and the key alone as a backup. Fretting and fatigue under the reversing slip swing are not checked (BL-070), nor the keyway corner's stress concentration, the gearbox's own shaft, bearings, seal and heat soak, the bond strain from fitting a bonded hub, the rotor chill from a cryogenic shaft or the key's shear. Both shafts are Ø10 through one bore input until the owner gives their sizes (BL-071); the interference bands (inner hub 13-18 µm, cup 15-22 µm) are placeholders until the real fits are set; the plating model is electroless nickel while the hub and cup are UltraOx with the bore masked (BL-069). The open facts are in BL-064's notes and 04-memory."
```
(`02-system.yaml` does not parse as YAML at 0687034, near line 14, before this branch: leave that alone; Step 12
parses each line this task writes on its own.)

- [ ] **Step 12: Check the YAML and the item shapes**

Run:
```bash
python - <<'EOF'
import yaml
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
ids = [x['id'] for x in b]
assert ids[:18] == ['BL-0%d' % n for n in range(56, 73)] + ['BL-055'], ids[:19]
keys = ('id', 'title', 'dimension', 'risk', 'evidence', 'acceptance', 'priority', 'status', 'notes')
assert all(tuple(x) == keys for x in b), [x['id'] for x in b if tuple(x) != keys]
item = {x['id']: x for x in b}
assert item['BL-064']['status'] == 'in_progress'
assert '(0 mm for a keyless shrink fit)' not in item['BL-064']['acceptance']
assert 'Hub is shrink fit with key only' in item['BL-064']['notes'] and 'ring ruling' in item['BL-064']['notes']
assert 'comes back with BL-064' not in item['BL-060']['notes']
assert 'move to five' in item['BL-066']['acceptance']
assert 'shaft_joint.required_Nm' in item['BL-067']['notes'] and 'one choice serves both' in item['BL-067']['acceptance']
for n, p in (('BL-069', 3), ('BL-070', 2), ('BL-071', 2), ('BL-072', 2)):
    assert (item[n]['dimension'], item[n]['risk'], item[n]['status'], item[n]['priority']) == ('physics', 'physics', 'open', p), n
assert '<SENTENCE' not in item['BL-070']['evidence']
assert 'E24' in item['BL-072']['acceptance'] and 'BL-057' in item['BL-072']['acceptance']
m = yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8'))
assert any(i.startswith('Magcoupler shaft joints (BL-064') for i in m['active_issues'])
assert any(q.startswith('Open (BL-064, owner facts') for q in m['open_questions'])
assert not any('BL-064 adds that joint check' in q for q in m['open_questions'])
yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8'))
lines = open('docs/ai/02-system.yaml', encoding='utf-8').read().splitlines()
for marker in ("The magcoupler's shaft joint check (BL-064, shaft_joint.check)", "The magcoupler's shaft-joint check (BL-064",
               'built on branch magcoupling/bl064-shaft-joint', 'kept paused by BL-064, owner decision 6A'):
    hit = [line for line in lines if marker in line]
    assert len(hit) == 1, (marker, len(hit))
    yaml.safe_load('k:\n' + hit[0])
print(len(b), len(m['active_issues']), len(m['open_questions']))
EOF
```
Expected: `72 5 66` (68 backlog items at 0687034 plus BL-069 to BL-072; 4 active issues plus BL-064's; 65 open
questions plus the owner-facts question). An `AssertionError` names the broken item: fix the edit, run again.

- [ ] **Step 13: The sweep again (green), the new facts, the docs tests and the suites**

Run Step 4's two commands again. Expected: both print nothing but the `---` line.

Then the new facts (each written by the task named):
```bash
grep -c 'src/engine/shaft_joint.rs' magcoupling-rs/README.md
grep -c 'records/shaft_joint.rs' magcoupling-rs/README.md
grep -c 'geometry, shaft_joint;' docs/ai/03-structure.yaml
grep -c 'tests/shaft_joint.rs' magcoupling-rs/README.md
grep -c 'nine chains' magcoupling-rs/README.md
grep -c -E '416 (of|explained)' magcoupling-rs/README.md
grep -c 'inner_onset_single_ring_C' magcoupling-rs/README.md docs/ai/03-structure.yaml docs/ai/02-system.yaml
grep -o 'addendum_only: \[[^]]*\]' docs/ai/03-structure.yaml
grep -c '193 paths' docs/ai/03-structure.yaml
grep -c '(193)' docs/ai/02-system.yaml
grep -c 'BL-069' docs/ai/backlog.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/02-system.yaml
```
Expected: at least `1` (T2's module row); at least `1` (T3's records row in the README); `1` (T3 Step 10, edit (b):
the `explain:` entry's `records/ (...)` list, which names chains without `.rs`); at least `1` (T2's test row); `2` (T3:
the module table and the test table); at least `2` (T1-T3's trace totals, ~:294 and the test table); at least `1`
in each of the three files (T1 in the README and `03-structure.yaml`, T2's invariant in `02-system.yaml`); a
list that contains `shaft_joint.rs` (T2); at least `1` (T3); at least `1` (T3); every count at least `1` (this
task). A `0` or an old count is the owning task's docs step missing: apply it from that task's step as the Rules
say, then run this step again.

Then the docs-count tests, the suites, the lints and the format check:
```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count_the_notes_and_the_steps
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count
cargo test --manifest-path magcoupling-rs/Cargo.toml 2>&1 | grep -E '^test result|FAILED|panicked'
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E '^test result|FAILED|panicked'
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
git status --short
```
Expected: the docs tests pass (`test result: ok. 1 passed` without features; `test result: ok. 2 passed` with
`--features app`, more only if T5 or T7 added a docs-count test, all passing); every `test result:` line of both
suites reads `ok` with `0 failed`, and no `FAILED` or `panicked` line; both clippy runs end without a warning or an
error; `cargo fmt --check` prints nothing; `git status --short` shows `M ` lines only for the files this task edited
(and any a Step 4 or Step 13 fallback edited) and `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

- [ ] **Step 14: Commit**

Write the message file `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T9.txt` (gitignored) with the Write tool,
exactly:
```text
docs(magcoupler): BL-064 docs pass: the tracker entry, 04-memory, BL-064's notes and status, BL-060, BL-066, BL-067, new BL-069 to BL-072

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
If Step 3 substituted a shipped text, or Step 4 or Step 13 made you apply an earlier task's docs step, add one body
line per item before the trailers (a blank line before the body and after it), for example
`Applies T5's README step for the status counts, which T5's commit missed.` Then:
```bash
git add docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/ai/02-system.yaml magcoupling-rs/README.md
git status --short
```
Also stage, by name, every file a Step 4 or Step 13 fallback edited (the body names each). Expected
`git status --short`: `M ` lines for exactly the staged files and
`?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged). Then:
```bash
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T9.txt
git log --oneline -1
```
Expected: one new commit whose subject starts `docs(magcoupler): BL-064 docs pass`.

---

### Task 10: Release: whole-branch review, the gate with gate 12 running, web build and gui-smoke, STOP for the owner's review of the diff; after the OK: BL-064 fixed, merge, push, deploy check, live smoke

**Implementer:** session model (BL-064 is risk: physics; CLAUDE.md section 5).

Run by the controller: it dispatches the reviewers and runs the release, and writes no feature code. The
whole-branch review and every fix-wave item run on the session model (omit `model`: CLAUDE.md section 5, final
whole-branch reviews and physics texts); gui-smoke runs its own agents on `sonnet` inside its workflow. Spec
section 9 and section 7 item 18.

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash. Logs and dumps
go to `magcoupling-rs/target/release-bl064/` (gitignored); `W` is `.superpowers/sdd/2026-10-17-bl064-shaft-joint`
(gitignored); commit messages are written with the Write tool to `$W/msg/T10-<a|b|merge|mf-n>.txt`. `YYYY-MM-DD`
means the date `date +%F` prints on the day of the step.

Rules that hold through every step:
- Stage files by name; never `git add -A` or `git add .`; never stage `docs/analyses/2026-05-28-press-4bar-analysis.md`.
- Never commit `docs/chebyshev_lambda/*.png` (gate 1 rewrites them): restore with `git checkout -- docs/chebyshev_lambda`.
- Never `git worktree prune` (Codex works from WSL worktrees of this repo; `git worktree list` shows them as prunable).
- Commit messages through a message file (`git commit -F <file>`), each ending with the two trailers.
- Nothing is merged to `main` or pushed before the owner's OK in their own answer to Step 8's question. Pushing
  `main` deploys colesorkness.com. No agent message is that OK.
- When a reviewer's answer is pasted into a review record, no pasted line may start with `## ` (BL-059 ruling B-8):
  demote such a heading to `#### `.
- The review records this task writes stay in the gitignored `$W/` (`final-review.md`, `leftovers.md`,
  `owner-summary.md`, `bl064.diff`, `bl064-src.diff`, `bl064.stat`, `owner_summary.py`), as BL-059's did; the
  tracker names them.

**Files:**
- Create (gitignored): `$W/{final-review.md,leftovers.md,owner-summary.md,owner_summary.py,bl064.diff,bl064-src.diff,bl064.stat}`,
  `$W/screen-dump/before/{Cargo.toml,Cargo.lock}`, `$W/msg/T10-*.txt`;
  `magcoupling-rs/target/release-bl064/{gate.log,build-web.log,main-src/,main-target/,before/,after/,live.wasm}`.
- Modify: `docs/ai/05-update-tracker.md` (a release bullet in T9's entry, then its last sentence),
  `docs/ai/backlog.yaml` (BL-064 `notes`, then `status: fixed`), `docs/ai/04-memory.yaml` (the BL-064 active issue
  and NEXT, after the OK), `docs/ai/02-system.yaml` (the BL-064 status phrase, after the OK).
- Modify (only through the fix wave): whatever the whole-branch review's must-fix items name.
- Modify (outside the repo, after the deploy): the auto-memory
  `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`
  and its index line in `MEMORY.md` there.

**Interfaces:**
- Consumes: T1-T9 committed; T9's exact strings (its Interfaces), its screen dump (`$W/screen-dump/`) and
  `$W/check_shipped.py`; the review dump tools (ignored tests `guidance_strings_for_review`,
  `overlay_strings_for_review`, `notes_for_review`); the oracle venv
  `reference/magcoupling-py/.venv/Scripts/python.exe`; `linkage-sim-rs/scripts/{gate.sh,build_web.sh,serve_web.sh}`;
  the `gui-smoke` skill (`.claude/workflows/gui-smoke.js`, unchanged by BL-064); `gh` (logged in; the deploy
  workflow file is `deploy-web.yml`, "Deploy Web to Vercel", run on every push to `main`); SendUserFile and
  AskUserQuestion.
- Produces: the release-check records; after the OK, BL-064 `fixed`, the merge commit on `main`, the push, a checked
  deploy and live smoke, the local branch deleted, the memory updated.

**gui-smoke needs no change for BL-064** (checked against `.claude/workflows/gui-smoke.js` and `src/app.rs`
`the_smoke_test_share_link_opens_its_design` at 0687034): the calculator step opens the pinned share link (the
default design, face gap 1.5 mm, Auto on the magnet length) at 1280 x 800 and checks the first view, the Show
spreadsheet details toggle, the share link, the Auto solve, the design file, the Why link under the dashboard's first
row, an input tooltip without spreadsheet details, and that after a 4000 px wheel the dashboard's last row, "Magnets
fit", and its "Why:" line are in view and its first row is not. BL-064 adds one row inside `DASHBOARD` (index 15) and
keeps Magnets fit drawn after `DASHBOARD`. Green or red, the row adds at most one line and one open block (BL-059
measured about 215 to 260 pt for a red block), well inside the 4000 px wheel; on the smoke design it is expected
green: Auto on length meets the requirement, so the joints must hold twice cold-high, the band times the requirement,
8.37 N·m, as at the defaults' Auto length, where both hold (10.26 and 16.90 N·m at Auto's own 13.7695 mm; Task 4's
plan deviation PD-15), and a longer magnet only
lengthens the inner hub. No smoke field counts rows or reads the status line. The
pinned share link and design file come from before BL-064: they load with the 15 new inputs at their defaults (spec
5.4, 7.16), and `the_smoke_test_share_link_opens_its_design` compares them with `Design::default()` plus their two
edits, which still holds. The key now drawn in the bore leaves the first view's "two rings of coloured magnet blocks
inside a grey cup around a shaft" true. What the smoke cannot show is that a new bundle is the one served. Steps 5 and
10 prove it instead: the shipped `.wasm` must contain the passing verdict text (`SHAFT_JOINTS_PASS`, read from
`shaft_joint.rs`) and no longer contain `main`'s whole stored-3D hover text (`STORED_3D_NOTE`, which R-15 extends).
Both were checked on the live file before BL-064 (fb79095's deploy, fetched 2026-10-18): "Both shaft joints hold and
can be fitted" 0 matches, `main`'s `STORED_3D_NOTE` 1.

- [ ] **Step 1: Preconditions and the release invariants**

Run:
```bash
git branch --show-current
git status --short
git log --oneline 0687034..HEAD | wc -l
git log --oneline -1
grep -rn -A2 '#\[ignore' magcoupling-rs/src magcoupling-rs/tests | grep -o 'fn [a-z_0-9]*' | sort -u
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
mkdir -p magcoupling-rs/target/release-bl064
BASE="$(git merge-base main HEAD)"; echo "$BASE"
git diff "$BASE" HEAD --stat -- reference magcoupling-rs/tests/data/python_schema.json magcoupling-rs/tests/data/differential magcoupling-rs/tests/data/deviations magcoupling-rs/tests/python_schema.rs magcoupling-rs/tests/parity.rs magcoupling-rs/tests/differential.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/src/engine/deviations.rs
echo ---
git diff "$BASE" HEAD --stat -- magcoupling-rs/src/engine/sizing.rs magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/engine/size_controls.rs
echo ---
grep -c '^pub const PAUSED: bool = true;' magcoupling-rs/src/engine/clamps.rs
git diff "$BASE" HEAD -- magcoupling-rs/src/engine/clamps.rs | grep '^[-+]' | grep -v '^[-+][-+]' | grep -v -E '^[-+]\s*///'
echo ---
python - "$BASE" <<'EOF'
import json, subprocess, sys
old = json.loads(subprocess.run(["git", "show", sys.argv[1] + ":magcoupling-rs/tests/data/input_schema.json"],
                                capture_output=True, encoding="utf-8", check=True).stdout)
new = json.load(open("magcoupling-rs/tests/data/input_schema.json", encoding="utf-8"))
o = {e["path"]: e for e in old["inputs"]}
n = {e["path"]: e for e in new["inputs"]}
changed = [p for p in o if p not in n or o[p] != n[p]]
added = [p for p in n if p not in o]
print(len(o), len(n), changed, all(p.startswith("shaft_joint.") for p in added), len(added))
EOF
```
Expected:
- `magcoupling/bl064-shaft-joint`; only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`;
- the commits since 0687034 hold every commit step of T1-T9, each with the subject that step names (T7 and T8 may
  add a review-fix commit each), after the plan's commit from Before Task 1 (and the commit of any later plan edit):
  at least `10`; the last commit is T9's (`docs(magcoupler): BL-064 docs pass ...`);
- the ignored functions are `fn guidance_strings_for_review`, `fn help_for_review`, `fn notes_before_snapshot`,
  `fn notes_for_review`, `fn notes_signoff_sheet`, `fn overlay_strings_for_review`, `fn review_sheet` (as at
  0687034), plus only a review tool the plan's T3-T8 steps add by name (check each extra name with
  `grep -c "fn <name>" docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md`, at least `1`);
- `cargo fmt --check` prints nothing; the merge base is fb79095's full hash (if `main` moved, the base is still
  fb79095 until Step 9 merges);
- the first `git diff --stat` prints nothing (no Python-locked row, data file or deviation changed: gate 12's parity
  is untouched); the second prints nothing (Auto unchanged, decision 21), or only a file whose diff is inside its
  `#[cfg(test)] mod tests` and is a test a plan step names (read `git diff "$BASE" HEAD -- <file>` to confirm);
- `1` (the clamp section still paused), and the `clamps.rs` diff filter prints nothing (only doc comments changed:
  owner decision 6A), or only lines inside its `mod tests` that a plan step names (read them);
- the schema check prints `172 187 [] True 15` (no existing input's metadata changed; the 15 new inputs are all
  `shaft_joint.*`).

Anything else: STOP and finish the owning task first.

- [ ] **Step 2: Whole-branch review (fresh context, session model)**

Dispatch one reviewer on the session model (omit `model`) with this prompt, verbatim:

> You are the whole-branch reviewer for BL-064, the magnetic coupling calculator's shaft-joint check (branch
> `magcoupling/bl064-shaft-joint` of C:/Users/Cole/source/repos/linkage_simulation, crate `magcoupling-rs`). Read the
> spec `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md` (binding: the owner decisions of section 2,
> 1A-8A, Bands, Process, Hub temp, Cup temp, Shaft temps, Material, Shafts, Release, R-15 and its ring ruling; the
> technical decisions 9-23; sections 3 to 8; the risks of section 11), the plan
> `docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md` (the controller rulings and plan deviations in its head are
> accepted changes to the spec), the review records `docs/analyses/2026-10-14-magcoupler-notes-review.md`
> (`## shaft_joint`) and `docs/analyses/2026-10-14-magcoupler-help-review.md` (`## BL-064 shaft joints`), then the
> whole diff `git diff main...magcoupling/bl064-shaft-joint`. Read-only: change no file and run no command that
> changes the repository (you may run `cargo test` with `CARGO_TARGET_DIR` set to a folder outside the repository).
> Check, each with file:line evidence:
> (1) every owner decision and technical decision 9-23 is implemented as the spec says, or the plan records a
> deviation the controller accepted; in particular 2A (the fit alone holds the safety factor times cold-high at its
> loosest; yield divided by `YIELD_FACTOR` at its tightest; the key alone is a backup at the same torque; fit and key
> capacities are never added, in the engine or in any text), 6A (`engine::clamps::PAUSED` is still `true`; `clamps.rs`
> changes only in its `PAUSED` doc, which records that un-pausing would show two joint methods, R-13), 7A (the needed
> fitting temperature of section 3.1, with the shaft's shrink from the NIST Fe-2.25Ni fit below 265 K, continuous at
> 265 K and falling strictly as the shaft warms), R-15 (`hub_fit_allowed`: the lowest of `shaft_joint.hub_fit_max_C`,
> the selected adhesive's design limit and the onset less `temperature::CURE_MARGIN_MIN_C`, the constant read, never
> a typed 10; NaN and no limit when any of the three is NaN, tested before any `py_min`; ties name the earlier limit;
> an onset of +inf never binds; the guide calls the same helper) and its ring ruling (the onset read is
> `temperature.demag.inner_onset_single_ring_C`, the inner ring's own `inner.onsets[3]`, never
> `temperature.demag.onset_single_ring_C`; the swapped-rings test, a ferrite inner ring with the NdFeB outer
> governing, tells the two apart; the mixed-ring pins of spec 7.1 set real grade ids or library parts and assert
> which ring governs);
> (2) the engine, `engine/shaft_joint.rs`, against section 3.1 (Lamé with Q = d/D; the holding's contact arc floored
> at zero; the bore's von Mises stress; the key's contact height capped at the keyway depth and its bearing length
> L − b, +inf and a fail where either is not above zero, no key term without a keyway and its margin +inf; no wall
> where Q >= 1, or a part diameter is typed at or below zero (plan deviation PD-1), or the keyway cuts through the wall,
> its Lamé rows NaN), section 3.3 (the loosest and tightest ends;
> the hot end the adhesive's design limit; the rotor-over-shaft allowance on the loosest side only), section 4.3 (the
> cause order; only a determinate joint judged; a determinate joint's red cause outranks the other joint's unknown;
> every comparison written so a NaN fails it), section 4.4 (the unknown test reads all 15 inputs and the 15 engine
> values before any min or max, and a determinate joint's compared terms; then a safety factor not above zero reads
> unknown, plan deviation PD-2) and section 4.2 (the verdict constants;
> the never-equal quoting through one helper pair); every new input and result is Rust-only with no cell;
> (3) every on-screen text the branch adds or changes (the 15 input labels and helps, the 24 result labels and helps
> and the inner onset's, the verdict texts, the guide's label, Problem sentences and Why, the four assumption labels
> and rationales, the section labels, `STORED_3D_NOTE`, the note `shaft_joint`, and any overlay the G-3 review
> changed) against the 22 traps of spec section 6.2, above all: 2 (cold-high "can reach" the joints; never "at most",
> "strongest" or "most expected"; read them at the BL-067 corner, op -40 °C, min 20 °C, variation 0.1, where the
> operating pull-out with its allowance is above cold-high), 9 (every fitting temperature names its limit's source,
> the hub's three forms each with its live number and the onset and the margin quoted with the allowed temperature's
> decimals; a needed temperature is what the fit would take, never one to use; never "safe"), 11 and 12 (no advice to
> cool the gearbox shaft below its rated range or to immerse it; no cryogenic temperature made to sound routine), 14
> and 15 (warm the part or cool the shaft; the owner's fitting order), 18 and 20 (never "permanent" or "no
> backlash"; nothing claims an unchecked item is fine: fretting, the keyway corner, the gearbox's shaft, bearings and
> seal, the bond strain), 21 (no "clamp", case-insensitive, in any shown text, path or section label) and 22 (plain
> words; no cell, id, "BL-064" or "Rust-only"; glyph-safe); and each number true at the defaults, the five probe
> designs, the designs of section 3.6 and any reachable input; the Problem's window quoted inward, its ends in order;
> (4) the note `shaft_joint` as committed matches its `## shaft_joint` record, and the code does what that record
> rules on the spec's risks R-1 (the stress side's smoothing), R-2, R-3, R-4 and R-14 (which body sits at the hot
> end); every other new on-screen text matches the `## BL-064 shaft joints` record: list any text edited after its
> review;
> (5) every test spec section 7 lists (items 1 to 18) exists and asserts values computed from the engine or named
> constants, the spec's deliberate literal pins excepted (the R-15 and ring-ruling numbers, the BL-067 corner, the
> never-equal cases); no test is newly ignored; every existing test the plan moves is moved as the plan says, and no
> other existing assertion was weakened;
> (6) the dashboard: the row at `DASHBOARD` index 15 (after the hidden clamp row and before the gearbox row; Magnets
> fit still last); it greys, uncounted and without a badge, while the end-effect model is out of range; it is in
> `STORED_3D_ROWS` with its hover's label, and the drawn label follows it in both pause states; the defaults read
> "fails 4 of 9 checks" paused and "4 of 10" unpaused; `passing_design()` passes all of them and
> `only_the_joint_fails()` fails 1; near the joint limit the hot minimum's length fix is held back; the three fixes
> held back by rule (spec 5.2), and no fix under an unknown;
> (7) the docs (`magcoupling-rs/README.md`, `docs/ai/02-system.yaml` to `05-update-tracker.md`, and BL-060, BL-064,
> BL-066, BL-067 and BL-069 to BL-072 in `docs/ai/backlog.yaml`) describe the code as it is: no stale count (rows,
> checks, guides, notes, Start here steps, inputs, results, explained results, scope paths, chains), no text that says
> the clamp section returns with BL-064, and BL-072 names the owner-approved E24 of 2026-10-01;
> (8) the share-link guard (`< 2_700`, its comment quoting the measured lengths) and that a design file and a share
> link from before BL-064 load with the new inputs at their defaults;
> (9) the shared rounding helpers (plan deviation PD-24: `apart_from`, `rounded_up`, `rounded_down`,
> `shown_decimals`) leave BL-059's `gearbox_problem`, `quoted_limit`, `quoted_load` and `cure_margin_texts`
> behaviour unchanged (their tests pass unedited); say so with file:line evidence.
> Return PASS, or a numbered list of must-fix items (file:line, the defect, the exact replacement text or the test to
> add) and a separate list of suggestions. A must-fix that changes the note's text names the note id `shaft_joint`;
> one that changes another on-screen text says so.

Save the reviewer's full answer to `.superpowers/sdd/2026-10-17-bl064-shaft-joint/final-review.md`.

- [ ] **Step 3: One fix wave, then the re-review (only if Step 2 returned must-fix items)**

For each must-fix item, in order, one commit each, on the session model (an implementer that does not spawn agents):
1. Write the failing test the item names (or one that shows the defect), run it, see it fail.
2. Apply the fix.
3. A fix that changes the note `shaft_joint` (its sentences, change entries, watch-outs or sources): dump it with
   `NOTES_FOR_REVIEW=shaft_joint NOTES_REVIEW_OUT="$PWD/magcoupling-rs/target/release-bl064/notes-shaft_joint.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture`,
   dispatch a fresh session-model physics reviewer on that dump (the engine code behind each claim, the traps of spec
   section 6.2), apply its must-fix items, move the note's review date to YYYY-MM-DD, and append its report to the
   note's existing section of `docs/analyses/2026-10-14-magcoupler-notes-review.md` as
   `### Re-review after the whole-branch review (YYYY-MM-DD)`: never a second `## shaft_joint` heading
   (`every_note_names_its_committed_review_section` allows exactly one, and
   `every_note_s_review_date_is_recorded_in_its_section` reads the date from that `###` heading).
4. A fix that changes another on-screen text: one fresh session-model G-3 reviewer on the changed strings (dump them
   with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture`,
   `... overlay_strings_for_review ...`, or, for an engine label or help, the screen dump:
   `CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path .superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-dump/after/Cargo.toml`),
   its report appended under `## BL-064 shaft joints` of `docs/analyses/2026-10-14-magcoupler-help-review.md` as
   `### Re-review after the whole-branch review (YYYY-MM-DD)`. A fix that rewords a `shaft_joint.*` input's label or
   help also re-blesses the input schema in the same commit
   (`MAGCOUPLING_BLESS=1 cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`; `tests/schema.rs`
   fails without it), and Step 1's schema check must still print `172 187 [] True 15`.
5. Run the touched suites (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app <filter>` and the
   same without `--features app`), both clippy runs
   (`cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and the same
   without `--features app`) and `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, update the docs the fix
   changes (a docs line that quotes the changed text: re-run T9's Step 2 dump and its Step 3 check with the new text
   in `QUOTES`), and
   write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-mf-<n>.txt`:
```text
fix(magcoupler): <what, in a few words>, from the BL-064 whole-branch review

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   stage the changed files by name and `git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-mf-<n>.txt`.

Then dispatch a fresh session-model reviewer on `git diff <the commit before the fix wave>..HEAD` with the must-fix
list, asking for PASS or the items still open. Append its answer to `final-review.md`. A second round of must-fix
items: apply them the same way and re-review once more; a third: STOP and report to the owner.

- [ ] **Step 4: The gate, with gate 12 running**

`gate.sh` finds the oracle's Python in this order: `$MAGCOUPLING_PYTHON` (used as given, so a wrong value fails the
gate), then `reference/magcoupling-py/.venv/Scripts/python.exe` (or `.venv/bin/python`) in this checkout or another
worktree. It prints `oracle Python: <path>` when gate 12 runs, and `SKIP gate 12/12: no Python for
reference/magcoupling-py ...` (without failing) when none is found. Naming the venv explicitly makes a SKIP
impossible.

Run in the background (about 6 minutes warm, longer cold):
```bash
MAGCOUPLING_PYTHON="$PWD/reference/magcoupling-py/.venv/Scripts/python.exe" bash linkage-sim-rs/scripts/gate.sh 2>&1 | tee magcoupling-rs/target/release-bl064/gate.log
```
When it exits, run:
```bash
L=magcoupling-rs/target/release-bl064/gate.log
tail -n 1 $L
grep -c 'SKIP gate 12/12' $L
grep -n 'oracle Python:' $L
grep -c '^== gate [0-9]*/12' $L
grep -E '^[0-9]+ passed' $L
grep -n 'differential data is current' $L
git checkout -- docs/chebyshev_lambda
git status --short
```
Expected: `GATE PASS`; `0`; `oracle Python: /c/Users/Cole/source/repos/linkage_simulation/reference/magcoupling-py/.venv/Scripts/python.exe`;
`12`; one pytest summary line `<N> passed ...` with no `failed` in it; `differential data is current (...)`; after the
restore, `git status --short` prints only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.
On a failure: STOP the release. The failing gate's output names the test; fix it as a must-fix of Step 3 (red test,
fix, the reviews it needs, commit) and run this step again from the top. Never edit `gate.sh` to pass.

- [ ] **Step 5: Web build, and the new check is in the bundle**

Run:
```bash
D=magcoupling-rs/target/release-bl064
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
B=$D/build-web.log
bash linkage-sim-rs/scripts/build_web.sh 2>&1 | tee $B
grep -n 'parity guard:\|Magcoupling build complete!' $B
WASM=linkage-sim-rs/web/tools/magcoupler/magcoupling-web_bg.wasm
PASS_TEXT="$(sed -n '/pub const SHAFT_JOINTS_PASS:/,/;/p' magcoupling-rs/src/engine/shaft_joint.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
OLD_NOTE="$(git show "$(git merge-base main HEAD)":magcoupling-rs/src/gui/dashboard.rs | sed -n '/^pub const STORED_3D_NOTE:/,/;/p' | grep -o '"[^"]*"' | head -1 | tr -d '"')"
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path $W/screen-dump/after/Cargo.toml > $D/screen-step5.tsv
NEW_NOTE="$(awk -F'\t' '$1=="stored 3D" && $2=="note" {print $3}' $D/screen-step5.tsv)"
echo "[$PASS_TEXT]"
echo "[$OLD_NOTE]"
[ -n "$PASS_TEXT" ] && grep -c -a -F "$PASS_TEXT" $WASM
[ -n "$OLD_NOTE" ] && grep -c -a -F "$OLD_NOTE" $WASM
case "$NEW_NOTE" in *"$OLD_NOTE"*) echo "main's note is inside the new one: the second count cannot tell the bundles apart";; *) echo "main's note is not inside the new one";; esac
git status --short
```
Expected: `parity guard: magcoupling-web builds without workbook-parity; magcoupling-rs features: ...` and
`Magcoupling build complete!`; the verdict text in brackets (`[Both shaft joints hold and can be fitted]` unless the
G-3 review reworded it; `[]` means the constant was not found: STOP and read `shaft_joint.rs`, since an empty text
would match everything); `main`'s stored-3D hover text in brackets (it starts `[The temperature checks use the
demagnetization reverse fields`); at least `1`; `0`; `main's note is not inside the new one` (if the other line
prints, the second count proves nothing: record that, and the first count alone proves the new bundle); `git status`
unchanged (the bundles are gitignored). If the release link fails with `LNK1104` (Windows MAX_PATH, 02-system
lessons), build from a short path: `subst X: "$(cygpath -w "$PWD")"`, then `cd /x && bash
linkage-sim-rs/scripts/build_web.sh`, then `subst X: /d`.

- [ ] **Step 6: gui-smoke on the local server**

Start the server in the background: `bash linkage-sim-rs/scripts/serve_web.sh 8080` (it serves
`linkage-sim-rs/web/` with `python -m http.server` at `http://localhost:8080/tools/`, `/tools/linkage/` and
`/tools/magcoupler/`). Then invoke the `gui-smoke` skill with no arguments (it smokes `http://localhost:8080`).
Expected: `passed: true`; for the linkage step `hub_page: true`, `canvas_present: true`, `title_warning: false`; for
the calculator `first_view`, `details_toggle`, `share_link_loaded`, `sizing_solved`, `design_file_loaded`,
`explorer_ready`, `why_link_opened`, `tooltip_plain` and `dashboard_scrolls` all `true` and `console_errors` empty.
If only a canvas-click field is false (`design_file_loaded`, `why_link_opened`, `tooltip_plain`, `dashboard_scrolls`)
and its notes say the target was not found, run the skill once more. A field false twice, or any console error: STOP
and treat it as a must-fix (Step 3), with the smoke's notes and screenshots as evidence. Stop the server afterwards
(stop its background task). `git status --short` must again print only the owner's untracked file (the smoke writes
and deletes `.playwright-mcp/magcoupling-smoke-design.json`, gitignored).

- [ ] **Step 7: Record the release checks and commit**

1. `docs/ai/05-update-tracker.md`: append a bullet as the last line of T9's BL-064 entry. The tracker has two
   `- Docs and backlog (T9):` bullets (BL-059's too), so anchor on the end of BL-064's: replace
```text
and the key's shear.

## 2026-10-16 — Magcoupler gearbox check
```
   with the same text plus this bullet between the first line and the blank line. Write `passed with no must-fix` if
   Step 2 returned PASS, else `passed after its fix wave of <N> items (re-reviewed)` with the count of Step 3's
   commits; if Step 5 printed that `main`'s note is inside the new one, end the web-build clause at
   `magcoupling-web_bg.wasm`:
```text
- T10 release checks (YYYY-MM-DD): the whole-branch review passed with no must-fix (record: .superpowers/sdd/2026-10-17-bl064-shaft-joint/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the check (its passing verdict text is in magcoupling-web_bg.wasm, main's stored-3D hover text is not); a gui-smoke run passed (hub, linkage; magcoupler: first view, the Show spreadsheet details toggle, share link, Auto solve, design file, a Why link opening a note, an input tooltip without spreadsheet details, the dashboard scrolling to Magnets fit at 1280 x 800, a clean console). Not yet done: the owner's review of the diff, the merge, the push and the deploy check.
```
2. `docs/ai/backlog.yaml`, BL-064: append to its `notes:` string, right before the closing `"`:
   ` Release checks done YYYY-MM-DD (T10): whole-branch review PASS, GATE PASS with gate 12 running, web build and gui-smoke PASS; waiting for the owner's review of the diff.`
   Anchor: `until the owner's OK and the push."` (with its closing quote) is BL-064's alone; BL-059's notes go on past
   that sentence.
3. Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-a.txt`:
```text
docs(magcoupler): record the BL-064 release checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
then:
```bash
git add docs/ai/05-update-tracker.md docs/ai/backlog.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-a.txt
```
Expected: the two files staged (`M `) and the owner's file untracked; one new commit.

- [ ] **Step 8: STOP: the owner's summary and the diff, then ask for the OK**

1. The texts as committed, from the branch (`after/`):
```bash
D=magcoupling-rs/target/release-bl064
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
mkdir -p $D/after $D/before
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > $D/after/guidance-strings.txt
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > $D/after/overlay-strings.txt
NOTES_REVIEW_OUT="$PWD/$D/after/notes.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path $W/screen-dump/after/Cargo.toml > $D/after/screen.tsv
grep -c '^test result: ok. 1 passed' $D/after/guidance-strings.txt $D/after/overlay-strings.txt
```
Expected: the notes run prints `21 notes at 5 designs: .../notes.txt` and `test result: ok. 1 passed` (no
`NOTES_FOR_REVIEW`: every note); the `grep -c` prints `1` for each dump file.

2. The same texts from `main` (`before/`), from a clean copy of the merge base (the first build there takes several
   minutes; it may run in the background from Step 4 on):
```bash
D=magcoupling-rs/target/release-bl064
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
BASE="$(git merge-base main HEAD)"
rm -rf $D/main-src && mkdir -p $D/main-src $D/before
git archive "$BASE" | tar -x -C $D/main-src
(cd $D/main-src && export CARGO_TARGET_DIR="$PWD/../main-target" &&
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > ../before/guidance-strings.txt &&
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > ../before/overlay-strings.txt &&
  NOTES_REVIEW_OUT="$PWD/../before/notes.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture)
mkdir -p $W/screen-dump/before
sed -e 's/screen-dump-after/screen-dump-before/g' -e "s#C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs#$(cygpath -m "$PWD/$D/main-src/magcoupling-rs")#" $W/screen-dump/after/Cargo.toml > $W/screen-dump/before/Cargo.toml
cp $D/main-src/magcoupling-rs/Cargo.lock $W/screen-dump/before/Cargo.lock
CARGO_TARGET_DIR="$PWD/$D/main-target" cargo run --quiet --manifest-path $W/screen-dump/before/Cargo.toml > $D/before/screen.tsv
grep -c '^test result: ok. 1 passed' $D/before/guidance-strings.txt $D/before/overlay-strings.txt
awk -F'\t' '$1=="count" && ($2=="inputs" || $2=="results")' $D/before/screen.tsv
git status --short
```
Expected: the notes run prints `20 notes at 5 designs: ...`; the `grep -c` prints `1` for each file; the counts
read `count inputs 172` and `count results 1087` (the dry run of this tool at 0687034 read the same: `main`'s code
equals 0687034's, `git diff fb79095 0687034 --stat -- magcoupling-rs` is empty); `git status` shows only the owner's
untracked file (`main-src/` and the dumps are under the gitignored `magcoupling-rs/target/`).

3. The diffs:
```bash
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
git diff main...HEAD > $W/bl064.diff
git diff main...HEAD -- . ':(exclude)magcoupling-rs/tests' ':(exclude)*_test*' > $W/bl064-src.diff
git diff --stat main...HEAD > $W/bl064.stat
tail -n 1 $W/bl064.stat
```
Expected: the stat's last line counts the files changed. `bl064-src.diff` is the production and docs part of the
diff: it leaves out `magcoupling-rs/tests` and test files, while the in-file `mod tests` blocks stay in the full diff
only.

4. Write `$W/leftovers.md` with the Write tool: the summary's sections 7 to 11, in this order and with these headings.
   Sections 8, 10 (its first sentence) and 11 are fixed text; 7 and 9 come from the records:
```markdown
## 7. The physics review's rulings on the spec's risks

<one line each for R-1 (the stress side's smoothing), R-2 (the from-memory constants: the yield factor 1.2, the
safety factor 2, friction 0.14, the 5 µm smoothing, the 10 µm joining clearance, the shaft's E, ν and expansion, the
NIST curve, the key flank allowable, the UltraOx and 4140 tempering temperatures), R-3 (the cup's expansion at
300 °C), R-4 (the seal side) and R-14 (which body sits at the hot end): the decision as the `## shaft_joint` record
states it, quoted, with the record's heading; "not ruled" where the record is silent>

## 8. Open facts for you (none blocks the merge; all in BL-064's notes and docs/ai/04-memory.yaml)

- The gearbox output shaft's real diameter, tolerance and key; the wheel shaft's diameter, tolerance and material. Both are Ø10 through one input until you give them (BL-071).
- The real interference bands: inner hub 13-18 µm and cup 15-22 µm are placeholders.
- The 4140 condition (annealed, 415 MPa, the check's conservative yield, or pre-hardened) and the UltraOx process and its temperature (the bare cup's 300 °C default and 400 °C slider top must stay below it; BL-069).
- The key length and the real keyway depth: the 1.7 mm keyway default matches no DIN 6885-1 row (3 × 3 has a hub depth t2 of 1.4 mm); the key contact input, 1.2 mm, is separate.
- The bondline thickness: at 0.05 mm the bond screen reaches the AA 326 lap shear near 113 °C, at 0.10 mm near 198 °C; the bonded hub's 100 °C default rests on the 0.05 mm figure.
- The four new backlog items, at the priorities you set in the plan review (drafted: BL-069 3; BL-070, BL-071 and BL-072 2; ruling C-9). BL-072 is the E24 correction you approved on 2026-10-01, which was to land with E21 (BL-057).

## 9. Leftovers from the reviews (none blocks the merge; your call)

<every deferred minor of the build ledger and every suggestion of final-review.md, one bullet each: what, where
(file:line), the offered fix; "on screen" when it changes a shown text (it then needs its own G-3 re-review); or the
single line "None.">

## 10. Also for you, not BL-064's

- docs/ai/02-system.yaml does not parse as strict YAML (near line 14), at 0687034, before this branch. It is left as it is here.

## 11. Checks done

The physics review of the new note and the G-3 review of every new text (records in docs/analyses/2026-10-14-magcoupler-notes-review.md and docs/analyses/2026-10-14-magcoupler-help-review.md), the whole-branch review, gate.sh GATE PASS with the Python oracle running (gate 12), the web build (the new check is in the bundle) and the web smoke test.
```
   Sections 7 and 9 are read from these, never from memory:
```bash
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
awk '/^## /{on=($0=="## shaft_joint")} on' docs/analyses/2026-10-14-magcoupler-notes-review.md | grep -n -E 'R-1[^0-9]|R-2|R-3|R-4|R-14'
L=.superpowers/sdd/2026-10-18-bl064-shaft-joint/progress.md
head -n 1 $L
grep -n 'minor (deferred)' $L
grep -n -i -A40 'suggestion' $W/final-review.md | head -80
python -c "import yaml; yaml.safe_load(open('docs/ai/02-system.yaml', encoding='utf-8'))" 2>&1 | tail -n 1
```
   `$L` is the build ledger (Global Constraints), not a file in `$W`. Its first line names this plan file
   (`# SDD ledger — plan: ...2026-10-18-bl064-shaft-joint.md`), and the grep prints every deferred minor of T1-T10, or
   nothing when no review deferred one. If `head` fails (no ledger), STOP and ask the controller where the task
   reviews' minor findings were recorded: section 9 is never written as "None." for want of the file.
   If the last command prints nothing (02-system.yaml parses now), drop section 10's bullet and write "Nothing."

5. Write `$W/owner_summary.py` with the Write tool, exactly:

```python
"""BL-064 T10 Step 8: the owner's summary of every on-screen text and number the branch
changes, before (main) -> after (the branch).

Both sides are read by the same tools from each checkout, in BEFORE_DIR and AFTER_DIR:
screen.tsv (the screen-dump tool: the dashboard, inputs, results, assumptions, notes, Start here,
the share link, the joint's verdicts and the counts at the example design), guidance-strings.txt
and overlay-strings.txt (the crate's review dumps) and notes.txt (notes_for_review). Nothing here
is typed from a draft; the owner's open facts are the spec's (section 2), the leftovers are the
controller's file.

Usage:
  python -I owner_summary.py BEFORE_DIR AFTER_DIR GUIDE_LABEL MAIN_SHORT STAT_LINE OUT [EXTRA.md ...]
"""

import collections
import re
import sys

TAG = re.compile(r"^\[([^\]]+)\] (.*)$")
DESIGN_KINDS = ("fix", "note", "problem")


def read_screen(path):
    """kind -> key -> text, in file order."""
    kinds = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                kind, key, text = line.split("\t", 2)
                kinds.setdefault(kind, collections.OrderedDict())[key] = text
    return kinds


def read_guidance(path):
    """tag -> its texts, in file order (a fix or note tag holds several)."""
    tags = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = TAG.match(line.rstrip("\r\n"))
            if m:
                tags.setdefault(m.group(1), []).append(m.group(2))
    return tags


def read_overlay(path):
    """(table, path) -> the text the overlay shows."""
    rows = collections.OrderedDict()
    key = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = TAG.match(line)
            if m:
                key = (m.group(1), m.group(2))
            elif key is not None and line.startswith("  now: "):
                rows[key] = line[len("  now: "):]
    return rows


def read_notes(path):
    """note id -> its block (title, sources and every probe design), as lines."""
    blocks = collections.OrderedDict()
    note = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = re.match(r"^==== (\S+) \(", line)
            if m:
                note = m.group(1)
                blocks[note] = []
            elif note is not None:
                blocks[note].append(line)
    return blocks


def at_default(lines):
    """A note as drawn at the default design: title, that design's lines, sources."""
    title, body, sources = "", [], []
    where, on = False, False
    for line in lines:
        if line.startswith("Title: "):
            title = line[len("Title: "):]
        elif line.startswith("Where this comes from:"):
            where = True
        elif line.startswith("-- at the "):
            where, on = False, line == "-- at the default design"
        elif where and line.startswith("  - "):
            sources.append(line[4:])
        elif on and line.strip() and not line.lstrip().startswith("aim "):
            body.append(line)
    return title, body, sources


def cell(text):
    if text is None or text == "":
        return "(none)"
    return text.replace("|", "\\|").replace("\n", "<br>")


def changes(before, after):
    """(key, before, after) for every key whose text differs, before's order first."""
    keys = list(before) + [k for k in after if k not in before]
    return [(k, before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)]


def table(rows, head=("Where", "Before (main)", "After (this branch)")):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return out


def get(screen, kind, key):
    """The text of one screen record, or None where the checkout has none."""
    return screen.get(kind, {}).get(key)


def on_main(text):
    """A before cell: main has none of the shaft-joint inputs, so an edit of one is not a design."""
    if text is not None and text.startswith("(not on this checkout"):
        return "(no such input on main)"
    return text


def key_drawn(width_record):
    """The geometry row's after cell, from the shaft key width input's record."""
    m = re.search(r" default (\S+)", width_record or "")
    if m is None:
        return None
    return f"drawn in both states from Shaft key width, {m.group(1)} mm by default (spec decision 20)"


def parts(tag):
    """(kind, label, design) of a guidance tag; design None where the tag has none."""
    bits = tag.split("/")
    if bits[0] in DESIGN_KINDS and len(bits) >= 3:
        return bits[0], "/".join(bits[1:-1]), bits[-1]
    return bits[0], "/".join(bits[1:]), None


def main(argv):
    before_dir, after_dir, label, main_short, stat, out_path = argv[1:7]
    extras = argv[7:]
    sb = read_screen(f"{before_dir}/screen.tsv")
    sa = read_screen(f"{after_dir}/screen.tsv")
    gb = read_guidance(f"{before_dir}/guidance-strings.txt")
    ga = read_guidance(f"{after_dir}/guidance-strings.txt")
    ob = read_overlay(f"{before_dir}/overlay-strings.txt")
    oa = read_overlay(f"{after_dir}/overlay-strings.txt")
    nb = read_notes(f"{before_dir}/notes.txt")
    na = read_notes(f"{after_dir}/notes.txt")
    lines = []
    w = lines.append

    w("# BL-064, the shaft joints: what changes on screen")
    w("")
    w(f"Branch `magcoupling/bl064-shaft-joint` against `main` ({main_short}): {stat.strip()}. The full diff")
    w("is attached (bl064.diff), and its production and docs part (bl064-src.diff). Every text below is")
    w("what the site shows at the example design (the defaults), with the shaft-clamp section paused,")
    w("unless a row names another design. Both columns were read by the same tools from each checkout.")
    w("")

    # 1. At a glance.
    new_rows = [k for k in sa.get("row", {}) if k not in sb.get("row", {})]
    steps_b = [v for k, v in sb.get("start here", {}).items() if k.startswith("step ")]
    steps_a = [v for k, v in sa.get("start here", {}).items() if k.startswith("step ")]
    new_steps = [s for s in steps_a if s not in steps_b]
    glance = [
        ("Dashboard status line", get(sb, "status", "paused"), get(sa, "status", "paused")),
        ("Status line with the clamp section shown", get(sb, "status", "unpaused"), get(sa, "status", "unpaused")),
        ("Dashboard rows shown", get(sb, "count", "dashboard rows shown while paused"), get(sa, "count", "dashboard rows shown while paused")),
        ("New dashboard row", "(no row)", "<br>".join(get(sa, "row", k) for k in new_rows) or None),
        ("Its hover", "(no row)", "<br>".join(get(sa, "row hover", k) for k in new_rows) or None),
        ("'Fixed 3D fields' label drawn under", get(sb, "stored 3D", "label drawn after"), get(sa, "stored 3D", "label drawn after")),
        ("Start here steps (paused)", str(len(steps_b)), f"{len(steps_a)}: new step {', '.join(new_steps) or '(none)'}"),
        ("Teaching notes shown", get(sb, "count", "notes shown"), get(sa, "count", "notes shown")),
        ("Share link of the default design", get(sb, "share link", "default design"), get(sa, "share link", "default design")),
        ("Inputs (Inputs sheet, design files)", get(sb, "count", "inputs"), get(sa, "count", "inputs")),
        ("Results (results table, CSV, JSON)", get(sb, "count", "results"), get(sa, "count", "results")),
        ("Spreadsheet Summary: dashboard lines", get(sb, "count", "dashboard rows"), get(sa, "count", "dashboard rows")),
        ("Geometry view: the key in the bore", "not drawn while the clamp section is paused",
         key_drawn(get(sa, "input", "shaft_joint.key_width_mm"))),
    ]
    w("## 1. At a glance")
    w("")
    lines += table([(f"{i}. {a}", b, c) for i, (a, b, c) in enumerate(glance, 1)])
    w("")

    # 2. The new check on other designs.
    w("## 2. The shaft-joint row at other designs (badge, and the status line there)")
    w("")
    rows = [(k, on_main(get(sb, "verdict", k)), v) for k, v in sa.get("verdict", {}).items()]
    lines += table(rows, ("Design", "Before (main)", "After (this branch)"))
    w("")

    # 3. The guide.
    w(f"## 3. Its guide ('{label}'): Why, and at each design where it shows, Problem, Fix and notes")
    w("")
    for kind in ("why", "learn"):
        for text in ga.get(f"{kind}/{label}", []):
            w(f"- **{kind.capitalize()}:** {text}")
    by_design = collections.OrderedDict()
    for tag, texts in ga.items():
        kind, lab, design = parts(tag)
        if lab == label and design is not None:
            by_design.setdefault(design, {}).setdefault(kind, []).extend(texts)
    w("")
    lines += table(
        [(d, "<br>".join(k.get("problem", [])), "<br>".join(k.get("fix", [])), "<br>".join(k.get("note", [])))
         for d, k in by_design.items()],
        ("Design", "Problem", "Fix", "Notes under the fixes"),
    )
    w("")

    # 4. Every other changed text, before -> after.
    w("## 4. Every other text or number that changes, before -> after")
    w("")
    designs_b = {parts(t)[2] for t in gb if parts(t)[2]}
    designs_a = {parts(t)[2] for t in ga if parts(t)[2]}
    new_designs = sorted(designs_a - designs_b)
    guide_rows = []
    for tag in list(gb) + [t for t in ga if t not in gb]:
        kind, lab, design = parts(tag)
        if lab == label or (design is not None and design not in designs_b):
            continue
        b, a = gb.get(tag), ga.get(tag)
        if b != a:
            guide_rows.append((tag, "<br>".join(b or []) or None, "<br>".join(a or []) or None))
    sections = [("Other guides and the Size tab (guidance dump)", guide_rows)]
    overlay_rows = [(f"{table_name} {path}", b, a) for (table_name, path), b, a in changes(ob, oa)]
    sections.append(("On-screen overlay labels and helps", overlay_rows))
    for kind, title in (
        ("row", "Dashboard rows (number, name, value, badge)"),
        ("row hover", "Dashboard row hovers"),
        ("guide label", "Check guides' labels"),
        ("guide why", "Check guides' Why sentences"),
        ("stored 3D", "The stored-3D label and its hover"),
        ("input section", "Input groups and sections (rows shown while paused)"),
        ("input", "Inputs (on-screen label, unit, default, slider, help)"),
        ("result", "Results (label, unit, value at the defaults, help)"),
        ("assumption", "Assumptions view"),
        ("note", "Teaching notes (title, shown or hidden)"),
    ):
        sections.append((title, changes(sb.get(kind, {}), sa.get(kind, {}))))
    hover_b, hover_a = get(sb, "start here", "hover"), get(sa, "start here", "hover")
    sections.append(("Start here", [
        ("Start here button hover", hover_b, hover_a),
        ("Start here steps, in order", "<br>".join(steps_b), "<br>".join(steps_a)),
    ] if (hover_b, steps_b) != (hover_a, steps_a) else []))
    for title, rows in sections:
        w(f"### {title}")
        w("")
        if rows:
            lines += table(rows)
        else:
            w("No change.")
        w("")
    if new_designs:
        w(f"The guidance dump also has test designs only on the branch ({', '.join(new_designs)}); their")
        w("lines for the other guides are new designs, not changed texts, and are in the full dump.")
        w("")

    # 5. Notes.
    w("## 5. Teaching notes")
    w("")
    new_notes = [n for n in na if n not in nb]
    changed_notes = [n for n in nb if n in na and nb[n] != na[n]]
    w(f"Notes whose drawn text changed at any probe design: {', '.join(changed_notes) or 'none'}.")
    w(f"New notes: {', '.join(new_notes) or 'none'}.")
    w("")
    shown = [(n, "new", None, na[n]) for n in new_notes]
    shown += [(n, "changed", nb[n], na[n]) for n in changed_notes]
    for n, state, before, after in shown:
        for side, block in (("before (main)", before), ("after (this branch)", after)):
            if block is None:
                continue
            title, body, sources = at_default(block)
            w(f"### {n} ({state}), {side}: \"{title}\" at the example design")
            w("")
            w("```text")
            lines.extend(body)
            w("Where this comes from:")
            lines.extend(f"  - {src}" for src in sources)
            w("```")
            w("")

    # 6. Counts.
    w("## 6. Pinned counts that moved")
    w("")
    lines += table(changes(sb.get("count", {}), sa.get("count", {})), ("Count", "Before (main)", "After (this branch)"))
    w("")

    for extra in extras:
        with open(extra, encoding="utf-8") as f:
            w(f.read().rstrip())
        w("")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    empty = [r for r in glance if r[2] in (None, "")]
    print(f"{out_path}: {len(lines)} lines; at-a-glance rows with no after text: {len(empty)}")


if __name__ == "__main__":
    main(sys.argv)
```

6. Build the summary and check it:
```bash
D=magcoupling-rs/target/release-bl064
W=.superpowers/sdd/2026-10-17-bl064-shaft-joint
GUIDE="$(awk -F'\t' '$1=="guide label" && $2=="shaft_joint.check" {print $3}' $D/after/screen.tsv)"
echo "[$GUIDE]"
python -I $W/owner_summary.py $D/before $D/after "$GUIDE" "$(git rev-parse --short "$(git merge-base main HEAD)")" "$(tail -n 1 $W/bl064.stat)" $W/owner-summary.md $W/leftovers.md
OUT=$W/owner-summary.md
grep -c '(not on this checkout' $OUT
grep -o '(no shaft-joint check)' $OUT | wc -l
grep -c '^## ' $OUT
sed -n '/^## 6\. Pinned counts that moved/,/^## 7/p' $OUT
```
Expected: the guide label in brackets (`[Shaft joints]` unless the G-3 review reworded it; `[]`: STOP, the screen
dump has no guide for the check); the script prints `.../owner-summary.md: <N> lines; at-a-glance rows with no after
text: 0`; `0` (every probe of section 2 is a design on the branch); `10` (the "before" cells of the five probes and
of the five edits `main` has inputs for; any more means the branch's column lacks the check somewhere); `11` (sections
1 to 6 and leftovers' 7 to 11); and section 6 lists exactly these 20 rows:

| Count | Before (main) | After (this branch) |
|---|---|---|
| inputs | 172 | 187 |
| results | 1087 | 1112 |
| results explained | 392 | 416 |
| scope paths | 169 | 193 |
| scope chains | 7 | 8 |
| dashboard rows | 18 | 19 |
| dashboard rows shown while paused | 17 | 18 |
| checks | 31 | 32 |
| guides | 9 | 10 |
| guides counted while paused | 8 | 9 |
| end-effect rows | 8 | 9 |
| stored-3D rows | 3 | 4 |
| notes | 20 | 21 |
| notes shown | 19 | 20 |
| start here steps shown | 13 | 14 |
| start here steps in all | 14 | 15 |
| assumptions | 14 | 18 |
| input sections | 18 | 19 |
| result chain groups | 8 | 9 |
| result package groups | 13 | 14 |

   A different "after" number is a finding: read the owning task's pinned test before sending anything. Then read
   the file once in full. Section 1 row 1 must read `Example design: fails 4 of 8 checks` -> the shipped status (T9's
   check read `Example design: fails 4 of 9 checks`); row 4 the new row with its passing verdict; section 2 the badges
   spec sections 3.6 and 4.2 give these designs (a pass at the probes but the 304 back iron's slip; the slip, the bore
   stress, cannot be fitted and the key backup each red at its design, the boss length 0 reading its slip first;
   amber at the reversed band; at the bore 16 mm in a 12 mm boss the inner hub's slip, red, outranks the cup's no
   wall, spec 4.3), and in each "after" cell a status line that counts 9 checks; section 5 `shaft_joint` as the one new
   note and no changed note (a changed note is a finding to check against the plan before sending); section 4 no
   change in "Results" outside `shaft_joint.*` and `temperature.demag.inner_onset_single_ring_C` (any other result
   that moved at the defaults is a finding).

7. Send the owner three files with SendUserFile: `$W/owner-summary.md`, `$W/bl064.diff` (the full diff) and
   `$W/bl064-src.diff` (the production and docs part of the diff).

8. Ask with AskUserQuestion (recommended options first), with these two questions (ask the second only if section 9
   lists at least one leftover):
   - Question 1: "1. BL-064 (the shaft-joint check) is built and checked: whole-branch review passed, GATE PASS with
     gate 12, web smoke passed. The summary and the diff are attached. Merge to main and push? (The push deploys
     colesorkness.com/tools/magcoupler.)"
     - Option "1A OK: merge, push and watch the deploy (recommended)": the diff and the texts are as reviewed; the
       open facts (section 8) stay open in BL-064's notes and do not block the release (spec section 2).
     - Option "1B Change texts first": name the summary's rows to change; each change is a must-fix of Step 3 (with
       its re-review), then Steps 4-8 again.
     - Option "1C Hold": do not merge yet; the branch stays as it is.
   - Question 2: "2. The reviews' leftovers (section 9 of the summary; none blocks the merge): what should happen to
     them?"
     - Option "2A Fold them in before the merge (recommended)": BL-059's precedent; each docs item is edited, each
       on-screen item gets its G-3 re-review, then Steps 4-6 (gate, web build, smoke) run again before the merge.
     - Option "2B File them as one backlog item and merge now".
     - Option "2C Leave them in the review records and merge now".

Do nothing further until the owner answers in their own message. 1B: run the changes as Step 3 says, then Steps 4-8
again. 1C: stop here and report the branch state. 1A with 2A: fold the leftovers in as Step 3 does (one commit per
item, `msg/T10-lo-<n>.txt`, subject `fix(magcoupler): <what>, from the BL-064 review leftovers`; each on-screen item
with its G-3 re-review appended under `## BL-064 shaft joints` as `### Re-review of the review leftovers
(YYYY-MM-DD)`), run Steps 4-6 again, then Step 9. 1A with 2B: file one backlog item (next free id, `risk` the
highest of its items, `status: open`) listing them, in Step 9's commit. 1A with 2C: Step 9.

- [ ] **Step 9: After the OK only: record it, merge, push**

1. `docs/ai/backlog.yaml`, BL-064: replace `  status: in_progress` (the only one in the file, BL-064's, between
   `priority: 2` and its `notes:`) with `  status: fixed`, and append to its `notes:` string, right before the
   closing `"` (anchor: `waiting for the owner's review of the diff."` with its closing quote, BL-064's alone):
   ` The owner reviewed the diff and gave the OK on YYYY-MM-DD (AGENTS.md risk: physics); merged to main and pushed the same day.`
   With 2A, write instead
   ` The owner reviewed the diff and gave the OK on YYYY-MM-DD (AGENTS.md risk: physics), after the review leftovers were folded in at the owner's choice (2A); merged to main and pushed the same day.`
2. `docs/ai/05-update-tracker.md`: in the T10 release bullet of Step 7, replace
   `Not yet done: the owner's review of the diff, the merge, the push and the deploy check.` with
   `The owner reviewed the diff and gave the OK on YYYY-MM-DD; merged to main and pushed; the deploy run and the live smoke are checked in the session that pushed.`
3. `docs/ai/04-memory.yaml`:
   - in the BL-064 active issue, replace `Magcoupler shaft joints (BL-064, branch magcoupling/bl064-shaft-joint; spec`
     with `Magcoupler shaft joints (BL-064; spec`, and replace
     `built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, gui-smoke) and then the owner's review of the diff come before the merge and the push (risk: physics).`
     with `released YYYY-MM-DD after the owner's review of the diff.`;
   - in the NEXT item, replace `then BL-064, the shaft joints` with `BL-064, the shaft joints, shipped YYYY-MM-DD`.
4. `docs/ai/02-system.yaml`: replace `built on branch magcoupling/bl064-shaft-joint` with `merged and deployed YYYY-MM-DD`.
5. Check the YAML still parses: run Task 9's Step 12 with its `'in_progress'` assertion changed to `'fixed'` and
   its 02-system marker `'built on branch magcoupling/bl064-shaft-joint'` changed to
   `'(BL-064, owner decisions 1A-8A, R-15 and its ring ruling) merged and deployed'` (and its 04-memory `startswith`
   still `Magcoupler shaft joints (BL-064`). Expected: `72 5 66`. With 2B's item, placed after BL-072, its first
   assertion's `range(56, 73)` becomes `range(56, 74)` and the expected line `73 5 66`.
6. Write `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-b.txt`:
```text
docs(magcoupler): record the owner's OK of BL-064, the shaft-joint check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   and `.superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-merge.txt`:
```text
Merge magcoupling/bl064-shaft-joint: the magcoupler shaft-joint check (BL-064)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
7. Commit on the branch, then merge and push:
```bash
git add docs/ai/backlog.yaml docs/ai/05-update-tracker.md docs/ai/04-memory.yaml docs/ai/02-system.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-b.txt
git checkout main
git pull --ff-only origin main
git merge --no-ff magcoupling/bl064-shaft-joint -F .superpowers/sdd/2026-10-17-bl064-shaft-joint/msg/T10-merge.txt
git status --short
git log --oneline -1
git push origin main
```
Expected: the four files staged and the owner's file untracked; the pull prints `Already up to date.` (main was
fb79095, as `origin/main`); if `main` moved, STOP: merge `main` into the branch, then Steps 4-6 and 8 again, then
here; the merge has no conflicts; `git status` shows only the owner's untracked file; the log's subject is
`Merge magcoupling/bl064-shaft-joint: the magcoupler shaft-joint check (BL-064)`; the push succeeds
(`fb79095..<merge> main -> main`).

- [ ] **Step 10: Watch the deploy; prove the new bundle is live; live smoke**

Run:
```bash
git rev-parse HEAD
gh run list --workflow deploy-web.yml --branch main --limit 1 --json databaseId,headSha,status,conclusion
```
Expected: the run's `headSha` equals `git rev-parse HEAD` (if the newest run is still fb79095's, wait a few seconds
and list again). Then, with that run's `databaseId`:
```bash
gh run watch <databaseId> --exit-status
```
Expected: the run ends `completed` with exit status 0 (BL-059's run, 39045017417, took the same path). On a failure:
`gh run view <databaseId> --log-failed`, report it to the owner, and fix on a new branch with its own red test; never
push again blindly.

Then the live bundle (Vercel serves the new files only once the deploy is done):
```bash
D=magcoupling-rs/target/release-bl064
curl -sS --compressed -o $D/live.wasm -w '%{http_code} %{size_download}\n' https://colesorkness.com/tools/magcoupler/magcoupling-web_bg.wasm
PASS_TEXT="$(sed -n '/pub const SHAFT_JOINTS_PASS:/,/;/p' magcoupling-rs/src/engine/shaft_joint.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
OLD_NOTE="$(git show fb79095:magcoupling-rs/src/gui/dashboard.rs | sed -n '/^pub const STORED_3D_NOTE:/,/;/p' | grep -o '"[^"]*"' | head -1 | tr -d '"')"
echo "[$PASS_TEXT]"
[ -n "$PASS_TEXT" ] && grep -c -a -F "$PASS_TEXT" $D/live.wasm
[ -n "$OLD_NOTE" ] && grep -c -a -F "$OLD_NOTE" $D/live.wasm
```
Expected: `200 <size>`; the verdict text in brackets, as in Step 5; at least `1`; `0` (before the deploy the live
file gave `0` and `1`; if Step 5 found `main`'s note inside the new one, this second count is `1` and proves
nothing). If it still gives the old counts, wait a minute and fetch again, at most three times; then report to the
owner.

Then invoke the `gui-smoke` skill with the arguments `{"url": "https://colesorkness.com"}` (the same steps against the
live site; `/` redirects to `/tools/` there). Expected: `passed: true` with every calculator field true and no console
error, as in Step 6. A failed field: report the smoke's notes and screenshots to the owner (the site is already live).

- [ ] **Step 11: Delete the local branch; update the memory; report**

1. Run:
```bash
git branch -d magcoupling/bl064-shaft-joint
git branch --show-current
git status --short
```
Expected: `Deleted branch magcoupling/bl064-shaft-joint (was <hash>).`; `main`; only the owner's untracked file. (The
branch was never pushed, so there is no remote branch to delete. Do not touch the other worktrees.)
2. The auto-memory (outside the repo), with the merge's short hash (`git rev-parse --short HEAD`) and the run id of
   Step 10:
   - In `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`,
     replace the whole bullet that starts `- BL-064 started 2026-10-17.` with:
```text
- BL-064 shaft joints SHIPPED YYYY-MM-DD: owner ruling "Hub is shrink fit with key only"; decisions 1A-8A (both joints in one counted row; the fit alone holds SF 2.0 x cold-high at its loosest, yield/1.2 at its tightest, the DIN 3x3 key a backup never added; per-joint bands 13-18 / 15-22 um, placeholders; "cannot be fitted" counted: bare cup shrunk onto the gearbox shaft first then bonded, hub bonded then shrunk onto the LN2-cooled wheel shaft; new note shaft_joint, cap 21) and R-15 with its ring ruling (the bonded hub's fitting cap at the INNER ring's own single-ring onset - 10 C: new result temperature.demag.inner_onset_single_ring_C; the row is a stored-3D row). The defaults pass; "fails 4 of 9" while the clamp section is paused (kept paused, 6A). Spec docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md, plan docs/superpowers/plans/2026-10-18-bl064-shaft-joint.md. The owner reviewed the diff and said OK; merge <short hash> on main, pushed; deploy run <run id> green; live wasm carries the new verdict; live gui-smoke PASS; branch deleted; the gitignored SDD workspace .superpowers/sdd/2026-10-17-bl064-shaft-joint/ is kept (final-review.md, owner-summary.md, the screen-dump tool). Open: the owner's facts (shaft sizes and tolerances, the real bands, the 4140 condition and UltraOx process, the key length and keyway depth, the bondline), BL-069..BL-072 (BL-072 = the approved E24, with E21/BL-057). Next on the roadmap: slice 3 (torque sensitivity card); re-confirm the deploy authorization per slice.
```
   - In `MEMORY.md` there, on the `project_magcoupler_guidance_overhaul.md` line, replace
     `shaft-clamp section PAUSED by owner 2026-10-15 (shrink fit likely)` with
     `shaft-clamp section PAUSED by owner 2026-10-15 (kept paused: BL-064 checks the keyed shrink fits)`, and
     `; next BL-064, slice 3` with `; BL-064 shaft joints shipped YYYY-MM-DD; next slice 3`.
3. Tell the owner, in a few lines: merged and deployed (the merge's short hash), the deploy run's link
   (`https://github.com/coleas33/linkage_sim/actions/runs/<run id>`), the live wasm check and the live smoke result,
   and what stays open: the owner's facts (section 8 of the summary), BL-069 to BL-072, then slice 3.

---

## Deviation index

PD-1 to PD-3 are in the head. The drafters' deviations follow, renumbered from PD-4, each with its drafter label, where it lives and its ruling. "Adopted" means the controller accepted it as drafted (ruling C-2 or the C-ruling named).

- **PD-4** (DEV-1; T2 Steps 8 and 10): the inputs panel's place of spec 5.3 (`SECTION_LABELS` 19, the `shaft` section's two key rows, the sections `shaft.fit` and `shaft.fitting`) and its eight GUI test moves come into Task 2 from Task 4: the 15 inputs fail those tests without a place. Adopted (C-6).
- **PD-5** (DEV-2; T2, T3, T8, T10): `tests/data/input_schema.json` is re-blessed in more places than Task 2: Task 3 for its four flags, and Task 8 and Task 10's fix wave when a `shaft_joint.*` label or help changes. Adopted (C-6; the Global Constraints' rule).
- **PD-6** (DEV-3; T2 Step 10c): "Shaft key width" ends with the paused clamp's "Key width", so the fix-label rule heads the clamp's fix text, and the fix-label test lists eight inputs. Adopted: that heading is never drawn while the section is paused.
- **PD-7** (DEV-4; T2 Step 3): `JointTerms::fails` keeps the spec's NaN-failing `!(a >= b)` form with `#[allow(clippy::neg_cmp_op_on_partial_ord)]` on that method alone. Adopted. Rulings C-3, C-4 and C-5 use the same form, each with the same `allow` on its statement.
- **PD-8** (DEV-5; T1 Step 1): the inner onset's `compute_all` pins sit in `tests/grades.rs` and its unit pins in `temperature.rs`; `tests/shaft_joint.rs` repeats the mixed rings at `passing()`. Adopted.
- **PD-9** (DV-B1; T3 Step 8): Task 3 re-blesses the input schema for its four `.assumption()` flags (exactly four lines). Adopted (C-6).
- **PD-10** (DV-B2; T3 Step 5): `SYMBOLS` gains 18 entries, not 15: the records also read the keyway depth and the steel's expansion and modulus. Adopted.
- **PD-11** (DV-B3; T3 Step 4): the allowed hub temperature's record has four arms instead of `min(min(…), … − 10)` with a NaN arm, because the markup's `min` drops a NaN. Bit for bit the helper's. Adopted.
- **PD-12** (DV-C1; T4): the inputs panel's place is Task 2's; Task 4 pins it in the exports. The three README phrases that placement leaves stale move into Task 2 Step 11 (j) to (l) (A-6). Adopted.
- **PD-13** (DV-C2; T4, T5): the XLSX Summary's joint row and the check's move to the headline group go to Task 5, which first has the `DASHBOARD` row. Adopted (C-6).
- **PD-14** (DV-C3; T4): Task 4 adds no result-group heading: `PACKAGE_LABELS` is Task 2's and `CHAIN_LABELS` Task 3's. Adopted (C-6).
- **PD-15** (DV-C4; T4 Step 10): Auto's measured numbers are pinned, 10.2647 / 16.8990 / 8.3654 N·m at 13.7695 mm, not spec 5.4's 10.27 / 16.90 / 8.37 (section 3.6's row at a rounded 13.77 mm). Adopted; Task 10's smoke paragraph quotes 10.26 (A-13).
- **PD-16** (DV-C5; T4 Step 13): `INERT_INPUTS`' reason for `clamps.key_width_mm` names the clamp drawing, which still reads it, not spec 7.17's draft "nothing reads it". Adopted.
- **PD-17** (DV-C6; T4 Step 13): the `PAUSED` doc is Task 4's. Adopted.
- **PD-18** (DV-C7; T4 Steps 6 and 10): two shared test designs in `test_support.rs`, and the BL-059 Auto test's solve moved into a helper (DRY beyond the spec). Adopted.
- **PD-19** (DV-C8; T9): Task 9's credits follow the moved work: `spreadsheet.rs:1196` is Task 5's, the inputs panel's sections Task 2's, the Summary's 19 lines Task 5's. Adopted (A-13).
- **PD-20** (D-1; T5 Step 1c(7)): the eleven `every_fix_moves_its_check_toward_passing_over_its_slider` cases land in Task 5 with the guide; Task 6 adds the stronger sweep. Adopted (C-6).
- **PD-21** (D-2; T5 Step 1e): the XLSX Summary row of spec 7.15 is Task 5's; the eighteen-row test is replaced whole. Adopted (C-6).
- **PD-22** (D-3; T5 Step 1f): Task 3's three result-groups lines flip in Task 5. Adopted.
- **PD-23** (D-4; T5 Step 1b(5)): the check-name guard's rule gains `name == "check"`, in Task 5's commit (beyond the spec). Adopted (C-8).
- **PD-24** (D-5; T5 Step 4(7), (9), (11)): the shared rounding helpers `apart_from`, `rounded_up`, `rounded_down` and `shown_decimals` serve BL-059's code too (DRY); its tests pass unedited. Adopted (C-8); Task 10's reviewer checks it (A-17).
- **PD-25** (D-6; T5, T7): the interim Why link, `Learn::None("No note covers the shaft joints yet.")`, from Task 5 until Task 7 flips it to `Learn::Note("shaft_joint")`. Adopted.
- **PD-26** (D-7; T5 Step 1b(7)): spec 7.6's stored-3D label assertions are dashboard tests: the last stored-3D line, and a new drawn-once test. Adopted.
- **PD-27** (D-8; T5 Step 1c(11)): the never-order pin covers the check's labels, helps, verdict constants and Problems; the note's half is Task 7's. Adopted.
- **PD-28** (D-9; T6): Task 6 adds no production code; its red is shown by planted faults. Adopted.
- **PD-29** (D-E; T7 Step 3b): the text of the note's change entry for Lower `shaft_joint.hub_interference_max_um` is the plan's draft, since the spec gives only its aim. Adopted; the physics reviewer sets the words.
- **PD-30** (D-E; T8 Step 3): Task 8 re-blesses the input schema when its review rewords a `shaft_joint.*` label or help. Adopted (C-6).
- **PD-31** (D-E; T8): the paused clamp's key-width help (`clamps.key_width_mm`) offered to the G-3 review as a should-fix (the drafts' C5). **Declined** (C-8): owner decision 6A keeps the paused section's texts unchanged, so Task 8 does not offer it.
- **PD-32** (D-E; T7 Step 9): a physics-review CHANGE to the engine's model (R-1, R-2, R-3 or R-14) stops Task 7 for the owner. Adopted (C-8).
- **PD-33** (D-F; T9): BL-056 is not edited. Adopted.
- **PD-34** (D-F; T9): the docs say the clamp section is "kept paused", not paused for good (decision 6A). Adopted.
- **PD-35** (D-F; T9 Step 10): BL-072 stays its own item, tied to the owner-approved E24 and to BL-057. Adopted (C-8).
- **PD-36** (D-F; T9 Step 10): backlog priorities BL-069 3, BL-070 2, BL-071 2, BL-072 2. The owner confirms or re-ranks them in the plan review (C-9).
- **PD-37** (D-F; T9 Steps 2 and 3): additions beyond spec 8: the screen-dump tool and the shipped-text check (`check_shipped.py`). Adopted.
- **PD-38** (D-F; T10 Step 3): Task 10's fix wave re-blesses the input schema when a fix rewords a `shaft_joint.*` label or help. Adopted (C-6).
- **PD-39** (D-F; T9 Step 7): the tracker entry is dated 2026-10-17, the date of the owner's decisions. Adopted.
- **PD-40** (D-F; T10): the release tools and records stay in the gitignored workspace, as BL-059's did. Adopted.
- **PD-41** (D-F; T9 Step 4): line ownership in the stale-text sweep: Task 9 keeps the narrative lines; the 02-system invariant is Task 2's with Task 5's sentence (A-5); the sweep's owners follow A-4, A-6 and A-13. Adopted.
- **PD-42** (D-F; T10 Step 8): the leftovers' ledger grep (`minor (deferred)` in `progress.md`) follows BL-059's progress-ledger convention, the subagent-driven-development skill's. Adopted: the build keeps that ledger (Global Constraints, Build ledger), and Step 8 reads it at `.superpowers/sdd/2026-10-18-bl064-shaft-joint/progress.md`, the workspace the skill derives from this plan's file name, and stops if it is missing.
- **PD-43** (D-F; T9 Step 3): the shipped-text substitution rule (BL-059 ruling B-1): where a quoted text differs from what ships, the shipped text is written and the commit body says so. Adopted.

## Per-task test counts

Measured counts first: Tasks 1 to 4 on the drafters' prototypes, Tasks 5 to 7 on the critic's integrated tree (A-12). Tasks 8 and 9 follow from those. "App lib" is `cargo test --features app --lib`; "lib" is the same without `--features app`. Every row also has: app lib 6 ignored, lib 1 ignored, `explain` 1 ignored, doc-tests 1 passed and 1 ignored.

| After | App lib | Lib | `explain` with app | `explain` without app | Other binaries that move |
|---|---|---|---|---|---|
| HEAD 0687034 | 886 | 229 | 58 | 57 | grades 11, material_links 12, robustness 12, assumptions 11, sizing 44 |
| T1 | 887 | 230 | 58 | 57 | grades 12, material_links 14 |
| T2 | 888 | 230 | 58 | 57 | shaft_joint 30 (new), robustness 13 |
| T3 | 889 | 231 | 61 | 60 | assumptions 12 |
| T4 | 892 | 231 | 61 | 60 | sizing 45 |
| T5 | 905 | 231 | 61 | 60 | none |
| T6 | 917 | 231 | 61 | 60 | none |
| T7 | 921 | 232 | 63 | 62 | none |
| T8 | 921, plus any assertion test a must-fix adds | 232, plus any | 63, plus any | 62, plus any | none |
| T9 | as after T8 (docs only) | as after T8 | as after T8 | as after T8 | none |
| T10 | as after T9, plus a test per fix-wave item | as after T9, plus any | as after T9, plus any | as after T9, plus any | none |

These never move: deviations 54, differential 19, parity 4, python_schema 7 (gate 12's parity untouched), gearbox 13, schema 11, material_library 4, size_controls 38, static_data 8; the two app binaries run 0 tests. The amendments folded in add assertions to existing tests only, so no count above moves for them.

## Amendment ledger

- **A-1** (C-1): T7 Step 1c (no import edit); T7's Files list (no `temperature` import); T7 Step 1e (the narrower `hub_fit_allowed` import).
- **A-2** (C-1): T5 Step 6a(27), whose Old text anchors on Task 4's clause.
- **A-3** (C-1): T3 Step 10, README edit (i).
- **A-4** (C-1): T2 Step 11, README edit (i); T9 Step 4's owner table (T2, Step 11i).
- **A-5** (C-2): T5 Step 6b(8); the `02-system.yaml` sentence of T2 Step 11; T9 Step 11 (its invariant item deleted), Step 12 (the marker) and Step 13 (T2's invariant).
- **A-6** (C-2): T2 Step 11, README edits (j) to (l); T4 Step 14 (seven README edits, relabelled (a) to (g)); T9 Step 4's owner table.
- **A-7** (C-2): T2 Step 11, README edit (m), which also names ruling C-4's pins.
- **A-8** (C-3, PD-1): T2 Step 3 (`joint`'s no-wall test with its `allow`, the `NoWall` doc, the notes); T2 Step 1 (`a_joint_with_no_wall_reads_nan_and_is_not_judged`, the boss typed at -22 mm); T2 Step 11 (README and `03-structure.yaml`); T3 Step 4 (the eight Lamé records' no-wall arm, the module doc, the transcription note); T9 Step 7 (the tracker); T10 Step 2, reviewer item (2).
- **A-9** (C-4, PD-2): T2 Step 3 (`UNKNOWN_SAFETY_FACTOR`, `verdict` and its `allow`, the module doc, the notes); T2 Step 1 (`the_verdict_texts_are_the_spec_s`; `degenerate_shaft_joint_inputs_never_pass` pins 0 and -1); T2 Step 2 (the robustness error count); T2 Step 11 (docs); T5 Steps 1b(3), 1c(2), 1c(6), 1c(11), 3(7), 4(5) and 4(10) (the Caution arm by prefix, the Problem sentence, no fix); T5 Step 2 (the error counts); T5 Step 6a(15) and (35); T8 (strings rows 3 and 7, Step 1's grep, the brief's typed values, strings and probe (3), the record); T9 Step 7 (the tracker); T10 Step 2, reviewer item (2).
- **A-10** (C-2): T5 Step 1c(11), the end of `the_shaft_joint_problem_quotes_the_live_numbers`.
- **A-11** (C-5, PD-3): T4 Step 4 (the local guard and its `allow`); T4 Step 2 (the keyway-0 check in both states); T4's head, Files and Interfaces; T4 Step 14 (README (f), `03-structure.yaml` (a)); T4 Step 16 (the message); T9 Step 7 (the tracker).
- **A-12** (C-2): T5 Steps 5 and 7, T6 Step 5, T7 Step 12; the Per-task test counts above.
- **A-13** (C-2): T2 Step 9 (Task 3's `CHAIN_LABELS`); T9 Step 4 (`spreadsheet.rs:1196` to T5), Step 7 (the tracker's panel bullet), Step 8 (BL-064's notes); T10's gui-smoke paragraph (10.26 and 16.90 N·m).
- **A-14** (C-6): the head's C-6 and Global Constraints; T3's PD-9, T8 Step 3 and T10 Step 3 carry the re-bless rule; the Deviation index.
- **A-15** (C-7): the head (its Open owner questions, and its pointers to the sections below); the Deviation index; the Per-task test counts.
- **A-16** (C-2): T7 Step 11, the expected heading lines.
- **A-17** (C-2): T10 Step 2, reviewer item (9).
- **A-18** (C-2): T8 Step 2, the brief's probe (4).
- **C-1**: A-1 to A-4, as above.
- **C-2**: A-5, A-6, A-7, A-10, A-12, A-13, A-16, A-17 and A-18, as above.
- **C-3**: A-8, as above; PD-1.
- **C-4**: A-9, as above; PD-2.
- **C-5**: A-11, as above; PD-3.
- **C-6**: T2 (the inputs panel's place, `PACKAGE_LABELS`); T3 (`CHAIN_LABELS`, the four-flag re-bless); T4's scope (its head, Files and Steps 7 to 14); T5 (the XLSX Summary row, the eleven slider cases); the head's re-bless rule; PD-4, PD-5, PD-9, PD-13, PD-14, PD-20, PD-21, PD-30, PD-38.
- **C-7**: the head.
- **C-8**: T5 Step 1b(5) (`name == "check"`, PD-23); T5 Steps 4(7), 4(9), 4(11) (the shared helpers, PD-24) and T10 Step 2 item (9); T5 Step 1d(3) (the guard's designs and the label and Why assertion); T5 Step 1f (the result-groups swap) and Step 6a(11) ("dashboard's 19 rows"); T2 Step 1 (`passing()`); T1 Step 5 and T9 Step 7 (the E18 observation); T8 (C5 removed from the Interfaces, the context list, Step 1's greps and expectations, the brief, Step 3 and the record; PD-31); T7 Step 9 (stop on an engine-model CHANGE, PD-32); T9 Step 10 (BL-072, PD-35).
- **C-9**: T9 Step 10 (the priorities note) and Step 12 (its assertion); T10 Step 8, the leftovers' section 8; PD-36.
