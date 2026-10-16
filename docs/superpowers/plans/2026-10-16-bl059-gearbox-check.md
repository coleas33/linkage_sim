# BL-059 gearbox check: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dashboard shows whether the coupling's cold-high torque is within the gearbox input rating. A counted check with a guide says by how much it is over and what to change. When no magnet size, gap or count can pass both this check and the hot minimum, the guide says so with numbers and names what would have to change instead.

**Architecture:**
- **Engine:** a new Rust-only result group `gearbox` (`magcoupling-rs/src/engine/gearbox.rs`) holds one text verdict, `gearbox.cold_high_check`. `api.rs` computes it beside the space claim from two values that already exist: `metal.cold_high_input_Nm` (cold-high ÷ ratio ÷ efficiency) and `coupling.gearbox_input_rating_Nm`. No Python-locked row changes, so gate 12 parity is unaffected.
- **GUI:**
  - one counted dashboard row and check (`DASHBOARD` index 15, `CHECKS`, greyed with the end-effect rows);
  - a slice-1 guide with a live-number Problem, the conflict sentence, a one-sentence Why, two geometry fixes, and a Why link to the torque-band note;
  - two overlay rows: the rating's help, and the ripple row's name and help.
- **Notes:** the two watch-outs that said nothing checks the gearbox are rewritten, with a fresh-context physics re-review.

**Tech Stack:** Rust 1.89; egui 0.32; the headless egui harness in `magcoupling-rs/src/gui/test_support.rs`; `linkage-sim-rs/scripts/gate.sh`, where gate 12 runs the Python oracle from `reference/magcoupling-py/.venv`.

**Spec:** `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`: owner decisions 1A-8A and R, approved by the owner 2026-10-16 at commit 3e63bec. The research, the plan contract and the drafts are in `.superpowers/sdd/2026-10-16-bl059-gearbox-check/` (gitignored).

**How this plan was made:**
1. The controller wrote a binding contract holding the fixed production code of T1 and T2.
2. Six session-model drafters wrote T1-T10 against it. Each prototyped its tasks red to green in a private copy of the repo.
3. A consistency and physics critic checked every task against the code, the contract and the spec, and prototyped every fix it proposed.
4. The controller ruled on the critique (the rulings below). The amendments are folded into the task text, so the tasks are what to execute.

Where a line number has moved because of an earlier task, trust the code and keep the intent.

## Global Constraints

- **Implementer model:** BL-059 is `risk: physics`, so every task runs on the session model (CLAUDE.md section 5: the session model wins on precedence). Reviews also run on the session model, in a fresh context.
- **Commit trailers**, exactly:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```
- **Commit messages** are written with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T<n>.txt` (or `T6-fix.txt`, `T8-fix.txt`, `T8-record.txt`, `T10-<a|b|merge|mf-n>.txt`), then `git commit -F <file>`. Never pipe a heredoc.
- **Staging:** files by name only. Never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`.
- **TDD:** write the failing test, see it fail for the expected reason, implement, see it pass.
- **Before each commit**, run all of these:
  - the touched suites;
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`;
  - the same without `--features app`;
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`.
- **Docs ship with the code, in the same task.** Each task names the README and `docs/ai` lines it owns.
- **Never edit:**
  - Python-locked input or result labels, help, cells and defaults (`tests/python_schema.rs`);
  - `reference/`;
  - `tests/data/*.json`;
  - `input_schema.json`;
  - the committed note sign-off sheet `docs/analyses/2026-10-14-magcoupler-notes-signoff.md` (owner decision 14A: no new sheet; the owner reviews the diff).
- **Gate 1** rewrites `docs/chebyshev_lambda/*.png`. Restore them with `git checkout -- docs/chebyshev_lambda` and never commit them.
- **The shaft-clamp section is paused** (`engine::clamps::PAUSED = true`). No on-screen text says "clamp".
- **On-screen text:**
  - plain words for engineers who are not magnet specialists;
  - no spreadsheet cells, ids, paths, "BL-0xx" or "Rust-only";
  - numbers come live from the results, in the readout format (4 significant digits);
  - glyph-safe (`->`, `<=`);
  - `plain_text_violations` is empty.
- **Caps:**

  | Text | Limit |
  |---|---|
  | Watch-outs | at most 30 words |
  | Help | at most 30 words, 1-2 sentences |
  | Why | one sentence, at most 30 words |
  | Overlay label | at most 47 characters |
- **Physics traps** (spec section 6). Never write any of these:
  - a torque without naming its side of the gearbox;
  - that efficiency "reduces the load" (it raises the input torque);
  - cold-high as a bound: write "up to" or "can reach", never "at most", and never as what every unit reaches;
  - that the coupling "protects the gearbox";
  - "peak" for the rating (its type is open, owner decision 2A); "peak" names only the compared torque;
  - "warm" for the ripple row: it is the nominal swing at the operating temperature;
  - that geometry narrows the band;
  - "hot-day" in these texts.
- **Release:** STOP for the owner's review of the diff before any merge or push. A push to `main` deploys.

## Review Focus

These five inputs or conditions are the most likely to bite a user; none is a main line of the spec. Each has a test in the task named.
1. **A rating at a rounding boundary or a slider end** (0.604 against 0.6041; 0.05; 5.0). A failing badge never quotes two equal numbers, and the ends give the expected verdict. T1.
2. **The gearbox ratio at 1 or the efficiency at 0.5** (their slider bottoms). Meeting the requirement is then over the rating even with a band of one, so sentence 2 offers no band (plan deviation PD-1). T3.
3. **The operating temperature below the minimum temperature** (both sliders reach it). Cold-high is then not the strongest pull-out:
   - sentence 2 is left out;
   - the check follows cold-high, as the spec says;
   - the ripple help claims no inequality;
   - the under-check is filed as BL-067.

   T3, T5, T9.
4. **The shaft-clamp section unpaused.** The status counts 9 checks ("fails 4 of 9" at the defaults). T2.
5. **Each check at the other's limit:**
   - at the length Auto solves, both gearbox fixes are held back, the gearbox block draws the no-fix line and its Problem carries the conflict sentence;
   - near the gearbox limit, the hot minimum's length fix is held back.

   T4.

## Controller rulings on the critique (binding; already folded into the tasks)

- **R-1 (A-1). One commit for T1 and T2.** T1 makes no commit. `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` fails until T2's `CHECKS` entry exists, and splitting `CHECKS` into T1 would leave a commit where the badge reads green at f_end ≤ 0. T1 ends with exactly that one failure; T2's commit stages T1's files.
- **R-2.** The rest are adopted as written: A-2 to A-4, A-6, A-7, A-9, A-10, A-13, A-14, A-17 to A-20.
- **R-3 (A-5), plan deviation PD-1: Problem sentence 2 is true wherever it shows.**
  - It shows only with the minimum temperature at or below the operating temperature.
  - Where the requirement alone is above the rating at the gearbox input (the band cannot go below one), it offers the gearbox or the requirement and no band.

  The owner confirms this in the plan review.
- **R-4 (A-15, A-16): the corner with the operating temperature below the minimum.** The engine is unchanged (spec 7A and 5A). The gearbox module doc is scoped to temperatures in order. A new BL-067 (`risk: physics`, priority 3) records the corner. The owner confirms this in the plan review.
- **R-5 (A-8).** The roadmap after BL-059 is BL-064, then slice 3. BL-065 is not placed without the owner.
- **R-6 (A-11, A-12). More tests are adopted:**
  - a test that holds the README and `03-structure.yaml` counts of dashboard rows and checks to the code;
  - the mirror of the near-limit case;
  - a sampled monotonicity test of both gearbox fixes.
- **R-7. The ripple help is T5's wording,** not the contract draft. The draft's "reads below cold-high at the gearbox input" is false with the operating temperature below the minimum.
- **R-8. The path `"gearbox.cold_high_check"` stays a literal in `DASHBOARD`, `CHECKS` and `verdict_level`,** as every other check's path is. The existing tests catch a typo: every check is listed and classified, and the guides cover every dashboard check. This is raised in the plan review (owner preference: DRY).
- **R-9. The notes:**
  - one physics reviewer for both notes;
  - each note's review date is the day its passing round ran (A-9);
  - the tests pin the reviewed words through constants;
  - the committed sign-off sheet is untouched.
- **R-10. Release and backlog:**
  - BL-066 priority 2;
  - the session-model physics reviews stand in for AGENTS.md's fbd-math-reviewer pass, which is specific to linkage FBDs; slice 2 did the same;
  - the owner gets the full diff and a diff without the tests (A-19);
  - the pre-existing YAML parse error in `docs/ai/02-system.yaml` is not BL-059's: it is reported to the owner, not fixed here.
- **R-11. The Python-locked rating help stays.** It shows as "Original spreadsheet note: GAM 5:1 peak input rating." only with Show spreadsheet details on. It is labelled as the spreadsheet's own note, and the T8 reviewer is told.
- **R-12.** When a NaN is set on the struct (unreachable from the UI), the gearbox guide still lists its static fixes. This is left as is.
- **R-13.** Between T2's commit and T6's, the two notes on the branch still say nothing checks the gearbox. Only the merge ships, and T2's commit body says so.

## Plan deviations from the spec

- **PD-1 (R-3).** Spec section 5's conflict predicate gains `metal.min_temp_C <= coupling.op_temp_C`. Sentence 2 gains a second remedy form: "It takes a gearbox rated for that or a requirement of at most {X}: the requirement alone puts {Y} on the gearbox input, so no narrower band settles it." That form is used when rating·ratio·efficiency / requirement < 1.
- **PD-2 (R-7).** The ripple help says what the row is and that the check compares cold-high instead. It claims no inequality with cold-high.
- **PD-3 (R-4).** New backlog item BL-067, beside BL-066.
- **PD-4 (R-1).** T1 and T2 land as one commit.

---

## Tasks

### Task 1: Engine: the Rust-only `gearbox` result group

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

Spec sections 4 and 7.1-7.2 (owner decision 1A: the input basis). The production code in Steps 4-6 and 10 is the controller's fixed code, with two adopted changes in `gearbox.rs`: the lint line `#[allow(non_snake_case)]` above `compute`, and the module doc's second sentence scoped to the temperatures in order (BL-067, ruling R-4).

**Files:**
- Create: `magcoupling-rs/src/engine/gearbox.rs`
- Create: `magcoupling-rs/tests/gearbox.rs`
- Modify: `magcoupling-rs/src/engine/mod.rs`, `magcoupling-rs/src/engine/api.rs`, `magcoupling-rs/src/gui/result_groups.rs`, `magcoupling-rs/src/gui/trace.rs`, `magcoupling-rs/tests/schema.rs`
- Modify (docs): `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`, `magcoupling-rs/src/lib.rs` (the crate doc)
- Test: `magcoupling-rs/tests/gearbox.rs` (new, 11 tests); `magcoupling-rs/tests/schema.rs` (`engine_owned_help_is_plain_and_short`, its coverage list extended); `magcoupling-rs/src/gui/trace.rs` (`an_input_no_explained_result_reads_traces_none_and_says_so`, pinned total moved); `magcoupling-rs/src/gui/result_groups.rs` (`every_chain_path_names_results_and_every_package_has_a_heading`, unchanged: the red for `PACKAGE_LABELS`)

**Interfaces:**
- Consumes: `metal.cold_high_input_Nm` (`MetalDesignResults`, Metal design C158: cold-high divided by ratio × efficiency, `metal_design.rs:509`); `coupling.gearbox_input_rating_Nm` (`CouplingInputs`, default 0.6 N·m, slider 0.05 to 5.0, step 0.01, `model.rs:196-198`); `engine::compat::{fmt_fixed, py_max}`; `engine::meta::{out_rust_only, results}`.
- Produces: `magcoupling::engine::gearbox::{GearboxResults, compute, WITHIN_THE_GEARBOX_RATING, ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN}`; the group `DesignResults::gearbox` (schema order: right after the `housing` group, before the `gap_sweep` table); the result `gearbox.cold_high_check` (Rust-only text, no cell, unit ""); `PACKAGE_LABELS` entry `("gearbox", "Gearbox")`. Task 2 reads the three constants and the path (`CHECKS`, `verdict_level`, `DASHBOARD`, `END_EFFECT_ROWS`, the guide).

**Commit boundary (plan deviation PD-4, ruling R-1).** This task cannot end green on its own. Once the group exists, `gui::dashboard::tests::every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` (`dashboard.rs:1317-1358`) fails with "gearbox.cold_high_check is not in CHECKS". That test requires every text result named `*_check` to be listed in `CHECKS` and classified, and that is Task 2's code. Task 1 therefore ends with the engine suite green and that one expected failure in the app suite. It makes no commit: Task 2's commit stages this task's files (Step 18).

- [ ] **Step 1: Write the failing engine tests**

Create `magcoupling-rs/tests/gearbox.rs`. It follows the space claim's precedent: a Rust-only engine result is tested end to end through `compute_all` in an integration file (`tests/sizing.rs:884-1070`; `housing.rs` has no unit-test module). The probe designs come from `probe_designs()`; the test pins each one's numbers, so a change to a probe fails it.

| design | cold-high at the output | at the gearbox input | verdict at 0.6 |
|---|---|---|---|
| `probe_designs()` "default" | 3.8233 | 0.8049 | "… 0.80 N·m against 0.60 N·m" |
| "8 magnets per ring" (`coupling.npole = 8`) | 2.7992 | 0.5893 | within |
| "304 back iron" (back iron `304_annealed`) | 2.4138 | 0.5082 | within |
| "Y30 ferrite, custom dimensions" (both rings custom dimensions, parts cleared, grade `Y30`) | 0.3381 | 0.0712 | within |
| "20 mm ring radius" (`with_ring_radius(&default, 20.0)`) | 3.6111 | 0.7602 | "… 0.76 N·m against 0.60 N·m" |
| built in the test: both rings custom dimensions, grade `N33AH` | 2.9401 (≤ 0.6 × 5 = 3.0) | 0.6190 | "… 0.62 N·m against 0.60 N·m" |

(Full-precision values from the prototype: 3.823310370488638 / 0.8049074464186606; 2.7992252237206623 / 0.5893105734148762; 2.4138467131384425 / 0.5081782553975668; 0.33806699474817237 / 0.07117199889435208; 3.611105809983419 / 0.7602328021017724; 2.940103051767475 / 0.6189690635299947.)

```rust
//! BL-059: the gearbox check (`gearbox.cold_high_check`, `src/engine/gearbox.rs`), end to end
//! through `compute_all` and on `gearbox::compute` itself: cold-high at the gearbox input
//! (`metal.cold_high_input_Nm`) against the gearbox input torque rating
//! (`coupling.gearbox_input_rating_Nm`), owner decision 1A. A Rust-only result, tested here as
//! the space claim is in `tests/sizing.rs`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::explain::probes::probe_designs;
use magcoupling::engine::gearbox::{
    self, ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN, WITHIN_THE_GEARBOX_RATING,
};
use magcoupling::engine::meta::{SetErrorKind, SliderRange, Value, input_rows, result_rows};

/// The failing verdict quoting a cold-high at the gearbox input and a rating, as the badge
/// quotes them (to 0.01 N·m).
fn above(cold_high_input: &str, rating: &str) -> String {
    format!("{ABOVE_THE_GEARBOX_RATING}: {cold_high_input} N·m against {rating} N·m")
}

/// The two numbers a failing verdict quotes: (cold-high at the gearbox input, the rating).
fn quoted(text: &str) -> (f64, f64) {
    let rest = text
        .strip_prefix(ABOVE_THE_GEARBOX_RATING)
        .and_then(|t| t.strip_prefix(": "))
        .unwrap_or_else(|| panic!("not a failing verdict: {text}"));
    let (torque, rating) = rest
        .split_once(" N·m against ")
        .unwrap_or_else(|| panic!("{text}"));
    let rating = rating
        .strip_suffix(" N·m")
        .unwrap_or_else(|| panic!("{text}"));
    (
        torque.parse().unwrap_or_else(|_| panic!("{text}")),
        rating.parse().unwrap_or_else(|_| panic!("{text}")),
    )
}

/// The default design with `edit` applied.
fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    edit(&mut inputs);
    inputs
}

/// The default design with both rings on their custom (manual) dimensions, `grade` on both.
fn custom_dimensions(grade: &str) -> DesignInputs {
    design(|i| {
        i.coupling.magnets.part_inner.clear();
        i.coupling.magnets.part_outer.clear();
        i.coupling.magnets.grade_inner = grade.to_owned();
        i.coupling.magnets.grade_outer = grade.to_owned();
    })
}

/// The rating's slider.
fn rating_range() -> SliderRange {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|row| row.path == "coupling.gearbox_input_rating_Nm")
        .and_then(|row| row.meta.range)
        .expect("the gearbox input rating has a slider")
}

/// The check's verdict with the rating at `rating` [N·m], the rest the defaults.
fn verdict_at_rating(rating: f64) -> String {
    compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = rating))
        .gearbox
        .cold_high_check
}

#[test]
fn the_defaults_fail_with_both_torques_quoted() {
    // Spec section 3: cold-high 3.8233 N·m at the gearbox output, 0.8049 N·m at its input
    // (÷ 5 ÷ 0.95), against the 0.6 N·m input rating.
    let inputs = DesignInputs::default();
    let r = compute_all(&inputs);
    let c = &inputs.coupling;
    assert_eq!(
        (c.gearbox_input_rating_Nm, c.gear_ratio, c.gear_efficiency),
        (0.6, 5.0, 0.95)
    );
    assert!(
        (r.metal.torque_cold_high_Nm - 3.8233).abs() < 5e-5,
        "{}",
        r.metal.torque_cold_high_Nm
    );
    assert!(
        (r.metal.cold_high_input_Nm - 0.8049).abs() < 5e-5,
        "{}",
        r.metal.cold_high_input_Nm
    );
    assert_eq!(r.gearbox.cold_high_check, above("0.80", "0.60"));
}

#[test]
fn a_rating_one_step_under_cold_high_quotes_the_torque_a_hundredth_above_it() {
    // At 0.80 N·m, one slider step, the rating sits under cold-high at the input (0.8049 N·m),
    // which rounds to 0.80 too: the badge quotes it at 0.81, never "0.80 against 0.80".
    let range = rating_range();
    let steps = (0.80 - range.min) / range.step;
    assert!(
        (steps - steps.round()).abs() < 1e-9,
        "0.80 is on the slider"
    );
    let x = compute_all(&DesignInputs::default())
        .metal
        .cold_high_input_Nm;
    assert!(x > 0.80 && x < 0.805, "{x}");
    assert_eq!(verdict_at_rating(0.80), above("0.81", "0.80"));
}

#[test]
fn each_probe_design_passes_exactly_when_cold_high_at_the_input_is_within_the_rating() {
    // The five probe designs of the notes (spec section 3), in `probe_designs()`' order: (name,
    // cold-high at the gearbox output and at its input [N·m], the verdict at the 0.6 N·m
    // rating). A change to a probe design moves its numbers here.
    let within = WITHIN_THE_GEARBOX_RATING.to_owned();
    let want: [(&str, f64, f64, String); 5] = [
        ("default", 3.8233, 0.8049, above("0.80", "0.60")),
        ("8 magnets per ring", 2.7992, 0.5893, within.clone()),
        ("304 back iron", 2.4138, 0.5082, within.clone()),
        ("Y30 ferrite, custom dimensions", 0.3381, 0.0712, within),
        ("20 mm ring radius", 3.6111, 0.7602, above("0.76", "0.60")),
    ];
    let designs = probe_designs();
    assert_eq!(designs.len(), want.len());
    for ((name, inputs), (want_name, cold_high, at_input, verdict)) in designs.into_iter().zip(want)
    {
        assert_eq!(name, want_name);
        let r = compute_all(&inputs);
        let c = &inputs.coupling;
        assert_eq!(c.gearbox_input_rating_Nm, 0.6, "{name}");
        assert!(
            (r.metal.torque_cold_high_Nm - cold_high).abs() < 5e-5,
            "{name}: {}",
            r.metal.torque_cold_high_Nm
        );
        assert!(
            (r.metal.cold_high_input_Nm - at_input).abs() < 5e-5,
            "{name}: {}",
            r.metal.cold_high_input_Nm
        );
        assert_eq!(r.gearbox.cold_high_check, verdict, "{name}");
        // The input basis is the workbook's reference at the output (rating × ratio ×
        // efficiency, 2.85 N·m): away from equality the two agree.
        assert_eq!(
            r.model.gearbox_reference_Nm,
            c.gearbox_input_rating_Nm * c.gear_ratio * c.gear_efficiency,
            "{name}"
        );
        assert_eq!(
            r.gearbox.cold_high_check == WITHIN_THE_GEARBOX_RATING,
            r.metal.torque_cold_high_Nm <= r.model.gearbox_reference_Nm,
            "{name}"
        );
    }
}

#[test]
fn n33ah_fails_the_input_basis_that_an_output_basis_without_the_efficiency_passes() {
    // Owner decision 1A: cold-high at the input against the input rating is cold-high against
    // rating × ratio × efficiency = 2.85 N·m at the output. Without the efficiency the output
    // limit would be 0.6 × 5 = 3.0 N·m, which this design (2.9401 N·m) would pass.
    let inputs = custom_dimensions("N33AH");
    let r = compute_all(&inputs);
    let c = &inputs.coupling;
    assert!(
        (r.metal.torque_cold_high_Nm - 2.9401).abs() < 5e-5,
        "{}",
        r.metal.torque_cold_high_Nm
    );
    assert!(
        (r.metal.cold_high_input_Nm - 0.6190).abs() < 5e-5,
        "{}",
        r.metal.cold_high_input_Nm
    );
    assert_eq!(r.gearbox.cold_high_check, above("0.62", "0.60"));
    assert!(r.metal.torque_cold_high_Nm <= c.gearbox_input_rating_Nm * c.gear_ratio);
    assert!(r.metal.torque_cold_high_Nm > r.model.gearbox_reference_Nm);
}

#[test]
fn exactly_at_the_rating_passes_and_one_float_above_fails() {
    // Equality passes, as the hot minimum check passes at the requirement.
    let rating: f64 = 0.6;
    assert_eq!(
        gearbox::compute(rating, rating).cold_high_check,
        WITHIN_THE_GEARBOX_RATING
    );
    assert_eq!(
        gearbox::compute(rating.next_up(), rating).cold_high_check,
        above("0.61", "0.60")
    );
    // Through the engine (the rating moves no torque): the rating at the defaults' cold-high
    // at the input passes; one float below it, so cold-high is the rating's next float up, fails.
    let x = compute_all(&DesignInputs::default())
        .metal
        .cold_high_input_Nm;
    let r = compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = x));
    assert_eq!(r.metal.cold_high_input_Nm, x);
    assert_eq!(r.gearbox.cold_high_check, WITHIN_THE_GEARBOX_RATING);
    let below = x.next_down();
    assert_eq!(below.next_up(), x);
    let r = compute_all(&design(|i| i.coupling.gearbox_input_rating_Nm = below));
    assert_eq!(r.metal.cold_high_input_Nm, x);
    assert_eq!(r.gearbox.cold_high_check, above("0.81", "0.80"));
}

#[test]
fn a_torque_that_is_not_a_number_reads_unknown() {
    for (cold_high_input, rating) in [
        (f64::NAN, 0.6),
        (0.5, f64::NAN),
        (f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NAN),
    ] {
        assert_eq!(
            gearbox::compute(cold_high_input, rating).cold_high_check,
            GEARBOX_CHECK_UNKNOWN,
            "{cold_high_input} against {rating}"
        );
    }
    // Through the engine: set() refuses NaN, but a struct, a design file or a share link can
    // hold one. A NaN rating, or a NaN ratio (cold-high at the input is then NaN), reads
    // unknown, and validate() names the input.
    type Put = fn(&mut DesignInputs);
    let cases: [(&str, Put); 2] = [
        ("coupling.gearbox_input_rating_Nm", |i| {
            i.coupling.gearbox_input_rating_Nm = f64::NAN
        }),
        ("coupling.gear_ratio", |i| i.coupling.gear_ratio = f64::NAN),
    ];
    for (path, put) in cases {
        let mut inputs = DesignInputs::default();
        put(&mut inputs);
        assert_eq!(
            compute_all(&inputs).gearbox.cold_high_check,
            GEARBOX_CHECK_UNKNOWN,
            "{path}"
        );
        let errors = inputs.validate().expect_err(path);
        assert_eq!(errors.len(), 1, "{path}: {errors:?}");
        assert_eq!(
            (errors[0].path.as_str(), &errors[0].kind),
            (path, &SetErrorKind::NotFinite)
        );
    }
}

#[test]
fn an_infinite_cold_high_at_the_gearbox_input_fails() {
    assert_eq!(
        gearbox::compute(f64::INFINITY, 0.6).cold_high_check,
        above("inf", "0.60")
    );
    // Through the engine: an efficiency of 0 set on the struct (its slider starts at 0.5)
    // divides cold-high by 0 at the input.
    let r = compute_all(&design(|i| i.coupling.gear_efficiency = 0.0));
    assert_eq!(r.metal.cold_high_input_Nm, f64::INFINITY);
    assert_eq!(r.gearbox.cold_high_check, above("inf", "0.60"));
}

#[test]
fn a_failing_check_never_quotes_two_equal_numbers() {
    // A rating of 0.604 quotes as 0.60, and cold-high at the input 0.6041 rounds to 0.60 too:
    // the badge quotes the torque 0.01 N·m above the rating as quoted.
    assert_eq!(
        gearbox::compute(0.6041, 0.604).cold_high_check,
        above("0.61", "0.60")
    );
    // Over the whole slider: at each step, just above it (quoted as the step) and half-way to
    // the next, a cold-high one float above the rating quotes above the rating.
    let range = rating_range();
    let steps = ((range.max - range.min) / range.step).round() as u32;
    for k in 0..=steps {
        let step = range.min + f64::from(k) * range.step;
        for rating in [step, step + 0.004, step + 0.005] {
            let text = gearbox::compute(rating.next_up(), rating).cold_high_check;
            let (torque, quoted_rating) = quoted(&text);
            assert!(torque > quoted_rating, "{rating}: {text}");
        }
    }
}

#[test]
fn the_rating_slider_s_ends_fail_and_pass_the_defaults() {
    let range = rating_range();
    assert_eq!((range.min, range.max, range.step), (0.05, 5.0, 0.01));
    assert_eq!(verdict_at_rating(range.min), above("0.80", "0.05"));
    assert_eq!(verdict_at_rating(range.max), WITHIN_THE_GEARBOX_RATING);
}

#[test]
fn the_rating_moves_only_the_gearbox_check_and_the_workbook_s_reference() {
    // The rating's two readers: this check and the workbook's reference (rating × ratio ×
    // efficiency). Every other result stays bit for bit (NaN equal to NaN).
    let bits = |inputs: &DesignInputs| -> Vec<(String, String)> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .map(|row| {
                let bits = match row.value {
                    Value::Num(x) => format!("{:016x}", x.to_bits()),
                    other => format!("{other:?}"),
                };
                (row.path, bits)
            })
            .collect()
    };
    let base = bits(&DesignInputs::default());
    let raised = bits(&design(|i| i.coupling.gearbox_input_rating_Nm = 1.0));
    assert_eq!(base.len(), raised.len());
    let moved: Vec<&str> = base
        .iter()
        .zip(&raised)
        .filter(|(a, b)| a != b)
        .map(|(a, _)| a.0.as_str())
        .collect();
    assert_eq!(
        moved,
        ["model.gearbox_reference_Nm", "gearbox.cold_high_check"]
    );
}

#[test]
fn the_check_is_a_rust_only_text_result_with_no_cell_after_the_housing_group() {
    let rows = result_rows(&compute_all(&DesignInputs::default()));
    let gearbox: Vec<&str> = rows
        .iter()
        .filter(|row| row.path.starts_with("gearbox."))
        .map(|row| row.path.as_str())
        .collect();
    assert_eq!(gearbox, ["gearbox.cold_high_check"]);
    let at = rows
        .iter()
        .position(|row| row.path == "gearbox.cold_high_check")
        .expect("the check");
    let row = &rows[at];
    assert!(row.meta.rust_only);
    assert_eq!((row.meta.cell, row.meta.unit), (None, ""));
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
    );
}
```

- [ ] **Step 2: Name the new result in the schema test's coverage list**

In `magcoupling-rs/tests/schema.rs`, `engine_owned_help_is_plain_and_short` (around :534), the list that proves the test is not vacuous gains the new Rust-only result. Its label and help then pass `every_result_label_and_help_passes_the_plain_words_guard` and this test's 30-word, two-sentence cap with no further edit (the help is two sentences, under 30 words; both tests passed in the prototype).

Old:
```rust
        "coupling.max_harmonic",
        "housing.space_claim_check",
        "metal.steel_cup_mass_g",
```

New:
```rust
        "coupling.max_harmonic",
        "housing.space_claim_check",
        "gearbox.cold_high_check",
        "metal.steel_cup_mass_g",
```

- [ ] **Step 3: Run the new tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test gearbox`

Expected: the test file does not compile (the module does not exist yet):
```text
error[E0432]: unresolved imports `magcoupling::engine::gearbox`, `magcoupling::engine::gearbox`
error[E0609]: no field `gearbox` on type `DesignResults`      (9 of these)
error: could not compile `magcoupling-rs` (test "gearbox") due to 10 previous errors
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema engine_owned_help_is_plain_and_short`

Expected:
```text
test engine_owned_help_is_plain_and_short ... FAILED
gearbox.cold_high_check is not engine-owned
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 10 filtered out
```

- [ ] **Step 4: Create the engine module**

Create `magcoupling-rs/src/engine/gearbox.rs` exactly as below. Two lines differ from the controller's first draft of this file:
- the line `#[allow(non_snake_case)] // unit suffixes, as the engine's names` above `compute`. Without it, both clippy runs fail with "variable `cold_high_input_Nm` should have a snake case name" and the same error for `rating_Nm`. The line copies `grades.rs:79` and `material_library.rs:53`; `metal_design.rs:380` does the same with "Python names and signature";
- the module doc's second sentence is scoped to the minimum magnet temperature at or below the operating one. With the operating temperature below the minimum (both sliders reach it: op -40 °C, min 20 °C, v 0), hot-low is 3.3246 N·m, above cold-high at 2.8930 N·m; that corner is backlog BL-067 (ruling R-4).

```rust
//! BL-059: the gearbox check. The coupling sits on the gearbox output (owner, 2026-10-14): when
//! the wheel is held it passes up to its pull-out before it slips, and while it slips the
//! gearbox carries a fully reversing swing of the same amplitude. With the minimum magnet
//! temperature at or below the operating one, the strongest pull-out expected is cold-high
//! (`metal.torque_cold_high_Nm`: the pull-out at the minimum magnet temperature with the
//! production allowance added); an operating temperature below the minimum is backlog BL-067.
//!
//! The check compares cold-high at the gearbox input, cold-high divided by the ratio and the
//! efficiency (`metal.cold_high_input_Nm`, Metal design C158), with the gearbox input torque
//! rating (`coupling.gearbox_input_rating_Nm`): the same test as cold-high at most rating ×
//! ratio × efficiency at the output, the workbook's `model.gearbox_reference_Nm` (owner
//! decision 1A). In the forward-drive half of each slip cycle the input carries T/(i·η), so
//! the input basis never checks less than the gearbox sees. A peak check only: the repeated
//! slip swing as a fatigue load is not checked (backlog BL-066).
//!
//! At the rating passes, as the hot minimum check passes at the requirement. The badge quotes
//! both torques to 0.01 N·m, the cold-high at least 0.01 N·m above the rating as quoted, so a
//! failing check never shows two equal numbers.

use super::compat::{fmt_fixed, py_max};
use super::meta::{out_rust_only, results};

results! {
    /// The gearbox check (BL-059). Rust-only.
    pub struct GearboxResults {
        fields {
            cold_high_check: String => out_rust_only("", "Cold-high torque at the gearbox input against its rating",
                "Compares the cold-high torque at the gearbox input, cold-high divided by the ratio and efficiency, with the gearbox input torque rating. A peak check only."),
        }
    }
}

/// `gearbox.cold_high_check` when cold-high at the gearbox input is at or under the rating.
pub const WITHIN_THE_GEARBOX_RATING: &str = "Within the gearbox input rating";

/// The start of `gearbox.cold_high_check` when cold-high at the gearbox input is above the
/// rating; ": {cold-high at the input} N·m against {rating} N·m" follows.
pub const ABOVE_THE_GEARBOX_RATING: &str = "Above the gearbox input rating";

/// `gearbox.cold_high_check` when cold-high at the gearbox input or the rating is not a number
/// (a value set on the struct; `DesignInputs::validate` names the input).
pub const GEARBOX_CHECK_UNKNOWN: &str = "Gearbox check unknown: a torque is not a number";

/// The gearbox check of a design: cold-high at the gearbox input [N·m] against the gearbox
/// input torque rating [N·m]. Equality passes; +∞ fails; NaN in either reads unknown.
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn compute(cold_high_input_Nm: f64, rating_Nm: f64) -> GearboxResults {
    let check = if cold_high_input_Nm.is_nan() || rating_Nm.is_nan() {
        GEARBOX_CHECK_UNKNOWN.to_owned()
    } else if cold_high_input_Nm > rating_Nm {
        let rating = fmt_fixed(rating_Nm, 2);
        // The rating as quoted, so the torque quoted beside it is at least 0.01 N·m above it.
        let quoted: f64 = rating.parse().unwrap_or(rating_Nm);
        format!(
            "{ABOVE_THE_GEARBOX_RATING}: {} N·m against {rating} N·m",
            fmt_fixed(py_max(cold_high_input_Nm, quoted + 0.01), 2)
        )
    } else {
        WITHIN_THE_GEARBOX_RATING.to_owned()
    };
    GearboxResults {
        cold_high_check: check,
    }
}
```

- [ ] **Step 5: Register the module**

In `magcoupling-rs/src/engine/mod.rs` (alphabetical: after `explain`, before `grades`):

Old:
```rust
pub mod deviations;
pub mod explain;
pub mod grades;
```

New:
```rust
pub mod deviations;
pub mod explain;
pub mod gearbox;
pub mod grades;
```

- [ ] **Step 6: Compute the group in `compute_all`**

In `magcoupling-rs/src/engine/api.rs`, four edits.

The import (rustfmt keeps it after the `cfg`-gated `deviations` import):

Old:
```rust
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::grades;
```

New:
```rust
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::gearbox::{self, GearboxResults};
use super::grades;
```

The group, after `housing` in `DesignResults` (around :69):

Old:
```rust
            warnings: WarningResults,
            housing: HousingResults,
        }
```

New:
```rust
            warnings: WarningResults,
            housing: HousingResults,
            gearbox: GearboxResults,
        }
```

The computation, after the space claim (around :356):

Old:
```rust
    // Addendum A1: the space claim, from the derived dimensions (both modes).
    let housing = housing::compute(&mdr, &axial);
```

New:
```rust
    // Addendum A1: the space claim, from the derived dimensions (both modes).
    let housing = housing::compute(&mdr, &axial);
    // BL-059: cold-high at the gearbox input against the gearbox input torque rating.
    let gearbox = gearbox::compute(mdr.cold_high_input_Nm, ci.gearbox_input_rating_Nm);
```

The struct literal (around :413):

Old:
```rust
        warnings: warn,
        housing,
        gap_sweep: gap,
```

New:
```rust
        warnings: warn,
        housing,
        gearbox,
        gap_sweep: gap,
```

- [ ] **Step 7: Run the new tests and see them pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test gearbox --test schema`

Expected: `tests/gearbox.rs` gives `test result: ok. 11 passed; 0 failed; 0 ignored`. Its tests are `the_defaults_fail_with_both_torques_quoted`, `a_rating_one_step_under_cold_high_quotes_the_torque_a_hundredth_above_it`, `each_probe_design_passes_exactly_when_cold_high_at_the_input_is_within_the_rating`, `n33ah_fails_the_input_basis_that_an_output_basis_without_the_efficiency_passes`, `exactly_at_the_rating_passes_and_one_float_above_fails`, `a_torque_that_is_not_a_number_reads_unknown`, `an_infinite_cold_high_at_the_gearbox_input_fails`, `a_failing_check_never_quotes_two_equal_numbers`, `the_rating_slider_s_ends_fail_and_pass_the_defaults`, `the_rating_moves_only_the_gearbox_check_and_the_workbook_s_reference` and `the_check_is_a_rust_only_text_result_with_no_cell_after_the_housing_group`. Then `tests/schema.rs` gives `test result: ok. 11 passed; 0 failed`.

- [ ] **Step 8: Run the engine suite (no features)**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`:

| binary | result |
|---|---|
| lib | 228 passed, 1 ignored |
| assumptions | 11 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 56 passed, 1 ignored |
| gearbox | 11 passed |
| grades | 11 passed |
| material_library | 4 passed |
| material_links | 12 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 12 passed |
| schema | 11 passed |
| size_controls | 38 passed |
| sizing | 43 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

The Python parity suites (`python_schema`, `parity`, `differential`, `deviations`) pass unchanged, at HEAD's counts: they skip `rust_only` results (`tests/python_schema.rs:12`, `:100`, `:159`, `:326-332`; `tests/differential.rs:35`, `:538`). `tests/robustness.rs` runs its extremes through the new group unchanged.

- [ ] **Step 9: Run the app library tests and see what the new group moves**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: `test result: FAILED. 846 passed; 6 failed; 6 ignored`. The failures:
```text
gui::result_groups::tests::every_chain_path_names_results_and_every_package_has_a_heading
    panicked at src\gui\result_groups.rs:359:13: no heading for gearbox
gui::result_groups::tests::every_result_is_in_exactly_one_group
    panicked at src\gui\result_groups.rs:243:9: left: 1086, right: 1087
gui::results_table::tests::the_lines_group_the_rows_with_only_the_headline_open_at_first
    panicked at src\gui\results_table.rs:1217:9: left: 1086, right: 1087
gui::spreadsheet::tests::every_result_is_on_one_row_by_physics_chain_with_its_value
    panicked at src\gui\spreadsheet.rs:938:9: left: 1086, right: 1087
gui::trace::tests::an_input_no_explained_result_reads_traces_none_and_says_so
    panicked at src\gui\trace.rs:194:9: left: (392, 1087), right: (392, 1086)
gui::dashboard::tests::every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed
    panicked at src\gui\dashboard.rs:1343:17: gearbox.cold_high_check is not in CHECKS
```
The first four fail because the group has no results-table heading yet: its row falls in no group (Step 10). The trace test moves with one more result (Step 11). The last one is Task 2's (PD-4).

- [ ] **Step 10: Give the group its results-table heading**

In `magcoupling-rs/src/gui/result_groups.rs`, `PACKAGE_LABELS` (around :74), in schema order after `housing`:

Old:
```rust
pub const PACKAGE_LABELS: [(&str, &str); 12] = [
```
```rust
    ("housing", "Housing"),
    ("gap_sweep", "Gap sweep"),
```

New:
```rust
pub const PACKAGE_LABELS: [(&str, &str); 13] = [
```
```rust
    ("housing", "Housing"),
    ("gearbox", "Gearbox"),
    ("gap_sweep", "Gap sweep"),
```

In this task's tree the check shows in the results table under Other results, Gearbox. Once Task 2 puts it on the dashboard, `groups_of` places it in the Headline group, which takes the `DASHBOARD` rows first. The Gearbox group is then empty and dropped (`groups.retain`), so the "Gearbox" heading no longer shows. The label stays required, because `every_chain_path_names_results_and_every_package_has_a_heading` asks every result's package for a heading.

- [ ] **Step 11: Move the trace totals**

In `magcoupling-rs/src/gui/trace.rs`. The new result has no equation record, so the explained count stays 392 and the total gains one.

Module doc, line 7. Old:
```rust
//! the dependencies of the results it explains (the A-3 scope: 392 of the 1086 results), so a
```
New:
```rust
//! the dependencies of the results it explains (the A-3 scope: 392 of the 1087 results), so a
```

`an_input_no_explained_result_reads_traces_none_and_says_so` (line 194). Old:
```rust
        assert_eq!((explained, table_entries().len()), (392, 1086));
```
New:
```rust
        assert_eq!((explained, table_entries().len()), (392, 1087));
```

- [ ] **Step 12: Run the app library tests again**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`

Expected: `test result: FAILED. 851 passed; 1 failed; 6 ignored`. The one failure is `gui::dashboard::tests::every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`, "gearbox.cold_high_check is not in CHECKS" (`dashboard.rs:1343`). It is expected and stays until Task 2 adds the `CHECKS` entry (PD-4).

- [ ] **Step 13: Docs: `magcoupling-rs/README.md`**

Five edits.

(a) The Layout table: insert this row after the `src/engine/housing.rs` row (around :146, unchanged):
```markdown
| `src/engine/gearbox.rs` | BL-059 gearbox check: `GearboxResults` (Rust-only, `gearbox.cold_high_check`): cold-high at the gearbox input (`metal.cold_high_input_Nm`, cold-high divided by the ratio and the efficiency) against the gearbox input torque rating (`coupling.gearbox_input_rating_Nm`), the same test as cold-high against the workbook's `model.gearbox_reference_Nm` at the output (owner decision 1A); `WITHIN_THE_GEARBOX_RATING`, `ABOVE_THE_GEARBOX_RATING` (the failing text's start: both torques to 0.01 N·m, the torque at least 0.01 N·m above the rating as quoted, so the two never read equal) and `GEARBOX_CHECK_UNKNOWN` (either torque not a number); at the rating passes, +inf fails. A peak check only: the repeated slip swing as a fatigue load is BL-066 |
```

(b) The trace paragraph (around :293). Old:
```text
  registry's `downstream`; the registry explains 392 of the 1086 results, so an input such as the
```
New:
```text
  registry's `downstream`; the registry explains 392 of the 1087 results, so an input such as the
```

(c) The `src/gui/inputs.rs`, … `src/gui/results_table.rs` row of the Tests table (around :667). Replace only this substring; Task 2 edits the "30 checks" count in the same row. Old:
```text
392 of the 1086 results are explained, every input traceable
```
New:
```text
392 of the 1087 results are explained, every input traceable
```

(d) The Tests table: insert this row after the `tests/sizing.rs` row (around :654, unchanged):
```markdown
| `tests/gearbox.rs` | BL-059 gearbox check end to end: the defaults fail quoting 0.80 against 0.60 N·m; with the rating one slider step under cold-high at the input (0.80) the torque reads 0.81, and over the whole rating slider a failing badge never quotes two equal numbers (cold-high 0.6041 against 0.604 reads 0.61 against 0.60); the five probe designs pass exactly when cold-high at the gearbox input is within the rating (8 magnets, 304 back iron and Y30 pass; the default and the 20 mm ring fail), as against the workbook's reference at the output; N33AH on both rings (2.9401 N·m at the output, 0.6190 at the input) fails although an output limit of 0.6 × 5 = 3.0 N·m would pass it (owner decision 1A); at the rating passes and one float above fails; a NaN rating or ratio reads unknown (`validate()` names the input), +inf fails; the slider's ends (0.05, 5.0) fail and pass the defaults; the rating moves only this check and the workbook's reference; the check is a Rust-only text result with no cell, after the housing group. |
```

(e) The Python-parity paragraph under "Regenerating test data" (around :692-696): a Rust-only result group. Old:
```text
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added.
```
New:
```text
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added. Nor does a Rust-only result group (`warnings`,
`housing`, `gearbox`: every field declared with `out_rust_only`): it has no `MODULES` entry
and no data file, and the parity, differential and Python schema tests skip `rust_only`
results, so gate 12 compares nothing new.
```
(Checked in the prototype: `warnings` 6 of 6, `housing` 7 of 7 and `gearbox` 1 of 1 rows are `rust_only`; every other group has at least one Python row. Gate 12 is "Python parity suite + gen_differential.py --check", `03-structure.yaml` gate line.)

- [ ] **Step 14: Docs: `docs/ai/03-structure.yaml`**

Three edits under `magcoupling_rs.engine_files` and `magcoupling_rs.tests`.

(a) Insert this entry after the `housing:` entry (around :52, unchanged):
```yaml
    gearbox: "gearbox.rs (BL-059 gearbox check: GearboxResults = DesignResults::gearbox, Rust-only cold_high_check, computed in api.rs beside housing::compute: cold-high at the gearbox input (metal.cold_high_input_Nm) against coupling.gearbox_input_rating_Nm, owner decision 1A, the same test as cold-high against model.gearbox_reference_Nm at the output; WITHIN_THE_GEARBOX_RATING, ABOVE_THE_GEARBOX_RATING (the failing text's start: both torques to 0.01 N m, the torque at least 0.01 above the rating as quoted), GEARBOX_CHECK_UNKNOWN (NaN in either); at the rating passes, +inf fails; a peak check only, the fatigue check is BL-066)"
```

(b) Around :63. Old:
```yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
```
New:
```yaml
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, gearbox.rs, explain/]
```

(c) The `tests:` line (around :73), which is an unquoted scalar, so the inserted text has no ": ". Old substring:
```text
assumptions.rs, sizing.rs (also solve_in, grid_in and InvalidRange), size_controls.rs
```
New:
```text
assumptions.rs, sizing.rs (also solve_in, grid_in and InvalidRange), gearbox.rs (the BL-059 gearbox check end to end; the defaults, the probe designs, the N33AH input-basis case, equality, NaN and +inf, the quoted numbers never equal), size_controls.rs
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`. A ": " inside the `tests:` scalar fails here with "mapping values are not allowed here".

- [ ] **Step 15: Docs: the crate doc (`magcoupling-rs/src/lib.rs`)**

The crate doc names each engine result the forward calculation reports beyond the workbook; it gains the new one.

Old:
```rust
//! forward calculation also reports the A1 space claim ([`engine::housing`],
//! with the axial housing that follows the length override). Beyond it:
```
New:
```rust
//! forward calculation also reports the A1 space claim ([`engine::housing`],
//! with the axial housing that follows the length override) and the BL-059
//! gearbox check ([`engine::gearbox`]: cold-high at the gearbox input against
//! its rating). Beyond it:
```

- [ ] **Step 16: Run the docs-count test**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count_the_notes_and_the_steps`

Expected: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 56 filtered out`. It reads `README.md` and `03-structure.yaml` through `include_str!`, and their note and step counts are unchanged.

- [ ] **Step 17: Clippy both ways and fmt**

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`

Expected: `Finished`, no warning.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`

Expected: `Finished`, no warning.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`

Expected: no output, exit 0. The code above is rustfmt's layout.

- [ ] **Step 18: End without a commit: both suites with the one expected failure, and the files Task 2 stages (PD-4)**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: as Step 8, every binary `ok`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: the lib gives `test result: FAILED. 851 passed; 1 failed; 6 ignored`. The one failure is exactly `gui::dashboard::tests::every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`: "gearbox.cold_high_check is not in CHECKS". The two app binaries give 0 tests, and every integration binary passes at Step 8's counts (gearbox 11). Any other failure: fix it before Task 2 starts.

Run: `git status --short`

Expected:
```text
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/api.rs
 M magcoupling-rs/src/engine/mod.rs
 M magcoupling-rs/src/gui/result_groups.rs
 M magcoupling-rs/src/gui/trace.rs
 M magcoupling-rs/src/lib.rs
 M magcoupling-rs/tests/schema.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
?? magcoupling-rs/src/engine/gearbox.rs
?? magcoupling-rs/tests/gearbox.rs
```

This task makes no commit. The controller's per-task review of T1 runs on `git diff` plus the two untracked files. Task 2's `git add` (Task 2, Step 9) names these ten files along with its own. It never stages `docs/analyses/2026-05-28-press-4bar-analysis.md`:
```text
magcoupling-rs/src/engine/gearbox.rs
magcoupling-rs/src/engine/mod.rs
magcoupling-rs/src/engine/api.rs
magcoupling-rs/src/gui/result_groups.rs
magcoupling-rs/src/gui/trace.rs
magcoupling-rs/tests/gearbox.rs
magcoupling-rs/tests/schema.rs
magcoupling-rs/src/lib.rs
magcoupling-rs/README.md
docs/ai/03-structure.yaml
```

Task 2's commit message opens with this task's paragraph (written out in Task 2, Step 9).

---

### Task 2: The gearbox check on the dashboard: its row, `CHECKS`, verdict levels, greying and the guide core

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

Line numbers are at HEAD `3e63bec`; T1 does not touch the `gui` files this task edits. Commands run from the
repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block below is rustfmt's layout.

Before you start: Task 1's changes are in the working tree, uncommitted (plan deviation PD-4), and
`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib` fails exactly one test,
`every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` ("gearbox.cold_high_check is not in
CHECKS"). This task's `CHECKS` entry turns it green, and this task's commit (Step 9) stages Task 1's files with its own.

**Files:**
- Modify: `magcoupling-rs/src/gui/dashboard.rs`: module doc `:5-8`; `DASHBOARD` `:153-185`; `END_EFFECT_ROWS` `:187-200`;
  `CHECKS` doc and array `:254-291`; `verdict_level` `:368-370`. Tests: imports `:997-1001`, `:1005-1016`, `:1026-1086`,
  `:1310-1313`, `:1361-1392`, `:1413-1457`, `:1495-1503`, `:1603-1661` (and a new test after it), `:1961-1964`,
  `:2148-2161`, `:2209-2223`.
- Modify: `magcoupling-rs/src/gui/guidance.rs`: check consts `:140-146`; after `CLAMP_GUIDE` `:202-212`; `CHECK_GUIDES`
  `:290-302`; `ROW_LEARN` `:343-345`; `design_status_text` doc `:1481-1484`; before `space_claim_problem` `:1622`. Tests:
  imports `:2065-2068`, after `unknown_claim` `:2150-2153`, `designs()` doc `:2171`, `every_text` `:2259`, the tripwire
  `:2462-2478`, `terms` `:2657-2662`, `:2822-2829`, `:3276-3346`, `:3461-3474`, `:3543-3551`, `:3626-3628`, a new test
  before `:4001`, `:4017-4065`, `:4102-4126`, a new test before `:4128`, `:4930-4935`.
- Modify: `magcoupling-rs/src/gui/test_support.rs`: `passing_design` `:25-37`; a new design before `:59`.
- Modify: `magcoupling-rs/src/gui/panel.rs` tests: `:3037-3045`, `:3064-3081`, `:3092`, `:3104-3108`, `:3141`,
  `:8687-8703`.
- Modify: `magcoupling-rs/src/gui/results_table.rs` test: `:1282`.
- Modify: `magcoupling-rs/src/gui/spreadsheet.rs` test comment `:1000` (six failing checks -> seven).
- Modify: `magcoupling-rs/tests/explain.rs`: a new test before `the_docs_count_the_notes_and_the_steps` (`:1241`).
- Modify: `magcoupling-rs/README.md` `:300-301`, `:317-322`, `:343`, `:353`, `:362`, `:656`, `:663`, `:667`, `:670`;
  `docs/ai/02-system.yaml:107`; `docs/ai/03-structure.yaml:66`.
- Test: the five `gui` files above with tests (`--features app`) and `tests/explain.rs`; the engine-only suite runs
  unchanged.
- Commit: Task 1's ten files with this task's (Step 9).

**Interfaces:**
- Consumes (T1): `crate::engine::gearbox::{WITHIN_THE_GEARBOX_RATING, ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN,
  compute}`; `DesignResults.gearbox.cold_high_check: String` at the path `gearbox.cold_high_check` (Rust-only, engine label
  "Cold-high torque at the gearbox input against its rating"); the existing `metal.cold_high_input_Nm`,
  `coupling.gearbox_input_rating_Nm` (default 0.6 N·m), `coupling.gear_ratio`, and `guidance::TORQUE_BAND_NOTE`
  (`"hot_low_cold_high"`).
- Produces:
  - `dashboard::DASHBOARD: [(&str, Option<&str>); 18]`, row 15 `("gearbox.cold_high_check", Some("gearbox.cold_high_check"))`,
    so `paths[15..]` = [gearbox, space claim, end effect]; `END_EFFECT_ROWS: [&str; 8]` (the gearbox check last);
    `CHECKS: [&str; 31]` (the gearbox check last, schema order); `verdict_level` Good / Caution / Bad (by prefix) for the
    three engine texts.
  - `guidance` (private to the module and its tests): `const GEARBOX`, `const GEARBOX_GUIDE` (label "Gearbox input
    torque", fixes Lower `MAGNET_LENGTH` then Raise `FACE_GAP`, learn `Learn::Note(TORQUE_BAND_NOTE)`),
    `CHECK_GUIDES: [CheckGuide; 9]` with the gearbox guide at index 5, its `ROW_LEARN` entry, `const GEARBOX_UNKNOWN`,
    `fn gearbox_problem(&DesignInputs, &DesignResults) -> Option<String>` whose `Level::Bad` arm is Problem sentence 1
    (T3 appends `gearbox_conflict` to that arm).
  - Tests: `test_support::only_the_gearbox_fails()` (the passing design with the example's gearbox rating: T4's
    "gearbox only" design is `("gearbox_only", only_the_gearbox_fails())`, T3's sentence-2-absent design is the same);
    the guidance test edit `unknown_rating(&mut DesignInputs)`; the `terms()` arm `GEARBOX` with the term
    `"gearbox input torque"` (margin: rating minus cold-high at the input; T4's cases use that name); the two
    `GEARBOX` cases of `every_fix_moves_its_check_toward_passing_over_its_slider` (they land with the guide: the test's
    table must list every slider fix of every guide); the `GEARBOX` case of `the_problems_quote_numbers_from_the_results`
    (the defaults against a minimum temperature of 0 °C; T3 extends that case for sentence 2).
  - `tests/explain.rs`: `the_docs_count_the_dashboard_rows_and_the_checks` (feature `gui`), which holds the README's
    dashboard row and check counts and `03-structure.yaml`'s CHECKS count to the code.

- [ ] **Step 1: Write the failing tests (test support, the new tests, the tripwire's replacement, every pinned test that moves)**

**1a. `magcoupling-rs/src/gui/test_support.rs`.** Replace `passing_design` and its doc (`:25-37`)

```rust
/// A design that passes all 8 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
/// to spare: the defaults with a 1.8 mm face gap (running clearance about 0.1 mm over its
/// target), a 2.0 mm cup wall (the flux needs about 1.85 mm at that gap) and a 1.6 N·m
/// requirement (hot-low torque about 2.0 N·m, a quarter over it: a hotter-rated grade, weaker
/// than N42SH, still meets it). The guidance tests break one check of it at a time, so a fix
/// that turns another check red shows.
pub(crate) fn passing_design() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.0;
    inputs.metal.required_min_Nm = 1.6;
    inputs
}
```

with

```rust
/// A design that passes all 9 guided checks (`gui::guidance::CHECK_GUIDES`), each with room
/// to spare: the defaults with a 1.8 mm face gap (running clearance about 0.1 mm over its
/// target), a 2.0 mm cup wall (the flux needs about 1.85 mm at that gap), a 1.6 N·m
/// requirement (hot-low torque about 2.0 N·m, a quarter over it: a hotter-rated grade, weaker
/// than N42SH, still meets it) and a 1.0 N·m gearbox input rating (cold-high at the gearbox
/// input about 0.71 N·m, 29 % under it; the rating changes no other result than the
/// workbook's gearbox reference, `model.gearbox_reference_Nm`, its only other reader). The
/// guidance tests break one check of it at a time, so a fix that turns another check red
/// shows.
pub(crate) fn passing_design() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.0;
    inputs.metal.required_min_Nm = 1.6;
    inputs.coupling.gearbox_input_rating_Nm = 1.0;
    inputs
}
```

Then insert, right before `/// The defaults at the clamp's limit: the clamp's safety factor raised until its capacity only` (`:59`):

```rust
/// [`passing_design`] with the gearbox input rating back at the example design's: it fails the
/// gearbox check alone (cold-high at the gearbox input about 0.71 N·m against 0.6 N·m).
pub(crate) fn only_the_gearbox_fails() -> crate::DesignInputs {
    let mut inputs = passing_design();
    inputs.coupling.gearbox_input_rating_Nm = crate::DesignInputs::default()
        .coupling
        .gearbox_input_rating_Nm;
    inputs
}

```

**1b. `magcoupling-rs/src/gui/dashboard.rs`, `mod tests`.**

(1) The imports (`:997-1001`): replace

```rust
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, hover_frames, only_the_clamp_fails, passing_design,
        primary_button, short_magnets, sized_frame, text_center, text_color, text_rect,
        tooltips_at_once, with_poles,
    };
```

with

```rust
    use crate::gui::test_support::{
        assert_glyphs, drawn_texts, hover_frames, only_the_clamp_fails, only_the_gearbox_fails,
        passing_design, primary_button, short_magnets, sized_frame, text_center, text_color,
        text_rect, tooltips_at_once, with_poles,
    };
```

(2) `the_dashboard_starts_with_the_headline_in_order` (`:1009-1012`): replace

```rust
        assert_eq!(
            paths[15..],
            ["housing.space_claim_check", "model.end_effect_check"]
        );
```

with

```rust
        assert_eq!(
            paths[15..],
            [
                "gearbox.cold_high_check",
                "housing.space_claim_check",
                "model.end_effect_check"
            ]
        );
```

(3) `every_verdict_each_check_gives_is_classified` (`:1058`): insert before `("housing.space_claim_check", DesignInputs::default(), Good),`

```rust
            ("gearbox.cold_high_check", DesignInputs::default(), Bad),
            ("gearbox.cold_high_check", passing_design(), Good),
            (
                "gearbox.cold_high_check",
                design(|i| i.coupling.gearbox_input_rating_Nm = f64::NAN),
                Caution,
            ),
```

and in the same test replace

```rust
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
        }
        // No back iron (E9): the wall rule does not apply, so the check gives no badge.
```

with

```rust
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
        }
        // The gearbox check's texts as its engine writes them: at the rating, above it, an
        // infinite torque, and a torque or a rating that is not a number.
        let gearbox = |torque: f64, rating: f64| {
            let text = crate::engine::gearbox::compute(torque, rating).cold_high_check;
            verdict_level("gearbox.cold_high_check", &text)
        };
        assert_eq!(gearbox(0.6, 0.6), Some(Good));
        assert_eq!(gearbox(0.8049, 0.6), Some(Bad));
        assert_eq!(gearbox(f64::INFINITY, 0.6), Some(Bad));
        assert_eq!(gearbox(f64::NAN, 0.6), Some(Caution));
        assert_eq!(gearbox(0.6, f64::NAN), Some(Caution));
        // No back iron (E9): the wall rule does not apply, so the check gives no badge.
```

(4) `every_verdict_of_every_other_check_is_classified` (`:1310-1311`): replace

```rust
        // No arm reads a text of another check, or of a path that is no check.
        assert_eq!(verdict_level("model.hub_check", "OK"), None);
```

with

```rust
        // No arm reads a text of another check, or of a path that is no check.
        assert_eq!(verdict_level("model.hub_check", "OK"), None);
        assert_eq!(
            verdict_level(
                "housing.space_claim_check",
                crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING
            ),
            None
        );
        assert_eq!(
            verdict_level(
                "gearbox.cold_high_check",
                crate::engine::housing::INSIDE_THE_SPACE_CLAIM
            ),
            None
        );
```

(5) `the_failing_checks_are_the_red_then_the_amber` (`:1366-1392`): replace `// The defaults fail four checks, all red.` with
`// The defaults fail five checks, all red.`; in both expected lists insert `("gearbox.cold_high_check", Bad),` right after
`("materials.cup_wall_check", Bad),` (in the manual-magnet list, before `("model.inner_temp_check", Caution),`). The two
lists then read

```rust
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("gearbox.cold_high_check", Bad),
            ]
```

and

```rust
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("gearbox.cold_high_check", Bad),
                ("model.inner_temp_check", Caution),
                ("model.outer_temp_check", Caution),
            ]
```

(6) The moved-rows helper and the end-effect test (`:1413-1447`): replace

```rust
    /// The headline paths whose value moves when `inputs` take any of `values`, from the
    /// defaults at back iron 1 and 0.
    fn headline_moved_by(
```

with

```rust
    /// The dashboard rows whose value moves when `inputs` take any of `values`, from the
    /// defaults at back iron 1 and 0.
    fn dashboard_moved_by(
```

replace its inner loop

```rust
                    for (_, result) in HEADLINE {
                        if before.get(result) != after.get(result) {
                            moved.insert(result);
                        }
                    }
```

with

```rust
                    for &(row, _) in &DASHBOARD {
                        if before.get(row) != after.get(row) {
                            moved.insert(row);
                        }
                    }
```

and replace the test

```rust
    #[test]
    fn the_end_effect_rows_are_the_headline_rows_that_move_with_the_end_effect_coefficient() {
        // c_end = 0 (f_end = 1) is the one value that flips the hot-minimum verdict.
        let moved = headline_moved_by(&["coupling.c_end".to_owned()], |_| {
            vec![0.0, 0.05, 0.3, 0.5]
        });
        assert_eq!(moved, END_EFFECT_ROWS.into_iter().collect());
    }
```

with

```rust
    #[test]
    fn the_end_effect_rows_are_the_dashboard_rows_that_move_with_the_end_effect_coefficient() {
        // c_end = 0 (f_end = 1) is the one value that flips the hot-minimum verdict.
        let moved = dashboard_moved_by(&["coupling.c_end".to_owned()], |_| {
            vec![0.0, 0.05, 0.3, 0.5]
        });
        assert_eq!(moved, END_EFFECT_ROWS.into_iter().collect());
        // The headline's rows, and one row past it: the gearbox check, computed from cold-high.
        let past_the_headline: Vec<&str> = END_EFFECT_ROWS
            .into_iter()
            .filter(|row| !HEADLINE.iter().any(|&(_, path)| path == *row))
            .collect();
        assert_eq!(past_the_headline, ["gearbox.cold_high_check"]);
    }
```

Then rename the three other calls: in `the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs`
`let moved = headline_moved_by(&inputs, |path| {` becomes `let moved = dashboard_moved_by(&inputs, |path| {`; in
`the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits`
`headline_moved_by(&fields("temperature.slip_loss."), ends),` becomes
`dashboard_moved_by(&fields("temperature.slip_loss."), ends),` and
`let by_reverse = headline_moved_by(&fields("temperature.demag."), ends);` becomes
`let by_reverse = dashboard_moved_by(&fields("temperature.demag."), ends);`. (Their assertions stay: the stored 3D
inputs move no row past the headline.)

(7) `out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter` (`:1603-1661`): replace

```rust
        // so the two checks among the greyed rows give no badge and are not listed as failing.
```

with

```rust
        // so the three checks among the greyed rows give no badge and are not listed as failing.
```

replace `assert_eq!(greyed, ["metal.hot_min_check", "clamps.recommended"]);` with

```rust
        assert_eq!(
            greyed,
            [
                "metal.hot_min_check",
                "clamps.recommended",
                "gearbox.cold_high_check"
            ]
        );
```

replace `// The verdict has a level of its own (red and green here): the rule drops it.` with
`// The verdict has a level of its own (red, green and green here): the rule drops it.`, and replace

```rust
        // In range nothing is dropped: the defaults' hot minimum is red, the screw green.
        let defaults = compute_all(&DesignInputs::default());
        assert!(!greyed_by_end_effect(&defaults, "metal.hot_min_check"));
        assert_eq!(check_level(&defaults, "metal.hot_min_check"), Some(Bad));
        assert_eq!(
            check_level(&defaults, "clamps.recommended"),
            Some(Level::Good)
        );
```

with

```rust
        // In range nothing is dropped: the defaults' hot minimum is red, the screw green, the
        // gearbox check red.
        let defaults = compute_all(&DesignInputs::default());
        assert!(!greyed_by_end_effect(&defaults, "metal.hot_min_check"));
        assert_eq!(check_level(&defaults, "metal.hot_min_check"), Some(Bad));
        assert_eq!(
            check_level(&defaults, "clamps.recommended"),
            Some(Level::Good)
        );
        assert!(!greyed_by_end_effect(&defaults, "gearbox.cold_high_check"));
        assert_eq!(check_level(&defaults, "gearbox.cold_high_check"), Some(Bad));
```

(8) New test: insert right before `#[test]` / `fn a_design_past_the_space_claim_shows_the_overshoot_in_red() {` (`:1663`):

```rust
    #[test]
    fn out_of_range_end_effect_greys_the_gearbox_check_whatever_its_text_reads() {
        // Cold-high comes from the pull-out, so while the end-effect model is out of range it is
        // no torque: on short magnets it is below zero and the engine's text reads within the
        // rating, a pass from a meaningless number; a rating below it (set on the struct) turns
        // the text red. Either way the row is greyed without a badge, the check has no level in
        // the dashboard, the results table or its failing filter, its guide shows no Problem,
        // and the status line counts it neither way.
        use crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING};
        use crate::gui::results_table::{entry_index, entry_level, table_entries};
        let gearbox = "gearbox.cold_high_check";
        let guide = guidance::guide(gearbox).expect("the gearbox check's guide");
        let entry = &table_entries()[entry_index(gearbox).expect("the gearbox check's row")];
        let mut below = short_magnets();
        below.coupling.gearbox_input_rating_Nm = -10.0;
        let mut statuses = Vec::new();
        for (inputs, text) in [
            (short_magnets(), WITHIN_THE_GEARBOX_RATING),
            (below, ABOVE_THE_GEARBOX_RATING),
        ] {
            let results = compute_all(&inputs);
            assert!(end_effect_out_of_range(&results).is_some());
            let torque = results.metal.cold_high_input_Nm;
            assert!(torque < 0.0, "{torque}");
            let verdict = &results.gearbox.cold_high_check;
            assert!(verdict.starts_with(text), "{verdict}");
            assert!(verdict_level(gearbox, verdict).is_some(), "{verdict}");
            assert!(greyed_by_end_effect(&results, gearbox));
            assert_eq!(check_level(&results, gearbox), None);
            let line = dashboard_lines(&results)
                .into_iter()
                .find(|l| l.path == gearbox)
                .expect("the gearbox row");
            assert!(line.greyed);
            assert_eq!(line.level, None);
            assert_eq!(entry_level(&results, entry), None);
            assert!(
                failing_checks(&results, false)
                    .iter()
                    .all(|&(path, _)| path != gearbox)
            );
            assert_eq!(guidance::level(guide, &inputs, &results), Level::Good);
            assert_eq!((guide.problem)(&inputs, &results), None);
            statuses.push(design_status(&inputs, &results, false));
        }
        assert_eq!(statuses[0], statuses[1]);
    }

```

(9) `while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard` (`:1963`): `"Your design: passes all 7 checks"`
becomes `"Your design: passes all 8 checks"`.

(10) `a_red_check_shows_its_block_an_amber_one_its_header_and_every_row_its_why_link` (`:2148-2161`): replace
`// The workbook example's three red checks are the ones with blocks.` with
`// The workbook example's four red checks are the ones with blocks.`, and the array

```rust
        let want = [
            "metal.hot_min_check",
            "metal.clearance_check",
            "materials.cup_wall_check",
        ];
```

with

```rust
        let want = [
            "metal.hot_min_check",
            "metal.clearance_check",
            "materials.cup_wall_check",
            "gearbox.cold_high_check",
        ];
```

(11) `the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour` (`:2210-2221`): replace

```rust
            // The workbook example: three red checks.
            (DesignInputs::default(), 3, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                2,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 3, 1, Level::Bad),
```

with

```rust
            // The workbook example: four red checks.
            (DesignInputs::default(), 4, 0, Level::Bad),
            // One fixed: the cup wall at its rule's 2.0 mm and up.
            (
                design(|i| i.metal.cup_wall_corner_mm = 2.4),
                3,
                0,
                Level::Bad,
            ),
            (passing_design(), 0, 0, Level::Good),
            // The passing design with the example's gearbox rating: the gearbox check alone.
            (only_the_gearbox_fails(), 1, 0, Level::Bad),
            // The hot day's temperature verdict is amber, counted apart from the red.
            (hot_day(), 4, 1, Level::Bad),
```

**1c. `magcoupling-rs/src/gui/guidance.rs`, `mod tests`.**

(1) The imports (`:2065-2068`): replace

```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, hot_day, only_the_clamp_fails, passing_design,
        short_magnets, small_boss, with_poles,
    };
```

with

```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, hot_day, only_the_clamp_fails, only_the_gearbox_fails,
        passing_design, short_magnets, small_boss, with_poles,
    };
```

(2) Insert right before `/// True arcs instead of flat blocks.` (`:2155`):

```rust
    /// A gearbox input rating that is not a number (set on the struct): the gearbox check asks
    /// for a look.
    fn unknown_rating(i: &mut DesignInputs) {
        i.coupling.gearbox_input_rating_Nm = f64::NAN;
    }

```

(3) `designs()` doc (`:2171`): `/// Every design the tests use: the defaults (they fail three checks), a passing design,` becomes
`/// Every design the tests use: the defaults (they fail four checks), a passing design,`.

(4) `every_text()` (`:2259`): replace `for fallback in [NO_FIX, TEMPERATURE_FALLBACK, SPACE_CLAIM_FALLBACK] {` with

```rust
        for fallback in [
            NO_FIX,
            TEMPERATURE_FALLBACK,
            SPACE_CLAIM_FALLBACK,
            GEARBOX_UNKNOWN,
        ] {
```

(5) The tripwire's replacement, in the same commit as the row (spec risk 1). Replace the whole of `:2462-2478`

```rust
    /// The torque-band note's second watch-out and the slip note's first say nothing checks
    /// cold-high at the gearbox input, or the slip swings, against the gearbox's rating. When
    /// such a check lands (BL-059) this fails, and those watch-outs change in the same commit.
    #[test]
    fn nothing_checks_a_gearbox_torque_against_its_rating_yet() {
        let gearbox_rows: Vec<(&str, Option<&str>)> = DASHBOARD
            .iter()
            .copied()
            .filter(|&(row, check)| {
                row.contains("gearbox") || check.is_some_and(|c| c.contains("gearbox"))
            })
            .collect();
        // The ripple row: a value with no check.
        assert_eq!(gearbox_rows, [("model.gearbox_input_ripple_Nm", None)]);
        assert!(CHECKS.iter().all(|c| !c.contains("gearbox")));
        assert!(CHECK_GUIDES.iter().all(|g| !g.check.contains("gearbox")));
    }
```

with

```rust
    /// The gearbox check (BL-059, replacing the test that said nothing checks a gearbox torque):
    /// one dashboard row of its own, after the headline's rows and before the space claim, a
    /// design check of the results, and a guide, between the clamp's and the space claim's,
    /// whose Why link and its row's lead to the torque-band note, which owns cold-high at the
    /// gearbox input. The ripple row stays a value with no check.
    #[test]
    fn the_gearbox_check_is_a_dashboard_row_with_its_guide() {
        let gearbox_rows: Vec<(usize, (&str, Option<&str>))> = DASHBOARD
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, (row, check))| {
                row.contains("gearbox") || check.is_some_and(|c| c.contains("gearbox"))
            })
            .collect();
        assert_eq!(
            gearbox_rows,
            [
                (5, ("model.gearbox_input_ripple_Nm", None)),
                (15, (GEARBOX, Some(GEARBOX))),
            ]
        );
        let checks: Vec<&str> = CHECKS
            .iter()
            .copied()
            .filter(|c| c.contains("gearbox"))
            .collect();
        assert_eq!(checks, [GEARBOX]);
        let gearbox = guide(GEARBOX).expect("the gearbox check has a guide");
        assert_eq!(gearbox.learn, Learn::Note(TORQUE_BAND_NOTE));
        assert_eq!(row_learn(GEARBOX), Some(gearbox.learn));
        let guided = CHECK_GUIDES
            .iter()
            .filter(|g| g.check.contains("gearbox"))
            .count();
        assert_eq!(guided, 1);
        let at = |check: &str| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        assert_eq!(at(GEARBOX), at(CLAMP) + 1);
        assert_eq!(at(SPACE_CLAIM), at(GEARBOX) + 1);
    }
```

(6) `terms()` (`:2657-2662`): insert before `END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0)],`

```rust
            GEARBOX => vec![at_least_zero(
                "gearbox input torque",
                inputs.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
            )],
```

(7) `every_fix_moves_its_check_toward_passing_over_its_slider` (its table must list every slider fix of every guide,
so the gearbox guide's two cases land with the guide, here and not in T4). Insert after the `CLAMP` /
`breaking(short_clamp)` case (after `:2828`, before the first `SPACE_CLAIM` case):

```rust
            (
                GEARBOX,
                DesignInputs::default(),
                MAGNET_LENGTH,
                Direction::Lower,
                "gearbox input torque",
            ),
            (
                GEARBOX,
                DesignInputs::default(),
                FACE_GAP,
                Direction::Raise,
                "gearbox input torque",
            ),
```

(8) `two_failing_checks_that_pull_one_input_apart_say_so` (`:3276-3346`). The magnet length can now be pulled by three
guides at once (the hot minimum raises it; the gearbox check and the space claim lower it), so the expected notes are
built per guide pulling the other way. Replace

```rust
        // both ways when two checks that pull it apart both fail already: here the hot-low
        // torque against an overall-length claim the stack is over (magnet length), and the
        // blocks against a diameter claim (Outside diameter). Each fix helps its own check, so
        // both stay, and each of the two blocks says, in the same words, that the input pulls
        // two ways and another input has to settle one of them (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 2] = [
            ("length_conflict", &[MAGNET_LENGTH]),
            ("diameter_conflict", &[RING]),
        ];
```

with

```rust
        // both ways when two checks that pull it apart both fail already: here the hot-low
        // torque against the gearbox check, on the example design and wherever both fail (magnet
        // length: no magnet size passes both), and against an overall-length claim the stack is
        // over (magnet length), and the blocks against a diameter claim (Outside diameter). Each
        // fix helps its own check, so both stay, and each of the two blocks says, in the same
        // words, that the input pulls two ways and another input has to settle one of them
        // (`trade_offs`).
        let pulled_apart: [(&str, &[&str]); 4] = [
            ("defaults", &[MAGNET_LENGTH]),
            ("bare_clearance", &[MAGNET_LENGTH]),
            ("length_conflict", &[MAGNET_LENGTH]),
            ("diameter_conflict", &[RING]),
        ];
```

then replace

```rust
            // The guides that raise and lower each such input carry its note; no other does.
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                let mut wanted = Vec::new();
                for &path in &both_ways {
                    let by = |direction| {
                        let at = all.iter().position(|f| f.contains(&(path, direction)));
                        CHECK_GUIDES[at.unwrap()].label
                    };
                    if all[index].iter().any(|&(fixed, _)| fixed == path) {
                        wanted.push(note(path, by(Direction::Raise), by(Direction::Lower)));
                    }
                }
                assert_eq!(trade_offs(&all, index), wanted, "{name}: {}", g.check);
            }
        }
        // The two cases in words: the same note in both blocks.
        let length = design(|i| i.metal.max_overall_axial_mm = 30.0);
        let results = compute_all(&length);
        let all = all_fixes(&length, &results, false);
        let expected = [
            "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Space claim needs it lowered, so change another input to settle one of them.",
        ];
        let at = |check| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        assert_eq!(trade_offs(&all, at(HOT_MIN)), expected);
        assert_eq!(trade_offs(&all, at(SPACE_CLAIM)), expected);
```

with

```rust
            // The guides that raise and lower each such input carry its note, one for each guide
            // that pulls it the other way; no other guide does.
            let pulling = |path: &str, direction: Direction| -> Vec<&str> {
                CHECK_GUIDES
                    .iter()
                    .zip(&all)
                    .filter(|(_, fixes)| fixes.contains(&(path, direction)))
                    .map(|(g, _)| g.label)
                    .collect()
            };
            for (index, g) in CHECK_GUIDES.iter().enumerate() {
                let mut wanted = Vec::new();
                for &(path, direction) in &all[index] {
                    if !both_ways.contains(&path) {
                        continue;
                    }
                    match direction {
                        Direction::Raise => {
                            for lowering in pulling(path, Direction::Lower) {
                                wanted.push(note(path, g.label, lowering));
                            }
                        }
                        Direction::Lower => {
                            for raising in pulling(path, Direction::Raise) {
                                wanted.push(note(path, raising, g.label));
                            }
                        }
                        Direction::Change | Direction::CustomHotterGrade => {}
                    }
                }
                assert_eq!(trade_offs(&all, index), wanted, "{name}: {}", g.check);
            }
        }
        // The cases in words. The overall-length claim: the space claim's note in its block and
        // in the hot-low torque's, which also carries the gearbox check's (it lowers the
        // magnets too).
        let at = |check| CHECK_GUIDES.iter().position(|g| g.check == check).unwrap();
        let length = design(|i| i.metal.max_overall_axial_mm = 30.0);
        let results = compute_all(&length);
        let all = all_fixes(&length, &results, false);
        let space = "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Space claim needs it lowered, so change another input to settle one of them.";
        let gearbox = note(MAGNET_LENGTH, HOT_MIN_GUIDE.label, GEARBOX_GUIDE.label);
        assert_eq!(trade_offs(&all, at(HOT_MIN)), [gearbox.as_str(), space]);
        assert_eq!(trade_offs(&all, at(GEARBOX)), [gearbox.as_str()]);
        assert_eq!(trade_offs(&all, at(SPACE_CLAIM)), [space]);
        // The example design: the hot-low torque against the gearbox check alone, the same note
        // in both blocks.
        let defaults = DesignInputs::default();
        let all = all_fixes(&defaults, &compute_all(&defaults), false);
        assert_eq!(trade_offs(&all, at(HOT_MIN)), [gearbox.as_str()]);
        assert_eq!(trade_offs(&all, at(GEARBOX)), [gearbox.as_str()]);
```

(The diameter-conflict lines after it stay as they are; they reuse `at`.)

(9) `each_problem_is_some_exactly_when_its_check_fails` (`:3461-3474`): replace
`// The passing design passes every guide; the defaults fail three, all red.` with
`// The passing design passes every guide; the defaults fail four, all red.`, and
`assert_eq!(failing, [(HOT_MIN, Bad), (CLEARANCE, Bad), (CUP_WALL, Bad)]);` with

```rust
        assert_eq!(
            failing,
            [
                (HOT_MIN, Bad),
                (CLEARANCE, Bad),
                (CUP_WALL, Bad),
                (GEARBOX, Bad)
            ]
        );
```

(10) `the_problems_quote_numbers_from_the_results` (it asserts one case per guide): insert after the `CLAMP` case (after
`&["clamps.required_Nm", "clamps.max_torque_Nm"],` / `),`, `:3551`). Both designs fail the check and both will show T3's
sentence 2, with different numbers (cold-high at the input 0.8049 and 0.7344 N·m; cold-high over hot-low 1.673 and
1.527), so T3 extends this same case:

```rust
            (
                GEARBOX,
                DesignInputs::default(),
                design(|i| i.metal.min_temp_C = 0.0),
                &["metal.cold_high_input_Nm"],
            ),
```

(11) `the_problems_quote_the_live_numbers` (`:3626-3628`): replace

```rust
        assert!(text.starts_with("Radial running gap 0.7768 mm"), "{text}");
        assert!(text.contains("leaves -0.003174 mm"), "{text}");
    }
```

with

```rust
        assert!(text.starts_with("Radial running gap 0.7768 mm"), "{text}");
        assert!(text.contains("leaves -0.003174 mm"), "{text}");
        // The gearbox check: cold-high at the gearbox input and the rating, each as its readout
        // shows it; the rating follows an edit.
        let results = compute_all(&defaults);
        let quoted = [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&defaults, "coupling.gearbox_input_rating_Nm"),
        ];
        assert_eq!(quoted, ["0.8049 N·m", "0.6000 N·m"]);
        let text = problem(GEARBOX, &defaults).unwrap();
        for number in &quoted {
            assert!(text.contains(number.as_str()), "{number} in {text}");
        }
        let rated = design(|i| i.coupling.gearbox_input_rating_Nm = 0.7);
        let text = problem(GEARBOX, &rated).unwrap();
        assert!(text.contains("0.7000 N·m"), "{text}");
    }
```

(12) New test: insert right before `#[test]` / `fn a_check_that_does_not_apply_gives_no_problem() {` (`:4001`):

```rust
    #[test]
    fn the_gearbox_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made() {
        // Red above the rating; amber when a torque it compares is not a number (the rating,
        // or cold-high at the gearbox input through a ratio set on the struct), with the text
        // that says so; green at or under the rating and while the end-effect model is out of
        // range (the row is greyed).
        use Level::{Bad, Caution, Good};
        let gearbox = guide(GEARBOX).unwrap();
        for (name, inputs, want) in [
            ("defaults", DesignInputs::default(), Bad),
            ("passing", passing_design(), Good),
            ("gearbox only", only_the_gearbox_fails(), Bad),
            ("unknown rating", breaking(unknown_rating), Caution),
            (
                "unknown ratio",
                breaking(|i| i.coupling.gear_ratio = f64::NAN),
                Caution,
            ),
            ("short magnets", short_magnets(), Good),
            ("40 poles", with_poles(40), Good),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(level(gearbox, &inputs, &results), want, "{name}");
            let text = (gearbox.problem)(&inputs, &results);
            match want {
                Good => assert_eq!(text, None, "{name}"),
                Caution => assert_eq!(text.as_deref(), Some(GEARBOX_UNKNOWN), "{name}"),
                Bad => {
                    let text = text.unwrap_or_else(|| panic!("{name}: no Problem"));
                    for number in [
                        result_text(&results, "metal.cold_high_input_Nm"),
                        input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
                    ] {
                        assert!(text.contains(&number), "{name}: {number} in {text}");
                    }
                }
            }
        }
    }

```

(13) `the_design_status_counts_the_failing_checks` (`:4017-4065`): replace

```rust
            DesignStatus {
                bad: 3,
                caution: 0,
                total: 8,
                is_default: true
            }
        );
        assert_eq!(
            design_status_text(&defaults),
            "Example design: fails 3 of 8 checks"
        );
        // One fix: the cup wall at its suggested 2.0 mm.
        let fixed = status(&design(|i| i.metal.cup_wall_corner_mm = 2.0));
        assert_eq!(
            design_status_text(&fixed),
            "Your design: fails 2 of 8 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 8 checks"
        );
        // Cautions are counted apart from the failures.
        assert_eq!(
            design_status_text(&status(&design(hot_day))),
            "Your design: fails 3 of 8 checks, 1 caution"
        );
```

with

```rust
            DesignStatus {
                bad: 4,
                caution: 0,
                total: 9,
                is_default: true
            }
        );
        assert_eq!(
            design_status_text(&defaults),
            "Example design: fails 4 of 9 checks"
        );
        // One fix: the cup wall at its suggested 2.0 mm.
        let fixed = status(&design(|i| i.metal.cup_wall_corner_mm = 2.0));
        assert_eq!(
            design_status_text(&fixed),
            "Your design: fails 3 of 9 checks"
        );
        assert_eq!(
            design_status_text(&status(&passing_design())),
            "Your design: passes all 9 checks"
        );
        // Cautions are counted apart from the failures.
        assert_eq!(
            design_status_text(&status(&design(hot_day))),
            "Your design: fails 4 of 9 checks, 1 caution"
        );
```

and replace

```rust
        assert_eq!(
            design_status_text(&status(&two_cautions)),
            "Your design: fails 3 of 8 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 8 checks"
        );
        // Any edit leaves the example, even one that keeps the counts.
        let edited = status(&design(|i| i.metal.hardware_g = 7.0));
        assert_eq!((edited.bad, edited.is_default), (3, false));
```

with

```rust
        assert_eq!(
            design_status_text(&status(&two_cautions)),
            "Your design: fails 4 of 9 checks, 2 cautions"
        );
        // The Magnets fit row counts as a check: 40 blocks fail it, the hot-low torque and the
        // clearance, and pass the cup wall and the gearbox check.
        assert_eq!(
            design_status_text(&status(&with_poles(40))),
            "Your design: fails 3 of 9 checks"
        );
        // Any edit leaves the example, even one that keeps the counts.
        let edited = status(&design(|i| i.metal.hardware_g = 7.0));
        assert_eq!((edited.bad, edited.is_default), (4, false));
```

(14) `a_design_failing_only_the_clamp_passes_while_the_section_is_paused` (`:4108-4125`): replace

```rust
        assert_eq!((shown.bad, shown.caution, shown.total), (1, 0, 8));
        assert_eq!(
            design_status_text(&shown),
            "Your design: fails 1 of 8 checks"
        );
        let paused = design_status(&inputs, &results, true);
        assert_eq!((paused.bad, paused.caution, paused.total), (0, 0, 7));
        assert_eq!(
            design_status_text(&paused),
            "Your design: passes all 7 checks"
        );
        // The example design fails the same three checks either way.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 3 of 7 checks"
        );
```

with

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
        );
        // The example design fails the same four checks either way.
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        assert_eq!(
            design_status_text(&design_status(&defaults, &results, true)),
            "Example design: fails 4 of 8 checks"
        );
```

(15) New test (the review focus: both counts): insert right before `#[test]` /
`fn a_paused_clamp_offers_no_fix_and_holds_back_no_other_guide_s_fix() {` (`:4128`):

```rust
    #[test]
    fn the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not() {
        // Owner decision 3A: counted, and red above the rating. The example design fails it
        // beside the hot minimum, the clearance and the cup wall; the passing design passes it,
        // and with the example's rating back fails it alone; a rating that is not a number asks
        // for a look. The clamp section's pause takes only the clamp out of the count.
        let status = |inputs: &DesignInputs, paused: bool| {
            design_status_text(&design_status(inputs, &compute_all(inputs), paused))
        };
        let defaults = DesignInputs::default();
        assert_eq!(
            status(&defaults, true),
            "Example design: fails 4 of 8 checks"
        );
        assert_eq!(
            status(&defaults, false),
            "Example design: fails 4 of 9 checks"
        );
        let passing = passing_design();
        assert_eq!(status(&passing, true), "Your design: passes all 8 checks");
        assert_eq!(status(&passing, false), "Your design: passes all 9 checks");
        let gearbox_only = only_the_gearbox_fails();
        let results = compute_all(&gearbox_only);
        let failing: Vec<(&str, Level)> = CHECK_GUIDES
            .iter()
            .map(|g| (g.check, level(g, &gearbox_only, &results)))
            .filter(|&(_, badge)| badge != Level::Good)
            .collect();
        assert_eq!(failing, [(GEARBOX, Level::Bad)]);
        assert_eq!(
            status(&gearbox_only, true),
            "Your design: fails 1 of 8 checks"
        );
        assert_eq!(
            status(&gearbox_only, false),
            "Your design: fails 1 of 9 checks"
        );
        assert_eq!(
            status(&breaking(unknown_rating), true),
            "Your design: passes all 8 checks, 1 caution"
        );
    }

```

(16) `raising_the_face_gap_says_it_lowers_the_failing_hot_low_torque` (`:4930-4935`): the gearbox guide shows the same
Raise face gap fix at the defaults, beside the failing hot-low torque, so it gets the note too. Replace

```rust
        // No other guide gets the note, and an index past the guides gets none.
        let results = compute_all(&defaults);
        let all = all_fixes(&defaults, &results, false);
        for index in (0..=CHECK_GUIDES.len()).filter(|&i| i != clearance) {
            assert!(side_effects(&all, index, &defaults, &results).is_empty());
        }
```

with

```rust
        // The gearbox guide's gap fix, shown at the defaults beside the failing hot-low torque,
        // brings the same note; no other guide gets it, and an index past the guides gets none.
        let gearbox = CHECK_GUIDES
            .iter()
            .position(|g| g.check == GEARBOX)
            .unwrap();
        let results = compute_all(&defaults);
        let all = all_fixes(&defaults, &results, false);
        assert!(all[gearbox].contains(&gap), "{:?}", all[gearbox]);
        assert_eq!(side_effects(&all, gearbox, &defaults, &results), [note]);
        for index in (0..=CHECK_GUIDES.len()).filter(|&i| i != clearance && i != gearbox) {
            assert!(side_effects(&all, index, &defaults, &results).is_empty());
        }
```

**1f. `magcoupling-rs/src/gui/spreadsheet.rs`, `mod tests` (`:1000`).** The comment's count moves with the new failing
check. Replace

```rust
        // The default design, one with six failing checks (manual magnets) and one out of the
```

with

```rust
        // The default design, one with seven failing checks (manual magnets) and one out of the
```

**1d. `magcoupling-rs/src/gui/panel.rs`, `mod tests`.**

(1) `the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` (`:3032-3097`): replace
`// Manual magnets: the defaults' four red checks and two amber rating checks.` with
`// Manual magnets: the defaults' five red checks and two amber rating checks.`; replace

```rust
        assert_eq!(failing.len(), 6);
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("6 of {total} results")), 1);
```

with

```rust
        assert_eq!(failing.len(), 7);
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("7 of {total} results")), 1);
```

replace `// Four red badges and two amber in the table, in the dashboard's colours: every badge` with
`// Five red badges and two amber in the table, in the dashboard's colours: every badge`; replace

```rust
            (table_badges(Level::Bad), table_badges(Level::Caution)),
            (4, 2)
```

with

```rust
            (table_badges(Level::Bad), table_badges(Level::Caution)),
            (5, 2)
```

and, since the gearbox check now fails and its path matches "gearbox", give the no-match search a word only passing rows
match (the ripple row's path): replace

```rust
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
```

with

```rust
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("ripple".to_owned())]);
```

(Only this occurrence: the inputs filter test at `:2757` also types "gearbox" and stays.)

(2) `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too` (`:3100-3182`): replace

```rust
        // without badges. In the table's headline (open) each check row carries exactly the
        // dashboard's badge, so the hot minimum (red by its text) and the screw sized from the
        // invalid pull-out (green by its text) carry none, and the failing filter leaves the hot
        // minimum out.
```

with

```rust
        // without badges. In the table's headline (open) each check row carries exactly the
        // dashboard's badge, so the hot minimum (red by its text), the screw sized from the
        // invalid pull-out and the gearbox check of a cold-high below zero (both green by their
        // texts) carry none, and the failing filter leaves the hot minimum out.
```

and `assert_eq!(checks.len(), 7);` with `assert_eq!(checks.len(), 8);`.

(3) `the_status_line_counts_the_checks_at_default_and_after_a_fix` (`:8682-8705`): replace

```rust
        // The workbook example fails its hot minimum, its clearance and its cup wall.
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (3, 0, 8, true)
        );
```

with

```rust
        // The workbook example fails its hot minimum, its clearance, its cup wall and its
        // gearbox check.
        assert_eq!(
            (status.bad, status.caution, status.total, status.is_default),
            (4, 0, 9, true)
        );
```

then `assert_eq!((status.bad, status.is_default), (2, false));` with `assert_eq!((status.bad, status.is_default), (3, false));`
and `assert_eq!((status.bad, status.is_default), (3, true));` with `assert_eq!((status.bad, status.is_default), (4, true));`.

**1e. `magcoupling-rs/src/gui/results_table.rs`**, `the_engine_order_and_the_failing_filter_are_flat` (`:1282`):
`assert_eq!(failing.len(), 6);` becomes `assert_eq!(failing.len(), 7);` (manual magnets: five red, two amber).

- [ ] **Step 2: Run the tests to see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gearbox`
Expected: FAIL to compile, 21 errors, all in `src/gui/guidance.rs`'s tests:
`error[E0425]: cannot find value \`GEARBOX\` in this scope` (18), `cannot find value \`GEARBOX_UNKNOWN\` in this scope`
(2), `cannot find value \`GEARBOX_GUIDE\` in this scope` (1).

(Beyond the compile errors: with only those three names declared as placeholders, 25 lib tests fail by assertion,
among them the four new or replaced tests and `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`,
which T1 leaves red.)

- [ ] **Step 3: Write the dashboard row, the check, its levels and its greying (`dashboard.rs`)**

(1) The module doc (`:5-8`): replace

```rust
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the A1 space
//! claim badge and the audit M9 end-effect flag. A row's badge comes from a check verdict
//! ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
```

with

```rust
//! [`DASHBOARD`] lists its rows: the 15 headline numbers in Python's order, then the gearbox
//! check (BL-059), the A1 space claim badge and the audit M9 end-effect flag. A row's badge
//! comes from a check verdict ([`verdict_level`]); its marker from the deviation registry
//! ([`crate::gui::corrections`]). Two notes from the user's decisions of 2026-09-30:
```

(2) `DASHBOARD` (`:155`): `pub const DASHBOARD: [(&str, Option<&str>); 17] = [` becomes
`pub const DASHBOARD: [(&str, Option<&str>); 18] = [`, and after `("clamps.recommended", Some("clamps.recommended")),`
insert

```rust
    ("gearbox.cold_high_check", Some("gearbox.cold_high_check")),
```

(rustfmt keeps this row on one line.)

(3) `END_EFFECT_ROWS` (`:187-200`): replace

```rust
/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the
/// headline rows that move with the end-effect coefficient (a test probes it at back iron 1
/// and 0). Per-row greying of the results table waits for plan A-3's dependency graph; the
/// table shows the banner instead (decision M41-11), and its two checks among these rows (the
/// hot minimum, the recommended screw) give no badge there either ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 7] = [
```

with

```rust
/// The dashboard rows computed from the pull-out, greyed when f_end ≤ 0 (audit M9): the rows
/// that move with the end-effect coefficient (a test probes them at back iron 1 and 0), the
/// headline's and the gearbox check, computed from cold-high. Per-row greying of the results
/// table waits for plan A-3's dependency graph; the table shows the banner instead (decision
/// M41-11), and its three checks among these rows (the hot minimum, the recommended screw, the
/// gearbox check) give no badge there either ([`greyed_by_end_effect`]).
pub const END_EFFECT_ROWS: [&str; 8] = [
```

and after its last entry `"clamps.recommended",` insert `"gearbox.cold_high_check",` (before `];`).

(4) `CHECKS` (`:254-291`): replace

```rust
/// screens, the material warnings and the space claim. Each gives a badge level
/// ([`verdict_level`]): the results table shows it, and its failing filter lists the checks
/// that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps' status
/// texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 30] = [
```

with

```rust
/// screens, the material warnings, the space claim and the gearbox check. Each gives a badge
/// level ([`verdict_level`]): the results table shows it, and its failing filter lists the
/// checks that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps'
/// status texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 31] = [
```

and after its last entry `"housing.space_claim_check",` insert `"gearbox.cold_high_check",` (schema order: the `gearbox`
group follows `housing` in `DesignResults`).

(5) `verdict_level` (`:370`): after
`("housing.space_claim_check", t) if t.starts_with("Exceeds the space claim:") => Some(Bad),` insert

```rust
        ("gearbox.cold_high_check", crate::engine::gearbox::WITHIN_THE_GEARBOX_RATING) => {
            Some(Good)
        }
        ("gearbox.cold_high_check", crate::engine::gearbox::GEARBOX_CHECK_UNKNOWN) => Some(Caution),
        ("gearbox.cold_high_check", t)
            if t.starts_with(crate::engine::gearbox::ABOVE_THE_GEARBOX_RATING) =>
        {
            Some(Bad)
        }
```

(rustfmt braces the Good arm, as shown.)

- [ ] **Step 4: Write the guide core (`guidance.rs`)**

(1) The check consts (`:144`): after `const CLAMP: &str = "clamps.recommended";` insert

```rust
const GEARBOX: &str = "gearbox.cold_high_check";
```

(2) After `CLAMP_GUIDE` (after its closing `};` at `:212`, before `const SPACE_CLAIM_GUIDE`) insert

```rust

const GEARBOX_GUIDE: CheckGuide = CheckGuide {
    check: GEARBOX,
    label: "Gearbox input torque",
    problem: gearbox_problem,
    why: "Up to cold-high torque reaches the gearbox output before the coupling slips, and in each swing while it slips; at the gearbox input it must stay within the rating.",
    // Geometry only (owner decision 6A): both lower cold-high, and hot-low with it, which the
    // trade-off and side-effect lines say where the hot minimum fails. Not the variation (an
    // assumption), the minimum temperature (an environment fact), or the rating or ratio (no
    // real gearbox changes one alone): the Problem names them where no geometry passes both.
    fixes: Fixes::Static(&[
        (MAGNET_LENGTH, Direction::Lower),
        (FACE_GAP, Direction::Raise),
    ]),
    learn: Learn::Note(TORQUE_BAND_NOTE),
};
```

(3) `CHECK_GUIDES` (`:290-302`): replace

```rust
/// The guides: the seven dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 8] = [
    HOT_MIN_GUIDE,
    CLEARANCE_GUIDE,
    CUP_WALL_GUIDE,
    TEMPERATURE_GUIDE,
    CLAMP_GUIDE,
    SPACE_CLAIM_GUIDE,
```

with

```rust
/// The guides: the eight dashboard checks in the dashboard's order, then the Magnets fit row.
/// Every Why and Problem template goes through the slice's physics review (spec G-3), and none
/// says what spec section 8 rules out.
pub static CHECK_GUIDES: [CheckGuide; 9] = [
    HOT_MIN_GUIDE,
    CLEARANCE_GUIDE,
    CUP_WALL_GUIDE,
    TEMPERATURE_GUIDE,
    CLAMP_GUIDE,
    GEARBOX_GUIDE,
    SPACE_CLAIM_GUIDE,
```

(4) `ROW_LEARN` (`:343`): after `(CLAMP, CLAMP_GUIDE.learn),` insert `(GEARBOX, GEARBOX_GUIDE.learn),`.

(5) `design_status_text`'s doc (`:1481-1484`): replace

```rust
/// The design status line: "Example design: fails 3 of 8 checks", "Your design: fails
/// 2 of 8 checks", "Your design: passes all 8 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 7 while the shaft-clamp section is paused.
```

with

```rust
/// The design status line: "Example design: fails 4 of 9 checks", "Your design: fails
/// 3 of 9 checks", "Your design: passes all 9 checks", each followed by ", 1 caution" or ", 2
/// cautions" when a check asks for a look. The count is of the guides that count ([`counts`]):
/// 8 while the shaft-clamp section is paused.
```

(6) The Problem, in the guides' order: insert right before
`fn space_claim_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {` (`:1622`)

```rust
/// The gearbox check's Problem when it cannot be made (it asks for a look): a value it
/// compares is not a number.
const GEARBOX_UNKNOWN: &str = "The gearbox check cannot be made: cold-high at the gearbox input or the rating is not a number.";

fn gearbox_problem(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    match level_of(GEARBOX, inputs, results) {
        Level::Good => None,
        Level::Caution => Some(GEARBOX_UNKNOWN.to_owned()),
        Level::Bad => Some(format!(
            "Cold-high at the gearbox input is {}, above its {} input rating.",
            result_text(results, "metal.cold_high_input_Nm"),
            input_text(inputs, "coupling.gearbox_input_rating_Nm"),
        )),
    }
}

```

(`Level` has exactly `Good`, `Caution` and `Bad`: the match is exhaustive as written.)

- [ ] **Step 5: Run the tests to see them pass**

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`
Expected: no output (the blocks above are its output).
Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`
Expected: `test result: ok. 855 passed; 0 failed; 6 ignored` (T1's 851 passing and its 1 expected failure, now green,
plus three new tests; the tripwire is replaced one for one).

- [ ] **Step 6: Run both suites**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every binary `ok`, `0 failed`: the lib 855 passed, 6 ignored; `tests/gearbox.rs` 11 passed; `tests/explain.rs`
56 passed, 1 ignored (Step 8 adds its 57th); every other binary at Task 1's Step 8 counts. `tests/explain.rs` passes
unchanged: its two gearbox watch-out tests still hold until T6 rewrites them.
Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every binary `ok`, `0 failed`, at Task 1's Step 8 counts (lib 228). No engine-only test moves in this task.

- [ ] **Step 7: Lint and format**

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
Expected: no warnings.
Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: no warnings.
Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check`
Expected: no output.

- [ ] **Step 8: The docs-count guard, then the docs (T2's lines; each anchor below is unique in its file)**

**8a. A permanent guard for the counts.** The README's "dashboard's N rows" and "every one of the N checks" and
`03-structure.yaml`'s "CHECKS (N design checks)" are hand-kept counts that went stale before. In
`magcoupling-rs/tests/explain.rs`, insert this test directly before `#[test]` /
`fn the_docs_count_the_notes_and_the_steps() {` (`:1241`; the file has CRLF line ends, which the Edit tool keeps):

```rust
/// BL-059 (the F36 lesson: counts in the docs went stale before): the README and
/// `03-structure.yaml` state the dashboard's rows and the design checks as the code counts them.
#[cfg(feature = "gui")]
#[test]
fn the_docs_count_the_dashboard_rows_and_the_checks() {
    use magcoupling::gui::dashboard::{CHECKS, DASHBOARD};
    let readme = include_str!("../README.md");
    let structure = include_str!("../../docs/ai/03-structure.yaml");
    for (doc, text, claim) in [
        (
            "README.md",
            readme,
            format!("dashboard's {} rows", DASHBOARD.len()),
        ),
        (
            "README.md",
            readme,
            format!("every one of the {} checks", CHECKS.len()),
        ),
        (
            "03-structure.yaml",
            structure,
            format!("CHECKS ({} design checks)", CHECKS.len()),
        ),
    ] {
        assert!(text.contains(&claim), "{doc} should say {claim:?}");
    }
}
```

Run (before the docs edits): `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count`
Expected: `test the_docs_count_the_dashboard_rows_and_the_checks ... FAILED` with
`README.md should say "dashboard's 18 rows"`; `test result: FAILED. 1 passed; 1 failed`
(`the_docs_count_the_notes_and_the_steps` passes).

**8b. The docs.**

`magcoupling-rs/README.md`:

- `:300-301`, replace

```
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the space claim
  badge and the end-effect flag (`DASHBOARD`).
```

with

```
- **Dashboard** (`dashboard.rs`): the 15 headline numbers in Python's order, the gearbox check
  (cold-high at the gearbox input against the gearbox input rating), the space claim badge and
  the end-effect flag (`DASHBOARD`).
```

(the rest of `:301`, ` Green, amber or red badges come from the check`, stays on the new third line).

- `:317`, `  dashboard, "Example design: fails N of 8 checks" while the inputs are the defaults (its hover:` becomes
  `  dashboard, "Example design: fails N of 9 checks" while the inputs are the defaults (its hover:`.
- `:319`, `  "Your design: fails N of 8 checks" or "passes all 8 checks" after an edit, with ", N cautions"` becomes
  `  "Your design: fails N of 9 checks" or "passes all 9 checks" after an edit, with ", N cautions"`.
- `:320-322`, replace

```
so the line reads "of 7 checks"). The eight guides are the
  dashboard's seven checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
  the recommended clamp, the space claim and the end-effect flag) and a **Magnets fit** row
```

with

```
so the line reads "of 8 checks": the example design "fails 4 of 8 checks"). The nine guides are the
  dashboard's eight checks (the hot minimum, the clearance, the cup wall, the thermal verdict,
  the recommended clamp, the gearbox check, the space claim and the end-effect flag) and a **Magnets fit** row
```

- `:343`, replace

```
  so above it only a hotter magnet or another adhesive helps. Every row, whatever its level, has a **Why:** link: a reviewed teaching note (the hot-low, hot-minimum and cold-high rows lead to "Hot-low and cold-high: the torque band", the gearbox ripple row to "When the coupling slips"), an
```

with

```
  so above it only a hotter magnet or another adhesive helps. The gearbox check's guide (BL-059)
  offers the geometry only, a shorter magnet length and a wider face gap; both lower hot-low too,
  so on the example design its block and the hot minimum's each say the magnet length pulls two
  ways, and its face-gap fix carries the face-gap note. Every row, whatever its level, has a **Why:** link: a reviewed teaching note (the hot-low, hot-minimum, cold-high and gearbox check rows lead to "Hot-low and cold-high: the torque band", the gearbox ripple row to "When the coupling slips"), an
```

- `:353`, `  dashboard's 17 rows, open; then the closed chain groups` becomes `  dashboard's 18 rows, open; then the closed chain groups`.
- `:362`, replace

```
  0 the hot minimum and the recommended screw, computed from the invalid pull-out, carry none, as
```

with

```
  0 the hot minimum, the recommended screw and the gearbox check, computed from the invalid
  pull-out, carry none, as
```

- `:656` (the `tests/explain.rs` test row), replace

```
`the_docs_count_the_notes_and_the_steps` holds this README's and
```

with

```
`the_docs_count_the_dashboard_rows_and_the_checks` (feature `gui`) holds this README's dashboard row and check counts and `docs/ai/03-structure.yaml`'s CHECKS count to the code. `the_docs_count_the_notes_and_the_steps` holds this README's and
```

  (This anchor is clear of T6's three anchors on the same row.)
- `:663` (the `src/gui/panel.rs` test row): `the failing filter shows exactly the failing checks with four red and two amber badges`
  becomes `the failing filter shows exactly the failing checks with five red and two amber badges`, and
  `carry exactly the dashboard's badges (none on the hot minimum and the screw) and the failing filter` becomes
  `carry exactly the dashboard's badges (none on the hot minimum, the screw and the gearbox check) and the failing filter`.
- `:667` (the `src/gui/dashboard.rs` test row), four replacements:
  - `The dashboard starts with `HEADLINE`; every verdict each check gives has a level (designs that reach each branch); `END_EFFECT_ROWS` are exactly the headline rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0,`
    becomes
    `The dashboard starts with `HEADLINE`, then the gearbox check, the space claim and the end-effect flag; every verdict each check gives has a level (designs that reach each branch; the gearbox check's texts also from its engine function at the rating, above it, at an infinite torque and with a value that is not a number); `END_EFFECT_ROWS` are exactly the dashboard rows that move with c_end (0 included: it flips the hot-minimum verdict) at back iron 1 and 0, the headline's and the gearbox check,`
  - `every one of the 30 checks is a text result` becomes `every one of the 31 checks is a text result`;
  - `out of the end-effect range the two checks among the greyed rows give no level and are not failing,` becomes
    `out of the end-effect range the three checks among the greyed rows (the hot minimum, the recommended screw, the gearbox check) give no level and are not failing, and the gearbox check greys whatever its text reads (a cold-high below zero reads within the rating, a rating set below it reads above): no badge on the dashboard or in the table, not failing, no Problem and no count in the status line;`
  - `design that fails only the clamp ("passes all 7 checks")` becomes `design that fails only the clamp ("passes all 8 checks")`.
- `:670` (the `src/gui/guidance.rs` test row), two replacements:
  - `a design that fails only the clamp passes all 7 counted checks while paused and fails 1 of 8 when not` becomes
    `a design that fails only the clamp passes all 8 counted checks while paused and fails 1 of 9 when not`;
  - `which explains each row's equation; no dashboard row, guide or check compares a gearbox torque with the gearbox rating (the note's watch-out says so until BL-059).`
    becomes
    `which explains each row's equation. BL-059: the gearbox check is one dashboard row right after the headline's, in `CHECKS`, with a guide between the clamp's and the space claim's whose Why link, as its row's, leads to the torque-band note (`the_gearbox_check_is_a_dashboard_row_with_its_guide`, which replaces the slice-2 test that no row, guide or check compared a gearbox torque with the rating); its Problem shows exactly when it fails or cannot be made, quoting cold-high at the gearbox input and the rating (the unknown text for a rating or ratio that is not a number); it counts in the status line whether the clamp section shows or not (the example design fails 4 of 8 checks while paused and 4 of 9 with it; the passing design, its gearbox rating 1.0 N·m, passes all 8, and with the example's rating back fails the gearbox check alone); its two fixes move it toward passing over their sliders; on the example design it and the hot minimum pull the magnet length apart, one note for each guide pulling the other way, and its face-gap fix carries the face-gap note.`
  - Leave the next sentence's clause "that note's first watch-out also says nothing checks the gearbox" (Slice 2, T15)
    for T6: it stays true until T6 rewrites that watch-out.

`docs/ai/02-system.yaml:107`, two replacements inside the quoted string:
- `CHECK_GUIDES holds one CheckGuide for each of the seven dashboard checks (metal.hot_min_check, metal.clearance_check, materials.cup_wall_check, temperature.summary.verdict, clamps.recommended, housing.space_claim_check, model.end_effect_check)`
  becomes
  `CHECK_GUIDES holds one CheckGuide for each of the eight dashboard checks (metal.hot_min_check, metal.clearance_check, materials.cup_wall_check, temperature.summary.verdict, clamps.recommended, gearbox.cold_high_check, housing.space_claim_check, model.end_effect_check)`;
- `design_status counts Bad and Caution rows over the eight guides` becomes `design_status counts Bad and Caution rows over the nine guides`.

`docs/ai/03-structure.yaml:66`, three replacements:
- `CHECKS (30 design checks)` becomes `CHECKS (31 design checks)`;
- `CHECK_GUIDES (the seven dashboard checks and FIT_CHECK)` becomes `CHECK_GUIDES (the eight dashboard checks and FIT_CHECK)`;
- `test designs only_the_clamp_fails (passing_design failing the clamp alone) and clamp_at_its_limit` becomes
  `test designs only_the_clamp_fails (passing_design failing the clamp alone), only_the_gearbox_fails (passing_design with the example design's gearbox rating, failing the gearbox check alone) and clamp_at_its_limit`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count`
Expected: `test result: ok. 2 passed; 0 failed`: the new guard reads 18 rows and 31 checks in both docs, and the note and
step counts `the_docs_count_the_notes_and_the_steps` reads are untouched.
Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain`
Expected: `test result: ok. 57 passed; 0 failed; 1 ignored`.
Run: `git diff --stat -- magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml`
Expected: three files changed, each edit above applied once (`docs/ai/02-system.yaml` already fails a strict YAML parse
at HEAD, near line 14, unrelated to this edit; do not reformat it).

- [ ] **Step 9: Commit Task 1's and this task's files (one commit, PD-4)**

Step 8 added a test, so run Step 7's three commands again (both clippy runs clean, `cargo fmt --check` silent). Then
write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T2.txt` (gitignored):

```
feat(magcoupler): the gearbox check, from the engine to the dashboard and its guide (BL-059 T1-T2)

Engine: a Rust-only result group, gearbox (engine/gearbox.rs, computed
beside housing::compute). gearbox.cold_high_check compares cold-high at
the gearbox input (metal.cold_high_input_Nm) with the gearbox input
torque rating, owner decision 1A: the same test as cold-high against the
workbook's 2.85 N·m reference at the output. At the rating passes, +inf
fails, NaN reads unknown; the badge quotes both torques to 0.01 N·m and
never two equal numbers. PACKAGE_LABELS gains "Gearbox"; the trace total
is 392 of 1087 results. tests/gearbox.rs (11 tests): the defaults, the
five probe designs, N33AH on both rings (fails the input basis that an
efficiency-free 3.0 N·m output limit would pass), equality and the next
float, NaN and +inf, the rating slider's ends, and the quoted numbers
over the whole slider.

The gearbox check (gearbox.cold_high_check: cold-high at the gearbox input
against the gearbox input rating) gets a dashboard row after the headline,
a CHECKS entry and its three verdict levels, greys with the end-effect rows,
and a counted guide between the clamp's and the space claim's: its label,
Problem sentence 1 and the unknown text, the Why, the geometry fixes (a
shorter magnet length, a wider face gap) and the torque-band note as its
Why link. The example design now fails 4 of 8 checks while the shaft-clamp
section is paused (4 of 9 with it shown).

The test that said nothing compares a gearbox torque with the rating is
replaced, in this commit, by the_gearbox_check_is_a_dashboard_row_with_its_guide.
passing_design() gets a 1.0 N·m gearbox rating so it still passes every
check; only_the_gearbox_fails() puts the example's rating back.
the_docs_count_the_dashboard_rows_and_the_checks holds the README's and
03-structure.yaml's row and check counts to the code.

The two notes' watch-outs still say nothing checks the gearbox; T6 rewrites
them with their physics re-review (only the merge ships).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run (never `git add -A` or `git add .`; never stage `docs/analyses/2026-05-28-press-4bar-analysis.md` or
`docs/chebyshev_lambda/`):

```bash
git add magcoupling-rs/src/engine/gearbox.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/gui/trace.rs magcoupling-rs/tests/gearbox.rs magcoupling-rs/tests/schema.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/tests/explain.rs magcoupling-rs/src/lib.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T2.txt
git status --short
```

Expected: one commit (18 files: Task 1's ten and this task's); `git status --short` lists only
`?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

#### Pinned tests that move (each confirmed by running; old -> new, reason)

| test | old -> new | reason |
|---|---|---|
| `dashboard::the_dashboard_starts_with_the_headline_in_order` | `paths[15..]` [space claim, end effect] -> [gearbox, space claim, end effect] | the row sits right after the headline |
| `dashboard::the_failing_checks_are_the_red_then_the_amber` | defaults 4 -> 5 (+ gearbox Bad); manual magnets 6 -> 7 (+ gearbox Bad before the amber) | 0.8049 N·m at the input against 0.6 in both |
| `dashboard::the_end_effect_rows_are_the_headline_rows_that_move_with_the_end_effect_coefficient` | renamed `..._dashboard_rows_...`; helper `headline_moved_by` -> `dashboard_moved_by` over `DASHBOARD`; set equality kept; + the only row past the headline is the gearbox check | the check moves with c_end but is no headline row (decision 10A) |
| `dashboard::out_of_range_end_effect_drops_the_pull_out_checks_from_the_table_and_the_failing_filter` | greyed [hot min, clamp] -> [hot min, clamp, gearbox]; + in range the gearbox is red and not greyed | `END_EFFECT_ROWS` gains the check |
| `dashboard::while_the_clamp_section_is_paused_its_row_and_guide_leave_the_dashboard` | "passes all 7 checks" -> "passes all 8 checks" | 8 guides count while paused |
| `dashboard::a_red_check_shows_its_block_an_amber_one_its_header_and_every_row_its_why_link` | the example's red blocks 3 -> 4 (+ gearbox) | the example fails the check |
| `dashboard::the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour` | defaults (3,0) -> (4,0); cup wall 2.4 (2,0) -> (3,0); hot day (3,1) -> (4,1); + `only_the_gearbox_fails()` (1,0, red) | same |
| `dashboard::the_stored_3d_rows_are_...`, `the_stored_3d_note_says_...` | call `dashboard_moved_by`; assertions unchanged | helper renamed; the stored 3D inputs move no row past the headline |
| `guidance::nothing_checks_a_gearbox_torque_against_its_rating_yet` | replaced by `the_gearbox_check_is_a_dashboard_row_with_its_guide` | the tripwire's purpose (spec risk 1) |
| `guidance::every_fix_moves_its_check_toward_passing_over_its_slider` | + 2 cases (gearbox: Lower length, Raise gap, from the defaults) | `listed == slider_fixes`: the cases land with the guide |
| `guidance::each_problem_is_some_exactly_when_its_check_fails` | defaults' failing [HOT_MIN, CLEARANCE, CUP_WALL] -> + (GEARBOX, Bad) | the example fails the check |
| `guidance::the_problems_quote_numbers_from_the_results` | + a GEARBOX case (defaults vs minimum 0 °C, key `metal.cold_high_input_Nm`) | the test requires one case per guide |
| `guidance::two_failing_checks_that_pull_one_input_apart_say_so` | pulled-apart designs 2 -> 4 (+ defaults, bare_clearance: length); notes built per guide pulling the other way; length_conflict: HOT_MIN [space] -> [gearbox, space], + GEARBOX [gearbox]; + the defaults pair | the hot minimum raises the length, the gearbox check lowers it (decision 4A's conflict) |
| `guidance::the_design_status_counts_the_failing_checks` | {3,0,8} -> {4,0,9}; "3 of 8" -> "4 of 9"; "2 of 8" -> "3 of 9"; "all 8" -> "all 9"; hot day "3 of 8, 1 caution" -> "4 of 9, 1 caution"; "3 of 8, 2 cautions" -> "4 of 9, 2 cautions"; 40 poles "3 of 8" -> "3 of 9" (the gearbox passes at 40 poles, 0.2980 N·m); edited (3,false) -> (4,false) | one more guide, failing at the defaults |
| `guidance::a_design_failing_only_the_clamp_passes_while_the_section_is_paused` | (1,0,8) -> (1,0,9); "1 of 8" -> "1 of 9"; (0,0,7) -> (0,0,8); "all 7" -> "all 8"; "Example design: fails 3 of 7" -> "4 of 8" | same |
| `guidance::raising_the_face_gap_says_it_lowers_the_failing_hot_low_torque` | "no other guide gets the note" -> the gearbox guide gets it too at the defaults (asserted), no other | its Raise face gap fix shows beside the failing hot-low torque |
| `panel::out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too` | dashboard rows in `CHECKS` 7 -> 8 | the new row; its table badge is none, as the dashboard's |
| `panel::the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` | failing 6 -> 7; "6 of" -> "7 of"; table badges (4,2) -> (5,2); the no-match search "gearbox" -> "ripple" | the check fails on manual magnets and its path matches "gearbox" |
| `panel::the_status_line_counts_the_checks_at_default_and_after_a_fix` | (3,0,8,true) -> (4,0,9,true); after the cup wall fix 2 -> 3; reset 3 -> 4 | same |
| `results_table::the_engine_order_and_the_failing_filter_are_flat` | failing on manual magnets 6 -> 7 | same |

Extended, not moved: `every_verdict_each_check_gives_is_classified` (+3 design cases, +5 engine-text cases),
`every_verdict_of_every_other_check_is_classified` (+2 cross-check negatives), `the_problems_quote_the_live_numbers`
(+ the gearbox's 0.8049 and 0.6000 N·m, and 0.7000 N·m after an edit). New: `the_gearbox_check_is_a_dashboard_row_with_its_guide`,
`out_of_range_end_effect_greys_the_gearbox_check_whatever_its_text_reads`,
`the_gearbox_problem_is_some_exactly_when_the_check_fails_or_cannot_be_made`,
`the_gearbox_check_counts_in_the_status_whether_the_clamp_section_shows_or_not`, and in `tests/explain.rs`
`the_docs_count_the_dashboard_rows_and_the_checks`. A comment moves: `spreadsheet.rs:1000` (six failing checks ->
seven). Not moved, still true:
`the_status_text_reads_each_form` (a formatter test with a literal total of 8), the trace totals (T1's `(392, 1087)`),
the spreadsheet tests (they count `DASHBOARD.len()`).

---

### Task 3: Problem sentence 2, the conflict line (owner decision 4A)

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

Every code block below is rustfmt's layout, prototyped red to green when the plan was made. Plan deviation PD-1 (ruling
R-3) shapes this task: sentence 2 shows only with the minimum temperature at or below the operating one, and where the
requirement alone is above the rating at the gearbox input it offers the gearbox or the requirement and no band.

**Files:**
- Modify: `magcoupling-rs/src/gui/guidance.rs`: the Bad arm of `gearbox_problem` (T2's), a new private
  `fn gearbox_conflict` right after it; in `mod tests`: one import, the `ConflictNumbers` oracle and two assertion
  helpers, five new tests, a sentence-2 block at the end of `the_problems_quote_numbers_from_the_results`, and the
  `every_text()` helper extended to the second remedy form.
- Modify: `magcoupling-rs/README.md`: one sentence in the **Check guides** paragraph; the `src/gui/guidance.rs` row
  of the test table.
- Test: `magcoupling-rs/src/gui/guidance.rs` (`mod tests`).

**Interfaces:**
- Consumes: from T1 `metal.cold_high_input_Nm` and the `gearbox` group. From T2 `GEARBOX`, `GEARBOX_GUIDE`,
  `gearbox_problem` (its Bad arm as Task 2 writes it), `passing_design()` with the rating at 1.0,
  `only_the_gearbox_fails()` (the passing design with the example's rating, imported from `test_support` by T2), and the
  `GEARBOX` case of `the_problems_quote_numbers_from_the_results` (that test's every-guide set check needs it at T2).
  Already in the module: `level_of`, `level`, `result_text`, `input_text`, `number_text`, `format_value`,
  `with_number`, `with`, `design`, `breaking`, `designs`, `every_text`, `InputCatalogue`, `FreeVariable`, `BTreeSet`.
  Also `crate::engine::sizing::{solve, SizingOutcome}` and `crate::gui::sizing::sized` (the Size tab's solve for Auto
  on length is `solve_in` over the length slider, which is `solve`: `panel.rs` `sizing_range`).
- Produces: `fn gearbox_conflict(inputs: &DesignInputs, results: &DesignResults) -> Option<String>` (private),
  appended to the Bad Problem after one space. Test helpers: `ConflictNumbers`, `assert_conflict_quotes`,
  `assert_gearbox_problem_consistent`.

Precondition (from T2): no T2 test pins the whole gearbox Problem at a design where the conflict holds (the
defaults, and every design built on them: `bare_clearance`, `length_conflict`, `diameter_conflict`, `twelve_poles`,
`design(|i| i.metal.face_gap_mm = 1.6)`). There the Problem gains sentence 2 in this task. T2 pins sentence 1 with
`contains`, or on `only_the_gearbox_fails()`, where the Problem is sentence 1 alone.

- [ ] **Step 1: Add the test import**

In `magcoupling-rs/src/gui/guidance.rs`, `mod tests`, replace:

```rust
    use crate::engine::model::ring_fits;
```

with:

```rust
    use crate::engine::model::ring_fits;
    use crate::engine::sizing::{SizingOutcome, solve};
```

The gearbox-only design is T2's `only_the_gearbox_fails()`, already imported.

- [ ] **Step 2: Write the five failing tests and their helpers**

In `mod tests`, insert this block directly before:

```rust
    #[test]
    fn the_temperature_problem_names_each_failing_term() {
```

(that is, after `the_problems_quote_the_live_numbers`):

```rust
    // ------------------------------------------------------------------ the gearbox conflict

    /// Problem sentence 2's numbers (owner decision 4A), worked out here from a design's
    /// results and inputs as the sentence's own check: the band (cold-high over hot-low), the
    /// requirement, cold-high at the gearbox input of a design that just meets the
    /// requirement, the largest requirement and the widest band the rating allows, and whether
    /// the conflict holds (the minimum temperature at or below the operating one, hot-low above
    /// zero, and the requirement times the band, at the gearbox input, above the rating).
    struct ConflictNumbers {
        band: f64,
        required: f64,
        input_at_required: f64,
        largest_required: f64,
        widest_band: f64,
        holds: bool,
    }

    impl ConflictNumbers {
        fn of(inputs: &DesignInputs, results: &DesignResults) -> Self {
            let (c, md) = (&inputs.coupling, &results.metal);
            let gear = c.gear_ratio * c.gear_efficiency;
            let rating = c.gearbox_input_rating_Nm;
            let band = md.torque_cold_high_Nm / md.torque_hot_low_Nm;
            let required = inputs.metal.required_min_Nm;
            let input_at_required = required * band / gear;
            Self {
                band,
                required,
                input_at_required,
                largest_required: rating * gear / band,
                widest_band: rating * gear / required,
                holds: inputs.metal.min_temp_C <= inputs.coupling.op_temp_C
                    && md.torque_hot_low_Nm > 0.0
                    && input_at_required > rating,
            }
        }

        /// Each number as sentence 2 quotes it (the readouts' format), in the sentence's order.
        fn quoted(&self, inputs: &DesignInputs) -> [String; 5] {
            [
                format_value(&Value::Num(self.band)),
                input_text(inputs, "metal.required_min_Nm"),
                number_text(self.input_at_required, "metal.cold_high_input_Nm"),
                number_text(self.largest_required, "metal.torque_hot_low_Nm"),
                if self.widest_band >= 1.0 {
                    format_value(&Value::Num(self.widest_band))
                } else {
                    // No band is offered: the requirement alone at the gearbox input.
                    number_text(
                        self.input_at_required / self.band,
                        "metal.cold_high_input_Nm",
                    )
                },
            ]
        }
    }

    /// Asserts that `conflict`, Problem sentence 2 of `inputs` (whose results are `results`),
    /// quotes its five numbers from the results in the sentence's order, and that each limit
    /// it names is, as quoted, below what it limits: no rounding makes it read against itself.
    fn assert_conflict_quotes(
        name: &str,
        inputs: &DesignInputs,
        results: &DesignResults,
        conflict: &str,
    ) {
        let numbers = ConflictNumbers::of(inputs, results);
        let mut from = 0;
        for text in numbers.quoted(inputs) {
            let at = conflict[from..]
                .find(&text)
                .unwrap_or_else(|| panic!("{name}: {text} after byte {from} of {conflict}"));
            from += at + text.len();
        }
        let as_quoted = |x: f64| -> f64 { format_value(&Value::Num(x)).parse().unwrap() };
        assert!(
            as_quoted(numbers.largest_required) < as_quoted(numbers.required),
            "{name}: {conflict}"
        );
        if numbers.widest_band >= 1.0 {
            assert!(
                as_quoted(numbers.widest_band) < as_quoted(numbers.band),
                "{name}: {conflict}"
            );
        } else {
            // No band is offered: the requirement alone is above the rating at the input.
            let alone = numbers.input_at_required / numbers.band;
            let rating = inputs.coupling.gearbox_input_rating_Nm;
            assert!(as_quoted(alone) > as_quoted(rating), "{name}: {conflict}");
        }
    }

    /// Holds the gearbox Problem of `inputs` to owner decision 4A: it is present exactly when
    /// the check fails or asks for a look; sentence 2 ([`gearbox_conflict`]) exists exactly
    /// when the conflict holds by the test's own numbers ([`ConflictNumbers`]), quotes them
    /// ([`assert_conflict_quotes`]), and follows sentence 1 after one space exactly when the
    /// check fails as well. Returns the check's level and whether sentence 2 is shown.
    fn assert_gearbox_problem_consistent(name: &str, inputs: &DesignInputs) -> (Level, bool) {
        let results = compute_all(inputs);
        let badge = level_of(GEARBOX, inputs, &results);
        let problem = gearbox_problem(inputs, &results);
        assert_eq!(
            problem.is_some(),
            badge != Level::Good,
            "{name}: {problem:?}"
        );
        let conflict = gearbox_conflict(inputs, &results);
        let holds = ConflictNumbers::of(inputs, &results).holds;
        assert_eq!(conflict.is_some(), holds, "{name}: {conflict:?}");
        let shown = match (&problem, &conflict) {
            (Some(problem), Some(conflict)) => problem.ends_with(&format!(" {conflict}")),
            _ => false,
        };
        assert_eq!(shown, badge == Level::Bad && holds, "{name}: {problem:?}");
        if let Some(conflict) = &conflict {
            assert_conflict_quotes(name, inputs, &results, conflict);
        }
        (badge, shown)
    }

    #[test]
    fn at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks() {
        // Owner decision 4A. Cold-high over hot-low does not depend on the geometry, so every
        // design that meets the default requirement puts the requirement times that band on
        // the gearbox output, above the rating at its input: sentence 2 follows sentence 1,
        // every number of both from the results and inputs.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        assert_eq!(
            assert_gearbox_problem_consistent("defaults", &inputs),
            (Level::Bad, true)
        );
        let problem = gearbox_problem(&inputs, &results).unwrap();
        let conflict = gearbox_conflict(&inputs, &results).unwrap();
        let first = problem.strip_suffix(&format!(" {conflict}")).unwrap();
        for text in [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
        ] {
            assert!(first.contains(&text), "{text} in {first}");
        }
    }

    #[test]
    fn a_design_failing_the_gearbox_check_alone_has_no_conflict_sentence() {
        // The passing design with the rating back at its default fails the gearbox check
        // alone. Its requirement times the band, at the gearbox input, is under the rating, so
        // a design that just meets its requirement passes both checks: no sentence 2.
        let inputs = only_the_gearbox_fails();
        let results = compute_all(&inputs);
        assert_eq!(
            assert_gearbox_problem_consistent("gearbox_only", &inputs),
            (Level::Bad, false)
        );
        let problem = gearbox_problem(&inputs, &results).unwrap();
        for text in [
            result_text(&results, "metal.cold_high_input_Nm"),
            input_text(&inputs, "coupling.gearbox_input_rating_Nm"),
        ] {
            assert!(problem.contains(&text), "{text} in {problem}");
        }
        // Not a word of the defaults' sentence 2 either: its lead, the words before its first
        // number, is nowhere in this Problem.
        let defaults = DesignInputs::default();
        let conflict = gearbox_conflict(&defaults, &compute_all(&defaults)).unwrap();
        let lead = conflict.split(|c: char| c.is_ascii_digit()).next().unwrap();
        assert!(!lead.is_empty() && !problem.contains(lead), "{problem}");
    }

    #[test]
    fn under_auto_on_length_the_gearbox_check_fails_exactly_when_the_conflict_holds() {
        // Spec 5 (Auto, owner decision 5A): Auto sizes for the hot minimum check alone, so a
        // solved design's hot-low is the requirement and its cold-high at the gearbox input
        // the requirement times the band over ratio and efficiency: the check fails exactly
        // where sentence 2 shows. At the defaults it does. With the minimum temperature at its
        // slider's top and no variation allowance the band narrows enough: Auto solves about
        // 11.90 mm, the check passes and there is no Problem.
        let catalogue = InputCatalogue::get();
        let range = |path: &str| catalogue.entry(path).unwrap().meta.range.unwrap();
        let narrow_band = design(|i| {
            i.metal.min_temp_C = range("metal.min_temp_C").max;
            i.metal.variation = range("metal.variation").min;
        });
        for (name, base, fails) in [
            ("defaults", DesignInputs::default(), true),
            ("narrow band", narrow_band, false),
        ] {
            // The Size tab's Auto on length: the solve over the length slider (the panel's
            // search range for every free variable but ring size), and the design it shows.
            let Ok(SizingOutcome::Solved(point)) =
                solve(&base, FreeVariable::AxialLength, base.metal.required_min_Nm)
            else {
                panic!("{name}: Auto on length solves")
            };
            let inputs = crate::gui::sizing::sized(FreeVariable::AxialLength, &base, point.value);
            let results = compute_all(&inputs);
            assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Good, "{name}");
            let (badge, shown) = assert_gearbox_problem_consistent(name, &inputs);
            assert_eq!((badge == Level::Bad, shown), (fails, fails), "{name}");
            if !fails {
                assert!((point.value - 11.902).abs() < 0.005, "{}", point.value);
                assert_eq!(gearbox_problem(&inputs, &results), None);
            }
        }
    }

    #[test]
    fn the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks() {
        // Review focus (owner decision 4A): over every test design, and the defaults with the
        // gearbox ratio and efficiency at each end of their sliders, sentence 2 shows exactly
        // when the check fails and the conflict holds, and every limit it quotes is below
        // what it limits.
        let catalogue = InputCatalogue::get();
        let mut cases: Vec<(String, DesignInputs)> = designs()
            .into_iter()
            .map(|(name, inputs)| (name.to_owned(), inputs))
            .collect();
        for path in ["coupling.gear_ratio", "coupling.gear_efficiency"] {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            for end in [range.min, range.max] {
                let inputs = with_number(&DesignInputs::default(), path, end).unwrap();
                cases.push((format!("{path} at {end}"), inputs));
            }
        }
        let (mut with_sentence, mut without_sentence) = (0, 0);
        for (name, inputs) in &cases {
            match assert_gearbox_problem_consistent(name, inputs) {
                (Level::Bad, true) => with_sentence += 1,
                (Level::Bad, false) => without_sentence += 1,
                _ => {}
            }
        }
        // Both forms of a failing check's Problem are reached.
        assert!(
            with_sentence > 0 && without_sentence > 0,
            "{with_sentence} with sentence 2, {without_sentence} without"
        );
    }

    #[test]
    fn an_operating_temperature_below_the_minimum_keeps_the_gearbox_problem_to_sentence_1() {
        // Review focus: the operating-temperature slider reaches below the minimum
        // temperature. There hot-low is taken colder than cold-high, the band can fall below
        // one and SmCo widens it, so sentence 2's levers would be false: the Problem is
        // sentence 1 alone wherever the check fails, and nothing panics.
        let catalogue = InputCatalogue::get();
        let range = |path: &str| catalogue.entry(path).unwrap().meta.range.unwrap();
        let coldest = range("coupling.op_temp_C").min;
        let mut forms = BTreeSet::new();
        for minimum in [-20.0, range("metal.min_temp_C").max] {
            for variation in [
                DesignInputs::default().metal.variation,
                range("metal.variation").min,
            ] {
                let inputs = design(|i| {
                    i.coupling.op_temp_C = coldest;
                    i.metal.min_temp_C = minimum;
                    i.metal.variation = variation;
                });
                assert!(inputs.coupling.op_temp_C < inputs.metal.min_temp_C);
                let name = format!("minimum {minimum} °C, variation {variation}");
                forms.insert(assert_gearbox_problem_consistent(&name, &inputs));
                for required in [
                    inputs.metal.required_min_Nm,
                    range("metal.required_min_Nm").max,
                ] {
                    let demanding = with(inputs.clone(), |i| i.metal.required_min_Nm = required);
                    let results = compute_all(&demanding);
                    assert_eq!(
                        gearbox_conflict(&demanding, &results),
                        None,
                        "{name}, {required}"
                    );
                    forms.insert(assert_gearbox_problem_consistent(&name, &demanding));
                }
            }
        }
        // Red in every case, never with sentence 2.
        assert_eq!(forms, BTreeSet::from([(Level::Bad, false)]));
    }
```

Then extend `every_text()`, so the plain-words, glyph and ring-size bans (and T5's protect ban) read sentence 2's
second remedy form too. In `every_text()`, replace

```rust
            texts.push(design_status_text(&design_status(&inputs, &results, false)));
        }
        texts
    }
```

with

```rust
            texts.push(design_status_text(&design_status(&inputs, &results, false)));
        }
        // The gearbox Problem's other remedy (no band under one settles it) shows only at the
        // gearbox ratio's and efficiency's slider bottoms, which no test design reaches.
        let catalogue = InputCatalogue::get();
        for path in ["coupling.gear_ratio", "coupling.gear_efficiency"] {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            let inputs = with_number(&DesignInputs::default(), path, range.min).unwrap();
            texts.extend(gearbox_problem(&inputs, &compute_all(&inputs)));
        }
        texts
    }
```

Notes on the tests (no typed numbers in any expected text; every expected text comes from the results or inputs
through the same formatters the Problem uses):
- `ConflictNumbers` is the test's own oracle of the predicate (with PD-1's temperature order) and the five numbers.
  `assert_conflict_quotes` finds them in the sentence in order, which catches two N·m numbers swapped. It also checks,
  on the quoted values, that the largest requirement reads below the requirement and the widest band below the band,
  or, where no band is offered (the widest band below one), that the requirement alone at the gearbox input reads
  above the rating.
- `assert_gearbox_problem_consistent` holds: Problem present exactly when the badge is not green;
  `gearbox_conflict` is `Some` exactly when the oracle's predicate holds; sentence 2 appended after one space exactly
  when the check is red and the predicate holds.
- Review focus (a) is `the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks`. It runs over
  `designs()` and over the ratio and efficiency slider ends read from the catalogue (1 and 20; 0.5 and 1.0). At the
  ratio's and efficiency's bottoms sentence 2 takes its second remedy form (no band). At this task `long_magnets`
  supplies the red-without-sentence-2 form (cold-high at the input 3.078 against 1.000; the predicate
  0.5636 <= 1.0). T4 adds the gearbox-only design to `designs()`.
- Review focus (b) is `an_operating_temperature_below_the_minimum_keeps_the_gearbox_problem_to_sentence_1`: operating
  temperature -40 °C (its slider's bottom), minimum -20 °C and 20 °C, variation 0.15 and 0, each at the default
  requirement and at the requirement slider's top. Probe values: band 1.293, 0.9557, 1.177 and 0.8702 (two of them below one;
  SmCo raises the 0.8702 to 0.9593, so "SmCo magnets narrow it" would be false there); the check is red in every case
  and sentence 2 never shows (`gearbox_conflict` is `None`; the corner is backlog BL-067).

- [ ] **Step 3: Extend `the_problems_quote_numbers_from_the_results` with the sentence-2 numbers**

Sentence 2's numbers are worked out from results, not read at a result path, so they cannot go in the case table's
`keys` column; they get their own block after the loop. In `the_problems_quote_numbers_from_the_results`, replace the
end of the loop and the function:

```rust
            assert_ne!(texts[0], texts[1], "{check}");
            for (text, results) in texts.iter().zip([&a, &b]) {
                for path in keys {
                    let number = quoted(results, path);
                    assert!(
                        text.contains(&number),
                        "{check}: {number} ({path}) in {text}"
                    );
                }
            }
        }
    }
```

with:

```rust
            assert_ne!(texts[0], texts[1], "{check}");
            for (text, results) in texts.iter().zip([&a, &b]) {
                for path in keys {
                    let number = quoted(results, path);
                    assert!(
                        text.contains(&number),
                        "{check}: {number} ({path}) in {text}"
                    );
                }
            }
        }
        // The gearbox Problem's sentence 2 (owner decision 4A) quotes numbers worked out from
        // the results: two designs with different bands (a warmer minimum temperature narrows
        // it) quote different numbers, each its own.
        let sentences: Vec<String> = [
            ("defaults", DesignInputs::default()),
            ("minimum -20 °C", design(|i| i.metal.min_temp_C = -20.0)),
        ]
        .iter()
        .map(|(name, inputs)| {
            let results = compute_all(inputs);
            let conflict = gearbox_conflict(inputs, &results).expect("the conflict holds");
            assert_conflict_quotes(name, inputs, &results, &conflict);
            conflict
        })
        .collect();
        assert_ne!(sentences[0], sentences[1]);
    }
```

(The two designs' bands, from the probe: 1.673 at -40 °C, 1.599 at -20 °C.)

- [ ] **Step 4: Run the new tests and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks a_design_failing_the_gearbox_check_alone_has_no_conflict_sentence under_auto_on_length_the_gearbox_check_fails_exactly_when_the_conflict_holds the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks an_operating_temperature_below_the_minimum_keeps_the_gearbox_problem_to_sentence_1 the_problems_quote_numbers_from_the_results`

Expected: the test build fails, five times (test (b) calls `gearbox_conflict` directly too):

```
error[E0425]: cannot find function `gearbox_conflict` in this scope
...
error: could not compile `magcoupling-rs` (lib test) due to 5 previous errors
```

(The runtime red: with a stub `gearbox_conflict` returning `None`, five of the six fail. Three fail in
`assert_gearbox_problem_consistent`'s `conflict.is_some() == holds`; for example
`the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks` fails with
`assertion left == right failed: defaults: None`. The gearbox-only test fails unwrapping the defaults' sentence 2,
and the extension fails with `the conflict holds`. The renamed test (b) passes against the stub: it is red only
against a sentence 2 without the temperature gate. After Step 5, removing `temperatures_in_order &&` from the
predicate turns it red; put it back.)

- [ ] **Step 5: Implement sentence 2 and append it to the Bad Problem**

In `magcoupling-rs/src/gui/guidance.rs`, in `fn gearbox_problem`, replace T2's Bad arm and the function's end:

```rust
        Level::Bad => Some(format!(
            "Cold-high at the gearbox input is {}, above its {} input rating.",
            result_text(results, "metal.cold_high_input_Nm"),
            input_text(inputs, "coupling.gearbox_input_rating_Nm"),
        )),
    }
}
```

with:

```rust
        Level::Bad => {
            let overload = format!(
                "Cold-high at the gearbox input is {}, above its {} input rating.",
                result_text(results, "metal.cold_high_input_Nm"),
                input_text(inputs, "coupling.gearbox_input_rating_Nm"),
            );
            // Owner decision 4A: where no geometry passes both checks, the Problem says so.
            Some(match gearbox_conflict(inputs, results) {
                Some(conflict) => format!("{overload} {conflict}"),
                None => overload,
            })
        }
    }
}

/// Problem sentence 2 of the gearbox check (owner decision 4A): where no magnet size, gap or
/// count can pass both it and the hot minimum check, the numbers and what would have to change
/// instead. Cold-high over hot-low (the band) does not depend on the geometry, the magnet count
/// or the back iron, only on the two temperatures, the grades' Br coefficients and the
/// variation allowance. So a design that just meets the requirement has cold-high the
/// requirement times the band at the gearbox output, and that over ratio × efficiency at its
/// input: above the rating, no geometry passes both checks. `None` where hot-low is not above
/// zero (the end-effect model is out of range, and the check is greyed) or a value is not a
/// number (a comparison with NaN is false). `None` also with the operating temperature below
/// the minimum (BL-067).
fn gearbox_conflict(inputs: &DesignInputs, results: &DesignResults) -> Option<String> {
    let (c, md) = (&inputs.coupling, &results.metal);
    let required = inputs.metal.required_min_Nm;
    let rating = c.gearbox_input_rating_Nm;
    // Output torque per input torque in forward drive: ratio × efficiency, never zero on the
    // sliders (ratio from 1, efficiency from 0.5).
    let gear = c.gear_ratio * c.gear_efficiency;
    let band = md.torque_cold_high_Nm / md.torque_hot_low_Nm;
    let input_at_required = required * band / gear;
    // With the operating temperature below the minimum, hot-low is taken colder than
    // cold-high: the band's levers and its floor of one no longer hold, so no sentence 2.
    let temperatures_in_order = inputs.metal.min_temp_C <= inputs.coupling.op_temp_C;
    (temperatures_in_order && md.torque_hot_low_Nm > 0.0 && input_at_required > rating).then(|| {
        // The requirement is a hot-low torque, at the gearbox output.
        let largest_required = number_text(rating * gear / band, "metal.torque_hot_low_Nm");
        let widest_band = rating * gear / required;
        // The band is at least one (cold-high is never below hot-low with the temperatures in
        // order): a band under one is out of the levers' reach, and only the gearbox or the
        // requirement can settle the conflict.
        let remedy = if widest_band >= 1.0 {
            format!(
                "It takes a gearbox rated for that, a requirement of at most {largest_required}, or a band of at most {} times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it).",
                format_value(&Value::Num(widest_band)),
            )
        } else {
            format!(
                "It takes a gearbox rated for that or a requirement of at most {largest_required}: the requirement alone puts {} on the gearbox input, so no narrower band settles it.",
                number_text(required / gear, "metal.cold_high_input_Nm"),
            )
        };
        format!(
            "No magnet size, gap or count passes both this and the hot minimum: cold-high is {} times hot-low here, so meeting the {} requirement puts {} on the gearbox input. {remedy}",
            format_value(&Value::Num(band)),
            input_text(inputs, "metal.required_min_Nm"),
            number_text(input_at_required, "metal.cold_high_input_Nm"),
        )
    })
}
```

The predicate and every term are read from the results and inputs, as spec section 5 lists them, with plan deviation
PD-1's extra term. At every reachable input it is well defined: the variation slider tops out at 0.5, so hot-low <= 0
only when the end-effect model is out of range (the row greys and the check is green, so no Problem). A NaN makes the
comparison false. Ratio >= 1 and efficiency >= 0.5 on their sliders, so ratio × efficiency is never 0.

The two PD-1 terms, and why each sentence stays true wherever it shows:
- **The temperature order.** With the minimum temperature at or below the operating one, the band
  B = th(Tmin)/th(Top)·(1+v)/(1−v) is at least one, and a warmer minimum, a smaller variation or SmCo narrows it. With
  the operating temperature below the minimum (both sliders reach it), hot-low is taken colder than cold-high: the band
  can fall below one and SmCo widens it, so sentence 2 is left out there (backlog BL-067).
- **The second remedy form.** Where rating·ratio·efficiency is below the requirement (the widest band below one: the
  ratio's slider bottom, 1, or the efficiency's, 0.5), no lever reaches a band that low: the requirement alone is above
  the rating at the gearbox input. The sentence then offers the gearbox or the requirement and names what the
  requirement alone puts on the gearbox input.

- [ ] **Step 6: Run the new tests and see them pass**

Run: the command of Step 4.

Expected:

```
test gui::guidance::tests::at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks ... ok
test gui::guidance::tests::a_design_failing_the_gearbox_check_alone_has_no_conflict_sentence ... ok
test gui::guidance::tests::an_operating_temperature_below_the_minimum_keeps_the_gearbox_problem_to_sentence_1 ... ok
test gui::guidance::tests::the_problems_quote_numbers_from_the_results ... ok
test gui::guidance::tests::the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks ... ok
test gui::guidance::tests::under_auto_on_length_the_gearbox_check_fails_exactly_when_the_conflict_holds ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; ...
```

The live text at the defaults (probe, for the record; no test types it):
"Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating. No magnet size, gap or count
passes both this and the hot minimum: cold-high is 1.673 times hot-low here, so meeting the 2.500 N·m requirement
puts 0.8806 N·m on the gearbox input. It takes a gearbox rated for that, a requirement of at most 1.703 N·m, or a
band of at most 1.140 times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it)."
The second remedy form at the defaults with the ratio at 1: "... It takes a gearbox rated for that or a requirement of
at most 0.3407 N·m: the requirement alone puts 2.632 N·m on the gearbox input, so no narrower band settles it." (with
the efficiency at 0.5 instead: 0.8966 and 1.000 N·m).
On the gearbox-only design: "Cold-high at the gearbox input is 0.7070 N·m, above its 0.6000 N·m input rating." Under
Auto on length: 13.770 mm at the defaults (0.8806 N·m at the input, red, sentence 2), and 11.902 mm with the minimum
at 20 °C and variation 0 (0.5664 N·m, green, no Problem).

- [ ] **Step 7: Run the suites**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every `test result: ok`; the lib `860 passed; 0 failed; 6 ignored` (Task 2's 855 plus the five new tests).
The new texts also go through `every_text()`, so `no_guide_text_carries_workbook_jargon`,
`every_guide_text_has_glyphs_in_the_default_fonts` and `no_guide_text_calls_the_ring_radius_ring_size` cover sentence 2
at every design that shows it, and its second remedy form at the ratio's and efficiency's slider bottoms (Step 2's
`every_text()` extension); they pass.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every `test result: ok` (no engine change in this task).

- [ ] **Step 8: Docs (README)**

In `magcoupling-rs/README.md`, the **Check guides** paragraph, replace:

```
  the numbers in them always come from the results.
```

with:

```
  the numbers in them always come from the results. Where no magnet size, gap or count can pass both the gearbox check and the hot minimum check (the requirement times cold-high over hot-low, taken to the gearbox input, is above the rating), the gearbox Problem adds a second, computed sentence that says so and names what would have to change instead: a gearbox rated for it, a lower requirement or a narrower band, or, where the requirement alone is above the rating at the gearbox input, only the first two (cold-high is never below hot-low with the minimum temperature at or below the operating one); with the operating temperature below the minimum it adds none (owner decision 4A; BL-067). Auto still sizes for the hot minimum alone (5A), so under Auto, with the temperatures in that order, the gearbox check fails exactly when that sentence shows.
```

In the test table, the `src/gui/guidance.rs` row, replace:

```
each note reviewed and shown. |
```

with:

```
each note reviewed and shown. BL-059 (T3): the gearbox check's Problem adds its second sentence exactly when the check fails and no magnet size, gap or count passes it and the hot minimum together (the requirement times cold-high over hot-low, at the gearbox input, above the rating), over every test design, the gearbox ratio and efficiency at their slider ends (at the ratio's and efficiency's slider bottoms the requirement alone is above the rating at the input and the sentence offers no band), and with an operating temperature below the minimum it never shows; every number in it is worked out from the results and quoted in order, and each limit it quotes reads below what it limits; at the defaults it shows, on the design that fails only the gearbox check it does not, and under Auto on length the check fails exactly when it shows (with the minimum temperature at its slider's top and no variation allowance Auto solves about 11.90 mm and the check passes). |
```

- [ ] **Step 9: Lint and format**

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
Expected: `Finished`, no warnings. A loop variable named `min_temp_C` trips `non_snake_case` under `-D warnings`,
which is why the test names it `minimum`.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: `Finished`, no warnings.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `git diff --stat`
Expected: only `magcoupling-rs/src/gui/guidance.rs` and `magcoupling-rs/README.md` changed. Do not commit
`docs/chebyshev_lambda/*.png` or `docs/analyses/2026-05-28-press-4bar-analysis.md`.

- [ ] **Step 10: Commit**

Write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T3.txt` (gitignored):

```
feat(magcoupler): BL-059 gearbox Problem says when no geometry passes both checks

Owner decision 4A. Cold-high over hot-low does not depend on the geometry, so where the
requirement times that band, taken to the gearbox input, is above the rating, no magnet
size, gap or count passes both the gearbox check and the hot minimum check. The gearbox
Problem then adds a computed second sentence: the band, the requirement, cold-high at the
input of a design that just meets it, and the largest requirement and widest band the rating
allows. Tests: the defaults, the design that fails only the gearbox check, Auto on length
(the defaults and a narrow band), every test design with the gearbox ratio and efficiency at
their slider ends, an operating temperature below the minimum, and the numbers from results.
Sentence 2 offers a band only where one at or above one settles the conflict, and is left out
with the operating temperature below the minimum (plan deviation; BL-067).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:

```bash
git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/README.md
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T3.txt
```

Expected: one commit, 2 files changed.

---

### Task 4: Fix coverage, the gearbox-only design, near the gearbox limit, and Auto (decision 5A)

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

This task adds tests only. The behaviour they pin came with T2 (the guide, its fixes, `helps` holding a reddening step
back) and T3 (sentence 2), so each new test passes on its first run. Step 5 shows how each can be made to fail by
mutation.

**Files:**
- Modify: `magcoupling-rs/src/gui/guidance.rs` (`mod tests` only): three imports; `designs()` gains the gearbox-only
  design; the fails-its-check list in `each_problem_is_some_exactly_when_its_check_fails` gains its entry;
  `fn near_the_gearbox_limit`, `near_the_gearbox_limit_the_hot_minimum_guide_shows_no_fix`, and its mirror and the
  sampled monotonicity test of the two gearbox fixes.
- Modify: `magcoupling-rs/tests/sizing.rs`: one import, `auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there`.
- Modify: `magcoupling-rs/README.md`: the `src/gui/guidance.rs` and `tests/sizing.rs` rows of the test table.
- Test: the same two files.

**Interfaces:**
- Consumes: T2's `only_the_gearbox_fails()` (imported from `test_support` by T2). T2's `GEARBOX_GUIDE` (fixes Lower
  `MAGNET_LENGTH`, Raise `FACE_GAP`) and the two `every_fix_moves_its_check_toward_passing_over_its_slider` cases: those
  landed in T2, see the note below. T3's `gearbox_conflict`, and its imports `crate::engine::sizing::{SizingOutcome,
  solve}`. Private in `gui::guidance`: `moves`, `helps`, `within_minimum_diameter`, `all_fixes`, `fixes`,
  `hot_min_problem`, `gearbox_problem`, `current_number`, `with_number`, `HOT_MIN_GUIDE`, `NO_FIX`. Also
  `crate::gui::dashboard::{FIX, dashboard_ui}`, `crate::gui::readouts::Readouts`,
  `crate::gui::test_support::{sized_frame, drawn_texts}`, `crate::gui::sizing::sized`, `InputCatalogue`,
  `FreeVariable`, the engine's `gearbox::ABOVE_THE_GEARBOX_RATING` and `sizing::solve`.
- Produces: tests only.

Note on `every_fix_moves_its_check_toward_passing_over_its_slider`: its two gearbox cases,
`(GEARBOX, DesignInputs::default(), MAGNET_LENGTH, Direction::Lower, "gearbox input torque")` and
`(GEARBOX, DesignInputs::default(), FACE_GAP, Direction::Raise, "gearbox input torque")`, landed in T2: the test
asserts that its table equals every slider fix of every guide (`assert_eq!(listed, slider_fixes)`), so it fails the
moment `GEARBOX_GUIDE` has fixes. This task re-runs it in Step 2. That test compares a slider's end with its start;
Step 4's sampled test holds both gearbox fixes monotone all along their sliders. A probe at plan time found the gearbox
margin (rating minus cold-high at the input) monotone over both sliders from the defaults: 400 samples each, length
12.7 -> 2 mm gives -0.2049 -> +0.5520 N·m, gap 1.4 -> 5 mm gives -0.2049 -> +0.3250 N·m.

- [ ] **Step 1: Register the gearbox-only design**

In `mod tests`, `fn designs()`, replace:

```rust
            ("bare_clearance", design(bare_clearance)),
```

with:

```rust
            ("bare_clearance", design(bare_clearance)),
            ("gearbox_only", only_the_gearbox_fails()),
```

In `each_problem_is_some_exactly_when_its_check_fails`, the list of designs that each fail the check they were built
for, replace:

```rust
            (design(bare_clearance), CLEARANCE, Bad),
        ] {
```

with:

```rust
            (design(bare_clearance), CLEARANCE, Bad),
            (only_the_gearbox_fails(), GEARBOX, Bad),
        ] {
```

- [ ] **Step 2: Re-probe every test design with the fix tests**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red each_problem_is_some_exactly_when_its_check_fails all_fixes_lists_each_guide_s_fixes_in_the_guides_order two_failing_checks_that_pull_one_input_apart_say_so every_fix_moves_its_check_toward_passing_over_its_slider no_guide_text_carries_workbook_jargon every_guide_text_has_glyphs_in_the_default_fonts no_guide_text_calls_the_ring_radius_ring_size thinning_the_sleeve_or_liner_says_no_check_covers_magnet_retention the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks`

Expected: `test result: ok. 10 passed` (no production change: these pass on the first run).

What the re-probe at plan time found (every `designs()` entry plus `gearbox_only`, clamp section shown and paused,
compared with HEAD 3e63bec):
- **No design's shown fixes change because of the gearbox guide.** No other guide loses or gains a fix: the gearbox
  check holds back no fix in any `designs()` entry, because it is red there already or has room.
- Only notes change. The new trade-off line "One input pulls two ways (Axial magnet length, both rings): Hot-low
  torque needs it raised and Gearbox input torque needs it lowered, …" appears in the Hot-low block of `defaults`,
  `bare_clearance` and `length_conflict` (in `length_conflict` beside the existing Space claim line, so that block
  has two). T2 pins those.
- The gearbox block shows Lower Axial magnet length and Raise Magnet face gap on `defaults`, `bare_clearance`,
  `length_conflict`, `diameter_conflict`, `twelve_poles`, `long_magnets` and `gearbox_only`.
- On `gearbox_only` the gearbox block has no note (the hot minimum passes, so neither the trade-off nor the face-gap
  side effect fires), and `each_failing_term…` passes there with both fixes.

- [ ] **Step 3: Add the imports the near-limit test draws the dashboard with**

In `mod tests`, replace:

```rust
    use crate::gui::dashboard::{CHECKS, DASHBOARD};
```

with:

```rust
    use crate::gui::dashboard::{CHECKS, DASHBOARD, FIX, dashboard_ui};
```

replace:

```rust
    use crate::gui::plain_text::label_for;
```

with:

```rust
    use crate::gui::plain_text::label_for;
    use crate::gui::readouts::Readouts;
```

and replace T2's list:

```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, hot_day, only_the_clamp_fails, only_the_gearbox_fails,
        passing_design, short_magnets, small_boss, with_poles,
    };
```

with (rustfmt's layout):

```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, drawn_texts, hot_day, only_the_clamp_fails,
        only_the_gearbox_fails, passing_design, short_magnets, sized_frame, small_boss, with_poles,
    };
```

- [ ] **Step 4: Write the near-limit design and its test, its mirror, and the sampled monotonicity test**

In `mod tests`, insert directly before:

```rust
    #[test]
    fn two_failing_checks_that_pull_one_input_apart_say_so() {
```

this block:

```rust
    /// The defaults with the gearbox input rating just above cold-high at the gearbox input:
    /// halfway into the rise one step of the magnet-length slider gives (spec 7, item 7). The
    /// rating feeds no other result, so the step and its rise are the defaults' own.
    fn near_the_gearbox_limit() -> DesignInputs {
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        let [longer] = &moves(&defaults, &results, MAGNET_LENGTH, Direction::Raise)[..] else {
            panic!("one step of longer magnets from the defaults")
        };
        let at = results.metal.cold_high_input_Nm;
        let rise = compute_all(longer).metal.cold_high_input_Nm - at;
        assert!(rise > 0.0, "longer magnets raise cold-high: {rise}");
        design(|i| i.coupling.gearbox_input_rating_Nm = at + rise / 2.0)
    }

    #[test]
    fn near_the_gearbox_limit_the_hot_minimum_guide_shows_no_fix() {
        // Spec 7, item 7, a test of its own rather than a `designs()` entry: there the hot
        // minimum guide's only fix is held back by construction. The gearbox check passes just
        // under its rating and the hot minimum check fails; one step of longer magnets, the
        // hot minimum's only fix, would turn the gearbox check red, so `helps` holds it back
        // and the hot minimum's open block draws the no-fix line (`dashboard::fix_line` draws
        // `NO_FIX` for an open block whose fixes are empty), with the shaft-clamp section
        // shown and paused.
        let inputs = near_the_gearbox_limit();
        let results = compute_all(&inputs);
        assert_eq!(level_of(GEARBOX, &inputs, &results), Level::Good);
        assert_eq!(problem(GEARBOX, &inputs), None);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Bad);
        let longer = (MAGNET_LENGTH, Direction::Raise);
        let [stepped] = &moves(&inputs, &results, longer.0, longer.1)[..] else {
            panic!("one step of longer magnets")
        };
        assert_eq!(
            level_of(GEARBOX, stepped, &compute_all(stepped)),
            Level::Bad
        );
        let hot_min = CHECK_GUIDES
            .iter()
            .position(|g| g.check == HOT_MIN)
            .unwrap();
        for paused in [false, true] {
            assert!(
                !helps(&inputs, &results, longer, None, paused),
                "paused {paused}"
            );
            // The fixes as the panel hands them to the dashboard (no Minimum OD set).
            let all = within_minimum_diameter(&all_fixes(&inputs, &results, paused), &inputs, None);
            assert_eq!(all[hot_min], [], "paused {paused}");
            let output = sized_frame(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                Vec::new(),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &results,
                        &all,
                        paused,
                        &mut Readouts::default(),
                    )
                },
            );
            let texts = drawn_texts(&output);
            let problem = hot_min_problem(&inputs, &results).unwrap();
            let open = texts
                .iter()
                .position(|t| *t == problem)
                .expect("the hot minimum's block is open");
            let fix = open + texts[open..].iter().position(|t| t == FIX).unwrap();
            assert_eq!(texts[fix + 1], NO_FIX, "paused {paused}");
        }
        // With the rating a whole rise further up, the step stays within it and the fix shows.
        let rise =
            2.0 * (inputs.coupling.gearbox_input_rating_Nm - results.metal.cold_high_input_Nm);
        let roomier = with(inputs, |i| i.coupling.gearbox_input_rating_Nm += rise);
        let results = compute_all(&roomier);
        assert_eq!(fixes(&HOT_MIN_GUIDE, &roomier, &results, false), [longer]);
    }

    #[test]
    fn at_the_hot_minimum_s_limit_the_gearbox_guide_shows_no_fix_and_its_problem_says_why() {
        // Spec 5 (5A) and 4A: the length Auto solves, set by hand, puts hot-low at the
        // requirement; cold-high over hot-low does not depend on the geometry, so the gearbox
        // check fails there. Both its fixes lower hot-low, so `helps` holds each back (one step
        // would turn the hot minimum red), the open gearbox block draws NO_FIX right after its
        // Fix label, and its Problem's sentence 2 says why, with the clamp section shown and
        // paused.
        let base = DesignInputs::default();
        let Ok(SizingOutcome::Solved(point)) =
            solve(&base, FreeVariable::AxialLength, base.metal.required_min_Nm)
        else {
            panic!("Auto on length solves")
        };
        let inputs = crate::gui::sizing::sized(FreeVariable::AxialLength, &base, point.value);
        let results = compute_all(&inputs);
        assert_eq!(level_of(HOT_MIN, &inputs, &results), Level::Good);
        assert_eq!(level_of(GEARBOX, &inputs, &results), Level::Bad);
        let gearbox = CHECK_GUIDES
            .iter()
            .position(|g| g.check == GEARBOX)
            .unwrap();
        let problem = gearbox_problem(&inputs, &results).unwrap();
        let conflict = gearbox_conflict(&inputs, &results).expect("no geometry passes both");
        assert!(problem.ends_with(&format!(" {conflict}")), "{problem}");
        for paused in [false, true] {
            for fix in [
                (MAGNET_LENGTH, Direction::Lower),
                (FACE_GAP, Direction::Raise),
            ] {
                assert!(
                    !helps(&inputs, &results, fix, None, paused),
                    "{fix:?} paused {paused}"
                );
            }
            let all = within_minimum_diameter(&all_fixes(&inputs, &results, paused), &inputs, None);
            assert_eq!(all[gearbox], [], "paused {paused}");
            let output = sized_frame(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                Vec::new(),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &results,
                        &all,
                        paused,
                        &mut Readouts::default(),
                    )
                },
            );
            let texts = drawn_texts(&output);
            let open = texts
                .iter()
                .position(|t| *t == problem)
                .expect("the gearbox block is open");
            let fix = open + texts[open..].iter().position(|t| t == FIX).unwrap();
            assert_eq!(texts[fix + 1], NO_FIX, "paused {paused}");
        }
    }

    #[test]
    fn each_gearbox_fix_moves_the_check_toward_passing_all_along_its_slider() {
        // Spec 5: both gearbox fixes are monotone over their sliders from the example design,
        // so a step anywhere along the way, not only the slider's end, moves cold-high at the
        // gearbox input down (200 points from the default to the end the fix points to).
        let catalogue = InputCatalogue::get();
        let defaults = DesignInputs::default();
        let results = compute_all(&defaults);
        for (path, direction) in [
            (MAGNET_LENGTH, Direction::Lower),
            (FACE_GAP, Direction::Raise),
        ] {
            let range = catalogue.entry(path).unwrap().meta.range.unwrap();
            let start = current_number(&defaults, &results, path).unwrap();
            let end = match direction {
                Direction::Lower => range.min,
                _ => range.max,
            };
            let mut last = results.metal.cold_high_input_Nm;
            for k in 1..=200 {
                let x = start + (end - start) * f64::from(k) / 200.0;
                let inputs = with_number(&defaults, path, x).unwrap();
                let now = compute_all(&inputs).metal.cold_high_input_Nm;
                assert!(now < last, "{path} at {x}: {now} after {last}");
                last = now;
            }
        }
    }
```

How the panel decides to draw `NO_FIX`: `dashboard::fix_line` draws it when the open block's fix list is empty.
`guide_ui` opens the block under red. The list is `within_minimum_diameter(FixCache::get(..))` for the guide's
index (`panel.rs`, the dashboard call). The test feeds the dashboard the same list, then checks that the text drawn
right after the hot minimum block's `Fix:` label is `NO_FIX`. It does so with the shaft-clamp section shown and
paused (the panel runs paused, `clamps::PAUSED`). The rise is computed from `compute_all` at the very step `helps`
judges (`moves`), not typed. In the prototype it is 0.000707 N·m at the input (0.00336 N·m at the output; the near-limit rating is 0.805261 N·m).

The mirror test sets by hand the length Auto solves at the defaults (13.770 mm): hot-low meets the requirement, and the
gearbox check fails there because cold-high over hot-low does not depend on the geometry. Both gearbox fixes lower
hot-low, so `helps` holds each back, the open gearbox block draws `NO_FIX` right after its `Fix:` label, and its Problem
ends with sentence 2, which says why (spec 5: under a solved Auto that sentence is the guide's real answer). The
monotonicity test walks 200 points from the default to the slider end each gearbox fix points to and asserts that cold-high
at the gearbox input falls at every step.

- [ ] **Step 5: Run them**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- near_the_gearbox_limit_the_hot_minimum_guide_shows_no_fix at_the_hot_minimum_s_limit_the_gearbox_guide_shows_no_fix_and_its_problem_says_why each_gearbox_fix_moves_the_check_toward_passing_all_along_its_slider`
Expected: `test result: ok. 3 passed`.

(Mutation checks made at plan time, each reverted. With the rating at `at + rise * 1.5` the near-limit test fails at
`level_of(GEARBOX, stepped, ..)`: `left: Good, right: Bad`. With `fix_line`'s `NO_FIX` label commented out it fails
at `assert_eq!(texts[fix + 1], NO_FIX, "paused false")`: `left: "Cold-high torque"`, the next row's label. With `helps`
letting the face-gap fix through, the mirror test fails.)

- [ ] **Step 6: Write the Auto test (decision 5A)**

The solver lives in the engine (`engine::sizing`), so the test goes in `magcoupling-rs/tests/sizing.rs` (engine only,
no feature). Replace:

```rust
use magcoupling::engine::api::DesignInputs;
```

with:

```rust
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::gearbox::ABOVE_THE_GEARBOX_RATING;
```

and insert directly before:

```rust
#[test]
fn a_short_override_shrinks_the_axial_housing() {
```

this test:

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
    assert_eq!(r.metal.hot_min_check, "Estimate covers hot min");
    assert!(r.metal.cold_high_input_Nm > base.coupling.gearbox_input_rating_Nm);
    assert!(
        r.gearbox
            .cold_high_check
            .starts_with(ABOVE_THE_GEARBOX_RATING),
        "{}",
        r.gearbox.cold_high_check
    );
}
```

(Tolerance 0.005 mm, as `a_design_sized_by_length_grows_its_housing_and_can_exceed_the_space_claim` uses for
the same 13.77 mm. The probe value is 13.7695 mm. T3's `under_auto_on_length_…` covers the guide side: the Problem's
sentence 2 at this design.)

- [ ] **Step 7: Run it**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test sizing auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there`
Expected: `test auto_sizes_for_the_hot_minimum_alone_and_the_gearbox_check_fails_there ... ok` and
`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 43 filtered out`.

- [ ] **Step 8: Docs (README test table)**

In `magcoupling-rs/README.md`, the `src/gui/guidance.rs` row (T3's text ends it), replace:

```
Auto solves about 11.90 mm and the check passes). |
```

with:

```
Auto solves about 11.90 mm and the check passes). BL-059 (T4): the test designs include one that fails only the gearbox check (the passing design with the rating back at its default); near the gearbox limit (the rating within one magnet-length step's rise of cold-high at the gearbox input) the hot minimum's longer-magnet fix is held back, with the shaft-clamp section shown or paused, and its open block draws the no-fix line under its Fix label; with the rating a step's rise further up the fix shows again; at the hot minimum's limit (the length Auto solves, set by hand) both gearbox fixes are held back, its block draws the no-fix line and its Problem carries the conflict sentence; each gearbox fix lowers cold-high at the gearbox input all along its slider. |
```

In the `tests/sizing.rs` row, replace:

```
a range with a non-finite end or min > max is `SizingError::InvalidRange`. |
```

with:

```
a range with a non-finite end or min > max is `SizingError::InvalidRange`. BL-059 (owner decision 5A): Auto on length at the defaults still sizes for the hot minimum alone (13.77 mm), and the gearbox check is red there. |
```

- [ ] **Step 9: Suites, lint, format**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every `test result: ok`; the lib `863 passed; 0 failed; 6 ignored` (Task 3's 860 plus the three new tests);
`tests/sizing.rs` 44 passed.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every `test result: ok`; `tests/sizing.rs` 44 passed.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and
`cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: both `Finished`, no warnings.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `git diff --stat`
Expected: `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/tests/sizing.rs`, `magcoupling-rs/README.md` only.

- [ ] **Step 10: Commit**

Write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T4.txt`:

```
test(magcoupler): BL-059 gearbox fix coverage, near-limit and Auto tests

The test designs gain the passing design with the gearbox rating back at its default: it
fails the gearbox check alone, and every fix test runs over it. Near the gearbox limit (the
rating within one magnet-length step's rise of cold-high at the gearbox input) the hot
minimum's only fix is held back and its block draws the no-fix line, shown or paused clamp
section alike; a step's rise further up the fix shows again. Its mirror: at the hot
minimum's limit (the length Auto solves, set by hand) both gearbox fixes are held back, the
gearbox block draws the no-fix line and its Problem carries the conflict sentence. Each
gearbox fix lowers cold-high at the gearbox input all along its slider (200 points). Owner
decision 5A: Auto on length at the defaults still sizes for the hot minimum alone
(13.77 mm), and the gearbox check is red there. No production change.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:

```bash
git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/tests/sizing.rs magcoupling-rs/README.md
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T4.txt
```

Expected: one commit, 3 files changed.

---

### Task 5: Overlays for the gearbox rating and the slip swing; no text says the coupling protects the gearbox

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

**Files:**
- Modify: `magcoupling-rs/src/engine/explain/plain.rs` (new `mentions_gearbox_protection` and its test)
- Modify: `magcoupling-rs/src/engine/explain/notes.rs` (the body of `no_note_says_the_coupling_protects_the_gearbox` calls the shared rule; its comment stays for T6)
- Modify: `magcoupling-rs/src/gui/plain_text.rs` (`LABELS` row and doc, two `HELP` rows; tests: canonical names, the overlay ban, the gearbox texts' claims)
- Modify: `magcoupling-rs/src/gui/guidance.rs` (test: the guide ban)
- Modify: `magcoupling-rs/src/gui/dashboard.rs` (moved test: the renamed rows, 3 -> 4)
- Docs: `magcoupling-rs/README.md` (test rows of `plain.rs` ~:661, `guidance.rs` ~:670, the notes ~:671, `plain_text.rs` ~:673), `docs/ai/02-system.yaml` (~:111), `docs/ai/03-structure.yaml` (~:62, ~:66)

**Interfaces:**
- Consumes: T1 `crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING}`, `DesignResults::gearbox.cold_high_check` and the engine label and help of `gearbox.cold_high_check` (the overlay ban reads them through `label_for` and `help_for`); T2 `GEARBOX_GUIDE` and T3's Problem sentence 2 through `guidance::tests::every_text()`; `engine::explain::probes::{probe_designs, step_input}`, `engine::explain::notes::Step`.
- Produces: `pub fn crate::engine::explain::plain::mentions_gearbox_protection(text: &str) -> bool`; a `LABELS` row and a `HELP` row for `model.gearbox_input_ripple_Nm`; the rewritten `HELP` row of `coupling.gearbox_input_rating_Nm`. T8's G-3 review sets their final words: if it rewords them, its fix commit updates `the_canonical_names_are_the_spec_s` (the label literal) and the word lists in `the_gearbox_texts_claim_what_the_engine_does`.

**Where the ban lives:** two tests, each in the module that owns its texts and run over that module's own collector (`guidance::tests::every_text()`; `plain_text::tests::every_row()` plus every label and help shown), sharing one predicate, `plain::mentions_gearbox_protection`, which the note ban calls too. The rule is typed once, and a new guide text or overlay row is covered where it is written.

**The ripple help claims no inequality with cold-high (plan deviation PD-2, ruling R-7).** An earlier draft said the swing "reads below cold-high at the gearbox input". That is false at reachable inputs. With the operating temperature at -40 °C, the minimum at 20 °C and the variation at 0.1 (all on their sliders), the swing is 0.6999 N·m and cold-high at the input is 0.6700 N·m (the corner is backlog BL-067). The help therefore says what the row is and what the check compares ("compares cold-high at the gearbox input instead"), and claims no inequality. The claims test pins both sides of that corner.

- [ ] **Step 1: Write the failing test for the shared rule**

In `magcoupling-rs/src/engine/explain/plain.rs`, at the end of `mod tests`, after the closing brace of `audit_ids_names_each_prefix_once_in_its_order`, add:

```rust

    #[test]
    fn the_gearbox_protection_rule_flags_both_words_in_any_case() {
        // BL-059: any text naming the gearbox beside any form of "protect" is flagged; either
        // word alone is not.
        for text in [
            "The coupling protects the gearbox.",
            "Gearbox protection up to its rating",
            "It PROTECTS the GEARBOX",
        ] {
            assert!(mentions_gearbox_protection(text), "{text}");
        }
        for text in [
            "The gearbox check compares cold-high at the gearbox input with its rating.",
            "The sleeve protects the magnets.",
            "",
        ] {
            assert!(!mentions_gearbox_protection(text), "{text}");
        }
    }
```

- [ ] **Step 2: Run it and see it fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib the_gearbox_protection_rule`
Expected: the build fails with `error[E0425]: cannot find function `mentions_gearbox_protection` in this scope` (twice, in `src\engine\explain\plain.rs`).

- [ ] **Step 3: Implement the rule**

In `magcoupling-rs/src/engine/explain/plain.rs`, replace:

```rust
/// Whether `c` continues a word: a letter, a digit or an underscore.
```

with:

```rust
/// Whether `text` names the gearbox and any form of "protect", in any case: the trap of the
/// slice-2 spec's section 8 that BL-059's check makes tempting. A pass of the gearbox check
/// means only that the modelled cold-high peak is within the gearbox input rating; backlash,
/// the ring-down at re-lock, resonance and the drive line's stored energy can exceed it. Not in
/// [`BANNED`]: the notes' ban (`notes::no_note_says_the_coupling_protects_the_gearbox`), the
/// guides' and the overlay's call this, so the rule is typed once.
pub fn mentions_gearbox_protection(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("gearbox") && lower.contains("protect")
}

/// Whether `c` continues a word: a letter, a digit or an underscore.
```

- [ ] **Step 4: Run it and see it pass**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib the_gearbox_protection_rule`
Expected: `test engine::explain::plain::tests::the_gearbox_protection_rule_flags_both_words_in_any_case ... ok`, `test result: ok. 1 passed`.

- [ ] **Step 5: Make the note ban call the shared rule**

In `magcoupling-rs/src/engine/explain/notes.rs`, in `no_note_says_the_coupling_protects_the_gearbox` (~:1582), replace:

```rust
            for text in note_texts(n) {
                let lower = text.to_lowercase();
                assert!(
                    !(lower.contains("gearbox") && lower.contains("protect")),
                    "{}: {text}",
                    n.id
                );
            }
```

with:

```rust
            for text in note_texts(n) {
                assert!(
                    !crate::engine::explain::plain::mentions_gearbox_protection(text),
                    "{}: {text}",
                    n.id
                );
            }
```

Leave the test's comment ("BL-059 and spec section 8: nothing checks ...") as it is. T6 rewrites it with the watch-outs.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib -- protect`
Expected: `the_gearbox_protection_rule_flags_both_words_in_any_case ... ok` and `no_note_says_the_coupling_protects_the_gearbox ... ok`; `test result: ok. 2 passed`.

- [ ] **Step 6: Add the guide ban and the overlay ban**

These two tests are guards. No text says it today, so they pass on arrival. Step 1's test shows the rule flags the phrase. At plan time, planting "It protects the gearbox." in the ripple help and "the coupling protects the gearbox." in the gearbox Why made each guard fail with that text (the plants were then removed).

In `magcoupling-rs/src/gui/guidance.rs`, in `mod tests`, replace:

```rust
    // ------------------------------------------------------------------ the Problems
```

with:

```rust
    #[test]
    fn no_guide_text_says_the_coupling_protects_the_gearbox() {
        // BL-059 and slice-2 spec section 8: the gearbox check's Problem, Why and fixes, and
        // every other guide text, say what the check compares; a pass is no protection.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        for text in every_text() {
            assert!(!mentions_gearbox_protection(&text), "{text:?}");
        }
    }

    // ------------------------------------------------------------------ the Problems
```

(The header is unique. The new test lands after `no_guide_text_calls_the_ring_radius_ring_size`.)

In `magcoupling-rs/src/gui/plain_text.rs`, in `mod tests`, replace the import:

```rust
    use crate::engine::explain::plain::{plain_text_violations, sentence_count, word_count};
```

with:

```rust
    use crate::engine::explain::plain::{
        mentions_gearbox_protection, plain_text_violations, sentence_count, word_count,
    };
```

and replace:

```rust
    #[test]
    fn each_display_label_names_one_quantity() {
```

with:

```rust
    #[test]
    fn no_label_help_or_hint_says_the_coupling_protects_the_gearbox() {
        // BL-059 and slice-2 spec section 8: a pass of the gearbox check means only that the
        // modelled cold-high peak is within the gearbox input rating. Every overlay row, and
        // every engine label and help shown where the overlay has none (the gearbox check's
        // own among them), keeps to the rule the notes and the guides keep.
        let inputs = InputCatalogue::get().all().map(|entry| entry.path.as_str());
        let results = table_entries().iter().map(|entry| entry.path.as_str());
        let shown = inputs
            .chain(results)
            .flat_map(|path| [label_for(path), help_for(path)]);
        let overlay = every_row().into_iter().map(|(_, _, text)| text);
        for text in overlay.chain(shown) {
            assert!(!mentions_gearbox_protection(text), "{text:?}");
        }
    }

    #[test]
    fn each_display_label_names_one_quantity() {
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- protect`
Expected: four tests `... ok`: `the_gearbox_protection_rule_flags_both_words_in_any_case`, `no_note_says_the_coupling_protects_the_gearbox`, `no_guide_text_says_the_coupling_protects_the_gearbox` and `no_label_help_or_hint_says_the_coupling_protects_the_gearbox`.

- [ ] **Step 7: Write the failing tests for the two overlay rows**

(a) In `magcoupling-rs/src/gui/plain_text.rs`, in `the_canonical_names_are_the_spec_s` (~:960), replace:

```rust
        // (T7), the name is "Ring radius".
        let want = [
```

with:

```rust
        // (T7), the name is "Ring radius". BL-059 adds the gearbox ripple row's: the nominal
        // slip swing, read beside the gearbox check, which compares cold-high instead.
        let want = [
```

then replace:

```rust
            ("metal.torque_cold_high_Nm", "Cold-high torque"),
        ];
        for (path, label) in want {
```

with:

```rust
            ("metal.torque_cold_high_Nm", "Cold-high torque"),
            (
                "model.gearbox_input_ripple_Nm",
                "Nominal slip swing at the gearbox input",
            ),
        ];
        for (path, label) in want {
```

and replace:

```rust
        assert_eq!(LABELS.len(), want.len(), "no row beyond the spec's table");
```

with:

```rust
        assert_eq!(LABELS.len(), want.len(), "no row beyond these");
```

(b) In `magcoupling-rs/src/gui/plain_text.rs`, in `mod batch_b_tests`, next to the joint-neutral claims test, replace:

```rust
    #[test]
    fn the_batch_b_claims_beyond_one_result_hold() {
```

with:

```rust
    /// BL-059: the gearbox rating's help names the check that now reads the rating, and the
    /// ripple row's label and help name the nominal slip swing, apart from the cold-high torque
    /// the check compares. The rating's type is open (owner decision 2A), so no text calls it a
    /// peak; the operating-temperature slider reaches below the minimum, so no text calls the
    /// swing warm. Each claim holds on the engine: at the five probe designs the swing is the
    /// nominal pull-out divided by the ratio and efficiency, one slider step of the operating
    /// temperature moves it and one of the minimum temperature or the variation does not; and
    /// with the rating halfway between the swing and cold-high at the gearbox input, the check
    /// follows cold-high on either side of the swing.
    #[test]
    fn the_gearbox_texts_claim_what_the_engine_does() {
        use crate::engine::explain::notes::Step;
        use crate::engine::explain::probes::{probe_designs, step_input};
        use crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING};
        const RATING: &str = "coupling.gearbox_input_rating_Nm";
        const RIPPLE: &str = "model.gearbox_input_ripple_Nm";
        // The words: what each text must name, then what none may say ("no check" was the
        // rating's help until the check landed).
        let names = |path: &str, text: &str, words: &[&str]| {
            let lower = text.to_lowercase();
            for word in words {
                assert!(lower.contains(word), "{path}: {word:?} in {text:?}");
            }
        };
        names(
            RATING,
            help_for(RATING),
            &["gearbox check", "cold-high", "gearbox input"],
        );
        names(
            RIPPLE,
            help_for(RIPPLE),
            &[
                "operating temperature",
                "without the production allowance",
                "ratio and efficiency",
                "cold-high at the gearbox input",
            ],
        );
        for text in [help_for(RATING), label_for(RIPPLE), help_for(RIPPLE)] {
            let lower = text.to_lowercase();
            for word in ["no check", "peak", "warm"] {
                assert!(!lower.contains(word), "{word:?} in {text:?}");
            }
        }
        // Under the spreadsheet-details toggle the rating keeps the spreadsheet's own note; the
        // ripple row's engine help is empty, so it shows none.
        assert!(original_help(RATING).is_some());
        assert_eq!(
            original_help(RATING),
            metadata(RATING).map(|(_, help)| help)
        );
        assert_eq!(original_help(RIPPLE), None);
        // The swing, on the engine.
        let number = |inputs: &DesignInputs, path: &str| match compute_all(inputs).get(path) {
            Some(Value::Num(x)) => x,
            other => panic!("{path}: {other:?}"),
        };
        for (name, design) in probe_designs() {
            let ripple = number(&design, RIPPLE);
            let ratio_and_efficiency = design.coupling.gear_ratio * design.coupling.gear_efficiency;
            let through = number(&design, "model.pullout_Nm") / ratio_and_efficiency;
            assert!(
                (ripple - through).abs() <= 1e-12 * through.abs(),
                "{name}: {ripple} against {through}"
            );
            for (input, moves) in [
                ("coupling.op_temp_C", true),
                ("metal.min_temp_C", false),
                ("metal.variation", false),
            ] {
                let stepped = step_input(&design, input, Step::Raise)
                    .unwrap_or_else(|e| panic!("{name}: {input}: {e:?}"));
                assert_eq!(
                    number(&stepped, RIPPLE) != ripple,
                    moves,
                    "{name}: {input} one step up"
                );
            }
        }
        // The check compares cold-high at the gearbox input, not the swing: with the rating
        // halfway between them it follows cold-high, above the swing at the defaults and below
        // it with the operating temperature under the minimum (-40 against 20 °C, variation
        // 0.1: both temperatures on their sliders).
        let mut colder_running = DesignInputs::default();
        colder_running.coupling.op_temp_C = -40.0;
        colder_running.metal.min_temp_C = 20.0;
        colder_running.metal.variation = 0.1;
        for (design, passes) in [(DesignInputs::default(), false), (colder_running, true)] {
            let results = compute_all(&design);
            let ripple = results.model.gearbox_input_ripple_Nm;
            let cold_high = results.metal.cold_high_input_Nm;
            assert_eq!(cold_high < ripple, passes, "{cold_high} against {ripple}");
            let mut between = design.clone();
            between.coupling.gearbox_input_rating_Nm = (ripple + cold_high) / 2.0;
            let verdict = compute_all(&between).gearbox.cold_high_check;
            if passes {
                assert_eq!(verdict, WITHIN_THE_GEARBOX_RATING);
            } else {
                assert!(verdict.starts_with(ABOVE_THE_GEARBOX_RATING), "{verdict}");
            }
        }
    }

    #[test]
    fn the_batch_b_claims_beyond_one_result_hold() {
```

(c) The moved test. In `magcoupling-rs/src/gui/dashboard.rs`, `each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label` (~:1830), replace:

```rust
        // The overlay renames three rows (spec 3.5); their catalogue labels are not drawn.
```

with:

```rust
        // The overlay renames four rows (spec 3.5's three and BL-059's ripple row); their
        // catalogue labels are not drawn.
```

and replace:

```rust
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm"
            ]
```

with:

```rust
                "metal.torque_hot_low_Nm",
                "metal.torque_cold_high_Nm",
                "model.gearbox_input_ripple_Nm"
            ]
```

- [ ] **Step 8: Run them and see them fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gearbox_texts canonical each_row_is_drawn_under`
Expected: `test result: FAILED. 0 passed; 3 failed`, with:
- `the_gearbox_texts_claim_what_the_engine_does`: `coupling.gearbox_input_rating_Nm: "gearbox check" in "Peak torque the gearbox input is rated for. Only an informational reference uses it: no check compares the coupling's torques with it."`
- `the_canonical_names_are_the_spec_s`: `assertion `left == right` failed: model.gearbox_input_ripple_Nm`, `left: "Estimated gearbox input torque ripple amplitude"`, `right: "Nominal slip swing at the gearbox input"`
- `each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label`: `left: ["model.pullout_Nm", "metal.torque_hot_low_Nm", "metal.torque_cold_high_Nm"]`, `right: [..., "model.gearbox_input_ripple_Nm"]`

- [ ] **Step 9: Add the rows**

In `magcoupling-rs/src/gui/plain_text.rs`, replace the `LABELS` doc and its last row:

```rust
/// The display label of an input or a result, by path: spec 3.5's canonical names. A row is
/// never longer than the catalogue label it replaces and names one quantity only (tests). A
/// name other modules draw too is a constant above, so it is typed once.
```

with:

```rust
/// The display label of an input or a result, by path: spec 3.5's canonical names, and the
/// gearbox ripple row's (BL-059: the nominal slip swing, beside the gearbox check, which
/// compares cold-high). A row is never longer than the catalogue label it replaces and names one
/// quantity only (tests). A name other modules draw too is a constant above, so it is typed
/// once.
```

and replace:

```rust
    ("metal.torque_cold_high_Nm", "Cold-high torque"),
];
```

with:

```rust
    ("metal.torque_cold_high_Nm", "Cold-high torque"),
    (
        "model.gearbox_input_ripple_Nm",
        "Nominal slip swing at the gearbox input",
    ),
];
```

(39 characters against the catalogue label's 47.)

Replace the rating's help (~:150):

```rust
        "coupling.gearbox_input_rating_Nm",
        "Peak torque the gearbox input is rated for. Only an informational reference uses it: no check compares the coupling's torques with it.",
```

with:

```rust
        "coupling.gearbox_input_rating_Nm",
        "The gearbox's input torque rating. The gearbox check compares the cold-high torque at the gearbox input with it.",
```

And at the end of `HELP` (~:736), replace:

```rust
        "Number of M3 screws in the optional adapter's joint, beside its pilot and dowel; the joint's slip torque rises in proportion.",
    ),
];
```

with:

```rust
        "Number of M3 screws in the optional adapter's joint, beside its pilot and dowel; the joint's slip torque rises in proportion.",
    ),
    // BL-059: the ripple row, read beside the gearbox check (G-3 review
    // docs/analyses/2026-10-14-magcoupler-help-review.md, BL-059 section).
    (
        "model.gearbox_input_ripple_Nm",
        "Torque swing while slipping: the pull-out at the operating temperature, without the production allowance, divided by the ratio and efficiency. The gearbox check compares cold-high at the gearbox input instead.",
    ),
];
```

(The ripple help is 30 words in 2 sentences and types no number. The rating help is 19 words in 2 sentences, with no digit, as batch A requires.)

- [ ] **Step 10: Run them and see them pass, with every overlay test**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- protect gearbox_texts canonical each_row_is_drawn_under`
Expected: `test result: ok. 7 passed`. The seven are `the_gearbox_protection_rule_flags_both_words_in_any_case`, `no_note_says_the_coupling_protects_the_gearbox`, `no_guide_text_says_the_coupling_protects_the_gearbox`, `no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, `the_gearbox_texts_claim_what_the_engine_does`, `the_canonical_names_are_the_spec_s` and `each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label`.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib plain_text`
Expected: no failure. This covers the caps, plain words, glyphs, coverage, no-digit and no-repeated-default tests, `label_for_falls_back_to_the_metadata_and_never_to_the_path` (39 <= 47 characters) and `each_display_label_names_one_quantity`.

- [ ] **Step 11: Docs**

Each edit replaces the exact text in the first block (one occurrence; the README rows are single long lines) with the second. None overlaps T2's README edits.

`magcoupling-rs/README.md`, `plain.rs` row (~:661). Replace:

```text
which the guard does not know and every note text is held to as well (A-13) |
```

with:

```text
which the guard does not know and every note text is held to as well (A-13); `mentions_gearbox_protection` flags a text that names the gearbox beside any form of "protect", in any case, and neither word alone (BL-059: the one rule the note, guide and overlay bans call) |
```

`magcoupling-rs/README.md`, `guidance.rs` row (~:670). Replace:

```text
no guide, fix, Problem or consequence text says "ring size" (the input row's name is Ring radius).;
```

with:

```text
no guide, fix, Problem or consequence text says "ring size" (the input row's name is Ring radius), and none says the coupling protects the gearbox (`no_guide_text_says_the_coupling_protects_the_gearbox`, BL-059).;
```

`magcoupling-rs/README.md`, the notes row (~:671). Replace:

```text
no note says the coupling protects the gearbox (`no_note_says_the_coupling_protects_the_gearbox`);
```

with:

```text
no note says the coupling protects the gearbox (`no_note_says_the_coupling_protects_the_gearbox`, by `plain::mentions_gearbox_protection`, the rule the guide and overlay bans share);
```

`magcoupling-rs/README.md`, `plain_text.rs` row (~:673), the canonical names. Replace:

```text
the nine canonical names of spec 3.5 and no other row,
```

with:

```text
the nine canonical names of spec 3.5 and the gearbox ripple row's (BL-059: Nominal slip swing at the gearbox input), and no other row,
```

`magcoupling-rs/README.md`, `plain_text.rs` row (~:673), the overlay ban. Replace:

```text
no display label is another input's or result's name. Every input has help
```

with:

```text
no display label is another input's or result's name; no overlay row, and no engine label or help shown where the overlay has none, says the coupling protects the gearbox (`no_label_help_or_hint_says_the_coupling_protects_the_gearbox`, BL-059). Every input has help
```

`magcoupling-rs/README.md`, `plain_text.rs` row (~:673), at its end, the claims test. Replace:

```text
and that wall breaks through at the slider top on the smallest ring |
```

with:

```text
and that wall breaks through at the slider top on the smallest ring; the gearbox texts claim what the engine does (BL-059, `the_gearbox_texts_claim_what_the_engine_does`): the rating's help names the gearbox check and the cold-high torque at the gearbox input, the ripple row's label and help the slip swing at the operating temperature without the production allowance, through the ratio and efficiency, and no text says peak, warm or no check (the rating's type is open; the operating temperature can sit below the minimum); at the five probe designs the swing is the nominal pull-out divided by the ratio and efficiency, and moves when the operating temperature rises one slider step but not when the minimum temperature or the variation does; and with the rating halfway between the swing and cold-high at the gearbox input the check follows cold-high, above the swing at the defaults and below it with the operating temperature under the minimum (-40 against 20 °C, variation 0.1) |
```

`docs/ai/02-system.yaml`, the "Display names and help" entry (~:111), a double-quoted scalar: keep the `\"` escapes. Replace:

```text
Batch B's help claims are pinned to the engine (plain_text::batch_b_tests::CLAIMS, one slider step on named designs), so an engine change that falsifies a help fails a test."
```

with:

```text
Batch B's help claims are pinned to the engine (plain_text::batch_b_tests::CLAIMS, one slider step on named designs), so an engine change that falsifies a help fails a test. BL-059: LABELS and HELP gain the gearbox ripple row (model.gearbox_input_ripple_Nm, \"Nominal slip swing at the gearbox input\": the pull-out at the operating temperature without the production allowance, through the ratio and efficiency, apart from the cold-high torque the gearbox check compares), and the gearbox rating's help names that check (no text calls the rating a peak: its type is open); the_gearbox_texts_claim_what_the_engine_does pins both to the engine. No overlay row, engine label or help shown, guide text or note names the gearbox beside \"protect\" (one rule, engine::explain::plain::mentions_gearbox_protection, three bans)."
```

`docs/ai/03-structure.yaml`, the `plain.rs` entry (~:62). Replace:

```text
audit_ids (TC-, PA-, CG-, which the notes' texts are held to as well), pure std so gate 4 runs it, re-exported by gui::guidance)
```

with:

```text
audit_ids (TC-, PA-, CG-, which the notes' texts are held to as well), mentions_gearbox_protection (BL-059: the one gearbox-protect rule the note, guide and overlay bans call), pure std so gate 4 runs it, re-exported by gui::guidance)
```

`docs/ai/03-structure.yaml`, the `plain_text.rs` entry (~:66), `LABELS`. Replace:

```text
LABELS (spec 3.5's nine canonical names;
```

with:

```text
LABELS (spec 3.5's nine canonical names and the gearbox ripple row's, BL-059;
```

`docs/ai/03-structure.yaml`, the `plain_text.rs` entry (~:66), `HELP`. Replace:

```text
and six overrides elsewhere, three of them results;
```

with:

```text
and six overrides elsewhere, three of them results; BL-059: the gearbox ripple row's help (a result) and the gearbox rating's, rewritten to name the gearbox check;
```

Check the YAML: `python -c "import yaml; [yaml.safe_load(l.strip()) for l in open('docs/ai/02-system.yaml', encoding='utf-8') if 'mentions_gearbox_protection' in l]; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8'))"` prints nothing (the whole 02-system.yaml already fails a strict parse at line 14, pre-existing, so only its edited entry is parsed).

- [ ] **Step 12: The suites**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every `test result: ok`, no `FAILED`; the lib `867 passed; 0 failed; 6 ignored` (Task 4's 863 plus the four
new tests: `the_gearbox_protection_rule_flags_both_words_in_any_case`, `no_guide_text_says_the_coupling_protects_the_gearbox`,
`no_label_help_or_hint_says_the_coupling_protects_the_gearbox` and `the_gearbox_texts_claim_what_the_engine_does`).
Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every `test result: ok`, the lib `229 passed` (228 plus the rule's test), including
`the_docs_count_the_notes_and_the_steps` in `tests/explain.rs`.

- [ ] **Step 13: Clippy both ways and fmt**

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`
Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: each ends `Finished`, with no warning.
Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `git diff --stat`. Expected: only the files of this task.

- [ ] **Step 14: Commit**

Write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T5.txt`:

```
feat(magcoupler): the gearbox rating's help names the check, the ripple row reads as the nominal slip swing, and no text says the coupling protects the gearbox (BL-059 T5)

The rating's overlay help no longer says no check reads it. The ripple row gets a label and help
for the nominal slip swing at the operating temperature, apart from the cold-high torque the
gearbox check compares; a test pins each claim to the engine, both sides of the
operating-below-minimum corner included. One rule, plain::mentions_gearbox_protection, now bans
"gearbox" beside "protect" in the notes, the guides and every label, help and hint shown.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/engine/explain/plain.rs magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/src/gui/plain_text.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T5.txt
git status --short
```
Expected: the commit lists those 8 files; `git status --short` still shows `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged). If gate 1 ran, restore `git checkout -- docs/chebyshev_lambda` first.

**Existing tests T5 moves:**
- `gui::dashboard::tests::each_row_is_drawn_under_its_display_label_and_exported_under_its_catalogue_label`: the renamed list goes from `[pull-out, hot-low, cold-high]` to the same three plus `"model.gearbox_input_ripple_Nm"`, and the comment from "three rows (spec 3.5)" to "four rows (spec 3.5's three and BL-059's ripple row)". Reason: the ripple row's new display label.
- `gui::plain_text::tests::the_canonical_names_are_the_spec_s`: `want` gains `("model.gearbox_input_ripple_Nm", "Nominal slip swing at the gearbox input")`, and the message goes from "no row beyond the spec's table" to "no row beyond these". Reason: `LABELS` gains a row outside spec 3.5's table.
- `engine::explain::notes::tests::no_note_says_the_coupling_protects_the_gearbox`: the inline `lower.contains("gearbox") && lower.contains("protect")` becomes `plain::mentions_gearbox_protection(text)` (the same rule, typed once). Its comment is left for T6.

---

### Task 6: Notes: the two gearbox watch-outs, their sources and dates, and their physics re-review

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

Two reviewed notes say that nothing checks the gearbox. The gearbox check from Task 1 makes both sentences false:
`hot_low_cold_high` watch_out[1] and `when_slipping` watch_out[0]. This task replaces them with the drafts in Step 5.
Each note cites the check and the rating's open datasheet question through one shared source line. Both notes
then go through a fresh-context physics re-review, which sets the final words. Their review dates move to the day the
review passes, and the record goes under each note's own section of the review report.

**What this task checked, and leaves alone** (verified at plan time in a private prototype of the repo with Tasks 1
and 2 applied):
- **The before-snapshot** (`docs/analyses/2026-10-14-magcoupler-notes-before.txt`, `include_str!` at `notes.rs` ~:1312)
  is not compared with the current notes:
  - `the_before_snapshot_format_reads_back` round-trips the current notes through the format;
  - `the_before_snapshot_holds_the_seventeen_notes_slice_2_started_from` checks only the structure of the frozen file;
  - neither note is in the snapshot: both are slice 2's `NEW_NOTES`.
  No change.
- **The sign-off sheet generator** (`gui/explorer.rs` ~:2812-2866). `the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`
  tests the generator in memory: every placeholder must be filled at the default design, and the new watch-outs keep
  `{metal.cold_high_input_Nm}`. The committed `docs/analyses/2026-10-14-magcoupler-notes-signoff.md` still carries the
  old watch-outs, but no test reads it.
  - The owner decided on no new sign-off sheet: the owner reviews the diff (spec section 2). Do **not** run
    `notes_signoff_sheet`, which overwrites the committed sheet.
  - Never run `-- --ignored` without one of the three review-tool names this plan uses (`notes_for_review`,
    `guidance_strings_for_review`, `overlay_strings_for_review`).
  - The record below says the sheet stays the slice-2 record.
- **No other test reads either watch-out's words.** `git grep -n "Nothing checks"` finds only the two literals. The
  watch-out tests in `gui/explorer.rs` use the fixture `RENDER_NOTE`, and `gui/panel.rs` ~:7995 renders
  `torque_from_shear`'s watch-out.
- **How the gates tie a note to its review:**
  - `release_notes_are_reviewed` (`tests/explain.rs` ~:1047) checks only that every linked note is `Review::Reviewed`.
  - `every_note_names_its_committed_review_section` (~:1093) checks three things: the reviewer string, that the date
    is a well-formed ISO date, and that `record == "<report>#<id>"` with exactly one `## <id>` line in the report.
  - **Nothing ties the review date to the report.** This task adds `every_note_s_review_date_is_recorded_in_its_section`:
    the date must appear in the section's "Date: <date>" or in a `###` heading that ends "(<date>)". A date then
    never moves without its record.

**Files:**
- Modify `magcoupling-rs/src/engine/explain/notes.rs`:
  - new `pub const GEARBOX_CHECK_SOURCE` before `pub const NOTES` (~:146);
  - `hot_low_cold_high`: watch_out[1] (~:584), sources (~:591), review date (~:595);
  - `when_slipping`: watch_out[0] (~:745), sources (~:754), review date (~:758);
  - the comment in `no_note_says_the_coupling_protects_the_gearbox` (~:1583).
- Modify `magcoupling-rs/tests/explain.rs`:
  - the import (~:37);
  - a new test after `every_note_names_its_committed_review_section` (ends ~:1125);
  - `the_torque_band_note_s_watch_outs_hold` (~:2475-2500) rewritten, with two helpers after it;
  - `the_slip_note_s_claims_hold_at_every_probe_design`: its doc (~:2774-2778), ~:2798 and ~:2834-2840.
- Modify `docs/analyses/2026-10-14-magcoupler-notes-review.md`: append `### BL-059 re-review (<date>)` at the end of
  `## hot_low_cold_high` (just before `## br_temperature`, ~:2279) and at the end of `## when_slipping` (just before
  `## slip_heating`, ~:2786).
- Modify `magcoupling-rs/README.md`: three clauses of the `tests/explain.rs` row (~:656) and one clause of the
  `src/gui/guidance.rs` row (~:670).
- Modify `docs/ai/03-structure.yaml` (~:73): the new test's name.
- Gitignored outputs:
  - `magcoupling-rs/target/notes-review-BL059.txt` and `magcoupling-rs/target/g3/BL059-notes.diff`;
  - `.superpowers/sdd/2026-10-16-bl059-gearbox-check/t6-review-r<N>.md`, the reviewer's answer for each round.

**Interfaces:**
- Consumes:
  - from Task 1: `magcoupling::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING}` and
    `DesignResults::gearbox.cold_high_check`;
  - from Task 2: `GEARBOX_GUIDE.learn == Learn::Note(TORQUE_BAND_NOTE)`, so `release_notes_are_reviewed` already covers
    the gearbox row's Why note;
  - from the codebase: `probes::probe_designs()`, `note_text::segments`, `result_rows`, `FieldType` and the review
    tool `notes_for_review`.
- Produces:
  - `pub const engine::explain::notes::GEARBOX_CHECK_SOURCE: &str`;
  - the test helpers `values_in` and `assert_the_gearbox_check_compares_cold_high_at_the_input` (`tests/explain.rs`);
  - the test `every_note_s_review_date_is_recorded_in_its_section`;
  - the final, reviewed watch-out texts (Task 8's reviewer reads them for consistency, and Task 9's docs quote them);
  - the two review dates.

**Pinned tests this task moves:**

| Test | Old assertion | New assertion | Reason |
|---|---|---|---|
| `the_torque_band_note_s_watch_outs_hold` (`tests/explain.rs`) | `r.metal.cold_high_input_Nm > c.gearbox_input_rating_Nm` on the example design; its comment cites the tripwire | at all 5 probe designs: watch_out[1] names "gearbox check", its only placeholder is `metal.cold_high_input_Nm`, and the check fails exactly when cold-high at the input is above the rating (it passes at equality, compares the peak and not the swing). The failing probes are `["default", "20 mm ring radius"]`. The only text result naming the gearbox is `gearbox.cold_high_check`. The sources hold `GEARBOX_CHECK_SOURCE`. | The watch-out now says what the check compares and that fatigue is unchecked. |
| `the_slip_note_s_claims_hold_at_every_probe_design` (`tests/explain.rs`) | `r.metal.cold_high_input_Nm > r.model.gearbox_input_ripple_Nm`, with the tripwire comment | the helper, which keeps that assertion and adds that the check compares this peak; watch_out[0] names "gearbox check" with the one placeholder; the source is cited | The watch-out now names the check. |
| `no_note_says_the_coupling_protects_the_gearbox` (`notes.rs`) | comment: "nothing checks the cold-high torque … against a rating" | comment: the check compares one modelled peak, so a pass protects nothing | The comment became false; the assertion is unchanged. |
| new: `every_note_s_review_date_is_recorded_in_its_section` | (none) | every shown note's date is in its section | Nothing tied the date to the record. |

- [ ] **Step 1: Write the failing tests (`magcoupling-rs/tests/explain.rs`)**

  1a. After `use magcoupling::engine::explain::{TermSource, Trace, note_text, notes, probes, render};` (~:37), add:

  ```rust
  use magcoupling::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, WITHIN_THE_GEARBOX_RATING};
  ```

  1b. After the closing `}` of `every_note_names_its_committed_review_section` (~:1125), and before
  `/// Spec 3.3 (R4), section 6 test 5 and ruling 11: the full Start here order is fourteen`, insert:

  ```rust
  /// A note's review date is recorded in its section of the review report: in the section's
  /// "Date: <date>" (the slice-2 review) or in a `###` re-review heading that ends "(<date>)" (a
  /// later review, such as BL-059's of the torque-band and slip notes). A note's date never moves
  /// without its record.
  #[test]
  fn every_note_s_review_date_is_recorded_in_its_section() {
      let report = include_str!("../../docs/analyses/2026-10-14-magcoupler-notes-review.md");
      for n in notes::NOTES.iter().filter(|n| notes::shown(n.id)) {
          let notes::Review::Reviewed { date, .. } = n.review else {
              panic!("{}: not reviewed", n.id);
          };
          let heading = format!("## {}", n.id);
          let section: Vec<&str> = report
              .lines()
              .skip_while(|line| line.trim_end() != heading)
              .skip(1)
              .take_while(|line| !line.starts_with("## "))
              .collect();
          let words = section.join(" ");
          let prose: Vec<&str> = words.split_whitespace().collect();
          let in_header = prose.join(" ").contains(&format!("Date: {date}"));
          let re_review = format!("({date})");
          let in_heading = section
              .iter()
              .any(|line| line.starts_with("### ") && line.trim_end().ends_with(&re_review));
          assert!(
              in_header || in_heading,
              "{}: its review date {date} is not recorded under {heading}",
              n.id
          );
      }
  }
  ```

  (The headers wrap "Date:" and the date across lines in some sections, `## slip_heating` and `## demagnetization`
  among them. That is why the section's words are joined before the search. The report has CRLF line ends in the work
  tree; `lines()` and `trim_end()` handle both kinds.)

  1c. Replace the doc comment and the function `the_torque_band_note_s_watch_outs_hold` (~:2475-2500, from
  ``/// `hot_low_cold_high`'s watch-outs on the example design.`` to its closing `}`) with the following. The
  function and the two helpers come after it:

  ```rust
  /// `hot_low_cold_high`'s watch-outs: the first on the example design, the second at the five
  /// probe designs.
  #[test]
  fn the_torque_band_note_s_watch_outs_hold() {
      // Watch-out 1: hot-low reads the operating temperature, not the hot-day start, and the
      // hot-day torque is compared with the requirement without the allowance: on the example
      // design hot-low fails while the hot-day torque meets the requirement.
      let defaults = DesignInputs::default();
      let r = compute_all(&defaults);
      let required = defaults.metal.required_min_Nm;
      assert!(r.metal.torque_hot_low_Nm < required);
      assert!(r.temperature.magnet_life.torque_hot_day_Nm >= required);
      assert_ne!(
          r.temperature.duty.hot_day_start_C,
          defaults.coupling.op_temp_C
      );
      let hotter_day = stepped(&defaults, "temperature.duty.hot_ambient_C", Step::Raise);
      assert_eq!(
          compute_all(&hotter_day).metal.torque_hot_low_Nm,
          r.metal.torque_hot_low_Nm
      );
      // Watch-out 2: the gearbox check compares cold-high at the gearbox input, the one value the
      // watch-out quotes, with the input rating, at every probe design; on the example design
      // and the 20 mm ring it fails, on the other three it passes, so both sides are probed.
      let note = notes::note("hot_low_cold_high").expect("the torque-band note");
      assert!(
          note.watch_out[1].contains("gearbox check"),
          "watch-out 2 names the gearbox check: {:?}",
          note.watch_out[1]
      );
      assert_eq!(values_in(note.watch_out[1]), ["metal.cold_high_input_Nm"]);
      let mut failing = Vec::new();
      for (name, design) in probes::probe_designs() {
          assert_the_gearbox_check_compares_cold_high_at_the_input(name, &design);
          let check = compute_all(&design).gearbox.cold_high_check;
          if check.starts_with(ABOVE_THE_GEARBOX_RATING) {
              failing.push(name);
          }
      }
      assert_eq!(failing, ["default", "20 mm ring radius"]);
      // "Repeated slips also load the gearbox in fatigue, which nothing checks": the peak check
      // is the engine's only check of the gearbox (a fatigue check, backlog BL-066, adds a
      // second text result here, and this watch-out changes with it).
      let gearbox_checks: Vec<String> = result_rows(&r)
          .into_iter()
          .filter(|row| row.meta.ty == FieldType::Text && row.path.contains("gearbox"))
          .map(|row| row.path)
          .collect();
      assert_eq!(gearbox_checks, ["gearbox.cold_high_check"]);
      // The note cites the check and the rating's open datasheet question.
      assert!(note.sources.contains(&notes::GEARBOX_CHECK_SOURCE));
  }

  /// The placeholders of one note text, in order.
  fn values_in(text: &'static str) -> Vec<&'static str> {
      note_text::segments(text)
          .unwrap_or_else(|fault| panic!("{fault}"))
          .into_iter()
          .filter_map(|piece| match piece {
              note_text::Segment::Value(path) => Some(path),
              note_text::Segment::Text(_) => None,
          })
          .collect()
  }

  /// What the torque-band note's second watch-out and the slip note's first say of the gearbox
  /// check (BL-059), at `design`: it compares cold-high at the gearbox input with the input
  /// rating and fails exactly when cold-high there is above the rating, so at the rating it
  /// passes (the rating 1 % either side of cold-high at the input, and at it); cold-high at the
  /// gearbox input lies above the ripple there, the nominal slip swing; and the check compares
  /// that peak, not the swing: with the rating between the two, it fails.
  fn assert_the_gearbox_check_compares_cold_high_at_the_input(name: &str, design: &DesignInputs) {
      let fails = |d: &DesignInputs| {
          let r = compute_all(d);
          let check = r.gearbox.cold_high_check;
          let above = check.starts_with(ABOVE_THE_GEARBOX_RATING);
          assert!(
              above || check == WITHIN_THE_GEARBOX_RATING,
              "{name}: {check}"
          );
          assert_eq!(
              above,
              r.metal.cold_high_input_Nm > d.coupling.gearbox_input_rating_Nm,
              "{name}: {check}"
          );
          above
      };
      let r = compute_all(design);
      let peak = r.metal.cold_high_input_Nm;
      let rated = |rating: f64| {
          let mut d = design.clone();
          d.coupling.gearbox_input_rating_Nm = rating;
          d
      };
      for (rating, want) in [(0.99 * peak, true), (peak, false), (1.01 * peak, false)] {
          assert_eq!(fails(&rated(rating)), want, "{name}: rating {rating}");
      }
      let swing = r.model.gearbox_input_ripple_Nm;
      assert!(
          swing < peak,
          "{name}: the ripple below cold-high at the gearbox input"
      );
      assert!(
          fails(&rated(0.5 * (swing + peak))),
          "{name}: a rating between the ripple and cold-high at the gearbox input"
      );
      // At the design's own rating.
      fails(design);
  }
  ```

  Every rating these probes set stays inside the rating's slider (0.05 to 5.0 N·m): the smallest is at the Y30 design,
  0.99 × 0.07117 = 0.0705 N·m, with a midpoint of 0.0574 N·m.

  1d. In `the_slip_note_s_claims_hold_at_every_probe_design`, update the doc comment (~:2774-2778). Old:

  ```rust
  /// and cold-high at the input lies above it (the first watch-out: cold, the swings can reach
  /// cold-high).
  ```

  New:

  ```rust
  /// and cold-high at the input lies above it (the first watch-out: cold, the swings can reach
  /// cold-high, and the gearbox check compares that peak with the input rating).
  ```

  1e. In the same test (~:2798), replace:

  ```rust
      let sentence_zero = notes::note("when_slipping").unwrap().sentences[0];
      assert!(
          sentence_zero.contains("gearbox"),
          "sentence 0 names the gearbox: {sentence_zero:?}"
      );
  ```

  with:

  ```rust
      let note = notes::note("when_slipping").unwrap();
      let sentence_zero = note.sentences[0];
      assert!(
          sentence_zero.contains("gearbox"),
          "sentence 0 names the gearbox: {sentence_zero:?}"
      );
      // Watch-out 1 names the gearbox check and quotes cold-high at the gearbox input, the value
      // it compares; the note cites the check and the rating's open datasheet question.
      assert!(
          note.watch_out[0].contains("gearbox check"),
          "watch-out 1 names the gearbox check: {:?}",
          note.watch_out[0]
      );
      assert_eq!(values_in(note.watch_out[0]), ["metal.cold_high_input_Nm"]);
      assert!(note.sources.contains(&notes::GEARBOX_CHECK_SOURCE));
  ```

  1f. In the same test's probe loop (~:2834-2840), replace:

  ```rust
          // Watch-out 1: cold, the swings reach cold-high, which at the gearbox input lies above
          // the ripple at the operating temperature (the GUI test
          // `nothing_checks_a_gearbox_torque_against_its_rating_yet` holds the "nothing checks").
          assert!(
              r.metal.cold_high_input_Nm > r.model.gearbox_input_ripple_Nm,
              "{name}: cold-high at the gearbox input"
          );
  ```

  with the following. The helper keeps the removed assertion: cold-high at the input lies above the ripple.

  ```rust
          // Watch-out 1: the gearbox output carries the swings (sentence 4's relation); cold they
          // can reach cold-high, which at the gearbox input lies above the ripple at the
          // operating temperature; and the gearbox check compares only that peak with the input
          // rating.
          assert_the_gearbox_check_compares_cold_high_at_the_input(name, &design);
  ```

  This removes both comments that named the tripwire (`nothing_checks_a_gearbox_torque_against_its_rating_yet`).

- [ ] **Step 2: Run the tests; they fail to compile**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain`
  Expected (the shared source does not exist yet):
  ```
  error[E0425]: cannot find value `GEARBOX_CHECK_SOURCE` in module `notes`
  error[E0425]: cannot find value `GEARBOX_CHECK_SOURCE` in module `notes`
  error: could not compile `magcoupling-rs` (test "explain") due to 2 previous errors
  ```

- [ ] **Step 3: Add the shared source line (`notes.rs`, before `/// Every note: 20, the cap the completeness test holds`, ~:146)**

  ```rust
  /// The source both gearbox watch-outs rest on (`hot_low_cold_high` and `when_slipping`; BL-059):
  /// the gearbox check, and the open question of which datasheet figure the rating is. Typed once.
  pub const GEARBOX_CHECK_SOURCE: &str = "Design ruling of 2026-10-16: the gearbox check compares cold-high at the gearbox input with the gearbox input torque rating; which datasheet figure that rating is (input or output side; nominal, acceleration or emergency stop) is still to be confirmed";
  ```

  The text is plain words: no "decision <n>" (banned by `plain::BANNED`), no "BL-059" and no "clamp". It does not
  call the rating a peak: its type is open (owner decision 2A). "the gearbox input torque rating" is the input's label
  (`engine/model.rs` ~:196).

- [ ] **Step 4: Run the tests; they fail on the old watch-outs**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- the_torque_band_note_s_watch_outs_hold the_slip_note_s_claims_hold_at_every_probe_design every_note_s_review_date_is_recorded_in_its_section`
  Expected:
  ```
  test every_note_s_review_date_is_recorded_in_its_section ... ok
  test the_slip_note_s_claims_hold_at_every_probe_design ... FAILED
  test the_torque_band_note_s_watch_outs_hold ... FAILED
  ... watch-out 1 names the gearbox check: "Nothing checks the gearbox against these swings: when cold they can reach the cold-high torque, on this design {metal.cold_high_input_Nm} at its input; check its rating."
  ... watch-out 2 names the gearbox check: "Nothing checks cold-high against the gearbox: at its input it is {metal.cold_high_input_Nm} here; compare that with the gearbox's input rating yourself."
  test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 55 filtered out
  ```

- [ ] **Step 5: Put the drafts in the notes (`notes.rs`); the review dates stay as they are for now**

  5a. `hot_low_cold_high` watch_out[1] (~:584). Old:
  ```rust
              "Nothing checks cold-high against the gearbox: at its input it is {metal.cold_high_input_Nm} here; compare that with the gearbox's input rating yourself.",
  ```
  New (26 words raw):
  ```rust
              "The gearbox check compares cold-high at the gearbox input, {metal.cold_high_input_Nm} here, with its input rating; repeated slips also load the gearbox in fatigue, which nothing checks.",
  ```

  5b. `hot_low_cold_high` sources (~:591). Old:
  ```rust
              "Guidance audit of 2026-10-13: which temperature each torque uses; the gearbox input",
          ],
  ```
  New:
  ```rust
              "Guidance audit of 2026-10-13: which temperature each torque uses; the gearbox input",
              GEARBOX_CHECK_SOURCE,
          ],
  ```

  5c. `when_slipping` watch_out[0] (~:745). Old:
  ```rust
              "Nothing checks the gearbox against these swings: when cold they can reach the cold-high torque, on this design {metal.cold_high_input_Nm} at its input; check its rating.",
  ```
  New (27 words raw):
  ```rust
              "The gearbox output carries these swings; cold they can reach cold-high, {metal.cold_high_input_Nm} at its input here, and the gearbox check compares only that peak with its rating.",
  ```

  5d. `when_slipping` sources (~:754). Old:
  ```rust
              "Guidance audit of 2026-10-13: what happens past pull-out; the rings re-lock only when the slip speed falls",
          ],
  ```
  New:
  ```rust
              "Guidance audit of 2026-10-13: what happens past pull-out; the rings re-lock only when the slip speed falls",
              GEARBOX_CHECK_SOURCE,
          ],
  ```

  5e. The comment in `no_note_says_the_coupling_protects_the_gearbox` (~:1583). Old:
  ```rust
          // BL-059 and spec section 8: nothing checks the cold-high torque at the gearbox input
          // against a rating, so no note may claim the coupling protects the gearbox.
  ```
  New:
  ```rust
          // Spec section 8 and BL-059: the gearbox check compares only the modelled cold-high
          // peak at the gearbox input with one rating; repeated slips, backlash and resonance are
          // not checked, so a pass protects nothing and no note may claim the coupling protects
          // the gearbox.
  ```

- [ ] **Step 6: Run the tests; they pass, and so do the note guards**

  Run, from the repo root:
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain`
    Expected: `test result: ok. 57 passed; 0 failed; 1 ignored`. That is the 56 at HEAD plus
    `every_note_s_review_date_is_recorded_in_its_section` (Task 2's docs-count guard needs the `gui` feature, so it does
    not run here). The run includes `every_note_reads_in_plain_words_at_every_probe_design`,
    `every_note_placeholder_names_a_number_result_and_renders_at_every_probe_design` (both placeholders sit in a
    "here" clause with no unit typed after them), `release_notes_are_reviewed` and
    `every_note_names_its_committed_review_section`.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain`
    Expected: `test result: ok. 58 passed; 0 failed; 1 ignored` (the 57 above plus Task 2's
    `the_docs_count_the_dashboard_rows_and_the_checks`). With the `gui` part on, this run also checks the gearbox row's
    Why note.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib engine::explain::notes`
    Expected: `test result: ok. 29 passed; 0 failed; 1 ignored`. The run includes `every_note_keeps_to_the_slice_2_caps`
    (watch-outs of at most 30 words: 26 and 27 here), `the_slip_notes_keep_the_caps`, `every_sentence_zero_is_plain_words`
    and `no_note_says_the_coupling_protects_the_gearbox`.
  - `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice every_teaching_note_shows_with_glyphs_in_the_default_fonts the_ripple_row_s_why_link_opens_the_slip_note every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`
    Expected: `test result: ok. 4 passed; 0 failed`. These check the generator's filled placeholders, glyphs (sources
    included) and the paused-clamp ban, which covers sources.

- [ ] **Step 7: Write the review dump and the diff for the reviewer**

  ```bash
  NOTES_FOR_REVIEW=hot_low_cold_high,when_slipping NOTES_REVIEW_OUT=target/notes-review-BL059.txt cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture
  mkdir -p magcoupling-rs/target/g3
  git diff -- magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/tests/explain.rs > magcoupling-rs/target/g3/BL059-notes.diff
  grep -c "<<<" magcoupling-rs/target/notes-review-BL059.txt
  ```
  Expected:
  - the first command prints `2 notes at 5 designs: target/notes-review-BL059.txt` and `test result: ok. 1 passed`;
  - the `grep -c` prints `0`: no text is refused by the parser or flagged by the guard;
  - watch-out lines in the dump at the five designs:
    - torque band: "The gearbox check compares cold-high at the gearbox input, 0.8049 N·m here, …" (then 0.5893,
      0.5082, 0.07117 and 0.7602 N·m);
    - slip note: "The gearbox output carries these swings; cold they can reach cold-high, 0.8049 N·m at its input
      here, …".

- [ ] **Step 8 (controller): the physics re-review, one fresh-context reviewer for both notes**

  **One reviewer for both notes.** The two watch-outs make one claim about one check, share one source line
  (`GEARBOX_CHECK_SOURCE`) and are pinned by one test helper. One fresh context checks that they agree and never words
  the shared source two ways. This follows slice 2, where T14 reviewed the torque-band note with `br_temperature` and
  T15 reviewed three slip notes together. Each note's record still goes under its own section.

  Dispatch it with the Agent tool: subagent `general-purpose`, `model` omitted (session model: physics, CLAUDE.md
  section 5), fresh context. Save each answer verbatim as `.superpowers/sdd/2026-10-16-bl059-gearbox-check/t6-review-r<N>.md`.
  The brief (round 1, verbatim):

  ```text
  You are a fresh-context physics reviewer (session model) for BL-059 in the magnetic coupling calculator
  (crate magcoupling-rs; repo C:/Users/Cole/source/repos/linkage_simulation, branch magcoupling/bl059-gearbox-check).
  Read-only: do not edit the repository and run no git command that changes state. Probe the engine only from a
  scratch crate outside the repository, in your scratchpad, with its own CARGO_TARGET_DIR there. Its Cargo.toml:
      [package]
      name = "bl059review"
      version = "0.1.0"
      edition = "2024"
      publish = false
      [dependencies]
      magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs" }
      [workspace]
  (the library is `magcoupling`: magcoupling::engine::api::{DesignInputs, compute_all}). The first build takes several
  minutes.

  BL-059 adds the gearbox check, gearbox.cold_high_check (engine/gearbox.rs): cold-high at the gearbox input
  (metal.cold_high_input_Nm = cold-high / (ratio x efficiency)) against the gearbox input torque rating
  (coupling.gearbox_input_rating_Nm, 0.6 N·m by default); at the rating it passes. Two reviewed notes said nothing
  checks the gearbox; their texts change:
  - hot_low_cold_high watch_out[1]. Old: "Nothing checks cold-high against the gearbox: at its input it is
    {metal.cold_high_input_Nm} here; compare that with the gearbox's input rating yourself." Draft: "The gearbox check
    compares cold-high at the gearbox input, {metal.cold_high_input_Nm} here, with its input rating; repeated slips also
    load the gearbox in fatigue, which nothing checks."
  - when_slipping watch_out[0]. Old: "Nothing checks the gearbox against these swings: when cold they can reach the
    cold-high torque, on this design {metal.cold_high_input_Nm} at its input; check its rating." Draft: "The gearbox
    output carries these swings; cold they can reach cold-high, {metal.cold_high_input_Nm} at its input here, and the
    gearbox check compares only that peak with its rating."
  - Both notes' sources gain one shared line, notes::GEARBOX_CHECK_SOURCE: "Design ruling of 2026-10-16: the gearbox
    check compares cold-high at the gearbox input with the gearbox input torque rating; which datasheet figure that
    rating is (input or output side; nominal, acceleration or emergency stop) is still to be confirmed"
  The diff: magcoupling-rs/target/g3/BL059-notes.diff. Both notes as drawn at the five probe designs:
  magcoupling-rs/target/notes-review-BL059.txt.

  Read: the spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md (sections 1, 2, 3, 6 and 10); the
  engine behind every claim: magcoupling-rs/src/engine/gearbox.rs; engine/api.rs (grep "gearbox::compute");
  engine/metal_design.rs (grep "cold_high_input_Nm" and "torque_cold_high_Nm"); engine/model.rs (grep
  "gearbox_input_ripple_Nm" and "gearbox_reference_Nm"); gui/guidance.rs (GEARBOX_GUIDE: its Why link opens
  hot_low_cold_high); gui/dashboard.rs (DASHBOARD, END_EFFECT_ROWS); the earlier review of both notes in
  docs/analyses/2026-10-14-magcoupler-notes-review.md, sections "## hot_low_cold_high" and "## when_slipping" (traps T-O
  and T-H, ruling 8, the SF-2 offered there); the slice-2 traps in
  docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md section 8. The ripple row's on-screen label is now
  "Nominal slip swing at the gearbox input" (T5's overlay); say whether when_slipping sentence 0 still answers that row.

  Traps (each one written is a must-fix): name the side of the gearbox for every torque; efficiency raises the input
  torque (never "reduces the load"); cold-high is an expectation ("up to", "can reach"; never "at most", never what
  every unit reaches); never "the coupling protects the gearbox" (a test bans "gearbox" with "protect" in any note
  text; a pass means only that the modelled cold-high peak is within the input rating); "peak" names the compared
  torque, never the rating (its type is open, owner decision 2A); the ripple row is the nominal swing at the operating
  temperature (no "warm": the operating-temperature slider reaches below the minimum); geometry cannot narrow the
  band; no "hot-day" in these texts; no internal paths, ids, cells, "BL-059", "Rust-only" or "clamp" on screen;
  glyph-safe ("->", "<=").

  Rules the words must keep (tests hold them):
  - a watch-out has at most 30 words, raw, counted as runs between whitespace; a note has at most 2 watch-outs;
  - a live value only as a {result.path} placeholder of a number result, inside a clause with the whole word "here"
    or "on this design", with no unit typed after it;
  - plain words: no cell, correction or audit id, no "decision <n>", no "BL-059", no "clamp";
  - a source holds no placeholder and has no word cap, but keeps the plain-words rule;
  - the tests pin the words "gearbox check" in both watch-outs and metal.cold_high_input_Nm as each one's only
    placeholder: if you change either, say so and say what the tests should pin instead.

  Probe at least:
  (1) the five probe designs (engine::explain::probes::probe_designs). Expected: cold-high at the gearbox input 0.8049,
      0.5893, 0.5082, 0.07117 and 0.7602 N·m; ripple at the input 0.5660, 0.4144, 0.3573, 0.04359 and 0.5346 N·m; the
      check above, within, within, within, above;
  (2) the rating at 1.0 (all within), at 0.80 ("Above the gearbox input rating: 0.81 N·m against 0.80 N·m"), and equal
      to cold-high at the input (within);
  (3) a NaN rating set on the struct ("Gearbox check unknown: a torque is not a number");
  (4) the end-effect model out of range (axial length 1.0 mm, or c_end 1.5 set on the struct): the placeholder renders
      "no prediction", and the dashboard greys the row (END_EFFECT_ROWS);
  (5) the operating temperature below the minimum temperature, both inside their sliders: op -40 °C with min 20 °C
      and variation 0 (cold-high at the input 0.6091 against a ripple of 0.6999 N·m), variation 0.05 (0.6395 against
      0.6999), and op 0 °C with min 20 °C and variation 0 (0.6091 against 0.6386). There the nominal swing exceeds
      cold-high: judge whether "cold they can reach cold-high" and "only that peak" stay true, and scope them if not
      (the corner is backlog BL-067; the engine stays as it is, decision 7A);
  (6) ratio 1 with efficiency 0.5 (cold-high at the input 7.647 N·m) and ratio 20 with efficiency 1.0 (0.1912 N·m);
  (7) the slip swing as a fatigue load: 16.7 cycles per 0.1 s event at the defaults and metal.magnetic_cycles per
      life. Is "which nothing checks" true? The check is a peak check only, and backlog BL-066 holds the fatigue check.

  Return, for the record (it is pasted verbatim under each note's section of the review report):
  1. "Reviewer: physics reviewer (session model), fresh context, <date>; round <n>."
  2. Read: what you read, with file and line.
  3. Numbers reproduced from compute_all: at the five probe designs, cold-high at the gearbox input, the ripple at the
     input and the check text; then every extra probe.
  4. Per field (hot_low_cold_high watch_out[1]; when_slipping watch_out[0]; GEARBOX_CHECK_SOURCE): PASS, or MUST-FIX
     with the exact replacement text (within the rules above), why (the engine fact or probe that makes the draft
     false or misleading) and the test assertion that would pin the claim; SHOULD-FIX offered, in the same form.
  5. Traps checked, one line each.
  6. Form: word counts, raw and rendered at the default design; the placeholder rule.
  7. Verdict for each note (PASS or MUST_FIX) and for the round.
  Your words win over the drafts within the rules above: write the text you would ship.
  ```

  Round 2 and round 3 briefs are the same, with three additions:
  - the round's changes, each field old -> new;
  - "Re-review only the changed fields and the fields whose meaning depends on them";
  - the earlier answers' paths.
  Each round is a new fresh context.

- [ ] **Step 9: Apply the must-fix items; at most three rounds**

  For each must-fix item, in order:
  1. Copy the reviewer's replacement text verbatim into `notes.rs`. Where the item names a claim that a test can check,
     write that assertion first: in `the_torque_band_note_s_watch_outs_hold` or
     `the_slip_note_s_claims_hold_at_every_probe_design`, or in the helper if both notes make the claim. Run it, see
     it fail on the old text, then apply the text.
  2. If the reviewer renames the check, or adds or removes a placeholder, change the `"gearbox check"` pin and the
     `values_in(...)` expectation to the reviewer's wording in the same edit, and update the comments that quote the
     watch-outs.
  3. Re-run the Step 6 commands (all green) and the Step 7 dump (`grep -c "<<<"` prints 0).

  Then dispatch the next round (Step 8, round 2). Stop when a round returns PASS for both notes.

  If round 3 still returns a must-fix, **stop**: report it to the controller with the open items. It is an owner
  decision; do not run a fourth round. Should-fix items stay offered and are not applied unless the controller says
  so, as in slice 2. The reviewer's words win over the drafts within the caps.

- [ ] **Step 10: Move both review dates to the day of the passing round; the date test fails**

  Take the date on the day the round that returned PASS ran: `PASS_DATE="$(date +%F)"; echo "$PASS_DATE"`. Below and in
  Step 11, `<PASS_DATE>` is the date it printed (a shell variable does not outlive its Bash call, so type the date in).

  In `notes.rs`, in both `Review::Reviewed { … }` blocks (`hot_low_cold_high` ~:595 and `when_slipping` ~:758),
  change `date: "2026-10-14",` to `date: "<PASS_DATE>",`.

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- every_note_s_review_date_is_recorded_in_its_section`
  Expected:
  ```
  test every_note_s_review_date_is_recorded_in_its_section ... FAILED
  hot_low_cold_high: its review date <PASS_DATE> is not recorded under ## hot_low_cold_high
  test result: FAILED. 0 passed; 1 failed
  ```

- [ ] **Step 11: Append the two records to `docs/analyses/2026-10-14-magcoupler-notes-review.md`; the date test passes**

  Do not rewrite history: append only, and keep the file's CRLF line ends.

  **(a) Under `## hot_low_cold_high`,** after its last line and just before the line `## br_temperature`, append the
  block below. Angle-bracket slots are filled from the reviewer's returned answers and nothing else; the rest is
  verbatim.

  ```markdown
  ### BL-059 re-review (<PASS_DATE>)

  - Why: BL-059 added the gearbox check (`gearbox.cold_high_check`: cold-high at the gearbox input
    against the gearbox input torque rating; spec
    `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`, sections 2 and 6), so
    watch_out[1], which said nothing checks cold-high against the gearbox, became false. The note's
    `review` date moves to the date in this heading. The slice-2 review above is not rewritten.
  - watch_out[1]:
    - Old: "Nothing checks cold-high against the gearbox: at its input it is
      {metal.cold_high_input_Nm} here; compare that with the gearbox's input rating yourself."
    - New (<n> words raw): "<the text as shipped>"
  - sources[4] (new; the same line as `when_slipping`'s last source, typed once as
    `notes::GEARBOX_CHECK_SOURCE`): "<the text as shipped>"
  - Reviewer: physics reviewer (session model), fresh context, one reviewer for this note and
    `when_slipping` (the two watch-outs make one claim about one check and share one source line).
    Rounds: <n>. Verdict: PASS. Each round's full answer is
    `.superpowers/sdd/2026-10-16-bl059-gearbox-check/t6-review-r<N>.md`.
  - Must-fix items applied (old -> new, and why): <"none", or one bullet per item with its round>.
  - Should-fix offered, not applied (open for the controller): <"none", or one bullet per item>.
  - Tests: `the_torque_band_note_s_watch_outs_hold` (`tests/explain.rs`) pins watch_out[1] at the
    five probe designs: it names the gearbox check, its one live value is cold-high at the gearbox
    input, and the check fails exactly when that is above the input rating (on the example design and
    the 20 mm ring), passes at the rating, and compares that peak, not the slip swing. The only gearbox
    check is that peak check, so "nothing checks" fatigue holds. The test failed on the old text
    ("watch-out 2 names the gearbox check") and passes on the new.
    `every_note_s_review_date_is_recorded_in_its_section` holds this heading to the note's date.
  - The dump the reviewer read: `magcoupling-rs/target/notes-review-BL059.txt` (both notes, five
    designs; git-ignored).
  - Sign-off: no new sign-off sheet (owner, spec section 2: the owner reviews the diff).
    `docs/analyses/2026-10-14-magcoupler-notes-signoff.md` stays the slice-2 record and shows the
    old text.

  The reviewer's report for the record, round by round, as returned:

  Round 1:

  <round 1's answer, verbatim>
  ```

  Add `Round 2:` and `Round 3:` blocks only for rounds that ran.

  **(b) Under `## when_slipping`,** after its last line and just before the line `## slip_heating`, append:

  ```markdown
  ### BL-059 re-review (<PASS_DATE>)

  - Why: as under `## hot_low_cold_high`, `### BL-059 re-review (<PASS_DATE>)`: watch_out[0], which
    said nothing checks the gearbox against these swings, became false. The note's `review` date
    moves to the date in this heading. The slice-2 review above is not rewritten.
  - watch_out[0]:
    - Old: "Nothing checks the gearbox against these swings: when cold they can reach the cold-high
      torque, on this design {metal.cold_high_input_Nm} at its input; check its rating."
    - New (<n> words raw): "<the text as shipped>"
  - sources[5] (new): `notes::GEARBOX_CHECK_SOURCE`, quoted under `## hot_low_cold_high`.
  - Reviewer and rounds: reviewed with `hot_low_cold_high` in the same rounds. The full answers are
    as under that note's `### BL-059 re-review (<PASS_DATE>)`. Verdict for this note: PASS.
  - Must-fix items applied to this note (old -> new, and why): <"none", or one bullet per item>.
  - Should-fix offered for this note, not applied: <"none", or one bullet per item>.
  - The ripple row's on-screen label: "Nominal slip swing at the gearbox input" (T5); sentence 0, reviewed in slice 2
    against "Estimated gearbox input torque ripple amplitude", <the reviewer's finding>.
  - Tests: `the_slip_note_s_claims_hold_at_every_probe_design` (`tests/explain.rs`) pins watch_out[0]
    at the five probe designs. It names the gearbox check, and its one live value is cold-high at the
    gearbox input, which lies above the ripple there. The check compares that peak with the input
    rating: it fails with the rating between the ripple and cold-high at the input. The test failed
    on the old text ("watch-out 1 names the gearbox check") and passes on the new.

  The reviewer's report for the record on this note's fields, round by round, as returned:

  Round 1:

  <the part of round 1's answer on when_slipping watch_out[0] and the shared source, verbatim>
  ```

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain -- every_note_s_review_date_is_recorded_in_its_section every_note_names_its_committed_review_section`
  Expected: `test result: ok. 2 passed; 0 failed`. The `###` headings do not count as a second `## <id>` section.

- [ ] **Step 12: Docs that ship with this change**

  **(a) `magcoupling-rs/README.md`, the `tests/explain.rs` row (~:656).** Three replacements; each old text occurs
  exactly once in the file (checked with `grep -o … | wc -l`). Each block is one line in the file; the backticks
  are literal.

  1. The torque-band watch-out test. Old:
     ```text
     on the example design hot-low fails while the hot-day torque meets the requirement, a hotter day leaves hot-low unchanged and cold-high at the gearbox input is above the input rating;
     ```
     New:
     ```text
     on the example design hot-low fails while the hot-day torque meets the requirement and a hotter day leaves hot-low unchanged; at the five probe designs the gearbox check, which the second watch-out names with cold-high at the gearbox input as its one live value, fails exactly when that is above the input rating (on the example design and the 20 mm ring), passes at the rating (the rating 1 % either side of cold-high at the input, and at it) and fails with the rating between the ripple and cold-high at the input, so it compares that peak and not the swing; the engine's only gearbox check is that peak check (no fatigue check), and both notes cite it and the rating's open datasheet question (`notes::GEARBOX_CHECK_SOURCE`);
     ```
  2. The slip-note test. Old:
     ```text
     the ripple at the gearbox input is the pull-out through the ratio and the efficiency, and cold-high at the input lies above it;
     ```
     New:
     ```text
     the ripple at the gearbox input is the pull-out through the ratio and the efficiency, and cold-high at the input lies above it; the first watch-out names the gearbox check, which compares that peak with the input rating (the torque band's assertions);
     ```
  3. The new test. Old (it ends with a space):
     ```text
     and that section to the report, once; 
     ```
     New (it ends with a space):
     ```text
     and that section to the report, once; `every_note_s_review_date_is_recorded_in_its_section` holds each such note's review date to its section (its "Date:" or a `###` re-review heading ending in that date, as BL-059's re-review of the torque-band and slip notes); 
     ```

  **(b) `magcoupling-rs/README.md`, the `src/gui/guidance.rs` row (~:670).** Old (it ends with a space):
  ```text
  the gearbox ripple row's Why link leads to the slip note, which owns its equation; that note's first watch-out also says nothing checks the gearbox. 
  ```
  New (it ends with a space):
  ```text
  the gearbox ripple row's Why link leads to the slip note, which owns its equation. 
  ```
  Task 2 rewrote the clause just before this one ("no dashboard row, guide or check compares a gearbox torque … until
  BL-059") and left this one for this task.

  **(c) `docs/ai/03-structure.yaml` (~:73).** Old:
  ```text
  release_notes_are_reviewed, every_note_names_its_committed_review_section, start_here_is_fourteen_steps_and_thirteen_while_paused
  ```
  New:
  ```text
  release_notes_are_reviewed, every_note_names_its_committed_review_section, every_note_s_review_date_is_recorded_in_its_section, start_here_is_fourteen_steps_and_thirteen_while_paused
  ```

  If the reviewer changed either watch-out, also run `git grep -n -F "<the draft text>" -- magcoupling-rs/README.md docs/ai`.
  It should print nothing. Task 9 quotes the final texts in `05-update-tracker.md`.

- [ ] **Step 13: Suites, clippy both ways, fmt**

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
  ```
  Expected:
  - every `test result:` line reads `ok` (the lib 867 passed with `--features app` and 229 without; tests/explain.rs:
    58 passed with `--features app` and 57 without, 1 ignored each);
  - both clippy runs end `Finished` with no warning;
  - `--check` prints nothing.

- [ ] **Step 14: Commit**

  Run `git status --short`. Expected: ` M` on exactly the five files below, and `?? docs/analyses/2026-05-28-press-4bar-analysis.md`
  (never staged). Write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T6.txt`:

  ```text
  fix(magcoupler): BL-059 gearbox watch-outs name the check (physics re-review PASS)

  The torque-band note's second watch-out and the slip note's first said nothing
  checks the gearbox; the gearbox check now compares cold-high at the gearbox input
  with the input rating. Both watch-outs name it; both notes cite it and the
  rating's open datasheet question (notes::GEARBOX_CHECK_SOURCE); both passed a
  fresh-context physics re-review, recorded as "### BL-059 re-review" under each
  note in docs/analyses/2026-10-14-magcoupler-notes-review.md, and carry its date.
  tests/explain.rs pins the watch-outs at the five probe designs (the check fails
  exactly when cold-high at the input is above the rating, passes at it, and
  compares that peak, not the slip swing) and now holds every note's review date
  to its record.

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/tests/explain.rs docs/analyses/2026-10-14-magcoupler-notes-review.md magcoupling-rs/README.md docs/ai/03-structure.yaml
  git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T6.txt
  ```

  A second commit, `fix(magcoupler): BL-059 watch-out re-review round <n>: …` (message file
  `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T6-fix.txt`), is used only if a review fix lands after this
  commit, for example when the controller asks for another round. The same date and record rules apply.

---

### Task 7: Panel and export coverage of the gearbox check

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

**Files:**
- Modify (tests only): `magcoupling-rs/src/gui/panel.rs` (a `GEARBOX_CHECK` test constant, the shared helper `table_badges_on`, three tests, the end-effect table test's closure replaced by the helper)
- Modify (tests only): `magcoupling-rs/src/gui/results_table.rs` (one test)
- Modify (tests only): `magcoupling-rs/src/gui/spreadsheet.rs` (one test)
- Docs: `magcoupling-rs/README.md` (test rows of `panel.rs` ~:663, `results_table.rs` ~:667, `spreadsheet.rs` ~:668)

**Interfaces:**
- Consumes: T1 `crate::engine::gearbox::{ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN, WITHIN_THE_GEARBOX_RATING}`, `DesignResults::gearbox.cold_high_check`, its engine label (no unit, no cell), schema order (the `gearbox` group after `housing`); T2 `DASHBOARD` (the row at index 15, between the clamp row and the space claim's), `END_EFFECT_ROWS` greying, `verdict_level`, `GEARBOX_GUIDE.learn` and `ROW_LEARN` = `Learn::Note(TORQUE_BAND_NOTE)`; `guidance::{TORQUE_BAND_NOTE, Learn, row_learn}`, `explorer::{note_fallback, NoteFallback, NEAREST_NOTE, NO_NOTE, reviewed_note}`; `results_table::{results_csv, results_json, csv_field, RUST_ONLY}`; the spreadsheet tests' `sheet`, `column`, `row_of`, `snapshot`, `short_magnets`.
- Produces: tests only, and the panel tests' helper `fn table_badges_on(output, visuals, row, in_table) -> Vec<Level>`.

**Counts already pinned and broken at T2 (T2's updates, not T7's):** the defaults' 5 red failing checks (`dashboard::tests::the_failing_checks_are_the_red_then_the_amber`, ~:1381); the manual-magnet variant's 7 (the same test; `panel::tests::the_failing_filter_shows_exactly_the_failing_checks_with_their_badges`, ~:3045 `failing.len()` 6 -> 7, ~:3047 "6 of" -> "7 of", ~:3081 badges `(4, 2)` -> `(5, 2)`; and `results_table::tests::the_engine_order_and_the_failing_filter_are_flat`, ~:1282, 6 -> 7); the end-effect table test's dashboard checks (`panel.rs` ~:3141, 7 -> 8). Not pinned, so self-adjusting and green at T2: the XLSX Summary's row count (`DASHBOARD.len()`), the CSV and JSON row counts (`result_rows`), Explain over every row (it iterates `DASHBOARD`), and every check's XLSX level (it iterates `CHECKS`). T7 adds the explicit tests below.

**These tests pass on arrival.** They pin wiring that T1 and T2 built, so they pass when written. Step 7 proves that each one fails when that wiring breaks: three planted faults, each reverted.

- [ ] **Step 1: Check that the wiring T7 pins is committed**

Run: `git diff --quiet HEAD -- magcoupling-rs/src/engine/gearbox.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/guidance.rs && echo clean`
Expected: `clean`. Step 7 edits these three files temporarily and must find them clean.

- [ ] **Step 2: Share the table-badge helper (panel.rs)**

In `magcoupling-rs/src/gui/panel.rs`, `mod tests`, replace:

```rust
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";
```

with:

```rust
    const MEASURED_DRAG: &str = "metal.measured_drag_Nm";
    const GEARBOX_CHECK: &str = "gearbox.cold_high_check";
```

Replace:

```rust
    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
```

with:

```rust
    /// The badge levels drawn on the line of `row`, a rect of the results table's text: the
    /// badge circles whose centre `in_table` places in the table, by x (the dashboard draws its
    /// own beside it).
    fn table_badges_on(
        output: &egui::FullOutput,
        visuals: &egui::Visuals,
        row: egui::Rect,
        in_table: impl Fn(f32) -> bool,
    ) -> Vec<Level> {
        let shapes = crate::gui::test_support::flat_shapes(output);
        [Level::Good, Level::Caution, Level::Bad]
            .into_iter()
            .filter(|level| {
                shapes.iter().any(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(visuals)
                            && in_table(c.center.x)
                            && (c.center.y - row.center().y).abs() < row.height())
                })
            })
            .collect()
    }

    #[test]
    fn out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too() {
```

In that test, replace the closure:

```rust
        let centre = |x: f32| x > INPUTS_WIDTH && x < dashboard_left;
        let badges_on = |output: &egui::FullOutput, row: egui::Rect| -> Vec<Level> {
            let shapes = flat_shapes(output);
            [Level::Good, Level::Caution, Level::Bad]
                .into_iter()
                .filter(|level| {
                    shapes.iter().any(|shape| {
                        matches!(shape, egui::Shape::Circle(c)
                            if c.fill == level.color(&visuals)
                                && centre(c.center.x)
                                && (c.center.y - row.center().y).abs() < row.height())
                    })
                })
                .collect()
        };
```

with:

```rust
        let centre = |x: f32| x > INPUTS_WIDTH && x < dashboard_left;
```

and its call:

```rust
            assert_eq!(badges_on(&output, row), want, "{}", line.path);
```

with:

```rust
            assert_eq!(
                table_badges_on(&output, &visuals, row, centre),
                want,
                "{}",
                line.path
            );
```

(The test's `use crate::gui::test_support::flat_shapes;` stays: its "Seven red badges" count still calls it.)

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib out_of_range_end_effect_drops_the_pull_out_checks_badges`
Expected: `... ok` (the same assertions, now through the helper).

- [ ] **Step 3: The three panel tests**

In `magcoupling-rs/src/gui/panel.rs`, replace:

```rust
    #[test]
    fn explain_never_shows_no_teaching_note_on_a_dashboard_row() {
```

with:

```rust
    #[test]
    fn explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it() {
        // BL-059: the gearbox check has no equation and no note of its own. With Explain on,
        // its row offers the note its Why link leads to, the torque band (a row's own link comes
        // first), never "no note", and the link beside the lead opens it on its own.
        use crate::engine::explain::notes::note_for;
        use crate::gui::explorer::{NEAREST_NOTE, NO_NOTE, NoteFallback, note_fallback};
        use crate::gui::guidance::{Learn, TORQUE_BAND_NOTE, row_learn};
        use crate::gui::typeset::glyph_safe;
        assert!(note_for(GEARBOX_CHECK).is_none());
        assert_eq!(
            row_learn(GEARBOX_CHECK),
            Some(Learn::Note(TORQUE_BAND_NOTE))
        );
        let NoteFallback::Note(note) = note_fallback(GEARBOX_CHECK) else {
            panic!("{GEARBOX_CHECK} offers no note");
        };
        assert_eq!(note.id, TORQUE_BAND_NOTE);
        let mut harness = Harness::on_screen(TALL);
        harness.panel.explorer.explain = true;
        harness.panel.explorer.open_path(GEARBOX_CHECK);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, NO_NOTE), 0);
        assert_eq!(count(&output, NEAREST_NOTE), 1);
        // The one copy of the title on the lead's line, left of the dashboard (whose rows draw
        // the same title in their Why lines).
        let lead = text_rect(&output, NEAREST_NOTE).expect("the lead");
        let beside: Vec<egui::Pos2> = last_characters(&output, &glyph_safe(note.title))
            .into_iter()
            .filter(|at| {
                at.x < TALL.x - DASHBOARD_WIDTH && (at.y - lead.center().y).abs() < lead.height()
            })
            .collect();
        assert_eq!(beside.len(), 1, "the link beside the lead: {beside:?}");
        harness.click(beside[0]);
        assert_eq!(harness.panel.explorer.note(), Some(TORQUE_BAND_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn explain_never_shows_no_teaching_note_on_a_dashboard_row() {
```

and replace:

```rust
    #[test]
    fn the_status_line_counts_the_checks_at_default_and_after_a_fix() {
```

with:

```rust
    #[test]
    fn the_gearbox_row_s_own_why_link_opens_the_torque_band_note() {
        // BL-059: the gearbox check's Why link leads to the torque-band note, as the hot-low,
        // hot-minimum and cold-high rows above it do, so those rows draw the note's title first.
        // The copy clicked is the gearbox row's own: on the dashboard, below the row's label and
        // above the next row's. The click edits nothing.
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::guidance::TORQUE_BAND_NOTE;
        use crate::gui::typeset::glyph_safe;
        let mut harness = Harness::on_screen(TALL);
        let output = harness.frame(Vec::new());
        let note = crate::gui::explorer::reviewed_note(TORQUE_BAND_NOTE)
            .expect("the torque-band note is reviewed");
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
            .position(|&(path, _)| path == GEARBOX_CHECK)
            .expect("the gearbox row");
        let (row, next) = (label_top(GEARBOX_CHECK), label_top(DASHBOARD[index + 1].0));
        let links: Vec<egui::Pos2> = last_characters(&output, &glyph_safe(note.title))
            .into_iter()
            .filter(|at| on_dashboard(at.x))
            .collect();
        assert!(links[0].y < row, "the rows above draw it first: {links:?}");
        let own: Vec<egui::Pos2> = links
            .iter()
            .copied()
            .filter(|at| at.y > row && at.y < next)
            .collect();
        assert_eq!(own.len(), 1, "one link under the gearbox row: {links:?}");
        harness.click(own[0]);
        assert!(harness.panel.explorer.open);
        assert_eq!(harness.panel.explorer.note(), Some(TORQUE_BAND_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_failing_filter_lists_the_gearbox_check_with_its_red_badge() {
        use crate::gui::dashboard::failing_checks;
        use crate::gui::results_table::{FAILING_ONLY, table_entries};
        // BL-059: at the defaults the gearbox check fails beside the cup ring, the hot minimum,
        // the clearance and the cup wall, all red. The filter lists the five, and the gearbox
        // check's row once in the table, with a red badge.
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results(), false);
        assert_eq!(failing.len(), 5, "{failing:?}");
        assert!(
            failing.iter().all(|&(_, level)| level == Level::Bad),
            "{failing:?}"
        );
        assert!(
            failing.contains(&(GEARBOX_CHECK, Level::Bad)),
            "{failing:?}"
        );
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("5 of {total} results")), 1);
        let in_table = |x: f32| x > INPUTS_WIDTH && x < SCREEN.x - DASHBOARD_WIDTH;
        let rows: Vec<egui::Rect> =
            text_rects(&output, result_info(GEARBOX_CHECK).unwrap().meta.label)
                .into_iter()
                .filter(|r| in_table(r.center().x))
                .collect();
        assert_eq!(rows.len(), 1, "the gearbox check's row in the table");
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(
            table_badges_on(&output, &visuals, rows[0], in_table),
            [Level::Bad]
        );
    }

    #[test]
    fn the_status_line_counts_the_checks_at_default_and_after_a_fix() {
```

- [ ] **Step 4: The CSV and JSON test (results_table.rs)**

In `magcoupling-rs/src/gui/results_table.rs`, `mod tests`, replace:

```rust
    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
```

with:

```rust
    #[test]
    fn the_exports_carry_the_gearbox_check_with_no_cell() {
        use crate::engine::gearbox::{
            ABOVE_THE_GEARBOX_RATING, GEARBOX_CHECK_UNKNOWN, WITHIN_THE_GEARBOX_RATING,
        };
        // BL-059: the gearbox check is a Rust-only result, like the space claim. The CSV and
        // JSON exports carry its engine label and verdict, last of the result groups (after the
        // housing group, before the sweeps), with no unit and no spreadsheet cell; each of its
        // three verdicts exports as the engine writes it.
        const PATH: &str = "gearbox.cold_high_check";
        let label = result_info(PATH).expect("a result").meta.label;
        let mut within = DesignInputs::default();
        within.coupling.gearbox_input_rating_Nm = 1.0;
        let mut unknown = DesignInputs::default();
        unknown.coupling.gearbox_input_rating_Nm = f64::NAN;
        for (inputs, verdict_start) in [
            (DesignInputs::default(), ABOVE_THE_GEARBOX_RATING),
            (within, WITHIN_THE_GEARBOX_RATING),
            (unknown, GEARBOX_CHECK_UNKNOWN),
        ] {
            let design = Design {
                inputs,
                ..Design::default()
            };
            let results = compute_all(&design.inputs);
            let verdict = results.gearbox.cold_high_check.clone();
            assert!(verdict.starts_with(verdict_start), "{verdict}");
            let csv = results_csv(&results);
            let lines: Vec<&str> = csv.split("\r\n").collect();
            let row = lines
                .iter()
                .position(|l| l.starts_with(&format!("{PATH},")))
                .expect("the gearbox check's CSV line");
            assert_eq!(
                lines[row],
                [PATH, label, verdict.as_str(), "", ""]
                    .map(csv_field)
                    .join(",")
            );
            assert!(lines[row - 1].starts_with("housing."), "{}", lines[row - 1]);
            assert!(
                lines[row + 1].starts_with("gap_sweep["),
                "{}",
                lines[row + 1]
            );
            let json: Json = serde_json::from_str(&results_json(&design, &results)).unwrap();
            let entry = json["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["path"] == PATH)
                .expect("the gearbox check's JSON row");
            assert_eq!(entry["label"], label);
            assert_eq!(entry["value"], Json::from(verdict.as_str()));
            assert_eq!(entry["unit"], "");
            assert_eq!(entry["cell"], Json::Null);
        }
    }

    #[test]
    fn the_json_export_holds_the_design_and_every_result() {
```

- [ ] **Step 5: The XLSX test (spreadsheet.rs)**

In `magcoupling-rs/src/gui/spreadsheet.rs`, `mod tests`, replace:

```rust
    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
```

with:

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

    #[test]
    fn the_summary_lists_the_warnings_that_fire_and_the_assumptions_banner() {
```

- [ ] **Step 6: Run the T7 tests**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_gearbox_row_s_own explain_offers_the_gearbox the_failing_filter_lists_the_gearbox the_exports_carry_the_gearbox the_summary_lists_eighteen out_of_range_end_effect_drops_the_pull_out_checks_badges`
Expected: `test result: ok. 6 passed`. The six are `the_exports_carry_the_gearbox_check_with_no_cell`, `the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge`, `the_gearbox_row_s_own_why_link_opens_the_torque_band_note`, `explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it`, `the_failing_filter_lists_the_gearbox_check_with_its_red_badge` and `out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too`.

- [ ] **Step 7: Prove each test fails when the wiring breaks (three planted faults, each reverted)**

Use the Step 6 command for every run. After each fault, undo the edit exactly and run `git diff --stat -- <that file>`. Expected: it prints nothing.

1. In `magcoupling-rs/src/gui/guidance.rs`, inside `GEARBOX_GUIDE`, change `    learn: Learn::Note(TORQUE_BAND_NOTE),` to `    learn: Learn::Note("when_slipping"),`. Expected: `the_gearbox_row_s_own_why_link_opens_the_torque_band_note` and `explain_offers_the_gearbox_row_its_why_note_and_the_link_opens_it` FAIL; the other four pass. Undo it.
2. In `magcoupling-rs/src/gui/dashboard.rs`, `verdict_level`, in the arm `("gearbox.cold_high_check", t) if t.starts_with(crate::engine::gearbox::ABOVE_THE_GEARBOX_RATING) => { Some(Bad) }`, change `Some(Bad)` to `Some(Caution)`. Expected: `the_failing_filter_lists_the_gearbox_check_with_its_red_badge` and `the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge` FAIL. Undo it.
3. In `magcoupling-rs/src/engine/gearbox.rs`, change `out_rust_only("", "Cold-high torque` to `out_rust_only("N·m", "Cold-high torque`. Expected: `the_exports_carry_the_gearbox_check_with_no_cell` and `the_summary_lists_eighteen_dashboard_rows_with_the_gearbox_badge` FAIL. Undo it.

Run: `git diff --quiet HEAD -- magcoupling-rs/src/engine/gearbox.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/guidance.rs && echo clean`
Expected: `clean`.

- [ ] **Step 8: Docs**

Each edit replaces the exact text in the first block (one occurrence) with the second.

`magcoupling-rs/README.md`, `panel.rs` row (~:663), at its end. Replace:

```text
with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither |
```

with:

```text
with Explain on, a value with nothing upstream to read draws `NO_NOTE` and `START_HERE_HINT` once each, and Explain off draws neither; BL-059: the gearbox row's own Why link, the copy of the torque-band title under the row's label (the hot-low, hot-minimum and cold-high rows draw it first), opens that note and changes no input; with Explain on the gearbox row, which has no equation or note of its own, offers its Why link's note as the closest note, and the link beside the lead opens it; at the defaults the failing filter lists five red checks, the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test) |
```

`magcoupling-rs/README.md`, `results_table.rs` row (~:667). Replace:

```text
`+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design;
```

with:

```text
`+inf` and `NaN` in CSV and JSON (E20's positive-beta design); the JSON export holds the design; the CSV and JSON exports carry the gearbox check (BL-059) with its engine label and each of its three verdicts, after the housing group and before the sweeps, with no unit and no cell;
```

`magcoupling-rs/README.md`, `spreadsheet.rs` row (~:668). Replace:

```text
the Summary (export time, version, link, sizing, both banners, the dashboard's rows, the warnings, the corrections applied);
```

with:

```text
the Summary (export time, version, link, sizing, both banners, the dashboard's rows, the warnings, the corrections applied); its 18 dashboard rows carry the gearbox check between the clamp's and the space claim's with its level (red at the defaults, green with a 1.0 N·m rating, none with the greyed note out of the end-effect range) and Rust-only for its cell, and the Results sheet's row carries the same level (BL-059);
```

- [ ] **Step 9: The suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`
Expected: every `test result: ok`, no `FAILED`; the lib `872 passed; 0 failed; 6 ignored` (867 after Task 6, plus the
five new tests).
Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`
Expected: every `test result: ok`.
Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings`
Expected: each ends `Finished`, with no warning.
Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `git diff --stat`. Expected: only `panel.rs`, `results_table.rs`, `spreadsheet.rs` and `README.md`.

- [ ] **Step 10: Commit**

Write the message with the Write tool to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T7.txt`:

```
test(magcoupler): the gearbox row's Why link, Explain, the failing filter and the exports cover the gearbox check (BL-059 T7)

The gearbox row's own Why link (not the first copy of the torque-band title, which the hot-low
row draws) opens the note; Explain offers the row its Why note; at the defaults the failing
filter lists the five red checks with the gearbox check's red badge; the CSV and JSON carry its
three verdicts with no unit or cell; the Summary lists 18 dashboard rows with its level (red,
green, greyed) and the Results sheet the same. The table-badge lookup is one shared helper.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/spreadsheet.rs magcoupling-rs/README.md
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T7.txt
git status --short
```
Expected: the commit lists those 4 files; `docs/analyses/2026-05-28-press-4bar-analysis.md` is still untracked and unstaged.

**Existing tests T7 moves:**
- `gui::panel::tests::out_of_range_end_effect_drops_the_pull_out_checks_badges_in_the_table_too`: its local `badges_on` closure is replaced by the module helper `table_badges_on(&output, &visuals, row, centre)`. The same check (the same circles, levels, x-range and row line); the new defaults filter test needs it too (DRY). T2's `checks.len()` 7 -> 8 edit in the same test is separate and untouched.

---

### Task 8: G-3 review of the gearbox check's on-screen texts, and its record

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

One fresh-context session-model review (slice-1 decision G-3) covers every new or changed on-screen string of BL-059
outside the notes. The audience is engineers who are not magnet specialists. Nothing may be false at the defaults, at
the probe designs or at any reachable input. Must-fix items are applied where each string lives, and tests follow
through the constants. The record goes into the help-review report under a new `## BL-059 gearbox check` section.

**Files:**
- Modify, only as the review asks:
  - `magcoupling-rs/src/engine/gearbox.rs`
  - `magcoupling-rs/src/gui/guidance.rs`
  - `magcoupling-rs/src/gui/plain_text.rs`
  - the tests that pin a reworded string by literal (found by grep in Step 3)
  - the doc lines that quote a reworded string
- Modify: `docs/analyses/2026-10-14-magcoupler-help-review.md` (append `## BL-059 gearbox check` at the end of the
  file, after `## Reachability strings (T19)`).
- Gitignored outputs:
  - `magcoupling-rs/target/g3/BL059-guidance.txt`, `BL059-overlay.txt` and `BL059-strings.diff`;
  - `.superpowers/sdd/2026-10-16-bl059-gearbox-check/t8-g3-r<N>.md`.

**Interfaces:**
- Consumes:
  - Task 1's engine constants, label and help;
  - Task 2's `GEARBOX_GUIDE` (label, why), `GEARBOX_UNKNOWN` and `gearbox_problem`;
  - Task 3's `gearbox_conflict`;
  - Task 5's overlay rows;
  - Task 6's final watch-outs (for consistency only);
  - the review tools `guidance_strings_for_review` and `overlay_strings_for_review`.
- Produces: the reviewed strings (Task 9 quotes them; Task 10's STOP summary lists them), and the G-3 record.

**The strings under review.**
- Approximate lines are from the plan-time prototype of Tasks 1-5; the final lines are what Step 1's grep prints.
- "Defaults" is the text as drawn at the default design, with the clamp section paused.

| # | String (draft) | Where it lives | Drawn |
|---|---|---|---|
| 1 | "Gearbox input torque" | `gui/guidance.rs`, `GEARBOX_GUIDE.label` (~:217) | heading of the gearbox block; inside the trade-off line ("… Hot-low torque needs it raised and Gearbox input torque needs it lowered …") |
| 2 | "Up to cold-high torque reaches the gearbox output before the coupling slips, and in each swing while it slips; at the gearbox input it must stay within the rating." | `GEARBOX_GUIDE.why` (~:219) | under the gearbox row |
| 3 | Problem sentence 1: "Cold-high at the gearbox input is {result_text(metal.cold_high_input_Nm)}, above its {input_text(coupling.gearbox_input_rating_Nm)} input rating." | `gearbox_problem` (~:1645-1660) | red block. Defaults: "Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating." |
| 4 | Problem sentence 2 (Task 3), two forms. First: "No magnet size, gap or count passes both this and the hot minimum: cold-high is {B} times hot-low here, so meeting the {required} requirement puts {R·B/(i·η)} on the gearbox input. It takes a gearbox rated for that, a requirement of at most {rating·i·η/B}, or a band of at most {rating·i·η/R} times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it)." Second, where rating·i·η < R: "No magnet size, gap or count passes both this and the hot minimum: cold-high is {B} times hot-low here, so meeting the {required} requirement puts {R·B/(i·η)} on the gearbox input. It takes a gearbox rated for that or a requirement of at most {rating·i·η/B}: the requirement alone puts {R/(i·η)} on the gearbox input, so no narrower band settles it." | `gearbox_conflict` (~:1662-1700) | after sentence 1, only when the conflict holds and the minimum temperature is at or below the operating one. Defaults (first form): 1.673, 2.500 N·m, 0.8806 N·m, 1.703 N·m, 1.140. Ratio 1 (second form): 0.3407 N·m and 2.632 N·m |
| 5 | "The gearbox check cannot be made: cold-high at the gearbox input or the rating is not a number." | `GEARBOX_UNKNOWN` (~:1643) | amber block (a NaN set on the struct) |
| 6 | "Within the gearbox input rating" | `engine/gearbox.rs`, `WITHIN_THE_GEARBOX_RATING` (~:33) | the row's value, green |
| 7 | "Above the gearbox input rating" + ": {q} N·m against {r} N·m" | `ABOVE_THE_GEARBOX_RATING` (~:37) and the `format!` in `compute` (~:54) | the row's value, red. Defaults: "Above the gearbox input rating: 0.80 N·m against 0.60 N·m" |
| 8 | "Gearbox check unknown: a torque is not a number" | `GEARBOX_CHECK_UNKNOWN` (~:41) | the row's value, amber |
| 9 | Engine label "Cold-high torque at the gearbox input against its rating" | the `results!` row in `engine/gearbox.rs` (~:26) | the dashboard row's label (no overlay label), the results table, exports |
| 10 | Engine help "Compares the cold-high torque at the gearbox input, cold-high divided by the ratio and efficiency, with the gearbox input torque rating. A peak check only." | same row (~:27) | the row's hover |
| 11 | Overlay help of `coupling.gearbox_input_rating_Nm`: "The gearbox's input torque rating. The gearbox check compares the cold-high torque at the gearbox input with it." | `gui/plain_text.rs` `HELP` (~:153-156, Task 5) | the input's hover. The Python-locked engine help stays: with Show spreadsheet details on, the input also shows "Original spreadsheet note: GAM 5:1 peak input rating.", the spreadsheet's own labelled note (ruling R-11; the reviewer is told) |
| 12 | Overlay label of `model.gearbox_input_ripple_Nm`: "Nominal slip swing at the gearbox input" | `plain_text.rs` `LABELS` (~:72-75, Task 5) | the ripple row's label |
| 13 | Overlay help of `model.gearbox_input_ripple_Nm`: "Torque swing while slipping: the pull-out at the operating temperature, without the production allowance, divided by the ratio and efficiency. The gearbox check compares cold-high at the gearbox input instead." (Task 5's text) | `plain_text.rs` `HELP` (~:157-160, Task 5) | the ripple row's hover |

Read in context: these are judged only for how the new words read in them, and are not reworded unless a new word
makes them wrong.
- the trade-off line and the face-gap side-effect line in the gearbox block at the defaults ("Raising Magnet face gap
  also lowers the pull-out, so Hot-low torque falls further.");
- the status line "Example design: fails 4 of 8 checks".

(The results table's "Gearbox" heading is never drawn: `groups_of` takes the dashboard rows into the Headline group
and drops the empty Gearbox group, so it is not reviewed.)

- [ ] **Step 1: Locate the strings and write the review sheets**

  ```bash
  git grep -n -e 'label: "Gearbox input torque"' -e 'why: "Up to cold-high' -e 'const GEARBOX_UNKNOWN' -e 'fn gearbox_problem' -e 'fn gearbox_conflict' -- magcoupling-rs/src/gui/guidance.rs
  git grep -n -e 'cold_high_check: String' -e 'pub const WITHIN_THE_GEARBOX_RATING' -e 'pub const ABOVE_THE_GEARBOX_RATING' -e 'pub const GEARBOX_CHECK_UNKNOWN' -e 'N·m against {rating} N·m' -- magcoupling-rs/src/engine/gearbox.rs
  git grep -n -e '"coupling.gearbox_input_rating_Nm",' -e '"model.gearbox_input_ripple_Nm",' -- magcoupling-rs/src/gui/plain_text.rs
  mkdir -p magcoupling-rs/target/g3
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/g3/BL059-guidance.txt
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/g3/BL059-overlay.txt
  git diff -U0 main -- magcoupling-rs/src/engine/gearbox.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/plain_text.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/result_groups.rs > magcoupling-rs/target/g3/BL059-strings.diff
  grep -n "Gearbox input torque" magcoupling-rs/target/g3/BL059-guidance.txt
  grep -n -A2 "gearbox_input_r" magcoupling-rs/target/g3/BL059-overlay.txt
  ```
  Expected:
  - Each of the three `git grep` commands prints its lines. Copy the line numbers into the review prompt below.
  - Each `cargo test` ends `test result: ok. 1 passed`.
  - The guidance grep prints these lines, with the defaults' Problem as in rows 3 and 4 above. Task 3's final wording
    may differ from the draft. With Task 4's "gearbox only" design in `designs()`, a
    `[problem/Gearbox input torque/gearbox_only]` line reads "Cold-high at the gearbox input is 0.7070 N·m, above its
    0.6000 N·m input rating." with no sentence 2.
    ```text
    [why/Gearbox input torque] Up to cold-high torque reaches the gearbox output before the coupling slips, …
    [fix/Gearbox input torque/defaults] Lower Axial magnet length, both rings
    [fix/Gearbox input torque/defaults] Raise Magnet face gap
    [note/Gearbox input torque/defaults] One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Gearbox input torque needs it lowered, so change another input to settle one of them.
    [note/Gearbox input torque/defaults] Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further.
    [problem/Gearbox input torque/defaults] Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating. No magnet size, …
    ```
  - The overlay grep prints the rating's `[help]` row ("was: GAM 5:1 peak input rating.") and the ripple row's
    `[label]` and `[help]` rows ("was: Estimated gearbox input torque ripple amplitude").

- [ ] **Step 2 (controller): the G-3 review, one fresh-context reviewer**

  Dispatch it with the Agent tool: subagent `general-purpose`, `model` omitted (session model: physics, CLAUDE.md
  section 5), fresh context. Save each answer verbatim as `.superpowers/sdd/2026-10-16-bl059-gearbox-check/t8-g3-r<N>.md`.
  The brief (round 1, verbatim; paste Step 1's grep output where it says so):

  ```text
  You are a fresh-context physics reviewer (session model) doing the G-3 review of BL-059's new on-screen texts in
  the magnetic coupling calculator (crate magcoupling-rs; repo C:/Users/Cole/source/repos/linkage_simulation, branch
  magcoupling/bl059-gearbox-check). Read-only: do not edit the repository and run no git command that changes state.
  Probe the engine only from a scratch crate outside the repository, in your scratchpad, with its own CARGO_TARGET_DIR
  there. Its Cargo.toml:
      [package]
      name = "bl059g3"
      version = "0.1.0"
      edition = "2024"
      publish = false
      [dependencies]
      magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
      [workspace]
  The library is `magcoupling`: magcoupling::engine::api::{DesignInputs, compute_all}; a guide's Problem is
  (g.problem)(&inputs, &results) for g in magcoupling::gui::guidance::CHECK_GUIDES. The test designs in
  gui::test_support are crate-private: build them from the inputs given below. The first build takes several minutes.

  Audience: engineers who are not magnet specialists. Plain words. Every string must be true at the default design, at
  the five probe designs (engine::explain::probes::probe_designs) and at every reachable input: every value inside
  its slider. Sliders: rating 0.05-5.0 N·m (step 0.01); ratio 1-20; efficiency 0.5-1.0; operating temperature -40 to
  150 °C; minimum temperature -60 to 20 °C; variation 0-0.5; requirement 0.1-10 N·m.

  What BL-059 is: the spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md (read sections 1-6). The
  owner's decisions:
  - 1A: the check compares cold-high at the gearbox input (metal.cold_high_input_Nm = cold-high / (ratio x efficiency))
    with the gearbox input torque rating (0.6 N·m), which is the same as 2.85 N·m at the output (ratio x efficiency x
    rating);
  - 2A: the rating's datasheet type is open, so no text may call the rating a peak or a nominal rating;
  - 4A: Problem sentence 2 says when no magnet size, gap or count passes both this and the hot minimum check;
  - 6A: the fixes are geometry only;
  - 7A: change only what became false or misleading;
  - 8A: fatigue from repeated slips is not checked (backlog).

  The strings under review, with file and line (from git grep):
  <paste Step 1's three git grep outputs>
  1 the guide label; 2 the Why; 3 Problem sentence 1; 4 Problem sentence 2 (the conflict); 5 the Problem when the check
  cannot be made; 6-8 the engine's three verdict texts (and the ": {q} N·m against {r} N·m" tail); 9 the engine label
  (the dashboard row's label); 10 the engine help; 11 the gearbox input torque rating's overlay help; 12 and 13 the
  ripple row's overlay label and help. With Show spreadsheet details on, the rating input also shows "Original
  spreadsheet note: GAM 5:1 peak input rating.": the spreadsheet's own labelled note, Python-locked, which stays (the
  owner's decision 2A binds our texts, not the spreadsheet's); judge only that it reads as the spreadsheet's note. Read
  in context only (judge just how the new words read there): the trade-off line, the face-gap side-effect line, the
  status line "Example design: fails 4 of 8 checks".
  Sheets: magcoupling-rs/target/g3/BL059-guidance.txt (every guide string at every test design; the gearbox guide's
  lines carry "Gearbox input torque"), BL059-overlay.txt (each overlay row, now and was) and BL059-strings.diff (the
  source diff). For consistency only, not for review: the two notes' gearbox watch-outs as Task 6 shipped them
  (engine/explain/notes.rs, hot_low_cold_high watch_out[1] and when_slipping watch_out[0]); the gearbox row's Why link
  opens hot_low_cold_high.

  Read the code behind every string: engine/gearbox.rs (compute: equality passes, +inf fails, NaN is unknown; the
  quoted cold-high is at least 0.01 above the quoted rating); engine/api.rs (grep "gearbox::compute");
  engine/metal_design.rs (grep "cold_high_input_Nm"); engine/model.rs (grep "gearbox_input_ripple_Nm",
  "gearbox_reference_Nm"); gui/guidance.rs (GEARBOX_GUIDE, gearbox_problem, gearbox_conflict, level_of, helps,
  trade_offs); gui/dashboard.rs (DASHBOARD, END_EFFECT_ROWS, verdict_level).

  Traps (each one written is a must-fix): name the side of the gearbox for every torque; efficiency raises the input
  torque (never "reduces the load"); cold-high is an expectation ("up to", "can reach"; never "at most", never what
  every unit reaches); never "the coupling protects the gearbox" (a pass means only that the modelled cold-high peak is
  within the input rating; tests ban "gearbox" with "protect" in guide, overlay and note texts); "peak" names the
  compared torque, never the rating (2A); the ripple row is the nominal swing at the operating temperature (no "warm":
  the operating-temperature slider reaches below the minimum); geometry cannot narrow the band (the band, cold-high /
  hot-low, depends only on the two temperatures, the grade's coefficient and the variation); no "hot-day" in these
  texts; no internal paths, ids, cells, "BL-059", "Rust-only" or "clamp" on screen; glyph-safe ("->", "<=");
  plain_text_violations empty.

  Caps (tests hold them):
  - the Why is one sentence of at most 30 words ending in a full stop, with no ". " inside;
  - each Problem sentence stays one sentence;
  - an overlay help has at most 30 words in 1-2 sentences, with no braces and no "guarantee";
  - an overlay label has at most 47 characters and names one quantity;
  - the engine label and help pass the plain-words guard, and the help has at most 30 words in 1-2 sentences;
  - the guide label is distinct from the other guides' labels.
  If the engine label reads too long as the dashboard row's label, you may ask for an overlay label (at most 47
  characters) instead of a new engine label.

  Probe at least, from compute_all (and gui::guidance for the Problems):
  (1) the defaults: verdict "Above the gearbox input rating: 0.80 N·m against 0.60 N·m"; the Problem with 0.8049,
      0.6000, 1.673, 2.500, 0.8806, 1.703 and 1.140, and sentence 2;
  (2) the five probe designs: cold-high at the input 0.8049, 0.5893, 0.5082, 0.07117 and 0.7602 N·m; above, within,
      within, within, above;
  (3) the rating at 0.80, one slider step above cold-high at the input: "Above the gearbox input rating: 0.81 N·m
      against 0.80 N·m", and sentence 2 still shows (0.8806 > 0.80); the rating at 0.05 and at 5.0;
  (4) the tests' passing design (the defaults with face gap 1.8 mm, cup wall 2.0 mm, requirement 1.6 N·m and the
      rating at 1.0 N·m): passes, no Problem; the "gearbox only" design (the same with the rating back at 0.6):
      0.7070 fails, R·B/(i·η) = 0.5636, so no sentence 2;
  (5) Auto on the axial length at the defaults (13.770 mm): hot-low 2.5000, 0.8806 at the input, red with sentence 2;
      the minimum temperature at 20 °C with variation 0 and Auto length (11.902 mm): 0.5664, passes, no Problem;
  (6) a NaN rating set on the struct: the unknown verdict and Problem 5;
  (7) the end-effect model out of range (the tests' short magnets: the defaults with c_end 0.5, both rings' parts
      cleared, and manual inner and outer lengths of 2.0 mm): the row is greyed, with no badge and no Problem (the
      engine text reads "Within" there on a negative torque, which is why the row greys);
  (8) the operating temperature below the minimum temperature, both in their sliders: op -40 °C, min 20 °C, variation
      0: cold-high 2.8930 below hot-low 3.3246, and at the input 0.6091 below the ripple 0.6999 N·m (variation 0.05:
      0.6395 against 0.6999; op 0 °C, min 20 °C, variation 0: 0.6091 against 0.6386). Problem sentence 2 is left out
      there (gearbox_conflict requires the minimum temperature at or below the operating one; the corner is backlog
      BL-067); judge the Why's "Up to cold-high" (a static text: "can reach" is true there, "up to" is not) and the
      ripple help there;
  (9) ratio 1 with efficiency 0.5 (cold-high at the input 7.647 N·m, ripple 5.377) and ratio 20 with efficiency 1.0
      (0.1912): efficiency raises the input torque;
  (10) at the defaults, no lever in sentence 2's parentheses passes alone: variation 0 gives 0.6509 at the input,
      minimum 20 °C gives 0.7663, SmCo 2:17 gives 0.7581. The sentence names targets, not single levers;
  (11) the ratio at 1, and the efficiency at 0.5, each alone at the defaults: sentence 2's second form, "It takes a
      gearbox rated for that or a requirement of at most 0.3407 N·m: the requirement alone puts 2.632 N·m on the
      gearbox input, so no narrower band settles it." (efficiency 0.5: 0.8966 and 1.000 N·m);
  (12) "meeting the {required} requirement puts {x} on the gearbox input": a design that meets the requirement with
      room puts more; say whether it should read "puts at least";
  (13) the check's on-screen name: the help, the watch-outs, the source and the unknown texts say "the gearbox check",
      while the row shows the engine label "Cold-high torque at the gearbox input against its rating" and the guide's
      block "Gearbox input torque"; say whether an engineer finds "the gearbox check" on the dashboard, or name it as
      the row reads.

  Return, for the record (it is pasted verbatim into docs/analyses/2026-10-14-magcoupler-help-review.md):
  1. "Reviewer: physics reviewer (session model), fresh context, <date>; round <n>."
  2. Read: what you read; the probes you ran, with their numbers.
  3. A table, one row per string 1-13: PASS or MUST-FIX, with the reason.
  4. Each MUST-FIX: the string's place (constant or row), Old, New (the exact text, within the caps), Why (the engine
     fact and the probe that make the old text false or misleading) and, where a claim can be pinned, the test
     assertion.
  5. SHOULD-FIX offered, in the same form.
  6. Traps checked, one line each. Typed numbers: every number typed in a string, and whether it is allowed (the
     strings should type none).
  7. The context lines (trade-off, side effect, status): fine, or why not.
  8. Verdict: PASS or MUST_FIX.
  Your words win over the drafts within the caps: write the text you would ship.
  ```

  Rounds 2 and 3 use the same brief, with three changes:
  - add the round's changes (each string old -> new);
  - add "Re-review only the changed strings and every string whose meaning depends on them; round 1's findings on
    the unchanged strings stand";
  - give the earlier answers' paths.
  Each round is a new fresh context.

- [ ] **Step 3: Apply the must-fix items: constants first, tests follow through the constants**

  For each must-fix item, edit the string only where it lives (the table's "Where it lives" column), once:
  - **The guide label or Why:** the field of `GEARBOX_GUIDE`. Tests read them through `GEARBOX_GUIDE` and
    `CHECK_GUIDES`: `every_why_is_one_short_sentence`, the label-distinct test, and the plain-words, glyph, clamp and
    protect bans over `every_text()`.
  - **The unknown Problem:** `GEARBOX_UNKNOWN`. `every_text()` pushes it, and Task 2's tests compare with the constant.
  - **Problem sentences 1 and 2:** the `format!` template in `gearbox_problem` or `gearbox_conflict`. Task 3's tests
    recompute the numbers with the same formatters. Change any words they pin in the same edit.
  - **The verdicts:** `WITHIN_THE_GEARBOX_RATING`, `ABOVE_THE_GEARBOX_RATING` and `GEARBOX_CHECK_UNKNOWN`.
    `verdict_level` and the classification tests read the constants. If the ": {q} N·m against {r} N·m" tail changes,
    edit the `format!` in `gearbox::compute`, then Task 1's tests that pin "0.80 N·m against 0.60 N·m" and
    "0.81 N·m against 0.80 N·m".
  - **The engine label or help:** the `out_rust_only(...)` row of `GearboxResults`. `tests/schema.rs` holds the
    plain-words guard and the 30-word, 2-sentence cap; Task 1's Rust-only test reads the metadata.
  - **An overlay label for the check row,** if the reviewer asks for one instead: a new `LABELS` row in
    `gui/plain_text.rs` of at most 47 characters. `the_canonical_names_are_the_spec_s` gains the row, as Task 5 did
    for the ripple row.
  - **The overlay rows:** their `HELP` or `LABELS` row. The overlay caps test and the canonical-names test read the
    table.

  Then:
  - run `git grep -n -F "<old text>" -- magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai`
    for each changed string. Replace every test literal that has a constant with the constant; update every remaining
    literal and every doc line to the new text;
  - where a must-fix names a claim a test can check, write that assertion first, see it fail on the old text, then
    apply the text;
  - run `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app` and
    `cargo test --manifest-path magcoupling-rs/Cargo.toml --test schema`; all `ok`;
  - rewrite the Step 1 sheets, then dispatch the next round (Step 2).
  At most three rounds. If round 3 still returns a must-fix, **stop** and report to the controller with the open
  items: it is an owner decision. Should-fix items stay offered and are not applied unless the controller says so.

- [ ] **Step 4: Write the record in `docs/analyses/2026-10-14-magcoupler-help-review.md`**

  Append the block below at the end of the file, after the last line of `## Reachability strings (T19)`, keeping the
  file's CRLF line ends. It has the shape of the T10 and T19 sections. Angle-bracket slots are filled from the
  reviewer's answers and from Step 3; everything else is verbatim. Leave out "Round 2" and "Round 3" if they did not
  run.

  ```markdown
  ## BL-059 gearbox check

  - Scope: not an overlay batch alone. These are BL-059's new and changed on-screen strings
    (spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`, section 6), reviewed
    under the same G-3 rule:
    - the gearbox guide (`gui/guidance.rs`): its label (`GEARBOX_GUIDE.label`), its Why
      (`GEARBOX_GUIDE.why`), and its Problem: sentence 1 (`gearbox_problem`), the conflict sentence
      2 (`gearbox_conflict`) and the text when the check cannot be made (`GEARBOX_UNKNOWN`);
    - the engine (`engine/gearbox.rs`): the three verdict texts (`WITHIN_THE_GEARBOX_RATING`,
      `ABOVE_THE_GEARBOX_RATING` with its ": {q} N·m against {r} N·m" tail,
      `GEARBOX_CHECK_UNKNOWN`), and the label and help of `gearbox.cold_high_check`, the
      dashboard row's label;
    - the overlay (`gui/plain_text.rs`): the help of the gearbox input torque rating
      (`coupling.gearbox_input_rating_Nm`), and the label and help of the ripple row
      (`model.gearbox_input_ripple_Nm`);
    - read in context, judged only for how the new words read there: the trade-off and face-gap
      side-effect lines in the gearbox block at the defaults and the status line ("Example design:
      fails 4 of 8 checks").
  - Read: the sheets `magcoupling-rs/target/g3/BL059-guidance.txt` (`guidance_strings_for_review`),
    `BL059-overlay.txt` (`overlay_strings_for_review`) and `BL059-strings.diff` (`git diff -U0
    main` of the source files); the spec, sections 1 to 6; the plan's trap list; the two notes'
    gearbox watch-outs as Task 6 shipped them, for consistency only; the engine and GUI code behind
    every string. Each round probed the engine from a scratch crate outside the repository:
    <the probes each round ran, with their numbers, as returned>.
  - Round 1: <n> must-fix items, text applied verbatim (constants first, tests through the
    constants):
    - `<the constant or row>`.
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
    strings through their constants">.
  - Traps: none of the plan's is written in any string: each torque names its side of the
    gearbox; efficiency raises the input torque; cold-high is an expectation; nothing says the
    coupling protects the gearbox; "peak" names the compared torque only; the ripple row is the
    nominal swing at the operating temperature; geometry does not narrow the band; no "hot-day";
    no path, id, "BL-059", "Rust-only" or "clamp"; glyph-safe. <Or the trap a round found, and its
    fix.>
  - Verdict: PASS (round <n>).

  ### Report for the record, round 1 (as returned)

  <round 1's answer, verbatim>

  ### Report for the record, round 2 (as returned)

  <round 2's answer, verbatim>
  ```

- [ ] **Step 5: Docs that quote a changed string**

  For every string the review changed, run
  `git grep -n -F "<old text>" -- magcoupling-rs/README.md docs/ai`. The command must print nothing once the README
  and `docs/ai` lines quote the new text. Tasks 2, 3 and 5 wrote those lines; this task updates only the words.

  If nothing changed, there is nothing to do here. Task 9 quotes the final texts in `05-update-tracker.md`.

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
  - every `test result:` line reads `ok`;
  - both clippy runs are clean;
  - `--check` prints nothing.

- [ ] **Step 7: Commit (two commits if the review changed strings)**

  Run `git status --short`: only the files this task touched, and `?? docs/analyses/2026-05-28-press-4bar-analysis.md`
  (never staged).

  **If any round changed strings,** commit the text fixes first. Stage each file by name: the source files, the tests
  and the doc lines from Steps 3 and 5. Write the message with the Write tool to
  `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T8-fix.txt`:

  ```text
  fix(magcoupler): BL-059 G-3 review: <strings changed>

  <one paragraph: each string old -> new in a few words, and the probe that made the
  old text false or misleading, from the record>

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add <each changed source, test and doc file, by name>
  git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T8-fix.txt
  ```

  Then commit the record, with the message file `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T8-record.txt`:

  ```text
  docs(magcoupler): BL-059 G-3 review of the gearbox check's on-screen texts (PASS at round <n>)

  One fresh-context session-model review of the gearbox guide's label, Why and
  Problems, the engine's three verdict texts, label and help, and the two overlay
  rows (the gearbox input torque rating's help, the ripple row's label and help),
  probed at the defaults, the five probe designs and the reachable edges. Recorded
  in docs/analyses/2026-10-14-magcoupler-help-review.md, "## BL-059 gearbox check".

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```

  ```bash
  git add docs/analyses/2026-10-14-magcoupler-help-review.md
  git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T8-record.txt
  ```

  **If no round changed strings,** make only the record commit.

---

### Task 9: Docs consistency pass, the docs/ai records and the backlog (BL-059 notes, BL-066, BL-067)

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

This task writes BL-059's physics basis note, the datasheet question, BL-066 and BL-067.

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash, on branch
`magcoupling/bl059-gearbox-check`. Edits are made with the Edit tool (exact old text -> exact new text). `YYYY-MM-DD`
anywhere below means the date `date +%F` prints on the day the step runs.

Rules for this task:
- Docs only.
- `docs/ai/05-update-tracker.md` history is never rewritten: this task adds one new top entry.
- A stale line that T1, T2, T5 or T6 owns (each task's docs step names its lines) and that is still stale here means
  that task's docs (or comment) step was not applied: apply that task's docs step text from this plan verbatim (no new
  wording here), and name it in the commit message. The lines no task owns get their exact text below.

**Files:**
- Modify: `docs/ai/04-memory.yaml` (the slice-2 active issue, a new BL-059 active issue, NEXT, a new datasheet open
  question, the ordering item's result count)
- Modify: `docs/ai/05-update-tracker.md` (a new top entry)
- Modify: `docs/ai/backlog.yaml` (BL-059 `status` and `notes`; new BL-066 and BL-067 after BL-065)
- Modify: `docs/ai/02-system.yaml` (`current_statuses.magcoupling`; one `known_limitations` entry)
- Modify: `magcoupling-rs/README.md` (the status paragraph's "Next:" sentence (~:46))
- Test: no new test; Step 10 runs `tests/explain.rs`'s docs-count tests (`the_docs_count_the_notes_and_the_steps` and,
  with the `gui` feature, `the_docs_count_the_dashboard_rows_and_the_checks`: they read the README and
  `03-structure.yaml`) and both suites.

**Interfaces:**
- Consumes: T1-T8 committed: `engine::gearbox` (`GearboxResults`, `compute`, `WITHIN_THE_GEARBOX_RATING`,
  `ABOVE_THE_GEARBOX_RATING`, `GEARBOX_CHECK_UNKNOWN`), `DASHBOARD` (18), `CHECKS` (31), `END_EFFECT_ROWS` (8),
  `GEARBOX_GUIDE`, `GEARBOX_UNKNOWN`, `gearbox_problem`, `gearbox_conflict`, `CHECK_GUIDES` (9), `passing_design()` with
  the rating at 1.0, the test `the_gearbox_check_is_a_dashboard_row_with_its_guide`, the review records
  (`### BL-059 re-review` subsections in `docs/analyses/2026-10-14-magcoupler-notes-review.md`;
  `## BL-059 gearbox check` in `docs/analyses/2026-10-14-magcoupler-help-review.md`), and T1, T2 and T5's README and
  docs/ai lines.
- Produces (T10 edits these exact strings after the owner's OK): the tracker heading
  `## 2026-10-16 — Magcoupler gearbox check: cold-high at the gearbox input against its rating (BL-059, branch magcoupling/bl059-gearbox-check)`;
  BL-059 `status: in_progress` and its notes' last sentence
  `Release (spec section 9) is T10: whole-branch review, GATE PASS with gate 12 running, web build and gui-smoke, then the owner's review of the diff; status stays in_progress until the owner's OK and the push.`;
  the 04-memory active issue that starts `  - "Magcoupler gearbox check (BL-059, branch magcoupling/bl059-gearbox-check;`;
  the NEXT parenthetical `then BL-059, the gearbox check`; the 02-system phrase
  `built on branch magcoupling/bl059-gearbox-check`.

- [ ] **Step 1: Preconditions**

Run:
```bash
git branch --show-current
git status --short
git log --oneline 3e63bec..HEAD
```
Expected: `magcoupling/bl059-gearbox-check`; `git status` prints only
`?? docs/analyses/2026-05-28-press-4bar-analysis.md`; the log shows the commits of T2-T8 (T1 lands in T2's commit): at
least seven; T6 and T8 may add one each. Anything else: STOP and finish the owning task first.

- [ ] **Step 2: The stale-text sweep, before the edits (red)**

Run:
```bash
STALE='of 7 checks|all 7 checks|seven (dashboard )?checks|eight guides|17 rows|30 (design )?checks|CheckGuide; 8\]|(^|[^0-9.])1086([^0-9]|$)|until BL-059|Nothing checks (cold-high against|the gearbox against)|says nothing checks|nothing_checks_a_gearbox_torque|no check compares|compares a gearbox torque with the gearbox rating|nothing checks the cold-high torque|hot minimum and the recommended screw|none on the hot minimum and the screw|3 N·m rating|Next: guidance slice 2'
grep -n -E "$STALE" magcoupling-rs/README.md docs/ai/01-meta.yaml docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml | cut -c1-150
echo ---
grep -rn -E "$STALE" magcoupling-rs/src magcoupling-rs/tests --include=*.rs | cut -c1-150
```
What each pattern catches (measured at 3e63bec: 13 lines from the first command, 14 from the second, every one a
line BL-059 changes): the paused status line "of 7 checks" and "passes all 7 checks" (README, and the pinned tests in
`guidance.rs` and `dashboard.rs` that T2 moves); "seven checks" and "seven dashboard checks" (README, `02-system.yaml`, `03-structure.yaml`,
`guidance.rs` doc comments); "eight guides"; "dashboard's 17 rows"; "30 checks" and "CHECKS (30 design checks)"; the old
`CHECK_GUIDES` type; the old result total 1086 (now 1087); "until BL-059"; the two old watch-outs; "says nothing checks"
(README and a `guidance.rs` doc comment); the tripwire's name (`guidance.rs`, the comments in `tests/explain.rs`); the old
rating help ("no check compares"); the README sentence "no dashboard row, guide or check compares a gearbox torque with
the gearbox rating"; the protect-ban comment ("nothing checks the cold-high torque"); the two end-effect badge sentences;
04-memory's "3 N·m rating"; the README's stale "Next: guidance slice 2". Patterns are chosen so the new texts do not
match them (the new torque-band watch-out ends "which nothing checks.", which no pattern catches).

Expected after T1-T8 (the text is what matters; T1 and T2 own and fix README ~:293, ~:362, ~:663 and ~:667 and the
`trace.rs` doc comment): exactly these first-command lines, the ones no other task owns, and nothing after `---`:
```
magcoupling-rs/README.md:46:  ... Next: guidance slice 2 (the note rewrite), then M3 (live 3D fields).
docs/ai/04-memory.yaml:7:  - "Magcoupler guidance slice 2 (BL-056): ... against the 3 N·m rating; ...
docs/ai/04-memory.yaml:18:  - "Magcoupling GUI ordering ... Tracing reaches only the 392 explained results of 1086. ...
---
```
Any other line: it belongs to T1, T2, T5 or T6 (each task's docs step names its lines). Apply that task's docs step
from this plan as the Rules above say, and note the item for the commit message (Step 11). A line that matches but is correct as it stands
after BL-059 (a false positive; none is expected): STOP and report it to the controller with the line, rather than
editing the pattern list.

- [ ] **Step 3: README: the status paragraph's "Next:" sentence (~:46)**

Replace
```text
Next: guidance slice 2 (the note rewrite), then M3 (live 3D fields).
```
with
```text
Guidance slice 2 complete (spec `docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md`, BL-056): the note rewrite with three new notes, display names and help for every input, Show spreadsheet details, and the shaft-clamp section paused. The gearbox check complete (spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`, BL-059, decisions 1A-8A): a counted dashboard check of cold-high at the gearbox input against the gearbox input rating, with its guide (see [The panel](#the-panel)). Next: BL-064 (the shaft-to-hub joint), then the torque sensitivity card (slice 3), then M3 (live 3D fields); BL-065, BL-066 and BL-067 are open in `docs/ai/backlog.yaml`.
```

- [ ] **Step 4: `docs/ai/04-memory.yaml`**

1. Replace the whole line 7 item
```text
  - "Magcoupler guidance slice 2 (BL-056): released 2026-10-16 after the owner's sign-off of docs/analyses/2026-10-14-magcoupler-notes-signoff.md. Open from it: BL-057..BL-065 (BL-059: the gearbox check, cold-high at the output against the 3 N·m rating; BL-064: the shaft-to-hub joint, the shaft-clamp section paused; BL-065: the end factor with magnetic steel behind the magnets)."
```
with these two items
```text
  - "Magcoupler guidance slice 2 (BL-056): released 2026-10-16 after the owner's sign-off of docs/analyses/2026-10-14-magcoupler-notes-signoff.md. Open from it: BL-057, BL-058 and BL-060..BL-065 (BL-064: the shaft-to-hub joint, the shaft-clamp section paused; BL-065: the end factor with magnetic steel behind the magnets); BL-059 is the gearbox check (next item)."
  - "Magcoupler gearbox check (BL-059, branch magcoupling/bl059-gearbox-check; spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md, plan docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md; owner decisions 1A-8A): built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, gui-smoke) and then the owner's review of the diff come before the merge and the push (risk: physics). The dashboard check compares cold-high at the gearbox input with the gearbox input rating (0.6 N·m, which is 2.85 N·m at the output with the 0.95 efficiency: decision 1A); the example design reads 'Example design: fails 4 of 8 checks' while the shaft-clamp section is paused. Open from it: BL-066 (the repeated slip swing against the gearbox's nominal rating), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out) and the datasheet question (open_questions)."
```
2. Replace the NEXT item (line 11). The owner's recorded order after BL-059 is BL-064, then slice 3; BL-065 is not
   placed without the owner (ruling R-5).
```text
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 built 2026-10-14): a torque sensitivity card; then guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Engine gaps split from slice 2: BL-057 (E21), BL-058 (E22), BL-059 (a gearbox cold-high check; the owner confirmed the 0.6 N·m input limit, ruling 8), BL-060 (the clamp boss link, paused with the clamp section), BL-064 (a shaft-to-hub joint check for the chosen method; the clamp section stays paused until it lands)."
```
with these two items (the NEXT item, then the datasheet question)
```text
  - "NEXT (owner order, 2026-10-13, G-1; slice 2 shipped 2026-10-16; then BL-059, the gearbox check): BL-064 (a shaft-to-hub joint check for the chosen method; the clamp section stays paused until it lands), then slice 3, a torque sensitivity card; then guided lessons, plot captions and drawing titles; guides for the results-table checks off the dashboard. Other engine gaps split from slice 2 and BL-059: BL-057 (E21), BL-058 (E22), BL-060 (the clamp boss link, paused with the clamp section), BL-061..BL-063 (verify first), BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it), BL-066 (the repeated slip swing against the gearbox's nominal rating; needs the owner's datasheet value), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out)."
  - "Open (BL-059, owner decision 2A, 2026-10-16): which GAM datasheet line is the 0.6 N·m gearbox input rating (coupling.gearbox_input_rating_Nm, Calculator!C47; its Python-locked engine help reads 'GAM 5:1 peak input rating.')? An input or an output figure (if it is an output rating divided by the 5:1 ratio, the check's input basis, which also divides by the 0.95 efficiency, is conservative by 5 %), and a nominal (T2N), acceleration (T2B, about 1000 load cycles per hour) or emergency-stop (T2NOT, about 1000 per life) rating? The check is built with 0.6 as an input rating, and no on-screen text calls it a peak rating until the owner answers; the answer goes into BL-059's notes, the torque-band note's sources and, for a nominal rating, BL-066. Also open: the old GAM catalog lists -10 or -20 °C as the gearbox's lowest operating temperature, while the coupling's minimum magnet temperature is -40 °C."
```
3. In the ordering item (line 18 before the inserts above, now line 20), replace
```text
Tracing reaches only the 392 explained results of 1086.
```
with
```text
Tracing reaches only the 392 explained results of 1087 (one more since BL-059's gearbox check).
```

- [ ] **Step 5: `docs/ai/05-update-tracker.md`: the new top entry**

Replace
```text
---

## 2026-10-14 — Magcoupler guidance slice 2: notes that say what to change, plain words on screen (BL-056, branch magcoupling/guidance-2)
```
with
```markdown
---

## 2026-10-16 — Magcoupler gearbox check: cold-high at the gearbox input against its rating (BL-059, branch magcoupling/bl059-gearbox-check)
- Why: nothing compared the coupling's strongest expected torque with the gearbox it sits on. The drive line is motor -> GAM 5:1 gearbox -> coupling (on the gearbox output) -> wheel (owner, 2026-10-14): when the wheel is held the coupling passes up to its pull-out before it slips, and while it slips the gearbox carries a fully reversing swing of the same amplitude; the strongest expected pull-out is cold-high (`metal.torque_cold_high_Nm`: the minimum magnet temperature, with the production allowance added). At the defaults cold-high at the gearbox input is 0.8049 N·m against the 0.6 N·m input rating. Spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`; plan `docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md` (T1-T10); research (gitignored) `.superpowers/sdd/2026-10-16-bl059-gearbox-check/`.
- Owner decisions (2026-10-16, all the recommended options): **1A** the basis is the gearbox input: `metal.cold_high_input_Nm` (cold-high divided by the ratio and the efficiency) against `coupling.gearbox_input_rating_Nm`, the same test as cold-high <= rating x ratio x efficiency = 2.85 N·m at the output (the workbook's `model.gearbox_reference_Nm`); it is never unconservative, since the forward-drive half of each slip cycle puts T/(i·η) on the input. BL-059's note of 2026-10-14 that called this the same as 3 N·m at the output is corrected. **2A** built with 0.6 now; which datasheet line it is stays open (BL-059 notes, 04-memory). **3A** counted and red. **4A** the conflict sentence in the Problem. **5A** Auto unchanged: it still sizes for the hot minimum check only. **6A** fixes Lower Axial magnet length and Raise Magnet face gap. **7A** only the texts that become false or misleading change. **8A** the repeated slip swing as a fatigue load is BL-066. Release: the owner's review of the diff before the merge (risk: physics).
- Engine (T1): `engine::gearbox`, Rust-only like `housing`: `GearboxResults { cold_high_check }` declared with `out_rust_only` (no cell) and `compute(cold_high_input_Nm, rating_Nm)`: at or under the rating `WITHIN_THE_GEARBOX_RATING`; strictly above, +∞ included, `ABOVE_THE_GEARBOX_RATING` followed by both torques to 0.01 N·m, the cold-high at least 0.01 N·m above the rating as quoted (the space claim's `quoted_overshoot` rule), so a red verdict never shows two equal numbers; a NaN in either `GEARBOX_CHECK_UNKNOWN`. `DesignResults.gearbox` is computed in `api.rs` beside `housing::compute`; `PACKAGE_LABELS` has 13 entries ("Gearbox"). No Python-ported input, default, range, label, help or result changed, so gate 12's parity is unaffected; the results table has 1087 results (392 explained).
- Dashboard and guide (T2-T4): `DASHBOARD` 18 rows, the check at index 15 after the 15 headline rows (Magnets fit is still drawn last); `CHECKS` 31; `verdict_level` Good, Bad (by the prefix) and Caution (unknown); `END_EFFECT_ROWS` 8: the check greys, uncounted, while the end-effect model is out of range (at c_end 1.5 the negative cold-high would otherwise read as a pass). `GEARBOX_GUIDE` joins `CHECK_GUIDES` (9, in the dashboard's order) and `ROW_LEARN`, with its Why link to the torque-band note (`hot_low_cold_high`). Its Problem quotes cold-high at the gearbox input and the rating; where meeting the requirement would put more than the rating on the gearbox input at the design's band (cold-high over hot-low, which no magnet size, gap or count changes), it adds the conflict sentence (`gearbox_conflict`: the band, what meeting the requirement puts on the gearbox input, and the targets that would settle it; 1.673, 0.8806, 1.703 and 1.140 at the defaults); where the requirement alone is above the rating at the gearbox input (the ratio's or efficiency's slider bottom) it offers no band, and with the operating temperature below the minimum it is left out (plan deviation PD-1, BL-067); a NaN reads `GEARBOX_UNKNOWN` (Caution). Fixes: Lower Axial magnet length and Raise Magnet face gap; both also lower hot-low, so at the defaults the length pulls two ways (the hot-low and gearbox blocks both say so) and the face-gap side-effect line prints in the gearbox block. The example design reads "Example design: fails 4 of 8 checks" while the shaft-clamp section is paused (4 of 9 with it shown). `passing_design()` takes a gearbox input rating of 1.0 N·m (cold-high at the input 0.7070, 29 % under it); with 0.6 back it fails the gearbox check alone. The tripwire `nothing_checks_a_gearbox_torque_against_its_rating_yet` became `the_gearbox_check_is_a_dashboard_row_with_its_guide`; a dedicated test covers a design just under the gearbox limit, where the hot minimum's length fix is held back and its block shows `NO_FIX`; Auto on length at the defaults (13.770 mm) leaves the check red (5A).
- On-screen texts (T5, T6, T8): the gearbox rating's on-screen help names the check (the engine help stays the workbook's); the ripple row gets an on-screen name, "Nominal slip swing at the gearbox input", and a help that says what it is (the pull-out at the operating temperature, without the production allowance, divided by the ratio and efficiency: 0.5660 N·m at the defaults, beside cold-high at the input, 0.8049) and that the gearbox check compares cold-high instead; it claims no inequality, since with the operating temperature below the minimum the swing exceeds cold-high (BL-067); the ban on saying the coupling protects the gearbox now covers the guide and overlay texts as well as the notes. The two watch-outs that said nothing checks the gearbox (`hot_low_cold_high` watch_out[1], `when_slipping` watch_out[0]) now say what the check compares, and that repeated slips also load the gearbox in fatigue, which nothing checks; their sources name the check and the open datasheet question. Both notes were physics re-reviewed (a `### BL-059 re-review` subsection under each note's section of `docs/analyses/2026-10-14-magcoupler-notes-review.md`), and every new on-screen text (the guide's label, Problem and Why, the three verdict texts, the engine label and help, the two overlay rows) passed one G-3 review (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## BL-059 gearbox check`). No new note sign-off sheet (decision 14A: the owner reviews the diff).
- Panel and exports (T7): the gearbox row's own Why link opens the torque-band note and the row opens in Explain; CSV and JSON carry `gearbox.cold_high_check` with an empty cell column; the XLSX Summary lists 18 dashboard lines; the results table's failing filter lists 5 red checks at the defaults (4 before).
- Docs and backlog (T9): the README, docs/ai 02-05 and BL-059 describe the check (BL-059's basis note corrected to 1A; the datasheet question recorded in BL-059 and 04-memory); BL-066 filed (risk: physics): a repeated-slip check of the slip swing against the gearbox's nominal (continuous) rating, which needs the owner's datasheet value as a new input; BL-067 filed (risk: physics, priority 3): an operating temperature below the minimum, where cold-high is not the strongest pull-out. Not covered by the check (spec section 3): the slip swing as a fatigue load (BL-066), backlash impacts at each reversal, the ring-down at re-lock, torsional resonance crossings as the slip frequency sweeps 0-167 Hz, the drive line's stored energy at a jam, and the gearbox's own temperature range (the old GAM catalog lists -10 or -20 °C; the coupling's minimum is -40 °C).

## 2026-10-14 — Magcoupler guidance slice 2: notes that say what to change, plain words on screen (BL-056, branch magcoupling/guidance-2)
```

- [ ] **Step 6: `docs/ai/backlog.yaml`: BL-059**

1. In the BL-059 item, replace `  status: open` (the one between its `priority: 2` and its `notes:`) with
   `  status: in_progress`. Use this exact old text, which is unique:
```text
  acceptance: "owner confirms the limit and where it comes from; then a red test, a dashboard check with its slice-1 guide (Problem, Why, Fix), and the when_slipping note updated; physics review; GATE PASS"
  priority: 2
  status: open
```
new text:
```text
  acceptance: "owner confirms the limit and where it comes from; then a red test, a dashboard check with its slice-1 guide (Problem, Why, Fix), and the when_slipping note updated; physics review; GATE PASS"
  priority: 2
  status: in_progress
```
2. Replace BL-059's whole `notes:` line
```text
  notes: "Split from BL-056 (S2-9). Owner confirmed 2026-10-14: the input limit 0.6 N m is right, equal to 3 N m at the 5:1 output; the coupling is mounted on the gearbox output, so the check compares the coupling's cold-high torque with the output rating (input rating × ratio, 3 N m) and, equivalently, the input-equivalent cold-high with 0.6 N m. Until then no note may say the coupling protects the gearbox. The slice-2 when_slipping note carries a watch-out (the cold-high swing at the gearbox input, unchecked) until this lands."
```
with
```text
  notes: "Split from BL-056 (S2-9). Owner confirmed 2026-10-14: the input limit 0.6 N m is right (3 N m at the 5:1 output, the efficiency left out); the coupling is mounted on the gearbox output. Corrected 2026-10-16 (BL-059 spec section 3, owner decision 1A): the comparison that note called equivalent is not: metal.cold_high_input_Nm (Metal design!C158) divides cold-high by the ratio and the efficiency, so cold-high at the gearbox input <= 0.6 N m means cold-high <= 0.6 × 5 × 0.95 = 2.85 N m at the output (the workbook's model.gearbox_reference_Nm, Calculator!C100), not 3 N m; the two bases disagree for cold-high in (2.85, 3.0] N m (N33AH on both rings, custom dimensions: 2.9401 N m at the output, 0.6190 at the input, fails 1A and would pass 3 N m). Decision 1A: the check compares on the input side, gearbox.cold_high_check = metal.cold_high_input_Nm against coupling.gearbox_input_rating_Nm, which is never unconservative: the forward-drive half of each slip cycle puts T/(i·η) on the input. Open datasheet question (decision 2A: built with 0.6 meanwhile): which GAM datasheet line is the 0.6 N m, an input or an output figure (if it is an output rating divided by the ratio, basis 1A is conservative by the efficiency, 5 %), and a nominal (T2N), acceleration (T2B) or emergency-stop (T2NOT) rating? No on-screen text calls it a peak rating until the owner answers (the Python-locked engine help 'GAM 5:1 peak input rating.' shows only with Show spreadsheet details on); the answer goes here, into the torque-band note's sources and, for a nominal rating, into BL-066. Also open: the old GAM catalog lists -10 or -20 C as the gearbox's lowest operating temperature; the coupling's minimum is -40 C. The ban stays and widens: no note, guide or overlay text says the coupling protects the gearbox (a pass means only that the modelled cold-high peak is within the input rating). Built on magcoupling/bl059-gearbox-check by docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md (T1-T10), spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md, owner decisions 1A-8A: engine::gearbox (Rust-only, no cell); a counted dashboard check, greyed with the end-effect rows; its guide (a Problem with live numbers and, where no magnet size, gap or count passes both it and the hot minimum, the conflict sentence; Why to the torque-band note; fixes Lower Axial magnet length and Raise Magnet face gap); the example design reads 'Example design: fails 4 of 8 checks' while the shaft-clamp section is paused; Auto unchanged (5A); the hot_low_cold_high and when_slipping watch-outs rewritten and physics re-reviewed (docs/analyses/2026-10-14-magcoupler-notes-review.md, the BL-059 re-review subsections); the guide, verdict, engine and overlay texts G-3 reviewed (docs/analyses/2026-10-14-magcoupler-help-review.md, BL-059 gearbox check). Plan deviations (controller rulings 2026-10-16): Problem sentence 2 shows only with the minimum temperature at or below the operating one, and where the requirement alone is above the rating at the gearbox input (the ratio's or efficiency's slider bottom) it offers no band, since no lever narrows the band below one; an operating temperature below the minimum is BL-067. AGENTS.md's fbd-math-reviewer pass is linkage-specific (its agent reviews 4-bar FBDs); BL-059's physics passes are the T6 note re-review, the T8 G-3 review and the T10 whole-branch review on the session model, as slice 2's were. Not covered (spec section 3): the repeated slip swing as a fatigue load (BL-066), backlash impacts at each reversal, the ring-down at re-lock, torsional resonance crossings and the drive line's stored energy at a jam. Release (spec section 9) is T10: whole-branch review, GATE PASS with gate 12 running, web build and gui-smoke, then the owner's review of the diff; status stays in_progress until the owner's OK and the push."
```
(The title, evidence and acceptance stay: like every other item, the title states the problem the item fixed.)

- [ ] **Step 7: `docs/ai/backlog.yaml`: BL-066 and BL-067**

Insert these two items between the last line of BL-065 (its `notes:` line, which ends `The end_effect note's 3D band
sentence names the steel case meanwhile."`) and `- id: BL-055`. Use the Edit tool with old text
```text
  notes: "Filed by the slice-2 whole-branch review fix wave (MF-4, SF-S of the T13 text re-reviews 1 and 2, 2026-10-16). The end_effect note's 3D band sentence names the steel case meanwhile."
- id: BL-055
```
and new text
```text
  notes: "Filed by the slice-2 whole-branch review fix wave (MF-4, SF-S of the T13 text re-reviews 1 and 2, 2026-10-16). The end_effect note's 3D band sentence names the steel case meanwhile."
- id: BL-066
  title: "Nothing checks the repeated slip swing against the gearbox's nominal (continuous) rating: each 0.1 s slip event reverses the gearbox torque about 33 times"
  dimension: physics
  risk: physics
  evidence: "BL-059 spec section 3 and owner decision 8A (docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md, 2026-10-16): while the coupling slips the gearbox output carries a fully reversing swing of up to the pull-out torque at metal.slip_freq_Hz (166.7 Hz at the defaults): 16.7 cycles, about 33 reversals, per 0.1 s slip event (metal.slip_event_s), and metal.magnetic_cycles = slip frequency × event time × life events (magcoupling-rs/src/engine/metal_design.rs:479) = 3.33e8 cycles per life at the defaults (2e7 slip events). BL-059's gearbox.cold_high_check (magcoupling-rs/src/engine/gearbox.rs) compares only the cold-high peak at the gearbox input (metal.cold_high_input_Nm, 0.8049 N m at the defaults) with one input rating (coupling.gearbox_input_rating_Nm, 0.6 N m). A peak rating is a strength limit, not a fatigue limit: gearbox catalogs (GAM, Ondrives) limit an acceleration-type peak (T2B) to about 1000 load cycles per hour and an emergency-stop peak (T2NOT) to about 1000 per life, and GAM's cycle-duty sizing applies a shock factor of up to 2.0 beyond 3000 cycles per hour. The nominal swing at the operating temperature, model.gearbox_input_ripple_Nm (magcoupling-rs/src/engine/model.rs:1156, the pull-out over ratio × efficiency), is 0.5660 N m at the gearbox input at the defaults, below 0.6 N m, and 0.6509 N m with the 15 % production allowance, above it"
  acceptance: "owner gives the gearbox datasheet line (model and size; its nominal, continuous T2N-type torque and the side it is rated at; any reversing-load or cycle-duty factor) and the slip duty to check (slip events per hour and per life); then red test first: a new Rust-only input for the nominal rating (input_schema.json re-blessed; design files and share links without it load at its default; the 'gearbox' filter tests that find exactly three gearbox inputs, in gui/inputs.rs, gui/panel.rs and gui/format.rs, move to four) and a counted dashboard check of the slip swing against it (which swing, the nominal at the operating temperature or cold, with or without the allowance, at the input or the output, decided with the owner and stated in the check's help), with its slice-1 guide (Problem, Why, Fix) and a reviewed Why note; the torque-band and slipping notes' fatigue watch-outs updated in the same commit; G-3 and physics reviews; the owner's review of the diff; GATE PASS with gate 12 running"
  priority: 2
  status: open
  notes: "Split from BL-059 (owner decision 8A, 2026-10-16; research .superpowers/sdd/2026-10-16-bl059-gearbox-check/research-physics.md, gitignored). Needs the owner's datasheet value as a new input, so BL-059's open datasheet question (which line the 0.6 N m is: input or output; T2N, T2B or T2NOT) comes first. If 0.6 N m turns out to be the nominal rating itself, this item needs no new input, and the owner decides which rating each check compares. Until it lands, the torque-band note's watch-out says repeated slips also load the gearbox in fatigue, which nothing checks."
- id: BL-067
  title: "An operating magnet temperature below the minimum magnet temperature is accepted: cold-high is then not the strongest pull-out, so the gearbox check can pass while the operating pull-out at the gearbox input is above the rating"
  dimension: physics
  risk: physics
  evidence: "BL-059 plan review (2026-10-16), probes on compute_all: coupling.op_temp_C slides to -40 C and metal.min_temp_C to 20 C (magcoupling-rs/src/engine/model.rs:174-176, metal_design.rs:28-30). At op -40 C, min 20 C, variation 0: hot-low 3.3246 N m above cold-high 2.8930 N m at the output; cold-high at the gearbox input 0.6091 N m below the nominal slip swing at the input (model.gearbox_input_ripple_Nm) 0.6999 N m (variation 0.1: 0.6700 against 0.6999). gearbox.cold_high_check compares cold-high only (magcoupling-rs/src/engine/gearbox.rs), so a rating between the two passes while the operating pull-out exceeds it. BL-059's static texts assume the minimum temperature at or below the operating one (the guide's Why and the two watch-outs); its Problem sentence 2 is left out there (gearbox_conflict)."
  acceptance: "owner chooses: (a) a caution when the operating temperature is below the minimum (which check carries it, and its text), or (b) the gearbox check compares the larger of cold-high and the operating pull-out with the allowance, at the gearbox input; then a red test at op -40 C, min 20 C, variation 0.1; G-3 review of any new text; GATE PASS"
  priority: 3
  status: open
  notes: "Split from BL-059's plan review (2026-10-16). BL-059 keeps the engine as specified (spec section 4, decision 7A) and leaves Problem sentence 2 out in this corner, since the band's levers and its floor of one do not hold there."
- id: BL-055
```

- [ ] **Step 8: `docs/ai/02-system.yaml`: the magcoupling status and a known limitation**

1. At the end of `current_statuses.magcoupling` (line ~485), replace
```text
; next: the torque sensitivity card, then M3 (live 3D fields)"
```
with
```text
; guidance slice 2 merged and deployed (3cc0952); the gearbox check (BL-059, decisions 1A-8A) built on branch magcoupling/bl059-gearbox-check: engine::gearbox (Rust-only) and a counted dashboard check of cold-high at the gearbox input against the gearbox input rating, with its guide (gui/guidance.rs GEARBOX_GUIDE); next: BL-064, then the torque sensitivity card (slice 3), then M3 (live 3D fields)"
```
2. Add one `known_limitations` item after the last one (the magcoupling design-file item). Replace
```text
  - "A magcoupling design file dropped on the linkage page is read as a mechanism (the status bar says the JSON import failed); the calculator's own page (colesorkness.com/tools/magcoupler) loads it with Load design."
```
with
```text
  - "A magcoupling design file dropped on the linkage page is read as a mechanism (the status bar says the JSON import failed); the calculator's own page (colesorkness.com/tools/magcoupler) loads it with Load design."
  - "The magcoupler's gearbox check (BL-059, gearbox.cold_high_check) is a peak check: cold-high at the gearbox input against one input rating. It does not check the repeated slip swing as a fatigue load (BL-066), backlash impacts, the ring-down at re-lock, torsional resonance crossings or the drive line's stored energy at a jam, each of which can exceed cold-high at the shaft; which datasheet rating the 0.6 N·m default is (input or output; nominal, acceleration or emergency stop) is open (owner decision 2A, 04-memory)."
```
(`02-system.yaml` does not parse as YAML at 3e63bec, near line 14, before this branch: leave that alone and leave it to
T10's owner summary, which reports it; Step 9 checks only the files this task can break.)

- [ ] **Step 9: Check the YAML and the item shapes**

Run:
```bash
python - <<'EOF'
import yaml
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
ids = [x['id'] for x in b]
assert ids[:12] == ['BL-056', 'BL-057', 'BL-058', 'BL-059', 'BL-060', 'BL-061', 'BL-062', 'BL-063', 'BL-064', 'BL-065', 'BL-066', 'BL-067'], ids[:13]
keys = ('id', 'title', 'dimension', 'risk', 'evidence', 'acceptance', 'priority', 'status', 'notes')
assert all(tuple(x) == keys for x in b), [x['id'] for x in b if tuple(x) != keys]
item = {x['id']: x for x in b}
assert item['BL-059']['status'] == 'in_progress'
assert 'equivalently' not in item['BL-059']['notes']
assert '2.85' in item['BL-059']['notes'] and 'T2NOT' in item['BL-059']['notes']
assert (item['BL-066']['risk'], item['BL-066']['status'], item['BL-066']['priority']) == ('physics', 'open', 2)
assert (item['BL-067']['risk'], item['BL-067']['status'], item['BL-067']['priority']) == ('physics', 'open', 3)
m = yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8'))
assert any(i.startswith('Magcoupler gearbox check (BL-059') for i in m['active_issues'])
assert any(q.startswith('Open (BL-059, owner decision 2A') for q in m['open_questions'])
print(len(b), len(m['active_issues']), len(m['open_questions']))
EOF
```
Expected: `67 4 65` (65 backlog items at 3e63bec plus BL-066 and BL-067; 3 active issues plus BL-059's; 64 open
questions plus the datasheet question). An `AssertionError` names the broken item: fix the edit, run again.

- [ ] **Step 10: The sweep again (green), the new facts, the docs tests and the suites**

Run Step 2's two commands again. Expected: both print nothing but the `---` line.

Then the new facts (each written by the task named):
```bash
grep -c '1087' magcoupling-rs/README.md
grep -c 'src/engine/gearbox.rs' magcoupling-rs/README.md
grep -o 'addendum_only: \[[^]]*\]' docs/ai/03-structure.yaml
grep -c 'of 9 checks' magcoupling-rs/README.md
grep -o 'CHECKS ([0-9]* design checks)' docs/ai/03-structure.yaml
grep -c 'BL-066' docs/ai/backlog.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/02-system.yaml
```
Expected: `2` (T1's two); at least `1` (T1's module row); a list that contains `gearbox.rs` (T1); at least `1` (T2's
unpaused status count); `CHECKS (31 design checks)` (T2); every count at least `1` (this task). A `0` or an old count is
the owning task's docs step missing: apply it from that task's step as the Rules say, then run this step again.

Then the test that reads the README and `03-structure.yaml`, the suites, the lints and the format check:
```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count_the_notes_and_the_steps
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count
cargo test --manifest-path magcoupling-rs/Cargo.toml 2>&1 | grep -E '^test result|FAILED|panicked'
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E '^test result|FAILED|panicked'
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
```
Expected: the docs tests pass (`test result: ok. 1 passed` without features; `test result: ok. 2 passed` with
`--features app`, which adds `the_docs_count_the_dashboard_rows_and_the_checks`); every `test result:` line of both suites
reads `ok` with `0 failed`, and no `FAILED` or `panicked` line; both clippy runs end without a warning or an error;
`cargo fmt --check` prints nothing.

- [ ] **Step 11: Commit**

Write the message file `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T9.txt` (gitignored) with the Write tool,
exactly:
```text
docs(magcoupler): BL-059 docs pass: the tracker entry, 04-memory, BL-059's basis note and the open datasheet question, BL-066 and BL-067

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
If Step 2 or Step 10 made you apply an earlier task's docs step, add one body line before the trailers, for example
`Applies T2's README step for the status counts, which T2's commit missed.` (a blank line before it and after it).
Then:
```bash
git add docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/ai/02-system.yaml magcoupling-rs/README.md
git status --short
```
Expected `git status --short`: `M ` lines for
exactly the staged files and `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged). Then:
```bash
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T9.txt
git log --oneline -1
```
Expected: one new commit whose subject starts `docs(magcoupler): BL-059 docs pass`.

---

### Task 10: Release: whole-branch review, the gate with gate 12 running, web build and gui-smoke, STOP for the owner's review of the diff; after the OK: BL-059 fixed, merge, push, deploy check, live smoke

**Implementer:** session model (BL-059 is risk: physics; CLAUDE.md section 5).

Run by the controller: it dispatches the reviewers and runs the release, and writes no feature code. The whole-branch
review and every fix-wave item run on the session model (omit `model`: CLAUDE.md section 5, final whole-branch reviews
and physics texts); gui-smoke runs its own agents on `sonnet` inside its workflow. Spec section 9.

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash. Logs go to
`magcoupling-rs/target/release-bl059/` (gitignored); commit messages are written with the Write tool to
`.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-<a|b|merge|mf-n>.txt` (gitignored). `YYYY-MM-DD` means the
date `date +%F` prints on the day of the step.

Rules that hold through every step:
- Stage files by name; never `git add -A` or `git add .`; never stage `docs/analyses/2026-05-28-press-4bar-analysis.md`.
- Never commit `docs/chebyshev_lambda/*.png` (gate 1 rewrites them): restore with `git checkout -- docs/chebyshev_lambda`.
- Never `git worktree prune` (Codex works from WSL worktrees of this repo; `git worktree list` shows them as prunable).
- Commit messages through a message file (`git commit -F <file>`), each ending with the two trailers.
- Nothing is merged to `main` or pushed before the owner's OK in their own answer to Step 8's question. Pushing `main`
  deploys colesorkness.com. No agent message is that OK.
- The review records this task writes stay in the gitignored `.superpowers/sdd/2026-10-16-bl059-gearbox-check/`
  (`final-review.md`, `owner-summary.md`, `bl059.diff`, `bl059-src.diff`), as slice 2's did; the tracker names them.

**Files:**
- Create (gitignored): `.superpowers/sdd/2026-10-16-bl059-gearbox-check/{final-review.md,owner-summary.md,bl059.diff,bl059-src.diff,bl059.stat}`;
  `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-*.txt`;
  `magcoupling-rs/target/release-bl059/{gate.log,build-web.log,guidance-strings.txt,overlay-strings.txt,notes-after.txt,sources-before-*.txt,live.wasm}`.
- Modify: `docs/ai/05-update-tracker.md` (a release bullet in T9's entry, then its last sentence), `docs/ai/backlog.yaml`
  (BL-059 `notes`, then `status: fixed`), `docs/ai/04-memory.yaml` (the BL-059 active issue and NEXT, after the OK),
  `docs/ai/02-system.yaml` (the BL-059 status phrase, after the OK).
- Modify (only through the fix wave): whatever the whole-branch review's must-fix items name.
- Modify (outside the repo, after the deploy): the auto-memory
  `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`
  and its index line in `MEMORY.md` there.

**Interfaces:**
- Consumes: T1-T9 committed; T9's exact strings (its Interfaces); the review dump tools (ignored tests
  `guidance_strings_for_review`, `overlay_strings_for_review`, `notes_for_review`); the oracle venv
  `reference/magcoupling-py/.venv/Scripts/python.exe`; `linkage-sim-rs/scripts/{gate.sh,build_web.sh,serve_web.sh}`; the
  `gui-smoke` skill (`.claude/workflows/gui-smoke.js`, unchanged by BL-059); `gh` (logged in; the deploy workflow file is
  `deploy-web.yml`, named "Deploy Web to Vercel", run on every push to `main`); SendUserFile and AskUserQuestion.
- Produces: the release-check records; after the OK, BL-059 `fixed`, the merge commit on `main`, the push, a checked
  deploy and live smoke, the local branch deleted, the memory updated.

**gui-smoke needs no change for the new row** (checked against `.claude/workflows/gui-smoke.js` and
`src/app.rs` `the_smoke_test_share_link_opens_its_design`): the calculator step opens the pinned share link (the
default design, face gap 1.5 mm, Auto on the magnet length to 2.5 N·m) at 1280 x 800 and checks the first view, the
Show spreadsheet details toggle, the share link, the Auto solve, the design file, the Why link under the dashboard's
first row (the nominal pull-out), an input tooltip without spreadsheet details, and that after a 4000 px wheel the
dashboard's last row, "Magnets fit", and its "Why:" line are in view and its first row is not. BL-059 adds one row
inside `DASHBOARD` (index 15) and keeps Magnets fit drawn after `DASHBOARD`; the smoke design now fails the gearbox check
too, which adds one row and one open red block (about 215 to 260 pt, research-surfaces section 7), well inside the
4000 px wheel; no smoke field counts rows, red blocks or reads the status line; no input was added, so the share link
and design file are still current (`app.rs` pins both). What the smoke cannot show is that a new bundle is the one
served, since no smoke field changes with BL-059 (slice 2 used the new toggle for that). Steps 5 and 10 prove it
instead: the shipped `.wasm` must contain the new verdict text and no longer contain the old watch-out (both checked
on the live file at 3e63bec's deploy: `Show spreadsheet details` 1 match, `Nothing checks cold-high against the
gearbox` 1, `Within the gearbox input rating` 0).

- [ ] **Step 1: Preconditions**

Run:
```bash
git branch --show-current
git status --short
git log --oneline 3e63bec..HEAD | wc -l
git log --oneline -1
grep -rn -A2 '#\[ignore' magcoupling-rs/src magcoupling-rs/tests | grep -o 'fn [a-z_0-9]*' | sort -u
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
mkdir -p magcoupling-rs/target/release-bl059
```
Expected: `magcoupling/bl059-gearbox-check`; only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`; at least `8`
commits since 3e63bec (T2-T9; T1 is in T2's commit; T6 and T8 may add one each; the plan's own commit too, if the
controller committed it on the branch); the last commit is T9's (`docs(magcoupler): BL-059 docs pass ...`); the ignored functions are exactly
`fn guidance_strings_for_review`, `fn help_for_review`, `fn notes_before_snapshot`, `fn notes_for_review`,
`fn notes_signoff_sheet`, `fn overlay_strings_for_review`, `fn review_sheet` (as at 3e63bec); `cargo fmt --check` prints
nothing. Anything else: STOP and finish the owning task first.

- [ ] **Step 2: Whole-branch review (fresh context, session model)**

Dispatch one reviewer on the session model (omit `model`) with this prompt, verbatim:

> You are the whole-branch reviewer for BL-059, the magnetic coupling calculator's gearbox check (branch
> `magcoupling/bl059-gearbox-check` of C:/Users/Cole/source/repos/linkage_simulation, crate `magcoupling-rs`). Read the
> spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md` (binding: the owner decisions 1A-8A and R of
> section 2, the technical decisions 9-14, sections 3-8), the plan `docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md`,
> then the whole diff `git diff main...magcoupling/bl059-gearbox-check`. Read-only: change no file and run no command
> that changes the repository (you may run `cargo test` with `CARGO_TARGET_DIR` set to a folder outside the repository).
> Check, each with file:line evidence:
> (1) every owner decision 1A-8A and technical decision 9-14 is implemented as the spec says, or the plan records a
> deviation the controller accepted;
> (2) the engine: `engine/gearbox.rs` `compute` passes at equality, fails strictly above and at +∞, reads unknown on a
> NaN in either value, quotes both torques to 0.01 N·m with the cold-high at least 0.01 N·m above the quoted rating;
> the group is Rust-only with no cell; `git diff main...magcoupling/bl059-gearbox-check --stat -- reference
> magcoupling-rs/tests/data magcoupling-rs/tests/python_schema.rs` prints nothing (no Python-locked row, data file or
> input schema changed); Auto is unchanged: the same diff over `magcoupling-rs/src/engine/sizing.rs
> magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/gui/size_controls.rs` prints nothing;
> (3) every on-screen text the branch adds or changes (the guide's label, Problem sentences 1 and 2 and the unknown
> text, the Why, the three verdict texts, the engine label and help, the rating help and the ripple row's label and
> help, the two rewritten watch-outs and their sources) against the traps of spec section 6: every torque names its
> side of the gearbox; efficiency raises the input torque, never "reduces the load"; cold-high is an expectation ("up
> to", "can reach"), never "at most" or what every unit reaches; never that the coupling protects the gearbox; "peak"
> names the compared torque, never the rating (its type is open, 2A); the ripple row is the nominal swing at the
> operating temperature, with no "warm"; nothing says geometry narrows the band; no "hot-day"; no internal path, id,
> cell, "BL-059", "Rust-only" or "clamp"; glyph-safe; and each number true at the defaults, the five probe designs and
> any reachable input. Check Problem sentence 2's condition (hot-low > 0 and required x (cold-high / hot-low) / (ratio
> x efficiency) > rating, every term read from the results and inputs) and that its claim, no magnet size, gap or count
> passes both checks, holds wherever it shows;
> (4) the two notes' texts match their `### BL-059 re-review` records in
> `docs/analyses/2026-10-14-magcoupler-notes-review.md`, and every new on-screen text as committed matches the
> `## BL-059 gearbox check` record in `docs/analyses/2026-10-14-magcoupler-help-review.md`: list any text edited after
> its review;
> (5) every test the spec's section 7 lists exists and asserts values computed from the engine or named constants, not
> copied literals (pinned on-screen strings excepted); the tripwire is replaced by
> `the_gearbox_check_is_a_dashboard_row_with_its_guide`; no test is newly ignored;
> (6) the gearbox row greys and is uncounted while the end-effect model is out of range; `passing_design()` passes all
> 8 counted checks while the clamp section is paused; the "near the gearbox limit" test shows the held-back length fix;
> (7) the docs (`magcoupling-rs/README.md`, `docs/ai/02-system.yaml` to `05-update-tracker.md`, `docs/ai/backlog.yaml`
> BL-059, BL-066 and BL-067) describe the code as it is, with no stale count (rows, checks, guides, results);
> (8) the plan's accepted deviations: PD-1 (Problem sentence 2 also requires the minimum temperature at or below the
> operating one, and offers no band where the requirement alone is above the rating at the gearbox input); the
> operating-below-minimum corner is BL-067 (the engine is unchanged by decision 7A): judge texts there against that
> record.
> Return PASS, or a numbered list of must-fix items (file:line, the defect, the exact replacement text or the test to
> add) and a separate list of suggestions. A must-fix that changes a note's text names the note id; one that changes
> another on-screen text says so.

Save the reviewer's full answer to `.superpowers/sdd/2026-10-16-bl059-gearbox-check/final-review.md`.

- [ ] **Step 3: One fix wave, then the re-review (only if Step 2 returned must-fix items)**

For each must-fix item, in order, one commit each, on the session model (an implementer that does not spawn agents):
1. Write the failing test the item names (or one that shows the defect), run it, see it fail.
2. Apply the fix.
3. A fix that changes a note's text (`hot_low_cold_high` or `when_slipping`; sentences, change entries, watch-outs or
   sources): dump the note with
   `NOTES_FOR_REVIEW=<id> NOTES_REVIEW_OUT="$PWD/magcoupling-rs/target/release-bl059/notes-<id>.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture`,
   dispatch a fresh session-model physics reviewer on that dump (the engine code behind each claim, spec section 6's
   traps), apply its must-fix items, move the note's review date to YYYY-MM-DD, and append its report to that note's
   existing section of `docs/analyses/2026-10-14-magcoupler-notes-review.md` as
   `### BL-059 re-review after the whole-branch review (YYYY-MM-DD)`: never a second `## <id>` heading
   (`every_note_names_its_committed_review_section` allows exactly one).
4. A fix that changes another on-screen text: one fresh session-model G-3 reviewer on the changed strings (dump them
   with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture`
   or `... overlay_strings_for_review ...`), its report appended under `## BL-059 gearbox check` of
   `docs/analyses/2026-10-14-magcoupler-help-review.md` as `### Re-review after the whole-branch review (YYYY-MM-DD)`.
5. Run the touched suites (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app <filter>` and the same
   without `--features app`), both clippy runs
   (`cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and the same
   without `--features app`) and `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, update the docs the fix
   changes, and write `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-mf-<n>.txt`:
```text
fix(magcoupler): <what, in a few words>, from the BL-059 whole-branch review

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   stage the changed files by name and `git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-mf-<n>.txt`.

Then dispatch a fresh session-model reviewer on `git diff <the commit before the fix wave>..HEAD` with the must-fix
list, asking for PASS or the items still open. Append its answer to `final-review.md`. A second round of must-fix
items: apply them the same way and re-review once more; a third: STOP and report to the owner.

- [ ] **Step 4: The gate, with gate 12 running**

`gate.sh` finds the oracle's Python in this order: `$MAGCOUPLING_PYTHON` (used as given, so a wrong value fails the
gate), then `reference/magcoupling-py/.venv/Scripts/python.exe` (or `.venv/bin/python`) in this checkout or another
worktree. It prints `oracle Python: <path>` when gate 12 runs, and `SKIP gate 12/12: no Python for
reference/magcoupling-py ...` (without failing) when none is found. Naming the venv explicitly makes a SKIP impossible.

Run in the background (about 6 minutes warm, longer cold):
```bash
MAGCOUPLING_PYTHON="$PWD/reference/magcoupling-py/.venv/Scripts/python.exe" bash linkage-sim-rs/scripts/gate.sh 2>&1 | tee magcoupling-rs/target/release-bl059/gate.log
```
When it exits, run:
```bash
L=magcoupling-rs/target/release-bl059/gate.log
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
B=magcoupling-rs/target/release-bl059/build-web.log
bash linkage-sim-rs/scripts/build_web.sh 2>&1 | tee $B
grep -n 'parity guard:\|Magcoupling build complete!' $B
WASM=linkage-sim-rs/web/tools/magcoupler/magcoupling-web_bg.wasm
VERDICT="$(sed -n '/pub const WITHIN_THE_GEARBOX_RATING:/,/;/p' magcoupling-rs/src/engine/gearbox.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
echo "[$VERDICT]"
[ -n "$VERDICT" ] && grep -c -a -F "$VERDICT" $WASM
grep -c -a -F 'Nothing checks cold-high against the gearbox' $WASM
git status --short
```
Expected: `parity guard: magcoupling-web builds without workbook-parity; magcoupling-rs features: ...` and
`Magcoupling build complete!`; the verdict text in brackets (`[Within the gearbox input rating]` unless the G-3 review
reworded it; `[]` means the constant was not found: STOP and read `gearbox.rs`, since an empty text would match
everything); at least `1`; `0`; `git status` unchanged (the bundles are gitignored). If the release link fails with `LNK1104`
(Windows MAX_PATH, 02-system lessons), build from a short path: `subst X: "$(cygpath -w "$PWD")"`, then
`cd /x && bash linkage-sim-rs/scripts/build_web.sh`, then `subst X: /d`.

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
(stop its background task). `git status --short` must again print only the owner's untracked file (the smoke writes and
deletes `.playwright-mcp/magcoupling-smoke-design.json`, gitignored).

- [ ] **Step 7: Record the release checks and commit**

1. `docs/ai/05-update-tracker.md`: append this bullet as the last line of T9's BL-059 entry (right after its
   `- Docs and backlog (T9): ...` bullet, before the blank line and `## 2026-10-14 — Magcoupler guidance slice 2`). Write
   `passed with no must-fix` if Step 2 returned PASS, else `passed after its fix wave of <N> items (re-reviewed)` with
   the count of Step 3's commits:
```text
- T10 release checks (YYYY-MM-DD): the whole-branch review passed with no must-fix (record: .superpowers/sdd/2026-10-16-bl059-gearbox-check/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the check (its verdict text is in magcoupling-web_bg.wasm, the old torque-band watch-out is not); a gui-smoke run passed (hub, linkage; magcoupler: first view, the Show spreadsheet details toggle, share link, Auto solve, design file, a Why link opening a note, an input tooltip without spreadsheet details, the dashboard scrolling to Magnets fit at 1280 x 800, a clean console). Not yet done: the owner's review of the diff, the merge, the push and the deploy check.
```
2. `docs/ai/backlog.yaml`, BL-059: append to its `notes:` string, right before the closing `"`:
   ` Release checks done YYYY-MM-DD (T10): whole-branch review PASS, GATE PASS with gate 12 running, web build and gui-smoke PASS; waiting for the owner's review of the diff.`
3. Write `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-a.txt`:
```text
docs(magcoupler): record the BL-059 release checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
then:
```bash
git add docs/ai/05-update-tracker.md docs/ai/backlog.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-a.txt
```
Expected: the two files staged (`M `) and the owner's file untracked; one new commit.

- [ ] **Step 8: STOP: the owner's summary and the diff, then ask for the OK**

1. The texts as committed, from the branch:
```bash
D=magcoupling-rs/target/release-bl059
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > $D/guidance-strings.txt
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > $D/overlay-strings.txt
NOTES_FOR_REVIEW=hot_low_cold_high,when_slipping NOTES_REVIEW_OUT="$PWD/$D/notes-after.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture
git diff main...HEAD > .superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059.diff
git diff main...HEAD -- . ':(exclude)magcoupling-rs/tests' ':(exclude)*_test*' > .superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059-src.diff
git diff --stat main...HEAD > .superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059.stat
grep -c '^test result: ok. 1 passed' $D/guidance-strings.txt $D/overlay-strings.txt
tail -n 1 .superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059.stat
```
Expected: the notes run prints `2 notes at 5 designs: .../notes-after.txt` and `test result: ok. 1 passed`; the
`grep -c` prints `1` for each dump file (the two dumps' test output went into the files); the stat's last line counts
the files changed. `bl059-src.diff` is the production and docs part of the diff: it leaves out `magcoupling-rs/tests`
and test files, while the in-file `mod tests` blocks stay in the full diff only.

2. Write the summary with this script (run it whole in one Bash call; it reads every "after" text from the dumps and
   the code, and the "before" texts are the ones at 3e63bec, which equal `main`'s: `git diff main 3e63bec --stat --
   magcoupling-rs` is empty; they were read from the same dumps run there):
```bash
D=magcoupling-rs/target/release-bl059
OUT=.superpowers/sdd/2026-10-16-bl059-gearbox-check/owner-summary.md
join() { awk -v ORS='' 'NR>1{print "<br>"} {print} END{if (NR==0) print "(none)"}'; }
g() { grep -F "[$1] " "$D/guidance-strings.txt" | sed 's/^\[[^]]*\] //' | join; }
o() { grep -F -x -A1 "[$1] $2" "$D/overlay-strings.txt" | sed -n 's/^  now: //p' | join; }
w() { awk '/^==== /{note=$2; on=0} /^-- at the /{d=($0=="-- at the default design"); on=0} /^Watch out:/{on=d; next} on && /^  - /{print note "\t" substr($0,5)}' "$D/notes-after.txt" | awk -F'\t' -v id="$1" -v n="$2" '$1==id {k++; if (k==n) print $2}' | join; }
src() { awk '/^==== /{note=$2; s=0} /^Where this comes from:/{s=1; next} /^-- at the /{s=0} s && /^  - /{print note "\t" substr($0,5)}' "$D/notes-after.txt" | awk -F'\t' -v id="$1" '$1==id {print $2}' | grep -v -x -F -f "$D/sources-before-$1.txt" | join; }
gconst() { sed -n "/pub const $1:/,/;/p" magcoupling-rs/src/engine/gearbox.rs | grep -o '"[^"]*"' | head -1 | tr -d '"'; }
engine() { sed -n '/cold_high_check: String => out_rust_only/,/),$/p' magcoupling-rs/src/engine/gearbox.rs | grep -o '"[^"]*"' | sed -n "$1p" | tr -d '"'; }
printf '%s\n' "Model assumption: the production variation allowance (symmetric; an engineering allowance, not measured)" "Math audit of 2026-09-29, confirmed: the hot and cold torque rows and the hot-minimum check" "Design rulings of 2026-10-14 and 2026-10-15: the coupling sits on the gearbox output, and whatever holds the hub on the shaft carries the cold-high torque" "Guidance audit of 2026-10-13: which temperature each torque uses; the gearbox input" > "$D/sources-before-hot_low_cold_high.txt"
printf '%s\n' "The torque-angle model: pull-out as the peak of the summed harmonics, one cycle per pole pair" "The slip-loss and thermal model: the drag, the heating and the slip fault trip time" "The demagnetization model: the like-poles-facing reverse field while slipping sets the magnet limit" "The gearbox rows: the ripple at the gearbox input is the pull-out through the gear ratio and efficiency" "Guidance audit of 2026-10-13: what happens past pull-out; the rings re-lock only when the slip speed falls" > "$D/sources-before-when_slipping.txt"
LABEL="$(sed -n '/^const GEARBOX_GUIDE/,/^};/p' magcoupling-rs/src/gui/guidance.rs | grep -o 'label: "[^"]*"' | head -1 | cut -d'"' -f2)"
ROW="$(o label gearbox.cold_high_check)"; [ "$ROW" = "(none)" ] && ROW="$(engine 2)"
ROWHELP="$(o help gearbox.cold_high_check)"; [ "$ROWHELP" = "(none)" ] && ROWHELP="$(engine 3)"
WITHIN="$(gconst WITHIN_THE_GEARBOX_RATING)"; ABOVE="$(gconst ABOVE_THE_GEARBOX_RATING)"; UNKNOWN="$(gconst GEARBOX_CHECK_UNKNOWN)"
STAT="$(tail -n 1 .superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059.stat)"
cat > "$OUT" <<EOF
# BL-059, the gearbox check: what changes on screen

Branch \`magcoupling/bl059-gearbox-check\` against \`main\` (3cc0952): $STAT. The full diff is attached
(bl059.diff). Every text below is at the example design (the defaults), with the shaft-clamp section paused, as the
site shows it.

| # | Where | Before (main) | After (this branch) |
|---|---|---|---|
| 1 | Dashboard status line | Example design: fails 3 of 7 checks | Example design: fails 4 of 8 checks |
| 2 | Dashboard rows | 17 | 18: the new row sits just above Space claim; Magnets fit stays last |
| 3 | New row: name and value | (no row) | $ROW: $ABOVE: 0.80 N·m against 0.60 N·m (red) |
| 4 | New row: hover help | (no row) | $ROWHELP |
| 5 | New row: verdicts on other designs | (no row) | at or under the rating: $WITHIN (green)<br>a value not a number: $UNKNOWN (amber) |
| 6 | Its guide: Problem | (none) | $(g "problem/$LABEL/defaults") |
| 7 | Its guide: Why, and its link | (none) | $(g "why/$LABEL")<br>Why: Hot-low and cold-high: the torque band |
| 8 | Its guide: Fix | (none) | $(g "fix/$LABEL/defaults") |
| 9 | Its guide: notes under the fixes | (none) | $(g "note/$LABEL/defaults") |
| 10 | Hot-low torque block: notes under its fixes | (none) | $(g "note/Hot-low torque/defaults") |
| 11 | Gearbox input torque rating: hover help | Peak torque the gearbox input is rated for. Only an informational reference uses it: no check compares the coupling's torques with it. | $(o help coupling.gearbox_input_rating_Nm) |
| 12 | Ripple row (dashboard): name | Estimated gearbox input torque ripple amplitude | $(o label model.gearbox_input_ripple_Nm) (value unchanged, 0.5660 N·m) |
| 13 | Ripple row: hover help | (no help) | $(o help model.gearbox_input_ripple_Nm) |
| 14 | Note "Hot-low and cold-high: the torque band", Watch out 2 | Nothing checks cold-high against the gearbox: at its input it is 0.8049 N·m here; compare that with the gearbox's input rating yourself. | $(w hot_low_cold_high 2) |
| 15 | Note "When the coupling slips", Watch out 1 | Nothing checks the gearbox against these swings: when cold they can reach the cold-high torque, on this design 0.8049 N·m at its input; check its rating. | $(w when_slipping 1) |
| 16 | The two notes' "Where this comes from": new or changed lines | (as before) | torque band: $(src hot_low_cold_high)<br>slipping: $(src when_slipping) |
| 17 | Results table, Failing checks only | 4 red | 5 red (the gearbox check added) |
| 18 | Exports | CSV and JSON: no gearbox row; spreadsheet Summary: 17 dashboard lines | CSV and JSON: a gearbox.cold_high_check row (no workbook cell); Summary: 18 dashboard lines |

**The example design now reads "fails 4 of 8 checks", and stays red on the gearbox.** Cold-high is 1.673 times hot-low
at the default temperatures and variation, whatever the magnet size, gap or count, so any design that meets the
2.500 N·m requirement puts 0.8806 N·m on the gearbox input. The Problem says so and names what would settle it (row
6). Auto is unchanged (decision 5A).

**Open question for you (decision 2A).** Which GAM datasheet line is the 0.6 N·m? An input or an output figure (if it
is an output rating divided by the 5:1 ratio, this input-side check is 5 % conservative, the efficiency), and a
nominal (T2N), acceleration (T2B) or emergency-stop (T2NOT) rating? The check is built with 0.6 as an input rating; no
on-screen text calls it a peak rating until you answer. A nominal rating also feeds BL-066 (the repeated slip swing as
a fatigue load, filed by this branch). The old GAM catalog also lists -10 or -20 °C as the gearbox's lowest operating
temperature; the coupling's minimum magnet temperature is -40 °C.

**Not covered by the check:** the repeated slip swing as a fatigue load (BL-066), backlash impacts at each reversal,
the ring-down at re-lock, torsional resonance crossings and the drive line's stored energy at a jam.

**A corner the plan review found:** with the operating temperature below the minimum (both on their sliders), cold-high
is not the strongest pull-out, so the check can pass falsely; the Problem's conflict sentence is left out there, and
BL-067 holds the fix for your choice (a caution, or comparing the larger torque).

**Also for you, not BL-059's:** docs/ai/02-system.yaml already fails a strict YAML parse (near line 14), at 3e63bec,
before this branch. It is left as it is here.

**Checks done:** the physics re-review of the two notes and the G-3 review of every new text (records in
docs/analyses/2026-10-14-magcoupler-notes-review.md and docs/analyses/2026-10-14-magcoupler-help-review.md), the
whole-branch review, gate.sh GATE PASS with the Python oracle running (gate 12), the web build and the web smoke test.
EOF
awk -F'|' 'NF>5 && $5 ~ /^ *\(none\)/' "$OUT"
grep -c '^| [0-9]' "$OUT"
```
Expected: the `awk` prints nothing (no "after" cell came out empty: an empty one means a dump line was not found, for
example a guide label the G-3 review renamed; read the dump, fix the lookup, run the script again); `18`. In row 16 a
`(none)` means that note's sources did not change, which is a finding to check against T6 (spec section 6 says both
notes' sources name the check). Read the file once in full: each "after" cell must be the text the branch shows, and
the "before" cells are fixed text checked at 3e63bec. (The script was dry-run on 3e63bec's dumps with a stand-in
`gearbox.rs` and guide: it fills every cell that exists there and flags rows 9, 10, 12 and 13, the texts T2-T5 add.)

3. Send the owner three files with SendUserFile: `.superpowers/sdd/2026-10-16-bl059-gearbox-check/owner-summary.md`,
   `.superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059.diff` (the full diff) and
   `.superpowers/sdd/2026-10-16-bl059-gearbox-check/bl059-src.diff` (the production and docs part of the diff).

4. Ask with AskUserQuestion (recommended option first), with this question and these options:
   - Question: "1. BL-059 (the gearbox check) is built and checked: whole-branch review passed, GATE PASS with gate 12,
     web smoke passed. The summary and the diff are attached. Merge to main and push? (The push deploys
     colesorkness.com/tools/magcoupler.)"
   - Option "1A OK: merge, push and watch the deploy (recommended)": the diff and the texts are as reviewed; the open
     datasheet question stays open in BL-059 and does not block the release (decision 2A).
   - Option "1B Change texts first": name the rows of the summary to change; each change is a must-fix of Step 3 (with
     its re-review), then Steps 4-8 again.
   - Option "1C Hold": do not merge yet; the branch stays as it is.

Do nothing further until the owner answers in their own message. 1B: run the changes as Step 3 says, then Steps 4-8
again. 1C: stop here and report the branch state.

- [ ] **Step 9: After the OK only: record it, merge, push**

1. `docs/ai/backlog.yaml`, BL-059: replace `  status: in_progress` (the one in BL-059, between `priority: 2` and its
   `notes:`) with `  status: fixed`, and append to its `notes:` string, right before the closing `"`:
   ` The owner reviewed the diff and gave the OK on YYYY-MM-DD (AGENTS.md risk: physics); merged to main and pushed the same day.`
2. `docs/ai/05-update-tracker.md`: in the T10 release bullet of Step 7, replace
   `Not yet done: the owner's review of the diff, the merge, the push and the deploy check.` with
   `The owner reviewed the diff and gave the OK on YYYY-MM-DD; merged to main and pushed; the deploy run and the live smoke are checked in the session that pushed.`
3. `docs/ai/04-memory.yaml`:
   - replace the whole active issue that starts `  - "Magcoupler gearbox check (BL-059, branch magcoupling/bl059-gearbox-check;` with
```text
  - "Magcoupler gearbox check (BL-059; spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md, plan docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md; owner decisions 1A-8A): released YYYY-MM-DD after the owner's review of the diff. The dashboard check compares cold-high at the gearbox input with the gearbox input rating (0.6 N·m, which is 2.85 N·m at the output with the 0.95 efficiency: decision 1A); the example design reads 'Example design: fails 4 of 8 checks' while the shaft-clamp section is paused. Open from it: BL-066 (the repeated slip swing against the gearbox's nominal rating), BL-067 (an operating temperature below the minimum, where cold-high is not the strongest pull-out) and the datasheet question (open_questions)."
```
   - in the NEXT item, replace `then BL-059, the gearbox check` with `BL-059, the gearbox check, shipped YYYY-MM-DD`.
4. `docs/ai/02-system.yaml`: replace `built on branch magcoupling/bl059-gearbox-check` with `merged and deployed YYYY-MM-DD`.
5. Check the YAML still parses: run Step 9 of Task 9 with its `'in_progress'` assertion changed to `'fixed'` (and its
   04-memory `startswith` still `Magcoupler gearbox check (BL-059`). Expected: `67 4 65`.
6. Write `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-b.txt`:
```text
docs(magcoupler): record the owner's OK of BL-059, the gearbox check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   and `.superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-merge.txt`:
```text
Merge magcoupling/bl059-gearbox-check: the magcoupler gearbox check (BL-059)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
7. Commit on the branch, then merge and push:
```bash
git add docs/ai/backlog.yaml docs/ai/05-update-tracker.md docs/ai/04-memory.yaml docs/ai/02-system.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-b.txt
git checkout main
git pull --ff-only origin main
git merge --no-ff magcoupling/bl059-gearbox-check -F .superpowers/sdd/2026-10-16-bl059-gearbox-check/msg/T10-merge.txt
git status --short
git log --oneline -1
git push origin main
```
Expected: the four files staged and the owner's file untracked; the pull prints `Already up to date.` (main was
3cc0952, as `origin/main`); if `main` moved, STOP: merge `main` into the branch, then Steps 4-6 again, then here; the
merge has no conflicts; `git status` shows only the owner's untracked file; the log's subject is
`Merge magcoupling/bl059-gearbox-check: the magcoupler gearbox check (BL-059)`; the push succeeds
(`3cc0952..<merge> main -> main`).

- [ ] **Step 10: Watch the deploy; prove the new bundle is live; live smoke**

Run:
```bash
git rev-parse HEAD
gh run list --workflow deploy-web.yml --branch main --limit 1 --json databaseId,headSha,status,conclusion
```
Expected: the run's `headSha` equals `git rev-parse HEAD` (if the newest run is still 3cc0952's, wait a few seconds and
list again). Then, with that run's `databaseId`:
```bash
gh run watch <databaseId> --exit-status
```
Expected: the run ends `completed` with exit status 0 (slice 2's run, 38904923933, took the same path). On a failure:
`gh run view <databaseId> --log-failed`, report it to the owner, and fix on a new branch with its own red test; never
push again blindly.

Then the live bundle (Vercel serves the new files only once the deploy is done):
```bash
D=magcoupling-rs/target/release-bl059
curl -sS --compressed -o $D/live.wasm -w '%{http_code} %{size_download}\n' https://colesorkness.com/tools/magcoupler/magcoupling-web_bg.wasm
VERDICT="$(sed -n '/pub const WITHIN_THE_GEARBOX_RATING:/,/;/p' magcoupling-rs/src/engine/gearbox.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
echo "[$VERDICT]"
[ -n "$VERDICT" ] && grep -c -a -F "$VERDICT" $D/live.wasm
grep -c -a -F 'Nothing checks cold-high against the gearbox' $D/live.wasm
```
Expected: `200 <size>`; the verdict text in brackets, as in Step 5; at least `1`; `0` (before the deploy the live file
gave `0` and `1`). If it still gives the old
counts, wait a minute and fetch again, at most three times; then report to the owner.

Then invoke the `gui-smoke` skill with the arguments `{"url": "https://colesorkness.com"}` (the same steps against the
live site; `/` redirects to `/tools/` there). Expected: `passed: true` with every calculator field true and no console
error, as in Step 6. A failed field: report the smoke's notes and screenshots to the owner (the site is already live).

- [ ] **Step 11: Delete the local branch; update the memory; report**

1. Run:
```bash
git branch -d magcoupling/bl059-gearbox-check
git branch --show-current
git status --short
```
Expected: `Deleted branch magcoupling/bl059-gearbox-check (was <hash>).`; `main`; only the owner's untracked file. (The
branch was never pushed, so there is no remote branch to delete. Do not touch the other worktrees.)
2. The auto-memory (outside the repo), with the merge's short hash (`git rev-parse --short HEAD`) and the run id of
   Step 10:
   - In `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`,
     replace the whole bullet that starts `- BL-059 gearbox check (2026-10-16):` with:
```text
- BL-059 gearbox check SHIPPED YYYY-MM-DD: owner decisions 1A-8A (input basis: cold_high_input vs 0.6, = 2.85 N·m at the output; counted red, the defaults read "fails 4 of 8"; the conflict sentence in the Problem; Auto unchanged; fixes Lower length / Raise gap; fatigue -> BL-066). Spec docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md, plan docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md. The owner reviewed the diff and said OK; merge <short hash> on main, pushed; deploy run <run id> green; live wasm carries the new verdict; live gui-smoke PASS; branch deleted; the gitignored SDD workspace .superpowers/sdd/2026-10-16-bl059-gearbox-check/ is kept (final-review.md, owner-summary.md). Open: the GAM datasheet line of the 0.6 N·m (input or output; T2N, T2B or T2NOT), BL-066, BL-067. Next on the roadmap: BL-064, then slice 3 (torque sensitivity card); BL-065, BL-066 and BL-067 open (the owner places BL-065); re-confirm deploy authorization per slice.
```
   - In `MEMORY.md` there, replace `; next BL-059 gearbox check` at the end of the `project_magcoupler_guidance_overhaul.md`
     line with `; BL-059 gearbox check shipped YYYY-MM-DD; next BL-064, slice 3`.
3. Tell the owner, in a few lines: merged and deployed (the merge's short hash), the deploy run's link
   (`https://github.com/coleas33/linkage_sim/actions/runs/<run id>`), the live wasm check and the live smoke result,
   and what stays open: the datasheet question (2A), BL-066, BL-067, then BL-064 and slice 3 (BL-065 to place).

---

## Amendment ledger

Where each critique amendment now lives (step numbers are this plan's).

- **A-1** (one commit for T1 and T2): T1 commit-boundary paragraph and Step 18 (no commit, the one expected failure, the controller reviews `git diff`, the ten files); T2 "Before you start", **Files** (Commit line) and Step 9 (A-1's staging list, subject, T1's paragraph first in the body, the notes paragraph last).
- **A-2** (commit counts): T9 Step 1 (at least seven, T2-T8); T10 Step 1 (at least 8, T2-T9).
- **A-3** (one gearbox-only design): T3 Interfaces, Precondition, Step 1 (import only) and Step 2 (`a_design_failing_the_gearbox_check_alone_has_no_conflict_sentence` on `only_the_gearbox_fails()`, its failing-list block gone); T4 Interfaces and Step 1 (`designs()` and `each_problem…` entries).
- **A-4** (T4's import anchor): T4 Step 3, third replacement (Old is T2's list; New in rustfmt's layout; the fallback removed).
- **A-5** (Problem sentence 2 true wherever it shows, PD-1): T3 opening paragraph, Step 2 (`ConflictNumbers` doc, `holds` and `quoted`; `assert_conflict_quotes`; test (b) renamed and rewritten; the `every_text()` extension; the notes), Step 4 (5 × E0425; the stub note), Step 5 (`gearbox_conflict` with the gate and the second remedy; its doc; the PD-1 explanation), Step 6 (renamed test; the second form's live text), Step 7 (coverage of the second form), Step 8 (README Check-guides sentence and test row), Step 10 (commit message sentence); T8 strings table row 4; T9 Steps 5 and 6.
- **A-6** (G-3 brief reviews the shipped texts): T8 strings table rows 4 and 13, "Read in context" list, Step 2 brief (context sentence, probe 8, probes 11-13), Step 4 record template, Step 7 ("If any round changed strings"; the fix subject).
- **A-7** (no re-owned README lines; the tracker's ripple claim): T9 Rules, **Files**, Step 2 (Expected), Step 3 (item 1 only; the old `trace.rs` step deleted), Step 5 (the "On-screen texts" and "Dashboard and guide" bullets), Step 10 (`2` is T1's two), Step 11 (no `trace.rs`).
- **A-8** (BL-065 not placed): T9 Step 3 (README "Next:"), Step 4 item 2 (NEXT), Step 8 item 1 (02-system status); T10 Step 11 (memory bullet, MEMORY.md line, report).
- **A-9** (T6 dates and the ripple rename): T6 Step 8 (brief: the ripple row's label), Step 10 (`PASS_DATE` from the passing round), Step 11 (`(<PASS_DATE>)` headings; the ripple-label bullet in (b)), Step 12(b) (conditional removed).
- **A-10** (`spreadsheet.rs:1000` comment): T2 **Files**, Step 1 (1f), Step 9 (staged).
- **A-11** (docs-count guard): T2 **Files**, Interfaces, Step 8 (8a the test and its red run; the README `:656` edit; the green runs, 2 and 57), Step 9 (message); T6 Steps 6 and 13 (58 with `--features app`, 57 without); T9 **Files** and Step 10 (the `the_docs_count` run).
- **A-12** (mirror and monotonicity tests): T4 **Files**, Interfaces, the note on `every_fix_moves…`, Step 4 (both tests and their explanation), Step 5 (three-test run; mutation), Step 8 (README row), Step 9 (863), Step 10 (message).
- **A-13** (probe designs through `probe_designs()`): T1 Step 1 (intro sentence, table, imports in rustfmt order, the rewritten test).
- **A-14** (crate doc): T1 **Files**, Step 15 (`src/lib.rs`), Step 18 (`git status` and the ten files); T2 Step 9 (staged).
- **A-15** (module doc scoped): T1 Step 4 (the doc's second sentence and why).
- **A-16** (BL-067): T9 title, **Files**, Step 5 ("Docs and backlog" bullet), Step 6 (BL-059 notes: plan deviations and the fbd-math-reviewer stand-in), Step 7 (the BL-067 item), Step 9 (ids to BL-067, its assertion, `67 4 65`), Step 11 (subject); T10 Step 8 (summary paragraph "A corner the plan review found") and Step 9 item 5 (`67 4 65`).
- **A-17** (one chain of test counts): T1 Steps 8, 12 and 18 (851, 1 failed); T2 Steps 5, 6 and 8 (855; explain 56 then 57); T3 Step 7 (860); T4 Steps 7 and 9 (863; sizing 44); T5 Step 12 (867; engine lib 229); T6 Steps 6 and 13 (867; explain 58 and 57); T7 Step 9 (872).
- **A-18** (implementer model): the **Implementer** line under every task heading, T1-T10; T9's `sonnet` executor line removed.
- **A-19** (review prompt and owner summary): T10 **Files** and Rules (`bl059-src.diff`), Step 2 (point 8), Step 8 items 1 (the production-and-docs diff), 2 ("names what would settle it (row 6)") and 3 (three files sent).
- **A-20** (one commit-message folder): T2 Step 9 (`msg/T2.txt`), T3 Step 10, T4 Step 10, T5 Step 14, T6 Step 14 (`msg/T6.txt`, `msg/T6-fix.txt`), T7 Step 10, T8 Step 7 (`msg/T8-fix.txt`, `msg/T8-record.txt`), T9 Step 11, T10 opening paragraph, Step 3 item 5 (`msg/T10-mf-<n>.txt`), Step 7 item 3 (`msg/T10-a.txt`) and Step 9 items 6-7 (`msg/T10-b.txt`, `msg/T10-merge.txt`).
