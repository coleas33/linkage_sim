# Agentic Test-and-Improve Loop — Design

**Date:** 2026-08-08
**Status:** Approved by user (interactive brainstorm, Q1–Q3 + machinery choice)
**Scope:** `linkage-sim-rs/` (Rust solver + GUI), plus `docs/ai/` coordination files

## Problem

The repo has accumulated many known and suspected issues: a doctest in
`core/linear_driver.rs` fails to parse, `cargo clippy --all-targets`
fails on 3 deny-level `approx_constant` errors (plus ~290 warnings),
CSV/HTML export NaN handling is unverified, reaction-force physics
validation is incomplete (current active focus), and there are suspected
DRY/perf/GUI issues nobody has cataloged. (2026-08-08 diagnostics: the
integration test crate compiles clean — earlier "broken tests/ crate"
intel was stale; all 9 test executables build and `cargo test --lib` is
716/716 green.) The user wants an agentic system that finds, verifies, fixes,
and regression-guards these issues with minimal manual cataloging.

## Decisions made during brainstorm

| Question | Decision |
|----------|----------|
| Backlog source | Hybrid: user braindump + multi-agent audit, merged |
| Autonomy / gate | Agents fix on a branch with per-item commits; user reviews and merges batches; nothing pushes to `main` (which auto-deploys) without user review |
| Audit dimensions | Physics correctness, test health/coverage, code quality/perf, GUI smoke (smoke-level only) |
| Machinery | Phased multi-agent Workflow campaigns driven in-session (ralph-loop tail and scheduled cloud runs explicitly rejected for v1) |

## Architecture: three phase types

### Phase 0 — Baseline repair (one-time, first)

- Fix `linkage-sim-rs/tests/` compile errors (stale struct literals) —
  scoped agent, edits `tests/` only.
- Fix the `core/linear_driver.rs` doctest parse failure.
- Exit gate: `cargo test --all` green, `cargo clippy` introduces no new
  warnings (pre-existing warnings are triaged into the backlog, not
  fixed here), WASM build succeeds. Small branch; user merges same day.
- Rationale: every later verification gate depends on a green baseline.

### Phase 1 — Audit (one-time, re-runnable)

- **Inputs:** user braindump of known/suspected issues + parallel finder
  agents, one set per dimension:
  - *Physics:* cross-check solver outputs against independent references —
    FBD-derived reactions at additional poses (use `fbd-derive` skill),
    energy balance over sweeps, virtual-work vs two-pass-statics
    consistency.
  - *Tests:* coverage gaps, NaN propagation into CSV/HTML export, error
    paths, non-Grashof edge cases.
  - *Quality/perf:* DRY violations, clippy findings worth acting on,
    two-pass statics double-SVD sweep cost.
  - *GUI smoke:* Playwright against local WASM build — app loads, all 30
    samples render, 14 plot tabs switch, zero console errors.
- **Verification:** every finding goes to independent skeptic agents
  prompted to refute it. Physics findings: multiple skeptics, majority
  must fail to refute. Mechanical findings: one verifier reproduces it.
  Unconfirmed findings are dropped (logged, not backlogged).
- **Merge:** confirmed findings + braindump items are deduped and
  prioritized into `docs/ai/backlog.yaml`. Braindump items without
  evidence get an audit agent assigned to produce evidence before they
  are fixable.
- **User gate:** user reviews the backlog (kill / reprioritize / add)
  before any fix campaign starts.

### Phase 2..N — Fix campaigns (repeating)

- Batch = top ~5–8 open items, one fresh branch per batch.
- Items run **sequentially** within a batch (parallel fixes on shared
  solver code conflict; parallelism is spent on audit/verify instead).
- Batch done → user reviews branch → merge to `main` → Vercel deploys.

## Backlog artifact: `docs/ai/backlog.yaml`

```yaml
- id: BL-007
  title: CSV export emits literal NaN strings for non-Grashof sweep rows
  dimension: tests          # physics | tests | quality | gui
  risk: mechanical          # physics | mechanical (controls review strictness)
  evidence: "src/gui/export/csv.rs:88 — no NaN filter; repro: non-Grashof sample → export"
  acceptance: "test proving export output parses; NaN rows skipped or empty-celled"
  priority: 2               # 1 = highest
  status: open              # open | in_progress | fixed | deferred | rejected
  notes: ""
```

Rules:

- Every item carries **evidence** (file:line or a reproduction). No
  vibes-based entries, including braindump items.
- **Acceptance is stated as a test** so "done" is never a judgment call.
- `risk: physics` items always get the `fbd-math-reviewer` pass and
  cannot be batch-merged without the user explicitly viewing that item's
  diff. `risk: mechanical` items ride normal batch review.
- `docs/ai/04-memory.yaml` `active_issues` points at this file rather
  than duplicating items.

## Per-item fix pipeline

1. **Red test first.** Write the acceptance test; confirm it fails for
   the expected reason. If it unexpectedly passes → `status: rejected`
   with notes; no fix is written against an unreproduced issue.
2. **Fix.** Minimum change that makes the red test pass; surgical-change
   rules (no adjacent improvements).
3. **Full gate.** `cargo test --all` + `cargo clippy` + WASM build
   (mandatory for any `src/gui/` touch; run always — it is cheap).
4. **Independent review.** Fresh-context reviewer agent on the diff:
   `fbd-math-reviewer` for physics items, general code-reviewer
   otherwise. Findings loop back to step 2. Two consecutive review
   failures escalate the item to the user.
5. **Commit.** One conventional commit per item referencing the backlog
   id; backlog status → `fixed`.

**Mutation step for new physics tests:** when a fix adds a physics
cross-check test, deliberately break the solver quantity under test and
confirm the new test goes red (institutionalizes the lesson from commit
`a88fd80`).

## Success criteria (campaign complete)

1. `cargo test --all` green, and green on every per-item commit.
2. Physics regression suite covers reactions at multiple poses across
   the cycle including force-zone-active poses, plus energy-balance and
   virtual-work-vs-statics cross-checks — as permanent tests.
3. `docs/ai/backlog.yaml` has zero `open` items (all `fixed`,
   `deferred` with reason, or `rejected` with refutation).
4. GUI smoke suite runs green locally.

## Docs coordination

Per batch (not per item), one agent applies the standing `docs/ai`
contract: update `02-system.yaml` (invariants/lessons), `04-memory.yaml`
(active issues), `05-update-tracker.md`. Backlog status updates happen
per item.

## Failure handling

- Solver failure during audit → finding stays unconfirmed, never enters
  backlog.
- Gate failure after a fix → item stays `open` with a `blocked:` note;
  branch resets to last good per-item commit (broken item never poisons
  the batch).
- Baseline breaks mid-campaign (outside merge to `main`) → campaign
  pauses until baseline is green.

## Out of scope (v1)

- Full GUI regression suite (screenshot diffing, interaction tests) —
  smoke only; revisit as its own project.
- Parallel fix execution in worktrees — sequential within batch.
- Overnight ralph-loop for the mechanical tail — possible later upgrade;
  backlog format already supports it.
- Scheduled cloud routines — local toolchain (Windows cargo, WASM,
  Playwright) makes this impractical.
