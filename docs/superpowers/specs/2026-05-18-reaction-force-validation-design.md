# Reaction-force validation for 4-bars with a LinearActuator — design

**Status:** options drafted, validation-reference choice open.
**Active focus:** matches `docs/ai/01-meta.yaml` `active_focus` from this date forward.

## Background

The recent J3-mismatch bug (commit `492fbb7` — "GUI/plot agreement +
lock plot x-bounds to data range") shipped because the test suite
checked solver *self-consistency* — lambdas finite, residual small,
plot data matches HashMap data — but not *absolute correctness*.
Pass-1 (rotational driver as prime mover) and pass-2 (LinearActuator
as prime mover) both pass every existing test, even though they
produce different reaction magnitudes on joints inside the actuator's
load path.

For the simulator's reaction outputs to be trusted in engineering
decisions (bearing sizing, mount design, fastener selection), we need
at least one external reference point — a configuration where we
*know* the correct answer by means other than the simulator itself —
and we need a continuous self-consistency check guarding every
subsequent computation against silent regressions.

This spec captures three approaches the user can choose from. The
choice changes the work scope by ~3×.

## Goal

Within roughly one week of effort, achieve high confidence that
`solver/reactions.rs::solve_reactions_with_actuator` produces
physically correct joint reactions for any 4-bar mechanism with a
LinearActuator in sizing mode (`la.force ≈ 0`), at any pose where the
mechanism solves cleanly.

"High confidence" defined as:
- one absolute-correctness anchor (a config where we proved the
  numbers from first principles or an independent tool), AND
- automated regression coverage that catches future sign-flips,
  Jacobian errors, double-counting, or frame-conversion mistakes
  before they ship.

## Option A — Closed-form free-body-diagram analysis (recommended)

Pick a textbook 4-bar geometry plus a representative LinearActuator
placement. Derive each joint's reaction force by hand at 4–6
representative poses:

1. Top-dead-center (crank along +X, coupler horizontal)
2. Bottom-dead-center
3. Mid-stroke (45° crank)
4. Near-singular toggle position (transmission angle ≈ 0 or π)
5. (Optional) one off-axis pose to break any zero-Y symmetry

For each pose, cut the linkage at each pin, draw a free-body diagram
per body, and write the equilibrium equations:

```
ΣF_x = 0
ΣF_y = 0
ΣM_cg = 0
```

Solve the linear system on paper (or in a one-off Python/Mathematica
script kept in `data/validation/`), record the expected
`(F_x, F_y)` at each joint, and bake them into a Rust regression
test that builds the same mechanism, runs `solve_reactions_with_actuator`,
and asserts each reaction matches within `1e-4 N` (statics tolerance).

**Scope:** ~250 LOC (test + fixture + derivation comments). One new file:
`linkage-sim-rs/tests/reaction_validation_4bar_actuator.rs` (lives in
the integration-tests directory so it stays close to the
`solver::reactions::tests` module but doesn't bloat the lib build).

**Pros**
- Zero external dependencies. Reproducible by anyone with a copy of
  the repo.
- Validates absolute correctness at the chosen poses; if pass-2 has a
  sign flip or a double-count, the test fails immediately.
- Forces us to write down the math, which doubles as documentation.

**Cons**
- Tedious to derive once. ~3–4 hours of careful FBD work.
- Only validates the specific geometry encoded. Different
  configurations could in principle still be wrong; broader coverage
  needs option B.

## Option B — Internal cross-checks expanded

No absolute-truth comparison. Instead, extend the simulator's own
consistency checks to cover the actuator-driven case:

1. Extend `analysis/virtual_work.rs` to assert the actuator power
   balance:
   ```
   F_act · (dL/dt) = Σ_bodies F_ext · v_cg + Σ_bodies τ_ext · ω
   ```
   when a LinearActuator is the prime mover (pass-2 ran). The existing
   `virtual_work_check` validates `τ_driver · q̇_driver = Σ F_ext · v`
   only, which is the rotational-driver case.
2. Add a per-body equilibrium check at every solved pose:
   ```
   ‖ΣF_on_body_i‖ < ε_F
   ‖ΣM_about_body_i_cg‖ < ε_M
   ```
   where the sum is over reaction forces from connected joints plus
   external forces applied to that body. ε's are scaled by the
   problem's natural force scale (e.g. `max(|F_act|, |gravity|) * 1e-6`).
3. Surface a green/red "validation passed" badge in the GUI
   property panel beside the Joint Reactions readout.
4. Run the equilibrium check as a `debug_assert!` in dev builds inside
   `solve_reactions_with_actuator`, so any future change that breaks
   self-consistency panics in CI.

**Scope:** ~150 LOC. Touches `analysis/virtual_work.rs`,
`solver/reactions.rs`, `gui/property_panel/mod.rs`.

**Pros**
- Catches regressions across *every* test mechanism, not just one.
- Live confidence indicator in the GUI gives the user instant feedback
  while editing geometry.
- Cheap to maintain — the checks are derived from the same physics
  the solver implements.

**Cons**
- This is fundamentally a *self*-consistency check. The solver could
  satisfy all internal checks while still being wrong in absolute
  terms (e.g., a sign-flipped measurement Jacobian would preserve
  internal consistency but produce wrong magnitudes).
- Cannot detect a class of bugs where the same flaw is present in
  both the forward solve and the validation check.

## Option C — External tool comparison

The user produces a reference dataset by running an equivalent
configuration in an independent multibody simulator — MATLAB
Simscape Multibody, MSC ADAMS, OpenSim, or a hand-coded Python solve
using `sympy.physics.mechanics`. Reactions are exported per sample as
CSV.

A new test crate `tests/external_validation.rs` loads the CSV,
constructs the same mechanism in this codebase, runs
`compute_sweep_data`, and asserts each joint reaction matches the
reference within an agreed tolerance (~1% of peak load is realistic
for cross-tool comparison; different solvers handle damping
differently).

**Scope:** ~100 LOC of harness. Open-ended on the user side
(producing the reference can be hours to days depending on the tool's
learning curve).

**Pros**
- Independent gold standard. Two physically grounded codebases
  agreeing is very strong evidence of correctness.
- Documents an explicit benchmark anyone can reproduce.

**Cons**
- Requires the user to produce or source the reference. Idle harness
  until they do.
- Tool-to-tool tolerance is inherently looser than analytical (~1%
  vs the 1e-4 N possible with option A). Sub-percent bugs slip
  through.
- Adds an external dependency to the project's mental model —
  "valid" now requires keeping a Simscape model in sync.

## Hybrid — A + B

Run option A this turn, then immediately do option B as a follow-up.
A pins absolute correctness at the chosen anchor. B then layers a
live consistency check on top so every other mechanism configuration
inherits regression protection without requiring its own FBD
derivation.

**Scope:** ~400 LOC across two commits. ~5–6 hours of work.

**Recommendation:** this is the best path if the user has the bandwidth.
Strict superset of either option in isolation.

## Recommendation

**A first**, decision on B/C/Hybrid deferred until the A test is green.
Reasoning: A is the only option that produces an absolute-correctness
anchor, which is what was missing when the J3 bug shipped. B is
strictly better with A in place (live check anchored on a verified
config); C only pays off if the user already has a Simscape/ADAMS
license and a model.

## Open question

**Which option does the user want?** Tracked in `docs/ai/04-memory.yaml`
until resolved. Decision blocks the implementation plan; once
resolved, invoke the `superpowers:writing-plans` skill to break the
chosen option into a numbered task list.

## Out of scope (explicit non-goals)

- Validating *non*-LinearActuator force elements (springs, dampers,
  motors). The recent bug was actuator-specific; springs already get
  exercised by existing virtual-work tests.
- Validating 6-bar or higher topologies. The user's focus is 4-bars;
  generalising the validation harness comes after the 4-bar anchor
  is in.
- Validating pass-1 reactions (no LinearActuator present). Pass-1 has
  not been implicated in any reported bug and has existing test
  coverage (`solver::statics::tests::fourbar_reactions_extract`).

## Files touched (per option)

| Path | Option A | Option B | Option C |
|---|---|---|---|
| `linkage-sim-rs/tests/reaction_validation_4bar_actuator.rs` | new | — | — |
| `linkage-sim-rs/src/analysis/virtual_work.rs` | — | extend | — |
| `linkage-sim-rs/src/solver/reactions.rs` | — | add equilibrium check | — |
| `linkage-sim-rs/src/gui/property_panel/mod.rs` | — | add validation badge | — |
| `linkage-sim-rs/tests/external_validation.rs` | — | — | new |
| `data/validation/4bar_actuator_reference.csv` | — | — | new (user-produced) |
| `data/validation/4bar_actuator_fbd.py` | maybe | — | — |
| `docs/ai/02-system.yaml` | append lesson | append lesson | append lesson |
| `docs/ai/05-update-tracker.md` | log | log | log |
