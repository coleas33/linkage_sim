# Agentic Test-and-Improve Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and boot the agentic test-and-improve loop from `docs/superpowers/specs/2026-08-08-agentic-test-improve-loop-design.md`: repair the baseline (Phase 0), create the backlog artifact and the three workflow scripts (audit, GUI smoke, fix campaign), and run the Phase 1 audit to produce a user-reviewed backlog.

**Architecture:** Phase 0 is two small direct fixes (doctest, clippy deny errors) plus a gate script that every later fix must pass. The loop machinery is three Workflow scripts in `.claude/workflows/` consumed by the orchestrating session, plus `docs/ai/backlog.yaml` as the single queue. Fix campaigns (Phase 2+) are operations, not build work — they reuse `fix-campaign.js` per batch and are out of scope here beyond selecting batch 1.

**Tech Stack:** Rust 1.89.0 (crate at `linkage-sim-rs/`), cargo test/clippy, wasm32-unknown-unknown + wasm-bindgen-cli 0.2.114, bash scripts run via Git Bash on Windows, Claude Code Workflow scripts (plain JS), Playwright MCP for GUI smoke.

## Global Constraints

- Repo root: `C:\Users\Cole\source\repos\linkage_simulation`; crate root: `linkage-sim-rs/` (all cargo commands run there).
- WASM compile gate (exact): `cargo check --target wasm32-unknown-unknown --bin linkage-web --no-default-features --features raster` (mirrors `.github/workflows/deploy-web.yml:38-50` minus release/link).
- Full WASM build (pre-merge only): `./scripts/build_web.sh`; serve: `./scripts/serve_web.sh` (port 8080).
- Conventional commits (`feat:`/`fix:`/`test:`/`docs:`/`refactor:`) ending with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- All work in this plan happens on branch `agent/phase0-baseline`. Agents NEVER push. The user merges to `main` (main auto-deploys via Vercel).
- After file-modifying work, apply the `docs/ai` coordination contract (`docs/ai/01-meta.yaml:18-25`) — Task 8 does this once for the whole plan.
- Windows host: run `.sh` scripts through the Bash tool (Git Bash), never PowerShell.
- Workflow scripts must not use `Date.now()`, `Math.random()`, or argless `new Date()`.

---

### Task 1: Create branch; fix the `linear_driver.rs` doctest

**Files:**
- Modify: `linkage-sim-rs/src/core/linear_driver.rs:6-8`

**Interfaces:**
- Produces: green `cargo test --doc`. Later tasks rely on `cargo test --all` passing, which includes doctests.

Background: line 7 of the module doc is a 4-space-indented line preceded and followed by blank doc lines, so Markdown promotes it to a code block and rustdoc compiles it as Rust (`|P_b - P_a|` parses as a closure). Verified failing 2026-08-08: 5 errors, `test result: FAILED. 0 passed; 1 failed`. The similar-looking line 22 is NOT a doctest (no preceding blank doc line) — leave it alone.

- [ ] **Step 1: Create the working branch**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation
git checkout -b agent/phase0-baseline
```

- [ ] **Step 2: Run the doctest to confirm the failure (red)**

Run: `cd linkage-sim-rs && cargo test --doc 2>&1 | tail -20`
Expected: `src\core\linear_driver.rs - core::linear_driver (line 7)` FAILED, "expected a pattern, found an expression".

- [ ] **Step 3: Fence the formula as text**

In `linkage-sim-rs/src/core/linear_driver.rs`, replace lines 6-8:

```rust
//!
//!     Phi = |P_b - P_a| - d(t) = 0
//!
```

with:

```rust
//!
//! ```text
//! Phi = |P_b - P_a| - d(t) = 0
//! ```
//!
```

- [ ] **Step 4: Run doctests to verify green**

Run: `cd linkage-sim-rs && cargo test --doc 2>&1 | tail -5`
Expected: `0 failed` (the text fence is not compiled).

- [ ] **Step 5: Commit**

```bash
git add linkage-sim-rs/src/core/linear_driver.rs
git commit -m "fix(docs): fence linear_driver formula as text so rustdoc stops compiling it

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Fix the 3 deny-level `clippy::approx_constant` errors

**Files:**
- Modify: `linkage-sim-rs/src/gui/undo.rs:251` and `:264`
- Modify: `linkage-sim-rs/tests/property_tests.rs:327`

**Interfaces:**
- Produces: `cargo clippy --all-targets` exits 0. Task 3's gate script depends on this.

Background: `approx_constant` is deny-by-default, so these are errors, not warnings. Verified failing 2026-08-08.

- [ ] **Step 1: Confirm clippy fails (red)**

Run: `cd linkage-sim-rs && cargo clippy --all-targets 2>&1 | grep -A2 approx_constant | head -20`
Expected: errors at `src\gui\undo.rs:251`, `src\gui\undo.rs:264`, `tests\property_tests.rs:327`.

- [ ] **Step 2: Replace the approximate constants**

`src/gui/undo.rs:251` — change `driver_omega: 3.14,` to:

```rust
driver_omega: std::f64::consts::PI,
```

`src/gui/undo.rs:264` — change `assert_eq!(restored.driver_omega, 3.14);` to:

```rust
assert_eq!(restored.driver_omega, std::f64::consts::PI);
```

(The two must stay equal — the test round-trips the value.)

`tests/property_tests.rs:327` — change `theta_0 in 0.0_f64..6.28` to:

```rust
theta_0 in 0.0_f64..std::f64::consts::TAU
```

(TAU = 6.2832 > 6.28 widens the property strategy range slightly; the property must hold on the full circle anyway, so this is a strengthening, not a behavior change.)

- [ ] **Step 3: Verify clippy exits 0 and touched tests pass**

Run: `cd linkage-sim-rs && cargo clippy --all-targets && cargo test --lib undo && cargo test --test property_tests`
Expected: clippy exit 0 (warnings still print — only errors are gone); undo tests pass; property_tests pass.

- [ ] **Step 4: Commit**

```bash
git add linkage-sim-rs/src/gui/undo.rs linkage-sim-rs/tests/property_tests.rs
git commit -m "fix(lint): use std PI/TAU constants, clearing deny-level approx_constant errors

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Gate script `linkage-sim-rs/scripts/gate.sh`

**Files:**
- Create: `linkage-sim-rs/scripts/gate.sh`

**Interfaces:**
- Produces: `bash scripts/gate.sh` (run from `linkage-sim-rs/`) exits 0 and prints `GATE PASS` iff `cargo test --all`, `cargo clippy --all-targets`, and the WASM check all succeed. `--full` additionally runs `./scripts/build_web.sh`. Every fixer agent in `fix-campaign.js` (Task 7) runs exactly this command.

Note on the spec's "no new warnings" clippy gate: v1 enforces clippy's exit code only (deny-level lints). Warning regression is enforced by the reviewer agent (Task 7 review prompt rejects diffs adding warnings on touched files) and warning debt is tracked as backlog item BL-003. A count-based baseline was considered and rejected as fragile (duplicate warnings across targets make counts noisy).

- [ ] **Step 1: Write the script**

Create `linkage-sim-rs/scripts/gate.sh`:

```bash
#!/usr/bin/env bash
# Verification gate for the agentic fix loop.
# Usage: scripts/gate.sh [--full]   (--full also runs the release WASM build)
# Exit 0 + "GATE PASS" = all gates green. Any failure exits non-zero.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== gate 1/3: cargo test --all =="
cargo test --all

echo "== gate 2/3: cargo clippy --all-targets =="
cargo clippy --all-targets

echo "== gate 3/3: WASM check =="
cargo check --target wasm32-unknown-unknown --bin linkage-web --no-default-features --features raster

if [[ "${1:-}" == "--full" ]]; then
  echo "== gate 4 (--full): release WASM build =="
  ./scripts/build_web.sh
fi

echo "GATE PASS"
```

- [ ] **Step 2: Make it executable and run it**

Run: `cd linkage-sim-rs && chmod +x scripts/gate.sh && bash scripts/gate.sh`
Expected: all three gates run; final line `GATE PASS`; exit 0. (Takes several minutes — test suite alone is ~3 min.) If the wasm32 target is missing, run `rustup target add wasm32-unknown-unknown` once and re-run.

- [ ] **Step 3: Commit**

```bash
git add linkage-sim-rs/scripts/gate.sh
git commit -m "feat(scripts): add gate.sh verification gate for the agentic fix loop

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Backlog artifact `docs/ai/backlog.yaml` + pointer in `04-memory.yaml`

**Files:**
- Create: `docs/ai/backlog.yaml`
- Modify: `docs/ai/04-memory.yaml:5` (`active_issues: []`)

**Interfaces:**
- Produces: the queue consumed by `audit-campaign.js` merges (Task 9) and `fix-campaign.js` batches (Task 7). Item shape (all keys required except `notes`): `id, title, dimension(physics|tests|quality|gui), risk(physics|mechanical), evidence, acceptance, priority(1-3), status(open|in_progress|fixed|deferred|rejected|escalated), notes`.

- [ ] **Step 1: Write the seed backlog**

Create `docs/ai/backlog.yaml` (seeds are all evidence-backed from the 2026-08-08 diagnostics; BL-001/BL-002 are recorded as fixed by Tasks 1-2 so the file is the single history of loop work):

```yaml
# Agentic test-and-improve loop queue.
# Spec: docs/superpowers/specs/2026-08-08-agentic-test-improve-loop-design.md
# Rules: every item carries evidence (file:line or repro); acceptance is a
# verifiable check; risk=physics items require fbd-math-reviewer + explicit
# user review of that item's diff before merge.
# status: open | in_progress | fixed | deferred | rejected | escalated

- id: BL-001
  title: linear_driver.rs module doc formula compiled as failing doctest
  dimension: tests
  risk: mechanical
  evidence: "src/core/linear_driver.rs:7 — indented block after blank doc line; cargo test --doc: 1 failed (2026-08-08)"
  acceptance: "cargo test --doc reports 0 failed"
  priority: 1
  status: fixed
  notes: "Fixed in Phase 0 (agent/phase0-baseline)."

- id: BL-002
  title: cargo clippy --all-targets fails on 3 deny-level approx_constant errors
  dimension: quality
  risk: mechanical
  evidence: "src/gui/undo.rs:251,264 (3.14 for PI); tests/property_tests.rs:327 (6.28 for TAU); clippy exit != 0 (2026-08-08)"
  acceptance: "cargo clippy --all-targets exits 0"
  priority: 1
  status: fixed
  notes: "Fixed in Phase 0 (agent/phase0-baseline)."

- id: BL-003
  title: "Clippy warning debt: ~289 lib warnings (167 collapsible_if; 234 auto-fixable)"
  dimension: quality
  risk: mechanical
  evidence: "cargo clippy --all-targets 2026-08-08: lib 289 warnings, lib-test 305 (278 dupes); top lints collapsible_if 167, too_many_arguments 17, empty_line_after_doc_comments 13"
  acceptance: "cargo clippy --all-targets produces 0 warnings on lib and lib-test targets; cargo test --all stays green"
  priority: 3
  status: open
  notes: "Mostly cargo clippy --fix + review; too_many_arguments items may be deferred individually with reasons."

- id: BL-004
  title: docs/guides/WASM_DEPLOYMENT.md drifted from actual build/deploy config
  dimension: quality
  risk: mechanical
  evidence: "WASM_DEPLOYMENT.md:36-40 omits --features raster (build_web.sh:20-24 and deploy-web.yml:38-50 include it); WASM_DEPLOYMENT.md:100 claims max-age=31536000 immutable but web/vercel.json sets max-age=0 must-revalidate"
  acceptance: "doc's build command and cache-control statements match build_web.sh and web/vercel.json verbatim"
  priority: 3
  status: open
  notes: ""

- id: BL-005
  title: .claude/agents/rust-test-fixer.md premise is stale (tests already fixed)
  dimension: quality
  risk: mechanical
  evidence: "Agent description claims tests/ struct literals miss fields; commit b919485 (2026-05-26) added them; cargo test --all --no-run exits 0 (2026-08-08)"
  acceptance: "agent description rewritten to a general 'repair broken test-crate compiles' role or the agent is deleted; no claim that tests/ is currently broken"
  priority: 3
  status: open
  notes: ""

- id: BL-006
  title: nom v1.2.4 flagged future-incompatible (will be rejected by a future rustc)
  dimension: quality
  risk: mechanical
  evidence: "cargo report during 2026-08-08 build: 'nom v1.2.4 ... will be rejected by a future version of Rust'"
  acceptance: "cargo tree -i nom documented in notes; dependency updated/replaced, or item deferred with the upstream blocker named"
  priority: 2
  status: open
  notes: ""

- id: BL-007
  title: CSV/HTML export behavior on NaN-padded (non-Grashof) sweep rows unverified
  dimension: tests
  risk: mechanical
  evidence: "docs/ai/02-system.yaml risks: 'CSV/HTML export needs verification' for NaN rows; no test in tests/ or src/gui/export/ covers a non-Grashof sweep export"
  acceptance: "test exports a non-Grashof sweep to CSV and HTML; CSV parses row-count-consistently with a documented NaN policy (skip or empty cell); HTML contains no literal 'NaN' artifacts"
  priority: 2
  status: open
  notes: ""

- id: BL-008
  title: mixed_script_confusables warning for `let δ = 1e-6` in IK derivatives
  dimension: quality
  risk: mechanical
  evidence: "solver/inverse_kinematics/derivatives.rs:217:13 — rustc mixed_script_confusables warning during doctest build (2026-08-08)"
  acceptance: "identifier renamed (e.g. delta); warning gone; cargo test --all green"
  priority: 3
  status: open
  notes: ""

- id: BL-009
  title: Extend reaction-force physics validation to force-zone poses + cross-checks
  dimension: physics
  risk: physics
  evidence: "docs/ai/01-meta.yaml active_focus; docs/ai/04-memory.yaml open question (validation reference choice, spec 2026-05-18 recommends FBD-first); existing FBD regression covers theta_2=60deg only (solver/reactions.rs tests)"
  acceptance: "FBD regression tests at additional poses including force-zone-active ones (use fbd-derive skill); energy-balance and virtual-work-vs-statics cross-check tests; each new check mutation-tested (break solver quantity, confirm test fails, restore)"
  priority: 1
  status: open
  notes: "Coordinate with fbd-math-reviewer conventions from tests::fbd_validates_pass2_reactions_at_60deg."
```

- [ ] **Step 2: Point `active_issues` at the backlog**

In `docs/ai/04-memory.yaml`, replace `active_issues: []` with:

```yaml
active_issues:
  - "Agentic loop queue: see docs/ai/backlog.yaml (open items tracked there, not here)"
```

- [ ] **Step 3: Validate YAML parses**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation && python -c "import yaml,sys; d=yaml.safe_load(open('docs/ai/backlog.yaml')); assert isinstance(d,list) and all('id' in i and 'evidence' in i and 'acceptance' in i for i in d); print(len(d),'items OK')" && python -c "import yaml; yaml.safe_load(open('docs/ai/04-memory.yaml')); print('memory OK')"`
Expected: `9 items OK` and `memory OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/ai/backlog.yaml docs/ai/04-memory.yaml
git commit -m "docs(ai): seed agentic-loop backlog with 9 evidence-backed items

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Audit workflow `.claude/workflows/audit-campaign.js`

**Files:**
- Create: `.claude/workflows/audit-campaign.js`

**Interfaces:**
- Consumes: `args.dimensions` (optional subset of `['physics','tests','quality','gui']`), `args.braindump` (optional array of user-suspected-issue strings), `args.selftest` (bool).
- Produces: `{ confirmed: Finding[], dropped: Finding[] }` where Finding = `{title, dimension, risk, evidence, acceptance, severity, notes?, confirmed, verdicts}`. The orchestrating session merges `confirmed` into `docs/ai/backlog.yaml` (Task 9 procedure).

- [ ] **Step 1: Write the workflow script**

Create `.claude/workflows/audit-campaign.js`:

```javascript
export const meta = {
  name: 'audit-campaign',
  description: 'Audit linkage-sim-rs across physics/tests/quality/gui; adversarially verify findings',
  whenToUse: 'Phase 1 (or re-audit) of the agentic test-and-improve loop',
  phases: [
    { title: 'Find', detail: 'one finder per dimension (+braindump chaser)' },
    { title: 'Verify', detail: 'skeptics try to refute each finding' },
  ],
}

const FINDINGS_SCHEMA = {
  type: 'object', required: ['findings'],
  properties: { findings: { type: 'array', items: {
    type: 'object',
    required: ['title', 'dimension', 'risk', 'evidence', 'acceptance', 'severity'],
    properties: {
      title: { type: 'string' },
      dimension: { enum: ['physics', 'tests', 'quality', 'gui'] },
      risk: { enum: ['physics', 'mechanical'] },
      evidence: { type: 'string' },
      acceptance: { type: 'string' },
      severity: { type: 'integer', minimum: 1, maximum: 3 },
      notes: { type: 'string' },
    } } } },
}
const VERDICT_SCHEMA = {
  type: 'object', required: ['confirmed', 'reason'],
  properties: { confirmed: { type: 'boolean' }, reason: { type: 'string' } },
}

if (args && args.selftest) {
  return { ok: true, dimensions: ['physics', 'tests', 'quality', 'gui'] }
}

const ROOT = 'C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs'
const COMMON = `Repo crate: ${ROOT}. Read ../docs/ai/02-system.yaml first (invariants, known risks — do not re-report items already in ../docs/ai/backlog.yaml). Every finding MUST carry concrete evidence: file:line, or a reproduction command plus its observed output. Acceptance must be phrased as a runnable check. severity: 1=correctness/physics, 2=robustness/perf, 3=hygiene. No pure style nitpicks. Return findings even if few; an empty list is a valid answer.`

const DIMS = {
  physics: `Dimension: physics correctness. Cross-check solver outputs against independent physics: (a) pick 2 sample mechanisms with force elements, derive expected joint reactions or input effort independently (FBD by hand — the repo skill fbd-derive documents the conventions; follow tests::fbd_validates_pass2_reactions_at_60deg in src/solver/reactions.rs), compare against solver values via a small cargo test or rust-script; (b) check energy balance: driver input power vs force-element power sinks over a sweep for one sample; (c) check virtual-work actuator force vs two-pass statics (docs/ai/02-system.yaml two_pass_statics_for_actuators). Report only discrepancies beyond numerical tolerance (state the tolerance you used).`,
  tests: `Dimension: test health/coverage. Hunt gaps where 02-system.yaml risks meet untested code: NaN-padded sweep rows entering src/gui/export/csv.rs and report.rs (BL-007 exists — look BEYOND it: SVG/GIF/DXF exporters, envelope/statistics code that consumes sweep channels); error paths in io/serialization.rs (malformed JSON, wrong schema_version); solver failure paths (singular Jacobian handling in solver/kinematics.rs). For each gap: name the specific untested branch (file:line) and the test that should exist.`,
  quality: `Dimension: code quality + performance. (a) DRY: find concrete repeated logic blocks >10 lines appearing 2+ times (file:line pairs) — the user wants repetition flagged aggressively; (b) perf: the two-pass statics solve doubles sweep cost (02-system.yaml risks) — identify whether pass-1 results can be reused and any allocation hot spots in the 361-iteration sweep loop (src/gui/sweep/mod.rs); (c) cargo tree -i nom to trace the future-incompat dep. Skip clippy warning bulk (BL-003 covers it).`,
  gui: `Dimension: GUI static review (live smoke is a separate workflow). Review src/gui/ for: state mutations missing the push_undo-before/rebuild-after contract (grep mutate sites against AppState::mutate_and_rebuild, see 02-system.yaml invariants); WASM/native divergence (rfd dialogs vs dropped_files paths — lessons_learned says both must exist for every import flow); DragValue::changed() used where drag_stopped()||lost_focus() is required. Report each violating site file:line.`,
}

const selected = ((args && args.dimensions) || Object.keys(DIMS))
  .filter(k => DIMS[k])
  .map(k => ({ key: k, prompt: DIMS[k] }))

const braindump = (args && args.braindump) || []
const finders = selected.map(d => ({
  key: d.key,
  prompt: `${COMMON}\n\n${d.prompt}`,
}))
if (braindump.length) {
  finders.push({
    key: 'braindump',
    prompt: `${COMMON}\n\nDimension: user braindump chase-down. The user suspects these issues (verbatim, may be vague):\n${braindump.map((b, i) => `${i + 1}. ${b}`).join('\n')}\nFor EACH: find concrete evidence (file:line or reproduction) that confirms, refines, or refutes it. Emit one finding per confirmed/refined item with dimension+risk you judge appropriate; omit refuted ones but mention them in notes of a summary finding titled "braindump triage".`,
  })
}

const results = await pipeline(
  finders,
  f => agent(f.prompt, { label: `find:${f.key}`, phase: 'Find', schema: FINDINGS_SCHEMA }),
  (found, f) => {
    if (!found || !found.findings || !found.findings.length) return []
    return parallel(found.findings.map(fi => () => {
      const lenses = fi.risk === 'physics'
        ? ['reproduce it from scratch', 'check the math, signs, frames, and units independently', 'find an alternative benign explanation']
        : ['reproduce it from scratch']
      return parallel(lenses.map(lens => () =>
        agent(`Adversarial skeptic. Try to REFUTE this finding via the lens: ${lens}. ${COMMON}\nFinding: ${JSON.stringify(fi)}\nRules: default confirmed=false unless you positively reproduce/validate the evidence yourself. State exactly what you ran or read.`,
          { label: `verify:${(fi.title || '').slice(0, 40)}`, phase: 'Verify', schema: VERDICT_SCHEMA })))
        .then(vs => {
          const good = vs.filter(Boolean)
          const yes = good.filter(v => v.confirmed).length
          return { ...fi, confirmed: good.length > 0 && yes > good.length / 2, verdicts: good.map(v => v.reason) }
        })
    }))
  },
)

const flat = results.filter(Boolean).flat().filter(Boolean)
const confirmed = flat.filter(f => f.confirmed)
const dropped = flat.filter(f => !f.confirmed)
log(`Audit complete: ${confirmed.length} confirmed, ${dropped.length} dropped`)
return { confirmed, dropped }
```

- [ ] **Step 2: Selftest the script**

Invoke the Workflow tool with `{scriptPath: '.claude/workflows/audit-campaign.js', args: {selftest: true}}` (from the orchestrating session).
Expected: returns `{ ok: true, dimensions: [...] }` with zero agents spawned — proves the script parses and the arg plumbing works.

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/audit-campaign.js
git commit -m "feat(workflows): audit-campaign workflow (find + adversarial verify)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: GUI smoke workflow `.claude/workflows/gui-smoke.js`

**Files:**
- Create: `.claude/workflows/gui-smoke.js`

**Interfaces:**
- Consumes: `args.url` (default `http://localhost:8080`), `args.selftest`. Precondition (caller's job): `./scripts/build_web.sh` has been run and `./scripts/serve_web.sh` is serving.
- Produces: `{ passed, canvas_present, console_errors, screenshot_note, notes }`.

Scope note (deviation from spec success-criterion 4, flagged to user at plan review): v1 smoke = page loads, canvas renders, zero console errors. Driving egui's canvas UI (loading all 30 samples, switching 14 tabs) needs pixel-coordinate clicking or an accessibility bridge — moved to the backlog as an audit-era investigation item rather than hand-waved here.

- [ ] **Step 1: Write the workflow script**

Create `.claude/workflows/gui-smoke.js`:

```javascript
export const meta = {
  name: 'gui-smoke',
  description: 'Smoke-test the locally served WASM build: loads, canvas present, console clean',
  whenToUse: 'Before merging a batch that touched src/gui/, and during Phase 1 audit',
  phases: [{ title: 'Smoke' }],
}

const SMOKE_SCHEMA = {
  type: 'object', required: ['passed', 'canvas_present', 'console_errors'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    screenshot_note: { type: 'string' },
    notes: { type: 'string' },
  },
}

if (args && args.selftest) { return { ok: true } }

const url = (args && args.url) || 'http://localhost:8080'
const result = await agent(
  `Smoke-test the WASM linkage app at ${url} using Playwright MCP tools (load them via ToolSearch, e.g. "select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_snapshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_wait_for,mcp__plugin_playwright_playwright__browser_close").
Steps: navigate to ${url}; wait 5 seconds for WASM init; snapshot the page and confirm a <canvas> element exists; collect console messages; take a screenshot and judge whether it shows a rendered app (dark CAD-style UI) vs a blank page; close the browser.
passed=true only if: page loaded, canvas present, zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (caller must have run scripts/serve_web.sh).`,
  { label: 'gui-smoke', phase: 'Smoke', schema: SMOKE_SCHEMA },
)
return result
```

- [ ] **Step 2: Selftest**

Invoke Workflow with `{scriptPath: '.claude/workflows/gui-smoke.js', args: {selftest: true}}`.
Expected: `{ ok: true }`, zero agents.

- [ ] **Step 3: Live smoke run (build + serve + run)**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs
./scripts/build_web.sh
```

Then start `./scripts/serve_web.sh` as a background process, invoke Workflow with `{scriptPath: '.claude/workflows/gui-smoke.js'}`, and stop the server afterward.
Expected: `passed: true, canvas_present: true, console_errors: []`. If it fails, the failure is itself a Phase 1 finding — record it; do not block this task on fixing the app.

- [ ] **Step 4: Commit**

```bash
git add .claude/workflows/gui-smoke.js
git commit -m "feat(workflows): gui-smoke workflow for the served WASM build

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Fix-campaign workflow `.claude/workflows/fix-campaign.js`

**Files:**
- Create: `.claude/workflows/fix-campaign.js`

**Interfaces:**
- Consumes: `args.items` — array of backlog items (Task 4 shape), already selected as a batch by the orchestrating session; the batch branch is already checked out. `args.selftest`.
- Produces: `{ results: [{id, outcome: 'fixed'|'rejected'|'blocked'|'escalated'|'agent-error', reviews?, notes?}] }`. The orchestrating session updates `docs/ai/backlog.yaml` statuses from this and presents escalations to the user.

- [ ] **Step 1: Write the workflow script**

Create `.claude/workflows/fix-campaign.js`:

```javascript
export const meta = {
  name: 'fix-campaign',
  description: 'Sequentially fix backlog items: red test, fix, gate, fresh-context review, commit',
  whenToUse: 'Phase 2+ batches of the agentic test-and-improve loop',
  phases: [{ title: 'Fix' }],
}

const FIX_SCHEMA = {
  type: 'object', required: ['status', 'notes'],
  properties: {
    status: { enum: ['fixed', 'rejected', 'blocked'] },
    red_test_confirmed: { type: 'boolean' },
    gate_pass: { type: 'boolean' },
    files_changed: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
}
const REVIEW_SCHEMA = {
  type: 'object', required: ['approved', 'findings'],
  properties: {
    approved: { type: 'boolean' },
    findings: { type: 'array', items: { type: 'string' } },
  },
}

if (args && args.selftest) { return { ok: true } }

const items = (args && args.items) || []
if (!items.length) return { error: 'pass args.items = array of backlog items (see docs/ai/backlog.yaml)' }

const ROOT = 'C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs'
const SPEC = 'docs/superpowers/specs/2026-08-08-agentic-test-improve-loop-design.md'
const results = []

for (const item of items) {
  log(`Fixing ${item.id}: ${item.title}`)

  const fix = await agent(
    `Fix ONE backlog item in ${ROOT}. The batch branch is already checked out — work in the main working tree. Do NOT commit. NEVER push. Spec: ../${SPEC}.
Item: ${JSON.stringify(item)}
Pipeline (all steps mandatory, in order):
1. RED TEST: write the test described by the item's acceptance field; run it; confirm it fails for the expected reason. If it unexpectedly PASSES, stop: return status "rejected" with what you observed. (Doc-only items: verify the drift exists instead, then treat the corrected doc as the fix.)
2. FIX: minimum change making the red test pass. No adjacent improvements, no drive-by refactors.
3. GATE: run "bash scripts/gate.sh" from ${ROOT}; it must exit 0 printing GATE PASS. If you cannot get it green after honest attempts, return status "blocked" with the failure output in notes.
4. MUTATION (only if you added a physics cross-check test): temporarily break the solver quantity under test, confirm your new test goes red, restore. Note the mutation you used.
Return files_changed as repo-relative paths.`,
    { phase: 'Fix', label: `fix:${item.id}`, schema: FIX_SCHEMA },
  )

  if (!fix || fix.status !== 'fixed') {
    results.push({ id: item.id, outcome: fix ? fix.status : 'agent-error', notes: fix ? fix.notes : 'agent died' })
    if (fix && fix.status !== 'rejected') {
      await agent(`In ${ROOT}: if "git status --porcelain" shows changes, run: git stash push -u -m "loop-${item.id}-${fix.status}". Confirm the working tree is clean afterward and report what you stashed.`,
        { phase: 'Fix', label: `clean:${item.id}` })
    }
    continue
  }

  let approved = false
  let lastReview = null
  for (let round = 1; round <= 2 && !approved; round++) {
    lastReview = await agent(
      `Fresh-context review of the UNCOMMITTED diff in ${ROOT} fixing backlog item: ${JSON.stringify(item)}.
Run "git status" and "git diff" yourself. Judge: (a) correctness of the change; (b) test quality — would the new test fail if the bug came back?; (c) scope — every changed hunk traces to this item; (d) run "cargo clippy" on the touched targets — reject if the diff introduces new warnings. Approve only if all four hold; otherwise list concrete, actionable findings.`,
      {
        phase: 'Fix', label: `review:${item.id}:r${round}`, schema: REVIEW_SCHEMA,
        agentType: item.risk === 'physics' ? 'fbd-math-reviewer' : 'feature-dev:code-reviewer',
      },
    )
    if (lastReview && lastReview.approved) { approved = true; break }
    if (round === 1) {
      const rework = await agent(
        `Address these review findings for backlog item ${item.id} in ${ROOT} (do NOT commit): ${JSON.stringify(lastReview ? lastReview.findings : ['review agent died'])}. Then re-run "bash scripts/gate.sh" — must print GATE PASS. Keep changes minimal.`,
        { phase: 'Fix', label: `rework:${item.id}`, schema: FIX_SCHEMA },
      )
      if (!rework || rework.status !== 'fixed') break
    }
  }

  if (!approved) {
    results.push({ id: item.id, outcome: 'escalated', reviews: 2, notes: JSON.stringify(lastReview ? lastReview.findings : 'review/rework failed') })
    await agent(`In ${ROOT}: run git stash push -u -m "loop-${item.id}-escalated" so the batch stays clean; the user will inspect the stash. Confirm working tree clean.`,
      { phase: 'Fix', label: `stash:${item.id}` })
    continue
  }

  await agent(
    `In ${ROOT}: commit ALL current working-tree changes as ONE conventional commit for backlog item ${item.id} ("${item.title}"). Choose the type from the change (fix:/test:/docs:/refactor:), mention ${item.id} in the subject, one-sentence body, and end the message with the footer line:
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Do NOT push. Confirm with "git log -1 --stat".`,
    { phase: 'Fix', label: `commit:${item.id}` },
  )
  results.push({ id: item.id, outcome: 'fixed', reviews: lastReview && lastReview.approved ? 1 : 2 })
}

return { results }
```

(Design notes, mirroring the spec: items run strictly sequentially — the `for` loop is intentional; escalated/blocked work is `git stash`ed rather than hard-reset so nothing is destroyed but the batch branch stays clean; two review rounds then escalate.)

- [ ] **Step 2: Selftest**

Invoke Workflow with `{scriptPath: '.claude/workflows/fix-campaign.js', args: {selftest: true}}` → expect `{ ok: true }`. Then invoke with `{args: {}}` → expect the `error: 'pass args.items...'` return (validates the guard).

- [ ] **Step 3: Commit**

```bash
git add .claude/workflows/fix-campaign.js
git commit -m "feat(workflows): fix-campaign workflow (red test, gate, fresh review, per-item commit)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: docs/ai coordination updates

**Files:**
- Modify: `docs/ai/02-system.yaml` (known_limitations + current_statuses + lessons_learned)
- Modify: `docs/ai/05-update-tracker.md` (append entry)

**Interfaces:**
- Consumes: facts established by Tasks 1-7. Produces: docs/ai contract satisfied for this branch.

- [ ] **Step 1: Update `02-system.yaml`**

Remove the stale limitation (lines around 125-127):

```yaml
  - Forward dynamics doctests for linear_driver (src/core/linear_driver.rs
    line 7) fail to parse as Rust code. Pre-existing; not blocking.
```

In `current_statuses`, update the solver line's test count from `614 lib tests pass` to `716 lib tests pass`.

Append to `lessons_learned`:

```yaml
  - Verify "known broken" intel against the compiler before scheduling repair
    work — the tests/ struct-literal breakage was already fixed in b919485,
    but stale docs and an agent description kept reporting it for months.
  - Indented lines in /// or //! docs become rustdoc doctests when preceded
    by a blank doc line. Fence ASCII math as ```text.
```

- [ ] **Step 2: Append to `05-update-tracker.md`**

Add an entry (match the file's existing entry format, newest at top or bottom as the file shows):

```markdown
## 2026-08-08 — Agentic test-and-improve loop bootstrapped
- Phase 0: fixed linear_driver doctest (text fence) + 3 deny-level
  approx_constant clippy errors; cargo test --all and clippy now green.
- Added scripts/gate.sh (test + clippy + WASM check gate).
- Added docs/ai/backlog.yaml (9 seed items) as the loop queue;
  04-memory active_issues now points at it.
- Added .claude/workflows/{audit-campaign,gui-smoke,fix-campaign}.js.
- Spec: docs/superpowers/specs/2026-08-08-agentic-test-improve-loop-design.md;
  plan: docs/superpowers/plans/2026-08-08-agentic-test-improve-loop.md.
```

- [ ] **Step 3: Validate YAML still parses**

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/02-system.yaml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add docs/ai/02-system.yaml docs/ai/05-update-tracker.md
git commit -m "docs(ai): record loop bootstrap; retire stale doctest limitation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Run Phase 1 — audit, merge, user review (operational)

**Files:**
- Modify: `docs/ai/backlog.yaml` (merge confirmed findings)

**Interfaces:**
- Consumes: `audit-campaign.js` (Task 5), `gui-smoke.js` (Task 6), the user's braindump (collected in-session; if the user said "later", ask now — this step cannot start without explicitly offering the braindump opportunity again).
- Produces: user-approved backlog; batch 1 selection. Fix campaigns (Phase 2+) then run `fix-campaign.js` per batch — that is post-plan operations.

- [ ] **Step 1: USER GATE — request review of branch `agent/phase0-baseline`**

Present the branch's commits (Tasks 1-8) to the user for review and merge to `main` before the audit runs (the audit reads the corrected docs and gate script from the merged state; running it on the branch is acceptable if the user prefers to merge once at the end — ask).

- [ ] **Step 2: Collect braindump**

Ask the user for known/suspected issues (physics doubts, GUI friction, known-broken, wishlist-adjacent). Pass as `args.braindump` (array of strings). An empty braindump is allowed but must be the user's explicit choice.

- [ ] **Step 3: Run the audit**

Invoke Workflow: `{scriptPath: '.claude/workflows/audit-campaign.js', args: {braindump: [...]}}`. Also run the live GUI smoke (Task 6 Step 3 procedure) and fold any failure into the findings.

- [ ] **Step 4: Merge findings into the backlog**

For each `confirmed` finding: dedupe against existing items (same file + same symptom = same item; keep the better evidence), assign the next `BL-NNN` id, map severity→priority (1→1, 2→2, 3→3), status `open`. Record `dropped` findings count in the commit body for the audit trail. Validate YAML (Task 4 Step 3 command). Commit:

```bash
git add docs/ai/backlog.yaml
git commit -m "docs(ai): merge Phase 1 audit findings into backlog

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: USER GATE — backlog review + batch 1 selection**

Present the full backlog to the user: they kill/reprioritize/add items, then approve. Propose batch 1 = top 5-8 open items by priority, honoring the spec rule that `risk: physics` items get user-visible diffs at merge time. Plan complete when the user approves the backlog and batch 1.

---

## Deviations from spec (flagged for user review)

1. **Phase 0 content changed:** spec's `tests/` crate repair is a no-op (already fixed in `b919485`); replaced by the 3 deny-level clippy errors, which really do break `cargo clippy --all-targets`.
2. **GUI smoke v1 is load/canvas/console only** — not "all 30 samples render, 14 tabs switch" (spec success-criterion 4). Driving egui's canvas needs coordinate-clicking or an AccessKit bridge; investigating that is an audit-era backlog candidate, not a smoke-test freebie.
3. **Clippy "no new warnings" gate** is enforced by reviewer agents on touched files plus backlog item BL-003, not by a count baseline in `gate.sh` (duplicate-warning counts across targets are too noisy to diff reliably).
4. **Escalated items are `git stash`ed, not reset** — same clean-batch guarantee, but nothing is destroyed.
