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

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) {
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

const selected = (Array.isArray(ARGS.dimensions) ? ARGS.dimensions : Object.keys(DIMS))
  .filter(k => DIMS[k])
  .map(k => ({ key: k, prompt: DIMS[k] }))

const braindump = ARGS.braindump || []
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
