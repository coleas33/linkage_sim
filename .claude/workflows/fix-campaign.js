export const meta = {
  name: 'fix-campaign',
  description: 'Sequentially fix backlog items: red test, fix, gate, fresh-context review, per-item commit',
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

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const items = ARGS.items || []
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
