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
    { phase: 'Fix', label: `fix:${item.id}`, schema: FIX_SCHEMA, ...(item.risk === 'physics' ? {} : { model: 'sonnet' }) },
  )

  if (!fix || fix.status !== 'fixed') {
    const outcome = fix ? fix.status : 'agent-error'
    results.push({ id: item.id, outcome, notes: fix ? fix.notes : 'agent died' })
    await agent(`In ${ROOT}: if "git status --porcelain" shows changes, run: git stash push -u -m "loop-${item.id}-${outcome}". Confirm the working tree is clean afterward and report what you stashed.`,
      { phase: 'Fix', label: `clean:${item.id}`, model: 'haiku', effort: 'low' })
    continue
  }

  let changedFiles = fix.files_changed || []
  let approved = false
  let lastReview = null
  let roundsUsed = 0
  for (let round = 1; round <= 2 && !approved; round++) {
    roundsUsed = round
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
        { phase: 'Fix', label: `rework:${item.id}`, schema: FIX_SCHEMA, ...(item.risk === 'physics' ? {} : { model: 'sonnet' }) },
      )
      if (!rework || rework.status !== 'fixed') break
      changedFiles = Array.from(new Set([...changedFiles, ...(rework.files_changed || [])]))
    }
  }

  if (!approved) {
    results.push({ id: item.id, outcome: 'escalated', reviews: roundsUsed, notes: JSON.stringify(lastReview ? lastReview.findings : 'review/rework failed') })
    await agent(`In ${ROOT}: run git stash push -u -m "loop-${item.id}-escalated" so the batch stays clean; the user will inspect the stash. Confirm working tree clean.`,
      { phase: 'Fix', label: `stash:${item.id}`, model: 'haiku', effort: 'low' })
    continue
  }

  await agent(
    `In ${ROOT}: stage EXACTLY these files (repo-relative paths) and commit them as ONE conventional commit for backlog item ${item.id} ("${item.title}"): ${JSON.stringify(changedFiles)}
For each listed file, run: git add -- <file>. Do not use "git add -A" or "git add .". Choose the commit type from the change (fix:/test:/docs:/refactor:), mention ${item.id} in the subject, one-sentence body, and end the message with the footer line:
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
NEVER stage docs/chebyshev_lambda/*.png or any file not in this list; if the list is empty, run git status, report it, and do NOT commit.
Do NOT push. Confirm with "git log -1 --stat".`,
    { phase: 'Fix', label: `commit:${item.id}`, model: 'haiku', effort: 'low' },
  )
  results.push({ id: item.id, outcome: 'fixed', reviews: roundsUsed })
}

return { results }
