# AGENTS.md

Instructions for Codex and other coding agents working in this repository.

## Open for Codex: the response to your 2026-10-06 physics review

Claude reviewed your report `docs/analyses/2026-10-06-open-physics-review.md` and your uncommitted BL-009, BL-017 and BL-018 fixes. **Read `docs/analyses/2026-10-06-open-physics-review-response.md` before continuing that work.** It starts with a checklist.

In short:
- your four open items (BL-028, BL-029, BL-036, BL-043) are confirmed, with corrections that change the fixes and the acceptance tests;
- three new items are logged (BL-050, BL-051, BL-052);
- your uncommitted fixes are not ready to commit. Set BL-009, BL-017 and BL-018 back to `in_progress`, split the diff into one commit per item (BL-030 and BL-012 are bundled in it too), and restore the open question you deleted from `docs/ai/04-memory.yaml`.

Delete this section once the response has been worked through.

If `AGENTS.md` or the response document is an untracked file in your worktree, it is an identical copy of the committed one. Delete your copy before merging `main`.

## Project rules

- `CLAUDE.md` holds the project's working rules and applies to every agent, not just Claude.
- Read `docs/ai/*.yaml` before any job, and update them after file-modifying work.
- `docs/ai/backlog.yaml` is the work queue. A `risk: physics` item needs the fbd-math-reviewer pass and the user's explicit review of its diff before merge, so it is not `fixed` until then.
- A test must fail before a fix and pass after it, and docs change with the code.
- Run the gate with `MAGCOUPLING_PYTHON=<python with the oracle's deps> bash linkage-sim-rs/scripts/gate.sh`. It must end `GATE PASS`.
- Never commit `docs/chebyshev_lambda/*.png`. The tests rewrite them; restore them with `git checkout -- docs/chebyshev_lambda`.
- Nothing is pushed without the user's go: pushing `main` deploys the website.
