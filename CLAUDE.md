Claude Code Prompt for Plan Mode #prompts

Review this plan thoroughly before making any code changes. For every issue or recommendation, explain the concrete tradeoffs, give me an opinionated recommendation, and ask for my input before assuming a direction.

My engineering preferences (use these to guide your recommendations):

﻿﻿DRY is important-flag repetition aggressively.
﻿﻿Well-tested code is non-negotiable; I'd rather have too many tests than too few.
﻿﻿I want code that's "engineered enough" - not under-engineered (fragile, hasky) and not over-engineered (premature abstraction, unnecessary complexity).
﻿﻿I err on the side of handling more edge cases, not fewer; thoughtfulness > speed.
﻿﻿Bias toward explicit over clever.
1. Architecture review

Evaluate:

﻿﻿Overall system design and component boundaries.
﻿﻿Dependency graph and coupling concerns.
﻿﻿Data flow patterns and potential bottlenecks.
﻿﻿Scaling characteristics and single points of failure.
﻿﻿Security architecture (auth, data access, API boundaries).
2. Code quality review

Evaluate:

﻿﻿Code organization and module structure.
﻿﻿DRY violations-be aggressive here.
﻿﻿Error handling patterns and missing edge cases (call these out explicitly).
﻿﻿Technical debt hotspots.
﻿﻿Areas that are over-engineered or under-engineered relative to my preferences.
3. Test review

Evaluate:

﻿﻿Test coverage gaps (unit, integration, e2e).
﻿﻿Test quality and assertion strength.
﻿﻿Missing edge case coverage-be thorough.
﻿﻿Untested failure modes and error paths.
4. Performance review

Evaluate:

﻿﻿N+1 queries and database access patterns.
﻿﻿Memory-usage concerns.
﻿﻿Caching opportunities.
﻿﻿Slow or high-complexity code paths.
For each issue you find

For every specific issue (bug, smell, design concern, or risk):

﻿﻿Describe the problem concretely, with file and line references.
﻿﻿Present 2-3 options, including "do nothing" where that's reasonable.
﻿﻿For each option, specify: implementation effort, risk, impact on other code, and maintenance burden.
﻿﻿Give me your recommended option and why, mapped to my preferences above.
﻿﻿Then explicitly ask whether I agree or want to choose a different direction before proceeding.
Workflow and interaction

﻿﻿Do not assume my priorities on timeline or scale.
﻿﻿After each section, pause and ask for my feedback before moving on.
BEFORE YOU START:

Ask if I want one of two options:

1/ BIG CHANGE: Work through this interactively, one section at a time (Architecture → Code Quality → Tests → Performance) with at most 4 top issues in each section.

2/ SMALL CHANGE: Work through interactively ONE question per review section

FOR EACH STAGE OF REVIEW: output the explanation and pros and cons of each stage's questions AND your opinionated recommendation and why, and then use AskUserQuestion. Also NUMBER issues and then give LETTERS for options and when using AskUserQuestion make sure each option clearly labels the issue NUMBER and option LETTER so the user doesn't get confused. Make the recommended option always the 1st option.

---

# Behavioral Guidelines

Reduce common LLM coding mistakes. Biased toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.
- Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Model Selection for Subagents and Workflows

This overrides the default of letting subagents inherit the session model. It applies whenever you dispatch a subagent (Agent tool) or write a workflow script, and always under ultracode.

- **Simple tasks run on `sonnet`** (the alias for the newest Sonnet). Simple means mechanical, well specified, low judgment:
  - exploring or mapping code; audit finders for the `tests`, `quality`, and `gui` dimensions;
  - implementing a plan task whose plan gives the exact code (transcription plus running tests);
  - the first attempt at a `risk: mechanical` backlog item in `fix-campaign` (red test, minimal fix, gate);
  - the per-item review of a `risk: mechanical` fix diff, first pass or re-review;
  - reproduce-it skeptics on non-physics findings;
  - doc and YAML edits, lint fixes, running commands and reporting output, GUI smoke runs.
- **Git housekeeping runs on `haiku` at `effort: 'low'`:** stash, cleanup, and scoped commit helpers.
- **Judgment-heavy work keeps the session model (omit `model`):**
  - physics, meaning every `risk: physics` backlog item (fix, rework, and review), `fbd-math-reviewer` passes, the `physics` finder dimension, all skeptic lenses on physics findings, and the magcoupling math audit;
  - architecture, design, and planning;
  - debugging without a known root cause, including braindump chase-downs;
  - feature or integration work that spans several files without exact code in a plan;
  - final whole-branch reviews.
- **Precedence:** when a task matches both lists, the session model wins.
- **Escalate, never downgrade.** A `sonnet` attempt has failed when it ends `blocked`, leaves the gate red, or is rejected in review. Every retry or rework of that task runs on the session model, as the `rework:` step in `fix-campaign.js` does. Escalate too when a task turns out mid-way to need judgment.
- **When unsure, use the session model.**
- **Make the choice visible in workflow scripts.** Set `model: 'sonnet'` or `model: 'haiku'` explicitly on every simple-task `agent()` call. For session-model calls, omit `model` on purpose and say why in a comment, such as `// session model: physics`. Never hard-code an Opus alias or version. For queues that mix risk levels, use the conditional spread in `.claude/workflows/fix-campaign.js` and `.claude/workflows/audit-campaign.js`: `...(item.risk === 'physics' ? {} : { model: 'sonnet' })`.
