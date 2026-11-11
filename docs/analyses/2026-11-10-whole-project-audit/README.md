# Whole-project audit, 2026-11-10

Read-only audit of main at 1048962: 15 finder agents by area, each finding re-checked by one of
8 skeptic agents. Both ran on Sonnet and ran no cargo or pytest command. Physics-suspect findings
were described, not judged.

Start with `../2026-11-10-whole-project-audit.md`: what was fixed, and the 168 findings left for
the owner, most important first.

- `INDEX.md`: one row per finding (id, title, the check's verdict, and what happened: fixed with
  its commit, left for the owner with the check's reason, or dropped).
- `<area>.md`: the finder's findings, with the files it did not read under "Not read".
  `<area>-verified.md`: the skeptic's check of each finding, with the lines it read.
- `prepared-fixes/`: what was applied and how.
  - `01` to `17`: the exact-replacement specs. All 17 are applied, in commits d6f5880 to c452531
    on the branch `audit/2026-11-10-whole-project`.
  - `status.json`: each fixed finding with its commit and what was left, and each finding held
    back although the check called it safe, with the reason.
  - `apply_edits.py <root> <spec> --write` applies a spec; it refuses a text that does not match
    the expected number of times. `fix_yaml_scalars.py <file> --write` is the BL-015 fix of
    docs/ai/02-system.yaml.
  - `make_audit_list.py` writes `INDEX.md`. `make_review_list.py` writes the review list and
    fails unless every finding is fixed, dropped or in exactly one of its sections.

27 findings are fixed, 168 are listed for the owner and 2 were dropped because the check could
not confirm them. `linkage-sim-rs/scripts/gate.sh` printed GATE PASS on the tree that holds all
the fixes together; it was not run again for each commit.