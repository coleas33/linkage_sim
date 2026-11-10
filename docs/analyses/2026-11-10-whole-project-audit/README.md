# Whole-project audit, 2026-11-10 (raw results, interim)

Read-only audit of main at 1048962: 15 finder agents by area, each re-checked by a skeptic agent.
Nothing has been fixed yet. This folder only preserves the results.

- `INDEX.md`: one row per finding (id, title, the check's verdict, status).
  `UNHANDLED` means the check called it a sure fix and it has not been applied yet.
- `<area>.md`: the finder's findings. `<area>-verified.md`: the skeptic's check of each.
- `prepared-fixes/`: exact-replacement specs (`01` to `10`) that were dry-run against this
  tree and matched, with the scripts that apply them (`apply_edits.py <root> <spec> --write`,
  `fix_yaml_scalars.py <file> --write` for docs/ai/02-system.yaml, BL-015). Not applied, not compiled.

Physics-suspect findings were described, not judged. Files the finders did not read are named
in each finder file's "Not read" section.
