"""Write the owner's review list of the 2026-11-10 audit.

Usage: python -I make_review_list.py <findings_dir> <status.json> <out.md>

The hand-written parts live here; titles and verdicts come from the skeptics' `-verified.md` files.
Exits 1 unless every finding id is in the fixed table, in exactly one review section, or dropped.
"""
import json
import re
import sys
from pathlib import Path

findings, status, out = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), Path(sys.argv[3])
fixed, held = status["fixed"], status["held"]
HEAD = re.compile(r"^## ([A-Z]+-\d+)\s*:\s*(.*)$")
title, verdict, area = {}, {}, {}
for f in sorted(findings.glob("*-verified.md")):
    cur = None
    for line in f.read_text(encoding="utf-8").splitlines():
        m = HEAD.match(line)
        if m:
            cur = m.group(1)
            title[cur], area[cur] = m.group(2).strip().rstrip("."), f.name[: -len("-verified.md")]
        elif cur and line.startswith("- Verdict") and cur not in verdict:
            verdict[cur] = line.split(":", 1)[1].strip()
dropped = sorted(i for i, v in verdict.items() if re.match(r"(REFUTED|NOT CONFIRMED|NOT A)", v, re.I))

S = "linkage-sim-rs/src/"
# (id, where, what a user sees, what the check did not prove, suggested first step)
A = [
    ("LGE-01", S + "gui/sweep/mod.rs:788-802, 955-957, 1334-1353",
     "A joint's reaction series is created at the first sample that solves. When early samples fail, the series is shorter than the angle axis, so every later reaction is plotted and exported against the wrong angle.",
     "", "pad a new series with NaN up to the current sample, with a test whose first sample fails."),
    ("LGE-02", S + "gui/sweep/mod.rs:626-630, 1252-1258",
     "A statics solve that fails is stored as a driver torque of 0.0 in the forward sweep and as NaN in trajectory mode. A failed sample therefore reads as \"no torque needed\" in the plots, the envelopes and the actuator force. The code's own comment at 743-744 calls the 0 fake.",
     "", "store NaN in both modes."),
    ("LGS-02", S + "gui/mod.rs:1022-1029, 1065-1073",
     "Autosave recovery deletes the autosave file even when loading it failed, on the desktop and in the browser. The only copy of the work is gone and no message is shown.",
     "", "delete the autosave only after a load that succeeded, and show the error."),
    ("LGS-12", S + "gui/state/file_io.rs:302-362",
     "A load that fails has already replaced the sweep mode, the sensors, the motion profile, the rated force, the duration, the parametric settings and the blueprint. The app is left half old and half new.",
     "", "build the new mechanism first and assign the fields only when it built."),
    ("LFA-05", S + "forces/elements/force_element.rs:164-209; io/from_json.rs:320-325",
     "A file whose force names a point on a body that does not exist crashes the loader (a map index that panics inside a function that returns a Result). The validator that would catch it, `validate_force_elements`, is called only by tests.",
     "No such file was loaded to see the crash.", "return an error from the lookup and call the validator on load."),
    ("LGS-17", S + "gui/state/entity_crud.rs:277-303",
     "Removing a body keeps the linear drivers that point at it. The next rebuild then fails on a body that is not there.",
     "The same sentence about force point references was not checked.", "drop the linear drivers of a removed body, as joints, drivers and forces already are."),
    ("LGS-16", S + "gui/state/undo_ops.rs:35-92",
     "Undo and redo restore the mechanism but skip two refreshes that a rebuild and a load make: the driver's display offset and the stroke range.",
     "", "call the same refresh steps in one shared function."),
    ("LGE-04", S + "gui/export/raster.rs:102-152",
     "GIF export skips every frame that fails and still returns success, so it can write an empty GIF. It also divides by the driver speed with no guard against zero.",
     "", "count the frames written and return an error when none were."),
    ("LGS-10", S + "gui/menu_bar.rs:139-228; gui/mod.rs:70-72, 1261-1284",
     "On the desktop a failed open, save or recovery is written to the log and nothing is shown. A save that failed looks like a save that worked.",
     "", "put the error in the status line, where the success message already goes."),
    ("LGS-11", S + "gui/state/file_io.rs:127-138, 252-269; gui/state/mod.rs:939-972",
     "The desktop autosave is never cleaned up, an autosave beside a saved file is never offered for recovery, and File > New brings the recovery prompt back.",
     "Found by reading the code, not by running it.", "decide the autosave's life cycle once (where it lives, when it is offered, when it is deleted)."),
    ("LGE-07", S + "gui/export/report.rs:150, 164, 383, 401, 474; svg.rs:147, 221; dxf.rs:48",
     "Body, joint, trace and sensor names go into the HTML report, the SVG and the report's script without escaping. The names come from the mechanism file, so a file from someone else can put markup or script into a report you export. The two escaping helpers that exist disagree with each other.",
     "No name with a quote mark was tried.", "one escaping helper, used at every place a name is written."),
    ("LGC-16", S + "gui/property_panel/health.rs:180-184, 473-488",
     "The Health panel prints toggle angles in the body frame while the plots show the display frame, so the two disagree by the display offset. In stroke mode it prints metres with a degree sign.",
     "", "shift and label the values the way the plots do."),
    ("LFA-07", S + "forces/compound.rs:187-253; io/from_json.rs:355-360",
     "When a compound cylinder cannot be expanded, the two bodies it added first stay in the mechanism with no joints.",
     "Found by reading; the effect on the solve was not checked.", "check the references before adding the bodies, or remove the bodies on failure."),
    ("LFA-01", S + "io/to_json.rs:213-227; io/from_json.rs:301-313",
     "A cosine-stroke linear driver is saved in a form that loads back as a constant-velocity driver. Nothing outside the tests builds a cosine driver today, so the app cannot reach this yet.",
     "", "your decision: delete the cosine driver or finish its save and load."),
    ("LGS-01", S + "gui/mod.rs:103-110",
     "Drag-and-drop import reads only the dropped file's bytes. The finder says the desktop build receives a path and no bytes, so a drop does nothing there.",
     "The check could not prove what the desktop build receives; it needs one try on the desktop app.", "try a drop on the desktop build, then read the path when there are no bytes."),
    ("LGE-14", S + "gui/samples/fourbar.rs:578-579; gui/samples/special.rs:1033-1034, 1083-1084",
     "Two samples ignore the driver joint that was asked for, and the Custom 6-bar can hand back a start position that did not solve, with only a log line.",
     "", "return an error where the joint cannot be honoured."),
    ("LGC-12", S + "gui/property_panel/pending_edits.rs:239-251",
     "Renaming a link makes no undo step and does not mark the file as changed.",
     "The finder's third point does not hold: the sweep is right to recompute, because the weight breakdown shows the names.", "push an undo step when the edit ends."),
    ("LGS-03", S + "gui/mod.rs:1166-1167, 1255-1285",
     "The shortcut help lists Ctrl+S and Ctrl+Shift+S in the browser build, where nothing handles them.",
     "", "hide the two rows in the browser build."),
    ("LCS-13", S + "solver/inverse_kinematics/control_target.rs:64, 124, 217",
     "A control target with a zero direction stops the program with an assert. The type can be read from a file.",
     "Whether a saved file can reach it was not traced.", "return an error in place of the assert."),
    ("LFA-18", S + "analysis/grashof.rs:74, 87",
     "`check_grashof` stops the program on a link length that is not a number.",
     "Not proven that such a length can reach it.", "compare as `motor_sizing.rs:89` does."),
    ("LFA-12", S + "analysis/virtual_work.rs:70-90",
     "The virtual-work check says \"Mechanism must be built before solving\" when the mechanism is built and simply has no driver.",
     "", "a message that names the missing driver (it needs a new error variant)."),
]
B = [
    ("LCS-02", S + "solver/forward_dynamics.rs:364-397, 563-571",
     "Forward dynamics integrates first and then projects every tenth stored sample back onto the constraints, without writing the projected state back. The drift it reports is measured after the projection, so it can read smaller than the drift the simulation had."),
    ("LCS-06", S + "solver/statics.rs:126-133",
     "The reaction of a prismatic joint is read as a force [Fx, Fy] from its two multipliers. The joint's two equations are a normal offset and a relative angle, so the second multiplier would be a moment, not a force."),
    ("LCS-14", S + "solver/inverse_kinematics/solver.rs:142-144, 185, 263-281",
     "`solve_for_target` reports \"SVD solve failed\" when no probe sample converged, and its branch-jump threshold is a rule of thumb taken from the bodies' centres of gravity (the code calls it \"heuristic only\")."),
    ("LFA-16", S + "forces/elements/evaluation.rs:241-246",
     "At full compression the gas spring's column is floored at 1e-10 m, so the force ratio is the stroke divided by 1e-10: about 2e9 times the nominal force for a 0.2 m stroke. The element has no end stop."),
    ("LFA-17", S + "forces/elements/evaluation.rs:363-368; analysis/motor_sizing.rs:61-68",
     "The motor element gives full stall torque at any reverse speed. The motor-sizing check uses the speed's size and so derates in both directions. The two disagree whenever the motor is driven backwards."),
    ("LGE-03", S + "gui/sweep/motion_profile.rs:155-163; gui/state/blueprint_ops.rs:1405-1433",
     "The motion profile reads the sweep's axis as absolute degrees over a full turn and is applied to every sweep, including a custom range and a stroke sweep, where the axis holds metres."),
]
# (title, intro, [id or (id, note) or (id, "rest", note)])
REST = [
    ("C. Tests that can pass without checking anything", "The repository's rule is more tests, not fewer, so none of these was changed. Making them strict may turn some red.", [
        ("INF-14", "`linkage-sim-rs/tests/property_tests.rs:134-178`, `golden_fixtures.rs:1012-1045`. First step: count the cases that reached an assertion and assert the count."),
        "PY-08", "PY-03", "LCS-16",
        ("LGS-14", "They leave \"test\", \"roundtrip\", \"a\", \"b\" and \"only\" in `~/.linkage-sim/templates/`, and \"only\" is never removed. `gui/state/templates.rs:96-112`."),
        "MT-07", "INF-07", "INF-02", "MT-06", "MEX-06", "MGP-11", "INF-11", "PY-07", "LGE-20",
        ("INF-15", "rest", "the property test `expression_driver_matches_constant_speed` exercises the `meval` crate and no code of this repository."),
    ]),
    ("D. Build, deploy, gate and local tooling", "", [
        ("INF-05", "The larger point sits in the same finding: the deploy job runs no test and no gate before `vercel deploy --prod`, so a push to main publishes whatever compiles. Pinning a toolchain changes every local build. Both are yours."),
        ("INF-06", "This is why the audit could not run as the repository's own campaign in a worktree. Both files hold your uncommitted edits, so they were not touched."),
        "INF-04", "INF-01", "INF-16", "LCS-01", "MEC-02", "DMH-01",
        ("INF-08", "A local, git-ignored file in your main checkout; not touched."),
        ("INF-09", "A local, git-ignored file in your main checkout; not touched."),
        ("INF-10", "Ready to apply on your word: the check gives the exact replacement text. The two files are git-ignored and exist only in your main checkout, so they were not touched."),
        ("DOC-13", "The agent files, the fbd-derive skill and the hooks exist only in your checkout, because `.gitignore` ignores `.claude/*` except the workflows. A clone cannot follow the rules that name them."),
        ("DOC-15", "CLAUDE.md is your own prompt text, so it was not touched."),
        ("INF-17", "Left as it is: the commented-out rules also record that the golden fixtures and the output GIFs are tracked on purpose, so deleting them is a wording choice, not a defect."),
        ("MGV-08", "Ready to apply on your word (the check gives the exact change). Harmless as it is, and some keep such lists explicit on purpose."),
    ]),
    ("E. Dead or unfinished code and files: delete or finish", "Each of these is something nothing uses. Whether it is rubbish or the start of something you still want is yours to say. Where the check gives the exact deletion it is marked ready.", [
        "LGS-07",
        "LGS-08",
        ("LGC-01", "Ready to apply on your word (the check gives the exact deletion). Held because each reads as a feature that was started: delete it or wire it up."),
        ("LGE-06", "Ready to apply on your word: delete the helper, or use it in the three places."),
        ("MGG-06", "Ready to apply on your word (`TermColors::is_empty`, `texts`). Marking them `#[cfg(test)]` shrinks the library's public surface."),
        ("MGV-03", "Ready to apply on your word (`Cell::as_text`, `package_label`). Moving them into the test modules shrinks the public surface."),
        "LCS-03",
        ("LCS-12", "rest", "`State::set_angle` and `JointConstraint::is_cam_follower` have no caller, and `rtol`/`atol` and the local-frame reactions are used only by tests."),
        ("LFA-03", "rest", "`JointJson::RevoluteDriver` is a legacy variant the canvas and blueprint code still match; old files may hold it."),
        "LFA-10",
        ("LGC-02", "rest", "the `SegmentHit` allow."),
        "LGE-05", "LGE-08", "LGE-09", "LGE-19", "LGS-13",
        ("LGS-19", "rest", "`update_prismatic_axis`, `move_attachment_point` and `sync_active_load_case`."),
        "DMH-06", "MEC-04", "MEX-04", "MEX-05", "MEX-07", "MEX-08", "MEX-09", "MGG-01", "MGG-07", "MGP-01", "MGP-08", "MGV-02",
        "PY-01", "PY-02",
        ("PY-10", "Removing them from the tree does not shrink the history."),
        "MT-03", "MT-08", "MT-09", "DMH-03", "DMH-04", "DMH-05",
    ]),
    ("F. Documents that need rewriting, not a line fixed", "The sure line fixes are in the fixed table. What is left needs new wording or a decision on what the document is for.", [
        ("DOC-07", "The largest of these."),
        "DOC-12", "DOC-01", "DOC-02", "DOC-03", "DOC-06", "DOC-10", "DOC-14", "DOC-17", "LGC-09", "MGG-08",
        ("DOC-04", "rest", "the force-margin and mobile-support lines of \"Planned / Future\"."),
        ("DOC-08", "rest", "the benchmark suite's format and planned sections."),
        ("DOC-11", "rest", "the modules `docs/ai/03-structure.yaml` does not list."),
        ("LFA-11", "rest", "two dated comments in `io/mod.rs` (lines 6 and 10) and a test module's name."),
        ("MEX-03", "rest", "the same stale sentence in `docs/ai/03-structure.yaml:49` and `magcoupling-rs/tests/assumptions.rs:6` and `:268`."),
        ("MGP-05", "rest", "`gui/panel.rs:1342` and line 33 of the README."),
        ("MT-05", "rest", "`gen_differential.py:35`, in the vendored oracle."),
    ]),
    ("G. Things that behave or read differently in two places", "", [
        "LGC-07", "LGC-17", "LGC-11", "LGC-20", "LGC-14", "LGC-08", "LGC-13", "LGE-10", "LGE-16", "LGE-17", "LGE-18",
        ("LCS-07", "rest", "the cam profile named \"Spline\" is piecewise linear; its guards and an `if let Ok` disagree with that."),
        "LCS-15", "LFA-09", "LGS-18", "MGP-06", "MGP-10", "MGV-01", "MGV-04", "PY-09",
    ]),
    ("H. Speed", "Nothing here was measured. Each is a place where the code does the same work more than once.", [
        "LCS-08", "LGC-18", "LGC-05", "LGS-20", "LCS-11", "LCS-09",
        ("LCS-10", "rest", "an expression driver parses its formula again on every evaluation."),
        "LFA-08", "MEX-01", "MEC-03", "MGG-02", "MGP-02", "MGP-03", "MGV-06", "MT-02",
    ]),
    ("I. The same code written more than once", "You asked for repetition to be flagged hard. None of these was merged: each merge is a refactor, and a refactor is a judgment.", [
        ("LGS-04", "More than repetition: the copies at `gui/state/mod.rs:1300` and `gui/state/file_io.rs:408, 423` divide by the driver speed without the zero guard the other copies have."),
        "LGC-10", "LGC-06", "LGS-15", "LGS-09", "LGE-15", "LGE-12", "LGE-13", "LFA-06", "LFA-04", "LFA-14", "LFA-15", "LFA-13",
        "LCS-05", "LGC-03", "LGC-04", "LGC-19", "LGS-05", "INF-12", "INF-13", "PY-06", "PY-04", "PY-05",
        ("MGV-05", "Ready to apply on your word: shared constants (the check's exact change) or a status enum. The strings are also the Python oracle's."),
        "MEX-10", "MEC-01",
        ("MGG-03", "rest", "`gui/guidance.rs:199` and `:1815`, whose constants live in another module or function."),
        "MGG-04", "MGG-05", "MGG-09", "MGG-10", "MGP-04", "MGP-09", "MGV-07", "MT-01",
    ]),
]

AREAS = [("DOC", "docs-living"), ("DMH", "docs-magcoupling-history"), ("LCS", "linkage-core-solver"),
         ("LFA", "linkage-forces-analysis-io"), ("LGC", "linkage-gui-canvas-panels"), ("LGE", "linkage-gui-export"),
         ("LGS", "linkage-gui-state"), ("MEC", "mag-engine-calc"), ("MEX", "mag-engine-explain"),
         ("MGG", "mag-gui-guidance"), ("MGP", "mag-gui-panel"), ("MGV", "mag-gui-views"), ("PY", "python"),
         ("INF", "tests-linkage-infra"), ("MT", "tests-magcoupling")]

seen, rest_seen, problems = [], [], []


def partly(fid):
    return " The check confirmed this only in part." if verdict[fid].upper().startswith("PARTLY") else ""


def item(entry):
    if isinstance(entry, str):
        seen.append(entry)
        return f"- **{entry}** {title[entry]}.{partly(entry)}"
    if len(entry) == 3:
        rest_seen.append(entry[0])
        return f"- **{entry[0]}** (the part not fixed) {entry[2]}"
    seen.append(entry[0])
    return f"- **{entry[0]}** {title[entry[0]]}.{partly(entry[0])} {entry[1]}"


L = []
w = L.append
w("# Whole-project audit, 2026-11-10: what was fixed and what is yours to decide")
w("")
w("You asked: \"Read this entire project and find everything that's broken, inconsistent, inefficient, or no longer needed. Fix what you're sure about and list the rest for me to review.\" This is that list.")
w("")
w("## In short")
w("")
w(f"- {len(verdict)} findings. {len(fixed)} are fixed, some only in their sure part. {len(verdict) - len(fixed) - len(dropped)} are listed below for you. {len(dropped)} were dropped because the check could not confirm them.")
w("- The fixes are 9 commits on the branch `audit/2026-11-10-whole-project` (worktree `C:\\Users\\Cole\\source\\repos\\lsim-audit`), cut from main at 1048962. Nothing is merged and nothing is pushed.")
w("- The gate (`linkage-sim-rs/scripts/gate.sh`, 13 gates) printed GATE PASS on the tree that holds all the fixes together. It was not run again for each commit.")
w("- No physics was changed and no physics was judged. Six findings are marked physics-suspect (section B).")
w("- What I would look at first: the five in section A that lose work or show a wrong number (LGE-01, LGE-02, LGS-02, LGS-12, LFA-05); the deploy that runs no test (INF-05, section D); the six in section B, which need a physics review before anything is changed; and the seven marked \"ready to apply on your word\".")
w("")
w("## How it was done, and its limits")
w("")
w("- 15 finder agents each read one area of the tree. 8 skeptic agents then checked every finding against the code. Both ran on Sonnet and were told to change nothing and to run no cargo or pytest command.")
w("- Most findings were confirmed by reading the code, not by running it. Where a finding says a crash or a wrong value, nobody has reproduced it unless the line says so.")
w("- A fix was applied only where the skeptic gave an exact change and called it safe without judgment, and where I agreed after reading it. Anything public, anything a test calls, anything that changes what a user sees and every deletion that might be wanted was left for you (the register `docs/analyses/2026-11-10-conservative-assumptions.md` in the main checkout, row A-16).")
w("- Not read by any finder: `.claude/settings.local.json` (it may hold secrets), build output, and the files each finder names under \"Not read\".")
w("- Not touched: CLAUDE.md, AGENTS.md and the two campaign workflows, which hold your uncommitted edits, and the git-ignored files under `.claude/` in your main checkout.")
w("")
w("## Where the detail is")
w("")
w("Every finding has a section with the lines the check read, in `docs/analyses/2026-11-10-whole-project-audit/<area>-verified.md`, and the finder's longer text in `<area>.md`. `INDEX.md` in that folder has one row per finding.")
w("")
w("| ids | area file |")
w("|---|---|")
for pre, name in AREAS:
    w(f"| {pre}-nn | `{name}` |")
w("")
w(f"## Fixed ({len(fixed)} findings, 9 commits)")
w("")
w("| id | commit | what changed, and what was left |")
w("|---|---|---|")
for fid in sorted(fixed, key=lambda i: (area[i], i)):
    w(f"| {fid} | `{fixed[fid]['commit']}` | {fixed[fid]['what'].replace('|', '/')} |")
w("")
w(f"## For your review ({len(verdict) - len(fixed) - len(dropped)} findings), most important first")
w("")
w(f"### A. A wrong result, lost work or a crash that a user can reach ({len(A)})")
w("")
w("All in the linkage simulator. In the order I would fix them. Each needs a test that fails first, so none was fixed here.")
w("")
for n, (fid, where, what, unproven, first) in enumerate(A, 1):
    seen.append(fid)
    w(f"{n}. **{fid}** {what}{' ' + unproven if unproven else ''} Where: `{where}`. First step: {first}")
w("")
w(f"### B. Physics-suspect: described, not judged ({len(B)})")
w("")
w("The skeptics confirmed that the code does what each line says. Whether that is wrong physics was not judged by anyone. By the repository's rule a physics item needs the physics review on Fable and your review of its diff, so nothing here was changed.")
w("")
for n, (fid, where, what) in enumerate(B, 1):
    seen.append(fid)
    w(f"{n}. **{fid}** {what} Where: `{where}`.")
w("")
for name, intro, entries in REST:
    w(f"### {name} ({sum(1 for e in entries if not (isinstance(e, tuple) and len(e) == 3))})")
    w("")
    if intro:
        w(intro)
        w("")
    for e in entries:
        w(item(e))
    w("")
w("### J. From the baseline tools, not from a finder")
w("")
w("- `cargo fmt --check`: 129 files of linkage-sim-rs are not rustfmt-clean (magcoupling-rs is clean). One formatting commit would touch them all, so it should land when no other branch is open.")
w("- `cargo clippy --all-targets` on linkage-sim-rs: 259 warnings in the library, 203 of them auto-fixable (backlog BL-003). Gate 2 does not fail on them (INF-02).")
w("- The gate runs no `cargo fmt --check`, no pytest and no mypy. Run by hand on 2026-11-10: pytest 1088 passed, 2 skipped; mypy strict clean on 54 files.")
w("- `cargo tree -d` on linkage-sim-rs: 212 lines of duplicate dependency versions, and `nom v1.2.4` will be rejected by a future Rust.")
w("- Not installed, so not run: ruff, cargo-machete, cargo-udeps. The unused-dependency findings (LCS-01, INF-16) rest on reading alone.")
w("- Paths in living documents that exist nowhere and were not corrected, because each needs a choice of what to say: `.claude/workflows/gui-smoke.js:102` (`src/web/events.rs`), `AGENTS.md:7`, `docs/ai/03-structure.yaml:29`, `docs/ai/04-memory.yaml:93`, `docs/architecture/ARCHITECTURE.md:265`, `EXTENSIBILITY.md:178`, `VALIDATION.md:135`, `docs/guides/WASM_DEPLOYMENT.md:56, 58`, `magcoupling-rs/README.md:734`.")
w("- Git: 11 local branches are fully merged into main and can go. Two Codex worktrees are prunable; they were left alone by your rule. AGENTS.md opens with a stale \"Open for Codex\" section, and `docs/ai/01-meta.yaml` names an old focus.")
w("")
w(f"### Dropped ({len(dropped)})")
w("")
for fid in dropped:
    w(f"- **{fid}** {title[fid]}. The check's verdict: {verdict[fid]}")
w("")
w("## What needs your yes")
w("")
w("- Merging this branch. A trial merge with main is clean. A trial merge with `magcoupling/off-default-fixes` conflicts in one file, `docs/ai/05-update-tracker.md`, where both branches add an entry at the top (keep both); every other file merges by itself, and the five `docs/ai` YAML files load in the merged tree. The merged tree was not built or tested.")
w("- Pushing anything. Pushing main deploys the website, and the deploy runs no test (INF-05).")
w("- Any item above. Tell me the ids, or a whole section, and I will fix them on this branch with a failing test first.")
w("")

every = set(verdict)
listed = every - set(fixed) - set(dropped)
for fid in sorted(listed - set(seen)):
    problems.append(f"not in any section: {fid}")
for fid in sorted(set(seen) - listed):
    problems.append(f"in a section but fixed, dropped or unknown: {fid}")
for fid in sorted({i for i in seen if seen.count(i) > 1}):
    problems.append(f"in more than one section: {fid}")
for fid in sorted(set(rest_seen) - set(fixed)):
    problems.append(f"marked as the unfixed part of a finding that is not fixed: {fid}")
for fid in sorted(set(held) - set(seen)):
    problems.append(f"held but not in a section: {fid}")
out.write_text("\n".join(L), encoding="utf-8", newline="\n")
print(f"{len(every)} findings: {len(fixed)} fixed, {len(set(seen))} in sections, {len(dropped)} dropped; {len(L)} lines")
if problems:
    sys.exit("\n".join(problems))