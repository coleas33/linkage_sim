# Audit findings: docs-magcoupling-history
Scope read: magcoupling-rs/README.md (all 929 lines skimmed; closely sampled: status 1-61, invalid inputs, differences table, layout, features, panel 178-300 and 354-366, versions and binaries 600-682, tests table 693-734 (first 700 chars of each row), regenerating test data, equation explorer, porting, translation rules, deviations; every backticked identifier (693) and every backticked file path checked against the code, section counts for grades, materials, warnings, notes, assumptions, guides, gate numbers, BL statuses checked); docs/ai/05-update-tracker.md structure (headings, order, sizes, duplicates, BL status claims); name-reference and size scan of all 77 files in docs/superpowers/plans, specs and docs/analyses (content not reviewed).

## DMH-01: README (and Cargo.toml comments) still say linkage-sim-rs depends on magcoupling-rs and that a gate checks lock agreement
- Kind: inconsistent
- Where: magcoupling-rs/README.md:166-167, :174 (and :601, :721 test text "the linkage Tools window's 1100 x 700"); magcoupling-rs/Cargo.toml:60, :63, :76 (comments)
- Evidence: README:166-167 "`linkage-sim-rs` depends on it by path (feature `gui`) for its Tools -> Magnetic coupling window"; README:174 "the linkage app in an `egui::Window` (M5)"; README:688-689 and 635-636 say the opposite: "`linkage-sim-rs` no longer depends on this crate", "egui and eframe no longer have to match `linkage-sim-rs`". linkage-sim-rs/Cargo.toml has no `magcoupl` match; gate.sh:110 fails if it ever does. Cargo.toml:63 "Cargo.lock pins linkage-sim-rs's 0.33.0 and gate 11 checks it"; gate.sh gate 11 (lines 123-132) only compares wasm-bindgen with deploy-web.yml's CLI pin, never egui, egui_plot or rust_xlsxwriter.
- Why it matters: One file tells a reader two opposite things, and three Cargo.toml comments promise a gate check that does not exist, so drift in egui_plot or rust_xlsxwriter between the crates would be believed guarded.
- Suggested action: Rewrite README:166-167 and :174 to the standalone-site state (point to the "In the linkage app" section); correct the three Cargo.toml comments (lines 60, 63, 76) to name only the wasm-bindgen check.
- Fix class: mechanical-safe
- Confidence: certain

## DMH-02: README status line lists BL-072 as open although it is fixed and shipped
- Kind: inconsistent
- Where: magcoupling-rs/README.md:46
- Evidence: README:46 "BL-065 to BL-072, BL-074 and BL-075 are open in `docs/ai/backlog.yaml`". backlog.yaml: BL-072 status fixed; README:29 and :124 (the E24 row) record BL-072 as shipped; git log 532f6a0 "E24, the cure margin from the lower ring's own onset (BL-072)". BL-073 is also fixed and correctly not listed.
- Why it matters: The status paragraph sends a reader to open work that is done.
- Suggested action: Change to "BL-065 to BL-071, BL-074 and BL-075".
- Fix class: mechanical-safe
- Confidence: certain

## DMH-03: 17 plan and spec files are named by no other tracked file (1.76 MB), and the whole history set is 15.4 MB
- Kind: not needed
- Where: docs/superpowers/plans/ 2026-03-17-crank-selection, 2026-03-18-animation-driver-selection, 2026-03-18-coupler-trace-multi, 2026-03-18-multi-pivot-body-editor, 2026-03-18-phase5-gui-mvp, 2026-03-20-gui-bugfix-ux-rework, 2026-03-21-click-to-place-forces, 2026-03-21-compound-force-bodies, 2026-03-21-force-mount-points, 2026-03-23-force-zones-labels-body-geometry, 2026-03-24-actuator-stroke-limits, 2026-03-29-actuator-driven-linkages, 2026-03-29-mounting-angle, 2026-03-30-mobile-friendly (14 files, 487,919 bytes together), 2026-09-29-payload-weights-track2.md (709,471), 2026-10-13-magcoupler-guidance-slice-1.md (557,916); docs/superpowers/specs/2026-04-09-dxf-import-design.md (9,541)
- Evidence: `git grep -l -F <file stem>` outside the file itself gives 0 hits for each of the 17 (the other 66 files in docs/superpowers and docs/analyses have 1 to 21 hits). Totals by folder: plans 11,282,016 bytes (20 files over 200 kB, three over 900 kB), specs 1,106,359 (of which the guidance-2 research folder, 9 files, 427,729, is named only by its design, synthesis and plan), analyses 3,022,673. The features the 17 describe all exist in code (for example gui/dxf_import.rs). The March plans' designs are named only by their own plan; no plan names its design back from any living doc.
- Why it matters: These are records of finished work that nothing points to; they add 1.76 MB (and the whole set 15.4 MB) to a tree whose largest tracked files they are, and every grep over docs/ drowns in them. Not load-bearing: tests/explain.rs:1544 and gui/explorer.rs:2845 read analyses/2026-10-14-magcoupler-notes-review.md and -signoff.md, tests read analyses/2026-09-30-magcoupling-addendum-a-data.json, and notes.rs cites sections of help-review.md and notes-review.md, so those analyses must stay.
- Suggested action: Move the 17 (and, if wanted, the March designs they belong to) under docs/history/ or drop them from the tree (git keeps them); do not touch the analyses that tests and the app cite. The 2026-10-02-magcoupling-m5-embed.md plan (215,569 bytes) describes the linkage-window embed that decision U-5 and the 2026-10-06 url-move plan removed (README:685-690), so it is superseded by that plan and can go in the same move.
- Fix class: owner-decision
- Confidence: certain on "named by nothing"; the owner decides whether records stay in the tree

## DMH-04: Update tracker is 313 kB in 110 entries; six entries hold 38 percent, and entries are split per task
- Kind: inefficient
- Where: docs/ai/05-update-tracker.md (entries at lines 52 (64,495 bytes), 17 (18,380), 29 (14,791), 41 (11,005), 78 (7,297), 8 (5,219)); per-task runs at 691-1079 (Payload weights Task 1 to 10, 10 entries) and 2026-09-29 to 2026-10-01 (M2, A-1 to A-3, M4-1 to M4-3, each with a separate "final review fix wave" entry)
- Evidence: parsed `## ` headings: 110 entries, 305,684 bytes of entries; the six newest magcoupler entries (2026-10-13 to 2026-11-04) are 117 kB, the slice-2 entry alone 64,495 bytes; one line (line 27) is 10,005 characters. Two spots break the header's "Reverse chronological (newest at top)": line 2231 (2026-04-30) follows 2026-04-29 entries, and 1993 (2026-04-30) sits above 2063 (2026-04-29) so the same day is split in two places. No duplicated entries (no two share a title; only two boilerplate lines repeat, 329/364 and 486/544). I found no entry that claims a BL item fixed that the backlog lists as open, or the reverse (checked every "BL-nnn ... fixed/shipped/open" phrase against backlog.yaml: 26 fixed, 49 open).
- Why it matters: The file is meant to tell a future session what changed; at 313 kB it cannot be read whole, and its newest entries restate what the README (same facts, same counts), backlog notes, plans and specs already hold.
- Suggested action: Cap each entry to what the README and backlog do not hold, or move entries older than the current milestone to docs/history/ and leave the "Earlier history" pointer at its end (lines 3125-3129) updated; fix the two out-of-order days.
- Fix class: owner-decision
- Confidence: certain on the sizes and the ordering; whether to trim is the owner's call

## DMH-05: Update tracker, backlog and memory cite records under the gitignored .superpowers/ that no clone has
- Kind: broken
- Where: docs/ai/05-update-tracker.md:18, 27, 30, 38, 39, 42, 49, 74, 76 (9 mentions, e.g. `.superpowers/sdd/2026-10-19-magcoupler-slice3/final-review.md`, `.../timing-T10.txt`, `.../smoke-local.json`); docs/ai/backlog.yaml (4 mentions); docs/ai/04-memory.yaml (1)
- Evidence: .gitignore has the line `.superpowers/`; `git ls-files .superpowers` is empty; the audit worktree has no such folder. The tracker cites final-review.md as the record of the whole-branch review passing for slice 3, BL-064, BL-059 and slice 2.
- Why it matters: The review records the entries point to cannot be opened from a fresh clone or worktree, so the "passed after its fix wave" claims cannot be checked from the repository.
- Suggested action: Either say once in the tracker's header that `.superpowers/sdd/` is a local, ungit-tracked workspace, or commit the four final-review.md files under docs/analyses/.
- Fix class: needs-judgment
- Confidence: certain that the paths are absent from the repository; likely the owner's main checkout has them

## DMH-06: set_keyboard_shortcuts exists for a host that no longer exists
- Kind: not needed
- Where: magcoupling-rs/src/gui/panel.rs:364, 445, 608-609, 873 (field and setter); README:601
- Evidence: README:601 "A host with its own undo can keep the keys (`set_keyboard_shortcuts(false)`: M5's linkage window)". Grep over the worktree finds callers only in panel.rs tests (4315, 4322); the standalone app (app.rs) never calls it; the linkage app embedding it was removed 2026-10-06 (README:685-690).
- Why it matters: A public setter and a branch in the shortcut handler serve only one removed host and its own test.
- Suggested action: Leave it if the panel is meant to stay hostable by other egui apps (README:174 says so); otherwise remove the field, setter, the `!self.keyboard_shortcuts` test and the README clause.
- Fix class: owner-decision
- Confidence: likely; settled by whether the owner wants the panel embeddable again

## Leads settled
- leads.md section 1, `magcoupling-rs` [dev-dependencies] `magcoupling-rs`: not real. It is the self dev-dependency that turns on `workbook-parity` for `cargo test`; Cargo.toml comments and README:668-681 explain it and gate 10 guards it.
- leads.md section 1, `wasm-bindgen-futures` (linkage-sim-rs) and `uuid`: outside my part, not settled.
- leads.md sections 2, 3, 4, 6: no candidate falls in my part (README, plans, specs, analyses, tracker).
- leads.md section 5 (files named nowhere): none of the listed files is in my part; I ran the same test on my part by name stem and report the result as DMH-03.

## Not read
- Plan, spec and analysis contents (by instruction); README rows of the tests table past their first 700 characters, README lines 300-353, 367-599 (the panel bullets) read only by identifier and count checks, not line by line; docs/ai/05-update-tracker.md entry bodies beyond the top five entries and the headings.
- Files not opened at all: 77 plan, spec, analysis and research files (all by instruction); count of files not read for the purpose of reading is 0 beyond those.
