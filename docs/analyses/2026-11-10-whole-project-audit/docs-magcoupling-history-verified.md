# Verified: docs-magcoupling-history

## DMH-01: README (and Cargo.toml comments) still say linkage-sim-rs depends on magcoupling-rs and that a gate checks lock agreement
- Verdict: CONFIRMED
- Checked: magcoupling-rs/README.md:166-167 ("`linkage-sim-rs` depends on it by path (feature `gui`)..."), :174 ("the linkage app in an `egui::Window` (M5)"), :601 ("M5's linkage window") against :635-636 and :685-689 ("no longer depends on this crate"). Search of linkage-sim-rs/Cargo.toml for "magcoupl": 0 hits. linkage-sim-rs/scripts/gate.sh:110-113 fails if it ever does. Gate 11 (gate.sh:123-132) compares only wasm-bindgen with deploy-web.yml's CLI pin; no egui, egui_plot or rust_xlsxwriter comparison. magcoupling-rs/Cargo.toml:57-60 ("gate.sh checks that both lock files agree"), :63 ("gate 11 checks it"), :76-77 ("gate 11 checks that both lock files agree") all promise a check that does not exist. Also the anchor `#in-the-linkage-app-m5` at README:167 no longer matches the heading `### In the linkage app` (README:683).
- Safe to apply without judgment: no. The Cargo.toml comments and README text are only prose, but README is read by include_str! tests (tests/explain.rs:1779-1860, gui/guidance.rs:6718), so the README edit needs a check that those pinned phrases are untouched (they are not on lines 166-174), and the replacement wording is a choice. The Cargo.toml comment edits alone cannot change behaviour.
- Exact change: n/a

## DMH-02: README status line lists BL-072 as open although it is fixed and shipped
- Verdict: CONFIRMED
- Checked: README.md:46 ends "BL-065 to BL-072, BL-074 and BL-075 are open in `docs/ai/backlog.yaml`." docs/ai/backlog.yaml BL-072 has `status: fixed`. README:29 and :124 (E24 row) record it shipped. The pinned README claims in tests/explain.rs (dashboard rows, checks, notes, steps, guides count) do not include this phrase; `git grep 'BL-065 to BL-07'` hits README:46 and two plan files (historical).
- Safe to apply without judgment: yes. Prose in one doc line; no test, script or document reads that phrase.
- Exact change: magcoupling-rs/README.md, line 46, old text "BL-065 to BL-072, BL-074 and BL-075 are open in `docs/ai/backlog.yaml`." new text "BL-065 to BL-071, BL-074 and BL-075 are open in `docs/ai/backlog.yaml`."

## DMH-03: 17 plan and spec files are named by no other tracked file (1.76 MB), and the whole history set is 15.4 MB
- Verdict: PARTLY. The "named by no other tracked file" fact holds for all 17 (git grep -l -F of each file stem: only the file itself, or for the 8 with a design/plan pair only the plan naming its own design; no hit in README, docs/ai, code or tests). Sizes of the 17 re-summed from single-file stems match in order of magnitude (payload-weights 709,471; slice-1 557,916; dxf design 9,541 exact). I did not re-add the 1.76 MB total or the 15.4 MB set total. The 2026-10-02 m5-embed plan is named by docs/ai/04-memory.yaml, backlog.yaml and README, so it is referenced, only superseded.
- Checked: git grep per stem as above; README:685-690.
- Safe to apply without judgment: no. Owner decision on removing records; backlog and memory name the m5 plan.
- Exact change: n/a

## DMH-04: Update tracker is 313 kB in 110 entries; six entries hold 38 percent, and entries are split per task
- Verdict: CONFIRMED
- Checked: docs/ai/05-update-tracker.md is 313,402 bytes, 3129 lines, 110 `## ` headings. Entries at lines 52, 17, 29, 41, 78, 8 measure 64.8, 18.5, 14.9, 11.1, 7.3, 5.2 kB (finder's figures within 0.5 percent; byte counting differs slightly). Line 27 is 10,012 characters. Order break: headings 2177 (2026-04-29), 2231 (2026-04-30), 2260 (2026-04-30); 1993 (2026-04-30) sits above 2063 (2026-04-29). Header says "Reverse chronological (newest at top)". "Earlier history" pointer at 3125-3129 exists. I did not re-verify the duplicate or BL-status cross-check.
- Safe to apply without judgment: no. Trimming or moving history is an owner call.
- Exact change: n/a

## DMH-05: Update tracker, backlog and memory cite records under the gitignored .superpowers/ that no clone has
- Verdict: CONFIRMED (count slightly off: the tracker has 10 mentions, not 9; the tenth is at line 1687)
- Checked: .gitignore:56 `.superpowers/`; `git ls-files .superpowers` empty (0); the audit worktree has no .superpowers folder (Test-Path False). Mentions of `.superpowers/` per file: 05-update-tracker.md 10 (lines 18,27,30,38,39,42,49,74,76,1687), backlog.yaml 4, 04-memory.yaml 1. backlog BL-064 notes itself says "owner-decisions.md, gitignored" for one path.
- Safe to apply without judgment: no. Needs a choice between a header note and committing the review files.
- Exact change: n/a

## DMH-06: set_keyboard_shortcuts exists for a host that no longer exists
- Verdict: CONFIRMED
- Checked: `git grep set_keyboard_shortcuts` in .rs: only panel.rs (docs :21, :363, :870; setter :608-609; test calls :4315, :4322); README:601 mentions it; app.rs never calls it; linkage-sim-rs has no use (it no longer depends on the crate). Field read at panel.rs:873. Plan files also mention it (historical). The `pub` setter is public API, so removal depends on whether other egui hosts are wanted (README:174 says any egui app can host the panel).
- Safe to apply without judgment: no. Removing a pub API and a test.
- Exact change: n/a

## Seen while checking
- README:167 anchor `#in-the-linkage-app-m5` is dead (heading is now "In the linkage app"); folded into DMH-01.
