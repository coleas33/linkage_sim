# Magcoupler UX overhaul, slice A (first arrival, part 1): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The calculator opens on a starting design that passes its nine checks (Required torque 1.35 N·m), keeps the spreadsheet's design one button away, and writes the design into the page's address so a reload or a bookmark brings it back. All of it is on today's screen.

**Architecture:**
- **A session baseline in the panel.** `MagcouplingPanel::with_baseline(inputs)` opens on a design and keeps it as the baseline for the session. Six on-screen surfaces compare with the baseline instead of the spreadsheet's values: the changed dot, a row's reset, the tooltip's "Default:" line, Reset all, the Equation panel's "changed" tag and the status line's name. `MagcouplingPanel::new()` keeps the spreadsheet's values as its baseline, so a panel built the old way behaves as today.
- **Two named designs.** The app owns the starting design (`app::starting_design`: `DesignInputs::default()` with six inputs set) and hands it to the panel. The status line reads "Starting design", "Example design" or "Your design". A header button, "Open the example design", opens the spreadsheet's values as one undo step.
- **The address follows the settled design.** Each frame the web app compares the panel's settled design (the undo history's current state) with the design the address stands for. When they differ it rewrites the address with `history.replaceState`: the clean address at the starting design, the Share link for any other design. There is one encoding, `session::share_link`.
- **Unchanged:** the engine, `DesignInputs::default`, the file and link format, every parity fixture, the layout and the look (the design's call C-1).

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, web-sys 0.3 (this plan adds its `History` feature); the headless egui harness (`magcoupling-rs/src/gui/test_support.rs` and the panel tests' `Harness`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. Slice A is its section 7, tests 1 to 5 of section 8.1, the text rules of section 8.3 and the docs and release rules of section 9. Executors read the spec's sections 1.3, 2.3, 7, 8.1, 8.3 and 9 beside this plan. Inventory: `docs/analyses/2026-11-09-magcoupler-ux-inventory.md`. Backlog item: **BL-077** (`docs/ai/backlog.yaml`). Open questions Q-1, Q-3, Q-4 and Q-5 stand on their recommendations (`docs/analyses/2026-11-10-conservative-assumptions.md`, rows A-2 to A-5).

**How this plan was made.** The slice was built once as a prototype on a copy of `main` at 7c2dc8d. Every Replace block below was then applied, in this plan's order, to a fresh copy of 7c2dc8d. The result equals the prototype byte for byte and passes `cargo test` with and without the app feature (1,029 lib tests pass with it, 25 of them new), both native clippy runs, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (75 checks). Each red step's "Expected" below is what that replay printed. Not run for this plan: gates 1 to 3 (linkage-sim-rs, which this slice does not touch), gate 12 (the Python oracle), the release WASM build and the browser. Task 8 runs them.

## Global Constraints

- **Today's screen (C-1).** No new look, no layout change. The one visible addition to the header is the button of Task 2.
- **No engine number moves** (the design's section 2.3). No file under `magcoupling-rs/src/engine/` or `magcoupling-rs/tests/` changes. `DesignInputs::default` is unchanged. Never edit `reference/`, `magcoupling-rs/tests/data/` or a Python-locked row. No fixture is re-blessed.
- **A panel built with `MagcouplingPanel::new()` behaves as today** (the design's test 2). Where a function gains a baseline argument, the existing test's call is edited and its assertion is not. Each task lists those calls. If an existing assertion fails, stop: do not change it.
- **The baseline never moves** after the panel is built: not by an undo, a redo, a file, a link, Reset all or the example design. It is in no design file, link or undo step.
- **Nothing is stored between visits (C-8).** The address is the only memory. Share always copies the full link (C-23).
- **Text rules (the design's section 8.3)** for every new on-screen string: it passes `plain_text_violations`; it does not say the coupling protects the gearbox; it has none of "sealed", "containment", "leak", "clamp", or a ring named by a colour; it carries no typed-in number. Every new string is a named constant, and tests read the constants. The clamp section stays paused.
- **The Replace blocks.** Each block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them.
- **TDD.** Red, then green. A red step that calls an API the next step adds fails to compile; that is its red, and the plan says so. Two tests pin code this slice does not change; they pass at once and the plan says how to see them bite.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1: it adds a BOM and treats backticks as escapes. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Docs ship with the code** (repository rule): each task updates the README text it makes stale. Task 7 sweeps `docs/ai/`.
- **Release.** Nothing is merged or pushed before the owner's review of the diff (BL-077 is `risk: physics`; the design's section 9.4). A push to `main` deploys.

## Model per task

CLAUDE.md section 5, and the controller's instruction for this plan: transcription from exact code runs on `sonnet`; every task that states or checks a physics claim, and every review, runs on `fable`; git housekeeping runs on `haiku` at low effort.

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The session baseline in the panel | `sonnet` |
| 2 | Two named designs: the status line and the example design's button | `sonnet` |
| 3 | The starting design, and the app opens on it | `fable` (it states the design's physics claims) |
| 4 | The design in the address bar | `sonnet` |
| 5 | gui-smoke: the clean address and the reload | `sonnet` |
| 6 | Text and physics review of the new strings and claims | `fable` (fresh context) |
| 7 | Docs sweep | `sonnet` |
| 8 | Release: whole-branch review, full gate, gui-smoke, STOP | `fable` controller; `haiku` for git |

A `sonnet` task that ends blocked, leaves a test red or is rejected in review is redone on `fable` (escalate, never downgrade). Each task's diff is reviewed by a fresh-context `fable` reviewer before the next task starts.

## Review Focus

Five inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named, except the last, which no headless test can reach.

1. **A design file or link that names only some inputs** (an old file, a hand-made one, the smoke run's own file). A person expects it to open the design it always opened. The inputs it leaves out take the spreadsheet's values, not the baseline's. Task 3: `a_file_that_names_one_input_reads_the_rest_from_the_spreadsheet_not_the_baseline`.
2. **A 1280-pixel window.** The new header button pushes "Show spreadsheet details" onto a second header row (measured in the prototype). A person expects the first view to keep its open torque card and its drawing. Task 3: `the_app_opens_the_starting_design` asserts the card's caption is drawn at 1280 x 800.
3. **Auto on at the starting design's inputs.** That is a different design from the starting design, so the address must carry it and a reload must bring Auto back. Task 4: `the_address_is_clean_at_the_starting_design_and_a_share_link_elsewhere` (its `auto_on` case).
4. **An edit in progress:** a held arrow key, a drag, a value being typed. A person expects the address to stay still until the edit ends, and Safari refuses more than 100 writes in 30 seconds. Task 4: `the_address_follows_settled_edits_undo_and_redo` and `the_settled_design_follows_settled_edits_undo_redo_and_an_opened_link`.
5. **A browser that refuses the write.** The code ignores a refused `replaceState` and writes again at the next settled edit. Only the browser smoke run exercises the call (Task 5, `address_follows`), and only in Chromium. Safari and Firefox are not verified by this plan; Task 8 says so on the STOP sheet.

## Decisions this plan takes

Each is a reading of the design where it was open, or a finding. The owner may overrule any of them; Task 8 lists them on the STOP sheet.

| Id | Decision | Why |
|---|---|---|
| P-1 | The starting design is `starting_design()` in `magcoupling-rs/src/app.rs`. | The design: "written once in the app's code; where exactly is the plan's". |
| P-2 | The baseline is a `DesignInputs`. Reset all returns its inputs with the default sizing state (Auto off). | Reset all turns Auto off today. Auto on is part of a design, not of "the default". |
| P-3 | The panel exposes `settled_design() -> &Design` (the undo history's current state) in place of the design's boolean "is settled" accessor. | The history already holds exactly the last settled design, after undo and redo too. The app compares two designs by reference each frame and clones only when it writes. A boolean would need a new field and a clone per frame. |
| P-4 | "Open the example design" is drawn only on a panel whose baseline is not the spreadsheet's design. | On a panel built with `new()` Reset all already opens it, and the design's test 2 keeps that panel as it is today. |
| P-5 | `guidance::status_line(status, inputs, baseline)` is added beside `design_status_text`. `DesignStatus` and `design_status` keep their signatures. | The roughly 40 existing test calls of `design_status` stay untouched; five calls of `dashboard_ui` and `status_ui` gain an argument. |
| P-6 | The Equation panel's "changed" tag: the registry (`engine::explain`) still compares with the spreadsheet's values, and the GUI sets the flag for input terms from the baseline. `explorer_ui` and `equation_body` take the baseline as an eighth argument, each with a narrow `#[allow(clippy::too_many_arguments)]`. | No engine line changes. The alternative, the baseline inside `NoteView`, edits nine test literals for the same allow on `explorer_ui`. |
| P-7 | The button's hover is the example design's existing hover text. Its status line is "Example design opened". Reset all still says "Default design restored". | The design: the words of the surfaces that follow the baseline do not change. One new status string. |
| P-8 | The address is clean only when the inputs are the starting design's and Auto is off. | Auto on is a different design (Review Focus 3). |
| P-9 | A link opened at start-up is not rewritten until the first edit, even where its text differs from the link the app would write for the same design. | The design's section 7.2: the address changes after a settled edit. |
| P-10 | **Finding.** At the starting design the Size tab draws one amber line: "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." The design did not know of it: its checks were read from the dashboard's functions, and this line is drawn by the Size tab (`view_ui`). The approved six values are kept, the line stays visible, and a test pins it. | The conservative answer (owner ruling of 2026-11-10): nothing is hidden and nothing approved is moved. A ring radius of 9.26 mm with a 42.2 mm cap thread clears the line and takes the cap skirt from 0.55 to 0.30 mm. Task 3 adds register row A-17. |
| P-11 | The smoke run's `address_follows` needs the drag. A drag that did not hold the handle now fails an ordinary run through that check. | The design's test 5: the smoke run "reloads the page after an edit and reads the design back". Without the edit the address was not checked. |
| P-12 | BL-077 is `risk: physics`. | The slice shows a new design and its numbers on first paint and says it passes every check. So the owner reviews the diff before the merge. |
| P-13 | The nine findings of the design's section 12.4 (the short link form of Q-5 among them) are not filed by this plan. One new finding is filed, as BL-078 in Task 7: the keyway line's rounding (at a ring radius of 9.25 mm the wall computes as 2.4999999999999990 mm, so the line still shows and reads "2.50 mm"). It is not fixed here. | The design's section 12.4: each of the nine is for the owner to accept or strike; Task 8 asks. Section 9.1: a new finding is filed, never fixed in passing. |
| P-14 | At 1280 pixels the header takes a second row. | C-1: slice A is on today's screen; slice B moves the button into the Design menu. The first view keeps its open card (Review Focus 2). |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/panel.rs` | The `baseline` field, `with_baseline`, `reset` to the baseline, the row's default from the baseline, `open_example_design`, the header button, `settled_design`, `share_base`; tests | 1, 2, 4 |
| `magcoupling-rs/src/gui/input_ui.rs` | `input_row` takes the default it shows | 1 |
| `magcoupling-rs/src/gui/inputs.rs` | `input_tooltip` takes the default its line shows; tests | 1 |
| `magcoupling-rs/src/gui/explorer.rs` | `explorer_ui` takes the baseline for the "changed" tag | 1 |
| `magcoupling-rs/src/gui/guidance.rs` | `STARTING_DESIGN`, `STARTING_DESIGN_HOVER`, `status_line`; the example design's hover loses "starting"; tests | 2 |
| `magcoupling-rs/src/gui/dashboard.rs` | `dashboard_ui` takes the baseline for the status line; tests | 1, 2 |
| `magcoupling-rs/src/gui/mod.rs` | Re-exports `OPEN_EXAMPLE` | 2 |
| `magcoupling-rs/src/gui/history.rs` | `History::committed`; test | 4 |
| `magcoupling-rs/src/app.rs` | `starting_design`, `address_of`, `replace_address`, the app's `address_design`; tests | 3, 4, 5 |
| `magcoupling-rs/Cargo.toml` | web-sys feature `History` | 4 |
| `.claude/workflows/gui-smoke.js` | The `starting_design` and `address_follows` checks | 5 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | The two checks in its list | 5 |
| `magcoupling-rs/README.md` | The baseline, the status line, the starting design, the address, the smoke run, the test rows | 1 to 5, 7 |
| `docs/analyses/2026-11-10-conservative-assumptions.md` | Row A-17 | 3 |
| `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` | The keyway finding in 7.1; the status and backlog lines of the header | 3, 7 |
| `docs/analyses/2026-11-17-magcoupler-slice-a-review.md` | New: the review record | 6 |
| `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml` | The docs sweep | 7, 8 |

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` (at or after the commit that holds this plan), create the branch `magcoupling/overhaul-a-first-arrival` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  ```

  Expected at 7c2dc8d: `test result: ok. 1004 passed; 0 failed; 6 ignored`. If `main` has moved and the count differs, note the new count: the counts in the tasks below shift by the same amount.

- [ ] **Step 3: Check the base.** `git log --oneline -1 -- magcoupling-rs/src` shows `e105f84` at 7c2dc8d (the last commit under `magcoupling-rs/src`). If it shows a later commit, a file this plan edits may have changed and a Replace block may no longer match once: the task that meets a mismatch stops and reports.

## Tasks

### Task 1: The session baseline in the panel

**Model:** `sonnet` (transcription; no physics claim).

Three behaviours, each with its own red and green: (A) a panel opens on its baseline and Reset all returns to it; (B) the changed dot, a row's reset and the tooltip's Default line follow it; (C) the Equation panel's "changed" tag follows it.

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/input_ui.rs`, `magcoupling-rs/src/gui/inputs.rs`, `magcoupling-rs/src/gui/explorer.rs`, `magcoupling-rs/src/gui/dashboard.rs` (one test call), `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces:
  - `MagcouplingPanel::with_baseline(baseline: DesignInputs) -> Self` (public). `MagcouplingPanel::new()` is `with_baseline(DesignInputs::default())`.
  - The private field `baseline: DesignInputs` of `MagcouplingPanel`.
  - `MagcouplingPanel::reset(&mut self)` returns to the baseline's inputs with `SizingState::default()`.
  - `input_ui::input_row(ui: &mut egui::Ui, entry: &InputEntry, current: &Value, default: &Value, seed: Option<f64>, locked: bool, show_details: bool) -> RowOutput`.
  - `inputs::input_tooltip(entry: &InputEntry, default: &Value, show_details: bool) -> String`.
  - `explorer::explorer_ui(ui, explorer, inputs, results, baseline: &DesignInputs, show_details, held, clamps_paused)`.
  - In the panel's test module: `Harness::with_baseline(baseline: DesignInputs, size: egui::Vec2) -> Harness` and `fn design_at(inputs: &DesignInputs) -> Design`.

**Existing test calls this task edits (the assertions stay):** `input_tooltip` in `inputs.rs` (`tooltips_carry_help_path_cell_range_and_default`, five calls through one local closure; `tooltips_name_the_path_cell_and_original_note_only_with_the_spreadsheet_details`, two; `a_tooltip_opens_with_the_help_the_overlay_gives`, one), in `dashboard.rs` (`every_hover_text_with_the_details_off_is_in_plain_words`) and in `panel.rs` (`an_input_row_s_hover_names_its_path_and_cell_only_with_the_spreadsheet_details`, `every_shown_text`); `explorer_ui` in `explorer.rs` (`swatch_colors`, `the_panel_s_pause_decides_the_drawn_used_by_links`, `hover_in_panel`, `the_equation_panel_s_value_line_names_the_cup_wall_input_by_its_label`). Each passes the spreadsheet's values (`&entry.default`, `&DesignInputs::default()`), so each test reads as before.

- [ ] **Step 1 (A, red): the harness for a baseline, and the three tests.** The first block adds `passing_design` to the test module's imports; the second turns the harness's `on_screen` into `with_baseline` plus a one-line `on_screen` (the lines that follow the old text become the tail of `with_baseline`); the rest add a helper and three tests.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        only_the_joint_fails, primary_button, row_line, select_all, short_magnets, sized_frame,
        sized_frame_at, smoke_inputs, text_center, text_clip_rect, text_rect, text_rects,
        text_shapes, tooltips_at_once, with_poles,
```

with:

```rust
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
        sized_frame, sized_frame_at, smoke_inputs, text_center, text_clip_rect, text_rect,
        text_rects, text_shapes, tooltips_at_once, with_poles,
```

Replace:

```rust
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
            let mut panel = MagcouplingPanel::new();
```

with:

```rust
        pub(crate) fn on_screen(size: egui::Vec2) -> Self {
            Self::with_baseline(DesignInputs::default(), size)
        }

        /// [`Harness::on_screen`] for a panel whose baseline is `baseline` (UX overhaul spec
        /// 7.1): the panel opens on it.
        fn with_baseline(baseline: DesignInputs, size: egui::Vec2) -> Self {
            let mut panel = MagcouplingPanel::with_baseline(baseline);
```

Replace:

```rust
    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

with:

```rust
    /// `inputs` as a design with the default sizing state: what a panel whose baseline is
    /// `inputs` opens on and Reset all returns to.
    fn design_at(inputs: &DesignInputs) -> Design {
        Design {
            inputs: inputs.clone(),
            sizing: SizingState::default(),
        }
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

Replace:

```rust
    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

with:

```rust
    #[test]
    fn a_new_panel_keeps_the_spreadsheet_s_values_as_its_baseline() {
        // UX overhaul spec 7.1 (C-21): a host or a test that builds the panel the old way sees
        // the old behaviour.
        assert_eq!(MagcouplingPanel::new().baseline, DesignInputs::default());
        assert_eq!(
            MagcouplingPanel::default().baseline,
            DesignInputs::default()
        );
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

Replace:

```rust
    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

with:

```rust
    #[test]
    fn a_panel_opens_on_its_baseline_and_reset_all_returns_to_it() {
        let baseline = passing_design();
        let start = design_at(&baseline);
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        assert_eq!(harness.panel.design(), start);
        assert_eq!(harness.panel.results(), &compute_all(&baseline));
        // The session starts there: nothing to undo.
        assert!(!harness.panel.history.can_undo(&start));
        // Auto on is part of the design, so Reset all turns it off again.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        assert_ne!(harness.panel.design(), start);
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), start);
        assert_eq!(harness.panel.results(), &compute_all(&baseline));
        // Reset all is an edit like any other: Undo brings the changed design back.
        harness.click_text(UNDO);
        assert_ne!(harness.panel.design(), start);
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

Replace:

```rust
    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

with:

```rust
    #[test]
    fn nothing_moves_the_baseline() {
        // UX overhaul spec 7.1: not an undo, a redo, a file, a link, a link opened as the
        // session's start or Reset all.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        harness.frame(Vec::new());
        harness
            .panel
            .load_share_payload(&encode_share_payload(&gap_design(1.5)))
            .unwrap();
        harness.frame(Vec::new());
        harness.click_text(UNDO);
        harness.click_text(REDO);
        harness.click_text(RESET_ALL);
        harness
            .panel
            .open_share_payload(&encode_share_payload(&Design::default()))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.baseline, baseline);
        assert_eq!(harness.panel.design(), Design::default());
        harness.panel.reset();
        assert_eq!(harness.panel.design(), design_at(&baseline));
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib baseline`
  Expected: it does not compile. `error[E0599]: no function or associated item named 'with_baseline' found for struct 'gui::panel::MagcouplingPanel'` and three `error[E0609]: no field 'baseline'`.

- [ ] **Step 3 (A, green): the field, the constructor and Reset all.**

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    /// The sizing mode and free variable (the target is the requirement input, decision G-2).
    sizing: SizingState,
```

with:

```rust
    /// The session's baseline (UX overhaul spec 7.1, C-21 and C-22): what "the default" means
    /// on screen (the changed dot, a row's reset, a tooltip's Default line, Reset all, the
    /// Equation panel's changed tags, the status line's name). Set when the panel is built and
    /// never moved: not by an undo, a file, a link or the example design. In no design file,
    /// link or undo step.
    baseline: DesignInputs,
    /// The sizing mode and free variable (the target is the requirement input, decision G-2).
    sizing: SizingState,
```

Replace:

```rust
    /// A panel at the default design.
    pub fn new() -> Self {
```

with:

```rust
    /// A panel at the default design (the spreadsheet's values), which is also its baseline.
    pub fn new() -> Self {
        Self::with_baseline(DesignInputs::default())
    }

    /// A panel that opens on `baseline` and keeps it as its baseline for the session (UX
    /// overhaul spec 7.1): the standalone app hands it the starting design.
    pub fn with_baseline(baseline: DesignInputs) -> Self {
```

Replace:

```rust
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        Self {
            inputs,
            sizing: SizingState::default(),
            runner: SizingRunner::default(),
            history: History::new(Design::default()),
```

with:

```rust
        let inputs = baseline.clone();
        let results = compute_all(&inputs);
        let design = Design {
            inputs: baseline.clone(),
            sizing: SizingState::default(),
        };
        Self {
            inputs,
            baseline,
            sizing: SizingState::default(),
            runner: SizingRunner::default(),
            history: History::new(design),
```

Replace:

```rust
    /// Back to the default design (inputs and sizing state); it can be undone.
    pub fn reset(&mut self) {
        self.replace_design(Design::default());
```

with:

```rust
    /// Back to the baseline (its inputs, with the default sizing state); it can be undone.
    pub fn reset(&mut self) {
        self.replace_design(Design {
            inputs: self.baseline.clone(),
            sizing: SizingState::default(),
        });
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- baseline reset_all`
  Expected: PASS, 6 tests (the three new ones, `reset_all_can_be_undone`, `reset_all_restores_the_default_design_and_headline`, `reset_all_and_a_load_keep_the_card_s_default_with_no_user_choice`).

- [ ] **Step 5 (B, red): the dot, the reset and the Default line.**

**`magcoupling-rs/src/gui/inputs.rs`**

Replace:

```rust
    #[test]
    fn tooltips_name_the_path_cell_and_original_note_only_with_the_spreadsheet_details() {
```

with:

```rust
    #[test]
    fn the_default_line_shows_the_default_it_is_handed() {
        // UX overhaul spec 7.1 (C-22): the panel hands the baseline's value; each arm of the
        // line reads it, not the catalogue's default.
        let catalogue = InputCatalogue::new();
        let line = |path: &str, default: Value| {
            let tooltip = input_tooltip(catalogue.entry(path).unwrap(), &default, false);
            tooltip
                .lines()
                .find(|line| line.starts_with("Default: "))
                .unwrap_or_else(|| panic!("{path}: no Default line in {tooltip}"))
                .to_owned()
        };
        assert_eq!(line("metal.face_gap_mm", Value::Num(1.8)), "Default: 1.800");
        assert_eq!(line("coupling.npole", Value::Int(8)), "Default: 8");
        assert_eq!(
            line("coupling.backiron", Value::Int(0)),
            "Default: 0 = no back iron"
        );
        assert_eq!(
            line("metal.measured_drag_Nm", Value::Num(0.05)),
            "Default: 0.05000"
        );
        assert_eq!(
            line("coupling.magnets.axial_length_mm", Value::None),
            "Default: blank"
        );
        assert_eq!(
            line("coupling.magnets.part_inner", Value::Text(String::new())),
            "Default: blank"
        );
    }

    #[test]
    fn tooltips_name_the_path_cell_and_original_note_only_with_the_spreadsheet_details() {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn the_axial_length_override_starts_blank_and_enters_at_the_ring_length() {
```

with:

```rust
    #[test]
    fn the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline() {
        // UX overhaul spec 7.1 (C-22): on screen "the default" is the baseline's value.
        use crate::gui::input_ui::TRACE_HINT;
        use crate::gui::inputs::input_tooltip;
        // The passing test design: its face gap is 1.8 mm, the spreadsheet's 1.4 mm.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        tooltips_at_once(&harness.ctx);
        // At the baseline no row carries the dot, though it differs from the spreadsheet's
        // values in two rows of the Key design group (the face gap and the cup wall).
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 0);
        assert_eq!(count(&output, RESET_LABEL), 0);
        // The label's hover gives the baseline's value as the default.
        let entry = InputCatalogue::get().entry(FACE_GAP).unwrap();
        let tooltip = input_tooltip(entry, &Value::Num(1.8), false);
        assert!(tooltip.ends_with("Default: 1.800"), "{tooltip}");
        let label = crate::gui::plain_text::label_for(FACE_GAP);
        let at = text_center(&output, label);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let want = format!("{tooltip}\n{TRACE_HINT}");
        assert!(drawn_texts(&output).contains(&want), "{want:?}");
        // One step off the baseline: the dot and the reset, whose hover names the baseline's
        // value; the reset returns it, not the spreadsheet's.
        harness.focus(FACE_GAP);
        harness.frame(key_tap(egui::Key::ArrowRight));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 1);
        assert_eq!(count(&output, RESET_LABEL), 1);
        let at = text_center(&output, RESET_LABEL);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Back to the default: 1.800"), 1);
        harness.click_text(RESET_LABEL);
        assert_eq!(harness.panel.inputs(), &baseline);
        assert_eq!(count(&harness.frame(Vec::new()), CHANGED_DOT), 0);
        // At the spreadsheet's values both rows carry the dot: they differ from the baseline.
        harness.panel.inputs = DesignInputs::default();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CHANGED_DOT), 2);
        assert_eq!(count(&output, RESET_LABEL), 2);
    }

    #[test]
    fn the_axial_length_override_starts_blank_and_enters_at_the_ring_length() {
```

- [ ] **Step 6: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_default_line`
  Expected: it does not compile. Two `error[E0061]: this function takes 2 arguments but 3 arguments were supplied` (the new tests call `input_tooltip` with a default).

- [ ] **Step 7 (B, green): the row and the tooltip take the default they show; the panel hands them the baseline's value.** The blocks in the test modules only add the argument to existing calls.

**`magcoupling-rs/src/gui/inputs.rs`**

Replace:

```rust
/// [`help_for`]), path, workbook cell, slider range, default, and whether it is a model
/// assumption.
```

with:

```rust
/// [`help_for`]), path, workbook cell, slider range, default, and whether it is a model
/// assumption. `default` is the value its Default line shows: the session's baseline value on
/// the panel (UX overhaul spec 7.1, C-22), `entry.default` (the spreadsheet's) elsewhere.
```

Replace:

```rust
pub fn input_tooltip(entry: &InputEntry, show_details: bool) -> String {
```

with:

```rust
pub fn input_tooltip(entry: &InputEntry, default: &Value, show_details: bool) -> String {
```

Replace:

```rust
    let default = match (&entry.default, meta.ty) {
```

with:

```rust
    let default = match (default, meta.ty) {
```

Replace:

```rust
        assert_eq!(
            input_tooltip(catalogue.entry("metal.face_gap_mm").unwrap(), true),
```

with:

```rust
        let tooltip = |path: &str| {
            let entry = catalogue.entry(path).unwrap();
            input_tooltip(entry, &entry.default, true)
        };
        assert_eq!(
            tooltip("metal.face_gap_mm"),
```

Replace:

```rust
        let backiron = input_tooltip(catalogue.entry("coupling.backiron").unwrap(), true);
```

with:

```rust
        let backiron = tooltip("coupling.backiron");
```

Replace:

```rust
        let drag = input_tooltip(catalogue.entry("metal.measured_drag_Nm").unwrap(), true);
```

with:

```rust
        let drag = tooltip("metal.measured_drag_Nm");
```

Replace:

```rust
        let harmonic = input_tooltip(catalogue.entry("coupling.max_harmonic").unwrap(), true);
```

with:

```rust
        let harmonic = tooltip("coupling.max_harmonic");
```

Replace:

```rust
        let npole = input_tooltip(catalogue.entry("coupling.npole").unwrap(), true);
```

with:

```rust
        let npole = tooltip("coupling.npole");
```

Replace:

```rust
            let on = input_tooltip(entry, true);
            let off = input_tooltip(entry, false);
```

with:

```rust
            let on = input_tooltip(entry, &entry.default, true);
            let off = input_tooltip(entry, &entry.default, false);
```

Replace:

```rust
            let tooltip = input_tooltip(entry, true);
```

with:

```rust
            let tooltip = input_tooltip(entry, &entry.default, true);
```

**`magcoupling-rs/src/gui/input_ui.rs`**

Replace:

```rust
//! dot when changed from default, a tooltip with help text (and, with the spreadsheet details,
```

with:

```rust
//! dot when changed from the default it is handed (the panel's baseline value, UX overhaul spec
//! 7.1), a tooltip with help text (and, with the spreadsheet details,
```

Replace:

```rust
    /// Back to the default.
```

with:

```rust
    /// Back to the default the row was drawn with.
```

Replace:

```rust
/// reset button) and the widget under it. `seed` is where an optional input starts when the
```

with:

```rust
/// reset button) and the widget under it. `default` is the value the dot, the reset button's
/// hover and the tooltip's Default line compare with and show: the session's baseline value
/// (UX overhaul spec 7.1, C-22). `seed` is where an optional input starts when the
```

Replace:

```rust
    seed: Option<f64>,
    locked: bool,
```

with:

```rust
    default: &Value,
    seed: Option<f64>,
    locked: bool,
```

Replace:

```rust
    let tooltip = || input_tooltip(entry, show_details);
    let changed = *current != entry.default;
```

with:

```rust
    let tooltip = || input_tooltip(entry, default, show_details);
    let changed = current != default;
```

Replace:

```rust
                                format_value(&entry.default)
```

with:

```rust
                                format_value(default)
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        let row = ui.scope(|ui| input_row(ui, entry, &current, seed, locked, show_details));
```

with:

```rust
        // "The default" on screen is the baseline's value (UX overhaul spec 7.1, C-22).
        let default = self.baseline.get(&entry.path).unwrap_or(Value::None);
        let row =
            ui.scope(|ui| input_row(ui, entry, &current, &default, seed, locked, show_details));
```

Replace:

```rust
                RowEdit::Reset => entry.default.clone(),
```

with:

```rust
                RowEdit::Reset => default,
```

Replace:

```rust
            let want = format!("{}\n{TRACE_HINT}", input_tooltip(entry, show_details));
```

with:

```rust
            let want = format!(
                "{}\n{TRACE_HINT}",
                input_tooltip(entry, &entry.default, show_details)
            );
```

Replace:

```rust
                .map(|entry| crate::gui::inputs::input_tooltip(entry, true)),
```

with:

```rust
                .map(|entry| crate::gui::inputs::input_tooltip(entry, &entry.default, true)),
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
            .map(|entry| crate::gui::inputs::input_tooltip(entry, false))
```

with:

```rust
            .map(|entry| crate::gui::inputs::input_tooltip(entry, &entry.default, false))
```

- [ ] **Step 8: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_default_line the_changed_dot tooltip a_changed_input`
  Expected: PASS. `the_changed_dot_a_row_s_reset_and_the_default_line_follow_the_baseline` and `the_default_line_shows_the_default_it_is_handed` are among them, and `a_changed_input_shows_the_dot_and_its_reset_restores_the_default` still passes untouched.

- [ ] **Step 9 (C, red): the Equation panel's "changed" tag.**

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn an_assumption_term_is_styled_in_the_equation_panel() {
```

with:

```rust
    #[test]
    fn the_equation_panel_s_changed_tag_follows_the_baseline() {
        // UX overhaul spec 7.1 (C-22). The passing test design asks for 1.6 N·m, the
        // spreadsheet for 2.5 N·m; the hot minimum check's equation reads the requirement.
        let input = "input: click to find its row";
        let changed = format!("{CHANGED_DOT} changed; {input}");
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        harness.panel.explorer.open_path("metal.hot_min_check");
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            (count(&output, input), count(&output, &changed)),
            (1, 0),
            "at the baseline"
        );
        harness.panel.inputs.metal.required_min_Nm = 2.5;
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            (count(&output, input), count(&output, &changed)),
            (0, 1),
            "at the spreadsheet's requirement"
        );
    }

    #[test]
    fn an_assumption_term_is_styled_in_the_equation_panel() {
```

- [ ] **Step 10: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_equation_panel_s_changed_tag`
  Expected: FAIL with `assertion 'left == right' failed: at the baseline`, `left: (0, 1)`, `right: (1, 0)`: at the baseline the requirement still carries the tag, because the registry compares with the spreadsheet's 2.5 N·m.

- [ ] **Step 11 (C, green): `explorer_ui` takes the baseline.** The last four blocks add the argument to the explorer tests' calls.

**`magcoupling-rs/src/gui/explorer.rs`**

Replace:

```rust
/// `held` says why an input is not free on the design shown ([`held_reason`]), for the notes'
/// change entries. `clamps_paused` is the panel's pause of the shaft-clamp section: the Start here
```

with:

```rust
/// `baseline` is the session's baseline (UX overhaul spec 7.1, C-22): an input term carries the
/// changed tag while it differs from the baseline's value.
/// `held` says why an input is not free on the design shown ([`held_reason`]), for the notes'
/// change entries. `clamps_paused` is the panel's pause of the shaft-clamp section: the Start here
```

Replace:

```rust
pub fn explorer_ui(
    ui: &mut egui::Ui,
```

with:

```rust
#[allow(clippy::too_many_arguments)] // the design shown, its results and its baseline, then the view's switches
pub fn explorer_ui(
    ui: &mut egui::Ui,
```

Replace:

```rust
    show_details: bool,
    held: &dyn Fn(&str) -> Option<String>,
```

with:

```rust
    baseline: &DesignInputs,
    show_details: bool,
    held: &dyn Fn(&str) -> Option<String>,
```

Replace:

```rust
                    others_marked,
                    explorer.explain,
```

with:

```rust
                    baseline,
                    others_marked,
                    explorer.explain,
```

Replace:

```rust
/// `others_marked`, a term's internal name on hover with `show_details`. Returns the term
/// clicked and the link clicked in the note or in the next best read.
```

with:

```rust
/// `others_marked`, a term's internal name on hover with `show_details`. An input term carries
/// the changed tag while it differs from `baseline`'s value. Returns the term clicked and the
/// link clicked in the note or in the next best read.
#[allow(clippy::too_many_arguments)] // the equation, its design and baseline, then the view's switches
```

Replace:

```rust
    others_marked: bool,
    explain: bool,
```

with:

```rust
    baseline: &DesignInputs,
    others_marked: bool,
    explain: bool,
```

Replace:

```rust
                        let style = registry()
                            .term_style(&row.path, terms.inputs)
                            .expect("every term is an input or a result (registry build)");
```

with:

```rust
                        let mut style = registry()
                            .term_style(&row.path, terms.inputs)
                            .expect("every term is an input or a result (registry build)");
                        // The registry compares with the spreadsheet's values; on screen
                        // "changed" is against the session's baseline.
                        if matches!(style.kind, TermKind::Input { .. }) {
                            style.changed_from_default =
                                terms.inputs.get(&row.path) != baseline.get(&row.path);
                        }
```

Replace:

```rust
                false,
                &nothing_held,
```

with:

```rust
                &DesignInputs::default(),
                false,
                &nothing_held,
```

Replace:

```rust
                    false,
                    &nothing_held,
                    paused,
```

with:

```rust
                    &DesignInputs::default(),
                    false,
                    &nothing_held,
                    paused,
```

Replace:

```rust
                    show_details,
                    &nothing_held,
```

with:

```rust
                    &DesignInputs::default(),
                    show_details,
                    &nothing_held,
```

Replace:

```rust
                    false,
                    &nothing_held,
                    clamps::PAUSED,
```

with:

```rust
                    &DesignInputs::default(),
                    false,
                    &nothing_held,
                    clamps::PAUSED,
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
                        show_details,
                        &held,
```

with:

```rust
                        &self.baseline,
                        show_details,
                        &held,
```

- [ ] **Step 12: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_equation_panel_s_changed_tag an_assumption_term a_term_s_tag`
  Expected: PASS, 3 tests.

- [ ] **Step 13: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
(the results search is no edit); reset all; save and load
```

with:

```markdown
(the results search is no edit); reset all (back to the baseline, below); save and load
```

Replace:

```markdown
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1)
```

with:

```markdown
- **Baseline** (UX overhaul slice A, BL-077; design `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, section 7.1): a panel holds one baseline design, set when it is built and fixed for the session. `MagcouplingPanel::with_baseline(inputs)` opens on it; `MagcouplingPanel::new()` keeps the spreadsheet's values as its baseline. On screen "the default" is the baseline's value: the changed dot, a row's reset and its hover, the "Default:" line of an input's tooltip, Reset all and the Equation panel's "changed" tag compare with it. The spreadsheet export's "Changed from default" column, the assumptions banner and the dots of the Assumptions view still compare with the spreadsheet's values. No undo, redo, file, link or opened design moves the baseline, and it is in no design file, link or undo step.
- **Sizing** (`sizing.rs`, Addendum A1; the Size tab's Auto since slice 1)
```

- [ ] **Step 14: The checks before a commit** (Global Constraints). Expected: `cargo test --features app` ends its lib run with `1010 passed; 0 failed; 6 ignored`; both clippy runs end without a warning.

- [ ] **Step 15: Commit.**

  Message (`.superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T1.txt`), then the trailers:

  ```text
  feat(magcoupler): the panel keeps a session baseline that "the default" on screen follows (BL-077 T1)

  MagcouplingPanel::with_baseline opens on a design and keeps it as the baseline for the
  session; new() keeps the spreadsheet's values, so a panel built the old way behaves as before.
  Reset all, the changed dot, a row's reset and its hover, the tooltip's Default line and the
  Equation panel's changed tag compare with the baseline. The spreadsheet export's flag and the
  assumptions banner still compare with the workbook. No engine line changes.

  Tests: six new (the baseline opens and Reset all returns to it, nothing moves it, new() keeps
  the spreadsheet's values; the dot, the reset and the Default line; the changed tag). Existing
  calls of input_tooltip and explorer_ui gain the argument; no assertion changes.
  ```

  ```bash
  git add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/input_ui.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/explorer.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T1.txt
  ```

### Task 2: Two named designs: the status line and the example design's button

**Model:** `sonnet` (transcription). The new strings are the design's own drafts; Task 6 reviews them.

Two behaviours: (A) the status line names the design against the baseline; (B) "Open the example design" opens the spreadsheet's values as one undo step.

**Files:**
- Modify: `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: Task 1's `baseline` field, `Harness::with_baseline` and `design_at`.
- Produces:
  - `guidance::STARTING_DESIGN: &str = "Starting design"`.
  - `guidance::STARTING_DESIGN_HOVER: &str = "A design that passes every check. It keeps the example design's gearbox rating, space and magnets, and asks for less torque."`
  - `guidance::EXAMPLE_DESIGN_HOVER` now reads `"The values from the original spreadsheet."`
  - `guidance::status_line(status: &DesignStatus, inputs: &DesignInputs, baseline: &DesignInputs) -> (String, Option<&'static str>)`: the line and its hover.
  - `dashboard::dashboard_ui(ui, inputs, baseline: &DesignInputs, results, fixes, clamps_paused, readouts)`.
  - `gui::OPEN_EXAMPLE: &str = "Open the example design"` (defined in `panel.rs`, re-exported), `panel::EXAMPLE_OPENED: &str = "Example design opened"`.
  - `MagcouplingPanel::open_example_design(&mut self)` (public).

**Existing test calls this task edits (the assertions stay):** `dashboard_ui` in `dashboard.rs` (`the_correction_ids_and_their_legend_show_only_with_the_spreadsheet_details`, `dashboard_frame_with`, `the_example_design_s_status_line_says_on_hover_where_its_values_come_from`) and in `guidance.rs` (`assert_open_block_draws_no_fix`); `status_ui` in `dashboard.rs` (`every_guide_line_has_glyphs_in_the_default_fonts_and_plain_words`). Each passes `&DesignInputs::default()`. The dashboard test module also imports `design_status_text` itself, because the module above it no longer does.

- [ ] **Step 1 (A, red): the status line's three tests.**

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
    #[test]
    fn the_design_status_counts_the_failing_checks() {
```

with:

```rust
    #[test]
    fn the_status_line_names_the_design_against_the_baseline() {
        // UX overhaul spec 7.1: "Starting design" while the design is the session's baseline
        // and the baseline is not the spreadsheet's design; "Example design" at the
        // spreadsheet's values, whatever the baseline; "Your design" otherwise.
        let line = |inputs: &DesignInputs, baseline: &DesignInputs| {
            let status = design_status(inputs, &compute_all(inputs), false);
            status_line(&status, inputs, baseline)
        };
        let defaults = DesignInputs::default();
        let baseline = passing_design();
        assert_eq!(
            line(&baseline, &baseline),
            (
                "Starting design: passes all 10 checks".to_owned(),
                Some(STARTING_DESIGN_HOVER)
            )
        );
        assert_eq!(
            line(&defaults, &baseline),
            (
                "Example design: fails 4 of 10 checks".to_owned(),
                Some(EXAMPLE_DESIGN_HOVER)
            )
        );
        let edited = design(|i| i.metal.cup_wall_corner_mm = 2.0);
        assert_eq!(
            line(&edited, &baseline),
            ("Your design: fails 3 of 10 checks".to_owned(), None)
        );
        // A panel built the old way has the spreadsheet's design as its baseline: it stays the
        // example design, no design is the starting design, and every line is
        // design_status_text's.
        for inputs in [&defaults, &baseline, &edited] {
            let status = design_status(inputs, &compute_all(inputs), false);
            assert_eq!(line(inputs, &defaults).0, design_status_text(&status));
        }
        assert_eq!(line(&defaults, &defaults).1, Some(EXAMPLE_DESIGN_HOVER));
        assert_eq!(line(&baseline, &defaults).1, None);
    }

    #[test]
    fn the_design_status_counts_the_failing_checks() {
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
```

with:

```rust
    #[test]
    fn the_starting_design_s_status_line_says_on_hover_what_it_is() {
        // UX overhaul spec 7.1: at the session's baseline, when that is not the spreadsheet's
        // design, the line reads "Starting design" and its hover says what the design is; the
        // same design on a panel whose baseline is the spreadsheet's is "Your design", with no
        // hover.
        let inputs = passing_design();
        let results = compute_all(&inputs);
        let fixes = all_fixes(&inputs, &results, false);
        for (baseline, name, starting) in [
            (passing_design(), guidance::STARTING_DESIGN, true),
            (DesignInputs::default(), guidance::YOUR_DESIGN, false),
        ] {
            let text = format!("{name}: passes all 10 checks");
            let (_, output) = hover_frames(
                &egui::Context::default(),
                egui::vec2(1280.0, 4000.0),
                |output| text_center(output, &text),
                |ui| {
                    dashboard_ui(
                        ui,
                        &inputs,
                        &baseline,
                        &results,
                        &fixes,
                        false,
                        &mut Readouts::default(),
                    )
                },
            );
            let texts = drawn_texts(&output);
            let hover = guidance::STARTING_DESIGN_HOVER.to_owned();
            assert_eq!(texts.contains(&hover), starting, "{text}");
            let example = guidance::EXAMPLE_DESIGN_HOVER.to_owned();
            assert!(!texts.contains(&example), "{text}");
        }
    }

    #[test]
    fn the_status_line_counts_the_guided_checks_in_the_worst_level_s_colour() {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn nothing_moves_the_baseline() {
```

with:

```rust
    #[test]
    fn the_status_line_reads_starting_design_at_the_baseline() {
        // UX overhaul spec 7.1: the name follows the design; the count is live, as before.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline, SCREEN);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.05;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Your design: passes all 10 checks"), 1);
        harness.click_text(RESET_ALL);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
    }

    #[test]
    fn nothing_moves_the_baseline() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib status_line`
  Expected: it does not compile. `error[E0425]: cannot find function 'status_line'`, `cannot find value 'STARTING_DESIGN'`, `cannot find value 'STARTING_DESIGN_HOVER'`, and `error[E0061]: this function takes 6 arguments but 7 arguments were supplied` (`dashboard_ui`).

- [ ] **Step 3 (A, green): the names, `status_line`, and the dashboard draws it.**

**`magcoupling-rs/src/gui/guidance.rs`**

Replace:

```rust
/// The hover text of the status line at the defaults: where the example's values come from.
pub const EXAMPLE_DESIGN_HOVER: &str = "The starting values from the original spreadsheet.";
```

with:

```rust
/// The hover text of the status line at the defaults, and of the header's Open the example
/// design button: where the example's values come from.
pub const EXAMPLE_DESIGN_HOVER: &str = "The values from the original spreadsheet.";

/// The design status line's start at the session's baseline, when that is not the
/// spreadsheet's design (UX overhaul spec 7.1, C-20): the standalone app's starting design.
pub const STARTING_DESIGN: &str = "Starting design";

/// The hover text of the status line at the starting design. It describes the standalone
/// app's starting design (`app::starting_design`, whose tests hold each claim).
pub const STARTING_DESIGN_HOVER: &str = "A design that passes every check. It keeps the example design's gearbox rating, space and magnets, and asks for less torque.";
```

Replace:

```rust
    let checks = if status.bad == 0 {
        format!("passes all {} checks", status.total)
```

with:

```rust
    format!("{design}: {}", checks_text(status))
}

/// The design status line as the panel draws it, and its hover text (UX overhaul spec 7.1):
/// [`design_status_text`], but named [`STARTING_DESIGN`] while `inputs` are the session's
/// `baseline` and are not the spreadsheet's values. The example design and the starting design
/// each have a hover; any other design has none. `status` is `inputs`' own
/// ([`design_status`]).
pub fn status_line(
    status: &DesignStatus,
    inputs: &DesignInputs,
    baseline: &DesignInputs,
) -> (String, Option<&'static str>) {
    let (design, hover) = if status.is_default {
        (EXAMPLE_DESIGN, Some(EXAMPLE_DESIGN_HOVER))
    } else if inputs == baseline {
        (STARTING_DESIGN, Some(STARTING_DESIGN_HOVER))
    } else {
        (YOUR_DESIGN, None)
    };
    (format!("{design}: {}", checks_text(status)), hover)
}

/// The status line after the design's name: how many checks fail, then the cautions.
fn checks_text(status: &DesignStatus) -> String {
    let checks = if status.bad == 0 {
        format!("passes all {} checks", status.total)
```

Replace:

```rust
    format!("{design}: {checks}{cautions}")
```

with:

```rust
    format!("{checks}{cautions}")
```

Replace:

```rust
            |ui| dashboard_ui(ui, inputs, results, &all, paused, &mut Readouts::default()),
```

with:

```rust
            |ui| {
                dashboard_ui(
                    ui,
                    inputs,
                    &DesignInputs::default(),
                    results,
                    &all,
                    paused,
                    &mut Readouts::default(),
                )
            },
```

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    design_status, design_status_text, fix_text, guide_notes, row_learn,
```

with:

```rust
    design_status, fix_text, guide_notes, row_learn, status_line,
```

Replace:

```rust
/// and the status line does not count its check.
```

with:

```rust
/// and the status line does not count its check. `baseline` is the session's baseline: the
/// status line names the design against it ([`guidance::status_line`]).
```

Replace:

```rust
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    clamps_paused: bool,
```

with:

```rust
    baseline: &DesignInputs,
    results: &DesignResults,
    fixes: &[Vec<(&'static str, Direction)>],
    clamps_paused: bool,
```

Replace:

```rust
    status_ui(ui, inputs, results, clamps_paused);
```

with:

```rust
    status_ui(ui, inputs, baseline, results, clamps_paused);
```

Replace:

```rust
/// The design status line (guidance spec 3.1): whose design it is and how many guided checks
/// fail (those that count with the shaft-clamp section's pause, `clamps_paused`), in the colour
/// of the worst level, over a separator; at the defaults its hover says where the example's
/// values come from (spec S2-5).
```

with:

```rust
/// The design status line (guidance spec 3.1): whose design it is against the session's
/// `baseline` and how many guided checks fail (those that count with the shaft-clamp section's
/// pause, `clamps_paused`), in the colour of the worst level, over a separator; at the example
/// design and at the starting design its hover says what the design is (spec S2-5; UX overhaul
/// spec 7.1).
```

Replace:

```rust
    results: &DesignResults,
    clamps_paused: bool,
```

with:

```rust
    baseline: &DesignInputs,
    results: &DesignResults,
    clamps_paused: bool,
```

Replace:

```rust
    let text = egui::RichText::new(design_status_text(&status))
```

with:

```rust
    let (text, hover) = status_line(&status, inputs, baseline);
    let text = egui::RichText::new(text)
```

Replace:

```rust
    if status.is_default {
        line.on_hover_text(guidance::EXAMPLE_DESIGN_HOVER);
```

with:

```rust
    if let Some(hover) = hover {
        line.on_hover_text(hover);
```

Replace:

```rust
    use crate::gui::guidance::{CHECK_GUIDES, Direction, Fixes, ROW_LEARN, all_fixes, trade_offs};
```

with:

```rust
    use crate::gui::guidance::{
        CHECK_GUIDES, Direction, Fixes, ROW_LEARN, all_fixes, design_status_text, trade_offs,
    };
```

Replace:

```rust
                dashboard_ui(ui, &inputs, &results, &fixes, false, &mut readouts);
```

with:

```rust
                dashboard_ui(
                    ui,
                    &inputs,
                    &DesignInputs::default(),
                    &results,
                    &fixes,
                    false,
                    &mut readouts,
                );
```

Replace:

```rust
                &results,
                &fixes,
```

with:

```rust
                &DesignInputs::default(),
                &results,
                &fixes,
```

Replace:

```rust
                        &inputs,
                        &results,
```

with:

```rust
                        &inputs,
                        &DesignInputs::default(),
                        &results,
```

Replace:

```rust
                status_ui(ui, &inputs, &results, false);
```

with:

```rust
                status_ui(ui, &inputs, &DesignInputs::default(), &results, false);
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
                                &self.results,
                                &fixes,
```

with:

```rust
                                &self.baseline,
                                &self.results,
                                &fixes,
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib status`
  Expected: PASS. The three new tests are among them (`the_status_line_names_the_design_against_the_baseline`, `the_starting_design_s_status_line_says_on_hover_what_it_is`, `the_status_line_reads_starting_design_at_the_baseline`), and `the_status_line_counts_the_checks_at_default_and_after_a_fix`, `the_design_status_counts_the_failing_checks` and `the_example_design_s_status_line_says_on_hover_where_its_values_come_from` pass with their assertions untouched.

- [ ] **Step 5 (B, red): the button's tests, the text rules, and the new texts in the glyph guard.** The last block adds the four new texts to the list `every_shown_text` collects, so the existing guard `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` holds them too.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn nothing_moves_the_baseline() {
```

with:

```rust
    #[test]
    fn open_the_example_design_is_one_undo_step_and_keeps_the_baseline() {
        // UX overhaul spec 7.1: the button opens the spreadsheet's design as an edit; the dots
        // then show what the baseline changed; Reset all still returns to the baseline.
        let baseline = passing_design();
        let mut harness = Harness::with_baseline(baseline.clone(), SCREEN);
        harness.click_text(OPEN_EXAMPLE);
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 1);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, EXAMPLE_OPENED), 1);
        assert_eq!(count(&output, "Example design: fails 4 of 10 checks"), 1);
        // The Key design group shows two of the inputs the passing test design changes: the
        // face gap and the cup wall.
        assert_eq!(count(&output, CHANGED_DOT), 2);
        harness.click_text(UNDO);
        assert_eq!(harness.panel.design(), design_at(&baseline));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Starting design: passes all 10 checks"), 1);
        assert_eq!(count(&output, CHANGED_DOT), 0);
        harness.click_text(REDO);
        assert_eq!(harness.panel.design(), Design::default());
        harness.click_text(RESET_ALL);
        assert_eq!(harness.panel.design(), design_at(&baseline));
    }

    #[test]
    fn nothing_moves_the_baseline() {
```

Replace:

```rust
    #[test]
    fn nothing_moves_the_baseline() {
```

with:

```rust
    #[test]
    fn the_first_arrival_s_new_texts_keep_the_text_rules() {
        // UX overhaul spec 8.3: plain words; nothing says the coupling protects the gearbox;
        // none of the words banned for new strings (no pressure boundary is modelled, a ring is
        // not named by a colour, the clamp section is paused); no typed-in number.
        use crate::engine::explain::plain::mentions_gearbox_protection;
        use crate::gui::guidance::{STARTING_DESIGN, STARTING_DESIGN_HOVER, plain_text_violations};
        for text in [
            STARTING_DESIGN,
            STARTING_DESIGN_HOVER,
            EXAMPLE_DESIGN_HOVER,
            OPEN_EXAMPLE,
            EXAMPLE_OPENED,
        ] {
            assert_eq!(plain_text_violations(text), Vec::<&str>::new(), "{text:?}");
            assert!(!mentions_gearbox_protection(text), "{text:?}");
            let lower = text.to_lowercase();
            for banned in [
                "sealed",
                "containment",
                "leak",
                "clamp",
                "red ring",
                "blue ring",
            ] {
                assert!(!lower.contains(banned), "{banned:?} in {text:?}");
            }
            assert!(!text.chars().any(|c| c.is_ascii_digit()), "{text:?}");
        }
    }

    #[test]
    fn nothing_moves_the_baseline() {
```

Replace:

```rust
    #[test]
    fn nothing_moves_the_baseline() {
```

with:

```rust
    #[test]
    fn a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button() {
        // Reset all already opens it there, and the header stays as it was (spec 8.1, test 2).
        let mut harness = Harness::new();
        assert_eq!(count(&harness.frame(Vec::new()), OPEN_EXAMPLE), 0);
        let mut harness = Harness::with_baseline(passing_design(), SCREEN);
        assert_eq!(count(&harness.frame(Vec::new()), OPEN_EXAMPLE), 1);
    }

    #[test]
    fn nothing_moves_the_baseline() {
```

Replace:

```rust
        // The hover texts that are named consts (an inline hover literal is not collected).
        texts.extend(
```

with:

```rust
        // The starting design's name and hover, and the example design's button and status
        // (UX overhaul spec 7.1), which a panel at the spreadsheet's baseline does not draw.
        texts.extend(
            [
                crate::gui::guidance::STARTING_DESIGN,
                crate::gui::guidance::STARTING_DESIGN_HOVER,
                OPEN_EXAMPLE,
                EXAMPLE_OPENED,
            ]
            .map(str::to_owned),
        );
        // The hover texts that are named consts (an inline hover literal is not collected).
        texts.extend(
```

- [ ] **Step 6: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib example`
  Expected: it does not compile. `error[E0425]: cannot find value 'OPEN_EXAMPLE' in this scope`, `cannot find value 'EXAMPLE_OPENED'`, `cannot find value 'EXAMPLE_DESIGN_HOVER'`.

- [ ] **Step 7 (B, green): the button.** It is drawn only where the baseline is not the spreadsheet's design (decision P-4).

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::clamps;
```

with:

```rust
use crate::engine::api::workbook_defaults;
use crate::engine::assumptions::{self, ASSUMPTIONS};
use crate::engine::clamps;
```

Replace:

```rust
    ConsequenceCache, Consequences, FixCache, SIZE_TAB_ALIASES, SIZE_TAB_NOTE, TORQUE_BAND_NOTE,
    within_minimum_diameter,
```

with:

```rust
    ConsequenceCache, Consequences, EXAMPLE_DESIGN_HOVER, FixCache, SIZE_TAB_ALIASES,
    SIZE_TAB_NOTE, TORQUE_BAND_NOTE, within_minimum_diameter,
```

Replace:

```rust

/// The session buttons.
```

with:

```rust

/// The header button that opens the spreadsheet's design (UX overhaul spec 7.1), shown on a
/// panel whose baseline is another design: there Reset all no longer reaches it.
pub const OPEN_EXAMPLE: &str = "Open the example design";

/// The status line after [`OPEN_EXAMPLE`].
pub const EXAMPLE_OPENED: &str = "Example design opened";

/// The session buttons.
```

Replace:

```rust
    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
```

with:

```rust
    /// Opens the example design (the spreadsheet's values, with the default sizing state): an
    /// edit like a loaded file, so it can be undone. The baseline stays where it is.
    pub fn open_example_design(&mut self) {
        self.replace_design(Design::default());
    }

    /// Draws the panel into `ui` and applies this frame's edits.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
```

Replace:

```rust
            }
            ui.separator();
            if ui.button(SAVE_DESIGN).clicked() {
```

with:

```rust
            }
            // Where the baseline is the spreadsheet's design, Reset all already opens it.
            if self.baseline != *workbook_defaults()
                && ui
                    .button(OPEN_EXAMPLE)
                    .on_hover_text(EXAMPLE_DESIGN_HOVER)
                    .clicked()
            {
                self.open_example_design();
                self.status = Some(EXAMPLE_OPENED.to_owned());
            }
            ui.separator();
            if ui.button(SAVE_DESIGN).clicked() {
```

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
    CentreView, MagcouplingPanel, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX, WELCOME,
```

with:

```rust
    CentreView, MagcouplingPanel, OPEN_EXAMPLE, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX,
    WELCOME,
```

- [ ] **Step 8: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- example first_arrival every_text_the_panel_shows`
  Expected: PASS. `open_the_example_design_is_one_undo_step_and_keeps_the_baseline`, `a_panel_whose_baseline_is_the_spreadsheet_s_design_has_no_example_button`, `the_first_arrival_s_new_texts_keep_the_text_rules` and `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` are among them.

- [ ] **Step 9: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
  "The starting values from the original spreadsheet."; slice 2 amends G-6),
```

with:

```markdown
  "The values from the original spreadsheet."; slice 2 amends G-6), "Starting design: ..." while the inputs are the panel's baseline and the baseline is not the spreadsheet's design (`status_line`; its hover: "A design that passes every check. It keeps the example design's gearbox rating, space and magnets, and asks for less torque."; UX overhaul slice A),
```

Replace:

```markdown
moves the baseline, and it is in no design file, link or undo step.
```

with:

```markdown
moves the baseline, and it is in no design file, link or undo step. On a panel whose baseline is not the spreadsheet's design the header shows **Open the example design** after Reset all: it opens the spreadsheet's values as one undo step (its hover: "The values from the original spreadsheet."), and Reset all still returns to the baseline. A panel built with `new()` has no such button: there Reset all already opens the example design.
```

- [ ] **Step 10: The checks before a commit.** Expected: the lib run ends `1016 passed; 0 failed; 6 ignored`.

- [ ] **Step 11: Commit.**

  ```text
  feat(magcoupler): the status line names the starting design, and a button opens the example design (BL-077 T2)

  guidance::status_line names the design against the panel's baseline: "Starting design" at the
  baseline when that is not the spreadsheet's design, "Example design" at the spreadsheet's
  values, "Your design" otherwise; the starting and the example design each have a hover. The
  example design's hover loses the word "starting". On a panel whose baseline is not the
  spreadsheet's design the header shows "Open the example design": one undo step, the baseline
  stays. A panel built with new() draws no such button and reads as before.

  Tests: six new (the three names, the hover on the dashboard, the line on the panel; the
  button's undo step, no button at the spreadsheet's baseline, the text rules of the design's
  section 8.3). The new texts join the glyph and paused-clamp guard. Calls of dashboard_ui and
  status_ui gain the baseline; no assertion changes.
  ```

  ```bash
  git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T2.txt
  ```

### Task 3: The starting design, and the app opens on it

**Model:** `fable`. This task states the design's physics claims (it passes every counted check; hot-low 1.448 N·m against 1.35 N·m; 0.510 N·m against the 0.60 N·m gearbox rating; the 0.35 + 1.00 N·m split) and pins them to `compute_all`. The code is exact; the judgment is in Steps 2, 5 and 6.

**Files:**
- Modify: `magcoupling-rs/src/app.rs`, `magcoupling-rs/README.md`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

**Interfaces:**
- Consumes: `MagcouplingPanel::with_baseline`, `open_example_design`, `reset`, `undo`, `redo` (Tasks 1 and 2); `guidance::{counted_guides, design_status, level}`; `gui::OPEN_EXAMPLE`.
- Produces:
  - `app::starting_design() -> DesignInputs` (public): `DesignInputs::default()` with `coupling.npole = 8`, `coupling.inner_back_apothem_mm = 9.0`, `metal.face_gap_mm = 1.8`, `metal.cup_wall_corner_mm = 2.2`, `metal.cap_thread_dia_mm = 41.7`, `metal.required_min_Nm = 1.35`.
  - `MagcouplingApp::new(ctx)` builds its panel with `with_baseline(starting_design())`. `MagcouplingApp` no longer derives `Default`.
  - In the app's test module: `STARTING_STATUS`, `EXAMPLE_STATUS`, `STARTING_CHANGES: [(&str, f64, f64); 6]`, `fn frame_texts(ctx, app) -> Vec<String>` (one frame at 1280 x 800), `fn number(inputs, path) -> f64`.

**One existing test changes by design:** `the_app_shows_the_panel_as_a_full_page` asserted the example design's status line. The app no longer opens on that design, so that assertion moves to `the_app_opens_the_starting_design`; the test keeps its heading and theme assertions.

- [ ] **Step 1 (red): the test module's head and the seven tests.** The first block replaces the module's imports and its first test with the new imports, constants, helpers and that test.

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    use super::*;
    use crate::gui::guidance::{design_status, design_status_text};
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::drawn_texts;
    use crate::{DesignInputs, compute_all};

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let input = || egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(input(), |ctx| app.ui(ctx));
        let texts = drawn_texts(&ctx.run(input(), |ctx| app.ui(ctx)));
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        // The design status line tops the dashboard: it was laid out (its lower rows scroll on
        // this screen, under the check guides).
        let defaults = DesignInputs::default();
        let status = design_status_text(&design_status(
            &defaults,
            &compute_all(&defaults),
            crate::engine::clamps::PAUSED,
        ));
        assert!(texts.contains(&status), "missing {status:?} in {texts:?}");
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
    }
```

with:

```rust
    use super::*;
    use crate::compute_all;
    use crate::engine::meta::{InputSet, Value};
    use crate::gui::dashboard::{Level, warning_lines};
    use crate::gui::guidance::{counted_guides, design_status, level};
    use crate::gui::inputs::InputCatalogue;
    use crate::gui::session::{Design, encode_share_payload};
    use crate::gui::test_support::drawn_texts;

    /// The status line at the starting design and at the example design, with the shaft-clamp
    /// section paused as the app runs it (nine counted checks).
    const STARTING_STATUS: &str = "Starting design: passes all 9 checks";
    const EXAMPLE_STATUS: &str = "Example design: fails 4 of 9 checks";

    /// The six inputs the starting design changes (UX overhaul spec 7.1): the path, the
    /// spreadsheet's value and the starting design's.
    const STARTING_CHANGES: [(&str, f64, f64); 6] = [
        ("coupling.npole", 10.0, 8.0),
        ("coupling.inner_back_apothem_mm", 10.15, 9.0),
        ("metal.required_min_Nm", 2.5, 1.35),
        ("metal.face_gap_mm", 1.4, 1.8),
        ("metal.cup_wall_corner_mm", 1.8, 2.2),
        ("metal.cap_thread_dia_mm", 41.0, 41.7),
    ];

    /// One frame of `app` on the laptop screen; the texts it drew.
    fn frame_texts(ctx: &egui::Context, app: &mut MagcouplingApp) -> Vec<String> {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        };
        drawn_texts(&ctx.run(input, |ctx| app.ui(ctx)))
    }

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
        match inputs.get(path) {
            Some(Value::Num(x)) => x,
            Some(Value::Int(i)) => i as f64,
            other => panic!("{path}: {other:?}"),
        }
    }

    #[test]
    fn the_app_shows_the_panel_as_a_full_page() {
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(
            texts.iter().any(|t| t == "Magnetic coupling calculator"),
            "{texts:?}"
        );
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
    }
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
        // UX overhaul spec 8.1, test 1, pinned to compute_all. If a later physics fix turns
        // this red the cure is a new starting design, never a loosened test (spec 12.2).
        let inputs = starting_design();
        let results = compute_all(&inputs);
        // Nine counted checks while the shaft-clamp section is paused, ten with it shown.
        for (clamps_paused, total) in [(true, 9), (false, 10)] {
            let status = design_status(&inputs, &results, clamps_paused);
            assert_eq!(
                (status.bad, status.caution, status.total),
                (0, 0, total),
                "paused: {clamps_paused}"
            );
            for guide in counted_guides(clamps_paused) {
                assert_eq!(
                    level(guide, &inputs, &results),
                    Level::Good,
                    "{}",
                    guide.label
                );
            }
        }
        let warnings = warning_lines(&results);
        assert!(warnings.is_empty(), "{:?}", warnings[0].1);
        // No drawing note asks for a look (the wall rule's hint has no level).
        let notes = crate::gui::geometry::geometry(&inputs, &results, true).notes;
        assert!(notes.iter().all(|note| note.level.is_none()), "{notes:?}");
        // The numbers the design document and the README quote, at the precision they quote.
        let close = |got: f64, want: f64, tolerance: f64| (got - want).abs() < tolerance;
        let hot_low = results.metal.torque_hot_low_Nm;
        assert!(close(hot_low, 1.448, 0.0005), "hot-low {hot_low}");
        assert!(
            close(hot_low / inputs.metal.required_min_Nm - 1.0, 0.073, 0.0005),
            "the hot-low margin over 1.35 N·m"
        );
        let at_gearbox = results.metal.cold_high_input_Nm;
        assert!(
            close(at_gearbox, 0.510, 0.0005),
            "at the gearbox {at_gearbox}"
        );
        assert_eq!(inputs.coupling.gearbox_input_rating_Nm, 0.6);
        assert!(close(results.model.pullout_Nm, 1.704, 0.0005));
        assert!(close(results.metal.torque_cold_high_Nm, 2.423, 0.0005));
        // It passes up to a Required torque of 1.44 N·m, and not at 1.45.
        for (required, bad) in [(1.44, 0), (1.45, 1)] {
            let mut asked = starting_design();
            asked.metal.required_min_Nm = required;
            let status = design_status(&asked, &compute_all(&asked), true);
            assert_eq!(status.bad, bad, "Required torque {required}");
        }
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_starting_design_changes_six_inputs_on_their_slider_steps_and_no_assumption() {
        // UX overhaul spec 7.1: six inputs differ from the spreadsheet's, each one a value a
        // slider step reaches, so a user can walk from one design to the other.
        let (start, example) = (starting_design(), DesignInputs::default());
        let mut differing = Vec::new();
        start.visit("", &mut |path, _, value| {
            if example.get(path) != Some(value) {
                differing.push(path.to_owned());
            }
        });
        differing.sort();
        let mut want = STARTING_CHANGES.map(|(path, _, _)| path.to_owned());
        want.sort();
        assert_eq!(differing, want);
        for (path, from, to) in STARTING_CHANGES {
            assert_eq!(number(&example, path), from, "{path}");
            assert_eq!(number(&start, path), to, "{path}");
            let range = InputCatalogue::get()
                .entry(path)
                .and_then(|entry| entry.meta.range)
                .unwrap_or_else(|| panic!("{path} has a slider"));
            let steps = (to - range.min) / range.step;
            assert!(
                (range.min..=range.max).contains(&to) && (steps - steps.round()).abs() < 1e-9,
                "{path}: {to} is not a step of {range:?}"
            );
        }
        // It changes no assumption, so the assumptions banner is quiet at both designs.
        for inputs in [&start, &example] {
            let modified = crate::engine::assumptions::modified(inputs);
            assert!(modified.is_empty(), "{:?}", modified[0].label);
        }
        // What the status line's hover says the design keeps (STARTING_DESIGN_HOVER): the
        // gearbox rating, the space and the magnets; and it asks for less torque.
        for path in [
            "coupling.gearbox_input_rating_Nm",
            "metal.max_diameter_mm",
            "metal.max_overall_axial_mm",
            "metal.max_large_dia_axial_mm",
            "coupling.magnets.part_inner",
            "coupling.magnets.part_outer",
            "coupling.magnets.axial_length_mm",
        ] {
            assert!(start.get(path).is_some(), "{path} is an input");
            assert_eq!(start.get(path), example.get(path), "{path}");
        }
        assert!(start.metal.required_min_Nm < example.metal.required_min_Nm);
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_app_opens_the_starting_design() {
        // UX overhaul spec 8.1, test 3: the app hands the panel the starting design as its
        // baseline, so the first view names it and Reset all returns to it.
        use crate::gui::input_ui::CHANGED_DOT;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        assert_eq!(app.panel().inputs(), &starting_design());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        let count = |text: &str| texts.iter().filter(|t| *t == text).count();
        assert_eq!(count(STARTING_STATUS), 1, "{texts:?}");
        assert_eq!(count(CHANGED_DOT), 0, "no input is changed at the baseline");
        assert_eq!(
            count(crate::gui::OPEN_EXAMPLE),
            1,
            "the example design's button"
        );
        // The readouts show the requirement and its margin.
        assert_eq!(count("1.35 N·m"), 1, "{texts:?}");
        assert_eq!(count("margin +0.10 N·m (+7.3 %)"), 1, "{texts:?}");
        // The header takes a second row at this width; the torque card still opens.
        assert_eq!(count(crate::gui::torque_card::CAPTION), 1, "{texts:?}");
        // An edit, then Reset all: the starting design again, not the spreadsheet's.
        app.panel
            .load_design_file(&crate::gui::session::design_to_json(&Design::default()))
            .unwrap();
        app.panel.reset();
        assert_eq!(app.panel().inputs(), &starting_design());
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_starting_design_draws_one_advisory_on_the_opening_tab() {
        // A finding of the slice A plan (2026-11-17), kept visible on purpose: at a ring radius
        // of 9.00 mm the hub wall past the keyway is 2.25 mm, under the Size tab's guidance of
        // about 2.5 mm, so the opening tab draws that one amber line. No counted check judges
        // it. A ring radius of 9.26 mm with a 42.2 mm cap thread clears it and takes the cap
        // skirt from 0.55 to 0.30 mm.
        let inputs = starting_design();
        let wall = compute_all(&inputs).model.hub_wall_past_key_mm;
        assert!((wall - 2.25).abs() < 1e-9, "{wall}");
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        let advisory = "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm.";
        assert!(texts.iter().any(|t| t == advisory), "{texts:?}");
        assert!(
            !texts.iter().any(|t| t.starts_with("Exploration only")),
            "{texts:?}"
        );
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_example_design_opens_as_one_undo_step_from_the_starting_design() {
        // UX overhaul spec 8.1, test 4, on the real starting design (the panel's own tests hold
        // the mechanism on a test baseline).
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        app.panel.open_example_design();
        assert_eq!(app.panel().design(), Design::default());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == EXAMPLE_STATUS), "{texts:?}");
        // One undo step back: the starting design, named so.
        app.panel.undo();
        assert_eq!(app.panel().inputs(), &starting_design());
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == STARTING_STATUS), "{texts:?}");
        // Redo, then Reset all: the baseline is still the starting design.
        app.panel.redo();
        assert_eq!(app.panel().design(), Design::default());
        app.panel.reset();
        assert_eq!(app.panel().inputs(), &starting_design());
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn a_file_that_names_one_input_reads_the_rest_from_the_spreadsheet_not_the_baseline() {
        // UX overhaul spec 7.1: a file or link made before this change opens to the same design
        // as before. An input a file leaves out takes the spreadsheet's value, whatever the
        // panel's baseline (the reader is not changed by this slice; this pins it).
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let file = r#"{"format": "magcoupling-design", "version": 1, "inputs": {"metal.face_gap_mm": 2.0}}"#;
        app.open_picked_file(Ok(file.to_owned()));
        let mut want = Design::default();
        want.inputs.metal.face_gap_mm = 2.0;
        assert_eq!(app.panel().design(), want);
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_spreadsheet_export_still_flags_against_the_workbook() {
        // UX overhaul spec 7.1 (C-22) and 8.1, test 4: in the export "changed" means "differs
        // from the workbook", whatever the panel's baseline: the six inputs at the starting
        // design, none at the example design. The export's code is not changed by this slice;
        // this pins it.
        use crate::gui::spreadsheet::{CellValue, INPUT_COLUMNS, INPUTS_SHEET, YES, sheets};
        use crate::gui::test_support::snapshot;
        let column = |name: &str| {
            INPUT_COLUMNS
                .iter()
                .position(|c| *c == name)
                .unwrap_or_else(|| panic!("no column {name}"))
        };
        let (changed, path) = (column("Changed from default"), column("Path"));
        let flagged = |inputs: &DesignInputs| -> Vec<String> {
            let results = compute_all(inputs);
            let book = sheets(&snapshot(inputs, &results));
            let sheet = book
                .iter()
                .find(|sheet| sheet.name == INPUTS_SHEET)
                .expect("the Inputs sheet");
            let mut paths: Vec<String> = sheet
                .rows
                .iter()
                .filter(|row| {
                    row.len() == INPUT_COLUMNS.len()
                        && row[changed].value == CellValue::Text(YES.to_owned())
                })
                .map(|row| match &row[path].value {
                    CellValue::Text(path) => path.clone(),
                    other => panic!("a path cell holds {other:?}"),
                })
                .collect();
            paths.sort();
            paths
        };
        let mut want = STARTING_CHANGES.map(|(path, _, _)| path.to_owned());
        want.sort();
        assert_eq!(flagged(&starting_design()), want);
        assert_eq!(flagged(&DesignInputs::default()), Vec::<String>::new());
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests`
  Expected: it does not compile. `error[E0425]: cannot find function 'starting_design' in this scope` (once per use), and `DesignInputs` is not in scope (`E0412`, `E0433`: Step 3 adds its import to the module above).

- [ ] **Step 3 (green): the starting design, and the app hands it to its panel.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
use crate::gui::session::LoadError;
use crate::gui::{MagcouplingPanel, PanelRequest};
```

with:

```rust
use crate::DesignInputs;
use crate::gui::session::LoadError;
use crate::gui::{MagcouplingPanel, PanelRequest};
```

Replace:

```rust
/// The standalone app: one [`MagcouplingPanel`], full page.
#[derive(Default)]
```

with:

```rust
/// The design the app opens on when the page's address carries no share link (UX overhaul
/// spec 7.1; owner rulings O-8 and O-9). It keeps the example design's given facts (the
/// 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock
/// length on both rings, the sleeve, the liner, the allowances and every assumption) and is
/// sized inside them to pass every check. Six inputs differ from the spreadsheet's, each on its
/// slider's step.
///
/// Required torque is 1.35 N·m: 0.35 N·m of hot wheel-bearing drag plus 1.00 N·m to accelerate
/// the wheel (the owner's figures; the tool takes the requirement as one number). The design
/// passes every check up to a Required torque of 1.44 N·m.
pub fn starting_design() -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = 8;
    inputs.coupling.inner_back_apothem_mm = 9.0;
    inputs.metal.face_gap_mm = 1.8;
    inputs.metal.cup_wall_corner_mm = 2.2;
    inputs.metal.cap_thread_dia_mm = 41.7;
    inputs.metal.required_min_Nm = 1.35;
    inputs
}

/// The standalone app: one [`MagcouplingPanel`], full page.
```

Replace:

```rust
    /// The app at the default design, with the linkage app's theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        Self::default()
```

with:

```rust
    /// The app at the starting design ([`starting_design`], the panel's baseline for the
    /// session), with the linkage app's theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        Self {
            panel: MagcouplingPanel::with_baseline(starting_design()),
            picker: files::DesignPicker::default(),
        }
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::tests`
  Expected: PASS, 12 tests.

  If `the_starting_design_passes_every_check_with_no_caution_and_no_warning` fails on your base, a physics fix merged after 7c2dc8d has moved a margin. Stop and report to the controller. The cure is a new starting design, which is a design decision (the design's section 12.2); never loosen the test.

- [ ] **Step 5: See the two pin tests bite.** `the_spreadsheet_export_still_flags_against_the_workbook` and `a_file_that_names_one_input_reads_the_rest_from_the_spreadsheet_not_the_baseline` pin code this slice does not change, so they pass at once. Plant each fault, see the test fail, and undo it:
  1. In `magcoupling-rs/src/gui/spreadsheet.rs`, change `flag(value != entry.default),` to `flag(false),`. Run `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_spreadsheet_export_still_flags`. Expected: FAIL (the flagged list is empty). Undo with `git checkout -- magcoupling-rs/src/gui/spreadsheet.rs`.
  2. In `magcoupling-rs/src/gui/session.rs`, in the function `inputs_from`, add the line `inputs.coupling.npole = 8;` directly after its line `let mut inputs = DesignInputs::default();` (a reader that starts from another design). Run `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib a_file_that_names_one_input`. Expected: FAIL (`npole: 8` against `npole: 10` in the two designs). Undo with `git checkout -- magcoupling-rs/src/gui/session.rs`.
  3. `git status --short` shows neither file modified.

- [ ] **Step 6: Check the claims the texts make.** Read the new doc comment on `starting_design`, the hover `STARTING_DESIGN_HOVER` (Task 2) and the README paragraph of Step 7 against the tests: every number in them is asserted by `the_starting_design_passes_every_check_with_no_caution_and_no_warning` or `the_starting_design_changes_six_inputs_on_their_slider_steps_and_no_assumption`, except three that are the owner's or are arithmetic and that no test can hold: the bearing drag of 0.35 N·m and the 1.00 N·m for acceleration (owner ruling O-8 and O-9; nothing in the repository measures them), and the cap skirt of 0.55 mm ((42.8 - 41.7) / 2 from `metal.cap_od_mm` and `metal.cap_thread_dia_mm`). If a text says more than the tests and those three hold, change the text, not the test.

- [ ] **Step 7: The README, the register and the design's section 7.1.** The register row records decision P-10 as an assumption for the owner.

**`magcoupling-rs/README.md`**

Replace:

```markdown
**Size and torque** is the first view (decision G-5; it replaced the geometry view, M42-2).
```

with:

```markdown
**The starting design** (UX overhaul slice A, BL-077; owner rulings O-8 and O-9). The standalone app opens on `app::starting_design()` and hands it to the panel as its baseline, so the first status line reads "Starting design: passes all 9 checks". It keeps the example design's given facts (the 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock length on both rings, the sleeve, the liner, the allowances and every assumption) and differs from the spreadsheet's values in six inputs, each on its slider's step:

| Input | Spreadsheet | Starting design |
|---|---|---|
| Magnets per ring | 10 | 8 |
| Ring radius | 10.15 mm | 9.00 mm |
| Magnet face gap | 1.40 mm | 1.80 mm |
| Minimum outer return ring wall | 1.80 mm | 2.20 mm |
| Cap thread nominal diameter | 41.0 mm | 41.7 mm |
| Required torque | 2.50 N·m | 1.35 N·m |

Required torque is 1.35 N·m: 0.35 N·m of hot wheel-bearing drag plus 1.00 N·m to accelerate the wheel (the owner's figures; the tool takes the requirement as one number). Hot-low torque is 1.448 N·m, 7.3 % over the requirement, and the design passes up to a Required torque of 1.44 N·m. Cold-high torque at the gearbox input is 0.510 N·m against the 0.60 N·m rating. A test pins this to `compute_all` (`the_starting_design_passes_every_check_with_no_caution_and_no_warning`): if a later physics fix turns it red, the cure is a new starting design, not a looser test. Two things no counted check judges: the cap's skirt is 0.55 mm (cap outside diameter 42.8 mm less the 41.7 mm thread, halved), and the hub wall past the keyway is 2.25 mm, under the Size tab's guidance of about 2.5 mm, so the opening tab draws that one amber line (a ring radius of 9.26 mm with a 42.2 mm cap thread clears it and takes the skirt to 0.30 mm). `DesignInputs::default` is unchanged: it is the example design, the spreadsheet's values.

**Size and torque** is the first view (decision G-5; it replaced the geometry view, M42-2).
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
| The audit report |

## Log
```

with:

```markdown
| The audit report |
| A-17 | The starting design draws one amber line on the opening tab: "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." The design document did not know of it (its checks were read from the dashboard's functions; this line is drawn by the Size tab). Keep the approved design, or move it? | Keep the approved six values and leave the line visible. A test pins the line (`the_starting_design_draws_one_advisory_on_the_opening_tab`). | Nothing is hidden, and nothing the owner approved is moved. The other answer trades one margin for another: a ring radius of 9.26 mm with a 42.2 mm cap thread clears the line and takes the cap skirt from 0.55 to 0.30 mm, which no check judges. | A new starting design (ring radius 9.26 mm, cap thread 42.2 mm: passes all 9 checks, hot-low 1.463 N·m, 0.515 N·m at the gearbox input), with test 1 and the README re-pinned. | BL-077; `magcoupling-rs/src/app.rs`, `starting_design`; plan `docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md` |

## Log
```

Replace:

```markdown
- 2026-11-16: the owner approved the list (see "The owner's approval" above).
```

with:

```markdown
- 2026-11-16: the owner approved the list (see "The owner's approval" above).
- 2026-11-17: A-17 added with the slice A plan (BL-077): the starting design's keyway-wall line. It is an assumption, not the owner's answer.
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown
- **Ring radius 9.00 mm is the slider's minimum.** A user cannot walk further down from it.
```

with:

```markdown
- **Ring radius 9.00 mm is the slider's minimum.** A user cannot walk further down from it.
- **The opening tab draws one amber line** (found by the slice A plan, 2026-11-17; register row A-17). At a ring radius of 9.00 mm the hub wall past the keyway is 2.25 mm, and the Size tab says so against its guidance of about 2.5 mm ("Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm."). No counted check judges it, so the status line still reads "passes all 9 checks". A ring radius of 9.26 mm clears the line and takes the cap's skirt from 0.55 to 0.30 mm (with the cap thread at 42.2 mm; at 9.25 mm the wall computes a hair under 2.5 mm and the line still shows). The approved values are kept and the line stays visible until the owner rules.
```

- [ ] **Step 8: The checks before a commit.** Expected: the lib run ends `1023 passed; 0 failed; 6 ignored`.

- [ ] **Step 9: Commit.**

  ```text
  feat(magcoupler): the app opens on the starting design, Required torque 1.35 N·m (BL-077 T3)

  app::starting_design is the spreadsheet's design with six inputs changed, each on its slider's
  step: 8 magnets, ring radius 9.00 mm, face gap 1.80 mm, outer return ring wall 2.20 mm, cap
  thread 41.7 mm, Required torque 1.35 N·m (0.35 N·m of hot bearing drag plus 1.00 N·m to
  accelerate the wheel: owner rulings O-8 and O-9). The app hands it to the panel as its
  baseline, so the page opens on "Starting design: passes all 9 checks". DesignInputs::default
  is unchanged.

  Tests: seven new. The design is pinned to compute_all (every counted check Good, no caution,
  no material warning, hot-low 1.448 N·m, 0.510 N·m at the gearbox input, passing up to 1.44
  N·m); six inputs differ and no assumption; the app opens it; the example design is one undo
  step away; the spreadsheet export still flags against the workbook; a file that names one
  input reads the rest from the spreadsheet.

  Finding, kept visible (register row A-17): at this design the Size tab draws one amber line,
  "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." No counted check judges it.
  A test pins it.
  ```

  ```bash
  git add magcoupling-rs/src/app.rs magcoupling-rs/README.md docs/analyses/2026-11-10-conservative-assumptions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T3.txt
  ```

### Task 4: The design in the address bar

**Model:** `sonnet` (transcription; no physics claim).

**Files:**
- Modify: `magcoupling-rs/src/gui/history.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/Cargo.toml`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `app::starting_design` and the app tests' `frame_texts`, `EXAMPLE_STATUS` (Task 3); `session::share_link(base: &str, design: &Design) -> String`; `MagcouplingPanel::open_example_design` (Task 2).
- Produces:
  - `History::committed(&self) -> &T`: the state as of the last settled change.
  - `MagcouplingPanel::settled_design(&self) -> &Design` and `MagcouplingPanel::share_base(&self) -> &str` (public).
  - `app::address_of(base: &str, design: &Design) -> String` (public): `base` alone at the starting design with Auto off, else `share_link(base, design)`.
  - The private field `address_design: Design` of `MagcouplingApp`, and the private `fn replace_address(address: &str)` (wasm32: `history.replaceState`; native: nothing).
  - In the app's test module: `fn frame_with(ctx, app, events: Vec<egui::Event>) -> Vec<String>`, `fn address(app) -> String`, `fn gap_file(gap: f64) -> String`, `fn starting() -> Design`.

- [ ] **Step 1 (red): the tests.** In `app.rs` the first two blocks turn the frame helper into `frame_with`, which takes events; the next four add helpers.

**`magcoupling-rs/src/gui/history.rs`**

Replace:

```rust
    #[test]
    fn an_unchanged_state_records_nothing() {
```

with:

```rust
    #[test]
    fn committed_is_the_last_settled_state() {
        let mut history = History::new(1);
        assert_eq!(*history.committed(), 1);
        // An edit in progress is not in it.
        assert!(!history.observe(&2, false));
        assert_eq!(*history.committed(), 1);
        assert!(history.observe(&2, true));
        assert_eq!(*history.committed(), 2);
        // After an undo or a redo it is the state that gave.
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(*history.committed(), 1);
        assert_eq!(history.redo(&1), Some(2));
        assert_eq!(*history.committed(), 2);
    }

    #[test]
    fn an_unchanged_state_records_nothing() {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    #[test]
    fn a_drag_is_one_undo_step() {
```

with:

```rust
    #[test]
    fn the_settled_design_follows_settled_edits_undo_redo_and_an_opened_link() {
        // UX overhaul spec 7.2: what the web app writes the page's address from.
        let mut harness = Harness::new();
        assert_eq!(harness.panel.settled_design(), &Design::default());
        assert_eq!(harness.panel.share_base(), PUBLIC_BASE_URL);
        // A key held down: the edit is in progress, so the settled design waits.
        harness.focus(FACE_GAP);
        harness.frame(vec![key_event(egui::Key::ArrowRight, true)]);
        assert_eq!(harness.panel.design(), gap_design(1.41));
        assert_eq!(harness.panel.settled_design(), &Design::default());
        harness.frame(vec![key_event(egui::Key::ArrowRight, false)]);
        assert_eq!(harness.panel.settled_design(), &gap_design(1.41));
        harness.click_text(UNDO);
        assert_eq!(harness.panel.settled_design(), &Design::default());
        harness.click_text(REDO);
        assert_eq!(harness.panel.settled_design(), &gap_design(1.41));
        // A loaded file settles in the frame that follows it.
        harness
            .panel
            .load_design_file(&design_to_json(&gap_design(2.0)))
            .unwrap();
        harness.frame(Vec::new());
        assert_eq!(harness.panel.settled_design(), &gap_design(2.0));
        // A link opened as the session's start is settled at once.
        harness
            .panel
            .open_share_payload(&encode_share_payload(&gap_design(1.6)))
            .unwrap();
        assert_eq!(harness.panel.settled_design(), &gap_design(1.6));
        harness
            .panel
            .set_share_base("http://localhost:8080/tools/magcoupler/");
        assert_eq!(
            harness.panel.share_base(),
            "http://localhost:8080/tools/magcoupler/"
        );
    }

    #[test]
    fn a_drag_is_one_undo_step() {
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    /// One frame of `app` on the laptop screen; the texts it drew.
    fn frame_texts(ctx: &egui::Context, app: &mut MagcouplingApp) -> Vec<String> {
```

with:

```rust
    /// One frame of `app` on the laptop screen with `events`; the texts it drew.
    fn frame_with(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        events: Vec<egui::Event>,
    ) -> Vec<String> {
```

Replace:

```rust
            ..Default::default()
        };
```

with:

```rust
            events,
            ..Default::default()
        };
```

Replace:

```rust
    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

with:

```rust
    /// One frame of `app` with no input; the texts it drew.
    fn frame_texts(ctx: &egui::Context, app: &mut MagcouplingApp) -> Vec<String> {
        frame_with(ctx, app, Vec::new())
    }

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

Replace:

```rust
    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

with:

```rust
    /// The address the page shows, as the app last wrote it (or opened on).
    fn address(app: &MagcouplingApp) -> String {
        address_of(app.panel.share_base(), &app.address_design)
    }

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

Replace:

```rust
    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

with:

```rust
    /// The design with the face gap at `gap`, as a design file's text.
    fn gap_file(gap: f64) -> String {
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = gap;
        crate::gui::session::design_to_json(&design)
    }

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

Replace:

```rust
    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

with:

```rust
    /// The starting design as a design (Auto off).
    fn starting() -> Design {
        Design {
            inputs: starting_design(),
            sizing: SizingState::default(),
        }
    }

    /// The number at `path` of `inputs`.
    fn number(inputs: &DesignInputs, path: &str) -> f64 {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_address_is_clean_at_the_starting_design_and_a_share_link_elsewhere() {
        // UX overhaul spec 7.2 and C-23: one encoding, the link Share copies.
        use crate::gui::session::{PUBLIC_BASE_URL, SHARE_PARAM, decode_share_payload};
        let base = PUBLIC_BASE_URL;
        assert_eq!(address_of(base, &starting()), base);
        // Share still copies the full link at the starting design: a pasted link names every
        // input, so it keeps its meaning if the starting design changes later.
        let ctx = egui::Context::default();
        let app = MagcouplingApp::new(&ctx);
        let prefix = format!("{base}?{SHARE_PARAM}=");
        let shared = app.panel().share_link();
        let payload = shared.strip_prefix(&prefix).expect("a full link");
        assert_eq!(decode_share_payload(payload).unwrap(), starting());
        // Any other design: its share link, which decodes to the design that made it.
        let mut auto_on = starting();
        auto_on.sizing.mode = crate::gui::sizing::SizingMode::TorqueToMagnets;
        let mut edited = starting();
        edited.inputs.metal.face_gap_mm = 1.81;
        for design in [Design::default(), auto_on, edited] {
            let address = address_of(base, &design);
            assert_eq!(address, share_link(base, &design));
            let payload = address.strip_prefix(&prefix).expect("a share link");
            assert_eq!(decode_share_payload(payload).unwrap(), design);
        }
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn a_link_stays_under_four_thousand_characters() {
        // UX overhaul spec 7.2: a link names every input (decision M41-5), about 2,700
        // characters whatever the design; the ceiling catches a format that grows past what
        // an address bar and a host take.
        let mut long_text = starting();
        long_text.inputs.coupling.magnets.part_inner = "X".repeat(200);
        long_text.inputs.coupling.magnets.part_outer = "Y".repeat(200);
        for design in [starting(), Design::default(), long_text] {
            let link = share_link(crate::gui::session::PUBLIC_BASE_URL, &design);
            assert!(link.len() <= 4_000, "{} characters", link.len());
            assert!(link.len() > 2_000, "{} characters: every input", link.len());
        }
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn the_address_follows_settled_edits_undo_and_redo() {
        // UX overhaul spec 7.2 and 8.1, test 5. The write itself (history.replaceState) is the
        // browser smoke run's; this holds which design the address stands for.
        use crate::gui::test_support::key_event;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        let base = app.panel().share_base().to_owned();
        frame_texts(&ctx, &mut app);
        assert_eq!(
            address(&app),
            base,
            "the starting design: the clean address"
        );
        // An edit in progress (a key held down) writes nothing; its release does.
        app.panel.load_design_file(&gap_file(2.0)).unwrap();
        frame_with(&ctx, &mut app, vec![key_event(egui::Key::F12, true)]);
        assert_eq!(address(&app), base, "the edit has not settled");
        frame_with(&ctx, &mut app, vec![key_event(egui::Key::F12, false)]);
        let edited = app.panel().design();
        assert_eq!(address(&app), share_link(&base, &edited));
        assert_ne!(address(&app), base);
        // Idle frames keep it; undo returns the clean address; redo the link.
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, edited);
        app.panel.undo();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), base);
        app.panel.redo();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), share_link(&base, &edited));
        // The example design is a design like any other: its full link.
        app.panel.open_example_design();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), share_link(&base, &Design::default()));
        // Reset all: the starting design, the clean address again.
        app.panel.reset();
        frame_texts(&ctx, &mut app);
        assert_eq!(address(&app), base);
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

Replace:

```rust
    #[test]
    fn a_share_link_opens_its_design() {
```

with:

```rust
    #[test]
    fn a_link_opened_at_start_up_is_the_address_until_the_first_edit() {
        // UX overhaul spec 7.2: a link's address is not rewritten when it opens; a link with
        // the spreadsheet's values opens as the example design; a link that cannot be read
        // changes nothing, says why and leaves the address until the first edit.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        app.open_share_payload(&encode_share_payload(&Design::default()))
            .unwrap();
        assert_eq!(app.address_design, Design::default(), "no write is pending");
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(texts.iter().any(|t| t == EXAMPLE_STATUS), "{texts:?}");
        assert_eq!(app.address_design, Design::default());
        // The baseline is still the starting design: Reset all returns to it.
        app.panel.reset();
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, starting());

        let mut app = MagcouplingApp::new(&ctx);
        assert!(app.open_share_payload("broken").is_err());
        assert_eq!(
            app.panel().design(),
            starting(),
            "a bad link changes nothing"
        );
        frame_texts(&ctx, &mut app);
        let texts = frame_texts(&ctx, &mut app);
        assert!(
            texts.iter().any(|t| t.contains("not URL-safe base64")),
            "the header says why: {texts:?}"
        );
        assert_eq!(app.address_design, starting(), "nothing is written");
        // The first edit writes the address.
        app.panel.load_design_file(&gap_file(2.0)).unwrap();
        frame_texts(&ctx, &mut app);
        assert_eq!(app.address_design, app.panel().design());
        assert_ne!(app.address_design, starting());
    }

    #[test]
    fn a_share_link_opens_its_design() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib settled`
  Expected: it does not compile: `no method named 'committed' found for struct 'history::History'`, `no method named 'settled_design'`, `no method named 'share_base'`, `cannot find function 'address_of'`, `no field 'address_design'`.

- [ ] **Step 3 (green): the accessors, the address and the write.** The `Cargo.toml` block adds web-sys's `History` feature, in the list's alphabetical order.

**`magcoupling-rs/src/gui/history.rs`**

Replace:

```rust
    /// The number of undo steps recorded (an unsettled edit not included).
    pub fn undo_len(&self) -> usize {
```

with:

```rust
    /// The state as of the last settled change: what [`History::observe`] last recorded, or the
    /// state the last undo or redo gave. An edit in progress is not in it.
    pub fn committed(&self) -> &T {
        &self.committed
    }

    /// The number of undo steps recorded (an unsettled edit not included).
    pub fn undo_len(&self) -> usize {
```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
    /// The share link of the design.
    pub fn share_link(&self) -> String {
```

with:

```rust
    /// The address share links point at ([`MagcouplingPanel::set_share_base`]).
    pub fn share_base(&self) -> &str {
        &self.share_base
    }

    /// The share link of the design.
    pub fn share_link(&self) -> String {
```

Replace:

```rust
    /// Loads a design file's text; on refusal the design is unchanged and the panel shows why.
    pub fn load_design_file(&mut self, text: &str) -> Result<(), LoadError> {
```

with:

```rust
    /// The design as of the last settled edit, undo, redo or load: what the undo history holds
    /// as current. An edit in progress (a drag, a held key, a value being typed) is not in it
    /// until it settles. The web app writes the page's address from it (UX overhaul spec 7.2).
    pub fn settled_design(&self) -> &Design {
        self.history.committed()
    }

    /// Loads a design file's text; on refusal the design is unchanged and the panel shows why.
    pub fn load_design_file(&mut self, text: &str) -> Result<(), LoadError> {
```

**`magcoupling-rs/Cargo.toml`**

Replace:

```toml
    "HtmlAnchorElement",
    "HtmlCanvasElement",
```

with:

```toml
    "History",
    "HtmlAnchorElement",
    "HtmlCanvasElement",
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
use crate::gui::session::LoadError;
```

with:

```rust
use crate::gui::session::{Design, LoadError, share_link};
use crate::gui::sizing::SizingState;
```

Replace:

```rust
/// The standalone app: one [`MagcouplingPanel`], full page.
pub struct MagcouplingApp {
```

with:

```rust
/// The page address that stands for `design` (UX overhaul spec 7.2): the clean address, `base`
/// alone, at the starting design with Auto off (what the page opens on without a link); for any
/// other design the share link Share copies. One encoding: [`share_link`].
pub fn address_of(base: &str, design: &Design) -> String {
    let starting = design.sizing == SizingState::default() && design.inputs == starting_design();
    if starting {
        base.to_owned()
    } else {
        share_link(base, design)
    }
}

/// Rewrites the page's address in place with `history.replaceState` (UX overhaul spec 7.2): no
/// history entry is added, so the browser's Back still leaves the tool. A refused write (Safari
/// refuses past 100 calls in 30 seconds) is ignored: the next settled edit writes again.
#[cfg(target_arch = "wasm32")]
fn replace_address(address: &str) {
    if let Some(history) = web_sys::window().and_then(|window| window.history().ok()) {
        let _ = history.replace_state_with_url(&wasm_bindgen::JsValue::NULL, "", Some(address));
    }
}

/// The native app has no address to write.
#[cfg(not(target_arch = "wasm32"))]
fn replace_address(_address: &str) {}

/// The standalone app: one [`MagcouplingPanel`], full page.
pub struct MagcouplingApp {
```

Replace:

```rust
    picker: files::DesignPicker,
}
```

with:

```rust
    picker: files::DesignPicker,
    /// The design the page's address stands for (UX overhaul spec 7.2): the design the app
    /// opened on, then the last one whose address it wrote. Compared with the panel's settled
    /// design every frame, so a frame that changes nothing encodes nothing.
    address_design: Design,
}
```

Replace:

```rust
        Self {
            panel: MagcouplingPanel::with_baseline(starting_design()),
            picker: files::DesignPicker::default(),
```

with:

```rust
        let panel = MagcouplingPanel::with_baseline(starting_design());
        let address_design = panel.design();
        Self {
            panel,
            picker: files::DesignPicker::default(),
            address_design,
```

Replace:

```rust
            Ok(()) => log::info!(
                "{SHARE_LINK_LOADED} (sizing: {})",
                self.panel.sizing().mode.label()
            ),
```

with:

```rust
            Ok(()) => {
                // The address already carries this design: nothing is written until an edit.
                self.address_design = self.panel.settled_design().clone();
                log::info!(
                    "{SHARE_LINK_LOADED} (sizing: {})",
                    self.panel.sizing().mode.label()
                );
            }
```

Replace:

```rust
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
    }
```

with:

```rust
                PanelRequest::OpenDesign => self.picker.pick(ctx),
            }
        }
        // The address follows the design once an edit has settled (UX overhaul spec 7.2).
        if *self.panel.settled_design() != self.address_design {
            self.address_design = self.panel.settled_design().clone();
            replace_address(&address_of(self.panel.share_base(), &self.address_design));
        }
    }
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- committed_is settled address a_link`
  Expected: PASS: `committed_is_the_last_settled_state`, `the_settled_design_follows_settled_edits_undo_redo_and_an_opened_link`, `the_address_is_clean_at_the_starting_design_and_a_share_link_elsewhere`, `the_address_follows_settled_edits_undo_and_redo`, `a_link_stays_under_four_thousand_characters`, `a_link_opened_at_start_up_is_the_address_until_the_first_edit`.

- [ ] **Step 5: The web build's code compiles.** The write is wasm32-only, so the native tests do not compile it.

  Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings`
  Expected: `Finished`, no warning (this is the second command of gate 9).

- [ ] **Step 6: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
the session's start (`open_share_payload`: the first Undo keeps it).
```

with:

```markdown
the session's start (`open_share_payload`: the first Undo keeps it). The web app also writes the design into the page's address (UX overhaul slice A, design section 7.2): after every settled edit, undo, redo, load or opened example design, `MagcouplingApp::ui` rewrites the address with `history.replaceState` to `app::address_of` the settled design (`MagcouplingPanel::settled_design`, the undo history's current state). That is the clean address at the starting design and the link Copy share link copies for any other design, so a reload or a bookmark brings the design back, and the browser's Back still leaves the tool (Undo is the tool's). Copy share link copies the full link at the starting design too. A link that cannot be read changes nothing, says why in the header, and leaves the address as it was until the first edit. The native app writes nothing. The Size tab's Minimum OD is a view preference, not a design input: a reload does not bring it back.
```

Replace:

```markdown
(the panel as a full page; does its requests),
```

with:

```markdown
(the panel as a full page; does its requests; on the web it writes the settled design into the page's address), `starting_design` (the design it opens on and hands the panel as its baseline), `address_of` (the page address that stands for a design),
```

- [ ] **Step 7: The checks before a commit.** Expected: the lib run ends `1029 passed; 0 failed; 6 ignored`.

- [ ] **Step 8: Commit.**

  ```text
  feat(magcoupler): the web app writes the design into the page's address (BL-077 T4)

  After every settled edit, undo, redo and load the app rewrites the page's address with
  history.replaceState, so a reload or a bookmark brings the design back and the browser's Back
  still leaves the tool. The address is app::address_of the panel's settled design (the undo
  history's current state): the clean address at the starting design, the link Share copies
  elsewhere. A link opened at start-up is left until the first edit; one that cannot be read
  changes nothing. The native app writes nothing. web-sys gains the History feature.

  Tests: six new (the history's committed state; the panel's settled design; the address at the
  starting design, with Auto on and after an edit; it follows settled edits, undo and redo and
  waits for an edit in progress; a link opened at start-up; a link stays under 4,000
  characters). The replaceState call itself is the browser smoke run's (T5).
  ```

  ```bash
  git add magcoupling-rs/src/gui/history.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/app.rs magcoupling-rs/Cargo.toml magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T4.txt
  ```

  `magcoupling-rs/Cargo.lock` does not change (web-sys is already locked; a feature adds no package). If `git status` shows it modified, stop and report.

### Task 5: gui-smoke: the clean address and the reload

**Model:** `sonnet` (transcription).

The calculator's smoke step gains two checks: `starting_design` (the clean address opens "Starting design: passes all 9 checks" and stays clean) and `address_follows` (after the drag the address has been rewritten, and a reload of it brings the edited design back). The second is the only check of the `history.replaceState` call (the design's test 5).

**Files:**
- Modify: `.claude/workflows/gui-smoke.js` (LF line endings: `.gitattributes` has `.claude/workflows/*.js text eol=lf`), `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `magcoupling-rs/src/app.rs` (the test module only), `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `app::starting_design`, `guidance::status_line`, `gui::OPEN_EXAMPLE`; the app test's `quoted`, `between` and `collect` (already in `the_smoke_test_share_link_opens_its_design`).
- Produces: in the script, the constant `STARTING_STATUS` and the required schema booleans `starting_design` and `address_follows`. The script's own rule (`CALCULATOR_CHECKS`) takes both in without an edit, because it reads the schema's required booleans.

- [ ] **Step 1 (red): the verdict test's list of checks.**

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
const CHECKS = ['canvas_present', 'first_view', 'details_toggle', 'share_link_loaded',
  'sizing_solved', 'design_file_loaded', 'explorer_ready', 'why_link_opened', 'tooltip_plain',
  'dashboard_scrolls', 'torque_card']
```

with:

```js
const CHECKS = ['canvas_present', 'starting_design', 'first_view', 'details_toggle',
  'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'explorer_ready', 'why_link_opened',
  'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'address_follows']
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1`
  Expected: `2 of 73 checks fail` (a reading with `starting_design` or `address_follows` false still passes the old script).

- [ ] **Step 3 (red): the app test holds the script to the app.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
```

with:

```rust
        // UX overhaul slice A (spec 7.1, 7.2 and 8.1 test 5): the clean address opens the
        // starting design, whose status line the script types once as STARTING_STATUS; the
        // step names the example design's button; and after the drag the address step reloads
        // the page. Each of the two new pages has its console messages collected.
        {
            use crate::gui::guidance::status_line;
            let start = starting_design();
            let status = design_status(&start, &compute_all(&start), crate::engine::clamps::PAUSED);
            let (line, _) = status_line(&status, &start, &start);
            assert_eq!(
                quoted("STARTING_STATUS"),
                line,
                "the script's STARTING_STATUS"
            );
            assert_eq!(
                script.matches(line.as_str()).count(),
                1,
                "the smoke prompt types the starting design's status once, in STARTING_STATUS"
            );
            assert!(
                script.contains(crate::gui::OPEN_EXAMPLE),
                "the smoke prompt names the example design's button"
            );
            assert!(
                between("starting_design=true", "it carries a share link").contains(collect),
                "the clean page's console messages are collected before the link is opened"
            );
            assert!(
                script.contains("await page.reload()"),
                "the address step reloads the page"
            );
            assert!(
                between("address_follows=true", "then close the browser").contains(collect),
                "the reloaded page's console messages are collected before the close"
            );
        }
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
```

- [ ] **Step 4: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link`
  Expected: FAIL, `no STARTING_STATUS`.

- [ ] **Step 5 (green): the script.** Eight replacements, in this order. The prompt's lines are long: each old text below is a part of its line that occurs once.

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls at 1280x800, the torque card shows its rows between the readouts and the drawing, and a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame)',
```

with:

```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator the clean address opens the starting design, a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls at 1280x800, the torque card shows its rows between the readouts and the drawing, a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame), and the address follows that edit so a reload brings the design back',
```

Replace:

```js
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'drag_moved', 'drag_frames_within_60fps'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    first_view: { type: 'boolean' },
```

with:

```js
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'drag_moved', 'drag_frames_within_60fps', 'address_follows'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    starting_design: { type: 'boolean' },
    address_follows: { type: 'boolean' },
    first_view: { type: 'boolean' },
```

Replace:

```js
// The torque card's heading, which the calculator's first view and its card step both look for:
```

with:

```js
// The status line of the design the calculator opens on at the clean address (no ?m=), the
// starting design (UX overhaul spec 7.1): typed once here. app.rs's smoke test holds it equal to
// the line the app draws for app::starting_design.
const STARTING_STATUS = 'Starting design: passes all 9 checks'

// The torque card's heading, which the calculator's first view and its card step both look for:
```

Replace:

```js
// The magnetic coupling calculator: deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque under the welcome
```

with:

```js
// The magnetic coupling calculator. First (UX overhaul slice A, spec 7.1 and 7.2) the clean
// address, with no share link: it opens the starting design and stays clean. Then
// deterministic state through its share link, checked in
// the console (the page is one canvas). A new page opens on Size and torque under the welcome
```

Replace:

```js
// drawing, and one drag of the gap slider is timed frame by frame. The script checks the agent's
```

with:

```js
// drawing, and one drag of the gap slider is timed frame by frame. Last (slice A, spec 7.2 and
// 8.1 test 5) the address: the drag was a settled edit, so the app rewrote the address with
// history.replaceState, and a reload of that address brings the edited design back; this is the
// one check of the replaceState call itself. The script checks the agent's
```

Replace:

```js
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size the dashboard must scroll at). Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

with:

```js
Steps: first set the window to 1280 x 800 CSS pixels with browser_resize (the laptop size the dashboard must scroll at). Then the starting design: navigate to ${url}/tools/magcoupler/ (the clean address, with no share link) and wait 8 seconds (WASM init). Take a screenshot and read the address's query with browser_run_code_unsafe, code: async (page) => page.evaluate(() => location.search). starting_design=true only if the dashboard (the right side) is topped by the line "${STARTING_STATUS}", the header row at the top of the page (it may wrap to a second line) shows the button "Open the example design" right after "Reset all", and the code returned an empty string (at the starting design the address stays clean). Collect the console messages at level "info" (this page's: the next step navigates away).
Then navigate to this exact URL (copy it whole; it carries a share link): ${page}
```

Replace:

```js
Collect the console messages at level "info" once more (it covers the reopened page and the drag), then close the browser.
```

with:

```js
Collect the console messages at level "info" once more (it covers the reopened page and the drag).
Then the address. The app rewrites it after every settled edit, and the drag was one. Read it, reload the page and read it again in one call of browser_run_code_unsafe, code: async (page) => { const before = await page.evaluate(() => location.href); await page.reload(); await page.waitForTimeout(8000); return { before, after: await page.evaluate(() => location.href) }; } Then take a screenshot. address_follows=true only if drag_moved, before contains "?m=" and is not the URL opened above (the drag's edit rewrote the address), after equals before, and on the reloaded page the Magnet face gap reads the value it read after the drag, not ${SMOKE_GAP_SHOWN} (the reload brought the edited design back). If drag_moved is false, address_follows=false: say so in notes. Collect the console messages at level "info" a last time (the reloaded page), then close the browser.
```

Replace:

```js
passed=true only if: page loaded, canvas present, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, torque_card, and zero console messages of type error
```

with:

```js
passed=true only if: page loaded, canvas present, starting_design, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, torque_card, address_follows, and zero console messages of type error
```

- [ ] **Step 6: Run to verify both pass.**

  Run: `node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs | tail -n 1`
  Expected: `all 75 checks hold`.

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link`
  Expected: PASS (cargo rebuilds: the script is an `include_str!` input of the test).

  Run: `git ls-files --eol .claude/workflows/gui-smoke.js`
  Expected: `i/lf w/lf`.

- [ ] **Step 7: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it opens
```

with:

```markdown
go. The web smoke test is the `gui-smoke` workflow (`.claude/workflows/gui-smoke.js`): it first opens `/tools/magcoupler/` at the clean address and checks the status line "Starting design: passes all 9 checks", the header's "Open the example design" button and that the address stays clean (UX overhaul slice A, `starting_design`); then it opens
```

Replace:

```markdown
a p95 within one 60 fps frame (16.7 ms) are required;
```

with:

```markdown
a p95 within one 60 fps frame (16.7 ms) are required; last (slice A, `address_follows`) it reads the address, which the app rewrote after the drag's settled edit, reloads the page and checks that the address is the same and the edited design is back (the one check of the `history.replaceState` call);
```

- [ ] **Step 8: The checks before a commit.** Expected: the lib run still ends `1029 passed; 0 failed; 6 ignored`.

- [ ] **Step 9: Commit.**

  ```text
  test(magcoupler): gui-smoke checks the starting design at the clean address and a reload after an edit (BL-077 T5)

  The calculator's smoke step first opens /tools/magcoupler/ with no share link: the status
  line reads "Starting design: passes all 9 checks", the header shows "Open the example
  design" and the address stays clean (starting_design). After the drag it reads the address,
  which the settled edit rewrote, reloads the page and checks that the address is the same and
  the edited design is back (address_follows): the one check of the history.replaceState call.
  A drag that did not hold now fails the run through that check.

  The verdict test (gate 13) lists both checks; app.rs holds the script's STARTING_STATUS to
  the line the app draws and the two new pages' console collection.
  ```

  ```bash
  git add .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs magcoupling-rs/src/app.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T5.txt
  ```

### Task 6: Text and physics review of the new strings and claims

**Model:** `fable`, one fresh-context reviewer (the design's section 8.3: the text review of decision G-3; a string that states physics also gets a physics review). The reviewer changes no code. It writes one record file.

**Files:**
- Create: `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`

**Interfaces:**
- Consumes: the five strings (`guidance::STARTING_DESIGN`, `STARTING_DESIGN_HOVER`, `EXAMPLE_DESIGN_HOVER`; `panel::OPEN_EXAMPLE`, `EXAMPLE_OPENED`); the doc comment of `app::starting_design`; the README's "The starting design" paragraph and its Baseline and address sentences; the design's sections 7.1, 8.3 and 12.3.
- Produces: the record, with one verdict per string and per claim. Task 7 names the record.

- [ ] **Step 1: Brief the reviewer** with this, and nothing of the builder's reasoning:

  > Review the on-screen texts and the claims that slice A of the magcoupler UX overhaul adds. Read `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` sections 2.3, 7.1 and 8.3, then the branch's diff of `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/app.rs` and `magcoupling-rs/README.md` against `main`. The audience is mechanically literate engineers who are not magnetics specialists.
  >
  > Text review, for each of the five strings ("Starting design"; its hover "A design that passes every check. It keeps the example design's gearbox rating, space and magnets, and asks for less torque."; the example design's hover "The values from the original spreadsheet."; the button "Open the example design"; the status "Example design opened"): is it plain, true, and does a first-time visitor tell "Starting design" from "Example design"? Does any of them break section 8.3?
  >
  > Physics review, for each claim, checked against the code and by running the tests named: (1) the starting design passes every counted check with no caution and no material warning (`app::tests::the_starting_design_passes_every_check_with_no_caution_and_no_warning`); (2) it keeps the example design's gearbox rating, space claim and magnets, and asks for less torque (`...changes_six_inputs_on_their_slider_steps_and_no_assumption`); (3) the numbers in the README paragraph and the doc comment (hot-low 1.448 N·m, 7.3 % over 1.35 N·m, passing up to 1.44 N·m, 0.510 N·m against the 0.60 N·m rating, the cap skirt of 0.55 mm); (4) the split of 1.35 N·m into 0.35 N·m of bearing drag and 1.00 N·m for acceleration is presented as the owner's figures and not as something the tool computes; (5) at this design the Size tab draws "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." beside a status line and a hover that say the design passes every check: is the hover still honest, and is the line's presence stated wherever the design is described?
  >
  > Write `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`: a table of the five strings and the five claims, each with PASS or FAIL, the evidence (a test name, a file and line, or a number you computed) and, for a FAIL, the exact replacement text. End with one line: APPROVED, or CHANGES NEEDED with their count. Do not edit any other file.

- [ ] **Step 2: Rule on the record.** APPROVED: go to Step 3. CHANGES NEEDED: a `fable` agent applies each replacement text to the constant and to every place that quotes it (the README; the design's section 7.1, where it quotes the hover), runs the checks before a commit, and the same reviewer re-reads only the changed texts and appends `## Re-review` with its verdict to the record. This plan's own text is not edited. At most two rounds; a third disagreement goes to the owner on the STOP sheet. A changed string that Task 2's or Task 3's tests quote literally ("Starting design: passes all 9 checks", "Example design: fails 4 of 9 checks") changes there and in `.claude/workflows/gui-smoke.js`'s `STARTING_STATUS` in the same commit.

- [ ] **Step 3: Commit the record** (and any text change of Step 2, in its own earlier commit).

  ```text
  docs(magcoupler): slice A text and physics review record (BL-077 T6)
  ```

  ```bash
  git add docs/analyses/2026-11-17-magcoupler-slice-a-review.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T6.txt
  ```

### Task 7: Docs sweep

**Model:** `sonnet` (doc and YAML edits from exact text).

**Files:**
- Modify: `magcoupling-rs/README.md` (three rows of its Tests table), `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

**Interfaces:**
- Consumes: the names Tasks 1 to 6 produced. Produces nothing code reads.

- [ ] **Step 1: Apply the sweep.** The YAML lines are very long: each old text below is a part of its line that occurs once. In `docs/ai/05-update-tracker.md`, change the entry's date if the build ran on a later day. In `docs/ai/backlog.yaml` the first block sets BL-077 to `in_progress`: it is not `fixed` until the owner has reviewed the diff (AGENTS.md). The second files the one new finding as BL-078 (decision P-13); if an item was filed since this plan was written and BL-078 is taken, use the next free id in both places of that block. `docs/ai/04-memory.yaml` gains the slice's line and the decisions of the design's section 1 that the slice makes real (the design's section 9.1).

**`magcoupling-rs/README.md`**

Replace:

```markdown
the changed dot and per-field reset;
```

with:

```markdown
the changed dot and per-field reset; the session baseline (UX overhaul slice A: a panel built `with_baseline` opens on it and Reset all returns to it, `new()` keeps the spreadsheet's values, nothing moves it; the changed dot, a row's reset and its hover, the tooltip's Default line and the Equation panel's changed tag follow it; the status line reads "Starting design" at it; Open the example design is one undo step and is not drawn on a panel whose baseline is the spreadsheet's design; the new texts keep the text rules of the design's section 8.3; the settled design follows settled edits, undo, redo and an opened link);
```

Replace:

```markdown
Undo and redo walk settled steps,
```

with:

```markdown
Undo and redo walk settled steps (`committed` is the last settled state),
```

Replace:

```markdown
The app draws the whole panel as a page in the CAD dark theme;
```

with:

```markdown
The app draws the whole panel as a page in the CAD dark theme and opens the starting design (its status line, no changed dot, the example design's button, the requirement and its margin on the readouts); the starting design passes every counted check with no caution and no material warning, pinned to `compute_all` with the numbers the README quotes, changes six inputs on their slider steps and no assumption, and draws its one keyway-wall line on the opening tab; the example design opens as one undo step and Reset all returns the starting design; the spreadsheet export still flags against the workbook (six inputs at the starting design, none at the example design); the page address is clean at the starting design and the share link elsewhere, follows settled edits, undo and redo and waits for an edit in progress; a link opened at start-up stays the address until the first edit, and one that cannot be read changes nothing; a link stays under 4,000 characters;
```

**`docs/ai/02-system.yaml`**

Replace:

```yaml
      - "The first view is Size and torque (decision G-5; reverses M42-2):
```

with:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077; design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md sections 7.1 and 7.2): DesignInputs::default() is the example design, the spreadsheet's values, and the overhaul never moves it. The standalone app opens on app::starting_design() (six inputs off the spreadsheet's, Required torque 1.35 N·m; its test pins it to compute_all: every counted check Good, no caution, no material warning, no assumption changed) and hands it to the panel as its baseline (MagcouplingPanel::with_baseline; new() keeps the spreadsheet's values). On screen 'the default' is the baseline: the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's 'Starting design'. The spreadsheet export's changed flag, the assumptions banner and the Assumptions view's dots stay against the workbook. Nothing moves the baseline (no undo, file, link or opened design), and it is in no file, link or undo step. Design files and links do not depend on it: they name every input, and a missing input still reads the spreadsheet's value. On the web the app writes app::address_of(the settled design) into the page's address with history.replaceState: the clean address at the starting design, the Share link elsewhere; Share always copies the full link. If a later physics fix makes the starting design fail, it gets a new starting design, never a looser test. The starting design draws one amber Size-tab line that no counted check judges (hub wall past the keyway 2.25 mm against about 2.5 mm guidance; docs/analyses/2026-11-10-conservative-assumptions.md row A-17)."
      - "The first view is Size and torque (decision G-5; reverses M42-2):
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
app: "src/app.rs (MagcouplingApp::new applies the theme; ui drains the panel's requests;
```

with:

```yaml
app: "src/app.rs (starting_design (UX overhaul slice A, BL-077: the design the app opens on and hands the panel as its baseline; Required torque 1.35 N·m) and address_of (the page address that stands for a design: the share base alone at the starting design, else its share link); MagcouplingApp::new applies the theme and builds the panel with_baseline(starting_design()); ui drains the panel's requests and, when the panel's settled design is not the one the address stands for, rewrites the page's address (replace_address: history.replaceState on wasm32, nothing natively);
```

Replace:

```yaml
panel.rs (MagcouplingPanel: header with Undo/Redo/Reset all/Save design/
```

with:

```yaml
panel.rs (MagcouplingPanel: new (the spreadsheet's values as its baseline) and with_baseline (UX overhaul slice A: the session's baseline, fixed when the panel is built, which the changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow), settled_design (the undo history's current state, which the web app writes the page's address from), share_base, open_example_design (the spreadsheet's design as one undo step); header with Undo/Redo/Reset all/Open the example design (OPEN_EXAMPLE, EXAMPLE_OPENED; drawn only where the baseline is not the spreadsheet's design)/Save design/
```

Replace:

```yaml
history.rs (History, MAX_UNDO_LEVELS 100)
```

with:

```yaml
history.rs (History, committed (the last settled state), MAX_UNDO_LEVELS 100)
```

Replace:

```yaml
design_status (over the guides that count: 9 while the clamp section is paused), design_status_text,
```

with:

```yaml
design_status (over the guides that count: 9 while the clamp section is paused), design_status_text, status_line (the line as the panel draws it against its baseline, with its hover: STARTING_DESIGN and STARTING_DESIGN_HOVER beside EXAMPLE_DESIGN and EXAMPLE_DESIGN_HOVER),
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
  - "Payload weights Track 2: before merge the user runs the hands-on checklist
```

with:

```yaml
  - "Magcoupler UX overhaul (design docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md, approved by the owner on 2026-11-16; six slices A to F, each with its own plan and branch). Slice A (BL-077; plan docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md; branch magcoupling/overhaul-a-first-arrival) is built and waits at its STOP for the owner's review of the diff before merge and push: the page opens on the starting design (Required torque 1.35 N·m, 'Starting design: passes all 9 checks'), the example design is one button away, and the web app writes the design into the page's address. For the owner at the STOP: the starting design draws one amber line on the opening tab (hub wall past the keyway 2.25 mm against about 2.5 mm guidance); a ring radius of 9.26 mm with a 42.2 mm cap thread clears it and takes the cap skirt from 0.55 to 0.30 mm (register row A-17). Next after the release: slice B, the instrument (BL-068, which it needs, is merged)."
  - "Payload weights Track 2: before merge the user runs the hands-on checklist
```

Replace:

```yaml
  - "NEXT (owner order, 2026-10-13, G-1;
```

with:

```yaml
  - "Decided (owner, 2026-11-08 to 2026-11-16; UX overhaul design sections 1.2 to 1.4, which hold the whole list; these are the ones slice A, BL-077, makes real): O-7 the passing example, the address bar and the seven fixes come first. O-8 and O-9 the opening design's requirement is 1.35 N·m ('Yes 1.35Nm'): 0.35 N·m of hot bearing drag plus 1.00 N·m for acceleration, on a design sized inside the given facts (the 0.6 N·m gearbox input rating at 5:1, the 43 mm space claim, B842SH magnets at their stock length: the design's call C-19). The design's own calls, which the owner approved with it on 2026-11-16: C-1 slice A is on today's screen; C-8 nothing is stored between visits, the address is the only memory; C-20 the opening design is named 'Starting design' and 'Example design' keeps meaning the spreadsheet's; C-21 and C-22 one baseline, fixed for the session, which 'the default' on screen follows while the spreadsheet export and the assumptions banner keep comparing with the workbook; C-23 Share always copies the full link; C-24 a reload does not keep the Size tab's Minimum OD; C-27 'Torque the coupling must carry for driving' stays at 0.7 N·m. Still open and standing on their recommendations (docs/analyses/2026-11-10-conservative-assumptions.md, rows A-2 to A-5): Q-1 the calculator's own theme, Q-3 slip heating is always judged, Q-4 slices A and B are released apart, Q-5 the long link stays. The slice A plan's own calls P-1 to P-14 are in docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md; row A-17 is the one that changes what a visitor sees."
  - "NEXT (owner order, 2026-10-13, G-1;
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown
## 2026-11-16 - Magcoupling tests: the differential JSON is parsed once per test binary
```

with:

```markdown
## 2026-11-17 - Magcoupler UX overhaul, slice A: the page opens on a starting design, and the design is in the address bar (BL-077; branch magcoupling/overhaul-a-first-arrival)
- What changed: the panel holds a session baseline (`MagcouplingPanel::with_baseline`; `new()` keeps the spreadsheet's values). The changed dot, a row's reset, the tooltip's Default line, Reset all, the Equation panel's changed tag and the status line's name follow it; the spreadsheet export's flag and the assumptions banner stay against the workbook. The app opens on `app::starting_design()` (8 magnets, ring radius 9.00 mm, face gap 1.80 mm, outer return ring wall 2.20 mm, cap thread 41.7 mm, Required torque 1.35 N·m = 0.35 N·m of hot bearing drag + 1.00 N·m to accelerate the wheel), which passes all 9 checks; "Open the example design" opens the spreadsheet's values as one undo step. On the web the app rewrites the page's address with `history.replaceState` after every settled edit, undo, redo and load (`app::address_of`: clean at the starting design, the share link elsewhere). `DesignInputs::default`, the engine and every parity fixture are unchanged.
- Tests: 25 new headless tests (panel, inputs, dashboard, guidance, history, app); the gui-smoke workflow gained `starting_design` and `address_follows`, and its verdict test (gate 13) their cases. The text and physics review is `docs/analyses/2026-11-17-magcoupler-slice-a-review.md`.
- For the owner: the starting design draws one amber line on the opening tab, "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." (register row A-17: kept visible, the approved values not moved). Not released until the owner's review of the diff (design 9.4).

## 2026-11-16 - Magcoupling tests: the differential JSON is parsed once per test binary
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: open
  notes: "Filed with the slice A plan on 2026-11-17 (the design's header: each slice's plan files one item).
```

with:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-a-first-arrival; the text and physics review is docs/analyses/2026-11-17-magcoupler-slice-a-review.md. in_progress, not fixed, until the owner has reviewed the diff (AGENTS.md, risk: physics). Filed with the slice A plan on 2026-11-17 (the design's header: each slice's plan files one item).
```

Replace:

```yaml
The keyway line's rounding at 2.50 mm is a new finding: the plan's Task 7 files it as its own item."
- id: BL-055
```

with:

```yaml
The keyway line's rounding at 2.50 mm is a new finding: the plan's Task 7 files it as its own item (BL-078)."
- id: BL-078
  title: "Magcoupler Size tab: the keyway-wall line shows at a wall of exactly 2.5 mm and reads '2.50 mm' against 'about 2.5 mm'"
  dimension: gui
  risk: mechanical
  evidence: "magcoupling-rs/src/gui/panel.rs, view_ui: the amber line 'Wall past keyway: {key_wall:.2} mm; existing guidance is about 2.5 mm.' is drawn while key_wall < 2.5, where key_wall is model.hub_wall_past_key_mm, a difference of floats. With the starting design's other values and a ring radius of 9.25 mm (a slider step) the wall computes as 2.4999999999999990 mm (compute_all at main 7c2dc8d, scratch probe of 2026-11-17), so the line shows and reads 'Wall past keyway: 2.50 mm; existing guidance is about 2.5 mm.'; at 9.26 mm (2.51 mm) it is gone. Found while planning BL-077, whose starting design sits at 2.25 mm."
  acceptance: "owner picks: compare the wall as the line prints it (rounded to 0.01 mm), or with a small tolerance; red test first at ring radius 9.25 mm (the line is not drawn at a wall that prints as 2.50 mm) and at 9.24 mm (it is, reading 2.49 mm); the line's words unchanged; GATE PASS"
  priority: 4
  status: open
  notes: "Filed by the BL-077 plan's docs sweep (the UX overhaul design's section 9.1: new findings are filed, never fixed in passing). It does not change BL-077's starting design, whose 2.25 mm is well under the guidance either way (register row A-17)."
- id: BL-055
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown
nothing here is built yet.**
```

with:

```markdown
slice A is built on the branch `magcoupling/overhaul-a-first-arrival` (BL-077) and waits for the owner's review of its diff; slices B to F are not built.**
```

Replace:

```markdown
- Backlog: none filed yet. Each slice
```

with:

```markdown
- Backlog: BL-077 is slice A's item (filed 2026-11-17 with its plan, `docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md`). Each later slice
```

- [ ] **Step 2: Check the YAML still parses and the doc tests pass.**

  ```bash
  python -c "import io, yaml; [yaml.safe_load(io.open('docs/ai/' + f, encoding='utf-8')) for f in ('02-system.yaml', '03-structure.yaml', '04-memory.yaml', 'backlog.yaml')]; print('parses')"
  cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count
  ```

  Expected: `parses`; the doc-count test passes (this sweep touches none of the counts it reads).

- [ ] **Step 3: Read the README's panel section once, top to bottom,** for a sentence the slice made stale that no task caught (for one: a sentence that says the page opens on the example design, or that Reset all restores the spreadsheet's values in the standalone app). If you find one, do not reword it yourself: report the sentence and where it is, and the controller sets the wording (a `fable` call). Do not touch a sentence that is still true.

- [ ] **Step 4: Commit.**

  ```text
  docs(magcoupler): slice A docs sweep: docs/ai, the tracker, the README's test rows, BL-077 in progress (BL-077 T7)
  ```

  ```bash
  git add magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md
  git commit -F .superpowers/sdd/2026-11-17-magcoupler-overhaul-a/msg/T7.txt
  ```

### Task 8: Release: whole-branch review, the full gate, gui-smoke, the STOP

**Model:** the `fable` controller; the whole-branch reviewer is a fresh-context `fable` agent; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

**Files:**
- Modify after the owner's OK: `docs/ai/backlog.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`

**Interfaces:**
- Consumes: the branch as Tasks 1 to 7 leave it; `linkage-sim-rs/scripts/gate.sh`, `build_web.sh` (run by the gate's `--full`), `serve_web.sh`; the `gui-smoke` workflow.

- [ ] **Step 1: Preconditions.**

  ```bash
  git status --short
  git diff --stat main...HEAD -- magcoupling-rs/src/engine magcoupling-rs/tests reference
  ```

  Expected: a clean tree but for the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`; the second command prints nothing (no engine, integration-test or oracle file changed).

- [ ] **Step 2: Whole-branch review** (fresh context, `fable`; give it the diff `git diff main...HEAD`, the design and this plan, not the builders' notes). It checks:
  1. The design's section 7 and tests 1 to 5 of 8.1, item by item, against the diff. A gap is a finding.
  2. The anti-goals of 2.3: no engine number moves; nothing is hidden (the keyway line, the assumptions banner, the fixed-3D notice); no text claims what the model does not support.
  3. Test 2: no existing assertion was edited. The only existing-test edits are the calls listed in Tasks 1 and 2 and the one app test Task 3 names.
  4. The address: one encoding; nothing written on an idle frame or during an edit; a refused write is ignored; the native app writes nothing.
  5. The decisions P-1 to P-14: each is what the code does, and none hides a behaviour the owner should rule on.

  One fix wave on `fable` for what it finds, each fix with its red test first, then the reviewer re-reads the fixes. Record the review in `.superpowers/sdd/2026-11-17-magcoupler-overhaul-a/final-review.md` (gitignored).

- [ ] **Step 3: The full gate.**

  ```bash
  MAGCOUPLING_PYTHON="$PWD/reference/magcoupling-py/.venv/Scripts/python.exe" bash linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 75 checks hold`; gate 14 built both web bundles. If a gate fails, fix the cause on the branch (red test first, the reviews it needs) and run this step again from the top. Never edit `gate.sh` to pass.

- [ ] **Step 4: gui-smoke on the local server.**
  1. In the background: `bash linkage-sim-rs/scripts/serve_web.sh 8080` (it serves `linkage-sim-rs/web/`, which Step 3's gate 14 just built).
  2. Run the `gui-smoke` workflow with the arguments `{"frame_check": true}`.
  3. Expected: `passed: true`; in `magcoupling`: `starting_design: true`, `address_follows: true`, `torque_card: true`, `drag_frames_within_60fps: true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-17-magcoupler-overhaul-a/smoke-local.json`.
  4. Stop the server.

  A failure of `starting_design` or `address_follows` is a finding about the build, not about the smoke run, until shown otherwise: open the page by hand at `http://localhost:8080/tools/magcoupler/`, edit a slider, and watch the address.

- [ ] **Step 5: The STOP sheet.** Write `.superpowers/sdd/2026-11-17-magcoupler-overhaul-a/stop-sheet.md` for the owner and stop. It holds:
  - What shipped, in five lines, and the commit list (`git log --oneline main..HEAD`).
  - The two status lines to look at in the built page: "Starting design: passes all 9 checks" at the clean address, and "Example design: fails 4 of 9 checks" after the button (the design's risk "Two names").
  - **The finding (P-10, register row A-17):** the opening tab's amber line "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm.", and the other answer: ring radius 9.26 mm with a 42.2 mm cap thread (passes all 9 checks, hot-low 1.463 N·m, cap skirt 0.30 mm instead of 0.55 mm). Which does the owner want?
  - The header's second row at 1280 pixels (P-14), until slice B.
  - The decisions P-1 to P-14, one line each, for the owner to overrule.
  - Not verified: Safari and Firefox (the address write was exercised in Chromium only); whether a 0.55 mm cap skirt can be made; the bearing drag figures; the live site until Step 7.
  - Asked: the owner's review of the diff (BL-077 is `risk: physics`), the go-ahead to merge and push (a push to `main` deploys), and whether to file the nine findings of the design's section 12.4 (P-13).

  **STOP here. Do not merge or push without the owner's OK.**

- [ ] **Step 6 (after the owner's OK): close the item and merge.**
  1. In `docs/ai/backlog.yaml`, in BL-077: `status: in_progress` becomes `status: fixed`, and its notes gain at their start `FIXED <date>: the owner reviewed the diff and gave the go-ahead to merge and push. `. In `docs/ai/04-memory.yaml` the slice A sentence "is built and waits at its STOP for the owner's review of the diff before merge and push" becomes "was released on <date> after the owner's review of the diff". In `docs/ai/05-update-tracker.md` the entry's last bullet becomes "Released <date> after the owner's review of the diff." Apply the owner's rulings on P-1 to P-14 first if any changes code (each its own commit, red test first, gate again).
  2. Commit (`docs(magcoupler): BL-077 fixed after the owner's review of the diff`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects, and push.

- [ ] **Step 7: The live smoke run.** When the deploy has finished (the `deploy-web.yml` run of the push is green), run the `gui-smoke` workflow with `{"url": "https://colesorkness.com"}`. Expected: `passed: true`, with `starting_design` and `address_follows` true. Its last step reloads a full-length link (about 2,700 characters) on the live host, which the design's section 12.3 lists as not verified. Report the result to the owner. If it fails, say so plainly: the site is live with the change.

## What this plan covers of the design

| Design item | Where |
|---|---|
| 7.1 The starting design: six values, Required torque 1.35 N·m, passes every counted check, changes no assumption, each value on a slider step | Task 3 (`starting_design`, its two tests) |
| 7.1 Two named designs; the status word; the example design's hover loses "starting" | Task 2 |
| 7.1 "Open the example design" on today's top bar, one undo step | Task 2 (the button), Task 3 (on the real starting design) |
| 7.1 The baseline: one constructor, fixed for the session, `new()` keeps the spreadsheet's values | Task 1 |
| 7.1 What follows the baseline (C-22): the dot, the reset, the Default line, Reset all, the "changed" tags | Task 1 |
| 7.1 What stays against the workbook: the export's flag, the assumptions banner | Task 3 (`the_spreadsheet_export_still_flags_against_the_workbook`; no assumption changed) |
| 7.1 A link or file made before this change opens to the same design | Task 3 (the one-input file), Task 4 (a link with the spreadsheet's values opens as the example design) |
| 7.1 The split of 1.35 N·m is written in the README and in a comment | Task 3 |
| 7.2 The address after every settled edit, undo, redo, load and the example design; `replaceState`; Back still leaves | Task 4, Task 5 (`address_follows`) |
| 7.2 The clean address at the starting design; Share copies the full link (C-23) | Task 4 |
| 7.2 A link that cannot be read changes nothing and leaves the address until the first edit | Task 4 |
| 7.2 The native app does nothing; view flags stay out of the address; Minimum OD is not kept (C-24) | Task 4 (the code and the README sentence) |
| 7.2 The length test (a ceiling of 4,000 characters); one full-length link on the live site | Task 4; Task 8 Step 7 |
| 8.1 Test 1 | Task 3 |
| 8.1 Test 2 | Tasks 1 and 2 (the listed calls; no assertion edited); Task 8 Step 2 |
| 8.1 Test 3 | Task 3 (`the_app_opens_the_starting_design`), Task 1 (the Default line and the tag on a test baseline) |
| 8.1 Test 4 | Task 2 (the button on a test baseline), Task 3 (on the starting design; the export; the assumptions). "Exactly the six differing inputs carry the dot" is held in two parts: a row carries the dot exactly when its value differs from the baseline's (Task 1), and exactly six inputs differ (Task 3). No test opens every input group to count six drawn dots. |
| 8.1 Test 5 | Tasks 4 and 5 |
| 8.3 Text rules | Task 2 (`the_first_arrival_s_new_texts_keep_the_text_rules`, the glyph guard), Task 6 |
| 9.1 Docs with each slice: the README, `docs/ai`, the backlog item, new findings filed, section 1's decisions in `04-memory.yaml`, the gui-smoke checklist | Each task's README step; Task 5 (the smoke run); Task 7 (the sweep, BL-078, the decisions) |
| 9.4 Release | Task 8 |
| Not in slice A: the theme and font (tests 6 and 7 of 8.1), the layout, the Explain panel | Slice B |
