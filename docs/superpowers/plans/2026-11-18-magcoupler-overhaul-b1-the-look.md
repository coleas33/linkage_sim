# Magcoupler UX overhaul, slice B, stage 1 (the look): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The calculator wears its own look, "Charcoal and crimson": one token table for every colour, the Inter font with tabular digits, a crimson brand mark and a crimson line under the active tab. The layout, the words and the numbers are today's.

**Architecture:**
- **One token table, in the app.** `app/theme.rs` holds the design's eleven tokens and `visuals()`, egui's dark visuals with the tokens in place of its colours. The app applies them; the panel still reads whatever `Visuals` its host gives it.
- **The font is the app's too.** `app/fonts.rs` embeds a 30 KB cut of Inter Regular and puts it ahead of egui's built-in fonts. The cut is built once by a committed script; its ten digits are Inter's tabular ones.
- **Two colours are the panel's own.** No egui theme has a field for a brand accent or for a passing check, so `gui/look.rs` owns them, with the two things the accent marks: the brand mark before the title and the active tab's underline.
- **Unchanged:** the engine, every on-screen string, every position and size constant, the exports, the linkage app.

**Tech Stack:** Rust (2024 edition); egui 0.32.3, eframe 0.32, egui_plot 0.33; fontTools 4.51.0 on Python 3.12 (to build the font file once; the app build does not run it); the headless egui harness (`magcoupling-rs/src/gui/test_support.rs` and the panel tests' `Harness`); `linkage-sim-rs/scripts/gate.sh`; the `gui-smoke` Playwright workflow (`.claude/workflows/gui-smoke.js`) and its verdict test under node (gate 13).

**Spec:** `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, approved by the owner on 2026-11-16. This stage is its section 6 (the look), tests 6 and 7 of section 8.1, the hub word of section 9.2 and the docs and release rules of section 9. Executors read the spec's sections 1.3, 2.2, 2.3, 6, 8.1 and 9 beside this plan. Inventory: `docs/analyses/2026-11-09-magcoupler-ux-inventory.md` (its section 7). The owner's chosen look: `.superpowers/brainstorm/522-1794188235/content/look-and-feel.html`, option 2 (gitignored, in the main checkout). Backlog item: **BL-079** (`docs/ai/backlog.yaml`). Question Q-1 stands on its recommendation (`docs/analyses/2026-11-10-conservative-assumptions.md`, row A-2). C-26 holds: BL-068 is merged (8285ea2).

**How this plan was made.** The stage was built once as a prototype on a copy of `main` at 17e4638, with the design's font spike (6.2) run first. Every block below was then applied, in this plan's order, to a fresh copy of 17e4638 with BL-079 filed. The result equals the prototype file for file (line endings aside) and passes `cargo test` with and without the app feature (1,045 lib tests pass with it: 13 new, and the parity test with the linkage app gone), both native clippy runs, both wasm32 clippy runs, `cargo fmt --check` and the gui-smoke verdict test (77 checks). Each red step's "Expected" below is what that replay printed. The font's build commands of Task 2 were run as written and gave the hashes they state. Not run for this plan: gates 1 to 3 (linkage-sim-rs, which this stage does not touch), gate 12 (the Python oracle), the release WASM build and the browser. Task 7 runs them.

## Why a stage, and what follows

The design names no stages inside slice B. This plan reads two in it (decision L-1):

- Section 8 keeps the look's tests (6 and 7, under "Slice A and the look") apart from slice B's tests (8 to 15).
- Section 6.2 puts the font spike before the plan "fixes sizes": the layout's sizes need the font first.
- The layout cannot be cut smaller. The check list drops the Problem, Why and Fix blocks, the Size column drops its hint lines, and both need the Explain panel to hold them, or something is lost (goal G-4).

So stage 1 is the look, which is visible on its own and changes no position. **What follows, as stage 2, with its own plan and backlog item:**

1. The screen (the design's sections 4 and 5, tests 8 to 15): the top bar and its Design menu, the Size column, the four tiles, the check list and the Explain panel with its orientation; the word budget and the nothing-lost table.
2. **The fixed-3D line on the first view (register row A-18).** Today's dashboard keeps "Fixed 3D fields from the example design" under the shaft joints row, by owner ruling R-15, and at 1280 x 800 that is below the first screen. The design's check list closes with the line (section 4.5), which puts it on the first view. Stage 2 pins that with a test. This stage does not move it: moving it on today's dashboard would overrule R-15 for a place stage 2 removes.
3. The slip-heating readout's stray unit (the design's 9.2: "never: steady state stays below the limit s"). The unit is appended wherever a result's value is formatted with its unit, in the dashboard, the plots, the results table and the Equation panel; stage 2 rebuilds the Plots tab that shows it.
4. The accent's third use, the one primary button (**Learn**), and state by shape as well as colour (C-4): both belong to widgets stage 2 adds.

## What the spike found

The design's section 6.2 asks for a throwaway spike before the plan. It was a headless renderer: the real `MagcouplingApp` drawn frame by frame, tessellated by egui and rasterized to a picture, at 1280 x 800 and 1920 x 1080. Seven pictures are in `.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/spike/` of the main checkout (gitignored): the first view before, and after at both designs, the Geometry tab, a plot and the Equation panel.

1. **Coverage.** The source has 54 characters outside ASCII. Inter 4.1 Regular has 47 of them. The other 7 are never drawn as text: the "element of" sign and the ceiling and floor brackets are strokes, "proportional to" is drawn as `~`, and the double plus is in no on-screen text. With the cut wired into the panel's glyph guards, every guarded text passes: no drawn character falls back to a built-in font.
2. **The equations.** `typeset.rs` needs no change. Inter's Greek, subscripts and operators set the equations cleanly at both sizes. `glyph_safe` stays as it is: the panel does not know its host's fonts.
3. **Tabular digits.** Inter's `tnum` feature substitutes 61 glyphs: the digits, and also the space, the full stop, the comma, the colon, the hyphen, the brackets and the operators. Frozen whole, it would widen every space in running text. The cut remaps the ten digits only (decision L-6). Their advance is 1328 of 2048 units.
4. **Row height.** Inter's rows are 1.21 em tall, taller than the built-in font's. At 1280 x 800 the first view still shows the whole centre; the dashboard shows one row fewer. The first view draws 612 words at the starting design (615 before).
5. **Size.** The cut is 386 characters, 435 glyphs, 30,684 bytes.
6. **One text colour.** egui's dark visuals have three text levels (label grey, white, and a weak grey of about 2.6:1). The tokens have two that pass 4.5:1. Body text, values and headings share the text colour, and `weak()` is the muted one (decision L-2). So `strong()` no longer sets a value apart from its label. That is plain on today's dashboard rows, which stage 2 replaces with tiles and check rows.
7. **The plots.** egui_plot draws its grid, ticks and axis labels from the text colour. With bright body text the grid was as loud as the curves. The plot now hands them the weak colour (Task 1).
8. **Selection.** egui fills every selected choice with `selection.bg_fill`. The accent there would put crimson on every selected choice of the page (the inputs' and the results' order, the Assumptions view, Auto's variable, the Equation panel's toggle, a picker's list), against C-5. A selected choice is a chip on the hairline colour with its text in the link blue (decision L-3).
9. **Two egui facts for the tests.** `Fonts::has_glyph` is false for a family's own replacement character (the question mark, when the family is the cut alone). A wrapping label's rect starts at its row's left edge, so the title's first glyph marks where it begins.

Not checked by the spike: a browser's own drawing of the font (the smoke run's `look`, Task 5), Safari and Firefox, and the release build's size.

## Global Constraints

- **Today's layout.** No position, size or spacing constant changes (`ITEM_SPACING`, `BUTTON_PADDING`, `INPUTS_WIDTH`, `DASHBOARD_WIDTH`). No on-screen string is added or reworded; the hub page's card loses one word (Task 4).
- **No engine number moves** (the design's section 2.3). No file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/` or `reference/` changes.
- **The linkage app is not touched** (O-1). No file under `linkage-sim-rs/src/` changes. The two files under `linkage-sim-rs/web/` are the calculator's page and the hub's card.
- **The panel sets no theme and no font** (the design's 2.2 and 6.3). Code under `magcoupling-rs/src/gui/` reads `ui.visuals()`. The only colours it owns are the ones it owned before, plus `look::ACCENT`.
- **The accent** (C-5) marks the brand mark and the active tab. It is never a text colour and never on a value or a check. Red means a failing check. Links stay egui's blue (C-13).
- **The exports do not change** (the design's 12.1).
- **The font file is built, never edited.** `magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf` has SHA-256 `6345f33a4302725884b90eb123e13d041a00a158e8bb2b79d70cea0d7c35fb32`. A file with another hash is not committed.
- **The blocks.** Each Replace block's old text occurs exactly once in its file at the moment its step applies it. Apply a step's blocks in the order given, top to bottom. Keep each file's line endings. If an old text is not found, or is found twice, stop and report: do not adapt the block. Line numbers are not given because earlier steps move them. `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-18-magcoupler-overhaul-b1\apply_plan.py` (gitignored, in the main checkout) applies one step's blocks this way: `python <it> <this plan> <task> <step>`, from the worktree's root. The replay used it.
- **TDD.** Red, then green. A red step that calls an API the next step adds fails to compile; that is its red, and the plan says so.
- **Before each commit**, from the repository root:
  ```bash
  cargo fmt --manifest-path magcoupling-rs/Cargo.toml
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
  ```
  After `cargo fmt`, `git status --short` shows only the files the task names. If it shows another, stop and report.
- **Commits.** Stage files by name: never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`, and never stage `docs/chebyshev_lambda/*.png` (gate 1 rewrites them; restore with `git checkout -- docs/chebyshev_lambda`). Write each commit message with the Write tool to `.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T<n>.txt` (gitignored), then `git commit -F <file>`. Do not pipe a here-string from PowerShell 5.1. End every message with the two trailer lines the controller gives the task: `Co-Authored-By:` naming the model that did the task, then the session's `Claude-Session:` line.
- **Shell.** On this machine `bash` typed in PowerShell is WSL, which fails on the checkout's line endings. Shell scripts run through Git Bash: `& 'C:\Program Files\Git\bin\bash.exe' <script>`. Python scripts are written with the Write tool, not a here-document: the shell mangles characters outside ASCII.
- **Docs ship with the code** (repository rule): each task updates the README text it makes stale. Task 6 sweeps `docs/ai/`.

## Model per task

CLAUDE.md section 5. BL-079 is `risk: mechanical` (decision L-12), and every code task transcribes exact blocks, so the builders and the per-task reviewers run on `sonnet`. The release and the whole-branch review run on `fable`. Git housekeeping runs on `haiku` at low effort.

| Task | What | Model |
|---|---|---|
| Before Task 1 | Branch and baseline run | `haiku` (git), controller |
| 1 | The theme: the token table, the visuals, the page colour, the pass green, the plots' axes | `sonnet` |
| 2 | The font: the cut of Inter, installed by the app; the glyph guards hold texts to it | `sonnet` |
| 3 | The accent's two marks: the brand mark and the active tab's line | `sonnet` |
| 4 | The hub page's card | `sonnet` |
| 5 | gui-smoke: the `look` check | `sonnet` |
| 6 | Docs sweep | `sonnet` |
| 7 | Release: the full gate, gui-smoke, the independent whole-branch review, the merge | `fable` controller; a fresh-context `fable` reviewer; `haiku` for git; the smoke workflow's own agents on `sonnet` |

A `sonnet` task that ends blocked, leaves a test red or is rejected in review is redone on `fable` (escalate, never downgrade). Each task's diff is reviewed by a fresh-context `sonnet` reviewer before the next task starts; it checks the diff against the task's blocks and nothing else.

## Review Focus

Five inputs or conditions the design implies and does not test, most likely first. Each has its test in the task named.

1. **A character outside the cut.** A later note or label uses a sign Inter's cut does not hold. A person expects one face; the app would draw that character in a built-in font in the middle of a word. Task 2: `assert_glyphs`, which every glyph guard of the panel calls, also holds each character to the cut, and its failure says how to add one.
2. **A failing design.** Red badges and red text beside a crimson mark. A person must not read the brand as an error, and the accent must sit on nothing that is judged. Task 3: `the_accent_marks_the_brand_and_the_active_tab_and_nothing_else` counts the accent's shapes at the starting design and at the failing example design.
3. **A window where the tab row wraps.** At 1280 points the eight tabs take two lines. A person expects the line under the tab they chose, wherever it sits. Task 3: `the_active_tab_is_underlined_in_the_accent_and_no_other` clicks a tab on the second line.
4. **A zoomed browser or a dense display.** At one and a half or two pixels per point egui rounds glyph advances to another grid. A person expects columns of numbers to stay aligned. Task 2: `the_ten_digits_have_one_advance_width` runs at three pixel densities and seven sizes.
5. **A host with another theme.** The panel is drawn in egui's default visuals by every panel test, and may be embedded again. A person expects the look's own pieces to work there. Task 1: `a_passing_check_is_the_ok_token_in_a_dark_theme` holds the light theme's green; Task 3's panel tests run in egui's default visuals; Task 2's `the_app_draws_its_text_in_inter` checks that a plain context keeps egui's fonts.

## Decisions this plan takes

Each is a reading of the design where it was open, or a finding. The owner may overrule any of them; Task 7 lists them in its report.

| Id | Decision | Why |
|---|---|---|
| L-1 | Slice B is built in stages, and this plan is the first: the look. | "Why a stage" above. The conservative answer to Q-4 is smaller releases. |
| L-2 | Body text, values and headings are the `text` token; `weak()` is `muted`. `strong()` and plain text are the same colour. | The token table gives body text and values one colour, and egui has no bold. The other mapping (plain text muted) would hide the greyed state of a row, which is drawn in the weak colour. |
| L-3 | A selected choice is a chip filled with the `border` token, its text in the link blue. The changed dot, the trace marks and the focus frame keep that blue. Selected text in a field sits on the same fill. | egui gives every selected choice `selection.bg_fill`; the accent there would be on every selected choice of the page (C-5). The `panel2` token, which the table names for a selected row, is too faint behind selected text in a field (1.15:1 against the field). Stage 2 draws its own selected row. |
| L-4 | The accent and the pass green are constants of the panel (`gui::look`), listed in the app's token table under their token names. | The design's 6.3: the colours the panel owns take the token values. egui's `Visuals` has no accent field once `selection` is neutral. A host that embeds the panel sees the calculator's accent. |
| L-5 | In this stage the accent marks the brand (an 8-point square before the title) and the active tab (a 2-point line under its name; the other tabs' names are muted). | Both are on today's header and tab row. Without them the look has no crimson. The primary button comes with the top bar. |
| L-6 | The cut remaps the ten digits to Inter's tabular ones, and nothing else of `tnum`. | Finding 3: the whole feature widens spaces and punctuation in running text. The design's 6.2 says "the `tnum` feature frozen in" and wants "tabular digits as its default digits"; the digits are what it wants. |
| L-7 | The cut is taken by ranges: Basic Latin, Latin-1, Greek, the dashes and quotes, superscripts and subscripts, arrows and 14 maths signs. 386 characters. | A fixed list from today's strings would drop the next Greek letter a note uses. The guards catch a character outside the ranges. |
| L-8 | One face, Regular. No type size changes. | The design's default plan. A second face is for the tiles (stage 2), if they need it. |
| L-9 | The plots take the weak text colour for their grid, ticks and axis labels, in the panel's code. | Finding 7. egui_plot has no grid colour of its own. In another host this draws them in that host's weak colour. |
| L-10 | The spreadsheet export's fills keep their palette: green 5AC878, and egui's dark amber and red. | The design's 12.1: no change to the exports' contents in slices A to C. The page's pass green moves to the token; the export's is pinned in `xlsx::fill`. |
| L-11 | The plots' three series colours, the drawing's part colours, the diagrams' pens and the equations' term palette do not change. | The design's 6.3: they take a token "where a token exists". A series or a part is not a failing check, a caution or a pass. C-7 keeps the drawing's pole colours. |
| L-12 | BL-079 is `risk: mechanical`, with no stop for the owner before the merge. | The stage shows no new number and makes no new physical claim (the design's header rule). The controller's instruction for this plan ends on the merge. |
| L-13 | Buttons, fields, menus and tooltips share one corner radius, 4 points. Menus and tooltips cast no shadow. | The chosen mockup's radius; C-6: hairlines, no gradients or glows. |
| L-14 | The smoke run gains one check, `look`, on the clean page. | It is the one place a browser's own drawing of the theme and the font is seen (the design's 12.3). |

## Files

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/app/theme.rs` | The token table, `visuals()`, `RADIUS`; its tests | 1 |
| `magcoupling-rs/src/gui/look.rs` | New: `ACCENT`, `OK`; then `MARK_SIDE`, `UNDERLINE_WIDTH`, `mark_ui`, `tab_ui` | 1, 3 |
| `magcoupling-rs/src/gui/mod.rs` | `pub mod look;` | 1 |
| `magcoupling-rs/src/gui/dashboard.rs` | `Level::Good` in a dark theme is `look::OK`; test | 1 |
| `magcoupling-rs/src/gui/xlsx.rs` | `fill` pins the export's green; its test renamed | 1 |
| `magcoupling-rs/src/gui/plots.rs` | The plot takes the weak text colour; test | 1 |
| `linkage-sim-rs/web/tools/magcoupler/index.html` | The page's background | 1 |
| `magcoupling-rs/assets/fonts/` | New: `subset_inter.py`, `Inter-Regular.subset.ttf`, `OFL.txt` | 2 |
| `magcoupling-rs/src/app/fonts.rs` | New: `INTER`, `INTER_NAME`, `install`, `cut_holds`; tests | 2 |
| `magcoupling-rs/src/gui/test_support.rs` | `assert_glyphs` also holds a text to the cut | 2 |
| `magcoupling-rs/src/gui/panel.rs` | The tab row and the header call `look`; tests | 3 |
| `magcoupling-rs/src/app.rs` | The app installs the font; tests | 1 to 5 |
| `linkage-sim-rs/web/tools/index.html` | The hub's card loses "clamps" | 4 |
| `.claude/workflows/gui-smoke.js` | The `look` check | 5 |
| `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs` | `look` in its list | 5 |
| `magcoupling-rs/README.md` | The theme, the font, the look, the smoke run, the test rows | 1 to 5 |
| `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml`; the register; the design | The docs sweep | 6, 7 |

**Existing tests this plan edits, and how.** No assertion is weakened.
- `app::theme::tests`: the module is replaced. `the_visuals_are_the_linkage_app_s` goes (Q-1) and `the_visuals_are_the_token_table` takes its place. `the_theme_is_dark_whatever_the_system_prefers` and `the_web_page_background_is_the_panel_colour` keep their assertions, calling `visuals()`; the first gains one on the button padding.
- `app::tests::the_app_shows_the_panel_as_a_full_page`: one call, `theme::cad_dark_visuals()` to `theme::visuals()`.
- `gui::xlsx::tests::the_level_fills_are_the_page_s_badge_colours` is renamed `the_level_fills_are_a_fixed_palette`; its three assertions stay.
- Helpers: `plots::tests::draw` calls a new `draw_in`; `app::tests::frame_with` calls a new `frame_output`; `test_support::assert_glyphs` gains the check of the cut; `app::tests::the_smoke_test_share_link_opens_its_design` gains a block.

## Before Task 1 (controller; git steps on `haiku` at low effort)

- [ ] **Step 1: Branch.** From `main` (at or after the commit that holds this plan), create the branch `magcoupling/overhaul-b1-the-look` in its own worktree (superpowers:using-git-worktrees). Never run `git worktree prune` in this repository (a WSL worktree depends on it).

- [ ] **Step 2: Baseline run.** From the worktree's root:

  ```bash
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib 2>&1 | tail -n 3
  ```

  Expected at 17e4638: `test result: ok. 1033 passed; 0 failed; 6 ignored`. If `main` has moved and the count differs, note the new count: the counts in the tasks below shift by the same amount.

- [ ] **Step 3: Check the base.** `git log --oneline -1 -- magcoupling-rs/src` shows `ea3fe3d`, and `git log --oneline -1 -- .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs linkage-sim-rs/web/tools` shows `bd47b2f`. If either shows a later commit, a file this plan edits may have changed and a block may no longer match once: the task that meets a mismatch stops and reports.

## Tasks

### Task 1: The theme

**Model:** `sonnet` (transcription).

Three behaviours, each with its own red and green: (A) the app's visuals are the token table, and the page's background follows; (B) a passing check is the `ok` token, and the export keeps its green; (C) the plots take the weak text colour for their axes.

**Files:**
- Create: `magcoupling-rs/src/gui/look.rs`
- Modify: `magcoupling-rs/src/app/theme.rs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/mod.rs`, `magcoupling-rs/src/gui/dashboard.rs`, `magcoupling-rs/src/gui/xlsx.rs`, `magcoupling-rs/src/gui/plots.rs`, `linkage-sim-rs/web/tools/magcoupler/index.html`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces:
  - `app::theme`: the constants `BG`, `PANEL`, `PANEL2`, `BORDER`, `TEXT`, `MUTED`, `ACCENT`, `BAD`, `WARN`, `OK`, `LINK` (`egui::Color32`), `RADIUS: u8`, `ITEM_SPACING`, `BUTTON_PADDING`; `pub fn visuals() -> egui::Visuals`; `pub fn apply(ctx: &egui::Context)`. `cad_dark_visuals` is gone.
  - `gui::look::ACCENT` and `gui::look::OK` (`egui::Color32`).
  - In the plots' test module: `fn draw_in(ctx: &egui::Context, kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput`.

- [ ] **Step 1 (A, red): the theme's tests.** The first block replaces the theme's whole test module; the second edits the one call of the old function in the app's tests.

**`magcoupling-rs/src/app/theme.rs`**

Replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The body of `fn cad_dark_visuals`, from its signature to the closing brace,
    /// with line endings normalized: a checkout may write either file with CRLF.
    fn body(source: &str) -> String {
        let source = source.replace("\r\n", "\n");
        let start = source
            .find("fn cad_dark_visuals() -> egui::Visuals {")
            .expect("the function is there");
        let end = source[start..].find("\n}\n").expect("it ends") + start;
        source[start..end].to_owned()
    }

    #[test]
    fn the_visuals_are_the_linkage_app_s() {
        let linkage = include_str!("../../../linkage-sim-rs/src/gui/theme.rs");
        let ours = include_str!("theme.rs");
        assert_eq!(body(ours), body(linkage));
        // And the spacing of its apply_cad_theme.
        assert!(linkage.contains("style.spacing.item_spacing = egui::vec2(6.0, 4.0);"));
        assert!(linkage.contains("style.spacing.button_padding = egui::vec2(8.0, 4.0);"));
        assert_eq!(
            (ITEM_SPACING, BUTTON_PADDING),
            (egui::vec2(6.0, 4.0), egui::vec2(8.0, 4.0))
        );
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
        let ctx = egui::Context::default();
        ctx.options_mut(|o| o.fallback_theme = egui::Theme::Light);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        assert_eq!(ctx.style().visuals, cad_dark_visuals());
        assert_eq!(ctx.style().spacing.item_spacing, ITEM_SPACING);
        // The system turning light changes nothing: the preference is dark.
        let light = egui::RawInput {
            system_theme: Some(egui::Theme::Light),
            ..Default::default()
        };
        let _ = ctx.run(light, |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
    }

    #[test]
    fn the_web_page_background_is_the_panel_colour() {
        let page = include_str!("../../../linkage-sim-rs/web/tools/magcoupler/index.html");
        let [r, g, b, _] = cad_dark_visuals().panel_fill.to_array();
        let colour = format!("background: #{r:02x}{g:02x}{b:02x};");
        assert!(page.contains(&colour), "index.html has no {colour}");
    }
}
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The WCAG 2 relative luminance of an opaque colour.
    fn luminance(colour: Color32) -> f64 {
        let linear = |channel: u8| {
            let s = f64::from(channel) / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * linear(colour.r()) + 0.7152 * linear(colour.g()) + 0.0722 * linear(colour.b())
    }

    /// The WCAG 2 contrast ratio of two opaque colours.
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (a, b) = (luminance(a), luminance(b));
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    #[test]
    fn the_tokens_are_the_spec_s_table() {
        // UX overhaul spec 6.1, value for value; the link blue is decision C-13's.
        let hex = |c: Color32| format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b());
        let table = [
            (BG, "#121417"),
            (PANEL, "#181b1f"),
            (PANEL2, "#1e2227"),
            (BORDER, "#262b31"),
            (TEXT, "#e6e8eb"),
            (MUTED, "#8a919a"),
            (ACCENT, "#c8102e"),
            (BAD, "#ff4d57"),
            (WARN, "#e0a100"),
            (OK, "#4cc38a"),
            (LINK, "#5aaaff"),
        ];
        for (token, value) in table {
            assert_eq!(hex(token), value);
        }
        assert_eq!(LINK, egui::Visuals::dark().hyperlink_color);
    }

    #[test]
    fn the_visuals_are_the_token_table() {
        // UX overhaul spec 8.1, test 6 (it replaces the_visuals_are_the_linkage_app_s: the
        // spec's question Q-1).
        let v = visuals();
        let hairline = Stroke::new(1.0, BORDER);
        let radius = CornerRadius::same(RADIUS);
        // Surfaces.
        assert_eq!(
            (v.panel_fill, v.window_fill, v.extreme_bg_color),
            (PANEL, PANEL2, BG)
        );
        assert_eq!((v.faint_bg_color, v.code_bg_color), (PANEL2, PANEL2));
        // Text: body and values, captions, links, a caution, a failing check.
        assert_eq!(
            (v.text_color(), v.strong_text_color(), v.weak_text_color()),
            (TEXT, TEXT, MUTED)
        );
        assert_eq!(
            (v.hyperlink_color, v.warn_fg_color, v.error_fg_color),
            (LINK, WARN, BAD)
        );
        // Hairlines, no shadow, one radius (decision C-6).
        assert_eq!(v.window_stroke, hairline);
        assert_eq!(
            (v.window_shadow, v.popup_shadow),
            (egui::Shadow::NONE, egui::Shadow::NONE)
        );
        assert_eq!(
            (v.window_corner_radius, v.menu_corner_radius),
            (radius, radius)
        );
        // What is selected: a chip on the border colour, its text in the link blue.
        assert_eq!(v.selection.bg_fill, BORDER);
        assert_eq!(v.selection.stroke, Stroke::new(1.0, LINK));
        // The widget states: fill, a button's fill, outline.
        let w = &v.widgets;
        for (name, state, fill, button, outline) in [
            ("noninteractive", w.noninteractive, PANEL2, PANEL2, hairline),
            ("inactive", w.inactive, BORDER, PANEL2, hairline),
            (
                "hovered",
                w.hovered,
                BORDER,
                BORDER,
                Stroke::new(1.0, MUTED),
            ),
            ("active", w.active, BORDER, BORDER, Stroke::new(1.0, TEXT)),
            ("open", w.open, PANEL2, PANEL2, hairline),
        ] {
            assert_eq!(
                (state.bg_fill, state.weak_bg_fill, state.bg_stroke),
                (fill, button, outline),
                "{name}"
            );
            assert_eq!(
                (state.fg_stroke.color, state.corner_radius),
                (TEXT, radius),
                "{name}"
            );
        }
        // Nothing else moved: with egui's values back in the fields above, these are egui's
        // dark visuals. A colour set outside the table fails here.
        let dark = egui::Visuals::dark();
        let mut rest = v.clone();
        rest.panel_fill = dark.panel_fill;
        rest.window_fill = dark.window_fill;
        rest.window_stroke = dark.window_stroke;
        rest.window_shadow = dark.window_shadow;
        rest.popup_shadow = dark.popup_shadow;
        rest.window_corner_radius = dark.window_corner_radius;
        rest.menu_corner_radius = dark.menu_corner_radius;
        rest.extreme_bg_color = dark.extreme_bg_color;
        rest.faint_bg_color = dark.faint_bg_color;
        rest.code_bg_color = dark.code_bg_color;
        rest.weak_text_color = dark.weak_text_color;
        rest.hyperlink_color = dark.hyperlink_color;
        rest.warn_fg_color = dark.warn_fg_color;
        rest.error_fg_color = dark.error_fg_color;
        rest.selection = dark.selection;
        for (ours, egui_s) in [
            (
                &mut rest.widgets.noninteractive,
                dark.widgets.noninteractive,
            ),
            (&mut rest.widgets.inactive, dark.widgets.inactive),
            (&mut rest.widgets.hovered, dark.widgets.hovered),
            (&mut rest.widgets.active, dark.widgets.active),
            (&mut rest.widgets.open, dark.widgets.open),
        ] {
            ours.bg_fill = egui_s.bg_fill;
            ours.weak_bg_fill = egui_s.weak_bg_fill;
            ours.bg_stroke = egui_s.bg_stroke;
            ours.fg_stroke.color = egui_s.fg_stroke.color;
            ours.corner_radius = egui_s.corner_radius;
        }
        assert_eq!(rest, dark);
        // The spacing the layout was built on.
        assert_eq!(
            (ITEM_SPACING, BUTTON_PADDING),
            (egui::vec2(6.0, 4.0), egui::vec2(8.0, 4.0))
        );
    }

    #[test]
    fn every_text_token_reads_on_every_surface_it_is_drawn_on() {
        // UX overhaul spec 6.1 and goal G-7: at least 4.5:1, computed here by the WCAG
        // formula. Text and links are drawn on all four surfaces. Captions and the three
        // level colours are not drawn on the border colour (a hovered button's and a selected
        // chip's fill, whose text is the text colour and the link blue): there they are under
        // 4.5:1, and the last assertions pin that so the rule is not lost.
        // Not counted: a disabled button's faded text (WCAG exempts an inactive control), and
        // a caption while its text is selected with the pointer.
        let everywhere = [BG, PANEL, PANEL2, BORDER];
        let off_the_fill = [BG, PANEL, PANEL2];
        let drawn: [(&str, Color32, &[Color32]); 6] = [
            ("text", TEXT, &everywhere),
            ("link", LINK, &everywhere),
            ("muted", MUTED, &off_the_fill),
            ("bad", BAD, &off_the_fill),
            ("warn", WARN, &off_the_fill),
            ("ok", OK, &off_the_fill),
        ];
        for (name, token, surfaces) in drawn {
            for &surface in surfaces {
                let ratio = contrast(token, surface);
                assert!(ratio >= 4.5, "{name} on {surface:?}: {ratio:.2}");
            }
        }
        // The spec's own figures (6.1), to one decimal.
        let one_decimal = |a, b| (contrast(a, b) * 10.0).round() / 10.0;
        assert_eq!(one_decimal(MUTED, PANEL), 5.4);
        assert_eq!(one_decimal(MUTED, PANEL2), 5.0);
        assert_eq!(one_decimal(BAD, PANEL), 5.3);
        assert_eq!(one_decimal(Color32::WHITE, ACCENT), 5.9);
        // Decision C-5: the accent is never a text colour. On the panel it is 2.9:1.
        assert_eq!(one_decimal(ACCENT, PANEL), 2.9);
        // Why captions and level colours stay off the border colour.
        assert!(contrast(MUTED, BORDER) < 4.5 && contrast(BAD, BORDER) < 4.5);
    }

    #[test]
    fn the_theme_is_dark_whatever_the_system_prefers() {
        let ctx = egui::Context::default();
        ctx.options_mut(|o| o.fallback_theme = egui::Theme::Light);
        apply(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
        assert_eq!(ctx.style().visuals, visuals());
        assert_eq!(ctx.style().spacing.item_spacing, ITEM_SPACING);
        assert_eq!(ctx.style().spacing.button_padding, BUTTON_PADDING);
        // The system turning light changes nothing: the preference is dark.
        let light = egui::RawInput {
            system_theme: Some(egui::Theme::Light),
            ..Default::default()
        };
        let _ = ctx.run(light, |_| {});
        assert_eq!(ctx.theme(), egui::Theme::Dark);
    }

    #[test]
    fn the_web_page_background_is_the_panel_colour() {
        let page = include_str!("../../../linkage-sim-rs/web/tools/magcoupler/index.html");
        let [r, g, b, _] = visuals().panel_fill.to_array();
        let colour = format!("background: #{r:02x}{g:02x}{b:02x};");
        assert!(page.contains(&colour), "index.html has no {colour}");
    }
}
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        assert_eq!(ctx.style().visuals, theme::cad_dark_visuals());
```

with:

```rust
        assert_eq!(ctx.style().visuals, theme::visuals());
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::theme`
  Expected: it does not compile. The new tests name what Step 3 adds: the errors are `error[E0425]: cannot find value` for each token (`PANEL2`, `BORDER`, `PANEL`, `MUTED`, `TEXT`, `LINK`, `BAD`, `BG`, `WARN`, `ACCENT`, `OK`), `cannot find function `visuals` in this scope` and `cannot find type `Color32` in this scope` (82 errors in all).

- [ ] **Step 3 (A, green): the look's two constants, the token table and the visuals, the page's colour.** The third block replaces everything of `theme.rs` above its test module.

**`magcoupling-rs/src/gui/look.rs`**

Create the file with:

```rust
//! The panel's own part of the look (UX overhaul spec 6.3). The panel takes its colours from
//! the host's `Visuals`, and an egui theme has no field for a brand accent or for a passing
//! check: the panel owns those two, at the values of the spec's token table (6.1). The
//! standalone app's theme (`app::theme`) lists them in its table under the same names.

/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
/// check: red means a failing check and nothing else.
pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(0xc8, 0x10, 0x2e);

/// A passing check, in a dark theme.
pub const OK: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xc3, 0x8a);
```

**`magcoupling-rs/src/gui/mod.rs`**

Replace:

```rust
mod panel;
pub mod pickers;
```

with:

```rust
pub mod look;
mod panel;
pub mod pickers;
```

**`magcoupling-rs/src/app/theme.rs`**

Replace:

```rust
//! The standalone app's theme: the linkage app's (spec M4 "Session": "Theme matches the
//! linkage app"). The linkage app applies its CAD dark visuals at start-up whatever the
//! system's light or dark preference; so does this app (decision M41-3), and the web page's
//! background is the same panel colour.
//!
//! The visuals are a copy of `cad_dark_visuals` in `linkage-sim-rs/src/gui/theme.rs` (the
//! crates are separate: linkage-sim-rs will depend on this one in M5, not the other way
//! round). A test compares the two function bodies, so they cannot drift apart. The panel
//! itself sets no theme: in M5 the linkage app's own theme applies.

/// Build the professional dark visuals inspired by CAD tools (SolidWorks, ANSYS).
pub fn cad_dark_visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();

    // Darker, more professional background tones
    v.panel_fill = egui::Color32::from_rgb(30, 32, 38);
    v.window_fill = egui::Color32::from_rgb(35, 37, 44);
    v.extreme_bg_color = egui::Color32::from_rgb(20, 22, 28);
    v.faint_bg_color = egui::Color32::from_rgb(38, 40, 48);

    // Accent color for selections and interactions
    v.selection.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Widget styling — more rounded, cleaner
    v.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(42, 44, 52);
    v.widgets.noninteractive.bg_stroke =
        egui::Stroke::new(0.5, egui::Color32::from_rgb(60, 62, 72));

    v.widgets.inactive.bg_fill = egui::Color32::from_rgb(50, 52, 62);
    v.widgets.inactive.bg_stroke = egui::Stroke::new(0.5, egui::Color32::from_rgb(70, 72, 82));

    v.widgets.hovered.bg_fill = egui::Color32::from_rgb(60, 65, 80);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(100, 140, 220));

    v.widgets.active.bg_fill = egui::Color32::from_rgb(40, 100, 200);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 160, 255));

    // Separator and window stroke
    v.window_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(55, 58, 68));

    v
}

/// The linkage app's spacing (`apply_cad_theme`): item spacing and button padding [points].
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// Applies the theme: dark whatever the system prefers, with the CAD visuals and the linkage
/// app's spacing.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = cad_dark_visuals();
        style.spacing.item_spacing = ITEM_SPACING;
        style.spacing.button_padding = BUTTON_PADDING;
    });
}
```

with:

```rust
//! The standalone app's theme: "Charcoal and crimson" (UX overhaul spec 6; owner decision
//! O-4). The app applies it at start-up whatever the system's light or dark preference, and
//! the web page's background is the panel colour.
//!
//! It is the calculator's own look. The linkage app keeps its CAD dark visuals, so this
//! reverses decision M41-3 ("Theme matches the linkage app": the spec's question Q-1, taken on
//! its recommendation). The panel itself sets no theme: it reads the host's `Visuals`, so an
//! app that embeds it (M5) shows it in that app's theme.

use egui::style::WidgetVisuals;
use egui::{Color32, CornerRadius, Stroke};

// The token table (spec 6.1). Every colour [`visuals`] sets is one of these.

/// Behind the drawing and the plots; a field's background.
pub const BG: Color32 = Color32::from_rgb(0x12, 0x14, 0x17);
/// The three columns and the top bar; the web page behind the canvas.
pub const PANEL: Color32 = Color32::from_rgb(0x18, 0x1b, 0x1f);
/// A button at rest, a menu, a tooltip.
pub const PANEL2: Color32 = Color32::from_rgb(0x1e, 0x22, 0x27);
/// Hairlines; the fill of a pressed or hovered button and of a selected choice.
pub const BORDER: Color32 = Color32::from_rgb(0x26, 0x2b, 0x31);
/// Values, headings, body text.
pub const TEXT: Color32 = Color32::from_rgb(0xe6, 0xe8, 0xeb);
/// Captions and hints (decision C-14: lightened from the mockup's #7c838c, which is under
/// 4.5:1 on [`PANEL2`]).
pub const MUTED: Color32 = Color32::from_rgb(0x8a, 0x91, 0x9a);
/// The brand accent (decision C-5), which the panel draws from its own constant.
pub const ACCENT: Color32 = crate::gui::look::ACCENT;
/// A failing check.
pub const BAD: Color32 = Color32::from_rgb(0xff, 0x4d, 0x57);
/// A caution.
pub const WARN: Color32 = Color32::from_rgb(0xe0, 0xa1, 0x00);
/// A passing check, which the panel draws from its own constant.
pub const OK: Color32 = crate::gui::look::OK;
/// A link, and the mark of what is selected or traced (decision C-13: egui's link blue, the
/// hue of the drawing's dimension lines; blue means "click me").
pub const LINK: Color32 = Color32::from_rgb(90, 170, 255);

/// The corner radius of buttons, fields, menus and tooltips [points].
pub const RADIUS: u8 = 4;

/// Item spacing and button padding [points]: the values the panel's layout was built on (the
/// linkage app's). The look changes no position.
pub const ITEM_SPACING: egui::Vec2 = egui::vec2(6.0, 4.0);
pub const BUTTON_PADDING: egui::Vec2 = egui::vec2(8.0, 4.0);

/// One widget state: its fill (a checkbox's box, a slider's rail), a button's fill and its
/// outline. Its text is in the text colour and its corners are [`RADIUS`].
fn state(state: &mut WidgetVisuals, fill: Color32, button: Color32, outline: Stroke) {
    state.bg_fill = fill;
    state.weak_bg_fill = button;
    state.bg_stroke = outline;
    state.fg_stroke.color = TEXT;
    state.corner_radius = CornerRadius::same(RADIUS);
}

/// The theme's visuals: egui's dark visuals with the tokens in place of its colours.
///
/// Body text, values and headings share [`TEXT`]; egui has no bold, so `strong()` no longer
/// sets a value apart from the text beside it (a second, heavier face is the cure if a later
/// slice needs one). `weak()` is [`MUTED`]. Borders are hairlines and nothing casts a shadow
/// (decision C-6).
pub fn visuals() -> egui::Visuals {
    let hairline = Stroke::new(1.0, BORDER);
    let radius = CornerRadius::same(RADIUS);
    let mut v = egui::Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL2;
    v.window_stroke = hairline;
    v.window_shadow = egui::Shadow::NONE;
    v.popup_shadow = egui::Shadow::NONE;
    v.window_corner_radius = radius;
    v.menu_corner_radius = radius;
    v.extreme_bg_color = BG;
    v.faint_bg_color = PANEL2;
    v.code_bg_color = PANEL2;
    v.weak_text_color = Some(MUTED);
    v.hyperlink_color = LINK;
    v.warn_fg_color = WARN;
    v.error_fg_color = BAD;
    // A selected choice is a filled chip with its text in the link blue; selected text in a
    // field sits on the same fill. The accent is not used here: it stays rare (decision C-5).
    v.selection.bg_fill = BORDER;
    v.selection.stroke = Stroke::new(1.0, LINK);
    state(&mut v.widgets.noninteractive, PANEL2, PANEL2, hairline);
    state(&mut v.widgets.inactive, BORDER, PANEL2, hairline);
    state(
        &mut v.widgets.hovered,
        BORDER,
        BORDER,
        Stroke::new(1.0, MUTED),
    );
    state(
        &mut v.widgets.active,
        BORDER,
        BORDER,
        Stroke::new(1.0, TEXT),
    );
    state(&mut v.widgets.open, PANEL2, PANEL2, hairline);
    v
}

/// Applies the theme: dark whatever the system prefers, with [`visuals`] and the spacing.
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.visuals = visuals();
        style.spacing.item_spacing = ITEM_SPACING;
        style.spacing.button_padding = BUTTON_PADDING;
    });
}
```

**`linkage-sim-rs/web/tools/magcoupler/index.html`**

Replace:

```html
            overflow: hidden;
            background: #1e2026;
```

with:

```html
            overflow: hidden;
            background: #181b1f;
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
//! `/tools/magcoupler/`). It applies the linkage app's theme ([`theme`]) and does the panel's
```

with:

```rust
//! `/tools/magcoupler/`). It applies the calculator's theme ([`theme`]) and does the panel's
```

Replace:

```rust
    /// session), with the linkage app's theme applied to `ctx`.
```

with:

```rust
    /// session), with the theme applied to `ctx`.
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib app::`
  Expected: PASS. `test result: ok. 25 passed; 0 failed`, the theme's five tests among them.

- [ ] **Step 5 (B, red): a passing check is the `ok` token.**

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    #[test]
    fn badge_colours_follow_the_theme() {
```

with:

```rust
    #[test]
    fn a_passing_check_is_the_ok_token_in_a_dark_theme() {
        // UX overhaul spec 6.3: the panel's own green takes the token's value (#4cc38a). A
        // light theme's green is not a token and stays.
        assert_eq!(
            Level::Good.color(&egui::Visuals::dark()),
            crate::gui::look::OK
        );
        assert_eq!(
            Level::Good.color(&egui::Visuals::light()),
            egui::Color32::from_rgb(20, 130, 60)
        );
    }

    #[test]
    fn badge_colours_follow_the_theme() {
```

- [ ] **Step 6: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib a_passing_check_is_the_ok_token`
  Expected: FAIL. `a_passing_check_is_the_ok_token_in_a_dark_theme` panics with `left: #5A_C8_78_FF`, `right: #4C_C3_8A_FF`: the badge is still the old green.

- [ ] **Step 7 (B, green): the badge takes the token, and the export keeps its green.** Without the `xlsx.rs` blocks the export's fill would follow the page: its test (renamed here; its assertions do not change) fails on `5AC878`.

**`magcoupling-rs/src/gui/dashboard.rs`**

Replace:

```rust
    /// The badge colour: green, the theme's warning colour, the theme's error colour.
    pub fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        match self {
            Level::Good if visuals.dark_mode => egui::Color32::from_rgb(90, 200, 120),
```

with:

```rust
    /// The badge colour: the panel's own green ([`crate::gui::look::OK`] in a dark theme), the
    /// theme's warning colour, the theme's error colour.
    pub fn color(self, visuals: &egui::Visuals) -> egui::Color32 {
        match self {
            Level::Good if visuals.dark_mode => crate::gui::look::OK,
```

**`magcoupling-rs/src/gui/xlsx.rs`**

Replace:

```rust
//! filled with the standalone page's badge colour, a fixed dark-theme palette (decision X-11;
```

with:

```rust
//! filled from a fixed palette, the badge colours of egui's dark visuals (decision X-11;
```

Replace:

```rust
/// The badge colour of `level` in egui's dark visuals, which the standalone page forces
/// (decision X-11): green, the warning amber, the error red. Black text reads on each. A fixed
/// palette: the linkage app's calculator window may draw in light visuals, whose badges differ
/// (green 14823C, amber FF6400), and the fills do not follow them.
pub fn fill(level: Level) -> Color {
    let color = level.color(&egui::Visuals::dark());
```

with:

```rust
/// The fill of `level` (decision X-11): green 5AC878, and the warning amber and the error red
/// of egui's dark visuals. Black text reads on each. A fixed palette: it follows neither the
/// standalone page's theme, whose badges moved to that theme's tokens (UX overhaul spec 6; the
/// exports' contents stay as they are, spec 12.1), nor a host's light visuals (green 14823C,
/// amber FF6400).
pub fn fill(level: Level) -> Color {
    let color = match level {
        Level::Good => egui::Color32::from_rgb(90, 200, 120),
        Level::Caution | Level::Bad => level.color(&egui::Visuals::dark()),
    };
```

Replace:

```rust
    fn the_level_fills_are_the_page_s_badge_colours() {
```

with:

```rust
    fn the_level_fills_are_a_fixed_palette() {
        // Renamed with the standalone page's own theme (UX overhaul spec 6): the fills were the
        // page's badge colours until then, and the export keeps them (spec 12.1).
```

- [ ] **Step 8: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- a_passing_check_is_the_ok_token the_level_fills_are_a_fixed_palette badge_colours_follow_the_theme`
  Expected: PASS, 3 tests.

- [ ] **Step 9 (C, red): the plots' axes.** The first two blocks turn the test helper `draw` into `draw_in`, which takes the context, plus a one-line `draw`.

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
    fn draw(kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput {
        let ctx = egui::Context::default();
```

with:

```rust
    fn draw(kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput {
        draw_in(&egui::Context::default(), kind, inputs)
    }

    /// [`draw`] in `ctx`, whose visuals the plot reads.
    fn draw_in(ctx: &egui::Context, kind: PlotKind, inputs: &DesignInputs) -> egui::FullOutput {
```

Replace:

```rust
        let mut output = None;
        for _ in 0..2 {
            output = Some(sized_frame(
                &ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results, &mut Readouts::default()),
```

with:

```rust
        let mut output = None;
        for _ in 0..2 {
            output = Some(sized_frame(
                ctx,
                egui::vec2(900.0, 600.0),
                Vec::new(),
                |ui| plot_ui(ui, kind, inputs, &results, &mut Readouts::default()),
```

Replace:

```rust
    #[test]
    fn the_legends_and_axes_use_the_on_screen_names() {
```

with:

```rust
    #[test]
    fn the_axes_take_the_weak_text_colour_and_the_readouts_do_not() {
        // egui_plot draws its grid, ticks and axis labels from the text colour. Under a theme
        // whose body text is as bright as its values (the standalone app's, UX overhaul spec 6)
        // the grid would be as loud as the curves: the plot hands it the weak colour.
        use crate::gui::test_support::text_color;
        let (text, weak) = (
            egui::Color32::from_rgb(250, 251, 252),
            egui::Color32::from_rgb(101, 102, 103),
        );
        let ctx = egui::Context::default();
        ctx.style_mut(|style| {
            style.visuals.widgets.noninteractive.fg_stroke.color = text;
            style.visuals.weak_text_color = Some(weak);
        });
        let output = draw_in(&ctx, PlotKind::TorqueTemperature, &DesignInputs::default());
        for label in ["Magnet temperature [°C]", "Torque [N·m]"] {
            assert_eq!(text_color(&output, label), Some(weak), "{label}");
        }
        // The line of values over the plot is a readout, not a caption.
        let readout = drawn_texts(&output)
            .into_iter()
            .find(|t| t.starts_with(&format!("{PULL_OUT}: ")))
            .expect("the pull-out's readout");
        assert_eq!(text_color(&output, &readout), Some(text));
    }

    #[test]
    fn the_legends_and_axes_use_the_on_screen_names() {
```

- [ ] **Step 10: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_axes_take_the_weak_text_colour`
  Expected: FAIL. `the_axes_take_the_weak_text_colour_and_the_readouts_do_not` panics with `assertion `left == right` failed: Magnet temperature [°C]`, `left: Some(#FA_FB_FC_FF)`, `right: Some(#65_66_67_FF)`: the axis label is in the text colour.

- [ ] **Step 11 (C, green): the plot hands egui_plot the weak colour.**

**`magcoupling-rs/src/gui/plots.rs`**

Replace:

```rust
    match kind {
        PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
        PlotKind::GapSweep => sweep_ui(
            ui,
            kind,
            "Corner gap [mm]",
            PULL_OUT,
            &results.gap_sweep,
            [results.model.corner_gap_mm, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::PoleSweep => sweep_ui(
            ui,
            kind,
            label_for("coupling.npole"),
            PULL_OUT_SMALLEST_APOTHEM,
            &results.pole_sweep,
            [inputs.coupling.npole as f64, results.model.pullout_Nm],
            results.model.required_floor_Nm,
        ),
        PlotKind::SlipHeating => slip_heating_ui(ui, results),
        PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
    }
```

with:

```rust
    // egui_plot draws the grid, the ticks and the axis labels from the text colour. Under a
    // theme whose body text is as bright as its values the grid is as loud as the curves, so
    // the plot takes the weak text colour for them. (A legend entry is a button: its text
    // keeps the button's colour.)
    ui.scope(|ui| {
        ui.visuals_mut().widgets.noninteractive.fg_stroke.color = ui.visuals().weak_text_color();
        match kind {
            PlotKind::TorqueTemperature => torque_temperature_ui(ui, inputs, results),
            PlotKind::GapSweep => sweep_ui(
                ui,
                kind,
                "Corner gap [mm]",
                PULL_OUT,
                &results.gap_sweep,
                [results.model.corner_gap_mm, results.model.pullout_Nm],
                results.model.required_floor_Nm,
            ),
            PlotKind::PoleSweep => sweep_ui(
                ui,
                kind,
                label_for("coupling.npole"),
                PULL_OUT_SMALLEST_APOTHEM,
                &results.pole_sweep,
                [inputs.coupling.npole as f64, results.model.pullout_Nm],
                results.model.required_floor_Nm,
            ),
            PlotKind::SlipHeating => slip_heating_ui(ui, results),
            PlotKind::TorqueAngle => torque_angle_ui(ui, inputs, results),
        }
    });
```

- [ ] **Step 12: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib gui::plots`
  Expected: PASS. `test result: ok. 19 passed; 0 failed`: every plot test, the new one among them.

- [ ] **Step 13: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
the sizing mode switch and the linkage app's theme (see
```

with:

```markdown
the sizing mode switch and the theme (the linkage app's then, the calculator's own since 2026-11; see
```

Replace:

```markdown
, `test_support.rs` (headless egui helpers
```

with:

```markdown
, `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries), `test_support.rs` (headless egui helpers
```

Replace:

```markdown
`app/theme.rs` (the linkage app's CAD dark visuals, forced dark)
```

with:

```markdown
`app/theme.rs` (the calculator's own theme, "Charcoal and crimson": the token table and the visuals built from it, forced dark)
```

Replace:

```markdown
  level filled with the standalone page's badge colour, a fixed dark-theme palette (green 5AC878,
  amber FF8F00, red FF0000). rust_xlsxwriter 0.99.1 writes
```

with:

```markdown
  level filled from a fixed palette, the badge colours of egui's dark visuals (green 5AC878,
  amber FF8F00, red FF0000); the fills do not follow the page's theme. rust_xlsxwriter 0.99.1 writes
```

Replace:

```markdown
- **Theme** (`app/theme.rs`): the standalone app applies the linkage app's CAD dark visuals and
  spacing, dark whatever the system prefers (M41-3; a test keeps the visuals equal to
  `linkage-sim-rs/src/gui/theme.rs`), and the web page's background is the same panel colour.
  The panel sets no theme: the host's applies.
```

with:

```markdown
- **Theme** (`app/theme.rs`; UX overhaul spec 6, slice B stage 1, BL-079): the standalone app
  applies its own theme, "Charcoal and crimson", dark whatever the system prefers. One token
  table: the surfaces `#121417` (behind the drawing and the plots, and in fields), `#181b1f` (the
  three columns and the top bar) and `#1e2227` (buttons, menus, tooltips), hairlines `#262b31`,
  text `#e6e8eb`, muted `#8a919a`, the accent `#c8102e`, and `#ff4d57`, `#e0a100` and `#4cc38a`
  for a failing check, a caution and a passing one; links keep egui's blue. `visuals()` is
  egui's dark visuals with the tokens in place of its colours: hairline borders, one corner
  radius, no shadows, and a selected choice as a chip on the hairline colour with its text in
  the link blue. Body text, values and headings share the text colour (egui has no bold, so
  `strong()` no longer sets a value apart from the text beside it); `weak()` is the muted
  colour. Every text token is at least 4.5:1 on every surface it is drawn on (a test computes
  it by the WCAG formula). The web page's background is the panel colour, and the spacing is
  the layout's own, so no position moved. This reverses M41-3 ("Theme matches the linkage app";
  the spec's question Q-1, taken on its recommendation): the linkage app keeps its visuals.
  The panel sets no theme: the host's applies. Two colours are the panel's own (`gui/look.rs`),
  because no egui theme has a field for them: the accent and a dark theme's pass green
  (`Level::Good`). In the plots the grid, the ticks and the axis labels take the weak text
  colour (egui_plot draws them from the text colour, which is now as bright as a curve). The
  spreadsheet export's fills keep their palette.
```

Replace:

```markdown
The app draws the whole panel as a page in the CAD dark theme and opens
```

with:

```markdown
The app draws the whole panel as a page in the calculator's theme and opens
```

Replace:

```markdown
the panel colour as its background; the visuals are `linkage-sim-rs`'s, function body for body; the theme stays dark when the system turns light |
```

with:

```markdown
the panel colour as its background; the tokens are the UX overhaul spec's table value for value, the visuals are that table field for field and egui's dark visuals in every other field, and every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures to one decimal (`the_tokens_are_the_spec_s_table`, `the_visuals_are_the_token_table`, `every_text_token_reads_on_every_surface_it_is_drawn_on`); the theme stays dark when the system turns light |
```

Replace:

```markdown
and a region a few points tall or none draws; a readout's hover names its path only with the spreadsheet details |
```

with:

```markdown
and a region a few points tall or none draws; a readout's hover names its path only with the spreadsheet details; the axis labels take the weak text colour and the readouts over the plot keep the text colour (`the_axes_take_the_weak_text_colour_and_the_readouts_do_not`) |
```

Replace:

```markdown
(`no_correction_s_title_says_it_is_not_approved`) |
```

with:

```markdown
(`no_correction_s_title_says_it_is_not_approved`); a passing check's badge is the look's `OK` in a dark theme and the old green in a light one (`a_passing_check_is_the_ok_token_in_a_dark_theme`) |
```

- [ ] **Step 14: The checks before a commit** (Global Constraints). Expected: `cargo test --features app` ends its lib run with `1037 passed; 0 failed; 6 ignored`; both clippy runs end without a warning.

- [ ] **Step 15: Commit.**

  Message (`.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T1.txt`), then the trailers:

  ```text
  feat(magcoupler): the calculator has its own theme, built from one token table (BL-079 T1)

  app::theme holds the UX overhaul design's tokens (section 6.1) and visuals(), egui's dark
  visuals with the tokens in place of its colours: charcoal surfaces, hairline borders, one
  corner radius, no shadows. The web page's background is the panel colour. This reverses
  decision M41-3 for the calculator alone (the design's question Q-1); the linkage app is not
  touched. A passing check's badge is the ok token; the spreadsheet export keeps its green.
  The plots take the weak text colour for their grid, ticks and axis labels, which egui_plot
  draws from the text colour. No position, string or engine line changes.

  Tests: the tokens are the design's table; the visuals are the tokens field for field and
  egui's everywhere else (this replaces the parity test with the linkage app); every text
  token is at least 4.5:1 on every surface it is drawn on; the pass green; the plots' axes.
  ```

  ```bash
  git add magcoupling-rs/src/app/theme.rs magcoupling-rs/src/app.rs magcoupling-rs/src/gui/look.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/xlsx.rs magcoupling-rs/src/gui/plots.rs linkage-sim-rs/web/tools/magcoupler/index.html magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T1.txt
  ```

### Task 2: The font

**Model:** `sonnet` (transcription, and one command that builds a file with a known hash).

**Files:**
- Create: `magcoupling-rs/assets/fonts/subset_inter.py`, `magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf` (built), `magcoupling-rs/assets/fonts/OFL.txt` (copied), `magcoupling-rs/src/app/fonts.rs`
- Modify: `magcoupling-rs/src/app.rs`, `magcoupling-rs/src/gui/test_support.rs`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `app::theme::apply` (Task 1), which `MagcouplingApp::new` already calls.
- Produces:
  - `app::fonts::INTER: &[u8]`, `app::fonts::INTER_NAME: &str` (`"Inter"`), `pub fn install(ctx: &egui::Context)`.
  - Tests only: `pub(crate) fn cut_holds(c: char) -> bool` in `app::fonts`.
  - `MagcouplingApp::new` calls `fonts::install(ctx)` after `theme::apply(ctx)`.

- [ ] **Step 1: The script that builds the cut.**

**`magcoupling-rs/assets/fonts/subset_inter.py`**

Create the file with:

```python
"""Builds the calculator's font: Inter Regular, cut to the characters the app draws, with
tabular digits as its default digits.

    python subset_inter.py Inter-Regular.ttf Inter-Regular.subset.ttf

Why a built file: egui cannot switch OpenType features, so the ten digits are remapped here to
Inter's tabular ones (its `tnum` feature, the digits only: the whole feature would also widen
every space, full stop, comma, colon, hyphen and bracket in running text). The app build runs
no font tools; it embeds the built file (`src/app/fonts.rs`).

Source: Inter 4.1, https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
  (SHA-256 9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e),
  its file extras/ttf/Inter-Regular.ttf
  (SHA-256 40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82).
Licence: SIL Open Font License 1.1 (OFL.txt beside this file, the zip's LICENSE.txt). Inter
  declares no Reserved Font Name, so the cut file keeps the name.
Tool: fontTools 4.51.0 on Python 3.12. The output does not depend on the date (the head table's
  timestamps are kept), so the same input and tool give the same bytes.
"""
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# What the app draws (UX overhaul spec 6.2): Latin, Greek, the maths and unit signs in use.
# A character outside these falls back to egui's built-in fonts.
UNICODES = (
    list(range(0x0020, 0x007F))  # Basic Latin
    + list(range(0x00A0, 0x0100))  # Latin-1: the degree sign, the middle dot, the micro sign
    + list(range(0x0370, 0x0400))  # Greek: the equations' symbols
    + list(range(0x2010, 0x2028))  # dashes, quotes, the bullet, the ellipsis
    + [0x2030, 0x2032, 0x2033]  # per mille, prime, double prime
    + list(range(0x2070, 0x209D))  # superscripts and subscripts
    + list(range(0x2190, 0x219A))  # arrows
    + [0x2202, 0x2206, 0x220F, 0x2211, 0x2212, 0x2215, 0x2219, 0x221A, 0x221E, 0x222B]
    + [0x2248, 0x2260, 0x2264, 0x2265]  # about equal, not equal, at most, at least
)


def main(source: str, target: str) -> None:
    font = TTFont(source, recalcTimestamp=False)
    order = set(font.getGlyphOrder())
    tabular = {name: f"{name}.tf" for name in DIGITS}
    missing = [glyph for glyph in tabular.values() if glyph not in order]
    if missing:
        sys.exit(f"{source} has no tabular digits: {missing}")
    for table in font["cmap"].tables:
        for codepoint, name in list(table.cmap.items()):
            if name in tabular:
                table.cmap[codepoint] = tabular[name]

    options = subset.Options()
    options.layout_features = []  # egui applies no GSUB or GPOS
    options.hinting = False  # egui's rasterizer does not hint
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14]  # the copyright and licence stay in the file
    options.notdef_outline = True
    options.recalc_timestamp = False
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=UNICODES)
    subsetter.subset(font)

    cmap = font.getBestCmap()
    widths = {font["hmtx"][cmap[ord(digit)]][0] for digit in "0123456789"}
    if len(widths) != 1:
        sys.exit(f"the digits have {len(widths)} advance widths, not one")
    font.save(target)
    print(f"{target}: {len(font.getGlyphOrder())} glyphs, {len(cmap)} characters, digit advance {widths.pop()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
```

- [ ] **Step 2: Build the cut and copy the licence.** From the worktree's root, in Git Bash. The downloaded zip and its two files go to a gitignored folder of the worktree; they are not committed.

  ```bash
  T=.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/font-build
  mkdir -p "$T"
  curl -sL -o "$T/Inter-4.1.zip" https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
  sha256sum "$T/Inter-4.1.zip"
  unzip -j -q "$T/Inter-4.1.zip" extras/ttf/Inter-Regular.ttf LICENSE.txt -d "$T"
  python -c "import fontTools; print(fontTools.version)"
  python magcoupling-rs/assets/fonts/subset_inter.py "$T/Inter-Regular.ttf" magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf
  cp "$T/LICENSE.txt" magcoupling-rs/assets/fonts/OFL.txt
  sha256sum magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf
  ```

  Expected, in order: the zip's hash `9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e`; `4.51.0`; the script's line `magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf: 435 glyphs, 386 characters, digit advance 1328`; the cut's hash `6345f33a4302725884b90eb123e13d041a00a158e8bb2b79d70cea0d7c35fb32` (30,684 bytes).

  If fontTools is another version, or there is no network: the main checkout holds the spike's copies in `C:\Users\Cole\source\repos\linkage_simulation\.superpowers\sdd\2026-11-18-magcoupler-overhaul-b1\font\` (`Inter-Regular.ttf`, `LICENSE.txt` and the built `Inter-Regular.subset.ttf`). Copy `Inter-Regular.ttf` and `LICENSE.txt` into `$T`, the built file to `magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf` and the licence to `magcoupling-rs/assets/fonts/OFL.txt`, and check the hash. If the hash of the file you have is not the one above, stop and report: do not commit another file.

- [ ] **Step 3 (red): the app installs the font, and the glyph guards hold every text to the cut.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
```

with:

```rust
    #[test]
    fn the_app_draws_its_text_in_inter() {
        // UX overhaul spec 6.2: the app installs its font; the panel alone does not.
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        let first = |ctx: &egui::Context| {
            ctx.fonts(|f| {
                f.lock().fonts.definitions().families[&egui::FontFamily::Proportional][0].clone()
            })
        };
        assert_eq!(first(&ctx), fonts::INTER_NAME);
        // A number keeps its width whatever its digits (the cut's tabular digits).
        let body = egui::TextStyle::Body.resolve(&ctx.style());
        let width = |digit| ctx.fonts(|f| f.glyph_width(&body, digit));
        assert_eq!(width('1'), width('8'));
        // A panel in a context no app set up keeps egui's fonts.
        let plain = egui::Context::default();
        let mut panel = MagcouplingPanel::new();
        let _ = plain.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.ui(ui));
        });
        assert_ne!(first(&plain), fonts::INTER_NAME);
    }

    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
```

**`magcoupling-rs/src/gui/test_support.rs`**

Replace:

```rust
pub(crate) fn assert_glyphs(ctx: &egui::Context, text: &str, what: &str) {
    let font = egui::FontId::proportional(14.0);
```

with:

```rust
///
/// With the app feature it also asserts that the standalone app's font holds the character
/// (UX overhaul spec 8.1, test 7): the app draws one outside its cut of Inter in a built-in
/// font, a different face in the middle of a word.
pub(crate) fn assert_glyphs(ctx: &egui::Context, text: &str, what: &str) {
    let font = egui::FontId::proportional(14.0);
```

Replace:

```rust
            "{what}: no glyph for {c:?} (U+{:04X}) in {text:?}",
            c as u32
```

with:

```rust
            "{what}: no glyph for {c:?} (U+{:04X}) in {text:?}",
            c as u32
        );
        #[cfg(feature = "app")]
        assert!(
            crate::app::fonts::cut_holds(c),
            "{what}: {c:?} (U+{:04X}) in {text:?} is not in the app's font: add it to UNICODES in assets/fonts/subset_inter.py and build the cut again",
            c as u32
```

- [ ] **Step 4: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_app_draws_its_text_in_inter`
  Expected: it does not compile. `error[E0433]: failed to resolve: could not find `fonts` in `app`` (the guard's call in `test_support.rs`) and `use of unresolved module or unlinked crate `fonts`` (the app's test).

- [ ] **Step 5 (green): the font module, and the app installs it.**

**`magcoupling-rs/src/app/fonts.rs`**

Create the file with:

```rust
//! The standalone app's font: Inter (UX overhaul spec 6.2), ahead of egui's built-in fonts.
//!
//! The file is a cut of Inter Regular, made once by `assets/fonts/subset_inter.py` and
//! committed with its licence (`assets/fonts/OFL.txt`, SIL Open Font License 1.1); the build
//! runs no font tools. Two things the cut does:
//!
//! - It holds the characters the app draws (Latin, Greek, the maths and unit signs in use),
//!   about 30 KB. A character outside it is drawn in a built-in font, which the glyph guards
//!   of the panel's tests refuse (`gui::test_support::assert_glyphs`).
//! - Its ten digits are Inter's tabular ones, so numbers keep their columns: egui cannot
//!   switch an OpenType feature on, so the file carries them as its default digits.
//!
//! One face. egui has no bold; a heavier face for numbers and headings is added only if a
//! later slice needs it. The panel itself installs no font: in another host it is drawn in
//! that host's fonts, so the typesetter's substitutions for egui's built-in fonts
//! (`gui::typeset::glyph_safe`) stay as they are.

use egui::epaint::text::{FontInsert, FontPriority, InsertFontFamily};

/// The cut of Inter Regular.
pub const INTER: &[u8] = include_bytes!("../../assets/fonts/Inter-Regular.subset.ttf");

/// Its name among egui's fonts.
pub const INTER_NAME: &str = "Inter";

/// Puts Inter first in the proportional family. egui's built-in fonts stay behind it, for a
/// character outside the cut, and the monospace family is not touched.
pub fn install(ctx: &egui::Context) {
    ctx.add_font(FontInsert::new(
        INTER_NAME,
        egui::FontData::from_static(INTER),
        vec![InsertFontFamily {
            family: egui::FontFamily::Proportional,
            priority: FontPriority::Highest,
        }],
    ));
}

/// A context whose only font is the cut.
#[cfg(test)]
fn inter_alone() -> egui::Context {
    let mut definitions = egui::FontDefinitions::empty();
    definitions.font_data.insert(
        INTER_NAME.to_owned(),
        std::sync::Arc::new(egui::FontData::from_static(INTER)),
    );
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        definitions
            .families
            .insert(family, vec![INTER_NAME.to_owned()]);
    }
    let ctx = egui::Context::default();
    ctx.set_fonts(definitions);
    // egui loads fonts on a context's first frame.
    let _ = ctx.run(egui::RawInput::default(), |_| {});
    ctx
}

/// Whether the cut itself holds `c`, whatever egui's built-in fonts hold: the panel's glyph
/// guards ask it of every text (`gui::test_support::assert_glyphs`). The font is loaded once
/// per test thread. egui reports a font's own replacement character as missing; with the cut
/// alone that is the question mark, which the cut has.
#[cfg(test)]
pub(crate) fn cut_holds(c: char) -> bool {
    thread_local! {
        static ALONE: egui::Context = inter_alone();
    }
    c == '?' || ALONE.with(|ctx| ctx.fonts(|f| f.has_glyph(&egui::FontId::proportional(14.0), c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{FontFamily, FontId};

    /// A context with the font installed and loaded.
    fn installed() -> egui::Context {
        let ctx = egui::Context::default();
        install(&ctx);
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        ctx
    }

    #[test]
    fn inter_leads_the_proportional_family_and_the_built_in_fonts_follow() {
        let definitions = installed().fonts(|f| f.lock().fonts.definitions().clone());
        let built_in = egui::FontDefinitions::default();
        let mut proportional = vec![INTER_NAME.to_owned()];
        proportional.extend(built_in.families[&FontFamily::Proportional].iter().cloned());
        assert_eq!(
            definitions.families[&FontFamily::Proportional],
            proportional
        );
        assert_eq!(
            definitions.families[&FontFamily::Monospace],
            built_in.families[&FontFamily::Monospace]
        );
    }

    #[test]
    fn the_ten_digits_have_one_advance_width() {
        // UX overhaul spec 8.1, test 7, at the sizes the app draws numbers in (small text, the
        // drawing's tags, body text, a term, a heading and the open equation), and at one, one
        // and a half and two pixels per point: a zoomed browser, a dense display.
        for pixels_per_point in [1.0, 1.5, 2.0] {
            let (ctx, alone) = (installed(), inter_alone());
            for ctx in [&ctx, &alone] {
                ctx.set_pixels_per_point(pixels_per_point);
                let _ = ctx.run(egui::RawInput::default(), |_| {});
            }
            for size in [9.0, 11.0, 12.0, 12.5, 15.0, 18.0, 22.0] {
                let font = FontId::proportional(size);
                let widths = |ctx: &egui::Context| -> Vec<f32> {
                    "0123456789"
                        .chars()
                        .map(|digit| ctx.fonts(|f| f.glyph_width(&font, digit)))
                        .collect()
                };
                let drawn = widths(&ctx);
                assert!(
                    drawn.iter().all(|&w| w > 0.0 && w == drawn[0]),
                    "{pixels_per_point} {size}: {drawn:?}"
                );
                // They are the cut's digits, not a built-in font's.
                assert_eq!(drawn, widths(&alone), "{pixels_per_point} {size}");
            }
        }
    }

    #[test]
    fn the_cut_holds_the_characters_the_app_draws() {
        // Printable ASCII, and every other character of the app's texts that Inter has (the
        // list the cut was made for, `assets/fonts/subset_inter.py`).
        let ascii = (0x21u8..0x7f).map(char::from);
        let others = "§¯°±²³µ·×Øé÷ΔΣΩαβδεηθλμνπρστφωϑ–—…•⁴⁵⁶⁻→−√∞∫≈≠≤≥".chars();
        let missing: String = ascii.chain(others).filter(|&c| !cut_holds(c)).collect();
        assert_eq!(missing, "", "not in the cut");
        // The check can fail: Inter has no "element of" sign, and the cut leaves Cyrillic out.
        assert!(!cut_holds('\u{2208}') && !cut_holds('\u{0416}'));
    }
}
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
//! `/tools/magcoupler/`). It applies the calculator's theme ([`theme`]) and does the panel's
//! platform work ([`files`]): saving files and picking a design file. The web entry also
//! opens a share link's design (`?m=`).

pub mod files;
```

with:

```rust
//! `/tools/magcoupler/`). It applies the calculator's theme ([`theme`]) and font ([`fonts`])
//! and does the panel's platform work ([`files`]): saving files and picking a design file.
//! The web entry also opens a share link's design (`?m=`).

pub mod files;
pub mod fonts;
```

Replace:

```rust
    /// session), with the theme applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
```

with:

```rust
    /// session), with the theme and the font applied to `ctx`.
    pub fn new(ctx: &egui::Context) -> Self {
        theme::apply(ctx);
        fonts::install(ctx);
```

- [ ] **Step 6: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`
  Expected: `1041 passed; 0 failed; 6 ignored`. The run includes every glyph guard of the panel (`every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp`, the notes', the guides', the torque card's, the diagrams'): each now also holds its texts to the cut.

- [ ] **Step 7: See the font's own tests bite.** The three tests of `fonts.rs` and the app's test pass as soon as the module exists. Put the uncut font in the cut's place, run them, and put the cut back:

  ```bash
  T=.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/font-build
  cp magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf "$T/cut.ttf"
  cp "$T/Inter-Regular.ttf" magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- fonts:: the_app_draws_its_text_in_inter
  cp "$T/cut.ttf" magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf
  sha256sum magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf
  ```

  Expected: three of the four fail. `the_ten_digits_have_one_advance_width` prints ten different widths (the one is the narrowest), `the_app_draws_its_text_in_inter` fails on the widths of 1 and 8, and `the_cut_holds_the_characters_the_app_draws` fails because the uncut font holds Cyrillic. The last line is the cut's hash again; if it is not, build the cut again as in Step 2.

- [ ] **Step 8: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
forced dark), `app/files.rs` (saving
```

with:

```markdown
forced dark), `app/fonts.rs` (the app's font: a cut of Inter Regular with tabular digits, ahead of egui's built-in fonts), `app/files.rs` (saving
```

Replace:

```markdown
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
```

with:

```markdown
| `src/bin/` | Feature `app`: `magcoupling_app.rs` (native window), `magcoupling_web.rs` (wasm32 entry, eframe WebRunner; opens a `?m=` share link) |
| `assets/fonts/` | The app's font (feature `app`): `Inter-Regular.subset.ttf`, the cut `src/app/fonts.rs` embeds; `subset_inter.py`, the command that built it (fontTools; the build does not run it); `OFL.txt`, Inter's licence |
```

Replace:

```markdown
they are drawn as θ, ¯ and ~ (`glyph_safe`, M43-6), the markup and the notes unchanged; a test
```

with:

```markdown
they are drawn as θ, ¯ and ~ (`glyph_safe`, M43-6), the markup and the notes unchanged (the
  standalone app's own font has the first two; the substitutions stay, since the panel does not
  know its host's fonts); a test
```

Replace:

```markdown
  spreadsheet export's fills keep their palette.
```

with:

```markdown
  spreadsheet export's fills keep their palette.
- **Font** (`app/fonts.rs`, `assets/fonts/`; UX overhaul spec 6.2): the standalone app draws its
  text in Inter, put first in the proportional family; egui's built-in fonts stay behind it and
  stay the monospace family. The file is a cut of Inter Regular (30,684 bytes: Latin, Greek and
  the maths and unit signs the app draws) whose ten digits are Inter's tabular ones, so a number
  keeps its width whatever its digits. egui cannot switch an OpenType feature on, so the file
  carries them as its default digits, and the digits only: the whole `tnum` feature would also
  widen every space, full stop, comma and bracket in running text. `assets/fonts/subset_inter.py`
  built the file once from Inter 4.1 with fontTools (its docstring has the source, the hashes and
  the command; the same input gives the same bytes), `OFL.txt` is the licence, and the app build
  runs no font tools. With the app feature the panel's glyph guards
  (`test_support::assert_glyphs`) also hold every text to the cut: a character outside it would
  be drawn in a built-in font, a different face in the middle of a word, and the cure is to add
  it to the script's list and build the cut again. One face: a heavier one for numbers and
  headings waits for a slice that needs it. The panel installs no font: another host draws it in
  its own.
```

Replace:

```markdown
; the theme stays dark when the system turns light |
```

with:

```markdown
; the app's font: Inter leads the proportional family and the built-in fonts follow, the ten digits have one advance width at seven sizes and are the cut's own, the cut holds printable ASCII and the other characters the app's texts use and not a character Inter lacks, and the app installs it while a panel in a plain context keeps egui's fonts (`src/app/fonts.rs`; `the_app_draws_its_text_in_inter`); the theme stays dark when the system turns light |
```

Replace:

```markdown
the bottom of its slider's 40.00 to 85.41 mm) |
```

with:

```markdown
the bottom of its slider's 40.00 to 85.41 mm); with the app feature `assert_glyphs`, which every glyph guard of the panel calls, also holds each character to the standalone app's cut of Inter (`app::fonts::cut_holds`) |
```

- [ ] **Step 9: The checks before a commit.** Expected: the lib run ends `1041 passed; 0 failed; 6 ignored`. Then the web build's code compiles with the embedded file:

  ```bash
  cargo clippy --manifest-path magcoupling-rs/Cargo.toml --features app --bin magcoupling-web --target wasm32-unknown-unknown -- -D warnings
  ```

  Expected: no warning.

- [ ] **Step 10: Commit.**

  ```text
  feat(magcoupler): the app draws its text in Inter, with tabular digits (BL-079 T2)

  app::fonts embeds a cut of Inter Regular (30,684 bytes: Latin, Greek, the maths and unit
  signs the app draws) and puts it ahead of egui's built-in fonts, which stay as the fallback.
  The cut's ten digits are Inter's tabular ones, so a number keeps its width whatever its
  digits; the rest of the tnum feature is left out, since it also widens spaces and
  punctuation. assets/fonts/subset_inter.py built the file once with fontTools 4.51.0 (the
  same input gives the same bytes); OFL.txt is the licence. The panel installs no font.

  Tests: Inter leads the proportional family; the digits have one advance width at seven
  sizes and three pixel densities; the cut holds the characters the app draws; and with the
  app feature every glyph guard of the panel holds its texts to the cut.
  ```

  ```bash
  git add magcoupling-rs/assets/fonts/subset_inter.py magcoupling-rs/assets/fonts/Inter-Regular.subset.ttf magcoupling-rs/assets/fonts/OFL.txt magcoupling-rs/src/app/fonts.rs magcoupling-rs/src/app.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T2.txt
  ```

### Task 3: The accent's two marks

**Model:** `sonnet` (transcription).

**Files:**
- Modify: `magcoupling-rs/src/gui/look.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `gui::look::ACCENT` (Task 1); `app::theme::BAD` (Task 1).
- Produces:
  - `gui::look::MARK_SIDE: f32` (8.0), `gui::look::UNDERLINE_WIDTH: f32` (2.0).
  - `pub fn mark_ui(ui: &mut egui::Ui)`: the brand mark, an accent square at the middle of a heading's height.
  - `pub fn tab_ui(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response`: a tab; the selected one has the accent line under it.
  - In the app's test module: `fn frame_output(ctx: &egui::Context, app: &mut MagcouplingApp, events: Vec<egui::Event>) -> egui::FullOutput`.

- [ ] **Step 1 (red): the tests.** In `panel.rs` the first block adds `flat_shapes` and `text_color` to the test module's imports. In `app.rs` the first three blocks turn `frame_with` into `frame_output` plus a short `frame_with`.

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, hot_day, hub_overstressed,
        key_event, key_tap, last_characters, no_back_iron, only_the_clamp_fails,
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
        sized_frame, sized_frame_at, smoke_inputs, text_center, text_clip_rect, text_rect,
        text_rects, text_shapes, tooltips_at_once, with_poles,
```

with:

```rust
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, flat_shapes, hot_day,
        hub_overstressed, key_event, key_tap, last_characters, no_back_iron, only_the_clamp_fails,
        only_the_joint_fails, passing_design, primary_button, row_line, select_all, short_magnets,
        sized_frame, sized_frame_at, smoke_inputs, text_center, text_clip_rect, text_color,
        text_rect, text_rects, text_shapes, tooltips_at_once, with_poles,
```

Replace:

```rust
    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

with:

```rust
    /// The line segments drawn in the brand accent, each as (left, right, y).
    fn accent_lines(output: &egui::FullOutput) -> Vec<(f32, f32, f32)> {
        flat_shapes(output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::LineSegment { points, stroke } if stroke.color == look::ACCENT => {
                    Some((points[0].x, points[1].x, points[0].y))
                }
                _ => None,
            })
            .collect()
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
    fn the_active_tab_is_underlined_in_the_accent_and_no_other() {
        // UX overhaul spec 6.1 (the accent's role) and decision C-5: one line, under the active
        // tab's label, as wide as the tab.
        let under = |output: &egui::FullOutput, label: egui::Rect| {
            let lines = accent_lines(output);
            assert_eq!(lines.len(), 1, "{lines:?}");
            let (left, right, y) = lines[0];
            assert!(
                left <= label.left() && label.right() <= right,
                "{lines:?} {label:?}"
            );
            assert!(
                label.bottom() <= y && y <= label.bottom() + 6.0,
                "{lines:?} {label:?}"
            );
        };
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        // The Size tab's name is drawn twice: the left side's heading, then the tab.
        let size_tab = *text_rects(&output, CentreView::SizeAndTorque.label())
            .last()
            .expect("the tab");
        under(&output, size_tab);
        // A tab still answers a click, and the line follows it on the next frame.
        let geometry = CentreView::Geometry.label();
        harness.click_text(geometry);
        assert_eq!(harness.panel.centre, CentreView::Geometry);
        let output = harness.frame(Vec::new());
        under(&output, text_rect(&output, geometry).expect("the tab"));
        // The active tab's label is in the strong text colour, another's in the weak one.
        let visuals = harness.ctx.style().visuals.clone();
        assert_eq!(
            text_color(&output, geometry),
            Some(visuals.strong_text_color())
        );
        assert_eq!(
            text_color(&output, PlotKind::GapSweep.label()),
            Some(visuals.weak_text_color())
        );
        // The row wraps at this width: a tab on its second line is underlined where it is.
        let last = CentreView::Results.label();
        harness.click_text(last);
        let output = harness.frame(Vec::new());
        let label = text_rect(&output, last).expect("the tab");
        assert!(label.top() > size_tab.bottom(), "{label:?} {size_tab:?}");
        under(&output, label);
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
    fn the_brand_mark_is_one_accent_square_before_the_title() {
        let mut harness = Harness::first_view();
        let output = harness.frame(Vec::new());
        let marks: Vec<egui::Rect> = flat_shapes(&output)
            .into_iter()
            .filter_map(|shape| match shape {
                egui::Shape::Rect(rect) if rect.fill == look::ACCENT => Some(rect.rect),
                _ => None,
            })
            .collect();
        assert_eq!(marks.len(), 1, "{marks:?}");
        let mark = marks[0];
        assert_eq!(mark.size(), egui::Vec2::splat(look::MARK_SIDE));
        // The title's rect starts at the row's left edge (a wrapping label is indented to
        // where its text goes): its first glyph marks where the title begins.
        let (_, title) = text_shapes(&output, HEADING).next().expect("the title");
        let first = title.galley.rows[0].glyphs[0].pos.x + title.pos.x;
        assert!(mark.right() < first, "{mark:?} {first}");
        let middle = title.galley.rect.translate(title.pos.to_vec2()).center().y;
        assert!((mark.center().y - middle).abs() <= 2.0, "{mark:?} {middle}");
    }

    #[test]
    fn save_design_queues_the_design_file_and_load_design_asks_for_one() {
```

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    /// One frame of `app` on the laptop screen with `events`; the texts it drew.
    fn frame_with(
```

with:

```rust
    /// One frame of `app` on the laptop screen with `events`.
    fn frame_output(
```

Replace:

```rust
        events: Vec<egui::Event>,
    ) -> Vec<String> {
```

with:

```rust
        events: Vec<egui::Event>,
    ) -> egui::FullOutput {
```

Replace:

```rust
        drawn_texts(&ctx.run(input, |ctx| app.ui(ctx)))
```

with:

```rust
        ctx.run(input, |ctx| app.ui(ctx))
    }

    /// [`frame_output`]; the texts it drew.
    fn frame_with(
        ctx: &egui::Context,
        app: &mut MagcouplingApp,
        events: Vec<egui::Event>,
    ) -> Vec<String> {
        drawn_texts(&frame_output(ctx, app, events))
```

Replace:

```rust
    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
```

with:

```rust
    #[test]
    fn the_accent_marks_the_brand_and_the_active_tab_and_nothing_else() {
        // Decision C-5, in the app's own theme: crimson is never a text colour and never on a
        // value or a check. At the starting design and at a failing one the accent is on two
        // shapes, the brand mark and the line under the active tab.
        use crate::gui::look::ACCENT;
        use crate::gui::test_support::flat_shapes;
        let ctx = egui::Context::default();
        let mut app = MagcouplingApp::new(&ctx);
        frame_texts(&ctx, &mut app);
        for failing in [false, true] {
            if failing {
                app.panel.open_example_design();
                assert!(frame_texts(&ctx, &mut app).contains(&EXAMPLE_STATUS.to_owned()));
            }
            let output = frame_output(&ctx, &mut app, Vec::new());
            let mut marks = 0;
            for shape in flat_shapes(&output) {
                match shape {
                    egui::Shape::Text(text) => {
                        let mut colours = vec![text.fallback_color];
                        colours.extend(text.override_text_color);
                        colours.extend(text.galley.job.sections.iter().map(|s| s.format.color));
                        assert!(!colours.contains(&ACCENT), "{:?}", text.galley.text());
                    }
                    egui::Shape::Rect(rect) if rect.fill == ACCENT => marks += 1,
                    egui::Shape::LineSegment { stroke, .. } if stroke.color == ACCENT => marks += 1,
                    egui::Shape::Circle(circle) => assert_ne!(circle.fill, ACCENT),
                    _ => {}
                }
            }
            assert_eq!(marks, 2, "failing: {failing}");
        }
        // A failing check is the theme's red, which is not the accent.
        assert_ne!(theme::BAD, ACCENT);
    }

    #[test]
    fn the_starting_design_passes_every_check_with_no_caution_and_no_warning() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_brand_mark`
  Expected: it does not compile. `error[E0433]: failed to resolve: use of unresolved module or unlinked crate `look`` in the panel's tests (the panel does not import `look` yet), three times.

- [ ] **Step 3 (green): the mark and the tab, and the panel draws them.**

**`magcoupling-rs/src/gui/look.rs`**

Replace:

```rust

/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
```

with:

```rust
//!
//! The accent marks two things and no more: the brand mark before the title ([`mark_ui`]) and
//! the active tab of the centre's row ([`tab_ui`]).

/// The brand accent, crimson (decision C-5). Never a text colour, and never on a value or a
```

Replace:

```rust
pub const OK: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xc3, 0x8a);

```

with:

```rust
pub const OK: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xc3, 0x8a);

/// The side of the brand mark [points].
pub const MARK_SIDE: f32 = 8.0;

/// The width of the line under the active tab [points].
pub const UNDERLINE_WIDTH: f32 = 2.0;

/// The brand mark: a small square in the accent, at the middle of a heading's height. The
/// title follows it.
pub fn mark_ui(ui: &mut egui::Ui) {
    let height = ui.text_style_height(&egui::TextStyle::Heading);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(MARK_SIDE, height), egui::Sense::hover());
    let square = egui::Rect::from_center_size(rect.center(), egui::Vec2::splat(MARK_SIDE));
    ui.painter().rect_filled(square, 2.0, ACCENT);
}

/// A tab of the centre's row. The active tab is marked by a line in the accent under its
/// label, not by a filled chip, and its label is in the strong text colour; the other tabs
/// are in the weak text colour until the pointer is on them.
pub fn tab_ui(ui: &mut egui::Ui, selected: bool, label: &str) -> egui::Response {
    let visuals = ui.visuals_mut();
    let was = (visuals.selection, visuals.widgets.inactive);
    visuals.selection.bg_fill = egui::Color32::TRANSPARENT;
    visuals.selection.stroke.color = visuals.widgets.active.fg_stroke.color;
    visuals.widgets.inactive.bg_stroke = egui::Stroke::NONE;
    visuals.widgets.inactive.fg_stroke.color = visuals.weak_text_color();
    let response = ui.selectable_label(selected, label);
    let visuals = ui.visuals_mut();
    (visuals.selection, visuals.widgets.inactive) = was;
    if selected {
        ui.painter().hline(
            response.rect.x_range(),
            response.rect.bottom(),
            egui::Stroke::new(UNDERLINE_WIDTH, ACCENT),
        );
    }
    response
}

```

**`magcoupling-rs/src/gui/panel.rs`**

Replace:

```rust
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
```

with:

```rust
use crate::gui::look;
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
```

Replace:

```rust
                    ui.selectable_value(&mut self.centre, view, view.label());
```

with:

```rust
                    if look::tab_ui(ui, self.centre == view, view.label()).clicked() {
                        self.centre = view;
                    }
```

Replace:

```rust
            ui.heading(HEADING);
            ui.separator();
```

with:

```rust
            look::mark_ui(ui);
            ui.heading(HEADING);
            ui.separator();
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_brand_mark the_active_tab the_accent_marks`
  Expected: PASS, 3 tests.

- [ ] **Step 5: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
, `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries), `test_support.rs`
```

with:

```markdown
, `look.rs` (the panel's own part of the look: the brand accent and the pass green, the two colours no egui theme carries; `mark_ui`, the brand mark before the title; `tab_ui`, a tab of the centre's row, underlined in the accent when it is the active one), `test_support.rs`
```

Replace:

```markdown
  (`Level::Good`). In the plots the grid,
```

with:

```markdown
  (`Level::Good`). The accent marks two things and no more (decision C-5): the brand mark, a
  small square before the title, and a line under the active tab of the centre's row, whose
  name is in the strong text colour while the other tabs' are muted (`look::mark_ui`,
  `look::tab_ui`). It is never a text colour and never on a value or a check: red means a
  failing check and nothing else. In the plots the grid,
```

Replace:

```markdown
(`the_card_s_measured_height_is_the_drawn_height`) |
```

with:

```markdown
(`the_card_s_measured_height_is_the_drawn_height`); the look: one line in the accent, as wide as the active tab and just under its name, which follows a click to another tab, the active tab's name in the strong text colour and another's in the weak one, and one accent square of 8 points before the title's first glyph, at the title's height (`the_active_tab_is_underlined_in_the_accent_and_no_other`, `the_brand_mark_is_one_accent_square_before_the_title`) |
```

Replace:

```markdown
; the theme stays dark when the system turns light |
```

with:

```markdown
; on the app's page, at the starting design and at the failing example design, the accent is on two shapes (the brand mark and the active tab's line), on no text and on no badge (`the_accent_marks_the_brand_and_the_active_tab_and_nothing_else`); the theme stays dark when the system turns light |
```

- [ ] **Step 6: The checks before a commit.** Expected: the lib run ends `1044 passed; 0 failed; 6 ignored`. Every earlier panel test still passes: the tabs answer the same clicks at the same places.

- [ ] **Step 7: Commit.**

  ```text
  feat(magcoupler): the accent marks the brand and the active tab, and nothing else (BL-079 T3)

  gui::look draws the brand mark, an 8-point crimson square before the title, and the centre
  row's tabs: the active tab has a 2-point crimson line under its name, in the strong text
  colour, and the other tabs' names are muted until the pointer is on them. The accent is on
  no text, value or check (the design's decision C-5): a failing check is the theme's red.

  Tests: one accent line, under the active tab, which follows a click, on the row's second
  line too; one accent square before the title; on the app's page the accent is on those two
  shapes and nothing else, at the starting design and at the failing example design.
  ```

  ```bash
  git add magcoupling-rs/src/gui/look.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/app.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T3.txt
  ```

### Task 4: The hub page's card

**Model:** `sonnet` (transcription).

**Files:**
- Modify: `linkage-sim-rs/web/tools/index.html`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: nothing a later task uses.

- [ ] **Step 1 (red): the test.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
    #[test]
    fn the_canvas_id_matches_the_web_page() {
```

with:

```rust
    #[test]
    fn the_hub_page_does_not_name_the_paused_clamp_section() {
        // UX overhaul spec 9.2 and 8.3: no "clamp" on screen while the section is paused
        // (`engine::clamps::PAUSED`), and the hub's card for the calculator is a screen too.
        let hub = include_str!("../../linkage-sim-rs/web/tools/index.html");
        assert!(hub.contains("Magnetic Coupling Calculator"));
        assert!(!hub.to_lowercase().contains("clamp"));
    }

    #[test]
    fn the_canvas_id_matches_the_web_page() {
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_hub_page`
  Expected: FAIL. `the_hub_page_does_not_name_the_paused_clamp_section` panics with `assertion failed: !hub.to_lowercase().contains("clamp")`.

- [ ] **Step 3 (green): the card loses the word.**

**`linkage-sim-rs/web/tools/index.html`**

Replace:

```html
            <p>Coaxial magnetic slip couplings: torque, end effects, temperature, clamps, and design checks.</p>
```

with:

```html
            <p>Coaxial magnetic slip couplings: torque, end effects, temperature, and design checks.</p>
```

- [ ] **Step 4: Run to verify it passes.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_hub_page`
  Expected: PASS.

- [ ] **Step 5: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
; the theme stays dark when the system turns light |
```

with:

```markdown
; the hub page's card for the calculator does not name the paused clamp section (`the_hub_page_does_not_name_the_paused_clamp_section`); the theme stays dark when the system turns light |
```

- [ ] **Step 6: The checks before a commit.** Expected: the lib run ends `1045 passed; 0 failed; 6 ignored`.

- [ ] **Step 7: Commit.**

  ```text
  fix(web): the hub page's card for the calculator does not list the paused clamps (BL-079 T4)

  The card read "torque, end effects, temperature, clamps, and design checks" while the
  shaft-clamp section is paused and no screen of the calculator names it (the UX overhaul
  design's sections 8.3 and 9.2). A test reads the page.
  ```

  ```bash
  git add linkage-sim-rs/web/tools/index.html magcoupling-rs/src/app.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T4.txt
  ```

### Task 5: gui-smoke: the look

**Model:** `sonnet` (transcription).

**Files:**
- Modify: `.claude/workflows/gui-smoke.js`, `linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`, `magcoupling-rs/src/app.rs`, `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `app::theme::PANEL` (Task 1).
- Produces: the calculator's smoke result gains the boolean `look`; the script's constant `PAGE_BACKGROUND`.

- [ ] **Step 1 (red): the verdict test's list of checks.**

**`linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`**

Replace:

```js
const CHECKS = ['canvas_present', 'starting_design', 'first_view', 'details_toggle',
```

with:

```js
const CHECKS = ['canvas_present', 'starting_design', 'look', 'first_view', 'details_toggle',
```

- [ ] **Step 2: Run to verify it fails.**

  Run: `node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs`
  Expected: its last line is `1 of 76 checks fail`, and the failing line is `not ok - the calculator's checks: look false with the agent's passed true fails the run: passed=true within=true frame_check=false`.

- [ ] **Step 3 (red): the app test holds the script to the app.**

**`magcoupling-rs/src/app.rs`**

Replace:

```rust
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
```

with:

```rust
        // UX overhaul slice B, stage 1 (spec 6): on the clean page the script reads the page's
        // background, typed once as PAGE_BACKGROUND and equal to the theme's panel colour as a
        // browser reports it, and judges the look in the screenshot: the brand mark before the
        // title and the line under the active tab.
        {
            let [r, g, b, _] = theme::PANEL.to_array();
            let background = format!("rgb({r}, {g}, {b})");
            assert_eq!(
                quoted("PAGE_BACKGROUND"),
                background,
                "the script's PAGE_BACKGROUND"
            );
            assert_eq!(
                script.matches(background.as_str()).count(),
                1,
                "the smoke prompt types the page's background once, in PAGE_BACKGROUND"
            );
            let clean_page = between("starting_design=true", "it carries a share link");
            for (what, text) in [
                ("judges the look", "look=true only if"),
                (
                    "reads the page's background",
                    "getComputedStyle(document.body).backgroundColor",
                ),
                ("names the title", "Magnetic coupling calculator"),
                (
                    "names the active tab",
                    crate::gui::CentreView::SizeAndTorque.label(),
                ),
            ] {
                assert!(
                    clean_page.contains(text),
                    "the clean page's step {what}: {text}"
                );
            }
        }
        // The app opens the link and the solve succeeds after its debounce.
        let ctx = egui::Context::default();
```

- [ ] **Step 4: Run to verify it fails.**

  Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test`
  Expected: FAIL. `the_smoke_test_share_link_opens_its_design` panics with `no PAGE_BACKGROUND`.

- [ ] **Step 5 (green): the script.** Seven replacements, in this order. The prompt's lines are long: each old text below is a part of its line that occurs once. The script's strings are in single quotes, so no new text has an apostrophe outside a back-quoted prompt.

**`.claude/workflows/gui-smoke.js`**

Replace:

```js
on the calculator the clean address opens the starting design, a Why link opens a note,
```

with:

```js
on the calculator the clean address opens the starting design in its own look (a charcoal page, the brand mark, the underlined tab, the app font), a Why link opens a note,
```

Replace:

```js
required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'share_link_loaded',
```

with:

```js
required: ['passed', 'canvas_present', 'console_errors', 'starting_design', 'look', 'share_link_loaded',
```

Replace:

```js
    starting_design: { type: 'boolean' },
    address_follows: { type: 'boolean' },
```

with:

```js
    starting_design: { type: 'boolean' },
    look: { type: 'boolean' },
    address_follows: { type: 'boolean' },
```

Replace:

```js
const STARTING_STATUS = 'Starting design: passes all 9 checks'
```

with:

```js
const STARTING_STATUS = 'Starting design: passes all 9 checks'

// The page's background as a browser reports it: the panel colour of the calculator's theme
// (UX overhaul spec 6.1, #181b1f), typed once here. app.rs's smoke test holds it equal to
// app::theme::PANEL.
const PAGE_BACKGROUND = 'rgb(24, 27, 31)'
```

Replace:

```js
// address, with no share link: it opens the starting design and stays clean. Then
```

with:

```js
// address, with no share link: it opens the starting design and stays clean, and (slice B,
// stage 1, spec 6) the page wears the calculator's own look: its background is the theme's
// panel colour, read from the page, and the screenshot shows the brand mark before the title,
// the line under the active tab and text in the app's font. Then
```

Replace:

```js
and the code returned an empty string (at the starting design the address stays clean). Collect the console messages at level "info" (this page's: the next step navigates away).
```

with:

```js
and the code returned an empty string (at the starting design the address stays clean). Then the look: read the page's background with browser_run_code_unsafe, code: async (page) => page.evaluate(() => getComputedStyle(document.body).backgroundColor). look=true only if the code returned "${PAGE_BACKGROUND}" and the screenshot shows the calculator's own look: a charcoal page; at the top left a small crimson square right before the title "Magnetic coupling calculator"; in the row of tabs the selected tab, "Size and torque", marked by a crimson line under its name and not by a filled button; and every text drawn in one clean sans-serif face, with no empty box where a character should be. Collect the console messages at level "info" (this page's: the next step navigates away).
```

Replace:

```js
passed=true only if: page loaded, canvas present, starting_design, first_view,
```

with:

```js
passed=true only if: page loaded, canvas present, starting_design, look, first_view,
```

- [ ] **Step 6: Run to verify both pass.**

  ```bash
  node linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test
  ```

  Expected: `all 77 checks hold`; the app test passes.

- [ ] **Step 7: The README.**

**`magcoupling-rs/README.md`**

Replace:

```markdown
and that the address stays clean (UX overhaul slice A, `starting_design`); then it opens
```

with:

```markdown
and that the address stays clean (UX overhaul slice A, `starting_design`), and on the same page that the calculator wears its own look (slice B stage 1, `look`: the page's background is the theme's panel colour, read from the page, and the screenshot shows a charcoal page, the crimson square before the title, the crimson line under the selected tab and every text in one sans-serif face with no empty box); then it opens
```

Replace:

```markdown
; the theme stays dark when the system turns light |
```

with:

```markdown
; the smoke script's `PAGE_BACKGROUND`, typed once, is the theme's panel colour as a browser reports it, and its clean-page step judges the look, reads the page's background and names the title and the active tab; the theme stays dark when the system turns light |
```

- [ ] **Step 8: The checks before a commit.** Expected: the lib run still ends `1045 passed; 0 failed; 6 ignored`.

- [ ] **Step 9: Commit.**

  ```text
  test(magcoupler): gui-smoke checks the calculator's look on the clean page (BL-079 T5)

  The clean page's step reads the page's background (the theme's panel colour, typed once as
  PAGE_BACKGROUND and held to app::theme::PANEL by the app's smoke test) and judges the
  screenshot: a charcoal page, the crimson square before the title, the crimson line under
  the selected tab, every text in one sans-serif face with no empty box. It is the one check
  of a browser's own drawing of the theme and the font. The verdict test (gate 13) has the
  case: 77 checks.
  ```

  ```bash
  git add .claude/workflows/gui-smoke.js linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs magcoupling-rs/src/app.rs magcoupling-rs/README.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T5.txt
  ```

### Task 6: Docs sweep

**Model:** `sonnet` (doc and YAML edits from exact blocks).

**Files:**
- Modify: `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`, `docs/ai/backlog.yaml`, `docs/analyses/2026-11-10-conservative-assumptions.md`, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`

**Interfaces:**
- Consumes: the branch as Tasks 1 to 5 leave it.
- Produces: BL-079 reads `status: in_progress`.

- [ ] **Step 1: Apply the sweep.** The YAML lines are very long: each old text below is a part of its line that occurs once. In `docs/ai/05-update-tracker.md` and in the register's two log lines, change the date if the build runs on another day than 2026-11-18.

**`docs/ai/02-system.yaml`**

Replace:

```yaml
undo/redo, design files and share links, the sizing mode and the linkage app's theme.
```

with:

```yaml
undo/redo, design files and share links, the sizing mode and the theme (the linkage app's then; the calculator's own since UX overhaul slice B, stage 1, BL-079).
```

Replace:

```yaml
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077;
```

with:

```yaml
      - "The look (UX overhaul slice B, stage 1, BL-079; design sections 6 and 8.1 tests 6 and 7): the standalone app owns its theme and its font, and the panel still sets neither. app::theme holds the one token table (BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK, LINK) and visuals(), egui's dark visuals with the tokens in place of its colours: a test holds every field to the table and every other field to egui's, and another holds every text token to 4.5:1 on every surface it is drawn on (muted and the level colours are under it on the border colour, the fill of a hovered button and of a selected chip, whose own text is TEXT and LINK; a caption whose text is selected with the pointer is not counted). Body text, values and headings share TEXT (egui has no bold: strong() no longer sets a value apart), weak() is MUTED. The accent is never a text colour and never on a value or a check: it marks the brand mark and the active tab only (gui::look::mark_ui and tab_ui; a test counts its shapes on the app's page). ACCENT and OK are the panel's own constants (gui::look), since no egui theme has a field for them; the spreadsheet export's fills keep their own palette (xlsx::fill). app::fonts installs a cut of Inter Regular first in the proportional family (egui's built-in fonts stay behind it and stay monospace); the cut is built once by assets/fonts/subset_inter.py, its ten digits are Inter's tabular ones, and with the app feature every glyph guard also holds its texts to the cut (test_support::assert_glyphs through app::fonts::cut_holds). The spacing is unchanged, and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone."
      - "The session baseline and the two named designs (UX overhaul slice A, BL-077;
```

Replace:

```yaml
  - "egui's default fonts have no U+2192 (the arrow) or U+2264:
```

with:

```yaml
  - "egui 0.32 and egui_plot 0.33, learned by the look (BL-079): a selected selectable_label is filled with visuals.selection.bg_fill, draws its text in selection.stroke.color and takes its outline from the widget state's bg_stroke, so an accent in selection.bg_fill fills every selected choice. egui_plot draws its grid, ticks and axis labels from visuals.text_color() (a legend entry from its button state), so a bright body text makes a loud grid. Fonts::has_glyph is false for a family's own replacement character (the question mark when the family has no U+25FB). Context::add_font with FontPriority::Highest puts a font ahead of the built-in ones before the first frame. Inter's tnum feature substitutes the space and the punctuation as well as the digits: remap the digits alone. In .claude/workflows/gui-smoke.js a prompt or description inside single quotes takes no apostrophe (gate 13 fails on the syntax error)."
  - "egui's default fonts have no U+2192 (the arrow) or U+2264:
```

**`docs/ai/03-structure.yaml`**

Replace:

```yaml
no eframe, no I/O): typeset.rs (the equation typesetter:
```

with:

```yaml
no eframe, no I/O): look.rs (the panel's own part of the look, UX overhaul slice B stage 1: ACCENT and OK, the two token colours no egui theme carries; mark_ui, the brand mark before the title; tab_ui, a tab of the centre's row, underlined in the accent when active), typeset.rs (the equation typesetter:
```

Replace:

```yaml
MagcouplingApp::new applies the theme and builds the panel
```

with:

```yaml
MagcouplingApp::new applies the theme and the font and builds the panel
```

Replace:

```yaml
src/app/theme.rs (cad_dark_visuals, a copy of linkage-sim-rs's checked body for body; apply forces ThemePreference::Dark)
```

with:

```yaml
src/app/theme.rs (the calculator's own theme, UX overhaul spec 6: the token table BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, BAD, WARN, OK and LINK; visuals(), egui's dark visuals with the tokens; apply forces ThemePreference::Dark), src/app/fonts.rs (INTER, the cut of Inter Regular embedded from assets/fonts/Inter-Regular.subset.ttf; install puts it first in the proportional family; cut_holds, tests only), assets/fonts/ (the cut, subset_inter.py that built it with fontTools, OFL.txt)
```

**`docs/ai/04-memory.yaml`**

Replace:

```yaml
Next after the release: slice B, the instrument (BL-068, which it needs, is merged)."
```

with:

```yaml
Slice A is merged (17e4638). Slice B is built in stages (the design names none; the stage 1 plan's reading, its decision L-1): stage 1, the look (BL-079; plan docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md; branch magcoupling/overhaul-b1-the-look), gives the calculator its own theme (one token table, charcoal and crimson) and font (a cut of Inter with tabular digits) on today's layout. Next: stage 2, the screen: the top bar and its Design menu, the Size column, the four tiles, the check list (which puts the fixed-3D line of register row A-18 on the first view) and the Explain panel; then the word budget, the nothing-lost table and the slip-heating readout's stray unit."
```

Replace:

```yaml
row A-17 is the one that changes what a visitor sees."
```

with:

```yaml
row A-17 is the one that changes what a visitor sees. Slice B's stage 1 (BL-079) makes O-4 and Q-1 real: the look is 'Charcoal and crimson' and decision M41-3 (theme matches the linkage app) is reversed for the calculator alone. Its plan's own calls L-1 to L-14 are in docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md; the ones a visitor sees: body text is as bright as values (one text colour, no bold), a selected choice is a grey chip with blue text, and the digits alone are tabular."
```

**`docs/ai/05-update-tracker.md`**

Replace:

```markdown
## 2026-11-17 - Magcoupler slice A, fixes from the independent review (BL-077; branch magcoupling/overhaul-a)
```

with:

```markdown
## 2026-11-17 - Magcoupler UX overhaul, slice B, stage 1: the look (BL-079; branch magcoupling/overhaul-b1-the-look)
- What changed: the calculator has its own theme and font, on today's layout. `app/theme.rs` holds one token table (UX overhaul spec 6.1: charcoal surfaces, hairlines, text, muted, the crimson accent, the three level colours; links keep egui's blue) and `visuals()`, egui's dark visuals with the tokens in place of its colours; the web page's background is the panel colour. This reverses M41-3 ("Theme matches the linkage app"; the spec's question Q-1, on its recommendation): the linkage app is not touched. `app/fonts.rs` installs Inter ahead of egui's built-in fonts: a 30,684-byte cut of Inter Regular whose ten digits are its tabular ones (`assets/fonts/`, with the script that built it and the licence). The accent marks the brand (a square before the title) and the active tab (a line under it), and nothing else (`gui/look.rs`). The plots' grid, ticks and axis labels take the weak text colour. The hub page's card no longer lists "clamps".
- Tests: the tokens are the spec's table and the visuals are the tokens (this replaces the parity test with the linkage app); every text token is at least 4.5:1 on every surface it is drawn on, with the spec's own figures; the digits have one advance width; every glyph guard of the panel also holds its texts to the cut of Inter (`test_support::assert_glyphs`, with the app feature); the accent is on two shapes of the app's page and on no text or badge. The gui-smoke workflow gained `look`, and its verdict test (gate 13) the case.
- What the spike found (design 6.2): the cut holds every character the guards collect; Inter's `tnum` feature also widens spaces and punctuation, so the digits alone are remapped; Inter's rows are taller than the built-in font's (1.21 em), and the first view at 1280 x 800 still shows the whole centre; `strong()` no longer sets a value apart (one text colour, no bold): a heavier face is the cure if the tiles of stage 2 need one.
- Not in this stage: the layout (the top bar, the Size column, the tiles, the check list) and the Explain panel, which are stage 2; the fixed-3D line on the first view (register row A-18), which stage 2's check list brings; the slip-heating readout's stray unit (design 9.2).

## 2026-11-17 - Magcoupler slice A, fixes from the independent review (BL-077; branch magcoupling/overhaul-a)
```

**`docs/analyses/2026-11-10-conservative-assumptions.md`**

Replace:

```markdown
- 2026-11-17: A-18 added by the slice A text and physics review (BL-077): the starting design's temperature verdicts use the example design's stored 3D fields. It is an assumption, not the owner's answer.
```

with:

```markdown
- 2026-11-17: A-18 added by the slice A text and physics review (BL-077): the starting design's temperature verdicts use the example design's stored 3D fields. It is an assumption, not the owner's answer.
- 2026-11-18: A-2 (Q-1) is built: slice B's stage 1 (BL-079) gives the calculator its own theme and font. The other answer puts the linkage app's visuals back: `visuals()` in `magcoupling-rs/src/app/theme.rs` and the page colour in `linkage-sim-rs/web/tools/magcoupler/index.html`. (The row's "slice C" is slice B.)
- 2026-11-18: A-18 is not changed by slice B's stage 1, which keeps today's layout: the line "Fixed 3D fields from the example design" stays under the shaft joints row (owner ruling R-15), below the dashboard's first screen at 1280 x 800. Stage 2's check list closes with it (the design's section 4.5), which puts it on the first view; that plan pins it with a test.
```

**`docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`**

Replace:

```markdown
slice A is built on the branch `magcoupling/overhaul-a` (BL-077) and waits for the owner's review of its diff; slices B to F are not built.**
```

with:

```markdown
slice A (BL-077) is merged (17e4638); slice B is built in stages, and its first, the look (section 6; BL-079), is built by the plan `docs/superpowers/plans/2026-11-18-magcoupler-overhaul-b1-the-look.md`; the rest of slice B (sections 4 and 5) and slices C to F are not built.**
```

Replace:

```markdown
Each later slice's plan files one item with the next free id (`dimension: gui`; `risk: physics`
```

with:

```markdown
BL-079 is the item of slice B's first stage, the look (filed 2026-11-18 with its plan). Each later plan files one item with the next free id (`dimension: gui`; `risk: physics`
```

Replace:

```markdown
  command that made it are committed; the app build does not run font tools.
```

with:

```markdown
  command that made it are committed; the app build does not run font tools. (Built 2026-11-18,
  BL-079: the ten digits alone are remapped to their tabular forms. The whole feature also
  substitutes the space, the full stop, the comma, the colon, the hyphen and the brackets, which
  would widen them in running text.)
```

Replace:

```markdown
  plots at 1280x800 and 1920x1080. The spike is throwaway; its findings go into the slice B plan.
```

with:

```markdown
  plots at 1280x800 and 1920x1080. The spike is throwaway; its findings go into the slice B plan.
  (Done 2026-11-17: the findings are in the stage 1 plan, under "What the spike found".)
```

Replace:

```markdown
- That egui 0.32 draws the frozen-`tnum` Inter subset correctly, and what it costs in wasm bytes (the spike).
```

with:

```markdown
- That egui 0.32 draws the frozen-`tnum` Inter subset correctly, and what it costs in wasm bytes (the spike).
  Checked 2026-11-17 in a headless render of the real app (the equations, the drawing's tags and two plots, at
  1280x800 and 1920x1080); the file is 30,684 bytes. A browser's own drawing of it is the smoke run's `look`.
```

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: open
  notes: "Filed with the plan of slice B's first stage on 2026-11-18
```

with:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b1-the-look (Tasks 1 to 6 of the plan); the full gate, the smoke run and the whole-branch review are its Task 7. Filed with the plan of slice B's first stage on 2026-11-18
```

- [ ] **Step 2: Check the YAML still parses and the doc tests pass.**

  ```bash
  python -c "import io, yaml; [yaml.safe_load(io.open('docs/ai/' + f, encoding='utf-8')) for f in ('02-system.yaml', '03-structure.yaml', '04-memory.yaml', 'backlog.yaml')]; print('parses')"
  cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count
  ```

  Expected: `parses`; the doc-count test passes (this sweep touches none of the counts it reads).

- [ ] **Step 3: Read the README's panel section once, top to bottom,** for a sentence the stage made stale that no task caught (for one: a sentence that says the app wears the linkage app's theme, or that the page is drawn in egui's default fonts). If you find one, do not reword it yourself: report the sentence and where it is, and the controller sets the wording (a `fable` call). Do not touch a sentence that is still true: the panel under a plain context still has egui's fonts and visuals, and the panel's tests still run there.

- [ ] **Step 4: Commit.**

  ```text
  docs(magcoupler): slice B stage 1 docs sweep: docs/ai, the tracker, the register, the design's header, BL-079 in progress (BL-079 T6)
  ```

  ```bash
  git add docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml docs/analyses/2026-11-10-conservative-assumptions.md docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md
  git commit -F .superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/msg/T6.txt
  ```

### Task 7: Release: the full gate, gui-smoke, the whole-branch review, the merge

**Model:** the `fable` controller; the whole-branch reviewer is a fresh-context `fable` agent; git steps run on `haiku` at low effort; the gui-smoke workflow runs its own agents on `sonnet`.

**Files:**
- Modify: `docs/ai/backlog.yaml` (Step 6)

**Interfaces:**
- Consumes: the branch as Tasks 1 to 6 leave it; `linkage-sim-rs/scripts/gate.sh`, `build_web.sh` (run by the gate's `--full`), `serve_web.sh`; the branch's own `.claude/workflows/gui-smoke.js`.

- [ ] **Step 1: Preconditions.**

  ```bash
  git status --short
  git diff --stat main...HEAD -- magcoupling-rs/src/engine magcoupling-rs/tests reference linkage-sim-rs/src
  ```

  Expected: a clean tree; the second command prints nothing (no engine, integration-test, oracle or linkage-app file changed).

- [ ] **Step 2: The full gate.** From PowerShell at the worktree's root. Plain `bash` there is WSL and fails on the script's line endings.

  ```powershell
  $env:MAGCOUPLING_PYTHON = "C:\Users\Cole\source\repos\linkage_simulation\reference\magcoupling-py\.venv\Scripts\python.exe"
  & 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/gate.sh --full
  git checkout -- docs/chebyshev_lambda
  ```

  Expected: the last line is `GATE PASS`; gate 12 printed `oracle Python:` and not a SKIP line; gate 13 printed `all 77 checks hold`; gate 14 built both web bundles. (The oracle's Python is the main checkout's: a worktree has no `.venv` of its own.) If a gate fails, fix the cause on the branch, red test first, and run this step again from the top. Never edit `gate.sh` to pass.

- [ ] **Step 3: gui-smoke, from the branch's own script.**
  1. In the background: `& 'C:\Program Files\Git\bin\bash.exe' linkage-sim-rs/scripts/serve_web.sh 8080` (it serves `linkage-sim-rs/web/`, which Step 2's gate 14 just built).
  2. Run the Workflow tool with `scriptPath` set to this worktree's `.claude/workflows/gui-smoke.js` and the arguments `{"frame_check": true}`. The `gui-smoke` skill of the session runs the main checkout's script, which has no `look`: do not use it for this run.
  3. Expected: `passed: true`; in `magcoupling`: `look: true`, `starting_design: true`, `first_view: true`, `address_follows: true`, `torque_card: true`, `drag_frames_within_60fps: true`, `console_errors: []`. Keep the result's JSON in `.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/smoke-local.json`, and the run's first screenshot beside it.
  4. Stop the server.

  A failure of `look` is a finding about the build until shown otherwise: open `http://localhost:8080/tools/magcoupler/` by hand and compare it with `.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/spike/after-first-view-1280.png` in the main checkout. The other checks read text off screenshots of a page whose font and colours changed: a check that fails because the agent could not read a label is a finding too, and it is reported, not waved through.

- [ ] **Step 4: The independent whole-branch review** (fresh context, `fable`). Give it `git diff main...HEAD`, the design, this plan, the spike's pictures and Step 3's screenshot, and none of the builders' notes. It checks:
  1. The design's section 6 and tests 6 and 7 of 8.1, item by item, against the diff. A gap is a finding.
  2. The decisions C-4 to C-7, C-13 and C-14: where the accent is drawn; that no text is in it; that links are blue; that the muted token is `#8a919a`.
  3. The anti-goals of 2.3: no engine number moves; nothing that was visible is hidden (the keyway line, the fixed-3D line where it was, the banners); no decoration that is not information.
  4. No existing assertion was weakened: the only edits to existing tests are the ones this plan lists under "Existing tests this plan edits".
  5. The panel sets no theme and no font, and still works in egui's default visuals. The linkage app's files are not in the diff.
  6. The decisions L-1 to L-14: each is what the code does, and none hides a behaviour the owner should rule on.
  7. The pictures: at the failing example design, does the crimson read as the brand and the red as the failure? (The design's risk "Crimson beside red".)

  One fix wave on `fable` for what it finds, each fix with its red test first; the reviewer re-reads the fixes. Record the review in `.superpowers/sdd/2026-11-18-magcoupler-overhaul-b1/final-review.md` (gitignored).

- [ ] **Step 5: If Step 4 changed a file, run Steps 2 and 3 again.** The merge needs a gate and a smoke run of the tree that is merged.

- [ ] **Step 6: Close the item and merge.** In `docs/ai/backlog.yaml` (use the day's date, and say what the review found if it found something):

**`docs/ai/backlog.yaml`**

Replace:

```yaml
  status: in_progress
  notes: "Built on branch magcoupling/overhaul-b1-the-look (Tasks 1 to 6 of the plan); the full gate, the smoke run and the whole-branch review are its Task 7. Filed
```

with:

```yaml
  status: fixed
  notes: "FIXED 2026-11-18: built on branch magcoupling/overhaul-b1-the-look; GATE PASS with --full, the gui-smoke run passed with look true, and the independent whole-branch review approved it. Filed
```

  Commit it (`docs(magcoupler): BL-079 fixed: the gate, the smoke run and the whole-branch review passed`), then merge the branch into `main` with a merge commit whose message lists the tasks' commit subjects. Remove the worktree with `git worktree remove`, never `git worktree prune`.

- [ ] **Step 7: Report, then the push (the controller's call).** A push to `main` deploys the site. The report to the owner holds: what changed on screen, in three sentences; the two pictures to look at (the first view, and the failing example design for "crimson beside red"); the decisions L-1 to L-14, one line each; what follows as stage 2, with the fixed-3D line of register row A-18 named; and what was not verified (Safari and Firefox; the release build's size). After a push, when the `deploy-web.yml` run is green, run the smoke workflow with `{"url": "https://colesorkness.com"}` and report the result. If it fails, say so plainly: the site is live with the change.

## What this plan covers of the design

| Design item | Where |
|---|---|
| 6.1 The token table, value for value | Task 1 (`the_tokens_are_the_spec_s_table`) |
| 6.1 Contrast: every text token at least 4.5:1, with the design's own figures | Task 1 (`every_text_token_reads_on_every_surface_it_is_drawn_on`) |
| 6.1 The accent's roles: the brand mark and the active tab's underline | Task 3. The one primary button is stage 2's. |
| 6.2 Inter, bundled and installed ahead of the built-in fonts, which stay as the fallback | Task 2 |
| 6.2 Tabular digits as the file's default digits; the file, its licence and the command committed | Task 2 (decision L-6: the digits alone) |
| 6.2 One face | Task 2 (decision L-8) |
| 6.2 The spike, and its findings in the plan | "What the spike found" |
| 6.3 The theme lives in `app/theme.rs`; the panel reads the host's `Visuals` | Tasks 1 and 3 (the panel's tests run in egui's default visuals) |
| 6.3 The Good green takes the token's value; palettes without a token stay | Task 1 (decisions L-10 and L-11) |
| 6.3 The parity test is replaced; the page's background moves to the new panel colour (Q-1) | Task 1 |
| 8.1 Test 6 | Task 1 |
| 8.1 Test 7 | Task 2: every glyph guard runs against the cut, and the built-in fonts stay behind it; the digits have one advance width |
| 8.3 Text rules | No on-screen string is added. The hub's card loses "clamps" (Task 4). |
| 9.1 Docs with each slice | Each task's README step; Task 5 (the smoke run); Task 6 |
| 9.2 The hub card's "clamps" | Task 4 |
| 9.2 The slip-heating readout's stray unit | Stage 2 |
| 9.4 Release | Task 7. No stop for the owner: BL-079 is `risk: mechanical` (decision L-12). |
| C-4 State by shape and word as well as colour | Stage 2 (the check rows). Today's badges keep their hover word. |
| C-5, C-13, C-14 | Tasks 1 and 3 |
| C-6 Hairlines, no gradients or glows; tabular digits | Tasks 1 and 2. Numbers leading their labels on tiles is stage 2's. |
| C-7 The drawing keeps its pole colours | Not touched (decision L-11) |
| Sections 4 and 5, tests 8 to 15; register row A-18 on the first view | Stage 2 |
